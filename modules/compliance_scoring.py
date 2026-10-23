# Stage 3: PySceneDetect, Falconsai Modesty & Gemini Fiqh Audit
import os
import json
import time
import cv2
import torch
from PIL import Image
from scenedetect import detect, ContentDetector
from transformers import pipeline as hf_pipeline
from google import genai
from google.genai import types
from core.pipeline import (
    PipelineStage, PipelineContext, ClipCandidate, register_stage, logger
)


@register_stage("scene_and_modesty_gate")
class SceneAndModestyGateStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        threshold_nsfw = self.config["compliance"]["nsfw_confidence_threshold"]
        device_idx = 0 if torch.cuda.is_available() else -1
        classifier = hf_pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device_idx
        )

        for video in context.selected_videos:
            v_path = str(video.raw_video_path)
            scenes = detect(v_path, ContentDetector(threshold=27.0, min_scene_len=15))

            cap = cv2.VideoCapture(v_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            raw_bounds = (
                [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
                if scenes else [(0.0, total_frames / fps)]
            )

            safe_scenes = []
            for start_s, end_s in raw_bounds:
                mid_frame = int(((start_s + end_s) / 2.0) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                if not ret:
                    continue
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                preds = {p["label"].lower(): p["score"] for p in classifier(pil_img)}
                if preds.get("nsfw", 0.0) < threshold_nsfw:
                    safe_scenes.append((round(start_s, 2), round(end_s, 2)))
                else:
                    logger.warning(f"Scene {start_s}-{end_s}s excluded by Visual Modesty Gate.")

            cap.release()
            video.scene_boundaries = safe_scenes
        return context


@register_stage("viral_fiqh_scoring")
class ViralFiqhScoringStage(PipelineStage):
    def _call_gemini_with_fallback(self, client: genai.Client, prompt: str) -> list:
        primary_model = self.config["compliance"].get("gemini_model", "gemini-3.8-flash")
        fallback_model = self.config["compliance"].get("fallback_gemini_model", "gemini-3.5-flash-lite")

        for attempt, model_id in enumerate([primary_model, fallback_model]):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                return json.loads(response.text)
            except Exception as exc:
                logger.warning(f"Gemini call on {model_id} failed ({exc}). Trying fallback...")
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("Both primary and fallback Gemini models failed.")

    def execute(self, context: PipelineContext) -> PipelineContext:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is missing.")

        client = genai.Client(api_key=api_key)
        num_clips = self.config["clipping"]["num_clips_per_video"]
        min_dur = self.config["clipping"]["min_clip_duration"]
        max_dur = self.config["clipping"]["max_clip_duration"]

        for video in context.selected_videos:
            if not video.words:
                logger.warning(f"No transcribed words found for {video.video_id}.")
                continue

            chunks = []
            curr_words, chunk_start = [], 0.0
            for w in video.words:
                if not curr_words:
                    chunk_start = w.start
                curr_words.append(w.word)
                if w.end - chunk_start >= 5.0:
                    chunks.append(f"[{chunk_start:.2f}s - {w.end:.2f}s] {' '.join(curr_words)}")
                    curr_words = []
            if curr_words:
                chunks.append(f"[{chunk_start:.2f}s - {video.words[-1].end:.2f}s] {' '.join(curr_words)}")

            prompt = f"""
Analyze the following timestamped transcript and safe visual scene boundaries from the video '{video.title}'.
Select up to {num_clips} self-contained short clips between {min_dur} and {max_dur} seconds long.

STRICT SELECTION & SHARIAH COMPLIANCE RULES:
1. Viral Hook & Completeness: Each clip must start with a compelling hook and end at a complete sentence boundary that aligns closely with safe scene boundaries: {video.scene_boundaries[:60]}.
2. Islamic Ethical Compliance: Reject any segment containing profanity, backbiting/slander (ghibah), promotion of riba (interest), gambling (maysir), intoxicants, or anti-Islamic theology.
3. Return ONLY a valid JSON list matching this schema:
[
  {{
    "start_time": 12.4,
    "end_time": 48.2,
    "hook_title": "Catchy halal viral title under 70 chars #Shorts",
    "caption_text": "Engaging description with ethical takeaways #halal #shorts",
    "virality_score": 92,
    "fiqh_compliance_passed": true,
    "fiqh_notes": "Clean educational segment on discipline."
  }}
]

TRANSCRIPT:
{chr(10).join(chunks)}
"""
            parsed = self._call_gemini_with_fallback(client, prompt)
            for idx, item in enumerate(parsed[:num_clips]):
                if not item.get("fiqh_compliance_passed", False):
                    continue
                s_time, e_time = float(item["start_time"]), float(item["end_time"])
                clip_words = [
                    w for w in video.words if s_time <= w.start and w.end <= e_time
                ]
                video.candidates.append(ClipCandidate(
                    clip_id=f"{video.video_id}_clip_{idx+1}",
                    source_video_id=video.video_id,
                    start_time=s_time,
                    end_time=e_time,
                    hook_title=item["hook_title"],
                    caption_text=item["caption_text"],
                    virality_score=int(item["virality_score"]),
                    fiqh_compliance_passed=True,
                    fiqh_notes=item.get("fiqh_notes", ""),
                    words=clip_words
                ))
        return context