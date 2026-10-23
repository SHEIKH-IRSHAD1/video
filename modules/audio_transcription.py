# Stage 2: Meta Demucs Vocal Isolation & Faster-Whisper ASR
import subprocess
import torch
from faster_whisper import WhisperModel
from core.pipeline import PipelineStage, PipelineContext, WordTiming, register_stage, logger


@register_stage("audio_purification")
class AudioPurificationStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        comp_cfg = self.config["compliance"]
        for video in context.selected_videos:
            if not video.raw_video_path or not video.raw_video_path.exists():
                continue

            raw_audio = context.work_dir / f"{video.video_id}_source.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(video.raw_video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(raw_audio)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not comp_cfg.get("strip_music_with_demucs", True):
                video.clean_vocal_path = raw_audio
                continue

            model_name = comp_cfg.get("demucs_model", "htdemucs")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            out_dir = context.work_dir / "separated"

            logger.info(f"Isolating human vocals via Demucs ({model_name}) on {device}...")
            cmd = [
                "python3", "-m", "demucs.separate",
                "--two-stems", "vocals",
                "-n", model_name,
                "-d", device,
                "-o", str(out_dir),
                str(raw_audio)
            ]
            subprocess.run(cmd, check=True)

            isolated_vocals = out_dir / model_name / raw_audio.stem / "vocals.wav"
            if not isolated_vocals.exists():
                raise FileNotFoundError(f"Expected Demucs vocal stem at {isolated_vocals}")

            video.clean_vocal_path = isolated_vocals
        return context


@register_stage("transcription")
class TranscriptionStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        has_gpu = torch.cuda.is_available()
        device = "cuda" if has_gpu else "cpu"
        compute_type = "float16" if has_gpu else "int8"
        model_size = "large-v3-turbo" if has_gpu else "small"

        logger.info(f"Loading faster-whisper ({model_size}) on {device} ({compute_type})...")
        whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)

        for video in context.selected_videos:
            audio_target = video.clean_vocal_path or video.raw_video_path
            segments, _ = whisper_model.transcribe(
                str(audio_target),
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            words: list[WordTiming] = []
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        words.append(WordTiming(
                            word=w.word.strip(),
                            start=round(w.start, 3),
                            end=round(w.end, 3),
                            probability=round(w.probability, 3)
                        ))
            video.words = words
            logger.info(f"Transcribed {len(words)} words for {video.video_id}.")
        return context