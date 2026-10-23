# Stage 4: FFmpeg 9:16 Vertical Reframing & Karaoke ASS Captions
import subprocess
from pathlib import Path
from core.pipeline import PipelineStage, PipelineContext, ClipCandidate, register_stage, logger


def format_ass_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


def generate_karaoke_ass(candidate: ClipCandidate, ass_path: Path):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,70,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,80,80,380,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    words = candidate.words
    offset = candidate.start_time

    for i in range(0, len(words), 3):
        group = words[i:i+3]
        for active_idx, active_w in enumerate(group):
            w_start = max(0.0, active_w.start - offset)
            w_end = max(w_start + 0.05, active_w.end - offset)
            rendered_tokens = []
            for j, token in enumerate(group):
                clean_txt = token.word.replace("{", "").replace("}", "").replace("\\", "")
                if j == active_idx:
                    rendered_tokens.append(r"{\c&H0034C9FC&}" + clean_txt + r"{\c&H00FFFFFF&}")
                else:
                    rendered_tokens.append(clean_txt)
            line_text = " ".join(rendered_tokens)
            events.append(
                f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},Default,,0,0,0,,{line_text}"
            )

    ass_path.write_text(header + "\n".join(events), encoding="utf-8")


@register_stage("vertical_composer")
class VerticalComposerStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        mode = self.config["clipping"].get("aspect_ratio_mode", "blurred_pad")
        for video in context.selected_videos:
            for clip in video.candidates:
                ass_file = context.work_dir / f"{clip.clip_id}.ass"
                generate_karaoke_ass(clip, ass_file)

                out_mp4 = context.work_dir / f"{clip.clip_id}_final.mp4"
                duration = round(clip.end_time - clip.start_time, 3)
                escaped_ass = str(ass_file).replace("\\", "/").replace(":", "\\:")

                if mode == "blurred_pad":
                    vf_graph = (
                        f"[0:v]split=2[bg][fg];"
                        f"[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
                        f"crop=1080:1920,gblur=sigma=20[bg_blur];"
                        f"[fg]scale=1080:-2[fg_scaled];"
                        f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2,"
                        f"ass='{escaped_ass}'[v_out]"
                    )
                else:
                    vf_graph = (
                        f"[0:v]scale=-2:1920,crop=1080:1920:(in_w-1080)/2:0,"
                        f"ass='{escaped_ass}'[v_out]"
                    )

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(clip.start_time), "-t", str(duration), "-i", str(video.raw_video_path),
                    "-ss", str(clip.start_time), "-t", str(duration), "-i", str(video.clean_vocal_path),
                    "-filter_complex", f"{vf_graph};[1:a]loudnorm=I=-14:TP=-1.5:LRA=11[a_out]",
                    "-map", "[v_out]", "-map", "[a_out]",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
                    str(out_mp4)
                ]
                logger.info(f"Rendering 9:16 vertical Short: {clip.clip_id}")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                clip.rendered_mp4_path = out_mp4
        return context