# Headless CLI runner for GitHub Actions automation
import argparse
import shutil
import uuid
from pathlib import Path
from core.pipeline import ModularPipelineOrchestrator, VideoItem, logger

import modules.discovery  # noqa: F401
import modules.audio_transcription  # noqa: F401
import modules.compliance_scoring  # noqa: F401
import modules.composer  # noqa: F401
import modules.publisher  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Headless Halal Shorts Automation Runner")
    parser.add_argument("--topic", type=str, default="Islamic Golden Age history")
    parser.add_argument("--video-url", type=str, default="", help="Optional direct video URL")
    parser.add_argument("--num-clips", type=int, default=3)
    parser.add_argument("--strip-music", type=str, default="true")
    parser.add_argument("--publish-youtube", type=str, default="false")
    parser.add_argument("--output-dir", type=str, default="output_shorts")
    args = parser.parse_args()

    orchestrator = ModularPipelineOrchestrator("config.yaml")
    orchestrator.config["discovery"]["default_topic"] = args.topic
    orchestrator.config["clipping"]["num_clips_per_video"] = int(args.num_clips)
    orchestrator.config["compliance"]["strip_music_with_demucs"] = args.strip_music.lower() == "true"
    orchestrator.config["publishing"]["enable_youtube_shorts"] = args.publish_youtube.lower() == "true"

    ctx = orchestrator.create_context(run_id=uuid.uuid4().hex[:8])

    if args.video_url.strip():
        ctx.selected_videos = [
            VideoItem(
                video_id=f"cli_{uuid.uuid4().hex[:6]}",
                title="Direct URL Video",
                channel_title="User Provided",
                url=args.video_url.strip(),
                views=0,
                published_at="",
                viral_velocity=0.0,
                license_type="user_verified"
            )
        ]

    try:
        ctx = orchestrator.run_stages(ctx)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for v in ctx.selected_videos:
            for c in v.candidates:
                if c.rendered_mp4_path and c.rendered_mp4_path.exists():
                    dest = out_dir / c.rendered_mp4_path.name
                    shutil.copy2(c.rendered_mp4_path, dest)
                    logger.info(f"Saved rendered Short to {dest} | Hook: {c.hook_title}")
    finally:
        orchestrator.cleanup(ctx)


if __name__ == "__main__":
    main()