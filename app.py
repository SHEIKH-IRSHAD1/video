# Interactive Gradio Web Dashboard (ZeroGPU enabled)
import uuid
import spaces
import gradio as gr
from core.pipeline import ModularPipelineOrchestrator, VideoItem, STAGE_REGISTRY

import modules.discovery  # noqa: F401
import modules.audio_transcription  # noqa: F401
import modules.compliance_scoring  # noqa: F401
import modules.composer  # noqa: F401
import modules.publisher  # noqa: F401

orchestrator = ModularPipelineOrchestrator("config.yaml")


def search_online_videos(topic: str, license_mode: str, max_videos: int):
    orchestrator.config["discovery"]["default_topic"] = topic
    orchestrator.config["discovery"]["license_mode"] = license_mode
    orchestrator.config["discovery"]["max_source_videos"] = int(max_videos)

    disc_stage = STAGE_REGISTRY["ethical_discovery"](orchestrator.config)
    items = disc_stage.search_viral_candidates(topic, max_results=int(max_videos))

    table_rows = [
        [v.video_id, v.title, v.channel_title, v.views, v.viral_velocity, v.license_type, v.url]
        for v in items
    ]
    choices = [f"{v.video_id} | {v.title[:60]} (Velocity: {v.viral_velocity}/hr)" for v in items]
    return table_rows, gr.update(choices=choices, value=choices[:1] if choices else []), items


@spaces.GPU(duration=180)
def run_gpu_clipping_pipeline(
        selected_labels: list[str],
        custom_url: str,
        cached_items: list[VideoItem],
        num_clips: int,
        min_dur: float,
        max_dur: float,
        strip_music: bool,
        aspect_mode: str,
        pub_yt: bool,
        pub_ig: bool,
        pub_tt: bool
):
    orchestrator.config["clipping"]["num_clips_per_video"] = int(num_clips)
    orchestrator.config["clipping"]["min_clip_duration"] = float(min_dur)
    orchestrator.config["clipping"]["max_clip_duration"] = float(max_dur)
    orchestrator.config["clipping"]["aspect_ratio_mode"] = aspect_mode
    orchestrator.config["compliance"]["strip_music_with_demucs"] = bool(strip_music)
    orchestrator.config["publishing"]["enable_youtube_shorts"] = bool(pub_yt)
    orchestrator.config["publishing"]["enable_instagram_reels"] = bool(pub_ig)
    orchestrator.config["publishing"]["enable_tiktok"] = bool(pub_tt)

    ctx = orchestrator.create_context(run_id=uuid.uuid4().hex[:8])

    selected_ids = {label.split(" | ")[0] for label in (selected_labels or [])}
    ctx.selected_videos = [v for v in (cached_items or []) if v.video_id in selected_ids]

    if custom_url and custom_url.strip():
        custom_vid_id = f"custom_{uuid.uuid4().hex[:6]}"
        ctx.selected_videos.append(VideoItem(
            video_id=custom_vid_id,
            title="Custom Direct URL Video",
            channel_title="User Provided",
            url=custom_url.strip(),
            views=0,
            published_at="",
            viral_velocity=0.0,
            license_type="user_verified"
        ))

    if not ctx.selected_videos:
        raise gr.Error("Please select at least one discovered video or enter a custom video URL.")

    ctx = orchestrator.run_stages(ctx)

    rendered_files = []
    summary_rows = []
    for v in ctx.selected_videos:
        for c in v.candidates:
            if c.rendered_mp4_path and c.rendered_mp4_path.exists():
                rendered_files.append(str(c.rendered_mp4_path))
            summary_rows.append([
                c.clip_id, c.hook_title, f"{c.start_time}s - {c.end_time}s",
                c.virality_score, c.fiqh_notes, str(c.published_urls)
            ])
    return rendered_files, summary_rows


with gr.Blocks(title="Ethical & Halal Viral Shorts Automation Studio") as demo:
    gr.Markdown("# 🎬 Cloud-Native Halal Viral Shorts Studio")
    gr.Markdown("Discover ethical Creative Commons videos, remove musical instruments with Meta Demucs, audit for Islamic compliance, and generate 9:16 vertical Shorts.")
    cached_videos_state = gr.State([])

    with gr.Tab("1. Discover & Select Viral Videos"):
        with gr.Row():
            topic_in = gr.Textbox(label="Search Topic / Niche", value="Islamic Golden Age History")
            license_in = gr.Dropdown(
                choices=["creativeCommon", "whitelist_only"],
                value="creativeCommon",
                label="Shariah IP Filter (License Mode)"
            )
            max_v_in = gr.Slider(1, 15, value=5, step=1, label="Max Candidates to Scan")
            search_btn = gr.Button("🔍 Scan Online Viral Videos", variant="primary")

        results_table = gr.Dataframe(
            headers=["Video ID", "Title", "Channel", "Views", "Velocity (Views/Hr)", "License", "URL"],
            interactive=False,
            label="Discovered Ethical Candidates"
        )
        selected_videos_chk = gr.CheckboxGroup(label="Select Discovered Viral Videos to Clip")
        custom_url_in = gr.Textbox(
            label="Or Paste a Direct Video URL (Optional - e.g., Whitelisted Lecture or CC-BY Link)",
            placeholder="https://www.youtube.com/watch?v=..."
        )

    with gr.Tab("2. Configure Clips, Fiqh Filters & Publish"):
        with gr.Row():
            num_clips_in = gr.Slider(1, 10, value=3, step=1, label="Number of Short Clips per Video")
            min_dur_in = gr.Slider(15, 45, value=25, step=5, label="Min Clip Duration (s)")
            max_dur_in = gr.Slider(30, 90, value=58, step=2, label="Max Clip Duration (s)")
        with gr.Row():
            strip_music_chk = gr.Checkbox(value=True, label="Isolate Vocals & Strip Musical Instruments (Meta Demucs v4)")
            aspect_in = gr.Radio(
                choices=["blurred_pad", "center_crop"],
                value="blurred_pad",
                label="9:16 Vertical Reframing Style"
            )
        with gr.Row():
            pub_yt_chk = gr.Checkbox(value=False, label="Auto-Upload to YouTube Shorts")
            pub_ig_chk = gr.Checkbox(value=False, label="Auto-Upload to Instagram Reels")
            pub_tt_chk = gr.Checkbox(value=False, label="Auto-Upload to TikTok")

        run_btn = gr.Button("🚀 Generate & Publish Halal Shorts", variant="primary")
        gallery_out = gr.Gallery(label="Rendered 9:16 Vertical Shorts (Click to Preview & Download)")
        clip_summary_out = gr.Dataframe(
            headers=["Clip ID", "Hook Title", "Time Span", "Viral Score", "Fiqh Audit Notes", "Published URLs"],
            label="Generated Clips & Distribution Status"
        )

    search_btn.click(
        search_online_videos,
        inputs=[topic_in, license_in, max_v_in],
        outputs=[results_table, selected_videos_chk, cached_videos_state]
    )
    run_btn.click(
        run_gpu_clipping_pipeline,
        inputs=[
            selected_videos_chk, custom_url_in, cached_videos_state, num_clips_in,
            min_dur_in, max_dur_in, strip_music_chk, aspect_in,
            pub_yt_chk, pub_ig_chk, pub_tt_chk
        ],
        outputs=[gallery_out, clip_summary_out]
    )

if __name__ == "__main__":
    demo.launch()