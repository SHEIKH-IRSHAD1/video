# Stage 1: CC-BY / Whitelist Viral Video Discovery & yt-dlp
import os
from datetime import datetime, timezone
import isodate
from googleapiclient.discovery import build
import yt_dlp
from core.pipeline import PipelineStage, PipelineContext, VideoItem, register_stage, logger


@register_stage("ethical_discovery")
class EthicalDiscoveryStage(PipelineStage):
    def search_viral_candidates(self, query: str, max_results: int = 10) -> list[VideoItem]:
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            raise EnvironmentError("YOUTUBE_API_KEY environment variable is missing.")

        youtube = build("youtube", "v3", developerKey=api_key)
        disc_cfg = self.config["discovery"]

        search_params = {
            "q": query,
            "part": "id,snippet",
            "type": "video",
            "order": "viewCount",
            "maxResults": min(50, max_results * 3),
            "videoDefinition": "high",
        }
        if disc_cfg.get("license_mode") == "creativeCommon":
            search_params["videoLicense"] = "creativeCommon"

        search_res = youtube.search().list(**search_params).execute()
        video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]
        if not video_ids:
            return []

        details_res = youtube.videos().list(
            id=",".join(video_ids),
            part="snippet,contentDetails,statistics,status"
        ).execute()

        now = datetime.now(timezone.utc)
        items: list[VideoItem] = []

        for v in details_res.get("items", []):
            duration_sec = isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()
            if not (disc_cfg["min_duration_seconds"] <= duration_sec <= disc_cfg["max_duration_seconds"]):
                continue

            license_str = v["status"].get("license", "youtube")
            channel_id = v["snippet"]["channelId"]
            if disc_cfg["license_mode"] == "whitelist_only" and channel_id not in disc_cfg["whitelisted_channel_ids"]:
                continue

            pub_dt = datetime.fromisoformat(v["snippet"]["publishedAt"].replace("Z", "+00:00"))
            hours_alive = max(1.0, (now - pub_dt).total_seconds() / 3600.0)
            views = int(v["statistics"].get("viewCount", 0))
            velocity = round(views / hours_alive, 2)

            items.append(VideoItem(
                video_id=v["id"],
                title=v["snippet"]["title"],
                channel_title=v["snippet"]["channelTitle"],
                url=f"https://www.youtube.com/watch?v={v['id']}",
                views=views,
                published_at=v["snippet"]["publishedAt"],
                viral_velocity=velocity,
                license_type=license_str
            ))

        items.sort(key=lambda x: x.viral_velocity, reverse=True)
        return items[:max_results]

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.selected_videos:
            topic = self.config["discovery"]["default_topic"]
            max_v = self.config["discovery"]["max_source_videos"]
            context.discovered_videos = self.search_viral_candidates(topic, max_results=max_v)
            context.selected_videos = context.discovered_videos[:1]

        for video in context.selected_videos:
            out_tmpl = str(context.work_dir / f"{video.video_id}_raw.%(ext)s")
            ydl_opts = {
                "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
                "outtmpl": out_tmpl,
                "merge_output_format": "mp4",
                "quiet": True,
                "no_warnings": True,
            }
            logger.info(f"Downloading video {video.video_id}: {video.title}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video.url])
            video.raw_video_path = context.work_dir / f"{video.video_id}_raw.mp4"

        return context