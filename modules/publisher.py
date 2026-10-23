# Stage 5: YouTube Shorts, Instagram Reels & TikTok Uploader
import os
import time
import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from core.pipeline import PipelineStage, PipelineContext, ClipCandidate, register_stage, logger


@register_stage("social_publisher")
class SocialPublisherStage(PipelineStage):
    def _upload_youtube_short(self, clip: ClipCandidate) -> str:
        creds = Credentials(
            token=None,
            refresh_token=os.environ["YT_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["YT_CLIENT_ID"],
            client_secret=os.environ["YT_CLIENT_SECRET"],
            scopes=["https://www.googleapis.com/auth/youtube.upload"]
        )
        creds.refresh(Request())
        youtube = build("youtube", "v3", credentials=creds)

        title = clip.hook_title if "#Shorts" in clip.hook_title else f"{clip.hook_title[:85]} #Shorts"
        body = {
            "snippet": {
                "title": title,
                "description": f"{clip.caption_text}\n\nLicense: Creative Commons Attribution #Shorts",
                "tags": ["shorts", "islamic", "ethical", "education"],
                "categoryId": "27"
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False
            }
        }
        media = MediaFileUpload(str(clip.rendered_mp4_path), chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        response = None
        while response is None:
            _, response = request.next_chunk()
        return f"https://youtube.com/shorts/{response['id']}"

    def _upload_instagram_reel(self, clip: ClipCandidate, public_video_url: str) -> str:
        ig_user_id = os.environ["IG_USER_ID"]
        access_token = os.environ["IG_ACCESS_TOKEN"]
        base_url = "https://graph.facebook.com/v23.0"

        create_resp = requests.post(
            f"{base_url}/{ig_user_id}/media",
            data={
                "media_type": "REELS",
                "video_url": public_video_url,
                "caption": clip.caption_text[:2200],
                "share_to_feed": "true",
                "access_token": access_token
            },
            timeout=30
        )
        create_resp.raise_for_status()
        container_id = create_resp.json()["id"]

        for _ in range(30):
            status_resp = requests.get(
                f"{base_url}/{container_id}",
                params={"fields": "status_code,status", "access_token": access_token},
                timeout=20
            )
            status_resp.raise_for_status()
            status_code = status_resp.json().get("status_code")
            if status_code == "FINISHED":
                break
            if status_code == "ERROR":
                raise RuntimeError(f"IG Reel container error: {status_resp.json()}")
            time.sleep(5)
        else:
            raise TimeoutError("Instagram Reel processing timed out.")

        pub_resp = requests.post(
            f"{base_url}/{ig_user_id}/media_publish",
            data={"creation_id": container_id, "access_token": access_token},
            timeout=30
        )
        pub_resp.raise_for_status()
        return f"https://www.instagram.com/reel/{pub_resp.json()['id']}"

    def _upload_tiktok_video(self, clip: ClipCandidate, public_video_url: str) -> str:
        tt_token = os.environ["TIKTOK_ACCESS_TOKEN"]
        headers = {
            "Authorization": f"Bearer {tt_token}",
            "Content-Type": "application/json; charset=UTF-8"
        }
        payload = {
            "post_info": {
                "title": clip.hook_title[:150],
                "privacy_level": "PUBLIC_TO_EVERYONE",
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False
            },
            "source_info": {
                "source": "PULL_FROM_URL",
                "video_url": public_video_url
            }
        }
        init_resp = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/video/init/",
            headers=headers,
            json=payload,
            timeout=30
        )
        init_resp.raise_for_status()
        publish_id = init_resp.json()["data"]["publish_id"]
        return f"tiktok_publish_id:{publish_id}"

    def execute(self, context: PipelineContext) -> PipelineContext:
        pub_cfg = self.config["publishing"]
        staging_base = pub_cfg.get("public_staging_bucket_url", "").rstrip("/")

        for video in context.selected_videos:
            for clip in video.candidates:
                if not clip.rendered_mp4_path or not clip.rendered_mp4_path.exists():
                    continue

                if pub_cfg.get("enable_youtube_shorts") and os.environ.get("YT_REFRESH_TOKEN"):
                    yt_url = self._upload_youtube_short(clip)
                    clip.published_urls["youtube"] = yt_url
                    logger.info(f"Published YouTube Short: {yt_url}")

                public_url = f"{staging_base}/{clip.rendered_mp4_path.name}" if staging_base else ""
                if pub_cfg.get("enable_instagram_reels") and public_url and os.environ.get("IG_ACCESS_TOKEN"):
                    ig_url = self._upload_instagram_reel(clip, public_url)
                    clip.published_urls["instagram"] = ig_url
                    logger.info(f"Published Instagram Reel: {ig_url}")

                if pub_cfg.get("enable_tiktok") and public_url and os.environ.get("TIKTOK_ACCESS_TOKEN"):
                    tt_id = self._upload_tiktok_video(clip, public_url)
                    clip.published_urls["tiktok"] = tt_id
                    logger.info(f"Initiated TikTok Publish: {tt_id}")

        return context