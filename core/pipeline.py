# Plug-and-play Stage Registry & PipelineOrchestrator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import tempfile
import shutil
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HalalShortsEngine")


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class ClipCandidate:
    clip_id: str
    source_video_id: str
    start_time: float
    end_time: float
    hook_title: str
    caption_text: str
    virality_score: int
    fiqh_compliance_passed: bool
    fiqh_notes: str
    words: List[WordTiming] = field(default_factory=list)
    rendered_mp4_path: Optional[Path] = None
    published_urls: Dict[str, str] = field(default_factory=dict)


@dataclass
class VideoItem:
    video_id: str
    title: str
    channel_title: str
    url: str
    views: int
    published_at: str
    viral_velocity: float
    license_type: str
    raw_video_path: Optional[Path] = None
    clean_vocal_path: Optional[Path] = None
    scene_boundaries: List[tuple] = field(default_factory=list)
    words: List[WordTiming] = field(default_factory=list)
    candidates: List[ClipCandidate] = field(default_factory=list)


@dataclass
class PipelineContext:
    config: Dict[str, Any]
    run_id: str
    work_dir: Path
    discovered_videos: List[VideoItem] = field(default_factory=list)
    selected_videos: List[VideoItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineStage(ABC):
    """Abstract base class for all modular pipeline components."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        pass


STAGE_REGISTRY: Dict[str, Type[PipelineStage]] = {}


def register_stage(name: str):
    """Decorator to register a stage into the plug-and-play registry."""
    def decorator(cls: Type[PipelineStage]):
        STAGE_REGISTRY[name] = cls
        return cls
    return decorator


class ModularPipelineOrchestrator:
    """Executes registered stages sequentially based on config.yaml."""
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def create_context(self, run_id: str) -> PipelineContext:
        base_tmp = Path(tempfile.mkdtemp(prefix=f"halal_shorts_{run_id}_"))
        return PipelineContext(config=self.config, run_id=run_id, work_dir=base_tmp)

    def run_stages(
            self,
            context: PipelineContext,
            stage_names: Optional[List[str]] = None
    ) -> PipelineContext:
        stages_to_run = stage_names or self.config["pipeline"]["active_stages"]
        for stage_name in stages_to_run:
            if stage_name not in STAGE_REGISTRY:
                raise ValueError(f"Stage '{stage_name}' is not registered.")
            logger.info(f"==> Executing Stage: {stage_name}")
            stage_instance = STAGE_REGISTRY[stage_name](self.config)
            context = stage_instance.execute(context)
        return context

    @staticmethod
    def cleanup(context: PipelineContext):
        if context.work_dir.exists():
            shutil.rmtree(context.work_dir, ignore_errors=True)