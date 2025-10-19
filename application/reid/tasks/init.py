# Import tasks so Celery sees them as "reid.tasks"
from .pipeline import pipeline_run
from .detect import detect_flank
from .reidentify import reid_sift

__all__ = ["pipeline_run", "detect_flank", "reid_sift"]
