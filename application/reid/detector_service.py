import os, logging
from functools import lru_cache
from django.conf import settings

log = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_detector():
    from .detector import build_model  #existing builder
    ckpt = getattr(settings, "DETECTOR_CHECKPOINT", None)
    if not ckpt:
        raise FileNotFoundError("DETECTOR_CHECKPOINT is not set in settings")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(getattr(settings, "BASE_DIR", ""), ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Detector checkpoint not found: {ckpt}")
    device = str(getattr(settings, "DETECTOR_DEVICE", "cpu")).strip() or "cpu"
    log.info("Detector: loading %s on %s", ckpt, device)
    return build_model(checkpoint_path=ckpt, device=device)
