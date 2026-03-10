# reid/tasks/detect.py
from __future__ import annotations

import os
import tempfile
import logging
from typing import Dict, Any, Tuple
from functools import lru_cache

from celery import shared_task
from django.conf import settings
from api.models import ReIDResult

# storage helpers (same package level)
from .storage import ensure_local, save_crop, ext_of

log = logging.getLogger(__name__)

# ---------- preprocessing to match test_mob.py ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _score_thresh() -> float:
    # default 0.5 (like CLI) — make configurable
    return float(getattr(settings, "DETECT_SCORE_THRESH", 0.5))

def _use_grayscale() -> bool:
    # CLI run used --grayscale; default True here to match
    return bool(getattr(settings, "DETECT_GRAYSCALE", True))

def _warmup_runs() -> int:
    # optional micro-warmup like --warmup (0 disables)
    return int(getattr(settings, "DETECT_WARMUP_ITERS", 0))

# ---------- model bootstrap  ----------
@lru_cache(maxsize=1)
def _get_detector():
    """
    Build/load the detector once. Reuses project's builder so the
    architecture matches the checkpoint exactly.
    """
    from ..detector import build_model  #existing factory
    ckpt = getattr(settings, "DETECTOR_CHECKPOINT", None)
    if not ckpt:
        raise FileNotFoundError("DETECTOR_CHECKPOINT is not set in settings")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(getattr(settings, "BASE_DIR", ""), ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Detector checkpoint not found: {ckpt}")
    device = str(getattr(settings, "DETECTOR_DEVICE", "cpu")).strip() or "cpu"
    model = build_model(checkpoint_path=ckpt, device=device)
    log.info("DETECT: model ready on %s from %s", device, ckpt)
    return model

# ---------- core detection ----------
def _prepare_tensor(local_path: str, grayscale: bool):
    import torchvision.transforms.functional as VF
    from PIL import Image
    img = Image.open(local_path).convert("RGB")
    if grayscale:
        # triplicate grayscale into 3 channels to match CLI
        img = img.convert("L")
        img = Image.merge("RGB", (img, img, img))
    t = VF.to_tensor(img)                       # [0,1]
    t = VF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t

def _infer_boxes(model, tensor, thresh: float) -> Tuple[list, list]:
    """
    Returns (boxes_xyxy, scores). Chooses the single best box (argmax)
    if it meets the threshold — same as test_mob.py behavior.
    """
    import torch
    device = next(model.parameters()).device
    tensor = tensor.unsqueeze(0).to(device)

    warms = max(0, _warmup_runs())
    if warms:
        model.eval()
        with torch.no_grad():
            for _ in range(warms):
                _ = model(tensor)

    model.eval()
    with torch.no_grad():
        preds = model(tensor)

    if not preds or len(preds) == 0:
        return [], []

    pred = preds[0]
    if ("boxes" not in pred) or ("scores" not in pred):
        log.warning("DETECT: prediction missing boxes/scores keys: %s", list(pred.keys()))
        return [], []

    boxes  = pred["boxes"].detach().float().cpu()
    scores = pred["scores"].detach().float().cpu()

    # log top scores (very helpful for threshold tuning)
    if scores.numel():
        k = min(5, scores.numel())
        topk_vals, _ = scores.topk(k=k)
        log.info("DETECT: top% d scores (thresh=%.3f): %s",
                 k, thresh, [round(float(v), 3) for v in topk_vals.tolist()])

    if scores.numel() == 0 or float(scores.max().item()) < thresh:
        return [], scores.tolist()

    best_idx = int(scores.argmax().item())
    xyxy = [int(v) for v in boxes[best_idx].tolist()]
    return [xyxy], scores.tolist()

def _write_crop(local_image_path: str, box_xyxy: list, into_dir: str) -> str:
    """
    Save the cropped region under into_dir, preserving extension when possible.
    """
    from PIL import Image
    os.makedirs(into_dir, exist_ok=True)
    x1, y1, x2, y2 = box_xyxy
    img = Image.open(local_image_path).convert("RGB")
    W, H = img.size
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop: {(x1,y1,x2,y2)} for image size {(W,H)}")

    crop = img.crop((x1, y1, x2, y2)).convert("RGB")
    suffix = ext_of(local_image_path) or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=into_dir)
    os.close(fd)

    if suffix.lower() in (".jpg", ".jpeg"):
        crop.save(tmp_path, "JPEG", quality=95)
    elif suffix.lower() == ".png":
        crop.save(tmp_path, "PNG")
    else:
        crop.save(tmp_path, "JPEG", quality=95)

    log.info("DETECT: wrote crop to %s", tmp_path)
    return tmp_path

class ObjectDetection:
    """
    Mirrors the CLI pipeline:
      • grayscale triplication (configurable)
      • ImageNet normalization
      • pick top-score box ≥ threshold
      • save crop (S3/local via storage helpers)
      • return a state dict for re-ID
    """

    def run(self, *, rec_id: int, src_ref: str, temp_dir: str, image_key_or_path: str) -> Dict[str, Any]:
        """
        :param rec_id: ReIDResult PK
        :param src_ref: storage key OR absolute local path (pipeline resolved/persisted source)
        :param temp_dir: pipeline-run directory to keep intermediates (same dir will be passed to re-ID)
        :param image_key_or_path: original reference (for naming/logging only)
        """
        ReIDResult.objects.filter(pk=rec_id).update(status="detecting")

        # always resolve to a local file inside temp_dir
        local_path, _ = ensure_local(src_ref, into_dir=temp_dir)
        log.info("DETECT: source -> %s", local_path)

        # build input tensor (match CLI)
        t = _prepare_tensor(local_path, grayscale=_use_grayscale())

        # run model
        model = _get_detector()
        thresh = _score_thresh()
        boxes, scores = _infer_boxes(model, t, thresh)

        if not boxes:
            ReIDResult.objects.filter(pk=rec_id).update(
                status="no_detection",
                votes_json={"reason": "no_detection", "threshold": thresh, "top_scores": scores[:5] if isinstance(scores, list) else None},
            )
            return {"status": "no_detection"}

        # take best box and write crop
        crop_local = _write_crop(local_path, boxes[0], into_dir=temp_dir)

        # persist crop (S3/local) — keep a friendly name for traceability
        stem = os.path.splitext(os.path.basename(image_key_or_path))[0]
        crop_name = f"{stem}_cropped{ext_of(crop_local) or '.jpg'}"
        stored_crop_id = save_crop(crop_local, original_name=crop_name)

        # return state for chained re-ID task
        state = {
            "status": "ok",
            "crop": {
                "stored_id": stored_crop_id,
                "local_path": crop_local,
                "name": crop_name,
            },
            "detect": {
                "threshold": thresh,
                "top_scores": scores[:5] if isinstance(scores, list) else None,
                "box": boxes[0],
            },
        }
        log.info("DETECT: crop stored_id=%s name=%s", stored_crop_id, crop_name)
        return state

# ---------- Celery task wrapper ----------
@shared_task(bind=True, max_retries=5, retry_backoff=10, retry_jitter=True, soft_time_limit=300)
def detect_flank(self, payload: dict):
    """
    Called from pipeline: detect_flank.s(payload) | reid_sift.s(rec_id=..., temp_dir=...)
    """
    try:
        return ObjectDetection().run(**payload)
    except FileNotFoundError as exc:
        # transient (e.g., storage race); retry
        raise self.retry(exc=exc)
    except Exception as exc:
        # bubble other errors to Celery with retry semantics
        raise self.retry(exc=exc)
