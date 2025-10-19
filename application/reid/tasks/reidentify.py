# reid/tasks/reidentify.py
from __future__ import annotations

import os
import shutil
import logging
from typing import Dict, Any, Tuple

from celery import shared_task
from django.conf import settings
from django.db import transaction

from api.models import ReIDResult, Animal, StoredImage
from .storage import ensure_local  # downloads from S3 or resolves local

log = logging.getLogger(__name__)

# -------------------------------
# Choose ReID engine (new vs old)
# -------------------------------
_HAVE_REID2 = False
_reid2 = None
_reid1 = None

try:
    # New engine (reid2): image-vote → ID rollup, rank weighting, per-ID normalization, caps, search_k mult
    from ..query_sift_reid2 import reidentify as _reid2  # type: ignore
    _HAVE_REID2 = True
except Exception:
    pass

try:
    # Legacy engine (reid1): straight distance-weight voting to IDs
    from ..query_sift_reid import reidentify as _reid1  # type: ignore
except Exception:
    _reid1 = None


def _choose_reid_impl():
    """
    Decide which implementation to use.
    Env override:
      REID_IMPLEMENTATION = "reid2" | "reid1" | "legacy" | "old"
    Default: use reid2 when available; else fallback to reid1.
    """
    force = str(getattr(settings, "REID_IMPLEMENTATION", "") or "").lower()
    if force in {"reid2", "v2", "new"} and _HAVE_REID2:
        return _reid2, "reid2"
    if force in {"reid1", "legacy", "old"} and _reid1 is not None:
        return _reid1, "reid1"
    if _HAVE_REID2:
        return _reid2, "reid2"
    if _reid1 is not None:
        return _reid1, "reid1"
    raise ImportError("No ReID implementation available: neither query_sift_reid2 nor query_sift_reid found.")


def _cleanup_dir(path: str | None):
    if not path:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
        log.info("REID: cleaned pipeline dir %s", path)
    except Exception as e:
        log.warning("REID: failed to cleanup %s: %s", path, e)


def _extract_ids_and_paths(payload: dict) -> tuple[str | None, str | None]:
    """
    Be tolerant to different payload shapes.
    Returns (stored_crop_id, local_crop_path).
    """
    crop = payload.get("crop") or {}
    stored_crop_id = (
        payload.get("stored_crop_id")
        or payload.get("stored_id")
        or payload.get("crop_id")
        or crop.get("stored_id")
        or crop.get("stored_crop_id")
    )
    local_crop_path = (
        payload.get("local_crop_path")
        or payload.get("local_path")
        or crop.get("local_path")
    )
    return stored_crop_id, local_crop_path


@shared_task(bind=True, max_retries=5, retry_backoff=10, retry_jitter=True, soft_time_limit=300)
def reid_sift(self, payload: dict, *, rec_id: int, temp_dir: str | None = None):
    """
    Re-identification task.
      1) Honor detect result "no_detection".
      2) Resolve stored crop → local file.
      3) Run chosen ReID implementation (reid2 preferred, legacy fallback).
      4) Persist result.
      5) Clean up temp dir.
    Optional setting:
      REID_IMPLEMENTATION = "reid2" | "reid1" | "legacy" | "old"
    """
    try:
        # 1) Early exit if detection found nothing
        if payload.get("status") == "no_detection":
            ReIDResult.objects.filter(pk=rec_id).update(
                status="no_detection",
                votes_json={"reason": "no_detection", **(payload.get("detect") or {})}
            )
            return {"status": "no_detection"}

        # 2) IDs / paths
        stored_crop_id, local_crop_path = _extract_ids_and_paths(payload)
        if not stored_crop_id:
            raise KeyError(
                "stored_crop_id (or crop.stored_id) is missing from payload; "
                f"payload keys: {list(payload.keys())}"
            )

        # Prefer handed-off local path; else download
        if local_crop_path and os.path.exists(local_crop_path):
            local = local_crop_path
            log.info("REID: using handed-off local crop %s", local)
        else:
            img = StoredImage.objects.get(pk=stored_crop_id)
            key = img.s3_key()  # 'uploads/<uuid>.<ext>'
            local, _ = ensure_local(key, into_dir=temp_dir)
            log.info("REID: downloaded crop into %s", local)

        # 3) Run ReID (new preferred)
        impl, impl_name = _choose_reid_impl()
        best_code, votes = impl(local)
        log.info("REID: impl=%s best_code=%s", impl_name, best_code)

        # 4) Persist
        from django.db import transaction
        with transaction.atomic():
            rec = ReIDResult.objects.select_for_update().get(pk=rec_id)
            try:
                animal = Animal.objects.get(code=best_code)
                rec.predicted_animal = animal
                rec.votes_json = votes
            except Animal.DoesNotExist:
                data = rec.votes_json or {}
                data.update({"guessed_code": best_code, "votes": votes})
                rec.votes_json = data
            rec.status = "completed"
            rec.save()

        return {"status": "completed", "best_code": best_code, "votes": votes}

    except FileNotFoundError as exc:
        raise self.retry(exc=exc)
    except Exception as exc:
        if (self.request.retries + 1) >= self.max_retries:
            ReIDResult.objects.filter(pk=rec_id).update(status="error")
        raise self.retry(exc=exc)
    finally:
        _cleanup_dir(temp_dir)
