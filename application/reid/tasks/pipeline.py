import logging
import time
from celery import shared_task, chain

from ..utils.temp import make_run_dir, cleanup_tree, make_temp_file  # make_temp_file kept for consistency
from .storage import ensure_in_storage, persisted_ref_for_source, persisted_ref_for_stored_image
from .detect import detect_flank
from .reidentify import reid_sift

log = logging.getLogger(__name__)


# Terminal task: logs total elapsed time for the whole chain.
@shared_task(bind=True)
def metrics_log(self, last_result, *, rec_id: int, pipe_start_ms: int, pipeline_job_id: str | None):
    end_ms = int(time.monotonic() * 1000)
    total_ms = end_ms - pipe_start_ms

    status = None
    if isinstance(last_result, dict):
        status = last_result.get("status") or last_result.get("state") or last_result.get("result")

    log.info(
        "PIPE: chain complete rec_id=%s pipeline_job_id=%s total_ms=%s status=%s",
        rec_id, pipeline_job_id, total_ms, status
    )
    return last_result


class PipelineService:
    def run(self, *, rec_id: int, image_key_or_path: str, request_id: str | None):
        from api.models import ReIDResult

        pipe_dir = make_run_dir("pipe", request_id)
        try:
            # Start timing as soon as we build the chain (captures queue + execution time)
            pipe_start_ms = int(time.monotonic() * 1000)

            # Ensure source is persisted (S3 if enabled), then run detect → reid
            src_ref = ensure_in_storage(image_key_or_path)

            # Build detect payload (returns detect_result)
            detect_payload = dict(
                rec_id=rec_id,
                src_ref=src_ref,
                temp_dir=pipe_dir,
                image_key_or_path=image_key_or_path,
            )

            # Chain: detect → reid → metrics logger
            c = chain(
                detect_flank.s(detect_payload),
                reid_sift.s(rec_id=rec_id, temp_dir=pipe_dir),  # kwargs appended after detect_result
                metrics_log.s(rec_id=rec_id, pipe_start_ms=pipe_start_ms, pipeline_job_id=request_id),
            )
            async_res = c.apply_async()

            # Immediate trace
            log.info(
                "PIPE: queued rec_id=%s pipeline_job_id=%s chain_last_id=%s",
                rec_id, request_id, getattr(async_res, "id", None)
            )

            # Record persisted refs immediately for traceability
            sw, sr = persisted_ref_for_source(src_ref)
            try:
                rec = ReIDResult.objects.get(pk=rec_id)
                data = rec.votes_json or {}
                data.setdefault("persist", {})
                data["persist"]["source"] = {"where": sw, "ref": sr}
                rec.votes_json = data
                rec.save(update_fields=["votes_json"])
            except Exception:
                log.debug("PIPE: persist record failed", exc_info=True)

        except Exception:
            # If chain didn't start, clean temp now; otherwise reid will cleanup
            cleanup_tree(pipe_dir)
            raise


@shared_task(bind=True, max_retries=3, retry_backoff=10, retry_jitter=True, soft_time_limit=60)
def pipeline_run(self, rec_id: int, image_key_or_path: str):
    try:
        req_id = getattr(getattr(self, "request", None), "id", None)
        log.info("PIPE: pipeline_run received rec_id=%s pipeline_job_id=%s", rec_id, req_id)
        return PipelineService().run(rec_id=rec_id, image_key_or_path=image_key_or_path, request_id=req_id)
    except Exception as exc:
        log.exception("PIPE: pipeline_run failed rec_id=%s", rec_id)
        raise self.retry(exc=exc)
