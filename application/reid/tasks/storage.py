# reid/tasks/storage.py
import os
import uuid as _uuid
import logging
import tempfile
from typing import Tuple, Optional

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

log = logging.getLogger(__name__)

# ------------- helpers -------------

def _default_storage_is_s3() -> bool:
    try:
        cls = default_storage.__class__.__name__.lower()
        return "s3" in cls or hasattr(default_storage, "bucket_name")
    except Exception:
        return False

def _use_s3_flag() -> bool:
    return bool(getattr(settings, "AWS_USE_S3", False))

def _get_s3():
    """
    Return (client, bucket) or (None, None) if not configured/available.
    We read creds/region from settings/env via boto3’s standard chain.
    """
    try:
        import boto3
    except Exception:
        return None, None

    bucket = getattr(settings, "AWS_STORAGE_BUCKET_NAME", "") or None
    region = getattr(settings, "AWS_S3_REGION_NAME", None)
    if not (_use_s3_flag() and bucket and region):
        return None, None

    try:
        client = boto3.client("s3", region_name=region)
        return client, bucket
    except Exception:
        return None, None

def use_s3() -> bool:
    """S3 'active' if flag is on and default_storage is S3 or a boto3 client is available."""
    return _use_s3_flag() and (_default_storage_is_s3() or _get_s3()[0] is not None)

def ext_of(*candidates: str, default: str = ".jpg") -> str:
    for c in candidates:
        e = (os.path.splitext(c)[1] or "").lower()
        if e:
            return e
    return default

def media_rel(local_path: str) -> Optional[str]:
    """MEDIA_ROOT-relative path (unix-style) or None if not under MEDIA_ROOT."""
    root = getattr(settings, "MEDIA_ROOT", None)
    if not root:
        return None
    try:
        rel = os.path.relpath(local_path, root).replace("\\", "/")
        return rel if not rel.startswith(".") else None
    except ValueError:
        return None

def _s3_put_exact(key: str, data: bytes, content_type: Optional[str] = None):
    """
    Put object to the EXACT S3 key (no renaming).
    - Uses default_storage’s client if possible; else falls back to our boto3 client.
    """
    # try through default_storage's underlying client first
    try:
        bucket = getattr(default_storage, "bucket_name", None) or getattr(settings, "AWS_STORAGE_BUCKET_NAME", "")
        client = getattr(getattr(getattr(default_storage, "connection", None), "meta", None), "client", None)
        if client and bucket:
            kwargs = {"Bucket": bucket, "Key": key, "Body": data}
            if content_type:
                kwargs["ContentType"] = content_type
            client.put_object(**kwargs)
            return client, bucket
    except Exception:
        pass

    # fallback to our own boto3 client
    client, bucket = _get_s3()
    if not client or not bucket:
        raise RuntimeError("S3 client not available to upload")
    kwargs = {"Bucket": bucket, "Key": key, "Body": data}
    if content_type:
        kwargs["ContentType"] = content_type
    client.put_object(**kwargs)
    return client, bucket

def _s3_head(client, bucket: str, key: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        etag = (head.get("ETag") or "").strip('"')
        ver = head.get("VersionId")
        return etag, ver
    except Exception:
        return None, None

def delete_from_s3(key: str) -> bool:
    """Delete an object from S3 if configured. Returns True if delete likely succeeded."""
    if not use_s3():
        return False
    try:
        bucket = getattr(default_storage, "bucket_name", None) or getattr(settings, "AWS_STORAGE_BUCKET_NAME", "")
        client = getattr(getattr(getattr(default_storage, "connection", None), "meta", None), "client", None)
        if client and bucket:
            client.delete_object(Bucket=bucket, Key=key)
            return True
    except Exception:
        pass
    client, bucket = _get_s3()
    if client and bucket:
        client.delete_object(Bucket=bucket, Key=key)
        return True
    return False

def _is_under(parent: str, child: str) -> bool:
    try:
        parent = os.path.abspath(parent)
        child = os.path.abspath(child)
        return os.path.commonpath([parent]) == os.path.commonpath([parent, child])
    except Exception:
        return False

def _maybe_delete_local_after_s3(local_path: str):
    """
    After uploading a local file to S3, remove it if it lives under MEDIA_ROOT
    to avoid keeping two persistent copies (S3 + local).
    """
    if not _use_s3_flag():
        return
    media_root = getattr(settings, "MEDIA_ROOT", None)
    if not media_root:
        return
    try:
        if os.path.exists(local_path) and _is_under(str(media_root), local_path):
            os.remove(local_path)
            log.info("S3: removed local source copy %s", local_path)
    except Exception as e:
        log.warning("S3: failed to remove local copy %s: %r", local_path, e)

def _uuid_if_cropped_name(original_name: str) -> Optional[str]:
    """
    If original_name looks like '<UUID>_cropped.<ext>' return that UUID as str,
    else None. This lets us enforce the exact physical key for crops.
    """
    base = os.path.splitext(os.path.basename(original_name))[0]  # '<UUID>_cropped'
    if not base.endswith("_cropped"):
        return None
    maybe_uuid = base[:-8]  # strip '_cropped'
    try:
        _ = _uuid.UUID(maybe_uuid)
        return maybe_uuid
    except Exception:
        return None

# ------------- public API -------------

def ensure_in_storage(path_or_key: str) -> str:
    """
    Ensure the source image is persisted in the configured backend and return its key (S3),
    or the original input (local). If we upload from a local file under MEDIA_ROOT to S3,
    delete the local source afterwards to prevent duplication.
    """
    if not use_s3():
        # Not using S3: treat input as-is (local path or MEDIA_ROOT-relative key)
        return path_or_key

    # Already a key? check existence
    if not os.path.isabs(path_or_key):
        if _default_storage_is_s3():
            try:
                if default_storage.exists(path_or_key):
                    return path_or_key
            except Exception as e:
                log.debug("S3 exists(%s) via default_storage failed: %r", path_or_key, e)
        else:
            client, bucket = _get_s3()
            if client:
                try:
                    client.head_object(Bucket=bucket, Key=path_or_key)
                    return path_or_key
                except Exception:
                    pass  # fall through to upload

    # Find local file to upload
    local = None
    if os.path.isabs(path_or_key) and os.path.exists(path_or_key):
        local = path_or_key
    else:
        root = getattr(settings, "MEDIA_ROOT", None)
        if root:
            cand = os.path.join(root, path_or_key)
            if os.path.exists(cand):
                local = cand
    if not local:
        raise FileNotFoundError(f"Original not found locally or in storage: {path_or_key}")

    rel = media_rel(local)
    key = rel if (rel and rel.startswith("uploads/")) else f"uploads/{_uuid.uuid4()}{ext_of(local)}"

    with open(local, "rb") as fh:
        data = fh.read()

    # For sources we can tolerate default_storage.save (renaming is OK here)
    if _default_storage_is_s3():
        default_storage.save(key, ContentFile(data))
    else:
        _s3_put_exact(key, data, content_type=None)

    log.info("S3: uploaded %s -> %s", local, key)
    _maybe_delete_local_after_s3(local)
    return key


def ensure_local(path_or_key: str, *, into_dir: str) -> tuple[str, bool]:
    """
    Return (local_path, is_temp).
    If S3 is active and given a key, download into 'into_dir'.
    """
    if os.path.isabs(path_or_key) and os.path.exists(path_or_key):
        return path_or_key, False

    if use_s3():
        suffix = os.path.splitext(path_or_key)[1] or ".jpg"
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=into_dir)
        os.close(fd)
        try:
            if _default_storage_is_s3():
                with default_storage.open(path_or_key, "rb") as fsrc, open(tmp, "wb") as fdst:
                    fdst.write(fsrc.read())
            else:
                client, bucket = _get_s3()
                if not client:
                    raise RuntimeError("S3 client not available to download")
                obj = client.get_object(Bucket=bucket, Key=path_or_key)
                body = obj["Body"].read()
                with open(tmp, "wb") as fdst:
                    fdst.write(body)
            return tmp, True
        except Exception as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            log.error("S3: download %s failed: %r", path_or_key, e)
            raise FileNotFoundError(f"Could not download '{path_or_key}' from storage")

    # local storage path
    root = getattr(settings, "MEDIA_ROOT", None)
    if not root:
        raise FileNotFoundError("MEDIA_ROOT not configured")
    cand = os.path.join(root, path_or_key)
    if os.path.exists(cand):
        return cand, False
    raise FileNotFoundError(f"Local file not found: {path_or_key}")


def save_crop(local_path: str, *, original_name: str) -> str:
    """
    Persist a crop and return StoredImage UUID string.

    ✳️ If original_name == '<ORIGINAL_UUID>_cropped.<ext>',
       we will physically store the crop at:
         uploads/<ORIGINAL_UUID>_cropped.<ext>
       on BOTH S3 and local.

    Otherwise we fall back to the canonical key 'uploads/<row.object_id>.<ext>'.

    We also record the final physical key in StoredImage.storage_key for reliable retrieval.
    """
    from api.models import StoredImage

    ext = ext_of(original_name, local_path, default=".jpg")
    ct = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    row = StoredImage(
        original_name=original_name,
        file_ext=ext,
        content_type=ct,
        size=os.path.getsize(local_path),
    )

    # Decide the exact physical key
    original_uuid = _uuid_if_cropped_name(original_name)  # None or 'xxxxxxxx-...'
    desired_key = f"uploads/{original_uuid}_cropped{ext}" if original_uuid else None
    key_to_use = desired_key or row.s3_key()  # fallback to canonical 'uploads/<row.object_id>.<ext>'

    # Read data once
    with open(local_path, "rb") as fh:
        data = fh.read()

    if use_s3():
        # If we need exact key, bypass default_storage.save to avoid renaming when FILE_OVERWRITE=False
        if desired_key:
            client, bucket = _s3_put_exact(key_to_use, data, content_type=ct)
            etag, ver = _s3_head(client, bucket, key_to_use) if (client and bucket) else (None, None)
            row.etag, row.version_id = etag, ver
        else:
            # canonical key is fine through default_storage
            if _default_storage_is_s3():
                default_storage.save(key_to_use, ContentFile(data))
                client, bucket = _get_s3()
                if client and bucket:
                    etag, ver = _s3_head(client, bucket, key_to_use)
                    row.etag, row.version_id = etag, ver
            else:
                client, bucket = _get_s3()
                if not client:
                    raise RuntimeError("S3 client not available to upload")
                client.put_object(Bucket=bucket, Key=key_to_use, Body=data, ContentType=ct)
                etag, ver = _s3_head(client, bucket, key_to_use)
                row.etag, row.version_id = etag, ver

        row.storage = "s3"
        if hasattr(row, "storage_key"):
            row.storage_key = key_to_use

    else:
        # Local-only: write to MEDIA_ROOT/<key_to_use>
        abs_path = os.path.join(settings.MEDIA_ROOT, key_to_use)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as fdst:
            fdst.write(data)
        row.local_path = abs_path
        row.storage = "local"
        if hasattr(row, "storage_key"):
            row.storage_key = key_to_use

    row.save()
    return str(row.object_id)


def persisted_ref_for_stored_image(stored_image_id: str) -> Tuple[str, str]:
    """
    Return ("s3", key) or ("local", absolute_path) for any StoredImage PK.
    Prefer the model's recorded storage; fall back to env if missing.
    """
    from api.models import StoredImage

    img = StoredImage.objects.get(pk=stored_image_id)
    key = getattr(img, "storage_key", None) or img.s3_key()

    backend = getattr(img, "storage", None)
    if backend == "s3":
        return "s3", key
    if backend == "local":
        if getattr(img, "local_path", None):
            return "local", img.local_path
        return "local", os.path.join(getattr(settings, "MEDIA_ROOT", ""), key)

    # Fallback (legacy rows without storage set)
    if use_s3():
        return "s3", key
    return "local", os.path.join(getattr(settings, "MEDIA_ROOT", ""), key)


def persisted_ref_for_source(src_ref: str) -> Tuple[str, str]:
    """
    For the source image fed to the pipeline.
    If S3 is active, treat non-absolute refs as keys; else resolve to local abs path.
    """
    if use_s3():
        return "s3", src_ref
    if os.path.isabs(src_ref):
        return "local", src_ref
    return "local", os.path.join(getattr(settings, "MEDIA_ROOT", ""), src_ref)
