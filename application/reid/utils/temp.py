# util/temp.py
import os
import uuid
import shutil
import tempfile
import logging
from typing import Optional
from django.conf import settings

log = logging.getLogger(__name__)

def _norm(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _is_under(parent: str, child: str) -> bool:
    parent = _norm(parent)
    child  = _norm(child)
    try:
        return os.path.commonpath([parent]) == os.path.commonpath([parent, child])
    except Exception:
        return False

def safe_root() -> str:
    """
    A safe temp root OUTSIDE MEDIA_ROOT.
    Uses settings.REID_TEMP_DIR if provided and not under MEDIA_ROOT,
    otherwise falls back to system temp (/tmp on *nix).
    """
    desired = getattr(settings, "REID_TEMP_DIR", None)
    media_root = getattr(settings, "MEDIA_ROOT", None)

    if desired:
        desired = _norm(str(desired).strip())
        if media_root and _is_under(media_root, desired):
            log.warning("TEMP: REID_TEMP_DIR=%s is under MEDIA_ROOT; using system temp instead.", desired)
            desired = None

    if not desired:
        desired = _norm(os.path.join(tempfile.gettempdir(), "reid_processing"))

    os.makedirs(desired, exist_ok=True)
    return desired

def make_run_dir(prefix: str, run_id: Optional[str] = None) -> str:
    """
    Create a unique temp working directory like: <safe_root>/<prefix>_<uuid>
    """
    root = safe_root()
    ident = (run_id or uuid.uuid4().hex).replace(os.sep, "_")
    path = os.path.join(root, f"{prefix}_{ident}")
    os.makedirs(path, exist_ok=True)
    log.info("TEMP: created %s", path)
    return path

def cleanup_tree(path: Optional[str]) -> None:
    """
    Recursively remove a directory tree. Ignores errors.
    """
    if not path:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
        log.info("TEMP: cleaned %s", path)
    except Exception as e:
        log.warning("TEMP: cleanup failed for %s: %s", path, e)

def make_temp_file(dir_path: Optional[str] = None, suffix: str = "") -> str:
    """
    Create a temp file inside dir_path (or safe_root if None) and return its path.
    Caller is responsible for deleting it.
    """
    base = dir_path or safe_root()
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir=base)
    os.close(fd)
    return tmp

__all__ = ["safe_root", "make_run_dir", "cleanup_tree", "make_temp_file"]
