# reid/query_sift_reid.py
import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from django.conf import settings

try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None


# ---------- global caches ----------
_index = None         # Annoy index
_meta: Optional[object] = None  # list or dict: annoy_id -> giraffe_id (code)


# ---------- helpers: settings ----------
def _s(name: str, default=None):
    """Read from Django settings using getattr(settings, name, default)."""
    return getattr(settings, name, default)


# ---------- index / meta loaders ----------
def _get_index() -> AnnoyIndex:
    """Load Annoy index once; dimension & metric must match how you built it."""
    if AnnoyIndex is None:
        raise RuntimeError("annoy is not installed (pip install annoy)")

    global _index
    if _index is not None:
        return _index

    idx_path = _s("REID_INDEX_PATH", None)
    dim      = int(_s("REID_INDEX_DIM", 128))
    metric   = str(_s("REID_INDEX_METRIC", "euclidean")).lower()

    if not idx_path or not os.path.exists(idx_path):
        raise FileNotFoundError(f"Annoy index not found at {idx_path}")
    if metric not in ("euclidean", "angular", "manhattan", "hamming", "dot"):
        metric = "euclidean"

    idx = AnnoyIndex(dim, metric)
    if not idx.load(idx_path):
        raise OSError(f"Failed to load Annoy index from {idx_path}")

    _index = idx
    return _index


def _get_meta():
    """Load the meta file and normalize it so _id_to_code can read it."""
    global _meta
    if _meta is not None:
        return _meta

    meta_path = getattr(settings, "REID_META_PATH", None)
    if not meta_path or not os.path.exists(meta_path):
        _meta = None
        return _meta

    with open(meta_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # CASE A: {"meta": [...], "config": {...}}
    if isinstance(obj, dict) and "meta" in obj and isinstance(obj["meta"], list):
        _meta = obj["meta"]  # list of dicts OR strings
        return _meta

    # CASE B: list (either list[str] or list[dict])
    if isinstance(obj, list):
        _meta = obj
        return _meta

    # CASE C: dict mapping {"0": "ZURI", "1": "NIA", ...} or {0: "ZURI", ...}
    if isinstance(obj, dict):
        try:
            as_int_keys = {int(k): v for k, v in obj.items()}
            # Normalize to list if contiguous from 0..N-1
            keys = sorted(as_int_keys.keys())
            if keys and keys[0] == 0 and keys[-1] == len(keys) - 1:
                _meta = [as_int_keys[i] for i in range(len(keys))]
            else:
                _meta = as_int_keys
        except Exception:
            _meta = obj
        return _meta

    _meta = None
    return _meta


# ---------- image preprocessing & SIFT ----------
def _preprocess_gray(img: np.ndarray, width: int, use_clahe: bool = True) -> np.ndarray:
    """Resize (keep aspect), convert to gray, optional CLAHE normalization."""
    h, w = img.shape[:2]
    new_w = int(width)
    new_h = max(1, int(round(h * (new_w / float(w)))))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img


def _load_mask_aligned(image_path: str, processed_shape, masks_root: Optional[str]) -> Optional[np.ndarray]:
    """
    Locate a binary mask (PNG) aligned to image_path.
    Priority:
      1) Same folder, same stem + '.png'
      2) If masks_root provided: replace leading 'gallery'/'test' with masks_root and mirror path
    Returns uint8 mask in processed size or None.
    """
    if not masks_root and not os.path.isfile(os.path.splitext(image_path)[0] + ".png"):
        return None

    H, W = processed_shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [os.path.join(os.path.dirname(image_path), stem + ".png")]

    if masks_root:
        parts = os.path.normpath(image_path).split(os.sep)
        anchor = "gallery" if "gallery" in parts else ("test" if "test" in parts else None)
        if anchor:
            rel = os.path.join(*parts[parts.index(anchor) + 1 :])
            candidates.append(os.path.join(masks_root, rel))

    for mp in candidates:
        if os.path.isfile(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            return m
    return None


def _rootsift(desc: np.ndarray) -> np.ndarray:
    if desc is None or desc.size == 0:
        return desc
    desc = desc.astype(np.float32)
    l1 = np.maximum(desc.sum(axis=1, keepdims=True), 1e-12)
    return np.sqrt(desc / l1)


def _extract_sift_desc(
    img_path: str,
    *,
    img_width: int,
    max_kpts: int,
    masks_root: Optional[str],
    flip: bool = False
) -> np.ndarray:
    """
    Extract RootSIFT descriptors from (optionally flipped) preprocessed image.
    Keeps top-N keypoints by response for stability.
    """
    read_flag = cv2.IMREAD_COLOR
    raw = cv2.imread(img_path, read_flag)
    if raw is None:
        return np.empty((0, 128), dtype=np.float32)

    # Prepare view (flip if requested)
    if flip:
        raw = cv2.flip(raw, 1)

    img = _preprocess_gray(raw, width=img_width, use_clahe=bool(_s("REID_USE_CLAHE", True)))

    # Optional binary mask
    mask = _load_mask_aligned(img_path, img.shape, _s("REID_MASKS_ROOT", None))
    if flip and mask is not None:
        mask = cv2.flip(mask, 1)

    # SIFT (requires opencv-contrib-python)
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        return np.empty((0, 128), dtype=np.float32)

    kps, desc = sift.detectAndCompute(img, mask)
    if desc is None or len(desc) == 0:
        return np.empty((0, 128), dtype=np.float32)

    # Keep strongest keypoints
    order = np.argsort([-kp.response for kp in kps])[: max_kpts]
    desc = desc[order]

    # RootSIFT
    desc = _rootsift(desc).astype(np.float32)
    return desc


# ---------- voting ----------
def _dist_to_weight(dist: float) -> float:
    # Convert Annoy distance to positive vote weight (smaller dist -> larger weight)
    return 1.0 / (1e-6 + float(dist))


def _id_to_code(nei_id: int) -> str:
    """Map Annoy item id -> your animal code, robust to various meta formats."""
    meta = _get_meta()
    if meta is None:
        return str(nei_id)

    # list of strings
    if isinstance(meta, list) and (not meta or isinstance(meta[0], str)):
        return str(meta[nei_id]) if 0 <= nei_id < len(meta) else str(nei_id)

    # list of dicts: pick common code fields
    if isinstance(meta, list) and meta and isinstance(meta[0], dict):
        rec = meta[nei_id] if 0 <= nei_id < len(meta) else {}
        for key in ("giraffe_id", "code", "animal_code", "id", "name"):
            if key in rec and rec[key]:
                return str(rec[key])
        return str(nei_id)

    # dict mapping index -> string or dict
    if isinstance(meta, dict):
        rec = meta.get(nei_id, None)
        if rec is None:
            return str(nei_id)
        if isinstance(rec, str):
            return str(rec)
        if isinstance(rec, dict):
            for key in ("giraffe_id", "code", "animal_code", "id", "name"):
                if key in rec and rec[key]:
                    return str(rec[key])
        return str(nei_id)

    return str(nei_id)


# ---------- public API ----------
def reidentify(crop_path: str) -> Tuple[str, Dict[str, float]]:
    """
    Match a crop against the descriptor-level Annoy index.
    Returns (best_code, votes_dict) where codes are your giraffe IDs from the gallery.
    - votes_dict keys: codes; values: normalized vote weights (sum to 1).
    """
    idx = _get_index()

    # ---- config from Django settings (via getattr) ----
    topk_per_desc: int = int(_s("REID_TOPK_PER_DESC", 5))
    # Backward-compat: allow either MAX_KPTS (preferred) or MAX_DESCRIPTORS
    max_kpts: int = int(_s("REID_MAX_KPTS", _s("REID_MAX_DESCRIPTORS", 150)))
    img_width: int = int(_s("REID_IMG_WIDTH", 256))
    flip_query: bool = bool(_s("REID_FLIP_QUERY", False))
    masks_root: Optional[str] = _s("REID_MASKS_ROOT", None)
    # Optional Annoy search depth (None = library default)
    annoy_search_k = _s("REID_ANNOY_SEARCH_K", None)
    if annoy_search_k is not None:
        try:
            annoy_search_k = int(annoy_search_k)
        except Exception:
            annoy_search_k = None

    # ---- descriptor extraction (original + optional flipped) ----
    desc1 = _extract_sift_desc(crop_path, img_width=img_width, max_kpts=max_kpts, masks_root=masks_root, flip=False)
    desc2 = _extract_sift_desc(crop_path, img_width=img_width, max_kpts=max_kpts, masks_root=masks_root, flip=True) if flip_query else np.empty((0, 128), dtype=np.float32)

    desc = np.vstack([d for d in (desc1, desc2) if d.size]) if (desc1.size or desc2.size) else np.empty((0, 128), dtype=np.float32)
    if desc.size == 0:
        return "unknown", {}

    # ---- voting over nearest neighbours ----
    votes: Dict[str, float] = {}

    if annoy_search_k is None:
        nn = lambda v: idx.get_nns_by_vector(v, topk_per_desc, include_distances=True)
    else:
        nn = lambda v: idx.get_nns_by_vector(v, topk_per_desc, search_k=annoy_search_k, include_distances=True)

    for d in desc:
        ids, dists = nn(d.tolist())
        for nei_id, dist in zip(ids, dists):
            code = _id_to_code(int(nei_id))
            votes[code] = votes.get(code, 0.0) + _dist_to_weight(dist)

    # Normalize to pseudo-probabilities
    total = float(sum(votes.values())) or 1.0
    for k in list(votes.keys()):
        votes[k] = float(votes[k] / total)

    # Pick best
    best_code = max(votes.items(), key=lambda kv: kv[1])[0] if votes else "unknown"
    return best_code, votes
