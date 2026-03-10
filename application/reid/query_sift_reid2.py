"""
Query-only SIFT/RootSIFT ReID (reid2 pipeline)
- No CLI, no evaluation helpers.
- Public API: reidentify(crop_path) -> (best_code, probs_dict)
- Reads settings from Django settings.* or environment variables.

Environment / settings supported (same as legacy + reid2 options):
  REID_INDEX_PATH, REID_META_PATH, REID_INDEX_DIM, REID_INDEX_METRIC
  REID_IMG_WIDTH, REID_MAX_KPTS, REID_FLIP_QUERY, REID_USE_CLAHE (on by default)
  REID_TOPK_PER_DESC (aka K_NEIGH), REID_ANNOY_SEARCH_K, REID_SEARCH_K_MULT
  REID_PER_IMAGE_MATCH_CAP, REID_USE_RANK_WEIGHT, REID_PER_ID_NORMALIZE
"""
from __future__ import annotations

import os, json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import logging
log = logging.getLogger(__name__)

import numpy as np
import cv2

# -----------------
# Settings access
# -----------------
try:
    from django.conf import settings as _dj_settings
except Exception:  # pragma: no cover
    _dj_settings = None


def _s(name: str, default: Any = None) -> Any:
    # Prefer Django settings if available; fall back to env; else default
    if _dj_settings is not None and hasattr(_dj_settings, name):
        return getattr(_dj_settings, name)
    return os.environ.get(name, default)

# ---------------
# Annoy binding
# ---------------
try:
    from annoy import AnnoyIndex
except Exception as _e:  # pragma: no cover
    AnnoyIndex = None  # type: ignore

_index: Optional[AnnoyIndex] = None
_records: Optional[List[Dict[str, str]]] = None
_id_images: Optional[Dict[str, List[str]]] = None
_id_image_counts: Optional[Dict[str, int]] = None


def _get_index() -> AnnoyIndex:
    if AnnoyIndex is None:
        raise RuntimeError("annoy is not installed (pip install annoy)")
    global _index
    if _index is not None:
        return _index

    idx_path = _s("REID_INDEX_PATH", None)
    dim = int(_s("REID_INDEX_DIM", 128))
    metric = str(_s("REID_INDEX_METRIC", "euclidean")).lower()
    if not idx_path or not os.path.exists(idx_path):
        raise FileNotFoundError(f"Annoy index not found at {idx_path}")
    if metric not in ("euclidean", "angular", "manhattan", "hamming", "dot"):
        metric = "euclidean"
    idx = AnnoyIndex(dim, metric)
    if not idx.load(idx_path):
        raise OSError(f"Failed to load Annoy index from {idx_path}")
    _index = idx
    return _index


# -----------------
# Meta & records
# -----------------

def _load_raw_meta():
    meta_path = _s("REID_META_PATH", None)
    if not meta_path or not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_records(raw) -> List[Dict[str, str]]:
    """
    Normalise diverse meta formats to a list of dicts:
      records[i] = {"code": <gallery_id>, "image": <path or "">}
    Supports:
      {"meta":[{"giraffe_id"/"gid"/"code","image_path"/"image"/"path"}, ...]}
      {"items":[{"gid","image"}, ...]}
      [{"gid","image"}, ...]
      {"0":{"gid","image"}, "1":{...}} (dense int keys)
      list[str] -> codes only
      dict[int|str] -> str or dict
    """
    recs: List[Dict[str, str]] = []
    if raw is None:
        return recs

    for it in raw["meta"]:
        code = str(it.get("identity"))
        img = str(it.get("image_path"))
        recs.append({"code": code, "image": img})
    return recs


def _ensure_records():
    global _records, _id_images, _id_image_counts
    if _records is not None and _id_images is not None and _id_image_counts is not None:
        return
    raw = _load_raw_meta()
    _records = _normalize_records(raw) if raw is not None else []
    images_by_code: Dict[str, List[str]] = {}
    for r in _records:
        c = r.get("code", "")
        p = r.get("image", "")
        images_by_code.setdefault(c, [])
        if p:
            images_by_code[c].append(p)
    _id_images = images_by_code
    _id_image_counts = {c: max(1, len(v)) for c, v in images_by_code.items()}


# -----------------
# Preprocess & RootSIFT
# -----------------

def _preprocess_color_to_gray(img, width):
    """
    Resize to a fixed width (preserve aspect), convert to grayscale,
    then apply CLAHE (local contrast equalization).
    """
    h, w = img.shape[:2]
    new_w = max(1, int(width))
    new_h = max(1, int(round(h * (new_w / float(w)))))
    # INTER_AREA is robust for downscaling (and acceptable for light upscaling)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE helps SIFT under shadows/bright spots
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _rootsift(desc):
    """
    Convert SIFT to RootSIFT
    """
    if desc is None or desc.size == 0:
        return desc
    eps = 1e-10
    desc = desc.astype(np.float32, copy=False)
    desc /= (np.sum(desc, axis=1, keepdims=True) + eps)
    np.sqrt(desc, out=desc)
    return desc


def _extract_sift_desc(img_path: str, *, img_width: int, max_kpts: int, flip: bool = False) -> np.ndarray:
    raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if raw is None:
        return np.empty((0, 128), dtype=np.float32)
    if flip:
        raw = cv2.flip(raw, 1)
    gray = _preprocess_color_to_gray(raw, width=img_width)
    try:
        sift = cv2.SIFT_create(nfeatures=max_kpts)
    except AttributeError:  # SIFT not available in this OpenCV build
        return np.empty((0, 128), dtype=np.float32)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        return np.empty((0, 128), dtype=np.float32)
    if max_kpts and len(desc) > max_kpts:
        keep_idx = np.argsort([-kp.response for kp in kps])[:max_kpts]
        desc = desc[keep_idx]
    return _rootsift(desc).astype(np.float32)


# -----------------
# Scoring
# -----------------

def _dist_to_weight(d: float) -> float:
    return 1.0 / (1e-6 + float(d))


def _match_query(
    desc: np.ndarray,
    idx: AnnoyIndex,
    records: List[Dict[str, str]],
    id_image_counts: Dict[str, int],
    *,
    k_neigh: int,
    search_k: int,
    per_image_cap: int,
    rank_weight: bool,
    per_id_normalize: bool,
) -> Dict[str, Any]:
    if desc is None or desc.size == 0:
        return {"ranked": [], "ranked_votes": [], "image_votes": {}, "image_scores": {}, "n": 0}

    image_votes = defaultdict(int)    # votes per gallery image
    image_scores = defaultdict(float)
    image_caps = defaultdict(int)     # per-image cap tracker

    # Map image path -> code for quick roll-up
    img_to_code: Dict[str, str] = {}
    for i, r in enumerate(records):
        if r.get("image"):
            img_to_code[r["image"]] = r.get("code", "")

    for d in desc:
        ids, dists = idx.get_nns_by_vector(d.tolist(), k_neigh, search_k=search_k, include_distances=True)
        for rank, (nei_id, dist) in enumerate(zip(ids, dists), start=1):
            if nei_id < 0 or nei_id >= len(records):
                continue
            rec = records[nei_id]
            g_img = rec.get("image", "") or f"{rec.get('code', '') or 'ID'}#{nei_id}"
            if per_image_cap and image_caps[g_img] >= per_image_cap:
                continue
            image_caps[g_img] += 1
            image_votes[g_img] += 1.0
            s = _dist_to_weight(dist) * (1.0 / rank if rank_weight else 1.0)
            image_scores[g_img] += s

    # roll-up: per-image -> per-ID
    id_votes: Dict[str, float] = defaultdict(float)
    id_scores: Dict[str, float] = defaultdict(float)

    for g_img, v in image_votes.items():
        code = img_to_code.get(g_img, g_img.split("#", 1)[0] if "#" in g_img else "")
        id_votes[code] += v
    for g_img, s in image_scores.items():
        code = img_to_code.get(g_img, g_img.split("#", 1)[0] if "#" in g_img else "")
        id_scores[code] += s

    if per_id_normalize:
        for gid in list(id_scores):
            id_scores[gid] /= max(1, id_image_counts.get(gid, 1))

    ranked_scores = sorted(id_scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked_votes  = sorted(id_votes.items(),  key=lambda kv: kv[1], reverse=True)

    return {
        "ranked": ranked_scores,
        "ranked_votes": ranked_votes,
        "image_votes": dict(image_votes),
        "image_scores": dict(image_scores),
        "n": int(len(desc)),
    }


# -----------------
# Public API
# -----------------

def reidentify(crop_path: str) -> Tuple[str, Dict[str, float]]:
    """
    ReID a single crop image. Returns (best_code, probs_dict)
    where probs_dict contains normalised per-ID scores (sums to 1.0).
    """
    idx = _get_index()
    _ensure_records()
    recs = _records or []
    id_image_counts = _id_image_counts or {}

    # Configuration
    img_width = int(_s("REID_IMG_WIDTH", 256))
    max_kpts = int(_s("REID_MAX_KPTS", 85))
    k_neigh = int(_s("REID_TOPK_PER_DESC", 11))
    flip = bool(str(_s("REID_FLIP_QUERY", False)).lower() in ("1", "true", "yes"))
    per_cap = int(_s("REID_PER_IMAGE_MATCH_CAP", 30))
    rank_wt = bool(str(_s("REID_USE_RANK_WEIGHT", True)).lower() in ("1", "true", "yes"))
    per_norm = bool(str(_s("REID_PER_ID_NORMALIZE", False)).lower() in ("1", "true", "yes"))

    # absolute vs multiplier search depth
    mult = float(_s("REID_SEARCH_K_MULT", 32.0))
    search_k = int(k_neigh * mult)
    log.info("=========================================================REID_desc: search_k %s, per_norm %s, rank_wt %s, per_cap %d, k_neigh %d, max_kpts %d, img_width %d", search_k, per_norm, rank_wt, per_cap, k_neigh, max_kpts, img_width)

    # descriptors (with optional horizontal flip)
    desc = _extract_sift_desc(crop_path, img_width=img_width, max_kpts=max_kpts, flip=False)
    #d2 = _extract_sift_desc(crop_path, img_width=img_width, max_kpts=max_kpts, flip=True) if flip else np.empty((0, 128), np.float32)
    #desc = np.vstack([d for d in (d1, d2) if d.size]) if (d1.size or d2.size) else np.empty((0, 128), np.float32)
    log.info("REID_desc: desc with size %s", desc.shape)
    
    if desc.size == 0:
        return "unknown0", {}

    res = _match_query(
        desc,
        idx,
        recs,
        id_image_counts,
        k_neigh=k_neigh,
        search_k=search_k,
        per_image_cap=per_cap,
        rank_weight=rank_wt,
        per_id_normalize=per_norm,
    )

    log.info("===========REID: %s", res)

    if not res["ranked"]:
        return "unknown_not", {}

    best = res["ranked"][0][0] or "unknown_unranked"
    base = dict(res["ranked"])  # ID -> score
    total = float(sum(base.values())) or 1.0
    probs = {k: float(v / total) for k, v in base.items() if k}
    return best, probs
