#!/usr/bin/env python3
"""
SIFT / RootSIFT Re-identification Evaluator (Annoy-backed)
— with Top-1/Top-2 metrics, confusion matrices, and per-query CSVs.

What this script does
---------------------
1) Loads a prebuilt Annoy gallery index (built from SIFT/RootSIFT descriptors)
   and its metadata mapping each descriptor back to (gallery_image, identity).
2) For each query image:
   • Preprocess (resize + CLAHE), extract (Root)SIFT descriptors
   • For each descriptor, retrieve k nearest gallery descriptors (Annoy)
   • Aggregate per-gallery-image votes/scores with caps to avoid domination
   • Aggregate per-identity votes/scores (optional per-ID normalisation)
   • Produce ranked identities (by score and by raw votes)
3) Computes:
   • Top-1 and Top-2 accuracy over the query set
   • Confusion matrix (counts + row-normalised %)
   • Per-query vote breakdowns (by ID and by gallery image)
   • Per-image timing (feature time, total time, descriptors used)
4) Writes all results to CSV files.

Inputs you must provide
-----------------------
• --index_path      : Annoy index (.ann) built from the gallery
• --meta_path       : JSON meta that maps Annoy item ids → {image, gid}
• EITHER:
    --test_dir      : directory of query images (GT inferred from parent folder)
  OR
    --query_image   : single query image (GT from its parent folder)
• --save_confmat    : output CSV path (base name for all result CSVs)

Key behaviors / assumptions
---------------------------
• Descriptor mode must match the gallery build (SIFT vs RootSIFT).
• Annoy distance metric is Euclidean; RootSIFT pairs well with it.
• Per-image caps (per_image_match_cap) prevent any single gallery image
  from contributing unlimited votes for one query.
• Per-ID score normalisation (on scores, not votes) divides identity scores
  by the number of distinct gallery images for that ID, to reduce bias.

Quick start
-----------
# Evaluate a full directory
python query_sift_reid.py eval \
  --test_dir data/test \
  --index_path out/sift_gallery.ann \
  --meta_path  out/sift_meta.json \
  --save_confmat out/confusion_counts.csv \
  --descriptor rootsift --img_width 256 --max_kpts 150 \
  --k_neigh 7 --search_k_mult 20 --per_image_match_cap 30

# Evaluate a single image
python query_sift_reid.py eval \
  --query_image data/test/NIA/img1.jpg \
  --index_path out/sift_gallery.ann \
  --meta_path  out/sift_meta.json \
  --save_confmat out/single_confusion.csv \
  --descriptor rootsift --img_width 256 --max_kpts 150 \
  --k_neigh 7 --search_k_mult 20 --per_image_match_cap 30

Outputs
-------
<base>_confusion_counts.csv                 # counts (rows=GT, cols=Pred)
<base>_confusion_counts_rowpct.csv          # row-normalised % 0..100
<base>_votes_per_query.csv                  # raw ID votes per query
<base>_votes_per_query_by_image.csv         # raw gallery-image votes per query
<base>_per_query_predictions.csv            # GT, winners, and per-ID votes
<base>_times.csv                            # per-query timing + descriptor count

Requirements
------------
pip install opencv-contrib-python annoy numpy
"""

import os, sys, time, json, csv, argparse, glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from annoy import AnnoyIndex

# ------------------- Defaults -------------------
IMG_WIDTH = 256
MAX_KPTS  = 150
K_NEIGH   = 7
SEARCH_K_MULT = 20
PER_IMAGE_MATCH_CAP = 30
PER_ID_NORMALIZE = True  # apply normalisation to scores (not raw votes)


# ------------------- SIFT / RootSIFT extractors -------------------
def _create_sift(max_kpts: int = MAX_KPTS):
    """Create a SIFT extractor (compat with OpenCV contrib variants)."""
    try:
        return cv2.SIFT_create(nfeatures=max_kpts)
    except Exception:
        return cv2.xfeatures2d.SIFT_create(nfeatures=max_kpts)

def _rootsift(des, eps=1e-12):
    """RootSIFT: L1-normalize rows then sqrt (works well with Euclidean)."""
    if des is None or len(des) == 0:
        return des
    des = des.astype(np.float32)
    des /= (des.sum(axis=1, keepdims=True) + eps)
    return np.sqrt(des)

def extract_descriptors(img_bgr: np.ndarray, mode: str = "rootsift",
                        max_kpts: int = MAX_KPTS) -> Tuple[Optional[np.ndarray], int]:
    """Return (descriptors, n_keypoints_detected)."""
    if img_bgr is None:
        return None, 0
    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = _create_sift(max_kpts=max_kpts)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(des) == 0:
        return None, 0
    if mode.lower() == "rootsift":
        des = _rootsift(des)
    elif mode.lower() == "sift":
        des = des.astype(np.float32)
    else:
        raise ValueError(f"Unsupported descriptor mode: {mode}")
    if max_kpts and len(des) > max_kpts:
        des = des[:max_kpts]
    return des, len(kps)


# ------------------- Preprocessing -------------------
def preprocess_image(path: str, width: int = IMG_WIDTH) -> Optional[np.ndarray]:
    """Read BGR image, resize (preserve aspect), and apply CLAHE (on V channel)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if width and w != width:
        new_w = int(width)
        new_h = int(round(h * (new_w / float(w))))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # CLAHE on V to handle harsh lighting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ------------------- Meta / index helpers -------------------
def load_meta(meta_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load meta mapping Annoy item id → {'image': str, 'gid': str}.
    Supports multiple formats:
      A) {"items":[{"image":"...","gid":"..."},...], "index":{...}}
      B) {"0":{"image":"...","gid":"..."}, "1":{...}}
      C) [{"image":"...","gid":"..."}, ...]
      D) {"meta":[{"giraffe_id":"...","image_path":"...", ...}], "config": {...}}  (from build_sift_index.py)
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mapping = {}

    # D) build_sift_index.py format
    if isinstance(meta, dict) and "meta" in meta and isinstance(meta["meta"], list):
        for i, it in enumerate(meta["meta"]):
            gid = it.get("gid") or it.get("giraffe_id")
            img = it.get("image") or it.get("image_path")
            mapping[i] = {"image": str(img), "gid": str(gid)}
        return mapping

    # A)
    if isinstance(meta, dict) and "items" in meta:
        for i, it in enumerate(meta["items"]):
            mapping[i] = {"image": it["image"], "gid": it["gid"]}
        return mapping

    # B)
    if isinstance(meta, dict):
        for k, it in meta.items():
            if k == "index":
                continue
            try:
                idx = int(k)
            except Exception:
                continue
            mapping[idx] = {"image": it["image"], "gid": it["gid"]}
        if mapping:
            return mapping

    # C)
    if isinstance(meta, list):
        for i, it in enumerate(meta):
            mapping[i] = {"image": it["image"], "gid": it["gid"]}
        return mapping

    raise ValueError("Unsupported meta JSON structure")

def build_id_image_maps(item_meta: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, int], List[str], List[str]]:
    """
    Build useful lookups:
      • id_image_counts: gid → #distinct gallery images present in meta
      • all_gids: sorted unique identities
      • all_gallery_images: sorted unique gallery image names
    """
    images_by_id: Dict[str, set] = defaultdict(set)
    gallery_images: set = set()
    for _, info in item_meta.items():
        gid = str(info["gid"])
        img = str(info["image"])
        images_by_id[gid].add(img)
        gallery_images.add(img)
    id_image_counts = {gid: len(imgs) for gid, imgs in images_by_id.items()}
    all_gids = sorted(images_by_id.keys())
    all_gallery_images = sorted(gallery_images)
    return id_image_counts, all_gids, all_gallery_images

def guess_annoy_dim(index_path: str) -> int:
    """Return descriptor dim; SIFT/RootSIFT are 128-D."""
    return 128


# ------------------- Matching (per query) -------------------
def match_query(
    des: np.ndarray,
    annoy: AnnoyIndex,
    item_meta: Dict[int, Dict[str, Any]],
    id_image_counts: Dict[str, int],
    k_neigh: int = K_NEIGH,
    search_k_mult: int = SEARCH_K_MULT,
    per_image_match_cap: int = PER_IMAGE_MATCH_CAP,
    per_id_normalize: bool = PER_ID_NORMALIZE
) -> Dict[str, Any]:
    """
    Retrieve k-NN for each descriptor and aggregate to identities.

    Returns:
      {
        "ranked":       [(gid, score)],  # score: sum of (1/dist) with light rank weight
        "ranked_votes": [(gid, votes)],  # raw vote counts per identity
        "image_votes":  {gallery_image: votes},
        "n_descriptors": int
      }
    """
    if des is None or len(des) == 0:
        return {"ranked": [], "ranked_votes": [], "image_votes": {}, "n_descriptors": 0}

    n_items = annoy.get_n_items()
    search_k = max(n_items * search_k_mult, k_neigh)

    image_votes: Dict[str, int] = defaultdict(int)    # votes per gallery image
    image_scores: Dict[str, float] = defaultdict(float)
    image_caps: Dict[str, int] = defaultdict(int)     # per-image cap tracker

    total_desc = 0
    for d in des:
        # retrieve neighbors (with distances) for this descriptor
        idxs, dists = annoy.get_nns_by_vector(d.tolist(), k_neigh, search_k=search_k, include_distances=True)
        total_desc += 1
        for rank, (idx, dist) in enumerate(zip(idxs, dists), start=1):
            meta = item_meta.get(idx)
            if not meta:
                continue
            img = meta["image"]
            # enforce per-image cap
            if per_image_match_cap and image_caps[img] >= per_image_match_cap:
                continue
            image_votes[img] += 1
            image_caps[img] += 1
            # inverse-distance score with light rank downweight
            s = 1.0 / (1e-6 + dist)
            s *= 1.0 / rank
            image_scores[img] += s

    # aggregate per ID
    id_votes: Dict[str, int] = defaultdict(int)
    id_scores: Dict[str, float] = defaultdict(float)
    for _, meta in item_meta.items():
        img = meta["image"]; gid = meta["gid"]
        if img in image_votes:
            id_votes[gid] += image_votes[img]
        if img in image_scores:
            id_scores[gid] += image_scores[img]

    # optional per-ID score normalisation (reduce bias from many gallery images)
    if per_id_normalize:
        for gid in list(id_scores):
            id_scores[gid] /= max(1, id_image_counts.get(gid, 1))

    ranked_scores = sorted(id_scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked_votes  = sorted(id_votes.items(),  key=lambda kv: kv[1], reverse=True)
    return {
        "ranked": ranked_scores,
        "ranked_votes": ranked_votes,
        "image_votes": dict(image_votes),
        "n_descriptors": int(total_desc)
    }


# ------------------- Evaluation over a set -------------------
def row_normalise_counts(counts: Dict[str, Dict[str, int]], classes: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
    """Row-normalise confusion counts to percent and probability per GT row."""
    rowpct, rowprob = [], []
    for gt in classes:
        row = [counts.get(gt, {}).get(pred, 0) for pred in classes]
        s = sum(row)
        if s == 0:
            rowpct.append([0.0] * len(classes))
            rowprob.append([0.0] * len(classes))
        else:
            rowpct.append([100.0 * c / s for c in row])
            rowprob.append([float(c) / s for c in row])
    return rowpct, rowprob

def evaluate_paths(
    qpaths: List[str],
    annoy: AnnoyIndex,
    item_meta: Dict[int, Dict[str, Any]],
    id_image_counts: Dict[str, int],
    classes: List[str],
    gallery_images: List[str],
    save_confmat: str,
    descriptor_mode: str,
    img_width: int,
    max_kpts: int,
    k_neigh: int,
    search_k_mult: int,
    per_image_match_cap: int,
    per_id_normalize: bool
) -> None:
    """Evaluate a list of query paths and write all CSV outputs."""

    # collectors
    counts: Dict[str, Dict[str, int]] = {gt: {pred: 0 for pred in classes} for gt in classes}
    top1_correct = 0
    top2_correct = 0
    total = 0

    votes_rows: List[List[Any]] = []
    byimg_rows: List[List[Any]] = []
    pred_rows:  List[List[Any]] = []
    time_rows:  List[List[Any]] = []

    header_ids   = ["query_image", "gt_id"] + classes
    header_byimg = ["query_image", "gt_id"] + gallery_images
    header_pred  = ["query_image", "gt_id", "pred_score", "pred_votes", "votes_pred"] + classes
    header_time  = ["query_image", "det_ms", "total_ms", "n_descriptors"]

    for i, qpath in enumerate(qpaths, start=1):
        t0 = time.time()
        img = preprocess_image(qpath, width=img_width)
        des, _ = extract_descriptors(img, mode=descriptor_mode, max_kpts=max_kpts)
        t1 = time.time()

        gt = Path(qpath).parent.name

        # rank identities for this query
        res = match_query(
            des, annoy, item_meta, id_image_counts,
            k_neigh=k_neigh, search_k_mult=search_k_mult,
            per_image_match_cap=per_image_match_cap,
            per_id_normalize=per_id_normalize
        )
        ranked        = res.get("ranked", [])
        ranked_votes  = res.get("ranked_votes", [])
        image_votes   = res.get("image_votes", {})
        n_desc        = res.get("n_descriptors", 0)

        # winners (by score, by votes); "__NONE__" if no candidates
        pred_score = ranked[0][0] if len(ranked) > 0 else "__NONE__"
        pred_votes = ranked_votes[0][0] if len(ranked_votes) > 0 else "__NONE__"

        # update confusion counts
        if gt not in counts:
            counts[gt] = {pred: 0 for pred in classes}
        if pred_score in counts[gt]:
            counts[gt][pred_score] += 1

        # top-k correctness
        total += 1
        if pred_score == gt:
            top1_correct += 1
        top2_ids = [p[0] for p in ranked[:2]]
        if gt in top2_ids:
            top2_correct += 1

        # per-query votes by ID
        votes_map = {gid: 0 for gid in classes}
        for gid, v in ranked_votes:
            if gid in votes_map:
                votes_map[gid] = int(v)
        relname = os.path.join(Path(qpath).parent.name, Path(qpath).name)
        votes_rows.append([relname, gt] + [votes_map[c] for c in classes])

        # per-query votes by gallery image
        byimg_map = {img_name: 0 for img_name in gallery_images}
        for img_name, v in image_votes.items():
            if img_name in byimg_map:
                byimg_map[img_name] = int(v)
        byimg_rows.append([relname, gt] + [byimg_map[c] for c in gallery_images])

        # per-query predictions summary
        votes_for_pred_votes = votes_map.get(pred_votes, 0) if isinstance(pred_votes, str) else 0
        pred_rows.append([relname, gt, pred_score, pred_votes, votes_for_pred_votes] + [votes_map[c] for c in classes])

        # timings
        t2 = time.time()
        det_ms = (t1 - t0) * 1000.0
        tot_ms = (t2 - t0) * 1000.0
        time_rows.append([relname, f"{det_ms:.1f}", f"{tot_ms:.1f}", n_desc])

        print(f"[{i}/{len(qpaths)}] {Path(qpath).name}: det={det_ms:.1f} ms total={tot_ms:.1f} ms dets={n_desc}")

    # ----- Write outputs -----
    counts_csv = save_confmat
    classes_hdr = ["GT\\Pred"] + classes

    # Confusion (counts)
    with open(counts_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(classes_hdr)
        for gt in classes:
            w.writerow([gt] + [counts.get(gt, {}).get(pred, 0) for pred in classes])
    print("[EVAL] Saved:", counts_csv)

    # Row-normalised percent
    rowpct, _ = row_normalise_counts(counts, classes)
    rowpct_csv = os.path.splitext(counts_csv)[0] + "_rowpct.csv"
    with open(rowpct_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(classes_hdr)
        for gt, vals in zip(classes, rowpct):
            w.writerow([gt] + [f"{v:.3f}" for v in vals])
    print("[EVAL] Saved:", rowpct_csv)

    # Per-query votes (by ID)
    votes_csv = os.path.splitext(counts_csv)[0] + "_votes_per_query.csv"
    with open(votes_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header_ids); w.writerows(votes_rows)
    print("[EVAL] Saved:", votes_csv)

    # Per-query votes (by gallery image)
    byimg_csv = os.path.splitext(counts_csv)[0] + "_votes_per_query_by_image.csv"
    with open(byimg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header_byimg); w.writerows(byimg_rows)
    print("[EVAL] Saved:", byimg_csv)

    # Per-query predictions (winners + per-ID votes)
    pred_csv = os.path.splitext(counts_csv)[0] + "_per_query_predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header_pred); w.writerows(pred_rows)
    print("[EVAL] Saved:", pred_csv)

    # Per-query timings
    times_csv = os.path.splitext(counts_csv)[0] + "_times.csv"
    with open(times_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header_time); w.writerows(time_rows)
    print("[EVAL] Saved:", times_csv)

    # Summary
    top1 = top1_correct / max(1, total)
    top2 = top2_correct / max(1, total)
    print(f"[EVAL] Top-1={top1:.4f}  Top-2={top2:.4f}  N={total}")


# ------------------- CLI -------------------
def collect_classes_from_test(test_dir: str) -> List[str]:
    """Infer class names from test_dir by using each file's parent folder name."""
    classes = set()
    for root, _, files in os.walk(test_dir):
        for fn in files:
            ext = os.path.splitext(fn.lower())[1]
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
                classes.add(Path(root).name)
    return sorted(classes)

def main():
    ap = argparse.ArgumentParser(
        description="SIFT/RootSIFT ReID evaluator with per-query predictions and confusion CSVs"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Evaluate a directory (or a single image) against an Annoy index")
    grp = pe.add_mutually_exclusive_group(required=True)
    grp.add_argument("--test_dir", help="Directory of query images; GT inferred from each file's parent folder")
    grp.add_argument("--query_image", help="Single query image path; GT inferred from its parent folder")

    pe.add_argument("--index_path", required=True, help="Annoy index (.ann) built from gallery descriptors")
    pe.add_argument("--meta_path",  required=True, help="JSON mapping Annoy item ids to {'image','gid'}")
    pe.add_argument("--save_confmat", required=True, help="Output CSV path for confusion counts (base for all outputs)")

    # Query feature extraction
    pe.add_argument("--descriptor", default="rootsift", choices=["rootsift", "sift"], help="Descriptor mode for queries")
    pe.add_argument("--img_width", type=int, default=IMG_WIDTH, help="Resize width (preserve aspect) before SIFT")
    pe.add_argument("--max_kpts",  type=int, default=MAX_KPTS, help="Max query keypoints/descriptors to use")

    # Annoy retrieval settings
    pe.add_argument("--k_neigh", type=int, default=K_NEIGH, help="k neighbours per descriptor")
    pe.add_argument("--search_k_mult", type=int, default=SEARCH_K_MULT, help="search_k = n_items * this multiplier")
    pe.add_argument("--per_image_match_cap", type=int, default=PER_IMAGE_MATCH_CAP, help="Cap matches counted per gallery image")

    # Scoring / normalisation
    pe.add_argument("--no_per_id_normalize", action="store_true", help="Disable per-ID score normalisation (scores only)")

    args = ap.parse_args()

    # Load gallery meta and index
    item_meta = load_meta(args.meta_path)
    id_image_counts, all_gids, all_gallery_images = build_id_image_maps(item_meta)

    dim = guess_annoy_dim(args.index_path)
    annoy = AnnoyIndex(dim, 'euclidean')
    if not annoy.load(args.index_path):
        print(f"[ERR] Failed to load Annoy index: {args.index_path}")
        sys.exit(2)

    # Sanity check: meta length vs index items
    n_items = annoy.get_n_items()
    if len(item_meta) != n_items:
        print(f"[WARN] Meta entries ({len(item_meta)}) != index items ({n_items}). "
              f"If meta is 0, check meta format/paths.")

    # Build list of query paths + class set
    if args.query_image:
        if not os.path.isfile(args.query_image):
            print(f"[ERR] --query_image not found: {args.query_image}")
            sys.exit(2)
        qpaths = [args.query_image]
        single_gt = Path(args.query_image).parent.name
        classes = sorted(set([single_gt]) | set(all_gids))  # include gallery IDs for aligned columns
    else:
        classes = collect_classes_from_test(args.test_dir) or all_gids
        classes = sorted(classes)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        qpaths = sorted([
            p for p in glob.glob(os.path.join(args.test_dir, "**", "*"), recursive=True)
            if p.lower().endswith(exts)
        ])
        if len(qpaths) == 0:
            print(f"[ERR] No images found under: {args.test_dir}")
            sys.exit(2)

    # Run evaluation
    evaluate_paths(
        qpaths, annoy, item_meta, id_image_counts, classes, all_gallery_images,
        save_confmat=args.save_confmat, descriptor_mode=args.descriptor,
        img_width=args.img_width, max_kpts=args.max_kpts,
        k_neigh=args.k_neigh, search_k_mult=args.search_k_mult,
        per_image_match_cap=args.per_image_match_cap,
        per_id_normalize=(not args.no_per_id_normalize)
    )

if __name__ == "__main__":
    main()
