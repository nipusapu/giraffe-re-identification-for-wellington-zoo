#!/usr/bin/env python3
"""
SIFT / RootSIFT Gallery Index Builder (Annoy) — with informativeness filtering and reproducibility metadata.

What this script does
---------------------
1) Walks a labeled gallery laid out as:
      GALLERY_DIR/
        NIA/     *.jpg, *.png, ...
        SUNNY/   ...
        ZAHARA/  ...
        ZURI/    ...
   Each subfolder name is treated as the identity (giraffe ID).

2) For each image:
   • Resize to a fixed width (keeps aspect)
   • Convert to grayscale
   • Apply CLAHE (helps under harsh lighting)
   • Extract SIFT descriptors
   • Keep the top-K keypoints by response (strongest)
   • Convert to RootSIFT (L1-normalize → sqrt)

3) Applies informativeness filtering:
   • Uses cross-validation to score each descriptor by margin ratio
   • Keeps the most informative descriptors per class
   • Skips if --final_target_per_class is not set

4) Adds every (filtered) descriptor as an item in an Annoy index (metric = euclidean)
   and records lightweight metadata (which image + which ID).

5) Saves:
   • Annoy index (e.g., gallery.ann)
   • JSON metadata (paths, IDs) + the exact config used (for reproducibility)

Inputs you must provide
-----------------------
• --gallery_dir   : folder with subfolders per identity (NIA, SUNNY, …)
• --index_path    : output .ann path
• --meta_path     : output JSON metadata path

Key behaviors / assumptions
---------------------------
• Descriptor dimension is 128 for (Root)SIFT.
• Annoy metric is **euclidean**; RootSIFT typically works best with it.
• Each descriptor becomes **one** Annoy item; metadata maps items back to (image, ID).
• CLAHE is applied to stabilize SIFT in varied lighting.

Quick start
-----------
# Without informativeness filtering (default):
python build_sift_index.py \
  --gallery_dir data/gallery \
  --index_path  out/sift_gallery_rootsift_100_unsegmented.ann \
  --meta_path   out/sift_gallery_rootsift_100_unsegmented.json \
  --img_width   256 \
  --max_kpts    150 \
  --num_trees   100 \
  --descriptor  rootsift

# With informativeness filtering (keep top 10,000 descriptors per class):
python build_sift_index.py \
  --gallery_dir data/gallery \
  --index_path  out/sift_gallery_rootsift_100_filtered.ann \
  --meta_path   out/sift_gallery_rootsift_100_filtered.json \
  --img_width   256 \
  --max_kpts    150 \
  --num_trees   100 \
  --descriptor  rootsift \
  --final_target_per_class 10000 \
  --k_neighbors 20 \
  --n_splits    5

Outputs
-------
• Console summary (number of images / descriptors)
• Saved Annoy index (.ann)
• JSON metadata with config snapshot (for reproducibility)

Requirements
------------
pip install opencv-contrib-python annoy scikit-learn
"""

import os, json, time, argparse
from typing import Optional
from pathlib import Path
import numpy as np
import cv2
from annoy import AnnoyIndex
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

BBOX_JSON = Path("bbox_annotations.json")
SEGMENT_TRAIN = False  # set True if you have masks for training images and want to apply them
MASKS_ROOT = Path("masks")            

def load_bbox_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_keypoints_vis(img_bgr, kps, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis = cv2.drawKeypoints(
        img_bgr,
        kps,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(str(out_path), vis)

def apply_mask_rgb(
    img_bgr: np.ndarray,
    mask,
    *,
    invert: bool = False,
    threshold: int = 127,
    background = (0, 0, 0),
    resize_interpolation: int = cv2.INTER_NEAREST,
    return_binary_mask: bool = False,
):
    """
    Apply a binary mask to a BGR OpenCV image.

    Args:
        img_bgr: HxWx3 uint8 BGR image.
        mask: Either:
              - path to a mask image (png/jpg/etc), or
              - numpy array mask (HxW or HxWxC).
        invert: If True, invert mask (swap foreground/background).
        threshold: Threshold used to binarize mask if it isn't already.
        background: Background fill value for masked-out pixels.
                    Either scalar int or 3-tuple BGR.
        resize_interpolation: Interpolation used if mask needs resizing.
        return_binary_mask: If True, also return the binary mask (uint8 0/255).

    Returns:
        masked_img (and binary_mask).
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"img_bgr must be HxWx3 BGR image, got shape {None if img_bgr is None else img_bgr.shape}")

    # Load mask if a path was provided
    if isinstance(mask, (str, Path)):
        mask_path = str(mask)
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Could not read mask at: {mask_path}")
    else:
        m = mask

    # Convert mask to single channel
    if m.ndim == 3:
        # If it has alpha, prefer alpha channel; else convert to gray
        if m.shape[2] == 4:
            m = m[:, :, 3]
        else:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    # Resize mask to match image if needed
    H, W = img_bgr.shape[:2]
    if m.shape[0] != H or m.shape[1] != W:
        m = cv2.resize(m, (W, H), interpolation=resize_interpolation)

    # Binarize
    if m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)
    _, binmask = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)

    if invert:
        binmask = cv2.bitwise_not(binmask)

    # Apply mask
    out = img_bgr.copy()
    keep = binmask.astype(bool)

    if isinstance(background, int):
        out[~keep] = background
    else:
        out[~keep] = np.array(background, dtype=out.dtype)

    if return_binary_mask:
        return out, binmask
    return out
    
def clamp_box(b, w, h):
    x1 = max(0, min(int(b["x1"]), w - 1))
    x2 = max(0, min(int(b["x2"]), w - 1))
    y1 = max(0, min(int(b["y1"]), h - 1))
    y2 = max(0, min(int(b["y2"]), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def find_mask_for_train_image(img_path: Path, masks_root: Path) -> Optional[Path]:
    """
    Masks are stored flat in one folder: masks/
    Tries:
      masks/<stem>.png
      masks/<stem>_mask.png
    Example:
      Dataset/train/classA/01_014_0006.jpg -> masks/01_014_0006.png (or ..._mask.png)
    """
    stem = img_path.stem
    cand1 = masks_root / f"{stem}.png"
    cand2 = masks_root / f"{stem}_mask.png"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


# ------------------- Image preprocessing -------------------
def preprocess_gray(img: np.ndarray, width: int) -> np.ndarray:
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


# ------------------- RootSIFT normalization -------------------
def rootsift(desc: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Convert SIFT to RootSIFT:
      • L1-normalize each row, then sqrt elementwise.
      • Works well with Euclidean distance.
    """
    if desc is None or desc.size == 0:
        return desc
    eps = 1e-10
    desc = desc.astype(np.float32, copy=False)
    desc /= (np.sum(desc, axis=1, keepdims=True) + eps)
    np.sqrt(desc, out=desc)
    return desc


# ------------------- Informativeness filtering -------------------
def filter_informativeness(X, y, k_neighbors: int, n_splits: int = 5, seed: int = 42):
    """
    Score each descriptor by CV margin:
      d+ = distance to nearest SAME-class descriptor in TRAIN fold
      d- = distance to nearest DIFFERENT-class descriptor in TRAIN fold
      score = d- / (d+ + eps)
    Returns:
      scores: (N,) float32, averaged over folds (each point scored in its test fold)
    """
    eps = 1e-12
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    N = len(X)
    scores = np.full(N, np.nan, dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_num, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"  [FILTER] Processing fold {fold_num}/{n_splits}...")
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        k = min(k_neighbors, len(Xtr))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(Xtr)

        dists, inds = nn.kneighbors(Xte, return_distance=True)

        fold_scores = np.empty(len(Xte), dtype=np.float32)

        for i in range(len(Xte)):
            neigh_labels = ytr[inds[i]]
            neigh_dists  = dists[i]

            same = np.where(neigh_labels == yte[i])[0]
            diff = np.where(neigh_labels != yte[i])[0]

            if same.size == 0 or diff.size == 0:
                fold_scores[i] = -np.inf  # shouldn't happen if each class appears in train
                continue

            d_plus  = neigh_dists[same[0]]
            d_minus = neigh_dists[diff[0]]
            fold_scores[i] = d_minus / (d_plus + eps)

        scores[te] = fold_scores

    # If any NaNs remain (rare), set them very low
    scores = np.where(np.isfinite(scores), scores, -np.inf).astype(np.float32)
    return scores


def select_top_per_class(X, y, scores, final_target_per_class: int):
    """
    Select top `final_target_per_class` descriptors per class by `scores`.
    Returns X_sel, y_sel, idx_sel.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    scores = np.asarray(scores)

    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = idx[np.argsort(scores[idx])[::-1]]  # descending
        keep.append(idx[:min(final_target_per_class, len(idx))])

    idx_sel = np.concatenate(keep)
    return X[idx_sel], y[idx_sel], idx_sel


# ------------------- Index builder -------------------
def build_index(
    gallery_dir: str,
    index_path: str,
    meta_path: str,
    descr_dim: int,
    num_trees: int,
    max_kpts: int,
    img_width: int,
    descriptor: str,
    final_target_per_class: int = None,
    k_neighbors: int = 20,
    n_splits: int = 5,
    seed: int = 42
) -> None:
    """
    Walk the gallery, extract (Root)SIFT, filter by informativeness, and build an Annoy index.

    Notes:
      • Annoy metric is 'euclidean' to match RootSIFT behavior.
      • Each descriptor becomes one Annoy item; metadata maps back to image & ID.
      • If final_target_per_class is set, applies informativeness filtering.
    """
    sift = cv2.SIFT_create()
    use_rootsift = (descriptor.lower() == "rootsift")
    bbox_map = load_bbox_json(BBOX_JSON)

    # Step 1: Extract all descriptors first
    print("Extracting descriptors from all images...")
    all_desc = []
    all_cls = []
    all_img = []
    image_counter = 0

    # Expect: gallery_dir/<identity>/*.jpg|png|...
    for identity in sorted(os.listdir(gallery_dir)):
        subdir = os.path.join(gallery_dir, identity)
        if not os.path.isdir(subdir):
            continue

        for fname in sorted(os.listdir(subdir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue

            path = os.path.join(subdir, fname)
            # Build key consistently (use relative path or whatever matches bbox json)
            key = os.path.join(identity, fname).replace("\\", "/")  # example stable key
            boxes = bbox_map.get(key)

            box = boxes[0] if boxes else None

            raw = cv2.imread(path)  # BGR
            if raw is None:
                continue

            img = raw  # work image is OpenCV array

            # Crop first (common for flank masks that match the full image), then mask the crop
            if box is not None:
                h, w = img.shape[:2]
                x1, y1, x2, y2 = clamp_box(box, w=w, h=h)
                img = img[y1:y2, x1:x2].copy()

            # Optional masking on the cropped image
            if SEGMENT_TRAIN:
                mask_path = find_mask_for_train_image(path, MASKS_ROOT)
                if mask_path is not None:
                    img = apply_mask_rgb(img, mask_path)  # will auto-resize mask if needed

            # SIFT expects gray
            gray = preprocess_gray(img, img_width)  # ensure preprocess_gray accepts BGR ndarray
            kps, desc = sift.detectAndCompute(gray, None)
            if not kps or desc is None or desc.size == 0:
                continue

            # Save visualisation
            out_vis_path = Path("debug_kps") / identity / f"{Path(fname).stem}_kp.jpg"
            save_keypoints_vis(img, kps, out_vis_path)

            # Keep top-K by keypoint response (strongest first)
            keep_idx = np.argsort([-kp.response for kp in kps])[:max_kpts]
            desc = desc[keep_idx]

            # Optional RootSIFT normalization
            if use_rootsift:
                desc = rootsift(desc)

            # Collect descriptors instead of adding directly to index
            desc = np.asarray(desc, dtype=np.float32)
            all_cls.append(np.full((desc.shape[0],), identity, dtype=object))
            all_desc.append(desc)
            all_img.append(np.full((desc.shape[0],), fname, dtype=object))
            image_counter += 1

    # Step 2: Concatenate all descriptors
    X = np.concatenate(all_desc, axis=0)       # (M, 128)
    y = np.concatenate(all_cls, axis=0)        # (M,)
    all_img = np.concatenate(all_img, axis=0)  # (M,)
    print(f"Images processed: {image_counter}")
    print(f"Total descriptors extracted: {X.shape[0]}")

    # Step 3: Optional informativeness filtering
    if final_target_per_class is not None and final_target_per_class > 0:
        print(f"Filtering descriptors by informativeness (target={final_target_per_class} per class)...")
        scores = filter_informativeness(X, y, k_neighbors, n_splits, seed)
        X_filtered, y_filtered, idx_filtered = select_top_per_class(X, y, scores, final_target_per_class)
        img_filtered = all_img[idx_filtered]
        print(f"Descriptors after filtering: {X_filtered.shape[0]}")
    else:
        X_filtered = X
        y_filtered = y
        img_filtered = all_img
        print("Skipping informativeness filtering (final_target_per_class not set)")

    # Step 4: Build image ID mapping
    unique_imgs = sorted(set(map(str, img_filtered)))
    img2id = {p: i for i, p in enumerate(unique_imgs)}

    # Step 5: Build Annoy index
    print("Building Annoy index...")
    index = AnnoyIndex(descr_dim, "euclidean")
    meta = []

    for i in range(X_filtered.shape[0]):
        index.add_item(i, X_filtered[i].astype(np.float32).tolist())
        meta.append({
            "identity": str(y_filtered[i]),
            "image_path": img_filtered[i],
            "image_id": img2id[img_filtered[i]]
        })

    # Build Annoy forest (NUM_TREES: recall vs build time)
    index.build(num_trees)
    index.save(index_path)

    # Save metadata + exact build config for reproducibility
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "config": {
                    "DESCR_DIM": descr_dim,
                    "NUM_TREES": num_trees,
                    "MAX_KPTS": max_kpts,
                    "IMG_WIDTH": img_width,
                    "descriptor": "RootSIFT" if use_rootsift else "SIFT",
                    "annoy_metric": "euclidean",
                    "final_target_per_class": final_target_per_class,
                    "k_neighbors": k_neighbors if final_target_per_class else None,
                    "n_splits": n_splits if final_target_per_class else None,
                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            },
            f,
            indent=2,
        )

    print(f"Index->{index_path}")
    print(f"Meta->{meta_path}")


# ------------------- CLI -------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build an Annoy index from a labeled gallery using SIFT or RootSIFT."
    )
    ap.add_argument(
        "--gallery_dir", required=True,
        help="Root folder with one subfolder per identity (e.g., GALLERY/NIA, GALLERY/SUNNY, ...)."
    )
    ap.add_argument(
        "--index_path", required=True,
        help="Output path for the Annoy index file (e.g., out/sift_gallery_rootsift_100_unsegmented.ann)."
    )
    ap.add_argument(
        "--meta_path", required=True,
        help="Output path for JSON metadata (e.g., out/sift_gallery_rootsift_100_unsegmented.json)."
    )
    ap.add_argument(
        "--img_width", type=int, default=256,
        help="Resize width before SIFT; height scales to keep aspect (default: 256)."
    )
    ap.add_argument(
        "--max_kpts", type=int, default=85,
        help="Keep at most this many strongest keypoints per image (default: 85)."
    )
    ap.add_argument(
        "--num_trees", type=int, default=100,
        help="Annoy NUM_TREES; higher = better recall, slower build (default: 100)."
    )
    ap.add_argument(
        "--descr_dim", type=int, default=128,
        help="Descriptor length (SIFT=128). Do not change unless using a custom extractor."
    )
    ap.add_argument(
        "--descriptor", choices=["sift", "rootsift"], default="rootsift",
        help="Which descriptor to store. RootSIFT is recommended with euclidean metric."
    )
    ap.add_argument(
        "--final_target_per_class", type=int, default=None,
        help="If set, apply informativeness filtering to keep this many descriptors per class (default: None, no filtering)."
    )
    ap.add_argument(
        "--k_neighbors", type=int, default=20,
        help="Number of neighbors for informativeness filtering (default: 20)."
    )
    ap.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of CV folds for informativeness filtering (default: 5)."
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for informativeness filtering (default: 42)."
    )
    args = ap.parse_args()

    build_index(
        args.gallery_dir, args.index_path, args.meta_path,
        args.descr_dim, args.num_trees, args.max_kpts, args.img_width, args.descriptor,
        args.final_target_per_class, args.k_neighbors, args.n_splits, args.seed
    )


if __name__ == "__main__":
    main()
