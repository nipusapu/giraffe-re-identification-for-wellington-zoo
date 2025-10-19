#!/usr/bin/env python3
"""
SIFT / RootSIFT Gallery Index Builder (Annoy) — with reproducibility metadata.

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
   • Optionally convert to RootSIFT (L1-normalize → sqrt)

3) Adds every descriptor as an item in an Annoy index (metric = euclidean)
   and records lightweight metadata (which image + which ID).

4) Saves:
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
python build_sift_index.py \
  --gallery_dir data/gallery \
  --index_path  out/sift_gallery.ann \
  --meta_path   out/sift_meta.json \
  --img_width   256 \
  --max_kpts    150 \
  --num_trees   50 \
  --descriptor  rootsift

Outputs
-------
• Console summary (number of images / descriptors)
• Saved Annoy index (.ann)
• JSON metadata with config snapshot (for reproducibility)

Requirements
------------
pip install opencv-contrib-python annoy
"""

import os, json, time, argparse
from typing import Optional
import numpy as np
import cv2
from annoy import AnnoyIndex


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


# ------------------- Index builder -------------------
def build_index(
    gallery_dir: str,
    index_path: str,
    meta_path: str,
    descr_dim: int,
    num_trees: int,
    max_kpts: int,
    img_width: int,
    descriptor: str
) -> None:
    """
    Walk the gallery, extract (Root)SIFT, and build an Annoy index.

    Notes:
      • Annoy metric is 'euclidean' to match RootSIFT behavior.
      • Each descriptor becomes one Annoy item; metadata maps back to image & ID.
    """
    sift = cv2.SIFT_create()
    index = AnnoyIndex(descr_dim, "euclidean")
    meta = []                     # metadata per descriptor (parallel to Annoy items)
    item_id = 0                   # monotonically increasing Annoy item id
    image_counter = 0             # counts images processed (not items)
    use_rootsift = (descriptor.lower() == "rootsift")

    # Expect: gallery_dir/<identity>/*.jpg|png|...
    for identity in sorted(os.listdir(gallery_dir)):
        subdir = os.path.join(gallery_dir, identity)
        if not os.path.isdir(subdir):
            continue

        for fname in sorted(os.listdir(subdir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue

            path = os.path.join(subdir, fname)
            raw = cv2.imread(path)
            if raw is None:
                # unreadable/corrupt – skip
                continue

            # Preprocess → SIFT
            gray = preprocess_gray(raw, img_width)
            kps, desc = sift.detectAndCompute(gray, None)
            if not kps or desc is None or desc.size == 0:
                # no features – skip
                continue

            # Keep top-K by keypoint response (strongest first)
            keep_idx = np.argsort([-kp.response for kp in kps])[:max_kpts]
            desc = desc[keep_idx]

            # Optional RootSIFT normalization
            if use_rootsift:
                desc = rootsift(desc)

            # Add each descriptor as an Annoy item and record metadata
            for d in desc:
                index.add_item(item_id, d.astype(np.float32).tolist())
                meta.append({
                    "identity": identity,     # folder name label
                    "image_path": path,      # source path
                    "image_id": image_counter
                })
                item_id += 1

            image_counter += 1

    print(f"[BUILD] Images processed: {image_counter}")
    print(f"[BUILD] Descriptors added: {item_id}")

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
                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            },
            f,
            indent=2,
        )

    print(f"[SAVE] Index  → {index_path}")
    print(f"[SAVE] Meta   → {meta_path}")


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
        help="Output path for the Annoy index file (e.g., out/sift_gallery.ann)."
    )
    ap.add_argument(
        "--meta_path", required=True,
        help="Output path for JSON metadata (e.g., out/sift_meta.json)."
    )
    ap.add_argument(
        "--img_width", type=int, default=256,
        help="Resize width before SIFT; height scales to keep aspect (default: 256)."
    )
    ap.add_argument(
        "--max_kpts", type=int, default=150,
        help="Keep at most this many strongest keypoints per image (default: 150)."
    )
    ap.add_argument(
        "--num_trees", type=int, default=50,
        help="Annoy NUM_TREES; higher = better recall, slower build (default: 50)."
    )
    ap.add_argument(
        "--descr_dim", type=int, default=128,
        help="Descriptor length (SIFT=128). Do not change unless using a custom extractor."
    )
    ap.add_argument(
        "--descriptor", choices=["sift", "rootsift"], default="rootsift",
        help="Which descriptor to store. RootSIFT is recommended with euclidean metric."
    )
    args = ap.parse_args()

    build_index(
        args.gallery_dir, args.index_path, args.meta_path,
        args.descr_dim, args.num_trees, args.max_kpts, args.img_width, args.descriptor
    )


if __name__ == "__main__":
    main()
