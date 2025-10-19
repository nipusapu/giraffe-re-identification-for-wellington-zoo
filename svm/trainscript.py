#!/usr/bin/env python3
"""
HOG + Linear SVM trainer script.
What this script does (high level):
1) Loads two image splits: TRAIN and VALIDATION. Each split has two folders:
      <split>/positives/*.jpg   (flank present)
      <split>/negatives/*.jpg   (no flank / background)
2) Converts each image to grayscale, resizes to a fixed size, and extracts HOG features.
3) Trains a LinearSVC (linear SVM) on TRAIN features.
4) Evaluates accuracy on both TRAIN and VALIDATION.
5) Optionally logs metrics and artifacts to Weights & Biases (W&B).
6) Saves the trained model as a .pkl file via joblib.

Why fixed HOG size?
- HOG turns an image into a fixed-length feature vector *only if* all images share the
  same width/height grid. Resizing guarantees the feature vectors all have the same length.

Expected folder layout:
    train_dir/
      ├── positives/
      │     └── *.jpg
      └── negatives/
            └── *.jpg
    val_dir/
      ├── positives/
      │     └── *.jpg
      └── negatives/
            └── *.jpg

Quick start
-----------
Install deps:
    pip install wandb scikit-image scikit-learn joblib opencv-python

(Optionally) log in to W&B:
    wandb login

Run:
    python trainscript.py \
      --train_dir output/crops/train \
      --val_dir   output/crops/val \
      --hog_size 128 128 \
      --wandb_project giraffe-flank-svm \
      --wandb_run_name hog-svm-trainval-128

Outputs:
- Console metrics for train/val accuracy
- Saved model: hog_svm.pkl (path set by --model_out)
- If W&B is installed: metrics + confusion matrix counts + model artifact
"""

import os
import glob
import argparse
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle

# --------------------------- HOG defaults (sensible starting points) --------------------------- #
DEFAULT_ORIENTATIONS = 6               # number of gradient orientation bins
DEFAULT_PX_PER_CELL = (4, 4)           # pixels per cell (x, y)
DEFAULT_CELLS_PER_BLOCK = (2, 2)       # cells per block (x, y)
DEFAULT_BLOCK_NORM = 'L2-Hys'          # normalization scheme recommended for HOG


def extract_hog(image_path, orientations, ppc, cpb, block_norm, hog_size):
    """
    Load one image and return its HOG feature vector.

    Steps:
      - Read BGR image from disk
      - Convert to grayscale (HOG uses intensity)
      - Resize to fixed (W,H) if provided
      - Compute HOG descriptor

    Args:
        image_path (str): Path to an image on disk.
        orientations (int): Number of orientation bins.
        ppc (tuple[int, int]): Pixels per cell, e.g., (4,4).
        cpb (tuple[int, int]): Cells per block, e.g., (2,2).
        block_norm (str): HOG block normalization mode, e.g., 'L2-Hys'.
        hog_size (tuple[int, int] or None): (W, H) to resize to. If None, keep original size.

    Returns:
        np.ndarray | None: 1D HOG feature vector (float32). None if the image failed to load.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure every image produces the exact same feature length by resizing first
    if hog_size is not None:
        w, h = hog_size  # (W, H)
        if w > 0 and h > 0 and (gray.shape[1] != w or gray.shape[0] != h):
            # INTER_AREA is usually safe for downscaling; OpenCV picks a reasonable default
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

    feat = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=ppc,
        cells_per_block=cpb,
        block_norm=block_norm,
        transform_sqrt=True,   # light contrast normalization helps robustness
    )
    return feat.astype(np.float32, copy=False)


def load_split_dir(split_dir, orientations, ppc, cpb, block_norm, hog_size, limit=None):
    """
    Load all images from one split (train OR val) and build X, y.

    Expected subfolders:
        split_dir/positives/*.jpg   -> label 1
        split_dir/negatives/*.jpg   -> label 0

    Args:
        split_dir (str): Root folder for the split.
        orientations, ppc, cpb, block_norm, hog_size: See extract_hog().
        limit (int|None): If set, cap the number of images **per class** for quick experiments.

    Returns:
        X (np.ndarray): Shape (N, D) feature matrix.
        y (np.ndarray): Shape (N,) integer labels (0 or 1).
    """
    pos_dir = os.path.join(split_dir, 'positives')
    neg_dir = os.path.join(split_dir, 'negatives')
    X, y = [], []

    pos_paths = sorted(glob.glob(os.path.join(pos_dir, '*.jpg')))
    neg_paths = sorted(glob.glob(os.path.join(neg_dir, '*.jpg')))

    if limit is not None:
        pos_paths = pos_paths[:limit]
        neg_paths = neg_paths[:limit]

    loaded_pos = loaded_neg = 0

    # Build features for positives (label = 1)
    for p in pos_paths:
        f = extract_hog(p, orientations, ppc, cpb, block_norm, hog_size)
        if f is not None:
            X.append(f); y.append(1); loaded_pos += 1

    # Build features for negatives (label = 0)
    for p in neg_paths:
        f = extract_hog(p, orientations, ppc, cpb, block_norm, hog_size)
        if f is not None:
            X.append(f); y.append(0); loaded_neg += 1

    # Handle empty split
    if len(X) == 0:
        return np.empty((0,)), np.empty((0,))

    # Sanity check: all feature vectors must have the same length
    feat_len = len(X[0])
    bad = [i for i, f in enumerate(X) if len(f) != feat_len]
    if bad:
        raise RuntimeError(
            f"Inconsistent HOG lengths (first={feat_len}, e.g. idx {bad[0]} len={len(X[bad[0]])}). "
            f"Set --hog_size so every crop is resized identically."
        )

    X = np.vstack(X).astype(np.float32, copy=False)
    y = np.array(y, dtype=np.int64)

    print(f"[load] {split_dir} -> positives={loaded_pos}, negatives={loaded_neg}, feats={X.shape}")
    X, y = shuffle(X, y, random_state=42)
    return X, y


def main():
    # --------------------------- CLI arguments --------------------------- #
    ap = argparse.ArgumentParser(
        description="Train a HOG+LinearSVC model and report TRAIN/VAL accuracy. "
                    "Requires 'positives' and 'negatives' subfolders in both splits."
    )
    ap.add_argument('--train_dir', required=True,
                    help='Path to TRAIN split (must contain positives/ and negatives/).')
    ap.add_argument('--val_dir',   required=True,
                    help='Path to VALIDATION split (must contain positives/ and negatives/).')

    # HOG configuration (keep consistent across train/val!)
    ap.add_argument('--orientations', type=int, default=DEFAULT_ORIENTATIONS,
                    help='Number of gradient orientation bins (e.g., 6, 9).')
    ap.add_argument('--px_per_cell', type=int, nargs=2, default=DEFAULT_PX_PER_CELL, metavar=('PX', 'PY'),
                    help='Pixels per cell (width height). Smaller cells capture finer detail.')
    ap.add_argument('--cells_per_block', type=int, nargs=2, default=DEFAULT_CELLS_PER_BLOCK, metavar=('CX', 'CY'),
                    help='Cells per block (width height). Controls local normalization window.')
    ap.add_argument('--block_norm', type=str, default=DEFAULT_BLOCK_NORM,
                    help="Block normalization mode: one of 'L1', 'L1-sqrt', 'L2', 'L2-Hys' (recommended).")

    # Resizing to a fixed grid guarantees fixed-length HOG vectors
    ap.add_argument('--hog_size', type=int, nargs=2, default=(128, 128), metavar=('W', 'H'),
                    help='Resize each crop to this (W H) before HOG. Must be the same for train and val.')

    ap.add_argument('--max_per_class', type=int, default=None,
                    help='Optional cap per class per split (debug/quick runs).')

    ap.add_argument('--model_out', type=str, default='hog_svm.pkl',
                    help='Where to save the trained model (.pkl).')

    # Optional: Weights & Biases logging
    ap.add_argument('--wandb_project', type=str, default='giraffe-flank-svm',
                    help='W&B project name (if wandb is installed).')
    ap.add_argument('--wandb_run_name', type=str, default=None,
                    help='W&B run name (optional).')
    args = ap.parse_args()

    # Lazy-import W&B so the script works even if wandb is not installed
    WANDB = None
    try:
        import wandb
        WANDB = wandb
    except Exception:
        print("[INFO] Weights & Biases (wandb) not installed; to enable logging: pip install wandb")

    # --------------------------- Data loading --------------------------- #
    X_train, y_train = load_split_dir(
        args.train_dir,
        args.orientations, tuple(args.px_per_cell), tuple(args.cells_per_block), args.block_norm,
        tuple(args.hog_size),
        args.max_per_class
    )
    X_val, y_val = load_split_dir(
        args.val_dir,
        args.orientations, tuple(args.px_per_cell), tuple(args.cells_per_block), args.block_norm,
        tuple(args.hog_size),
        args.max_per_class
    )
    print(f"[data] train: {len(y_train)}  val: {len(y_val)}")

    # Initialize W&B run (optional)
    run = None
    if WANDB is not None:
        run = WANDB.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=dict(
                orientations=args.orientations,
                px_per_cell=args.px_per_cell,
                cells_per_block=args.cells_per_block,
                block_norm=args.block_norm,
                hog_size=tuple(args.hog_size),
                max_per_class=args.max_per_class,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
            ),
        )

    # Basic guard: both splits must contain data
    if len(y_train) == 0 or len(y_val) == 0:
        print("[ERR] Need both train and val data. Check your directory paths and contents.")
        if run:
            run.finish()
        raise SystemExit(2)

    # --------------------------- Train model --------------------------- #
    # LinearSVC is a fast linear SVM optimized for large, dense feature vectors like HOG.
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train, y_train)

    # --------------------------- Evaluate --------------------------- #
    ytr = clf.predict(X_train)
    yva = clf.predict(X_val)
    train_acc = float(accuracy_score(y_train, ytr))
    val_acc   = float(accuracy_score(y_val, yva))
    print(f"[metrics] train_accuracy={train_acc:.4f}  val_accuracy={val_acc:.4f}")

    # Optional: log metrics + confusion matrix to W&B
    if run:
        WANDB.log({"train_accuracy": train_acc, "val_accuracy": val_acc})
        cm = confusion_matrix(y_val, yva, labels=[0, 1])  # [neg, pos] order
        WANDB.log({"val_confusion_00_01_10_11": cm.flatten().tolist()})

    # --------------------------- Save model --------------------------- #
    joblib.dump(clf, args.model_out)
    print(f"[save] model -> {args.model_out}")

    # Also upload to W&B as an artifact (if enabled)
    if run:
        art = WANDB.Artifact('hog_svm_model', type='model')
        art.add_file(args.model_out)
        WANDB.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()
