#!/usr/bin/env python3
"""
MobileNetV3-Large Faster R-CNN Test Runner (service-aligned)
— evaluates detections vs COCO GT, computes VOC 11-point AP,
  saves crops (pass/fail/no_gt), and writes JSON metrics+detections.

What this script does
---------------------
1) Loads your trained detector checkpoint into a Faster R-CNN
   (MobileNetV3-Large+FPN) model with the custom 1024-d ROI head.
2) Reads a COCO JSON and robustly resolves each image path under --images_dir.
3) Runs inference (optionally warm up first), with service-aligned preprocessing:
   • grayscale triplication if enabled  (NO CLAHE/sharpen here)
   • ImageNet normalization
4) Filters detections by score, optionally keeps only top-1 (service-like),
   computes IoU with each image's GT, and classifies detections as
   PASS / FAIL / NO_GT for crop saving.
5) Aggregates detections across the run to compute VOC 11-point AP at --iou_thresh,
   recall, mean IoU over TPs, and other counts.
6) Writes:
   • <out_dir>/metrics.json     → summary metrics + run metadata
   • <out_dir>/detections.json  → raw detections with coords & IoU
   • Per-dataset crop folders: pass/ fail/ no_gt/ (unless --dry_run)

Inputs you must provide
-----------------------
• --images_dir   : root folder where images live (may have subfolders per dataset)
• --coco_json    : COCO JSON referencing those images (file_name, optional dataset_name)
• --checkpoint   : .pth weights for the detector
• --out_dir      : where to save crops and JSONs

Key behaviors / assumptions
---------------------------
• Defaults can mirror your service via env vars:
    DETECT_SCORE_THRESH, DETECT_GRAYSCALE, DETECT_WARMUP_ITERS
• --top1 keeps only the best-scoring box per image (service-like);
  use --all to keep all boxes ≥ score threshold.
• We only do IoU-based pass/fail (single category assumption).
• Evaluation uses VOC 11-point AP at the chosen IoU threshold.

Quick start
-----------
python testscript.py \
  --images_dir data/images \
  --coco_json  data/annotations.json \
  --checkpoint checkpoints/model_epoch50.pth \
  --out_dir    eval_out \
  --score_thresh 0.7 --iou_thresh 0.5 --top1

Outputs
-------
• Console progress and counts per image
• Crops under eval_out/<dataset>/{pass,fail,no_gt}/
• JSONs: eval_out/metrics.json and eval_out/detections.json
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ----------------- ROI head (keeps your 256 -> 1024 head shape) -----------------
class ConvHead(nn.Module):
    """
    Tiny ROI head: conv3x3 → ReLU → conv3x3 → ReLU → GAP → Linear(->rep_size).
    Matches the 1024-d representation used during training.
    """
    def __init__(self, in_ch: int, rep_size: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.avg   = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(in_ch, rep_size)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.avg(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ----------------- Model -----------------
def build_model(checkpoint_path: str, device: torch.device, num_classes: int = 2, verbose: bool = False):
    """
    Build Faster R-CNN (MobileNetV3-Large+FPN) with the custom 1024-d head,
    then load your checkpoint (tolerant to common key wrapping patterns).
    """
    det_wts = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    bb_wts  = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=det_wts, weights_backbone=bb_wts)
    in_ch = model.backbone.out_channels  # typically 256
    model.roi_heads.box_head      = ConvHead(in_ch, rep_size=1024)
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, num_classes)

    # Load checkpoint (support dicts like {'state_dict': ...} or raw state dicts)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and any(k in state for k in ("model", "state_dict", "model_state")):
        for key in ("model", "state_dict", "model_state"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:    print("        missing keys (first 8):", missing[:8])
        if unexpected: print("        unexpected keys (first 8):", unexpected[:8])
    model.to(device).eval()
    return model


# ----------------- Path + COCO helpers -----------------
def resolve_img_path(images_root: str, file_name: str, dataset_name: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Resolve an image path robustly:
      1) <root>/<dataset>/<file> (if dataset_name given)
      2) Lower/UPPER dataset variants
      3) Search immediate subfolders for <file>
      4) Fallback: <root>/<file>
    Returns (absolute_path, dataset_used_or_None).
    """
    if dataset_name:
        cand = os.path.abspath(os.path.join(images_root, dataset_name, file_name))
        if os.path.exists(cand): return cand, dataset_name
        lower = os.path.abspath(os.path.join(images_root, str(dataset_name).lower(), file_name))
        upper = os.path.abspath(os.path.join(images_root, str(dataset_name).upper(), file_name))
        if os.path.exists(lower): return lower, str(dataset_name).lower()
        if os.path.exists(upper): return upper, str(dataset_name).upper()
    try:
        for sub in os.listdir(images_root):
            p = os.path.abspath(os.path.join(images_root, sub, file_name))
            if os.path.exists(p): return p, sub
    except FileNotFoundError:
        pass
    return os.path.abspath(os.path.join(images_root, file_name)), dataset_name


def load_coco_annotations(
    coco_json_path: str,
    images_root: str,
    category_name: Optional[str] = None,
    category_id: Optional[int]   = None,
    verbose: bool = False,
) -> Tuple[Dict[str, List[Tuple[float, float, float, float]]], Dict[str, str], List[Tuple[str, Optional[str]]]]:
    """
    Parse a COCO JSON and build:
      - gt_by_file: abs_path -> list of GT boxes (xyxy)
      - img_to_dataset: abs_path -> dataset name used for that image (if known)
      - img_files: [(abs_path, dataset), ...] for every image entry in JSON

    Notes:
      • If both category_name and category_id are absent, use all categories.
      • If category_name is given and id is not, we resolve it from 'categories'.
      • Uses resolve_img_path(...) so paths are robust to folder casing and nesting.
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    name_to_id = {c["name"]: c["id"] for c in coco.get("categories", []) if "id" in c and "name" in c}
    if category_name and category_id is None:
        category_id = name_to_id.get(category_name)

    id_to_imgrec: Dict[int, Tuple[str, Optional[str]]] = {}
    for im in coco.get("images", []):
        id_to_imgrec[im["id"]] = (im.get("file_name", ""), im.get("dataset_name"))

    gt_by_file: Dict[str, List[Tuple[float, float, float, float]]] = {}
    img_to_dataset: Dict[str, str] = {}
    img_files: List[Tuple[str, Optional[str]]] = []

    # Build image list using robust resolver
    for im in coco.get("images", []):
        fn, dset = im.get("file_name", ""), im.get("dataset_name")
        if not fn:
            continue
        path, used = resolve_img_path(images_root, fn, dset)
        img_files.append((path, used))

    # Build GT using resolved paths
    for ann in coco.get("annotations", []):
        if "image_id" not in ann or "bbox" not in ann:
            continue
        if category_id is not None and ann.get("category_id") != category_id:
            continue
        fn, dset = id_to_imgrec.get(ann["image_id"], ("", None))
        if not fn:
            continue
        key, used = resolve_img_path(images_root, fn, dset)
        x, y, w, h = ann["bbox"]
        x1, y1 = float(x), float(y)
        x2, y2 = x1 + float(w), y1 + float(h)
        gt_by_file.setdefault(key, []).append((x1, y1, x2, y2))
        if used:
            img_to_dataset[key] = used

    # Ensure every image has a key (even if it has no GT in this category)
    for path, used in img_files:
        gt_by_file.setdefault(path, gt_by_file.get(path, []))
        if used:
            img_to_dataset.setdefault(path, used)

    if verbose:
        print(f"[JSON] images: {len(img_files)}  annotations-linked: {sum(1 for v in gt_by_file.values() if v)}")
    return gt_by_file, img_to_dataset, img_files


# ----------------- Geometry -----------------
def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """
    Intersection-over-Union between two boxes (x1,y1,x2,y2).
    Returns 0 if no overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


# ----------------- Crop helpers -----------------
def clamp_box(x1, y1, x2, y2, W, H, pad: int = 0) -> Tuple[int, int, int, int]:
    """
    Clamp and (optionally) pad a box to image bounds. Ensures x2>x1 and y2>y1.
    """
    x1 = int(round(x1)) - pad
    y1 = int(round(y1)) - pad
    x2 = int(round(x2)) + pad
    y2 = int(round(y2)) + pad
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def save_crop(bgr: np.ndarray, clamped_box, out_dir: str, base: str, idx: int, score: float, iou: Optional[float],
              dry_run: bool, verbose: bool):
    """
    Save a detection crop with filename encoding score and IoU (or 'na' if none).
    """
    x1, y1, x2, y2 = clamped_box
    iou_str = "na" if iou is None else f"{iou:.3f}"
    fn = f"{base}_det{idx:02d}_sc{score:.3f}_iou{iou_str}.jpg"
    out_path = os.path.join(out_dir, fn)
    if verbose:
        print(f"      -> {out_path}")
    if not dry_run:
        crop = bgr[y1:y2, x1:x2]
        cv2.imwrite(out_path, crop)


# ----------------- AP computation (VOC 11-pt) -----------------
def compute_voc11_ap(det_records, gt_by_file, processed_paths_set, iou_thr: float):
    """
    Compute  mAP over the images processed in this run.

    det_records: list of dicts with keys: img, score, best_iou, best_gt_idx
    gt_by_file : dict path -> list of GT boxes (xyxy)
    processed_paths_set: set of image paths included in this run

    Returns dict with:
      gt_total, tp, fp, recall_max, mAP_voc11, mean_iou_over_TP, detections_after_score_filter
    """
    gt_for_run = {p: gt_by_file.get(p, []) for p in processed_paths_set}
    total_gt = sum(len(v) for v in gt_for_run.values())
    gt_matched = {p: [False]*len(gt_for_run[p]) for p in gt_for_run}

    dets = sorted(det_records, key=lambda d: d["score"], reverse=True)
    tp_flags, fp_flags, ious_tp = [], [], []

    for d in dets:
        p = d["img"]; bgi = d.get("best_gt_idx", -1); biou = float(d.get("best_iou", 0.0))
        if p not in gt_for_run:
            tp_flags.append(0); fp_flags.append(1); continue
        if bgi >= 0 and biou >= iou_thr and not gt_matched[p][bgi]:
            tp_flags.append(1); fp_flags.append(0); gt_matched[p][bgi] = True; ious_tp.append(biou)
        else:
            tp_flags.append(0); fp_flags.append(1)

    precisions, recalls = [], []
    tp_cum = fp_cum = 0
    for tp, fp in zip(tp_flags, fp_flags):
        tp_cum += tp; fp_cum += fp
        denom = tp_cum + fp_cum
        precisions.append(tp_cum / denom if denom > 0 else 0.0)
        recalls.append(tp_cum / total_gt if total_gt > 0 else 0.0)

    ap11 = 0.0
    for thr in [i/10 for i in range(11)]:
        p_at_r = 0.0
        for r, p in zip(recalls, precisions):
            if r >= thr and p > p_at_r: p_at_r = p
        ap11 += p_at_r
    ap11 /= 11.0
    mean_iou_tp = float(sum(ious_tp) / len(ious_tp)) if ious_tp else 0.0

    return {
        "gt_total": int(total_gt),
        "tp": int(sum(tp_flags)),
        "fp": int(sum(fp_flags)),
        "recall_max": float(max(recalls) if recalls else 0.0),
        "mAP_voc11": float(ap11),
        "mean_iou_over_TP": mean_iou_tp,
        "detections_after_score_filter": int(len(dets)),
    }


# ----------------- Main -----------------
def main():
    # Service-aligned defaults via env (can be overridden by CLI)
    env_score = float(os.getenv("DETECT_SCORE_THRESH", "0.7"))
    env_gray  = os.getenv("DETECT_GRAYSCALE", "true").lower() in ("1","true","yes","y","on")
    env_warm  = int(os.getenv("DETECT_WARMUP_ITERS", "0"))

    ap = argparse.ArgumentParser("Service-aligned detector test with IoU eval + JSON outputs")
    ap.add_argument("--images_dir", required=True, help="Root dir containing dataset subfolders (e.g., nia/, sunny/)")
    ap.add_argument("--coco_json", required=True, help="COCO JSON referencing images; optional 'dataset_name' is respected.")
    ap.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint.")
    ap.add_argument("--out_dir", required=True, help="Output folder for crops and JSON files.")

    # Eval thresholds & run options
    ap.add_argument("--score_thresh", type=float, default=env_score, help=f"Default from DETECT_SCORE_THRESH (now {env_score})")
    ap.add_argument("--iou_thresh",   type=float, default=0.50, help="IoU used for pass/fail and AP evaluation.")
    ap.add_argument("--padding", type=int, default=0, help="Pixels of padding around crops.")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (0=all).")
    ap.add_argument("--verbose", action="store_true", help="Print extra details.")
    ap.add_argument("--dry_run", action="store_true", help="Run without writing crops to disk.")
    ap.add_argument("--warmup_iters", type=int, default=env_warm, help=f"Default from DETECT_WARMUP_ITERS (now {env_warm})")

    # top-1 (service-like) vs all detections ≥ threshold
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--top1", action="store_true", default=True, help="Keep only best-scoring box (service-like).")
    g.add_argument("--all",  dest="top1", action="store_false", help="Keep all boxes with score ≥ threshold.")

    # Optional category filter
    ap.add_argument("--coco_category", type=str, default=None, help="Filter annotations by this category name.")
    ap.add_argument("--coco_category_id", type=int, default=None, help="Filter annotations by this numeric category id.")

    # Outputs
    ap.add_argument("--metrics_json", type=str, default=None, help="Default: <out_dir>/metrics.json")
    ap.add_argument("--detections_json", type=str, default=None, help="Default: <out_dir>/detections.json")

    # Preprocessing (service-aligned)
    ap.add_argument("--grayscale", action="store_true", default=env_gray,
                    help=f"Triplicate grayscale (default from DETECT_GRAYSCALE, now {env_gray}).")
    ap.add_argument("--no-grayscale", dest="grayscale", action="store_false")

    args = ap.parse_args()

    if args.verbose:
        print(f"[ARGS] score_thresh={args.score_thresh}  iou_thresh={args.iou_thresh}  top1={args.top1}  pad={args.padding}")
        print(f"[ARGS] grayscale={args.grayscale}  warmup_iters={args.warmup_iters}")
        print(f"[PATH] images_dir={args.images_dir}")
        print(f"[PATH] coco_json={args.coco_json}")
        print(f"[PATH] out_dir={args.out_dir}  dry_run={args.dry_run}")

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path    = args.metrics_json    or os.path.join(args.out_dir, "metrics.json")
    detections_path = args.detections_json or os.path.join(args.out_dir, "detections.json")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(f"[DEV ] device={device}")

    # Model
    model  = build_model(args.checkpoint, device=device, verbose=args.verbose)

    # Load JSON + build image list
    gt_by_file, img_to_dataset, img_files = load_coco_annotations(
        args.coco_json, images_root=args.images_dir,
        category_name=args.coco_category, category_id=args.coco_category_id,
        verbose=args.verbose
    )
    if not img_files:
        raise SystemExit("No images to run (check --images_dir and --coco_json).")

    # ImageNet normalization (must match training)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    # Create per-dataset output folders on demand
    out_cache: Dict[str, Dict[str, str]] = {}
    def ensure_out_dirs(dataset: Optional[str]) -> Dict[str, str]:
        """
        Create and cache pass/fail/no_gt subfolders under out_dir/<dataset>/.
        """
        ds = dataset or "unknown"
        if ds in out_cache: return out_cache[ds]
        base = os.path.join(args.out_dir, ds)
        p = os.path.join(base, "pass"); f = os.path.join(base, "fail"); n = os.path.join(base, "no_gt")
        if not args.dry_run:
            os.makedirs(p, exist_ok=True); os.makedirs(f, exist_ok=True); os.makedirs(n, exist_ok=True)
        out_cache[ds] = {"pass": p, "fail": f, "no_gt": n}
        return out_cache[ds]

    # Optional warmup (on a dummy tensor with the first image's shape)
    if args.warmup_iters > 0:
        # try to read first available image to get shape
        for (p, _) in img_files:
            bgr0 = cv2.imread(p)
            if bgr0 is not None:
                if args.grayscale:
                    gray = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
                    img0 = np.stack([gray, gray, gray], axis=2)
                else:
                    img0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
                ten0 = torch.from_numpy(np.ascontiguousarray(img0)).permute(2, 0, 1).float().div(255)
                ten0 = F.normalize(ten0, mean=norm_mean, std=norm_std).to(device)
                model.eval()
                with torch.no_grad():
                    for _ in range(args.warmup_iters):
                        model([ten0])
                if args.verbose:
                    print(f"[WARM] ran {args.warmup_iters} iters")
                break

    totals = {"pass": 0, "fail": 0, "no_gt": 0}
    th    = float(args.score_thresh)
    iouth = float(args.iou_thresh)

    det_records = []  # for mAP/recall eval
    det_dump    = []  # raw detections dump for detections.json
    processed_paths = []

    N = len(img_files) if args.limit <= 0 else min(args.limit, len(img_files))
    if args.verbose:
        print(f"[RUN ] processing {N} image(s)")

    # ------------- Main inference loop -------------
    for i, (path, dset) in enumerate(img_files[:N], 1):
        processed_paths.append(os.path.abspath(path))
        if args.verbose:
            print(f"\n[{i}/{N}] {dset or 'unknown'} :: {path}")

        bgr = cv2.imread(path)
        if bgr is None:
            print(f"[WARN] cannot read {path}")
            continue

        # Service-like preprocessing (no CLAHE/sharpen)
        if args.grayscale:
            gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            img   = np.stack([gray, gray, gray], axis=2)
        else:
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        ten = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)
        ten = F.normalize(ten, mean=norm_mean, std=norm_std).to(device)

        with torch.no_grad():
            out = model([ten])[0]

        boxes  = out.get("boxes",  torch.empty(0, 4, device=device)).detach().to("cpu").numpy()
        scores = out.get("scores", torch.empty(0,    device=device)).detach().to("cpu").numpy()

        if args.verbose:
            print(f"  raw dets: {len(scores)}")
            if scores.size:
                topk = np.sort(scores)[-5:][::-1]
                print("  top scores (up to 5):", [float(f"{s:.3f}") for s in topk])

        # Keep all scores >= threshold; optionally reduce to top-1
        keep = np.flatnonzero(scores >= th) if scores.size else np.array([], dtype=np.int64)
        if keep.size == 0:
            print(f"[{i}/{N}] {os.path.basename(path)}: no dets >= score {th}")
            continue

        boxes_k  = boxes[keep]
        scores_k = scores[keep]

        order = np.argsort(-scores_k)
        if args.top1:
            order = order[:1]
        if args.verbose:
            print(f"  kept >= {th}: {len(scores_k)}  evaluating/saving: {len(order)}")

        base = os.path.splitext(os.path.basename(path))[0]
        gts  = gt_by_file.get(os.path.abspath(path), [])
        has_gt = len(gts) > 0
        ddirs = ensure_out_dirs(dset)

        H, W = bgr.shape[:2]

        for j, idx in enumerate(order.tolist(), start=1):
            sc = float(scores_k[idx])
            if sc < th:   # safety net
                continue

            box = boxes_k[idx].astype(float)
            clamped = clamp_box(*box, W, H, pad=args.padding)

            # compute best IoU vs GT (if any)
            best_iou = 0.0
            best_gt_idx = -1
            if has_gt:
                for gi, gt in enumerate(gts):
                    val = iou_xyxy(tuple(box), gt)
                    if val > best_iou:
                        best_iou = val
                        best_gt_idx = gi

                passed = best_iou >= iouth
                out_sub = ddirs["pass"] if passed else ddirs["fail"]
                if args.verbose:
                    print(f"    box {j}: score={sc:.3f}  best_iou={best_iou:.3f}  -> {'PASS' if passed else 'FAIL'}")
                save_crop(bgr, clamped, out_sub, base, j, sc, best_iou, args.dry_run, args.verbose)
                totals["pass" if passed else "fail"] += 1
            else:
                if args.verbose:
                    print(f"    box {j}: score={sc:.3f}  (no GT) -> NO_GT")
                save_crop(bgr, clamped, ddirs["no_gt"], base, j, sc, None, args.dry_run, args.verbose)
                totals["no_gt"] += 1

            # record for metrics
            det_records.append({
                "img": os.path.abspath(path),
                "score": sc,
                "best_iou": float(best_iou),
                "best_gt_idx": int(best_gt_idx),
            })
            # record raw detection
            det_dump.append({
                "img": os.path.abspath(path),
                "dataset": dset or "unknown",
                "box_xyxy": [int(clamped[0]), int(clamped[1]), int(clamped[2]), int(clamped[3])],
                "score": sc,
                "best_iou": float(best_iou),
                "best_gt_idx": int(best_gt_idx),
            })

        if not args.verbose:
            print(f"[{i}/{N}] {dset or 'unknown'}/{os.path.basename(path)}: saved {len(order)} crop(s)")

    # ---- Compute metrics (mAP/Recall/GT/TP/FP) over processed images ----
    processed_set = set(processed_paths)
    metrics = compute_voc11_ap(
        det_records=det_records,
        gt_by_file=gt_by_file,
        processed_paths_set=processed_set,
        iou_thr=iouth,
    )
    # Add run metadata
    metrics.update({
        "images_total_in_json": len(img_files),
        "images_processed": len(processed_paths),
        "score_thresh": float(th),
        "iou_thresh": float(iouth),
        "top1_mode": bool(args.top1),
        "grayscale": bool(args.grayscale),
        "warmup_iters": int(args.warmup_iters),
        "crops_saved": dict(totals),
    })

    # Save JSONs
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(detections_path, "w", encoding="utf-8") as f:
        json.dump(det_dump, f, indent=2)

    print("\nDone.")
    print(f"Saved: pass={totals['pass']}  fail={totals['fail']}  no_gt={totals['no_gt']}")
    print(f"[METRICS] wrote {metrics_path}")
    print(f"[DETS   ] wrote {detections_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
