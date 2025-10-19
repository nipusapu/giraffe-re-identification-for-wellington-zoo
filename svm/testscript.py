#!/usr/bin/env python3
"""
HOG + Linear SVM sliding-window detector script
What this script does
---------------------
1) Loads a **LinearSVC** model trained on HOG features.
2) Scans each input image with a multi-scale sliding window (pyramid + step).
3) Applies an optional gradient-magnitude prefilter to skip low-texture windows.
4) Runs per-image NMS and **keeps only the single top-1 detection** per image.
5) Evaluates detections against COCO-format ground truth:
      • Average Precision (VOC 11-point) at a user-chosen IoU
      • Mean IoU over true positives
6) Measures timing (per image + summary) and, if psutil is installed, CPU% and RAM RSS.
7) Logs per-image and summary metrics to **Weights & Biases** (W&B).
8) Optionally saves visualization images with GT (red) and top-1 detection (green).

Inputs you must provide
-----------------------
• --model_path      : joblib-saved LinearSVC trained on matching HOG params
• --images_dir      : folder containing the evaluation images
• --coco_json       : COCO annotations file matching those images
• (Pick one) --coco_category  or  --coco_category_id  to filter the class

Key behaviors / assumptions
---------------------------
• HOG size and other HOG params **must match training** (feature length check included).
• If --max_side > 0, images are downscaled for speed; boxes are remapped to original size.
• NMS is applied, then the top-scoring box is retained (0 or 1 box per image).
• AP is computed using the VOC 11-point interpolation at --eval_iou.

Quick start
-----------
python testscript.py \
  --model_path path/to/hog_svm.pkl \
  --images_dir path/to/images \
  --coco_json  path/to/annotations.json \
  --coco_category giraffe_flank \
  --hog_size 128 128 \
  --wandb_project msc-detector \
  --save_vis out_vis

Outputs
-------
• Console summary (timings, AP, IoU, TP/FP, CPU/RAM)
• W&B logs: per-image + summary metrics
• Optional visualizations written to --save_vis
"""

import os, glob, time, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2, joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC

# psutil is optional; if present we sample CPU% and RAM RSS
try:
    import psutil   # type: ignore
except Exception:
    psutil = None

# ------------------- HOG defaults -------------------
DEFAULT_ORIENTATIONS = 6
DEFAULT_PX_PER_CELL = (4, 4)
DEFAULT_CELLS_PER_BLOCK = (2, 2)
DEFAULT_BLOCK_NORM = 'L2-Hys'


# ------------------- IoU / NMS utilities -------------------
def iou_xyxy(a, b) -> float:
    """
    Intersection-over-Union between two boxes in (x1,y1,x2,y2).
    Returns 0.0 if there is no overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / (denom + 1e-9) if denom > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """
    Greedy Non-Maximum Suppression; returns indices to keep.
    """
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        rest = order[1:]
        if len(rest) == 0:
            break
        # Vectorized IoU vs remaining boxes
        xx1 = np.maximum(boxes[i,0], boxes[rest,0])
        yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2])
        yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_r = (boxes[rest,2]-boxes[rest,0]) * (boxes[rest,3]-boxes[rest,1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return keep


# ------------------- COCO GT loader -------------------
def load_coco_annotations(coco_json_path: str, images_dir: str,
                          category_name: Optional[str] = None,
                          category_id: Optional[int] = None
                          ) -> Dict[str, List[Tuple[float,float,float,float]]]:
    """
    Load COCO JSON and collect GT boxes for the chosen category.
    Returns a dict: {abs_image_path: [(x1,y1,x2,y2), ...], ...}
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Resolve category id from name if not provided
    name_to_id = {c["name"]: c["id"] for c in coco.get("categories", []) if "id" in c and "name" in c}
    if category_name and category_id is None:
        category_id = name_to_id.get(category_name)

    id_to_fname = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    gt_by_file: Dict[str, List[Tuple[float,float,float,float]]] = {}

    # Gather annotations (COCO bbox is xywh; convert to xyxy)
    for ann in coco.get("annotations", []):
        if "image_id" not in ann or "bbox" not in ann:
            continue
        if category_id is not None and ann.get("category_id") != category_id:
            continue
        fname = id_to_fname.get(ann["image_id"])
        if not fname:
            continue
        x, y, w, h = ann["bbox"]
        x1, y1 = float(x), float(y)
        x2, y2 = x1 + float(w), y1 + float(h)
        key = os.path.abspath(os.path.join(images_dir, fname))
        gt_by_file.setdefault(key, []).append((x1, y1, x2, y2))

    # Ensure every image listed in JSON has a key (possibly with empty list)
    for img in coco.get("images", []):
        key = os.path.abspath(os.path.join(images_dir, img["file_name"]))
        gt_by_file.setdefault(key, gt_by_file.get(key, []))
    return gt_by_file


# ------------------- AP (VOC 11-pt) -------------------
class Prediction:
    """
    Simple container for a single detection used by AP computation.
    """
    def __init__(self, image_path: str, bbox: Tuple[float,float,float,float], score: float):
        self.image_path = image_path
        self.bbox = bbox
        self.score = score


def compute_ap_iou(preds: List[Prediction],
                   gt_by_file: Dict[str, List[Tuple[float,float,float,float]]],
                   iou_thresh: float = 0.5) -> Tuple[float, float, int, int]:
    """
    Compute VOC 11-point AP at a fixed IoU threshold and report:
    (AP_11pt, mean_iou_over_true_positives, TP_count, FP_count)
    """
    if not preds:
        return 0.0, 0.0, 0, 0

    preds_sorted = sorted(preds, key=lambda p: p.score, reverse=True)
    gt_matched = {k: [False]*len(v) for k, v in gt_by_file.items()}
    tp_flags, fp_flags, ious_tp = [], [], []

    # Greedy matching by score
    for p in preds_sorted:
        key = os.path.abspath(p.image_path)
        gts = gt_by_file.get(key, [])
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            iou = iou_xyxy(p.bbox, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0 and not gt_matched[key][best_j]:
            tp_flags.append(1); fp_flags.append(0)
            gt_matched[key][best_j] = True
            ious_tp.append(best_iou)
        else:
            tp_flags.append(0); fp_flags.append(1)

    total_gt = sum(len(v) for v in gt_by_file.values())
    tp_cum = 0; fp_cum = 0; precisions = []; recalls = []
    for tp, fp in zip(tp_flags, fp_flags):
        tp_cum += tp; fp_cum += fp
        precisions.append(tp_cum / max(tp_cum + fp_cum, 1e-9))
        recalls.append(tp_cum / max(total_gt, 1e-9))

    # VOC 11-point interpolation
    ap = 0.0
    for thr in [i / 10 for i in range(11)]:
        p_at_r = 0.0
        for r, p in zip(recalls, precisions):
            if r >= thr and p > p_at_r:
                p_at_r = p
        ap += p_at_r
    ap /= 11.0

    mean_iou = float(np.mean(ious_tp)) if ious_tp else 0.0
    return ap, mean_iou, int(sum(tp_flags)), int(sum(fp_flags))


# ------------------- CPU/RAM tracker -------------------
class CpuUsageTracker:
    """
    Lightweight process CPU% (approx) and RAM RSS sampler.
    Uses process_time/perf_counter for a portable CPU% estimate; psutil for RAM samples.
    """
    def __init__(self):
        self.t0_wall = time.perf_counter()
        self.t0_cpu  = time.process_time()
        self.proc = psutil.Process() if psutil else None
        if self.proc:
            try:
                self.proc.cpu_percent(interval=None)  # prime psutil sampling
            except Exception:
                pass
        self.rss_samples = []

    def sample_mem(self):
        if not self.proc:
            return
        try:
            self.rss_samples.append(self.proc.memory_info().rss / (1024 * 1024))
        except Exception:
            pass

    def overall_cpu_percent(self) -> float:
        dtw = max(1e-9, time.perf_counter() - self.t0_wall)
        dtc = max(0.0, time.process_time() - self.t0_cpu)
        return (dtc / dtw) * 100.0

    def psutil_cpu_percent(self, interval: float = 0.1):
        if not self.proc:
            return None
        try:
            return self.proc.cpu_percent(interval=interval)
        except Exception:
            return None


# ------------------- Detector -------------------
def detect_image(
    img_bgr: np.ndarray,
    clf: LinearSVC,
    hog_size: Tuple[int,int],
    step: int,
    pyramid_scale: float,
    score_thresh: float,
    orientations: int,
    px_per_cell: Tuple[int,int],
    cells_per_block: Tuple[int,int],
    block_norm: str,
    prefilter_percentile: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run multi-scale sliding-window detection.
    Returns:
        boxes_xyxy (M,4) in the ORIGINAL image coordinate system
        scores (M,)
    Notes:
        • Uses a gradient-magnitude integral image to skip low-texture windows quickly.
        • Applies the LinearSVC in a "manual" way: score = dot(w, feat) + b for speed.
    """
    H0, W0 = img_bgr.shape[:2]
    gray0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Precompute gradient magnitude + integral image (once per image; reused across scales)
    gx = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=1)
    mag0 = cv2.add(np.abs(gx), np.abs(gy))
    thr_pixel = 0.0
    if prefilter_percentile > 0:
        thr_pixel = float(np.percentile(mag0, prefilter_percentile))
    int0 = cv2.integral(mag0)  # shape (H0+1, W0+1), float64

    boxes, scores = [], []
    scale = 1.0
    win_w, win_h = hog_size

    # Pull classifier weights once
    w = clf.coef_.reshape(-1).astype(np.float32)
    b = float(clf.intercept_[0])

    while True:
        H = int(H0 / scale); W = int(W0 / scale)
        if H < win_h or W < win_w:
            break

        # Resize image view for this scale (keeps coordinates compact)
        gray = cv2.resize(gray0, (W, H), interpolation=cv2.INTER_AREA)

        # s maps this scale's window back to ORIGINAL coords
        s = scale
        for y in range(0, H - win_h + 1, step):
            y1o = int(y * s); y2o = int((y + win_h) * s)
            for x in range(0, W - win_w + 1, step):
                # Cheap prefilter: skip windows whose mean gradient magnitude is below percentile threshold
                if prefilter_percentile > 0:
                    x1o = int(x * s); x2o = int((x + win_w) * s)
                    S = int0[y2o, x2o] - int0[y1o, x2o] - int0[y2o, x1o] + int0[y1o, x1o]
                    mean_mag = S / float((x2o - x1o) * (y2o - y1o) + 1e-9)
                    if mean_mag < thr_pixel:
                        continue

                patch = gray[y:y+win_h, x:x+win_w]
                feat = hog(
                    patch,
                    orientations=orientations,
                    pixels_per_cell=px_per_cell,
                    cells_per_block=cells_per_block,
                    block_norm=block_norm,
                    transform_sqrt=True
                ).astype(np.float32, copy=False)

                # Guard against HOG length mismatches
                if feat.shape[0] != w.shape[0]:
                    continue

                sscore = float(np.dot(feat, w) + b)
                if sscore >= score_thresh:
                    x1 = int(x * s); y1 = int(y * s)
                    x2 = int((x + win_w) * s); y2 = int((y + win_h) * s)
                    boxes.append((x1, y1, x2, y2))
                    scores.append(sscore)

        scale *= pyramid_scale

    if not boxes:
        return np.zeros((0,4), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    return np.array(boxes, dtype=np.int32), np.array(scores, dtype=np.float32)


# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser(
        description="Fast HOG+SVM sliding-window detection with optional downscale + prefilter, "
                    "and evaluation (AP@IoU, mean IoU of TPs, CPU/RAM)."
    )
    ap.add_argument("--model_path", required=True, help="Path to joblib LinearSVC model (trained on matching HOG params).")
    ap.add_argument("--images_dir", required=True, help="Directory with images referenced by the COCO JSON.")
    ap.add_argument("--glob", default="*.jpg;*.jpeg;*.png;*.JPG;*.JPEG;*.PNG",
                    help="Optional glob filter (semicolon-separated). Applied after matching JSON files.")

    # HOG params (must match training)
    ap.add_argument("--hog_size", type=int, nargs=2, default=(128,128), metavar=("W","H"),
                    help="Sliding-window size (W H) used for HOG feature extraction.")
    ap.add_argument("--orientations", type=int, default=DEFAULT_ORIENTATIONS, help="HOG orientations (e.g., 6 or 9).")
    ap.add_argument("--px_per_cell", type=int, nargs=2, default=DEFAULT_PX_PER_CELL, help="HOG pixels per cell (PX PY).")
    ap.add_argument("--cells_per_block", type=int, nargs=2, default=DEFAULT_CELLS_PER_BLOCK, help="HOG cells per block (CX CY).")
    ap.add_argument("--block_norm", type=str, default=DEFAULT_BLOCK_NORM, help="HOG block normalization (e.g., L2-Hys).")

    # Sliding window / detection controls
    ap.add_argument("--step", type=int, default=16, help="Stride in pixels between windows at each scale.")
    ap.add_argument("--pyramid_scale", type=float, default=1.5, help="Scale factor (>1). Larger means fewer scales (faster).")
    ap.add_argument("--score_thresh", type=float, default=0.0, help="Linear score threshold; keep windows with score >= threshold.")
    ap.add_argument("--nms_iou", type=float, default=0.3, help="NMS IoU threshold.")
    ap.add_argument("--max_images", type=int, default=0, help="If >0, evaluate only the first N images.")
    ap.add_argument("--save_vis", type=str, default=None, help="If set, save visualizations to this directory.")
    ap.add_argument("--max_side", type=int, default=0, help="If >0, downscale images so max(H,W) <= max_side for speed.")
    ap.add_argument("--prefilter_percentile", type=float, default=0.0,
                    help="0 disables. Typical 60–80 skips low-texture windows via gradient-magnitude percentile.")

    # Visualization cosmetics
    ap.add_argument("--vis_det_color", type=int, nargs=3, default=(0, 255, 0), metavar=("B","G","R"),
                    help="Detection box color (B G R). Default: 0 255 0 (green).")
    ap.add_argument("--vis_gt_color", type=int, nargs=3, default=(0, 0, 255), metavar=("B","G","R"),
                    help="Ground-truth box color (B G R). Default: 0 0 255 (red).")
    ap.add_argument("--vis_det_thickness", type=int, default=6,
                    help="Line thickness for detection boxes. Default: 6.")
    ap.add_argument("--vis_gt_thickness", type=int, default=5,
                    help="Line thickness for ground-truth boxes. Default: 5.")

    # Evaluation (COCO)
    ap.add_argument("--coco_json", required=True, help="COCO JSON with ground-truth boxes.")
    ap.add_argument("--coco_category", type=str, default=None, help="Category name (use if you know the name).")
    ap.add_argument("--coco_category_id", type=int, default=None, help="Category id (use if you know the id).")
    ap.add_argument("--eval_iou", type=float, default=0.5, help="IoU threshold for TP matching and AP computation.")

    # Logging (W&B)
    ap.add_argument("--wandb_project", type=str, required=True, help="W&B project name.")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name.")
    args = ap.parse_args()

    # Initialize W&B
    import wandb
    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # CPU/RAM tracker
    cpu = CpuUsageTracker()

    # Load model and sanity-check expected HOG length
    clf: LinearSVC = joblib.load(args.model_path)
    if not hasattr(clf, "coef_"):
        raise SystemExit("Model does not look like a LinearSVC (missing coef_).")

    W, H = args.hog_size
    dummy = np.zeros((H, W), dtype=np.uint8)
    feat_len = hog(
        dummy, orientations=args.orientations,
        pixels_per_cell=tuple(args.px_per_cell),
        cells_per_block=tuple(args.cells_per_block),
        block_norm=args.block_norm,
        transform_sqrt=True
    ).shape[0]
    expected = clf.coef_.shape[1]
    if feat_len != expected:
        raise SystemExit(f"HOG length {feat_len} != model expected {expected}. "
                         f"Match --hog_size and HOG params to training.")

    # Load COCO ground truth
    gt_by_file = load_coco_annotations(
        args.coco_json, images_dir=args.images_dir,
        category_name=args.coco_category, category_id=args.coco_category_id
    )
    total_gt = sum(len(v) for v in gt_by_file.values())

    # Build image list from JSON (safer than free-globbing), then optionally filter by glob
    img_list = sorted(gt_by_file.keys())
    if args.glob:
        patterns = [g.strip() for g in args.glob.split(";") if g.strip()]
        allowed = set()
        for pat in patterns:
            for p in glob.glob(os.path.join(args.images_dir, pat)):
                allowed.add(os.path.abspath(p))
        img_list = [p for p in img_list if p in allowed] if allowed else img_list

    if args.max_images > 0:
        img_list = img_list[:args.max_images]

    if not img_list:
        raise SystemExit("No images to evaluate (check --images_dir, --coco_json, and --glob).")

    if args.save_vis:
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)

    det_times, total_times, num_dets_list = [], [], []
    preds: List[Prediction] = []

    # ------------------- Per-image loop -------------------
    for idx, p in enumerate(img_list, start=1):
        t_total0 = time.perf_counter()
        img0 = cv2.imread(p)
        if img0 is None:
            print(f"[WARN] cannot read {p}")
            continue

        # Optional downscale for speed; remember scale to map boxes back
        scale_to_orig = (1.0, 1.0)
        if args.max_side and max(img0.shape[:2]) > args.max_side:
            h, w = img0.shape[:2]
            if h >= w:
                new_h = args.max_side; new_w = int(round(w * (args.max_side / h)))
            else:
                new_w = args.max_side; new_h = int(round(h * (args.max_side / w)))
            img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_to_orig = (w / new_w, h / new_h)  # (sx, sy)
        else:
            img = img0

        # Detect
        t0 = time.perf_counter()
        boxes_s, scores = detect_image(
            img, clf,
            hog_size=tuple(args.hog_size),
            step=args.step,
            pyramid_scale=args.pyramid_scale,
            score_thresh=args.score_thresh,
            orientations=args.orientations,
            px_per_cell=tuple(args.px_per_cell),
            cells_per_block=tuple(args.cells_per_block),
            block_norm=args.block_norm,
            prefilter_percentile=args.prefilter_percentile,
        )

        # Map boxes back to ORIGINAL image if we downscaled
        if len(boxes_s):
            sx, sy = scale_to_orig
            boxes = boxes_s.astype(np.float32)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            boxes = boxes.astype(np.int32)
        else:
            boxes = boxes_s

        # NMS and keep only the top-1 detection for evaluation & visualization
        keep = nms(boxes, scores, args.nms_iou) if len(boxes) else []
        if len(keep):
            boxes_nms = boxes[keep]; scores_nms = scores[keep]
            top = int(np.argmax(scores_nms))
            boxes_top1 = boxes_nms[top:top+1]; scores_top1 = scores_nms[top:top+1]
        else:
            boxes_top1 = np.zeros((0,4), dtype=np.int32)
            scores_top1 = np.zeros((0,), dtype=np.float32)

        t1 = time.perf_counter()

        # Timing
        detect_ms = (t1 - t0) * 1000.0
        total_ms  = (time.perf_counter() - t_total0) * 1000.0
        det_times.append(detect_ms); total_times.append(total_ms)
        num = int(len(boxes_top1)); num_dets_list.append(num)

        # RAM sample
        cpu.sample_mem()

        # Per-image W&B log (+prediction collection for AP)
        log_data = {"image_index": idx, "detect_time_ms": detect_ms, "total_time_ms": total_ms, "num_detections": num}
        if num == 1:
            x1, y1, x2, y2 = map(int, boxes_top1[0])
            sc = float(scores_top1[0])
            log_data.update({"top1_score": sc, "top1_x1": x1, "top1_y1": y1, "top1_x2": x2, "top1_y2": y2})
            preds.append(Prediction(os.path.abspath(p), (float(x1), float(y1), float(x2), float(y2)), sc))
        wandb.log(log_data)

        # Optional visualization
        if args.save_vis:
            vis = img0.copy()
            # Draw GT in RED
            gts = gt_by_file.get(os.path.abspath(p), [])
            for gx1, gy1, gx2, gy2 in gts:
                cv2.rectangle(
                    vis, (int(gx1), int(gy1)), (int(gx2), int(gy2)),
                    tuple(args.vis_gt_color), int(args.vis_gt_thickness)
                )
            # Draw top-1 detection in GREEN (score as text)
            if num == 1:
                x1, y1, x2, y2 = map(int, boxes_top1[0])
                sc = float(scores_top1[0])
                cv2.rectangle(
                    vis, (x1, y1), (x2, y2),
                    tuple(args.vis_det_color), int(args.vis_det_thickness)
                )
                cv2.putText(
                    vis, f"{sc:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    tuple(args.vis_det_color),
                    max(1, int(args.vis_det_thickness // 2)),
                    cv2.LINE_AA
                )
            out_p = os.path.join(args.save_vis, Path(p).stem + "_det.jpg")
            cv2.imwrite(out_p, vis)

        print(f"[{idx}/{len(img_list)}] {Path(p).name}: det={detect_ms:.1f} ms total={total_ms:.1f} ms dets={num}")

    # -------- Timing summary (based on detection stage only) --------
    det_arr = np.array(det_times, dtype=np.float32) if det_times else np.array([], dtype=np.float32)
    mean_ms   = float(det_arr.mean()) if det_arr.size else 0.0
    median_ms = float(np.median(det_arr)) if det_arr.size else 0.0
    p95_ms    = float(np.percentile(det_arr, 95)) if det_arr.size else 0.0
    fps_avg   = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    # -------- AP / IoU evaluation --------
    ap_val, mean_iou_tp, tp, fp = compute_ap_iou(preds, gt_by_file, iou_thresh=args.eval_iou)

    # -------- CPU / RAM --------
    cpu_overall = cpu.overall_cpu_percent()
    cpu_ps = cpu.psutil_cpu_percent(0.1) if psutil else None
    ram_mean = ram_p95 = ram_peak = 0.0
    if cpu.rss_samples:
        rss = np.array(cpu.rss_samples, dtype=np.float32)
        ram_mean = float(rss.mean()); ram_p95 = float(np.percentile(rss, 95)); ram_peak = float(rss.max())

    # -------- Print + W&B --------
    print("\n==== SUMMARY (HOG+SVM, fast) ====")
    print(f"Inference: mean={mean_ms:.1f} ms   median={median_ms:.1f} ms   p95={p95_ms:.1f} ms   fps≈{fps_avg:.2f}")
    print(f"Eval: AP@{args.eval_iou:.2f}={ap_val:.3f}   mean IoU(TP)={mean_iou_tp:.3f}   TP={tp} FP={fp}  GT={total_gt}")
    print(f"CPU overall: {cpu_overall:.1f}%")
    if cpu_ps is not None:
        print(f"CPU psutil: {cpu_ps:.1f}%")
    if cpu.rss_samples:
        print(f"RAM RSS: mean={ram_mean:.1f} MB  p95={ram_p95:.1f} MB  peak={ram_peak:.1f} MB")
    else:
        print("RAM RSS: psutil not available (install psutil).")

    wandb.log({
        "mean_detect_time_ms": mean_ms,
        "median_detect_time_ms": median_ms,
        "p95_detect_time_ms": p95_ms,
        "fps_avg": fps_avg,
        "num_images": len(img_list),
        "total_detections": int(sum(num_dets_list)),

        "AP": ap_val,
        "AP_iou_thresh": args.eval_iou,
        "mean_iou_tp": mean_iou_tp,
        "tp": tp, "fp": fp, "num_gt": total_gt,

        "cpu_overall_percent": cpu_overall,
        **({"cpu_psutil_percent": cpu_ps} if cpu_ps is not None else {}),
        "mem_rss_mb_mean": ram_mean,
        "mem_rss_mb_p95": ram_p95,
        "mem_rss_mb_peak": ram_peak,
    })

    run.finish()


if __name__ == "__main__":
    main()
