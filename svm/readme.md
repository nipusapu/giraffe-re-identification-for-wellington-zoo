# HOG + SVM Trainer & Detector

## Requirements

* Python **3.10**
* Install packages:

```bash
pip install numpy opencv-python scikit-image scikit-learn joblib
# Optional but recommended:
pip install wandb psutil
```

---

## dir structure

### Training (binary classification)

You need **positives** and **negatives** folders in each split:

```
data/
  train/
    positives/   # crops with a giraffe flank
    negatives/   # background / no flank
  val/
    positives/
    negatives/
```

### Testing (detection + evaluation)

You need a folder of images and a **COCO JSON** file that describes the boxes.

```
det_eval/
  images/
    img_0001.jpg
    img_0002.jpg
    ...
  annotations.json   # COCO format (bbox = [x,y,w,h])
```

Minimal COCO example (shortened):

```json
{
  "images": [
    {"id": 1, "file_name": "img_0001.jpg", "width": 1280, "height": 720}
  ],
  "annotations": [
    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [420, 210, 280, 330]}
  ],
  "categories": [
    {"id": 1, "name": "giraffe_flank"}
  ]
}
```

> `testscript.py` matches images by `file_name`. Keep names identical to what’s in the JSON.

---

## Quick start

### Train HOG+SVM

```bash
python trainscript.py \
  --train_dir data/train \
  --val_dir   data/val \
  --hog_size 128 128 \
  --model_out hog_svm.pkl \
  --wandb_project giraffe-flank-svm \
  --wandb_run_name hog-svm-128
```

You’ll get:

* Train/Val accuracy in the console
* Saved model: `hog_svm.pkl`
* Optional W&B logs if you’re logged in (`wandb login`)

### Run detector + evaluation

```bash
python testscript.py \
  --model_path hog_svm.pkl \
  --images_dir det_eval/images \
  --coco_json  det_eval/annotations.json \
  --coco_category giraffe_flank \
  --hog_size 128 128 \
  --eval_iou 0.50 \
  --wandb_project msc-detector \
  --save_vis det_vis
```

You’ll see:

* Per-image times and top-1 detection info
* Summary with **AP@IoU**, **mean IoU (TP)**, **TP/FP**, and timing (mean/median/p95)
* Visualizations in `det_vis/` (GT = red, detection = green)
* Optional W&B charts

---

## Important: Match HOG settings

**HOG parameters must match** between training and testing.
If you trained with `--hog_size 128 128`, `--orientations 6`, `--px_per_cell 4 4`, `--cells_per_block 2 2`, use the **same values** for `testscript.py`.
Otherwise you’ll hit a “HOG length != model expected” error (this is intentional).

---

## Useful flags

### `trainscript.py`

* `--hog_size W H` – resize crops before HOG; gives fixed feature length.
* `--orientations` – gradient bins (6 or 9 are common).
* `--px_per_cell PX PY` – smaller cells capture finer detail (larger feature vectors).
* `--cells_per_block CX CY` – normalization window size.
* `--max_per_class N` – cap images per class (quick tests).
* `--model_out` – where to save the model (`.pkl`).
* `--wandb_project`, `--wandb_run_name` – optional W&B logging.

### `testscript.py`

* `--hog_size W H` – **must match training**; also the sliding-window size.
* `--step` – stride (pixels) between windows. Larger = faster, coarser.
* `--pyramid_scale` – scale factor per level (>1). Bigger = fewer scales, faster.
* `--score_thresh` – raw SVM score threshold (start at `0.0`).
* `--nms_iou` – IoU for NMS (0.3 is a good start).
* `--max_side` – downscale big images so max(H,W) ≤ this.
* `--prefilter_percentile` – skip low-texture windows (e.g., 70–80).
* `--eval_iou` – IoU used for AP (0.50 or 0.75 are typical).
* `--save_vis DIR` – write images with GT (red) + top-1 detection (green).
* `--wandb_project` – optional W&B logging.

---

## Metrics explained

* **AP@IoU**: average precision (VOC 11-point) at a fixed IoU.
  Higher IoU is stricter, so AP usually drops as IoU increases.
* **mean IoU (TP)**: average IoU of the **correct** detections.
* **TP / FP**: counts of correct vs incorrect detections.
* **Timing**: average/median/95th-percentile ms per image.
* **CPU/RAM**: rough usage (needs `psutil` for RAM samples).


## Reproducibility checklist

* Save the exact HOG settings you used.
* Keep `hog_svm.pkl` with a small note:

  * HOG params
  * Train/Val accuracy
  * Dataset version + date

---

## Examples

**Train (quick run):**

```bash
python trainscript.py --train_dir data/train --val_dir data/val \
  --hog_size 128 128 --max_per_class 500 --model_out hog_svm.pkl
```

**Test at IoU 0.50:**

```bash
python testscript.py --model_path hog_svm.pkl \
  --images_dir det_eval/images --coco_json det_eval/annotations.json \
  --coco_category giraffe_flank --hog_size 128 128 --eval_iou 0.50 \
  --save_vis det_vis
```

**Test at IoU 0.75 with speed tweaks:**

```bash
python testscript.py --model_path hog_svm.pkl \
  --images_dir det_eval/images --coco_json det_eval/annotations.json \
  --coco_category giraffe_flank --hog_size 128 128 --eval_iou 0.75 \
  --step 24 --max_side 900 --prefilter_percentile 75 --save_vis det_vis_075
```

---
