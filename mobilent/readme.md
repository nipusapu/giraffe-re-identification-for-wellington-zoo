# MobileNetV3 Faster R-CNN — Train & Test

## Requirements

* Python **3.10**
* Install packages:

```bash
# Install PyTorch/torchvision first (pick the command for your OS/CUDA):
# https://pytorch.org/get-started/locally/
pip install torch torchvision

# Then the rest:
pip install pycocotools opencv-python numpy wandb
```

---

## dir structure

### Training (COCO-style)

You need an images root and **two COCO JSONs** (train/val). `file_name` in each JSON should be **relative to `images_dir`**.

```
dataset/
  images/
    NIA/img_0001.jpg
    SUNNY/img_0002.jpg
    ...
  annotations_train.json
  annotations_val.json
```

### Testing (evaluation)

Use one **COCO JSON** for the images you want to evaluate, and point to the same `images_dir` root.

```
dataset/
  images/
    NIA/img_0001.jpg
    SUNNY/img_0002.jpg
    ...
  annotations_eval.json
```

> The test script can optionally respect a `dataset_name` field per image in the JSON to help resolve paths. Keep it simple if you can: just match `file_name` under `images/`.

---

## Quick start

### Train

```bash
python trinscript.py \
  --images_dir dataset/images \
  --train_json dataset/annotations_train.json \
  --val_json   dataset/annotations_val.json \
  --output_dir checkpoints \
  --epochs 50 \
  --batch_size 2 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --project giraffe-reid
```

You’ll get:

* Console logs per epoch (train/val loss)
* W&B logs (`wandb login` to view them online)
* Checkpoints in `checkpoints/` (every 5 epochs + final)

### Test (evaluate a checkpoint)

```bash
python testscript.py \
  --images_dir dataset/images \
  --coco_json  dataset/annotations_eval.json \
  --checkpoint checkpoints/model_epoch50.pth \
  --out_dir    eval_out \
  --score_thresh 0.7 \
  --iou_thresh 0.50 \
  --top1
```

You’ll see:

* Per-image progress and saved crops
* Summary JSONs in `eval_out/`:

  * `metrics.json` (VOC 11-pt AP@IoU, mean IoU(TP), TP/FP, recall, run config)
  * `detections.json` (raw detections with coords, scores, IoU)
* Crops in `eval_out/<dataset>/{pass,fail,no_gt}/`

---

## Important: settings & preprocessing

* **Single class**: scripts are set up for **one foreground class** (background + 1).
* **Grayscale**: training converts images to **grayscale** (replicated to 3 channels) with CLAHE + sharpen; testing uses **service-style grayscale triplication** (no CLAHE/sharpen).
* **Normalization**: both train/test use **ImageNet mean/std** (must match).
* **Head size**: both scripts use a **1024-d ROI head**; checkpoints and test code are aligned.
* **IoU**: evaluation uses **VOC 11-point AP** at your chosen `--iou_thresh` (e.g., 0.50).

---

## Useful flags

### `trinscript.py`

* `--images_dir` – root where all images live.
* `--train_json`, `--val_json` – COCO JSONs for train/val.
* `--output_dir` – where to write `model_epoch*.pth`.
* `--epochs` – total training epochs (default 50).
* `--batch_size` – detector batches are memory-heavy; 2 is a safe start.
* `--lr`, `--weight_decay` – AdamW hyper-params (defaults: 1e-4 / 1e-4).
* `--project` – W&B project name for logging.

*Notes:*

* Backbone is mostly frozen; the **last stages are unfrozen** for fine-tuning.
* LR uses **cosine decay** with **5% warmup** (per step).

### `testscript.py`

* `--images_dir` – images root.
* `--coco_json` – COCO JSON for the eval set (same `file_name` convention).
* `--checkpoint` – the `.pth` file to evaluate.
* `--out_dir` – where to store crops + JSON outputs.
* `--score_thresh` – keep detections with score ≥ this (e.g., 0.7).
* `--iou_thresh` – IoU threshold for PASS/FAIL and AP (e.g., 0.50).
* `--top1` / `--all` – keep only best box per image (service-like) or keep all ≥ threshold.
* `--padding` – pixels to pad crops when saving (cosmetic).
* `--limit` – run only the first N images (debug).
* `--warmup_iters` – optional warmup passes before timing/inference.
* `--grayscale` / `--no-grayscale` – service-style grayscale triplication on/off (defaults can be read from env vars).
* `--coco_category`, `--coco_category_id` – filter GT to a single category if your JSON has many.
* `--metrics_json`, `--detections_json` – custom paths (defaults go inside `out_dir/`).

---

## Metrics explained

* **AP@IoU (VOC 11-point)**: average precision at a fixed IoU (e.g., 0.50).  Higher IoU is stricter → AP usually decreases as IoU increases.
* **mean IoU (TP)**: average IoU over **true positives** only.
* **TP / FP**: counts of correct vs incorrect detections.
* **Recall max**: highest recall achieved along the precision–recall curve.
* **Timing** (from console): per-image progress; total crops saved summary.

---

## Reproducibility checklist

* Save command lines used for **train** and **test**.
* Keep `checkpoints/model_epoch*.pth` with notes:

  * epochs, batch size, lr, weight decay
  * train/val loss curves (in W&B)
  * dataset version + date
* Record test config:

  * `score_thresh`, `iou_thresh`, `top1` vs `all`
  * grayscale on/off, warmup iters
* Store the produced `metrics.json` and `detections.json` alongside the checkpoint.

---

## Examples

**Train (quick, smaller run):**

```bash
python trinscript.py \
  --images_dir dataset/images \
  --train_json dataset/annotations_train.json \
  --val_json   dataset/annotations_val.json \
  --output_dir checkpoints \
  --epochs 20 --batch_size 2 --lr 1e-4 --weight_decay 1e-4 \
  --project giraffe-reid
```

**Test (service-like, top-1 only):**

```bash
python testscript.py \
  --images_dir dataset/images \
  --coco_json  dataset/annotations_eval.json \
  --checkpoint checkpoints/model_epoch20.pth \
  --out_dir    eval_out \
  --score_thresh 0.7 --iou_thresh 0.50 --top1
```

**Test (keep all detections, add padding, warm up):**

```bash
python testscript.py \
  --images_dir dataset/images \
  --coco_json  dataset/annotations_eval.json \
  --checkpoint checkpoints/model_epoch50.pth \
  --out_dir    eval_out_all \
  --score_thresh 0.5 --iou_thresh 0.50 --all \
  --padding 8 --warmup_iters 5
```

---
