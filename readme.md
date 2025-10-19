# giraffe-re-identification-for-zoo

Identifying giraffes by their spot patterns is vital for research, industry, and education, but many tools are expert-oriented and require training. Zoo visitors often have no simple way to identify an individual on site. Each giraffe has unique markings—spot patterns, horn shape, ear notches—that are easy to miss without guidance during a short visit.

This project delivers a simple three-part solution:

1. a friendly **web page** to upload a photo,
2. a **service** that finds the giraffe’s flank and
3. a **re-identification engine** that matches it against a gallery.

The back end is lightweight and runs on a laptop or in the cloud. It’s budget-friendly (CPU-first processing, on-upload resizing, caching) and returns results in real time with a clear interface for non-experts. The design is modular, so it can extend to zebras, tigers, and beyond with minimal changes.

---

## Table of Contents

- [giraffe-re-identification-for-zoo](#giraffe-re-identification-for-zoo)
  - [Table of Contents](#table-of-contents)
  - [What’s in this repo](#whats-in-this-repo)
  - [System overview](#system-overview)
  - [Folder guide](#folder-guide)
  - [Quick start](#quick-start)
    - [Run the web app with Docker Compose](#run-the-web-app-with-docker-compose)
    - [Run locally (Python + Node)](#run-locally-python--node)
    - [C) Try the algorithms from the CLI](#c-try-the-algorithms-from-the-cli)
  - [Configuration (.env)](#configuration-env)
  - [Swapping models](#swapping-models)

---

## What’s in this repo

* **`application/`** – Django REST API + Celery pipeline + Next.js web UI
* **`mobilent/`** – MobileNetV3 Faster R-CNN training & testing scripts
* **`sift/`** – SIFT/RootSIFT gallery index builder & query evaluator
* **`svm/`** – HOG + Linear SVM trainer & sliding-window detector

Every folder has its **own README** for deep details and examples.

---

## System overview

```
User Upload -> Django API  -> Celery Pipeline -> Detector (MobileNetV3 FRCNN)
            (POST /upload)                     -> Crop giraffe flank (top-1 box)
                                              -> ReID (SIFT + Annoy)
                                              -> Predicted ID + votes
          <- Frontend polls /result/<id> and renders name/info
```

* **Detector**: MobileNetV3-Large + FPN, grayscale pipeline to match training.
* **Re-ID**: RootSIFT descriptors + Annoy nearest neighbors + voting/normalization.
* **Storage**: Local media by default; optional S3 with presigned URLs.
* **Auth**: API key via `X-API-Key: <prefix>.<secret>` (see app README for generation).

---

## Folder guide

| Path           | Purpose                            | Key files                                                                                   |
| -------------- | ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `application/` | Web application (API + tasks + UI) | `api/`, `reid/`, `config/`, `ui/`, `Dockerfile.api`, `Dockerfile.web`, `docker-compose.yml` |
| `mobilent/`    | Detector training & evaluation     | `trinscript.py` (train), `testscript.py` (service-style test)                               |
| `sift/`        | SIFT/RootSIFT Re-ID                | `build_sift_index.py` (build Annoy), `query_sift_reid.py` (evaluate)                        |
| `svm/`         | Baseline HOG + SVM                 | `trainscript.py` (train), `testscript.py` (detect + AP eval)                                |

> See the README inside each folder for usage and examples.

---

## Quick start

### Run the web app with Docker Compose

```bash
# 1) Clone and enter
git clone <your-repo-url> giraffe-re-identification-for-zoo
cd giraffe-re-identification-for-zoo

# 2) Create env file
cp .env.example .env
# edit: SECRET_KEY, DATABASE_URL (use db service), REDIS_URL, model paths, CORS

# 3) Prepare model/index files (mounted into the API)
mkdir -p models out
# put your detector checkpoint and SIFT index/meta here
# e.g., models/model_epoch50.pth, out/sift_gallery.ann, out/sift_meta.json

# 4) Build and run
docker compose up --build -d

# 5) Migrate DB and create admin
docker compose exec api python manage.py migrate
docker compose exec api python manage.py createsuperuser

# 6) (If Celery isn’t a separate service) start worker in the api container
docker compose exec -d api celery -A celery worker -l info
```

Open:

* **Web** (Next.js): [http://localhost:3000](http://localhost:3000)
* **API** (Django): [http://localhost:8000](http://localhost:8000)

### Run locally (Python + Node)

**Backend**

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env, then export (bash):
export $(grep -v '^#' .env | xargs)

# start dependencies (or use local installs)
docker run --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15-alpine
docker run --name redis -p 6379:6379 -d redis:7-alpine

python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

**Celery worker (new terminal)**

```bash
celery -A celery worker -l info
```

**Frontend**

```bash
cd application/ui
npm install
npm run dev  # http://localhost:3000
```

> Tip: In `next.config.js` add a rewrite so `/api/*` hits Django during dev:

```js
module.exports = {
  async rewrites() {
    return [{ source: '/api/:path*', destination: 'http://localhost:8000/api/:path*' }];
  },
};
```

### C) Try the algorithms from the CLI

**SIFT Re-ID**

```bash
# Build gallery index
python sift/build_sift_index.py \
  --gallery_dir GALLERY \
  --index_path  out/sift_gallery.ann \
  --meta_path   out/sift_meta.json \
  --img_width   256 --max_kpts 150 --num_trees 50 --descriptor rootsift

# Evaluate queries (Top-1/Top-2 + confusion CSVs)
python sift/query_sift_reid.py eval \
  --test_dir TEST \
  --index_path out/sift_gallery.ann \
  --meta_path  out/sift_meta.json \
  --save_confmat out/confusion_counts.csv \
  --descriptor rootsift --img_width 256 --max_kpts 150 \
  --k_neigh 7 --search_k_mult 20 --per_image_match_cap 30
```

**MobileNetV3 detector**

```bash
# Train
python mobilent/trinscript.py \
  --images_dir dataset/images \
  --train_json dataset/annotations_train.json \
  --val_json   dataset/annotations_val.json \
  --output_dir checkpoints --epochs 50 --batch_size 2 --lr 1e-4

# Test (service-style top-1)
python mobilent/testscript.py \
  --images_dir dataset/images \
  --coco_json  dataset/annotations_eval.json \
  --checkpoint checkpoints/model_epoch50.pth \
  --out_dir    eval_out --score_thresh 0.7 --iou_thresh 0.50 --top1
```

**HOG + SVM baseline**

```bash
# Train
python svm/trainscript.py \
  --train_dir data/train --val_dir data/val \
  --hog_size 128 128 --model_out hog_svm.pkl

# Detect + evaluate
python svm/testscript.py \
  --model_path hog_svm.pkl \
  --images_dir det_eval/images \
  --coco_json  det_eval/annotations.json \
  --coco_category giraffe_flank \
  --hog_size 128 128 --eval_iou 0.50 --save_vis det_vis
```

---

## Configuration (.env)

A complete example lives at **`.env.example`**—copy it to `.env` and edit.

Key settings you’ll likely touch:

* **Detector**
  `DETECTOR_CHECKPOINT`, `DETECTOR_DEVICE` (`cpu`/`cuda`), `DETECT_SCORE_THRESH`, `DETECT_GRAYSCALE`

* **Re-ID (SIFT + Annoy)**
  `REID_INDEX_PATH`, `REID_META_PATH`, `REID_TOPK_PER_DESC`, `REID_SEARCH_K_MULT`, `REID_PER_IMAGE_MATCH_CAP`, `REID_PER_ID_NORMALIZE`

* **Storage**
  `AWS_USE_S3` (+ `AWS_*` when on) or keep local media

* **Auth & CORS**
  API keys via `X-API-Key`. Allowed web origins via `DJANGO_CORS_ALLOWED_ORIGINS`.

> The Django README (in `application/`) includes step-by-step **API key generation** and **S3 vs local storage** notes.

---

## Swapping models

* **Change detector**
  Drop the new `.pth` into `models/` (or your mounted folder) and set:

  ```
  DETECTOR_CHECKPOINT=/app/models/model_epochNEW.pth
  DETECTOR_DEVICE=cpu|cuda
  ```

  Restart the API and the Celery worker (Compose: `docker compose restart api`).

* **Change Re-ID gallery**
  Rebuild Annoy with `sift/build_sift_index.py`, then set:

  ```
  REID_INDEX_PATH=/app/out/sift_gallery_NEW.ann
  REID_META_PATH=/app/out/sift_meta_NEW.json
  ```

  Restart the API/worker.

* **Toggle engine**

  ```
  REID_IMPLEMENTATION=reid2   # preferred (or reid1)
  ```

---

