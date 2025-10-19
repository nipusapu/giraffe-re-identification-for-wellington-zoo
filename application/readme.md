# Django Web App
---

## What’s inside

```
application/
  api/                 # Django REST API
  reid/tasks/          # Celery tasks: storage, detect, reidentify, pipeline
  config/              # settings, urls, wsgi
  media/               # local file storage
  ui/app/              # Next.js app router

Dockerfile.api         # container for Django API
Dockerfile.web         # container for Next.js web
docker-compose.yml     # Postgres + Redis + API + Web
```

### Data model (simplified)

* **StoredImage** — where/how the file is stored (S3 or local), UUID id.
* **ReIDResult** — a job row: status, predicted animal, votes JSON.
* **Animal** — catalog of known IDs (e.g., `NIA`, `ZURI`).
* **ImageTag** — (optional) tag images with animals (auto/manual).
* **ApiKey** — hashed API keys (for production auth).

### Pipeline (Celery chain)

`upload → StoredImage → ReIDResult(queued) → detect_flank → reid_sift → Completed/Error`

* **detect_flank**: loads the MobileNetV3 Faster‑R‑CNN checkpoint → picks top box (if ≥ threshold) → writes crop.
* **reid_sift**: SIFT/RootSIFT + Annoy voting (reid2 preferred; legacy fallback) → pick best code → persist.
* **storage** helpers: move files to S3 or local, presign URLs.

---

## Requirements

* **Python 3.10+**, **Node 18+**, **Docker** (optional), **Redis**, **Postgres**.
* Python packages are installed via `pip` (or Dockerfile).

```bash
# API
pip install -r requirements.txt

# Web (Next.js)
cd ui && npm install
```

---

## Environment (.env)

Create a file named **`.env`** and load it before running the server.
**Booleans** accept `1/0`, `true/false`.
**Paths** can be **absolute** or **relative** to the project root; `~` is expanded.

### Minimal for local host development

```env
# --- Django ---
SECRET_KEY=dev-insecure
DEBUG=1
ALLOWED_HOSTS=127.0.0.1,localhost
DJANGO_CORS_ALLOWED_ORIGINS=http://localhost:3000
PRESIGN_EXPIRES=300

# --- Database ---
DATABASE_URL=postgres://postgres:postgres@localhost:5432/postgres
DB_CONN_MAX_AGE=60

# --- Redis / Celery ---
REDIS_URL=redis://localhost:6379/0

# --- Detector ---
DETECTOR_CHECKPOINT=./models/model_epoch50.pth
DETECTOR_DEVICE=cpu
DETECT_SCORE_THRESH=0.70
DETECT_GRAYSCALE=1
DETECT_WARMUP_ITERS=0

# --- ReID (SIFT + Annoy) ---
REID_INDEX_PATH=./out/sift_gallery.ann
REID_META_PATH=./out/sift_meta.json
REID_IMG_WIDTH=256
REID_MAX_KPTS=150
REID_TOPK_PER_DESC=7
REID_SEARCH_K_MULT=20
REID_PER_IMAGE_MATCH_CAP=30
REID_PER_ID_NORMALIZE=1
REID_IMPLEMENTATION=

# --- Storage ---
AWS_USE_S3=0
```

### Production template (edit to fit your infra)

```env
# --- Django ---
SECRET_KEY=replace-with-a-strong-secret
DEBUG=0
ALLOWED_HOSTS=api.example.com
DJANGO_CORS_ALLOWED_ORIGINS=https://example.com
PRESIGN_EXPIRES=600

# --- Database (prefer DATABASE_URL) ---
DATABASE_URL=postgres://user:pass@db:5432/app
DB_CONN_MAX_AGE=300

# --- Redis / Celery ---
REDIS_URL=redis://redis:6379/0

# --- Detector ---
DETECTOR_CHECKPOINT=/app/models/model_epoch50.pth
DETECTOR_DEVICE=cuda
DETECT_SCORE_THRESH=0.70
DETECT_GRAYSCALE=1
DETECT_WARMUP_ITERS=2

# --- ReID (SIFT + Annoy) ---
REID_INDEX_PATH=/app/out/sift_gallery.ann
REID_META_PATH=/app/out/sift_meta.json
REID_IMG_WIDTH=256
REID_MAX_KPTS=150
REID_TOPK_PER_DESC=7
REID_SEARCH_K_MULT=20
REID_PER_IMAGE_MATCH_CAP=30
REID_PER_ID_NORMALIZE=1
REID_IMPLEMENTATION=reid2

# --- S3 (turn on for production) ---
AWS_USE_S3=1
AWS_STORAGE_BUCKET_NAME=my-bucket
AWS_S3_REGION_NAME=ap-southeast-2
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
# For LocalStack or MinIO (optional)
# AWS_S3_ENDPOINT_URL=https://s3.localstack.cloud:4566
```

### Variable glossary

| Name                                          |         Type | Default                    | Purpose                                             |
| --------------------------------------------- | -----------: | -------------------------- | --------------------------------------------------- |
| `SECRET_KEY`                                  |          str | `dev-only-insecure-key`    | Django secret (prod: set a strong value)            |
| `DEBUG`                                       |         bool | `True`                     | Development on/off                                  |
| `ALLOWED_HOSTS`                               |          csv | `127.0.0.1,localhost`      | Allowed hostnames for Django                        |
| `DJANGO_CORS_ALLOWED_ORIGINS`                 |          csv | –                          | Web origins allowed to call the API                 |
| `PRESIGN_EXPIRES`                             |      int (s) | `300`                      | Time a presigned URL remains valid                  |
| `DATABASE_URL`                                |          url | –                          | Postgres DSN (preferred over POSTGRES_*)            |
| `DB_CONN_MAX_AGE`                             |      int (s) | `60`                       | Persistent DB connections                           |
| `REDIS_URL`                                   |          url | `redis://localhost:6379/0` | Celery broker/result backend                        |
| `AWS_USE_S3`                                  |         bool | `False`                    | Toggle S3 storage on/off                            |
| `AWS_STORAGE_BUCKET_NAME`                     |          str | –                          | S3 bucket name                                      |
| `AWS_S3_REGION_NAME`                          |          str | `ap-southeast-2`           | S3 region                                           |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` |          str | –                          | S3 credentials                                      |
| `AWS_S3_ENDPOINT_URL`                         |          url | –                          | Custom S3 endpoint (LocalStack/MinIO)               |
| `DETECTOR_CHECKPOINT`                         |         path | –                          | Path to `*.pth` model file                          |
| `DETECTOR_DEVICE`                             | `cpu`/`cuda` | `cpu`                      | Inference device                                    |
| `DETECT_SCORE_THRESH`                         |        float | `0.5`                      | Min score to accept a detection                     |
| `DETECT_GRAYSCALE`                            |         bool | `True`                     | Service-style grayscale triplication                |
| `DETECT_WARMUP_ITERS`                         |          int | `0`                        | Optional warmup passes                              |
| `REID_INDEX_PATH` / `REID_META_PATH`          |         path | –                          | SIFT Annoy index + metadata                         |
| `REID_INDEX_DIM`                              |          int | `128`                      | Descriptor dimensionality                           |
| `REID_INDEX_METRIC`                           |          str | `euclidean`                | Annoy metric                                        |
| `REID_FLIP_QUERY`                             |         bool | `False`                    | Flip query images at test time                      |
| `REID_IMG_WIDTH`                              |          int | `256`                      | Resize width before SIFT                            |
| `REID_MAX_KPTS`                               |          int | `150`                      | Max keypoints/descriptors per image                 |
| `REID_USE_CLAHE`                              |         bool | `True`                     | Apply CLAHE in preproc                              |
| `REID_TOPK_PER_DESC`                          |          int | `7`                        | k-NN per descriptor                                 |
| `REID_ANNOY_SEARCH_K`                         |     int/None | `None`                     | Annoy `search_k` (if 0, use multiplier)             |
| `REID_SEARCH_K_MULT`                          |        float | `20.0`                     | `search_k = n_items * MULT` when above is 0         |
| `REID_PER_IMAGE_MATCH_CAP`                    |          int | `30`                       | Cap votes per gallery image                         |
| `REID_USE_RANK_WEIGHT`                        |         bool | `True`                     | Weight neighbors by `1/rank`                        |
| `REID_PER_ID_NORMALIZE`                       |         bool | `True`                     | Normalise scores by gallery size per ID             |
| `REID_IMPLEMENTATION`                         |          str | `""`                       | Engine selector (`reid2`, `reid1`, blank=auto)      |
| `REID_ASSIGN_THRESHOLD`                       |        float | `0.35`                     | Auto-assign threshold for predicted label (if used) |
| `REID_ALLOW_NUMERIC_PK_MATCH`                 |         bool | `False`                    | Allow numeric pk ID matching (if used)              |
| `REID_TEMP_DIR`                               |    path/None | –                          | Custom temp directory                               |
| `REID_CLEANUP_INTERVAL`                       |      int (s) | `3600`                     | Temp cleanup interval                               |

### Tips

* When running **with Docker Compose**, prefer container paths such as `/app/models/model.pth` and mount host folders via volumes.
* For **local dev**, relative paths like `./output/sift_gallery.ann` are resolved from the project root.
* If you enable S3, make sure the bucket exists and the credentials are valid; otherwise the app falls back to local storage.

## Quick start (Docker Compose)

These steps bring up **Postgres + Redis + Django API + Next.js** using the provided Dockerfiles and `docker-compose.yml`.

**Create `.env`** (see the **Environment** section). For Compose, use service hostnames:

   * `DATABASE_URL=postgres://postgres:postgres@db:5432/postgres`
   * `REDIS_URL=redis://redis:6379/0`
   * Set model paths you will mount (see the *Models* note below).

**Prepare model files** (host machine):

   ```bash
   mkdir -p models checkpoints out
   # put your detector here
   cp /path/to/model_epoch50.pth models/
   # put your SIFT index & meta here
   cp /path/to/sift_gallery.ann out/
   cp /path/to/sift_meta.json    out/
   ```

**Start the stack:**

   ```bash
   docker compose up --build -d
   ```

**Run DB migrations:**

   ```bash
   docker compose exec api python manage.py migrate
   ```

** start the worker:**

   ```bash
   docker compose exec api celery -A celery worker -l info
   ```

**Open the apps:**

   * Web (Next.js): [http://localhost:3000](http://localhost:3000)
   * API (Django):  [http://localhost:8000](http://localhost:8000)

> **Models in containers**: the sample compose mounts `./models` and `./out` into the API container so the API can read `DETECTOR_CHECKPOINT`, `REID_INDEX_PATH`, and `REID_META_PATH`. If your compose doesn’t yet do that, add:
>
> ```yaml
> services:
>   api:
>     volumes:
>       - ./models:/app/models:ro
>       - ./out:/app/out:ro
> ```
>
> Then set in `.env`:
>
> ```env
> DETECTOR_CHECKPOINT=/app/models/model_epoch50.pth
> REID_INDEX_PATH=/app/out/sift_gallery.ann
> REID_META_PATH=/app/out/sift_meta.json
> ```

## Quick start (Local Dev: Python + Node)

### Backend (Django API)

 **Create & activate venv, install deps**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

 **Set environment** (from your `.env`)

```bash
# export everything from .env (bash-compatible)
export $(grep -v '^#' .env | xargs)
```

**Databases & broker**

* EITHER run Postgres/Redis locally, OR use Docker one-liners:

```bash
# Postgres
docker run --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15-alpine
# Redis
docker run --name redis -p 6379:6379 -d redis:7-alpine
```

Update `.env` if needed:

```env
DATABASE_URL=postgres://postgres:postgres@localhost:5432/postgres
REDIS_URL=redis://localhost:6379/0
```

**Migrate**

```bash
python manage.py migrate
```

**Run the API**

```bash
python manage.py runserver 0.0.0.0:8000
```

**Run Celery worker** (another terminal)

```bash
celery -A celery worker -l info
```

**Model files**
Place your detector and SIFT index where your paths point (e.g., `./checkpoints/model.pth`, `./out/sift_gallery.ann`, `./out/sift_meta.json`).

### Front‑end (Next.js)

```bash
cd ui
npm install
npm run dev  # http://localhost:3000
```

**Proxy /api → Django** (optional): add in `next.config.js` so browser calls to `/api/*` hit Django while developing.

```js
module.exports = {
  async rewrites() {
    return [{ source: '/api/:path*', destination: 'http://localhost:8000/api/:path*' }];
  },
};
```

## API reference (minimal)

Base path is typically `/api`.

### `POST /api/upload`

**Body:** `multipart/form-data` with `image` field.
**Returns:** `{ reid_id, image_id, status, image_url }`.
**Notes:**

* Rejects non‑image and files > 10MB.
* Writes to S3 if enabled; otherwise to `MEDIA_ROOT`.
* Enqueues Celery pipeline.

### `GET /api/result/{reid_id}`

**Returns:**

```json
{
  "id": 12,
  "status": "completed",
  "predicted_animal": "ZURI",
  "display_name": "Zuri",
  "description": "…",
  "image_id": "<uuid>",
  "image_url": "https://…",
  "votes": {"ZURI": 12, "NIA": 3}
}
```

### `GET /api/presign/{image_uuid}?expires=300`

Return a **presigned** URL (S3) or a local media URL.

> **Docs:** Swagger/Redoc can be enabled (DRF‑YASG). Common paths: `/swagger/` or `/redoc/`.

---

## Auth & CORS

This API uses **API keys** via the `X-API-Key` header. Keys are stored in the `ApiKey` table and validated by a constant‑time hash. The header format is:

```
X-API-Key: <prefix>.<secret>
```

When a request is authenticated, `request.user` is an anonymous placeholder and the key record is attached to `request.auth`.

### Generate an API key (Django shell)

Open a shell:

```bash
python manage.py shell
```

Run this (replace `<yourapp>` with your app/module name that contains `models.py`):

```python
from <yourapp>.models import ApiKey
import secrets, hashlib

prefix = secrets.token_hex(6)       # short id you can share/log
secret = secrets.token_urlsafe(32)  # store this securely client-side
salt   = secrets.token_hex(16)
hashed = hashlib.sha256((secret + salt).encode()).hexdigest()
rec = ApiKey.objects.create(prefix=prefix, salt=salt, hashed_key=hashed, is_active=True)
print('API KEY (copy now):', f'{prefix}.{secret}')
```

> The **secret is shown only once**. Store it securely (e.g., in a vault or env var). You cannot recover it from the database later.

### Use the key

```bash
# Upload
curl -H "X-API-Key: <prefix>.<secret>" \
     -F "image=@/path/to/photo.jpg" \
     http://localhost:8000/api/upload

# Fetch result
curl -H "X-API-Key: <prefix>.<secret>" \
     http://localhost:8000/api/result/1
```

### Rotate / revoke keys

* **Rotate**: create a new key, update clients, then mark the old record `is_active=False`.
* **Revoke**: set `is_active=False` for the key; it will immediately stop working.]

> Header mapping note: DRF reads `X-API-Key` from `request.META['HTTP_X_API_KEY']` under the hood.

## Detector & Re‑ID configuration

**Detector (MobileNetV3 Faster‑R‑CNN)**

* `DETECTOR_CHECKPOINT` — path to `.pth`.
* `DETECTOR_DEVICE` — `cpu` or `cuda`.
* `DETECT_SCORE_THRESH` — keep best box only if ≥ threshold (service‑like).
* `DETECT_GRAYSCALE` — grayscale triplication to match training.
* `DETECT_WARMUP_ITERS` — optional warmup passes.

**Re‑ID (SIFT + Annoy)**

* `REID_INDEX_PATH` / `REID_META_PATH` — built by the SIFT builder.
* `REID_TOPK_PER_DESC`, `REID_SEARCH_K_MULT`, `REID_PER_IMAGE_MATCH_CAP` — search/voting behavior.
* `REID_PER_ID_NORMALIZE=1` — balances IDs with many photos.
* `REID_IMPLEMENTATION` — `reid2` preferred, `reid1` legacy.

> Keep **build/query** SIFT settings consistent (image width, max kpts, descriptor type).

### Swapping models (Detector & Re‑ID)

** Swap the DETECTOR checkpoint**

1. Put the new `.pth` on disk (or in the mounted volume).
2. Update `.env`:

   ```env
   DETECTOR_CHECKPOINT=/app/models/model_epochNEW.pth   # Docker path
   DETECTOR_DEVICE=cuda   # or cpu
   DETECT_SCORE_THRESH=0.70
   ```
3. **Restart** the API container and Celery worker (Compose):

   ```bash
   docker compose restart api
   docker compose exec api pkill -f 'celery' || true
   docker compose exec -d api celery -A celery worker -l info
   ```

   For local dev, stop/start `runserver` and the Celery process.

** Swap the Re‑ID gallery**

1. Build a new Annoy index + meta (see SIFT README).
2. Update `.env`:

   ```env
   REID_INDEX_PATH=/app/out/sift_gallery_NEW.ann
   REID_META_PATH=/app/out/sift_meta_NEW.json
   REID_PER_ID_NORMALIZE=1
   ```
3. **Restart** the API container and Celery worker.

**Toggle Re‑ID implementation**

* If your code supports both, set:

  ```env
  REID_IMPLEMENTATION=reid2   # preferred
  # REID_IMPLEMENTATION=reid1 # legacy fallback
  ```
* Keep the index/meta compatible with the chosen implementation.


**Rolling changes without downtime (Compose)**

* Build a new image tag with updated defaults, then:

  ```bash
  docker compose pull && docker compose up -d
  ```
* Or restart only the `api` service after updating `.env`/mounted files.

---

## S3 vs Local storage

* Toggle with `AWS_USE_S3`.
* When S3 is **on**, uploads and crops are written **only to S3**; presign with `PRESIGN_EXPIRES`.
* When S3 is **off**, files live under `MEDIA_ROOT` at keys like `uploads/<uuid>.jpg`.
* Crops may be stored at a **fixed key** `uploads/<ORIGINAL_UUID>_cropped.ext` for traceability.

---

## Running the front‑end (Next.js)

Local dev:

```bash
cd ui
npm install
npm run dev   # http://localhost:3000
```