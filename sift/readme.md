# SIFT Re‑ID

## Requirements

* Python **3.10**
* Install packages:

```bash
pip install opencv-contrib-python annoy numpy
```

---

## dir structure

### Gallery (labeled identities)

Each subfolder name is treated as the **identity label**:

```
GALLERY/
  NIA/      img_001.jpg img_002.jpg ...
  SUNNY/    ...
  ZAHARA/   ...
  ZURI/     ...
```

### Queries (evaluation)

Queries are grouped by their ground‑truth identity (inferred from parent folder):

```
TEST/
  NIA/      q_001.jpg q_002.jpg ...
  SUNNY/    ...
  ZAHARA/   ...
  ZURI/     ...
```

> `query_sift_reid.py` infers **ground truth** from each image’s parent folder name — **no COCO JSON needed**.

---

## Quick start

### Build SIFT index

```bash
python build_sift_index.py \
  --gallery_dir GALLERY \
  --index_path  out/sift_gallery.ann \
  --meta_path   out/sift_meta.json \
  --img_width   256 \
  --max_kpts    150 \
  --num_trees   50 \
  --descriptor  rootsift
```

You’ll get:

* Annoy index: `out/sift_gallery.ann`
* Metadata JSON (with build config): `out/sift_meta.json`

### Run re‑ID + evaluation

```bash
python query_sift_reid.py eval \
  --test_dir TEST \
  --index_path out/sift_gallery.ann \
  --meta_path  out/sift_meta.json \
  --save_confmat out/confusion_counts.csv \
  --descriptor rootsift --img_width 256 --max_kpts 150 \
  --k_neigh 7 --search_k_mult 20 --per_image_match_cap 30
```

You’ll see:

* Per‑query times and descriptor counts
* **Top‑1 / Top‑2** accuracy
* CSVs: confusion counts, row‑percent, per‑query votes (by ID and by gallery image), per‑query predictions, and timings

---

## Important: Match SIFT settings

**Descriptor mode must match** between build and query (`--descriptor rootsift` vs `sift`).
Keep `--img_width` and `--max_kpts` consistent across build and query for stable results.
Descriptors are 128‑D (SIFT/RootSIFT) and Annoy metric is **euclidean**.

---

## Useful flags

### `build_sift_index.py`

* `--gallery_dir` – root with one subfolder per identity (e.g., `GALLERY/NIA`).
* `--index_path` – output Annoy index (`.ann`).
* `--meta_path` – output metadata JSON (includes a config snapshot).
* `--img_width` – resize width (keeps aspect) before SIFT; default `256`.
* `--max_kpts` – keep at most this many strongest keypoints per image; default `150`.
* `--num_trees` – Annoy trees; higher = better recall, slower build; default `50`.
* `--descr_dim` – descriptor length; SIFT uses `128` (leave default).
* `--descriptor` – `rootsift` (recommended) or `sift`.

### `query_sift_reid.py eval`

* `--test_dir` / `--query_image` – evaluate a directory or a single image.
* `--index_path`, `--meta_path` – paths produced by the build step.
* `--save_confmat` – **base** CSV path; other CSVs use this base name.
* `--descriptor` – must match build (`rootsift` / `sift`).
* `--img_width`, `--max_kpts` – query extraction settings (match build).
* `--k_neigh` – neighbors per descriptor (start at `7`).
* `--search_k_mult` – Annoy search multiplier; `search_k = n_items * mult` (start at `20`).
* `--per_image_match_cap` – cap votes per gallery image to avoid domination (start at `30`).
* `--no_per_id_normalize` – disable per‑ID score normalization (enabled by default).

---

## Metrics explained

* **Top‑1 / Top‑2**: fraction of queries where GT is ranked 1st / within top‑2 by score.
* **Confusion matrix**: rows = GT IDs, columns = predicted IDs (look for a strong diagonal).
* **Per‑query votes**: raw vote counts per ID and per gallery image (helps debug bias).
* **Timing**: descriptor extraction time and total per‑query time (ms).

## Reproducibility checklist

* Keep `out/sift_meta.json` (it stores the build config: `IMG_WIDTH`, `MAX_KPTS`, `NUM_TREES`, descriptor, timestamp).
* Record your query settings (`--k_neigh`, `--search_k_mult`, `--per_image_match_cap`, normalization on/off).
* Save the exact command lines used for build and evaluation.

---

## Examples

**Build (quick run):**

```bash
python build_sift_index.py --gallery_dir GALLERY \
  --index_path out/sift_gallery.ann --meta_path out/sift_meta.json \
  --img_width 256 --max_kpts 150 --num_trees 50 --descriptor rootsift
```

**Evaluate directory:**

```bash
python query_sift_reid.py eval --test_dir TEST \
  --index_path out/sift_gallery.ann --meta_path out/sift_meta.json \
  --save_confmat out/confusion_counts.csv \
  --descriptor rootsift --img_width 256 --max_kpts 150 \
  --k_neigh 7 --search_k_mult 20 --per_image_match_cap 30
```

**Evaluate single image:**

```bash
python query_sift_reid.py eval --query_image TEST/NIA/q_001.jpg \
  --index_path out/sift_gallery.ann --meta_path out/sift_meta.json \
  --save_confmat out/single_confusion.csv \
  --descriptor rootsift --img_width 256 --max_kpts 150 \
  --k_neigh 7 --search_k_mult 20 --per_image_match_cap 30
```

---
