# SIFT Re‑ID

## Requirements

* Python **3.10**
* Install packages:

```bash
pip install opencv-contrib-python annoy numpy scikit-learn
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

### Build SIFT index (with informativeness filtering)

```bash
python build_sift_index.py \
  --gallery_dir GALLERY \
  --index_path  out/sift_gallery_rootsift_100_unsegmented.ann \
  --meta_path   out/sift_gallery_rootsift_100_unsegmented.json \
  --img_width   256 \
  --max_kpts    100 \
  --num_trees   100 \
  --descriptor  rootsift \
  --final_target_per_class 10000 \
  --k_neighbors 20 \
  --n_splits    5
```

You'll get:

* Annoy index: `out/sift_gallery_rootsift_100_unsegmented.ann`
* Metadata JSON (with build config): `out/sift_gallery_rootsift_100_unsegmented.json`

### Run re‑ID + evaluation

```bash
python query_sift_reid.py eval \
  --test_dir TEST \
  --index_path out/sift_gallery_rootsift_100_unsegmented.ann \
  --meta_path  out/sift_gallery_rootsift_100_unsegmented.json \
  --save_confmat out/confusion_counts.csv \
  --bbox_json bbox_annotations.json \
  --descriptor rootsift --img_width 256 --max_kpts 85 \
  --k_neigh 11 --search_k_mult 32 --per_image_match_cap 30
```

You'll see:

* Per‑query times and descriptor counts
* **Top‑1 / Top‑2** accuracy
* **mAP** (mean Average Precision) and **mean rank**
* CSVs: confusion counts, row‑percent, per‑query votes (by ID and by gallery image), per‑query predictions (with rank + AP), and timings

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
* `--max_kpts` – keep at most this many strongest keypoints per image; default `85`.
* `--num_trees` – Annoy trees; higher = better recall, slower build; default `100`.
* `--descr_dim` – descriptor length; SIFT uses `128` (leave default).
* `--descriptor` – `rootsift` (recommended) or `sift`.
* **Informativeness filtering**:
  * `--final_target_per_class` – if set, filter descriptors using cross-validation margin scoring; keeps this many per class (default: `None`, no filtering).
  * `--k_neighbors` – neighbors used for informativeness scoring; default `20`.
  * `--n_splits` – cross-validation folds for informativeness scoring; default `5`.
  * `--seed` – random seed; default `42`.

### `query_sift_reid.py eval`

* `--test_dir` / `--query_image` – evaluate a directory or a single image.
* `--index_path`, `--meta_path` – paths produced by the build step.
* `--save_confmat` – **base** CSV path; other CSVs use this base name.
* `--descriptor` – must match build (`rootsift` / `sift`).
* `--img_width`, `--max_kpts` – query extraction settings (match build).
* `--k_neigh` – neighbors per descriptor (start at `11`).
* `--search_k_mult` – Annoy search multiplier; `search_k = n_items * mult` (start at `32`).
* `--per_image_match_cap` – cap votes per gallery image to avoid domination (start at `30`).
* `--no_per_id_normalize` – disable per‑ID score normalization (enabled by default).
* `--bbox_json` – JSON file with bounding box annotations for image cropping (default: use the annotations in root).

---

## Metrics explained

* **Top‑1 / Top‑2**: fraction of queries where GT is ranked 1st / within top‑2 by score.
* **mAP** (mean Average Precision): average of `1 / rank(GT)` over all queries (assumes single relevant item per query).
* **mean rank**: average position of ground truth in the rankings (across all queries).
* **Confusion matrix**: rows = GT IDs, columns = predicted IDs (look for a strong diagonal).
* **Per‑query votes**: raw vote counts per ID and per gallery image (helps debug bias).
* **gt_rank** (in per-query predictions CSV): the rank position of the ground truth for that query.
* **ap_single** (in per-query predictions CSV): per-query AP, computed as `1 / rank(GT)` for that query.
* **Timing**: descriptor extraction time and total per‑query time (ms).

## Reproducibility checklist

* Keep `out/sift_gallery_rootsift_100_unsegmented.json` (it stores the build config: `IMG_WIDTH`, `MAX_KPTS`, `NUM_TREES`, descriptor, timestamp).
* Record query settings (`--k_neigh`, `--search_k_mult`, `--per_image_match_cap`, normalization on/off).
* Save the exact command lines used for build and evaluation.

---

## Examples

**Build (with informativeness filtering):**

```bash
python build_sift_index.py --gallery_dir GALLERY \
  --index_path out/sift_gallery_rootsift_100_filtered.ann --meta_path out/sift_gallery_rootsift_100_filtered.json \
  --img_width 256 --max_kpts 100 --num_trees 100 --descriptor rootsift \
  --final_target_per_class 10000 --k_neighbors 20 --n_splits 5
```

**Evaluate directory:**

```bash
python query_sift_reid.py eval --test_dir TEST \
  --index_path out/sift_gallery_rootsift_100_unsegmented.ann --meta_path out/sift_gallery_rootsift_100_unsegmented.json \
  --save_confmat out/confusion_counts.csv \
  --bbox_json bbox_annotations.json \
  --descriptor rootsift --img_width 256 --max_kpts 85 \
  --k_neigh 11 --search_k_mult 32 --per_image_match_cap 30
```

**Evaluate single image:**

```bash
python query_sift_reid.py eval --query_image TEST/NIA/q_001.jpg \
  --index_path out/sift_gallery_rootsift_100_unsegmented.ann --meta_path out/sift_gallery_rootsift_100_unsegmented.json \
  --save_confmat out/single_confusion.csv \
  --descriptor rootsift --img_width 256 --max_kpts 85 \
  --k_neigh 11 --search_k_mult 32 --per_image_match_cap 30
```

---
