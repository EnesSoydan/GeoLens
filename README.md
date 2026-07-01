# GeoLens

> Retrieval-first visual geo-localization with explainability — MVP on the
> MSLS **Amsterdam** subset.

Given a street-level photo, GeoLens retrieves the most visually similar
reference images from a geo-tagged database, predicts a coordinate, and shows
*why* via an Eigen-CAM heatmap and the nearest reference on a map.

<!-- Sprint status: S0 skeleton complete. Sections below are filled per sprint. -->

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [License](#license)

## Overview
<!-- TODO(S2/S4): elevator pitch, hero GIF, key numbers. -->
Retrieval-first pipeline: **DINOv2 ViT-B/14 + SALAD** (8448-dim) embeddings
indexed with **FAISS** (`IndexFlatIP` via `IndexIDMap`). See
[`docs/architecture/`](docs/architecture) for the full design rationale.

## Architecture
<!-- TODO(S2): embed the Mermaid pipeline diagram. -->
Layered FastAPI backend (`router → service → repository`) with a single shared
GPU semaphore; Gradio + Leaflet frontend. See
[`docs/architecture/00-genel-bakis.md`](docs/architecture/00-genel-bakis.md).

## Dataset
Mapillary Street-Level Sequences (**MSLS**), Amsterdam validation city
(CC-BY-SA, Mapillary registration required). The dataset is **license-gated**
and not redistributed here.

1. Register and accept the license at <https://www.mapillary.com/dataset/places>.
2. Download the `train_val` archive.
3. Filter the Amsterdam subset:
   ```bash
   python scripts/download_msls.py --source <extracted_dir_or_zip>
   ```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or CUDA build
pip install -e ".[dev]"
```

## Usage
<!-- TODO(S2/S4): run the API + Gradio UI. -->
```bash
uvicorn app.main:app --reload --app-dir backend
```

## Evaluation
<!-- TODO(S5): official MSLS-val protocol (740 queries, 25 m AND <=40 deg GT),
     Recall@1/5/10 + median km error, NetVLAD baseline on the same subset. -->

## Deployment
<!-- TODO(S5): Docker + Hugging Face Spaces. -->

## Roadmap
- **MVP (S0–S5):** Amsterdam, DINOv2+SALAD, FAISS Flat, Eigen-CAM, Gradio.
- **v2 (out of scope):** multi-city, Next.js, PostGIS, Qdrant, VLM rationale.

## License
[MIT](LICENSE). Dataset under its own CC-BY-SA terms.
