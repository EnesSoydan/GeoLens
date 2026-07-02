# GeoLens

> Retrieval-first visual geo-localization with explainability — MVP on the
> official MSLS **validation** split (**Copenhagen + San Francisco**).

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
Mapillary Street-Level Sequences (**MSLS**), official **validation** cities
Copenhagen (`cph`) and San Francisco (`sf`) — these are the two cities the
standard 740-query MSLS-val protocol evaluates on, and they carry public ground
truth (Amsterdam is a *training* city and out of scope). CC-BY-SA, Mapillary
registration required. The dataset is **license-gated** and not redistributed
here.

1. Register and accept the license at <https://www.mapillary.com/dataset/places>.
2. Download the `train_val` archive.
3. Filter the MSLS-val subset (both cities by default):
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
- **MVP (S0–S5):** MSLS-val (cph+sf), DINOv2+SALAD, FAISS Flat, Eigen-CAM, Gradio.
- **v2 (out of scope):** multi-city, Next.js, PostGIS, Qdrant, VLM rationale.

## Dependencies & Licensing
- **Project code:** MIT (see [LICENSE](LICENSE)).
- **Dataset:** MSLS under CC-BY-SA (Mapillary terms).
- **Model:** the DINOv2 + SALAD descriptor is loaded at runtime via `torch.hub`
  from [`serizba/salad`](https://github.com/serizba/salad), which is licensed
  **GPL-3.0**. SALAD is therefore a runtime dependency of GeoLens (fetched via
  the hub), and it is executed by the served application — including the public
  Hugging Face Spaces demo planned in the deployment design. GeoLens keeps its
  own source under MIT and does not vendor SALAD's code, but the running/served
  application incorporates GPL-3.0 code. For commercial or official use, consult
  legal counsel to confirm exact license compliance.

## License
[MIT](LICENSE) for GeoLens's own code. Third-party components retain their own
licenses — see [Dependencies & Licensing](#dependencies--licensing).
