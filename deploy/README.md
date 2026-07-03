---
title: GeoLens
emoji: 🌍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GeoLens — Hugging Face Space

Retrieval-first visual geo-localization with explainability, served as a single
container (FastAPI API at `/api/v1` + mounted Gradio UI at `/ui`). MVP scope:
MSLS-val Copenhagen + San Francisco.

This file is the Space configuration; the image is built from `deploy/Dockerfile`.

## Deploying this Space

1. **Create a Docker Space** and push this repository to it. Place this file at
   the Space repo root as `README.md` (HF reads the YAML frontmatter above:
   `sdk: docker`, `app_port: 7860`). Point the Space at `deploy/Dockerfile`.

2. **Upload the FAISS index to an HF *dataset* repo** (it is ~639 MB and is not
   baked into the image). From a machine that has the built artifacts:

   ```bash
   huggingface-cli upload <user>/geolens-msls-val-index \
       data/index/index.faiss data/index/metadata.db data/index/manifest.json \
       --repo-type dataset
   ```

3. **Set Space variables/secrets:**
   - `GEOLENS_HF_INDEX_REPO = <user>/geolens-msls-val-index` (variable)
   - `HF_TOKEN` (secret) only if the dataset repo is private.

   On startup `deploy/entrypoint.sh` downloads the three index files into
   `GEOLENS_INDEX_DIR` (default `/app/data/index`) and then launches uvicorn.
   The DINOv2 + SALAD weights (~1.4 GB) are baked into the image at build time.

## Resource fit (free tier: 16 GB RAM / 50 GB disk, CPU-only)

Measured on the reference machine with the image built as-is (pytorch-lightning
kept, no lean-inference rewrite):

| Constraint | Measured | Free-tier limit |
|---|---|---|
| Peak RSS (CPU, model + full 18,916-vec index + one `/predict` w/ heatmap) | ~2.1 GB | 16 GB |
| Image disk (CPU torch + baked weights + fetched index) | ~3 GB | 50 GB |

Both fit with large headroom, so no lean-inference module is needed.

## Dependencies & Licensing

- **Project code:** MIT.
- **Dataset:** MSLS under CC-BY-SA (Mapillary terms).
- **Model:** the DINOv2 + SALAD descriptor is loaded at runtime via `torch.hub`
  from [`serizba/salad`](https://github.com/serizba/salad), which is licensed
  **GPL-3.0**. SALAD is therefore a runtime dependency of GeoLens and is executed
  by this served application — including this public Hugging Face Space. GeoLens
  keeps its own source under MIT and does not vendor SALAD's code, but the
  running/served application incorporates GPL-3.0 code. For commercial or
  official use, consult legal counsel to confirm exact license compliance.
