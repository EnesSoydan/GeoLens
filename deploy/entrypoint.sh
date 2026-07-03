#!/usr/bin/env bash
# Container startup: make sure the FAISS index artifacts are present, then serve.
#
# The ~639 MB index (index.faiss + metadata.db + manifest.json) is NOT baked into
# the image. Per docs/architecture/07-deployment.md it is pulled from a Hugging
# Face Hub *dataset* repo at startup and cached on the Space's persistent disk.
# Set GEOLENS_HF_INDEX_REPO to that dataset repo id (e.g. "user/geolens-msls-val-index").
# If the artifacts already exist (baked/mounted), the fetch is skipped.
set -euo pipefail

INDEX_DIR="${GEOLENS_INDEX_DIR:-/app/data/index}"
mkdir -p "${INDEX_DIR}"

if [[ ! -f "${INDEX_DIR}/index.faiss" ]]; then
    if [[ -n "${GEOLENS_HF_INDEX_REPO:-}" ]]; then
        echo "entrypoint: fetching index artifacts from ${GEOLENS_HF_INDEX_REPO} -> ${INDEX_DIR}"
        # Use the stable hf_hub_download API (the CLI name drifts across
        # huggingface_hub 0.x/1.x). HF_TOKEN (a Space secret) is read
        # automatically for private repos.
        GEOLENS_HF_INDEX_REPO="${GEOLENS_HF_INDEX_REPO}" INDEX_DIR="${INDEX_DIR}" python - <<'PY'
import os

from huggingface_hub import hf_hub_download

repo = os.environ["GEOLENS_HF_INDEX_REPO"]
dest = os.environ["INDEX_DIR"]
for name in ("index.faiss", "metadata.db", "manifest.json"):
    path = hf_hub_download(repo, name, repo_type="dataset", local_dir=dest)
    print(f"entrypoint: fetched {name} -> {path}", flush=True)
PY
    else
        echo "entrypoint: WARNING index missing and GEOLENS_HF_INDEX_REPO unset;" \
             "the app will boot NOT_READY (health model_ready/index_ready=false)." >&2
    fi
fi

export GEOLENS_INDEX_DIR="${INDEX_DIR}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-7860}"
