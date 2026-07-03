# GeoLens single-container image (FastAPI + mounted Gradio UI), CPU-only.
#
# Target: Hugging Face Spaces free tier (16 GB RAM / 50 GB disk, CPU-only).
# Measured footprint of this image as-is: ~2.1 GB peak RSS at serving time
# (model + full 18,916-vector FAISS index + one /predict with heatmap) and
# ~3 GB on disk (CPU torch + weights baked below; the ~639 MB index is fetched
# at startup, not baked). Both sit far under the free-tier limits, so the image
# is deployed with pytorch-lightning kept as-is (no lean-inference rewrite).
#
# Build context is the repo root:  docker build -f deploy/Dockerfile -t geolens .
FROM python:3.12-slim

# libGL / libglib are needed by opencv (pulled in by grad-cam) at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000; create a matching non-root user so
# the torch.hub cache and fetched index live in a writable, predictable home.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    # The Gradio UI calls the API on the same container over HTTP; both are
    # served by app.main:app. backend/ + repo root must be on PYTHONPATH.
    PYTHONPATH=/app/backend:/app \
    # Bake + serve the model weights from here (see the warm step below).
    TORCH_HOME=/home/user/.cache/torch \
    # Serve on CPU: the HF free tier has no GPU.
    GEOLENS_SERVING_DEVICE=cpu

WORKDIR /app

# Install CPU-only torch/torchvision FIRST from the CPU wheel index so the
# generic "torch>=2.2" in pyproject resolves to the small CPU build (the default
# PyPI Linux wheel bundles CUDA and is multiple GB larger).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2,<2.13" "torchvision>=0.17"

# Application code + project metadata, then install the project (which pulls the
# remaining runtime dependencies from pyproject; CPU torch above already
# satisfies the torch/torchvision pins). The app is served from source via
# PYTHONPATH, so the installed package is only for its dependency closure.
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY frontend ./frontend
RUN pip install --no-cache-dir .

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && chown -R user:user /app /home/user

USER user

# Warm the DINOv2 + SALAD weights into TORCH_HOME at build time (~1.4 GB) so the
# model loads without a cold-start download. trust_repo=True mirrors the loader.
RUN python -c "import torch; torch.hub.load('serizba/salad', 'dinov2_salad', pretrained=True, trust_repo=True)"

# HF Spaces routes public traffic to app_port (7860, declared in deploy/README.md).
EXPOSE 7860
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
