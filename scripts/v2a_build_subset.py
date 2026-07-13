"""S-V2-1 production: build the v2a global-balanced OSV5M subset.

The shard-alignment probe proved OSV5M shards are geographically *mixed* (each
shard is a near-uniform global sample), so the first N shards already form a
globally balanced subset. This script materialises that subset for v2a:

Phase A  read each target shard's id list from the zip central directory over
         HTTP Range (no 2.5 GB download) -> the union of ids we will index.
Phase B  stream ``train.csv`` once, keep only rows whose id is in that union,
         and write ``metadata.csv`` (id, lat, lon, country, city, region,
         shard). Skipped if the file already exists (idempotent).
Phase C  for each shard: download it, embed every JPEG straight from the zip
         (SALAD 8448, GPU), save ``emb_<shard>.npy`` + ``ids_<shard>.json``,
         then delete the shard. Shards already embedded are skipped (resume).

Disk discipline (Faz 4): peak disk stays at one shard (~2.5 GB) plus the growing
embedding store (~1.7 GB/shard); a 50 GB headroom guard runs before each
download and every shard is removed in a ``finally`` block.

Usage::

    PYTHONPATH=backend:. .venv/Scripts/python scripts/v2a_build_subset.py \
        [--shards 00,01,02,03,04,05] [--device cuda] [--limit-per-shard N]
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from PIL import Image

from app.services.embedding import EmbeddingService
from scripts.osv5m_shard_probe import SHARD_URL_TMPL, _shard_ids

REPO_ID = "osv5m/osv5m"
CSV_URL = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/train.csv"
TMP_DIR = Path("data/osv5m_tmp")
OUT_DIR = Path("data/osv5m_v2a")
META_COLS = ("id", "latitude", "longitude", "country", "city", "region")
HEADROOM_GB = 50.0
GB = 1_000_000_000


def _free_gb(path: Path) -> float:
    """Return free disk space in GB for the volume holding ``path``."""
    target = path if path.exists() else path.anchor or Path(".")
    return shutil.disk_usage(target).free / GB


def _target_ids(shards: list[str]) -> dict[str, str]:
    """Map every image id in the target shards to its shard (central-dir read)."""
    id_to_shard: dict[str, str] = {}
    for shard in shards:
        url = SHARD_URL_TMPL.format(shard=shard)
        ids = _shard_ids(url)
        for i in ids:
            id_to_shard[i] = shard
        print(f"[ids] shard-{shard}: {len(ids)} id", flush=True)
    return id_to_shard


def _write_metadata(id_to_shard: dict[str, str], out_path: Path) -> int:
    """Stream train.csv, keep rows whose id is a target, write a CSV subset."""
    target = set(id_to_shard)
    kept: list[pd.DataFrame] = []
    t0 = time.time()
    with requests.get(CSV_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        reader = pd.read_csv(
            r.raw,
            usecols=lambda c: c in META_COLS,
            chunksize=200_000,
            on_bad_lines="skip",
        )
        seen = 0
        for chunk in reader:
            seen += len(chunk)
            chunk["id"] = chunk["id"].astype(str)
            hit = chunk[chunk["id"].isin(target)]
            if not hit.empty:
                kept.append(hit)
            found = sum(len(k) for k in kept)
            print(f"    csv taranan={seen} eslesen={found}/{len(target)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    meta = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=META_COLS)
    meta["shard"] = meta["id"].map(id_to_shard)
    meta.to_csv(out_path, index=False)
    return len(meta)


def _embed_shard(
    embedding: EmbeddingService, shard: str, batch_size: int, limit: int | None
) -> tuple[np.ndarray, list[str]]:
    """Download one shard, embed its JPEGs (zip order), delete it; return (vecs, ids)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    free_before = _free_gb(TMP_DIR)
    if free_before < HEADROOM_GB:
        raise RuntimeError(f"serbest disk {free_before:.1f} GB < {HEADROOM_GB} GB esigi")

    shard_path: str | None = None
    try:
        print(f"[dl] shard-{shard} indiriliyor (~2.5 GB)...", flush=True)
        t0 = time.time()
        shard_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"images/train/{shard}.zip",
            local_dir=str(TMP_DIR),
        )
        print(f"[dl] tamam: {time.time() - t0:.0f} s", flush=True)

        ids: list[str] = []
        vectors: list[np.ndarray] = []
        with zipfile.ZipFile(shard_path) as zf:
            jpgs = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg"))]
            if limit is not None:
                jpgs = jpgs[:limit]
            print(f"[embed] shard-{shard}: {len(jpgs)} goruntu...", flush=True)
            t1 = time.time()
            for start in range(0, len(jpgs), batch_size):
                batch = jpgs[start : start + batch_size]
                images: list[Image.Image] = []
                for name in batch:
                    ids.append(name.split("/")[-1].rsplit(".", 1)[0])
                    with zf.open(name) as fh:
                        images.append(Image.open(io.BytesIO(fh.read())).convert("RGB"))
                vectors.append(embedding.embed_batch(images))
                done = min(start + batch_size, len(jpgs))
                rate = done / max(time.time() - t1, 1e-9)
                print(f"    {done}/{len(jpgs)} ({rate:.1f} img/s)", flush=True)
        return np.vstack(vectors).astype(np.float32), ids
    finally:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        print(f"[disk] serbest (shard temizlik sonrasi): {_free_gb(Path('.')):.1f} GB", flush=True)


def main() -> int:
    """Materialise the v2a OSV5M subset (metadata + per-shard embeddings)."""
    parser = argparse.ArgumentParser(description="v2a OSV5M subset builder (S-V2-1).")
    parser.add_argument("--shards", default="00,01,02,03,04,05", help="comma-separated shards.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit-per-shard", type=int, default=None, help="Cap images/shard (smoke runs)."
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    shards = [s.strip() for s in args.shards.split(",") if s.strip()]
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"

    # Phase A + B: target ids and metadata (idempotent) --------------------------
    if meta_path.exists():
        print(f"[meta] {meta_path} mevcut -> atlaniyor (idempotent)", flush=True)
    else:
        print(f"[ids] {len(shards)} shard id listesi okunuyor (indirmesiz)...", flush=True)
        id_to_shard = _target_ids(shards)
        print(f"[ids] toplam hedef id: {len(id_to_shard)}", flush=True)
        print("[meta] train.csv akitiliyor + filtreleniyor (~2.9GB, tek gecis)...", flush=True)
        n_meta = _write_metadata(id_to_shard, meta_path)
        print(f"[meta] {n_meta} satir -> {meta_path}", flush=True)

    # Phase C: embed shards (resume) ---------------------------------------------
    print(f"[model] VPR model yukleniyor ({args.device})...", flush=True)
    embedding = EmbeddingService.load(pretrained=True, device=args.device)

    total = 0
    for shard in shards:
        emb_path = out_dir / f"emb_{shard}.npy"
        ids_path = out_dir / f"ids_{shard}.json"
        if emb_path.exists() and ids_path.exists():
            n = len(json.loads(ids_path.read_text(encoding="utf-8")))
            total += n
            print(f"[skip] shard-{shard} zaten var ({n} vektor)", flush=True)
            continue
        vecs, ids = _embed_shard(embedding, shard, args.batch_size, args.limit_per_shard)
        np.save(emb_path, vecs)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        total += len(ids)
        print(f"[saved] shard-{shard}: {vecs.shape} -> {emb_path.name}", flush=True)

    print(f"\n[ok] v2a alt-kume hazir: {total} vektor, {len(shards)} shard", flush=True)
    print(f"[ok] cikti dizini: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
