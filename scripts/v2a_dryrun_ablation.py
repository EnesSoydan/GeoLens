"""S-V2-1 dry-run ablation: a cheap early signal on PCA-1024 recall loss.

Before committing to a multi-hour OSV5M embed, this runs the full v2a chain end
to end at tiny scale to measure whether PCA 8448->1024 keeps the R@1 loss within
the +/-2 point acceptance gate (Faz 3 / Faz 6):

1. Download ONE OSV5M shard, embed its first ~N images -> PCA-fit corpus. This
   corpus is independent of the cph+sf eval set -> overfit prevention (Faz 3).
2. Fit PCA 8448->1024 on that corpus (randomized SVD).
3. Reuse the v1 MSLS-val harness (``scripts/evaluate.py``): reconstruct the 18916
   database descriptors straight from the v1 FAISS index (no re-embed) and
   re-embed the 750 official val queries.
4. Baseline: R@1 on 8448-Flat (should reproduce the published ~91.9%).
5. PCA-1024: apply the fitted PCA to db+query, L2-normalise, R@1 on 1024-Flat.
   Flat (not HNSW) isolates the PCA loss from any ANN approximation.
6. Report the R@1 delta versus the 2-point gate.

Disk discipline (Faz 4): peak disk stays at one shard (~2.5 GB); the shard is
deleted in a ``finally`` block, with a 50 GB headroom guard before download.

Usage::

    PYTHONPATH=backend:. .venv/Scripts/python scripts/v2a_dryrun_ablation.py \
        [--device cuda] [--pca-images 10000] [--pca-dim 1024]
"""

from __future__ import annotations

import argparse
import io
import shutil
import sqlite3
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.decomposition import PCA

# scripts/ is a package run with PYTHONPATH=backend:. so both ``app`` and
# ``scripts`` are importable; reuse the v1 eval harness verbatim.
from app.core.config import get_settings
from app.ml.faiss_io import read_index
from app.services.embedding import EmbeddingService
from scripts.evaluate import (
    DEFAULT_KS,
    GT_THRESHOLD_M,
    _embed_queries,
    _load_database,
    _load_queries,
    _read_allowed_query_keys,
    evaluate,
)

REPO_ID = "osv5m/osv5m"
SHARD = "images/train/00.zip"
TMP_DIR = Path("data/osv5m_tmp")
HEADROOM_GB = 50.0
GB = 1_000_000_000
GATE_POINTS = 2.0


def _free_gb(path: Path) -> float:
    """Return free disk space in GB for the volume holding ``path``."""
    target = path if path.exists() else path.anchor or Path(".")
    return shutil.disk_usage(target).free / GB


def _embed_osv5m_corpus(
    embedding: EmbeddingService, n_images: int, batch_size: int
) -> np.ndarray:
    """Download one shard, embed its first ``n_images`` JPEGs, delete the shard.

    Images are read straight from the zip (no extraction); peak disk is one shard.
    Returns a (n_images, 8448) matrix of unit descriptors for the PCA fit.
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    free_before = _free_gb(TMP_DIR)
    print(f"[disk] serbest (indirme oncesi): {free_before:.1f} GB", flush=True)
    if free_before < HEADROOM_GB:
        raise RuntimeError(f"serbest disk {free_before:.1f} GB < {HEADROOM_GB} GB esigi")

    shard_path: str | None = None
    try:
        print(f"[dl] {SHARD} indiriliyor (~2.5 GB, tek shard)...", flush=True)
        t0 = time.time()
        shard_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=SHARD,
            local_dir=str(TMP_DIR),
        )
        print(f"[dl] tamam: {time.time() - t0:.0f} s", flush=True)

        vectors: list[np.ndarray] = []
        with zipfile.ZipFile(shard_path) as zf:
            jpgs = [
                n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg"))
            ][:n_images]
            print(f"[embed] {len(jpgs)} OSV5M goruntusu PCA-corpus icin...", flush=True)
            t1 = time.time()
            for start in range(0, len(jpgs), batch_size):
                batch = jpgs[start : start + batch_size]
                images: list[Image.Image] = []
                for name in batch:
                    with zf.open(name) as fh:
                        images.append(Image.open(io.BytesIO(fh.read())).convert("RGB"))
                vectors.append(embedding.embed_batch(images))
                done = min(start + batch_size, len(jpgs))
                rate = done / max(time.time() - t1, 1e-9)
                print(f"    embedded {done}/{len(jpgs)} ({rate:.1f} img/s)", flush=True)
        return np.vstack(vectors)
    finally:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        print(f"[disk] serbest (temizlik sonrasi): {_free_gb(Path('.')):.1f} GB", flush=True)


def _build_flat_ip(vectors: np.ndarray, ids: np.ndarray) -> faiss.Index:
    """Build an IndexIDMap(IndexFlatIP) from (N, d) vectors keyed by ``ids``."""
    index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), ids)
    return index


def _apply_pca_norm(pca: PCA, vectors: np.ndarray) -> np.ndarray:
    """PCA-project then L2-normalise -> unit vectors for cosine (inner-product) search."""
    reduced = pca.transform(vectors).astype(np.float32)
    reduced = np.ascontiguousarray(reduced)
    faiss.normalize_L2(reduced)
    return reduced


def main() -> int:
    """Run the tiny end-to-end PCA-1024 ablation and print a pass/fail verdict."""
    parser = argparse.ArgumentParser(description="v2a dry-run PCA-1024 ablation.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--pca-images", type=int, default=10_000, help="PCA-fit corpus size.")
    parser.add_argument("--pca-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=GT_THRESHOLD_M)
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = settings.msls_raw_dir or (settings.raw_dir / "msls_images")
    index_path = settings.index_dir / "index.faiss"
    db_path = settings.index_dir / "metadata.db"
    for path in (index_path, db_path):
        if not path.exists():
            print(f"error: missing artifact {path} (run build_index.py first)", file=sys.stderr)
            return 2
    if not raw_dir.exists():
        print(f"error: MSLS raw dir not found: {raw_dir}", file=sys.stderr)
        return 2

    cities = settings.target_cities
    ks = DEFAULT_KS

    print(f"[model] VPR model yukleniyor ({args.device})...", flush=True)
    embedding = EmbeddingService.load(pretrained=True, device=args.device)

    # 1-2: independent OSV5M PCA-fit corpus + PCA fit -----------------------------
    corpus = _embed_osv5m_corpus(embedding, args.pca_images, args.batch_size)
    print(f"[pca] corpus={corpus.shape} -> PCA {corpus.shape[1]}->{args.pca_dim}...", flush=True)
    pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=0)
    pca.fit(corpus)
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"[pca] aciklanan varyans (ilk {args.pca_dim} bilesen): {evr:.1%}", flush=True)

    # 3: MSLS-val database (reconstruct) + queries (re-embed) ---------------------
    v1index = cast(Any, read_index(index_path))
    flat = faiss.downcast_index(v1index.index)
    n_db = flat.ntotal
    db_vecs = np.asarray(flat.reconstruct_n(0, n_db), dtype=np.float32)  # unit norm
    db_ids_stored = np.asarray(faiss.vector_to_array(v1index.id_map)).astype(np.int64)
    print(f"[msls] veritabani reconstruct: {db_vecs.shape}", flush=True)

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    db_ids, db_lat, db_lon = _load_database(conn)
    allowed_by_city = {c: _read_allowed_query_keys(raw_dir, c, "all") for c in cities}
    queries = _load_queries(conn, cities, allowed_by_city)
    conn.close()
    if not queries:
        print("error: no queries selected (check data)", file=sys.stderr)
        return 2
    print(f"[msls] sorgu={len(queries)} (subtask=all, {','.join(cities)})", flush=True)
    q_vecs = _embed_queries(embedding, queries, args.batch_size)  # (Q, 8448), unit norm

    # 4: baseline R@1 on 8448-Flat -----------------------------------------------
    base_index = _build_flat_ip(db_vecs, db_ids_stored)
    base = evaluate(base_index, db_ids, db_lat, db_lon, queries, q_vecs, ks, args.threshold)

    # 5: PCA-1024 R@1 on 1024-Flat -----------------------------------------------
    db_pca = _apply_pca_norm(pca, db_vecs)
    q_pca = _apply_pca_norm(pca, q_vecs)
    pca_index = _build_flat_ip(db_pca, db_ids_stored)
    red = evaluate(pca_index, db_ids, db_lat, db_lon, queries, q_pca, ks, args.threshold)

    # 6: verdict -----------------------------------------------------------------
    base_r1 = float(cast(float, base["recall@1"]))
    pca_r1 = float(cast(float, red["recall@1"]))
    delta = (base_r1 - pca_r1) * 100.0
    print("\n=== v2a dry-run ablasyon (kucuk olcek, erken sinyal) ===", flush=True)
    print(f"  PCA-fit corpus       : {args.pca_images} OSV5M (bagimsiz)", flush=True)
    print(f"  eval                 : {base['n_queries_evaluated']} MSLS-val sorgusu", flush=True)
    print(f"  8448-Flat  R@1/5/10  : {base['recall@1']:.4f} / "
          f"{base['recall@5']:.4f} / {base['recall@10']:.4f}", flush=True)
    print(f"  PCA{args.pca_dim}-Flat R@1/5/10  : {red['recall@1']:.4f} / "
          f"{red['recall@5']:.4f} / {red['recall@10']:.4f}", flush=True)
    print(f"  R@1 kaybi            : {delta:+.2f} puan (kapi: <= {GATE_POINTS:.0f})", flush=True)
    verdict = "GECTI" if delta <= GATE_POINTS else "KALDI"
    print(f"[VERDICT] {verdict} (PCA-{args.pca_dim} {'kabul' if verdict == 'GECTI' else 'RED'})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
