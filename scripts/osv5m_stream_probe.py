"""S-V2-0 streaming-download proof for OSV5M.

Downloads a *single* OSV5M image shard (``images/train/00.zip``, ~2.5 GB) with
``hf_hub_download`` (file-by-file, NOT ``snapshot_download`` of the whole 246 GB
set), reads a few JPEGs straight from the zip *without extracting*, measures the
real disk footprint, then deletes the shard in a ``finally`` block.

This validates the Faz 4 "iki-geçişli streaming" precondition: peak disk stays at
one shard (~2.5 GB), not the full dataset. If this mechanism works, v2a's disk
budget (~10-30 GB peak) holds. Run:

    .venv/Scripts/python scripts/osv5m_stream_probe.py

It prints a short report only; it does not build any index.
"""

from __future__ import annotations

import io
import shutil
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "osv5m/osv5m"
SHARD = "images/train/00.zip"
TMP_DIR = Path("data/osv5m_tmp")
HEADROOM_GB = 50.0  # Faz 4: stop if free disk would drop below this.
GB = 1_000_000_000


def _free_gb(path: Path) -> float:
    """Return free disk space in GB for the volume holding ``path``."""
    target = path if path.exists() else path.anchor or Path(".")
    return shutil.disk_usage(target).free / GB


def main() -> int:
    """Download one shard, read images from the zip, delete it, report disk usage."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    free_before = _free_gb(TMP_DIR)
    print(f"[disk] serbest (indirme oncesi): {free_before:.1f} GB")

    # Faz 4 discipline: never fill the disk. One shard is ~2.5 GB; abort if the
    # headroom safety margin would be violated.
    if free_before < HEADROOM_GB:
        print(f"[abort] serbest disk {free_before:.1f} GB < {HEADROOM_GB} GB esigi")
        return 2

    shard_path: str | None = None
    try:
        print(f"[dl] {SHARD} indiriliyor (~2.5 GB, tek shard)...")
        t0 = time.time()
        shard_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=SHARD,
            local_dir=str(TMP_DIR),
        )
        dl_s = time.time() - t0
        size_gb = Path(shard_path).stat().st_size / GB
        free_after_dl = _free_gb(TMP_DIR)
        rate = size_gb * 1000 / dl_s if dl_s else 0.0
        print(f"[dl] tamam: {size_gb:.2f} GB, {dl_s:.0f} s, ~{rate:.0f} MB/s")
        print(f"[disk] serbest (indirme sonrasi): {free_after_dl:.1f} GB")
        print(f"[disk] shard ayak izi: {free_before - free_after_dl:.2f} GB")

        # Read a few images straight from the zip WITHOUT extracting -> peak disk
        # stays at the shard size, not shard + extracted copy.
        with zipfile.ZipFile(shard_path) as zf:
            names = zf.namelist()
            jpgs = [n for n in names if n.lower().endswith((".jpg", ".jpeg"))]
            print(f"[zip] girdi sayisi: {len(names)}, jpg: {len(jpgs)}")
            print(f"[zip] ornek yollar: {jpgs[:3]}")
            for n in jpgs[:3]:
                with zf.open(n) as fh:
                    img = Image.open(io.BytesIO(fh.read()))
                    img.load()
                    print(f"[img] {n}: {img.mode} {img.size}")
        free_peak = _free_gb(TMP_DIR)
        print(f"[disk] serbest (zip okuma sirasinda, cikarim YOK): {free_peak:.1f} GB")
    finally:
        # Guaranteed cleanup even if the read fails mid-way (Faz 4 try/finally).
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        free_final = _free_gb(Path("."))
        print(f"[disk] serbest (temizlik sonrasi): {free_final:.1f} GB")
        print(f"[disk] geri kazanildi: {free_final - free_before:+.2f} GB (~0 beklenir)")

    print("[ok] streaming-indirme kaniti: tek shard indir->oku->sil calisiyor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
