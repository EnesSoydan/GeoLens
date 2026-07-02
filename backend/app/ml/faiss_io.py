"""Unicode-safe FAISS index read/write helpers.

FAISS's C++ file IO opens paths through the narrow ``char*`` API, which fails on
Windows for paths containing non-ASCII characters (e.g. a ``Masaüstü`` / Desktop
folder): ``could not open ... for writing: No such file or directory``. Python's
own file IO is unicode-aware, so we round-trip through an ASCII temporary path
(under the system temp dir) and move/copy with :mod:`shutil`.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import faiss


def write_index(index: faiss.Index, path: Path) -> None:
    """Write ``index`` to ``path``, tolerating non-ASCII destination paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "index.faiss"
        faiss.write_index(index, str(tmp_path))
        shutil.move(str(tmp_path), str(path))


def read_index(path: Path) -> faiss.Index:
    """Read a FAISS index from ``path``, tolerating non-ASCII source paths."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "index.faiss"
        shutil.copy(str(path), str(tmp_path))
        return faiss.read_index(str(tmp_path))
