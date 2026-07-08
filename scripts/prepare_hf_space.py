"""Assemble a self-contained Hugging Face Docker Space build context.

A HF Docker Space only looks for ``Dockerfile`` (and the frontmatter
``README.md``) at the **repository root** — it has no option to point at a
nested path. This project keeps its deployment files tidy under ``deploy/``, so
this script materialises a throwaway ``hf_space_build/`` directory that IS a
valid root build context, and which is the only thing pushed to the Space repo.

The key invariant: ``deploy/Dockerfile`` is copied to the build root **byte for
byte** (never rewritten), and every path it ``COPY``s (``backend``, ``frontend``,
``pyproject.toml``, ``README.md``, ``deploy/entrypoint.sh``) is staged at the
same relative location — so no COPY line can break. As a guard, the script then
parses the copied Dockerfile and asserts each local COPY source resolves inside
the build directory; if a future Dockerfile edit references an unstaged path,
this fails loudly instead of producing a Space that errors mid-build.

Usage::

    python scripts/prepare_hf_space.py            # regenerate hf_space_build/
    python scripts/prepare_hf_space.py --out dist/space

The generated directory is git-ignored; push only its contents to the Space
(from inside it: git init, git add ., git commit, then git push to the Space
remote). See deploy/README.md for the full go-live checklist.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Repo root: scripts/prepare_hf_space.py -> parents[1].
REPO_ROOT = Path(__file__).resolve().parents[1]

# Ignore transient Python artifacts when copying source trees.
_IGNORE = shutil.ignore_patterns("__pycache__", "*.py[cod]", "*.egg-info", ".DS_Store")

# What the build context must contain for deploy/Dockerfile's COPY lines to work.
# (src relative to REPO_ROOT, dest relative to the build dir). Dockerfile itself
# is relocated to the build root; everything else keeps its relative path so the
# COPY instructions stay valid unchanged.
_FILES: tuple[tuple[str, str], ...] = (
    ("deploy/Dockerfile", "Dockerfile"),
    ("deploy/README.md", "README.md"),
    (".dockerignore", ".dockerignore"),
    ("pyproject.toml", "pyproject.toml"),
    ("deploy/entrypoint.sh", "deploy/entrypoint.sh"),
)
_DIRS: tuple[str, ...] = ("backend", "frontend")


def _clean_build_dir(out_dir: Path) -> None:
    """Empty ``out_dir`` of generated content but keep an existing ``.git``.

    Preserving ``.git`` means a regenerate-then-push cycle reuses the Space's git
    history and remote (no need to re-add the remote after every regeneration),
    and it sidesteps the read-only git object files that break a full ``rmtree``
    on Windows. Only the regenerated source/config is removed and rewritten.
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        return
    for child in out_dir.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _copy_into(out_dir: Path) -> None:
    """Copy the required files and source trees into ``out_dir`` (keeps ``.git``)."""
    _clean_build_dir(out_dir)

    for src_rel, dest_rel in _FILES:
        src = REPO_ROOT / src_rel
        if not src.is_file():
            raise FileNotFoundError(f"expected source file missing: {src_rel}")
        dest = out_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    for dir_rel in _DIRS:
        src = REPO_ROOT / dir_rel
        if not src.is_dir():
            raise FileNotFoundError(f"expected source directory missing: {dir_rel}")
        shutil.copytree(src, out_dir / dir_rel, ignore=_IGNORE)


def _copy_sources(dockerfile: Path) -> list[str]:
    """Return the local (non-URL) source paths referenced by ``COPY`` lines.

    Each ``COPY <src>... <dest>`` instruction lists one or more sources followed
    by the destination; build flags (``--chown=`` etc.) and remote URLs are
    skipped. Only the sources are returned, as those must exist in the context.
    """
    sources: list[str] = []
    for raw in dockerfile.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.upper().startswith("COPY "):
            continue
        tokens = [t for t in line.split()[1:] if not t.startswith("--")]
        if len(tokens) < 2:
            continue  # malformed COPY (need at least one src + dest)
        for src in tokens[:-1]:  # drop the destination
            if "://" not in src:
                sources.append(src)
    return sources


def _verify_copy_paths(out_dir: Path) -> list[str]:
    """Return COPY sources in the built Dockerfile that are missing from ``out_dir``."""
    return [src for src in _copy_sources(out_dir / "Dockerfile") if not (out_dir / src).exists()]


def main() -> int:
    """Materialise the Space build context and verify the Dockerfile's COPY paths."""
    parser = argparse.ArgumentParser(description="Prepare a HF Docker Space build context.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "hf_space_build",
        help="Output directory (default: hf_space_build/).",
    )
    args = parser.parse_args()
    out_dir = args.out.resolve()

    _copy_into(out_dir)
    missing = _verify_copy_paths(out_dir)
    if missing:
        print(
            "error: Dockerfile COPYs sources absent from the build context: "
            + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        return 1

    rel = out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir
    print(f"prepared HF Space build context at {rel}/ (Dockerfile + README at root)")
    print("all Dockerfile COPY sources verified present.")
    print(f"next: push the contents of {rel}/ to your HF Docker Space repo (see deploy/README.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
