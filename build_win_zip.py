#!/usr/bin/env python3
"""Package a Windows-ready Python 3.12 ZIP from the current tree."""
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).parent.resolve()
OUT_ZIP = REPO / "NEWPROJECT-win-py312.zip"

EXCLUDE_DIRS = {".git", ".pytest_cache", "__pycache__", "cpp/build"}
EXCLUDE_SUFFIXES = {".zip", ".png", ".jpg", ".pkl", ".npz", ".stp", ".pdf"}


def should_include(rel: Path) -> bool:
    parts = rel.parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return False
    if any(part.startswith("NEWPROJECT-win-") and part.endswith(".zip") for part in parts):
        return False
    if rel.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    return True


def build() -> Path:
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(REPO):
            # prune excluded dirs to avoid descending
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                src = Path(root) / f
                rel = src.relative_to(REPO.parent).resolve()
                if not should_include(rel):
                    continue
                arcname = "NEWPROJECT" / src.relative_to(REPO)
                zf.write(src, arcname)
    return OUT_ZIP


if __name__ == "__main__":
    print("Building", OUT_ZIP)
    path = build()
    print("Wrote", path, f"({path.stat().st_size / 1024 / 1024:.1f} MB)")
