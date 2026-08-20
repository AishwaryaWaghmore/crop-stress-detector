from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and is_image(p)]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_label(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def find_label_from_path(path: Path, dataset_root: Path) -> str:
    rel = path.relative_to(dataset_root)
    parts = rel.parts
    if len(parts) < 2:
        return "unknown"
    return normalize_label(parts[-2])


def ensure_non_empty_dir(path: Path, purpose: str) -> None:
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"{purpose} directory '{path}' is missing or empty. "
            "Run dataset download/preparation first."
        )


def chunked(items: Iterable[Path], size: int):
    bucket = []
    for item in items:
        bucket.append(item)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
