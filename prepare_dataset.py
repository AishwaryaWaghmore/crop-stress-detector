from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

from dataset_utils import find_label_from_path, list_images, safe_mkdir


def collect_images(dataset_root: Path) -> Dict[str, List[Path]]:
    labeled = defaultdict(list)
    for img_path in list_images(dataset_root):
        label = find_label_from_path(img_path, dataset_root)
        if label != "unknown":
            labeled[label].append(img_path)
    return labeled


def copy_images(paths: List[Path], target_dir: Path) -> None:
    safe_mkdir(target_dir)
    for src in paths:
        dst = target_dir / src.name
        stem = src.stem
        suffix = src.suffix
        counter = 1
        while dst.exists():
            dst = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare combined train/val dataset from PlantVillage + PlantDoc")
    parser.add_argument("--plantvillage-dir", default="dataset/raw/plantvillage", help="Extracted PlantVillage directory")
    parser.add_argument("--plantdoc-dir", default="dataset/raw/plantdoc", help="Extracted PlantDoc directory")
    parser.add_argument("--output-dir", default="dataset/combined", help="Output directory for train/val splits")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--min-images-per-class", type=int, default=20, help="Minimum images per class to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    pv_root = Path(args.plantvillage_dir)
    pd_root = Path(args.plantdoc_dir)
    out_root = Path(args.output_dir)

    if out_root.exists():
        shutil.rmtree(out_root)

    train_root = out_root / "train"
    val_root = out_root / "val"
    safe_mkdir(train_root)
    safe_mkdir(val_root)

    pv_labeled = collect_images(pv_root)
    pd_labeled = collect_images(pd_root)

    merged = defaultdict(list)
    for label, items in pv_labeled.items():
        merged[label].extend(items)
    for label, items in pd_labeled.items():
        merged[label].extend(items)

    kept_classes = 0
    for label, items in merged.items():
        if len(items) < args.min_images_per_class:
            continue

        train_items, val_items = train_test_split(
            items,
            test_size=args.val_size,
            random_state=args.seed,
            shuffle=True,
        )

        copy_images(train_items, train_root / label)
        copy_images(val_items, val_root / label)
        kept_classes += 1

    if kept_classes == 0:
        raise RuntimeError(
            "No classes met min image requirement. Reduce --min-images-per-class or check dataset paths."
        )

    print(f"Prepared dataset at: {out_root}")
    print(f"Classes kept: {kept_classes}")


if __name__ == "__main__":
    main()
