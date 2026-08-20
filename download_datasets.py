from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DEFAULT_PLANTVILLAGE_SLUG = "emmarex/plantdisease"
DEFAULT_PLANTDOC_SLUG = "hasnain4236/plantdoc"


def download_and_extract(api: KaggleApi, slug: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{slug.split('/')[-1]}.zip"

    print(f"Downloading '{slug}' into '{out_dir}'...")
    api.dataset_download_files(slug, path=str(out_dir), quiet=False)

    downloaded = list(out_dir.glob("*.zip"))
    if not downloaded:
        raise FileNotFoundError(f"No zip downloaded for dataset '{slug}'.")

    target_zip = downloaded[0]
    if zip_path != target_zip:
        zip_path = target_zip

    print(f"Extracting '{zip_path.name}'...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    print(f"Done: {slug}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PlantVillage and PlantDoc from Kaggle")
    parser.add_argument("--plantvillage", default=DEFAULT_PLANTVILLAGE_SLUG, help="Kaggle dataset slug for PlantVillage")
    parser.add_argument("--plantdoc", default=DEFAULT_PLANTDOC_SLUG, help="Kaggle dataset slug for PlantDoc")
    parser.add_argument("--output", default="dataset/raw", help="Output directory")
    args = parser.parse_args()

    output = Path(args.output)
    plantvillage_dir = output / "plantvillage"
    plantdoc_dir = output / "plantdoc"

    api = KaggleApi()
    api.authenticate()

    download_and_extract(api, args.plantvillage, plantvillage_dir)
    download_and_extract(api, args.plantdoc, plantdoc_dir)

    print("All dataset downloads completed.")


if __name__ == "__main__":
    main()
