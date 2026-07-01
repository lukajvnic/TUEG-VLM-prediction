#!/usr/bin/env python3
"""Generate TUEP PNG previews and labels.csv in the common dataset format.

Reads EDF files from v3.1.0/{00_epilepsy,01_no_epilepsy} by default and writes
flat PNG files to data/ plus a labels.csv mapping image_path to boolean labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = ROOT / "v3.1.0"
DATA_DIR = ROOT / "data"
LABELS = ["epilepsy", "no_epilepsy"]
CLASS_DIRS = {
    "epilepsy": "00_epilepsy",
    "no_epilepsy": "01_no_epilepsy",
}


def edf_to_png(edf_path: Path, png_path: Path) -> None:
    """Render an EDF file to a PNG preview."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    fig = raw.plot(scalings="auto", title=str(edf_path), show=False)
    fig.set_size_inches(20, 12)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def image_name(label: str, patient_dir: Path, edf_path: Path) -> str:
    """Build a unique flat PNG filename with the class prefix included."""
    relative = edf_path.relative_to(patient_dir).with_suffix("")
    return "__".join((label, patient_dir.name, *relative.parts)) + ".png"


def patient_edfs(patient_dir: Path) -> list[Path]:
    return sorted(patient_dir.rglob("*.edf"))


def generate(dataset_root: Path, limit: int | None, overwrite: bool) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generated = skipped = failed = written = 0

    with (ROOT / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["image_path", *LABELS])
        writer.writeheader()

        for label, source_name in CLASS_DIRS.items():
            source_dir = dataset_root / source_name
            if not source_dir.exists():
                print(f"Missing source directory: {source_dir}", file=sys.stderr)
                failed += 1
                continue

            patient_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
            edf_count_for_label = 0

            for patient_dir in patient_dirs:
                for edf_path in patient_edfs(patient_dir):
                    if limit is not None and edf_count_for_label >= limit:
                        break

                    png_path = DATA_DIR / image_name(label, patient_dir, edf_path)
                    try:
                        if png_path.exists() and not overwrite:
                            print(f"Skipping existing image: {png_path}")
                            skipped += 1
                        else:
                            print(f"Generating {png_path} from {edf_path}")
                            edf_to_png(edf_path, png_path)
                            generated += 1

                        writer.writerow({
                            "image_path": str(png_path.relative_to(ROOT)),
                            "epilepsy": str(label == "epilepsy").lower(),
                            "no_epilepsy": str(label == "no_epilepsy").lower(),
                        })
                        written += 1
                        edf_count_for_label += 1
                    except Exception as exc:
                        print(f"Failed to generate image for {edf_path}: {exc}", file=sys.stderr)
                        failed += 1

                if limit is not None and edf_count_for_label >= limit:
                    break

            print(f"Found {edf_count_for_label} EDFs for {label}")

    print(f"Done. Labels: {written}, generated: {generated}, skipped: {skipped}, failed: {failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render TUEP EDFs to flat PNGs and labels.csv.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--limit", type=int, help="EDFs per class; omit for all.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    return generate(args.dataset_root, args.limit, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
