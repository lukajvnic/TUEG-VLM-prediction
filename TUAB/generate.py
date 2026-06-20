#!/usr/bin/env python3
"""Generate TUAB PNG previews and labels.csv in the common dataset format.

Reads EDF files from v3.0.1/edf/train/{normal,abnormal} by default and writes
flat PNG files to data/ plus a labels.csv mapping image_path to boolean labels.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_ROOT = ROOT / "v3.0.1" / "edf" / "train"
DATA_DIR = ROOT / "data"
LABELS = ["abnormal", "normal"]
CLASSES = ("normal", "abnormal")


def edf_to_png(edf_path: Path, png_path: Path) -> None:
    """Render one EDF file to a PNG preview."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    fig = raw.plot(scalings="auto", title=str(edf_path), show=False)
    fig.set_size_inches(20, 12)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def image_name(class_name: str, class_dir: Path, edf_path: Path) -> str:
    """Build a unique flat PNG filename with the class prefix included."""
    relative = edf_path.relative_to(class_dir).with_suffix("")
    return "__".join((class_name, *relative.parts)) + ".png"


def select_edfs(class_dir: Path, limit: int | None, shuffle: bool, seed: int) -> list[Path]:
    edfs = sorted(class_dir.rglob("*.edf"))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(edfs)
    return edfs if limit is None else edfs[:limit]


def generate(
    train_root: Path,
    limit: int | None,
    overwrite: bool,
    shuffle: bool,
    seed: int,
) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generated = skipped = failed = 0

    with (ROOT / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["image_path", *LABELS])
        writer.writeheader()

        for class_name in CLASSES:
            class_dir = train_root / class_name
            if not class_dir.exists():
                print(f"Missing source directory: {class_dir}", file=sys.stderr)
                failed += 1
                continue

            edfs = select_edfs(class_dir, limit=limit, shuffle=shuffle, seed=seed)
            print(f"Found {len(edfs)} EDFs for {class_name}")

            for index, edf_path in enumerate(edfs, start=1):
                png_path = DATA_DIR / image_name(class_name, class_dir, edf_path)
                try:
                    if png_path.exists() and not overwrite:
                        print(f"[{class_name} {index}/{len(edfs)}] Skipping existing: {png_path}")
                        skipped += 1
                    else:
                        print(f"[{class_name} {index}/{len(edfs)}] Generating {png_path}")
                        edf_to_png(edf_path, png_path)
                        generated += 1

                    writer.writerow({
                        "image_path": str(png_path.relative_to(ROOT)),
                        "abnormal": str(class_name == "abnormal").lower(),
                        "normal": str(class_name == "normal").lower(),
                    })
                except Exception as exc:
                    print(f"Failed to generate {png_path} from {edf_path}: {exc}", file=sys.stderr)
                    failed += 1

    print(f"Done. Generated: {generated}, skipped: {skipped}, failed: {failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render TUAB EDFs to flat PNGs and labels.csv.")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--limit", type=int, default=500, help="EDFs per class; use -1 for all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    limit = None if args.limit == -1 else args.limit
    if limit is not None and limit < 1:
        parser.error("--limit must be at least 1, or -1 for all")

    return generate(args.train_root, limit, args.overwrite, args.shuffle, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
