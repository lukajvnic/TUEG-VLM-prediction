#!/usr/bin/env python3
"""Generate PNG previews for 500 normal and 500 abnormal TUAB EDF files.

By default this reads EDF files from v3.0.1/edf/train/{normal,abnormal}
and writes PNGs to data/{normal,abnormal}.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_ROOT = SCRIPT_DIR / "v3.0.1" / "edf" / "train"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "data"
CLASSES = ("normal", "abnormal")


def edf_to_png(edf_path: Path, png_path: Path) -> None:
    """Render one EDF file to a PNG preview."""
    # Preload to avoid MNE edge-artifact warnings for EDFs with mixed sampling frequencies.
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    fig = raw.plot(scalings="auto", title=str(edf_path), show=False)
    fig.set_size_inches(20, 12)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def output_name(class_dir: Path, edf_path: Path) -> str:
    """Build a unique PNG filename from an EDF path relative to its class dir."""
    relative = edf_path.relative_to(class_dir).with_suffix("")
    return "__".join(relative.parts) + ".png"


def select_edfs(class_dir: Path, limit: int, shuffle: bool, seed: int) -> list[Path]:
    edfs = sorted(class_dir.rglob("*.edf"))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(edfs)
    return edfs[:limit]


def generate_images(
    train_root: Path,
    output_root: Path,
    limit: int,
    overwrite: bool,
    shuffle: bool,
    seed: int,
) -> int:
    generated = 0
    skipped = 0
    failed = 0

    for class_name in CLASSES:
        class_dir = train_root / class_name
        output_dir = output_root / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not class_dir.exists():
            print(f"Missing source directory: {class_dir}", file=sys.stderr)
            failed += 1
            continue

        edfs = select_edfs(class_dir, limit=limit, shuffle=shuffle, seed=seed)
        print(f"Found {len(edfs)} EDFs for {class_name}; writing to {output_dir}")

        if len(edfs) < limit:
            print(
                f"Warning: requested {limit} {class_name} EDFs, but only found {len(edfs)}",
                file=sys.stderr,
            )

        for index, edf_path in enumerate(edfs, start=1):
            png_path = output_dir / output_name(class_dir, edf_path)

            if png_path.exists() and not overwrite:
                print(f"[{class_name} {index}/{len(edfs)}] Skipping existing: {png_path}")
                skipped += 1
                continue

            try:
                print(f"[{class_name} {index}/{len(edfs)}] Generating {png_path}")
                edf_to_png(edf_path, png_path)
                generated += 1
            except Exception as exc:  # keep going if one EDF cannot be rendered
                print(f"Failed to generate {png_path} from {edf_path}: {exc}", file=sys.stderr)
                failed += 1

    print(f"Done. Generated: {generated}, skipped: {skipped}, failed: {failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate PNG previews from TUAB train EDF files."
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
        help="Directory containing normal/ and abnormal/ EDF folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where normal/ and abnormal/ PNG folders are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of EDFs to process from each class.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate PNGs even if they already exist.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly choose EDFs instead of taking the first sorted EDFs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --shuffle.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    return generate_images(
        train_root=args.train_root,
        output_root=args.output_root,
        limit=args.limit,
        overwrite=args.overwrite,
        shuffle=args.shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
