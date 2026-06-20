#!/usr/bin/env python3
"""Generate annotated TUEP PNG previews.

TUEP is a recording-level binary classification dataset, so there are no
interval annotations to highlight. The annotation added here is the class label
(epilepsy/no-epilepsy) as a colored title/banner on each rendered EDF preview.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = ROOT / "v3.1.0"
DATA_DIR = ROOT / "data-annotated"
CLASS_DIRS = {
    "epilepsy": "00_epilepsy",
    "no_epilepsy": "01_no_epilepsy",
}
COLORS = {
    "epilepsy": "#e74c3c",
    "no_epilepsy": "#2ecc71",
}
DISPLAY_LABELS = {
    "epilepsy": "EPILEPSY",
    "no_epilepsy": "NO EPILEPSY",
}


def image_name(label: str, patient_dir: Path, edf_path: Path) -> str:
    relative = edf_path.relative_to(patient_dir).with_suffix("")
    return "__".join((label, patient_dir.name, *relative.parts)) + ".png"


def patient_edfs(patient_dir: Path) -> list[Path]:
    return sorted(patient_dir.rglob("*.edf"))


def render(edf_path: Path, png_path: Path, label: str) -> None:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    color = COLORS[label]

    fig = raw.plot(scalings="auto", title=str(edf_path), show=False)
    fig.set_size_inches(20, 12)

    fig.text(
        0.5,
        0.985,
        DISPLAY_LABELS[label],
        ha="center",
        va="top",
        fontsize=20,
        weight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": color},
    )
    fig.patch.set_edgecolor(color)
    fig.patch.set_linewidth(10)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate(dataset_root: Path, limit: int | None, overwrite: bool) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generated = skipped = failed = 0

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
                        render(edf_path, png_path, label)
                        generated += 1

                    edf_count_for_label += 1
                except Exception as exc:
                    print(f"Failed to generate image for {edf_path}: {exc}", file=sys.stderr)
                    failed += 1

            if limit is not None and edf_count_for_label >= limit:
                break

        print(f"Found {edf_count_for_label} EDFs for {label}")

    print(f"Done. Generated: {generated}, skipped: {skipped}, failed: {failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render annotated TUEP EDF previews.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--limit", type=int, help="EDFs per class; omit for all.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    return generate(args.dataset_root, args.limit, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
