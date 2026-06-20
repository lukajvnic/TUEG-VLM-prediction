#!/usr/bin/env python3
"""Generate EEG PNG previews for epilepsy and no-epilepsy patients.

This creates one PNG image for every EDF file found under each patient's
directory. Images are generated with the same plotting method used in
visualize.py.
"""

from pathlib import Path
import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne

DATASET_ROOT = Path("v3.1.0")
CLASS_DIRS = {
    "00_epilepsy": Path("data/epilepsy"),
    "01_no_epilepsy": Path("data/no-epilepsy"),
}


def edf_to_png(edf_path: Path, png_path: Path) -> None:
    """Render an EDF file to a PNG using the same method as visualize.py."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True)
    fig = raw.plot(scalings="auto", title=str(edf_path), show=False)
    fig.set_size_inches(20, 12)
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def patient_edfs(patient_dir: Path) -> list[Path]:
    return sorted(patient_dir.rglob("*.edf"))


def png_name_for_edf(patient_dir: Path, edf_path: Path) -> str:
    """Create a unique PNG filename for an EDF within a patient directory."""
    relative = edf_path.relative_to(patient_dir).with_suffix("")
    return "__".join(relative.parts) + ".png"


def generate_images(overwrite: bool = False) -> int:
    total = 0
    skipped = 0
    failed = 0

    for source_name, output_dir in CLASS_DIRS.items():
        source_dir = DATASET_ROOT / source_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            print(f"Missing source directory: {source_dir}", file=sys.stderr)
            failed += 1
            continue

        patient_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())

        for patient_dir in patient_dirs:
            edfs = patient_edfs(patient_dir)
            if not edfs:
                print(
                    f"No EDF files found for patient {patient_dir.name}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            for edf_path in edfs:
                png_path = output_dir / png_name_for_edf(patient_dir, edf_path)

                if png_path.exists() and not overwrite:
                    print(f"Skipping existing image: {png_path}")
                    skipped += 1
                    continue

                try:
                    print(f"Generating {png_path} from {edf_path}")
                    edf_to_png(edf_path, png_path)
                    total += 1
                except Exception as exc:
                    print(
                        f"Failed to generate image for {edf_path}: {exc}", file=sys.stderr
                    )
                    failed += 1

    print(f"Done. Generated: {total}, skipped: {skipped}, failed: {failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one EEG PNG per EDF file."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate PNGs even if they already exist.",
    )
    args = parser.parse_args()

    return generate_images(overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
