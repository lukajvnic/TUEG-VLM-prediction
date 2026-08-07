#!/usr/bin/env python3
"""Generate TUEP PNG previews and labels.csv in the common dataset format.

Reads EDF files from v3.1.0/{00_epilepsy,01_no_epilepsy} by default and writes
PNG files split into train/ and test/ (no patient shared across splits) plus a
labels.csv mapping image_path to a split and boolean labels.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _render import process


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = ROOT / "v3.1.0"
LABELS = ["epilepsy", "no_epilepsy"]
CLASS_DIRS = {
    "epilepsy": "00_epilepsy",
    "no_epilepsy": "01_no_epilepsy",
}


SUBJECT = re.compile(r"aaaa[a-z]{4}")


def patient_of(edf):
    # TUEG anonymised subject id (aaaaXXXX) when present, else the containing dir
    # (handles TUEV eval files, which are named by label rather than subject).
    match = SUBJECT.search(edf.name)
    return match.group(0) if match else edf.parent.name


def scan_names(edfs):
    # <patient>_<index>.png: the patient id is anonymised and label-free; the index
    # (stable, by sorted scan order) distinguishes a patient's multiple scans.
    counters = defaultdict(int)
    names = {}
    for edf in sorted(edfs, key=str):
        patient = patient_of(edf)
        names[edf] = f"{patient}_{counters[patient]}"
        counters[patient] += 1
    return names


def assign_splits(edfs, test_frac, seed):
    # Keep every scan from one patient in the same split so no individual appears
    # in both train and test.
    by_patient = defaultdict(list)
    for edf in edfs:
        by_patient[patient_of(edf)].append(edf)
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    target = round(len(edfs) * test_frac)
    test, count = set(), 0
    for patient in patients:
        if count >= target:
            break
        test.add(patient)
        count += len(by_patient[patient])
    return {edf: ("test" if patient_of(edf) in test else "train") for edf in edfs}


def patient_edfs(patient_dir: Path) -> list[Path]:
    return sorted(patient_dir.rglob("*.edf"))


def generate(dataset_root: Path, limit: int | None, overwrite: bool, test_frac: float, seed: int,
             window: float, train_windows: int, test_windows: int) -> int:
    selected: list[tuple[Path, str]] = []
    for label, source_name in CLASS_DIRS.items():
        source_dir = dataset_root / source_name
        if not source_dir.exists():
            print(f"Missing source directory: {source_dir}", file=sys.stderr)
            continue

        count = 0
        for patient_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            for edf_path in patient_edfs(patient_dir):
                if limit is not None and count >= limit:
                    break
                selected.append((edf_path, label))
                count += 1
            if limit is not None and count >= limit:
                break
        print(f"Found {count} EDFs for {label}")

    splits = assign_splits([edf for edf, _ in selected], test_frac, seed)
    names = scan_names([edf for edf, _ in selected])
    for name in ("train", "test"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)

    generated = failed = 0
    with (ROOT / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["image_path", "split", *LABELS])
        writer.writeheader()

        for index, (edf_path, label) in enumerate(selected, start=1):
            split = splits[edf_path]
            label_row = {
                "epilepsy": str(label == "epilepsy").lower(),
                "no_epilepsy": str(label == "no_epilepsy").lower(),
            }
            def label_fn(start, stop, label_row=label_row):
                return label_row, False
            try:
                for image_path, row in process(edf_path, names[edf_path], split, ROOT, label_fn,
                                               window, train_windows, test_windows, seed, overwrite):
                    writer.writerow({"image_path": image_path, "split": split, **row})
                    generated += 1
                print(f"[{index}/{len(selected)}] [{split}] {names[edf_path]}")
            except Exception as exc:
                print(f"Failed {edf_path}: {exc}", file=sys.stderr)
                failed += 1

    print(f"Done. Generated: {generated}, failed: {failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render TUEP EDFs to flat PNGs and labels.csv.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--limit", type=int, help="EDFs per class; omit for all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=float, default=20)
    parser.add_argument("--train-windows", type=int, default=4)
    parser.add_argument("--test-windows", type=int, default=10)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    return generate(args.dataset_root, args.limit, args.overwrite, args.test_frac, args.seed,
                    args.window, args.train_windows, args.test_windows)


if __name__ == "__main__":
    raise SystemExit(main())
