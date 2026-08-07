#!/usr/bin/env python3
"""Generate TUAB PNG previews and labels.csv in the common dataset format.

Reads EDF files from v3.0.1/edf/train/{normal,abnormal} by default and writes
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
DEFAULT_TRAIN_ROOT = ROOT / "v3.0.1" / "edf" / "train"
LABELS = ["abnormal", "normal"]
CLASSES = ("normal", "abnormal")


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
    test_frac: float,
    window: float,
    train_windows: int,
    test_windows: int,
) -> int:
    selected: list[tuple[Path, str]] = []
    for class_name in CLASSES:
        class_dir = train_root / class_name
        if not class_dir.exists():
            print(f"Missing source directory: {class_dir}", file=sys.stderr)
            continue
        edfs = select_edfs(class_dir, limit=limit, shuffle=shuffle, seed=seed)
        print(f"Found {len(edfs)} EDFs for {class_name}")
        selected.extend((edf, class_name) for edf in edfs)

    splits = assign_splits([edf for edf, _ in selected], test_frac, seed)
    names = scan_names([edf for edf, _ in selected])
    for name in ("train", "test"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)

    generated = failed = 0
    with (ROOT / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["image_path", "split", *LABELS])
        writer.writeheader()

        for index, (edf_path, class_name) in enumerate(selected, start=1):
            split = splits[edf_path]
            label = {
                "abnormal": str(class_name == "abnormal").lower(),
                "normal": str(class_name == "normal").lower(),
            }
            def label_fn(start, stop, label=label):
                return label, False
            try:
                for image_path, row_label in process(edf_path, names[edf_path], split, ROOT, label_fn,
                                                     window, train_windows, test_windows, seed, overwrite):
                    writer.writerow({"image_path": image_path, "split": split, **row_label})
                    generated += 1
                print(f"[{index}/{len(selected)}] [{split}] {names[edf_path]}")
            except Exception as exc:
                print(f"Failed {edf_path}: {exc}", file=sys.stderr)
                failed += 1

    print(f"Done. Generated: {generated}, failed: {failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render TUAB EDFs to flat PNGs and labels.csv.")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--limit", type=int, default=500, help="EDFs per class; use -1 for all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--window", type=float, default=20)
    parser.add_argument("--train-windows", type=int, default=4)
    parser.add_argument("--test-windows", type=int, default=10)
    args = parser.parse_args()

    limit = None if args.limit == -1 else args.limit
    if limit is not None and limit < 1:
        parser.error("--limit must be at least 1, or -1 for all")

    return generate(args.train_root, limit, args.overwrite, args.shuffle, args.seed, args.test_frac,
                    args.window, args.train_windows, args.test_windows)


if __name__ == "__main__":
    raise SystemExit(main())
