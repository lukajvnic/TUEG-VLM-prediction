#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, random, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _render import process


ROOT = Path(__file__).resolve().parent
EDF_ROOT = ROOT / "v2.0.1" / "edf"
LABELS = ["bckg", "spsw", "gped", "pled", "eyem", "artf"]
REC_LABELS = {"1": "spsw", "2": "gped", "3": "pled", "4": "eyem", "5": "artf", "6": "bckg"}


def labels_for(edf: Path, start: float, stop: float) -> set[str]:
    rec = edf.with_suffix(".rec")
    if not rec.exists(): return set()
    found = set()
    for line in rec.read_text(errors="ignore").splitlines():
        try: _, a, b, code = [x.strip() for x in line.split(",")]
        except ValueError: continue
        label = REC_LABELS.get(code)
        if label in LABELS and float(a) < stop and float(b) > start: found.add(label)
    return found


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


def main() -> int:
    p = argparse.ArgumentParser(description="Render TUEV EDF windows to PNGs and write multi-label event booleans for that window.")
    p.add_argument("--limit", type=int); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--window", type=float, default=20)
    p.add_argument("--train-windows", type=int, default=4); p.add_argument("--test-windows", type=int, default=10)
    p.add_argument("--test-frac", type=float, default=0.3); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    edfs = sorted(EDF_ROOT.rglob("*.edf"))[:a.limit]
    splits = assign_splits(edfs, a.test_frac, a.seed)
    names = scan_names(edfs)
    for d in ("train", "test"): (ROOT / d).mkdir(exist_ok=True)
    with (ROOT / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["image_path", "split", *LABELS]); w.writeheader()
        for n, edf in enumerate(edfs, 1):
            split = splits[edf]
            def label_fn(start, stop, edf=edf):
                labs = labels_for(edf, start, stop)
                return {x: str(x in labs).lower() for x in LABELS}, bool(labs - {"bckg"})
            try:
                for image_path, label in process(edf, names[edf], split, ROOT, label_fn,
                                                 a.window, a.train_windows, a.test_windows, a.seed, a.overwrite):
                    w.writerow({"image_path": image_path, "split": split, **label})
                print(f"{n}/{len(edfs)} [{split}] {names[edf]}")
            except Exception as e: print(f"failed {edf}: {e}", file=sys.stderr)
    return 0

if __name__ == "__main__": raise SystemExit(main())
