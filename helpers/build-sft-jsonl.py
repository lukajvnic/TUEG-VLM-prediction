#!/usr/bin/env python3
"""Build sft_train.jsonl / sft_val.jsonl for the fine-tune from labels.csv.

One line per train-split window with a ground-truth rationale:
- instruction: the exact eval prompt (config.yml `prompts:` via structure.py),
  so the fine-tuned model is trained on the question eval will ask it.
- output: the eval schema JSON (label booleans + text_rationale), validated
  against structure.get_structure() before writing.
- images: [path relative to the dataset dir], as the collator expects.

The val split is patient-level: patients are ranked by a stable hash and the
first VAL_FRACTION of them (at least MIN_VAL_PATIENTS, so tiny datasets keep a
usable val set) go to sft_val.jsonl; no patient appears in both files.
Test-split rows never qualify (path prefix filter), preserving the
patient-level train/test split.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "models"))
from helpers.pipeline import DATASETS, RATIONALE, ROOT, read_csv
from structure import BINARY, classes, get_structure, prompt

VAL_FRACTION = 0.05   # of train patients
MIN_VAL_PATIENTS = 3  # floor so tiny datasets still get a usable val set


def patient(path):
    return Path(path).name.split("_")[0]


def val_patients(patients):
    ranked = sorted(patients, key=lambda p: hashlib.md5(p.encode()).hexdigest())
    count = max(math.ceil(len(ranked) * VAL_FRACTION), MIN_VAL_PATIENTS)
    return set(ranked[:min(count, len(ranked) // 2)])


def output_json(dataset, row):
    if dataset in BINARY:
        field, pos, _ = BINARY[dataset]
        payload = {field: row[pos].strip().lower() == "true"}
    else:
        payload = {f"has_{c}": row[c].strip().lower() == "true" for c in classes(dataset)}
    payload["text_rationale"] = row[RATIONALE].strip()
    get_structure(dataset)(**payload)
    return json.dumps(payload)


def build(dataset, dry_run):
    folder = ROOT / "datasets" / dataset
    rows = read_csv(folder / "labels.csv")
    eligible = [r for r in rows if r["path"].startswith("train/") and r[RATIONALE].strip()]
    instruction = prompt(dataset)

    lines, missing = [], 0
    for row in eligible:
        if not (folder / row["path"]).exists():
            missing += 1
            continue
        lines.append({"instruction": instruction, "input": "",
                      "output": output_json(dataset, row), "images": [row["path"]]})

    held_out = val_patients({patient(l["images"][0]) for l in lines})
    train = [l for l in lines if patient(l["images"][0]) not in held_out]
    val = [l for l in lines if patient(l["images"][0]) in held_out]

    train_patients = {patient(l["images"][0]) for l in train}
    assert not train_patients & held_out, "patient overlap between train and val"
    print(f"{dataset}: {len(train)} train ({len(train_patients)} patients), "
          f"{len(val)} val ({len(held_out)} patients), "
          f"{missing} missing images skipped, "
          f"{len(rows) - len(eligible)} rows ineligible (test split or no rationale)")

    if dry_run:
        return
    for name, lines in (("sft_train.jsonl", train), ("sft_val.jsonl", val)):
        with (folder / name).open("w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=["all"],
                        help="datasets to build (default: all six)")
    parser.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    args = parser.parse_args()
    targets = DATASETS if args.datasets == ["all"] else args.datasets
    for dataset in targets:
        if dataset not in DATASETS:
            sys.exit(f"unknown dataset {dataset!r} (choose from {', '.join(DATASETS)})")
        build(dataset, args.dry_run)


if __name__ == "__main__":
    main()
