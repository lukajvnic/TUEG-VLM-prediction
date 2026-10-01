import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "models"))
from helpers.pipeline import DATASETS, RATIONALE, ROOT, TARGETS, config, db, positives, read_csv, sft_file
from structure import BINARY, classes, get_structure, prompt


def val_patients(entries, fraction, min_patients):
    # entries: (patient, labels). Patients ranked by a stable hash; every class with >= 2 patients gets one
    # val patient first (rarest class first), then hash order fills the rest, so val is never single-class
    by_patient = defaultdict(set)
    for patient, labels in entries:
        by_patient[patient] |= labels
    ranked = sorted(by_patient, key=lambda p: hashlib.md5(p.encode()).hexdigest())
    count = min(max(math.ceil(len(ranked) * fraction), min_patients), len(ranked) // 2)
    by_class = defaultdict(list)
    for patient in ranked:
        for cls in by_patient[patient]:
            by_class[cls].append(patient)
    chosen = []
    for cls, patients in sorted(by_class.items(), key=lambda kv: (len(kv[1]), kv[0])):
        if len(patients) >= 2 and len(chosen) < count and not any(p in chosen for p in patients):
            chosen.append(patients[0])
    chosen += [p for p in ranked if p not in chosen][:max(count - len(chosen), 0)]
    return set(chosen)


def output_json(dataset, row, structure):
    if dataset in BINARY:
        field, pos, _ = BINARY[dataset]
        payload = {field: row[pos].strip().lower() == "true"}
    else:
        payload = {f"has_{c}": row[c].strip().lower() == "true" for c in classes(dataset)}
    if "text_rationale" in structure.model_fields:
        payload["text_rationale"] = row[RATIONALE].strip()
    return json.dumps(structure(**payload).model_dump())  # key order follows the schema (train.rationale-first)


def check_splits():
    # helpers/check-splits.py, loaded by file path for its hyphen
    spec = importlib.util.spec_from_file_location("check_splits", Path(__file__).with_name("check-splits.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_scope(dataset):
    # which train windows the fine-tune sees is decided by train/scripts/sample-train-split.py, not here
    rows = db().execute("SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'rationale'", (dataset,))
    return {p.split("/", 1)[1] for (p,) in rows}


def entries_for(dataset, cfg, leakage):
    # ({target: jsonl line}, merged patient, labels) per eligible window. Both targets get the same windows, so the
    # label-only and rationale runs share one split and differ only in the answer
    folder = ROOT / "datasets" / dataset
    rows = read_csv(folder / "labels.csv")
    scope = train_scope(dataset)
    if not scope:
        sys.exit(f"{dataset}: no train windows in scope - run train/scripts/sample-train-split.py first")
    assert all(p.startswith("train/") for p in scope), "test-split window in fine-tune scope"

    eligible = [r for r in rows if r["path"] in scope and r[RATIONALE].strip()]
    rationale = {target: target == "rationale" for target in TARGETS}
    instructions = {target: prompt(dataset, rationale[target]) for target in TARGETS}
    structures = {target: get_structure(dataset, cfg["rationale-first"], rationale[target]) for target in TARGETS}
    entries, missing = [], 0
    for row in eligible:
        if not (folder / row["path"]).exists():
            missing += 1
            continue
        lines = {target: {"dataset": dataset, "instruction": instructions[target], "input": "",
                          "output": output_json(dataset, row, structures[target]), "images": [row["path"]]}
                 for target in TARGETS}
        entries.append((lines, leakage.patient(row["path"]), positives(row)))
    print(f"{dataset}: {len(entries)} eligible, {missing} missing images skipped, "
          f"{len(scope)} in scope, {len(scope) - len(eligible)} of those still without a rationale")
    return entries


def split(dataset, entries, cfg):
    held_out = val_patients([(patient, labels) for _, patient, labels in entries],
                            cfg["val"]["fraction"], cfg["val"]["min-patients"])
    train = [lines for lines, patient, _ in entries if patient not in held_out]
    val = [lines for lines, patient, _ in entries if patient in held_out]

    train_patients = {patient for _, patient, _ in entries if patient not in held_out}
    val_classes = Counter(c for _, patient, labels in entries if patient in held_out for c in labels)
    print(f"{dataset}: {len(train)} train ({len(train_patients)} patients), "
          f"{len(val)} val ({len(held_out)} patients; {', '.join(f'{c} {n}' for c, n in sorted(val_classes.items()))})")
    return train, val


def write(dataset, train, val):
    for target in TARGETS:
        for split_name, entries in (("train", train), ("val", val)):
            with sft_file(dataset, split_name, target).open("w") as f:
                for lines in entries:
                    f.write(json.dumps(lines[target]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="build datasets/<DS>/sft_[labels_]{train,val}.jsonl, same split for both targets")
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    parser.add_argument("--dry-run", action="store_true", help="print counts and the leakage check, write nothing")
    args = parser.parse_args()
    unknown = [ds for ds in args.datasets if ds not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset {', '.join(unknown)} (choose from {', '.join(DATASETS)})")

    cfg = config()["train"]
    splits = check_splits()
    leakage = splits.Leakage()  # also merges patient tokens that share an image, for the val split

    built, leaks = {}, []
    for dataset in args.datasets:
        train, val = split(dataset, entries_for(dataset, cfg, leakage), cfg)
        train_images = [lines["rationale"]["images"][0] for lines in train]
        val_images = [lines["rationale"]["images"][0] for lines in val]
        found = leakage.overlaps(dataset, train_images, val_images)
        for warning in splits.excerpt_warnings(dataset, train_images, val_images):
            print(f"warning: {warning}")
        if any(found.values()):
            leaks.append(f"{dataset}: {splits.describe(found)}")
        if not train or not val:
            leaks.append(f"{dataset}: empty {'train' if not train else 'val'} split (no rationales yet?)")
        built[dataset] = train, val

    # every dataset is checked before any file is touched, so a leak anywhere leaves all the old files in place
    if leaks:
        sys.exit("train/val/test overlap or empty split, nothing written:\n  " + "\n  ".join(leaks))
    if args.dry_run:
        print("no overlap (dry run, nothing written)")
        return

    for dataset, (train, val) in built.items():
        write(dataset, train, val)
    print(f"no overlap; wrote sft_[labels_]train/val.jsonl for {', '.join(built)}")


if __name__ == "__main__":
    main()
