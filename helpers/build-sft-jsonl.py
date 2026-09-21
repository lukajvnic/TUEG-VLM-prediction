import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "models"))
from helpers.pipeline import DATASETS, RATIONALE, ROOT, config, db, duplicate_images, parse_name, positives, read_csv
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
    payload["text_rationale"] = row[RATIONALE].strip()
    return json.dumps(structure(**payload).model_dump())  # key order follows the schema (train.rationale-first)


POOLED = "pooled"  # datasets/pooled/sft_*.jsonl: all six corpora in one file, image paths prefixed with the corpus


def train_scope(dataset):
    # which train windows the fine-tune sees is decided by train/sample-train-split.py, not here
    rows = db().execute("SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'rationale'", (dataset,))
    return {p.split("/", 1)[1] for (p,) in rows}


def entries_for(dataset, cfg, canon, pooled):
    # (jsonl line, canonical patient, labels) per eligible window; pooled lines carry the corpus in the image
    # path (relative to datasets/) and in a `dataset` field, and their labels are corpus-prefixed for the val split
    folder = ROOT / "datasets" / dataset
    rows = read_csv(folder / "labels.csv")
    scope = train_scope(dataset)
    if not scope:
        sys.exit(f"{dataset}: no train windows in scope - run train/sample-train-split.py first")
    assert all(p.startswith("train/") for p in scope), "test-split window in fine-tune scope"
    eligible = [r for r in rows if r["path"] in scope and r[RATIONALE].strip()]
    instruction = prompt(dataset)
    structure = get_structure(dataset, cfg["rationale-first"])
    entries, missing = [], 0
    for row in eligible:
        if not (folder / row["path"]).exists():
            missing += 1
            continue
        line = {"dataset": dataset, "instruction": instruction, "input": "",
                "output": output_json(dataset, row, structure),
                "images": [f"{dataset}/{row['path']}" if pooled else row["path"]]}
        patient = parse_name(row["path"])[0]
        labels = {f"{dataset}:{c}" for c in positives(row)} if pooled else positives(row)
        entries.append((line, canon.get(patient, patient), labels))
    print(f"{dataset}: {len(entries)} eligible, {missing} missing images skipped, "
          f"{len(scope)} in scope, {len(scope) - len(eligible)} of those still without a rationale")
    return entries


def split_and_write(name, entries, folder, cfg, dry_run):
    held_out = val_patients([(patient, labels) for _, patient, labels in entries],
                            cfg["val"]["fraction"], cfg["val"]["min-patients"])
    train = [line for line, patient, _ in entries if patient not in held_out]
    val = [line for line, patient, _ in entries if patient in held_out]
    train_patients = {patient for _, patient, _ in entries if patient not in held_out}
    assert not train_patients & held_out, "patient overlap between train and val"
    val_classes = Counter(c for _, patient, labels in entries if patient in held_out for c in labels)
    print(f"{name}: {len(train)} train ({len(train_patients)} patients), "
          f"{len(val)} val ({len(held_out)} patients; {', '.join(f'{c} {n}' for c, n in sorted(val_classes.items()))})")
    if dry_run:
        return
    folder.mkdir(exist_ok=True)
    for file, lines in (("sft_train.jsonl", train), ("sft_val.jsonl", val)):
        with (folder / file).open("w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")


def build(dataset, dry_run):
    cfg = config()["train"]
    _, _, canon = duplicate_images()  # patient tokens sharing an image are one patient for the val split too
    if dataset == POOLED:
        # one patient-level split over all six corpora: a patient is train or val everywhere, never both
        entries = [e for ds in DATASETS for e in entries_for(ds, cfg, canon, pooled=True)]
        split_and_write(POOLED, entries, ROOT / "datasets" / POOLED, cfg, dry_run)
        return
    split_and_write(dataset, entries_for(dataset, cfg, canon, pooled=False), ROOT / "datasets" / dataset, cfg, dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=["all"],
                        help=f"datasets to build (default: all six, one file pair each); `{POOLED}` = one combined pair "
                             f"under datasets/{POOLED}/ for the pooled run")
    parser.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    args = parser.parse_args()
    targets = DATASETS if args.datasets == ["all"] else args.datasets
    for dataset in targets:
        if dataset not in DATASETS and dataset != POOLED:
            sys.exit(f"unknown dataset {dataset!r} (choose from {', '.join(DATASETS)}, {POOLED})")
        build(dataset, args.dry_run)


if __name__ == "__main__":
    main()
