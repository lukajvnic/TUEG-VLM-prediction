# train/val/test leakage check for the per-dataset fine-tune splits; build-sft-jsonl.py runs it before writing
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import DATASETS, ROOT, config, duplicate_images, parse_name, read_csv, sft_file  # noqa: E402

WITHIN = ("patient", "token", "recording", "image")  # train vs val of one dataset
AGAINST_TEST = ("test patient", "test image")  # train + val vs the test splits
SHOW = 3


class Leakage:
    """Keys a fine-tune window can share with another split: the merged patient (tokens with a byte-identical
    image are one patient, as in the samplers), the raw token, the recording and the image md5. The test side
    holds every test image of all six datasets, and every test patient of all six unless
    exclude-cross-dataset-test-patients is off, then only the dataset's own."""

    def __init__(self):
        self.hashes, _, self.canon = duplicate_images()
        everywhere = config()["settings"]["train-sample"]["exclude-cross-dataset-test-patients"]

        tests = {ds: self.keys(ds, [r["path"] for r in read_csv(ROOT / "datasets" / ds / "labels.csv")
                                    if r["path"].startswith("test/")])
                 for ds in DATASETS}
        self.test_images = set().union(*(keys["image"] for keys in tests.values()))
        everyone = set().union(*(keys["patient"] for keys in tests.values()))
        self.test_patients = {ds: everyone if everywhere else tests[ds]["patient"] for ds in DATASETS}

    def patient(self, path):
        token = parse_name(path)[0]
        return self.canon.get(token, token)

    def keys(self, dataset, paths):
        unhashed = [p for p in paths if p not in self.hashes[dataset]]
        if unhashed:
            sys.exit(f"{dataset}: {len(unhashed)} images without a hash (first {unhashed[0]}) - "
                     f"run helpers/hash-images.py {dataset}")

        return {"patient": {self.patient(p) for p in paths},
                "token": {parse_name(p)[0] for p in paths},
                "recording": {parse_name(p)[1] for p in paths},
                "image": {self.hashes[dataset][p] for p in paths}}

    def overlaps(self, dataset, train, val):
        # train, val: image paths relative to datasets/<dataset>/. Maps each check to the keys found on both sides
        train_keys, val_keys = self.keys(dataset, train), self.keys(dataset, val)

        found = {check: train_keys[check] & val_keys[check] for check in WITHIN}
        found["test patient"] = (train_keys["patient"] | val_keys["patient"]) & self.test_patients[dataset]
        found["test image"] = (train_keys["image"] | val_keys["image"]) & self.test_images
        return found


def describe(found):
    parts = []
    for check, keys in found.items():
        if keys:
            shown = ", ".join(sorted(keys)[:SHOW]) + (", ..." if len(keys) > SHOW else "")
            parts.append(f"{len(keys)} {check} ({shown})")
    return "; ".join(parts)


def excerpt_warnings(dataset, train, val):
    # TUEV's eval files carry a numeric excerpt id where the subject token goes, and one subject can sit under
    # several ids (knowledge/datasets.md), so for these only the image-hash merge stands between train and val
    warnings = []
    for split, paths in (("train", train), ("val", val)):
        tokens = Counter(token for token in (parse_name(p)[0] for p in paths) if token.isdigit())
        if tokens:
            listed = ", ".join(f"{token} {n}" for token, n in sorted(tokens.items()))
            warnings.append(f"{dataset}: {len(tokens)} numeric excerpt ids in {split} ({sum(tokens.values())} "
                            f"windows), subject-disjointness unprovable for them: {listed}")
    return warnings


def read_split(dataset, split, target):
    path = sft_file(dataset, split, target)
    if not path.exists():
        return None

    with path.open() as f:
        return [json.loads(line)["images"][0] for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="check the train.target jsonl (datasets/<DS>/sft_[labels_]{train,val}.jsonl) for train/val/test overlap")
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()
    unknown = [ds for ds in args.datasets if ds not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset {', '.join(unknown)} (choose from {', '.join(DATASETS)})")

    target = config()["train"]["target"]
    leakage = Leakage()
    missing, warnings, leaks = [], [], []
    print(f"{'':6}{'train (patients)':>18}{'val (patients)':>16}" + "".join(f"{c:>14}" for c in WITHIN + AGAINST_TEST))
    for ds in args.datasets:
        train, val = read_split(ds, "train", target), read_split(ds, "val", target)
        if train is None or val is None:
            missing.append(ds)
            continue

        found = leakage.overlaps(ds, train, val)
        sizes = [f"{len(paths)} ({len({leakage.patient(p) for p in paths})})" for paths in (train, val)]
        print(f"{ds:6}{sizes[0]:>18}{sizes[1]:>16}" + "".join(f"{len(found[c]):>14}" for c in WITHIN + AGAINST_TEST))
        warnings += excerpt_warnings(ds, train, val)
        if any(found.values()):
            leaks.append(f"{ds}: {describe(found)}")
    for warning in warnings:
        print(f"warning: {warning}")
    if missing:
        print(f"no {sft_file('<DS>', 'train', target).name} / val yet: {', '.join(missing)}")
    for leak in leaks:
        print(f"LEAK {leak}")
    sys.exit(1 if leaks else 0)


if __name__ == "__main__":
    main()
