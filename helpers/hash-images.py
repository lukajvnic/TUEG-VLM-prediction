"""md5 of every rendered PNG, written to datasets/<DS>/hashes.csv (path,md5), sorted by path.

The corpora overlap at the recording level and the same window renders byte-identically, so a hash match is
the same EEG even when the patient token differs (TUEV's numeric eval dirs, aaaaaduk vs aaaaadul, ...).
The train sampler uses these files to keep any test image out of the fine-tune; they ship via git next to
labels.csv so the cluster never re-hashes 76k files (about 5 min locally).
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import DATASETS, HASHES, ROOT, read_csv, read_hashes


def hash_dataset(dataset):
    folder = ROOT / "datasets" / dataset
    rows = [(f"{split}/{png.name}", hashlib.md5(png.read_bytes()).hexdigest())
            for split in ("train", "test") for png in sorted((folder / split).glob("*.png"))]
    with (folder / HASHES).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "md5"])
        writer.writerows(rows)
    print(f"{dataset}: {len(rows)} images hashed")


def check(dataset):
    labeled = {r["path"] for r in read_csv(ROOT / "datasets" / dataset / "labels.csv")}
    hashed = set(read_hashes(dataset))
    missing, extra = labeled - hashed, hashed - labeled
    print(f"{dataset}: {len(labeled)} labels, {len(hashed)} hashes, {len(missing)} unhashed, {len(extra)} unlabeled")
    return not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    parser.add_argument("--check", action="store_true", help="compare hashes.csv against labels.csv, hash nothing")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if all([check(ds) for ds in args.datasets]) else "some labels.csv rows have no hash - re-run without --check")
    for ds in args.datasets:
        hash_dataset(ds)


if __name__ == "__main__":
    main()
