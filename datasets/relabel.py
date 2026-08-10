"""Add the `assessed` column to every labels.csv, and fix TUSZ background labels.

Why this exists
---------------
`labels_for()` in each generate.py returns the set of source annotations that
overlap a window. It returns the *empty set* in two situations that mean
completely different things:

  1. the annotations cover this window and say "nothing is happening", and
  2. the annotations never covered this window at all.

Case 1 is a background window -- a real negative, and a useful one. Case 2 is
*unknown* -- grading a model against it is grading against nothing. Both came
out of generation as an all-false row, so `score.py` silently treated case 2 as
"every class absent", where a window can only ever produce false positives and
can never contribute to recall.

Which case applies is a property of the corpus, established by measurement (see
knowledge/datasets.md, "Annotation coverage"):

  TUSZ  event annotations are complete -- verified against the official
        AAREADME checksums (seizure event counts and total seizure seconds match
        exactly) -- and no recording carries both `seiz` and `bckg` rows: the
        lone `bckg` row in a seizure-free recording is a token marker spanning a
        median of 0.2% of the duration, not a time-localised annotation. So
        absence of annotation *is* background. Unannotated -> bckg.

  TUEV  annotations cover a median 1.4% of each recording's duration, and
  TUSL  3.3%, and `bckg` is annotated explicitly. Absence therefore means
        "outside the annotated excerpt" = unknown, not background.
  TUAR  covers a median 29%, and generate.py drops the source `bckg` rows
        entirely, so an empty row cannot be distinguished from unassessed.
        All three: unannotated -> assessed=false.

  TUAB  labelled at the recording/patient level from the directory tree, so
  TUEP  every window inherits a known label. Always assessed.

Effect: no image is re-rendered and no window is re-derived from the EDFs. This
rewrites labels.csv only, so the images on disk (and on the cluster) stay valid.

Idempotent -- running it twice changes nothing.

Usage:  python datasets/relabel.py [--dry-run]
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Datasets whose labels come from the directory tree, not from time-localised
# annotations: every window inherits the recording/patient label.
RECORDING_LEVEL = {"TUAB", "TUEP"}

# Datasets where an unannotated window is genuinely background (see module docstring).
ABSENCE_IS_BACKGROUND = {"TUSZ"}

BACKGROUND_LABEL = "bckg"
ASSESSED = "assessed"


def relabel(rows, dataset):
    """Return (new_rows, stats). Pure function over labels.csv rows."""
    label_cols = [c for c in rows[0] if c not in ("image_path", "split", ASSESSED)]
    stats = Counter()
    out = []

    for row in rows:
        row = dict(row)
        positives = [c for c in label_cols if row[c].strip().lower() == "true"]

        if dataset in RECORDING_LEVEL or positives:
            row[ASSESSED] = "true"
            stats["assessed"] += 1
        elif dataset in ABSENCE_IS_BACKGROUND:
            # No annotation overlaps this window, and for this corpus that means
            # background. Promote it to a real negative rather than a blank row.
            row[BACKGROUND_LABEL] = "true"
            row[ASSESSED] = "true"
            stats["relabelled-background"] += 1
            stats["assessed"] += 1
        else:
            # Outside the annotated excerpt: we do not know what is in this window.
            row[ASSESSED] = "false"
            stats["unassessed"] += 1

        out.append(row)

    return out, stats


def process(dataset_dir, dry_run):
    dataset = dataset_dir.name
    labels_csv = dataset_dir / "labels.csv"
    with labels_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return

    if dataset in ABSENCE_IS_BACKGROUND and BACKGROUND_LABEL not in rows[0]:
        raise SystemExit(f"{dataset}: expected a '{BACKGROUND_LABEL}' column, found {list(rows[0])}")

    new_rows, stats = relabel(rows, dataset)
    fieldnames = [*rows[0], ASSESSED] if ASSESSED not in rows[0] else list(rows[0])

    test = [r for r in new_rows if r.get("split") == "test"]
    kept = sum(r[ASSESSED] == "true" for r in test)
    print(f"{dataset:5s} test {len(test):6d} -> {kept:6d} assessed "
          f"({kept / len(test):5.1%})   "
          + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))

    if dry_run:
        return

    with labels_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    for dataset_dir in sorted(ROOT.iterdir()):
        if (dataset_dir / "labels.csv").is_file():
            process(dataset_dir, args.dry_run)

    if args.dry_run:
        print("\n(dry run -- nothing written)")


if __name__ == "__main__":
    main()
