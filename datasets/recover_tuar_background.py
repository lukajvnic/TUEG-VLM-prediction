"""Recover TUAR's discarded `bckg` labels from the source annotations.

TUAR's generate.py keeps only `ARTIFACTS` and `SEIZURES` when reading the source
annotation CSVs, so its 3122 `bckg` annotation rows are dropped on the floor.
Two things follow, and both are wrong:

  * `structure.py` offers `has_bckg` and the eval prompt lists BCKG as a
    selectable category, but BCKG is never present in TUAR ground truth. A model
    that correctly reports a clean window is marked wrong every time -- an
    unwinnable option that only ever produces false positives.
  * Every remaining assessed TUAR window is positive for something, so the test
    set has no true negatives at all.

This restores the dropped labels. No image is re-rendered: a window's time span
is recoverable from its filename (`<patient>_<scan>_<index>` -> index*20 s), and
the recording it came from is recoverable from generate.py's own deterministic
`scan_names` map. The script proves that mapping is exact before writing
anything -- it recomputes every existing label from source and refuses to
continue unless all of them reproduce byte-for-byte.

Run this BEFORE relabel.py (which then marks the newly-labelled background
windows as assessed).

Usage:  python datasets/recover_tuar_background.py [--dry-run]
"""

import argparse
import csv
import importlib.util
from collections import Counter
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TUAR = ROOT / "TUAR"
WINDOW_SECONDS = 20.0
BACKGROUND = "bckg"


def load_generate():
    spec = importlib.util.spec_from_file_location("tuar_generate", TUAR / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_labels(edf, start, stop):
    """Every annotation label overlapping [start, stop), including `bckg`.

    generate.py's `labels_for` filters to the artifact/seizure vocabularies; this
    keeps everything so background and coverage can be distinguished.
    """
    found = set()
    for annotation in (edf.with_suffix(".csv"), edf.with_name(edf.stem + "_seiz.csv")):
        if not annotation.exists():
            continue
        lines = [x for x in annotation.read_text(errors="ignore").splitlines()
                 if x and not x.startswith("#")]
        for row in csv.DictReader(StringIO("\n".join(lines))):
            if float(row["start_time"]) >= stop or float(row["stop_time"]) <= start:
                continue
            found.update(row.get("label", "").split("_"))
    return found


def window_span(image_path):
    patient, scan, index = Path(image_path).stem.rsplit("_", 2)
    start = int(index) * WINDOW_SECONDS
    return f"{patient}_{scan}", start, start + WINDOW_SECONDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    generate = load_generate()
    edfs = sorted(generate.EDF_ROOT.rglob("*.edf"))
    by_scan = {name: edf for edf, name in generate.scan_names(edfs).items()}

    labels_csv = TUAR / "labels.csv"
    with labels_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    known = [c for c in rows[0] if c not in ("image_path", "split", "assessed", BACKGROUND)]

    # Safety gate: the recovered labels are only trustworthy if the existing ones
    # reproduce exactly from the same (recording, window) derivation.
    stats = Counter()
    updated = []
    for row in rows:
        scan, start, stop = window_span(row["image_path"])
        edf = by_scan.get(scan)
        if edf is None:
            raise SystemExit(f"no source recording for {row['image_path']} (scan {scan})")

        recomputed = generate.labels_for(edf, start, stop)
        existing = {c for c in known if row[c].strip().lower() == "true"}
        if recomputed != existing:
            raise SystemExit(
                f"refusing to write: {row['image_path']} does not reproduce from source "
                f"(labels.csv={sorted(existing)}, source={sorted(recomputed)})"
            )
        stats["verified"] += 1

        row = dict(row)
        is_background = BACKGROUND in source_labels(edf, start, stop)
        row[BACKGROUND] = str(is_background).lower()
        stats["background" if is_background else "not-background"] += 1
        updated.append(row)

    fieldnames = ["image_path", "split", BACKGROUND, *known]
    if "assessed" in rows[0]:
        fieldnames.append("assessed")

    test = [r for r in updated if r.get("split") == "test"]
    print(f"verified {stats['verified']}/{len(rows)} existing labels reproduce from source")
    print(f"new {BACKGROUND} positives: {stats['background']} of {len(rows)} rows "
          f"({sum(r[BACKGROUND] == 'true' for r in test)} in the test split of {len(test)})")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    with labels_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)
    print(f"wrote {labels_csv}")


if __name__ == "__main__":
    main()
