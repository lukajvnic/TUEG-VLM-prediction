"""Deterministic stratified subsampling of the test split.

The benchmark is scored at the **recording** level (summarize.py aggregates a
recording's windows into one prediction), and for TUEP the ground truth is
actually per *patient*. So the precision of every headline metric is set by the
number of recordings/patients -- while inference cost scales with the number of
windows. Densely tiling 10-40 background windows per recording buys almost no
statistical precision and costs the majority of the compute.

This module cuts the redundancy without touching what the benchmark measures:

  * every window carrying an under-supported class is kept, in full, always --
    rare-class recall is the number the benchmark exists to report, and it is
    numerically identical before and after sampling;
  * abundant signatures (and background) are capped per recording, keeping the
    recording present and its aggregate label unchanged;
  * for patient-level TUEP, a patient's near-duplicate recordings are capped too.

Selection is by even spacing over the window index -- no RNG -- so every model
evaluates the identical set and the sample is reproducible from labels.csv
alone. There is no sample file to go stale.

Usage (inspection):  python eval/scripts/sample.py [--policy-json '{...}']
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Mutually-exclusive datasets labelled at the recording level; every window
# carries the same label, so signature-based stratification is meaningless.
BINARY = {"TUEP", "TUAB"}

# Ground truth is per patient, so a patient's recordings are near-duplicates.
PATIENT_LEVEL = {"TUEP"}

BACKGROUND = {"bckg"}

DEFAULT_POLICY = {
    # A class present in fewer than this many test recordings is "under-supported":
    # every window carrying it is protected from capping. Set at 10x summarize.py's
    # MIN_SUPPORT=20 so no class near the reporting threshold is ever thinned.
    "abundant-class-recordings": 200,
    # Cap per (recording, label-signature) for signatures made only of abundant classes.
    "windows-per-signature": 3,
    # Cap per recording for background-only / all-negative windows.
    "background-per-recording": 2,
    # Cap per recording for the recording-level binary datasets (majority vote).
    "binary-windows-per-recording": 4,
    # Cap per patient for patient-level datasets (TUEP).
    "recordings-per-patient": 4,
}


def parse_name(image_path):
    """`test/<patient>_<scan>_<window>.png` -> (patient, recording, window index)."""
    patient, scan, window = Path(image_path).stem.rsplit("_", 2)
    return patient, f"{patient}_{scan}", int(window)


def spread(items, cap):
    """Evenly-spaced deterministic subset of `items`, preserving order.

    Even spacing rather than "first N" so a capped recording still samples across
    its whole duration instead of clustering at the start.
    """
    if cap is None or cap <= 0 or cap >= len(items):
        return list(items)
    if cap == 1:
        return [items[len(items) // 2]]
    picked = {round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)}
    return [items[i] for i in sorted(picked)]


def label_columns(rows):
    return [c for c in rows[0] if c not in ("image_path", "split", "assessed")]


def positives(row, columns):
    return frozenset(c for c in columns if row[c].strip().lower() == "true")


def abundant_classes(rows, columns, threshold):
    """Classes appearing in >= `threshold` distinct test recordings."""
    by_class = defaultdict(set)
    for row in rows:
        _, recording, _ = parse_name(row["image_path"])
        for cls in positives(row, columns):
            by_class[cls].add(recording)
    return {cls for cls, recordings in by_class.items() if len(recordings) >= threshold}


def select(rows, dataset, policy=None):
    """Return the sorted list of image_path values to evaluate.

    `rows` must already be filtered to the test split. Unassessed windows are
    dropped here (a window the source annotations never covered cannot be graded).
    """
    policy = {**DEFAULT_POLICY, **(policy or {})}
    rows = [r for r in rows if r.get("assessed", "true").strip().lower() != "false"]
    if not rows:
        return []

    columns = label_columns(rows)
    abundant = abundant_classes(rows, columns, policy["abundant-class-recordings"])

    by_recording = defaultdict(list)
    patients = defaultdict(set)
    for row in rows:
        patient, recording, window = parse_name(row["image_path"])
        by_recording[recording].append((window, row))
        patients[patient].add(recording)

    if dataset in PATIENT_LEVEL:
        keep_recordings = set()
        for patient in sorted(patients):
            keep_recordings.update(
                spread(sorted(patients[patient]), policy["recordings-per-patient"])
            )
    else:
        keep_recordings = set(by_recording)

    chosen = []
    for recording in sorted(keep_recordings):
        windows = [row for _, row in sorted(by_recording[recording])]

        if dataset in BINARY:
            chosen += spread(windows, policy["binary-windows-per-recording"])
            continue

        by_signature = defaultdict(list)
        for row in windows:
            by_signature[positives(row, columns)].append(row)

        for signature, group in sorted(by_signature.items(), key=lambda kv: sorted(kv[0])):
            if signature - abundant - BACKGROUND:
                chosen += group                                    # under-supported class: keep all
            elif not signature - BACKGROUND:
                chosen += spread(group, policy["background-per-recording"])
            else:
                chosen += spread(group, policy["windows-per-signature"])

    return sorted(row["image_path"] for row in chosen)


def probe(rows, dataset, size, policy=None):
    """A small screening subset of `select`, stratified by label signature.

    Stage 1 of the two-stage design: run every model over this, and promote only
    the models whose recording-level macro-F1 confidence interval clears the
    constant-predictor baseline (see summarize.py). Most general-purpose VLMs are
    expected to sit at chance on EEG waveforms, and establishing that does not
    need the full test set -- but it does need the rare classes to be present,
    which is why this stratifies by signature instead of slicing the front of a
    sorted list the way `limit` does.
    """
    paths = select(rows, dataset, policy)
    if not size or size >= len(paths):
        return paths

    by_path = {row["image_path"]: row for row in rows}
    columns = label_columns(rows)
    groups = defaultdict(list)
    for path in paths:
        groups[positives(by_path[path], columns)].append(path)

    # Rarest signatures first, so a tight budget still covers the rare classes.
    order = sorted(groups, key=lambda sig: (len(groups[sig]), sorted(sig)))
    quota = {sig: 0 for sig in order}
    for sig in order:                                   # floor of one per signature
        if sum(quota.values()) >= size:
            break
        quota[sig] = 1
    remaining = size - sum(quota.values())
    total = sum(len(groups[sig]) for sig in order)
    for sig in order:                                   # then proportional to size
        if remaining <= 0:
            break
        extra = min(round(remaining * len(groups[sig]) / total), len(groups[sig]) - quota[sig])
        quota[sig] += max(extra, 0)

    chosen = [p for sig in order for p in spread(groups[sig], quota[sig])]
    return sorted(chosen)


def load_test_rows(dataset_dir):
    with (dataset_dir / "labels.csv").open(newline="", encoding="utf-8") as file:
        return [r for r in csv.DictReader(file) if r.get("split") == "test"]


def describe(dataset, rows, kept_paths):
    """Prove the sample preserves what the benchmark reports."""
    columns = label_columns(rows)
    kept = set(kept_paths)
    subset = [r for r in rows if r["image_path"] in kept]

    def support(sample, level):
        by_class = defaultdict(set)
        for row in sample:
            patient, recording, _ = parse_name(row["image_path"])
            unit = patient if level == "patient" else recording
            for cls in positives(row, columns):
                by_class[cls].add(unit)
        return {cls: len(v) for cls, v in by_class.items()}

    # The scoring unit: patient for patient-level truth, recording otherwise.
    level = "patient" if dataset in PATIENT_LEVEL else "recording"
    assessed = [r for r in rows if r.get("assessed", "true").strip().lower() != "false"]
    before, after = support(assessed, level), support(subset, level)
    lost = {c: (before[c], after.get(c, 0)) for c in before if after.get(c, 0) != before[c]}

    print(f"\n=== {dataset} ===")
    print(f"  test rows {len(rows)} -> assessed {len(assessed)} -> sampled {len(subset)} "
          f"({len(subset) / len(rows):.1%} of original)")
    print(f"  recordings: {len({parse_name(r['image_path'])[1] for r in subset})}, "
          f"patients: {len({parse_name(r['image_path'])[0] for r in subset})}")
    print(f"  {level}-level class support: {dict(sorted(after.items()))}")
    if lost:
        print(f"  ~~ {level}-level support changed by sampling: {lost}")
    else:
        print(f"  {level}-level support identical to the full assessed set")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-json", default="{}", help="JSON overrides for the default policy")
    args = parser.parse_args()
    policy = {**DEFAULT_POLICY, **json.loads(args.policy_json)}

    root = Path(__file__).resolve().parents[2] / "datasets"
    total_before = total_after = 0
    for dataset_dir in sorted(root.iterdir()):
        if not (dataset_dir / "labels.csv").is_file():
            continue
        rows = load_test_rows(dataset_dir)
        if not rows:
            continue
        kept = select(rows, dataset_dir.name, policy)
        describe(dataset_dir.name, rows, kept)
        total_before += len(rows)
        total_after += len(kept)

    print(f"\nTOTAL test images {total_before} -> {total_after} ({total_after / total_before:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
