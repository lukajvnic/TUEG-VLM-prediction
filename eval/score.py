"""Score an eval run with metrics that match what the benchmark is actually
trying to measure.

Two things the raw per-window `correct` flag gets wrong:

  * Accuracy on a background-dominated, window-tiled test set flatters a model
    that just says "background". So we report per-class precision/recall/F1
    (recall on the rare event classes is the number that matters).

  * TUAB/TUEP are labelled at the recording level (abnormal / epilepsy), so a
    single normal-looking window of an abnormal recording is not a fair unit to
    grade. We aggregate a recording's windows into one prediction and score at
    the recording level — the meaningful clinical unit, and it answers "did the
    model catch the finding somewhere in the recording".

Usage:  python eval/score.py <run-name>   (run dir under eval/runs/)
"""

import argparse
import ast
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Mutually-exclusive, recording-level datasets -> aggregate by majority vote.
# Everything else is multi-label event detection -> aggregate by any-positive.
BINARY = {"TUEP", "TUAB"}

# Classes below this many positives are too rare at the source to measure
# reliably; they're reported (with their support) but excluded from macro-F1.
MIN_SUPPORT = 20


def parse_set(text):
    text = text.strip()
    if text in ("set()", ""):
        return frozenset()
    return frozenset(ast.literal_eval(text))


def recording_of(image_path):
    # image name is <patient>_<scan>_<window>; the recording is <patient>_<scan>.
    return Path(image_path).stem.rsplit("_", 1)[0]


def read_windows(results_csv):
    with results_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            yield recording_of(row["path"]), parse_set(row["prediction"]), parse_set(row["actual"])


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def per_class(pairs):
    """pairs: list of (pred_set, true_set) -> {class: (precision, recall, f1, support)}."""
    classes = set()
    for pred, true in pairs:
        classes |= pred | true
    out = {}
    for cls in sorted(classes):
        tp = sum(1 for pred, true in pairs if cls in pred and cls in true)
        fp = sum(1 for pred, true in pairs if cls in pred and cls not in true)
        fn = sum(1 for pred, true in pairs if cls not in pred and cls in true)
        out[cls] = (*prf(tp, fp, fn), tp + fn)
    return out


def aggregate_to_recordings(windows, dataset):
    by_recording = defaultdict(lambda: {"preds": [], "trues": []})
    for recording, pred, true in windows:
        by_recording[recording]["preds"].append(pred)
        by_recording[recording]["trues"].append(true)

    pairs = []
    for group in by_recording.values():
        true_rec = frozenset().union(*group["trues"])  # positive if present in any window
        if dataset in BINARY:
            votes = Counter(cls for pred in group["preds"] for cls in pred)
            pred_rec = frozenset([votes.most_common(1)[0][0]]) if votes else frozenset()
        else:
            pred_rec = frozenset().union(*group["preds"])  # detected anywhere
        pairs.append((pred_rec, true_rec))
    return pairs


def report(title, stats):
    scored = {c: v for c, v in stats.items() if v[3] >= MIN_SUPPORT}
    under = {c: v for c, v in stats.items() if v[3] < MIN_SUPPORT}
    macro = sum(v[2] for v in scored.values()) / len(scored) if scored else 0.0
    print(f"  {title}: macro-F1={macro:.3f} over {len(scored)} classes with support>={MIN_SUPPORT}")
    for cls, (precision, recall, f1, support) in scored.items():
        print(f"    {cls:12} P={precision:.3f} R={recall:.3f} F1={f1:.3f}  (support={support})")
    if under:
        print("    under-supported (excluded from macro-F1): "
              + ", ".join(f"{c}(n={v[3]})" for c, v in under.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "runs" / args.run / "results"
    files = sorted(results_dir.glob("*.csv"))
    if not files:
        sys.exit(f"No results CSVs in {results_dir}")

    for results_csv in files:
        model, dataset = results_csv.stem.rsplit("-", 1)
        windows = list(read_windows(results_csv))
        if not windows:
            continue
        window_pairs = [(pred, true) for _, pred, true in windows]
        recording_pairs = aggregate_to_recordings(windows, dataset)

        print(f"\n=== {model}  |  {dataset}  "
              f"({len(windows)} windows, {len(recording_pairs)} recordings) ===")
        report("window-level", per_class(window_pairs))
        report("recording-level", per_class(recording_pairs))


if __name__ == "__main__":
    main()
