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
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Mutually-exclusive, recording-level datasets -> aggregate by majority vote.
# Everything else is multi-label event detection -> aggregate by any-positive.
BINARY = {"TUEP", "TUAB"}

# Classes below this many positives are too rare at the source to measure
# reliably; they're reported (with their support) but excluded from macro-F1.
MIN_SUPPORT = 20

# Resamples for the bootstrap confidence interval on macro-F1.
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0

# Below this many independent patients, a dataset cannot support an inferential
# claim no matter how many windows were rendered from them; report it, but say so.
MIN_PATIENTS = 20

# A model that emits one constant answer for everything can post a respectable
# macro-F1 on a class-imbalanced set without reading the image at all.
DEGENERATE_FRACTION = 0.95


def parse_set(text):
    text = text.strip()
    if text in ("set()", ""):
        return frozenset()
    return frozenset(ast.literal_eval(text))


def recording_of(image_path):
    # image name is <patient>_<scan>_<window>; the recording is <patient>_<scan>.
    return Path(image_path).stem.rsplit("_", 1)[0]


def patient_of(image_path):
    return Path(image_path).stem.split("_")[0]


def read_windows(results_csv):
    with results_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            yield (row["path"], recording_of(row["path"]),
                   parse_set(row["prediction"]), parse_set(row["actual"]))


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
    for _, recording, pred, true in windows:
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


def scored_classes(stats):
    return sorted(cls for cls, value in stats.items() if value[3] >= MIN_SUPPORT)


def macro_f1_from_counts(totals):
    """totals: (n_classes, 3) array of tp/fp/fn -> macro-F1."""
    tp, fp, fn = totals[:, 0], totals[:, 1], totals[:, 2]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(tp), where=denominator > 0)
    return float(f1.mean()) if len(f1) else 0.0


def bootstrap_macro_f1(groups, classes, iterations=BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Percentile CI for macro-F1, resampling whole recordings with replacement.

    Windows from one recording are correlated (same patient, same montage, same
    session), so resampling individual windows would understate the interval.
    The resampling unit is therefore the recording -- a cluster bootstrap -- which
    is also the unit the headline metric is computed over.
    """
    if not groups or not classes or iterations <= 0:
        return None

    counts = np.zeros((len(groups), len(classes), 3), dtype=np.float64)
    for i, pairs in enumerate(groups):
        for j, cls in enumerate(classes):
            for pred, true in pairs:
                counts[i, j, 0] += cls in pred and cls in true
                counts[i, j, 1] += cls in pred and cls not in true
                counts[i, j, 2] += cls not in pred and cls in true

    flat = counts.reshape(len(groups), -1)
    rng = np.random.default_rng(seed)
    n = len(groups)
    samples = np.empty(iterations)
    for b in range(iterations):
        # Multiplicities of a with-replacement resample of the n recordings.
        multiplicity = rng.multinomial(n, np.full(n, 1.0 / n))
        samples[b] = macro_f1_from_counts((multiplicity @ flat).reshape(len(classes), 3))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def baseline_macro_f1(pairs, classes, dataset):
    """Best macro-F1 reachable by a constant predictor that ignores the image.

    The floor any real result has to clear -- the exact shapes a model falls into
    when it stops looking and starts guessing the majority. A background-dominated
    multi-label set rewards these more than intuition suggests, which is why the
    comparison needs to be printed rather than assumed.

    Candidates are restricted to answers the dataset's schema can actually emit:
    TUAB/TUEP resolve to exactly one class (`is_abnormal` is a single bool), so
    "predict every class at once" is not available to them and must not be used
    to set their bar.
    """
    if not pairs or not classes:
        return 0.0
    candidates = [frozenset([cls]) for cls in classes]
    if dataset not in BINARY:
        candidates += [frozenset(classes), frozenset()]
    best = 0.0
    for guess in candidates:
        stats = per_class([(guess, true) for _, true in pairs])
        scored = [stats[cls][2] for cls in classes if cls in stats]
        if scored:
            best = max(best, sum(scored) / len(classes))
    return best


def degeneracy(pairs):
    """How concentrated the model's predictions are, regardless of correctness."""
    if not pairs:
        return None
    counts = Counter(pred for pred, _ in pairs)
    top, hits = counts.most_common(1)[0]
    return {
        "distinct": len(counts),
        "fraction": hits / len(pairs),
        "answer": ", ".join(sorted(top)) if top else "(nothing)",
    }


def report(title, stats, interval=None):
    scored = {c: v for c, v in stats.items() if v[3] >= MIN_SUPPORT}
    under = {c: v for c, v in stats.items() if v[3] < MIN_SUPPORT}
    macro = sum(v[2] for v in scored.values()) / len(scored) if scored else 0.0
    span = f"  95% CI [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else ""
    print(f"  {title}: macro-F1={macro:.3f}{span} "
          f"over {len(scored)} classes with support>={MIN_SUPPORT}")
    for cls, (precision, recall, f1, support) in scored.items():
        print(f"    {cls:12} P={precision:.3f} R={recall:.3f} F1={f1:.3f}  (support={support})")
    if under:
        print("    under-supported (excluded from macro-F1): "
              + ", ".join(f"{c}(n={v[3]})" for c, v in under.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP,
                        help="resamples for the macro-F1 CI (0 disables)")
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
        window_pairs = [(pred, true) for _, _, pred, true in windows]
        recording_pairs = aggregate_to_recordings(windows, dataset)
        patients = {patient_of(path) for path, _, _, _ in windows}

        print(f"\n=== {model}  |  {dataset}  ({len(windows)} windows, "
              f"{len(recording_pairs)} recordings, {len(patients)} patients) ===")

        # Both bootstraps resample the same clusters: whole recordings.
        by_recording = defaultdict(list)
        for _, recording, pred, true in windows:
            by_recording[recording].append((pred, true))
        recording_order = sorted(by_recording)

        window_stats = per_class(window_pairs)
        recording_stats = per_class(recording_pairs)
        window_groups = [by_recording[r] for r in recording_order]
        recording_groups = [[pair] for pair in recording_pairs]

        report("window-level", window_stats,
               bootstrap_macro_f1(window_groups, scored_classes(window_stats), args.bootstrap))
        recording_classes = scored_classes(recording_stats)
        recording_ci = bootstrap_macro_f1(recording_groups, recording_classes, args.bootstrap)
        report("recording-level", recording_stats, recording_ci)

        # The stage-1 promotion rule, computed rather than eyeballed.
        baseline = baseline_macro_f1(recording_pairs, recording_classes, dataset)
        if recording_ci:
            verdict = "ABOVE" if recording_ci[0] > baseline else "NOT above"
            print(f"  constant-predictor baseline: macro-F1={baseline:.3f} -- "
                  f"this model is {verdict} it (CI lower bound {recording_ci[0]:.3f})")

        shape = degeneracy(window_pairs)
        if shape and shape["fraction"] >= DEGENERATE_FRACTION:
            print(f"  !! degenerate output: {shape['fraction']:.1%} of windows answered "
                  f"\"{shape['answer']}\" ({shape['distinct']} distinct answers) -- "
                  f"this score does not reflect reading the image")
        elif shape:
            print(f"  output diversity: {shape['distinct']} distinct answers, "
                  f"most common \"{shape['answer']}\" on {shape['fraction']:.1%} of windows")

        if len(patients) < MIN_PATIENTS:
            print(f"  ~~ only {len(patients)} independent patients: report descriptively, "
                  f"not as an inferential result")


if __name__ == "__main__":
    main()
