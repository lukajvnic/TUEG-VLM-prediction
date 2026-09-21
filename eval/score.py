import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from helpers.pipeline import DATASETS, ROOT, db, parse_name, read_csv
from structure import BINARY, labels

MIN_SUPPORT = 20          # recordings a class needs before it counts toward macro-F1 / balanced accuracy
# classes left out of the recording-level mean: TUSZ bckg is "no seizure", the complement of the other classes,
# and is true in 99% of test recordings (2,418 of 2,443 sampled), so its recording-level balanced accuracy is
# ~0.5 for every model and only dilutes the seizure classes. Still reported per class and at window level.
RECORDING_EXCLUDE = {"TUSZ": {"bckg"}}
MIN_PATIENTS = 20         # below this a dataset is descriptive only
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
DEGENERATE_FRACTION = 0.95  # one answer for >=95% of windows: the model is not reading the image

FIELDS = ("model", "dataset", "windows", "coverage", "recordings", "patients",
          "recording_balanced_accuracy", "recording_macro_f1", "recording_ci_low", "recording_ci_high",
          "baseline_macro_f1", "clears_baseline", "window_balanced_accuracy", "window_macro_f1",
          "distinct_answers", "top_answer", "top_answer_fraction", "degenerate",
          "scored_classes", "low_support_classes", "few_patients")
CLASS_FIELDS = ("model", "dataset", "level", "class", "precision", "recall", "f1", "support", "in_macro")


def divide(a, b):
    return a / b if b else 0.0


def counts(pairs, cls):
    tp = sum(1 for pred, true in pairs if cls in pred and cls in true)
    fp = sum(1 for pred, true in pairs if cls in pred and cls not in true)
    fn = sum(1 for pred, true in pairs if cls not in pred and cls in true)
    return tp, fp, fn, len(pairs) - tp - fp - fn


def per_class(pairs, classes):
    out = {}
    for cls in classes:
        tp, fp, fn, _ = counts(pairs, cls)
        precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
        out[cls] = (precision, recall, divide(2 * precision * recall, precision + recall), tp + fn)
    return out


def scored(stats, dataset=None, level="window"):
    excluded = RECORDING_EXCLUDE.get(dataset, set()) if level == "recording" else set()
    return [cls for cls, value in stats.items() if value[3] >= MIN_SUPPORT and cls not in excluded]


def recording_classes(dataset, classes):
    # the classes the recording-level mean is taken over, support aside (the trainer's val metric uses this)
    return [c for c in classes if c not in RECORDING_EXCLUDE.get(dataset, set())]


def macro_f1(stats, classes):
    return divide(sum(stats[c][2] for c in classes), len(classes))


def balanced_accuracy(pairs, classes):
    # mean over classes of (recall + specificity) / 2; any constant answer scores exactly 0.5
    scores = []
    for cls in classes:
        tp, fp, fn, tn = counts(pairs, cls)
        scores.append((divide(tp, tp + fn) + divide(tn, tn + fp)) / 2)
    return divide(sum(scores), len(scores))


def to_recordings(windows, dataset):
    by_recording = defaultdict(lambda: ([], []))
    for recording, pred, true in windows:
        by_recording[recording][0].append(pred)
        by_recording[recording][1].append(true)
    pairs = []
    for preds, trues in by_recording.values():
        true = frozenset().union(*trues)
        if dataset in BINARY:
            votes = Counter(c for pred in preds for c in pred)  # majority vote across the recording's windows
            pred = frozenset([votes.most_common(1)[0][0]]) if votes else frozenset()
        else:
            pred = frozenset().union(*preds)  # detected anywhere in the recording
        pairs.append((pred, true))
    return pairs


def macro_f1_from_counts(totals):
    tp, fp, fn = totals[:, 0], totals[:, 1], totals[:, 2]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return float(f1.mean()) if len(f1) else 0.0


def bootstrap(groups, classes, iterations):
    # percentile CI resampling whole recordings: windows of one recording are correlated
    if not groups or not classes or iterations <= 0:
        return None
    table = np.zeros((len(groups), len(classes), 3))
    for i, pairs in enumerate(groups):
        for j, cls in enumerate(classes):
            tp, fp, fn, _ = counts(pairs, cls)
            table[i, j] = (tp, fp, fn)
    flat = table.reshape(len(groups), -1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(groups)
    samples = np.empty(iterations)
    for b in range(iterations):
        multiplicity = rng.multinomial(n, np.full(n, 1.0 / n))
        samples[b] = macro_f1_from_counts((multiplicity @ flat).reshape(len(classes), 3))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def baseline(pairs, classes, dataset):
    # best macro-F1 a constant, image-blind answer can reach; binary schemas can only emit one class
    if not pairs or not classes:
        return 0.0
    candidates = [frozenset([c]) for c in classes]
    if dataset not in BINARY:
        candidates += [frozenset(classes), frozenset()]
    return max(macro_f1(per_class([(guess, true) for _, true in pairs], classes), classes) for guess in candidates)


def positives(row, classes):
    return frozenset(c for c in classes if row[c].strip().lower() == "true")


def evaluate(model, dataset, rows, truth, scope, iterations):
    classes = labels(dataset)
    windows, seen = [], set()
    for row in rows:
        if row["path"] in seen or row["path"] not in scope:
            continue  # duplicates from overlapping runs, or rows from an older, larger sample
        seen.add(row["path"])
        windows.append((parse_name(row["path"])[1], positives(row, classes), positives(truth[row["path"]], classes)))
    if not windows:
        return None
    window_pairs = [(pred, true) for _, pred, true in windows]
    recording_pairs = to_recordings(windows, dataset)
    by_recording = defaultdict(list)
    for recording, pred, true in windows:
        by_recording[recording].append((pred, true))
    window_stats, recording_stats = per_class(window_pairs, classes), per_class(recording_pairs, classes)
    window_classes, kept = scored(window_stats), scored(recording_stats, dataset, "recording")
    ci = bootstrap([[pair] for pair in recording_pairs], kept, iterations)
    floor = baseline(recording_pairs, kept, dataset)
    answers = Counter(pred for pred, _ in window_pairs)
    top, hits = answers.most_common(1)[0]
    patients = {parse_name(p)[0] for p in seen}
    return {
        "model": model, "dataset": dataset, "windows": len(windows),
        "coverage": round(len(windows) / len(scope), 4), "recordings": len(recording_pairs), "patients": len(patients),
        "recording_balanced_accuracy": balanced_accuracy(recording_pairs, kept),
        "recording_macro_f1": macro_f1(recording_stats, kept),
        "recording_ci_low": ci[0] if ci else "", "recording_ci_high": ci[1] if ci else "",
        "baseline_macro_f1": floor, "clears_baseline": ci[0] > floor if ci else "",
        "window_balanced_accuracy": balanced_accuracy(window_pairs, window_classes),
        "window_macro_f1": macro_f1(window_stats, window_classes),
        "distinct_answers": len(answers), "top_answer": ", ".join(sorted(top)) or "(nothing)",
        "top_answer_fraction": hits / len(window_pairs), "degenerate": hits / len(window_pairs) >= DEGENERATE_FRACTION,
        "scored_classes": len(kept), "low_support_classes": len(classes) - len(kept),
        "few_patients": len(patients) < MIN_PATIENTS,
        "_classes": [(level, cls, *stats[cls][:3], stats[cls][3], cls in kept)
                     for level, stats, kept in (("window", window_stats, window_classes),
                                                ("recording", recording_stats, kept))
                     for cls in classes],
    }


def fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def report(dataset, summaries):
    print(f"\n{dataset}: {summaries[0]['recordings']} recordings scored at full coverage, "
          f"floor macro-F1 {summaries[0]['baseline_macro_f1']:.3f}, "
          f"{summaries[0]['scored_classes']} classes with support >= {MIN_SUPPORT}"
          + (" (few patients: descriptive only)" if summaries[0]["few_patients"] else ""))
    print(f"  {'model':34} {'cov':>5} {'rec-BA':>7} {'rec-F1':>7} {'95% CI':>15} {'clears':>6} {'degen':>5}  top answer")
    for s in summaries:
        ci = f"[{s['recording_ci_low']:.3f}, {s['recording_ci_high']:.3f}]" if s["recording_ci_low"] != "" else ""
        print(f"  {s['model']:34} {s['coverage']:5.2f} {s['recording_balanced_accuracy']:7.3f} "
              f"{s['recording_macro_f1']:7.3f} {ci:>15} {str(s['clears_baseline']):>6} "
              f"{'yes' if s['degenerate'] else '':>5}  {s['top_answer']} ({s['top_answer_fraction']:.0%})")


def write(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="recording-level scores per model per dataset from eval-baseline.csv")
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    parser.add_argument("--models", nargs="*", help="restrict to these model names")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP, help="resamples for the CI (0 = skip)")
    args = parser.parse_args()
    conn = db()
    for dataset in args.datasets:
        folder = ROOT / "datasets" / dataset
        scope = {p.split("/", 1)[1] for (p,) in conn.execute(
            "SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'full'", (dataset,))}
        truth = {r["path"]: r for r in read_csv(folder / "labels.csv")}
        by_model = defaultdict(list)
        for row in read_csv(folder / "eval-baseline.csv"):
            by_model[row["model"]].append(row)
        summaries = [s for model, rows in sorted(by_model.items())
                     if (not args.models or model in args.models)
                     and (s := evaluate(model, dataset, rows, truth, scope, args.bootstrap))]
        if not summaries:
            print(f"\n{dataset}: no scored rows")
            continue
        summaries.sort(key=lambda s: (-s["coverage"], -s["recording_balanced_accuracy"]))
        report(dataset, summaries)
        write(folder / "summary.csv", FIELDS, [{k: s[k] for k in FIELDS} for s in summaries])
        write(folder / "summary-classes.csv", CLASS_FIELDS,
              [dict(zip(CLASS_FIELDS, (s["model"], dataset, *row))) for s in summaries for row in s["_classes"]])


if __name__ == "__main__":
    main()
