"""Score an eval run and write the summary artifacts.

Two things the raw per-window `correct` flag gets wrong:

  * Accuracy on a background-dominated, window-tiled test set flatters a model
    that just says "background". So we report per-class precision/recall/F1
    (recall on the rare event classes is the number that matters).

  * TUAB/TUEP are labelled at the recording level (abnormal / epilepsy), so a
    single normal-looking window of an abnormal recording is not a fair unit to
    grade. We aggregate a recording's windows into one prediction and score at
    the recording level — the meaningful clinical unit, and it answers "did the
    model catch the finding somewhere in the recording".

Window-level accuracy and balanced accuracy are still written to `summary.csv`
for continuity, but the ranking and the chart are driven by recording-level
macro-F1, and every row carries the caveats that decide whether its score means
anything: the constant-predictor baseline it has to clear, the bootstrap CI,
whether the output was degenerate, and how many independent patients it saw.

Usage:  python eval/summarize.py <run-name> [--bootstrap N]
"""

import argparse
import ast
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as DrawPath

# ------------------------------------------------------------------- chart
#
# Metric-agnostic on purpose: `write_bar_chart` takes labels and values on a
# 0-based scale, so accuracy, balanced accuracy or recall all render the same
# way. Callers own the wording; this section owns the drawing.

DPI = 150
SLOT_PX = 40           # horizontal budget per bar, air included
BAR_MAX_PX = 24        # bars never fill their slot
BAR_GAP_PX = 16        # air between neighbouring bars
CORNER_PX = 4          # rounded data-end, square at the baseline
PLOT_HEIGHT_PX = 380
PLOT_MIN_WIDTH_PX = 640
MARGIN_PX = {"left": 90, "right": 48, "top": 76, "bottom": 150}

SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

DEFAULT_TICKS = (0, 0.25, 0.5, 0.75, 1.0)
HEADROOM = 1.06        # space above the tallest bar for its direct label


def make_figure(bar_count):
    plot_width = max(PLOT_MIN_WIDTH_PX, bar_count * SLOT_PX)
    width = MARGIN_PX["left"] + plot_width + MARGIN_PX["right"]
    height = MARGIN_PX["top"] + PLOT_HEIGHT_PX + MARGIN_PX["bottom"]
    figure = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor=SURFACE)
    axes = figure.add_axes((
        MARGIN_PX["left"] / width,
        MARGIN_PX["bottom"] / height,
        plot_width / width,
        PLOT_HEIGHT_PX / height,
    ))
    axes.set_facecolor(SURFACE)
    return figure, axes


def data_per_pixel(axes):
    inverse = axes.transData.inverted()
    origin_x, origin_y = inverse.transform((0, 0))
    unit_x, unit_y = inverse.transform((1, 1))
    return unit_x - origin_x, unit_y - origin_y


def rounded_bar(x, height, width, corner_x, corner_y):
    """A bar with rounded data-end and square baseline, as a closed path."""
    corner_x, corner_y = min(corner_x, width / 2), min(corner_y, height)
    left, right = x - width / 2, x + width / 2
    vertices = [
        (left, 0),
        (left, height - corner_y),
        (left, height), (left + corner_x, height),
        (right - corner_x, height),
        (right, height), (right, height - corner_y),
        (right, 0),
        (left, 0),
    ]
    codes = [
        DrawPath.MOVETO,
        DrawPath.LINETO,
        DrawPath.CURVE3, DrawPath.CURVE3,
        DrawPath.LINETO,
        DrawPath.CURVE3, DrawPath.CURVE3,
        DrawPath.LINETO,
        DrawPath.CLOSEPOLY,
    ]
    return PathPatch(DrawPath(vertices, codes), facecolor=SERIES, edgecolor="none", zorder=2)


def draw_bars(axes, values):
    """One hue for every bar — length already carries the magnitude."""
    unit_x, unit_y = data_per_pixel(axes)
    slot_px = axes.get_window_extent().width / max(len(values), 1)
    width_px = min(BAR_MAX_PX, max(1.0, slot_px - BAR_GAP_PX))
    for x, value in enumerate(values):
        axes.add_patch(rounded_bar(x, value, width_px * unit_x, CORNER_PX * unit_x, CORNER_PX * unit_y))


def style_axes(axes, labels, y_label, y_ticks, y_limits):
    axes.set_xlim(-0.5, len(labels) - 0.5)
    axes.set_ylim(*y_limits)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(f"{tick:.2f}" for tick in y_ticks)
    axes.set_ylabel(y_label, color=INK_SECONDARY, fontsize=10, labelpad=10)
    axes.tick_params(axis="both", length=0, colors=INK_MUTED, labelsize=9)
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=GRIDLINE, linewidth=1, linestyle="-")
    axes.xaxis.grid(False)
    for side, spine in axes.spines.items():
        spine.set_visible(side == "bottom")
    axes.spines["bottom"].set(color=BASELINE, linewidth=1)


def add_titles(axes, title, subtitle=None):
    axes.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=30)
    if subtitle:
        annotate(axes, subtitle, (0, 1), ("axes fraction", "axes fraction"), (0, 12), INK_SECONDARY, 9,
                 ha="left", va="bottom")


def add_reference(axes, value, label):
    """The threshold rides in the margin — over the bars it would collide."""
    axes.axhline(value, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=3)
    annotate(axes, label, (1, value), ("axes fraction", "data"), (8, 0), INK_MUTED, 8,
             ha="left", va="center")


def label_peak(axes, values, fmt):
    """Direct-label the extreme only; the axis carries the rest."""
    peak = max(range(len(values)), key=lambda index: values[index])
    annotate(axes, fmt.format(values[peak]), (peak, values[peak]), ("data", "data"), (0, 7), INK, 9,
             ha="center", va="bottom")


def zoomed_range(values, pad=0.05, floor=0.0, ceiling=1.0):
    """[min-pad, max+pad], clamped to [floor, ceiling]. Widens gaps between
    near-identical bars at the cost of exaggerating noise — caller must own
    that tradeoff in the caption."""
    lo = max(floor, min(values) - pad)
    hi = min(ceiling, max(values) + pad)
    return lo, hi if hi > lo else lo + pad


def nice_ticks(lo, hi, count=5):
    span = hi - lo
    step = span / (count - 1) if count > 1 else span
    return [lo + step * index for index in range(count)]


def annotate(axes, text, xy, coords, offset, color, size, **align):
    axes.annotate(
        text, xy=xy, xycoords=coords, xytext=offset, textcoords="offset points",
        color=color, fontsize=size, annotation_clip=False, **align,
    )


def write_bar_chart(path, labels, values, title, y_label, subtitle=None, reference=None,
                    y_ticks=DEFAULT_TICKS, value_format="{:.3f}", y_range=None):
    if not values:
        return
    if y_range is not None:
        lo, hi = y_range
        y_ticks = nice_ticks(lo, hi)
        y_limits = (lo, hi + (hi - lo) * (HEADROOM - 1))
    else:
        y_limits = (0, max(y_ticks) * HEADROOM)
    figure, axes = make_figure(len(values))
    style_axes(axes, labels, y_label, y_ticks, y_limits)
    add_titles(axes, title, subtitle)
    draw_bars(axes, values)
    label_peak(axes, values, value_format)
    if reference:
        add_reference(axes, *reference)
    figure.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)


# ------------------------------------------------------------------ scoring

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

SUMMARY_FIELDS = (
    "model", "dataset", "status", "windows", "recordings", "patients",
    "accuracy", "balanced_accuracy",
    "window_macro_f1", "window_ci_low", "window_ci_high",
    "recording_macro_f1", "recording_ci_low", "recording_ci_high",
    "baseline_macro_f1", "clears_baseline",
    "distinct_answers", "top_answer", "top_answer_fraction", "degenerate",
    "scored_classes", "low_support_classes", "few_patients",
    "tp", "fp", "tn", "fn",
)
CLASS_FIELDS = ("model", "dataset", "level", "class",
                "precision", "recall", "f1", "support", "in_macro")
RANK_FIELDS = ("rank", "model", "recording_macro_f1", "baseline_macro_f1",
               "balanced_accuracy", "datasets", "clears_baseline", "degenerate")


def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_labels(value):
    try:
        labels = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return frozenset()
    return frozenset(labels) if isinstance(labels, (set, frozenset, list, tuple)) else frozenset()


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def recording_of(image_path):
    # image name is <patient>_<scan>_<window>; the recording is <patient>_<scan>.
    return Path(image_path).stem.rsplit("_", 1)[0]


def patient_of(image_path):
    return Path(image_path).stem.split("_")[0]


def load_statuses(run_dir):
    """{sanitised-stem: status row} — the only place a model's real name survives.

    Results files are named with `safe_filename(model)`, which flattens the `:`
    in an Ollama tag, so the stem alone cannot be turned back into a model name.
    """
    path = run_dir / "status.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {f"{safe_filename(row['model'])}-{row['dataset']}": row
                for row in csv.DictReader(file)}


def load_tasks(run_dir):
    """Every results CSV in the run, named and status-tagged where possible.

    Globs `results/` rather than walking `status.csv` so a task that produced
    rows but never reported success is scored (and labelled) instead of silently
    dropped — the failure mode that makes a partial run look complete.
    """
    statuses = load_statuses(run_dir)
    tasks = []
    for results_csv in sorted((run_dir / "results").glob("*.csv")):
        row = statuses.get(results_csv.stem)
        if row:
            model, dataset, status = row["model"], row["dataset"], row["status"]
        else:
            model, dataset, status = (*results_csv.stem.rsplit("-", 1), "unknown")
        with results_csv.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows:
            tasks.append((model, dataset, status, rows))
    return tasks


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


def scored_classes(stats):
    return sorted(cls for cls, value in stats.items() if value[3] >= MIN_SUPPORT)


def macro_f1(stats):
    scored = [value[2] for value in stats.values() if value[3] >= MIN_SUPPORT]
    return sum(scored) / len(scored) if scored else 0.0


def pooled_confusion(pairs):
    """Label-summed tp/fp/tn/fn plus balanced accuracy averaged over labels."""
    classes = set()
    for pred, true in pairs:
        classes |= pred | true
    totals = [0, 0, 0, 0]
    balanced = []
    for cls in sorted(classes):
        tp = sum(1 for pred, true in pairs if cls in pred and cls in true)
        fp = sum(1 for pred, true in pairs if cls in pred and cls not in true)
        fn = sum(1 for pred, true in pairs if cls not in pred and cls in true)
        tn = len(pairs) - tp - fp - fn
        totals = [totals[0] + tp, totals[1] + fp, totals[2] + tn, totals[3] + fn]
        balanced.append((safe_divide(tp, tp + fn) + safe_divide(tn, tn + fp)) / 2)
    return (*totals, safe_divide(sum(balanced), len(balanced)))


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
    counts = Counter(pred for pred, _ in pairs)
    top, hits = counts.most_common(1)[0]
    return {
        "distinct": len(counts),
        "fraction": hits / len(pairs),
        "answer": ", ".join(sorted(top)) if top else "(nothing)",
    }


def evaluate(model, dataset, status, rows, iterations):
    windows = [(row["path"], recording_of(row["path"]),
                parse_labels(row["prediction"]), parse_labels(row["actual"]))
               for row in rows]
    window_pairs = [(pred, true) for _, _, pred, true in windows]
    recording_pairs = aggregate_to_recordings(windows, dataset)
    patients = {patient_of(path) for path, _, _, _ in windows}

    by_recording = defaultdict(list)
    for _, recording, pred, true in windows:
        by_recording[recording].append((pred, true))

    window_stats = per_class(window_pairs)
    recording_stats = per_class(recording_pairs)
    recording_classes = scored_classes(recording_stats)

    # Both bootstraps resample the same clusters: whole recordings.
    window_ci = bootstrap_macro_f1([by_recording[r] for r in sorted(by_recording)],
                                   scored_classes(window_stats), iterations)
    recording_ci = bootstrap_macro_f1([[pair] for pair in recording_pairs],
                                      recording_classes, iterations)

    baseline = baseline_macro_f1(recording_pairs, recording_classes, dataset)
    shape = degeneracy(window_pairs)
    tp, fp, tn, fn, balanced = pooled_confusion(window_pairs)

    return {
        "model": model,
        "dataset": dataset,
        "status": status,
        "windows": len(windows),
        "recordings": len(recording_pairs),
        "patients": len(patients),
        "accuracy": safe_divide(sum(row["correct"].lower() == "true" for row in rows), len(rows)),
        "balanced_accuracy": balanced,
        "window_macro_f1": macro_f1(window_stats),
        "window_ci_low": window_ci[0] if window_ci else "",
        "window_ci_high": window_ci[1] if window_ci else "",
        "recording_macro_f1": macro_f1(recording_stats),
        "recording_ci_low": recording_ci[0] if recording_ci else "",
        "recording_ci_high": recording_ci[1] if recording_ci else "",
        "baseline_macro_f1": baseline,
        # The stage-1 promotion rule: the CI lower bound must clear the floor,
        # not the point estimate. Blank rather than False when there is no CI.
        "clears_baseline": recording_ci[0] > baseline if recording_ci else "",
        "distinct_answers": shape["distinct"],
        "top_answer": shape["answer"],
        "top_answer_fraction": shape["fraction"],
        "degenerate": shape["fraction"] >= DEGENERATE_FRACTION,
        "scored_classes": len(recording_classes),
        "low_support_classes": len(recording_stats) - len(recording_classes),
        "few_patients": len(patients) < MIN_PATIENTS,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "_window_stats": window_stats,
        "_recording_stats": recording_stats,
        "_window_ci": window_ci,
        "_recording_ci": recording_ci,
    }


def class_rows(summary):
    for level, stats in (("window", summary["_window_stats"]),
                         ("recording", summary["_recording_stats"])):
        for cls, (precision, recall, f1, support) in sorted(stats.items()):
            yield {
                "model": summary["model"], "dataset": summary["dataset"],
                "level": level, "class": cls,
                "precision": precision, "recall": recall, "f1": f1,
                "support": support, "in_macro": support >= MIN_SUPPORT,
            }


def report_level(title, stats, interval):
    scored = {c: v for c, v in stats.items() if v[3] >= MIN_SUPPORT}
    under = {c: v for c, v in stats.items() if v[3] < MIN_SUPPORT}
    span = f"  95% CI [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else ""
    print(f"  {title}: macro-F1={macro_f1(stats):.3f}{span} "
          f"over {len(scored)} classes with support>={MIN_SUPPORT}")
    for cls, (precision, recall, f1, support) in scored.items():
        print(f"    {cls:12} P={precision:.3f} R={recall:.3f} F1={f1:.3f}  (support={support})")
    if under:
        print("    under-supported (excluded from macro-F1): "
              + ", ".join(f"{c}(n={v[3]})" for c, v in under.items()))


def report(summary):
    print(f"\n=== {summary['model']}  |  {summary['dataset']}  "
          f"({summary['windows']} windows, {summary['recordings']} recordings, "
          f"{summary['patients']} patients, status={summary['status']}) ===")
    report_level("window-level", summary["_window_stats"], summary["_window_ci"])
    report_level("recording-level", summary["_recording_stats"], summary["_recording_ci"])

    if summary["_recording_ci"]:
        verdict = "ABOVE" if summary["clears_baseline"] else "NOT above"
        print(f"  constant-predictor baseline: macro-F1={summary['baseline_macro_f1']:.3f} -- "
              f"this model is {verdict} it (CI lower bound {summary['_recording_ci'][0]:.3f})")

    if summary["degenerate"]:
        print(f"  !! degenerate output: {summary['top_answer_fraction']:.1%} of windows answered "
              f"\"{summary['top_answer']}\" ({summary['distinct_answers']} distinct answers) -- "
              f"this score does not reflect reading the image")
    else:
        print(f"  output diversity: {summary['distinct_answers']} distinct answers, "
              f"most common \"{summary['top_answer']}\" on "
              f"{summary['top_answer_fraction']:.1%} of windows")

    if summary["few_patients"]:
        print(f"  ~~ only {summary['patients']} independent patients: report descriptively, "
              f"not as an inferential result")


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rank_models(summaries):
    """Ranked on mean recording-level macro-F1 — the reportable metric.

    Only tasks that ran to completion are ranked; a task that died halfway would
    otherwise be compared against full ones on a fraction of the test set.
    """
    complete = [row for row in summaries if row["status"] in ("success", "unknown")]
    by_model = defaultdict(list)
    for row in complete:
        by_model[row["model"]].append(row)

    ranking = []
    for model, rows in by_model.items():
        ranking.append({
            "model": model,
            "recording_macro_f1": sum(r["recording_macro_f1"] for r in rows) / len(rows),
            "baseline_macro_f1": sum(r["baseline_macro_f1"] for r in rows) / len(rows),
            "balanced_accuracy": sum(r["balanced_accuracy"] for r in rows) / len(rows),
            "datasets": len(rows),
            "clears_baseline": sum(1 for r in rows if r["clears_baseline"] is True),
            "degenerate": sum(1 for r in rows if r["degenerate"]),
        })
    ranking.sort(key=lambda row: (-row["recording_macro_f1"], row["model"]))
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking


def write_rank_chart(path, ranking, run):
    """Reference line is the constant-predictor floor, not 0.5.

    Chance is not a meaningful landmark for macro-F1 on a class-imbalanced
    multi-label set; the floor an image-blind model reaches is.
    """
    values = [row["recording_macro_f1"] for row in ranking]
    baseline = sum(row["baseline_macro_f1"] for row in ranking) / len(ranking)
    write_bar_chart(
        path,
        labels=[row["model"] for row in ranking],
        values=values,
        title=f"Model ranking — {run}",
        y_label="Recording-level macro-F1",
        reference=(baseline, f"constant predictor {baseline:.2f}"),
        y_range=zoomed_range(values + [baseline]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP,
                        help="resamples for the macro-F1 CI (0 disables)")
    args = parser.parse_args()

    set_csv_field_limit()
    run_dir = Path(__file__).parent / "runs" / args.run
    tasks = load_tasks(run_dir)
    if not tasks:
        sys.exit(f"No results CSVs in {run_dir / 'results'}")

    summaries = []
    for model, dataset, status, rows in tasks:
        summary = evaluate(model, dataset, status, rows, args.bootstrap)
        report(summary)
        summaries.append(summary)

    ranking = rank_models(summaries)
    write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(run_dir / "classes.csv", CLASS_FIELDS,
              [row for summary in summaries for row in class_rows(summary)])
    write_csv(run_dir / "rank.csv", RANK_FIELDS, ranking)
    if ranking:
        write_rank_chart(run_dir / "rank.png", ranking, args.run)

    print(f"\nwrote summary.csv, classes.csv, rank.csv, rank.png to {run_dir}")


if __name__ == "__main__":
    main()
