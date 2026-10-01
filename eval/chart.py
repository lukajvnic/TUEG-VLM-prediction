"""Per-dataset bar chart: every base as a pair of bars, zero-shot (Ollama row) next to its fine-tune on that dataset.

  python eval/chart.py                      # every dataset with a summary.csv, recording balanced accuracy
  python eval/chart.py TUAB --metric recording_macro_f1

Reads datasets/<DS>/summary.csv (eval/score.py) and writes datasets/<DS>/chart-<metric>.png; score.py calls
write_dataset_chart() itself after writing the summary. The renderer is the one from the pre-redesign
summarize.py (rounded data-ends, recessive grid, a reference line for chance, one direct label per series),
extended to two series per model (2026-09-29). Bases come from config.yml `bases:`; a base with only one of the
two rows still gets its bar, the other slot stays empty.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as DrawPath  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import DATASETS, ROOT, config, run_name  # noqa: E402

DPI = 150
SLOT_PX = 64            # horizontal budget per model (two bars, air included)
BAR_MAX_PX = 20
PAIR_GAP_PX = 2         # surface gap between the two bars of a pair
SLOT_GAP_PX = 18        # air between neighbouring pairs
CORNER_PX = 4
PLOT_HEIGHT_PX = 380
PLOT_MIN_WIDTH_PX = 640
MARGIN_PX = {"left": 90, "right": 60, "top": 92, "bottom": 170}

SERIES = {"zero-shot": "#2a78d6", "fine-tuned": "#eb6834"}  # categorical slots 1 and 2, validated 2026-09-29
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRIDLINE, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TICKS = (0, 0.25, 0.5, 0.75, 1.0)
HEADROOM = 1.08

FULL_COVERAGE = 0.99    # below this the row is a partial predict run: drawn hatched and lighter, not comparable yet


def number(row, field):
    # summary.csv cells are strings; an absent or blank one is None
    raw = row.get(field) if row else None
    return None if raw in (None, "") else float(raw)


def chance(rows):
    return 0.5


def constant_predictor(rows):
    floors = [number(r, "baseline_macro_f1") for r in rows]
    return max(f for f in floors if f is not None)


METRICS = {  # metric -> (axis label, reference-line label, reference value from the complete rows)
    "recording_balanced_accuracy": ("recording-level balanced accuracy", "chance", chance),
    "recording_macro_f1": ("recording-level macro-F1", "constant predictor", constant_predictor),
}


def make_figure(model_count):
    plot_width = max(PLOT_MIN_WIDTH_PX, model_count * SLOT_PX)
    width = MARGIN_PX["left"] + plot_width + MARGIN_PX["right"]
    height = MARGIN_PX["top"] + PLOT_HEIGHT_PX + MARGIN_PX["bottom"]
    figure = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor=SURFACE)
    axes = figure.add_axes((MARGIN_PX["left"] / width, MARGIN_PX["bottom"] / height,
                            plot_width / width, PLOT_HEIGHT_PX / height))
    axes.set_facecolor(SURFACE)
    return figure, axes


def data_per_pixel(axes):
    inverse = axes.transData.inverted()
    ox, oy = inverse.transform((0, 0))
    ux, uy = inverse.transform((1, 1))
    return ux - ox, uy - oy


def rounded_bar(x, height, width, corner_x, corner_y, color, partial=False):
    corner_x, corner_y = min(corner_x, width / 2), min(corner_y, height)
    left, right = x - width / 2, x + width / 2
    vertices = [(left, 0), (left, height - corner_y), (left, height), (left + corner_x, height),
                (right - corner_x, height), (right, height), (right, height - corner_y), (right, 0), (left, 0)]
    codes = [DrawPath.MOVETO, DrawPath.LINETO, DrawPath.CURVE3, DrawPath.CURVE3, DrawPath.LINETO,
             DrawPath.CURVE3, DrawPath.CURVE3, DrawPath.LINETO, DrawPath.CLOSEPOLY]
    if partial:  # texture + lighter fill: the value is over a subset of the windows
        return PathPatch(DrawPath(vertices, codes), facecolor=color, alpha=0.45, edgecolor=color, hatch="////",
                         linewidth=0, zorder=2)
    return PathPatch(DrawPath(vertices, codes), facecolor=color, edgecolor="none", zorder=2)


def draw_pairs(axes, series, partial):
    # series: {"zero-shot": [v or None], "fine-tuned": [...]}, partial: same shape of bools;
    # bar i of model x sits at x -/+ half a bar + gap
    ux, uy = data_per_pixel(axes)
    slot_px = axes.get_window_extent().width / max(len(series["zero-shot"]), 1)
    bar_px = min(BAR_MAX_PX, max(1.0, (slot_px - SLOT_GAP_PX - PAIR_GAP_PX) / 2))
    offset = (bar_px + PAIR_GAP_PX) / 2 * ux
    for name, sign in (("zero-shot", -1), ("fine-tuned", 1)):
        for x, value in enumerate(series[name]):
            if value is not None:
                axes.add_patch(rounded_bar(x + sign * offset, value, bar_px * ux, CORNER_PX * ux, CORNER_PX * uy,
                                           SERIES[name], partial[name][x]))
    return offset


def annotate(axes, text, xy, coords, offset, color, size, **align):
    axes.annotate(text, xy=xy, xycoords=coords, xytext=offset, textcoords="offset points",
                  color=color, fontsize=size, annotation_clip=False, **align)


def label_peaks(axes, series, offset, fmt):
    # one direct label per series, at its maximum; the axis carries the rest
    for name, sign in (("zero-shot", -1), ("fine-tuned", 1)):
        values = series[name]
        present = [i for i, v in enumerate(values) if v is not None]
        if not present:
            continue
        peak = max(present, key=lambda i: values[i])
        # text runs away from the partner bar, which may be the taller one
        annotate(axes, fmt.format(values[peak]), (peak + sign * offset, values[peak]), ("data", "data"), (2 * sign, 6),
                 INK, 8, ha="right" if sign < 0 else "left", va="bottom")


def style_axes(axes, labels, y_label):
    axes.set_xlim(-0.5, len(labels) - 0.5)
    axes.set_ylim(0, max(TICKS) * HEADROOM)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.set_yticks(TICKS)
    axes.set_yticklabels(f"{t:.2f}" for t in TICKS)
    axes.set_ylabel(y_label, color=INK_SECONDARY, fontsize=10, labelpad=10)
    axes.tick_params(axis="both", length=0, colors=INK_MUTED, labelsize=9)
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    axes.xaxis.grid(False)
    for side, spine in axes.spines.items():
        spine.set_visible(side == "bottom")
    axes.spines["bottom"].set(color=BASELINE, linewidth=1)


def write_pair_chart(path, labels, series, partial, title, y_label, subtitle, reference):
    figure, axes = make_figure(len(labels))
    style_axes(axes, labels, y_label)
    axes.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=44)
    annotate(axes, subtitle, (0, 1), ("axes fraction", "axes fraction"), (0, 26), INK_SECONDARY, 9, ha="left", va="bottom")
    offset = draw_pairs(axes, series, partial)
    label_peaks(axes, series, offset, "{:.3f}")
    value, label = reference
    axes.axhline(value, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=3)
    annotate(axes, label, (1, value), ("axes fraction", "data"), (8, 0), INK_MUTED, 8, ha="left", va="center")
    handles = [Patch(facecolor=c, label=n) for n, c in SERIES.items()]
    if any(any(v) for v in partial.values()):
        handles.append(Patch(facecolor=INK_MUTED, alpha=0.45, edgecolor=INK_MUTED, hatch="////", linewidth=0,
                             label=f"partial (< {FULL_COVERAGE:.0%} of windows scored)"))
    axes.legend(handles=handles, loc="lower left",
                bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
                handlelength=1.2, handleheight=0.9, borderaxespad=0)
    figure.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)


def pair_rows(rows, cfg, dataset):
    # one entry per `bases:` key that has a zero-shot row (its Ollama name) or a fine-tuned row of the current
    # experiment (<key>-sft-<DS>-<train.experiment>: the fine-tune on this dataset alone)
    by_model = {r["model"]: r for r in rows}
    tuned_name = run_name(dataset, cfg["train"]["experiment"])
    pairs = []
    for key in cfg["bases"]:
        zero, tuned = by_model.get(key), by_model.get(f"{key}-sft-{tuned_name}")
        if zero or tuned:
            pairs.append((key, zero, tuned))
    return pairs


def is_partial(row):
    coverage = number(row, "coverage")
    return row is not None and coverage is not None and coverage < FULL_COVERAGE


def is_degenerate(row):
    return row is not None and str(row.get("degenerate", "")).lower() == "true"


def order(pairs, metric):
    # complete fine-tunes first, then partial ones, then zero-shot-only bases; by score within each group
    def key(pair):
        _, zero, tuned = pair
        group = 2 if tuned is None else 1 if is_partial(tuned) else 0
        return group, -(number(tuned, metric) or 0), -(number(zero, metric) or 0)
    return sorted(pairs, key=key)


def write_dataset_chart(dataset, rows, folder, metric="recording_balanced_accuracy", cfg=None):
    pairs = order(pair_rows(rows, cfg or config(), dataset), metric)
    series = {"zero-shot": [number(z, metric) for _, z, _ in pairs],
              "fine-tuned": [number(t, metric) for _, _, t in pairs]}
    if not any(v is not None for values in series.values() for v in values):
        return None
    partial = {"zero-shot": [is_partial(z) for _, z, _ in pairs], "fine-tuned": [is_partial(t) for _, _, t in pairs]}
    labels = [key + (" \u2020" if is_degenerate(zero) or is_degenerate(tuned) else "") for key, zero, tuned in pairs]
    present = [r for _, z, t in pairs for r in (z, t) if r]
    complete = [r for r in present if not is_partial(r)] or present
    n_tuned, n_partial = sum(1 for _, _, t in pairs if t), sum(partial["fine-tuned"]) + sum(partial["zero-shot"])
    subtitle = (f"{complete[0].get('recordings', '?')} test recordings, {complete[0].get('patients', '?')} patients; "
                f"{n_tuned} of {len(pairs)} bases fine-tuned (LoRA on the {dataset} train split)"
                + (f", {n_partial} bars partial" if n_partial else "") + "; \u2020 = one answer for \u2265 95% of windows")
    y_label, reference_label, reference = METRICS[metric]
    path = folder / f"chart-{metric}.png"
    write_pair_chart(path, labels, series, partial, f"{dataset}: zero-shot vs fine-tuned", y_label, subtitle,
                     (reference(complete), reference_label))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    parser.add_argument("--metric", choices=list(METRICS), default="recording_balanced_accuracy")
    args = parser.parse_args()
    cfg = config()
    for dataset in args.datasets:
        folder = ROOT / "datasets" / dataset
        summary = folder / "summary.csv"
        if not summary.exists():
            print(f"{dataset}: no summary.csv (run eval/score.py {dataset} first)")
            continue
        with summary.open() as f:
            rows = list(csv.DictReader(f))
        path = write_dataset_chart(dataset, rows, folder, args.metric, cfg)
        print(f"{dataset}: {path if path else 'no base has a row'}")


if __name__ == "__main__":
    main()
