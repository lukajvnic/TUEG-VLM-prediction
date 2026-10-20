# one grouped bar chart per corpus from the results/summary-*.csv files: test AUROC per model, zero-shot (the base,
# untrained, same images and scorer) beside fine-tuned (main seed), each with its 95% patient-bootstrap CI, and the
# 0.5 chance line. Models keep one order in every chart (mean fine-tuned AUROC over the 4 corpora). PNG and PDF per
# corpus plus a 2x2 overview, in results/figures/
#   python side/vision-freeze/figures.py [RESULTS_DIR]
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tables

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
QUESTIONS = {"TUSZ": "seizure (any type)", "TUAB": "abnormal EEG", "TUAR": "eye movement artifact",
             "TUEV": "epileptiform discharge"}
# the dataviz skill's validated reference palette, light mode: categorical slots 1-2 and the text/grid tokens
ZERO, TUNED = "#2a78d6", "#eb6834"
SURFACE, TEXT, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
WIDTH = 0.3  # of a model's slot: two bars and air, ~22 px each at 100 dpi


def values(ds):
    out = {}
    for short in tables.MODELS:
        for kind, arm in (("zero", f"z-{short}-{ds.lower()}"), ("tuned", tables.trained(short, ds))):
            s = tables.summary(arm)
            if s:
                out[short, kind] = tuple(float(s[k]) for k in ("auroc", "auroc_ci_low", "auroc_ci_high"))
    return out


def order(data):
    means = {}
    for short in tables.MODELS:
        tuned = [data[ds][short, "tuned"][0] for ds in data if (short, "tuned") in data[ds]]
        if tuned:
            means[short] = np.mean(tuned)
    return sorted(means, key=lambda s: -means[s])


def draw(ax, ds, data, models, legend=True):
    ax.set_facecolor(SURFACE)
    x = np.arange(len(models))
    for offset, kind, color, label in ((-WIDTH / 2 - 0.01, "zero", ZERO, "Zero-shot (untrained)"),
                                       (WIDTH / 2 + 0.01, "tuned", TUNED, "Fine-tuned")):
        present = [i for i, m in enumerate(models) if (m, kind) in data]
        auroc = np.array([data[models[i], kind][0] for i in present])
        low = np.array([data[models[i], kind][1] for i in present])
        high = np.array([data[models[i], kind][2] for i in present])
        ax.bar(x[present] + offset, auroc, WIDTH, color=color, edgecolor=SURFACE, linewidth=1, label=label, zorder=2)
        ax.errorbar(x[present] + offset, auroc, yerr=[auroc - low, high - auroc], fmt="none", ecolor=SECONDARY,
                    elinewidth=0.9, capsize=0, zorder=3)
    best = max((m for m in models if (m, "tuned") in data), key=lambda m: data[m, "tuned"][0])
    i = models.index(best)
    ax.annotate(f"{data[best, 'tuned'][0]:.3f}", (i + WIDTH / 2 + 0.01, data[best, "tuned"][2] + 0.015),
                ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(1.005, 0.5, "chance", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, color=MUTED,
            clip_on=False)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([tables.MODELS[m] for m in models], rotation=40, ha="right", fontsize=8, color=TEXT)
    ax.tick_params(axis="y", labelsize=8, colors=SECONDARY)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Test AUROC", fontsize=9, color=SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.set_title(f"{ds}: {QUESTIONS[ds]}", loc="left", fontsize=11, color=TEXT, fontweight="bold")
    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=2, frameon=False, fontsize=8, labelcolor=TEXT)


def main():
    tables.RESULTS = RESULTS
    out = RESULTS / "figures"
    out.mkdir(exist_ok=True)
    data = {ds: values(ds) for ds in QUESTIONS}
    models = order(data)
    note = "Bars: test AUROC; whiskers: 95% patient-bootstrap CI; fine-tuned = main seed."
    for ds in QUESTIONS:
        fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=SURFACE)
        draw(ax, ds, data[ds], models)
        fig.text(0.01, 0.01, note, fontsize=7, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 0.97, 1))
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"auroc-{ds.lower()}.{suffix}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=SURFACE)
    for k, (ax, ds) in enumerate(zip(axes.flat, QUESTIONS)):
        draw(ax, ds, data[ds], models, legend=k == 0)
    fig.text(0.01, 0.005, note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 0.98, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"auroc-all.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"{len(models)} models -> {out}")


if __name__ == "__main__":
    main()
