# one grouped bar chart per corpus from the results/summary-*.csv files: test AUROC per model, zero-shot (the base,
# untrained, same images and scorer) beside fine-tuned (main seed), each with its 95% patient-bootstrap CI, and the
# 0.5 chance line; and the same charts for balanced accuracy at a 0.5 threshold (ba-*), its CI bootstrapped here from
# the per-window scores (1,000 patient resamples, seed 0). Models keep one order in every chart (mean fine-tuned AUROC over the 4 corpora). PNG and PDF per
# corpus plus a 2x2 overview, in results/figures/. Also the loss curves of every main-seed run, from the last
# checkpoint's trainer_state.json (results/trainer-states/<arm>/checkpoint-N/, an rsync of the cluster's checkpoints/
# keeping only those files): one chart per model and corpus in results/figures/loss/<ds>/, and a 4x4 grid per corpus
#   python side/vision-freeze/figures.py [RESULTS_DIR]
import csv
import json
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
METRICS = {"auroc": ("Test AUROC", "Bars: test AUROC"),
           "ba": ("Balanced accuracy (threshold 0.5)", "Bars: test balanced accuracy at a 0.5 threshold")}


def balanced_accuracy(arm):
    # (BA, CI low, CI high): BA at 0.5 on the arm's test scores, CI from 1,000 patient resamples
    path = RESULTS / f"scores-{arm}.csv"
    if not path.exists():
        return None
    with path.open() as f:
        rows = [r for r in {r["path"]: r for r in csv.DictReader(f)}.values() if r["p_true"] != ""]
    patients = sorted({Path(r["path"]).stem.split("_")[0] for r in rows})
    index = {p: i for i, p in enumerate(patients)}
    counts = np.zeros((len(patients), 4))  # true positives, false negatives, true negatives, false positives
    for r in rows:
        positive, said = r["seizure"] == "true", float(r["p_true"]) >= 0.5
        counts[index[Path(r["path"]).stem.split("_")[0]], [1, 0, 2, 3][2 * (not positive) + said]] += 1

    def ba(total):
        sensitivity = total[..., 0] / np.maximum(total[..., 0] + total[..., 1], 1)
        specificity = total[..., 2] / np.maximum(total[..., 2] + total[..., 3], 1)
        return (sensitivity + specificity) / 2

    draws = np.random.default_rng(0).multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=1000)
    low, high = np.percentile(ba(draws @ counts), [2.5, 97.5])
    return float(ba(counts.sum(0))), float(low), float(high)


def values(ds, metric):
    out = {}
    for short in tables.MODELS:
        for kind, arm in (("zero", f"z-{short}-{ds.lower()}"), ("tuned", tables.trained(short, ds))):
            if metric == "ba":
                found = balanced_accuracy(arm)
            else:
                s = tables.summary(arm)
                found = tuple(float(s[k]) for k in ("auroc", "auroc_ci_low", "auroc_ci_high")) if s else None
            if found:
                out[short, kind] = found
    return out


def order(data):
    means = {}
    for short in tables.MODELS:
        tuned = [data[ds][short, "tuned"][0] for ds in data if (short, "tuned") in data[ds]]
        if tuned:
            means[short] = np.mean(tuned)
    return sorted(means, key=lambda s: -means[s])


def draw(ax, ds, data, models, legend=True, ylabel="Test AUROC"):
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
    ax.set_ylabel(ylabel, fontsize=9, color=SECONDARY)
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
    models = order({ds: values(ds, "auroc") for ds in QUESTIONS})  # one model order for both metrics
    for metric, (ylabel, bars) in METRICS.items():
        data = {ds: values(ds, metric) for ds in QUESTIONS}
        note = f"{bars}; whiskers: 95% patient-bootstrap CI; fine-tuned = main seed."
        for ds in QUESTIONS:
            fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=SURFACE)
            draw(ax, ds, data[ds], models, ylabel=ylabel)
            fig.text(0.01, 0.01, note, fontsize=7, color=MUTED)
            fig.tight_layout(rect=(0, 0.03, 0.97, 1))
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"{metric}-{ds.lower()}.{suffix}", dpi=200, facecolor=SURFACE)
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=SURFACE)
        for k, (ax, ds) in enumerate(zip(axes.flat, QUESTIONS)):
            draw(ax, ds, data[ds], models, legend=k == 0, ylabel=ylabel)
        fig.text(0.01, 0.005, note, fontsize=8, color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 0.98, 1))
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{metric}-all.{suffix}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
    print(f"{len(models)} models -> {out}")
    losses(out, models)
    tuep_scan(out)


def tuep_scan(out):
    # TUEP patient-level AUROC of the spike scan, one bar per TUEV detector (main-seed gm- arms and the s- detectors),
    # 95% patient-bootstrap CI, the chance line; the direct window-level fine-tune goes in the caption
    rows = []
    for path in sorted(RESULTS.glob("summary-tuep-scan-*.csv")):
        with path.open() as f:
            s = next(csv.DictReader(f))
        arm = s["arm"]
        short = arm.split("-", 1)[1].rsplit("-", 1)[0]
        if arm.endswith("-s2") or short not in tables.MODELS:
            continue
        name = tables.MODELS[short] + (" (vision-only detector)" if arm.startswith("s-") else "")
        rows.append((name, float(s["auroc_patient"]), float(s["auroc_patient_ci_low"]), float(s["auroc_patient_ci_high"])))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(rows))
    auroc = np.array([r[1] for r in rows])
    ax.barh(y, auroc, 0.55, color=ZERO, edgecolor=SURFACE, linewidth=1, zorder=2)
    ax.errorbar(auroc, y, xerr=[auroc - [r[2] for r in rows], np.array([r[3] for r in rows]) - auroc], fmt="none",
                ecolor=SECONDARY, elinewidth=0.9, capsize=0, zorder=3)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.5, len(rows) - 0.35, "chance", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=TEXT)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8, colors=SECONDARY)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Patient-level AUROC (epilepsy vs not)", fontsize=9, color=SECONDARY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title("TUEP: epilepsy diagnosis from a spike scan", loc="left", fontsize=11, color=TEXT, fontweight="bold")
    direct = tables.summary("r6-tuep-epilepsy")
    note = ("Each TUEV-trained model scores 12 windows per TUEP test recording; recording = mean of its top 5, patient = "
            "mean of their recordings; 107 patients (53 epilepsy). Whiskers: 95% patient-bootstrap CI.")
    if direct:
        note += (f"\nDirect fine-tuning on TUEP windows (Qwen3-VL 2B, earlier images): window AUROC "
                 f"{float(direct['auroc']):.2f} [{float(direct['auroc_ci_low']):.2f}, {float(direct['auroc_ci_high']):.2f}].")
    import textwrap
    note = "\n".join(textwrap.fill(line, 150) for line in note.split("\n"))
    fig.text(0.01, 0.01, note, fontsize=7, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"tuep-scan.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def history(arm):
    # the last checkpoint's log covers the whole run, resumed parts included
    checkpoints = sorted((RESULTS / "trainer-states" / arm).glob("checkpoint-*/trainer_state.json"),
                         key=lambda p: int(p.parent.name.split("-")[1]))
    return json.loads(checkpoints[-1].read_text()) if checkpoints else None


def smooth(values, window=5):
    return np.convolve(np.pad(values, (window // 2, window - 1 - window // 2), mode="edge"), np.ones(window) / window,
                       mode="valid")


def draw_loss(ax, state, small=False):
    ax.set_facecolor(SURFACE)
    train = [(h["epoch"], h["loss"]) for h in state["log_history"] if "loss" in h]
    val = [(h["epoch"], h["eval_loss"]) for h in state["log_history"] if "eval_loss" in h]
    epochs, loss = np.array(train).T
    ax.plot(epochs, loss, color=ZERO, linewidth=0.6, alpha=0.35, zorder=2)
    ax.plot(epochs, smooth(loss), color=ZERO, linewidth=1.5 if small else 2, label="Training loss", zorder=3)
    if val:
        e, v = np.array(val).T
        ax.plot(e, v, color=TUNED, linewidth=1.5 if small else 2, marker="o", markersize=3 if small else 5,
                label="Validation loss", zorder=4)
    kept = state.get("best_model_checkpoint")
    if kept:
        step = int(kept.rstrip("/").rsplit("-", 1)[1])
        epoch = next((h["epoch"] for h in state["log_history"] if h.get("step") == step), None)
        if epoch is not None:
            ax.axvline(epoch, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1, label="Kept checkpoint")
    ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7 if small else 8, colors=SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def losses(out, models):
    note = {"TUSZ": "Training and validation loss: answer + teacher rationale tokens (answer-first).",
            "multitask": "Training loss: answer lines + rationale lines (multitask); validation loss: answer lines only."}
    for ds in QUESTIONS:
        folder = out / "loss" / ds.lower()
        folder.mkdir(parents=True, exist_ok=True)
        text = note["TUSZ" if ds == "TUSZ" else "multitask"]
        runs = [(m, history(tables.trained(m, ds))) for m in models]
        runs = [(m, state) for m, state in runs if state]
        for m, state in runs:
            summary = tables.summary(tables.trained(m, ds))
            fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
            draw_loss(ax, state)
            ax.set_title(f"{tables.MODELS[m]}, {ds}", loc="left", fontsize=11, color=TEXT, fontweight="bold")
            if summary:
                ax.set_title(f"test AUROC {float(summary['auroc']):.3f}", loc="right", fontsize=9, color=SECONDARY)
            ax.set_xlabel("Epoch", fontsize=9, color=SECONDARY)
            ax.set_ylabel("Loss (log scale)", fontsize=9, color=SECONDARY)
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper right")
            fig.text(0.01, 0.01, text, fontsize=7, color=MUTED)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(folder / f"{tables.MODELS[m].lower().replace(' ', '-')}.png", dpi=200, facecolor=SURFACE)
            plt.close(fig)
        fig, axes = plt.subplots(4, 4, figsize=(16, 13), facecolor=SURFACE, sharey=True)
        for ax, (m, state) in zip(axes.flat, runs):
            draw_loss(ax, state, small=True)
            summary = tables.summary(tables.trained(m, ds))
            auroc = f"  (AUROC {float(summary['auroc']):.2f})" if summary else ""
            ax.set_title(f"{tables.MODELS[m]}{auroc}", loc="left", fontsize=9, color=TEXT)
        for ax in list(axes.flat)[len(runs):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=TEXT)
        fig.suptitle(f"{ds}: {QUESTIONS[ds]}, loss by epoch", x=0.01, ha="left", fontsize=13, color=TEXT,
                     fontweight="bold")
        fig.text(0.01, 0.005, text + " Log scale; dashed line: the checkpoint that was scored.", fontsize=8,
                 color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"loss-{ds.lower()}.{suffix}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"{ds}: {len(runs)} loss curves -> {folder}")


if __name__ == "__main__":
    main()
