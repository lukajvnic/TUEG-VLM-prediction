#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, sys
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
import mne

ROOT = Path(__file__).resolve().parent
EDF_ROOT = ROOT / "v2.0.1" / "edf"
DATA_DIR = ROOT / "data-annotated"
LABELS = ["bckg", "seiz", "slow"]
COLORS = {"bckg": "#95a5a6", "seiz": "#c0392b", "slow": "#f1c40f"}


def csv_rows(path: Path, start: float, stop: float) -> list[dict]:
    if not path.exists(): return []
    lines = [x for x in path.read_text(errors="ignore").splitlines() if x and not x.startswith("#")]
    out = []
    for r in csv.DictReader(StringIO("\n".join(lines))):
        a, b, label = float(r["start_time"]), float(r["stop_time"]), r.get("label", "")
        if label in LABELS and a < stop and b > start: out.append({"start": max(a, start), "stop": min(b, stop), "label": label})
    return out


def annotations(edf: Path, start: float, stop: float) -> list[dict]:
    anns = []
    for ann in [edf.with_suffix(".csv"), *edf.parent.glob(edf.stem + "_*.csv")]: anns += csv_rows(ann, start, stop)
    seen = set(); out = []
    for x in anns:
        k = (round(x["start"], 3), round(x["stop"], 3), x["label"])
        if k not in seen: seen.add(k); out.append(x)
    return out


def image_name(edf: Path) -> str:
    return "__".join(edf.relative_to(EDF_ROOT).with_suffix("").parts) + ".png"


def render(edf: Path, png: Path, anns: list[dict], seconds: float, channels: int) -> None:
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    picks = list(mne.pick_types(raw.info, eeg=True, exclude=[]))[:channels] or list(range(min(channels, len(raw.ch_names))))
    data, times = raw.get_data(picks=picks, stop=min(raw.n_times, int(seconds * raw.info["sfreq"])), return_times=True)
    fig, axes = plt.subplots(len(picks), 1, sharex=True, figsize=(16, max(4, len(picks) * .75)))
    if len(picks) == 1: axes = [axes]
    for ax, pick, y in zip(axes, picks, data * 1e6):
        ax.plot(times, y - y.mean(), color="black", lw=.5, zorder=2); ax.set_ylabel(raw.ch_names[pick], rotation=0, ha="right", va="center", fontsize=6); ax.set_yticks([]); ax.set_xlim(0, seconds)
    for ann in anns:
        color = COLORS.get(ann["label"], "#7f8c8d")
        for ax in axes: ax.add_patch(Rectangle((ann["start"], 0), ann["stop"] - ann["start"], 1, transform=ax.get_xaxis_transform(), facecolor=to_rgba(color, .18), edgecolor=color, linewidth=1.2, zorder=1))
        axes[0].text(ann["start"], .92, ann["label"], transform=axes[0].get_xaxis_transform(), fontsize=7, color=color, weight="bold")
    used = sorted({x["label"] for x in anns})
    if used: fig.legend([Patch(color=COLORS.get(x, "#7f8c8d"), alpha=.35) for x in used], used, loc="upper right")
    axes[-1].set_xlabel("seconds"); fig.suptitle(edf.name); fig.tight_layout(); png.parent.mkdir(exist_ok=True); fig.savefig(png, dpi=130); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Render annotated TUSL EDF previews.")
    p.add_argument("--limit", type=int); p.add_argument("--overwrite", action="store_true"); p.add_argument("--seconds", type=float, default=20); p.add_argument("--channels", type=int, default=19)
    a = p.parse_args(); DATA_DIR.mkdir(exist_ok=True); edfs = sorted(EDF_ROOT.rglob("*.edf"))[:a.limit]
    for n, edf in enumerate(edfs, 1):
        png = DATA_DIR / image_name(edf); anns = annotations(edf, 0, a.seconds)
        try:
            if a.overwrite or not png.exists(): render(edf, png, anns, a.seconds, a.channels)
            print(f"{n}/{len(edfs)} {png}")
        except Exception as e: print(f"failed {edf}: {e}", file=sys.stderr)
    return 0

if __name__ == "__main__": raise SystemExit(main())
