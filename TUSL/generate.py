#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, sys
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parent
EDF_ROOT = ROOT / "v2.0.1" / "edf"
DATA_DIR = ROOT / "data"
LABELS = ["bckg", "seiz", "slow"]


def csv_labels(path: Path, start: float, stop: float) -> set[str]:
    lines = [x for x in path.read_text(errors="ignore").splitlines() if x and not x.startswith("#")]
    return {r["label"] for r in csv.DictReader(StringIO("\n".join(lines))) if r.get("label") in LABELS and float(r["start_time"]) < stop and float(r["stop_time"]) > start}


def labels_for(edf: Path, start: float, stop: float) -> set[str]:
    anns = [edf.with_suffix(".csv"), *edf.parent.glob(edf.stem + "_*.csv")]
    found = set()
    for ann in anns:
        if ann.exists(): found |= csv_labels(ann, start, stop)
    return found


def image_name(edf: Path) -> str:
    return "__".join(edf.relative_to(EDF_ROOT).with_suffix("").parts) + ".png"


def edf_to_png(edf: Path, png: Path, seconds: float, channels: int) -> None:
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    picks = list(mne.pick_types(raw.info, eeg=True, exclude=[]))[:channels]
    if not picks: picks = list(range(min(channels, len(raw.ch_names))))
    data, times = raw.get_data(picks=picks, stop=min(raw.n_times, int(seconds * raw.info["sfreq"])), return_times=True)
    data = data * 1e6
    fig, axes = plt.subplots(len(picks), 1, sharex=True, figsize=(16, max(4, len(picks) * .7)))
    if len(picks) == 1: axes = [axes]
    for ax, pick, y in zip(axes, picks, data):
        ax.plot(times, y - y.mean(), color="black", lw=.5)
        ax.set_ylabel(raw.ch_names[pick], rotation=0, ha="right", va="center", fontsize=6)
        ax.set_yticks([])
    axes[-1].set_xlabel("seconds"); fig.suptitle(edf.name); fig.tight_layout(); png.parent.mkdir(exist_ok=True)
    fig.savefig(png, dpi=120); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Render TUSL EDFs to PNGs and write file-level labels.")
    p.add_argument("--limit", type=int); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seconds", type=float, default=20); p.add_argument("--channels", type=int, default=19)
    a = p.parse_args(); DATA_DIR.mkdir(exist_ok=True)
    edfs = sorted(EDF_ROOT.rglob("*.edf"))[:a.limit]
    with (ROOT / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["image_path", *LABELS]); w.writeheader()
        for n, edf in enumerate(edfs, 1):
            png = DATA_DIR / image_name(edf)
            try:
                if a.overwrite or not png.exists(): edf_to_png(edf, png, a.seconds, a.channels)
                labs = labels_for(edf, 0, a.seconds); w.writerow({"image_path": str(png.relative_to(ROOT)), **{x: str(x in labs).lower() for x in LABELS}})
                print(f"{n}/{len(edfs)} {png}")
            except Exception as e: print(f"failed {edf}: {e}", file=sys.stderr)
    return 0

if __name__ == "__main__": raise SystemExit(main())
