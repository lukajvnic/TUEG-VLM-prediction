# a second rendering of the TUSZ windows the vision-freeze test trains and scores on, written to side/vision-freeze/
# images/ and never mixed with the shipped PNGs (render.py does not reproduce those byte-for-byte, so a model could
# learn the render style as a cue). Same windows, labels and image size; three changes, each from the 2026-10-06
# research pass (knowledge/known-issues.md):
#   - bipolar longitudinal "double banana" (ACNS LB-18.1, 18 rows), EEG electrodes only: the shipped images plot
#     the stored referential channels, and 41% of their rows are not EEG (EKG, EMG, IBI/BURSTS/SUPPR)
#   - one gain per recording, shared by every row: amplitude relative to the recording's background separates seizure
#     from background windows (AUROC 0.60-0.66), and the shipped per-window, per-channel autoscale erases it
#   - soft compression instead of clipping: a row shows H*tanh(x/H), H = GAIN x the recording's background level,
#     so background draws near-linearly and large swings bend toward the row edge without going flat
# Also lists extra windows for the "more" arm: every seizure window of the training recordings that the shipped
# set never rendered, plus as many background-only windows of the same recordings
#   python side/vision-freeze/rerender.py lists          # data/windows.csv + data/more-train.jsonl (no rendering)
#   python side/vision-freeze/rerender.py render [N]     # render data/windows.csv into images/ with N processes
#   python side/vision-freeze/rerender.py sample         # 4 windows into images-sample/ to look at
#   python side/vision-freeze/rerender.py background [N] # data/background.csv (see below), rendered into images/
#   python side/vision-freeze/rerender.py datasets [N]   # the other corpora (see below), rendered into images/<DS>/
#   python side/vision-freeze/rerender.py core [N]       # every corpus in the core design (see below), images-core/
#   python side/vision-freeze/rerender.py tuep-scan [N]  # TUEP test recordings for the spike-detector scan (see below)
#   python side/vision-freeze/rerender.py heldout [N]    # the final TUSZ test: the 62 patients in no core split
#   python side/vision-freeze/rerender.py extra DS         # one question per class of EXTRA[DS], same windows (data/core/)
#   python side/vision-freeze/rerender.py tusz-types       # one question per TUSZ seizure type with training patients
#   python side/vision-freeze/rerender.py dataset-all      # each dataset's question lines merged for one model per dataset
# background: the scored TUSZ test set's background is 97.8% first/last windows of its recording (known-issues,
# 2026-10-14). This lists, from the same 200 test patients, shipped test windows never scored that are background
# only and not a recording's first or last window, up to 2 per recording spread through it (seed 0), with each
# window's distance in windows to the nearest seizure window of its recording (from the annotations)
# datasets (2026-10-14): does the recipe transfer? One yes/no question per other corpus (QUESTIONS), on the
# fine-tune's vetted train/val windows (sft_labels_*.jsonl: patient caps, cross-corpus image screen) and the
# benchmark's scored test windows. Labels come from labels.csv and must reproduce from the source annotations.
# Writes data/<DS>-{train,val,test}.jsonl and data/<DS>-windows.csv, renders into images/<DS>/ (window names repeat
# across corpora)
# core (2026-10-14, knowledge/methodology-decisions.md "Image redesign"): the same windows in the core design. TUH's
# 22-row TCP montage plus Fz-Cz, Cz-Pz and EKG (25 rows, coloured by chain), on a 2688x896 canvas (~127 px/s; both
# sides multiples of 224, so neither Qwen3-VL nor Qwen2.5-VL resizes it), 1 pt lines, no row frames, a 1 s grid. The
# gain stays one per recording; a 100 uV bar, whose length follows that gain, keeps the absolute scale in the image.
# Every corpus goes to images-core/<DS>/{train,val,test}/ with its lines in data/core/, and the build stops unless
# patients and recordings are disjoint across the three splits. TUSZ: train = the "more" lines (the pilot's train
# lines are a subset, data/core/TUSZ-pilot-train.jsonl), val = the pilot's, test = the scored windows, plus the
# interior background windows (data/core/TUSZ-background.jsonl)
import csv
import functools
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))

DATASET = ROOT / "datasets" / "TUSZ"
EDF_ROOT = DATASET / "v2.0.6" / "edf"
IMAGES, DATA = SIDE / "images", SIDE / "data"
LABELS = ["bckg", "absz", "cpsz", "fnsz", "gnsz", "mysz", "spsz", "tcsz", "tnsz"]
SEIZURES = set(LABELS) - {"bckg"}
BANANA = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
          ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
          ("FZ", "CZ"), ("CZ", "PZ"),
          ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
          ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2")]
GAIN = 2.0  # H = GAIN x the recording's background level (median over windows and rows of the window p99)
SEED = 0
SUBJECT = re.compile(r"aaaa[a-z]{4}")
NAME = re.compile(r"^(train|test)/(aaaa[a-z]{4}_\d+)_(\d+)\.png$")
ANY_NAME = re.compile(r"^(train|test)/(.+_\d+)_(\d+)\.png$")  # TUEV's excerpt patients are numbers ("000_0_2.png")
# corpus: (question, the labels.csv classes that make it true, what the prompt asks about)
QUESTIONS = {"TUAB": ("abnormal", ("abnormal",), "abnormal EEG activity"),
             "TUEP": ("epilepsy", ("epilepsy",), "EEG from a patient with epilepsy"),
             "TUAR": ("eyem", ("eyem",), "an eye movement artifact (EYEM)"),
             "TUEV": ("epileptiform", ("spsw", "gped", "pled"),
                      "an epileptiform discharge (a spike or sharp wave, GPED or PLED)"),
             "TUSL": ("slowing", ("slow",), "slowing (focal or generalized)")}
# extra(): one more yes/no question per learnable class, on a corpus's core windows and images (maintainer, 2026-10-25:
# TUAR asks eye movement, muscle and electrode artifact; TUEV asks spike, GPED and PLED separately)
EXTRA = {"TUAR": {"musc": "a muscle artifact (MUSC)", "elec": "an electrode artifact (ELEC)"},
         "TUEV": {"spsw": "a spike or sharp wave (SPSW)", "gped": "a generalized periodic epileptiform discharge (GPED)",
                  "pled": "a periodic lateralized epileptiform discharge (PLED)"}}
TUSZ_TYPES = {"fnsz": "a focal non-specific seizure (FNSZ)", "gnsz": "a generalized non-specific seizure (GNSZ)",
              "cpsz": "a complex partial seizure (CPSZ)", "absz": "an absence seizure (ABSZ)",
              "tcsz": "a tonic-clonic seizure (TCSZ)"}  # tusz_types()
CORE_IMAGES, CORE_DATA = SIDE / "images-core", DATA / "core"
# TUEV train/val windows dropped from the core lists: a signal-fingerprint audit (2026-10-14) found 2 TUEV recordings
# copied into test under other names (007 = 021, 008 = 022, so every 007 and 008 window goes) and 2 patients with
# recordings on both sides once TUEV's separately anonymised names are traced through the other corpora, plus 1
# patient shared by train and val
EXCLUDE = {"TUEV": SIDE / "tuev-exclude.csv"}
CORE_SIZE, CORE_DPI = (2688, 896), 100
CORE_GUTTER, CORE_MARGIN = 110, 56  # px: row labels on the left, the calibration bar on the right
EKG_FLOOR = 20e-6  # V: the EKG row is scaled per window to at least this, so a flat EKG stays flat
CORE_CLIP = 0.85  # a row's trace stays within 0.85 row pitches of its centre (tanh)
# ACNS order (left above right, chains grouped), then TUH's transverse chain, then EKG
CORE_GROUPS = [("#1f4e9c", [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1")]),
               ("#1b7f75", [("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1")]),
               ("#222222", [("FZ", "CZ"), ("CZ", "PZ")]),
               ("#a8561a", [("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")]),
               ("#b0202a", [("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2")]),
               ("#5e3a96", [("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"), ("T4", "A2")])]
EKG_COLOR, MISSING_COLOR = "#6b6b6b", "#bdbdbd"
GRID_COLORS = ("#e2e2e2", "#c4c4c4")
ANNOTATED_HEADER = 60  # px: annotate.py's title and legend strip above the plot


@functools.cache
def core_palette():
    # one palette for every image: each drawn colour blended with white in 20 steps, so antialiasing shades map to
    # the right hue whatever else is in the image (a per-image palette turned the grey EKG label purple)
    from PIL import Image

    bases = [c for c, _ in CORE_GROUPS] + [EKG_COLOR, MISSING_COLOR, "#000000", *GRID_COLORS]
    colors = []
    for base in bases:
        rgb = [int(base[i:i + 2], 16) for i in (1, 3, 5)]
        for t in range(20):
            color = tuple(round(c + (255 - c) * t / 20) for c in rgb)
            if color not in colors:
                colors.append(color)
    colors.append((255, 255, 255))
    palette = Image.new("P", (1, 1))
    palette.putpalette([v for c in colors for v in c] + [255] * 3 * (256 - len(colors)))
    return palette


def patient_of(edf):
    match = SUBJECT.search(edf.name)
    return match.group(0) if match else edf.parent.name


def scans():
    # render.py's scan_names over every TUSZ EDF: <patient>_<n> by sorted path, numbered per patient
    counters, names = defaultdict(int), {}
    for edf in sorted(EDF_ROOT.rglob("*.edf"), key=str):
        patient = patient_of(edf)
        names[f"{patient}_{counters[patient]}"] = edf
        counters[patient] += 1
    return names


def electrode(name):
    match = re.match(r"EEG (\w+)-(REF|LE)$", name.upper())
    return match.group(1) if match else None


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def window_labels(edf, n):
    # preprocess-TUSZ.py's labels_for: any annotation row overlapping the window, else bckg
    from render import WINDOW, csv_labels

    return [csv_labels([edf.with_suffix(".csv")], i * WINDOW, (i + 1) * WINDOW, LABELS) or {"bckg"} for i in range(n)]


def lists():
    # every window to render (shipped train/val/test windows of the fine-tune and scoring, plus the extras) and the
    # "more" arm's balanced training lines. Extras come only from recordings already in the fine-tune's train split:
    # those passed the sampler's patient and cross-corpus screens, and no TUSZ train EDF matches a test EDF of any
    # corpus by signal fingerprint (data audit, 2026-10-06)
    import mne
    from render import WINDOW

    names = scans()
    train = [r["images"][0] for r in read_jsonl(DATASET / "sft_labels_train.jsonl")]
    val = [r["images"][0] for r in read_jsonl(DATASET / "sft_labels_val.jsonl")]
    with sqlite3.connect(f"file:{ROOT / 'pipeline.db'}?mode=ro", uri=True) as db:
        test = sorted(p.split("/", 1)[1] for (p,) in db.execute(
            "SELECT DISTINCT path FROM pipeline WHERE dataset = 'TUSZ' AND scope = 'full'"))
    shipped = {r["path"]: {c for c in LABELS if r[c] == "true"} for r in csv.DictReader((DATASET / "labels.csv").open())}

    have = set(train) | set(val)
    seizure_extra, calm_extra, mismatches = [], [], 0
    recordings = sorted({NAME.match(p).group(2) for p in train})
    for k, scan in enumerate(recordings):
        edf = names[scan]
        header = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
        n = max(1, int((header.n_times / header.info["sfreq"]) // WINDOW))  # render.py's window_bounds count
        for i, found in enumerate(window_labels(edf, n)):
            path = f"train/{scan}_{i}.png"
            if path in shipped:
                mismatches += shipped[path] != found
            if path in have:
                continue
            (seizure_extra if found & SEIZURES else calm_extra).append((path, found))
        if k % 200 == 0:
            print(f"{k}/{len(recordings)} recordings", flush=True)
    print(f"label check: {mismatches} shipped train windows whose labels differ from the annotations")
    if mismatches:
        sys.exit("labels do not reproduce: stop")

    # the "more" arm: every seizure window of these recordings, and as many background-only ones, the shipped
    # background windows first, then extras spread over recordings (seeded)
    rng = random.Random(SEED)
    old = [(p, shipped[p]) for p in train]
    seizure = [(p, l) for p, l in old if l & SEIZURES] + seizure_extra
    calm_old = [(p, l) for p, l in old if not l & SEIZURES]
    rng.shuffle(calm_extra)
    calm = calm_old + calm_extra[:max(0, len(seizure) - len(calm_old))]
    calm = calm[:len(seizure)]
    shipped_train = set(train)
    picked_extra = {p for p, _ in seizure_extra} | {p for p, _ in calm if p not in shipped_train}

    DATA.mkdir(parents=True, exist_ok=True)
    rows = [(p, "train") for p in train] + [(p, "val") for p in val] + [(p, "test") for p in test] + \
           [(p, "extra") for p in sorted(picked_extra)]
    with (DATA / "windows.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "role", "scan", "index", "edf"])
        for path, role in rows:
            _, scan, index = NAME.match(path).groups()
            writer.writerow([path, role, scan, index, str(names[scan].relative_to(ROOT))])

    prompt = {r["instruction"] for r in read_jsonl(ROOT / "side" / "multilabel-pilot" / "data" / "perclass-train.jsonl")
              if r["question"] == "any_seizure"}
    assert len(prompt) == 1
    prompt = prompt.pop()
    lines = [{"dataset": "TUSZ", "question": "any_seizure", "instruction": prompt, "input": "",
              "output": json.dumps({"present": bool(l & SEIZURES)}), "images": [p]} for p, l in seizure + calm]
    rng.shuffle(lines)
    with (DATA / "more-train.jsonl").open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    print(f"windows: train {len(train)}, val {len(val)}, test {len(test)}, extra {len(picked_extra)} "
          f"(seizure {len(seizure_extra)}, background {len(picked_extra) - len(seizure_extra)}) -> {DATA / 'windows.csv'}")
    print(f"more-train: {len(lines)} lines, {len(seizure)} seizure / {len(calm)} background -> {DATA / 'more-train.jsonl'}")


def bipolar(raw):
    # the 18 LB-18.1 derivations; an electrode the recording lacks gives a flat row, so every image has the same rows
    index = {electrode(n): i for i, n in enumerate(raw.ch_names) if electrode(n)}
    data = raw.get_data()
    flat = data[0] * 0
    rows = [data[index[a]] - data[index[b]] if a in index and b in index else flat for a, b in BANANA]
    names = [f"{a.capitalize()}-{b.capitalize()}" for a, b in BANANA]
    return rows, names


def background_level(rows, sfreq, windows):
    import numpy as np

    stacked = np.array(rows)
    p99 = [np.percentile(np.abs(stacked[:, int(s * sfreq):int(e * sfreq)]), 99, axis=1) for s, e in windows]
    values = np.array(p99)
    values = values[values > 0]
    return float(np.median(values)) if values.size else 1.0


def draw(rows, names, level, sfreq, start, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from render import DPI, FIG_SIZE, LABEL_PT, LINEWIDTH, WINDOW

    a, b = int(round(start * sfreq)), int(round((start + WINDOW) * sfreq))
    height = GAIN * level
    seconds = np.arange(b - a) / sfreq
    fig, axes = plt.subplots(len(rows), 1, figsize=(FIG_SIZE, FIG_SIZE), dpi=DPI, sharex=True)
    for ax, name, y in zip(axes, names, rows):
        segment = np.full(b - a, np.nan)  # a recording shorter than one window (134 of them) leaves the rest blank
        part = y[a:b]
        segment[:len(part)] = part
        ax.plot(seconds, height * np.tanh(segment / height), linewidth=LINEWIDTH, color="black")
        ax.set_ylim(-height, height)
        ax.set_ylabel(name, fontsize=LABEL_PT, weight="bold", rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.set_xlim(0, WINDOW)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, dpi=DPI, format="png")
    plt.close(fig)
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.open(buffer).convert("L").save(png, optimize=True)  # grayscale: the plot is black on white anyway


def core_rows(raw):
    # (label, colour, signal or None when the recording lacks an electrode) per row, EKG last
    index = {electrode(n): i for i, n in enumerate(raw.ch_names) if electrode(n)}
    data = raw.get_data()
    rows = [(f"{a.capitalize()}-{b.capitalize()}", color, data[index[a]] - data[index[b]] if a in index and b in index else None)
            for color, pairs in CORE_GROUPS for a, b in pairs]
    ekg = [i for i, n in enumerate(raw.ch_names) if re.search(r"EKG|ECG", n.upper())]
    return rows + [("EKG", EKG_COLOR, data[ekg[0]] if ekg else None)]


def draw_core(rows, level, sfreq, start, png, spans=(), header=None):
    # spans and header are for annotate.py's human-facing copies: (row label or None for every row, start s, stop s,
    # colour) shaded behind the traces, and a title strip with a legend above the plot. Without them the image is the
    # core design, byte for byte
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    from PIL import Image
    from render import WINDOW

    width, height = CORE_SIZE
    if header is None:
        fig = plt.figure(figsize=(width / CORE_DPI, height / CORE_DPI), dpi=CORE_DPI)
        ax = fig.add_axes([CORE_GUTTER / width, 6 / height, 1 - (CORE_GUTTER + CORE_MARGIN) / width, 1 - 12 / height])
    else:
        total = height + ANNOTATED_HEADER
        fig = plt.figure(figsize=(width / CORE_DPI, total / CORE_DPI), dpi=CORE_DPI)
        ax = fig.add_axes([CORE_GUTTER / width, 6 / total, 1 - (CORE_GUTTER + CORE_MARGIN) / width, (height - 12) / total])
        title, legend = header
        fig.text(CORE_GUTTER / width, 1 - 30 / total, title, ha="left", va="center", fontsize=13, weight="bold")
        if legend:
            from matplotlib.patches import Patch
            fig.legend([Patch(facecolor=c, alpha=0.35, edgecolor=c) for _, c in legend], [name for name, _ in legend],
                       loc="upper right", bbox_to_anchor=(1 - 8 / width, 1 - 6 / total), ncol=min(len(legend), 4),
                       frameon=False, fontsize=11)
        else:
            fig.text(1 - CORE_MARGIN / width, 1 - 30 / total, "no annotated event in this window", ha="right",
                     va="center", fontsize=11, color="#6b6b6b")
    a, b = int(round(start * sfreq)), int(round((start + WINDOW) * sfreq))
    seconds = np.arange(b - a) / sfreq
    gain = GAIN * level
    n = len(rows)
    centers = {label.upper(): n - 1 - k for k, (label, _, _) in enumerate(rows)}
    for row, x0, x1, color in spans:
        low, high = (centers[row] - 0.5, centers[row] + 0.5) if row in centers else (-0.6, n - 0.4)
        ax.fill_between([x0, x1], low, high, color=color, alpha=0.28, linewidth=0, zorder=0.5)
    lines, colors, missing = [], [], []
    for k, (label, color, y) in enumerate(rows):
        center = n - 1 - k
        ax.text(-0.12, center, label, ha="right", va="center", fontsize=9, color=color, weight="bold")
        if y is None:
            missing.append([(0, center), (WINDOW, center)])
            continue
        segment = np.full(b - a, np.nan)  # a recording shorter than one window leaves the rest blank
        part = y[a:b]
        segment[:len(part)] = part
        if label == "EKG":  # shape only: its amplitude says nothing about the EEG
            scale = max(np.nanpercentile(np.abs(part), 99) if len(part) else 0, EKG_FLOOR)
            trace = 0.8 * np.tanh(segment / scale)
        else:
            trace = CORE_CLIP * np.tanh(segment / gain)
        lines.append(np.column_stack([seconds, center + trace]))
        colors.append(color)
    for s in range(1, int(WINDOW)):
        ax.axvline(s, color="#e2e2e2" if s % 5 else "#c4c4c4", linewidth=0.6, zorder=0)
    if missing:
        ax.add_collection(LineCollection(missing, colors=MISSING_COLOR, linewidths=1.0, linestyles=(0, (4, 4))))
    ax.add_collection(LineCollection(lines, colors=colors, linewidths=1.0))
    # 100 uV at this recording's gain (the tanh's slope at zero), so the absolute scale is visible: a quiet recording
    # draws a tall bar, a loud one a short bar
    bar = min(CORE_CLIP * 100e-6 / gain, n - 1.0)  # capped only below ~1.8 uV of background (flat recordings)
    x = WINDOW + 0.12
    ax.plot([x, x], [-0.6, -0.6 + bar], color="black", linewidth=1.5, clip_on=False)
    ax.text(x + 0.12, -0.6 + bar / 2, "100 µV", ha="left", va="center", rotation=90, fontsize=7, clip_on=False)
    ax.set_xlim(0, WINDOW)
    ax.set_ylim(-1.0, n - 0.2)
    ax.axis("off")
    buffer = BytesIO()
    fig.savefig(buffer, dpi=CORE_DPI, format="png")
    plt.close(fig)
    png.parent.mkdir(parents=True, exist_ok=True)
    # one fixed 256-colour palette (no dithering): a third of the size. Written to a
    # temporary name first: an interrupted run must not leave a truncated PNG that a rerun would skip
    image = Image.open(buffer).convert("RGB")
    partial = png.with_suffix(".partial")
    if header is None:
        image.quantize(palette=core_palette(), dither=Image.Dither.NONE).save(partial, "PNG", optimize=True)
    else:  # the shaded spans add colours the fixed palette lacks
        image.quantize(colors=256, dither=Image.Dither.NONE).save(partial, "PNG", optimize=True)
    os.replace(partial, png)


def render_recording(job):
    from render import read, window_bounds

    edf, items, out, style = job
    todo = [(path, int(index)) for path, index in items if not (out / path).exists()]
    if not todo:
        return len(items), 0, None
    try:
        raw = read(ROOT / edf)
        sfreq = raw.info["sfreq"]
        windows = window_bounds(raw)
        if style == "core":
            rows = core_rows(raw)
            eeg = [y for label, _, y in rows if y is not None and label != "EKG"]
            level = background_level(eeg, sfreq, windows) if eeg else 1.0
            for path, index in todo:
                draw_core(rows, level, sfreq, windows[index][0], out / path)
            return len(items), len(todo), None
        rows, names = bipolar(raw)
        level = background_level(rows, sfreq, windows)
        for path, index in todo:
            draw(rows, names, level, sfreq, windows[index][0], out / path)
        return len(items), len(todo), None
    except Exception as e:
        return len(items), 0, f"{edf}: {type(e).__name__}: {e}"


def background(processes):
    import mne
    from render import WINDOW

    names = scans()
    with sqlite3.connect(f"file:{ROOT / 'pipeline.db'}?mode=ro", uri=True) as db:
        scope = {p.split("/", 1)[1] for (p,) in db.execute(
            "SELECT DISTINCT path FROM pipeline WHERE dataset = 'TUSZ' AND scope = 'full'")}
    shipped = {r["path"]: {c for c in LABELS if r[c] == "true"} for r in csv.DictReader((DATASET / "labels.csv").open())
               if r["path"].startswith("test/")}
    by_recording = defaultdict(list)
    for path in shipped:
        _, scan, index = NAME.match(path).groups()
        by_recording[scan].append((int(index), path))

    rng, rows, mismatches = random.Random(SEED), [], 0
    for scan in sorted(by_recording):
        edf = names[scan]
        header = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
        n = max(1, int((header.n_times / header.info["sfreq"]) // WINDOW))
        found = window_labels(edf, n)
        seizures = [i for i, labels in enumerate(found) if labels & SEIZURES]
        candidates = []
        for index, path in sorted(by_recording[scan]):
            mismatches += shipped[path] != found[index]
            if path not in scope and not shipped[path] & SEIZURES and 0 < index < n - 1:
                candidates.append((index, path))
        if len(candidates) > 2:  # two spread through the recording: one from each half of the candidates
            half = len(candidates) // 2
            candidates = [rng.choice(candidates[:half]), rng.choice(candidates[half:])]
        for index, path in candidates:
            distance = min(abs(index - s) for s in seizures) if seizures else ""
            rows.append([path, "background", scan, index, str(edf.relative_to(ROOT)), n, bool(seizures), distance])
    print(f"label check: {mismatches} shipped test windows whose labels differ from the annotations")
    if mismatches:
        sys.exit("labels do not reproduce: stop")

    DATA.mkdir(parents=True, exist_ok=True)
    table = DATA / "background.csv"
    with table.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "role", "scan", "index", "edf", "recording_windows", "seizure_recording",
                         "windows_to_seizure"])
        writer.writerows(rows)
    in_seizure = sum(r[6] for r in rows)
    print(f"{len(rows)} background windows from {len({r[2] for r in rows})} recordings ({in_seizure} in seizure "
          f"recordings) -> {table}")
    render_all(processes, table=table)


def sources(ds):
    # the EDFs preprocess-<DS>.py rendered, and its label function, so window names map back to recordings exactly
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"preprocess_{ds}", ROOT / "preprocessing" / f"preprocess-{ds}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "select"):  # TUAB, TUEP: one class per recording, from its folder
        classes = module.select()
        return sorted(classes), lambda edf, start, stop: {classes[edf]}
    return sorted(module.RAW.rglob("*.edf")), module.labels_for


def datasets(processes):
    from render import WINDOW, scan_names

    prompt = config_intro()
    with sqlite3.connect(f"file:{ROOT / 'pipeline.db'}?mode=ro", uri=True) as db:
        for ds, (question, positive, what) in QUESTIONS.items():
            edfs, labels_for = sources(ds)
            names = {name: edf for edf, name in scan_names(edfs).items()}
            labels = {r["path"]: r for r in csv.DictReader((ROOT / "datasets" / ds / "labels.csv").open())}
            # bckg is left out: recover_tuar_background.py set it on TUAR windows the annotations leave empty (517 here)
            classes = [c for c in next(iter(labels.values())) if c not in ("path", "ground_truth_rationale", "bckg")]
            split = {s: [r["images"][0] for r in read_jsonl(ROOT / "datasets" / ds / f"sft_labels_{s}.jsonl")]
                     for s in ("train", "val")}
            split["test"] = sorted(p.split("/", 1)[1] for (p,) in db.execute(
                "SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'full'", (ds,)))

            rows, mismatches = [], 0
            for role, paths in split.items():
                for path in paths:
                    _, scan, index = ANY_NAME.match(path).groups()
                    edf, index = names[scan], int(index)
                    shipped = {c for c in classes if labels[path][c] == "true"}
                    mismatches += shipped != set(labels_for(edf, index * WINDOW, (index + 1) * WINDOW)) & set(classes)
                    rows.append((path, role, scan, index, edf, bool(shipped & set(positive))))
            print(f"{ds}: label check, {mismatches} windows whose labels differ from the annotations")
            if mismatches:
                sys.exit("labels do not reproduce: stop")

            DATA.mkdir(parents=True, exist_ok=True)
            table = DATA / f"{ds}-windows.csv"
            with table.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "role", "scan", "index", "edf"])
                writer.writerows([p, r, sc, i, str(e.relative_to(ROOT))] for p, r, sc, i, e, _ in rows)
            text = f"{prompt}Based only on these waveforms, does this window show {what}? Set present to true only when it does."
            for role in split:
                with (DATA / f"{ds}-{role}.jsonl").open("w") as f:
                    for p, r, _, _, _, present in rows:
                        if r == role:
                            f.write(json.dumps({"dataset": ds, "question": question, "instruction": text, "input": "",
                                                "output": json.dumps({"present": present}), "images": [p]}) + "\n")
            counts = {role: (sum(r == role for _, r, *_ in rows), sum(r == role and x for _, r, *_, x in rows)) for role in split}
            print(f"{ds} '{question}': " + ", ".join(f"{role} {n} ({k} true)" for role, (n, k) in counts.items()), flush=True)
            render_all(processes, out=IMAGES / ds, table=table)


def config_intro():
    import yaml

    return yaml.safe_load((ROOT / "config.yml").read_text())["prompts"]["intro"]


def core_splits():
    # every corpus's windows by split, from the lists the current render was built from: (path, present, scan,
    # index, edf). TUSZ's val windows are named train/... like every fine-tune window, so core paths are
    # <split>/<name>
    tusz = {r["path"]: r for r in csv.DictReader((DATA / "windows.csv").open())}
    tusz.update({r["path"]: r for r in csv.DictReader((DATA / "background.csv").open())})

    def tusz_lines(rows):
        return [(r["images"][0], json.loads(r["output"])["present"]) for r in rows]

    pilot = ROOT / "side" / "multilabel-pilot" / "data"
    pilot_lines = lambda split: [r for r in read_jsonl(pilot / f"perclass-{split}.jsonl") if r["question"] == "any_seizure"]
    labels = {r["path"]: any(r[c] == "true" for c in SEIZURES) for r in csv.DictReader((DATASET / "labels.csv").open())}
    with sqlite3.connect(f"file:{ROOT / 'pipeline.db'}?mode=ro", uri=True) as db:
        test = sorted(p.split("/", 1)[1] for (p,) in db.execute(
            "SELECT DISTINCT path FROM pipeline WHERE dataset = 'TUSZ' AND scope = 'full'"))
    background = [r["path"] for r in csv.DictReader((DATA / "background.csv").open())]
    splits = {"TUSZ": {"train": tusz_lines(read_jsonl(DATA / "more-train.jsonl")),
                       "pilot-train": tusz_lines(pilot_lines("train")), "val": tusz_lines(pilot_lines("val")),
                       "test": [(p, labels[p]) for p in test], "background": [(p, False) for p in background]}}
    tables = {"TUSZ": tusz}
    for ds in QUESTIONS:
        tables[ds] = {r["path"]: r for r in csv.DictReader((DATA / f"{ds}-windows.csv").open())}
        drop = {r["path"]: r["role"] for r in csv.DictReader(EXCLUDE[ds].open())} if ds in EXCLUDE else {}
        assert all(role in ("train", "val") for role in drop.values()), f"{ds}: an excluded window is not train/val"
        splits[ds] = {s: [(p, x) for p, x in tusz_lines(read_jsonl(DATA / f"{ds}-{s}.jsonl")) if p not in drop]
                      for s in ("train", "val", "test")}
        removed = sum(len(read_jsonl(DATA / f"{ds}-{s}.jsonl")) for s in ("train", "val")) - \
            len(splits[ds]["train"]) - len(splits[ds]["val"])
        assert removed == len(drop), f"{ds}: dropped {removed} of {len(drop)} listed windows"
    return splits, tables


def missing_rows(edf):
    # the drawn electrodes (and EKG) a recording lacks: those rows are dashed in every one of its windows
    import mne

    names = mne.io.read_raw_edf(str(ROOT / edf), preload=False, verbose="ERROR").ch_names
    have = {electrode(n) for n in names}
    needed = sorted({e for _, pairs in CORE_GROUPS for pair in pairs for e in pair})
    ekg = [] if any(re.search(r"EKG|ECG", n.upper()) for n in names) else ["EKG"]
    return tuple(e for e in needed if e not in have) + tuple(ekg)


def balance(rows, table):
    # within each missing-row pattern, trim the over-represented class to the split's overall positive rate, so a
    # dashed row says nothing about the label. In TUSZ's "more" lines, windows without EKG were 23% seizure against
    # 51% (2026-10-14 audit), and without Fz/Pz 38% against 50%
    rate = sum(x for _, x in rows) / len(rows)
    patterns, groups = {}, defaultdict(list)
    for path, present in rows:
        edf = table[path]["edf"]
        if edf not in patterns:
            patterns[edf] = missing_rows(edf)
        groups[patterns[edf]].append((path, present))
    rng, keep, report = random.Random(SEED), set(), []
    for pattern, members in sorted(groups.items()):
        positive = sorted(p for p, x in members if x)
        negative = sorted(p for p, x in members if not x)
        if pattern and 0 < rate < 1:
            if len(positive) > rate * len(members):
                positive = rng.sample(positive, round(len(negative) * rate / (1 - rate)))
            else:
                negative = rng.sample(negative, min(len(negative), round(len(positive) * (1 - rate) / rate)))
            report.append(f"{'/'.join(pattern)}: kept {len(positive) + len(negative)} of {len(members)}")
        keep |= set(positive) | set(negative)
    return [(p, x) for p, x in rows if p in keep], report


def core(processes):
    # every corpus's lists are built and gated before any rendering, so a failed gate costs seconds, not hours
    splits, tables = core_splits()
    CORE_DATA.mkdir(parents=True, exist_ok=True)
    prompt = config_intro()
    for ds, parts in splits.items():
        # the leakage gate: no patient and no recording in more than one of train / val / test
        role = {"train": "train", "pilot-train": "train", "val": "val", "test": "test", "background": "test"}
        patients, recordings = defaultdict(set), defaultdict(set)
        for split, rows in parts.items():
            for path, _ in rows:
                scan = ANY_NAME.match(path).group(2)
                patients[role[split]].add(scan.rsplit("_", 1)[0])
                recordings[role[split]].add(tables[ds][path]["edf"])
        for one, two in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = (patients[one] & patients[two], recordings[one] & recordings[two])
            if any(shared):
                sys.exit(f"{ds}: {one} and {two} share {len(shared[0])} patients and {len(shared[1])} recordings, e.g. "
                         f"{sorted(shared[0])[:3]} {sorted(shared[1])[:2]}: stop")

        for split in ("train", "pilot-train"):
            if split in parts:
                parts[split], report = balance(parts[split], tables[ds])
                print(f"{ds} {split}: missing-row balance, " + ("; ".join(report) or "nothing missing"), flush=True)

        windows = {}
        for split, rows in parts.items():
            text = None
            with (CORE_DATA / f"{ds}-{split}.jsonl").open("w") as f:
                for path, present in rows:
                    name = f"{role[split]}/{path.split('/', 1)[1]}"
                    row = tables[ds][path]
                    windows[name] = (row["scan"], row["index"], row["edf"])
                    if text is None:
                        text = question_text(ds, prompt)
                    f.write(json.dumps({"dataset": ds, "question": question_of(ds), "instruction": text, "input": "",
                                        "output": json.dumps({"present": present}), "images": [name]}) + "\n")
        table = CORE_DATA / f"{ds}-windows.csv"
        with table.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "role", "scan", "index", "edf"])
            writer.writerows([name, name.split("/")[0], *windows[name]] for name in sorted(windows))
        counts = ", ".join(f"{split} {len(rows)} ({sum(x for _, x in rows)} true)" for split, rows in parts.items())
        print(f"{ds}: {counts}; patients train {len(patients['train'])} / val {len(patients['val'])} / test "
              f"{len(patients['test'])}, no overlap -> {table}", flush=True)
    for ds in splits:
        render_all(processes, out=CORE_IMAGES / ds, table=CORE_DATA / f"{ds}-windows.csv", style="core")
    check_core()


def check_core():
    # every listed image exists and decodes in full, before anything is packed for the cluster
    from PIL import Image

    bad, total = [], 0
    for table in sorted(CORE_DATA.glob("*-windows.csv")):
        ds = table.name.split("-")[0]  # TUEP-scan's images are in TUEP/scan/, TUSZ-heldout's in TUSZ/heldout/
        for row in csv.DictReader(table.open()):
            total += 1
            try:
                with Image.open(CORE_IMAGES / ds / row["path"]) as image:
                    image.load()
                    assert image.size == CORE_SIZE, image.size
            except Exception as e:
                bad.append(f"{ds}/{row['path']}: {type(e).__name__}: {e}")
    print(f"check: {total} images, {len(bad)} bad", *bad[:10], sep="\n")
    if bad:
        sys.exit("bad core images: stop")


def tuep_scan(processes, per_recording=24):
    # TUEP's label is the patient's diagnosis, and most 20 s windows of an epileptic patient look normal, so the
    # patient-level system scans whole recordings with a TUEV spike detector (run.py --scan). Every TUEP test recording
    # of the core lists gives up to 24 windows spread evenly through it (its first and last window left out), each
    # labelled with the patient's diagnosis; rendered in the core design into images-core/TUEP/scan/
    import mne
    from render import WINDOW

    recordings = {r["scan"]: r["edf"] for r in csv.DictReader((CORE_DATA / "TUEP-windows.csv").open()) if r["role"] == "test"}
    rows = []
    for scan, edf in sorted(recordings.items()):
        header = mne.io.read_raw_edf(str(ROOT / edf), preload=False, verbose="ERROR")
        n = max(1, int((header.n_times / header.info["sfreq"]) // WINDOW))
        if n <= per_recording + 2:
            indices = list(range(n))
        else:
            indices = sorted({round(1 + k * (n - 3) / (per_recording - 1)) for k in range(per_recording)})
        rows += [(f"scan/{scan}_{i}.png", scan, i, edf, "00_epilepsy" in edf) for i in indices]

    table = CORE_DATA / "TUEP-scan-windows.csv"
    with table.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "role", "scan", "index", "edf"])
        writer.writerows([path, "scan", scan, i, edf] for path, scan, i, edf, _ in rows)
    with (CORE_DATA / "TUEP-scan.jsonl").open("w") as f:
        for path, scan, _, _, epilepsy in rows:
            f.write(json.dumps({"images": [path], "patient": scan.rsplit("_", 1)[0], "epilepsy": epilepsy}) + "\n")
    patients = {scan.rsplit("_", 1)[0] for _, scan, *_ in rows}
    print(f"TUEP scan: {len(rows)} windows from {len(recordings)} recordings, {len(patients)} patients "
          f"({sum(r[4] for r in rows)} windows epilepsy) -> {table}", flush=True)
    render_all(processes, out=CORE_IMAGES / "TUEP", table=table, style="core")


def heldout(processes):
    # the final TUSZ test: every shipped window of the 62 patients in no core TUSZ split (the sampler held them out as
    # other corpora's test patients, so no model trained on them). Scored once per finalist (run.py --heldout)
    import mne
    from render import WINDOW

    names = scans()
    used = {r["scan"].rsplit("_", 1)[0] for r in csv.DictReader((CORE_DATA / "TUSZ-windows.csv").open())}
    trained = set()
    for table in CORE_DATA.glob("*-windows.csv"):
        trained |= {SUBJECT.search(r["edf"]).group(0) for r in csv.DictReader(table.open())
                    if r["role"] in ("train", "val") and SUBJECT.search(r["edf"])}
    shipped = {r["path"]: {c for c in LABELS if r[c] == "true"} for r in csv.DictReader((DATASET / "labels.csv").open())}
    by_recording = defaultdict(list)
    for path in shipped:
        _, scan, index = NAME.match(path).groups()
        if scan.rsplit("_", 1)[0] not in used:
            by_recording[scan].append((int(index), path))
    patients = {scan.rsplit("_", 1)[0] for scan in by_recording}
    if patients & trained:
        sys.exit(f"{len(patients & trained)} held-out patients are in a core train/val split, e.g. "
                 f"{sorted(patients & trained)[:3]}: stop")

    rows, mismatches = [], 0
    for scan in sorted(by_recording):
        edf = names[scan]
        header = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
        found = window_labels(edf, max(1, int((header.n_times / header.info["sfreq"]) // WINDOW)))
        for index, path in sorted(by_recording[scan]):
            mismatches += shipped[path] != found[index]
            rows.append((f"heldout/{scan}_{index}.png", scan, index, str(edf.relative_to(ROOT)), bool(shipped[path] & SEIZURES)))
    print(f"label check: {mismatches} held-out windows whose labels differ from the annotations")
    if mismatches:
        sys.exit("labels do not reproduce: stop")

    table = CORE_DATA / "TUSZ-heldout-windows.csv"
    with table.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "role", "scan", "index", "edf"])
        writer.writerows([path, "heldout", scan, index, edf] for path, scan, index, edf, _ in rows)
    text = question_text("TUSZ", config_intro())
    with (CORE_DATA / "TUSZ-heldout.jsonl").open("w") as f:
        for path, *_, present in rows:
            f.write(json.dumps({"dataset": "TUSZ", "question": "any_seizure", "instruction": text, "input": "",
                                "output": json.dumps({"present": present}), "images": [path]}) + "\n")
    print(f"held-out: {len(rows)} windows ({sum(r[4] for r in rows)} seizure) from {len(by_recording)} recordings, "
          f"{len(patients)} patients, none in any core train/val split -> {table}", flush=True)
    render_all(processes, out=CORE_IMAGES / "TUSZ", table=table, style="core")


def tuep_random():
    # the EMBC 2026 TUEP setup's split (2026-10-22): whole recordings assigned to train / val / test at random, so a
    # patient can sit on both sides, against our patient-disjoint split. Same windows, images and split sizes; only the
    # split rule differs. Writes data/core/TUEP-random-{train,val,test}.jsonl (seed 0)
    lines = {split: read_jsonl(CORE_DATA / f"TUEP-{split}.jsonl") for split in ("train", "val", "test")}
    sizes = {split: len(rows) for split, rows in lines.items()}
    core_name = re.compile(r"^(train|val|test)/(.+_\d+)_(\d+)\.png$")
    by_recording = defaultdict(list)
    for rows in lines.values():
        for r in rows:
            by_recording[core_name.match(r["images"][0]).group(2)].append(r)
    recordings = sorted(by_recording)
    random.Random(SEED).shuffle(recordings)
    out = {"test": [], "val": [], "train": []}
    for scan in recordings:
        split = "test" if len(out["test"]) < sizes["test"] else "val" if len(out["val"]) < sizes["val"] else "train"
        out[split] += by_recording[scan]
    patients = {split: {core_name.match(r["images"][0]).group(2).rsplit("_", 1)[0] for r in rows} for split, rows in out.items()}
    for split, rows in out.items():
        with (CORE_DATA / f"TUEP-random-{split}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        positives = sum(json.loads(r["output"])["present"] for r in rows)
        print(f"{split}: {len(rows)} windows ({positives} epilepsy), {len(patients[split])} patients")
    shared = patients["test"] & (patients["train"] | patients["val"])
    print(f"test patients also in train/val: {len(shared)} of {len(patients['test'])}")


def extra(ds):
    # each class of EXTRA[ds] as its own question: the core windows and images of the corpus, labelled for that class
    # from labels.csv, each training split balanced again within missing-row patterns for its own label
    splits, tables = core_splits()
    labels = {r["path"]: r for r in csv.DictReader((ROOT / "datasets" / ds / "labels.csv").open())}
    prompt = config_intro()
    for cls, what in EXTRA[ds].items():
        text = f"{prompt}Based only on these waveforms, does this window show {what}? Set present to true only when it does."
        for split, rows in splits[ds].items():
            rows = [(path, labels[path][cls] == "true") for path, _ in rows]
            if split == "train":
                rows, report = balance(rows, tables[ds])
                print(f"{ds} {cls} train: missing-row balance, " + ("; ".join(report) or "nothing missing"))
            with (CORE_DATA / f"{ds}-{cls}-{split}.jsonl").open("w") as f:
                for path, present in rows:
                    f.write(json.dumps({"dataset": ds, "question": cls, "instruction": text, "input": "",
                                        "output": json.dumps({"present": present}),
                                        "images": [f"{split}/{path.split('/', 1)[1]}"]}) + "\n")
            print(f"{ds} {cls} {split}: {len(rows)} ({sum(x for _, x in rows)} true)")


def tusz_types():
    # each seizure type with training patients as its own question (maintainer, 2026-10-25: the paper detects the
    # annotated events). Train: every window of the type in the core TUSZ training windows, plus as many other windows
    # (background and other types, seed 0), then the missing-row balance; val and test: the core windows, labelled
    # for the type from the annotations. tnsz has no training window, spsz and mysz one patient and no test window
    splits, tables = core_splits()
    prompt = config_intro()
    found = {}
    for split in ("train", "val", "test"):
        by_edf = defaultdict(list)
        for path, _ in splits["TUSZ"][split]:
            by_edf[tables["TUSZ"][path]["edf"]].append((path, int(tables["TUSZ"][path]["index"])))
        for edf, items in by_edf.items():
            labels = window_labels(ROOT / edf, max(i for _, i in items) + 1)
            found.update({path: labels[i] for path, i in items})
    rng = random.Random(SEED)
    for cls, what in TUSZ_TYPES.items():
        text = f"{prompt}Based only on these waveforms, does this window show {what}? Set present to true only when it does."
        for split in ("train", "val", "test"):
            rows = [(path, cls in found[path]) for path, _ in splits["TUSZ"][split]]
            if split == "train":
                positive = [r for r in rows if r[1]]
                negative = rng.sample([r for r in rows if not r[1]], len(positive))
                rows, report = balance(sorted(positive + negative), tables["TUSZ"])
                print(f"TUSZ {cls} train: missing-row balance, " + ("; ".join(report) or "nothing missing"))
            with (CORE_DATA / f"TUSZ-{cls}-{split}.jsonl").open("w") as f:
                for path, present in rows:
                    f.write(json.dumps({"dataset": "TUSZ", "question": cls, "instruction": text, "input": "",
                                        "output": json.dumps({"present": present}),
                                        "images": [f"{split}/{path.split('/', 1)[1]}"]}) + "\n")
            print(f"TUSZ {cls} {split}: {len(rows)} ({sum(x for _, x in rows)} true)")


def dataset_all():
    # one model per dataset (maintainer, 2026-10-25: "three separate prompts, but all to train the same model"): each
    # dataset's question lines merged into <DS>-all-{train,val,test}.jsonl, one yes/no line per window and question as
    # each question's own lines have it. TUSZ windows whose only labels are cut seizure types answer no to both questions
    # but have a teacher text describing a seizure, so they are marked "explain": false and get no explain line
    # (maintainer: "cut them")
    sources = {"TUSZ": {"fnsz": "TUSZ-fnsz", "gnsz": "TUSZ-gnsz"},
               "TUAR": {"eyem": "TUAR", "musc": "TUAR-musc", "elec": "TUAR-elec"},
               "TUEV": {"spsw": "TUEV-spsw", "gped": "TUEV-gped", "pled": "TUEV-pled"}, "TUAB": {"abnormal": "TUAB"}}
    splits, tables = core_splits()
    found = {}
    for path, _ in splits["TUSZ"]["train"]:
        found.setdefault(tables["TUSZ"][path]["edf"], []).append((path, int(tables["TUSZ"][path]["index"])))
    seizure_only = set()
    for edf, items in found.items():
        labels = window_labels(ROOT / edf, max(i for _, i in items) + 1)
        for path, i in items:
            if labels[i] & (SEIZURES - {"fnsz", "gnsz"}) and not labels[i] & {"fnsz", "gnsz"}:
                seizure_only.add(f"train/{path.split('/', 1)[1]}")
    for ds, questions in sources.items():
        for split in ("train", "val", "test"):
            rows = []
            for question, stem in questions.items():
                for r in read_jsonl(CORE_DATA / f"{stem}-{split}.jsonl"):
                    assert r["question"] == question, (stem, r["question"])
                    no_explain = ds == "TUSZ" and r["images"][0] in seizure_only  # names repeat across corpora
                    rows.append({**r, **({"explain": False} if no_explain else {})})
            with (CORE_DATA / f"{ds}-all-{split}.jsonl").open("w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            marked = len({r["images"][0] for r in rows if r.get("explain") is False})
            print(f"{ds}-all {split}: {len(rows)} lines, {len({r['images'][0] for r in rows})} windows"
                  + (f", {marked} windows without an explain line" if marked else ""))


def question_of(ds):
    return "any_seizure" if ds == "TUSZ" else QUESTIONS[ds][0]


def question_text(ds, prompt):
    what = "a seizure of any type" if ds == "TUSZ" else QUESTIONS[ds][2]
    return f"{prompt}Based only on these waveforms, does this window show {what}? Set present to true only when it does."


def render_all(processes, out=IMAGES, only=None, table=DATA / "windows.csv", style="current"):
    by_edf = defaultdict(list)
    for row in csv.DictReader(table.open()):
        if only is None or row["path"] in only:
            by_edf[row["edf"]].append((row["path"], row["index"]))
    jobs = [(edf, items, out, style) for edf, items in sorted(by_edf.items(), key=lambda kv: -len(kv[1]))]
    started, done, drawn, errors = time.time(), 0, 0, []
    with Pool(processes) as pool:
        for k, (n, new, error) in enumerate(pool.imap_unordered(render_recording, jobs), 1):
            done, drawn = done + n, drawn + new
            if error:
                errors.append(error)
            if k % 100 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} recordings, {done} windows ({drawn} drawn now), "
                      f"{(time.time() - started) / 60:.1f} min", flush=True)
    for error in errors:
        print("FAILED", error)
    print(f"{drawn} drawn, {len(errors)} recordings failed -> {out}")


def main():
    if sys.argv[1] == "lists":
        lists()
    elif sys.argv[1] == "render":
        render_all(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif sys.argv[1] == "background":
        background(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif sys.argv[1] == "datasets":
        datasets(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif sys.argv[1] == "core":
        core(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif sys.argv[1] == "check-core":
        check_core()
    elif sys.argv[1] == "tuep-scan":
        tuep_scan(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif sys.argv[1] == "tuep-random":
        tuep_random()
    elif sys.argv[1] == "extra":
        extra(sys.argv[2])
    elif sys.argv[1] == "tusz-types":
        tusz_types()
    elif sys.argv[1] == "dataset-all":
        dataset_all()
    elif sys.argv[1] == "heldout":
        heldout(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif sys.argv[1] == "sample":
        # 2 seizure and 2 background training windows, beside their shipped PNGs, to look at
        shipped = {r["path"]: any(r[c] == "true" for c in SEIZURES) for r in csv.DictReader((DATASET / "labels.csv").open())}
        train = [r["path"] for r in csv.DictReader((DATA / "windows.csv").open()) if r["role"] == "train"]
        sample = [p for p in train if shipped[p]][:2] + [p for p in train if not shipped[p]][:2]
        render_all(4, SIDE / "images-sample", only=set(sample))
        print("\n".join(f"{p} seizure={shipped[p]}" for p in sample))


if __name__ == "__main__":
    main()
