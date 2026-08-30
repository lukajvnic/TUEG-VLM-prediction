import csv
import math
import random
import re
import sys
from collections import defaultdict
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

FIG_SIZE, DPI = 16, 96
BANDPASS = (0.5, 70.0)
NOTCH_HZ = 60.0
NON_SIGNAL = ("IBI", "BURST", "SUPPR", "PHOTIC")
SCALE_PCT, SCALE_PAD = 99, 1.15
LINEWIDTH = 0.4
LABEL_PT = 8
WINDOW = 20.0
TRAIN_WINDOWS = 4
TEST_WINDOWS = 10
TEST_FRAC = 0.3
SEED = 0
RATIONALE = "ground_truth_rationale"

SUBJECT = re.compile(r"aaaa[a-z]{4}")


def patient_of(edf):
    match = SUBJECT.search(edf.name)
    return match.group(0) if match else edf.parent.name


def scan_names(edfs):
    counters, names = defaultdict(int), {}
    for edf in sorted(edfs, key=str):
        patient = patient_of(edf)
        names[edf] = f"{patient}_{counters[patient]}"
        counters[patient] += 1
    return names


def assign_splits(edfs):
    by_patient = defaultdict(list)
    for edf in edfs:
        by_patient[patient_of(edf)].append(edf)
    patients = sorted(by_patient)
    random.Random(SEED).shuffle(patients)
    target = round(len(edfs) * TEST_FRAC)
    test, count = set(), 0
    for patient in patients:
        if count >= target:
            break
        test.add(patient)
        count += len(by_patient[patient])
    return {edf: ("test" if patient_of(edf) in test else "train") for edf in edfs}


def csv_labels(files, start, stop, vocab):
    vocab, found = set(vocab), set()
    for path in files:
        if not path.exists():
            continue
        lines = [x for x in path.read_text(errors="ignore").splitlines() if x and not x.startswith("#")]
        for row in csv.DictReader(StringIO("\n".join(lines))):
            if float(row["start_time"]) < stop and float(row["stop_time"]) > start:
                found |= set(row.get("label", "").split("_")) & vocab
    return found


def signal_picks(raw):
    return [i for i, ch in enumerate(raw.ch_names) if not any(tag in ch.upper() for tag in NON_SIGNAL)]


def read(edf):
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")  # lazy loading edge-artifacts, mne#10635
    picks = signal_picks(raw)
    if picks:
        raw.filter(*BANDPASS, picks=picks, verbose="ERROR")
        if NOTCH_HZ < raw.info["sfreq"] / 2:
            raw.notch_filter(NOTCH_HZ, picks=picks, verbose="ERROR")
    return raw


def window_bounds(raw):
    duration = raw.n_times / raw.info["sfreq"]
    count = max(1, int(math.floor(duration / WINDOW)))
    return [(i * WINDOW, (i + 1) * WINDOW) for i in range(count)]


def pick_train(n, event_flags, rng):
    events = [i for i in range(n) if event_flags[i]]
    background = [i for i in range(n) if not event_flags[i]]
    n_events = min(TRAIN_WINDOWS // 2, len(events))
    n_background = min(TRAIN_WINDOWS - n_events, len(background))
    n_events = min(len(events), TRAIN_WINDOWS - n_background)
    rng.shuffle(events)
    rng.shuffle(background)
    return sorted(events[:n_events] + background[:n_background])


def pick_test(n, event_flags):
    events = {i for i in range(n) if event_flags[i]}
    if TEST_WINDOWS >= n:
        spaced = set(range(n))
    else:
        spaced = {round(j * (n - 1) / (TEST_WINDOWS - 1)) for j in range(TEST_WINDOWS)}
    return sorted(events | spaced)


def render_window(raw, png, start):
    sfreq = raw.info["sfreq"]
    data, _ = raw[:, int(round(start * sfreq)):int(round((start + WINDOW) * sfreq))]
    seconds = np.arange(data.shape[1]) / sfreq
    fig, axes = plt.subplots(len(data), 1, figsize=(FIG_SIZE, FIG_SIZE), dpi=DPI, sharex=True)
    axes = [axes] if len(data) == 1 else axes
    for ax, name, y in zip(axes, raw.ch_names, data):
        ax.plot(seconds, y, linewidth=LINEWIDTH, color="black")
        scale = float(np.percentile(np.abs(y), SCALE_PCT)) * SCALE_PAD
        if scale > 0:
            ax.set_ylim(-scale, scale)
        ax.set_ylabel(name, fontsize=LABEL_PT, weight="bold", rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.set_xlim(0, WINDOW)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=DPI)
    plt.close(fig)


def process(edf, scan, split, root, label_fn):
    raw = read(edf)
    bounds = window_bounds(raw)
    windows = [label_fn(start, stop) for start, stop in bounds]
    flags = [event for _, event in windows]
    if split == "test":
        indices = pick_test(len(bounds), flags)
    else:
        indices = pick_train(len(bounds), flags, random.Random(f"{SEED}:{scan}"))  # seeded per recording
    for i in indices:
        png = root / split / f"{scan}_{i}.png"
        if not png.exists():
            render_window(raw, png, bounds[i][0])
        yield f"{split}/{png.name}", windows[i][0]


def write_labels(dataset, labels, rows):
    out = dataset / "labels.csv"
    old = {}
    if out.exists():
        with out.open(newline="") as f:
            old = {r["path"]: r.get(RATIONALE, "") for r in csv.DictReader(f) if "path" in r}
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["path", *labels, RATIONALE])
        writer.writeheader()
        for path in sorted(rows):
            writer.writerow({"path": path, **rows[path], RATIONALE: old.get(path, "")})


def generate(dataset, edfs, labels, labels_for, events):
    events = set(events)
    splits, names = assign_splits(edfs), scan_names(edfs)
    rows = {}
    for i, edf in enumerate(edfs, 1):
        def label_fn(start, stop, edf=edf):
            labs = labels_for(edf, start, stop)
            return {x: str(x in labs).lower() for x in labels}, bool(labs & events)
        try:
            for path, label in process(edf, names[edf], splits[edf], dataset, label_fn):
                rows[path] = label
            print(f"{i}/{len(edfs)} [{splits[edf]}] {names[edf]}")
        except Exception as e:
            print(f"failed {edf}: {e}", file=sys.stderr)
    write_labels(dataset, labels, rows)
