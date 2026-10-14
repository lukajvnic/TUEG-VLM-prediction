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
# background: the scored TUSZ test set's background is 97.8% first/last windows of its recording (known-issues,
# 2026-10-14). This lists, from the same 200 test patients, shipped test windows never scored that are background
# only and not a recording's first or last window, up to 2 per recording spread through it (seed 0), with each
# window's distance in windows to the nearest seizure window of its recording (from the annotations)
# datasets (2026-10-14): does the recipe transfer? One yes/no question per other corpus (QUESTIONS), on the
# fine-tune's vetted train/val windows (sft_labels_*.jsonl: patient caps, cross-corpus image screen) and the
# benchmark's scored test windows. Labels come from labels.csv and must reproduce from the source annotations.
# Writes data/<DS>-{train,val,test}.jsonl and data/<DS>-windows.csv, renders into images/<DS>/ (window names repeat
# across corpora)
import csv
import json
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
                      "an epileptiform discharge (a spike or sharp wave, GPED or PLED)")}


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


def render_recording(job):
    from render import read, window_bounds

    edf, items, out = job
    todo = [(path, int(index)) for path, index in items if not (out / path).exists()]
    if not todo:
        return len(items), 0, None
    try:
        raw = read(ROOT / edf)
        sfreq = raw.info["sfreq"]
        rows, names = bipolar(raw)
        windows = window_bounds(raw)
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


def render_all(processes, out=IMAGES, only=None, table=DATA / "windows.csv"):
    by_edf = defaultdict(list)
    for row in csv.DictReader(table.open()):
        if only is None or row["path"] in only:
            by_edf[row["edf"]].append((row["path"], row["index"]))
    jobs = [(edf, items, out) for edf, items in sorted(by_edf.items(), key=lambda kv: -len(kv[1]))]
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
    elif sys.argv[1] == "sample":
        # 2 seizure and 2 background training windows, beside their shipped PNGs, to look at
        shipped = {r["path"]: any(r[c] == "true" for c in SEIZURES) for r in csv.DictReader((DATASET / "labels.csv").open())}
        train = [r["path"] for r in csv.DictReader((DATA / "windows.csv").open()) if r["role"] == "train"]
        sample = [p for p in train if shipped[p]][:2] + [p for p in train if not shipped[p]][:2]
        render_all(4, SIDE / "images-sample", only=set(sample))
        print("\n".join(f"{p} seizure={shipped[p]}" for p in sample))


if __name__ == "__main__":
    main()
