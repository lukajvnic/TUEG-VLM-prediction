# human-facing copies of the core images with every annotated event marked (2026-10-23): the same 25-row render, the
# annotated rows shaded for the seconds the event covers (TUSZ, TUAR, TUSL annotate per TCP channel; TUEV numbers the
# same channels), a title strip with the window's question and label, and a legend of the events present. TUAB and
# TUEP carry one label per recording, so their images get the title only. Written to images-annotated/<DS>/<split>/
# under the core image's name; the model never sees these
#   python side/vision-freeze/annotate.py sample        # 5 windows (TUSZ, TUAR, TUEV, TUSL, TUAB) into images-annotated-sample/
#   python side/vision-freeze/annotate.py all [N]       # every core window, N processes
import csv
import json
import os
import sys
from collections import defaultdict
from io import StringIO
from multiprocessing import Pool
from pathlib import Path

import rerender
from rerender import CORE_DATA, ROOT, SIDE, core_rows, background_level, draw_core, read_jsonl

OUT = SIDE / "images-annotated"
SAMPLE = SIDE / "images-annotated-sample"
NAMES = {"absz": "absence seizure", "cpsz": "complex partial seizure", "fnsz": "focal non-specific seizure",
         "gnsz": "generalized non-specific seizure", "mysz": "myoclonic seizure", "spsz": "simple partial seizure",
         "tcsz": "tonic-clonic seizure", "tnsz": "tonic seizure", "eyem": "eye movement", "musc": "muscle artifact",
         "chew": "chewing", "shiv": "shivering", "elpp": "electrode pop", "elec": "electrode artifact",
         "artf": "artifact", "spsw": "spike / sharp wave", "gped": "generalized periodic discharge",
         "pled": "lateralized periodic discharge", "seiz": "seizure", "slow": "slowing"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDER = list(NAMES)  # a label keeps its colour slot in every image of a corpus
TUEV_CHANNELS = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "A1-T3", "T3-C3", "C3-CZ",
                 "CZ-C4", "C4-T4", "T4-A2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]
TUEV_CODES = {"1": "spsw", "2": "gped", "3": "pled", "4": "eyem", "5": "artf"}  # 6 (bckg) is not drawn
QUESTION = {"TUSZ": "seizure", "TUAB": "abnormal EEG", "TUAR": "eye movement artifact", "TUEV": "epileptiform discharge",
            "TUEP": "epilepsy patient", "TUSL": "slowing"}
TABLES = [("TUSZ", "TUSZ-windows.csv"), ("TUSZ", "TUSZ-heldout-windows.csv"), ("TUAB", "TUAB-windows.csv"),
          ("TUAR", "TUAR-windows.csv"), ("TUEV", "TUEV-windows.csv"), ("TUEP", "TUEP-windows.csv"),
          ("TUEP", "TUEP-scan-windows.csv"), ("TUSL", "TUSL-windows.csv")]


def events(ds, edf):
    # (TCP channel, start s, stop s, label) for every annotated event of the recording
    edf = ROOT / edf
    if ds == "TUEV":
        rec = edf.with_suffix(".rec")
        out = []
        for line in rec.read_text(errors="ignore").splitlines() if rec.exists() else []:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 4 and parts[3] in TUEV_CODES:
                out.append((TUEV_CHANNELS[int(parts[0])], float(parts[1]), float(parts[2]), TUEV_CODES[parts[3]]))
        return out
    if ds in ("TUSZ", "TUAR", "TUSL"):
        path = edf.with_suffix(".csv")
        if not path.exists():
            return []
        rows = csv.DictReader(StringIO("\n".join(x for x in path.read_text(errors="ignore").splitlines() if x and not x.startswith("#"))))
        return [(r["channel"].upper(), float(r["start_time"]), float(r["stop_time"]), label)
                for r in rows for label in r["label"].split("_") if label in NAMES]
    return []  # TUAB, TUEP: one label per recording


def labels():
    # image path (core name) -> the window's answer, per corpus
    out = defaultdict(dict)
    for path in CORE_DATA.glob("*.jsonl"):
        ds = path.name.split("-")[0]
        for r in read_jsonl(path):
            if "output" in r:
                out[ds][r["images"][0]] = json.loads(r["output"])["present"]
            elif "epilepsy" in r:  # the TUEP scan lists
                out[ds][r["images"][0]] = r["epilepsy"]
    return out


def render_recording(job):
    from render import WINDOW, read, window_bounds

    ds, edf, items, out = job
    todo = [(path, int(index), present) for path, index, present in items if not (out / ds / path).exists()]
    if not todo:
        return len(items), 0, None
    try:
        raw = read(ROOT / edf)
        sfreq, windows = raw.info["sfreq"], window_bounds(raw)
        rows = core_rows(raw)
        eeg = [y for label, _, y in rows if y is not None and label != "EKG"]
        level = background_level(eeg, sfreq, windows) if eeg else 1.0
        found = events(ds, edf)
        colours = {}
        for path, index, present in todo:
            start = windows[index][0]
            spans, legend = [], []
            for channel, a, b, label in found:
                if b <= start or a >= start + WINDOW:
                    continue
                colour = colours.setdefault(label, COLORS[ORDER.index(label) % len(COLORS)])
                spans.append((channel, max(a, start) - start, min(b, start + WINDOW) - start, colour))
                if (NAMES[label], colour) not in legend:
                    legend.append((NAMES[label], colour))
            answer = "unknown" if present is None else "yes" if present else "no"
            scope = " (whole recording)" if ds in ("TUAB", "TUEP") else ""
            title = (f"{ds}  ·  {QUESTION[ds]}: {answer}{scope}  ·  {Path(path).stem}  ·  {start:.0f}-{start + WINDOW:.0f} s "
                     f"of the recording")
            draw_core(rows, level, sfreq, start, out / ds / path, spans, (title, legend))
        return len(items), len(todo), None
    except Exception as e:
        return len(items), 0, f"{edf}: {type(e).__name__}: {e}"


def jobs(out, only=None):
    answers = labels()
    by_edf = defaultdict(list)
    for ds, table in TABLES:
        for r in csv.DictReader((CORE_DATA / table).open()):
            if only is None or (ds, r["path"]) in only:
                by_edf[ds, r["edf"]].append((r["path"], r["index"], answers[ds].get(r["path"])))
    return [(ds, edf, items, out) for (ds, edf), items in sorted(by_edf.items(), key=lambda kv: -len(kv[1]))]


def run(todo, processes):
    done = drawn = 0
    errors = []
    with Pool(processes) as pool:
        for k, (n, new, error) in enumerate(pool.imap_unordered(render_recording, todo), 1):
            done, drawn = done + n, drawn + new
            if error:
                errors.append(error)
            if k % 200 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} recordings, {done} windows ({drawn} drawn now)", flush=True)
    print(*[f"FAILED {e}" for e in errors], f"{drawn} drawn, {len(errors)} recordings failed", sep="\n")


def sample():
    # one window per corpus that has something to show: an annotated event inside it, or (TUAB) an abnormal recording
    answers = labels()
    picks = set()
    for ds, table in TABLES:
        if ds in {d for d, _ in picks} or ds == "TUEP":
            continue
        for r in csv.DictReader((CORE_DATA / table).open()):
            if r["role"] != "test" or not answers[ds].get(r["path"]):
                continue
            if ds == "TUAB" or any(a < (int(r["index"]) + 1) * 20 and b > int(r["index"]) * 20 for _, a, b, _ in events(ds, r["edf"])):
                picks.add((ds, r["path"]))
                break
    run(jobs(SAMPLE, picks), 5)
    for ds, path in sorted(picks):
        print(SAMPLE / ds / path)


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample()
    else:
        run(jobs(OUT), int(sys.argv[2]) if len(sys.argv) > 2 else 12)
