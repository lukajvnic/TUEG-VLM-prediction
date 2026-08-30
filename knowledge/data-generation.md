# Data generation

Turns raw EDF recordings into windowed waveform PNGs + a `labels.csv` manifest,
split patient-wise into `train/` and `test/`.

## Files

- `datasets/<DS>/generate.py` — one per dataset; dataset-specific label logic.
- `datasets/render.py` — **shared** windowing + rendering engine (imported by all
  six via `sys.path` insertion). Single source of truth so images stay identical
  across datasets. Was `datasets/_render.py`, which was gitignored and got lost;
  see "Reconstruction" below.
- `datasets/verify_render.py` — checks `render.py` against the shipped corpus.
  Run it before trusting `render.py` to generate anything.
- `generate-all.sh` — runs all six `generate.py` in parallel with `--overwrite`.

## Reconstruction (17 August 2026)

The original `datasets/_render.py` was gitignored (`.gitignore` listed it
explicitly) and so was never committed. It was not on disk, in HEAD, in either
stash, or on any branch, and nothing had absorbed it. All six `generate.py`
still imported it, so dataset generation was broken.

`datasets/render.py` is a rewrite from this document, verified against the
corpus by `datasets/verify_render.py`:

- **Selection is exact.** The chosen window indices match the shipped
  `labels.csv` for 240/240 sampled recordings, 40 per dataset, both splits.
  This is what found the two spec errors noted above.
- **Rendering is structurally equal, not byte-identical.** Same 1536x1536
  canvas, same ink fraction to within about 0.5%, layout box within 3 px. About
  8-13% of pixels differ, because MNE's filter output and matplotlib's
  rasteriser both moved between versions. `linewidth=0.4` was fixed by matching
  ink fraction against the shipped PNGs.

So it reproduces the method, not the exact bytes. **Do not mix newly rendered
images into the existing corpus**, and re-run `verify_render.py` before using
it to regenerate anything.

## The shared engine: `datasets/render.py`

Key constants:
```python
FIG_W, FIG_H, DPI = 16, 16, 96      # -> every image is 1536 x 1536 px
BANDPASS = (0.5, 70.0)              # display bandpass (Hz)
NOTCH_HZ = 60.0                    # US mains hum
NON_SIGNAL = ("IBI","BURST","SUPPR","PHOTIC")  # derived channels, not filtered
```

Functions:
- **`read(edf)`** — `read_raw_edf(preload=True)` (mixed sampling rates + lazy load
  cause edge artifacts, MNE #10635), then bandpass + notch filter on real signal
  channels only (derived/marker channels are skipped to avoid ringing).
- **`window_bounds(raw, window)`** — non-overlapping `window`-second windows
  (`floor(duration/window)`, at least 1).
- **`pick_train(n, event_flags, k, rng)`** — a **half-and-half** sample of `k`
  windows: `k // 2` event windows and the rest background, with one pool
  covering for the other when it is short. The events are shuffled first, then
  the background, both off the same generator, so that ordering is part of the
  contract. `rng` is `random.Random(f"{seed}:{scan_name}")`, seeded per
  recording so a recording's sample does not depend on how many were processed
  before it.
- **`pick_test(n, event_flags, cap)`** — `cap` evenly-spaced windows over the
  **whole recording**, `round(j * (n-1) / (cap-1))`, **∪ every event window**.

  Both of these were documented wrongly here until 17 August 2026, and both
  were corrected by testing against the shipped `labels.csv` rather than by
  reading code (the code was gone). `pick_train` was described as
  "prefer event windows, fill with background", which is not what the corpus
  shows: `aaaaasoc_0` has 27 event windows and 4 train slots, and only 2 events
  were taken. `pick_test`'s spacing was described as running over the
  background windows, but it runs over all `n`, which is why an event-heavy
  recording keeps its even coverage instead of having it squeezed.
- **`render_window(raw, png, start, window)`** — draws each channel in its own
  subplot row (never overlapping):
  - all channels, no cap
  - per-channel scale `= percentile(|y|, 99) * 1.15` with `ylim` clipping —
    fills the row with readable EEG, only rare extreme peaks clip at the edge
  - channel labels `fontsize=8, weight="bold"` (VLM-readable; verified ~26/30
    read back correctly)
  - x-axis rebased to 0 (`seconds`)
- **`process(edf, scan_name, split, root, label_fn, window, train_windows,
  test_windows, seed, overwrite)`** — loads the recording once, computes each
  window's `(label_dict, is_event)` via `label_fn`, chooses windows
  (`pick_test` for test, `pick_train` for train), renders them, and yields
  `(image_path, label_dict)` rows.

## Per-dataset `generate.py`

Each provides:
- **`patient_of(edf)`** — `aaaa[a-z]{4}` subject token from the filename, else the
  parent dir name (handles TUEV's numeric eval dirs).
- **`scan_names(edfs)`** — `<patient>_<index>.png` base names (index by sorted
  scan order, stable across runs). Label-free and dataset-free by design.
- **`assign_splits(edfs, test_frac, seed)`** — group by patient, shuffle patients
  (seeded), fill `test` until `~test_frac` of **scans**, rest `train`. Whole
  patients never span splits.
- **`label_fn(start, stop)`** — dataset-specific. Multi-label datasets call
  `labels_for(edf, start, stop)` and flag `is_event = bool(labs - {"bckg"})`.
  Binary datasets return the recording's global label with `is_event=False`.

Final image name is `<patient>_<scan>_<window>.png`, e.g. `aaaaaajy_0_8.png`
(patient `aaaaaajy`, scan 0, window 8). Recording = `<patient>_<scan>`.

## `labels.csv` format

One row per rendered window:
```
image_path,split,<label columns...>
train/aaaaaajy_0_8.png,train,false,true,...
test/aaaaaajy_0_10.png,test,...
```
- `image_path` is relative to the dataset dir and encodes the split (`train/`…).
- `split` column is explicit (`train`/`test`).
- Written in `"w"` mode → **truncated and rewritten fresh every run** (streamed as
  rendering proceeds; a crashed run leaves a partial file, a full run leaves a
  complete one).

## CLI args (per `generate.py`)

Common: `--overwrite`, `--limit`, `--seed` (0), `--test-frac` (0.3),
`--window` (20), `--train-windows` (4), `--test-windows` (10; `-1` = all).
TUAB also has `--train-root`, `--shuffle`; TUEP has `--dataset-root`.
`generate-all.sh` always passes `--overwrite` and forwards extra flags (but not
`--shuffle`, which only TUAB accepts).

## Rendering history / gotchas (so they aren't re-discovered)

- Images are **waveform plots**, never spectrograms (the prompts once wrongly
  said "spectrogram").
- The **EDF filename must never be rendered into the image** (old code set the
  plot title/suptitle to it → the model could OCR the class and cheat). Fixed.
- **Scaling metric matters:** `median`-based scaling clips (EEG has ~12x crest
  factor); `8x median` buried EEG at ~12% of the row (looked flat). `p99*1.15`
  is the fix.
- **Font, not resolution, drives label legibility.** 1536 vs 2048 gave identical
  read-back; `fontsize 5 -> 8 bold` fixed it. `tight_layout` reserves left margin,
  so labels never bleed off-frame even for long names (`EEG RESP1-REF`).
- **All channels are kept** (30–41 per recording, incl. EKG and derived
  IBI/BURSTS/SUPPR). Per-channel scaling is required because these have wildly
  different amplitudes.
