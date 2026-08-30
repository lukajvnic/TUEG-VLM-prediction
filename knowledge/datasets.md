# Datasets

Six sub-corpora of the Temple University EEG corpus (TUEG). Each lives under
`datasets/<NAME>/` with the raw EDFs in a versioned subdir and a `generate.py`
that renders them. Raw EDFs are **not** in git (too large); they live on the
cluster / locally.

## Source layout per dataset

| Dataset | Source dir | EDF path pattern | Label level |
|---|---|---|---|
| TUAB | `v3.0.1/edf/train/{normal,abnormal}/01_tcp_ar/` | `<subj>_s<sess>_t<tok>.edf` | recording |
| TUAR | `v3.0.1/edf/<montage>/` | `<subj>_s<sess>_t<tok>.edf` | window |
| TUEP | `v3.1.0/{00_epilepsy,01_no_epilepsy}/<subj>/s<sess>/01_tcp_ar/` | `<subj>_s<sess>_t<tok>.edf` | patient |
| TUEV | `v2.0.1/edf/{train/<subj>, eval/<NNN>}/` | `<subj>_00000001.edf` **or** `<label>_<NNN>_a_.edf` | window |
| TUSL | `v2.0.1/edf/<subj>/s<sess>/01_tcp_ar/` | `<subj>_s<sess>_t<tok>.edf` | window |
| TUSZ | `v2.0.6/edf/train/<subj>/s<sess>/<montage>/` | `<subj>_s<sess>_t<tok>.edf` | window |

- **Subject/patient id** is the anonymised `aaaaXXXX` token (8 lowercase letters).
- **TUEV is the odd one out:** its `train/` files are subject-named, but its
  `eval/` files are named `pled_024_a_.edf` etc. — the *label* is in the filename
  and the patient is the numeric parent dir (`024`). This is why patient
  extraction falls back to the parent directory when no `aaaaXXXX` token exists.

## Labels

| Dataset | Classes | Annotation source |
|---|---|---|
| TUAB | `abnormal`, `normal` | directory (normal/ vs abnormal/) |
| TUEP | `epilepsy`, `no_epilepsy` | directory (00_epilepsy/ vs 01_no_epilepsy/) |
| TUEV | `bckg, spsw, gped, pled, eyem, artf` | `.rec` files (numeric codes via `REC_LABELS`) |
| TUAR | `chew, elec, elpp, eyem, musc, shiv` + `seiz` + `cpsz, fnsz, gnsz, tcsz` | `<edf>.csv` (artifacts) + `<edf>_seiz.csv` (seizures) |
| TUSL | `bckg, seiz, slow` | `<edf>.csv` |
| TUSZ | `bckg, absz, cpsz, fnsz, gnsz, mysz, spsz, tcsz, tnsz` | `<edf>.csv` |

- **Binary datasets (TUAB, TUEP)** are labelled at the recording/patient level:
  every window inherits the whole-recording label (weak supervision).
- **Multi-label datasets (TUEV, TUAR, TUSZ, TUSL)** have **time-localized**
  annotations: a window's labels come from `labels_for(edf, start, stop)`, which
  reads the annotation file and keeps labels overlapping `[start, stop]`.
- `bckg` = background (no event). A window is an "event window" if it has any
  non-`bckg` label (`labs - {"bckg"}`). TUAR has no `bckg` label, so any artifact
  makes it an event window.

## Annotation coverage (why the `assessed` column exists)

Measured, because it decides whether an unannotated window is a usable negative
or ungradeable. Coverage = union of annotated intervals / recording duration.

| Dataset | Median coverage | `bckg` annotated explicitly? | Unannotated window means |
|---|---|---|---|
| TUSZ | 0.3% | no — see below | **background** (usable negative) |
| TUEV | 1.4% | yes (code 6, 73,372 rows) | outside the annotated excerpt — **unknown** |
| TUSL | 3.3% | yes (4,350 rows) | outside the annotated excerpt — **unknown** |
| TUAR | 29% | yes (3,122 rows, dropped by generate.py) | **unknown** |

**TUSZ is the exception, and it is safe.** Its annotations mark seizures only:
of 8,140 `.csv_bi` files, 7,043 are `bckg`-only and 1,097 are `seiz`-only —
*no file contains both*. The lone `bckg` row in a seizure-free recording is a
token marker spanning a median 0.2% of the duration, not a time-localised
annotation. And the seizure annotations are provably complete: the local copy
reproduces the official `AAREADME.txt` statistics exactly (train/dev/eval seizure
events 1812/492/360 and total seizure seconds 119425.2/26775.2/23912.1). So
absence of annotation genuinely is background, and TUSZ's 17,742 unannotated test
windows were relabelled `bckg` rather than dropped.

For TUEV/TUSL/TUAR, `bckg` *is* annotated where the annotators looked, so absence
means they did not look — those windows are marked `assessed=false` and excluded.

Every recording in all four datasets has its annotation file present (0 missing
of 518/112/310/8,140), so there is no third "file missing" case to handle.

## Rare-class scarcity (important, and real)

Several classes have too few **source recordings** to evaluate reliably. Counts
of recordings containing each class:

| Class | Dataset | Source recordings |
|---|---|---|
| `mysz` | TUSZ | 2 |
| `spsz` | TUSZ | 4 |
| `elpp` | TUAR | 4 |
| `tnsz` | TUSZ | 10 |
| `absz` | TUSZ | 19 |
| `fnsz` | TUSZ | 655 (for contrast) |

With 2–10 recordings, a patient-level split gives 2/0 or 1/1 — no reliable
metric is possible, and the class often lands entirely in one split. This is a
**data-availability limit, not a bug**. Handled by a support threshold at scoring
time (see [known-issues.md](known-issues.md) and `eval/summarize.py`).

## Approximate sizes (after windowing)

Rough post-generation row counts (varies with `--train-windows`/`--test-windows`):
TUSZ is by far the largest (~46k images total), then TUEP (~15k), TUAB (~5.8k),
TUAR (~4.8k), TUEV (~3.2k), TUSL (~0.7k). Total ~76k images at defaults.

## What the test split actually is

Cost scales with windows; precision scales with the scoring unit. The gap is
large, and it is why the test set is sampled (see
[methodology-decisions.md](methodology-decisions.md)).

| Dataset | test windows | assessed | sampled | recordings | patients |
|---|---|---|---|---|---|
| TUSZ | 25,189 | 25,189 | 8,460 | 2,443 | 200 |
| TUEP | 8,610 | 8,610 | 1,084 | 285 | **113** |
| TUAR | 3,915 | 3,664 | 3,655 | **94** | 67 |
| TUAB | 3,000 | 3,000 | 1,200 | 300 | 222 |
| TUEV | 1,729 | **354** | 354 | 151 | 115 |
| TUSL | 400 | **99** | 97 | 34 | **8** |
| total | 42,843 | 41,316 | **14,850** | | |

TUSL has 8 independent patients and one class above `MIN_SUPPORT`; `summarize.py`
flags any dataset under 20 patients as descriptive-only rather than inferential.
