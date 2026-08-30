# bckg annotation semantics

`bckg` is an explicit "reviewed, nothing found" assertion, distinct from unannotated time
("nobody looked"). The label exists because these corpora train/score event detectors:
the negative class must be explicit so detectors get negative examples and false alarms
can be scored over all time.

## TUSZ: absence IS an annotation

From the official NEDC annotation guidelines (Melles, Oymann, Obeid & Picone, 2026,
https://isip.piconepress.com/publications/reports/2026/eeg/annotations/):

> "If there is no seizure in the file, we mark the first second in the file on the
> channels Fp1-F7 as 'bckg' so that every edf file has a non-empty annotation file.
> This makes database curation much easier."

Verified against all 8140 TUSZ v2.0.6 recordings (2026-08-26):

- every `.csv_bi` is either bckg-only (7043: 6284 one-second stubs, 759 full-length)
  or seiz-spans-only (1097) — never both
- 229,336 of 264,260 twenty-second windows have no overlapping annotation row;
  nearly all sit in stub-marked seizure-free recordings
- seizure recordings carry full per-channel bckg/seiz delineation in `.csv`

So a TUSZ window with no overlapping annotation is verifiably seizure-free: either its
recording carries the whole-file stub, or it lies in the complement of exhaustively
annotated seizure spans. TUSZ `bckg` means "no seizure", not "nothing happening".
This is why `preprocess-TUSZ.py` maps empty windows to bckg (`or {"bckg"}`).

## TUAR / TUEV / TUSL: absence means unknown

No stub convention. Annotation coverage is genuinely partial (~29% / ~1.4% / ~3.3% of
time); unannotated stretches may contain unmarked artifacts/events. Absence must NOT
become bckg. Hence: no annotation -> all label columns false -> the window is excluded
from eval scope and from rationale generation (a window qualifies only with >=1 true
label; there is deliberately no "none" fallback in the teacher prompt, which would
mint false ground truth for merely-unassessed windows).

## The rule

bckg=true only where the corpus verifiably asserts it (explicit annotation, or TUSZ's
documented whole-file screening). Absence of knowledge is never converted into a label.
bckg never counts as an "event" for window selection (EVENTS excludes it): it marks the
absence of findings, so it must not attract the event-biased sampling.

Sources: guidelines above; TUSZ paper https://pmc.ncbi.nlm.nih.gov/articles/PMC6246677/
