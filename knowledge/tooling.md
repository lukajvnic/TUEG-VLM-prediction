# Tooling inventory

Every script in the repo, what it is for, and whether it is live. Written because
several of these had no documentation at all and it was not obvious from the
filename which were load-bearing and which were leftovers.

## Live — data pipeline

| Script | Purpose |
|---|---|
| `datasets/<DS>/generate.py` | Render EDFs → windowed PNGs + `labels.csv`. One per dataset. |
| `datasets/render.py` | Shared windowing/rendering/filtering used by all six. Rewritten and tracked on 17 August 2026 after the gitignored `_render.py` was lost; see [data-generation.md](data-generation.md). |
| `datasets/verify_render.py` | Checks `render.py` against the shipped `labels.csv` and PNGs (window selection, canvas, layout box, ink). Run before regenerating anything. |
| `generate-all.sh` | Run all six `generate.py` in parallel (`--overwrite` by default). |
| `datasets/relabel.py` | Add the `assessed` column; relabel TUSZ's unannotated windows `bckg`. Pure `labels.csv` transform, idempotent, `--dry-run`. *Deleted from the working tree; still in HEAD. Its output is already applied.* |
| `datasets/recover_tuar_background.py` | Restore TUAR's discarded `bckg` labels from source. Self-verifying, `--dry-run`. *Deleted from the working tree; still in HEAD. Its output is already applied.* |
| `qc.py` | QC gate — missing/blank images, size drift, patient leakage, class balance. **Gitignored and currently absent from the working tree, so unrecoverable.** |

Neither relabel script re-renders anything, so both are safe to run against
images already shipped to the cluster (`labels.csv` travels via git).

## Live — evaluation

See [eval-pipeline.md](eval-pipeline.md) for detail. `run-eval.py` →
`run_array.sbatch` → `eval.py` (+ `sample.py`, `structure.py`), then `status.py`
to monitor, `create-retry-config.py` to retry, `scripts/merge-runs.py` to union
the retry back into the base run, then `summarize.py` to score and report
(`summary.csv`, `classes.csv`, `rank.csv`, one `rank-<DATASET>.png` per
dataset, plotting recording-level balanced accuracy; `--metric macro-f1` charts
that instead). It also draws `agreement-<DATASET>.png` from the judge stage's
`agreement/*.csv` when that directory exists, so one command produces every chart
in a run. `summarize.py` also
**contains** the metric-agnostic bar-chart renderer that draws those charts
(formerly `eval/chart.py`), which makes it the only eval script needing
matplotlib.

Layout convention: the things **you** invoke live in `eval/`
(`run-eval.py`, `run-judge.py`, `status.py`, `summarize.py`,
`create-retry-config.py`); the things a **Slurm task or submitter** invokes live
in `eval/scripts/`.

`eval/scripts/merge-runs.py <base> <retry> [--into NAME] [--dry-run]` exists because a
retry lands in a **new** run dir and `resume-from` only suppresses re-work — it
never copies base rows forward, so neither dir is scoreable alone. Union keyed on
`path` (the two are disjoint by construction), retry status wins per task, and it
refuses to overwrite an existing destination.

## Live — rationale agreement

Grades *why* a model answered, not just *what*. See the "Rationale agreement"
section of [eval-pipeline.md](eval-pipeline.md).

| Script | Purpose |
|---|---|
| `eval/run-judge.py` | Driver: `check` (join coverage, no GPU), `submit` (array job), `report` (aggregate CSVs). |
| `eval/scripts/judge.py` | Worker — one array task per model; joins results ↔ references, judges each pair, resumable per pair. |
| `eval/scripts/judge_array.sbatch` | Per-task Ollama-in-Apptainer wrapper for `judge.py`. |
| `eval/config.yml` → `judge:` key | Judge model + resources. Read live, never from a run's frozen copy. |

**Prerequisite:** `sbatch --export=ALL,SPLIT=test eval/scripts/generate-rationales.sbatch`.
Without it there are no reference rationales and every join is empty.
`run-judge.py check` is the free local guard against discovering that after
queueing 35 tasks.

## Live — rationales / fine-tune

See [rationale-generation.md](rationale-generation.md).
`eval/scripts/generate-rationales.sbatch` → `eval/scripts/generate-rationales.py`
(`--split train|test`, `--dry-run`). Degenerate output is caught at write time,
so the old `scrub-degenerate.py` cleanup script was removed (2026-08-16).

It sits under `eval/` despite feeding the fine-tune because it is the teacher
pass the agreement stage depends on, and because it shares `sample.py` and
`config.yml`'s `test-sample` policy with the eval track — as a sibling it can now
`from sample import select` instead of loading that module by path.
`train/scripts/finetune_sample.py` is the fine-tune entry point, and reads
`rationales.csv` **only**: `rationales-test.csv` holds test patients and must
never reach it.

## Live — data acquisition

| Script | Purpose |
|---|---|
| `eval/scripts/download.sh` | rsync a TUEG subcorpus from `isip.piconepress.com`. Usage: `download.sh <remote_path> [destination]`. |
| `zip-datasets.sh` | Bundle `train/` + `test/` per dataset into `zips/`. |
| `hf-install.py` / `hf-install.sbatch` | Download Hugging Face snapshots of the models (weights + processor + chat template) for the fine-tune track. Gated repos need license acceptance first. *Both deleted from the working tree; still in HEAD. Restore before resuming the fine-tune track.* |

## Report

`report/project-report.tex`, built with `tectonic report/project-report.tex`
(10 pages as of 17 August 2026). **Audience is the PI**, and the brief is in
`Planning.md` at the repo root: a professional technical and architectural
write-up of what has been done. It follows that outline — Ollama setup, Hugging
Face setup, dataset generation, rationale generation, eval — rather than an
arbitrary structure, so rewrites should start from `Planning.md`.

**Typography (17 August 2026).** Tectonic runs XeTeX, so the preamble uses
`libertinus` (serif + sans + mono + math from one family), `hyphenat[htt]` so
file paths never hyphenate mid-token, `tcolorbox[most]` for the summary panel and
callouts, and `longtable` for the glossary. Headings are sans in the `accent`
navy with a hairline rule; the §6 status table uses coloured chips. All packages
resolve from Tectonic's bundle, but a **cold cache needs network** on first build.

**The Google Doc and the LaTeX have diverged (17 August 2026).** There is a
review copy at `docs.google.com/document/d/16Z7oKlputqVM8FH5UksI-pi4eG-8H5g4jJEtjVMJUxM`,
styled to match the LaTeX: Times New Roman, black only, justified body, centred
title block, booktabs-style tables (horizontal rules, no fills, no verticals).
No colour and no sans anywhere — that was tried and rejected.
Luka reviewed it in Suggesting mode and those suggestions were accepted, which cut
material the LaTeX still carries: the whole "What the project is testing" section,
the `def-milad777` account name, the `rsync -auvxL` clause, the images-vs-labels
transfer paragraph, and `<DS>` became `<TU??>`. The Doc is therefore the *shorter*
version and sections renumber 1--6. **Reconcile before either is treated as
canonical** — do not regenerate one from the other assuming they match.

**Register (17 August 2026).** Plain words and short sentences, but written for a PI,
not for Slack. An earlier pass leaned too far into `write-like-luka` and read as
casual ("repos", "forty-odd", "killed the process", "eats most of the compute"). Keep
the simplicity, keep the formal register. Also: no italics anywhere, and no bold
labels leading bullet items.

**Two editorial rules the report now follows**, both from PI review comments:
1. **Gloss the jargon on first use** (MIG, Apptainer, GGUF, LoRA, flash
   attention, KV cache, MoE, macro-F1, structured output, vision tokens). There
   is also an appendix glossary, `\appendix` §A.
2. **Name the incident, not just the design.** Where a choice traces to something
   that broke — the login-node HF download, the compute-node `ollama pull` that
   cost a judge array, the `JobIDRaw` bug, the two-script scoring trap, the
   `_render.py` scaling attempts — the report says so. That is the detail the PI
   asked for and it is what makes the design defensible.

The report summarises this knowledge base for an outside reader, so
**`knowledge/` stays the source of truth** and the report is regenerated from it,
never the other way round.

There is exactly one report file, and there should stay exactly one. A 22-page
version existed until 17 August 2026 and was deleted after checking every fact in
it already appeared here (the 2,488 GB to 1,008 GB RAM retune, the `JobIDRaw`
bug, the Ollama env vars, `p99 * 1.15` scaling and the crest-factor rationale all
did). `report/` is untracked, so nothing in it is recoverable from git.

Note the report names two things this directory did not previously record: there
is **no `ollama-install.py`** despite `Planning.md` asking for one (setup is the
manual `$SCRATCH/ollama/ollama.sif` Apptainer build), and several documented
scripts are missing from the working tree (see `known-issues.md`).

## Situational

| Script | Purpose |
|---|---|
| `datasets/<DS>/generate-annotated.py` | Render windows with annotation overlays drawn on. A **debugging/inspection** aid for checking that labels line up with the waveforms — not part of the benchmark pipeline, and its output must never be fed to a model (the overlay leaks the label). |
| `temp/port_legacy.py` | One-off import of an archived pre-restructure eval run into the current `eval/runs/` layout. *Deleted from the working tree; still in HEAD.* |

## Cruft — safe to delete

| Script | Why |
|---|---|
| `train/run.py` | Empty file (0 bytes). |
| `datasets/TUEP/scan.py` | Ad-hoc row counter that reads `row[1]` as a label. `row[1]` is the `split` column, so it has counted 0 epileptic rows since the split column was added — it predates the current schema and was never updated. |
| `README.md` (repo root) | Empty. The real documentation is this directory. |

## Note on what is gitignored

`.gitignore` excludes `knowledge/` and `qc.py`. So **this knowledge base does not
travel to the cluster** with `git pull`, and neither does the QC gate. That is
deliberate (they are local working aids), but it means anything the cluster needs
must live in a tracked file, not here.
