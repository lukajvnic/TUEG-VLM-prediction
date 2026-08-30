# Rationale generation

A VLM is shown an image *plus the known label* and asked to write a grounded
justification. The same generator serves two different consumers, selected by
`--split`:

| `--split` | writes | consumer |
|---|---|---|
| `train` (default) | `datasets/<DS>/rationales.csv` | fine-tune targets (goal B) |
| `test` | `datasets/<DS>/rationales-test.csv` | **reference** rationales for `eval/run-judge.py` |

**The two files are separate on purpose, and the separation is load-bearing.**
`rationales-test.csv` is written from held-out **test** patients. Training on it
would void the patient-level split, so nothing may glob `rationales*.csv` into
the fine-tune. It also protects the train file mechanically: `init_csv()`
rebuilds its target file from `labels.csv` on every run, so a shared filename
would blank one split's work each time the other ran.

## Files

- `eval/scripts/generate-rationales.py` — the generator (both splits).
- `eval/scripts/generate-rationales.sbatch` — the Slurm job; `SPLIT` selects which.
- `train/config.yml` — fine-tune hyperparameters.
- `train/scripts/finetune_sample.py` — the fine-tune script.

The generator lives under **`eval/`**, not `train/`, even though `--split train`
feeds the fine-tune: it is the teacher pass the agreement stage depends on, and
it shares `eval/scripts/sample.py` and `eval/config.yml`'s `test-sample` policy
with the eval track. As a sibling of `sample.py` it imports `select` directly
instead of loading that module by path.

## `generate-rationales.py`

Config:
```python
MODEL = "gemma3:12b"
MAX_OUTPUT_TOKENS = 512
PARALLEL_REQUESTS = int(os.environ.get("OLLAMA_NUM_PARALLEL", "4"))
FLUSH_EVERY = 25
MAX_CONSECUTIVE_DEGENERATE = 20
```

### Why the teacher is gemma3:12b and not a Qwen
Qwen2.5-VL in Ollama falls into **infinite single-token repetition** on these
plots — the reply is one character repeated for the whole `num_predict` budget.
It affects 7b/32b/72b alike (ollama/ollama#10767), so a bigger Qwen is not the
fix. Ollama has no repetition-loop detection (#16179) and exposes neither DRY nor
XTC (#7504), so no sampler setting prevents it either. Changing model family was
the only lever. `OLLAMA_FLASH_ATTENTION=1` is set for the same family of bugs
(the non-FA path mispredicts and never emits EOS).

Behaviour:
- **Row scope is `target_rows(dataset, split, policy, run)`**, computed once per
  dataset by `build_plan` and shared by `init_csv` / `pending_count` /
  `build_batch`, so those three can never disagree about what is in scope.
  - `train` → rows with `split == "train"` (or no split column, for legacy data).
  - `test` → the **sampled** test windows, via `eval/scripts/sample.py`'s
    `select()`, imported directly now that the two are siblings under
    `eval/scripts/`. Not raw `labels.csv`: generating outside the sampled set is wasted
    GPU, because the eval run produces no rationale to pair against it.
    **Verified 2026-08-12:** `--split test --dry-run` reports 1200/3655/1084/354/
    97/8460 for TUAB/TUAR/TUEP/TUEV/TUSL/TUSZ = **14,850**, identical to
    `python eval/scripts/sample.py`.
  - `--split test --run <run>` instead takes the union of image basenames present
    in that run's `results/*.csv`. Use it when the sampling policy changed after
    the run was submitted; it guarantees every scored window has a reference, at
    the cost of having to wait for results to exist.
- The test-split sampling caps are read from `eval/config.yml`'s
  `settings.test-sample`, not duplicated — one number drives both the evaluated
  set and the reference set.
- **`--dry-run`** reports target / written / pending per dataset and writes
  nothing. Cheap, local, and the way to confirm the train path is untouched
  before running the test path.
- **Reconciling `init_csv`:** on each run it rebuilds *its split's* file from the
  current `labels.csv`, **carrying over any rationales already written**. This
  keeps the files in sync so a changed `labels.csv` never desyncs the lookup and
  crashes the job (a real bug that was fixed).
- **Bounded concurrency:** a `ThreadPoolExecutor` fed with a sliding window — it
  only base64-encodes the next image when a slot frees, so memory stays bounded
  even across TUSZ's ~21k train images.
- **Periodic flush:** writes the CSV every `FLUSH_EVERY` completions (not every
  image) to avoid O(N²) rewrites, plus a final flush.
- **Resumable & fail-tolerant:** a failed request leaves the row blank (retried
  next run); resubmitting skips completed rows.
- **Degenerate-output guard:** `is_degenerate(text)` rejects a reply that is too
  short, or has too few distinct characters/words — the signature of a wedged
  runner. Greedy decoding makes that output identical every time, so it passes
  every check *except* this one. A degenerate reply leaves the row blank (so a
  later run retries) rather than baking garbage into the fine-tuning set, and
  `MAX_CONSECUTIVE_DEGENERATE = 20` aborts the job outright, since once the
  runner goes bad it stays bad and would spend the rest of the allocation writing
  garbage.
- **Label columns are selected by exclusion:** `create_prompt` builds the label
  list from any column whose value is `"true"`, so `NON_LABEL_COLUMNS`
  (`image_path`, `split`, `assessed`) must exclude bookkeeping flags by name.
  `assessed` in particular has the literal value `"true"` and would otherwise be
  injected into every prompt as if it were a class.
- **Grounded prompts:** the rationale prompt mirrors the eval rubric — it opens
  with "multi-channel EEG waveform plot" and asks for per-finding evidence
  (which channel, where in time, what the curve looks like). This keeps the
  fine-tune target aligned with what the eval measures.

## `generate-rationales.sbatch`

- Resources: `--nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32G
  --time=2-00:00:00`.
- Uses **Apptainer**, same as the eval array job. Native `ollama` is *not* on the
  compute node's `PATH` — assuming it was left the job spinning until it timed
  out. Runs `apptainer exec --nv "$OLLAMA_IMAGE" ollama serve`, waits for the
  server, and pulls the model only if `ollama show` misses.
- Sets `OLLAMA_CONTEXT_LENGTH=8192` (a 1536×1536 image + prompt is ~3050 tokens,
  leaving ample room for the 512-token report), `OLLAMA_NUM_PARALLEL=4`,
  `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_IMAGE=$SCRATCH/ollama/ollama.sif`,
  `OLLAMA_MODELS=$SCRATCH/ollama/models`.
- Reads the model tag out of `generate-rationales.py` with `sed`, so `MODEL` stays
  the single place to change it.
- **`SPLIT` env var** picks the split, defaulting to `train`:
  ```bash
  sbatch eval/scripts/generate-rationales.sbatch                          # train (fine-tune targets)
  sbatch --export=ALL,SPLIT=test eval/scripts/generate-rationales.sbatch  # test  (reference rationales)
  ```
  Both are resumable, so the test pass fits the same 2-day walltime; 14,850
  images at the same teacher is roughly 40% of the train pass's 33,491.
- Logs `ollama --version` with the run: vision-model regressions land often
  (ollama/ollama#13549 broke image input in 0.13.3+), so the version matters as
  much as the model choice.
- **Submit from the project dir**: `sbatch eval/scripts/generate-rationales.sbatch`.
  The script `cd`s to `$SLURM_SUBMIT_DIR` and uses project-root-relative paths, so
  where it is submitted from matters; where the file lives does not.

## Degenerate rationales are caught at write time

A wedged Ollama runner returns one token repeated for the whole `num_predict`
budget (a 512-char run of `?`). Greedy decoding makes it identical every time, so
it looks like a real answer to every check except `is_degenerate`. Such a row is
left **blank** rather than saved, so a later run retries it, and 20 consecutive
degenerate responses (`MAX_CONSECUTIVE_DEGENERATE`) aborts the job rather than
spending the rest of the allocation writing garbage.

This matters most for `rationales-test.csv`: garbage there is not merely wasted,
`eval/run-judge.py` would score it as a genuine reference and every model would be
judged against noise.

There used to be a `scrub-degenerate.py` at the repo root that blanked rows
written *before* this guard existed. **Removed 2026-08-16** — the guard now runs
at write time, so there is nothing left for it to clean.

If you ever do suspect stale garbage in a rationales file, the check is
`is_degenerate` in `eval/scripts/generate-rationales.py`; blanking the offending
`ground_truth_rationales` cells makes the next run regenerate them.

## Prerequisites on the cluster

`generate-rationales.py` reads `datasets/<DS>/labels.csv` and the `train/` PNGs, so
before submitting:
1. `labels.csv` present **and matching the images** (comes via git; the images via
   the transferred zips). A stale `labels.csv` breaks path resolution — verify with
   `python qc.py` or a labels↔images set comparison.
2. `train/*.png` extracted into `datasets/<DS>/train/`.
3. `.venv` present; `$SCRATCH/ollama/ollama.sif` and `$SCRATCH/ollama/models`
   in place (Apptainer image, not a PATH `ollama`).

## Fine-tuning (`train/config.yml`)

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`; `dataset: TUEP`,
  `output-dir: checkpoints/TUEP`.
- LoRA: rank 16, alpha 32, dropout 0.05.
- Training: 3 epochs, batch 1, grad-accum 8, lr 2e-4, eval/save every 100 steps,
  `save-total-limit: 2`.
- The fine-tune's quality is capped by the rationale generator (`gemma3:12b`) —
  see [known-issues.md](known-issues.md).
- **`train/run.py` is currently an empty file**; `train/scripts/finetune_sample.py`
  is the fine-tune entry point. There is no Slurm wrapper for the fine-tune yet.
