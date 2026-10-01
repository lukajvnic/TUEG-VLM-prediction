# scope: what each row needs

Every pipeline.db row carries a `scope` column naming its compute tier, derived
(like everything else) at sync time:

- full      - sampled test window: needs rationale + evaled + judged + judged_gpt
- rationale - sampled train window with >=1 true label: needs rationale only
              (fine-tuning targets; never evaluated or judged)
- none      - unsampled test, unsampled train, or unlabeled train: needs nothing, ever

`sampled` is set per split by two samplers with the same shape:
`eval/sample-test-split.py` (policy `settings.test-sample`) flags test rows and
`train/scripts/sample-train-split.py` (policy `settings.train-sample`, `--dry-run`)
flags train rows. Each resets only its own split, so running one never
disturbs the other. Since 2026-09-16 the train side is sampled too: the
fine-tune and the teacher pass both read scope, so the teacher only writes
rationales for windows the fine-tune will use (TUSZ train 21,207 -> 3,556).
**After pulling that change, run the train sampler once**, or every train row
is scope=none and `build-sft-jsonl.py` refuses to build.

`sampled` is a property of the path, but it is stored per (path, model) row and
the samplers only touch rows that exist when they run. Rows for a model added
to `config.yml` later used to come in with `sampled=0` and sit in scope=none
forever (found 2026-09-16 when the first fine-tune entry was added). `sync()`
now copies the flag onto any row whose path is sampled elsewhere, so adding a
model never needs the samplers re-run. A model entry may also carry
`datasets: [..]`; `sync()` then creates rows only for those datasets.

Done-ness per tier: full <=> rationale AND evaled AND judged AND judged_gpt;
rationale <=> rationale; none <=> always done. This is materialized as a
`done` column (DONE_UPDATE in helpers/pipeline.py), recomputed after every
scope re-tier in both sync() and sample-test-split.py - query it directly:

  sqlite3 pipeline.db "SELECT scope, done, COUNT(*) FROM pipeline GROUP BY scope, done"

`zero_shot` (2026-09-21) is 1 for the Ollama roster and 0 for `backend: hf`
entries (the HF base-as-is and the fine-tunes), set by sync from config.yml, so
the zero-shot benchmark's remaining work is one filter:

  sqlite3 pipeline.db "SELECT model, dataset, COUNT(*) FROM pipeline
                       WHERE zero_shot AND scope = 'full' AND NOT done GROUP BY 1, 2"

`judged_gpt` mirrors `judged` but syncs from judge-gpt.csv (the independent
gpt-5.6-luna judge pass, judge/judge-openai.py) instead of judge-baseline.csv.
Both columns are added by an idempotent ALTER TABLE migration in db(), so a
pre-existing pipeline.db upgrades in place on first use.

`labeled` (the ">=1 true label" fact from labels.csv) is stored alongside;
scope = CASE test AND sampled -> full, train AND sampled AND labeled ->
rationale, else none. sync() and both samplers apply the same normalization,
so a sampling policy change re-tiers rows automatically once the sampler runs.

All dispatchers and status.py read their pending sets through scope - it is
the single source of truth for "does this row need computation". Quick check:

  sqlite3 pipeline.db "SELECT scope, COUNT(*) FROM pipeline GROUP BY scope"

Stage flags on scope='none' rows (e.g. evals imported from old, larger
samples) are inert history, not discrepancies.
