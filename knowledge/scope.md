# scope: what each row needs

Every pipeline.db row carries a `scope` column naming its compute tier, derived
(like everything else) at sync time:

- full      - sampled test window: needs rationale + evaled + judged + judged_gpt
- rationale - train window with >=1 true label: needs rationale only
              (fine-tuning targets; never evaluated or judged)
- none      - unsampled test or unlabeled train: needs nothing, ever

Done-ness per tier: full <=> rationale AND evaled AND judged AND judged_gpt;
rationale <=> rationale; none <=> always done. This is materialized as a
`done` column (DONE_UPDATE in helpers/pipeline.py), recomputed after every
scope re-tier in both sync() and sample-test-split.py - query it directly:

  sqlite3 pipeline.db "SELECT scope, done, COUNT(*) FROM pipeline GROUP BY scope, done"

`judged_gpt` mirrors `judged` but syncs from judge-gpt.csv (the independent
gpt-5.6-luna judge pass, judge/judge-openai.py) instead of judge-baseline.csv.
Both columns are added by an idempotent ALTER TABLE migration in db(), so a
pre-existing pipeline.db upgrades in place on first use.

`labeled` (the ">=1 true label" fact from labels.csv) is stored alongside;
scope = CASE sampled -> full, train AND labeled -> rationale, else none. Both
sync() and sample-test-split.py apply the same normalization, so a sampling
policy change re-tiers rows automatically.

All dispatchers and status.py read their pending sets through scope - it is
the single source of truth for "does this row need computation". Quick check:

  sqlite3 pipeline.db "SELECT scope, COUNT(*) FROM pipeline GROUP BY scope"

Stage flags on scope='none' rows (e.g. evals imported from old, larger
samples) are inert history, not discrepancies.
