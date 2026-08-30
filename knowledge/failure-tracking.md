# failure tracking

A failed unit (eval pair, judge pair, rationale) writes no output row, so its
pipeline.db boolean stays 0 and the next dispatcher run retries it. Relaunching
after a fix is always the same command (run.py / generate-rationales.py /
judge.py) - they only ever submit pending work.

Why something failed is recorded in two append-only files under logs/:

- failures.csv (time, stage, dataset, path, model, job, task, error) - one row
  per failed attempt, written by every worker via helpers.pipeline.log_failure.
  job/task point at the Slurm log: logs/<jobname>-<job>_<task>.out.
- tasks.csv (time, job, task, jobname, event, detail) - start/end records
  written by the generated sbatch script. start without end = still running or
  killed hard (walltime/OOM, check sacct); end with nonzero code = crashed.

`python helpers/status.py` joins these with pipeline.db and prints: stage
progress per dataset; pending counts split into failed vs never-attempted;
open failures grouped by (stage, model, dataset) with the top error strings
and an example log path; dead and nonzero-exit tasks.

"Open" means the unit is still pending - once a retry succeeds, its old
failure records stop being reported (history stays in the file). failures.csv
starts at the first run after this change; arrays launched before it only
reported to their .out files.

The debug loop: status.py -> read the error / open the log -> fix -> re-run
the dispatcher.
