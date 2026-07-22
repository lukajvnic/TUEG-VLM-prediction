# Port legacy runs

## 1. Update the repository, not the data

Commit and push the new `eval/` code from `TUEG-VLM-restructure` to the `TUEG-VLM-prediction` repository first.

On the Alliance login node, back up the datasets and update the existing clone in place:

```bash
REPO=~/dev/TUEG-VLM-prediction
rsync -a "$REPO/datasets/" "$SCRATCH/TUEG-VLM-datasets-backup/"
cd "$REPO"
git status --short datasets
git pull --ff-only
```

Do not delete or reclone the repository. `git pull` preserves the downloaded dataset files, which are expected to remain at `datasets/`.

## 2. Freeze the legacy jobs

Before cancelling, archive their config, logs, results, and Slurm state. Replace the IDs below.

```bash
JOBS="<original-job-id>,<large-tier-job-id>,<retry-job-id>"
STAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE="$SCRATCH/tueg-vlm-legacy/$STAMP"
mkdir -p "$ARCHIVE"

squeue -r -j "$JOBS" -o "%i|%T|%M|%R" | tee "$ARCHIVE/squeue-before.txt"
sacct -X -j "$JOBS" --format=JobIDRaw,State,Reason | tee "$ARCHIVE/sacct-before.txt"
cp "$REPO/config.yml" "$ARCHIVE/config.yml"

scancel "$JOBS"
while squeue -h -j "$JOBS" | grep -q .; do sleep 10; done

rsync -a "$REPO/results/" "$ARCHIVE/results/"
rsync -a "$REPO/logs/" "$ARCHIVE/logs/"
sacct -X -j "$JOBS" --format=JobIDRaw,State,Reason | tee "$ARCHIVE/sacct-after.txt"
```

## 3. Import legacy outcomes

Create one imported run per old run:

- completed pairs become `success` rows and have their result CSV copied into the new result layout;
- known failures become `fail` rows with their legacy reason;
- interrupted/unstarted pairs remain unfinished and will be rerun from the start.

Do not import partial CSVs as successful results. The new evaluator does not yet resume a partially completed model/dataset task per image.

## 4. Resume only remaining work

1. Run `create-retry-config.py` on the imported status to activate failed pairs.
2. Manually uncomment only interrupted/unstarted pairs too.
3. Keep completed pairs commented.
4. Review failures and adjust affected models' `time`, `ram`, `gpus`, and possibly `array-concurrency`.
5. Submit the remaining work:

```bash
cd "$REPO/eval"
python run.py
```

A new run directory is expected and is fine. Imported legacy results and new resumed results can remain separate.
