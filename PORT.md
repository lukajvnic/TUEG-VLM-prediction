# Port active legacy runs

## 1. Freeze and archive on CCDB

Run this **before** pulling the new code. Replace the three job IDs.

```bash
REPO=~/dev/TUEG-VLM-prediction
JOBS="<original>,<large-tier>,<retry>"
STAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE="$SCRATCH/tueg-vlm-legacy/$STAMP"
mkdir -p "$ARCHIVE"

rsync -a "$REPO/datasets/" "$SCRATCH/tueg-vlm-datasets-backup/"
cp "$REPO/config.yml" "$ARCHIVE/config.yml"
squeue -r -j "$JOBS" -o "%i|%T|%M|%R" > "$ARCHIVE/squeue-before.txt"

scancel "$JOBS"
while squeue -h -j "$JOBS" | grep -q .; do sleep 10; done

rsync -a "$REPO/results/" "$ARCHIVE/results/"
rsync -a "$REPO/logs/" "$ARCHIVE/logs/"
sacct -X -j "$JOBS" --format=JobIDRaw,State,Reason > "$ARCHIVE/sacct.txt"
```

## 2. Pull the new code

```bash
cd "$REPO"
git pull --ff-only
source .venv/bin/activate
pip install -r requirements.txt
```

The in-repo `datasets/` directory remains in place. Do not run `git clean`.

## 3. Import all old arrays ((CONTINUE HERE))

Run this once. It reads every saved job ID and Slurm task log from the archive, then imports all matching legacy outcomes.

```bash
python temp/port_legacy.py "$ARCHIVE"
```

The command derives each task's dataset from its archived Slurm output, prints the imported run directory name, imports verified completed results as `success`, and imports legacy failures as `fail`. Interrupted tasks are not imported and must be rerun.

## 4. Submit unfinished and failed work

Create a retry config from the imported run:

```bash
cd "$REPO/eval"
python create-retry-config.py <imported-run-name>
```

This activates failed pairs only. Manually uncomment interrupted/unstarted pairs too. Keep completed pairs commented.

Review failed-model resources (`time`, `ram`, `gpus`) and `array-concurrency`, then submit:

```bash
python run.py
```
