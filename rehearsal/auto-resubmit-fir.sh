# Fir's watcher (2026-10-21), the same rules as Narval's auto-resubmit.sh: a grid / seed / zero-shot arm that timed out
# is resubmitted (training resumes from its last complete checkpoint, scoring from its last scored window); one that
# failed after its manifest was written is resubmitted once; any other failure (no manifest: the image check, the
# smoke, or out of memory on a 40 GB slice) is reported as FAILED for a decision by hand
cd /scratch/luka/TUEG-VLM-prediction
R=side/vision-freeze/results
records=$(sacct -u luka -S 2026-10-18T00:00 -X -n -P -o JobName,State | grep -E "^eeg-vlm-vision-freeze-(g-[^|]*-tusz|gm-|s-|z-)" | sed "s/^eeg-vlm-vision-freeze-//")
queued=$(squeue -u luka -h -o %j | sed "s/^eeg-vlm-vision-freeze-//")
done_arm() {
  case $1 in
    z-*-tusz) [ -f $R/summary-$1.csv ] && [ -f $R/summary-heldout-$1.csv ] && [ -f $R/summary-background-$1.csv ];;
    z-*|s-*) [ -f $R/summary-$1.csv ];;
    *) [ -f $R/summary-rationale-$1-answer-first.csv ] || [ -f $R/summary-rationale-$1-explain.csv ];;
  esac
}
submit() {
  module load StdEnv/2023 python/3.11 cuda arrow 2>/dev/null
  source .venv/bin/activate
  python side/vision-freeze/run.py $1 2>&1 | grep -v -i warn | tail -1
}
for arm in $(echo "$records" | awk -F"|" '$2 ~ /TIMEOUT/ {print $1}' | sort -u); do
  done_arm $arm && continue
  echo "$queued" | grep -qx "$arm" && continue
  echo "resubmitting $arm (timed out)"
  submit $arm
done
for arm in $(echo "$records" | awk -F"|" '$2 ~ /FAILED|OUT_OF_ME|NODE_FAIL/ {print $1}' | sort -u); do
  done_arm $arm && continue
  echo "$queued" | grep -qx "$arm" && continue
  failures=$(echo "$records" | grep -E "^$arm\|(FAILED|OUT_OF_ME|NODE_FAIL)" | wc -l)
  if [ -f side/vision-freeze/checkpoints/$arm/manifest.json ] && [ "$failures" -lt 2 ]; then
    echo "resubmitting $arm (trained, failed after training once)"
    submit $arm
  else
    echo "FAILED $arm ($failures failures, manifest: $([ -f side/vision-freeze/checkpoints/$arm/manifest.json ] && echo yes || echo no))"
  fi
done
