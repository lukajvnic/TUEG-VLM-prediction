#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p logs

datasets=(TUAB TUAR TUEP TUEV TUSL TUSZ)
pids=()
for ds in "${datasets[@]}"; do
    python3 "preprocess-$ds.py" > "logs/$ds.log" 2>&1 &
    pids+=($!)
done

fail=0
for i in "${!datasets[@]}"; do
    wait "${pids[$i]}" || { echo "${datasets[$i]} failed - see logs/${datasets[$i]}.log" >&2; fail=1; }
done
wc -l ../datasets/*/labels.csv
exit $fail
