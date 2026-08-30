import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ROOT, SUMMARY, sync

TOP_ERRORS = 3
JOB_NAMES = {"eval": "eeg-vlm-eval", "judge": "eeg-vlm-judge", "rationale": "eeg-vlm-rationales"}

PENDING = {
    "rationale": "SELECT DISTINCT dataset, path FROM pipeline "
                 "WHERE preprocessed = 1 AND rationale = 0 AND (split = 'train' OR sampled = 1)",
    "eval": "SELECT dataset, path, model FROM pipeline WHERE sampled = 1 AND evaled = 0",
    "judge": "SELECT dataset, path, model FROM pipeline "
             "WHERE sampled = 1 AND evaled = 1 AND rationale = 1 AND judged = 0",
}


def read_rows(name, header):
    path = ROOT / "logs" / name
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return [dict(zip(header, r)) for r in rows if r and r[0] != header[0]]


def pending_sets(conn):
    return {stage: {(r[0], r[1].split("/", 1)[1], *r[2:]) for r in conn.execute(query)}
            for stage, query in PENDING.items()}


def failure_key(row):
    key = (row["dataset"], row["path"])
    return key if row["stage"] == "rationale" else (*key, row["model"])


def report_pending(pending, open_failures):
    failed = defaultdict(set)
    for row in open_failures:
        failed[row["stage"]].add(failure_key(row))
    print()
    for stage, units in pending.items():
        print(f"{stage}: {len(units)} pending ({len(failed[stage])} with failures, "
              f"{len(units) - len(failed[stage])} not yet attempted)")


def report_failures(open_failures):
    if not open_failures:
        print("\nno open failures")
        return
    groups = defaultdict(list)
    for row in open_failures:
        groups[(row["stage"], row["model"], row["dataset"])].append(row)
    print(f"\nopen failures ({len(open_failures)} records):")
    for (stage, model, ds), rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        example = next((r for r in rows if r["job"]), rows[-1])
        where = (f"logs/{JOB_NAMES[stage]}-{example['job']}_{example['task']}.out"
                 if example["job"] else "local run")
        print(f"  {stage} {model} {ds}: {len({r['path'] for r in rows})} units, "
              f"{len(rows)} attempts ({where})")
        for error, n in Counter(r["error"][:110] for r in rows).most_common(TOP_ERRORS):
            print(f"    {n}x {error}")


def report_tasks(tasks):
    started, ended = {}, {}
    for row in tasks:
        key = (row["job"], row["task"], row["name"])
        (started if row["event"] == "start" else ended)[key] = row["detail"] or row["time"]
    dead = sorted(k for k in started if k not in ended)
    bad = sorted((k, ended[k]) for k in ended if ended[k] not in ("", "0"))
    if dead:
        print("\ntasks started but not finished (running, or killed by walltime/OOM):")
        for job, task, name in dead:
            print(f"  {name} {job}_{task} - check: sacct -j {job}_{task}")
    if bad:
        print("\ntasks exited nonzero:")
        for (job, task, name), code in bad:
            print(f"  {name} {job}_{task} exit {code} - logs/{name}-{job}_{task}.out")


def main():
    conn = sync()
    for ds, images, sampled, rationales, evaled, judged in conn.execute(SUMMARY):
        print(f"{ds}: {images} images, {sampled} sampled, {rationales} rationales, "
              f"{evaled} evaled, {judged} judged")
    pending = pending_sets(conn)
    failures = read_rows("failures.csv",
                         ["time", "stage", "dataset", "path", "model", "job", "task", "error"])
    open_failures = [r for r in failures if failure_key(r) in pending.get(r["stage"], set())]
    report_pending(pending, open_failures)
    report_failures(open_failures)
    report_tasks(read_rows("tasks.csv", ["time", "job", "task", "name", "event", "detail"]))


if __name__ == "__main__":
    main()
