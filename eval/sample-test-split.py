import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import DATASETS, ROOT, config, db, read_csv

BINARY = {"TUEP", "TUAB"}
PATIENT_LEVEL = {"TUEP"}
BACKGROUND = {"bckg"}
NON_LABELS = ("path", "ground_truth_rationale")


def parse_name(path):
    patient, scan, window = Path(path).stem.rsplit("_", 2)
    return patient, f"{patient}_{scan}", int(window)


def spread(items, cap):
    if cap is None or cap <= 0 or cap >= len(items):
        return list(items)
    if cap == 1:
        return [items[len(items) // 2]]
    picked = {round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)}
    return [items[i] for i in sorted(picked)]


def positives(row):
    return frozenset(c for c in row if c not in NON_LABELS and row[c].strip().lower() == "true")


def abundant_classes(rows, threshold):
    by_class = defaultdict(set)
    for row in rows:
        _, recording, _ = parse_name(row["path"])
        for cls in positives(row):
            by_class[cls].add(recording)
    return {cls for cls, recordings in by_class.items() if len(recordings) >= threshold}


def select(rows, dataset, policy):
    rows = [r for r in rows if positives(r)]
    if not rows:
        return []
    abundant = abundant_classes(rows, policy["abundant-class-recordings"])

    by_recording, patients = defaultdict(list), defaultdict(set)
    for row in rows:
        patient, recording, window = parse_name(row["path"])
        by_recording[recording].append((window, row))
        patients[patient].add(recording)

    if dataset in PATIENT_LEVEL:
        keep = set()
        for patient in sorted(patients):
            keep.update(spread(sorted(patients[patient]), policy["recordings-per-patient"]))
    else:
        keep = set(by_recording)

    chosen = []
    for recording in sorted(keep):
        windows = [row for _, row in sorted(by_recording[recording])]
        if dataset in BINARY:
            chosen += spread(windows, policy["binary-windows-per-recording"])
            continue
        by_signature = defaultdict(list)
        for row in windows:
            by_signature[positives(row)].append(row)
        for signature, group in sorted(by_signature.items(), key=lambda kv: sorted(kv[0])):
            if signature - abundant - BACKGROUND:
                chosen += group
            elif not signature - BACKGROUND:
                chosen += spread(group, policy["background-per-recording"])
            else:
                chosen += spread(group, policy["windows-per-signature"])

    return sorted(row["path"] for row in chosen)


def main():
    policy = config()["settings"]["test-sample"]
    conn = db()
    conn.execute("UPDATE pipeline SET sampled = 0")
    total = 0
    for ds in DATASETS:
        rows = [r for r in read_csv(ROOT / "datasets" / ds / "labels.csv")
                if r["path"].startswith("test/")]
        chosen = select(rows, ds, policy)
        conn.executemany("UPDATE pipeline SET sampled = 1 WHERE path = ?",
                         [(f"{ds}/{p}",) for p in chosen])
        recordings = len({parse_name(p)[1] for p in chosen})
        print(f"{ds}: {len(rows)} test -> {len(chosen)} sampled ({recordings} recordings)")
        total += len(chosen)
    conn.commit()
    print(f"total: {total}")


if __name__ == "__main__":
    main()
