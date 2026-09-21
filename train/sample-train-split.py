import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import (DATASETS, DONE_UPDATE, RATIONALE, ROOT, SCOPE_UPDATE, config, db, duplicate_images,
                              parse_name, positives, rationale_problems, read_csv, spread)

BINARY = {"TUEP", "TUAB"}
BACKGROUND = {"bckg"}
SHOW = 5


class Corpus:
    """What the screens need across all six corpora: image hashes (a byte-identical render is the same EEG even
    under another patient token), the test-side hashes, and patient tokens merged through shared images."""

    def __init__(self, policy):
        self.hashes, self.groups, self.canon = duplicate_images()
        self.everywhere = policy["exclude-cross-dataset-test-patients"]
        self.test_hashes = defaultdict(set)  # dataset -> md5s of its test images
        self.test_patients = defaultdict(set)
        for ds in DATASETS:
            for row in read_csv(ROOT / "datasets" / ds / "labels.csv"):
                if row["path"].startswith("test/"):
                    self.test_hashes[ds].add(self.hashes[ds][row["path"]])
                    self.test_patients[ds].add(self.patient(row["path"]))

    def patient(self, path):
        patient = parse_name(path)[0]
        return self.canon.get(patient, patient)

    def held_out(self, dataset):
        # with the flag on, a patient (or image) in any corpus's test split is never trained on; off, only its own
        sources = DATASETS if self.everywhere else [dataset]
        return (set().union(*(self.test_patients[ds] for ds in sources)),
                set().union(*(self.test_hashes[ds] for ds in sources)))


def screen(rows, dataset, policy, corpus):
    # drops rows that must not be trained on; a rejected rationale drops its window from scope (regenerating
    # at temperature 0 would give the same text back), and the budget picks another window instead
    held_out, test_hashes = corpus.held_out(dataset)
    hashes = corpus.hashes[dataset]
    labels_by_hash = defaultdict(set)
    for row in rows:
        if row["path"] not in hashes:
            sys.exit(f"{dataset}: no hash for {row['path']} - run helpers/hash-images.py {dataset}")
        labels_by_hash[hashes[row["path"]]].add(positives(row))
    reasons, examples, seen, kept = Counter(), defaultdict(list), set(), []
    for row in sorted(rows, key=lambda r: r["path"]):
        labels = positives(row)
        if not labels:
            continue  # unassessed: no ground truth
        if corpus.patient(row["path"]) in held_out:
            reasons["test patient"] += 1
            continue
        if hashes[row["path"]] in test_hashes:
            reasons["identical to a test image"] += 1
            continue
        if len(labels_by_hash[hashes[row["path"]]]) > 1:
            reasons["duplicate with conflicting labels"] += 1
            continue
        text = row[RATIONALE].strip()
        if policy["reject-bad-rationales"] and text:
            problems = rationale_problems(text, labels)
            if text in seen:
                problems.append("duplicate")
            seen.add(text)
            if problems:
                for problem in problems:
                    reasons[problem] += 1
                    examples[problem].append((row["path"], text))
                continue
        kept.append(row)
    return kept, reasons, examples


def round_robin(groups, budget):
    # one item per group per pass, groups in sorted order, so a budget cut lands evenly across patients
    queues = [list(items) for _, items in sorted(groups.items()) if items]
    picked = []
    while queues and len(picked) < budget:
        for queue in queues:
            if len(picked) < budget:
                picked.append(queue.pop(0))
        queues = [q for q in queues if q]
    return picked


def select_binary(by_patient, policy):
    by_class = defaultdict(lambda: defaultdict(list))
    for patient, recordings in by_patient.items():
        for recording in spread(sorted(recordings), policy["recordings-per-patient"]):
            for _, row in sorted(recordings[recording]):
                by_class[next(iter(positives(row)))][patient].append(row)
    sizes = {cls: sum(len(v) for v in patients.values()) for cls, patients in by_class.items()}
    budget = int(min(sizes.values()) * policy["majority-ratio"]) if len(sizes) > 1 else max(sizes.values())
    chosen = []
    for cls, patients in by_class.items():
        chosen += round_robin(patients, min(budget, sizes[cls]))
    return chosen


def select_multilabel(by_patient, policy):
    events, background = [], defaultdict(dict)
    for patient, recordings in by_patient.items():
        for recording, windows in recordings.items():
            rows = [row for _, row in sorted(windows)]
            events += [r for r in rows if positives(r) - BACKGROUND]
            quiet = [r for r in rows if not positives(r) - BACKGROUND]
            if quiet:
                background[patient][recording] = spread(quiet, policy["background-per-recording"])
    per_patient = {}
    for patient, recordings in background.items():
        kept = spread(sorted(recordings), policy["recordings-per-patient"])
        per_patient[patient] = [row for recording in kept for row in recordings[recording]]
    total = sum(len(v) for v in per_patient.values())
    budget = int(len(events) * policy["majority-ratio"]) if events else total
    return events + round_robin(per_patient, min(budget, total))


def select(rows, dataset, policy, corpus):
    by_patient = defaultdict(lambda: defaultdict(list))
    for row in rows:
        _, recording, window = parse_name(row["path"])
        by_patient[corpus.patient(row["path"])][recording].append((window, row))
    if not by_patient:
        return []
    chosen = (select_binary if dataset in BINARY else select_multilabel)(by_patient, policy)
    return sorted(chosen, key=lambda r: r["path"])


def describe(rows, corpus):
    patients = Counter(corpus.patient(r["path"]) for r in rows)
    classes = Counter(c for r in rows for c in positives(r))
    listed = ", ".join(f"{c} {n}" for c, n in sorted(classes.items()))
    return f"{len(patients)} patients, max {max(patients.values(), default=0)}/patient; {listed}"


def main():
    parser = argparse.ArgumentParser(description="flag the train windows the fine-tune uses (sampled=1 in pipeline.db)")
    parser.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    parser.add_argument("--show", metavar="REASON", help=f"print up to {SHOW} rejected rationales per dataset for this reason")
    args = parser.parse_args()
    policy = config()["settings"]["train-sample"]
    corpus = Corpus(policy)
    conn = None if args.dry_run else db()
    if conn:
        conn.execute("UPDATE pipeline SET sampled = 0 WHERE split = 'train'")
    total = 0
    for ds in DATASETS:
        rows = [r for r in read_csv(ROOT / "datasets" / ds / "labels.csv") if r["path"].startswith("train/")]
        eligible, reasons, examples = screen(rows, ds, policy, corpus)
        chosen = select(eligible, ds, policy, corpus)
        if conn:
            conn.executemany("UPDATE pipeline SET sampled = 1 WHERE path = ?",
                             [(f"{ds}/{r['path']}",) for r in chosen])
        dropped = "; dropped " + ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())) if reasons else ""
        print(f"{ds}: {len(rows)} train -> {len(chosen)} sampled ({describe(chosen, corpus)}{dropped})")
        for path, text in examples.get(args.show, [])[:SHOW]:
            print(f"    {path}: {text[:200]!r}")
        total += len(chosen)
    if conn:
        conn.execute(SCOPE_UPDATE)
        conn.execute(DONE_UPDATE)
        conn.commit()
    print(f"total: {total}{' (dry run, nothing written)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
