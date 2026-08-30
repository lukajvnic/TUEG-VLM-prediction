import ast
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "models"))
from helpers.pipeline import DATASETS, RATIONALE, ROOT, SUMMARY, config, read_csv, sync
from structure import labels

RUNS = ROOT / "eval" / "runs"
BACKUP = Path.home() / "tueg-legacy"
JUDGE_HEADER = ["path", "model", "correct_predictions", "correct_rationale", "correct_rationale_reason"]


def set_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_set(value):
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return set()
    return set(parsed) if isinstance(parsed, (set, frozenset, list, tuple)) else set()


def true_labels(row):
    return any(v.strip().lower() == "true" for c, v in row.items() if c not in ("path", RATIONALE))


def legacy_files(kind, models):
    for path in sorted(RUNS.glob(f"*/{kind}/*.csv")):
        safe, ds = path.stem.rsplit("-", 1)
        yield path, models.get(safe), ds


def append(path, header, rows):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        writer.writerows(rows)


def import_evals(models):
    counts = {"imported": 0, "dupes": 0, "skipped_files": 0}
    seen = {ds: {(r["path"], r["model"]) for r in read_csv(ROOT / "datasets" / ds / "eval-baseline.csv")}
            for ds in DATASETS}
    new = {ds: [] for ds in DATASETS}
    for path, model, ds in legacy_files("results", models):
        if ds not in DATASETS or model is None:
            counts["skipped_files"] += 1
            continue
        cols = labels(ds)
        for row in read_csv(path):
            rel = f"test/{Path(row['path']).name}"
            if (rel, model) in seen[ds]:
                counts["dupes"] += 1
                continue
            seen[ds].add((rel, model))
            prediction = parse_set(row["prediction"])
            new[ds].append([rel, model, *[str(c.upper() in prediction).lower() for c in cols],
                            row["text_rationale"]])
            counts["imported"] += 1
    for ds in DATASETS:
        if new[ds]:
            append(ROOT / "datasets" / ds / "eval-baseline.csv",
                   ["path", "model", *labels(ds), "rationale"], new[ds])
    return counts


def import_judgements(models):
    counts = {"imported": 0, "dupes": 0, "controls": 0, "orphans": 0, "skipped_files": 0}
    cache = {}
    new = {ds: [] for ds in DATASETS}
    for path, model, ds in legacy_files("agreement", models):
        if ds not in DATASETS or model is None:
            counts["skipped_files"] += 1
            continue
        if ds not in cache:
            folder = ROOT / "datasets" / ds
            cache[ds] = ({r["path"]: r for r in read_csv(folder / "labels.csv")},
                         {(r["path"], r["model"]): r for r in read_csv(folder / "eval-baseline.csv")},
                         {(r["path"], r["model"]) for r in read_csv(folder / "judge-baseline.csv")})
        truths, evals, done = cache[ds]
        cols = labels(ds)
        for row in read_csv(path):
            if row["control"].strip().lower() == "true":
                counts["controls"] += 1
                continue
            rel = f"test/{row['image']}"
            if (rel, model) in done:
                counts["dupes"] += 1
                continue
            pred, truth = evals.get((rel, model)), truths.get(rel)
            if pred is None or truth is None:
                counts["orphans"] += 1
                continue
            done.add((rel, model))
            correct = all(pred[c].strip().lower() == truth[c].strip().lower() for c in cols)
            agree = (row["same_conclusion"].strip().lower() == "true"
                     and row["same_evidence"].strip().lower() == "true")
            new[ds].append([rel, model, str(correct).lower(), str(agree).lower(), row["reason"]])
            counts["imported"] += 1
    for ds in DATASETS:
        if new[ds]:
            append(ROOT / "datasets" / ds / "judge-baseline.csv", JUDGE_HEADER, new[ds])
    return counts


def import_rationales():
    counts = {"filled": 0, "blocked_unassessed": 0, "join_misses": 0, "sources": 0}
    for ds in DATASETS:
        folder = ROOT / "datasets" / ds
        mapping = {}
        for name in ("rationales.csv", "rationales-test.csv"):
            for base in (BACKUP / "datasets" / ds, folder):
                for r in read_csv(base / name):
                    text = (r.get("ground_truth_rationales") or "").strip()
                    if text:
                        mapping.setdefault(r["image_path"], text)
        if not mapping:
            continue
        counts["sources"] += len(mapping)
        rows = read_csv(folder / "labels.csv")
        by_path = {r["path"]: r for r in rows}
        filled = 0
        for path, text in mapping.items():
            row = by_path.get(path)
            if row is None:
                counts["join_misses"] += 1
            elif not true_labels(row):
                counts["blocked_unassessed"] += 1
            elif not row[RATIONALE].strip():
                row[RATIONALE] = text
                filled += 1
        if filled:
            tmp = (folder / "labels.csv").with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(folder / "labels.csv")
        counts["filled"] += filled
    return counts


def main():
    set_field_limit()
    models = {safe_filename(m): m for m in config()["models"]}
    print("evals:", import_evals(models))
    print("judgements:", import_judgements(models))
    print("rationales:", import_rationales())
    for ds, images, sampled, rationales, evaled, judged in sync().execute(SUMMARY):
        print(f"{ds}: {images} images, {sampled} sampled, {rationales} rationales, "
              f"{evaled} evaled, {judged} judged")


if __name__ == "__main__":
    main()
