import argparse
import ast
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from chart import write_bar_chart

CHANCE_LEVEL = 0.5


def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_labels(value):
    try:
        labels = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return set()
    return set(labels) if isinstance(labels, (set, list, tuple)) else set()


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0


def load_completed_tasks(run_dir):
    with (run_dir / "status.csv").open(newline="") as file:
        return [row for row in csv.DictReader(file) if row["status"] == "success"]


def load_results(run_dir, task):
    path = run_dir / "results" / f"{safe_filename(task['model'])}-{task['dataset']}.csv"
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def get_confusion(actuals, predictions, label):
    tp = fp = tn = fn = 0
    for actual, prediction in zip(actuals, predictions):
        if label in actual and label in prediction:
            tp += 1
        elif label in prediction:
            fp += 1
        elif label in actual:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def summarize_task(task, results):
    actuals = [parse_labels(row["actual"]) for row in results]
    predictions = [parse_labels(row["prediction"]) for row in results]
    labels = set().union(*actuals, *predictions) if results else set()
    confusion_matrices = [get_confusion(actuals, predictions, label) for label in labels]
    tp, fp, tn, fn = (sum(values) for values in zip(*confusion_matrices)) if labels else (0, 0, 0, 0)
    balanced_accuracies = [
        (safe_divide(label_tp, label_tp + label_fn) + safe_divide(label_tn, label_tn + label_fp)) / 2
        for label_tp, label_fp, label_tn, label_fn in confusion_matrices
    ]

    return {
        "model": task["model"],
        "dataset": task["dataset"],
        "accuracy": safe_divide(sum(row["correct"].lower() == "true" for row in results), len(results)),
        "balanced_accuracy": safe_divide(sum(balanced_accuracies), len(balanced_accuracies)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_models(summary):
    scores = defaultdict(list)
    for row in summary:
        scores[row["model"]].append(row["balanced_accuracy"])

    ranking = []
    for model, values in scores.items():
        ranking.append({"model": model, "balanced_accuracy": sum(values) / len(values), "datasets": len(values)})
    ranking.sort(key=lambda row: (-row["balanced_accuracy"], row["model"]))
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking


def describe_ranking(ranking):
    counts = {row["datasets"] for row in ranking} or {0}
    span = str(max(counts)) if len(counts) == 1 else f"up to {max(counts)}"
    return (
        f"{len(ranking)} models · mean window-level balanced accuracy across {span} datasets"
        " · quick leaderboard, not the reportable metric"
    )


def write_rank_chart(path, ranking, run):
    write_bar_chart(
        path,
        labels=[row["model"] for row in ranking],
        values=[row["balanced_accuracy"] for row in ranking],
        title=f"Model ranking — {run}",
        subtitle=describe_ranking(ranking),
        y_label="Balanced accuracy",
        reference=(CHANCE_LEVEL, f"chance {CHANCE_LEVEL:.2f}"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()

    set_csv_field_limit()
    run_dir = Path(__file__).parent / "runs" / args.run
    summary = [summarize_task(task, load_results(run_dir, task)) for task in load_completed_tasks(run_dir)]
    ranking = rank_models(summary)
    write_csv(run_dir / "summary.csv", ("model", "dataset", "accuracy", "balanced_accuracy", "tp", "fp", "tn", "fn"), summary)
    write_csv(run_dir / "rank.csv", ("rank", "model", "balanced_accuracy", "datasets"), ranking)
    write_rank_chart(run_dir / "rank.png", ranking, args.run)


if __name__ == "__main__":
    main()
