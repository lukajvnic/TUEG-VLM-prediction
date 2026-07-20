import ast
import csv
import json
import re
import sys
from pathlib import Path

max_csv_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_csv_field_size)
        break
    except OverflowError:
        max_csv_field_size = int(max_csv_field_size / 10)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def parse_set(value: str) -> set[str]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return set()

    if isinstance(parsed, set):
        return {str(item) for item in parsed}
    if isinstance(parsed, (list, tuple)):
        return {str(item) for item in parsed}
    if isinstance(parsed, str):
        return {parsed}
    return set()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def model_name_from_file(path: Path) -> str:
    # Files are written as <safe-model>-<dataset>.csv. Keep the full stem if the
    # dataset suffix cannot be identified.
    match = re.match(r"(.+)-TU[A-Z0-9]+$", path.stem)
    return match.group(1) if match else path.stem


def load_task_statuses(results_dir: Path) -> dict[tuple[str, str], str]:
    """Return each task's terminal state; absent status means still partial."""
    status_dir = PROJECT_ROOT / "logs" / results_dir.name / "task_status"
    statuses = {}
    if not status_dir.exists():
        return statuses

    for path in status_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            model, dataset = payload["model"], payload["dataset"]
            statuses[(model, dataset)] = payload.get("outcome", "partial")
        except (OSError, ValueError, KeyError):
            print(f"WARNING: could not read task status {path}", file=sys.stderr)
    return statuses


def run_status(outcome: str | None) -> str:
    if outcome == "completed":
        return "success"
    if outcome == "failed":
        return "fail"
    return "partial"


def summarize_file(path: Path, status: str) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    model = model_name_from_file(path)
    total = len(rows)
    exact_correct = sum(1 for row in rows if str(row.get("correct", "")).lower() == "true")
    exact_accuracy = safe_div(exact_correct, total)

    actuals = [parse_set(row.get("actual", "")) for row in rows]
    predictions = [parse_set(row.get("prediction", "")) for row in rows]
    labels = sorted(set().union(*actuals, *predictions)) if rows else []

    label_summaries = []
    for label in labels:
        tp = fp = tn = fn = 0
        for actual, prediction in zip(actuals, predictions):
            actual_positive = label in actual
            predicted_positive = label in prediction
            if actual_positive and predicted_positive:
                tp += 1
            elif not actual_positive and predicted_positive:
                fp += 1
            elif not actual_positive and not predicted_positive:
                tn += 1
            else:
                fn += 1

        sensitivity = safe_div(tp, tp + fn)  # recall / true positive rate
        specificity = safe_div(tn, tn + fp)  # true negative rate
        precision = safe_div(tp, tp + fp)
        f1 = safe_div(2 * precision * sensitivity, precision + sensitivity)
        label_accuracy = safe_div(tp + tn, tp + fp + tn + fn)
        balanced_accuracy = (sensitivity + specificity) / 2

        label_summaries.append({
            "model": model,
            "file": path.name,
            "label": label,
            "scope": "label",
            "run_status": status,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "support_positive": tp + fn,
            "support_negative": tn + fp,
            "total": total,
            "exact_correct": exact_correct,
            "exact_accuracy": exact_accuracy,
            "label_accuracy": label_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "sensitivity_recall": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
        })

    if label_summaries:
        aggregate = {
            "model": model,
            "file": path.name,
            "label": "ALL",
            "scope": "model",
            "run_status": status,
            "tp": sum(int(row["tp"]) for row in label_summaries),
            "fp": sum(int(row["fp"]) for row in label_summaries),
            "tn": sum(int(row["tn"]) for row in label_summaries),
            "fn": sum(int(row["fn"]) for row in label_summaries),
            "support_positive": sum(int(row["support_positive"]) for row in label_summaries),
            "support_negative": sum(int(row["support_negative"]) for row in label_summaries),
            "total": total,
            "exact_correct": exact_correct,
            "exact_accuracy": exact_accuracy,
            "label_accuracy": sum(float(row["label_accuracy"]) for row in label_summaries) / len(label_summaries),
            "balanced_accuracy": sum(float(row["balanced_accuracy"]) for row in label_summaries) / len(label_summaries),
            "sensitivity_recall": sum(float(row["sensitivity_recall"]) for row in label_summaries) / len(label_summaries),
            "specificity": sum(float(row["specificity"]) for row in label_summaries) / len(label_summaries),
            "precision": sum(float(row["precision"]) for row in label_summaries) / len(label_summaries),
            "f1": sum(float(row["f1"]) for row in label_summaries) / len(label_summaries),
        }
        return [aggregate, *label_summaries]

    return [{
        "model": model,
        "file": path.name,
        "label": "ALL",
        "scope": "model",
        "run_status": status,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "support_positive": 0,
        "support_negative": 0,
        "total": total,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_accuracy,
        "label_accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "sensitivity_recall": 0.0,
        "specificity": 0.0,
        "precision": 0.0,
        "f1": 0.0,
    }]


def main() -> None:
    results_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    if not results_dir.exists():
        raise SystemExit(f"ERROR: results directory does not exist: {results_dir}")

    csv_files = sorted(
        path for path in results_dir.glob("*.csv")
        if path.name != "summary.csv" and path.is_file()
    )

    task_statuses = load_task_statuses(results_dir)
    expected_files = {
        f"{safe_filename_part(model)}-{dataset}.csv": (model, dataset)
        for model, dataset in json.loads((results_dir / "tasks.json").read_text(encoding="utf-8"))
    } if (results_dir / "tasks.json").exists() else {}

    fieldnames = [
        "model",
        "file",
        "label",
        "scope",
        "run_status",
        "tp",
        "fp",
        "tn",
        "fn",
        "support_positive",
        "support_negative",
        "total",
        "exact_correct",
        "exact_accuracy",
        "label_accuracy",
        "balanced_accuracy",
        "sensitivity_recall",
        "specificity",
        "precision",
        "f1",
    ]

    rows = []
    processed_files = set()
    for path in csv_files:
        task = expected_files.get(path.name)
        outcome = task_statuses.get(task) if task else None
        rows.extend(summarize_file(path, run_status(outcome)))
        processed_files.add(path.name)

    # Include tasks that failed before creating a results CSV (or are still running).
    for filename, (model, dataset) in expected_files.items():
        if filename not in processed_files:
            rows.append({
                "model": model, "file": filename, "label": "ALL", "scope": "model",
                "run_status": run_status(task_statuses.get((model, dataset))),
                "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "support_positive": 0, "support_negative": 0, "total": 0,
                "exact_correct": 0, "exact_accuracy": 0.0, "label_accuracy": 0.0,
                "balanced_accuracy": 0.0, "sensitivity_recall": 0.0,
                "specificity": 0.0, "precision": 0.0, "f1": 0.0,
            })

    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} summary row(s) to {summary_path}")


if __name__ == "__main__":
    main()
