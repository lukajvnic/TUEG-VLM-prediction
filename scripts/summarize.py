import ast
import csv
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


def model_name_from_file(path: Path) -> str:
    # Files are written as <safe-model>-<dataset>.csv. Keep the full stem if the
    # dataset suffix cannot be identified.
    match = re.match(r"(.+)-TU[A-Z0-9]+$", path.stem)
    return match.group(1) if match else path.stem


def summarize_file(path: Path) -> list[dict[str, object]]:
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

    fieldnames = [
        "model",
        "file",
        "label",
        "scope",
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
    for path in csv_files:
        rows.extend(summarize_file(path))

    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} summary row(s) to {summary_path}")


if __name__ == "__main__":
    main()
