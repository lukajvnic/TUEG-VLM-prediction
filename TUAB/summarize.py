#!/usr/bin/env python3
"""Summarize TUAB batch classification results.

Reads a CSV with at least `actual` and `predicted` columns and prints a
confusion matrix plus accuracy. By default it summarizes
`gemini-3.5-flash_batch_results_TUAB.csv`.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_CSV = "gemini-3.5-flash_batch_results_TUAB.csv"
POSITIVE_LABEL = "abnormal"
NEGATIVE_LABEL = "normal"


def normalize_label(value: str | None) -> str:
    """Normalize labels for consistent counting."""
    return (value or "").strip().lower()


def summarize(csv_path: Path) -> tuple[Counter[tuple[str, str]], int, int]:
    """Return (confusion counts, total rows, correct rows)."""
    counts: Counter[tuple[str, str]] = Counter()
    total = 0
    correct = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"actual", "predicted"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            actual = normalize_label(row.get("actual"))
            predicted = normalize_label(row.get("predicted"))
            if not actual or not predicted:
                continue

            counts[(actual, predicted)] += 1
            total += 1
            correct += actual == predicted

    return counts, total, correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute confusion matrix and accuracy for TUAB results CSV.")
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV, help=f"CSV file to summarize (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    counts, total, correct = summarize(csv_path)

    true_positive = counts[(POSITIVE_LABEL, POSITIVE_LABEL)]
    false_negative = counts[(POSITIVE_LABEL, NEGATIVE_LABEL)]
    false_positive = counts[(NEGATIVE_LABEL, POSITIVE_LABEL)]
    true_negative = counts[(NEGATIVE_LABEL, NEGATIVE_LABEL)]
    accuracy = correct / total if total else 0.0

    summary_path = Path("summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["file", str(csv_path)])
        writer.writerow(["total_evaluated", total])
        writer.writerow(["correct", correct])
        writer.writerow(["accuracy", f"{accuracy:.4f}"])
        writer.writerow(["accuracy_percent", f"{accuracy * 100:.2f}"])
        writer.writerow(["true_positive", true_positive])
        writer.writerow(["true_negative", true_negative])
        writer.writerow(["false_positive", false_positive])
        writer.writerow(["false_negative", false_negative])
        writer.writerow([])
        writer.writerow(["confusion_matrix", "pred_normal", "pred_abnormal"])
        writer.writerow(["actual_normal", true_negative, false_positive])
        writer.writerow(["actual_abnormal", false_negative, true_positive])

    print(f"Wrote summary to {summary_path}")

    other_counts = {
        pair: value
        for pair, value in counts.items()
        if pair[0] not in {NEGATIVE_LABEL, POSITIVE_LABEL} or pair[1] not in {NEGATIVE_LABEL, POSITIVE_LABEL}
    }
    if other_counts:
        print()
        print("Other label pairs found:")
        for (actual, predicted), value in sorted(other_counts.items()):
            print(f"actual={actual!r}, predicted={predicted!r}: {value}")


if __name__ == "__main__":
    main()
