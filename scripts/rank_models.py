#!/usr/bin/env python3
"""Rank models by their mean balanced accuracy across datasets in a summary CSV."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

DATASET_RE = re.compile(r"-(TU[A-Z0-9]+)\.csv$")


def dataset_from_row(row: dict[str, str]) -> str | None:
    """Extract the dataset name from a summary row's source filename."""
    match = DATASET_RE.search(row["file"])
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank models by unweighted mean balanced accuracy across datasets."
    )
    parser.add_argument("summary", nargs="?", type=Path, default=Path("summary.csv"))
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include models without a valid result for every dataset (ranked on available datasets).",
    )
    parser.add_argument(
        "--output", type=Path, help="Optionally write the ranking to this CSV file."
    )
    args = parser.parse_args()

    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        datasets: set[str] = set()
        for row in rows:
            # The ALL/model rows contain each dataset's macro balanced accuracy.
            if row.get("scope") != "model" or row.get("label") != "ALL":
                continue
            dataset = dataset_from_row(row)
            if dataset is None:
                continue
            datasets.add(dataset)
            if int(row.get("total") or 0) > 0:
                scores[row["model"]][dataset] = float(row["balanced_accuracy"])

    if not datasets:
        raise SystemExit("ERROR: no dataset-level model rows found in the summary CSV")

    dataset_columns = sorted(datasets)
    ranking = []
    for model, model_scores in scores.items():
        complete = len(model_scores) == len(dataset_columns)
        if not complete and not args.include_incomplete:
            continue
        mean_score = sum(model_scores.values()) / len(model_scores)
        ranking.append((model, mean_score, model_scores, complete))

    ranking.sort(key=lambda entry: (-entry[1], entry[0]))
    if not ranking:
        raise SystemExit(
            "ERROR: no models have results for every dataset; use --include-incomplete to rank partial results"
        )

    output_rows = []
    for rank, (model, mean_score, model_scores, complete) in enumerate(ranking, start=1):
        result = {
            "rank": rank,
            "model": model,
            "mean_balanced_accuracy": f"{mean_score:.6f}",
            "datasets_scored": f"{len(model_scores)}/{len(dataset_columns)}",
            "complete": complete,
        }
        result.update({dataset: f"{model_scores.get(dataset, float('nan')):.6f}" for dataset in dataset_columns})
        output_rows.append(result)

    fieldnames = ["rank", "model", "mean_balanced_accuracy", "datasets_scored", "complete", *dataset_columns]
    writer_target = args.output.open("w", newline="", encoding="utf-8") if args.output else None
    try:
        writer = csv.DictWriter(writer_target or sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    finally:
        if writer_target:
            writer_target.close()


if __name__ == "__main__":
    main()
