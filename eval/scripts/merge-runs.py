"""Union a base run with its retry run into a single scoreable run dir.

`create-retry-config.py` + `run-eval.py` put retried work in a *new* run dir, and
`resume-from` only tells the retry which images to skip -- it never copies the
base run's rows forward. So neither dir is scoreable alone: the base is missing
the retried images, and the retry holds only those images. The two are disjoint
by construction (that is what resume-from guarantees), so merging is a union
keyed on `path`.

    python eval/scripts/merge-runs.py <base-run> <retry-run> [--into NAME] [--dry-run]
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
STATUS_FIELDS = ("model", "dataset", "status", "reason", "total_images", "completed_images")


def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def merge_results(base_dir, retry_dir):
    """{filename: (fields, rows)} -- base rows first, retry rows appended."""
    merged = {}
    names = sorted({path.name for directory in (base_dir, retry_dir)
                    for path in (directory / "results").glob("*.csv")})

    for name in names:
        fields, rows, seen = None, [], set()
        duplicates = 0
        for directory in (base_dir, retry_dir):
            path = directory / "results" / name
            if not path.exists():
                continue
            source_fields, source_rows = read_rows(path)
            fields = fields or source_fields
            for row in source_rows:
                if row["path"] in seen:
                    duplicates += 1
                    continue
                seen.add(row["path"])
                rows.append(row)
        merged[name] = (fields, rows, duplicates)
    return merged


def merge_status(base_dir, retry_dir):
    """Retry outcome wins where it exists; base carries the rest."""
    statuses = {}
    for directory in (base_dir, retry_dir):
        path = directory / "status.csv"
        if not path.exists():
            continue
        _, rows = read_rows(path)
        for row in rows:
            statuses[row["model"], row["dataset"]] = row
    return statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("retry")
    parser.add_argument("--into", help="destination run name (default: <retry>-merged)")
    parser.add_argument("--dry-run", action="store_true", help="report and write nothing")
    args = parser.parse_args()

    set_csv_field_limit()
    base_dir, retry_dir = RUNS / args.base, RUNS / args.retry
    for directory in (base_dir, retry_dir):
        if not directory.is_dir():
            sys.exit(f"no such run: {directory}")

    destination = RUNS / (args.into or f"{args.retry}-merged")
    if destination.exists() and not args.dry_run:
        sys.exit(f"destination already exists: {destination}")

    merged = merge_results(base_dir, retry_dir)
    statuses = merge_status(base_dir, retry_dir)

    total = duplicates = 0
    for name, (_, rows, dupes) in merged.items():
        total += len(rows)
        duplicates += dupes
        base_count = len(read_rows(base_dir / "results" / name)[1]) if (base_dir / "results" / name).exists() else 0
        retry_count = len(read_rows(retry_dir / "results" / name)[1]) if (retry_dir / "results" / name).exists() else 0
        if retry_count:
            print(f"  {name}: {base_count} base + {retry_count} retry -> {len(rows)}"
                  f"{f' ({dupes} duplicate paths dropped)' if dupes else ''}")

    succeeded = sum(1 for row in statuses.values() if row["status"] == "success")
    print(f"\n{len(merged)} result files, {total} windows, {duplicates} duplicates dropped")
    print(f"status: {succeeded}/{len(statuses)} tasks successful")

    if args.dry_run:
        print(f"\n--dry-run: would write {destination}")
        return

    (destination / "results").mkdir(parents=True)
    for name, (fields, rows, _) in merged.items():
        with (destination / "results" / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    with (destination / "status.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(statuses.values())

    shutil.copy(retry_dir / "config.yml", destination / "config.yml")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
