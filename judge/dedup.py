import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import DATASETS, ROOT


def main():
    ap = argparse.ArgumentParser(description="Drop duplicate (path, model) rows, keeping the first.")
    ap.add_argument("--file", default="judge-gpt.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for ds in DATASETS:
        path = ROOT / "datasets" / ds / args.file
        if not path.exists():
            print(f"{ds}: no {args.file}, skipped")
            continue
        with path.open(newline="") as f:
            header, *body = csv.reader(f)
        seen, kept = set(), []
        for row in body:
            if (row[0], row[1]) not in seen:
                seen.add((row[0], row[1]))
                kept.append(row)
        dropped = len(body) - len(kept)
        action = "dry run" if args.dry_run else ("rewritten" if dropped else "unchanged")
        print(f"{ds}: {len(body)} rows, {dropped} duplicates ({action})")
        if args.dry_run or not dropped:
            continue
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(kept)
        tmp.replace(path)


if __name__ == "__main__":
    main()
