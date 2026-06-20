import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


DATASETS = ("TUEP", "TUAB", "TUEV", "TUAR", "TUSZ", "TUSL")
REPORT_COLUMN = "clinician_text_path"
EXCLUDED_DIRS = {".venv", "__pycache__", "DOCS"}
EXCLUDED_FILENAMES = {"AAREADME.txt", "requirements.txt"}


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def is_report_file(path: Path) -> bool:
    if path.name in EXCLUDED_FILENAMES:
        return False
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    return path.suffix.lower() == ".txt"


def build_report_index(dataset_dir: Path) -> dict[str, list[Path]]:
    index = defaultdict(list)

    for report_path in dataset_dir.rglob("*.txt"):
        if not is_report_file(report_path):
            continue

        relative_path = report_path.relative_to(dataset_dir)
        keys = {
            report_path.stem,
            normalize_key(report_path.stem),
        }

        for key in keys:
            index[key].append(relative_path)

    return index


def image_candidate_keys(image_path: str) -> list[str]:
    stem = Path(image_path).stem
    candidates = [stem]

    if "__" in stem:
        parts = stem.split("__")
        candidates.extend(parts)
        candidates.append(parts[-1])

    # Common TUH recording name pattern: patient_session_token, e.g. aaaaaaju_s005_t000.
    recording_match = re.search(r"([a-z]+_s\d+_t\d+)", stem, flags=re.IGNORECASE)
    if recording_match:
        recording = recording_match.group(1)
        candidates.append(recording)
        candidates.append(re.sub(r"_t\d+$", "", recording))

    # Common session pattern: patient_session_year, e.g. aaaaaajy__s001_2003.
    session_match = re.search(r"([a-z]+)__?(s\d+_\d{4})", stem, flags=re.IGNORECASE)
    if session_match:
        candidates.append("_".join(session_match.groups()))

    normalized = []
    seen = set()
    for candidate in candidates:
        for key in (candidate, normalize_key(candidate)):
            if key and key not in seen:
                normalized.append(key)
                seen.add(key)

    return normalized


def find_report(image_path: str, report_index: dict[str, list[Path]]) -> str:
    matches = []

    for key in image_candidate_keys(image_path):
        matches.extend(report_index.get(key, []))

    unique_matches = sorted({match.as_posix() for match in matches})
    if len(unique_matches) == 1:
        return unique_matches[0]

    return ""


def associate_dataset(dataset_dir: Path, overwrite: bool, dry_run: bool) -> tuple[int, int]:
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.exists():
        print(f"Skipping {dataset_dir}: no labels.csv")
        return 0, 0

    report_index = build_report_index(dataset_dir)

    with labels_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if REPORT_COLUMN not in fieldnames:
        fieldnames.append(REPORT_COLUMN)

    associated_count = 0
    for row in rows:
        if row.get(REPORT_COLUMN) and not overwrite:
            associated_count += 1
            continue

        report_path = find_report(row["image_path"], report_index)
        row[REPORT_COLUMN] = report_path
        if report_path:
            associated_count += 1

    if not dry_run:
        backup_path = labels_path.with_suffix(".csv.bak")
        if not backup_path.exists():
            backup_path.write_text(labels_path.read_text(encoding="utf-8"), encoding="utf-8")

        with labels_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(
        f"{dataset_dir.name}: associated {associated_count}/{len(rows)} rows "
        f"using {sum(len(paths) for paths in report_index.values())} report index entries"
    )
    return associated_count, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Add clinician_text_path associations to dataset labels.csv files."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=DATASETS,
        help="Dataset directories to process. Defaults to all known datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing clinician_text_path values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing labels.csv.",
    )
    args = parser.parse_args()

    total_associated = 0
    total_rows = 0
    for dataset in args.datasets:
        associated, rows = associate_dataset(Path(dataset), args.overwrite, args.dry_run)
        total_associated += associated
        total_rows += rows

    print(f"Total: associated {total_associated}/{total_rows} rows")


if __name__ == "__main__":
    main()
