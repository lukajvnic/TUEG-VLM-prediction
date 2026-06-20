#!/usr/bin/env python3
"""Find the shortest and longest EDF files in the TUEV dataset.

This reads only each EDF header, so it does not require pyedflib/mne and is fast.
EDF duration is: number_of_data_records * duration_of_each_data_record.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, NamedTuple


class EdfInfo(NamedTuple):
    path: Path
    duration_seconds: float
    num_records: int
    record_duration_seconds: float


def iter_edf_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.edf"))


def _read_ascii_field(header: bytes, start: int, end: int) -> str:
    return header[start:end].decode("ascii", errors="ignore").strip()


def read_edf_duration(path: Path) -> EdfInfo:
    """Return duration info for one EDF file by parsing the fixed EDF header.

    EDF/EDF+ fixed header offsets:
      bytes 236:244 = number of data records
      bytes 244:252 = duration of one data record in seconds
    """
    with path.open("rb") as f:
        header = f.read(256)

    if len(header) < 256:
        raise ValueError("file is too small to contain a valid EDF header")

    num_records_text = _read_ascii_field(header, 236, 244)
    record_duration_text = _read_ascii_field(header, 244, 252)

    try:
        num_records = int(num_records_text)
        record_duration = float(record_duration_text)
    except ValueError as exc:
        raise ValueError(
            f"invalid EDF duration fields: records={num_records_text!r}, "
            f"record_duration={record_duration_text!r}"
        ) from exc

    if num_records < 0:
        raise ValueError("EDF has unknown number of data records (-1)")

    return EdfInfo(
        path=path,
        duration_seconds=num_records * record_duration,
        num_records=num_records,
        record_duration_seconds=record_duration,
    )


def format_duration(seconds: float) -> str:
    whole_seconds = int(seconds)
    hours, rem = divmod(whole_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    suffix = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    if seconds != whole_seconds:
        suffix += f" ({seconds:.6f}s)"
    else:
        suffix += f" ({whole_seconds}s)"
    return suffix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find shortest and longest .edf files under a TUEV directory."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="v2.0.1/edf",
        type=Path,
        help="Directory to search recursively (default: v2.0.1/edf)",
    )
    args = parser.parse_args()

    edf_files = list(iter_edf_files(args.root))
    if not edf_files:
        print(f"No .edf files found under {args.root}")
        return 1

    infos: list[EdfInfo] = []
    errors: list[tuple[Path, Exception]] = []

    for path in edf_files:
        try:
            infos.append(read_edf_duration(path))
        except Exception as exc:  # keep scanning even if one file is malformed
            errors.append((path, exc))

    if not infos:
        print("No valid EDF durations could be read.")
        return 1

    shortest = min(infos, key=lambda item: item.duration_seconds)
    longest = max(infos, key=lambda item: item.duration_seconds)

    print(f"Scanned {len(infos)} EDF files under {args.root}")
    if errors:
        print(f"Skipped {len(errors)} files with errors")

    print("\nShortest EDF:")
    print(f"  path: {shortest.path}")
    print(f"  length: {format_duration(shortest.duration_seconds)}")
    print(f"  records: {shortest.num_records}")
    print(f"  record duration: {shortest.record_duration_seconds}s")

    print("\nLongest EDF:")
    print(f"  path: {longest.path}")
    print(f"  length: {format_duration(longest.duration_seconds)}")
    print(f"  records: {longest.num_records}")
    print(f"  record duration: {longest.record_duration_seconds}s")

    if errors:
        print("\nErrors:")
        for path, exc in errors:
            print(f"  {path}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
