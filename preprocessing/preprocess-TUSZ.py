from pathlib import Path

from render import csv_labels, generate

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "TUSZ"
RAW = DATASET / "v2.0.6" / "edf"
LABELS = ["bckg", "absz", "cpsz", "fnsz", "gnsz", "mysz", "spsz", "tcsz", "tnsz"]
EVENTS = set(LABELS) - {"bckg"}


def labels_for(edf, start, stop):
    return csv_labels([edf.with_suffix(".csv")], start, stop, LABELS) or {"bckg"}


if __name__ == "__main__":
    generate(DATASET, sorted(RAW.rglob("*.edf")), LABELS, labels_for, EVENTS)
