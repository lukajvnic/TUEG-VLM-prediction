from pathlib import Path

from render import csv_labels, generate

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "TUAR"
RAW = DATASET / "v3.0.1" / "edf"
LABELS = ["bckg", "chew", "elec", "elpp", "eyem", "musc", "shiv"]
EVENTS = set(LABELS) - {"bckg"}


def labels_for(edf, start, stop):
    return csv_labels([edf.with_suffix(".csv")], start, stop, LABELS)


if __name__ == "__main__":
    generate(DATASET, sorted(RAW.rglob("*.edf")), LABELS, labels_for, EVENTS)
