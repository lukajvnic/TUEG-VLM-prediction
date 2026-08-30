from pathlib import Path

from render import csv_labels, generate

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "TUSL"
RAW = DATASET / "v2.0.1" / "edf"
LABELS = ["bckg", "seiz", "slow"]
EVENTS = set(LABELS) - {"bckg"}


def labels_for(edf, start, stop):
    files = [edf.with_suffix(".csv"), *edf.parent.glob(edf.stem + "_*.csv")]
    return csv_labels(files, start, stop, LABELS)


if __name__ == "__main__":
    generate(DATASET, sorted(RAW.rglob("*.edf")), LABELS, labels_for, EVENTS)
