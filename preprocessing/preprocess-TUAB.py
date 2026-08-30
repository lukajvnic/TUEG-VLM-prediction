from pathlib import Path

from render import generate

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "TUAB"
RAW = DATASET / "v3.0.1" / "edf" / "train"
LABELS = ["abnormal", "normal"]
EVENTS = ()
PER_CLASS = 500


def select():
    return {edf: cls for cls in LABELS for edf in sorted((RAW / cls).rglob("*.edf"))[:PER_CLASS]}


if __name__ == "__main__":
    classes = select()

    def labels_for(edf, start, stop):
        return {classes[edf]}

    generate(DATASET, sorted(classes), LABELS, labels_for, EVENTS)
