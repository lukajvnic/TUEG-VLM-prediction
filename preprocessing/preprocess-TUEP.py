from pathlib import Path

from render import generate

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "TUEP"
RAW = DATASET / "v3.1.0"
LABELS = ["epilepsy", "no_epilepsy"]
EVENTS = ()
CLASS_DIRS = {"epilepsy": "00_epilepsy", "no_epilepsy": "01_no_epilepsy"}


def select():
    return {edf: cls for cls, d in CLASS_DIRS.items() for edf in sorted((RAW / d).rglob("*.edf"))}


if __name__ == "__main__":
    classes = select()

    def labels_for(edf, start, stop):
        return {classes[edf]}

    generate(DATASET, sorted(classes), LABELS, labels_for, EVENTS)
