from pathlib import Path

from render import generate

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "TUEV"
RAW = DATASET / "v2.0.1" / "edf"
LABELS = ["bckg", "spsw", "gped", "pled", "eyem", "artf"]
EVENTS = set(LABELS) - {"bckg"}
CODES = {"1": "spsw", "2": "gped", "3": "pled", "4": "eyem", "5": "artf", "6": "bckg"}


def labels_for(edf, start, stop):
    rec = edf.with_suffix(".rec")
    if not rec.exists():
        return set()
    found = set()
    for line in rec.read_text(errors="ignore").splitlines():
        try:
            _, a, b, code = [x.strip() for x in line.split(",")]
        except ValueError:
            continue
        if code in CODES and float(a) < stop and float(b) > start:
            found.add(CODES[code])
    return found


if __name__ == "__main__":
    generate(DATASET, sorted(RAW.rglob("*.edf")), LABELS, labels_for, EVENTS)
