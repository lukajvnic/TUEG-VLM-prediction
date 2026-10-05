# the pilot's two training sets, from the label-only TUSZ split (datasets/TUSZ/sft_labels_{train,val}.jsonl), so the
# windows and the patient split are the labels round's:
#   perclass:  one yes/no line per (window, question), every question balanced 50/50
#   joint-aux: the original joint lines with seizure vs seizure-free balanced 50/50, plus the perclass lines
# Both use the perclass val lines, so both pick their checkpoint on the same balanced metric. Deterministic (seed 0)
#   python side/multilabel-pilot/build.py
import json
import random
import sys
from collections import Counter
from pathlib import Path

SIDE = Path(__file__).resolve().parent
sys.path.insert(0, str(SIDE))
sys.path.insert(0, str(SIDE.parents[1]))
from helpers.pipeline import sft_file  # noqa: E402
from questions import DATASET, QUESTIONS, SEIZURES, answer, question_prompt, truth  # noqa: E402

SEED = 0
DATA = SIDE / "data"


def read(split):
    with sft_file(DATASET, split, "labels").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        row["labels"] = json.loads(row["output"])
    return rows


def question_line(row, question, value):
    return {"dataset": DATASET, "question": question, "instruction": question_prompt(question), "input": "",
            "output": answer(value), "images": row["images"]}


def balanced(rows, question, rng):
    # as many negatives as positives (or the reverse when negatives are fewer). A seizure type's negatives are half
    # other seizure types and half seizure-free windows, so the question has to tell types apart, not only seizure
    # from background
    pos = [r for r in rows if truth(r["labels"], question)]
    neg = [r for r in rows if not truth(r["labels"], question)]
    n = min(len(pos), len(neg))

    if question in SEIZURES:
        other = [r for r in neg if truth(r["labels"], "any_seizure")]
        calm = [r for r in neg if not truth(r["labels"], "any_seizure")]
        k = min(len(other), n // 2)
        chosen = rng.sample(other, k) + rng.sample(calm, n - k)
    else:
        chosen = rng.sample(neg, n)

    return [question_line(r, question, True) for r in rng.sample(pos, n)] + \
           [question_line(r, question, False) for r in chosen]


def joint(rows, rng):
    # the labels round's own lines, unchanged, with background-only windows cut to the number of seizure windows
    seizure = [r for r in rows if truth(r["labels"], "any_seizure")]
    calm = [r for r in rows if not truth(r["labels"], "any_seizure")]
    n = min(len(seizure), len(calm))
    keep = rng.sample(seizure, n) + rng.sample(calm, n)
    return [{**{k: v for k, v in r.items() if k != "labels"}, "question": "joint"} for r in keep]


def is_positive(line):
    labels = json.loads(line["output"])
    return truth(labels, "any_seizure") if line["question"] == "joint" else labels["present"]


def write(name, lines, rng):
    lines = list(lines)
    rng.shuffle(lines)
    with (DATA / f"{name}.jsonl").open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    counts = Counter((line["question"], is_positive(line)) for line in lines)
    questions = sorted({q for q, _ in counts}, key=lambda q: (q != "joint", q))
    print(f"{name}: {len(lines)} lines; " + ", ".join(f"{q} {counts[(q, True)]}/{counts[(q, False)]}" for q in questions))


def main():
    rng = random.Random(SEED)
    train, val = read("train"), read("val")
    perclass_train = [line for q in QUESTIONS for line in balanced(train, q, rng)]
    perclass_val = [line for q in QUESTIONS for line in balanced(val, q, rng)]
    joint_train = joint(train, rng)

    DATA.mkdir(exist_ok=True)
    print(f"from {len(train)} train / {len(val)} val windows; counts are true/false per question "
          "(joint: true = the window has a seizure)")
    write("perclass-train", perclass_train, rng)
    write("perclass-val", perclass_val, rng)
    write("joint-aux-train", joint_train + perclass_train, rng)
    write("joint-aux-val", perclass_val, rng)


if __name__ == "__main__":
    main()
