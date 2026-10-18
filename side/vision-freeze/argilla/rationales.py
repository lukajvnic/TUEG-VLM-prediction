# expert review of rationales in Argilla: per corpus, 50 test windows (25 with the class, 25 without) from the
# rationale evaluation of that corpus's best fine-tuned model. Each record shows the window, the question, the TUH
# label and two rationales in random order, the fine-tuned model's and the teacher's (gemma3:12b, label-stripped as
# in training). Annotators pick the better one, rate both and may write a correction: a human score for the paper
# and the first preference pairs for DPO. These are test windows, so their pairs must never enter DPO training;
# training-window pairs need a generation pass over train windows first
#   python side/vision-freeze/argilla/rationales.py build LABELS_ROOT records.jsonl  # .venv-rehearsal (imports run.py)
#   python side/vision-freeze/argilla/rationales.py push records.jsonl              # a venv with argilla 2.x
# push reads ARGILLA_API_URL, ARGILLA_API_KEY and HF_TOKEN (the Space is private)
import base64
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

SIDE = Path(__file__).resolve().parents[1]
PER_CLASS = 25
BEST = {"TUSZ": ("g-8b-tusz", "answer-first"), "TUAB": ("gm-gemma4-e4b-tuab", "explain"),
        "TUAR": ("gm-medgemma15-4b-tuar", "explain"), "TUEV": ("gm-granite-2b-tuev", "explain")}
GUIDELINES = """Each record is a 20-second EEG window: 25 rows, TUH bipolar montage (left chains above right, then the
transverse chain and EKG), time left to right with a grid line every second, and a 100 µV bar on the right. The label
is TUH's annotation.

Read both rationales and judge them against what you see in the window.

- preference: which rationale better describes the evidence in this window (A, B, tie, or both wrong).
- rating A / rating B: 5 = accurate, names the right channels, timing and waveform features; 3 = partly right or
  vague; 1 = wrong, or generic text that would fit any window.
- grounded: does the better rationale point to the right channels and time?
- correction (optional): a short rationale you would accept.

Rationale order is random. One rationale comes from a fine-tuned model, the other from the model that taught it."""


def read(path):
    csv.field_size_limit(10 ** 9)
    with path.open() as f:
        return list(csv.DictReader(f))


def build(labels_root, out):
    sys.argv = ["run.py"]
    spec = importlib.util.spec_from_file_location("run", SIDE / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    rng, records = random.Random(0), []
    for ds, (arm, fmt) in BEST.items():
        teacher = {r["path"]: r["ground_truth_rationale"] for r in read(labels_root / ds / "labels.csv")}
        rows = read(SIDE / "results" / f"rationale-{arm}-{fmt}.csv")
        picks = [rng.sample([r for r in rows if r["seizure"] == value], PER_CLASS) for value in ("true", "false")]
        for r in picks[0] + picks[1]:
            try:
                model_text = json.loads(r["reply"])["text_rationale"]
            except (json.JSONDecodeError, KeyError):
                model_text = r["reply"]
            teacher_text = run.CHANNEL.sub(r"\1", run.strip_label(teacher[r["path"]], ds))
            model_first = rng.random() < 0.5
            records.append({
                "dataset": ds, "id": f"{ds}-{Path(r['path']).stem}", "image": str(SIDE / "images-core" / ds / r["path"]),
                "question": "Does this window show " + run.prompt_text(arm).split("does this window show ", 1)[1].split("?")[0] + "?",
                "label": "present" if r["seizure"] == "true" else "absent",
                "rationale_a": model_text if model_first else teacher_text,
                "rationale_b": teacher_text if model_first else model_text,
                "a_source": "model" if model_first else "teacher", "model": arm})
    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"{len(records)} records ({PER_CLASS * 2} per corpus) -> {out}")


def push(path):
    import argilla as rg

    client = rg.Argilla(api_url=os.environ["ARGILLA_API_URL"], api_key=os.environ["ARGILLA_API_KEY"],
                        headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
    records = [json.loads(line) for line in path.open()]
    for ds in BEST:
        settings = rg.Settings(
            guidelines=GUIDELINES,
            fields=[rg.ImageField(name="eeg", title="EEG window"), rg.TextField(name="question", title="Question"),
                    rg.TextField(name="label", title="TUH label"), rg.TextField(name="rationale_a", title="Rationale A"),
                    rg.TextField(name="rationale_b", title="Rationale B")],
            questions=[rg.LabelQuestion(name="preference", title="Which rationale better explains this window?",
                                        labels=["A", "B", "tie", "both wrong"]),
                       rg.RatingQuestion(name="rating_a", title="Rating, rationale A", values=[1, 2, 3, 4, 5]),
                       rg.RatingQuestion(name="rating_b", title="Rating, rationale B", values=[1, 2, 3, 4, 5]),
                       rg.LabelQuestion(name="grounded", title="Does the better rationale point to the right channels "
                                        "and time?", labels=["yes", "partly", "no"]),
                       rg.TextQuestion(name="correction", title="Correction (optional)", required=False)],
            metadata=[rg.TermsMetadataProperty(name="label"),
                      rg.TermsMetadataProperty(name="a_source", visible_for_annotators=False),
                      rg.TermsMetadataProperty(name="model", visible_for_annotators=False)],
        )
        dataset = rg.Dataset(name=f"{ds.lower()}-rationales", workspace="argilla", settings=settings, client=client)
        dataset.create()
        mine = [r for r in records if r["dataset"] == ds]
        dataset.records.log([rg.Record(
            id=r["id"],
            fields={"eeg": "data:image/png;base64," + base64.b64encode(Path(r["image"]).read_bytes()).decode(),
                    "question": r["question"], "label": r["label"], "rationale_a": r["rationale_a"],
                    "rationale_b": r["rationale_b"]},
            metadata={"label": r["label"], "a_source": r["a_source"], "model": r["model"]}) for r in mine])
        print(f"{dataset.name}: {len(mine)} records")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        push(Path(sys.argv[2]))
