# the pilot's per-class questions for TUSZ: one yes/no question per class plus "any seizure", each answered
# {"present": true|false}. Shared by build.py (training lines), train.py and score.py (scoring prompts)
import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers.pipeline import config  # noqa: E402

DATASET = "TUSZ"
NAMES = {"absz": "an absence seizure", "bckg": "background (non-seizure) activity", "cpsz": "a complex partial seizure",
         "fnsz": "a focal non-specific seizure", "gnsz": "a generalized non-specific seizure",
         "mysz": "a myoclonic seizure", "spsz": "a simple partial seizure", "tcsz": "a tonic-clonic seizure",
         "tnsz": "a tonic seizure"}
SEIZURES = [c for c in NAMES if c != "bckg"]
QUESTIONS = ["any_seizure", *NAMES]


@functools.cache
def intro():
    return config()["prompts"]["intro"]


def question_prompt(question):
    what = "a seizure of any type" if question == "any_seizure" else f"{NAMES[question]} ({question.upper()})"
    return f"{intro()}Based only on these waveforms, does this window show {what}? Set present to true only when it does."


def answer(value):
    return json.dumps({"present": value})


def truth(labels, question):
    # labels: the joint answer's booleans, {"has_absz": false, ...}
    if question == "any_seizure":
        return any(labels[f"has_{c}"] for c in SEIZURES)
    return labels[f"has_{question}"]
