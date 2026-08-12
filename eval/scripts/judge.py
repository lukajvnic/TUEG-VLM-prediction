"""Judge whether a model's zero-shot rationale says the same thing as the reference.

For every test window the benchmark scored there are two descriptions of the same
waveform plot:

  * the **reference**, written by the gemma3:12b teacher which was told the true
    label (datasets/<DS>/rationales-test.csv, from generate-rationales.py --split test);
  * the **candidate**, written by the benchmarked model, which was not
    (eval/runs/<run>/results/<model>-<dataset>.csv, column text_rationale).

A text-only judge decides whether they agree. One array task per model, iterating
that model's datasets, so the judge's weights load once per ~14.8k pairs instead
of once per dataset.

Usage:  python scripts/judge.py --run <run> --model <model> [--dataset DS ...] [--limit N]
"""

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from sample import spread


# Retrying helps when the server was momentarily unreachable; it never helps when
# the model returned well-formed JSON that does not match the schema. Same
# reasoning (and the same constants) as eval.py.
TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)
RETRY_SLEEP_SECONDS = 10
PROGRESS_EVERY = 100

FIELDNAMES = (
    "image", "control", "verdict", "same_conclusion", "same_evidence",
    "correct", "prediction", "actual", "reason",
)

JUDGE_PROMPT = """Two descriptions of the SAME multi-channel EEG waveform plot.

REFERENCE -- written by a reader who was told the correct label:
{reference}

CANDIDATE -- written by a model that was not told the label:
{candidate}

Judge only what the two texts actually say.

same_conclusion: does the candidate reach the same overall assessment as the reference -- the same finding(s) reported present, and none of the reference's findings denied? Ignore wording, ordering and hedging.

same_evidence: does the candidate point at the same waveform evidence -- the same channel(s), the same region of the time axis, and the same morphology (shape, frequency, amplitude)? Overlap on the reference's main finding counts as true. A different channel, a different part of the recording, or a different morphology counts as false.

A confident tone is not correctness. A generic description that would fit any EEG recording is not matching evidence -- mark same_evidence false for it."""


class AgreementVerdict(BaseModel):
    same_conclusion: bool = Field(
        description="True if both texts report the same finding(s) present or absent."
    )
    same_evidence: bool = Field(
        description="True if both texts cite the same channels, time region and morphology."
    )
    reason: str = Field(description="At most 15 words.")


def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def eval_dir():
    return Path(__file__).resolve().parents[1]


def load_judge_config():
    # Read live, not from the run's frozen config.yml: a sweep submitted before
    # this stage existed has no judge settings in its copy, and judge.yml is
    # never rewritten by create-retry-config.py.
    with (eval_dir() / "judge.yml").open() as file:
        return yaml.safe_load(file)["judge"]


# ------------------------------------------------------------------- pairing


def recording_of(image):
    """`aaaaaajy_0_12.png` -> `aaaaaajy_0` (patient + scan)."""
    patient, scan, _ = Path(image).stem.rsplit("_", 2)
    return f"{patient}_{scan}"


def load_results(run_dir, model, dataset):
    path = run_dir / "results" / f"{safe_filename(model)}-{dataset}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_reference(dataset):
    """{image basename: reference rationale} for one dataset.

    Keyed on the basename because results CSVs hold resolved absolute paths while
    the rationales file holds dataset-relative ones; basenames are unique within
    a dataset (<patient>_<scan>_<window>.png).
    """
    path = Path(__file__).resolve().parents[2] / "datasets" / dataset / "rationales-test.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {
            Path(row["image_path"]).name: row["ground_truth_rationales"].strip()
            for row in csv.DictReader(file)
            if row.get("ground_truth_rationales", "").strip()
        }


def clip(text, limit):
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def build_pairs(run_dir, model, dataset, config, limit=None):
    """Join a model's rationales to the references, then append control pairs."""
    reference = load_reference(dataset)
    chars = config["max-rationale-chars"]

    pairs = []
    for row in load_results(run_dir, model, dataset):
        image = Path(row["path"]).name
        candidate = row.get("text_rationale", "").strip()
        if not candidate or image not in reference:
            continue
        pairs.append({
            "key": image,
            "image": image,
            "candidate": clip(candidate, chars),
            "reference": clip(reference[image], chars),
            "correct": row.get("correct", ""),
            "prediction": row.get("prediction", ""),
            "actual": row.get("actual", ""),
            "control": False,
        })

    pairs.sort(key=lambda pair: pair["image"])
    if limit:
        # Even spacing rather than the front of the list, so a limited run still
        # covers the whole recording set. Same rule as sample.py.
        pairs = spread(pairs, limit)
    return pairs + build_controls(pairs, config["control-fraction"])


def build_controls(pairs, fraction):
    """Pairs deliberately mismatched to a different recording's reference.

    The judge's agreement rate on these is its false-agreement floor. Selection
    and partnering are both positional, so the control set is reproducible.
    """
    if not fraction or len(pairs) < 2:
        return []

    step = max(round(1 / fraction), 1)
    offset = max(len(pairs) // 2, 1)
    controls = []
    for index in range(0, len(pairs), step):
        pair = pairs[index]
        for shift in range(len(pairs)):
            other = pairs[(index + offset + shift) % len(pairs)]
            if recording_of(other["image"]) != recording_of(pair["image"]):
                controls.append({
                    **pair,
                    "key": f"{pair['image']}#control",
                    "reference": other["reference"],
                    "control": True,
                })
                break
    return controls


# ------------------------------------------------------------------ judging


def init_judge(config):
    kwargs = {
        "model": config["model"],
        "temperature": 0,
        "num_ctx": config["num-ctx"],
        "num_predict": config["num-predict"],
    }
    if base_url := os.environ.get("OLLAMA_BASE_URL"):
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs).with_structured_output(
        AgreementVerdict, method="json_schema", include_raw=True
    )


def verdict_of(parsed):
    if parsed.same_conclusion and parsed.same_evidence:
        return "agree"
    if parsed.same_conclusion or parsed.same_evidence:
        return "partial"
    return "disagree"


def judge_pair(judge, pair, retries):
    prompt = JUDGE_PROMPT.format(reference=pair["reference"], candidate=pair["candidate"])
    for attempt in range(retries + 1):
        try:
            parsed = judge.invoke(prompt)["parsed"]
            if parsed is None:
                raise ValueError("Structured output could not be parsed")
            return {
                "image": pair["image"],
                "control": pair["control"],
                "verdict": verdict_of(parsed),
                "same_conclusion": parsed.same_conclusion,
                "same_evidence": parsed.same_evidence,
                "correct": pair["correct"],
                "prediction": pair["prediction"],
                "actual": pair["actual"],
                "reason": parsed.reason.strip(),
            }, None
        except Exception as error:
            if attempt == retries:
                return None, error
            if isinstance(error, TRANSIENT_ERRORS):
                time.sleep(RETRY_SLEEP_SECONDS)


def init_output(run_dir, model, dataset):
    directory = run_dir / "agreement"
    directory.mkdir(exist_ok=True)
    path = directory / f"{safe_filename(model)}-{dataset}.csv"
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()
    return path


def completed_keys(path):
    with path.open(newline="", encoding="utf-8") as file:
        return {
            f"{row['image']}#control" if row["control"] == "True" else row["image"]
            for row in csv.DictReader(file)
        }


def judge_dataset(judge, pairs, output, config, dataset):
    """Judge `pairs`, appending each verdict as it lands. Returns (ok, failed)."""
    ok = failed = 0
    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        with ThreadPoolExecutor(max_workers=config["parallel-requests"]) as executor:
            futures = [
                executor.submit(judge_pair, judge, pair, config["judge-retries"])
                for pair in pairs
            ]
            for future in as_completed(futures):
                row, error = future.result()
                if error is not None:
                    # A pair that failed stays out of the CSV, so a resubmit retries it.
                    failed += 1
                    print(f"{dataset} FAIL: {error}", file=sys.stderr, flush=True)
                    continue
                writer.writerow(row)
                ok += 1
                if (ok + failed) % PROGRESS_EVERY == 0:
                    file.flush()
                    print(f"[{ok + failed}/{len(pairs)}] {dataset}", flush=True)
    return ok, failed


# ----------------------------------------------------------------------- cli


def load_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory name under eval/runs/")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", action="append", dest="datasets",
                        help="repeatable; defaults to every dataset this model has results for")
    parser.add_argument("--limit", type=int, help="judge at most this many pairs per dataset")
    return parser.parse_args()


def datasets_for(run_dir, model):
    prefix = f"{safe_filename(model)}-"
    return sorted(
        path.stem.removeprefix(prefix)
        for path in (run_dir / "results").glob(f"{prefix}*.csv")
    )


def main():
    set_csv_field_limit()
    args = load_args()
    config = load_judge_config()
    run_dir = eval_dir() / "runs" / args.run
    datasets = args.datasets or datasets_for(run_dir, args.model)

    judge = init_judge(config)
    totals = {"pairs": 0, "judged": 0, "failed": 0, "skipped": 0}
    for dataset in datasets:
        pairs = build_pairs(run_dir, args.model, dataset, config, args.limit)
        output = init_output(run_dir, args.model, dataset)
        done = completed_keys(output)
        pending = [pair for pair in pairs if pair["key"] not in done]
        print(
            f"{args.model} {dataset}: {len(pairs)} pairs, {len(pairs) - len(pending)} already judged",
            flush=True,
        )
        ok, failed = judge_dataset(judge, pending, output, config, dataset)
        totals["pairs"] += len(pairs)
        totals["judged"] += ok
        totals["failed"] += failed
        totals["skipped"] += len(pairs) - len(pending)

    print(
        f"{args.model}: {totals['pairs']} pairs, {totals['judged']} judged, "
        f"{totals['skipped']} resumed, {totals['failed']} failed",
        flush=True,
    )
    # Exit 0 even with failures: they stay out of the CSV and a resubmit retries
    # them, exactly like a partially-failed eval task.


if __name__ == "__main__":
    main()
