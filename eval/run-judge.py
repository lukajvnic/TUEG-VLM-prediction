"""Rationale-agreement stage: does a model's zero-shot rationale say the same
thing as the reference rationale written for the same window?

    python eval/run-judge.py check  <run>   # join coverage, no GPU -- run this first
    python eval/run-judge.py submit <run>   # one array task per model
    python eval/run-judge.py report <run>   # aggregate to CSV

`check` is the pre-flight: it reports, per model x dataset, how many of the
evaluated windows actually have a reference rationale. A low number means the
teacher pass has not run or has not finished --

    sbatch --export=ALL,SPLIT=test eval/scripts/generate-rationales.sbatch

-- and queueing 35 judge tasks against a missing reference set would waste the
whole allocation.

`report` writes two files into the run directory:

  rationale-agreement.csv           one row per model x dataset
  rationale-agreement-by-model.csv  one row per model, macro-averaged over its
                                    datasets so TUSZ does not dominate

The headline column is `agreement_rate`. `agreement_rate_correct` is the one
worth reading next to it: overall agreement conflates "the model predicted the
wrong label, so of course its rationale differs" with "the model predicted the
right label for the wrong reasons". Restricting to correctly-labelled windows
separates right-for-the-right-reason from right-by-luck, which is the question a
capability probe exists to answer.

`control_agreement_rate` is the floor: the same judge, on the same candidate
rationales, paired against a *different* recording's reference. A model whose
agreement rate does not clear its own control rate is producing EEG boilerplate
that would match any plot.
"""

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml


DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
EVAL_DIR = Path(__file__).resolve().parent

TASK_FIELDS = (
    "model", "dataset", "pairs", "agree", "partial", "disagree",
    "agreement_rate", "lenient_rate",
    "pairs_correct", "agreement_rate_correct",
    "pairs_incorrect", "agreement_rate_incorrect",
    "control_pairs", "control_agreement_rate",
)

MODEL_FIELDS = (
    "rank", "model", "datasets", "pairs",
    "agreement_rate", "agreement_rate_correct", "lenient_rate",
    "control_agreement_rate", "clears_control",
)


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


def rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else 0.0


def mean(values):
    return round(sum(values) / len(values), 4) if values else 0.0


def load_judge_config():
    with (EVAL_DIR / "config.yml").open() as file:
        return yaml.safe_load(file)["judge"]


def run_dir_for(run):
    path = EVAL_DIR / "runs" / run
    if not path.is_dir():
        raise SystemExit(f"No such run: {path}")
    return path


def model_store():
    root = os.environ.get("OLLAMA_MODELS")
    if not root and (scratch := os.environ.get("SCRATCH")):
        root = f"{scratch}/ollama/models"
    return Path(root) if root else None


def judge_model_staged(model):
    """True/False if the store can be inspected, None if there is no store here.

    Compute nodes have no route to the Ollama registry, so a judge model missing
    from $OLLAMA_MODELS cannot be fetched at task start -- all 35 array tasks die
    on it. Reading the manifest path needs no server and no apptainer, so this
    works from a login node, which is where submission happens anyway.
    """
    store = model_store()
    if store is None or not store.is_dir():
        return None
    name, _, tag = model.partition(":")
    if "/" not in name:
        name = f"library/{name}"
    return (store / "manifests" / "registry.ollama.ai" / name / (tag or "latest")).is_file()


def load_tasks(run_dir):
    """Successful (model, dataset) pairs from the run's status.csv."""
    with (run_dir / "status.csv").open(newline="", encoding="utf-8") as file:
        return [
            (row["model"], row["dataset"])
            for row in csv.DictReader(file)
            if row["status"] == "success"
        ]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------- check


def reference_names(dataset):
    path = DATASETS_DIR / dataset / "rationales-test.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as file:
        return {
            Path(row["image_path"]).name
            for row in csv.DictReader(file)
            if row.get("ground_truth_rationales", "").strip()
        }


def evaluated_names(run_dir, model, dataset):
    path = run_dir / "results" / f"{safe_filename(model)}-{dataset}.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as file:
        return {
            Path(row["path"]).name
            for row in csv.DictReader(file)
            if row.get("text_rationale", "").strip()
        }


def report_judge_model():
    """Print whether the configured judge is staged. Returns True if it is absent."""
    model = load_judge_config()["model"]
    staged = judge_model_staged(model)
    if staged is None:
        print(f"judge model {model} -- no Ollama store here, not checked\n")
        return False
    if staged:
        print(f"judge model {model} -- staged in {model_store()}\n")
        return False
    print(f"!! judge model {model} is NOT in {model_store()}")
    print("   Compute nodes cannot reach the registry, so every judge task would")
    print("   fail on it. Stage it from a login node first:")
    print(f"     apptainer exec $SCRATCH/ollama/ollama.sif ollama pull {model}\n")
    return True


def check(run_dir):
    missing_model = report_judge_model()
    tasks = load_tasks(run_dir)
    references = {dataset: reference_names(dataset) for _, dataset in set(tasks)}

    missing = sorted(name for name, names in references.items() if names is None)
    if missing:
        print(f"!! no rationales-test.csv for: {', '.join(missing)}")
        print("   generate it with: sbatch --export=ALL,SPLIT=test "
              "eval/scripts/generate-rationales.sbatch\n")

    print(f"{'model':<30} {'dataset':<7} {'evaluated':>9} {'joinable':>9} {'coverage':>9}")
    shortfall = 0
    by_model = defaultdict(lambda: [0, 0])
    for model, dataset in sorted(tasks):
        available = references.get(dataset) or set()
        evaluated = evaluated_names(run_dir, model, dataset)
        joinable = len(evaluated & available)
        print(f"{model:<30} {dataset:<7} {len(evaluated):>9} {joinable:>9} "
              f"{rate(joinable, len(evaluated)):>9.1%}")
        by_model[model][0] += len(evaluated)
        by_model[model][1] += joinable
        shortfall += len(evaluated) - joinable

    total_evaluated = sum(values[0] for values in by_model.values())
    total_joinable = sum(values[1] for values in by_model.values())
    print(f"\n{len(by_model)} models, {len(set(tasks))} tasks")
    print(f"evaluated rationales {total_evaluated}, joinable {total_joinable} "
          f"({rate(total_joinable, total_evaluated):.1%})")
    if shortfall:
        print(f"~~ {shortfall} evaluated windows have no reference rationale; they "
              f"will be skipped by the judge.")
    return 1 if missing_model or not total_joinable else 0


# ------------------------------------------------------------------ submit


def submit(run_dir, args):
    config = load_judge_config()
    models = sorted({model for model, _ in load_tasks(run_dir)})
    if not models:
        raise SystemExit(f"No successful tasks in {run_dir / 'status.csv'}")

    if judge_model_staged(config["model"]) is False:
        raise SystemExit(
            f"judge model {config['model']} is not in {model_store()}.\n"
            f"Compute nodes cannot reach the Ollama registry, so all {len(models)} "
            f"tasks would fail on it. Stage it from a login node first:\n"
            f"  apptainer exec $SCRATCH/ollama/ollama.sif ollama pull {config['model']}"
        )

    concurrency = min(config["array-concurrency"], len(models))
    command = [
        "sbatch",
        f"--chdir={EVAL_DIR}",
        f"--array=0-{len(models) - 1}%{concurrency}",
        f"--time={config['time']}",
        f"--mem={config['ram']}",
        f"--gres=gpu:{config['gpus']}",
        f"--export=ALL,RUN_DIR={run_dir},RUN_NAME={run_dir.name},"
        f"MODELS_B64={base64.b64encode(json.dumps(models).encode()).decode()}",
        f"--output={run_dir}/logs/%x-%A_%a.out",
        f"--error={run_dir}/logs/%x-%A_%a.err",
    ]
    if args.nice is not None:
        command.append(f"--nice={args.nice}")
    command.append(str(EVAL_DIR / "scripts" / "judge_array.sbatch"))

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout.strip())
    print(f"{len(models)} models, judge {config['model']}, run {run_dir.name}")
    return 0


# ------------------------------------------------------------------ report


def load_agreement(run_dir, model, dataset):
    path = run_dir / "agreement" / f"{safe_filename(model)}-{dataset}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize_task(model, dataset, rows):
    real = [row for row in rows if row["control"] != "True"]
    controls = [row for row in rows if row["control"] == "True"]
    if not real:
        return None

    verdicts = [row["verdict"] for row in real]
    correct = [row for row in real if row["correct"] == "True"]
    incorrect = [row for row in real if row["correct"] != "True"]

    return {
        "model": model,
        "dataset": dataset,
        "pairs": len(real),
        "agree": verdicts.count("agree"),
        "partial": verdicts.count("partial"),
        "disagree": verdicts.count("disagree"),
        "agreement_rate": rate(verdicts.count("agree"), len(real)),
        "lenient_rate": rate(verdicts.count("agree") + verdicts.count("partial"), len(real)),
        "pairs_correct": len(correct),
        "agreement_rate_correct": rate(
            sum(row["verdict"] == "agree" for row in correct), len(correct)
        ),
        "pairs_incorrect": len(incorrect),
        "agreement_rate_incorrect": rate(
            sum(row["verdict"] == "agree" for row in incorrect), len(incorrect)
        ),
        "control_pairs": len(controls),
        "control_agreement_rate": rate(
            sum(row["verdict"] == "agree" for row in controls), len(controls)
        ),
    }


def summarize_models(tasks):
    """Macro-average each model over its datasets, so TUSZ does not dominate."""
    by_model = defaultdict(list)
    for task in tasks:
        by_model[task["model"]].append(task)

    ranking = []
    for model, rows in by_model.items():
        agreement = mean([row["agreement_rate"] for row in rows])
        control = mean([row["control_agreement_rate"] for row in rows
                        if row["control_pairs"]])
        ranking.append({
            "model": model,
            "datasets": len(rows),
            "pairs": sum(row["pairs"] for row in rows),
            "agreement_rate": agreement,
            "agreement_rate_correct": mean(
                [row["agreement_rate_correct"] for row in rows if row["pairs_correct"]]
            ),
            "lenient_rate": mean([row["lenient_rate"] for row in rows]),
            "control_agreement_rate": control,
            "clears_control": agreement > control,
        })

    ranking.sort(key=lambda row: (-row["agreement_rate"], row["model"]))
    for position, row in enumerate(ranking, start=1):
        row["rank"] = position
    return ranking


def report(run_dir):
    tasks = [
        summary
        for model, dataset in sorted(load_tasks(run_dir))
        if (summary := summarize_task(model, dataset, load_agreement(run_dir, model, dataset)))
    ]
    if not tasks:
        raise SystemExit(
            f"No judged pairs under {run_dir / 'agreement'} -- run "
            f"`python eval/run-judge.py submit {run_dir.name}` first."
        )

    models = summarize_models(tasks)
    write_csv(run_dir / "rationale-agreement.csv", TASK_FIELDS, tasks)
    write_csv(run_dir / "rationale-agreement-by-model.csv", MODEL_FIELDS, models)

    print(f"{'rank':>4}  {'model':<30} {'agree':>7} {'|correct':>9} {'control':>8}  clears")
    for row in models:
        print(f"{row['rank']:>4}  {row['model']:<30} {row['agreement_rate']:>7.1%} "
              f"{row['agreement_rate_correct']:>9.1%} {row['control_agreement_rate']:>8.1%}"
              f"  {'yes' if row['clears_control'] else 'NO'}")
    print(f"\nwrote {run_dir / 'rationale-agreement.csv'}")
    print(f"wrote {run_dir / 'rationale-agreement-by-model.csv'}")
    return 0


# --------------------------------------------------------------------- cli


def load_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("check", "submit", "report"))
    parser.add_argument("run", help="run directory name under eval/runs/")
    parser.add_argument(
        "--nice",
        type=int,
        help="submit only: lower these jobs' priority by this much (Slurm --nice).",
    )
    return parser.parse_args()


def main():
    set_csv_field_limit()
    args = load_args()
    run_dir = run_dir_for(args.run)
    return {"check": check, "report": report}.get(
        args.command, lambda directory: submit(directory, args)
    )(run_dir)


if __name__ == "__main__":
    sys.exit(main())
