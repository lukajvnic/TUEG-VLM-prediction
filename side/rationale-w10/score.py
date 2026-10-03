# side run: score the side adapter on the test split; predictions and metrics stay in side/rationale-w10/results/
import argparse
import importlib.util
import sys
from pathlib import Path

SIDE = Path(__file__).resolve().parent
sys.path.insert(0, str(SIDE.parents[1]))
sys.path.insert(0, str(SIDE.parents[1] / "eval" / "models"))
from helpers.pipeline import DATASETS, ROOT, append_row, config, db, read_csv, script_module  # noqa: E402
from helpers.slurm import queued, script, submit  # noqa: E402
from structure import get_structure, labels, prompt, to_labels  # noqa: E402

EXPERIMENT = "rationale-w10"
RESULTS = SIDE / "results"
TIME, RAM = "12:00:00", "64G"  # ~1,200 TUAB windows; round-1 qwen2.5vl:7b with a rationale measured 14.8 s/image (~4.9 h)


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(model, dataset):
    return SIDE / "checkpoints" / model.replace(":", "-") / dataset


def model_name(model, dataset):
    return f"{model}-sft-{dataset}-{EXPERIMENT}"


def job_name(model, dataset):
    return f"eeg-vlm-side-w10-score-{model.replace(':', '-')}-{dataset}"


def test_scope(dataset):
    rows = db().execute("SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'full'", (dataset,))
    return {p.split("/", 1)[1] for (p,) in rows}


def predict_all(model, dataset):
    # resumes: windows already in the predictions file are skipped
    runner = script_module("eval")
    name = model_name(model, dataset)
    out = RESULTS / f"eval-{name.replace(':', '-')}.csv"
    header = ["path", "model", *labels(dataset), "rationale"]
    done = {r["path"] for r in read_csv(out)}
    todo = sorted(test_scope(dataset) - done)
    print(f"{name}: {len(todo)} windows to score, {len(done)} already done", flush=True)

    if not todo:
        return out

    loaded, processor, manifest = runner.load({"base": model, "checkpoint": str(checkpoint(model, dataset).relative_to(ROOT))})
    structure = get_structure(dataset, manifest.get("rationale_first", False), manifest.get("target") == "rationale")
    text = prompt(dataset, manifest.get("target") == "rationale")
    prefix_fn = runner.enforcer(processor, structure)
    folder = ROOT / "datasets" / dataset

    for i, rel in enumerate(todo, 1):
        try:
            parsed = runner.predict(loaded, processor, structure, folder / rel, text, prefix_fn)
        except Exception as e:
            append_row(RESULTS / "failures.csv", ["path", "model", "error"], [rel, name, f"{type(e).__name__}: {e}"[:300]])
            continue
        values = to_labels(parsed, dataset)
        append_row(out, header, [rel, name, *[str(v).lower() for v in values.values()],
                                 getattr(parsed, "text_rationale", "")])
        if i % 50 == 0:
            print(f"{i}/{len(todo)}", flush=True)
    return out


def summarise(model, dataset, predictions):
    # eval/score.py's recording-level metrics, written beside the predictions instead of datasets/<DS>/summary.csv
    score = load_path("eval_score", ROOT / "eval" / "score.py")
    truth = {r["path"]: r for r in read_csv(ROOT / "datasets" / dataset / "labels.csv")}
    summary = score.evaluate(model_name(model, dataset), dataset, read_csv(predictions), truth, test_scope(dataset),
                             score.BOOTSTRAP)
    if summary is None:
        sys.exit(f"no scored rows in {predictions} (see {RESULTS / 'failures.csv'})")
    score.write(RESULTS / "summary.csv", score.FIELDS, [{k: summary[k] for k in score.FIELDS}])
    print(f"recording balanced accuracy {summary['recording_balanced_accuracy']:.3f}, coverage {summary['coverage']:.2f}, "
          f"top answer {summary['top_answer']} ({summary['top_answer_fraction']:.0%}) -> {RESULTS / 'summary.csv'}")


def submit_job(cfg, model, dataset, dry_run):
    if not (checkpoint(model, dataset) / "manifest.json").exists():
        sys.exit(f"{checkpoint(model, dataset)} has no manifest.json: training has not finished")
    if job_name(model, dataset) in queued():
        sys.exit(f"{job_name(model, dataset)} is already queued or running")

    (SIDE / "logs").mkdir(exist_ok=True)
    text = script(job_name(model, dataset), TIME, RAM, cfg["train"]["eval"]["cpus"], 1,
                  f"python {SIDE}/score.py {model} {dataset} --here")
    text = text.replace(str(ROOT / "logs"), str(SIDE / "logs"))
    job = "dry run" if dry_run else submit(text)
    print(f"{job} - score {model_name(model, dataset)} on the {dataset} test split -> {RESULTS}")


def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="a config.yml bases: key")
    parser.add_argument("dataset", choices=DATASETS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--here", action="store_true", help="score in this process (what the Slurm job runs)")
    mode.add_argument("--dry-run", action="store_true", help="print the job, submit nothing")
    args = parser.parse_args()

    if args.model not in cfg["bases"]:
        parser.error(f"unknown model {args.model!r}: not a config.yml bases: key")

    if args.here:
        RESULTS.mkdir(exist_ok=True)
        summarise(args.model, args.dataset, predict_all(args.model, args.dataset))
        return

    submit_job(cfg, args.model, args.dataset, args.dry_run)


if __name__ == "__main__":
    main()
