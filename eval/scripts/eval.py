import argparse
import base64
import csv
import fcntl
import json
import mimetypes
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
import wandb

from sample import probe, select
from structure import get_structure


# How often (in processed images) to refresh status.csv with live progress.
STATUS_EVERY = 50

# Retrying helps when the server was momentarily unreachable; it never helps when
# the model returned well-formed JSON that does not match the schema, or no JSON
# at all. Sleeping before those retries just burns walltime -- at 36 models x
# thousands of images a 10 s pause on a deterministic failure is hours of GPU time.
TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)
RETRY_SLEEP_SECONDS = 10


def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def load_relevant_config(args):
    with (Path(os.environ["RUN_DIR"]) / "config.yml").open() as file:
        config = yaml.safe_load(file)

    settings = config["settings"]
    model = config.get("models", {}).get(args.model, {})
    return {
        "prompt": config["datasets"][args.dataset]["prompt"],
        "eval-retries": settings["eval-retries"],
        "limit": settings["limit"],
        # A big model on a full A100 can keep more requests in flight than a 2B
        # model on a 20 GB MIG slice, so the per-model value wins when present.
        "parallel-requests": model.get("parallel-requests", settings["parallel-requests"]),
        "resume-from": settings.get("resume-from"),
        "model-kwargs": settings["model-kwargs"],
        "structured-output": settings["structured-output"],
        "test-sample": settings.get("test-sample"),
        "probe-images": settings.get("probe-images"),
    }


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def init_csv(run_dir, args):
    path = run_dir / "results" / f"{safe_filename(args.model)}-{args.dataset}.csv"

    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                ["path", "prediction", "actual", "correct", "text_rationale", "api_json"]
            )

    return path


def get_dataset_dir(args):
    return Path(__file__).parents[2] / "datasets" / args.dataset


def load_dataset(args, config):
    dataset_dir = get_dataset_dir(args)
    with (dataset_dir / "labels.csv").open(newline="", encoding="utf-8") as file:
        # Evaluate on held-out test patients only (split column absent = old
        # unsplit dataset, so fall through and use everything).
        rows = [row for row in csv.DictReader(file) if row.get("split") in (None, "test")]

    ground_truths = {
        (dataset_dir / row["image_path"]).resolve(): {
            label.upper()
            for label, value in row.items()
            if label not in ("image_path", "split", "assessed") and value.strip().lower() == "true"
        }
        for row in rows
    }

    # `select` drops windows the source annotations never covered (assessed=false --
    # they cannot be graded) and applies the stratified per-recording cap. It is
    # deterministic, so every model evaluates the identical set. `probe-images`
    # narrows that further for the stage-1 screening run.
    if config["probe-images"]:
        selected = probe(rows, args.dataset, config["probe-images"], config["test-sample"])
    else:
        selected = select(rows, args.dataset, config["test-sample"])
    paths = sorted(
        path for path in ((dataset_dir / name).resolve() for name in selected) if path.is_file()
    )

    limit = config["limit"]
    if limit != -1:
        paths = paths[:limit]
    return paths, ground_truths


def load_completed_paths(run_dir, args, config):
    csv_name = f"{safe_filename(args.model)}-{args.dataset}.csv"
    sources = [run_dir / "results" / csv_name]
    if config["resume-from"]:
        sources.append(Path(__file__).parents[1] / "runs" / config["resume-from"] / "results" / csv_name)

    completed = set()
    for source in sources:
        if source.exists():
            with source.open(newline="", encoding="utf-8") as file:
                completed.update(row["path"] for row in csv.DictReader(file))
    return completed


def get_prediction(parsed, dataset):
    if dataset == "TUEP":
        return {"EPILEPSY"} if parsed.has_epilepsy else {"NO_EPILEPSY"}
    if dataset == "TUAB":
        return {"ABNORMAL"} if parsed.is_abnormal else {"NORMAL"}

    return {
        field.removeprefix("has_").upper()
        for field in type(parsed).model_fields
        if field.startswith("has_") and getattr(parsed, field)
    }


def update_status(run_dir, args, status, reason, total_images, completed_images):
    path = run_dir / "status.csv"
    with path.open("a+", newline="", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        rows = [
            row for row in csv.DictReader(file)
            if (row["model"], row["dataset"]) != (args.model, args.dataset)
        ]
        rows.append(
            {
                "model": args.model,
                "dataset": args.dataset,
                "status": status,
                "reason": reason,
                "total_images": total_images,
                "completed_images": completed_images,
            }
        )
        file.seek(0)
        file.truncate()
        writer = csv.DictWriter(
            file,
            fieldnames=("model", "dataset", "status", "reason", "total_images", "completed_images"),
        )
        writer.writeheader()
        writer.writerows(rows)


def log_result(run_dir, args, results_csv, image_path, prediction, ground_truth, rationale, raw, error=None):
    reason = str(error) if error else ""
    raw = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw

    with (run_dir / "logs" / f"{safe_filename(args.model)}-{args.dataset}.jsonl").open("a") as file:
        file.write(json.dumps({"path": str(image_path), "raw": raw, "error": reason}, default=str) + "\n")

    if not error:
        with results_csv.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [image_path, prediction, ground_truth, prediction == ground_truth, rationale, json.dumps(raw, default=str)]
            )




def build_message(path, prompt):
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    image = base64.b64encode(path.read_bytes()).decode()
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{image}"},
        ]
    )


def evaluate_image(model, path, args, config, ground_truths):
    messages = [build_message(path, config["prompt"])]
    for attempt in range(config["eval-retries"] + 1):
        try:
            response = model.invoke(messages)
            parsed = response["parsed"]
            if parsed is None:
                raise ValueError("Structured output could not be parsed")

            return {
                "image_path": path,
                "prediction": get_prediction(parsed, args.dataset),
                "ground_truth": ground_truths[path],
                "rationale": parsed.text_rationale,
                "raw": response["raw"],
                "error": None,
            }
        except Exception as error:
            if attempt == config["eval-retries"]:
                return {
                    "image_path": path,
                    "prediction": None,
                    "ground_truth": None,
                    "rationale": "",
                    "raw": None,
                    "error": error,
                }
            # Back off only when the server might recover. A schema-validation or
            # JSON-parse failure is deterministic: retry immediately or not at all.
            if isinstance(error, TRANSIENT_ERRORS):
                time.sleep(RETRY_SLEEP_SECONDS)


def record_result(result, run_dir, args, results_csv, progress):
    log_result(
        run_dir,
        args,
        results_csv,
        result["image_path"],
        result["prediction"],
        result["ground_truth"],
        result["rationale"],
        result["raw"],
        result["error"],
    )
    if result["error"] is None:
        progress["completed_images"] += 1
        progress["correct_images"] += result["prediction"] == result["ground_truth"]
    else:
        progress["failed_images"] += 1


def send(model, paths, run_dir, args, results_csv, config, progress, ground_truths, total_images):
    with ThreadPoolExecutor(max_workers=config["parallel-requests"]) as executor:
        futures = [
            executor.submit(evaluate_image, model, path, args, config, ground_truths)
            for path in paths
        ]
        for future in as_completed(futures):
            result = future.result()
            record_result(result, run_dir, args, results_csv, progress)
            result["raw"] = None
            done = progress["completed_images"] + progress["failed_images"]
            tag = "ok" if result["error"] is None else "FAIL"
            print(f"[{done}/{total_images}] {args.dataset} {tag}: {Path(result['image_path']).name}", flush=True)
            if done % STATUS_EVERY == 0:
                update_status(run_dir, args, "running", "", total_images, progress["completed_images"])
                # Aggregate rather than per-image: one wandb event per image is
                # ~15k events per task and buys nothing that the results CSV and
                # score.py do not already give us.
                wandb.log(
                    {
                        "done": done,
                        "completed_images": progress["completed_images"],
                        "failed_images": progress["failed_images"],
                        "running_accuracy": progress["correct_images"] / max(progress["completed_images"], 1),
                    }
                )


def init_wandb(args, config):
    wandb.init(
        project="eeg-vlm-inference",
        name=f"{safe_filename(args.model)}-{args.dataset}",
        config={"model": args.model, "dataset": args.dataset, **config},
    )


def init_model(args, config):
    init_wandb(args, config)
    model_kwargs = config["model-kwargs"].copy()
    if base_url := os.environ.get("OLLAMA_BASE_URL"):
        model_kwargs["base_url"] = base_url

    return ChatOllama(model=args.model, **model_kwargs).with_structured_output(
        get_structure(args.dataset),
        include_raw=True,
        **config["structured-output"],
    )


def main():
    set_csv_field_limit()
    args = load_args()
    run_dir = Path(os.environ["RUN_DIR"])
    total_images = 0
    progress = {"completed_images": 0, "failed_images": 0, "correct_images": 0}

    try:
        config = load_relevant_config(args)
        paths, ground_truths = load_dataset(args, config)
        total_images = len(paths)

        results_csv = init_csv(run_dir, args)
        completed = load_completed_paths(run_dir, args, config)
        paths = [path for path in paths if str(path) not in completed]
        progress["completed_images"] = total_images - len(paths)
        update_status(run_dir, args, "running", "", total_images, progress["completed_images"])

        model = init_model(args, config)
        send(model, paths, run_dir, args, results_csv, config, progress, ground_truths, total_images)
    except Exception as error:
        update_status(run_dir, args, "fail", str(error), total_images, progress["completed_images"])
        raise
    else:
        failed = progress["failed_images"]
        status, reason = ("fail", f"{failed} images failed") if failed else ("success", "")
        update_status(run_dir, args, status, reason, total_images, progress["completed_images"])
        wandb.summary.update(
            {
                "total_images": total_images,
                "completed_images": progress["completed_images"],
                "failed_images": failed,
                "window_accuracy": progress["correct_images"] / max(progress["completed_images"], 1),
            }
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()