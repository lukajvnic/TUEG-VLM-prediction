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
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
import wandb

from structure import get_structure


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
    return {
        "prompt": config["datasets"][args.dataset]["prompt"],
        "eval-retries": settings["eval-retries"],
        "limit": settings["limit"],
        "model-kwargs": settings["model-kwargs"],
        "structured-output": settings["structured-output"],
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


def get_paths(args, config):
    dataset_dir = Path(__file__).parents[2] / "datasets" / args.dataset
    with (dataset_dir / "labels.csv").open(newline="", encoding="utf-8") as file:
        paths = sorted(
            (dataset_dir / row["image_path"]).resolve()
            for row in csv.DictReader(file)
            if (dataset_dir / row["image_path"]).is_file()
        )

    limit = config["limit"]
    return paths if limit == -1 else paths[:limit]


def get_ground_truth(path):
    dataset_dir = next(parent for parent in path.parents if (parent / "labels.csv").exists())
    image_path = path.relative_to(dataset_dir).as_posix()

    with (dataset_dir / "labels.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["image_path"] == image_path:
                return {
                    label.upper()
                    for label, value in row.items()
                    if label != "image_path" and value.strip().lower() == "true"
                }

    raise KeyError(f"No labels found for {image_path}")


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
        wandb.log({"path": str(image_path), "correct": prediction == ground_truth})




def send(model, batch, run_dir, args, results_csv, config, progress):
    for image_path, messages in batch:
        for attempt in range(config["eval-retries"] + 1):
            try:
                response = model.invoke(messages)
                parsed = response["parsed"]
                if parsed is None:
                    raise ValueError("Structured output could not be parsed")

                prediction = get_prediction(parsed, args.dataset)
                ground_truth = get_ground_truth(image_path)
                log_result(
                    run_dir,
                    args,
                    results_csv,
                    image_path,
                    prediction,
                    ground_truth,
                    parsed.text_rationale,
                    response["raw"],
                )
                progress["completed_images"] += 1
                break
            except Exception as error:
                if attempt == config["eval-retries"]:
                    log_result(
                        run_dir, args, results_csv, image_path, None, None, "", None, error
                    )
                    raise
                time.sleep(10)


def build_batch(paths, prompt):
    batch = []
    for path in paths:
        mime_type = mimetypes.guess_type(path)[0] or "image/png"
        image = base64.b64encode(path.read_bytes()).decode()
        batch.append(
            (
                path,
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:{mime_type};base64,{image}"},
                        ]
                    )
                ],
            )
        )
    return batch


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
    progress = {"completed_images": 0}

    try:
        config = load_relevant_config(args)
        paths = get_paths(args, config)
        total_images = len(paths)
        update_status(run_dir, args, "running", "", total_images, progress["completed_images"])

        results_csv = init_csv(run_dir, args)
        batch = build_batch(paths, config["prompt"])
        model = init_model(args, config)
        send(model, batch, run_dir, args, results_csv, config, progress)
    except Exception as error:
        update_status(run_dir, args, "fail", str(error), total_images, progress["completed_images"])
        raise
    else:
        update_status(run_dir, args, "success", "", total_images, progress["completed_images"])
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()