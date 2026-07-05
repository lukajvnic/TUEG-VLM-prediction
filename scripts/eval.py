import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.structure import get_structure
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


CONFIG_PATH = PROJECT_ROOT / "config.yml"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_CSV = PROJECT_ROOT / "results.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_config(model_override: str | None = None):
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if model_override:
        config["model"] = model_override

    ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
    if ollama_base_url:
        config.setdefault("model-kwargs", {})["base_url"] = ollama_base_url

    return config


def get_actual(path):
    image_path = Path(path).resolve()

    dataset = None
    for parent in image_path.parents:
        if (parent / "labels.csv").exists():
            dataset = parent
            break

    if dataset is None:
        return {image_path.parent.name.replace("-", "_").upper()}

    relative_path = image_path.relative_to(dataset).as_posix()
    labels_path = dataset / "labels.csv"

    with labels_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("image_path") == relative_path:
                return {
                    column.upper()
                    for column, value in row.items()
                    if column != "image_path"
                    and str(value).strip().lower() == "true"
                }

    raise KeyError(f"No labels found for {relative_path} in {labels_path}")


def get_prediction(parsed, dataset: str):
    if dataset == "TUEP":
        return {"EPILEPSY"} if parsed.has_epilepsy else {"NO_EPILEPSY"}

    if dataset == "TUAB":
        return {"ABNORMAL"} if parsed.is_abnormal else {"NORMAL"}

    return {
        field_name.removeprefix("has_").upper()
        for field_name in type(parsed).model_fields
        if field_name.startswith("has_") and getattr(parsed, field_name)
    }


def send(llm, batch: list[tuple[Path, list[HumanMessage]]], dataset: str, wandb_table=None):
    inputs = [messages for _, messages in batch]
    results = []
    start_time = time.monotonic()

    for completed_count, (index, result) in enumerate(llm.batch_as_completed(inputs), start=1):
        image_path = batch[index][0]
        parsed = result["parsed"]
        raw = result["raw"]
        prediction = get_prediction(parsed, dataset)
        actual = get_actual(image_path)
        latency = time.monotonic() - start_time
        error = None
        output_text = parsed.text_rationale
        ground_truth_label = ",".join(sorted(actual)) if actual else image_path.parent.name

        wandb.log({
            "image_index": completed_count,
            "images_processed": completed_count,
            "total_images": len(batch),
            "latency_seconds": latency,
            "had_error": error is not None,
        })

        if wandb_table is not None:
            wandb_table.add_data(
                str(image_path),
                ground_truth_label,
                output_text,
                latency,
                error,
            )

        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                str(image_path),
                str(prediction),
                str(actual),
                prediction == actual,
                parsed.text_rationale,
                json.dumps(raw.model_dump(mode="json"), ensure_ascii=False),
            ])

        print(f"Completed {completed_count}/{len(batch)} - {prediction}")
        results.append(result)

    return results


def init_csv():
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        return

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "prediction", "actual", "correct", "text_rationale", "api_json"])


def get_seen_paths() -> set[str]:
    if not RESULTS_CSV.exists():
        return set()

    with open(RESULTS_CSV, "r", newline="", encoding="utf-8") as file:
        return {row["path"] for row in csv.DictReader(file) if row.get("path")}


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def archive_results(model: str, dataset: str):
    if not RESULTS_CSV.exists():
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = safe_filename_part(model)
    destination = RESULTS_DIR / f"{safe_model}-{dataset}-{timestamp}.csv"
    RESULTS_CSV.rename(destination)
    print(f"Moved results to {destination}")


def get_batch_paths(config) -> list[Path]:
    dataset = config["dataset"]
    dataset_dir = DATASETS_DIR / dataset
    labels_path = dataset_dir / "labels.csv"

    if labels_path.exists():
        paths = []
        with labels_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                image_path = dataset_dir / row["image_path"]
                if image_path.is_file():
                    paths.append(image_path.resolve())
        return sorted(paths)

    data_dir = dataset_dir / config[dataset]["data-directory"]
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    return sorted(
        image_path.resolve()
        for image_path in data_dir.rglob("*.png")
        if image_path.is_file()
    )


def image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def build_batch(batch_paths, prompt) -> list[tuple[Path, list[HumanMessage]]]:
    batch = []

    for path in batch_paths:
        image_path = Path(path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_data_url(image_path)},
            ]
        )
        batch.append((image_path, [message]))

    return batch


def init_model(config, out_struct):
    model_kwargs = config.get("model-kwargs") or {}
    structured_output_kwargs = config.get("structured-output") or {}

    return ChatOllama(
        model=config["model"],
        **model_kwargs,
    ).with_structured_output(
        out_struct,
        include_raw=True,
        **structured_output_kwargs,
    )



def run_evaluation(config):
    global RESULTS_CSV

    print(f"Running {config['dataset']} with model {config['model']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    run_id = os.environ.get("SLURM_ARRAY_JOB_ID", datetime.now().strftime("%Y%m%d-%H%M%S"))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        run_id = f"{run_id}_{task_id}"
    RESULTS_CSV = RESULTS_DIR / f"{safe_filename_part(config['model'])}-{config['dataset']}-{run_id}.csv"

    out_struct = get_structure(config["dataset"])
    model = init_model(config, out_struct)
    prompt = config[config["dataset"]]["prompt"]
    image_dir = DATASETS_DIR / config["dataset"] / config[config["dataset"]]["data-directory"]

    wandb.init(
        project="eeg-vlm-inference",
        name=f"{config['model'].replace(':', '-')}-baseline",
        config={
            "model": config["model"],
            "prompt": prompt,
            "image_dir": str(image_dir),
            "output_path": str(RESULTS_CSV),
            "num_predict": 500,
            "stop": ["END"],
        },
    )

    init_csv()
    seen_paths = get_seen_paths()
    all_batch_paths = get_batch_paths(config)
    batch_paths = [path for path in all_batch_paths if str(path) not in seen_paths]
    skipped_count = len(all_batch_paths) - len(batch_paths)

    limit = config.get("limit", -1)
    if limit != -1:
        batch_paths = batch_paths[:limit]

    print(f"Skipping {skipped_count} already-completed item(s).")
    print(f"Sending {len(batch_paths)} item(s).")

    wandb_table = wandb.Table(
        columns=[
            "image_path",
            "ground_truth_label",
            "model_output",
            "latency_seconds",
            "error",
        ]
    )

    batch = build_batch(batch_paths, prompt)
    send(model, batch, config["dataset"], wandb_table)
    archive_results(config["model"], config["dataset"])
    wandb.log({"vlm_outputs": wandb_table})
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Run TUEG VLM evaluation.")
    parser.add_argument("--model", help="Override the model in config.yml.")
    args = parser.parse_args()

    config = load_config(args.model)

    if config["model"] == "all":
        for model_name in config.get("models", {}):
            run_evaluation({**config, "model": model_name})
    else:
        run_evaluation(config)


if __name__ == "__main__":
    main()
