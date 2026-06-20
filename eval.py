import base64
import csv
import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path
import yaml
from dotenv import load_dotenv
from structure import get_structure
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


RESULTS_CSV = Path("results.csv")
RESULTS_DIR = Path("results")


def load_config():
    with open("config.yml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def send(llm, batch: list[tuple[Path, list[HumanMessage]]], dataset: str):
    inputs = [messages for _, messages in batch]
    results = []

    for completed_count, (index, result) in enumerate(llm.batch_as_completed(inputs), start=1):
        image_path = batch[index][0]
        parsed = result["parsed"]
        raw = result["raw"]
        prediction = get_prediction(parsed, dataset)
        actual = get_actual(image_path)

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


def archive_results(dataset: str):
    if not RESULTS_CSV.exists():
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = RESULTS_DIR / f"{dataset}-{timestamp}.csv"
    RESULTS_CSV.rename(destination)
    print(f"Moved results to {destination}")


def get_batch_paths(config) -> list[Path]:
    dataset = config["dataset"]
    dataset_dir = Path(dataset)
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


def main():
    load_dotenv()
    config = load_config()

    out_struct = get_structure(config["dataset"])
    model = init_chat_model(model=config["model"], model_provider=config["provider"]).with_structured_output(out_struct, include_raw=True)
    prompt = config[config["dataset"]]["prompt"]
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

    batch = build_batch(batch_paths, prompt)
    send(model, batch, config["dataset"])
    archive_results(config["dataset"])


if __name__ == "__main__":
    main()
