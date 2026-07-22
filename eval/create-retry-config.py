import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml


def load_config(path):
    with path.open() as file:
        return yaml.safe_load(file)


def load_retry_config(run_dir):
    run_config = load_config(run_dir / "config.yml")
    base_config = load_config(Path(__file__).parent / "scripts" / "config-base.yml")
    base_config["settings"] = run_config["settings"]
    base_config["datasets"] = run_config["datasets"]
    return base_config


def load_failed_tasks(run_dir):
    with (run_dir / "status.csv").open(newline="") as file:
        return {
            (row["model"], row["dataset"])
            for row in csv.DictReader(file)
            if row["status"] == "fail"
        }


def comment_block(text):
    lines = []
    for line in text.splitlines(keepends=True):
        indentation = line[: len(line) - len(line.lstrip())]
        lines.append(f"{indentation}# {line[len(indentation):]}" if line.strip() else line)
    return "".join(lines)


def indent(text):
    return "".join(f"  {line}" if line.strip() else line for line in text.splitlines(keepends=True))


def render_datasets(config):
    output = "datasets:\n"
    for dataset, dataset_config in config["datasets"].items():
        output += indent(yaml.safe_dump({dataset: dataset_config}, sort_keys=False))
    return output


def render_model(model, model_config, failed_datasets):
    block = yaml.safe_dump({model: model_config}, sort_keys=False)
    lines = []
    for line in block.splitlines(keepends=True):
        dataset = line.strip().removeprefix("- ")
        if line.strip().startswith("- ") and dataset not in failed_datasets:
            line = comment_block(line)
        lines.append(line)
    return indent("".join(lines))


def render_models(config, failed_tasks):
    failed_by_model = defaultdict(set)
    for model, dataset in failed_tasks:
        failed_by_model[model].add(dataset)

    output = "models:\n"
    for model, model_config in config["models"].items():
        block = render_model(model, model_config, failed_by_model.get(model, set()))
        output += block if model in failed_by_model else comment_block(block)
    return output


def render_config(config, failed_tasks):
    settings = yaml.safe_dump({"settings": config["settings"]}, sort_keys=False)
    return settings + render_datasets(config) + render_models(config, failed_tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()

    eval_dir = Path(__file__).parent
    run_dir = eval_dir / "runs" / args.run
    retry_config = render_config(load_retry_config(run_dir), load_failed_tasks(run_dir))
    (eval_dir / "config.yml").write_text(retry_config)


if __name__ == "__main__":
    main()
