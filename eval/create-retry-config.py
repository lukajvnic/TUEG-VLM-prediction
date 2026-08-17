import argparse
import csv
from pathlib import Path

import yaml


def load_config(path):
    with path.open() as file:
        return yaml.safe_load(file)


def load_retry_config(run_dir):
    # config-base.yml supplies only `models` (the full, uncommented list); every
    # other key comes from the run being retried. render_config must emit all of
    # them -- a key that is loaded but not rendered is dropped on the first retry.
    run_config = load_config(run_dir / "config.yml")
    base_config = load_config(Path(__file__).parent / "scripts" / "config-base.yml")
    base_config["judge"] = run_config["judge"]
    base_config["settings"] = run_config["settings"]
    base_config["datasets"] = run_config["datasets"]
    return base_config


def load_successful_tasks(run_dir):
    with (run_dir / "status.csv").open(newline="") as file:
        return {
            (row["model"], row["dataset"])
            for row in csv.DictReader(file)
            if row["status"] == "success"
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


def render_model(model, model_config, active_datasets):
    block = yaml.safe_dump({model: model_config}, sort_keys=False)
    lines = []
    for line in block.splitlines(keepends=True):
        dataset = line.strip().removeprefix("- ")
        if line.strip().startswith("- ") and dataset not in active_datasets:
            line = comment_block(line)
        lines.append(line)
    return indent("".join(lines))


def render_models(config, successful_tasks):
    output = "models:\n"
    for model, model_config in config["models"].items():
        # Pairs absent from status.csv have not run and must remain enabled.
        active_datasets = {
            dataset for dataset in model_config["datasets"]
            if (model, dataset) not in successful_tasks
        }
        block = render_model(model, model_config, active_datasets)
        output += block if active_datasets else comment_block(block)
    return output


def render_config(config, successful_tasks):
    header = yaml.safe_dump(
        {"judge": config["judge"], "settings": config["settings"]}, sort_keys=False
    )
    return header + render_datasets(config) + render_models(config, successful_tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()

    eval_dir = Path(__file__).parent
    run_dir = eval_dir / "runs" / args.run

    config = load_retry_config(run_dir)
    # Resume the retried run's partially-completed tasks from where they stopped.
    config["settings"]["resume-from"] = args.run

    retry_config = render_config(config, load_successful_tasks(run_dir))
    (eval_dir / "config.yml").write_text(retry_config)


if __name__ == "__main__":
    main()
