import base64
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import yaml


def load_config():
    with (Path(__file__).parent / "config.yml").open() as file:
        return yaml.safe_load(file)


def group_runs(config):
    groups = {}

    for model, model_config in config["models"].items():
        resources = (
            model_config["time"],
            model_config["ram"],
            model_config["gpus"],
        )
        runs = [(model, dataset) for dataset in model_config["datasets"]]
        groups.setdefault(resources, []).extend(runs)

    return groups


def setup_run():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run = Path(__file__).parent / "runs" / f"run-{timestamp}"

    (run / "logs").mkdir(parents=True)
    (run / "results").mkdir()
    (run / "status.csv").write_text(
        "model,dataset,status,reason,total_images,completed_images\n"
    )
    shutil.copy2(Path(__file__).parent / "config.yml", run / "config.yml")

    return run


def save_tasks(run, job_id, runs):
    path = run / "tasks.json"
    tasks = json.loads(path.read_text()) if path.exists() else []
    tasks.extend(
        {"job_id": job_id, "task_id": task_id, "model": model, "dataset": dataset}
        for task_id, (model, dataset) in enumerate(runs)
    )
    path.write_text(json.dumps(tasks, indent=2) + "\n")


def dispatch_job(config, run, resources, runs):
    time, ram, gpus = resources
    concurrency = min(config["settings"]["array-concurrency"], len(runs))
    tasks = base64.b64encode(json.dumps(runs).encode()).decode()

    result = subprocess.run(
        [
            "sbatch",
            f"--chdir={Path(__file__).parent}",
            f"--array=0-{len(runs) - 1}%{concurrency}",
            f"--time={time}",
            f"--mem={ram}",
            f"--gres=gpu:{gpus}",
            f"--export=ALL,RUN_DIR={run},RUNS_B64={tasks}",
            f"--output={run}/logs/%x-%A_%a.out",
            f"--error={run}/logs/%x-%A_%a.err",
            str(Path(__file__).parent / "scripts" / "run_array.sbatch"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.split()[-1]
    print(result.stdout.strip())
    return job_id


def main():
    config = load_config()
    resource_groups = group_runs(config)
    run = setup_run()

    for resources, runs in resource_groups.items():
        job_id = dispatch_job(config, run, resources, runs)
        save_tasks(run, job_id, runs)


if __name__ == "__main__":
    main()