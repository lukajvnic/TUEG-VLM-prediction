import argparse
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


def load_args():
    parser = argparse.ArgumentParser(
        description="Submit the eval benchmark as one Slurm array job per resource group."
    )
    parser.add_argument(
        "--after",
        metavar="JOBID",
        help="hold every array until JOBID has started running. Use this when a "
             "higher-priority job of yours is still pending: the eval arrays cannot "
             "then be scheduled ahead of it, or burn the account's fair-share while "
             "it waits.",
    )
    parser.add_argument(
        "--nice",
        type=int,
        help="lower these jobs' priority by this much (Slurm --nice). Keeps the eval "
             "sweep from outranking your other pending work for the whole run, not "
             "just at submission time. 10000 is a strong deprioritisation.",
    )
    return parser.parse_args()


def dispatch_job(config, run, resources, runs, args):
    time, ram, gpus = resources
    concurrency = min(config["settings"]["array-concurrency"], len(runs))
    tasks = base64.b64encode(json.dumps(runs).encode()).decode()

    command = [
        "sbatch",
        f"--chdir={Path(__file__).parent}",
        f"--array=0-{len(runs) - 1}%{concurrency}",
        f"--time={time}",
        f"--mem={ram}",
        f"--gres=gpu:{gpus}",
        f"--export=ALL,RUN_DIR={run},RUNS_B64={tasks}",
        f"--output={run}/logs/%x-%A_%a.out",
        f"--error={run}/logs/%x-%A_%a.err",
    ]
    # `after` (not afterok): release once the other job is *running*, since at that
    # point it holds its GPU and nothing here can take it away.
    if args.after:
        command.append(f"--dependency=after:{args.after}")
    if args.nice is not None:
        command.append(f"--nice={args.nice}")
    command.append(str(Path(__file__).parent / "scripts" / "run_array.sbatch"))

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.split()[-1]
    print(result.stdout.strip())
    return job_id


def main():
    args = load_args()
    config = load_config()
    resource_groups = group_runs(config)
    run = setup_run()

    for resources, runs in resource_groups.items():
        job_id = dispatch_job(config, run, resources, runs, args)
        save_tasks(run, job_id, runs)

    print(f"\nrun directory: {run}")


if __name__ == "__main__":
    main()