"""Resume selected model/dataset evaluations from an existing run without recomputing rows."""
import argparse
import base64
import csv
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SLURM_SCRIPT = PROJECT_ROOT / "scripts" / "run_array.sbatch"
LOG_DIR = PROJECT_ROOT / "logs"

# These models emitted invalid schema-constrained responses in the failed run.
JSON_MODE_MODELS = {
    "gemma4:12b",
    "qwen2.5vl:3b",
    "qwen3-vl:2b-thinking",
    "qwen3-vl:4b-thinking",
    "qwen3-vl:8b-thinking",
}


def parse_duration(value: str) -> int:
    days = 0
    if "-" in value:
        day_part, value = value.split("-", 1)
        days = int(day_part)
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: int) -> str:
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02}:{minutes:02}:{seconds:02}"


def load_pairs(path: Path, models: dict) -> list[tuple[str, str, str]]:
    pairs = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            model, dataset, reason = row.get("model"), row.get("dataset"), row.get("reason", "retry")
            if not model or not dataset:
                raise SystemExit(f"ERROR: invalid row in {path}: {row}")
            if model not in models:
                raise SystemExit(f"ERROR: {model!r} in {path} is not in the source run config")
            if (model, dataset) not in seen:
                pairs.append((model, dataset, reason))
                seen.add((model, dataset))
    if not pairs:
        raise SystemExit(f"ERROR: no pairs found in {path}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume selected pairs from an existing TUEG VLM run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Original results/run-* directory to resume")
    parser.add_argument("--pairs", type=Path, default=PROJECT_ROOT / "fixable-resume-pairs.csv")
    parser.add_argument("--timeout-multiplier", type=int, default=3)
    args = parser.parse_args()

    source_run = args.run_dir.resolve()
    source_config = source_run / "config.yml"
    if not source_config.exists():
        raise SystemExit(f"ERROR: source run config not found: {source_config}")
    if args.timeout_multiplier < 1:
        raise SystemExit("ERROR: --timeout-multiplier must be at least 1")

    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    models = config.get("models") or {}
    pairs = load_pairs(args.pairs.resolve(), models)

    # Make an immutable retry config with narrowly scoped recovery settings.
    resume_id = "resume-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    resume_dir = PROJECT_ROOT / "results" / resume_id
    resume_log_dir = LOG_DIR / resume_id
    resume_dir.mkdir(parents=True, exist_ok=False)
    resume_log_dir.mkdir(parents=True, exist_ok=False)
    resume_config = resume_dir / "config.yml"
    retry_config = json.loads(json.dumps(config))  # independent nested dictionaries
    for model in JSON_MODE_MODELS & {model for model, _, _ in pairs}:
        retry_config["models"][model]["structured-output"] = {"method": "json_mode"}
    if "minicpm-v4.6:1b" in {model for model, _, _ in pairs}:
        retry_config["models"]["minicpm-v4.6:1b"]["ram"] = "32G"
    resume_config.write_text(yaml.safe_dump(retry_config, sort_keys=False), encoding="utf-8")
    (resume_dir / "source_run.txt").write_text(f"{source_run}\n", encoding="utf-8")
    (resume_dir / "tasks.json").write_text(
        json.dumps([{"model": model, "dataset": dataset, "reason": reason} for model, dataset, reason in pairs], indent=2) + "\n",
        encoding="utf-8",
    )

    groups = defaultdict(list)
    for model, dataset, reason in pairs:
        spec = retry_config["models"][model] or {}
        time_limit = spec.get("time", "12:00:00")
        if reason == "timeout":
            time_limit = format_duration(parse_duration(time_limit) * args.timeout_multiplier)
        key = (time_limit, spec.get("ram", "32G"), int(spec.get("gpus", 1)))
        groups[key].append((model, dataset))

    source_run_id = source_run.name
    job_ids = []
    for (time_limit, ram, gpus), tasks in groups.items():
        encoded_tasks = base64.b64encode(json.dumps(tasks).encode("utf-8")).decode("ascii")
        export = (
            f"ALL,PI_RUN_ID={source_run_id},PI_LOG_DIR={resume_log_dir},PI_CONFIG_PATH={resume_config},"
            f"PI_TASK_LIST_B64={encoded_tasks}"
        )
        command = [
            "sbatch", f"--chdir={PROJECT_ROOT}", f"--array=0-{len(tasks) - 1}%{len(tasks)}",
            f"--time={time_limit}", f"--mem={ram}", f"--gres=gpu:{gpus}", f"--export={export}",
            f"--output={resume_log_dir}/%x-%A_%a.out", f"--error={resume_log_dir}/%x-%A_%a.err",
            str(SLURM_SCRIPT),
        ]
        print(f"Submitting {len(tasks)} resume task(s): gpu:{gpus} mem:{ram} time:{time_limit}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        job_ids.append(result.stdout.strip().split()[-1])

    (resume_dir / "job_ids.txt").write_text(",".join(job_ids) + "\n", encoding="utf-8")
    if job_ids:
        dependency = "afterany:" + ":".join(job_ids)
        command = [
            "sbatch", f"--chdir={PROJECT_ROOT}", f"--dependency={dependency}",
            "--job-name=tueg-vlm-resume-summary", "--account=def-milad777", "--time=00:30:00", "--mem=4G",
            f"--output={resume_log_dir}/%x-%j.out", f"--error={resume_log_dir}/%x-%j.err",
            "--wrap", f"source .venv/bin/activate && python scripts/summarize.py {source_run}",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        summary_job_id = result.stdout.strip().split()[-1]
        (resume_dir / "summary_job_id.txt").write_text(summary_job_id + "\n", encoding="utf-8")
        print(result.stdout.strip())

    print(f"Source results resumed in place: {source_run}")
    print(f"Resume logs: {resume_log_dir}")


if __name__ == "__main__":
    main()
