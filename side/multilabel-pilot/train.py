# side run: two ways to fine-tune qwen3-vl:2b-instruct on TUSZ, each trained with train/train.py's trainer (one epoch,
# 5 evals) and scored in the same job by score.py. Data from build.py; everything stays in side/multilabel-pilot/
#   python side/multilabel-pilot/train.py perclass|joint-aux [--dry-run]   # submit one variant's job
#   python side/multilabel-pilot/train.py smoke                            # <=1 h job: both variants, 2 steps, 24 windows
#   python side/multilabel-pilot/train.py VARIANT --here [--smoke]         # what the jobs run
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

SIDE = Path(__file__).resolve().parent
sys.path.insert(0, str(SIDE))
sys.path.insert(0, str(SIDE.parents[1]))
sys.modules.setdefault("datasets", None)  # same guard as train/train.py: the repo's datasets/ is not the HF library
from helpers.pipeline import ROOT, config  # noqa: E402
from helpers.slurm import queued, script, submit  # noqa: E402
from questions import DATASET  # noqa: E402

MODEL = "qwen3-vl:2b-instruct"
VARIANTS = ("perclass", "joint-aux")
EVALS = 5
GPUS, RAM = "a100_3g.20gb:1", "40G"  # the probe's 12.0 GB peak for this base fits the slice
# training at the labels round's 17.4 s/step (762 / 1,031 steps); scoring measured on smoke job 4727011: perclass
# 5.2 s/window (10 questions), joint 2.8 s/window, x 8,460 test windows: ~16 h and ~12 h before margins
TIME = {"perclass": "1-06:00:00", "joint-aux": "1-00:00:00", "smoke": "02:00:00"}
SMOKE_LINES = (16, 8)


def main_trainer():
    spec = importlib.util.spec_from_file_location("train_main", ROOT / "train" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = main_trainer()


def base_dir(smoke):
    return SIDE / "smoke" if smoke else SIDE


def output_dir(variant, smoke=False):
    return base_dir(smoke) / "checkpoints" / variant


def model_name(variant):
    return f"{MODEL}-sft-{DATASET}-pilot-{variant}"


def pilot_config(cfg, variant, smoke):
    settings = main.load_config(cfg, MODEL, DATASET)
    settings.update({"experiment": f"pilot-{variant}", "target": "labels", "output-dir": str(output_dir(variant, smoke))})
    return settings


def load_data(variant, smoke):
    data = {split: main.read_jsonl(SIDE / "data" / f"{variant}-{split}.jsonl") for split in ("train", "val")}
    if smoke:
        data = {"train": data["train"][:SMOKE_LINES[0]], "val": data["val"][:SMOKE_LINES[1]]}
    return {"train": data["train"], "validation": data["val"]}


def start_wandb(settings, data, resume, smoke):
    # same project as the main runs, group multilabel-pilot; offline under the side dir, uploaded later with `wandb sync`
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    out = Path(settings["output-dir"])
    id_file = out / "wandb-id"
    run_id = id_file.read_text().strip() if resume and id_file.exists() else None
    logged = {k: settings[k] for k in ("model", "dataset", "experiment", "target", "base", "lora", "training")}
    logged.update({"train_rows": len(data["train"]), "val_rows": len(data["validation"])})
    logs = base_dir(smoke) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    run = wandb.init(entity=settings["wandb"]["entity"], project=settings["wandb"]["project"],
                     name=main.wandb_name(settings), group="multilabel-pilot", tags=[DATASET, settings["experiment"]],
                     dir=logs, id=run_id, resume="allow" if run_id else None, config=logged)

    out.mkdir(parents=True, exist_ok=True)
    id_file.write_text(run.id + "\n")
    return run


def train(settings, variant, smoke):
    out = Path(settings["output-dir"])
    if (out / "manifest.json").exists():  # a resubmitted job whose scoring was cut off: go straight to score.py
        print(f"{out} is already trained (manifest.json exists)", flush=True)
        return
    resume = main.resume_point(out)
    data = load_data(variant, smoke)
    training = settings["training"]
    steps = math.ceil(len(data["train"]) / (training["batch-size"] * training["gradient-accumulation"]))
    every = 1 if smoke else math.ceil(steps / EVALS)  # save-steps must equal eval-steps for the best-checkpoint reload
    settings["training"] = {**training, "epochs": 1, "eval-steps": every, "save-steps": every}

    print(f"{MODEL} on {DATASET}, pilot {variant}: {len(data['train'])} train / {len(data['validation'])} val lines, "
          f"{steps} steps, eval every {every} -> {out}", flush=True)
    if resume:
        print(f"resuming from {resume}", flush=True)

    model, processor = main.init_model(settings)
    run = start_wandb(settings, data, resume, smoke)
    trainer = main.init_trainer(settings, model, processor, data, ROOT / "datasets" / DATASET)
    if smoke:
        trainer.args.max_steps = 2

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    processor.save_pretrained(out)

    main.write_manifest(settings, data, run.id)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update({"variant": variant, "data": str((SIDE / "data").relative_to(ROOT)), "smoke": smoke})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run.finish()


def submit_job(cfg, variant, dry_run):
    name = f"eeg-vlm-pilot-{variant}"
    if name in queued():
        sys.exit(f"{name} is already queued or running")
    for v in VARIANTS:
        if not (SIDE / "data" / f"{v}-train.jsonl").exists():
            sys.exit("no training data: python side/multilabel-pilot/build.py")

    if variant == "smoke":
        command = "\n".join(f"python {SIDE}/train.py {v} --here --smoke && python {SIDE}/score.py {v} --smoke"
                            for v in VARIANTS)
    else:
        command = f"python {SIDE}/train.py {variant} --here && python {SIDE}/score.py {variant}"
    logs = base_dir(variant == "smoke") / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    text = script(name, TIME[variant], RAM, cfg["train"]["cpus"], GPUS, command)
    text = text.replace(str(ROOT / "logs"), str(logs))
    print(f"{'dry run' if dry_run else submit(text)} - {name}, {TIME[variant]}/{RAM}/gpu:{GPUS}, log in {logs}")
    if dry_run:
        print(text)


def main_cli():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=[*VARIANTS, "smoke"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--here", action="store_true", help="train in this process (what the job runs)")
    mode.add_argument("--dry-run", action="store_true", help="print the job, submit nothing")
    parser.add_argument("--smoke", action="store_true", help="with --here: 2 steps on 16 lines, under smoke/")
    args = parser.parse_args()

    if args.here:
        if args.variant == "smoke":
            parser.error("--here takes a variant")
        os.environ.setdefault("WANDB_MODE", "offline")
        train(pilot_config(cfg, args.variant, args.smoke), args.variant, args.smoke)
        return

    submit_job(cfg, args.variant, args.dry_run)


if __name__ == "__main__":
    main_cli()
