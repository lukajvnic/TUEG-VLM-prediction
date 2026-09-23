"""Submit the custom-code bases (config.yml bases: with `family:`) that train/run.py refuses.

  minicpm       MiniCPM-V 2.6 and 4.5 through LLaMA-Factory (template minicpm_v), venv .venv-llamafactory.
                The yaml is written next to the checkpoint from config.yml train: so the hyperparameters match the
                generic runs; data from helpers/export-sft.py <DS> --format llamafactory.
  deepseek-ocr  DeepSeek-OCR through Unsloth (train/custom/unsloth_deepseek_ocr.py), venv .venv-unsloth,
                reads sft_*.jsonl directly.
  moondream     no local LoRA path (FINETUNE-TODO item 21); always skipped.

Same skip rule as run.py: a finished manifest.json means no resubmission. Everything here is unverified on a GPU.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ROOT, base_spec, checkpoint_dir, config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import job_name, queued, script, submit  # noqa: E402

sys.path.insert(0, str(ROOT / "train"))
import importlib
snapshot_dir = importlib.import_module("hf-install").snapshot_dir

STACKS = {"minicpm": ("llamafactory", ".venv-llamafactory"), "deepseek-ocr": ("unsloth", ".venv-unsloth")}
TEMPLATES = {"openbmb/MiniCPM-V-2_6": "minicpm_v", "openbmb/MiniCPM-V-4_5": "minicpm_v"}  # LLaMA-Factory template names


def llamafactory_yaml(cfg, model, dataset, base, out):
    train, lora, tr = cfg["train"], cfg["train"]["lora"], cfg["train"]["training"]
    export = ROOT / "datasets" / dataset / "export-llamafactory"
    return {
        "model_name_or_path": str(snapshot_dir(base["repo"])), "trust_remote_code": True,
        "image_max_pixels": 1536 * 1536,  # the plots' native size; the default 768*768 would downscale them
        "stage": "sft", "do_train": True, "finetuning_type": "lora",
        "lora_rank": lora["rank"], "lora_alpha": lora["alpha"], "lora_dropout": lora["dropout"], "lora_target": "all",
        "freeze_vision_tower": not lora["vision"], "freeze_multi_modal_projector": not lora["vision"],
        "dataset": f"eeg_{dataset}_train", "eval_dataset": f"eeg_{dataset}_val", "dataset_dir": str(export),
        "template": TEMPLATES[base["repo"]], "cutoff_len": 4096,
        "preprocessing_num_workers": train["cpus"], "dataloader_num_workers": 2,
        "output_dir": str(out), "logging_steps": tr["logging-steps"], "save_steps": tr["save-steps"],
        "save_total_limit": tr["save-total-limit"], "plot_loss": True, "report_to": "none",
        "per_device_train_batch_size": tr["batch-size"], "gradient_accumulation_steps": tr["gradient-accumulation"],
        "learning_rate": tr["learning-rate"], "num_train_epochs": float(tr["epochs"]),
        "lr_scheduler_type": tr["lr-scheduler"], "warmup_ratio": tr["warmup-ratio"], "weight_decay": tr["weight-decay"],
        "bf16": True, "gradient_checkpointing": True,
        "per_device_eval_batch_size": 1, "eval_strategy": "steps", "eval_steps": tr["eval-steps"],
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",  # no recording metric in this stack
        "resume_from_checkpoint": train.get("resume-from-checkpoint") or None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", help="one `bases:` key; default every base with a supported `family`")
    parser.add_argument("--dataset")
    args = parser.parse_args()
    cfg = config()
    spec = cfg["train"]
    dataset = args.dataset or spec["dataset"]
    keys = [args.model] if args.model else [k for k, v in cfg["bases"].items() if v.get("family") in STACKS]
    pending = queued()
    for model in keys:
        base = base_spec(cfg, model)
        family = base.get("family")
        if family not in STACKS:
            print(f"{model}: family {family!r} has no local training path, skipping")
            continue
        stack, venv = STACKS[family]
        out = checkpoint_dir(model, dataset)
        if (out / "manifest.json").exists():
            print(f"{model} on {dataset}: {out}/manifest.json exists (finished), skipping")
            continue
        if job_name(model) in pending:
            print(f"{model}: {job_name(model)} is already queued or running, skipping")
            continue
        if stack == "llamafactory":
            export = ROOT / "datasets" / dataset / "export-llamafactory" / "dataset_info.json"
            if not args.dry_run and not export.exists():
                sys.exit(f"missing {export} - run helpers/export-sft.py {dataset} --format llamafactory first")
            out.mkdir(parents=True, exist_ok=True)
            yaml_path = out / "llamafactory.yaml"
            yaml_path.write_text(yaml.safe_dump(llamafactory_yaml(cfg, model, dataset, base, out), sort_keys=False))
            command = (f"llamafactory-cli train {yaml_path}\n"
                       f"python {ROOT}/train/custom/finish.py --model {model} --dataset {dataset} --stack llamafactory")
        else:
            if not args.dry_run and not (ROOT / "datasets" / dataset / "sft_train.jsonl").exists():
                sys.exit(f"missing datasets/{dataset}/sft_train.jsonl - run helpers/build-sft-jsonl.py {dataset} first")
            command = f"python {ROOT}/train/custom/unsloth_deepseek_ocr.py"
        if not (ROOT / venv / "bin" / "activate").exists() and not args.dry_run:
            sys.exit(f"{model}: {venv} missing - see knowledge/operations.md 4d (custom stacks)")
        text = script(job_name(model), base["time"], base["ram"], spec["cpus"], base["gpus"], command, venv=venv)
        env = {"TRAIN_MODEL": model, "TRAIN_DATASET": dataset}
        line = f"{model} on {dataset} -> {out}  stack: {stack} ({venv})  {base['time']}/{base['ram']}/gpu:{base['gpus']}"
        if args.dry_run:
            print(text)
            print(f"# {line}")
            continue
        print(f"{submit(text, env)} - {line}")


if __name__ == "__main__":
    main()
