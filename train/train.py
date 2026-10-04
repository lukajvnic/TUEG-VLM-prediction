import argparse
import base64
import datetime
import functools
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# with the repo root on sys.path, transformers 5.14 imports the repo's datasets/ folder as the Hugging Face `datasets`
# library (not installed here) and the Trainer dies on `datasets.Dataset` when it builds the first dataloader (jobs
# 4342993 and 4400514, 2026-10-03); marking it absent makes transformers skip that path
sys.modules.setdefault("datasets", None)
from helpers.pipeline import (DATASETS, ROOT, base_spec, checkpoint_dir, config, run_name, script_module,  # noqa: E402
                              sft_file, sync)
from helpers.slurm import job_name, queued, script, submit  # noqa: E402

snapshot_dir = script_module("hf-install").snapshot_dir
runner = script_module("eval")


def parse_args(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="a config.yml bases: key, or all")
    parser.add_argument("dataset", choices=[*DATASETS, "all"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--here", action="store_true", help="train in this process (what the Slurm job runs)")
    mode.add_argument("--dry-run", action="store_true", help="print the jobs, submit nothing")
    args = parser.parse_args()

    if args.model != "all" and args.model not in cfg["bases"]:
        parser.error(f"unknown model {args.model!r}: not a config.yml bases: key")
    if args.here and "all" in (args.model, args.dataset):
        parser.error("--here trains one model on one dataset")

    return args


def skip_reason(model, base, dataset, run, target, pending):
    # a submission that cannot succeed costs a queue slot and, at worst, a day; say why instead
    if base.get("status"):
        return f"{base['status']} (config.yml bases:)"
    if base.get("family"):
        return "custom-code base: train/run-custom.py"
    if (checkpoint_dir(model, run) / "manifest.json").exists():
        return "finished (manifest.json exists)"
    if job_name(model, run) in pending:
        return "already queued or running"
    if not (snapshot_dir(base["repo"]) / "config.json").exists():
        return f"weights not staged: python train/scripts/hf-install.py {model}"
    for split in ("train", "val"):
        if not sft_file(dataset, split, target).exists():
            return f"missing {sft_file(dataset, split, target).relative_to(ROOT)}: python helpers/build-sft-jsonl.py {dataset}"
    leak = leaks(dataset, target)
    if leak:
        return f"sft jsonl leaks {leak}: python helpers/build-sft-jsonl.py {dataset}"
    return None


@functools.cache
def split_checker():
    # helpers/check-splits.py (hyphenated, so loaded by path) and its test-side keys, built once and only when a
    # pair gets past the cheaper checks
    spec = importlib.util.spec_from_file_location("check_splits", ROOT / "helpers" / "check-splits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.Leakage()


@functools.cache
def leaks(dataset, target):
    # a jsonl built before test screening (the cluster's 2026-09-16 ones hold 44 TUAB and 50 TUSZ test patients)
    # would void the test scores; checked once per dataset
    module, leakage = split_checker()
    found = leakage.overlaps(dataset, module.read_split(dataset, "train", target),
                             module.read_split(dataset, "val", target))
    return ", ".join(f"{len(keys)} {check}" for check, keys in found.items() if keys)


def job_command(model, dataset, run):
    # train, then score the pair's test split in the same allocation: a separate scoring job waits in the queue
    # again (~2 days at the 2026-10 fairshare). `&&`: no scoring after a failed run. The scoring step is
    # train/scripts/eval.py as run-eval.py would run it; its pipeline.db rows come from sync() (run helpers/pipeline.py
    # once the checkpoint dir exists) and a rerun of run-eval.py resumes whatever it leaves pending
    task = base64.b64encode(json.dumps([{"model": f"{model}-sft-{run}", "dataset": dataset}]).encode()).decode()
    return (f"python {ROOT}/train/train.py {model} {dataset} --here && "
            f"SLURM_ARRAY_TASK_ID=0 SFT_EVAL_TASKS={task} python {ROOT}/train/scripts/eval.py")


def submit_jobs(cfg, models, datasets, dry_run):
    spec = cfg["train"]
    target = spec["target"]
    pending = queued()
    submitted = 0

    for model in models:
        base = base_spec(cfg, model)
        for dataset in datasets:
            run = run_name(dataset, spec["experiment"])
            reason = skip_reason(model, base, dataset, run, target, pending)

            if reason:
                print(f"{model} on {run}: {reason}, skipping")
                continue

            text = script(job_name(model, run), base["time"], base["ram"], spec["cpus"], base["gpus"],
                          job_command(model, dataset, run))
            job = "dry run" if dry_run else submit(text)
            if not dry_run:
                checkpoint_dir(model, run).mkdir(parents=True, exist_ok=True)
                submitted += 1

            print(f"{job} - {model} on {run} -> {checkpoint_dir(model, run)}, "
                  f"{base['time']}/{base['ram']}/gpu:{base['gpus']}")

    if submitted:
        # the job's scoring step reads its test windows from pipeline.db, and sync() only registers a model whose
        # checkpoint dir exists: the dirs were just made, so register them now, hours before any job scores
        print(f"registering the test windows of {submitted} submitted pairs in pipeline.db (sync) ...", flush=True)
        sync()


def load_config(cfg, model, dataset):
    train = cfg["train"]
    train["model"] = model
    train["dataset"] = dataset
    train["base"] = base_spec(cfg, model)
    train["output-dir"] = str(checkpoint_dir(model, run_name(dataset, train["experiment"])))
    return train


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(config):
    return {"train": read_jsonl(sft_file(config["dataset"], "train", config["target"])),
            "validation": read_jsonl(sft_file(config["dataset"], "val", config["target"]))}


def resolve_model(repo_id):
    local = snapshot_dir(repo_id)
    return str(local) if (local / "config.json").exists() else repo_id


def get_messages(example, include_answer, answer_suffix=""):
    prompt = example["instruction"]
    if example.get("input"):
        prompt = f"{prompt}\n\n{example['input']}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if include_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": example["output"] + answer_suffix}]})
    return messages


class DataCollator:
    def __init__(self, processor, dataset_dir, base):
        self.processor = processor
        self.dataset_dir = dataset_dir
        self.answer_suffix = base.get("answer-suffix", "")

    def __call__(self, examples):
        images, prompts, full = [], [], []
        empty = runner.EMPTY_THOUGHT

        for example in examples:
            with Image.open(self.dataset_dir / example["images"][0]) as image:
                images.append(image.convert("RGB"))

            prompt = runner.render(self.processor, get_messages(example, False), True)
            text = runner.render(self.processor, get_messages(example, True, self.answer_suffix), False)

            if not text.startswith(prompt) and prompt.endswith(empty) and text.startswith(prompt[:-len(empty)]):
                text = prompt + text[len(prompt) - len(empty):]
            if not text.startswith(prompt):
                raise ValueError("chat template: the full turn does not start with the generation prompt; "
                                 "answer masking would be wrong for this base")

            prompts.append(prompt)
            full.append(text)

        extra = runner.encode_kwargs(self.processor, full[0])
        batch = self.processor(text=full, images=images, padding=True, return_tensors="pt", **extra)
        prompt_lengths = self.processor(text=prompts, images=images, padding=True,
                                        return_tensors="pt", **extra)["attention_mask"].sum(dim=1)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for row, mask in enumerate(batch["attention_mask"]):
            first = int(mask.nonzero()[0])
            labels[row, :first + int(prompt_lengths[row])] = -100

        batch["labels"] = labels
        return batch


def init_model(config):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = config["base"]
    source = resolve_model(base["repo"])
    loading = dict(dtype=dtype, device_map="auto", trust_remote_code=base.get("trust-remote-code", False))

    processor = runner.load_processor(source, base, loading["trust_remote_code"])
    model = AutoModelForImageTextToText.from_pretrained(source, **loading)
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora = config["lora"]
    language, vision = runner.lora_targets(model)
    targets = language + (vision if lora["vision"] else [])

    if not language:
        raise RuntimeError("no language Linear modules found to adapt; check the loaded architecture")

    model = get_peft_model(model, LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=targets))

    model.print_trainable_parameters()
    return model, processor


def build_optimizer(model, config):
    import torch

    lr, scale = config["training"]["learning-rate"], config["lora"]["vision-lr-scale"]
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": [p for name, p in params if not runner.is_vision(name)], "lr": lr},
              {"params": [p for name, p in params if runner.is_vision(name)], "lr": lr * scale}]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, weight_decay=config["training"]["weight-decay"])


def boolean_token_ids(tokenizer):
    ids = {"true": set(), "false": set()}

    for token_id in range(len(tokenizer)):
        word = tokenizer.decode([token_id]).strip().lower()
        if word in ids:
            ids[word].add(token_id)

    return ids


def label_metrics(booleans):
    import numpy as np

    def compute(eval_prediction):
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids
        preds, labels = preds[:, :-1], labels[:, 1:]
        per_class = []

        for word, ids in booleans.items():
            positions = np.isin(labels, list(ids))
            if positions.any():
                per_class.append(float(np.isin(preds[positions], list(ids)).mean()))

        return {"label_balanced_accuracy": sum(per_class) / len(per_class) if per_class else 0.0,
                "label_positions": int(sum(np.isin(labels, list(i)).sum() for i in booleans.values()))}

    return compute


def argmax_only(logits, labels):
    return (logits[0] if isinstance(logits, tuple) else logits).argmax(-1)


def wandb_name(config):
    return f"{config['model']}-{run_name(config['dataset'], config['experiment'])}"


def resume_point(out):
    # a job that died or hit its walltime left checkpoints and no manifest: the resubmission continues from the latest
    # complete one. trainer_state.json is written last, so a checkpoint dir without it was cut off mid-save and would
    # crash the resume (transformers' get_last_checkpoint takes the highest number regardless)
    if (out / "manifest.json").exists() or not out.is_dir():
        return None
    complete = [p for p in out.glob("checkpoint-*") if (p / "trainer_state.json").exists()]
    return str(max(complete, key=lambda p: int(p.name.split("-")[1]))) if complete else None


def start_wandb(config, data, resume):
    # compute nodes have no internet: runs land in logs/wandb/ and are uploaded with `wandb sync` from a login node.
    # A resumed job reuses the run id so the curve continues; the HF callback logs into this run
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    out = Path(config["output-dir"])
    id_file = out / "wandb-id"
    run_id = id_file.read_text().strip() if resume and id_file.exists() else None
    settings = {"model": config["model"], "dataset": config["dataset"], "experiment": config["experiment"],
                "target": config["target"],
                "base": config["base"], "train_rows": len(data["train"]), "val_rows": len(data["validation"]),
                "rationale_first": config["rationale-first"], "lora": config["lora"], "training": config["training"]}

    (ROOT / "logs").mkdir(exist_ok=True)
    run = wandb.init(entity=config["wandb"]["entity"], project=config["wandb"]["project"], name=wandb_name(config),
                     group=config["experiment"], tags=[config["dataset"], config["target"]], dir=ROOT / "logs", id=run_id,
                     resume="allow" if run_id else None, config=settings)

    out.mkdir(parents=True, exist_ok=True)
    id_file.write_text(run.id + "\n")
    return run


def init_trainer(config, model, processor, data, dataset_dir):
    from transformers import Trainer, TrainingArguments

    settings = config["training"]

    arguments = TrainingArguments(
        output_dir=config["output-dir"],
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch-size"],
        per_device_eval_batch_size=settings["batch-size"],
        gradient_accumulation_steps=settings["gradient-accumulation"],
        learning_rate=settings["learning-rate"],
        warmup_steps=settings["warmup-ratio"],
        lr_scheduler_type=settings["lr-scheduler"],
        weight_decay=settings["weight-decay"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=settings["eval-steps"],
        save_strategy="steps",
        save_steps=settings["save-steps"],
        save_total_limit=settings["save-total-limit"],
        load_best_model_at_end=True,
        metric_for_best_model="label_balanced_accuracy",
        greater_is_better=True,
        logging_steps=settings["logging-steps"],
        dataloader_num_workers=4,
        report_to="wandb",
        run_name=wandb_name(config),
        remove_unused_columns=False,
    )

    return Trainer(
        model=model,
        args=arguments,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollator(processor, dataset_dir, config["base"]),
        processing_class=processor,
        optimizers=(build_optimizer(model, config), None),
        compute_metrics=label_metrics(boolean_token_ids(processor.tokenizer)),
        preprocess_logits_for_metrics=argmax_only,
    )


def write_manifest(config, data, wandb_id):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                capture_output=True).stdout.strip()
    except OSError:
        commit = ""
    manifest = {"dataset": config["dataset"], "model": config["model"], "base": config["base"],
                "experiment": config["experiment"], "target": config["target"],
                "rationale_first": config["target"] == "rationale" and config["rationale-first"],
                "train_rows": len(data["train"]), "val_rows": len(data["validation"]),
                "lora": config["lora"], "training": config["training"], "val": config["val"],
                "commit": commit, "wandb_id": wandb_id,
                "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    (Path(config["output-dir"]) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def train(config):
    dataset_dir = ROOT / "datasets" / config["dataset"]
    resume = resume_point(Path(config["output-dir"]))

    print(f"{config['model']} ({config['base']['repo']}) on {config['dataset']}, experiment {config['experiment']} "
          f"(target {config['target']}) "
          f"-> {config['output-dir']}", flush=True)
    if resume:
        print(f"resuming from {resume}", flush=True)

    data = load_data(config)
    model, processor = init_model(config)
    run = start_wandb(config, data, resume)
    trainer = init_trainer(config, model, processor, data, dataset_dir)

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    processor.save_pretrained(config["output-dir"])

    write_manifest(config, data, run.id)
    run.finish()


def main():
    cfg = config()
    args = parse_args(cfg)

    if args.here:
        train(load_config(cfg, args.model, args.dataset))
        return

    models = list(cfg["bases"]) if args.model == "all" else [args.model]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    submit_jobs(cfg, models, datasets, args.dry_run)


if __name__ == "__main__":
    main()
