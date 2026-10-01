# side run: train/train.py's trainer on the rationale target with the loss on the JSON's boolean values weighted 10x
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

SIDE = Path(__file__).resolve().parent
sys.path.insert(0, str(SIDE.parents[1]))
sys.path.insert(0, str(SIDE.parents[1] / "eval" / "models"))
from helpers.pipeline import DATASETS, ROOT, base_spec, config, sft_file  # noqa: E402
from helpers.slurm import queued, script, submit  # noqa: E402
from structure import FIELDS  # noqa: E402

EXPERIMENT = "rationale-w10"
TARGET = "rationale"
LABEL_WEIGHT = 10


def main_trainer():
    # train/train.py, reused unchanged; loaded by path because train/ is not a package
    spec = importlib.util.spec_from_file_location("train_main", ROOT / "train" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = main_trainer()


def output_dir(model, dataset):
    return SIDE / "checkpoints" / model.replace(":", "-") / dataset


def job_name(model, dataset):
    return f"eeg-vlm-side-w10-{model.replace(':', '-')}-{dataset}"


def parse_args(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="a config.yml bases: key")
    parser.add_argument("dataset", choices=DATASETS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--here", action="store_true", help="train in this process (what the Slurm job runs)")
    mode.add_argument("--dry-run", action="store_true", help="print the job, submit nothing")
    args = parser.parse_args()

    if args.model not in cfg["bases"]:
        parser.error(f"unknown model {args.model!r}: not a config.yml bases: key")

    return args


def skip_reason(model, base, dataset):
    if base.get("status") or base.get("family"):
        return "not a generic-path base"
    if (output_dir(model, dataset) / "manifest.json").exists():
        return "finished (manifest.json exists)"
    if job_name(model, dataset) in queued():
        return "already queued or running"
    if not (main.snapshot_dir(base["repo"]) / "config.json").exists():
        return f"weights not staged: python train/scripts/hf-install.py {model}"
    for split in ("train", "val"):
        if not sft_file(dataset, split, TARGET).exists():
            return f"missing {sft_file(dataset, split, TARGET).relative_to(ROOT)}: python helpers/build-sft-jsonl.py {dataset}"
    leak = main.leaks(dataset, TARGET)
    if leak:
        return f"sft jsonl leaks {leak}: python helpers/build-sft-jsonl.py {dataset}"
    return None


def submit_job(cfg, model, dataset, dry_run):
    base = base_spec(cfg, model)
    reason = skip_reason(model, base, dataset)

    if reason:
        print(f"{model} on {dataset} ({EXPERIMENT}): {reason}, skipping")
        return

    (SIDE / "logs").mkdir(exist_ok=True)
    text = script(job_name(model, dataset), base["time"], base["ram"], cfg["train"]["cpus"], base["gpus"],
                  f"python {SIDE}/train.py {model} {dataset} --here")
    text = text.replace(str(ROOT / "logs"), str(SIDE / "logs"))  # Slurm log and tasks.csv stay in the side dir
    job = "dry run" if dry_run else submit(text)

    print(f"{job} - {model} on {dataset} ({EXPERIMENT}) -> {output_dir(model, dataset)}, "
          f"{base['time']}/{base['ram']}/gpu:{base['gpus']}")
    if dry_run:
        print(text)


def side_config(cfg, model, dataset):
    settings = main.load_config(cfg, model, dataset)
    settings.update({"experiment": EXPERIMENT, "target": TARGET, "output-dir": str(output_dir(model, dataset))})
    return settings


def weighted_collator(base_collator, booleans, fields):
    # the parent collator's batch plus label_weights: 1 on every target token, LABEL_WEIGHT on the JSON's boolean
    # values. The target is rationale first, so those values are the last `fields` true/false tokens; a "true" or
    # "false" inside the prose comes earlier and keeps weight 1
    import torch

    ids = torch.tensor(sorted(booleans["true"] | booleans["false"]))

    def collate(examples):
        batch = base_collator(examples)
        labels = batch["labels"]
        weights = (labels != -100).float()

        for row in range(labels.shape[0]):
            positions = torch.isin(labels[row], ids).nonzero().flatten()
            if len(positions) < fields:
                raise ValueError(f"found {len(positions)} true/false tokens in a target, expected at least {fields}")
            weights[row, positions[-fields:]] = LABEL_WEIGHT

        batch["label_weights"] = weights
        return batch

    return collate


def weighted_trainer_class():
    import torch.nn.functional as F
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # compute_loss normalises by num_items_in_batch itself, so training_step must not also divide by the
            # gradient-accumulation steps
            self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            labels, weights = inputs.pop("labels"), inputs.pop("label_weights")
            outputs = model(**inputs)

            targets, weights = labels[:, 1:], weights[:, 1:]
            mask = targets != -100
            losses = F.cross_entropy(outputs.logits[:, :-1][mask].float(), targets[mask], reduction="none")
            count = num_items_in_batch if num_items_in_batch is not None else mask.sum()
            loss = (losses * weights[mask]).sum() / count

            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def build_trainer(settings, model, processor, data):
    # the main trainer's TrainingArguments, optimizer and metrics, with the weighted collator and loss
    dataset_dir = ROOT / "datasets" / settings["dataset"]
    base = main.init_trainer(settings, model, processor, data, dataset_dir)
    booleans = main.boolean_token_ids(processor.tokenizer)
    collate = weighted_collator(base.data_collator, booleans, len(FIELDS[settings["dataset"]]))

    return weighted_trainer_class()(
        model=base.model, args=base.args, train_dataset=data["train"], eval_dataset=data["validation"],
        data_collator=collate, processing_class=processor, optimizers=(base.optimizer, None),
        compute_metrics=base.compute_metrics, preprocess_logits_for_metrics=base.preprocess_logits_for_metrics)


def start_wandb(settings, data, resume):
    # same project as the main runs, own group; offline under the side dir, uploaded later with `wandb sync`
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    out = Path(settings["output-dir"])
    id_file = out / "wandb-id"
    run_id = id_file.read_text().strip() if resume and id_file.exists() else None
    logged = {k: settings[k] for k in ("model", "dataset", "experiment", "target", "base", "lora", "training")}
    logged.update({"label_weight": LABEL_WEIGHT, "train_rows": len(data["train"]), "val_rows": len(data["validation"])})

    (SIDE / "logs").mkdir(exist_ok=True)
    run = wandb.init(entity=settings["wandb"]["entity"], project=settings["wandb"]["project"],
                     name=main.wandb_name(settings), group=EXPERIMENT,
                     tags=[settings["dataset"], TARGET, f"label-weight-{LABEL_WEIGHT}"], dir=SIDE / "logs",
                     id=run_id, resume="allow" if run_id else None, config=logged)

    out.mkdir(parents=True, exist_ok=True)
    id_file.write_text(run.id + "\n")
    return run


def train(settings):
    out = Path(settings["output-dir"])
    resume = main.resume_point(out)

    print(f"{settings['model']} on {settings['dataset']}, {EXPERIMENT} (target {TARGET}, boolean values x{LABEL_WEIGHT}) "
          f"-> {out}", flush=True)
    if resume:
        print(f"resuming from {resume}", flush=True)

    data = main.load_data(settings)
    model, processor = main.init_model(settings)
    run = start_wandb(settings, data, resume)
    trainer = build_trainer(settings, model, processor, data)

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    processor.save_pretrained(out)

    main.write_manifest(settings, data, run.id)
    manifest = json.loads((out / "manifest.json").read_text())
    (out / "manifest.json").write_text(json.dumps({**manifest, "label_weight": LABEL_WEIGHT}, indent=2) + "\n")
    run.finish()


def main_cli():
    cfg = config()
    args = parse_args(cfg)

    if args.here:
        train(side_config(cfg, args.model, args.dataset))
        return

    submit_job(cfg, args.model, args.dataset, args.dry_run)


if __name__ == "__main__":
    main_cli()
