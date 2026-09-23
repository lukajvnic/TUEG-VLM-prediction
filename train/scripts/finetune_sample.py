#!/usr/bin/env python3
"""LoRA fine-tune of one HF vision-language base on one EEG SFT dataset (config.yml train: + bases:).

Model-agnostic on purpose (2026-09-21): every base in `bases:` loads through AutoModelForImageTextToText and its own
chat template; the answer span is found by tokenising the prompt-only template, and LoRA targets are enumerated
from the loaded modules instead of hard-coded names. TRAIN_MODEL / TRAIN_DATASET (set by train/run.py) override
config.yml train.model / train.dataset.
"""

import datetime
import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "config.yml"
for folder in ("", "train", "eval", "eval/models"):
    sys.path.insert(0, str(ROOT / folder))
from helpers.pipeline import base_spec, checkpoint_dir, parse_name, spread  # noqa: E402
from structure import get_structure, labels as label_names, to_labels  # noqa: E402
import score  # noqa: E402  the scoreboard's recording aggregation and balanced accuracy, so val measures the same thing
snapshot_dir = import_module("hf-install").snapshot_dir
runner = import_module("predict")  # shares generate(), lora_targets() and the schema enforcer with the scoring script

# Each JSONL line (from helpers/build-sft-jsonl.py) looks like:
# {"instruction": "This image is a multi-channel EEG...", "input": "",
#  "output": "{\"text_rationale\": \"...\", \"has_epilepsy\": true}", "images": ["train/aaaaaaaa_0_3.png"]}


def load_config():
    with CONFIG_PATH.open() as file:
        cfg = yaml.safe_load(file)
    train = cfg["train"]
    train["model"] = os.environ.get("TRAIN_MODEL") or train["model"]
    train["dataset"] = os.environ.get("TRAIN_DATASET") or train["dataset"]
    train["base"] = base_spec(cfg, train["model"])
    train["quantize-4bit"] = train["base"].get("quantize-4bit", train["quantize-4bit"])  # llama4 needs it
    train["output-dir"] = str(checkpoint_dir(train["model"], train["dataset"]))
    return train


POOLED = "pooled"  # datasets/pooled/sft_*.jsonl from build-sft-jsonl.py: every line carries its `dataset`


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(dataset_dir):
    data = {"train": read_jsonl(dataset_dir / "sft_train.jsonl"),
            "validation": read_jsonl(dataset_dir / "sft_val.jsonl")}
    for name, rows in data.items():
        if not rows:
            raise RuntimeError(f"{dataset_dir}/sft_{name}.jsonl is empty - run helpers/build-sft-jsonl.py where rationales exist")
    return data


def resolve_model(repo_id):
    # compute nodes have no internet: use the snapshot train/hf-install.py staged, else fall back to the hub id
    local = snapshot_dir(repo_id)
    return str(local) if (local / "config.json").exists() else repo_id


def get_messages(example, include_answer, answer_suffix=""):
    prompt = example["instruction"]
    if example.get("input"):
        prompt = f"{prompt}\n\n{example['input']}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if include_answer:  # answer_suffix: `bases:` answer-suffix, the stop token for templates that render none
        messages.append({"role": "assistant", "content": [{"type": "text", "text": example["output"] + answer_suffix}]})
    return messages


class DataCollator:
    # loss only on the assistant turn. The prompt span is measured by running the prompt-only template (the
    # same text generation starts from) through the processor, so image-token expansion and the template's
    # own markers are counted for whatever base this is; no per-model marker string

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
                text = prompt + text[len(prompt) - len(empty):]  # gemma4: keep the empty thinking channel in the prompt
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
            first = int(mask.nonzero()[0])  # left padding shifts the real tokens right
            labels[row, :first + int(prompt_lengths[row])] = -100
        batch["labels"] = labels
        return batch


def init_model(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning requires a CUDA GPU.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = config["base"]
    source = resolve_model(base["repo"])
    print(f"loading {source}", flush=True)
    loading = dict(dtype=dtype, device_map="auto", trust_remote_code=base.get("trust-remote-code", False))
    if config["quantize-4bit"]:
        from transformers import BitsAndBytesConfig
        loading["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            # never quantise the image side. Passing any list drops transformers' default skips, so lm_head is
            # named too. bitsandbytes matches these as module-name prefixes/suffixes, not substrings: it covers
            # llama4 (top-level `vision_model`, `multi_modal_projector`), not e.g. Qwen's `model.visual`
            llm_int8_skip_modules=[*runner.VISION_HINTS, "lm_head"])
    processor = runner.load_processor(source, base, loading["trust_remote_code"])
    model = AutoModelForImageTextToText.from_pretrained(source, **loading)
    model.config.use_cache = False
    if config["quantize-4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.enable_input_require_grads()  # gradient checkpointing needs a grad path through frozen base weights

    lora = config["lora"]
    language, vision = runner.lora_targets(model)
    targets = language + (vision if lora["vision"] else [])
    if not language:
        raise RuntimeError("no language Linear modules found to adapt; check the loaded architecture")
    model = get_peft_model(model, LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    print(f"LoRA on {len(targets)} modules: {len(language)} language, {len(targets) - len(language)} vision "
          f"(vision side has {len(vision)} Linear modules)", flush=True)
    model.print_trainable_parameters()
    return model, processor


def build_optimizer(model, config):
    # the vision tower gets a scaled learning rate; Trainer builds the schedule on top of these groups
    lr, scale = config["training"]["learning-rate"], config["lora"]["vision-lr-scale"]
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": [p for name, p in params if not runner.is_vision(name)], "lr": lr},
              {"params": [p for name, p in params if runner.is_vision(name)], "lr": lr * scale}]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, weight_decay=config["training"]["weight-decay"])


def label_set(structure, dataset, text):
    return frozenset(c for c, v in to_labels(structure(**json.loads(text)), dataset).items() if v)


def dataset_of(example, config):
    return example.get("dataset", config["dataset"])


def val_slice(examples, budget, config):
    # whole recordings, evenly spread over the val set, about `budget` windows: the scoreboard aggregates
    # windows per recording (majority vote / any-positive), so a per-window slice would measure something else
    by_recording = {}
    for example in examples:
        by_recording.setdefault((dataset_of(example, config), parse_name(example["images"][0])[1]), []).append(example)
    recordings = sorted(by_recording)
    per_recording = max(1, len(examples) // len(recordings))
    picked = spread(recordings, max(1, budget // per_recording))
    return [example for recording in picked for example in by_recording[recording]]


def class_scores(windows, dataset, classes):
    # windows: (recording, predicted labels, true labels); classes scored are those with a positive in the slice
    window_pairs = [(pred, true) for _, pred, true in windows]
    recording_pairs = score.to_recordings(windows, dataset)
    present = [c for c in classes if any(c in true for _, true in recording_pairs)]
    kept = score.recording_classes(dataset, present)  # same exclusions as the scoreboard (TUSZ bckg)
    return {"recording_balanced_accuracy": score.balanced_accuracy(recording_pairs, kept),
            "recording_macro_f1": score.macro_f1(score.per_class(recording_pairs, kept), kept),
            "balanced_accuracy": score.balanced_accuracy(window_pairs, present),
            "macro_f1": score.macro_f1(score.per_class(window_pairs, present), present)}


class GenerationTrainer(Trainer):
    # eval loss measures how gemma-like the prose is; generating on a fixed slice of val and scoring the
    # labels the way eval/score.py does (per recording) measures the task, and that picks the checkpoint

    def __init__(self, *args, generation, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation = generation

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        g = self.generation
        if not g["examples"]:
            return metrics
        self.model.eval()
        windows, ok = {}, 0
        for example in g["examples"]:
            ds = dataset_of(example, g["config"])
            truth = label_set(g["structures"][ds], ds, example["output"])
            try:
                text = runner.generate(self.model, g["processor"], g["dataset_dir"] / example["images"][0],
                                       example["instruction"], g["prefix_fns"][ds])
                pred = label_set(g["structures"][ds], ds, text)
                ok += 1
            except Exception as e:  # unparseable or truncated reply counts as predicting nothing
                print(f"val generation failed: {e}", flush=True)
                pred = frozenset()
            windows.setdefault(ds, []).append((parse_name(example["images"][0])[1], pred, truth))
        per_dataset = {ds: class_scores(rows, ds, label_names(ds)) for ds, rows in windows.items()}
        # the headline is the mean over corpora, so a pooled run is judged the way the paper reports it
        scores = {k: sum(v[k] for v in per_dataset.values()) / len(per_dataset) for k in next(iter(per_dataset.values()))}
        if len(per_dataset) > 1:
            scores.update({f"{k}_{ds}": v[k] for ds, v in per_dataset.items() for k in v})
        scores["json_ok"] = ok / sum(len(rows) for rows in windows.values())
        scores = {f"{metric_key_prefix}_{k}": v for k, v in scores.items()}
        self.log(scores)
        metrics.update(scores)
        return metrics


def init_trainer(config, model, processor, data, dataset_dir):
    settings, dataset = config["training"], config["dataset"]
    generate_n = config["val"]["generate"]
    use_bf16 = torch.cuda.is_bf16_supported()
    arguments = TrainingArguments(
        output_dir=config["output-dir"],
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch-size"],
        per_device_eval_batch_size=settings["batch-size"],
        gradient_accumulation_steps=settings["gradient-accumulation"],
        learning_rate=settings["learning-rate"],
        warmup_steps=settings["warmup-ratio"],  # a float < 1 is a ratio; warmup_ratio is deprecated in transformers 5
        lr_scheduler_type=settings["lr-scheduler"],
        weight_decay=settings["weight-decay"],
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=settings["eval-steps"],
        save_strategy="steps",
        save_steps=settings["save-steps"],
        save_total_limit=settings["save-total-limit"],
        load_best_model_at_end=True,
        metric_for_best_model="recording_balanced_accuracy" if generate_n else "loss",
        greater_is_better=bool(generate_n),
        logging_steps=settings["logging-steps"],
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
    )
    corpora = sorted({dataset_of(e, config) for e in data["train"] + data["validation"]})
    structures = {ds: get_structure(ds, config["rationale-first"]) for ds in corpora}
    try:
        prefix_fns = {ds: runner.enforcer(processor, structures[ds]) for ds in corpora}
    except ImportError:
        prefix_fns = dict.fromkeys(corpora)
        print("lm-format-enforcer not installed: val generation is unconstrained", flush=True)
    generation = {"examples": val_slice(data["validation"], generate_n, config) if generate_n else [],
                  "processor": processor, "config": config, "dataset_dir": dataset_dir,
                  "structures": structures, "prefix_fns": prefix_fns}
    return GenerationTrainer(
        model=model,
        args=arguments,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollator(processor, dataset_dir, config["base"]),
        processing_class=processor,
        optimizers=(build_optimizer(model, config), None),
        generation=generation,
    )


def write_manifest(config, data):
    # train/predict.py refuses a checkpoint without this: it marks a finished run and says what went in
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                capture_output=True).stdout.strip()
    except OSError:
        commit = ""
    manifest = {"dataset": config["dataset"], "model": config["model"], "base": config["base"],
                "corpora": sorted({dataset_of(e, config) for e in data["train"]}),
                "rationale_first": config["rationale-first"],
                "train_rows": len(data["train"]), "val_rows": len(data["validation"]),
                "lora": config["lora"], "training": config["training"], "val": config["val"],
                "quantize-4bit": config["quantize-4bit"], "commit": commit,
                "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    (Path(config["output-dir"]) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    config = load_config()
    dataset_dir = ROOT / "datasets" if config["dataset"] == POOLED else ROOT / "datasets" / config["dataset"]
    print(f"{config['model']} ({config['base']['repo']}) on {config['dataset']} -> {config['output-dir']}", flush=True)
    data = load_data(ROOT / "datasets" / config["dataset"])
    model, processor = init_model(config)
    trainer = init_trainer(config, model, processor, data, dataset_dir)
    trainer.train(resume_from_checkpoint=config.get("resume-from-checkpoint"))
    trainer.save_model()  # the best checkpoint by the val metric, since load_best_model_at_end is on
    processor.save_pretrained(config["output-dir"])
    write_manifest(config, data)


if __name__ == "__main__":
    main()
