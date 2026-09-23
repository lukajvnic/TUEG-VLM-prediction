"""LoRA fine-tune DeepSeek-OCR with Unsloth on datasets/<DS>/sft_*.jsonl, same targets as the generic trainer.

DeepSeek-OCR ships its own model code with no HF processor, so the generic trainer cannot collate it. This follows
Unsloth's `Deepseek_OCR_(3B).ipynb` (unslothai/notebooks, read 2026-09-23): FastVisionModel over AutoModel with
trust_remote_code, and their DeepSeekOCRDataCollator copied below, which builds input_ids and image tensors with the
model's own `format_messages` / `text_encode` / `dynamic_preprocess` helpers. Hyperparameters come from config.yml
train:. Adds an eval pass on sft_val.jsonl (loss only) and writes manifest.json at the end. Unverified on a GPU.

Runs in .venv-unsloth (knowledge/operations.md 4d): the notebook pins transformers==4.56.2 and trl==0.22.2.
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ROOT, base_spec, checkpoint_dir, config  # noqa: E402
from custom.finish import write_manifest  # noqa: E402

os.environ.setdefault("UNSLOTH_WARN_UNINITIALIZED", "0")


def load_settings():
    cfg = config()
    train = cfg["train"]
    model = os.environ.get("TRAIN_MODEL") or "deepseek-ocr:3b"
    dataset = os.environ.get("TRAIN_DATASET") or train["dataset"]
    return cfg, train, model, dataset, base_spec(cfg, model), checkpoint_dir(model, dataset)


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def conversations(dataset, rows):
    # the notebook's format: `<image>` in the user text, PIL images alongside; roles are the model's own tags
    folder = ROOT / "datasets" if dataset == "pooled" else ROOT / "datasets" / dataset
    return [{"messages": [{"role": "<|User|>", "content": "<image>\n" + row["instruction"],
                           "images": [Image.open(folder / row["images"][0]).convert("RGB")]},
                          {"role": "<|Assistant|>", "content": row["output"]}]} for row in rows]


class DeepSeekOCRDataCollator:
    """Verbatim from Unsloth's notebook apart from taking the model module explicitly (its helpers live in the
    remote-code module transformers registers, which is not importable by a fixed name)."""

    def __init__(self, tokenizer, model, module, image_size=640, base_size=1024, crop_mode=True,
                 train_on_responses_only=True):
        self.tokenizer = tokenizer
        self.model = model
        self.text_encode = module.text_encode
        self.dynamic_preprocess = module.dynamic_preprocess
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype
        self.train_on_responses_only = train_on_responses_only
        self.image_transform = module.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        self.patch_size = 16
        self.downsample_ratio = 4
        self.bos_id = tokenizer.bos_token_id if getattr(tokenizer, "bos_token_id", None) is not None else 0

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        images_list, images_crop_list, images_spatial_crop = [], [], []
        if self.crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio, images_crop_raw = (1, 1), []
            else:
                images_crop_raw, crop_ratio = self.dynamic_preprocess(
                    image, min_num=2, max_num=9, image_size=self.image_size, use_thumbnail=False)
            global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                       color=tuple(int(x * 255) for x in self.image_transform.mean))
            images_list.append(self.image_transform(global_view).to(self.dtype))
            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img).to(self.dtype))
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                    num_queries * height_crop_num)
        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])
            if self.base_size <= 640:
                resized = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized).to(self.dtype))
            else:
                global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                           color=tuple(int(x * 255) for x in self.image_transform.mean))
                images_list.append(self.image_transform(global_view).to(self.dtype))
            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]
        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
        images = [img.convert("RGB") for m in messages for img in m.get("images") or [] if img is not None]
        if not images:
            raise ValueError("No images found in sample.")
        tokenized_str, images_seq_mask = [self.bos_id], [False]
        images_list, images_crop_list, images_spatial_crop = [], [], []
        prompt_token_count, assistant_started, image_idx = -1, False, 0
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True
                content = f"{content.strip()} {self.tokenizer.eos_token}"
            text_splits = content.split("<image>")
            for i, text_sep in enumerate(text_splits):
                tokenized_sep = self.text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))
                if i < len(text_splits) - 1:
                    if image_idx >= len(images):
                        raise ValueError("Found '<image>' token but no corresponding image.")
                    img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(images[image_idx])
                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)
                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))
                    image_idx += 1
        if image_idx != len(images):
            raise ValueError(f"Found {len(images)} images but only {image_idx} '<image>' tokens were used.")
        if not assistant_started:
            prompt_token_count = len(tokenized_str)
        images_ori = torch.stack(images_list, dim=0)
        images_crop = torch.stack(images_crop_list, dim=0) if images_crop_list else \
            torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)
        return {"input_ids": torch.tensor(tokenized_str, dtype=torch.long),
                "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
                "images_ori": images_ori, "images_crop": images_crop,
                "images_spatial_crop": torch.tensor(images_spatial_crop, dtype=torch.long),
                "prompt_token_count": prompt_token_count}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data = [self.process_single_sample(f["messages"]) for f in features]  # a bad sample should raise, not be skipped
        pad = self.tokenizer.pad_token_id
        input_ids = pad_sequence([b["input_ids"] for b in batch_data], batch_first=True, padding_value=pad)
        images_seq_mask = pad_sequence([b["images_seq_mask"] for b in batch_data], batch_first=True, padding_value=False)
        labels = input_ids.clone()
        labels[labels == pad] = -100
        labels[images_seq_mask] = -100
        if self.train_on_responses_only:
            for idx, count in enumerate(b["prompt_token_count"] for b in batch_data):
                if count > 0:
                    labels[idx, :count] = -100
        return {"input_ids": input_ids, "attention_mask": (input_ids != pad).long(), "labels": labels,
                "images": [(b["images_crop"], b["images_ori"]) for b in batch_data],
                "images_seq_mask": images_seq_mask,
                "images_spatial_crop": torch.cat([b["images_spatial_crop"] for b in batch_data], dim=0)}


def main():
    from unsloth import FastVisionModel
    from transformers import AutoModel, Trainer, TrainingArguments
    sys.path.insert(0, str(ROOT / "train"))
    import importlib
    snapshot_dir = importlib.import_module("hf-install").snapshot_dir

    cfg, train, model_key, dataset, base, out = load_settings()
    lora, tr = train["lora"], train["training"]
    source = str(snapshot_dir(base["repo"]))
    print(f"{model_key} ({source}) on {dataset} -> {out}", flush=True)
    folder = ROOT / "datasets" / dataset
    train_rows, val_rows = read_jsonl(folder / "sft_train.jsonl"), read_jsonl(folder / "sft_val.jsonl")
    if not train_rows or not val_rows:
        sys.exit(f"{folder}/sft_*.jsonl empty - run helpers/build-sft-jsonl.py {dataset} first")

    model, tokenizer = FastVisionModel.from_pretrained(
        source, load_in_4bit=bool(base.get("quantize-4bit", train["quantize-4bit"])), auto_model=AutoModel,
        trust_remote_code=True, unsloth_force_compile=True, use_gradient_checkpointing="unsloth")
    module = sys.modules[type(model).__module__]  # the remote-code module: format_messages, text_encode, ...
    # leaf names, matched across all experts (model.layers.N.mlp.experts.M.*, shared_experts) and, when the
    # vision side is on, the SAM (qkv/proj/lin1/lin2) and CLIP (q/k/v/out_proj, fc1/fc2) towers. The single
    # linear projector is left alone. unsloth's finetune_*_layers flags would only reach attention and the
    # dense layer-0 MLP here (its regex wants model.layers.N.{self_attn,mlp}.<leaf> or a vision tag)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if lora["vision"]:
        targets += ["qkv", "proj", "lin1", "lin2", "out_proj", "fc1", "fc2"]
    model = FastVisionModel.get_peft_model(
        model, target_modules=targets, r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        bias="none", random_state=3407)
    model.print_trainable_parameters()
    FastVisionModel.for_training(model)

    collator = DeepSeekOCRDataCollator(tokenizer, model, module)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, data_collator=collator,
        train_dataset=conversations(dataset, train_rows), eval_dataset=conversations(dataset, val_rows),
        args=TrainingArguments(
            output_dir=str(out), per_device_train_batch_size=tr["batch-size"], per_device_eval_batch_size=1,
            gradient_accumulation_steps=tr["gradient-accumulation"], num_train_epochs=tr["epochs"],
            learning_rate=tr["learning-rate"], lr_scheduler_type=tr["lr-scheduler"], warmup_ratio=tr["warmup-ratio"],
            weight_decay=tr["weight-decay"], logging_steps=tr["logging-steps"],
            eval_strategy="steps", eval_steps=tr["eval-steps"], save_steps=tr["save-steps"],
            save_total_limit=tr["save-total-limit"], load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, bf16=True, optim="adamw_torch", seed=3407, report_to="none",
            dataloader_num_workers=2, remove_unused_columns=False))
    trainer.train(resume_from_checkpoint=train.get("resume-from-checkpoint") or None)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    write_manifest(model_key, dataset, "unsloth", {"train_rows": len(train_rows), "val_rows": len(val_rows),
                                                    "corpora": sorted({r.get("dataset", dataset) for r in train_rows})})


if __name__ == "__main__":
    main()
