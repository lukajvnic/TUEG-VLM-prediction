#!/usr/bin/env python3
"""Fine-tune Qwen2.5-VL with LoRA on one EEG SFT dataset."""

from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "train" / "config.yml"

# Each JSONL line is expected to look like:
# {"instruction": "Review this EEG image...", "input": "", "output":
# "{\"has_epilepsy\": true, \"text_rationale\": \"...\"}",
# "images": ["data/example.png"]}


def load_config():
    with CONFIG_PATH.open() as file:
        return yaml.safe_load(file)


def get_messages(example, include_answer):
    prompt = example["instruction"]
    if example.get("input"):
        prompt = f"{prompt}\n\n{example['input']}"

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    if include_answer:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": example["output"]}],
        })
    return messages


class DataCollator:
    def __init__(self, processor, dataset_dir):
        self.processor = processor
        self.dataset_dir = dataset_dir

    def __call__(self, examples):
        images = []
        texts = []
        prefix_lengths = []

        for example in examples:
            with Image.open(self.dataset_dir / example["images"][0]) as image:
                image = image.convert("RGB")
            images.append(image)
            texts.append(self.processor.apply_chat_template(
                get_messages(example, include_answer=True),
                tokenize=False,
                add_generation_prompt=False,
            ))
            prefix = self.processor.apply_chat_template(
                get_messages(example, include_answer=False),
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_lengths.append(self.processor(
                text=prefix,
                images=image,
                return_tensors="pt",
            )["input_ids"].shape[1])

        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for index, length in enumerate(prefix_lengths):
            labels[index, :length] = -100
        batch["labels"] = labels
        return batch


def load_data(dataset_dir):
    return load_dataset("json", data_files={
        "train": dataset_dir / "sft_train.jsonl",
        "validation": dataset_dir / "sft_val.jsonl",
    })


def init_model(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning requires a CUDA GPU.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(config["model"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model"],
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    return model, processor


def init_trainer(config, model, processor, data, dataset_dir):
    settings = config["training"]
    output_dir = ROOT / config["output-dir"]
    use_bf16 = torch.cuda.is_bf16_supported()
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch-size"],
        per_device_eval_batch_size=settings["batch-size"],
        gradient_accumulation_steps=settings["gradient-accumulation"],
        learning_rate=settings["learning-rate"],
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=settings["eval-steps"],
        save_strategy="steps",
        save_steps=settings["save-steps"],
        save_total_limit=settings["save-total-limit"],
        logging_steps=settings["logging-steps"],
        report_to="none",
        remove_unused_columns=False,
    )
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollator(processor, dataset_dir),
        processing_class=processor,
    )


def main():
    config = load_config()
    dataset_dir = ROOT / "datasets" / config["dataset"]
    data = load_data(dataset_dir)
    model, processor = init_model(config)
    trainer = init_trainer(config, model, processor, data, dataset_dir)
    trainer.train(resume_from_checkpoint=config.get("resume-from-checkpoint"))
    trainer.save_model()
    processor.save_pretrained(ROOT / config["output-dir"])


if __name__ == "__main__":
    main()
