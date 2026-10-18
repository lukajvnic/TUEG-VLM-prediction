# P(true) for 20 s EEG windows rendered in the paper's core design, from one packaged fine-tuned model, computed the
# way the paper's scorer does (run.py p_true: the answer is teacher-forced and P(true) is read at its boolean token).
# --explain also generates the model's rationale. The base model downloads from Hugging Face on first use
#   python predict.py models/<name> IMAGE [IMAGE ...] [--explain] [--out predictions.csv]
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

EMPTY_THOUGHT = "<|channel>thought\n<channel|>"  # Gemma 4's template writes an empty thought block into the prompt
MAX_NEW_TOKENS = 512


def load(model_dir):
    spec = json.loads((model_dir / "model.json").read_text())
    # on CPU everything stays in RAM: device_map="auto" offloads to disk when RAM is short, which peft cannot merge into
    model = AutoModelForImageTextToText.from_pretrained(spec["base"], dtype=torch.bfloat16,
                                                        device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(model, str(model_dir / "adapter")).merge_and_unload()
    model.eval()
    text = model.config.get_text_config()
    if getattr(text, "final_logit_softcapping", None):  # trained and scored without Gemma 4's logit cap
        text.final_logit_softcapping = None
    return spec, model, AutoProcessor.from_pretrained(str(model_dir / "adapter"))


def boolean_ids(tokenizer):
    ids = {"true": [], "false": []}
    for token_id in range(len(tokenizer)):
        word = tokenizer.decode([token_id]).strip().lower()
        if word in ids:
            ids[word].append(token_id)
    return ids


def messages(prompt, answer=None):
    turns = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        turns.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return turns


def encode(processor, text, image):
    bos = getattr(processor.tokenizer, "bos_token", None)
    extra = {"add_special_tokens": False} if bos and text.startswith(bos) else {}
    return processor(text=[text], images=[image], return_tensors="pt", **extra)


def p_true(model, processor, ids, prompt, answer, image):
    head = processor.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(messages(prompt, answer), tokenize=False, add_generation_prompt=False)
    if not full.startswith(head) and head.endswith(EMPTY_THOUGHT) and full.startswith(head[:-len(EMPTY_THOUGHT)]):
        full = head + full[len(head) - len(EMPTY_THOUGHT):]
    assert full.startswith(head), "chat template: the answer turn does not extend the prompt"
    batch = encode(processor, full, image)
    start = int(encode(processor, head, image)["attention_mask"].sum())
    tokens = batch["input_ids"][0]
    every = torch.tensor(ids["true"] + ids["false"])
    position = start + int(torch.isin(tokens[start:], every).nonzero()[0])
    with torch.inference_mode():
        logits = model(**batch.to(model.device)).logits[0, position - 1].float()
    logp = logits.log_softmax(-1)
    return float(torch.sigmoid(logp[ids["true"]].logsumexp(0) - logp[ids["false"]].logsumexp(0)))


def explain(model, processor, prompt, image):
    head = processor.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
    batch = encode(processor, head, image).to(model.device)
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.batch_decode([out[0, batch["input_ids"].shape[1]:]], skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--explain", action="store_true", help="also generate the model's rationale (slower)")
    parser.add_argument("--out", type=Path, help="write a CSV here as well as to stdout")
    args = parser.parse_args()

    spec, model, processor = load(args.model_dir)
    ids = boolean_ids(processor.tokenizer)
    print(f"{spec['name']}: {spec['question']}", file=sys.stderr)
    rows = []
    for path in args.images:
        with Image.open(path) as image:
            image = image.convert("RGB")
            p = p_true(model, processor, ids, spec["prompt"], spec["answer"], image)
            row = {"image": str(path), "p_true": round(p, 6), "predicted": p >= 0.5}
            if args.explain:
                row["rationale"] = explain(model, processor, spec["explain_prompt"], image)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
