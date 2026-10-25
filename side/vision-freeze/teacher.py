# teacher rationales on the core images (maintainer, 2026-10-25): the old ones were written by gemma3:12b on the
# shipped images (referential montage, per-channel scaling), so the windows that have one (datasets/<DS>/
# sft_train.jsonl) get a new one from a stronger reader of the full image, with the same label-conditioned prompt
# (config.yml prompts.rationale). Shards run as Slurm jobs; assemble writes datasets/<DS>/sft_train.jsonl in place
# (the old file kept beside it), which held arms read when they start
#   python side/vision-freeze/teacher.py submit MODEL SHARDS [--smoke]   # --smoke: 10 windows, one job
#   python side/vision-freeze/teacher.py run MODEL SHARD SHARDS [--smoke]
#   python side/vision-freeze/teacher.py assemble MODEL
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
sys.path.insert(0, str(ROOT))
from helpers.pipeline import append_row, base_spec, config, read_csv  # noqa: E402
from helpers.slurm import script, submit  # noqa: E402

DATASETS = ("TUSZ", "TUAR", "TUEV", "TUAB")
IMAGES = Path(os.environ.get("VF_CORE_ROOT", SIDE / "images-core"))
OUT = SIDE / "results" / "teacher"
MAX_NEW_TOKENS = 512  # the old teacher's budget
RESOURCES = {"qwen3-vl:32b-instruct": ("a100:2", "128G", "06:00:00"),
             "qwen3-vl:30b-a3b-instruct": ("a100:2", "128G", "04:00:00"),
             "gemma3:12b": ("a100:1", "64G", "04:00:00")}


def short(model):
    return model.replace(":", "-")


def windows():
    # (dataset, core image path, prompt) for every window with an old teacher rationale and a core training image
    prompts = config()["prompts"]
    out = []
    for ds in DATASETS:
        labels = {r["path"]: r for r in read_csv(ROOT / "datasets" / ds / "labels.csv")}
        for line in open(ROOT / "datasets" / ds / "sft_train.jsonl"):
            name = Path(json.loads(line)["images"][0]).name
            image = f"train/{name}"
            if not (IMAGES / ds / image).exists():
                continue
            row = labels[f"train/{name}"]
            true = [c for c, v in row.items() if c not in ("path", "ground_truth_rationale") and v.strip().lower() == "true"]
            clause = prompts["rationale"]["label-clauses"][ds].format(labels=", ".join(true))
            out.append((ds, image, f"{prompts['intro']}{clause} {prompts['rationale']['grounded']}"))
    return out


def smoke_windows(all_windows):
    # 10 windows: TUSZ seizure x3 and background x2, TUAR x2, TUEV epileptiform x2, TUAB x1 (seed 0)
    rng, picks = random.Random(0), []
    for ds, test, k in (("TUSZ", lambda p: "bckg" not in p, 3), ("TUSZ", lambda p: "bckg" in p, 2),
                        ("TUAR", lambda p: True, 2), ("TUEV", lambda p: any(e in p for e in ("spsw", "gped", "pled")), 2),
                        ("TUAB", lambda p: True, 1)):
        pool = [w for w in all_windows if w[0] == ds and test(w[2].split(" labels: ")[-1] if ds == "TUSZ" else w[2])]
        picks += rng.sample(pool, k)
    return picks


def run(model, shard, shards, smoke):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    sys.argv = ["train.py"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_main", ROOT / "train" / "train.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)

    todo = windows()
    todo = smoke_windows(todo) if smoke else todo[shard::shards]
    out = OUT / short(model) / (f"smoke.csv" if smoke else f"shard-{shard:03d}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["dataset"], r["path"]) for r in read_csv(out)}
    todo = [w for w in todo if (w[0], w[1]) not in done]
    base = base_spec(config(), model)
    source = main.resolve_model(base["repo"])
    print(f"{model} ({source}): {len(todo)} windows -> {out}", flush=True)
    model_ = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="auto")
    model_.eval()
    processor = main.runner.load_processor(source, base, False)
    started = time.time()
    for n, (ds, image, prompt) in enumerate(todo, 1):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = main.runner.render(processor, messages, True)
        with Image.open(IMAGES / ds / image) as picture:
            inputs = processor(text=[chat], images=[picture.convert("RGB")], return_tensors="pt",
                               **main.runner.encode_kwargs(processor, chat)).to(model_.device)
        with torch.inference_mode():
            generated = model_.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        reply = processor.batch_decode([generated[0, inputs["input_ids"].shape[1]:]], skip_special_tokens=True)[0].strip()
        append_row(out, ["dataset", "path", "prompt", "reply"], [ds, image, prompt, reply])
        if n % 10 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} ({(time.time() - started) / n:.1f} s/window)", flush=True)


def submit_jobs(model, shards, smoke):
    gpus, ram, limit = RESOURCES[model]
    if os.environ.get("CC_CLUSTER") in ("fir", "nibi"):  # 80 GB H100s: a 30-32B teacher fits one, a 12B one a 40 GB slice
        gpus = "nvidia_h100_80gb_hbm3_3g.40gb:1" if gpus == "a100:1" else "h100:1"
    for shard in [0] if smoke else range(shards):
        name = f"eeg-vlm-teacher-{short(model)}-" + ("smoke" if smoke else f"{shard:03d}")
        command = f"python {SIDE}/teacher.py run {model} {shard} {shards}" + (" --smoke" if smoke else "")
        text = script(name, "01:30:00" if smoke else limit, ram, config()["train"]["cpus"], gpus, command)
        text = text.replace(str(ROOT / "logs"), str(SIDE / "logs"))
        print(submit(text), "-", name, flush=True)


def assemble(model):
    # every shard's reply replaces that window's old rationale in datasets/<DS>/sft_train.jsonl; windows the new teacher
    # did not reach keep no rationale (they train on the answer alone), and the old file stays as sft_train-old.jsonl
    replies = {}
    for path in sorted((OUT / short(model)).glob("shard-*.csv")):
        for r in read_csv(path):
            if r["reply"].strip():
                replies[(r["dataset"], Path(r["path"]).name)] = r["reply"].strip()
    for ds in DATASETS:
        source = ROOT / "datasets" / ds / "sft_train.jsonl"
        backup = ROOT / "datasets" / ds / "sft_train-old.jsonl"
        if not backup.exists():
            backup.write_bytes(source.read_bytes())
        kept = []
        for line in backup.read_text().splitlines():
            row = json.loads(line)
            reply = replies.get((ds, Path(row["images"][0]).name))
            if reply:
                output = json.loads(row["output"])
                output["text_rationale"] = reply
                kept.append(json.dumps({**row, "output": json.dumps(output)}))
        source.write_text("\n".join(kept) + "\n")
        print(f"{ds}: {len(kept)} windows with a new rationale -> {source}")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    if args[0] == "submit":
        submit_jobs(args[1], int(args[2]), smoke)
    elif args[0] == "run":
        run(args[1], int(args[2]), int(args[3]), smoke)
    else:
        assemble(args[1])
