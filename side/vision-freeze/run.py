# side test: does training the vision side with the language model frozen (NeuroCanvas's recipe, arXiv 2602.04769)
# make qwen3-vl:2b-instruct see TUSZ seizures? Frozen vision features already separate seizure from background at
# AUROC ~0.70 (linear probe, 2026-10-06) while the LoRA fine-tunes reached 0.52-0.56. One balanced "any seizure"
# yes/no question (the multilabel pilot's lines), three arms on the same data:
#   freeze: vision tower + merger fully trained (fp32 master weights), language model frozen
#   lora:   the labels round's LoRA recipe on the same lines, the control
#   base:   no training, the same question asked of the untrained model
# Every arm scores P(true) on all TUSZ test windows; results/summary-<arm>.csv has window AUROC with a patient
# bootstrap CI, within seizure recordings, and on the probe's 1,500 test windows
# Round 2 (2026-10-07), the freeze recipe for 4 epochs: on the shipped images (freeze-4ep), on rerender.py's second
# rendering of the same windows (freeze-4ep-render: bipolar montage, one gain per recording, soft compression), and on
# that rendering with every seizure window of the training recordings (freeze-4ep-render-more, 4,664 / 4,664 lines)
# Round 3 (2026-10-10): freeze-4ep-render with gemma3:12b's rationale in the target ({"text_rationale", "present"},
# rationale first), plain and with the boolean token's loss weighted 10x (side/rationale-w10's collator and loss).
# Only the pilot's windows have rationales, so these compare with freeze-4ep-render. They keep the final checkpoint
# (the val metric would read the label off the teacher's label-stating rationale) and are scored by generating the
# rationale, then reading P(true) at the boolean, on the probe's 1,500 test windows (generation is ~10x slower)
#   python side/vision-freeze/run.py ARM [--dry-run]                 # submit one arm's MIG job
#   python side/vision-freeze/run.py smoke                           # <=2 h job: SMOKE_ARMS, 2 steps, 24 windows
#   python side/vision-freeze/run.py ARM --train|--score [--smoke]   # what the jobs run
import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "models"))
sys.modules.setdefault("datasets", None)  # same guard as train/train.py: the repo's datasets/ is not the HF library
from helpers.pipeline import append_row, base_spec, config, db, read_csv  # noqa: E402
from helpers.slurm import queued, script, submit  # noqa: E402

MODEL, DATASET, QUESTION = "qwen3-vl:2b-instruct", "TUSZ", "any_seizure"
# recipe: what trains (None: nothing); images: which PNGs; lines: the pilot's balanced any_seizure lines, or
# rerender.py's "more" set (train only; val is always the pilot's)
ARMS = {
    "freeze": {"recipe": "freeze", "epochs": 2, "images": "shipped", "lines": "pilot"},
    "lora": {"recipe": "lora", "epochs": 2, "images": "shipped", "lines": "pilot"},
    "base": {"recipe": None, "epochs": 0, "images": "shipped", "lines": "pilot"},
    "freeze-4ep": {"recipe": "freeze", "epochs": 4, "images": "shipped", "lines": "pilot"},
    "freeze-4ep-render": {"recipe": "freeze", "epochs": 4, "images": "render", "lines": "pilot"},
    "freeze-4ep-render-more": {"recipe": "freeze", "epochs": 4, "images": "render", "lines": "more"},
    "freeze-4ep-render-rationale": {"recipe": "freeze", "epochs": 4, "images": "render", "lines": "pilot",
                                    "target": "rationale", "weight": 1},
    "freeze-4ep-render-rationale-w10": {"recipe": "freeze", "epochs": 4, "images": "render", "lines": "pilot",
                                        "target": "rationale", "weight": 10},
}
# round 1's smoke (4833152) covered freeze, lora and base; round 2's (4961204) the rerender and the "more" lines
SMOKE_ARMS = ("freeze-4ep-render-rationale-w10",)
RATIONALES = ROOT / "datasets" / "TUSZ"  # sft_{train,val}.jsonl: gemma3:12b's label-conditioned rationales, same windows
# the rationales name the shipped images' referential channels ("EEG F8-REF", "Fp1-LE"); the rerender shows bipolar
# pairs ("F8-T4"), so channel names are cut to the electrode, which the bipolar row labels contain
RATIONALE_CHARS = 1200
CHANNEL = re.compile(r"\b(?:EEG\s+)?([A-Za-z0-9]+)-(?:REF|LE)\b")
IMAGE_ROOTS = {"shipped": ROOT / "datasets" / "TUSZ", "render": SIDE / "images"}
SOURCE = ROOT / "side" / "multilabel-pilot" / "data"  # the pilot's balanced lines; this test keeps only any_seizure
MORE = SIDE / "data" / "more-train.jsonl"  # rerender.py lists
SEIZURES = ("absz", "cpsz", "fnsz", "gnsz", "mysz", "spsz", "tcsz", "tnsz")
ANSWER = '{"present": true}'  # the scored row's answer: only the logits before its boolean are read
VISION_LR = 1e-5  # full fine-tune of a ViT: ViTST 2e-5, TimeMaster 2e-6; NeuroCanvas does not report one
EVALS = 5
SCORE_BATCH = 4  # test windows per forward
GPUS, RAM = "a100_3g.20gb:1", "40G"
# measured in round 1: freeze 13.1 s/step, scoring 0.60 s/window (~1.4 h for 8,460); 4 epochs of the pilot lines =
# 1,080 steps (~4 h), of the "more" lines = 4,664 steps (~17 h); 5 checkpoint saves of ~8 GB on top
TIME = {"freeze": "08:00:00", "lora": "08:00:00", "base": "03:00:00", "freeze-4ep": "10:00:00",
        "freeze-4ep-render": "10:00:00", "freeze-4ep-render-more": "1-06:00:00",
        "freeze-4ep-render-rationale": "14:00:00", "freeze-4ep-render-rationale-w10": "14:00:00", "smoke": "02:00:00"}
SMOKE_LINES, SMOKE_WINDOWS = {"train": 16, "val": 8}, 24
PROBE = Path(os.environ.get("SCRATCH", "/scratch/luka")) / "rehearsal" / "research" / "probe" / "manifest.csv"
BOOTSTRAP = 1000


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = load_path("train_main", ROOT / "train" / "train.py")


def base_dir(smoke):
    return SIDE / "smoke" if smoke else SIDE


def out_dir(arm, smoke=False):
    return base_dir(smoke) / "checkpoints" / arm


def lines(split, smoke, arm="freeze"):
    if ARMS[arm]["lines"] == "more" and split == "train":
        rows = main.read_jsonl(MORE)
    else:
        rows = [r for r in main.read_jsonl(SOURCE / f"perclass-{split}.jsonl") if r["question"] == QUESTION]
    if ARMS[arm].get("target") == "rationale":
        rows = with_rationale(rows, split)
    return rows[:SMOKE_LINES[split]] if smoke else rows


def rationale_prompt():
    p = config()["prompts"]
    tail = " Set present to true only when it does."
    base = prompt_text()
    assert base.endswith(tail), base
    return base[:-len(tail)] + " " + p["eval"]["binary-rationale"] + p["no-meta"] + tail


def with_rationale(rows, split):
    # the same windows and answers, with each window's teacher rationale first
    source = {r["images"][0]: json.loads(r["output"])["text_rationale"]
              for r in main.read_jsonl(RATIONALES / f"sft_{split}.jsonl")}
    text = rationale_prompt()
    return [{**r, "instruction": text,
             "output": json.dumps({"text_rationale": CHANNEL.sub(r"\1", source[r["images"][0]]),
                                   "present": json.loads(r["output"])["present"]})} for r in rows]


def image_root(arm):
    return IMAGE_ROOTS[ARMS[arm]["images"]]


def missing_images(arm):
    # every training, val and test image this arm reads, checked before the job is submitted
    paths = {r["images"][0] for split in ("train", "val") for r in lines(split, False, arm)} | set(test_windows())
    return sorted(p for p in paths if not (image_root(arm) / p).exists())


def prompt_text():
    texts = {r["instruction"] for r in main.read_jsonl(SOURCE / "perclass-train.jsonl") if r["question"] == QUESTION}
    assert len(texts) == 1, texts
    return texts.pop()


def arm_config(cfg, arm, smoke):
    settings = main.load_config(cfg, MODEL, DATASET)
    settings.update({"experiment": f"vision-freeze-{arm}", "target": "labels", "output-dir": str(out_dir(arm, smoke))})
    return settings


def vision_module(model):
    names = [n for n, _ in model.named_modules() if n.split(".")[-1] == "visual"]
    assert len(names) == 1, names
    return names[0], model.get_submodule(names[0])


def init_frozen_language(settings):
    # every weight frozen, then the vision tower (patch embedding, blocks, merger, deepstack mergers) unfrozen in fp32:
    # a 1e-5 step is below bf16's resolution for most weights, so bf16 weights would barely move
    import torch
    from transformers import AutoModelForImageTextToText

    base = settings["base"]
    source = main.resolve_model(base["repo"])
    processor = main.runner.load_processor(source, base, False)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.config.keys_to_ignore_at_inference = main.EVAL_IGNORE
    model.requires_grad_(False)

    name, vision = vision_module(model)
    vision.float()
    vision.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable / 1e6:.0f}M of {total / 1e6:.0f}M parameters ({name}, fp32); language model frozen",
          flush=True)
    return model, processor


def start_wandb(settings, data, resume, smoke):
    # same project as the main runs, group vision-freeze; offline under the side dir
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    out = Path(settings["output-dir"])
    id_file = out / "wandb-id"
    run_id = id_file.read_text().strip() if resume and id_file.exists() else None
    logged = {k: settings[k] for k in ("model", "dataset", "experiment", "target", "base", "lora", "training")}
    logged.update({"train_rows": len(data["train"]), "val_rows": len(data["validation"]), "vision_lr": VISION_LR})
    logs = base_dir(smoke) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    run = wandb.init(entity=settings["wandb"]["entity"], project=settings["wandb"]["project"],
                     name=main.wandb_name(settings), group="vision-freeze", tags=[DATASET, settings["experiment"]],
                     dir=logs, id=run_id, resume="allow" if run_id else None, config=logged)

    out.mkdir(parents=True, exist_ok=True)
    id_file.write_text(run.id + "\n")
    return run


def train(settings, arm, smoke):
    import torch

    out = Path(settings["output-dir"])
    if (out / "manifest.json").exists():  # a resubmitted job whose scoring was cut off: go straight to scoring
        print(f"{out} is already trained (manifest.json exists)", flush=True)
        return
    resume = main.resume_point(out)
    data = {"train": lines("train", smoke, arm), "validation": lines("val", smoke, arm)}
    training, epochs = settings["training"], ARMS[arm]["epochs"]
    steps = epochs * math.ceil(len(data["train"]) / (training["batch-size"] * training["gradient-accumulation"]))
    every = 1 if smoke else math.ceil(steps / EVALS)  # save-steps must equal eval-steps for the best-checkpoint reload
    settings["training"] = {**training, "epochs": epochs, "eval-steps": every, "save-steps": every}

    print(f"{MODEL} on {DATASET} '{QUESTION}', arm {arm} {ARMS[arm]}: {len(data['train'])} train / "
          f"{len(data['validation'])} val lines, {steps} steps, eval every {every}, images {image_root(arm)} -> {out}",
          flush=True)
    if resume:
        print(f"resuming from {resume}", flush=True)

    freeze = ARMS[arm]["recipe"] == "freeze"
    model, processor = init_frozen_language(settings) if freeze else main.init_model(settings)
    run = start_wandb(settings, data, resume, smoke)
    trainer = main.init_trainer(settings, model, processor, data, image_root(arm))
    if freeze:
        trainer.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=VISION_LR,
                                              weight_decay=training["weight-decay"])
        trainer.args.learning_rate = VISION_LR
    if ARMS[arm].get("target") == "rationale":
        # the teacher-forced val metric reads the boolean after the gold rationale, which names the label
        trainer.args.load_best_model_at_end = False
    if ARMS[arm].get("weight", 1) > 1:
        trainer = weighted(trainer, processor, data, ARMS[arm]["weight"])
    if smoke:
        trainer.args.max_steps = 2

    trainer.train(resume_from_checkpoint=resume)
    print(f"peak GPU memory {torch.cuda.max_memory_reserved() / 1e9:.1f} GB", flush=True)
    trainer.save_model()
    processor.save_pretrained(out)

    main.write_manifest(settings, data, run.id)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update({"arm": arm, **ARMS[arm], "image_root": str(image_root(arm).relative_to(ROOT)),
                     "question": QUESTION, "smoke": smoke, "vision_lr": VISION_LR if freeze else None})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run.finish()


def weighted(base, processor, data, weight):
    # side/rationale-w10's recipe: weight `weight` on the answer's boolean token (the last true/false target token,
    # since the rationale comes first), 1 on every other target token, loss = sum(w * CE) / target tokens
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    words = main.boolean_token_ids(processor.tokenizer)
    ids = torch.tensor(sorted(words["true"] | words["false"]))
    collate_base = base.data_collator

    def collate(examples):
        batch = collate_base(examples)
        labels = batch["labels"]
        weights = (labels != -100).float()
        for row in range(labels.shape[0]):
            positions = torch.isin(labels[row], ids).nonzero().flatten()
            if not len(positions):
                raise ValueError("no true/false token in a target")
            weights[row, positions[-1]] = weight
        batch["label_weights"] = weights
        return batch

    class WeightedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = True  # compute_loss divides by num_items_in_batch itself

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

    return WeightedTrainer(model=base.model, args=base.args, train_dataset=data["train"], eval_dataset=data["validation"],
                           data_collator=collate, processing_class=processor, optimizers=(base.optimizer, None),
                           compute_metrics=base.compute_metrics,
                           preprocess_logits_for_metrics=base.preprocess_logits_for_metrics)


def load_for_scoring(arm, smoke):
    import torch
    from transformers import AutoModelForImageTextToText

    base = base_spec(config(), MODEL)
    if ARMS[arm]["recipe"] == "lora":
        model, processor, _ = main.runner.load({"base": MODEL, "checkpoint": str(out_dir(arm, smoke).relative_to(ROOT))})
        return model, processor, base

    source = main.resolve_model(base["repo"]) if ARMS[arm]["recipe"] is None else str(out_dir(arm, smoke))
    if ARMS[arm]["recipe"] and not (out_dir(arm, smoke) / "manifest.json").exists():
        sys.exit(f"{out_dir(arm, smoke)} has no manifest.json: training has not finished")
    print(f"loading {source}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, main.runner.load_processor(source, base, False), base


def p_true(model, collator, ids, text, paths):
    # the collator builds exactly the training input, answer included, so the boolean token's position is known
    # and only the logits before it are kept
    import torch

    every = torch.cat([ids["true"], ids["false"]]).cpu()
    batch = collator([{"instruction": text, "input": "", "output": ANSWER, "images": [p]} for p in paths])
    targets = batch.pop("labels")
    positions = [int(torch.isin(row, every).nonzero()[0]) for row in targets]
    keep = targets.shape[1] - min(positions) + 1

    with torch.inference_mode():
        logits = model(**batch.to(model.device), logits_to_keep=keep).logits.float()

    offset = targets.shape[1] - keep
    probabilities = []
    for row, position in enumerate(positions):
        logp = logits[row, position - 1 - offset].log_softmax(-1)
        probabilities.append(float(torch.sigmoid(logp[ids["true"]].logsumexp(0) - logp[ids["false"]].logsumexp(0))))
    return probabilities


def rationale_structure():
    # maxLength makes the enforcer close the rationale and reach the boolean: the round-3 smoke model (2 steps)
    # rambled to the 512-token cap at 24 s/window, which would not fit 1,500 windows in the walltime. The teacher's
    # rationales are at most 1,112 characters (median 628)
    from pydantic import Field, create_model

    return create_model("SeizureRationale",
                        text_rationale=(str, Field(max_length=RATIONALE_CHARS, description="Text rationale for the prediction.")),
                        present=(bool, Field(description="Whether a seizure of any type is present.")))


def generate_p_true(model, processor, runner, prefix_fn, ids, text, image_path):
    # the deployed path: constrained decoding writes the rationale, then the boolean; P(true) from that step's raw
    # logits. None when the reply never reached a boolean (cut at max_new_tokens)
    import torch
    from PIL import Image

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = runner.render(processor, messages, True)
    with Image.open(image_path) as image:
        inputs = processor(text=[chat], images=[image.convert("RGB")], return_tensors="pt",
                           **runner.encode_kwargs(processor, chat)).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=runner.MAX_NEW_TOKENS, prefix_allowed_tokens_fn=prefix_fn.fresh(),
                             use_cache=True, do_sample=False, output_logits=True, return_dict_in_generate=True)
    generated = out.sequences[0, inputs["input_ids"].shape[1]:].cpu()
    every = torch.cat([ids["true"], ids["false"]]).cpu()
    steps = torch.isin(generated, every).nonzero().flatten()
    reply = processor.batch_decode([generated], skip_special_tokens=True)[0].strip()
    if not len(steps):
        return None, reply
    logp = out.logits[int(steps[-1])][0].float().log_softmax(-1)
    return float(torch.sigmoid(logp[ids["true"]].logsumexp(0) - logp[ids["false"]].logsumexp(0))), reply


def probe_windows():
    return sorted(r["rel"] for r in read_csv(PROBE) if r["ds"] == DATASET and r["split"] == "test")


def test_windows():
    rows = db().execute("SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'full'", (DATASET,))
    return sorted(p.split("/", 1)[1] for (p,) in rows)


def truth():
    return {r["path"]: any(r[c] == "true" for c in SEIZURES)
            for r in read_csv(ROOT / "datasets" / DATASET / "labels.csv")}


def score(arm, smoke):
    import torch

    results = base_dir(smoke) / "results"
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"scores-{arm}.csv"
    done = {r["path"] for r in read_csv(out)}
    rationale = ARMS[arm].get("target") == "rationale"
    todo = [p for p in (probe_windows() if rationale else test_windows()) if p not in done]
    if smoke:
        todo = todo[::max(len(todo) // SMOKE_WINDOWS, 1)][:SMOKE_WINDOWS]
    print(f"arm {arm}: {len(todo)} windows to score, {len(done)} already done -> {out}", flush=True)

    if todo:
        model, processor, base = load_for_scoring(arm, smoke)
        collator = main.DataCollator(processor, image_root(arm), base)
        words = main.boolean_token_ids(processor.tokenizer)
        ids = {w: torch.tensor(sorted(words[w]), device=model.device) for w in ("true", "false")}
        text, labels = prompt_text(), truth()
        started = time.time()

        if rationale:
            text = rationale_prompt()
            prefix_fn = main.runner.enforcer(processor, rationale_structure())
            for n, path in enumerate(todo, 1):
                p, reply = generate_p_true(model, processor, main.runner, prefix_fn, ids, text, image_root(arm) / path)
                append_row(out, ["path", "p_true", "seizure", "reply"],
                           [path, "" if p is None else f"{p:.6f}", str(labels[path]).lower(), reply])
                if n % 50 == 0 or n == len(todo) or smoke and n <= 2:
                    print(f"{n}/{len(todo)} ({(time.time() - started) / n:.2f} s/window)", flush=True)
            todo = []

        for start in range(0, len(todo), SCORE_BATCH):
            chunk = todo[start:start + SCORE_BATCH]
            for path, p in zip(chunk, p_true(model, collator, ids, text, chunk)):
                append_row(out, ["path", "p_true", "seizure"], [path, f"{p:.6f}", str(labels[path]).lower()])
            n = start + len(chunk)
            if n % 200 < SCORE_BATCH or n == len(todo) or smoke and start == 0:
                print(f"{n}/{len(todo)} ({(time.time() - started) / n:.2f} s/window)", flush=True)

    if not smoke:
        summarise(arm, out)


def auroc(scores, positive):
    # Mann-Whitney, ties at their average rank
    scores, positive = np.asarray(scores, float), np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return float("nan")
    order = scores.argsort()
    _, inverse, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = (np.bincount(inverse, np.arange(1, len(scores) + 1)) / counts)[inverse]
    n = positive.sum()
    return float((ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(scores) - n)))


def patient_ci(scores, positive, patients):
    # 95% interval over 1,000 resamples of patients, windows of a patient kept together
    rng = np.random.default_rng(0)
    groups = {}
    for i, patient in enumerate(patients):
        groups.setdefault(patient, []).append(i)
    keys = list(groups)
    values = []
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        values.append(auroc(scores[idx], positive[idx]))
    return np.nanpercentile(values, [2.5, 97.5])


def summarise(arm, path):
    rows = list({r["path"]: r for r in read_csv(path)}.values())
    unscored = [r for r in rows if r["p_true"] == ""]
    if unscored:
        print(f"{len(unscored)} windows whose reply never reached the boolean are left out", flush=True)
    rows = [r for r in rows if r["p_true"] != ""]
    p = np.array([float(r["p_true"]) for r in rows])
    y = np.array([r["seizure"] == "true" for r in rows])
    names = [Path(r["path"]).stem for r in rows]
    patients = [n.split("_")[0] for n in names]
    recordings = ["_".join(n.split("_")[:2]) for n in names]
    with_seizure = {rec for rec, positive in zip(recordings, y) if positive}
    within = np.array([rec in with_seizure for rec in recordings])
    probe_paths = {r["rel"] for r in read_csv(PROBE) if r["ds"] == DATASET and r["split"] == "test"}
    in_probe = np.array([r["path"] in probe_paths for r in rows])
    low, high = patient_ci(p, y, patients)
    predicted = p >= 0.5

    summary = {
        "arm": arm, "windows": len(rows), "seizure_windows": int(y.sum()), "patients": len(set(patients)),
        "auroc": round(auroc(p, y), 4), "auroc_ci_low": round(low, 4), "auroc_ci_high": round(high, 4),
        "auroc_within_seizure_recordings": round(auroc(p[within], y[within]), 4),
        "auroc_probe_windows": round(auroc(p[in_probe], y[in_probe]), 4) if in_probe.any() else "",
        "probe_windows": int(in_probe.sum()),
        "balanced_accuracy_at_0.5": round(float((predicted[y].mean() + (~predicted[~y]).mean()) / 2), 4),
        "predicted_seizure_fraction": round(float(predicted.mean()), 4),
    }
    with (path.parent / f"summary-{arm}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary), flush=True)


def submit_job(cfg, arm, dry_run):
    name = f"eeg-vlm-vision-freeze-{arm}"
    if name in queued():
        sys.exit(f"{name} is already queued or running")
    if not (SOURCE / "perclass-train.jsonl").exists():
        sys.exit("no training lines: python side/multilabel-pilot/build.py")
    for a in SMOKE_ARMS if arm == "smoke" else (arm,):
        missing = missing_images(a)
        if missing:
            sys.exit(f"{a}: {len(missing)} images missing under {image_root(a)}, e.g. {missing[:3]}")

    me = f"python {SIDE}/run.py"
    modes = lambda a: ("score",) if ARMS[a]["recipe"] is None else ("train", "score")
    steps = {a: [f"{me} {a} --{m}" for m in modes(a)] for a in ARMS}
    steps["smoke"] = [f"{me} {a} --{m} --smoke" for a in SMOKE_ARMS for m in modes(a)]
    logs = base_dir(arm == "smoke") / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    text = script(name, TIME[arm], RAM, cfg["train"]["cpus"], GPUS, " && ".join(steps[arm]))
    text = text.replace(str(ROOT / "logs"), str(logs))
    print(f"{'dry run' if dry_run else submit(text)} - {name}, {TIME[arm]}/{RAM}/gpu:{GPUS}, log in {logs}")
    if dry_run:
        print(text)


def main_cli():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=[*ARMS, "smoke"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="train this arm in this process")
    mode.add_argument("--score", action="store_true", help="score this arm on the TUSZ test windows")
    mode.add_argument("--dry-run", action="store_true", help="print the job, submit nothing")
    parser.add_argument("--smoke", action="store_true", help="with --train/--score: 2 steps, 24 windows, under smoke/")
    args = parser.parse_args()

    if args.train or args.score:
        if args.arm == "smoke" or args.train and ARMS[args.arm]["recipe"] is None:
            parser.error(f"--{'train' if args.train else 'score'} does not apply to {args.arm}")
        os.environ.setdefault("WANDB_MODE", "offline")
        if args.train:
            train(arm_config(cfg, args.arm, args.smoke), args.arm, args.smoke)
        else:
            score(args.arm, args.smoke)
        return

    submit_job(cfg, args.arm, args.dry_run)


if __name__ == "__main__":
    main_cli()
