# teacher rationales on the core images (maintainer, 2026-10-25): the old ones were written by gemma3:12b on the
# shipped images (referential montage, per-channel scaling), so the windows that have one (datasets/<DS>/
# sft_train.jsonl) get a new one from a stronger reader of the full image, with the same label-conditioned prompt
# (config.yml prompts.rationale). Shards run as Slurm jobs; assemble writes datasets/<DS>/sft_train.jsonl in place
# (the old file kept beside it), which held arms read when they start
#   python side/vision-freeze/teacher.py submit MODEL SHARDS [--smoke]   # --smoke: 10 windows, one job
#   python side/vision-freeze/teacher.py run MODEL SHARD SHARDS [--smoke]
#   python side/vision-freeze/teacher.py assemble MODEL [--dry-run]     # refuses unless every window has a new reply
#   python side/vision-freeze/teacher.py submit-refs MODEL SHARDS        # test-side references (the graded windows)
#   python side/vision-freeze/teacher.py assemble-refs MODEL [--dry-run]
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
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
# the old prompt never said how long the window is, so teachers placed findings at "30-45 s" or "100-200 s" (10-window
# tests, 2026-10-25); this sentence, added after the intro for the teacher only (maintainer, 2026-10-25), states it
TIME_NOTE = "The window is 20 seconds long, with a vertical grid line every second. "
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
        old = ROOT / "datasets" / ds / "sft_train-old.jsonl"  # the list stays the same after assemble
        for line in open(old if old.exists() else ROOT / "datasets" / ds / "sft_train.jsonl"):
            name = Path(json.loads(line)["images"][0]).name
            image = f"train/{name}"
            if not (IMAGES / ds / image).exists():
                continue
            row = labels[f"train/{name}"]
            true = [c for c, v in row.items() if c not in ("path", "ground_truth_rationale") and v.strip().lower() == "true"]
            clause = prompts["rationale"]["label-clauses"][ds].format(labels=", ".join(true))
            out.append((ds, image, f"{prompts['intro']}{TIME_NOTE}{clause} {prompts['rationale']['grounded']}"))
    return out


def test_windows():
    # test-side references (maintainer, 2026-10-25: "lets go with A"): only the windows the rationale evaluation grades,
    # run.py's rationale_eval_windows of each dataset model (seeded and label-only, so the same for every base model),
    # prompted with the window's TUH test labels like the training windows
    import importlib.util

    sys.argv = ["run.py"]
    spec = importlib.util.spec_from_file_location("vf_run", SIDE / "run.py")
    run_py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_py)
    prompts = config()["prompts"]
    out = []
    for ds in DATASETS:
        labels = {r["path"]: r for r in read_csv(ROOT / "datasets" / ds / "labels.csv")}
        for path in run_py.rationale_eval_windows(f"gm-8b-{ds.lower()}-all"):
            row = labels[path]
            true = [c for c, v in row.items() if c not in ("path", "ground_truth_rationale") and v.strip().lower() == "true"]
            clause = prompts["rationale"]["label-clauses"][ds].format(labels=", ".join(true))
            out.append((ds, path, f"{prompts['intro']}{TIME_NOTE}{clause} {prompts['rationale']['grounded']}"))
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


def run(model, shard, shards, smoke, refs=False, missing=None):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    sys.argv = ["train.py"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_main", ROOT / "train" / "train.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)

    todo = test_windows() if refs else windows()
    todo = smoke_windows(todo) if smoke else todo[shard::shards]
    out = OUT / short(model) / ("smoke.csv" if smoke else f"{'refs-' if refs else ''}shard-{shard:03d}.csv")
    if missing:  # run-missing: the training windows of these datasets that no shard has a reply for (a window whose
        # image was rendered after the shards started is not in their lists)
        replies = replies_of(model, "")
        todo = [w for w in windows() if w[0] in missing and (w[0], Path(w[1]).name) not in replies]
        out = OUT / short(model) / "shard-missing.csv"
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


def submit_jobs(model, shards, smoke, refs=False, only=None):
    gpus, ram, limit = RESOURCES[model]
    if os.environ.get("CC_CLUSTER") in ("fir", "nibi", "trillium"):  # 80 GB H100s: a 30-32B teacher fits one
        # (Trillium has no slices, so a 12B teacher takes a whole H100 there too)
        gpus = "nvidia_h100_80gb_hbm3_3g.40gb:1" if gpus == "a100:1" and os.environ["CC_CLUSTER"] != "trillium" else "h100:1"
    for shard in [0] if smoke else ([only] if only is not None else range(shards)):  # --only N: one dead shard again
        name = f"eeg-vlm-teacher-{short(model)}-" + ("smoke" if smoke else f"{'refs-' if refs else ''}{shard:03d}")
        command = f"python {SIDE}/teacher.py {'run-refs' if refs else 'run'} {model} {shard} {shards}" + (" --smoke" if smoke else "")
        text = script(name, "01:30:00" if smoke else limit, ram, config()["train"]["cpus"], gpus, command)
        text = text.replace(str(ROOT / "logs"), str(SIDE / "logs"))
        if os.environ.get("CC_CLUSTER") == "trillium":  # whole GPUs by --gpus-per-node, memory comes with the GPU
            text = text.replace("#SBATCH --gres=gpu:", "#SBATCH --gpus-per-node=")
            text = "\n".join(line for line in text.split("\n") if not line.startswith("#SBATCH --mem="))
        print(submit(text), "-", name, flush=True)


def replies_of(model, prefix):
    out = {}
    for path in sorted((OUT / short(model)).glob(f"{prefix}shard-*.csv")):
        for r in read_csv(path):
            out[(r["dataset"], Path(r["path"]).name)] = (r["reply"].strip(), r["prompt"])
    return out


def gate(wanted, replies, what, datasets=DATASETS):
    # every window needs a non-empty reply written with the current prompt (TIME_NOTE), or nothing is written
    missing = defaultdict(int)
    wanted = [w for w in wanted if w[0] in datasets]
    for ds, image, _ in wanted:
        reply, prompt = replies.get((ds, Path(image).name), ("", ""))
        if not reply or TIME_NOTE not in prompt:
            missing[ds] += 1
    for ds in datasets:
        total = sum(1 for d, _, _ in wanted if d == ds)
        print(f"{what} {ds}: {total - missing[ds]} of {total} windows have a new reply")
    return sum(missing.values())


def assemble(model, dry_run, datasets=DATASETS):
    # the new rationales replace the old ones in datasets/<DS>/sft_train.jsonl (the old file is kept as
    # sft_train-old.jsonl, which the first 8B runs' "rationales": "old" pin reads); refuses unless every window of the
    # chosen datasets has a new-prompt reply, so no window silently loses its rationale; --dry-run only reports.
    # --datasets=TUSZ,TUAR lets the finished datasets go ahead of a slow shard
    wanted, replies = windows(), replies_of(model, "")
    if gate(wanted, replies, "train", datasets) and not dry_run:
        sys.exit("windows without a new reply: not assembling")
    if dry_run:
        return
    for ds in datasets:
        source = ROOT / "datasets" / ds / "sft_train.jsonl"
        backup = ROOT / "datasets" / ds / "sft_train-old.jsonl"
        if not backup.exists():
            backup.write_bytes(source.read_bytes())
        kept = []
        for line in backup.read_text().splitlines():
            row = json.loads(line)
            reply = replies.get((ds, Path(row["images"][0]).name), ("", ""))[0]
            if reply:
                output = json.loads(row["output"])
                output["text_rationale"] = reply
                kept.append(json.dumps({**row, "output": json.dumps(output)}))
        source.write_text("\n".join(kept) + "\n")
        digest = hashlib.md5(source.read_bytes()).hexdigest()
        (ROOT / "datasets" / ds / "sft_train.md5").write_text(digest + "\n")  # run.py's gate for dataset arms
        print(f"{ds}: {len(kept)} windows with a new rationale -> {source} (md5 {digest})")


def assemble_refs(model, dry_run):
    # the new test-side references in datasets/<DS>/references-test.jsonl, which run.py's rationale evaluation reads
    # instead of labels.csv's gemma3:12b text; same gate
    wanted, replies = test_windows(), replies_of(model, "refs-")
    if gate(wanted, replies, "test") and not dry_run:
        sys.exit("test windows without a new reply: not assembling")
    if dry_run:
        return
    for ds in DATASETS:
        target = ROOT / "datasets" / ds / "references-test.jsonl"
        rows = [{"path": image, "text_rationale": replies[(d, Path(image).name)][0]} for d, image, _ in wanted if d == ds]
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{ds}: {len(rows)} test references -> {target} (md5 {hashlib.md5(target.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    dry_run = "--dry-run" in args
    only = next((int(a.split("=")[1]) for a in args if a.startswith("--only=")), None)
    chosen = next((tuple(a.split("=")[1].split(",")) for a in args if a.startswith("--datasets=")), DATASETS)
    assert set(chosen) <= set(DATASETS), chosen
    args = [a for a in args if a != "--dry-run" and not a.startswith(("--only=", "--datasets="))]
    if args[0] in ("submit", "submit-refs"):
        submit_jobs(args[1], int(args[2]), smoke, refs=args[0] == "submit-refs", only=only)
    elif args[0] in ("run", "run-refs"):
        run(args[1], int(args[2]), int(args[3]), smoke, refs=args[0] == "run-refs")
    elif args[0] == "run-missing":
        run(args[1], 0, 1, False, missing=chosen)
    elif args[0] == "submit-missing":
        gpus, ram, _ = RESOURCES[args[1]]
        command = f"python {SIDE}/teacher.py run-missing {args[1]} --datasets={','.join(chosen)}"
        text = script(f"eeg-vlm-teacher-{short(args[1])}-missing", "01:00:00", ram, config()["train"]["cpus"], "h100:1", command)
        text = text.replace(str(ROOT / "logs"), str(SIDE / "logs"))
        if os.environ.get("CC_CLUSTER") == "trillium":
            text = text.replace("#SBATCH --gres=gpu:", "#SBATCH --gpus-per-node=")
            text = "\n".join(line for line in text.split("\n") if not line.startswith("#SBATCH --mem="))
        print(submit(text), "-", f"eeg-vlm-teacher-{short(args[1])}-missing", flush=True)
    elif args[0] == "assemble-refs":
        assemble_refs(args[1], dry_run)
    else:
        assemble(args[1], dry_run, chosen)
