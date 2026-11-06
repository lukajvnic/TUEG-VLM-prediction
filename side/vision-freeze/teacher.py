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
#   python side/vision-freeze/teacher.py loc-list                        # TUEV prompts with the annotated locations
#   python side/vision-freeze/teacher.py submit-loc MODEL SHARDS [--smoke]
#   python side/vision-freeze/teacher.py assemble-loc MODEL [--dry-run]
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
# windows added for the teacher after the first run (rerender.py tuar-teacher-extra: TUAR's 1,080, maintainer
# 2026-10-25), listed after every original window so the first run's shard lists are unchanged
EXTRA = {"TUAR": SIDE / "data" / "tuar-teacher-extra.csv"}
CODES = {"TUAR": ("bckg", "chew", "elec", "elpp", "eyem", "musc", "shiv")}
IMAGES = Path(os.environ.get("VF_CORE_ROOT", SIDE / "images-core"))
OUT = SIDE / "results" / "teacher"
MAX_NEW_TOKENS = 512  # the old teacher's budget
# the old prompt never said how long the window is, so teachers placed findings at "30-45 s" or "100-200 s" (10-window
# tests, 2026-10-25); this sentence, added after the intro for the teacher only (maintainer, 2026-10-25), states it
TIME_NOTE = "The window is 20 seconds long, with a vertical grid line every second. "
# location-aware teacher (maintainer, 2026-11-06: "lets get a little one of these running on a 2b model"): the
# localization judge found the teacher places its findings little better than the label implies, having never been
# told where the events are. loc-list gives TUEV's training windows that have an annotated event a prompt that also
# lists each event's channels and seconds; it reads the .rec files, so it runs where they are, and its CSV travels by
# scp like the rest of data/. assemble-loc writes datasets/TUEV/sft_train-loc.jsonl, which gm-2b-tuev-all-loc trains on
LOC = SIDE / "data" / "tuev-teacher-loc.csv"
LOC_NOTE = "Annotated locations, in seconds from the left edge of the window: "
RESOURCES = {"qwen3-vl:32b-instruct": ("a100:2", "128G", "06:00:00"),
             "qwen3-vl:30b-a3b-instruct": ("a100:2", "128G", "04:00:00"),
             "gemma3:12b": ("a100:1", "64G", "04:00:00")}


def short(model):
    return model.replace(":", "-")


def windows():
    # (dataset, core image path, prompt) for every window with an old teacher rationale and a core training image, then
    # the windows added for the teacher
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
    for ds, table in EXTRA.items():
        if not table.exists():  # data/ is not in git: without the list the added windows would silently drop out
            sys.exit(f"{table} is missing: copy it from the repo checkout that ran rerender.py tuar-teacher-extra")
        for r in read_csv(table):
            clause = prompts["rationale"]["label-clauses"][ds].format(labels=", ".join(r["labels"].split(";")))
            out.append((ds, r["path"], f"{prompts['intro']}{TIME_NOTE}{clause} {prompts['rationale']['grounded']}"))
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


def spans(seconds):
    # (3, 4, 5, 9, 12) -> "3-6 s, 9-10 s and 12-13 s"
    runs = []
    for k in seconds:
        if runs and runs[-1][1] == k:
            runs[-1][1] = k + 1
        else:
            runs.append([k, k + 1])
    return listed([f"{a}-{b} s" for a, b in runs])


def listed(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def locations(found):
    # "gped on all channels at 0-20 s. spsw on FP1-F7 and F7-T3 at 4-5 s." from (channel, start s, stop s, label) in
    # window time; seconds are TUEV's 1 s epochs, and channels with the same seconds are listed together, top to bottom
    import annotate

    seconds = defaultdict(lambda: defaultdict(set))
    for ch, a, b, label in found:
        seconds[label][ch] |= {k for k in range(20) if a < k + 1 and b > k}
    parts = []
    for label in [c for c in annotate.TUEV_CODES.values() if c in seconds]:
        groups = defaultdict(list)
        for ch in annotate.TUEV_CHANNELS:
            if seconds[label].get(ch):
                groups[tuple(sorted(seconds[label][ch]))].append(ch)
        for ks, chs in groups.items():
            where = "all channels" if len(chs) == len(annotate.TUEV_CHANNELS) else listed(chs)
            parts.append(f"{label} on {where} at {spans(ks)}")
    return LOC_NOTE + ". ".join(parts) + "."


def loc_list():
    # data/tuev-teacher-loc.csv: every TUEV training window (the 335 of data/core/TUEV-all-train.jsonl) with an annotated
    # event, and the teacher prompt with the event locations between the label clause and the instructions
    import annotate

    prompts = config()["prompts"]
    table = {r["path"]: r for r in read_csv(annotate.CORE_DATA / "TUEV-windows.csv")}
    labels = {r["path"]: r for r in read_csv(ROOT / "datasets" / "TUEV" / "labels.csv")}
    paths = sorted({json.loads(line)["images"][0] for line in open(annotate.CORE_DATA / "TUEV-all-train.jsonl")})
    rows = []
    for path in paths:
        start = int(table[path]["index"]) * 20
        found = [(ch, max(a, start) - start, min(b, start + 20) - start, label)
                 for ch, a, b, label in annotate.events("TUEV", table[path]["edf"]) if a < start + 20 and b > start]
        if not found:
            continue
        true = [c for c, v in labels[path].items() if c not in ("path", "ground_truth_rationale") and v.strip().lower() == "true"]
        clause = prompts["rationale"]["label-clauses"]["TUEV"].format(labels=", ".join(true))
        rows.append(["TUEV", path, f"{prompts['intro']}{TIME_NOTE}{clause} {locations(found)} Place each finding at its "
                                   f"annotated channels and times. {prompts['rationale']['grounded']}"])
    with LOC.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "path", "prompt"])
        writer.writerows(rows)
    print(f"{len(rows)} of {len(paths)} TUEV training windows have an annotated event -> {LOC}")


def loc_windows():
    if not LOC.exists():  # data/ is not in git
        sys.exit(f"{LOC} is missing: run teacher.py loc-list where the TUEV .rec files are and copy it here")
    return [(r["dataset"], r["path"], r["prompt"]) for r in read_csv(LOC)]


def smoke_windows(all_windows):
    # 10 windows: TUSZ seizure x3 and background x2, TUAR x2, TUEV epileptiform x2, TUAB x1 (seed 0)
    rng, picks = random.Random(0), []
    for ds, test, k in (("TUSZ", lambda p: "bckg" not in p, 3), ("TUSZ", lambda p: "bckg" in p, 2),
                        ("TUAR", lambda p: True, 2), ("TUEV", lambda p: any(e in p for e in ("spsw", "gped", "pled")), 2),
                        ("TUAB", lambda p: True, 1)):
        pool = [w for w in all_windows if w[0] == ds and test(w[2].split(" labels: ")[-1] if ds == "TUSZ" else w[2])]
        picks += rng.sample(pool, k)
    return picks


def run(model, shard, shards, smoke, refs=False, missing=None, extra=False, loc=False):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    sys.argv = ["train.py"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_main", ROOT / "train" / "train.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)

    todo = loc_windows() if loc else test_windows() if refs else windows()
    if extra:  # run-extra: only the windows added for the teacher (EXTRA), sharded over their own list
        added = {(ds, r["path"]) for ds, table in EXTRA.items() for r in read_csv(table)}
        todo = [w for w in todo if (w[0], w[1]) in added]
    todo = (todo[:3] if loc else smoke_windows(todo)) if smoke else todo[shard::shards]
    name = f"refs-shard-{shard:03d}" if refs else f"shard-extra-{shard:03d}" if extra else \
        f"loc-shard-{shard:03d}" if loc else f"shard-{shard:03d}"
    out = OUT / short(model) / (f"{'loc-' * loc}smoke.csv" if smoke else f"{name}.csv")
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


def submit_jobs(model, shards, smoke, refs=False, only=None, extra=False, loc=False):
    gpus, ram, limit = RESOURCES[model]
    limit = "02:00:00" if loc else limit  # 335 windows at most, over several shards
    if os.environ.get("CC_CLUSTER") in ("fir", "nibi", "trillium"):  # 80 GB H100s: a 30-32B teacher fits one
        # (Trillium has no slices, so a 12B teacher takes a whole H100 there too)
        gpus = "nvidia_h100_80gb_hbm3_3g.40gb:1" if gpus == "a100:1" and os.environ["CC_CLUSTER"] != "trillium" else "h100:1"
    for shard in [0] if smoke else ([only] if only is not None else range(shards)):  # --only N: one dead shard again
        kind = "refs-" if refs else "extra-" if extra else "loc-" if loc else ""
        name = f"eeg-vlm-teacher-{short(model)}-" + (f"{kind}smoke" if smoke else f"{kind}{shard:03d}")
        command = f"python {SIDE}/teacher.py {'run-' + kind[:-1] if kind else 'run'} {model} {shard} {shards}" + \
            (" --smoke" if smoke else "")
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
        if ds in EXTRA:  # rows for the windows added for the teacher, in the format of the original ones
            template = json.loads(backup.read_text().splitlines()[0])
            for r in read_csv(EXTRA[ds]):
                found = set(r["labels"].split(";"))
                output = {"text_rationale": replies[(ds, Path(r["path"]).name)][0],
                          **{f"has_{code}": code in found for code in CODES[ds]}}
                kept.append(json.dumps({**template, "output": json.dumps(output), "images": [r["path"]]}))
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


def assemble_loc(model, dry_run):
    # datasets/TUEV/sft_train-loc.jsonl: the assembled rationales (sft_train.jsonl, pinned by sft_train.md5) with every
    # loc-list window's rationale replaced by its location-prompt reply; refuses unless each has a reply to its exact
    # prompt, so no window keeps an old rationale by accident
    wanted, replies = loc_windows(), replies_of(model, "loc-")
    missing = [w for w in wanted if replies.get((w[0], Path(w[1]).name), ("", ""))[1] != w[2]
               or not replies[w[0], Path(w[1]).name][0]]
    print(f"loc TUEV: {len(wanted) - len(missing)} of {len(wanted)} windows have a reply to their location prompt")
    if missing and not dry_run:
        sys.exit("windows without a location reply: not assembling")
    if dry_run:
        return
    folder = ROOT / "datasets" / "TUEV"
    digest = hashlib.md5((folder / "sft_train.jsonl").read_bytes()).hexdigest()
    if digest != (folder / "sft_train.md5").read_text().strip():
        sys.exit(f"{folder / 'sft_train.jsonl'} is not the assembled new-teacher file (md5 {digest})")
    names = {Path(image).name for _, image, _ in wanted}
    kept, replaced = [], 0
    for line in (folder / "sft_train.jsonl").read_text().splitlines():
        row = json.loads(line)
        name = Path(row["images"][0]).name
        if name in names:
            output = json.loads(row["output"])
            output["text_rationale"] = replies["TUEV", name][0]
            row, replaced = {**row, "output": json.dumps(output)}, replaced + 1
        kept.append(json.dumps(row))
    if replaced != len(names):
        sys.exit(f"{replaced} of {len(names)} location windows are in {folder / 'sft_train.jsonl'}: not assembling")
    target = folder / "sft_train-loc.jsonl"
    target.write_text("\n".join(kept) + "\n")
    print(f"TUEV: {replaced} of {len(kept)} rationales replaced -> {target} (md5 {hashlib.md5(target.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    dry_run = "--dry-run" in args
    only = next((int(a.split("=")[1]) for a in args if a.startswith("--only=")), None)
    chosen = next((tuple(a.split("=")[1].split(",")) for a in args if a.startswith("--datasets=")), DATASETS)
    assert set(chosen) <= set(DATASETS), chosen
    args = [a for a in args if a != "--dry-run" and not a.startswith(("--only=", "--datasets="))]
    if args[0] == "loc-list":
        loc_list()
    elif args[0] == "assemble-loc":
        assemble_loc(args[1], dry_run)
    elif args[0] in ("submit", "submit-refs", "submit-extra", "submit-loc"):
        submit_jobs(args[1], int(args[2]), smoke, refs=args[0] == "submit-refs", only=only, extra=args[0] == "submit-extra",
                    loc=args[0] == "submit-loc")
    elif args[0] in ("run", "run-refs", "run-extra", "run-loc"):
        run(args[1], int(args[2]), int(args[3]), smoke, refs=args[0] == "run-refs", extra=args[0] == "run-extra",
            loc=args[0] == "run-loc")
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
