# dress rehearsal of a fine-tune before it goes to the queue. It runs the real code end to end on a sample of the
# real data: train/train.py (or side/rationale-w10/train.py) through train(), a crash and resume, the best-checkpoint
# reload, save and manifest, then scoring a few test windows with the trained adapter.
# All output goes to $SCRATCH/rehearsal/<id>/, never to checkpoints/, logs/, datasets/ or wandb.
#
# What is swapped, and only this: where outputs go, how long the run is, and in the cpu tier the model (a 2-layer
# random copy of the base's architecture) and the precision. Everything else is the code the job will run.
#
#   python rehearsal/rehearse.py cpu  qwen2.5vl:7b TUAB [--side]   # login node, ~10 min, no GPU
#   python rehearsal/rehearse.py gpu  qwen2.5vl:7b TUAB [--side]   # submits a <=1 h Slurm job with the real model
import argparse
import datetime
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("datasets", None)  # same guard as train/train.py: the repo's datasets/ is not the HF library
from helpers.pipeline import DATASETS, config, read_csv  # noqa: E402

SANDBOX = Path(os.environ.get("SCRATCH", "/tmp")) / "rehearsal"
STEPS = {"cpu": 6, "gpu": 8}  # optimizer steps; eval + save every half, crash injected after the first save
ROWS = {"cpu": (24, 6, 3), "gpu": (96, 16, 6)}  # train rows, val rows, test windows scored


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Report:
    def __init__(self, folder):
        self.folder, self.lines, self.failed = folder, [], False

    def check(self, label, ok, detail=""):
        self.failed |= not ok
        self.say(f"{'PASS' if ok else 'FAIL'}  {label}" + (f": {detail}" if detail else ""))

    def say(self, line):
        print(line, flush=True)
        self.lines.append(line)
        (self.folder / "report.txt").write_text("\n".join(self.lines) + "\n")


def spread(rows, n):
    # evenly spaced rows, so the sample keeps the file's class mix
    step = max(len(rows) // n, 1)
    return rows[::step][:n]


def tiny_snapshot(base_repo, snapshot_dir, out):
    # random weights in the base's own architecture, shrunk to 2 layers; the real processor and chat template.
    # Only Qwen2.5-VL / Qwen3-VL style configs (text + vision sub-configs) are shrunk here
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    if (out / "config.json").exists():
        return out
    source = snapshot_dir(base_repo)
    cfg = AutoConfig.from_pretrained(source)
    text = getattr(cfg, "text_config", cfg)
    head = text.hidden_size // text.num_attention_heads
    text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads = 2, 2, 1
    text.hidden_size, text.intermediate_size = 2 * head, 4 * head
    vision = cfg.vision_config
    vhead = vision.hidden_size // vision.num_heads
    vision.depth, vision.num_heads, vision.hidden_size, vision.intermediate_size = 2, 2, 2 * vhead, 4 * vhead
    vision.out_hidden_size, vision.fullatt_block_indexes = text.hidden_size, [1]
    if text is not cfg:
        cfg.hidden_size = text.hidden_size

    model = AutoModelForImageTextToText.from_config(cfg)
    model.save_pretrained(out)
    AutoProcessor.from_pretrained(source).save_pretrained(out)
    return out


def patch_trainer_module(t, tier, folder, rows, steps, report, tiny=None):
    # outputs to the sandbox, a sample of the data, a short run; cpu tier: tiny model, no bf16 autocast
    import transformers

    original_load_data, original_init_trainer = t.load_data, t.init_trainer
    t.checkpoint_dir = lambda key, run: folder / "checkpoints" / key.replace(":", "-") / run

    def load_data(settings):
        data = original_load_data(settings)
        return {"train": spread(data["train"], rows[0]), "validation": spread(data["validation"], rows[1])}

    def init_trainer(settings, model, processor, data, dataset_dir):
        trainer = original_init_trainer(settings, model, processor, data, dataset_dir)
        args = trainer.args
        args.max_steps, args.eval_steps, args.save_steps, args.logging_steps = steps, steps // 2, steps // 2, 1
        return trainer

    t.load_data, t.init_trainer = load_data, init_trainer
    if tier == "cpu":
        import torch

        t.resolve_model = lambda repo: str(tiny)
        torch.cuda.is_bf16_supported = lambda *a, **k: True  # init_model then loads bf16, as on the A100

        class CPUArguments(transformers.TrainingArguments):
            def __init__(self, *args, **kwargs):
                kwargs.update(bf16=False, use_cpu=True, dataloader_num_workers=2)
                super().__init__(*args, **kwargs)

        transformers.TrainingArguments = CPUArguments


def patch_wandb(folder):
    # our own wandb.init call runs with its real arguments, but offline and inside the sandbox
    import wandb

    original = wandb.init

    def init(*args, **kwargs):
        kwargs.update(dir=str(folder), mode="offline")
        (folder / "wandb-init.json").write_text(json.dumps({k: str(v) for k, v in kwargs.items()}, indent=2))
        return original(*args, **kwargs)

    wandb.init = init


def crash_after_first_save(trainer_module, steps):
    # a TrainerCallback that kills the run right after the first checkpoint, like a walltime or node failure
    from transformers import TrainerCallback

    class Crash(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == steps // 2:
                raise RuntimeError("rehearsal: injected crash after the first checkpoint")

    original = trainer_module.init_trainer

    def init_trainer(*args, **kwargs):
        trainer = original(*args, **kwargs)
        trainer.add_callback(Crash())
        return trainer

    return original, init_trainer


def rehearse_training(kind, t, train_fn, settings_fn, report, steps):
    # run 1 crashes after the first save; run 2 must resume from it and finish
    out = Path(settings_fn()["output-dir"])
    original, crashing = crash_after_first_save(t, steps)
    t.init_trainer = crashing
    started = time.time()
    try:
        train_fn(settings_fn())
        report.check(f"{kind}: injected crash fired", False, "training finished without crashing")
    except RuntimeError as e:
        report.check(f"{kind}: run 1 trained to the first checkpoint, then crashed as injected",
                     "injected crash" in str(e) and (out / f"checkpoint-{steps // 2}").is_dir(),
                     f"{time.time() - started:.0f} s")
    t.init_trainer = original

    started = time.time()
    train_fn(settings_fn())
    elapsed = time.time() - started
    manifest = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    states = sorted(out.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    state = json.loads(states[-1].read_text()) if states else {}
    history = state.get("log_history", [])
    losses = [e["loss"] for e in history if "loss" in e]
    evals = [e for e in history if "eval_label_balanced_accuracy" in e]

    report.check(f"{kind}: run 2 resumed and finished", bool(manifest) and state.get("global_step") == steps,
                 f"global_step {state.get('global_step')}, {elapsed:.0f} s")
    report.check(f"{kind}: train loss logged and finite", bool(losses) and all(l == l and l < 1e4 for l in losses),
                 f"{[round(l, 3) for l in losses]}")
    report.check(f"{kind}: eval logged label accuracy with label positions", bool(evals) and
                 all(e.get("eval_label_positions", 0) > 0 for e in evals),
                 f"{[(e['step'], round(e['eval_label_balanced_accuracy'], 3), e.get('eval_label_positions')) for e in evals]}")
    report.check(f"{kind}: best checkpoint chosen and adapter saved",
                 bool(state.get("best_model_checkpoint")) and any(out.glob("adapter_model.*")),
                 f"{state.get('best_model_checkpoint')}")
    report.check(f"{kind}: manifest complete", all(k in manifest for k in ("experiment", "target", "wandb_id", "commit")),
                 f"experiment {manifest.get('experiment')}, target {manifest.get('target')}")
    return out, manifest, elapsed


def rehearse_scoring(kind, out, manifest, base_key, dataset, n, report, tiny=None):
    # train/scripts/eval.py's own load / enforcer / predict on a few test windows, then eval/score.py's evaluate
    from structure import get_structure, labels, prompt, to_labels

    runner = load("rehearsal_eval", ROOT / "train" / "scripts" / "eval.py")
    if tiny is not None:
        runner.snapshot_dir = lambda repo: tiny
    score = load("rehearsal_score", ROOT / "eval" / "score.py")

    started = time.time()
    model, processor, loaded = runner.load({"base": base_key, "checkpoint": str(out)})
    rationale = loaded.get("target", "rationale") == "rationale"
    structure = get_structure(dataset, loaded.get("rationale_first", False), rationale)
    prefix_fn = runner.enforcer(processor, structure)
    truth = {r["path"]: r for r in read_csv(ROOT / "datasets" / dataset / "labels.csv")}
    windows = sorted(p for p in truth if p.startswith("test/"))[:n]
    rows, errors = [], []
    for rel in windows:
        try:
            parsed = runner.predict(model, processor, structure, ROOT / "datasets" / dataset / rel,
                                    prompt(dataset, rationale), prefix_fn)
        except Exception as e:  # the tiny model's random text may not parse; the real model's must
            errors.append(f"{type(e).__name__}: {str(e)[:120]}")
            continue
        values = to_labels(parsed, dataset)
        rows.append({"path": rel, "model": f"rehearsal-{kind}", **{c: str(v).lower() for c, v in values.items()},
                     "rationale": getattr(parsed, "text_rationale", "")})
    elapsed = time.time() - started
    report.check(f"{kind}: scoring loaded base + adapter and decoded with the enforcer", bool(rows) or bool(errors),
                 f"{len(rows)} parsed, {len(errors)} failed, {elapsed:.0f} s" + (f"; first error {errors[0]}" if errors else ""))
    if rows:
        summary = score.evaluate(f"rehearsal-{kind}", dataset, rows, truth, set(windows), 0)
        report.check(f"{kind}: eval/score.py evaluate() accepts the rows", summary is not None,
                     f"columns {list(labels(dataset))}")
    del model
    gc.collect()


def run_tier(tier, base_key, dataset, side, report, folder):
    import torch

    cfg = config()
    rows, steps = ROWS[tier], STEPS[tier]
    patch_wandb(folder)
    kinds = [("main", ROOT / "train" / "train.py")] + ([("side", ROOT / "side" / "rationale-w10" / "train.py")] if side else [])

    for kind, path in kinds:
        report.say(f"== {kind}: {path.relative_to(ROOT)} on {base_key} / {dataset}, {tier} tier")
        module = load(f"rehearsal_{kind}", path)
        t = module if kind == "main" else module.main
        tiny = tiny_snapshot(cfg["bases"][base_key]["repo"], t.snapshot_dir, SANDBOX / "tiny" / base_key.replace(":", "-")) \
            if tier == "cpu" else None
        patch_trainer_module(t, tier, folder / kind, rows, steps, report, tiny)
        if kind == "main":
            settings_fn = lambda: t.load_config(config(), base_key, dataset)  # noqa: E731
            train_fn = t.train
        else:
            module.output_dir = lambda key, ds: folder / kind / "checkpoints" / key.replace(":", "-") / ds
            settings_fn = lambda: module.side_config(config(), base_key, dataset)  # noqa: E731
            train_fn = module.train

        try:
            out, manifest, elapsed = rehearse_training(kind, t, train_fn, settings_fn, report, steps)
            if tier == "gpu":
                per_step = elapsed / (steps - steps // 2)
                total = len(t.read_jsonl(t.sft_file(dataset, "train", settings_fn()["target"]))) * \
                    cfg["train"]["training"]["epochs"] // cfg["train"]["training"]["gradient-accumulation"]
                walltime = cfg["bases"][base_key].get("time", cfg["train"]["time"])
                report.say(f"     {kind}: {per_step:.1f} s/step incl. one eval -> ~{per_step * total / 3600:.1f} h for "
                           f"{total} steps (walltime {walltime}); peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
            rehearse_scoring(kind, out, manifest, base_key, dataset, rows[2], report, tiny)
        except Exception:
            report.check(f"{kind}: no exception", False, traceback.format_exc().strip().splitlines()[-1])
            report.say(traceback.format_exc())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def submit_gpu(base_key, dataset, side, folder):
    from helpers.slurm import script, submit

    cfg = config()
    base = cfg["bases"][base_key]
    command = f"python {ROOT}/rehearsal/rehearse.py gpu {base_key} {dataset} --here --folder {folder}" + (" --side" if side else "")
    text = script(f"eeg-vlm-rehearsal-{base_key.replace(':', '-')}-{dataset}", "01:00:00", base.get("ram", cfg["train"]["ram"]),
                  cfg["train"]["cpus"], base.get("gpus", cfg["train"]["gpus"]), command)
    text = text.replace(str(ROOT / "logs"), str(folder))
    job = submit(text)
    print(f"{job}: GPU rehearsal; report in {folder}/report.txt, Slurm log in {folder}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=["cpu", "gpu"])
    parser.add_argument("model")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--side", action="store_true", help="also rehearse side/rationale-w10/train.py")
    parser.add_argument("--here", action="store_true", help=argparse.SUPPRESS)  # the gpu job itself
    parser.add_argument("--folder", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "offline")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(args.folder) if args.folder else SANDBOX / f"{stamp}-{args.tier}-{args.model.replace(':', '-')}-{args.dataset}"
    folder.mkdir(parents=True, exist_ok=True)

    if args.tier == "gpu" and not args.here:
        submit_gpu(args.model, args.dataset, args.side, folder)
        return

    report = Report(folder)
    report.say(f"rehearsal {args.tier} {args.model} {args.dataset} -> {folder}")
    run_tier(args.tier, args.model, args.dataset, args.side, report, folder)
    report.say("RESULT: " + ("FAIL" if report.failed else "PASS"))
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
