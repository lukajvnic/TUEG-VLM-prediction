# peak GPU memory and speed of one real training step, one eval step and one scored window per base, on whatever GPU
# the job got (a MIG slice is the point): decides which bases can train on a 20 GB a100_3g.20gb slice. Each base runs
# in its own process, so one OOM doesn't hide the next. Output under $SCRATCH/rehearsal/, nothing in the repo
#   python rehearsal/probe-memory.py submit KEY [KEY ...]     # a <=2 h job on an a100_3g.20gb slice
#   python rehearsal/probe-memory.py one KEY                  # inside the job, per base
import datetime
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "models"))
sys.modules.setdefault("datasets", None)
DATASET = "TUSZ"  # the longest label-only answer (9 fields) and the largest scoring set
SLICE = "a100_3g.20gb:1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(key, out):
    os.environ["WANDB_MODE"] = "disabled"  # no run to log, and a compute node has no internet for an online one
    import torch

    from helpers.pipeline import config
    from structure import get_structure, prompt

    t = load("train_main", ROOT / "train" / "train.py")
    settings = t.load_config(config(), key, DATASET)
    settings["output-dir"] = str(out / key.replace(":", "-"))
    data = t.load_data(settings)
    data = {"train": data["train"][:8], "validation": data["validation"][:2]}
    result = {"key": key, "device": torch.cuda.get_device_name(0),
              "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}

    started = time.time()
    model, processor = t.init_model(settings)
    result["load_s"] = round(time.time() - started)
    trainer = t.init_trainer(settings, model, processor, data, ROOT / "datasets" / DATASET)
    trainer.args.max_steps, trainer.args.eval_steps, trainer.args.save_strategy = 1, 1, "no"
    trainer.args.load_best_model_at_end, trainer.args.report_to = False, []
    started = time.time()
    trainer.train()
    result["step_s"] = round(time.time() - started, 1)  # one optimizer step = 8 micro-batches, plus the eval
    result["train_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)

    torch.cuda.reset_peak_memory_stats()
    runner = t.runner
    structure = get_structure(DATASET, False, rationale=False)
    model.eval()
    window = next(p for p in sorted((ROOT / "datasets" / DATASET / "test").glob("*.png")))
    started = time.time()
    runner.predict(model, processor, structure, window, prompt(DATASET, False), runner.enforcer(processor, structure))
    result["score_s"] = round(time.time() - started, 1)
    result["score_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    return result


def run_one(key):
    out = Path(os.environ["PROBE_DIR"])
    try:
        result = probe(key, out)
    except Exception as e:  # torch.OutOfMemoryError included: that is the answer for this base
        result = {"key": key, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    with (out / "results.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


def submit(keys):
    from helpers.pipeline import config
    from helpers.slurm import script
    from helpers.slurm import submit as sbatch

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(os.environ.get("SCRATCH", Path.home())) / "rehearsal" / f"{stamp}-probe-memory"
    out.mkdir(parents=True)
    runs = "\n".join(f"PROBE_DIR={out} python {ROOT}/rehearsal/probe-memory.py one {k}" for k in keys)
    text = script("eeg-vlm-probe-memory", "02:00:00", "40G", config()["train"]["cpus"], SLICE, runs)  # the MIG jobs' RAM
    text = text.replace("set -euo pipefail", "set -uo pipefail")  # one base failing must not stop the rest
    print(sbatch(text.replace(str(ROOT / "logs"), str(out))), "->", out / "results.jsonl")


if __name__ == "__main__":
    if sys.argv[1] == "submit":
        submit(sys.argv[2:])
    else:
        run_one(sys.argv[2])
