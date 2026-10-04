# sizes every (base, dataset) job of train.experiment into train/resources.csv, which train/train.py reads when it
# submits: a walltime from measured speeds, a MIG slice for the small bases the memory probe cleared, RAM, and a nice
# value that orders our own batch so the pairs needed first start first
#   python helpers/plan-resources.py [--probe $SCRATCH/rehearsal/<stamp>-probe-memory/results.jsonl]
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import DATASETS, ROOT, base_spec, checkpoint_dir, config, db, run_name, sft_file  # noqa: E402

# s per optimizer step (8 micro-batches), tqdm rate of each base's round-1 pooled run (2026-09-23..28,
# archive/round1-pooled/logs). It includes that run's generation evals, so it overstates: qwen2.5vl:7b showed 36.8
# there and 26.7 in the 2026-10-03 label-only run, which is used for it instead. gemma4:12b died at step 300; its
# rate is from those steps
S_PER_STEP = {
    "bakllava:7b": 12.1, "gemma3:4b": 14.9, "gemma3:12b": 20.3, "gemma3:27b": 31.5, "gemma4:e2b": 16.1,
    "gemma4:e4b": 19.3, "gemma4:12b": 20.0, "gemma4:26b": 23.7, "gemma4:31b": 32.1, "glm-ocr:latest": 15.9,
    "granite3.2-vision:2b": 28.5, "llava:7b": 12.1, "llava:13b": 16.0, "llava:34b": 55.2, "llava-llama3:8b": 22.9,
    "llava-phi3:3.8b": 16.2, "medgemma:4b": 15.2, "medgemma1.5:4b": 15.4, "minicpm-v4.6:1b": 25.9,
    "mistral-small3.1:24b": 57.0, "mistral-small3.2:24b": 57.9, "qwen2.5vl:7b": 26.7, "qwen2.5vl:32b": 77.1,
    "qwen3-vl:2b-instruct": 14.9, "qwen3-vl:4b-instruct": 20.2, "qwen3-vl:8b-instruct": 26.0,
    "qwen3-vl:30b-a3b-instruct": 31.0, "qwen3-vl:32b-instruct": 60.2,
}
REFERENCE = 36.8  # qwen2.5vl:7b's round-1 s/step: speeds scale against it
# s per generated token of constrained decoding at the reference speed: qwen2.5vl:7b label-only TUAB measured 1.0
# s/window for ~10 tokens (2026-10-03). Gemma-family decoding measured ~3x slower per token in round 1 (gemma3:27b,
# 29.6 s/image for ~170 tokens), plausibly its 262k vocabulary in the enforcer
TOKEN_S = {"default": 0.06, "gemma": 0.20}
PREFILL_S = 0.5
ANSWER_TOKENS = {"TUAB": 10, "TUEP": 10, "TUAR": 58, "TUEV": 50, "TUSL": 26, "TUSZ": 74}  # ~8 per boolean field
# two processes (train, then score), each importing from Lustre and loading the weights: qwen2.5vl:7b (16.6 GB) took
# ~12 min to its first step and ~10 min to its first scored window (2026-10-03); load time grows with the weights
STARTUP_H, STARTUP_H_PER_GB = 0.4, 0.015
EVAL_ROW_S = 0.03  # x s/step per val row (qwen2.5vl:7b: 114 s for 140 rows at 26.7 s/step)
TRAIN_MARGIN, SCORE_MARGIN = 1.5, 1.25  # a cut-off in training costs a resubmission; in scoring, run-eval.py resumes
MIG_SLICE, MIG_RAM, MIG_HEADROOM_GB = "a100_3g.20gb:1", "40G", 17.0  # 15% under the slice's 20 GB
LOW_PRIORITY = {"llava:7b", "llava:13b", "bakllava:7b", "llava-llama3:8b", "llava-phi3:3.8b"}  # 336 px, plan.md


def nice(key, dataset, first):
    # order within our own batch only: a few hundred points move us past our own jobs, not other groups'
    if key in LOW_PRIORITY:
        return 300
    if key == first or dataset in ("TUAB", "TUEP"):
        return 0
    return 200 if dataset == "TUSZ" else 100


def test_windows():
    rows = db().execute("SELECT dataset, COUNT(DISTINCT path) FROM pipeline WHERE scope = 'full' GROUP BY dataset")
    return dict(rows)


def weights_gb(base):
    snapshot = load_snapshot_dir()(base["repo"])
    return sum(f.stat().st_size for f in snapshot.glob("*.safetensors")) / 1e9


def load_snapshot_dir():
    import importlib.util

    spec = importlib.util.spec_from_file_location("hf_install", ROOT / "train" / "scripts" / "hf-install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.snapshot_dir


def hours(key, dataset, rows, val_rows, windows, cfg, probe, gb):
    training = cfg["train"]["training"]
    steps = 2 * math.ceil(rows / (training["batch-size"] * training["gradient-accumulation"]))
    evals = steps // training["eval-steps"] + 1
    if probe:  # measured on the MIG slice: one optimizer step and one TUSZ window
        s_step = probe["step_s"]
        tokens = ANSWER_TOKENS[dataset]
        s_window = probe["score_s"] * (0.1 + 0.9 * tokens / ANSWER_TOKENS["TUSZ"])
    else:
        s_step = S_PER_STEP[key]
        family = "gemma" if "gemma" in key else "default"
        s_window = (s_step / REFERENCE) * (PREFILL_S + ANSWER_TOKENS[dataset] * TOKEN_S[family])
    train_h = (steps * s_step + evals * val_rows * EVAL_ROW_S * s_step) / 3600
    score_h = windows * s_window / 3600
    startup = STARTUP_H + STARTUP_H_PER_GB * gb
    return steps, train_h, score_h, startup + TRAIN_MARGIN * train_h + SCORE_MARGIN * score_h


def walltime(h):
    # capped at Narval's longest GPU tier (3 days); a job that needs more loses only the tail of its scoring, which
    # run-eval.py resumes
    h = min(max(2, math.ceil(h)), 71)  # 2 h floor: a short job's whole cost is its startup, which varies
    return f"{h // 24}-{h % 24:02d}:00:00" if h >= 24 else f"{h:02d}:00:00"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", help="results.jsonl of rehearsal/probe-memory.py: bases that fit go on a MIG slice")
    args = parser.parse_args()
    cfg = config()
    experiment, target = cfg["train"]["experiment"], cfg["train"]["target"]
    probe = {}
    if args.probe:
        for line in Path(args.probe).read_text().splitlines():
            r = json.loads(line)
            if "error" not in r and max(r["train_peak_gb"], r["score_peak_gb"]) < MIG_HEADROOM_GB:
                probe[r["key"]] = r
    windows = test_windows()
    keys = [k for k, b in cfg["bases"].items() if not b.get("status") and not b.get("family")]
    plan = []
    for key in keys:
        base = base_spec(cfg, key)
        gb = weights_gb(base)
        for dataset in DATASETS:
            if (checkpoint_dir(key, run_name(dataset, experiment)) / "manifest.json").exists():
                continue
            rows = sum(1 for _ in sft_file(dataset, "train", target).open())
            val_rows = sum(1 for _ in sft_file(dataset, "val", target).open())
            steps, train_h, score_h, total = hours(key, dataset, rows, val_rows, windows[dataset], cfg, probe.get(key), gb)
            on_mig = key in probe
            plan.append({"base": key, "dataset": dataset, "steps": steps, "train_h": round(train_h, 1),
                         "score_h": round(score_h, 1), "time": walltime(total),
                         "gpus": MIG_SLICE if on_mig else base["gpus"], "ram": MIG_RAM if on_mig else base["ram"],
                         "nice": nice(key, dataset, "qwen2.5vl:7b")})
    with (ROOT / "train" / "resources.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(plan[0]))
        writer.writeheader()
        writer.writerows(plan)
    gpu_hours = sum((r["train_h"] + r["score_h"]) * (1 if "g." in str(r["gpus"]) else int(r["gpus"])) for r in plan)
    print(f"{len(plan)} pairs -> train/resources.csv; {sum(1 for r in plan if 'g.' in str(r['gpus']))} on MIG; "
          f"~{gpu_hours:.0f} GPU-hours estimated (MIG slices counted as whole GPUs)")
    for r in sorted(plan, key=lambda r: (r["nice"], r["dataset"], r["base"])):
        print(f"  nice {r['nice']:3d} {r['base']:28s} {r['dataset']} steps {r['steps']:4d} train {r['train_h']:5.1f} h "
              f"score {r['score_h']:5.1f} h -> {r['time']:>11s} gpu:{r['gpus']} {r['ram']}")


if __name__ == "__main__":
    main()
