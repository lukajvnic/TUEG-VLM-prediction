"""Write checkpoints/<key>/<dataset>/manifest.json after a custom-stack run (LLaMA-Factory, Unsloth).

train/predict.py refuses a checkpoint without a manifest; the generic trainer writes its own. The custom
stacks know nothing about it, so the sbatch calls this after the trainer exits 0. Refuses if no adapter
weights are in the directory, so a crashed run cannot be scored.
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers.pipeline import ROOT, base_spec, checkpoint_dir, config


def write_manifest(model, dataset, stack, extra=None):
    cfg = config()
    out = checkpoint_dir(model, dataset)
    if not any((out / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        sys.exit(f"{out}: no adapter_model.* - training did not finish, no manifest written")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    train = cfg["train"]
    manifest = {"dataset": dataset, "model": model, "base": base_spec(cfg, model), "stack": stack,
                "rationale_first": train["rationale-first"], "lora": train["lora"], "training": train["training"],
                "commit": commit, "finished": datetime.datetime.now().isoformat(timespec="seconds"), **(extra or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}/manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="`bases:` key")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stack", required=True, help="what trained it, e.g. llamafactory")
    args = parser.parse_args()
    write_manifest(args.model, args.dataset, args.stack)


if __name__ == "__main__":
    main()
