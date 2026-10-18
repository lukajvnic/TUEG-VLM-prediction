# assembles the hand-over package: the best fine-tuned model per dataset as adapter + processor, the exact prompts
# run.py scores with (model.json), 4 sample test images with the P(true) the cluster gave them, predict.py, the
# Dockerfile and the README. Adapters come from WEIGHTS/<arm>/ (rsync of the cluster's checkpoints/<arm>/ top dir)
#   python side/vision-freeze/package/build.py WEIGHTS OUT
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIDE = HERE.parent
# best test AUROC per dataset among the 11 focus models, 2026-10-18
MODELS = {"g-8b-tusz": "tusz-seizure-qwen3-vl-8b", "gm-gemma4-e4b-tuab": "tuab-abnormal-gemma4-e4b",
          "gm-medgemma15-4b-tuar": "tuar-eye-movement-medgemma1.5-4b", "gm-granite-2b-tuev": "tuev-epileptiform-granite3.2-vision-2b"}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "chat_template.jinja", "processor_config.json",
                 "tokenizer.json", "tokenizer_config.json")


def run_module():
    sys.argv = ["run.py"]
    spec = importlib.util.spec_from_file_location("run", SIDE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    weights, out = Path(sys.argv[1]), Path(sys.argv[2])
    run = run_module()
    for name in ("predict.py", "Dockerfile", "requirements.txt", "README.md"):
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(HERE / name, out / name)

    for arm, name in MODELS.items():
        target = out / "models" / name
        (target / "adapter").mkdir(parents=True, exist_ok=True)
        for file in ADAPTER_FILES:
            shutil.copy(weights / arm / file, target / "adapter" / file)
        answer_first = run.order_of(arm) == "answer-first"
        summary = read(SIDE / "results" / f"summary-{arm}.csv")[0]
        spec = {
            "name": name, "arm": arm, "dataset": run.dataset_of(arm),
            "question": run.prompt_text(arm).rsplit("Based only on these waveforms, ", 1)[1],
            "base": run.base_spec(run.config(), run.model_of(arm))["repo"],
            "prompt": run.rationale_prompt(arm) if answer_first else run.prompt_text(arm),
            "answer": run.AF_ANSWER if answer_first else run.ANSWER,
            # answer-first models answer and explain in one reply; multitask models were taught a separate explain line
            "explain_prompt": run.rationale_prompt(arm) if answer_first else run.explain_prompt(arm),
            "test": {k: summary[k] for k in ("windows", "seizure_windows", "patients", "auroc", "auroc_ci_low",
                                             "auroc_ci_high", "balanced_accuracy_at_0.5")},
            "training": {k: v for k, v in run.ARMS[arm].items() if k in ("recipe", "order", "epochs", "strip", "weight")},
        }
        (target / "model.json").write_text(json.dumps(spec, indent=2) + "\n")

        # 2 positive and 2 negative test windows, with the cluster's P(true), to check an install against
        scores = read(SIDE / "results" / f"scores-{arm}.csv")
        picks = [r for r in scores if r["seizure"] == "true"][:2] + [r for r in scores if r["seizure"] != "true"][:2]
        (target / "samples").mkdir(exist_ok=True)
        with (target / "samples" / "expected.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label", "p_true_cluster"])
            for r in picks:
                image = Path(r["path"]).name
                shutil.copy(SIDE / "images-core" / spec["dataset"] / r["path"], target / "samples" / image)
                writer.writerow([image, r["seizure"], r["p_true"]])
        print(f"{name}: {spec['base']}, test AUROC {summary['auroc']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
