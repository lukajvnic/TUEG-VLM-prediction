# fast pre-submission check of every (base, dataset) pair, no GPU and no weights. For each pair, with config.yml's
# train.target: the base's own processor and chat template build a real training batch from 2 rows of the jsonl,
# the label masking leaves exactly the answer, the true/false positions the label metric reads are all there, and
# the scoring enforcer accepts the trained answer token by token
#   python rehearsal/check-pairs.py [MODEL ...]      # default: every base train/train.py would submit
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "models"))
sys.modules.setdefault("datasets", None)
from helpers.pipeline import DATASETS, base_spec, config, sft_file  # noqa: E402
from structure import FIELDS, get_structure  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answer_accepted(enforcer, prompt_ids, answer_ids):
    # feed the generation prompt, then the trained answer one token at a time; every token must be allowed
    prefix_fn = enforcer.fresh()
    sequence = list(prompt_ids)
    for token in answer_ids:
        if token not in set(prefix_fn(0, __import__("torch").tensor(sequence))):
            return False
        sequence.append(token)
    return True


def check_base(t, cfg, key, target, report):
    import torch

    base = base_spec(cfg, key)
    processor = t.runner.load_processor(t.resolve_model(base["repo"]), base, base.get("trust-remote-code", False))
    tokenizer = processor.tokenizer
    booleans = t.boolean_token_ids(tokenizer)
    ids = torch.tensor(sorted(booleans["true"] | booleans["false"]))
    suffix = base.get("answer-suffix", "")

    for dataset in DATASETS:
        try:
            rows = t.read_jsonl(sft_file(dataset, "train", target))[:2]
            collator = t.DataCollator(processor, t.ROOT / "datasets" / dataset, base)
            structure = get_structure(dataset, cfg["train"]["rationale-first"], target == "rationale")
            enforcer = t.runner.enforcer(processor, structure)
            problems = []
            for row in rows:
                # one example per batch, as the trainer sees them (per_device batch size 1)
                labels = collator([row])["labels"][0]
                answer = labels[labels != -100]
                decoded = tokenizer.decode(answer)
                if not decoded.startswith(row["output"] + suffix):
                    problems.append(f"masked target {decoded[:60]!r} does not start with the answer")
                positions = int(torch.isin(answer, ids).sum())
                if positions < len(FIELDS[dataset]):
                    problems.append(f"{positions} true/false tokens for {len(FIELDS[dataset])} fields")
                # the exact tokens it trains on, up to the end of the JSON (the end-of-turn tokens follow)
                trained = []
                for token in answer.tolist():
                    trained.append(token)
                    if len(tokenizer.decode(trained)) >= len(row["output"]):
                        break
                prompt = t.runner.render(processor, t.get_messages(row, False), True)
                prompt_ids = tokenizer(prompt, **t.runner.encode_kwargs(processor, prompt))["input_ids"]
                if not answer_accepted(enforcer, prompt_ids, trained):
                    problems.append("the scoring enforcer rejects the trained answer tokens")
            report(key, dataset, sorted(set(problems)))
        except Exception as e:
            report(key, dataset, [f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"])
            traceback.print_exc()


def main():
    t = load("train_main", ROOT / "train" / "train.py")
    cfg = config()
    target = cfg["train"]["target"]
    keys = sys.argv[1:] or [k for k, b in cfg["bases"].items() if not b.get("status") and not b.get("family")]
    failures = []

    def report(key, dataset, problems):
        line = f"{'FAIL' if problems else 'ok  '}  {key} {dataset}" + (": " + "; ".join(problems) if problems else "")
        print(line, flush=True)
        if problems:
            failures.append(line)

    for key in keys:
        try:
            check_base(t, cfg, key, target, report)
        except Exception as e:
            report(key, "*", [f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"])
    print(f"RESULT: {'FAIL' if failures else 'PASS'} ({len(keys)} bases x {len(DATASETS)} datasets, target {target})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
