# scores one pilot variant on the full TUSZ test split; predictions and metrics stay in side/multilabel-pilot/results/
#   perclass:  every window gets every question. P(true) is read at the answer's boolean token, teacher-forced exactly
#              as in training (the collator builds the input), and a class is present at P(true) >= 0.5
#   joint-aux: the original joint prompt with constrained greedy decoding, what the labels round and zero-shot ran
# Resumes: windows already in the predictions file are skipped
#   python side/multilabel-pilot/score.py VARIANT [--smoke]
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

SIDE = Path(__file__).resolve().parent
sys.path.insert(0, str(SIDE))
sys.path.insert(0, str(SIDE.parents[1]))
sys.path.insert(0, str(SIDE.parents[1] / "eval" / "models"))
sys.modules.setdefault("datasets", None)  # same guard as train/train.py
from helpers.pipeline import ROOT, append_row, db, read_csv, script_module  # noqa: E402
from questions import DATASET, QUESTIONS, SEIZURES, answer, question_prompt  # noqa: E402
from structure import get_structure, labels, prompt, to_labels  # noqa: E402

CHUNK = 5  # questions per forward pass: 5 copies of the image fit a 20 GB slice with room to spare
SMOKE_WINDOWS = 24
PROGRESS_EVERY = 50


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_path("pilot_train", SIDE / "train.py")


def test_scope():
    rows = db().execute("SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'full'", (DATASET,))
    return {p.split("/", 1)[1] for (p,) in rows}


def boolean_ids(processor, device):
    import torch

    ids = pilot.main.boolean_token_ids(processor.tokenizer)
    return {word: torch.tensor(sorted(ids[word]), device=device) for word in ("true", "false")}


def question_probabilities(model, collator, ids, rel):
    # one forward per CHUNK questions; the collator builds exactly the training input, answer included, so the
    # boolean token's position is known and only the logits before it are needed
    import torch

    probabilities = {}
    every_id = torch.cat([ids["true"], ids["false"]]).cpu()

    for start in range(0, len(QUESTIONS), CHUNK):
        chunk = QUESTIONS[start:start + CHUNK]
        batch = collator([{"instruction": question_prompt(q), "input": "", "output": answer(True), "images": [rel]}
                          for q in chunk])
        targets = batch.pop("labels")
        positions = [int(torch.isin(row, every_id).nonzero()[0]) for row in targets]
        keep = targets.shape[1] - min(positions) + 1

        with torch.inference_mode():
            logits = model(**batch.to(model.device), logits_to_keep=keep).logits.float()

        offset = targets.shape[1] - keep
        for row, (question, position) in enumerate(zip(chunk, positions)):
            logp = logits[row, position - 1 - offset].log_softmax(-1)
            gap = logp[ids["true"]].logsumexp(0) - logp[ids["false"]].logsumexp(0)
            probabilities[question] = float(torch.sigmoid(gap))

    return probabilities


def predict_all(variant, smoke):
    runner = script_module("eval")
    results = pilot.base_dir(smoke) / "results"
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"eval-{variant}.csv"
    name = pilot.model_name(variant)
    classes = labels(DATASET)
    header = ["path", "model", *classes, "rationale"] + ([f"p_{q}" for q in QUESTIONS] if variant == "perclass" else [])
    done = {r["path"] for r in read_csv(out)}
    todo = sorted(test_scope() - done)
    if smoke:
        todo = todo[::max(len(todo) // SMOKE_WINDOWS, 1)][:SMOKE_WINDOWS]
    print(f"{name}: {len(todo)} windows to score, {len(done)} already done -> {out}", flush=True)

    if not todo:
        return out

    checkpoint = pilot.output_dir(variant, smoke)
    model, processor, manifest = runner.load({"base": pilot.MODEL, "checkpoint": str(checkpoint.relative_to(ROOT))})
    folder = ROOT / "datasets" / DATASET
    if variant == "perclass":
        collator = pilot.main.DataCollator(processor, folder, manifest["base"])
        ids = boolean_ids(processor, model.device)
    else:
        structure = get_structure(DATASET, False, rationale=False)
        text, prefix_fn = prompt(DATASET, False), runner.enforcer(processor, structure)

    started = time.time()
    for i, rel in enumerate(todo, 1):
        try:
            if variant == "perclass":
                p = question_probabilities(model, collator, ids, rel)
                row = [*[str(p[c] >= 0.5).lower() for c in classes], "", *[f"{p[q]:.5f}" for q in QUESTIONS]]
            else:
                values = to_labels(runner.predict(model, processor, structure, folder / rel, text, prefix_fn), DATASET)
                row = [*[str(v).lower() for v in values.values()], ""]
        except Exception as e:
            append_row(results / "failures.csv", ["path", "model", "error"], [rel, name, f"{type(e).__name__}: {e}"[:300]])
            print(f"failed {rel}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        append_row(out, header, [rel, name, *row])
        if i % PROGRESS_EVERY == 0 or smoke and i <= 3:
            print(f"{i}/{len(todo)} ({(time.time() - started) / i:.2f} s/window)" +
                  (f" {dict(zip(QUESTIONS, row[len(classes) + 1:]))}" if smoke and variant == "perclass" else ""),
                  flush=True)
    print(f"{len(todo)} windows in {(time.time() - started) / 60:.1f} min", flush=True)
    return out


def auroc(scores, positive):
    # Mann-Whitney, ties at their average rank
    scores, positive = np.asarray(scores, float), np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return None
    order = scores.argsort()
    _, inverse, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = (np.bincount(inverse, np.arange(1, len(scores) + 1)) / counts)[inverse]
    n = positive.sum()
    return (ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(scores) - n))


def summarise(variant, predictions):
    # eval/score.py's recording-level metrics, written beside the predictions; perclass also gets window AUROC per
    # question from its probabilities
    score = load_path("eval_score", ROOT / "eval" / "score.py")
    results = predictions.parent
    scope, name = test_scope(), pilot.model_name(variant)
    rows = [r for r in {r["path"]: r for r in read_csv(predictions)}.values() if r["path"] in scope]
    truth = {r["path"]: r for r in read_csv(ROOT / "datasets" / DATASET / "labels.csv")}
    summary = score.evaluate(name, DATASET, rows, truth, scope, score.BOOTSTRAP)
    if summary is None:
        sys.exit(f"no scored rows in {predictions}")

    score.write(results / f"summary-{variant}.csv", score.FIELDS, [{k: summary[k] for k in score.FIELDS}])
    score.write(results / f"classes-{variant}.csv", score.CLASS_FIELDS,
                [dict(zip(score.CLASS_FIELDS, (name, DATASET, *c))) for c in summary["_classes"]])
    print(f"{name}: recording balanced accuracy {summary['recording_balanced_accuracy']:.3f}, recording macro-F1 "
          f"{summary['recording_macro_f1']:.3f} [{summary['recording_ci_low']}, {summary['recording_ci_high']}] vs floor "
          f"{summary['baseline_macro_f1']}, coverage {summary['coverage']:.2f}, top answer {summary['top_answer']} "
          f"({summary['top_answer_fraction']:.0%})", flush=True)

    if variant == "perclass":
        seizure = [any(truth[r["path"]][c] == "true" for c in SEIZURES) for r in rows]
        lines = []
        for q in QUESTIONS:
            positive = seizure if q == "any_seizure" else [truth[r["path"]][q] == "true" for r in rows]
            value = auroc([float(r[f"p_{q}"]) for r in rows], positive)
            lines.append({"question": q, "positives": sum(positive), "windows": len(rows),
                          "auroc": "" if value is None else round(value, 4)})
            print(f"  {q:12s} AUROC {lines[-1]['auroc']} ({sum(positive)} positive windows)", flush=True)
        score.write(results / "auroc-perclass.csv", ["question", "positives", "windows", "auroc"], lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=pilot.VARIANTS)
    parser.add_argument("--smoke", action="store_true", help="the smoke run's checkpoint, 24 windows, no summary")
    args = parser.parse_args()

    predictions = predict_all(args.variant, args.smoke)
    if not args.smoke:
        summarise(args.variant, predictions)


if __name__ == "__main__":
    main()
