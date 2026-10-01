# per (base, dataset, experiment) fine-tune: the val curve from its trainer state, then zero-shot vs fine-tuned summary.csv rows
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ROOT, checkpoint_dir, config, read_csv, run_name, trained_runs  # noqa: E402


def latest_state(folder):
    # the newest checkpoint's trainer_state.json holds the whole log_history and the best checkpoint so far
    states = sorted(folder.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1]))
    return json.loads(states[-1].read_text()) if states else None


def val_curve(state):
    # one dict per eval step; merged by step, since a step's loss and metrics may be logged as separate entries
    evals = {}
    for entry in state["log_history"]:
        if any(k.startswith("eval_") for k in entry):
            evals.setdefault(entry["step"], {}).update(entry)
    return [evals[step] for step in sorted(evals)]


def fmt(value, width=6, digits=3):
    return f"{value:{width}.{digits}f}" if value is not None else f"{'-':>{width}}"


def print_curve(folder):
    state = latest_state(folder)
    if state is None:
        print("  no checkpoint-*/trainer_state.json yet")
        return

    best = Path(state.get("best_model_checkpoint") or "").name
    print(f"  val: {'step':>6} {'epoch':>6} {'loss':>6} {'labelBA':>7}")
    for e in val_curve(state):
        mark = "  best" if f"checkpoint-{e['step']}" == best else ""
        print(f"       {e['step']:6d} {fmt(e.get('epoch'), digits=2)} {fmt(e.get('eval_loss'))} "
              f"{fmt(e.get('eval_label_balanced_accuracy'), 7)}{mark}")
    if best:
        print(f"  best {best}, metric {fmt(state.get('best_metric'), 0)}")


def print_test(key, dataset, run):
    rows = {r["model"]: r for r in read_csv(ROOT / "datasets" / dataset / "summary.csv")}
    pairs = [(label, rows[m]) for label, m in (("zero", key), ("tuned", f"{key}-sft-{run}")) if m in rows]
    if not pairs:
        print(f"  test: no rows in datasets/{dataset}/summary.csv (run eval/score.py {dataset})")
        return

    print(f"  test:{'':5} {'cov':>4} {'recBA':>6} {'winBA':>6}  top answer")
    for label, r in pairs:
        print(f"       {label:5} {float(r['coverage']):4.2f} {float(r['recording_balanced_accuracy']):6.3f} "
              f"{float(r['window_balanced_accuracy']):6.3f}  {r['top_answer'][:30]} "
              f"({float(r['top_answer_fraction']):.0%}), {r['distinct_answers']} distinct")


def main():
    cfg = config()
    keys = sys.argv[1:] or [k for k in cfg["bases"] if trained_runs(k)]
    if not keys:
        print("no base has a checkpoints/<key>/<DS>-<experiment>/ dir")
        return

    for key in keys:
        runs = trained_runs(key)
        if not runs:
            print(f"\n=== {key}: no checkpoints/{key.replace(':', '-')}/<DS>-<experiment>/ dir")
        for dataset, experiment in runs:
            run = run_name(dataset, experiment)
            folder = checkpoint_dir(key, run)
            status = "finished" if (folder / "manifest.json").exists() else "no manifest.json: still training, or died"
            print(f"\n=== {key} on {run}  ({status})")
            print_curve(folder)
            print_test(key, dataset, run)


if __name__ == "__main__":
    main()
