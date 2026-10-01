# one experiment's results: every base's zero-shot row beside its fine-tune of that experiment, per dataset, from
# datasets/<DS>/summary.csv (run eval/score.py first); printed and written to experiments/<experiment>.csv
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import DATASETS, ROOT, config, read_csv, run_name  # noqa: E402

COLUMNS = ["experiment", "base", "dataset", "zero_shot_ba", "fine_tuned_ba", "delta", "coverage",
           "top_answer", "top_answer_fraction", "distinct_answers"]


def rows_for(experiment, bases):
    rows = []
    for dataset in DATASETS:
        summary = {r["model"]: r for r in read_csv(ROOT / "datasets" / dataset / "summary.csv")}
        for base in bases:
            tuned = summary.get(f"{base}-sft-{run_name(dataset, experiment)}")
            if tuned is None:
                continue

            zero = summary.get(base)
            zero_ba = float(zero["recording_balanced_accuracy"]) if zero else None
            tuned_ba = float(tuned["recording_balanced_accuracy"])
            rows.append({"experiment": experiment, "base": base, "dataset": dataset,
                         "zero_shot_ba": zero_ba, "fine_tuned_ba": tuned_ba,
                         "delta": None if zero_ba is None else round(tuned_ba - zero_ba, 4),
                         "coverage": tuned["coverage"], "top_answer": tuned["top_answer"],
                         "top_answer_fraction": tuned["top_answer_fraction"],
                         "distinct_answers": tuned["distinct_answers"]})
    return rows


def main():
    cfg = config()
    experiment = sys.argv[1] if len(sys.argv) > 1 else cfg["train"]["experiment"]
    rows = rows_for(experiment, cfg["bases"])

    if not rows:
        sys.exit(f"no <base>-sft-<DS>-{experiment} rows in any datasets/<DS>/summary.csv (run eval/score.py first)")

    out = ROOT / "experiments" / f"{experiment}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'base':28} {'dataset':7} {'zero':>6} {'tuned':>6} {'delta':>6} {'cov':>5}  top answer")
    for r in rows:
        zero = "-" if r["zero_shot_ba"] is None else f"{r['zero_shot_ba']:.3f}"
        delta = "-" if r["delta"] is None else f"{r['delta']:+.3f}"
        print(f"{r['base']:28} {r['dataset']:7} {zero:>6} {r['fine_tuned_ba']:6.3f} {delta:>6} "
              f"{float(r['coverage']):5.2f}  {r['top_answer'][:24]} ({float(r['top_answer_fraction']):.0%})")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
