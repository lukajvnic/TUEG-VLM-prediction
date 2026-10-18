# the results to hand over, as plain CSVs in results/export/: metrics.csv (one row per model x dataset x kind),
# predictions.csv (every scored window: P(true) and the boolean at 0.5), rationales.csv (the generated rationales of
# the rationale evaluation) and tuep_scan.csv, plus a README. Grid arms only (g-, gm-, z-, s-); the earlier
# exploratory rounds stay in results/
#   python side/vision-freeze/export.py [RESULTS_DIR]
import csv
import re
import sys
from pathlib import Path

import tables
from tables import MODELS

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
ARM = re.compile(r"^(g|gm|z|s)-(.+)-(tusz|tuab|tuar|tuev)(-s2)?(-last)?$")
KIND = {"g": "fine-tuned", "gm": "fine-tuned", "z": "zero-shot", "s": "detector (vision only, answer only)"}
QUESTION = {"TUSZ": "seizure (any type)", "TUAB": "abnormal EEG", "TUAR": "eye movement artifact",
            "TUEV": "epileptiform discharge (spike/sharp wave, GPED, PLED)"}
README = """# EEG VLM results export

Every number comes from teacher-forced P(true) for the yes/no question of each dataset, read at the answer's
boolean token. "predicted" is P(true) >= 0.5. AUROC is the headline metric (the 0.5 threshold is not calibrated for
every model); its 95% CI is a patient-level bootstrap (1,000 resamples). All splits are patient-disjoint.

Datasets and questions: TUSZ seizure (any type), TUAB abnormal EEG, TUAR eye movement artifact, TUEV epileptiform
discharge. Images: 20 s windows, 25-row TCP bipolar montage, 2688x896.

kind: zero-shot = the base model untrained; fine-tuned = vision tower trained in full plus LoRA on the language
model, trained on the label with a teacher rationale (TUSZ answer-first, the others in the multitask format);
variant "last" = the run's final checkpoint instead of the best validation one, "seed2" = a second seed.

## summary_table.csv
The headline table: test AUROC per model and dataset, zero-shot against fine-tuned, with the 95% CI, plus the final
checkpoint ("last") and a second seed ("s2") where they exist.

## metrics.csv
model, dataset, question, kind, variant, arm, set (test / heldout = 62 TUSZ patients no model trained on /
interior-background = 4,471 mid-recording background windows, scored against the test seizures), windows,
positives, patients, auroc, auroc_ci_low, auroc_ci_high, balanced_accuracy, sensitivity, specificity, precision,
f1, predicted_positive_fraction (all at 0.5).

## predictions.csv
model, dataset, kind, variant, set, image, patient, label (true = the class is present), p_true, predicted.

## rationales.csv
The 300-window rationale evaluation (150 positive, 150 negative test windows) per fine-tuned model: format
(answer-first or explain), label, generated_present (answer-first only), reference_nll (mean per-token NLL of the
teacher's label-stripped rationale under the model; lower = closer to the teacher), rationale (the model's own
generated text).

## tuep_scan.csv
TUEP patient-level epilepsy diagnosis by scanning each test recording with a TUEV epileptiform detector: a
recording scores the mean of its top-5 window probabilities over 12 evenly spaced windows, a patient the mean over
their recordings. 107 patients (53 epilepsy).
"""


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def auroc(scores, labels):
    # Mann-Whitney with ties at their average rank
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    n = sum(label for _, label in pairs)
    if n in (0, len(pairs)):
        return ""
    return round((sum(r for r, (_, label) in zip(ranks, pairs) if label) - n * (n + 1) / 2) / (n * (len(pairs) - n)), 4)


def rates(p, y):
    tp = sum(pi >= 0.5 and yi for pi, yi in zip(p, y))
    fp = sum(pi >= 0.5 and not yi for pi, yi in zip(p, y))
    fn = sum(pi < 0.5 and yi for pi, yi in zip(p, y))
    tn = len(y) - tp - fp - fn
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"balanced_accuracy": round((sens + spec) / 2, 4), "sensitivity": round(sens, 4),
            "specificity": round(spec, 4), "precision": round(prec, 4), "f1": round(f1, 4),
            "predicted_positive_fraction": round((tp + fp) / len(y), 4)}


def describe(arm):
    match = ARM.match(arm)
    if not match or match.group(2) not in MODELS:
        return None
    prefix, short, ds, seed, last = match.groups()
    if prefix == "g" and ds != "tusz":  # answer-first on the all-rationale corpora, replaced by the multitask gm- runs
        return None
    variant = "+".join(v for v, on in (("seed2", seed), ("last", last)) if on) or "main"
    return {"model": MODELS[short], "dataset": ds.upper(), "kind": KIND[prefix], "variant": variant,
            "arm": arm.removesuffix("-last")}


def scored(rows, positive_key):
    rows = list({r["path"]: r for r in rows if r["p_true"] != ""}.values())
    return rows, [float(r["p_true"]) for r in rows], [r[positive_key] == "true" for r in rows]


def main():
    out = RESULTS / "export"
    out.mkdir(exist_ok=True)
    metrics, predictions = [], []
    for path in sorted(RESULTS.glob("scores-*.csv")):
        info = describe(path.stem.removeprefix("scores-"))
        if not info:
            continue
        sets = [("test", path, f"summary-{path.stem.removeprefix('scores-')}")]
        for prefix, name in (("heldout", "heldout"), ("background", "interior-background")):
            extra = RESULTS / f"{prefix}-{path.stem.removeprefix('scores-')}.csv"
            if extra.exists():
                sets.append((name, extra, f"summary-{prefix}-{path.stem.removeprefix('scores-')}"))
        test_rows = None
        for name, source, summary_name in sets:
            rows, p, y = scored(read(source), "seizure")
            if name == "test":
                test_rows = (rows, p, y)
            for r, pi, yi in zip(rows, p, y):
                predictions.append({**{k: info[k] for k in ("model", "dataset", "kind", "variant")}, "set": name,
                                    "image": r["path"], "patient": Path(r["path"]).stem.split("_")[0],
                                    "label": str(yi).lower(), "p_true": r["p_true"], "predicted": str(pi >= 0.5).lower()})
            if name == "interior-background":  # scored against the test set's seizure windows
                seizures = [(r, pi) for r, pi, yi in zip(*test_rows) if yi]
                rows, p, y = [r for r, _ in seizures] + rows, [pi for _, pi in seizures] + p, [True] * len(seizures) + y
            summary = (read(RESULTS / f"{summary_name}.csv") or [{}])[0] if (RESULTS / f"{summary_name}.csv").exists() else {}
            low = summary.get("auroc_ci_low", "")
            high = summary.get("auroc_ci_high", "")
            metrics.append({**info, "question": QUESTION[info["dataset"]], "set": name, "windows": len(y),
                            "positives": sum(y), "patients": len({Path(r["path"]).stem.split("_")[0] for r in rows}),
                            "auroc": auroc(p, y), "auroc_ci_low": low, "auroc_ci_high": high, **rates(p, y)})

    rationales = []
    for path in sorted(RESULTS.glob("rationale-*.csv")):
        match = re.match(r"rationale-(.+)-(answer-first|explain|rationale-first)$", path.stem)
        info = describe(match.group(1)) if match else None
        if not info:
            continue
        for r in read(path):
            rationales.append({**{k: info[k] for k in ("model", "dataset", "kind", "variant")}, "format": match.group(2),
                               "image": r["path"], "label": r["seizure"], "generated_present": r["generated_present"],
                               "reference_nll": r["reference_nll"], "rationale": r["reply"]})

    tables.RESULTS = RESULTS
    _, header, rows = tables.grid()
    with (out / "summary_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    tuep = [s for path in sorted(RESULTS.glob("summary-tuep-scan-*.csv")) for s in read(path)]
    for name, rows in (("metrics", metrics), ("predictions", predictions), ("rationales", rationales), ("tuep_scan", tuep)):
        with (out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}.csv: {len(rows)} rows")
    (out / "README.md").write_text(README)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
