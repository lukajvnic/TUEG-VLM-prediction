# the paper's tables from every results/summary-*.csv (run.py writes one per scored arm): the grid against its
# zero-shot baselines, TUSZ's final test, the rationale evaluation and the TUEP scan. Markdown to stdout and
# results/tables/tables.md, one CSV per table beside it. Runs wherever results/ is (the cluster, or a copy)
#   python side/vision-freeze/tables.py [RESULTS_DIR]
import csv
import sys
from pathlib import Path

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
MODELS = {"2b": "Qwen3-VL 2B", "4b": "Qwen3-VL 4B", "8b": "Qwen3-VL 8B", "q25-7b": "Qwen2.5-VL 7B",
          "gemma3-4b": "Gemma 3 4B", "medgemma-4b": "MedGemma 4B", "medgemma15-4b": "MedGemma 1.5 4B",
          "gemma4-e2b": "Gemma 4 E2B", "gemma4-e4b": "Gemma 4 E4B", "granite-2b": "Granite Vision 3.2 2B",
          "llava-llama3": "LLaVA-Llama3 8B", "gemma3-12b": "Gemma 3 12B", "llava-7b": "LLaVA 1.5 7B",
          "llava-phi3": "LLaVA-Phi3 3.8B", "gemma4-12b": "Gemma 4 12B", "minicpm46-1b": "MiniCPM-V 4.6 1B",
          "glm-ocr": "GLM-OCR"}
DATASETS = ["TUSZ", "TUAB", "TUAR", "TUEV"]


def summary(name):
    path = RESULTS / f"summary-{name}.csv"
    if not path.exists():
        return None
    with path.open() as f:
        return next(csv.DictReader(f), None)


def trained(short, ds):
    # TUSZ trains answer-first (g-), the all-rationale corpora in the multitask format (gm-)
    return f"{'g' if ds == 'TUSZ' else 'gm'}-{short}-{ds.lower()}"


def number(value, digits=3):
    return f"{float(value):.{digits}f}" if value not in (None, "") else ""


def with_ci(row, key="auroc", low="auroc_ci_low", high="auroc_ci_high"):
    return f"{number(row[key])} [{number(row[low])}, {number(row[high])}]" if row else ""


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def grid():
    # test AUROC with the 95% patient-bootstrap CI; "last" is the run's final checkpoint (run.py --last), "s2" a second
    # seed, where they exist
    header = ["model"] + [f"{ds} {kind}" for ds in DATASETS for kind in ("zero-shot", "fine-tuned")]
    rows = []
    for short, name in MODELS.items():
        row = [name]
        for ds in DATASETS:
            arm = trained(short, ds)
            cell = with_ci(summary(arm))
            extra = [f"{label} {number(s['auroc'])}" for label, s in (("last", summary(f"{arm}-last")),
                                                                     ("s2", summary(f"{arm}-s2"))) if s]
            row += [number((summary(f"z-{short}-{ds.lower()}") or {}).get("auroc")),
                    cell + (f" ({', '.join(extra)})" if extra else "")]
        if any(row[1:]):
            rows.append(row)
    return "Test AUROC [95% patient-bootstrap CI]", header, rows


def final_test():
    header = ["model", "kind", "test AUROC", "held-out AUROC", "held-out BA@0.5", "interior-background AUROC",
              "sensitivity", "specificity interior", "specificity 1-2 windows from seizure"]
    rows = []
    for short, name in MODELS.items():
        for kind, arm in (("fine-tuned", f"g-{short}-tusz"), ("zero-shot", f"z-{short}-tusz")):
            test, held, back = summary(arm), summary(f"heldout-{arm}"), summary(f"background-{arm}")
            if held or back:
                rows.append([name, kind, number((test or {}).get("auroc")), with_ci(held),
                             number((held or {}).get("balanced_accuracy_at_0.5")),
                             with_ci(back, "auroc_interior_background"), number((back or {}).get("sensitivity")),
                             number((back or {}).get("specificity_interior")),
                             number((back or {}).get("specificity_1_2_windows_from_seizure"))])
    return "TUSZ final test: 62 held-out patients, and 4,471 interior background windows", header, rows


def rationales():
    # reference NLL: mean per-token NLL of the teacher's (label-stripped) rationale under the model, lower = closer;
    # generated answer BA: the boolean of the model's own generated reply
    header = ["model", "dataset", "format", "reference NLL", "generated answer BA", "median chars", "at 1,200-char cap"]
    rows = []
    for short, name in MODELS.items():
        for ds in DATASETS:
            arm = trained(short, ds)
            fmt = "answer-first" if ds == "TUSZ" else "explain"
            s = summary(f"rationale-{arm}-{fmt}")
            if s:
                rows.append([name, ds, fmt, number(s["reference_nll"]), number(s.get("generated_answer_ba")),
                             s["rationale_chars_median"], s["rationale_at_cap"]])
    return "Rationale evaluation (300 stratified test windows)", header, rows


def tuep():
    header = ["detector", "window AUROC", "patient AUROC", "patient BA", "sensitivity", "specificity", "patients"]
    rows = []
    for path in sorted(RESULTS.glob("summary-tuep-scan-*.csv")):
        with path.open() as f:
            s = next(csv.DictReader(f))
        rows.append([s["arm"], number(s["auroc_window"]),
                     with_ci(s, "auroc_patient", "auroc_patient_ci_low", "auroc_patient_ci_high"),
                     number(s["balanced_accuracy_patient"]), number(s["sensitivity_patient"]),
                     number(s["specificity_patient"]), f"{s['patients']} ({s['epilepsy_patients']} epilepsy)"])
    return "TUEP patient-level scan (TUEV detector, top-5 of 12 windows per recording)", header, rows


def main():
    out = RESULTS / "tables"
    out.mkdir(exist_ok=True)
    text = []
    for build in (grid, final_test, rationales, tuep):
        title, header, rows = build()
        with (out / f"{build.__name__}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        text.append(f"## {title}\n\n" + (markdown(header, rows) if rows else "(no results yet)"))
    (out / "tables.md").write_text("\n\n".join(text) + "\n")
    print("\n\n".join(text))


if __name__ == "__main__":
    main()
