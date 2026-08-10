#!/usr/bin/env python3
"""Blank out degenerate rationales already written to datasets/<DS>/rationales.csv.

A wedged Ollama runner returns one token repeated for the whole num_predict budget
(a 512-char run of "?"). Those rows were saved as if they were real rationales, so
generate-rationales.py now treats them as complete and skips them. Blanking them
here makes the next run regenerate them.

Reuses is_degenerate() from generate-rationales.py so the two never disagree about
what counts as garbage. Run it with the venv active, from anywhere.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

GENERATE = Path(__file__).parent / "train" / "scripts" / "generate-rationales.py"

spec = importlib.util.spec_from_file_location("generate_rationales", GENERATE)
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)


def main():
    for dataset in generate.get_datasets():
        fields, rows = generate.load_rationales(dataset)
        scrubbed = 0
        for row in rows:
            text = row["ground_truth_rationales"].strip()
            if text and generate.is_degenerate(text):
                row["ground_truth_rationales"] = ""
                scrubbed += 1
        if scrubbed:
            generate.save_rationales(dataset, fields, rows)
        print(f"{dataset.name}: blanked {scrubbed} / {len(rows)}")


if __name__ == "__main__":
    main()
