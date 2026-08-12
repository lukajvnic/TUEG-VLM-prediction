#!/usr/bin/env python3
"""Blank out degenerate rationales already written to datasets/<DS>/rationales*.csv.

A wedged Ollama runner returns one token repeated for the whole num_predict budget
(a 512-char run of "?"). Those rows were saved as if they were real rationales, so
generate-rationales.py now treats them as complete and skips them. Blanking them
here makes the next run regenerate them.

Covers both splits: rationales.csv (fine-tune targets) and rationales-test.csv
(the reference rationales eval/agreement.py judges against). Garbage in the test
file is just as harmful -- it would be scored as a real reference.

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


def scrub(dataset, split):
    if not generate.rationales_path(dataset, split).exists():
        return
    fields, rows = generate.load_rationales(dataset, split)
    scrubbed = 0
    for row in rows:
        text = row[generate.RATIONALE_COLUMN].strip()
        if text and generate.is_degenerate(text):
            row[generate.RATIONALE_COLUMN] = ""
            scrubbed += 1
    if scrubbed:
        generate.save_rationales(dataset, fields, rows, split)
    print(f"{dataset.name} [{split}]: blanked {scrubbed} / {len(rows)}")


def main():
    for split in sorted(generate.RATIONALE_FILES):
        for dataset in generate.get_datasets():
            scrub(dataset, split)


if __name__ == "__main__":
    main()
