"""Re-export datasets/<DS or pooled>/sft_*.jsonl for the fine-tuning stacks the custom-code bases need.

The generic trainer (train/scripts/finetune_sample.py) covers every base transformers can load with a chat
template. Four cannot: MiniCPM-V 2.6 and 4.5, moondream2 and DeepSeek-OCR ship their own model code. Their
official or best-supported LoRA paths take a sharegpt-style JSON, which this writes from the same lines the
generic trainer trains on, so the data (windows, val split, rationale-first targets) is identical.

  --format llamafactory  data/<name>.json + a dataset_info.json entry (MiniCPM-V 4.5 via template minicpm_v,
                         also any generic base if LLaMA-Factory is preferred)
  --format minicpm       the OpenBMB finetune/ JSON (id, image, conversations) for MiniCPM-V 2.6's finetune_lora.sh
  --format unsloth       messages-with-image JSONL for Unsloth's generic vision trainer. Not used by
                         train/custom/unsloth_deepseek_ocr.py, which reads sft_*.jsonl itself (DeepSeek-OCR's
                         collator wants its own role tags); kept for any other base tried through Unsloth.

Image paths are absolute so the consumer can run from anywhere. Nothing here decides what is trained on.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ROOT

FORMATS = ("llamafactory", "minicpm", "unsloth")


def read_jsonl(path):
    if not path.exists():
        sys.exit(f"{path} missing - run helpers/build-sft-jsonl.py first")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def image_path(dataset, line):
    folder = ROOT / "datasets" if dataset == "pooled" else ROOT / "datasets" / dataset
    return str(folder / line["images"][0])


def llamafactory(dataset, lines):
    # "<image>\n": MiniCPM's chat() joins image and text with a newline, so score-time prompts match training
    return [{"conversations": [{"from": "human", "value": "<image>\n" + line["instruction"]},
                               {"from": "gpt", "value": line["output"]}],
             "images": [image_path(dataset, line)]} for line in lines]


def minicpm(dataset, lines):
    return [{"id": str(i), "image": image_path(dataset, line),
             "conversations": [{"role": "user", "content": "<image>\n" + line["instruction"]},
                               {"role": "assistant", "content": line["output"]}]}
            for i, line in enumerate(lines)]


def unsloth(dataset, lines):
    return [{"messages": [{"role": "user", "content": [{"type": "image", "image": image_path(dataset, line)},
                                                       {"type": "text", "text": line["instruction"]}]},
                          {"role": "assistant", "content": [{"type": "text", "text": line["output"]}]}]}
            for line in lines]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="a corpus name or `pooled`")
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default datasets/<dataset>/export-<format>/")
    args = parser.parse_args()
    folder = ROOT / "datasets" / args.dataset
    out = args.output_dir or folder / f"export-{args.format}"
    out.mkdir(parents=True, exist_ok=True)
    convert = {"llamafactory": llamafactory, "minicpm": minicpm, "unsloth": unsloth}[args.format]
    info = {}
    for split in ("train", "val"):
        lines = read_jsonl(folder / f"sft_{split}.jsonl")
        rows = convert(args.dataset, lines)
        name = f"eeg_{args.dataset}_{split}"
        if args.format == "unsloth":
            with (out / f"{name}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        else:
            (out / f"{name}.json").write_text(json.dumps(rows, indent=1))
        info[name] = {"file_name": f"{name}.json", "formatting": "sharegpt",
                      "columns": {"messages": "conversations", "images": "images"}}
        print(f"{name}: {len(rows)} rows -> {out}")
    if args.format == "llamafactory":
        (out / "dataset_info.json").write_text(json.dumps(info, indent=1))
        print(f"dataset_info.json: copy its entries into LLaMA-Factory's data/dataset_info.json (or point --dataset_dir here)")


if __name__ == "__main__":
    main()
