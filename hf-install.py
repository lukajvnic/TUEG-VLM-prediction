#!/usr/bin/env python3
"""Download the Hugging Face equivalents listed in HF_REPOS below.

Downloads complete repository snapshots, including model weights, processor, and
chat-template files. Models requiring acceptance of a license must be approved
on Hugging Face before this script can download them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Ollama model tag -> canonical Transformers/Hugging Face checkpoint.
# None means there is currently no corresponding public HF checkpoint.
HF_REPOS = {
    "bakllava:7b": "llava-hf/bakLlava-v1-hf",
    "deepseek-ocr:3b": "deepseek-ai/DeepSeek-OCR",
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma3:12b": "google/gemma-3-12b-it",
    "gemma3:27b": "google/gemma-3-27b-it",
    "gemma4:e2b": "google/gemma-4-E2B-it",
    "gemma4:e4b": "google/gemma-4-E4B-it",
    "gemma4:12b": "google/gemma-4-12B-it",
    "gemma4:26b": "google/gemma-4-26B-A4B-it",
    "gemma4:31b": "google/gemma-4-31B-it",
    "glm-ocr:latest": "zai-org/GLM-OCR",
    "granite3.2-vision:2b": "ibm-granite/granite-vision-3.2-2b",
    "llama4:scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama4:16x17b": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama4:128x17b": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "llava:7b": "llava-hf/llava-1.5-7b-hf",
    "llava:13b": "llava-hf/llava-1.5-13b-hf",
    "llava:34b": "liuhaotian/llava-v1.5-34b",
    "llava:v1.6": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-llama3:8b": "xtuner/llava-llama-3-8b-v1_1-transformers",
    "llava-phi3:3.8b": "microsoft/Phi-3-vision-128k-instruct",
    "medgemma:4b": "google/medgemma-4b-it",
    "medgemma1.5:4b": "google/medgemma-1.5-4b-it",
    "minicpm-v:8b": "openbmb/MiniCPM-V-2_6",
    "minicpm-v4.5:8b": "openbmb/MiniCPM-V-4_5",
    "minicpm-v4.6:1b": "openbmb/MiniCPM-V-4_6",
    "mistral-small3.1:24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistral-small3.2:24b": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "moondream:1.8b": "vikhyatk/moondream2",
    "moondream:v2": "vikhyatk/moondream2",
    "qwen2.5vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5vl:32b": "Qwen/Qwen2.5-VL-32B-Instruct",
}
for size in ("2B", "4B", "8B", "30B-A3B", "32B", "235B-A22B"):
    for mode in ("Instruct", "Thinking"):
        ollama_size = size.lower().replace("b", "b")
        suffix = "-instruct" if mode == "Instruct" else "-thinking"
        HF_REPOS[f"qwen3-vl:{ollama_size}{suffix}"] = f"Qwen/Qwen3-VL-{size}-{mode}"


def safe_dirname(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id)


def is_complete(repo_id: str, destination: Path, token: str | None) -> bool:
    """Check whether every file in the current HF revision is already local."""
    if not destination.is_dir():
        return False
    try:
        files = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=destination,
            token=token,
            dry_run=True,
        )
    except (HfHubHTTPError, OSError):
        return False
    return bool(files) and all(not file.will_download for file in files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("hf-checkpoints"))
    parser.add_argument("--dry-run", action="store_true", help="Print resolved repositories only")
    parser.add_argument(
        "--pause-between",
        action="store_true",
        help="Wait for Enter after each repository before starting the next download",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Check existing checkpoints again instead of skipping complete ones",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--token", help="HF token (otherwise uses `hf auth login` credentials or HF_TOKEN)")
    args = parser.parse_args()

    unavailable = [name for name, repo_id in HF_REPOS.items() if repo_id is None]
    if unavailable:
        print("No public HF equivalent configured for: " + ", ".join(unavailable), file=sys.stderr)

    # Preserve dictionary order while downloading an identical checkpoint only once.
    repos = list(dict.fromkeys(repo_id for repo_id in HF_REPOS.values() if repo_id))
    for repo_id in repos:
        destination = args.output_dir / safe_dirname(repo_id)
        print(f"{repo_id} -> {destination}")
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    failed = False
    for index, repo_id in enumerate(repos, start=1):
        destination = args.output_dir / safe_dirname(repo_id)
        if args.resume and is_complete(repo_id, destination, args.token):
            print(f"\n[{index}/{len(repos)}] Already complete; skipping {repo_id}", flush=True)
            continue

        try:
            print(f"\n[{index}/{len(repos)}] Downloading {repo_id}", flush=True)
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=destination,
                token=args.token,
                max_workers=8,
            )
            results[repo_id] = str(path)
        except (HfHubHTTPError, OSError) as error:
            failed = True
            print(f"FAILED {repo_id}: {error}", file=sys.stderr)

        if args.pause_between and index < len(repos):
            input("Press Enter to download the next model (Ctrl-C to stop): ")

    (args.output_dir / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
