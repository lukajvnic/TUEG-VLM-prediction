import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import ROOT, base_spec, config


def checkpoint_root():
    # HF_CHECKPOINTS wins, so an existing snapshot dir can be pointed at without moving it
    if os.environ.get("HF_CHECKPOINTS"):
        return Path(os.environ["HF_CHECKPOINTS"])
    scratch = os.environ.get("SCRATCH")
    return Path(scratch) / "hf-checkpoints" if scratch else ROOT / "hf-checkpoints"


def snapshot_dir(repo_id, root=None):
    return (root or checkpoint_root()) / re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id)


def is_complete(repo_id, destination, token):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
    if not destination.is_dir():
        return False
    try:
        files = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=destination,
                                  token=token, dry_run=True)
    except (HfHubHTTPError, OSError):
        return False
    return bool(files) and all(not f.will_download for f in files)


def main():
    parser = argparse.ArgumentParser(
        description="download HF snapshots for the fine-tune (run from a login node: compute nodes have no internet)")
    parser.add_argument("repos", nargs="*", help="HF repo ids or `bases:` keys (default: config.yml train.model)")
    parser.add_argument("--all", action="store_true", help="every base in config.yml bases: without a `status`")
    parser.add_argument("--output-dir", type=Path, default=checkpoint_root())
    parser.add_argument("--dry-run", action="store_true", help="print destinations, download nothing")
    parser.add_argument("--token", help="HF token (else `hf auth login` credentials or HF_TOKEN)")
    args = parser.parse_args()
    cfg = config()
    keys = [k for k, b in cfg.get("bases", {}).items() if not b.get("status")] if args.all else \
        (args.repos or [cfg["train"]["model"]])
    repos = [base_spec(cfg, k)["repo"] for k in keys]

    for repo_id in repos:
        print(f"{repo_id} -> {snapshot_dir(repo_id, args.output_dir)}")
    if args.dry_run:
        return
    from huggingface_hub import snapshot_download
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for repo_id in repos:
        destination = snapshot_dir(repo_id, args.output_dir)
        if is_complete(repo_id, destination, args.token):
            print(f"complete, skipping {repo_id}", flush=True)
            continue
        try:
            snapshot_download(repo_id=repo_id, repo_type="model", local_dir=destination,
                              token=args.token, max_workers=8)
        except Exception as e:  # gated repo without licence acceptance, network, disk
            failed.append(repo_id)
            print(f"FAILED {repo_id}: {e}", file=sys.stderr)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
