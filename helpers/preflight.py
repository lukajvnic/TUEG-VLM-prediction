"""Preflight one cluster venv without a GPU: imports, versions, and every base's processor + chat template.

  python helpers/preflight.py main          # .venv: generic trainer + predict; every `bases:` entry without a family
  python helpers/preflight.py llamafactory  # .venv-llamafactory: MiniCPM-V 2.6 / 4.5 remote code + LLaMA-Factory bounds
  python helpers/preflight.py unsloth       # .venv-unsloth: DeepSeek-OCR remote code + unsloth

Loads configs, tokenizers and processors from the staged snapshots (no weights), renders the training template
the way the collator does, tokenises it, and builds one schema enforcer, so a missing package, a removed
transformers symbol, an unstaged snapshot or a broken template fails here on the login node, not on a compute node
after an hour in the queue (2026-09-26, after three batches died on exactly those). Exit 1 on any failure.
"""
import argparse
import sys
import traceback
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ("", "train", "eval", "eval/models"):
    sys.path.insert(0, str(ROOT / folder))
from helpers.pipeline import base_spec, config  # noqa: E402

failures = []


def check(label, fn):
    try:
        result = fn()
        print(f"ok    {label}" + (f": {result}" if result not in (None, True) else ""))
        return result
    except Exception as e:  # report and carry on so one run lists everything
        failures.append(label)
        print(f"FAIL  {label}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        if "-v" in sys.argv:
            traceback.print_exc()
        return None


def version(name):
    from importlib.metadata import version as v
    return v(name)


def imports(names):
    for name in names:
        check(f"import {name}", lambda n=name: import_module(n) and version_of(n))


def version_of(module):
    try:
        return version({"PIL": "Pillow", "yaml": "PyYAML", "lmformatenforcer": "lm-format-enforcer",
                        "cut_cross_entropy": "cut-cross-entropy", "google.protobuf": "protobuf"}.get(module, module))
    except Exception:
        return ""


def snapshot(base):
    snapshot_dir = import_module("hf-install").snapshot_dir
    local = snapshot_dir(base["repo"])
    if not (local / "config.json").exists():
        raise FileNotFoundError(f"{local} not staged: python train/hf-install.py <key>")
    return local


def remote_class(local, auto_map_key):
    # imports the repo's modeling/tokenizer module the way from_pretrained would, weights untouched
    import json
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = json.loads((local / "config.json").read_text())
    ref = cfg["auto_map"][auto_map_key]
    return get_class_from_dynamic_module(ref, str(local)).__name__


def generic_base(key, base, runner):
    local = check(f"{key}: snapshot", lambda: snapshot(base))
    if local is None:
        return
    from transformers import AutoConfig
    check(f"{key}: config", lambda: type(AutoConfig.from_pretrained(local, trust_remote_code=base.get("trust-remote-code", False))).__name__)
    processor = check(f"{key}: processor", lambda: runner.load_processor(str(local), base, base.get("trust-remote-code", False)))
    if processor is None:
        return

    def template():
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q?"}]}]
        full = msgs + [{"role": "assistant", "content": [{"type": "text", "text": '{"a": 1}' + base.get("answer-suffix", "")}]}]
        prompt = runner.render(processor, msgs, True)
        text = runner.render(processor, full, False)
        empty = runner.EMPTY_THOUGHT
        if not text.startswith(prompt) and prompt.endswith(empty) and text.startswith(prompt[:-len(empty)]):
            text = prompt + text[len(prompt) - len(empty):]
        if not text.startswith(prompt):
            raise ValueError(f"full turn does not start with the generation prompt: {prompt[-40:]!r} vs {text[-60:]!r}")
        ids = processor.tokenizer(text, **runner.encode_kwargs(processor, text))["input_ids"]
        return f"target {text[len(prompt):]!r}, {len(ids)} text tokens"
    check(f"{key}: template", template)
    return processor


def main_venv():
    imports(["torch", "torchvision", "transformers", "peft", "accelerate", "PIL", "yaml", "pydantic",
             "sentencepiece", "google.protobuf", "tiktoken", "lmformatenforcer"])
    check("torch.cuda (False on a login node is fine)", lambda: import_module("torch").cuda.is_available())
    check("bitsandbytes (only 4-bit bases need it)", lambda: version_of("bitsandbytes"))
    runner = import_module("predict")
    from structure import get_structure
    cfg = config()
    first = None
    for key, base in cfg["bases"].items():
        if base.get("status") or base.get("family"):
            continue
        processor = generic_base(key, base, runner)
        first = first or processor
    # the enforcer's transformers integration reports any failed import as "transformers is not installed";
    # import each name it needs separately so the real one shows, after the shim predict.enforcer() installs
    check("predict.py carries the tokenization_utils shim",
          lambda: "tokenization_utils_base" in (ROOT / "train/predict.py").read_text() or (_ for _ in ()).throw(RuntimeError("not pulled")))
    runner.shim_tokenization_utils()
    for mod, name in [("transformers", "AutoModelForCausalLM"), ("transformers.generation.logits_process", "LogitsProcessor"),
                      ("transformers.generation.logits_process", "PrefixConstrainedLogitsProcessor"),
                      ("transformers.tokenization_utils", "PreTrainedTokenizerBase"), ("lmformatenforcer", "JsonSchemaParser"),
                      ("lmformatenforcer.integrations.transformers", "build_transformers_prefix_allowed_tokens_fn")]:
        check(f"enforcer import {mod}.{name}", lambda m=mod, n=name: getattr(import_module(m), n) and "")
    if first is not None:
        check("lm-format-enforcer under this transformers", lambda: runner.enforcer(first, get_structure("TUAB", True)) and "built")
    check("datasets/pooled/sft_train.jsonl", lambda: (ROOT / "datasets/pooled/sft_train.jsonl").stat().st_size)


def bounded(name, low, high):
    # what LLaMA-Factory's require_version does; Alliance's `+computecanada` local tag sorts above the bare version
    from packaging.version import Version
    v = Version(version(name))
    if not (Version(low) <= v <= Version(high)):
        raise RuntimeError(f"{name} {v} outside [{low}, {high}] (a +computecanada tag counts as newer)")
    return str(v)


def custom_base(key, base):
    local = check(f"{key}: snapshot", lambda: snapshot(base))
    if local is None:
        return
    from transformers import AutoConfig, AutoTokenizer
    check(f"{key}: config", lambda: type(AutoConfig.from_pretrained(local, trust_remote_code=True)).__name__)
    check(f"{key}: remote modeling module imports", lambda: remote_class(local, "AutoModel"))
    check(f"{key}: tokenizer", lambda: type(AutoTokenizer.from_pretrained(local, trust_remote_code=True)).__name__)


def llamafactory_venv():
    imports(["torch", "torchvision", "transformers", "peft", "trl", "accelerate", "datasets", "llamafactory", "PIL", "yaml", "pydantic", "pyarrow"])
    check("transformers within LLaMA-Factory's bound", lambda: bounded("transformers", "4.55.0", "5.8.0"))
    check("datasets within LLaMA-Factory's bound", lambda: bounded("datasets", "2.16.0", "4.0.0"))
    cfg = config()
    for key, base in cfg["bases"].items():
        if base.get("family") == "minicpm":
            custom_base(key, base)
    check("datasets/pooled/export-llamafactory/dataset_info.json",
          lambda: (ROOT / "datasets/pooled/export-llamafactory/dataset_info.json").stat().st_size)


def unsloth_venv():
    imports(["torch", "torchvision", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "triton", "PIL", "yaml",
             "pydantic", "matplotlib", "einops", "addict", "easydict", "tqdm"])
    check("import unsloth (slow, compiles nothing)", lambda: version_of("unsloth") or import_module("unsloth").__version__)
    cfg = config()
    for key, base in cfg["bases"].items():
        if base.get("family") == "deepseek-ocr":
            custom_base(key, base)
    check("datasets/pooled/sft_train.jsonl", lambda: (ROOT / "datasets/pooled/sft_train.jsonl").stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("venv", choices=["main", "llamafactory", "unsloth"])
    parser.add_argument("-v", action="store_true", help="full tracebacks")
    args = parser.parse_args()
    print(f"python {sys.version.split()[0]} at {sys.prefix}")
    {"main": main_venv, "llamafactory": llamafactory_venv, "unsloth": unsloth_venv}[args.venv]()
    if failures:
        sys.exit(f"\n{len(failures)} failed: " + ", ".join(failures))
    print("\nall checks passed")


if __name__ == "__main__":
    main()
