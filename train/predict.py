import json
import os
import re
import sys
import time
from base64 import b64decode
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers.pipeline import ROOT, append_row, base_spec, config, db, eval_file, is_degenerate, log_failure, read_csv
from structure import get_structure, labels, prompt, to_labels

snapshot_dir = import_module("hf-install").snapshot_dir

MAX_NEW_TOKENS = 512
RETRY_TEMPERATURE = 0.3
PROGRESS_EVERY = 50
# module-name fragments that mark the image side of a VLM (Qwen `visual`, SigLIP/CLIP `vision_tower`, projectors)
MINICPM_SYSTEM = "You are a helpful assistant."  # LLaMA-Factory template minicpm_v default_system
VISION_HINTS = ["visual", "vision", "multi_modal_projector", "mm_projector", "merger", "image_newline"]
SKIP_HINTS = ["audio"]  # gemma4's audio tower and embed_audio: neither language nor vision, left frozen
# gemma4 26B/31B: the generation prompt ends with an empty thinking channel that the rendered assistant turn
# never contains; the collator splices it in so the trained target starts exactly where generation starts
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"
_TEMPLATE_KWARGS = {}  # id(processor) -> `bases:` template-kwargs (e.g. enable_thinking: false)


def load_processor(source, base, remote):
    # `bases:` may carry processor-kwargs (llava repos without processor_config.json need patch_size etc.),
    # a chat-template (repos that ship none) and template-kwargs (thinking switches), applied identically by
    # the trainer and by scoring
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=remote, **base.get("processor-kwargs", {}))
    if base.get("chat-template"):
        processor.chat_template = base["chat-template"]  # saved with the checkpoint by processor.save_pretrained
    _TEMPLATE_KWARGS[id(processor)] = base.get("template-kwargs", {})
    return processor


def render(processor, messages, add_generation_prompt):
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         **_TEMPLATE_KWARGS.get(id(processor), {}))


def encode_kwargs(processor, text):
    # apply_chat_template(tokenize=False) already wrote the template's BOS (gemma, mistral); tokenising that text
    # with add_special_tokens would prepend a second one
    bos = getattr(getattr(processor, "tokenizer", None), "bos_token", None)
    return {"add_special_tokens": False} if bos and text.startswith(bos) else {}


def is_vision(name):
    return any(hint in name for hint in VISION_HINTS)


def lora_targets(model):
    # every Linear except the output head and embeddings, split language / vision by module path; the trainer
    # passes the full names so PEFT adapts exactly these, whatever the architecture calls them
    import torch
    language, vision = [], []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name.endswith("lm_head") or "embed" in name:
            continue
        if any(hint in name for hint in SKIP_HINTS):
            continue
        (vision if is_vision(name) else language).append(name)
    return language, vision


def task():
    tasks = json.loads(b64decode(os.environ["PREDICT_TASKS"]))
    return tasks[int(os.environ["SLURM_ARRAY_TASK_ID"])]


def pending(dataset, model, out):
    done = {r["path"] for r in read_csv(out) if r["model"] == model}
    rows = db().execute("SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND model = ? AND scope = 'full'",
                        (dataset, model))
    todo = [rel for (p,) in rows if (rel := p.split("/", 1)[1]) not in done]
    assert all(rel.startswith("test/") for rel in todo), "non-test window in eval scope"
    return todo


class CustomFamily:
    """Bases whose model code is their own (config.yml bases: `family`): loaded and prompted through their API.
    No logits access from those APIs, so decoding is unconstrained; the JSON is parsed and retried like the rest.
    Written from the model cards (2026-09-21), unverified on a GPU."""

    def __init__(self, name, source, checkpoint):
        import torch
        self.name = name
        if name == "minicpm":  # MiniCPM-V 2.6 / 4.5: AutoModel + model.chat(image=, msgs=, tokenizer=)
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            model = AutoModel.from_pretrained(source, trust_remote_code=True, dtype=torch.bfloat16,
                                              attn_implementation="sdpa")
        elif name == "moondream":  # moondream2: AutoModelForCausalLM + model.query(image, question)
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True, device_map={"": "cuda"})
        elif name == "deepseek-ocr":  # DeepSeek-OCR: AutoModel + model.infer(tokenizer, prompt=, image_file=)
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            model = AutoModel.from_pretrained(source, trust_remote_code=True, use_safetensors=True,
                                              dtype=torch.bfloat16)
        else:
            raise ValueError(f"unknown custom family {name!r}")
        if checkpoint is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(checkpoint), trust_remote_code=True).merge_and_unload()
        self.model = model.eval().cuda()

    def generate(self, image_path, text, temperature=None):
        from PIL import Image
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.name == "minicpm":
                return str(self.model.chat(image=None, msgs=[{"role": "user", "content": [image, text]}],
                                           tokenizer=self.tokenizer, sampling=bool(temperature),
                                           system_prompt=MINICPM_SYSTEM,  # what LLaMA-Factory's minicpm_v template trained with
                                           temperature=temperature or 0.0, max_new_tokens=MAX_NEW_TOKENS)).strip()
            if self.name == "moondream":
                return str(self.model.query(image, text)["answer"]).strip()
        if self.name == "deepseek-ocr":
            import tempfile
            # infer() makedirs output_path unconditionally and only returns the text with eval_mode=True
            # (otherwise it streams to stdout and returns None); save_results=False keeps the dir empty
            out = self.model.infer(self.tokenizer, prompt=f"<image>\n{text}", image_file=str(image_path),
                                   output_path=tempfile.gettempdir(), base_size=1024, image_size=640,
                                   crop_mode=True, save_results=False, eval_mode=True)
            return str(out).strip()
        raise ValueError(self.name)


def load(spec):
    # a spec without `checkpoint` is the base model as-is: the like-for-like "before" for its fine-tunes.
    # `base` is a `bases:` key (Ollama tag) or an HF repo id; any architecture AutoModelForImageTextToText knows
    import torch
    from transformers import AutoModelForImageTextToText
    base = base_spec(config(), spec["base"])
    local = snapshot_dir(base["repo"])
    source = str(local) if (local / "config.json").exists() else base["repo"]
    remote = base.get("trust-remote-code", False)
    if base.get("family"):
        checkpoint = ROOT / spec["checkpoint"] if "checkpoint" in spec else None
        manifest = json.loads((checkpoint / "manifest.json").read_text()) if checkpoint else \
            {"rationale_first": spec.get("rationale-first", False)}
        print(f"loading {source} via custom family {base['family']}" + (f" + {checkpoint}" if checkpoint else ""), flush=True)
        return CustomFamily(base["family"], source, checkpoint), None, manifest
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="auto",
                                                        trust_remote_code=remote)
    if "checkpoint" not in spec:
        print(f"loading {source} (no adapter)", flush=True)
        model.eval()
        return model, load_processor(source, base, remote), {"rationale_first": spec.get("rationale-first", False)}
    from peft import PeftModel
    checkpoint = ROOT / spec["checkpoint"]
    if not (checkpoint / "manifest.json").exists():
        sys.exit(f"{checkpoint} has no manifest.json - training did not finish")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    print(f"loading {source} + {checkpoint}", flush=True)
    processor = load_processor(str(checkpoint), base, remote)
    model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload()
    model.eval()
    return model, processor, manifest


_TOKENIZER_DATA = {}


def enforcer(processor, structure):
    # same schema Ollama's json mode constrains the zero-shot models to. The vocabulary table is built once per
    # tokenizer (it decodes every token id); a pooled run builds six enforcers from it
    try:
        import transformers.tokenization_utils  # noqa: F401  slow-tokenizer module, removed in transformers 5
    except ImportError:
        # lm-format-enforcer (0.11.3, 2026-09-25) still imports PreTrainedTokenizerBase from there and reports the
        # failure as "transformers is not installed"; it only needs the class, which lives in tokenization_utils_base
        import transformers.tokenization_utils_base as tokenization_utils
        sys.modules["transformers.tokenization_utils"] = tokenization_utils
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (build_token_enforcer_tokenizer_data,
                                                             build_transformers_prefix_allowed_tokens_fn)
    tokenizer = processor.tokenizer
    if id(tokenizer) not in _TOKENIZER_DATA:
        _TOKENIZER_DATA[id(tokenizer)] = build_token_enforcer_tokenizer_data(tokenizer)
    return build_transformers_prefix_allowed_tokens_fn(_TOKENIZER_DATA[id(tokenizer)],
                                                       JsonSchemaParser(structure.model_json_schema()))


def generate(model, processor, image_path, text, prefix_fn, temperature=None):
    import torch
    from PIL import Image
    if isinstance(model, CustomFamily):
        return model.generate(image_path, text, temperature)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = render(processor, messages, True)
    with Image.open(image_path) as image:
        inputs = processor(text=[chat], images=[image.convert("RGB")], return_tensors="pt",
                           **encode_kwargs(processor, chat)).to(model.device)
    sampling = dict(do_sample=True, temperature=temperature) if temperature else dict(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, prefix_allowed_tokens_fn=prefix_fn,
                             use_cache=True, **sampling)
    reply = out[0, inputs["input_ids"].shape[1]:]
    return processor.batch_decode([reply], skip_special_tokens=True)[0].strip()


def parse_json(reply):
    # the enforcer makes generic-path replies valid JSON; custom families return free text, so strip code fences
    # and take the first balanced object before giving up (a preamble or trailing sentence is not a failure)
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", reply.strip(), flags=re.M)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {reply[:80]!r}")
    return json.JSONDecoder().raw_decode(cleaned[start:])[0]


def predict(model, processor, structure, image_path, text, prefix_fn):
    error = None
    for temperature in (None, RETRY_TEMPERATURE):  # greedy first; a truncated or looping reply gets one sampled retry
        try:
            reply = generate(model, processor, image_path, text, prefix_fn, temperature)
            parsed = structure(**parse_json(reply))
            if is_degenerate(parsed.text_rationale):
                raise ValueError(f"degenerate rationale: {parsed.text_rationale[:60]!r}")
            return parsed
        except Exception as e:
            error = e
    raise error


def main():
    name, dataset = task()["model"], task()["dataset"]
    spec = config()["models"][name]
    folder = ROOT / "datasets" / dataset
    out = eval_file(folder, name, spec)  # datasets/<DS>/eval-<model>.csv for hf models
    header = ["path", "model", *labels(dataset), "rationale"]
    todo = pending(dataset, name, out)
    if not todo:
        print(f"{name} {dataset}: nothing pending")
        return
    model, processor, manifest = load(spec)
    structure = get_structure(dataset, manifest.get("rationale_first", False))  # the order the checkpoint was trained on
    text = prompt(dataset)
    prefix_fn = enforcer(processor, structure) if processor is not None else None  # custom families: unconstrained
    ok = failed = 0
    started = time.time()
    for rel in todo:
        try:
            parsed = predict(model, processor, structure, folder / rel, text, prefix_fn)
        except Exception as e:
            failed += 1
            print(f"failed {rel}: {e}", file=sys.stderr, flush=True)
            log_failure("predict", dataset, rel, name, e)
            continue
        values = to_labels(parsed, dataset)
        append_row(out, header, [rel, name, *[str(v).lower() for v in values.values()], parsed.text_rationale])
        ok += 1
        if ok % PROGRESS_EVERY == 0:
            print(f"{ok}/{len(todo)} ({(time.time() - started) / ok:.1f} s/image)", flush=True)
    print(f"{name} {dataset}: {ok} ok, {failed} failed, {len(todo)} attempted")


if __name__ == "__main__":
    main()
