import json
import os
import sys
import time
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers.pipeline import ROOT, append_row, config, db, log_failure, read_csv
from helpers.pipeline import image_message
from structure import get_structure, labels, prompt, to_labels

TRANSIENT = (ConnectionError, TimeoutError, OSError)
RETRY_SLEEP = 10
PROGRESS_EVERY = 50


def task():
    tasks = json.loads(b64decode(os.environ["EVAL_TASKS"]))
    return tasks[int(os.environ["SLURM_ARRAY_TASK_ID"])]


def pending(dataset, model, out):
    done = {r["path"] for r in read_csv(out) if r["model"] == model}
    rows = db().execute(
        "SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'full'", (dataset,))
    return [rel for (p,) in rows if (rel := p.split("/", 1)[1]) not in done]


def init_model(model, dataset, settings):
    from langchain_ollama import ChatOllama
    kwargs = dict(settings["model-kwargs"])
    kwargs["base_url"] = os.environ.get("OLLAMA_BASE_URL", kwargs.get("base_url"))
    return ChatOllama(model=model, **kwargs).with_structured_output(
        get_structure(dataset), include_raw=True, **settings["structured-output"])


def evaluate(llm, image, text, retries):
    for attempt in range(retries + 1):
        try:
            result = llm.invoke([image_message(image, text)])
            if result["parsed"] is None:
                raise ValueError(str(result.get("parsing_error")))
            return result["parsed"]
        except Exception as e:
            error = e
            if attempt < retries and isinstance(e, TRANSIENT):
                time.sleep(RETRY_SLEEP)
    raise error


def main():
    settings = config()["settings"]
    model, dataset = task()["model"], task()["dataset"]
    folder = ROOT / "datasets" / dataset
    out = folder / "eval-baseline.csv"
    header = ["path", "model", *labels(dataset), "rationale"]
    todo = pending(dataset, model, out)
    llm = init_model(model, dataset, settings)
    text = prompt(dataset)
    ok = failed = 0
    with ThreadPoolExecutor(max_workers=task()["parallel"]) as pool:
        futures = {pool.submit(evaluate, llm, folder / rel, text, settings["eval-retries"]): rel
                   for rel in todo}
        for future in as_completed(futures):
            rel = futures[future]
            try:
                parsed = future.result()
            except Exception as e:
                failed += 1
                print(f"failed {rel}: {e}", file=sys.stderr)
                log_failure("eval", dataset, rel, model, e)
                continue
            values = to_labels(parsed, dataset)
            append_row(out, header, [rel, model, *[str(v).lower() for v in values.values()],
                                     parsed.text_rationale])
            ok += 1
            if ok % PROGRESS_EVERY == 0:
                print(f"{ok}/{len(todo)}", flush=True)
    print(f"{model} {dataset}: {ok} ok, {failed} failed, {len(todo)} attempted")


if __name__ == "__main__":
    main()
