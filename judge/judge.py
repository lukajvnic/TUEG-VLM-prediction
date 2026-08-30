import json
import os
import sys
import time
from base64 import b64decode, b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pipeline import DATASETS, RATIONALE, ROOT, append_row, config, log_failure, read_csv, submit_array, sync

TRANSIENT = (ConnectionError, TimeoutError, OSError)
RETRY_SLEEP = 10
PROGRESS_EVERY = 100
HEADER = ["path", "model", "correct_predictions", "correct_rationale", "correct_rationale_reason"]

JUDGE_PROMPT = """Two descriptions of the SAME multi-channel EEG waveform plot.

REFERENCE -- written by a reader who was told the correct label:
{reference}

CANDIDATE -- written by a model that was not told the label:
{candidate}

Judge only what the two texts actually say.

same_conclusion: does the candidate reach the same overall assessment as the reference -- the same finding(s) reported present, and none of the reference's findings denied? Ignore wording, ordering and hedging.

same_evidence: does the candidate point at the same waveform evidence -- the same channel(s), the same region of the time axis, and the same morphology (shape, frequency, amplitude)? Overlap on the reference's main finding counts as true. A different channel, a different part of the recording, or a different morphology counts as false.

A confident tone is not correctness. A generic description that would fit any EEG recording is not matching evidence -- mark same_evidence false for it."""


class AgreementVerdict(BaseModel):
    same_conclusion: bool = Field(description="True if both texts report the same finding(s) present or absent.")
    same_evidence: bool = Field(description="True if both texts cite the same channels, time region and morphology.")
    reason: str = Field(description="At most 15 words.")


def pairs(dataset, model):
    folder = ROOT / "datasets" / dataset
    truths = {r["path"]: r for r in read_csv(folder / "labels.csv")}
    done = {r["path"] for r in read_csv(folder / "judge-baseline.csv") if r["model"] == model}
    return [(row, truths[row["path"]])
            for row in read_csv(folder / "eval-baseline.csv")
            if row["model"] == model and row["path"] not in done
            and row["path"] in truths and truths[row["path"]][RATIONALE].strip()]


def correct_predictions(eval_row, truth):
    cols = [c for c in truth if c not in ("path", RATIONALE)]
    return all(eval_row[c].strip().lower() == truth[c].strip().lower() for c in cols)


def judge_pair(llm, eval_row, truth, max_chars, retries):
    candidate = eval_row["rationale"].strip()[:max_chars]
    if not candidate:
        return False, "empty candidate rationale"
    text = JUDGE_PROMPT.format(reference=truth[RATIONALE].strip()[:max_chars], candidate=candidate)
    for attempt in range(retries + 1):
        try:
            result = llm.invoke(text)
            if result["parsed"] is None:
                raise ValueError(str(result.get("parsing_error")))
            verdict = result["parsed"]
            return verdict.same_conclusion and verdict.same_evidence, verdict.reason
        except Exception as e:
            error = e
            if attempt < retries and isinstance(e, TRANSIENT):
                time.sleep(RETRY_SLEEP)
    raise error


def judge_model(model):
    from langchain_ollama import ChatOllama
    cfg = config()
    jc = cfg["judge"]
    llm = ChatOllama(model=jc["model"], base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                     num_ctx=jc["num-ctx"], num_predict=jc["num-predict"], temperature=0,
                     ).with_structured_output(AgreementVerdict, include_raw=True,
                                              **cfg["settings"]["structured-output"])
    for dataset in DATASETS:
        todo = pairs(dataset, model)
        if not todo:
            continue
        out = ROOT / "datasets" / dataset / "judge-baseline.csv"
        ok = failed = 0
        with ThreadPoolExecutor(max_workers=jc["parallel-requests"]) as pool:
            futures = {pool.submit(judge_pair, llm, e, t, jc["max-rationale-chars"],
                                   jc["judge-retries"]): (e, t) for e, t in todo}
            for future in as_completed(futures):
                eval_row, truth = futures[future]
                try:
                    correct_rationale, reason = future.result()
                except Exception as e:
                    failed += 1
                    print(f"failed {dataset}/{eval_row['path']}: {e}", file=sys.stderr, flush=True)
                    log_failure("judge", dataset, eval_row["path"], model, e)
                    continue
                append_row(out, HEADER, [eval_row["path"], model,
                                         str(correct_predictions(eval_row, truth)).lower(),
                                         str(correct_rationale).lower(), reason])
                ok += 1
                if ok % PROGRESS_EVERY == 0:
                    print(f"{dataset}: {ok}", flush=True)
        print(f"{model} {dataset}: {ok} ok, {failed} failed", flush=True)


def main():
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        models = json.loads(b64decode(os.environ["JUDGE_MODELS"]))
        judge_model(models[int(os.environ["SLURM_ARRAY_TASK_ID"])])
        return
    conn = sync()
    models = [m for (m,) in conn.execute(
        "SELECT DISTINCT model FROM pipeline WHERE evaled = 1 AND rationale = 1 AND judged = 0 "
        "ORDER BY model")]
    if not models:
        print("nothing to judge")
        return
    jc = config()["judge"]
    payload = b64encode(json.dumps(models).encode()).decode()
    out = submit_array("eeg-vlm-judge", jc["time"], jc["ram"], jc["gpus"], len(models) - 1,
                       jc["array-concurrency"], jc["parallel-requests"],
                       f"python {ROOT}/judge/judge.py", {"JUDGE_MODELS": payload})
    print(f"{out} - {len(models)} models")


if __name__ == "__main__":
    main()
