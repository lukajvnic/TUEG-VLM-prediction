import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers.pipeline import DATASETS, ROOT, append_row, config, log_failure, read_csv
from judge import HEADER, JUDGE_PROMPT, AgreementVerdict, correct_predictions, pairs

MODEL = "gpt-5.6-luna"
OUT_NAME = "judge-gpt.csv"
REASONING_EFFORT = "low"
PARALLEL = 16
RETRIES = 3
RETRY_SLEEP = 15
PROGRESS_EVERY = 200


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("OPENAI_API_KEY not found in .env")


def judge_pair(client, eval_row, truth, max_chars):
    candidate = eval_row["rationale"].strip()[:max_chars]
    if not candidate:
        return False, "empty candidate rationale"
    text = JUDGE_PROMPT.format(reference=truth["ground_truth_rationale"].strip()[:max_chars],
                               candidate=candidate)
    for attempt in range(RETRIES + 1):
        try:
            response = client.chat.completions.parse(
                model=MODEL, reasoning_effort=REASONING_EFFORT,
                messages=[{"role": "user", "content": text}],
                response_format=AgreementVerdict)
            verdict = response.choices[0].message.parsed
            return verdict.same_conclusion and verdict.same_evidence, verdict.reason
        except Exception as e:
            error = e
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP * (attempt + 1))
    raise error


def main():
    from openai import OpenAI
    client = OpenAI(api_key=load_key())
    max_chars = config()["judge"]["max-rationale-chars"]
    models = sorted({r["model"] for ds in DATASETS
                     for r in read_csv(ROOT / "datasets" / ds / "eval-baseline.csv")})
    for dataset in DATASETS:
        out = ROOT / "datasets" / dataset / OUT_NAME
        todo = [(e, t) for model in models for e, t in pairs(dataset, model, OUT_NAME)]
        print(f"{dataset}: {len(todo)} pairs", flush=True)
        ok = failed = 0
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            futures = {pool.submit(judge_pair, client, e, t, max_chars): (e, t) for e, t in todo}
            for future in as_completed(futures):
                eval_row, truth = futures[future]
                try:
                    correct_rationale, reason = future.result()
                except Exception as e:
                    failed += 1
                    print(f"failed {dataset}/{eval_row['path']} {eval_row['model']}: {e}",
                          file=sys.stderr, flush=True)
                    log_failure("judge-gpt", dataset, eval_row["path"], eval_row["model"], e)
                    continue
                append_row(out, HEADER, [eval_row["path"], eval_row["model"],
                                         str(correct_predictions(eval_row, truth)).lower(),
                                         str(correct_rationale).lower(), reason])
                ok += 1
                if ok % PROGRESS_EVERY == 0:
                    print(f"{dataset}: {ok}/{len(todo)}", flush=True)
        print(f"{dataset}: {ok} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
