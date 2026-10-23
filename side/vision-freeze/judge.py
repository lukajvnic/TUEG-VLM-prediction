# the LLM judge for the vision-freeze rationale evaluations (2026-10-22): every results/rationale-<arm>-<fmt>.csv of a
# grid arm (g-*-tusz, gm-*, z-*, their -s2 seeds) is judged against the teacher's rationale for the same window,
# cleaned the way training cleaned it (run.py strip_label, CHANNEL), by the original benchmark's judge: judge/judge.py's
# prompt and config.yml judge: (mistral-small3.2:24b via Ollama, temperature 0, one sampled retry), so the numbers
# compare with the benchmark's. Both verdicts are kept. Control: 10% of each file's windows (seed 0) are judged a second
# time against the reference of a different window with the same label, which shows the agreement expected by chance
#   python side/vision-freeze/judge.py submit   # a smoke job, then the full job once the smoke passes (afterok)
#   python side/vision-freeze/judge.py run [--smoke]   # inside a job, with Ollama serving
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
sys.path.insert(0, str(ROOT))
from helpers.pipeline import append_row, config, read_csv  # noqa: E402
from helpers.slurm import queued  # noqa: E402

RESULTS = SIDE / "results"
LABELS_ROOT = ROOT / "datasets"
FILE = re.compile(r"^rationale-(.+)-(answer-first|explain|rationale-first)\.csv$")
ARM = re.compile(r"^(g-.+-tusz|gm-.+|z-.+)(-s2)?$")
SKIP = ("minicpm46-1b",)  # its image never reached the model
CONTROL = 0.10
HEADER = ["path", "label", "control", "reference_path", "same_conclusion", "same_evidence", "reason"]
RETRY_TEMPERATURE, RETRY_SLEEP, TRANSIENT = 0.3, 10, (ConnectionError, TimeoutError, OSError)
JOB = "eeg-vlm-judge-rationales"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load("run", SIDE / "run.py")
benchmark = load("benchmark_judge", ROOT / "judge" / "judge.py")
JC = config()["judge"]


def files():
    out = []
    for path in sorted(RESULTS.glob("rationale-*.csv")):
        match = FILE.match(path.name)
        if not match or not ARM.match(match.group(1)) or any(s in match.group(1) for s in SKIP):
            continue
        if match.group(1) in run.ARMS:
            out.append((path, match.group(1), match.group(2)))
    return out


def references(dataset, cache={}):
    if dataset not in cache:
        rows = read_csv(LABELS_ROOT / dataset / "labels.csv")
        cache[dataset] = {r["path"]: run.CHANNEL.sub(r"\1", run.strip_label(r["ground_truth_rationale"], dataset))
                          for r in rows if r.get("ground_truth_rationale", "").strip()}
    return cache[dataset]


def candidate(reply):
    try:
        return json.loads(reply)["text_rationale"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return reply


def out_path(arm, fmt, name="judge"):
    return RESULTS / f"{name}-{arm}-{fmt}.csv"


def tasks(path, arm, fmt):
    # every window against its own reference, and 10% of them against a different same-label window's reference
    rows = list({r["path"]: r for r in read_csv(path)}.values())
    refs = references(run.dataset_of(arm))
    rows = [r for r in rows if r["path"] in refs]
    done = {(r["path"], r["control"]) for r in read_csv(out_path(arm, fmt))}
    rng = random.Random(0)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["seizure"]].append(r["path"])
    control = rng.sample(sorted(r["path"] for r in rows), round(CONTROL * len(rows)))
    todo = [(r["path"], r["seizure"], "false", r["path"], candidate(r["reply"])) for r in rows]
    by_path = {r["path"]: r for r in rows}
    for p in control:
        others = [q for q in by_label[by_path[p]["seizure"]] if q != p]
        if others:
            todo.append((p, by_path[p]["seizure"], "true", rng.choice(others), candidate(by_path[p]["reply"])))
    return [t for t in todo if (t[0], t[2]) not in done], refs


def llms():
    from langchain_ollama import ChatOllama

    def make(temperature):
        return ChatOllama(model=JC["model"], base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                          num_ctx=JC["num-ctx"], num_predict=JC["num-predict"], temperature=temperature,
                          ).with_structured_output(benchmark.AgreementVerdict, include_raw=True,
                                                   **config()["settings"]["structured-output"])
    return make(0), make(RETRY_TEMPERATURE)


def judge_pair(models, reference, text):
    # judge/judge.py's judge_pair, keeping both verdicts: transient errors retry greedy, a parse failure gets one
    # sampled attempt at the end
    text = text.strip()[:JC["max-rationale-chars"]]
    if not text:
        return "false", "false", "empty candidate rationale"
    prompt = benchmark.JUDGE_PROMPT.format(reference=reference.strip()[:JC["max-rationale-chars"]], candidate=text)
    greedy, sampled = models
    attempts = [greedy] * (JC["judge-retries"] + 1) + [sampled]
    for i, llm in enumerate(attempts):
        try:
            result = llm.invoke(prompt)
            if result["parsed"] is None:
                raise ValueError(str(result.get("parsing_error")))
            verdict = result["parsed"]
            return str(verdict.same_conclusion).lower(), str(verdict.same_evidence).lower(), verdict.reason
        except Exception as e:
            error = e
            if isinstance(e, TRANSIENT):
                if i >= JC["judge-retries"]:
                    raise
                time.sleep(RETRY_SLEEP)
    raise error


def summarise(arm, fmt):
    rows = read_csv(out_path(arm, fmt))
    rate = lambda rs, key: round(sum(r[key] == "true" for r in rs) / len(rs), 4) if rs else ""
    both = lambda rs: round(sum(r["same_conclusion"] == r["same_evidence"] == "true" for r in rs) / len(rs), 4) if rs else ""
    summary = {"arm": arm, "format": fmt}
    for group, label in (("all", None), ("positive", "true"), ("negative", "false")):
        own = [r for r in rows if r["control"] == "false" and (label is None or r["label"] == label)]
        ctrl = [r for r in rows if r["control"] == "true" and (label is None or r["label"] == label)]
        summary.update({f"{group}_n": len(own), f"{group}_same_conclusion": rate(own, "same_conclusion"),
                        f"{group}_same_evidence": rate(own, "same_evidence"), f"{group}_both": both(own),
                        f"{group}_control_n": len(ctrl), f"{group}_control_same_conclusion": rate(ctrl, "same_conclusion"),
                        f"{group}_control_same_evidence": rate(ctrl, "same_evidence"), f"{group}_control_both": both(ctrl)})
    with out_path(arm, fmt, "summary-judge").open("w") as f:
        f.write(",".join(summary) + "\n" + ",".join(str(v) for v in summary.values()) + "\n")
    print(json.dumps({k: summary[k] for k in ("arm", "format", "all_n", "all_both", "all_control_both")}), flush=True)


def judge_all(smoke):
    models = llms()
    work = [(path, arm, fmt) for path, arm, fmt in files()]
    if smoke:
        path, arm, fmt = work[0]
        todo, refs = tasks(path, arm, fmt)
        picks = [t for t in todo if t[2] == "false"][:4] + [t for t in todo if t[2] == "true"][:1]
        for p, label, control, ref, text in picks:
            verdict = judge_pair(models, refs[ref], text)
            print(f"smoke {arm} {fmt} {p} label={label} control={control}: {verdict}", flush=True)
        print(f"smoke passed: {len(work)} files to judge", flush=True)
        return
    workers = JC["parallel-requests"]
    for path, arm, fmt in work:
        todo, refs = tasks(path, arm, fmt)
        started, failed = time.time(), 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(judge_pair, models, refs[t[3]], t[4]): t for t in todo}
            for future in as_completed(futures):
                p, label, control, ref, _ = futures[future]
                try:
                    conclusion, evidence, reason = future.result()
                except Exception as e:
                    failed += 1
                    print(f"failed {arm} {p}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    continue
                append_row(out_path(arm, fmt), HEADER, [p, label, control, ref, conclusion, evidence, reason])
        print(f"{arm} {fmt}: {len(todo) - failed} judged, {failed} failed, {time.time() - started:.0f} s", flush=True)
        summarise(arm, fmt)


SBATCH = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --account=def-milad777
#SBATCH --time={time}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:{gpus}
#SBATCH --output={logs}/{job}-%j.out
{dependency}
set -euo pipefail
module load StdEnv/2023 apptainer/1.4.5 2>/dev/null || module load StdEnv/2023 apptainer  # Fir has 1.3.5
source {root}/.venv/bin/activate
port=$((20000 + SLURM_JOB_ID % 40000))
export OLLAMA_HOST=127.0.0.1:$port
export OLLAMA_BASE_URL=http://127.0.0.1:$port
export OLLAMA_MODELS=$SCRATCH/ollama/models
export OLLAMA_CONTEXT_LENGTH={context}
export OLLAMA_NUM_PARALLEL={parallel}
export APPTAINERENV_OLLAMA_MODELS=$OLLAMA_MODELS
export APPTAINERENV_OLLAMA_HOST=$OLLAMA_HOST
export APPTAINERENV_OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL
export APPTAINERENV_OLLAMA_CONTEXT_LENGTH=$OLLAMA_CONTEXT_LENGTH
apptainer exec --nv $SCRATCH/ollama/ollama.sif ollama serve > {logs}/ollama-$SLURM_JOB_ID.log 2>&1 &
trap 'kill %1 2>/dev/null || true' EXIT
for i in $(seq 120); do curl -s $OLLAMA_BASE_URL > /dev/null && break; sleep 2; done
curl -s $OLLAMA_BASE_URL > /dev/null || {{ echo "ollama never became ready" >&2; exit 1; }}
cd {root}
python {script} run {mode}
"""


def submit():
    if any(name.startswith(JOB) for name in queued()):
        sys.exit(f"{JOB} is already queued or running")
    logs = SIDE / "logs"
    logs.mkdir(exist_ok=True)
    gpus = "nvidia_h100_80gb_hbm3_3g.40gb:1" if os.environ.get("CC_CLUSTER") == "fir" else "a100:1"
    ids = []
    for mode, limit, job in (("--smoke", "00:30:00", JOB + "-smoke"), ("", "08:00:00", JOB)):
        dependency = f"#SBATCH --dependency=afterok:{ids[0]}" if ids else ""
        text = SBATCH.format(job=job, time=limit, gpus=gpus, logs=logs, dependency=dependency, root=ROOT,
                             context=JC["num-ctx"], parallel=JC["parallel-requests"], script=Path(__file__), mode=mode)
        result = subprocess.run(["sbatch", "--parsable"], input=text, text=True, capture_output=True)
        if result.returncode:
            sys.exit(f"sbatch failed: {result.stderr.strip()}")
        ids.append(result.stdout.strip().split(";")[0])
        print(f"submitted {job}: {ids[-1]}", flush=True)
    n = sum(len(tasks(*f)[0]) for f in files())
    print(f"{len(files())} files, {n} judge calls pending", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["submit"]:
        submit()
    elif sys.argv[1:2] == ["run"]:
        judge_all("--smoke" in sys.argv)
    else:
        sys.exit("usage: judge.py submit | run [--smoke]")
