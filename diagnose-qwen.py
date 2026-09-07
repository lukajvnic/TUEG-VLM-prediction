import atexit
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval" / "models"))
from helpers.pipeline import ROOT, read_csv
from structure import get_structure

MODEL = "qwen2.5vl:3b"
FAILING_IMAGES = 2
TIMEOUT = 420
NUM_PREDICT = 128
BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def reachable():
    try:
        urllib.request.urlopen(BASE, timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def start_server():
    if reachable():
        return
    print("starting cpu ollama via apptainer...")
    env = {**os.environ, "OLLAMA_MODELS": f"{os.environ['SCRATCH']}/ollama/models"}
    proc = subprocess.Popen(
        ["bash", "-lc", "module load StdEnv/2023 apptainer/1.4.5 && "
         f"exec apptainer exec {os.environ['SCRATCH']}/ollama/ollama.sif ollama serve"],
        env=env, stdout=open("/tmp/diagnose-ollama.log", "w"), stderr=subprocess.STDOUT)
    atexit.register(proc.kill)
    for _ in range(60):
        if reachable():
            return
        time.sleep(2)
    sys.exit("ollama never became ready - see /tmp/diagnose-ollama.log")


def pick_images():
    failures = read_csv(ROOT / "logs" / "failures.csv")
    failing = []
    for r in failures:
        if r["model"] == MODEL and r["stage"] == "eval" and "No data" in r["error"]:
            full = f"{r['dataset']}/{r['path']}"
            if full not in failing and (ROOT / "datasets" / full).exists():
                failing.append(full)
        if len(failing) >= FAILING_IMAGES:
            break
    dataset = failing[0].split("/")[0]
    control = next(f"{dataset}/{r['path']}" for r in read_csv(ROOT / "datasets" / dataset / "eval-baseline.csv")
                   if r["model"] == MODEL and f"{dataset}/{r['path']}" not in failing
                   and (ROOT / "datasets" / dataset / r["path"]).exists())
    return failing, control


def generate(image, schema):
    payload = {"model": MODEL, "prompt": "Describe this EEG plot briefly.", "stream": False,
               "images": [base64.b64encode((ROOT / "datasets" / image).read_bytes()).decode()],
               "options": {"num_predict": NUM_PREDICT, "temperature": 0}}
    if schema is not None:
        payload["format"] = schema
    request = urllib.request.Request(f"{BASE}/api/generate", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            text = json.load(response).get("response", "")
        return f"{time.time() - started:5.0f}s  {len(text):4d} chars  {' '.join(text.split())[:90]!r}"
    except Exception as e:
        return f"{time.time() - started:5.0f}s  FAILED: {str(e)[:90]}"


def main():
    start_server()
    failing, control = pick_images()
    for image in [control, *failing]:
        kind = "CONTROL" if image == control else "FAILING"
        schema = get_structure(image.split("/")[0]).model_json_schema()
        print(f"\n{kind} {image}")
        print(f"  constrained:   {generate(image, schema)}", flush=True)
        print(f"  unconstrained: {generate(image, None)}", flush=True)
    print("\nif CONTROL itself times out, the cpu is too slow and the test is inconclusive")
    print("reading: constrained FAILED/empty + unconstrained fine on failing images = grammar stall")
    print("both fail on failing images (control fine) = model seizes on the image itself")


if __name__ == "__main__":
    main()
