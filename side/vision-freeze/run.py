# side test: does training the vision side with the language model frozen (NeuroCanvas's recipe, arXiv 2602.04769)
# make qwen3-vl:2b-instruct see TUSZ seizures? Frozen vision features already separate seizure from background at
# AUROC ~0.70 (linear probe, 2026-10-06) while the LoRA fine-tunes reached 0.52-0.56. One balanced "any seizure"
# yes/no question (the multilabel pilot's lines), three arms on the same data:
#   freeze: vision tower + merger fully trained (fp32 master weights), language model frozen
#   lora:   the labels round's LoRA recipe on the same lines, the control
#   base:   no training, the same question asked of the untrained model
# Every arm scores P(true) on all TUSZ test windows; results/summary-<arm>.csv has window AUROC with a patient
# bootstrap CI, within seizure recordings, and on the probe's 1,500 test windows
# Round 2 (2026-10-07), the freeze recipe for 4 epochs: on the shipped images (freeze-4ep), on rerender.py's second
# rendering of the same windows (freeze-4ep-render: bipolar montage, one gain per recording, soft compression), and on
# that rendering with every seizure window of the training recordings (freeze-4ep-render-more, 4,664 / 4,664 lines)
# Round 3 (2026-10-10): freeze-4ep-render with gemma3:12b's rationale in the target ({"text_rationale", "present"},
# rationale first), plain and with the boolean token's loss weighted 10x (side/rationale-w10's collator and loss).
# Only the pilot's windows have rationales, so these compare with freeze-4ep-render. They keep the final checkpoint
# (the val metric would read the label off the teacher's label-stating rationale) and are scored by generating the
# rationale, then reading P(true) at the boolean, on the probe's 1,500 test windows (generation is ~10x slower)
# Round 4 (2026-10-11), a sweep over how the rationale enters the target, all freeze-4ep-render otherwise:
#   order: answer-first ({"present", "text_rationale"}), rationale-first, or multitask (the plain answer question plus
#   a separate "explain" question per window); rationale: as written, or with the label-naming sentences and label
#   codes stripped (strip_label); boolean weight 1 or 10; and two arms on the "more" windows (rationale where the
#   teacher wrote one, the answer alone elsewhere). Answer-first and multitask score teacher-forced on all 8,460
#   windows like the label-only arms and keep the best checkpoint; rationale-first generates (probe windows)
# Round 5 (2026-10-13), rationale learning: the "hybrid" recipe trains the vision tower fully (fp32, lr 1e-5, saved
# whole through PEFT's modules_to_save) and LoRA on the language model (the labels round's r16 at lr 2e-4), so the
# text can learn the teacher's rationales while the answer stays image-driven. --rationale-eval measures rationale
# learning on 300 test windows (150 seizure): the mean NLL of the label-stripped teacher reference rationale, and
# the model's own generated rationale (saved, for reading and for the judge)
#   python side/vision-freeze/run.py rationale-eval ARM [ARM ...]    # one job evaluating finished arms
#   python side/vision-freeze/run.py background-test ARM [ARM ...]   # score finished arms on data/background.csv
#   python side/vision-freeze/run.py final-test ARM                  # held-out TUSZ patients + interior background
# background-test (2026-10-14): the scored test background is 97.8% first/last windows; rerender.py background lists
# 4,471 interior background windows of the same test patients. Each arm's P(true) on them, against its existing
# scores on the 3,754 test seizure windows, gives AUROC and specificity on background spread through the recordings
# Round 6 (2026-10-14), does the recipe transfer? freeze-4ep-render-more's recipe (answer only) on one yes/no question
# per other corpus (rerender.py datasets: the fine-tune's vetted train/val windows, the benchmark's scored test
# windows), with more epochs on the small sets (about TUSZ's ~1,000 steps, at most 12 epochs); and on qwen3-vl:4b
# (pilot TUSZ lines, a full A100: the 2b already peaks at 17-20 GB of the 20 GB slice)
#   python side/vision-freeze/run.py ARM [--dry-run]                 # submit one arm's MIG job
#   python side/vision-freeze/run.py smoke                           # <=2 h job: SMOKE_ARMS, 2 steps, 24 windows
#   python side/vision-freeze/run.py ARM --train|--score [--smoke]   # what the jobs run
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "models"))
sys.modules.setdefault("datasets", None)  # same guard as train/train.py: the repo's datasets/ is not the HF library
from helpers.pipeline import append_row, base_spec, config, db, read_csv  # noqa: E402
from helpers.slurm import queued, script, submit  # noqa: E402

MODEL, DATASET, QUESTION = "qwen3-vl:2b-instruct", "TUSZ", "any_seizure"
# recipe: what trains (None: nothing); images: which PNGs; lines: the pilot's balanced any_seizure lines, or
# rerender.py's "more" set (train only; val is always the pilot's)
ARMS = {
    "freeze": {"recipe": "freeze", "epochs": 2, "images": "shipped", "lines": "pilot"},
    "lora": {"recipe": "lora", "epochs": 2, "images": "shipped", "lines": "pilot"},
    "base": {"recipe": None, "epochs": 0, "images": "shipped", "lines": "pilot"},
    "freeze-4ep": {"recipe": "freeze", "epochs": 4, "images": "shipped", "lines": "pilot"},
    "freeze-4ep-render": {"recipe": "freeze", "epochs": 4, "images": "render", "lines": "pilot"},
    "freeze-4ep-render-more": {"recipe": "freeze", "epochs": 4, "images": "render", "lines": "more"},
    "freeze-4ep-render-rationale": {"recipe": "freeze", "epochs": 4, "images": "render", "lines": "pilot",
                                    "target": "rationale", "weight": 1},
    "freeze-4ep-render-rationale-w10": {"recipe": "freeze", "epochs": 4, "images": "render", "lines": "pilot",
                                        "target": "rationale", "weight": 10},
}
R4 = {"recipe": "freeze", "epochs": 4, "images": "render", "target": "rationale"}
for _name, _lines, _order, _strip, _weight in (
        ("r4-af-orig", "pilot", "answer-first", False, 1), ("r4-af-orig-w10", "pilot", "answer-first", False, 10),
        ("r4-af-strip", "pilot", "answer-first", True, 1), ("r4-af-strip-w10", "pilot", "answer-first", True, 10),
        ("r4-rf-strip", "pilot", "rationale-first", True, 1), ("r4-rf-strip-w10", "pilot", "rationale-first", True, 10),
        ("r4-mt-strip", "pilot", "multitask", True, 1), ("r4-mt-strip-w10", "pilot", "multitask", True, 10),
        ("r4-af-strip-more", "more", "answer-first", True, 1), ("r4-mt-strip-more", "more", "multitask", True, 1),
        # added 2026-10-12 after r4-af-strip-w10 matched the answer-only arm (-0.016 [-0.043, +0.010]) and beat both
        # r4-af-strip and r4-af-orig-w10 by ~0.10: the winning pilot recipe on the "more" windows
        ("r4-af-strip-w10-more", "more", "answer-first", True, 10)):
    ARMS[_name] = {**R4, "lines": _lines, "order": _order, "strip": _strip, "weight": _weight}
# round 5: the hybrid recipe on round 4's best formats; each compares with its frozen-LM twin
for _name, _lines, _order, _weight in (
        ("r5-hy-af-strip-w10", "pilot", "answer-first", 10), ("r5-hy-mt-strip", "pilot", "multitask", 1),
        ("r5-hy-af-strip-more", "more", "answer-first", 1), ("r5-hy-mt-strip-more", "more", "multitask", 1)):
    ARMS[_name] = {**R4, "recipe": "hybrid", "lines": _lines, "order": _order, "strip": True, "weight": _weight}
# the untrained model on the rerendered images: the rationale baseline, and a zero-shot reference on those images
ARMS["base-render"] = {"recipe": None, "epochs": 0, "images": "render", "lines": "pilot"}
# round 6: lines "dataset" reads rerender.py's data/<DS>-{train,val,test}.jsonl
for _ds, _question, _epochs in (("TUAB", "abnormal", 4), ("TUEP", "epilepsy", 10), ("TUAR", "eyem", 12),
                                ("TUEV", "epileptiform", 12)):
    ARMS[f"r6-{_ds.lower()}-{_question}"] = {"recipe": "freeze", "epochs": _epochs, "images": "render",
                                             "lines": "dataset", "dataset": _ds, "question": _question}
ARMS["r6-4b-freeze-4ep-render"] = {**ARMS["freeze-4ep-render"], "model": "qwen3-vl:4b-instruct", "gpus": "a100:1"}
# core design (2026-10-14): round 6's arms and the TUSZ "more" and pilot sets on rerender.py core's images and lines
# (images-core/<DS>/{train,val,test}/, data/core/); "train" names the training lines file when it is not <DS>-train
CORE = {"recipe": "freeze", "images": "core", "lines": "core"}
ARMS["c-tusz-more"] = {**CORE, "epochs": 4}
ARMS["c-tusz-pilot"] = {**CORE, "epochs": 4, "train": "pilot-train"}
for _ds, _question, _epochs in (("TUAB", "abnormal", 4), ("TUEP", "epilepsy", 10), ("TUAR", "eyem", 12),
                                ("TUEV", "epileptiform", 12)):
    ARMS[f"c-{_ds.lower()}-{_question}"] = {**CORE, "epochs": _epochs, "dataset": _ds, "question": _question}
# the split control for TUEP (2026-10-22): the EMBC 2026 paper split TUEP images at random, so patients sat on both
# sides; c-tuep-random is c-tuep-epilepsy on rerender.py tuep-random's split (whole recordings at random, same windows
# and sizes), so the two differ only in the split rule
ARMS["c-tuep-random"] = {**ARMS["c-tuep-epilepsy"], "splits": "random"}
# round 7 (2026-10-15), the grid: the hybrid recipe (answer first, label-stripped teacher rationale, weight 1) on the
# core images, for every Qwen-VL model that fits one A100 x five corpora. The 8b and the 7b have never run it, so
# their jobs start with a 2-step smoke (smoke-first) and stop early if it fails
GRID_MODELS = {"2b": ("qwen3-vl:2b-instruct", {}), "4b": ("qwen3-vl:4b-instruct", {"gpus": "a100:1", "ram": "64G"}),
               "8b": ("qwen3-vl:8b-instruct", {"gpus": "a100:1", "ram": "64G"}),
               "q25-7b": ("qwen2.5vl:7b", {"gpus": "a100:1", "ram": "64G"})}  # 2b: the default MIG slice
# round 7b (2026-10-15, "as many models as possible"): every other base that fits one A100 and has a separate image
# side (vision_modules): Gemma 3/4, MedGemma, Granite, LLaVA. They see less of the 2688x896 image (Gemma resizes to
# 896^2, LLaVA-1.5 to 336^2, Granite tiles), which is part of the result. All start with the smoke
GRID_EXTRA = {"gemma3-4b": "gemma3:4b", "gemma3-12b": "gemma3:12b", "medgemma-4b": "medgemma:4b",
              "medgemma15-4b": "medgemma1.5:4b", "gemma4-e2b": "gemma4:e2b", "gemma4-e4b": "gemma4:e4b",
              "granite-2b": "granite3.2-vision:2b", "llava-7b": "llava:7b", "llava-phi3": "llava-phi3:3.8b",
              "llava-llama3": "llava-llama3:8b",
              # round 7c (2026-10-18, "as many as possible"): the other bases on Narval that fit one A100
              "gemma4-12b": "gemma4:12b", "minicpm46-1b": "minicpm-v4.6:1b", "glm-ocr": "glm-ocr:latest"}
GRID_MODELS.update({short: (model, {"gpus": "a100:1", "ram": "64G"}) for short, model in GRID_EXTRA.items()})
# round 7d (2026-10-21, "as many remaining models as you can", all on Fir for the Narval outage): HF-native bases our
# collator and hybrid recipe take as they are (checked on the meta device and with each chat template). Left out:
# GLM-4.1V-9B-Thinking (its template prints the assistant's content list as text), LLaVA-OneVision (the answer turn
# does not extend the generation prompt), PaliGemma 2 (no chat template), InternVL3 14B (too big for one 40 GB GPU),
# Llama 3.2 Vision, Aya Vision and PerceptionLM (licences not accepted). They train on a 40 GB H100 slice, which starts
# within minutes on Fir, and every job first checks that the image reaches the model (--image-check)
GRID_FIR = {"internvl3-1b": "internvl3:1b", "internvl3-2b": "internvl3:2b", "internvl3-8b": "internvl3:8b",
            "lfm2vl-450m": "lfm2-vl:450m", "lfm2vl-16b": "lfm2-vl:1.6b", "q25-3b": "qwen2.5vl:3b",
            "smolvlm2-22b": "smolvlm2:2.2b", "idefics3-8b": "idefics3:8b", "llava-next-7b": "llava-next:7b",
            "gemma3n-e2b": "gemma3n:e2b", "gemma3n-e4b": "gemma3n:e4b"}
GRID_EXTRA.update(GRID_FIR)
GRID_MODELS.update({short: (model, {"gpus": "a100:1", "ram": "64G", "fir-gpus": "nvidia_h100_80gb_hbm3_3g.40gb:1",
                                    "image-check": True}) for short, model in GRID_FIR.items()})
# round 8 (2026-10-28, maintainer: "what are all the big models we can finetune ... i want to finetune them"): every
# base over 12B that our loader, collator and hybrid recipe take as they are (image side, LoRA targets and each chat
# template's answer span checked on the meta device). The number is the H100s a training job takes: one when the bf16
# weights plus the image side's fp32 copy and optimizer state stay near 65 GB (Gemma 3 27B: 63 GB), two above that
# (Qwen3-VL 32B: 78 GB; device_map="auto" splits the layers). Trillium takes 1 GPU or a whole node, so the two-GPU ones
# run on Fir. Scoring holds no optimizer state and takes one whole H100 (the largest, Qwen3.5-35B-A3B, is 72 GB).
# Left out: InternVL3 38B (its 6B vision tower trained in full needs ~110 GB of optimizer state), the 72-90B bases and
# Llama 4 (2-4 H100s each and ~10x the 8B's compute), Aya Vision 32B and Llama 3.2 90B (licences not accepted).
# Qwen2.5-VL 72B joined the same day (maintainer: "lets do qwen2.5 72B"): 161 GB static, so four H100s, or three of
# TamIA's H200s in its packs. Then Qwen3-VL 235B-A22B (481 GB) and Qwen3.5 122B-A10B (253 GB) (maintainer: "lets do
# both 235 and 122"), on TamIA's H200s: 6 and 4 per training run, since device_map's balanced split with GPU 0's 20% cap
# leaves layers on the CPU with one fewer; 4 and 3 per zero-shot arm. The counts below are H100s
GRID_BIG = {"gemma3-27b": ("gemma3:27b", 1), "medgemma-27b": ("medgemma:27b", 1), "gemma4-26b": ("gemma4:26b", 1),
            "mistral32-24b": ("mistral-small3.2:24b", 1), "mistral31-24b": ("mistral-small3.1:24b", 1),
            "ministral3-14b": ("ministral3:14b", 1), "internvl3-14b": ("internvl3:14b", 1), "llava-13b": ("llava:13b", 1),
            "gemma4-31b": ("gemma4:31b", 2), "30b-a3b": ("qwen3-vl:30b-a3b-instruct", 2),
            "q35-35b-a3b": ("qwen3.5:35b-a3b", 2), "q35-27b": ("qwen3.5:27b", 2), "32b": ("qwen3-vl:32b-instruct", 2),
            "q25-32b": ("qwen2.5vl:32b", 2), "llava-next-34b": ("llava:34b", 2),
            "q25-72b": ("qwen2.5vl:72b", 4), "235b-a22b": ("qwen3-vl:235b-a22b-instruct", 8),
            "q35-122b-a10b": ("qwen3.5:122b-a10b", 6)}
GRID_EXTRA.update({short: model for short, (model, _) in GRID_BIG.items()})
# Qwen3.5 trains on Killarney's L40S (2026-10-28): its linear attention needs flash-linear-attention's kernels to run at
# a usable speed (the torch fallback took 41.6 s/step for 27B on 2 H100s), and fla refuses its backward on Hopper GPUs
# under Triton 3.7.1 (wrong gradients, fla #640; the wheelhouse has 3.6.0, PyPI nothing newer). On an L40S fla matched
# the fallback (output 5.9e-3, gradients < 1e-2 relative error) at 18.7x its speed. 35B-A3B (72 GB) takes a whole node.
# 122B-A10B fits no L40S node, so it trains on Hopper with the fallback (hopper_linear_attention)
KILLARNEY_GPUS = {"q35-27b": "l40s:3", "q35-35b-a3b": "l40s:4"}
GRID_MODELS.update({short: (model, {"gpus": f"h100:{n}", "ram": f"{128 * n}G", "fir-gpus": f"h100:{n}", "h100s": n,
                                    "big": True, "image-check": True,
                                    **({"killarney-gpus": KILLARNEY_GPUS[short]} if short in KILLARNEY_GPUS else {})})
                    for short, (model, n) in GRID_BIG.items()})
# gm-: TUAB, TUAR and TUEV in the multitask format (each window's answer line plus a separate "explain" line). Every
# line of these corpora has a teacher rationale, so answer-first at weight 1 drowned the answer token (~1 in 150
# target tokens): the g- answer-first runs scored AUROC 0.49-0.79 against round 6's 0.78-0.97, with train loss near
# 0.08 and val loss rising. Multitask matched 10x weighting on TUSZ's all-rationale pilot lines (0.841 vs 0.853
# frozen, hybrid 0.855) without weighting tokens (maintainer's choice, 2026-10-15)
GRID_MULTITASK = ("TUAB", "TUAR", "TUEV")
# the TUEP system (2026-10-15): TUEP's label is the patient's diagnosis and most windows of an epileptic patient look
# normal (window AUROC 0.52 for every direct recipe), so a TUEV epileptiform detector scans whole TUEP test recordings
# (--scan) and each patient is scored by their top spike probabilities. The detectors are the best TUSZ recipe
# (vision trained, language frozen, answer only); every grid TUEV model also gets a scan
ARMS["s-2b-tuev"] = {**CORE, "epochs": 12, "dataset": "TUEV", "question": "epileptiform"}
ARMS["s-4b-tuev"] = {**ARMS["s-2b-tuev"], "model": "qwen3-vl:4b-instruct", "gpus": "a100:1", "ram": "64G"}
GRID_DATA = {"TUSZ": ("any_seizure", 4), "TUAB": ("abnormal", 4), "TUEP": ("epilepsy", 10), "TUAR": ("eyem", 12),
             "TUEV": ("epileptiform", 12)}
for _short, (_model, _resources) in GRID_MODELS.items():
    for _ds, (_question, _epochs) in GRID_DATA.items():
        if _ds in GRID_MULTITASK:
            ARMS[f"gm-{_short}-{_ds.lower()}"] = {
                **CORE, "recipe": "hybrid", "target": "rationale", "order": "multitask", "strip": True, "weight": 1,
                "epochs": _epochs, "dataset": _ds, "question": _question, "model": _model, **_resources,
                "smoke-first": _short in ("8b", "q25-7b") or _short in GRID_EXTRA}
        ARMS[f"g-{_short}-{_ds.lower()}"] = {
            **CORE, "recipe": "hybrid", "target": "rationale", "order": "answer-first", "strip": True, "weight": 1,
            "epochs": _epochs, "dataset": _ds, "question": _question, "model": _model, **_resources,
            "smoke-first": _short in ("8b", "q25-7b") or _short in GRID_EXTRA}
# second seeds for the two collapsed runs of otherwise strong models (2026-10-17): every earlier run used the Trainer's
# default seed 42; "seed" also seeds model setup (LoRA init) and the data order
# (2026-10-18: seed 2 took gm-8b-tuar from 0.524 to 0.776, so every collapsed or weak pair got one; 2026-10-20: seed 2
# moved 9 small-corpus cells by 0.06-0.45 both ways, so every pair of every valid grid model gets one, maintainer's
# choice; minicpm46-1b is left out, its image never reaches the model)
SEED_2 = tuple(f"{'g' if ds == 'TUSZ' else 'gm'}-{short}-{ds.lower()}" for short in GRID_MODELS if short != "minicpm46-1b"
               for ds in ("TUSZ", "TUAB", "TUAR", "TUEV"))
for _arm in SEED_2:
    ARMS[f"{_arm}-s2"] = {**ARMS[_arm], "seed": 2}
# zero-shot baselines (2026-10-17): every grid model untrained, scored on the core test lines and images with the
# fine-tunes' scorer, so every fine-tune has a baseline on the same images and question. The 33-model Ollama benchmark
# ran on the shipped images and stands as the off-the-shelf result. TUSZ's also run the final test (final-test)
ZERO_SHOT = list(GRID_MODELS)
for _short in ZERO_SHOT:
    for _ds in ("TUSZ", "TUAB", "TUAR", "TUEV"):
        # 2026-10-23: TUSZ's fine-tuned arms answer first, so its zero-shot arm is scored with the same answer-first
        # prompt and answer (it was the plain question before; those results are kept as *-plain files), and every
        # zero-shot arm also writes rationales in its corpus's trained format for the judge ("eval-format")
        ARMS[f"z-{_short}-{_ds.lower()}"] = {**CORE, "recipe": None, "dataset": _ds, "question": GRID_DATA[_ds][0],
                                             "model": GRID_MODELS[_short][0], **GRID_MODELS[_short][1],
                                             "final-test": _ds == "TUSZ",
                                             **({"target": "rationale", "order": "answer-first", "strip": True}
                                                if _ds == "TUSZ" else {}),
                                             "eval-format": "answer-first" if _ds == "TUSZ" else "explain"}
# pseudo-arms: jobs that evaluate finished arms; the single ones take one arm per job and ask for its GPU
PSEUDO = ("rationale-eval", "background-test", "tuep-scan", "score-last", "score-step", "final-test", "dpo-gen")
SINGLE = ("tuep-scan", "score-last", "score-step", "final-test", "dpo-gen")
RATIONALE_EVAL_WINDOWS = 300  # stratified 150 seizure / 150 not, from the probe's test windows, seed 0
DPO_WINDOWS = 30  # training windows per question for the expert DPO review (dpo_gen): half with the class, seed 0
# TUAB has one question, so 90 windows to match TUAR's and TUEV's three (maintainer, 2026-10-25: "lets up tuab")
DPO_PER_QUESTION = {"TUAB": 90}
# round 1's smoke (4833152) covered freeze, lora and base; round 2's (4961204) the rerender and the "more" lines;
# round 3's (5155519) rationale-first, the 10x weight and generation scoring; round 4's (5399669) answer-first and
# multitask
SMOKE_ARMS = ("r5-hy-af-strip-w10", "r5-hy-mt-strip")
# label codes the teacher wrote into its rationales, removed by strip_label; TUAB's and TUEP's labels are plain words
# ("normal", "epilepsy"), so only their label-naming sentences go
LABEL_CODES = {"TUSZ": r"absz|bckg|cpsz|fnsz|gnsz|mysz|spsz|tcsz|tnsz", "TUAR": r"bckg|chew|elec|elpp|eyem|musc|shiv",
               "TUEV": r"spsw|gped|pled|eyem|artf|bckg", "TUSL": r"bckg|seiz|slow"}
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# a class code left in a kept sentence becomes its plain name: Qwen3-VL-32B heads its paragraphs with the code
# ("**Eyem artifact**:") and names classes by code mid-sentence, which deleting the code turned into "** artifact**:"
CODE_NAMES = {"absz": "absence seizure", "bckg": "background", "cpsz": "complex partial seizure",
              "fnsz": "focal seizure", "gnsz": "generalized seizure", "mysz": "myoclonic seizure",
              "spsz": "simple partial seizure", "tcsz": "tonic-clonic seizure", "tnsz": "tonic seizure",
              "chew": "chewing", "elec": "electrode", "elpp": "electrode pop", "eyem": "eye movement", "musc": "muscle",
              "shiv": "shivering", "spsw": "spike or sharp wave", "gped": "GPED", "pled": "PLED", "artf": "artifact",
              "seiz": "seizure", "slow": "slowing"}
AF_ANSWER = '{"present": true, "text_rationale": ""}'  # answer-first scoring: the boolean comes before any text
RATIONALES = ROOT / "datasets"  # <DS>/sft_{train,val}.jsonl: the teacher's label-conditioned rationales (Qwen3-VL-32B once teacher.py assemble ran; gemma3:12b in sft_train-old.jsonl), same windows
# the rationales name the shipped images' referential channels ("EEG F8-REF", "Fp1-LE"); the rerender shows bipolar
# pairs ("F8-T4"), so channel names are cut to the electrode, which the bipolar row labels contain
RATIONALE_CHARS = 1200
CHANNEL = re.compile(r"\b(?:EEG\s+)?([A-Za-z0-9]+)-(?:REF|LE)\b")
# VF_RENDER_ROOT moves the rerendered images off $SCRATCH: on 2026-10-14 scratch took ~2-3.5 s to create one small
# file (project: ~0.01 s), so the background-test images live under /project
IMAGE_ROOTS = {"shipped": ROOT / "datasets", "render": Path(os.environ.get("VF_RENDER_ROOT", SIDE / "images")),
               "core": Path(os.environ.get("VF_CORE_ROOT", SIDE / "images-core"))}
SOURCE = ROOT / "side" / "multilabel-pilot" / "data"  # the pilot's balanced lines; this test keeps only any_seizure
MORE = SIDE / "data" / "more-train.jsonl"  # rerender.py lists
BACKGROUND = SIDE / "data" / "background.csv"  # rerender.py background
HELDOUT = SIDE / "data" / "core" / "TUSZ-heldout.jsonl"  # rerender.py heldout: the final TUSZ test
# the TUEP spike scan (2026-10-15): rerender.py tuep-scan's windows of every TUEP test recording, and the TUEP test
# patients left out because they share a recording or a person with TUEV train/val (signal-fingerprint audit)
TUEP_SCAN, TUEP_EXCLUDE = SIDE / "data" / "core" / "TUEP-scan.jsonl", SIDE / "tuep-scan-exclude.csv"
SCAN_TOP = (1, 3, 5)  # a recording's score: the mean of its top-k window spike probabilities (top-5 feeds the patient)
SCAN_WINDOWS = 12  # every recording is scored on exactly this many windows, evenly spaced; shorter ones are left out
SEIZURES = ("absz", "cpsz", "fnsz", "gnsz", "mysz", "spsz", "tcsz", "tnsz")
ANSWER = '{"present": true}'  # the scored row's answer: only the logits before its boolean are read
VISION_LR = 1e-5  # full fine-tune of a ViT: ViTST 2e-5, TimeMaster 2e-6; NeuroCanvas does not report one
EVALS = 5
SCORE_BATCH = 4  # test windows per forward (Qwen; score_batch)
GPUS, RAM = "a100_3g.20gb:1", "40G"
# Fir (2026-10-18), the second site for Narval's Oct 27 - Nov 3 outage, runs the same jobs on H100s ($CC_CLUSTER picks
# it). A 2b arm peaks near 20 GB, so Narval's 20 GB slice becomes a 40 GB one; the full-A100 arms peak at ~42 GB, too
# close to 40, so they get a whole H100
FIR_GPUS = {"a100_3g.20gb:1": "nvidia_h100_80gb_hbm3_3g.40gb:1", "a100:1": "h100:1", "h100:1": "h100:1", "h100:2": "h100:2"}
# measured in round 1: freeze 13.1 s/step, scoring 0.60 s/window (~1.4 h for 8,460); 4 epochs of the pilot lines =
# 1,080 steps (~4 h), of the "more" lines = 4,664 steps (~17 h); 5 checkpoint saves of ~8 GB on top
TIME = {"freeze": "08:00:00", "lora": "08:00:00", "base": "03:00:00", "freeze-4ep": "10:00:00",
        "freeze-4ep-render": "10:00:00", "freeze-4ep-render-more": "1-06:00:00",
        "freeze-4ep-render-rationale": "14:00:00", "freeze-4ep-render-rationale-w10": "14:00:00", "smoke": "02:00:00",
        # round 4, from round 2-3's 13-14 s/step: answer-first 1,080 steps + teacher-forced scoring; rationale-first
        # 1,080 steps + ~5 h of generation; multitask 2,156 steps (two lines per window); "more" 4,664 / 5,966 steps
        "r4-af-orig": "10:00:00", "r4-af-orig-w10": "10:00:00", "r4-af-strip": "10:00:00", "r4-af-strip-w10": "10:00:00",
        "r4-rf-strip": "14:00:00", "r4-rf-strip-w10": "14:00:00", "r4-mt-strip": "16:00:00", "r4-mt-strip-w10": "16:00:00",
        "r4-af-strip-more": "1-04:00:00", "r4-mt-strip-more": "1-08:00:00", "r4-af-strip-w10-more": "1-04:00:00",
        # round 5: train (~14 s/step with LoRA on top), score, then the rationale evaluation (~1.2 h)
        "r5-hy-af-strip-w10": "12:00:00", "r5-hy-mt-strip": "18:00:00", "r5-hy-af-strip-more": "1-08:00:00",
        "r5-hy-mt-strip-more": "1-12:00:00", "rationale-eval": "10:00:00", "base-render": "03:00:00",
        # the scan (5,867 windows at 0.4-1.5 s) and the detectors (503 steps) need 1-3 h; short limits backfill sooner
        "background-test": "06:00:00", "s-2b-tuev": "03:00:00", "s-4b-tuev": "03:00:00", "tuep-scan": "03:00:00",
        "score-last": "04:00:00", "dpo-gen": "02:00:00",
        # score-step: one model's views from one checkpoint; TUSZ's 2 x 8,460 windows at 0.13-0.17 s/window (1.6B)
        "score-step": "03:00:00",
        # held-out (5,929) and interior background (4,471) windows at the measured 0.14-0.69 s/window, plus loading
        "final-test": "03:00:00",
        # round 6, at ~13 s/step on the slice: TUAB 1,142 steps, TUEP 1,045, TUAR 557 and TUEV 549, then 354-3,540
        # test windows; the 4b on a full A100 at an unmeasured speed, so the pilot's 1,080 steps get room
        "r6-tuab-abnormal": "08:00:00", "r6-tuep-epilepsy": "08:00:00", "r6-tuar-eyem": "06:00:00",
        "r6-tuev-epileptiform": "05:00:00", "r6-4b-freeze-4ep-render": "12:00:00",
        # core: round 6's step counts, plus scoring; TUSZ as freeze-4ep-render(-more)
        "c-tusz-more": "1-06:00:00", "c-tusz-pilot": "10:00:00", "c-tuab-abnormal": "08:00:00",
        "c-tuep-epilepsy": "08:00:00", "c-tuep-random": "08:00:00", "c-tuar-eyem": "06:00:00", "c-tuev-epileptiform": "05:00:00"}
# grid walltimes: steps x s/step + test windows x s/window + the rationale eval (300 windows) + loading and saves,
# with 50% margin. Measured on the grid's first jobs (2026-10-15): s/step 2b 15.5-16.1 (slice), 4b 10.1-10.8 (one run
# 26.3), 8b 14.6-15.9, qwen2.5vl:7b 24.9-41.5; scoring 0.40-0.65 s/window; rationale eval 6.4-7.4 s/window. The other
# families are not measured: they see 256-2,000 image tokens against Qwen's 2,352-3,072, so these are loose guesses
GRID_SPEED = {"2b": (16.5, 0.7, 8), "4b": (14.0, 0.7, 8), "8b": (16.5, 0.7, 8), "q25-7b": (42.0, 1.0, 10)}
GRID_SPEED.update({short: (14.0, 1.0, 10) for short in ("gemma3-4b", "medgemma-4b", "medgemma15-4b", "gemma4-e2b",
                                                        "gemma4-e4b", "granite-2b", "llava-phi3")})
GRID_SPEED.update({short: (20.0, 1.0, 12) for short in ("llava-7b", "llava-llama3")})
GRID_SPEED["gemma3-12b"] = GRID_SPEED["gemma4-12b"] = (26.0, 1.5, 15)
GRID_SPEED.update({short: (14.0, 1.0, 10) for short in GRID_FIR})  # unmeasured: the 4b-class guess
GRID_SPEED.update({short: (14.0, 1.0, 10) for short in ("minicpm46-1b", "glm-ocr")})
# the big bases (round 8), unmeasured: the 8B's whole-H100 speed (6.5 s/step, 0.3 s/window, 4 s per rationale; its
# four dataset runs took 5.3-10.4 h on Trillium) times each base's compute against it (language parameters x tokens,
# plus the image side): Gemma sees 252-256 image tokens, Mistral 1,045, Qwen3-VL and LLaVA-NeXT ~2,350, Qwen2.5-VL
# 3,072, InternVL3 3,328. The MoE bases count their active parameters with the 8B's speed as a floor. Few-token bases
# cost more than their tokens say, and big ones less: against the 8B's first steps (~12 s), the smokes ran Gemma 3
# 27B at 1.6x (debug job 1183917), InternVL3 14B at 1.9x (1183933) and Qwen3-VL 32B on two GPUs at 1.45x (1184144).
# The rest are guesses from those three. Not steady-state speeds, so 1.5x more on top of the walltime's own margin
BIG_COMPUTE = {"gemma3-27b": 1.6, "medgemma-27b": 1.6, "gemma4-26b": 1.3, "mistral32-24b": 2.0, "mistral31-24b": 2.0,
               "ministral3-14b": 1.3, "internvl3-14b": 2.0, "llava-13b": 1.0, "gemma4-31b": 2.0, "30b-a3b": 1.5,
               "q35-35b-a3b": 1.5, "q35-27b": 2.5, "32b": 2.0, "q25-32b": 2.5, "llava-next-34b": 2.0,
               "q25-72b": 5.0,  # 72B, a guess: Qwen2.5-VL 32B measured 26 s/step (2.7x) on TamIA, 72B 2.3x its language
               "235b-a22b": 3.0, "q35-122b-a10b": 4.0}  # guesses: 22B active against 32B's 14.7 s/step; 27B's fallback 41.6
GRID_SPEED.update({short: (1.5 * 6.5 * f, 1.5 * 0.3 * f, 1.5 * 4.0 * f) for short, f in BIG_COMPUTE.items()})
GRID_LINES = {"TUSZ": (8760, 8460), "TUAB": (2283, 1200), "TUEP": (826, 1084), "TUAR": (367, 3540), "TUEV": (335, 354)}
for _short, (_step, _window, _rationale) in GRID_SPEED.items():
    for _ds, (_train, _test) in GRID_LINES.items():
        _hours = 1.5 * (math.ceil(_train * GRID_DATA[_ds][1] / 8) * _step + _test * _window + 300 * _rationale + 1800) / 3600
        TIME[f"g-{_short}-{_ds.lower()}"] = f"{int(_hours // 24)}-{int(_hours % 24):02d}:{int(_hours % 1 * 60):02d}:00"
        if _ds in GRID_MULTITASK:  # an explain line per window doubles the training lines
            _hours = 1.5 * (math.ceil(2 * _train * GRID_DATA[_ds][1] / 8) * _step + _test * _window + 300 * _rationale + 1800) / 3600
            TIME[f"gm-{_short}-{_ds.lower()}"] = f"{int(_hours // 24)}-{int(_hours % 24):02d}:{int(_hours % 1 * 60):02d}:00"
for _arm in SEED_2:
    TIME[f"{_arm}-s2"] = TIME[_arm]
# zero-shot: the test windows (TUSZ: plus the 5,929 held-out and 4,471 interior background windows) and loading
for _short in ZERO_SHOT:
    for _ds, (_, _test) in GRID_LINES.items():
        if f"z-{_short}-{_ds.lower()}" in ARMS:
            _hours = max(1.0, 1.5 * ((_test + 10400 * (_ds == "TUSZ")) * GRID_SPEED[_short][1] + 1200
                                     + 300 * GRID_SPEED[_short][2] * (_short in GRID_BIG)) / 3600)  # a big base's rationales
            TIME[f"z-{_short}-{_ds.lower()}"] = f"{int(_hours // 24)}-{int(_hours % 24):02d}:{int(_hours % 1 * 60):02d}:00"
# one question per learnable class (maintainer, 2026-10-25): TUAR asks muscle and electrode artifact beside eye movement,
# TUEV asks spike, GPED and PLED separately; rerender.py extra's lines, the same windows, images, recipe and epochs as
# the corpus's grid arms (gm-<short>-tuar / -tuev), for every valid grid model, both seeds, and a zero-shot baseline.
# TUEV's val sets hold 1-2 positives per class, so its arms keep the final checkpoint ("select": "last")
EXTRA_QUESTIONS = {"TUAR": ("musc", "elec"), "TUEV": ("spsw", "gped", "pled")}
for _short in GRID_MODELS:
    if _short == "minicpm46-1b":
        continue
    for _ds, _questions in EXTRA_QUESTIONS.items():
        for _question in _questions:
            _base, _arm = f"gm-{_short}-{_ds.lower()}", f"gm-{_short}-{_ds.lower()}-{_question}"
            ARMS[_arm] = {**ARMS[_base], "question": _question, "splits": _question,
                          **({"select": "last"} if _ds == "TUEV" else {})}
            ARMS[f"{_arm}-s2"] = {**ARMS[_arm], "seed": 2}
            ARMS[f"z-{_short}-{_ds.lower()}-{_question}"] = {**ARMS[f"z-{_short}-{_ds.lower()}"], "question": _question,
                                                             "splits": _question}
            TIME[_arm] = TIME[f"{_arm}-s2"] = TIME[_base]
            TIME[f"z-{_short}-{_ds.lower()}-{_question}"] = TIME[f"z-{_short}-{_ds.lower()}"]
# TUSZ, one question per seizure type with training patients (maintainer, 2026-10-25), on rerender.py tusz-types' lines
# (every window of the type plus as many others). Multitask format, as TUAB/TUAR/TUEV: the teacher wrote each TUSZ
# rationale about the whole window, so behind a per-type answer it could describe a seizure the answer says is absent;
# a separate explain line keeps the answer clean. Focal and generalised keep TUSZ's 4 epochs; the smaller types get the
# small corpora's 12 and keep the final checkpoint (2-4 positives in val). The zero-shot baselines score the test only
TUSZ_TYPES = {"fnsz": (4, 5699), "gnsz": (4, 3547), "cpsz": (12, 1050), "absz": (12, 225), "tcsz": (12, 249)}  # epochs, lines
for _short in GRID_MODELS:
    if _short == "minicpm46-1b":
        continue
    for _question, (_epochs, _lines) in TUSZ_TYPES.items():
        _arm = f"gm-{_short}-tusz-{_question}"
        ARMS[_arm] = {**ARMS[f"g-{_short}-tusz"], "order": "multitask", "question": _question, "splits": _question,
                      "epochs": _epochs, **({"select": "last"} if _epochs == 12 else {})}
        ARMS[f"{_arm}-s2"] = {**ARMS[_arm], "seed": 2}
        ARMS[f"z-{_short}-tusz-{_question}"] = {**ARMS[f"z-{_short}-tuar"], "dataset": "TUSZ", "question": _question,
                                                "splits": _question}
        _step, _window, _rationale = GRID_SPEED[_short]
        _hours = 1.5 * (math.ceil(_lines * _epochs / 8) * _step + 8460 * _window + 300 * _rationale + 1800) / 3600
        TIME[_arm] = TIME[f"{_arm}-s2"] = f"{int(_hours // 24)}-{int(_hours % 24):02d}:{int(_hours % 1 * 60):02d}:00"
        _hours = max(1.0, 1.5 * (8460 * _window + 1200) / 3600)
        TIME[f"z-{_short}-tusz-{_question}"] = f"{int(_hours // 24)}-{int(_hours % 24):02d}:{int(_hours % 1 * 60):02d}:00"
# the 8B per-annotation runs that trained on gemma3:12b's rationales of the shipped images (2026-10-25, before teacher.py):
# their dpo-gen pairs keep those rationales (the DPO fallback, option A)
for _arm in [f"gm-8b-tusz-{q}" for q in TUSZ_TYPES] + [f"gm-8b-tuev-{q}" for q in EXTRA_QUESTIONS["TUEV"]] + \
        [f"gm-8b-tuar-{q}" for q in EXTRA_QUESTIONS["TUAR"]] + ["gm-8b-tuar", "gm-8b-tuab"]:
    ARMS[_arm]["rationales"] = "old"
# one model per dataset (maintainer, 2026-10-25: "three separate prompts, but all to train the same model"): gm-<model>-
# <ds>-all trains on its dataset's question lines merged (rerender.py dataset-all) plus one explain line per window, and
# each question is scored through a view arm gm-<model>-<ds>-all-<q> that reads the dataset model's directory and that
# question's own test lines and prompt. Questions as decided 2026-10-25 (complex partial, absence, tonic-clonic and
# chewing cut). Checkpoint: best val balanced accuracy for TUSZ and TUAB, the last one for TUAR and TUEV, whose val sets
# are too small to choose on (maintainer: "do what you originally said")
DATASET_QUESTIONS = {"TUSZ": ("fnsz", "gnsz"), "TUAR": ("eyem", "musc", "elec"), "TUEV": ("spsw", "gped", "pled"),
                     "TUAB": ("abnormal",)}
# TUAR trains on every annotated window of its training recordings (5,722 windows, rerender.py tuar_all_windows): one
# pass over them costs about what 12 passes over the old 367-window sample did (maintainer, 2026-10-25: "go ahead for
# tuar"). Its explain lines cover 1,440 distinct windows once each, the teacher's 360 plus 1,080 added for it
# (rerender.py tuar-teacher-extra), the same explain budget as 360 repeated 4 times (maintainer: "okay go ahead")
# TUSZ 4 -> 3 epochs for runs that start after 2026-10-26 (maintainer: "do what you can to eliminate overfitting in
# future tusz runs"): in the first finished runs val loss is highest at epochs 3.2-4.0 and the kept checkpoints sit at
# 2.4-3.2. The 8 TUSZ runs that started with 4 are redone at the end if time allows
DATASET_EPOCHS = {"TUSZ": 3, "TUAB": 4, "TUAR": 1, "TUEV": 12}
DATASET_WHAT = {"TUSZ": "a focal or a generalized seizure", "TUAR": "an eye movement, muscle or electrode artifact",
                "TUEV": "a spike or sharp wave, a GPED or a PLED", "TUAB": "abnormal EEG activity"}
DATASET_LINES = {"TUSZ": (7347, 1525, 8460), "TUAR": (17138, 1440, 3540), "TUEV": (1005, 335, 354),
                 "TUAB": (2283, 2283, 1200)}  # answer lines, explain lines, test windows
# on Fir and Nibi every dataset model trains on a 40 GB slice: every base trained within Narval's 40 GB A100s (no CUDA
# out-of-memory in any Narval or Fir log; peaks of 41-42 GB are the allocator filling the card). The 8B (the DPO model,
# speed) and the three largest (closest to the limit) keep a whole H100
WHOLE_H100 = {"8b", "q25-7b", "gemma3-12b", "gemma4-12b", *GRID_BIG}
for _short in GRID_MODELS:
    if _short == "minicpm46-1b":
        continue
    for _ds, _questions in DATASET_QUESTIONS.items():
        _arm = f"gm-{_short}-{_ds.lower()}-all"
        _spec = {k: v for k, v in ARMS[f"gm-{_short}-tuar"].items() if k not in ("rationales", "select", "seed")}
        ARMS[_arm] = {**_spec, "dataset": _ds, "splits": "all", "questions": _questions, "question": "all",
                      "epochs": DATASET_EPOCHS[_ds], **({"select": "last"} if _ds in ("TUAR", "TUEV") else {})}
        if _short not in WHOLE_H100:
            ARMS[_arm]["fir-gpus"] = "nvidia_h100_80gb_hbm3_3g.40gb:1"
        ARMS[f"{_arm}-s2"] = {**ARMS[_arm], "seed": 2}
        _step, _window, _rationale = GRID_SPEED[_short]
        _answers, _explains, _test = DATASET_LINES[_ds]
        _hours = 1.5 * (math.ceil((_answers + _explains) * DATASET_EPOCHS[_ds] / 8) * _step
                        + _test * len(_questions) * _window + 300 * _rationale + 1800) / 3600
        for _name in (_arm, f"{_arm}-s2"):
            TIME[_name] = f"{int(_hours // 24)}-{int(_hours % 24):02d}:{int(_hours % 1 * 60):02d}:00"
            for _question in _questions:
                ARMS[f"{_name}-{_question}"] = {**ARMS[_name], "view": _name, "question": _question,
                                                "splits": None if _question in ("eyem", "abnormal") else _question}
                del ARMS[f"{_name}-{_question}"]["questions"]
                TIME[f"{_name}-{_question}"] = TIME[_name]
# expert DPO on the four 8B dataset models (maintainer, 2026-10-25: "sure 4 runs one per 8b"): gm-8b-<ds>-all-dpo adds
# a fresh LoRA, dropout off, to the language model of the trained dataset model (vision tower and projector frozen) and
# trains it on the neurologist's preferences (argilla/rationales.py export -> DPO_ANSWERS). The reference is the same
# model with that adapter disabled, so the two agree exactly at step 0. Log-ratios are per-token means: summed ones
# reward short rationales (review, 2026-10-25). Views gm-8b-<ds>-all-dpo-<q> score it like the dataset model
DPO_ANSWERS = SIDE / "results" / "dpo-answers.jsonl"
DPO_SETTINGS = {"beta": 5.0, "alpha": 1.0, "learning-rate": 2e-5, "epochs": 3, "accumulation": 2, "rank": 16,
                "lora-alpha": 32}
for _ds, _questions in DATASET_QUESTIONS.items():
    _from = f"gm-8b-{_ds.lower()}-all"
    _arm = f"{_from}-dpo"
    ARMS[_arm] = {**ARMS[_from], "dpo_from": _from}
    _step, _window, _rationale = GRID_SPEED["8b"]
    _hours = 1.5 * (2 * 90 * DPO_SETTINGS["epochs"] * _step + DATASET_LINES[_ds][2] * len(_questions) * _window
                    + 300 * _rationale + 1800) / 3600
    TIME[_arm] = f"{int(_hours // 24)}-{int(_hours % 24):02d}:{int(_hours % 1 * 60):02d}:00"
    for _question in _questions:
        ARMS[f"{_arm}-{_question}"] = {**ARMS[_arm], "view": _arm, "question": _question,
                                       "splits": None if _question in ("eyem", "abnormal") else _question}
        del ARMS[f"{_arm}-{_question}"]["questions"]
        TIME[f"{_arm}-{_question}"] = TIME[_arm]
SMOKE_LINES, SMOKE_WINDOWS = {"train": 16, "val": 8}, 24
PROBE = Path(os.environ.get("SCRATCH", "/scratch/luka")) / "rehearsal" / "research" / "probe" / "manifest.csv"
BOOTSTRAP = 1000


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = load_path("train_main", ROOT / "train" / "train.py")


def base_dir(smoke):
    return SIDE / "smoke" if smoke else SIDE


# --last scores a run's final checkpoint instead of the one load_best_model_at_end kept: on TUAR/TUEV's tiny val sets
# (31 / 16 lines) every eval of some runs tied at BA <= 0.5 and the Trainer kept the first (~20% of training), so the
# scored model was undertrained (review, 2026-10-17). Results then go to <file>-<arm>-last.csv. --step N scores
# checkpoint-N instead (the overfitting check: an earlier checkpoint against the scored one), to <file>-<arm>-stepN.csv
LAST = False
STEP = None


def tag(arm):
    return f"{arm}-last" if LAST else f"{arm}-step{STEP}" if STEP else arm


def model_dir(arm, smoke=False):
    if STEP:
        checkpoint = out_dir(arm, smoke) / f"checkpoint-{STEP}"
        assert checkpoint.is_dir(), f"{checkpoint} does not exist"
        print(f"scoring checkpoint {checkpoint}", flush=True)
        return checkpoint
    if not LAST:
        return out_dir(arm, smoke)
    checkpoints = sorted(out_dir(arm, smoke).glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[1]))
    assert checkpoints, f"{out_dir(arm, smoke)} has no checkpoint-*"
    print(f"scoring the last checkpoint: {checkpoints[-1]}", flush=True)
    return checkpoints[-1]


def out_dir(arm, smoke=False):
    return base_dir(smoke) / "checkpoints" / ARMS.get(arm, {}).get("view", arm)


def dataset_of(arm):
    return ARMS[arm].get("dataset", DATASET)


def model_of(arm):
    return ARMS[arm].get("model", MODEL)


def question_of(arm):
    return ARMS[arm].get("question", QUESTION)


def dataset_lines(arm, split):
    folder = SIDE / "data" / ("core" if ARMS[arm]["lines"] == "core" else "")
    name = f"{dataset_of(arm)}-{ARMS[arm]['splits']}" if ARMS[arm].get("splits") else dataset_of(arm)
    return main.read_jsonl(folder / f"{name}-{split}.jsonl")


def lines(split, smoke, arm="freeze"):
    if ARMS[arm]["lines"] in ("dataset", "core"):
        rows = dataset_lines(arm, ARMS[arm].get("train", "train") if split == "train" else split)
    elif ARMS[arm]["lines"] == "more" and split == "train":
        rows = main.read_jsonl(MORE)
    else:
        rows = [r for r in main.read_jsonl(SOURCE / f"perclass-{split}.jsonl") if r["question"] == QUESTION]
    if ARMS[arm].get("target") == "rationale":
        rows = with_rationale(rows, split, arm)
    return rows[:SMOKE_LINES[split]] if smoke else rows


def order_of(arm):
    return ARMS[arm].get("order", "rationale-first") if ARMS[arm].get("target") == "rationale" else None


def strip_label(text, dataset=DATASET):
    # the teacher wrote each rationale knowing the label and usually names it ('justifying the "fnsz" seizure
    # labeling'); drop the sentences about the labeling or introducing the report, then any label code left inline
    keep = [s for s in SENTENCE.split(text) if s.strip() and not re.search(r"label|here'?s an? |^\**report", s.strip(), re.I)]
    out = " ".join(s.strip() for s in keep)
    if dataset in LABEL_CODES:
        codes = LABEL_CODES[dataset]
        out = re.sub(rf'\s*\(\s*\\?"?({codes})\\?"?\s*\)', "", out, flags=re.I)  # "eye movement (eyem)"
        name = lambda m: CODE_NAMES[m.group(1).lower()]  # noqa: E731
        out = re.sub(rf'\\?"({codes})\\?"', name, out, flags=re.I)
        out = re.sub(rf"\b({codes})\b", name, out, flags=re.I)
    return re.sub(r"\s{2,}", " ", out).strip()


def explain_prompt(arm="freeze"):
    # what the question asks about: "a seizure" for TUSZ, the yes/no question's subject for the other corpora
    p = config()["prompts"]
    if ARMS[arm].get("questions"):
        what = DATASET_WHAT[dataset_of(arm)]
    elif dataset_of(arm) == "TUSZ" and not ARMS[arm].get("splits"):
        what = "a seizure"
    else:
        what = re.search(r"does this window show (.+)\? Set", prompt_text(arm)).group(1)
    return (p["intro"] + f"Based only on these waveforms, describe the specific evidence about whether this window shows "
            f"{what} in the text_rationale field: which channel(s), where along the time axis, and what the curve "
            "looks like there (shape, frequency, amplitude, symmetry). " + p["no-meta"])


def rationale_prompt(arm="freeze"):
    p = config()["prompts"]
    tail = " Set present to true only when it does."
    base = prompt_text(arm)
    assert base.endswith(tail), base
    return base[:-len(tail)] + " " + p["eval"]["binary-rationale"] + p["no-meta"] + tail


def with_rationale(rows, split, arm):
    # the same windows and answers with each window's teacher rationale, placed by the arm's order. A window without
    # a teacher rationale (the "more" extras) keeps the plain answer question. Rationales are matched by window name:
    # the core lists put the val windows under val/, the teacher's files under train/
    spec, dataset = ARMS[arm], dataset_of(arm)
    order = spec.get("order", "rationale-first")
    # "rationales": "old" pins an arm to the rationales it trained on once teacher.py assemble has replaced them
    # (sft_<split>-old.jsonl): the first 8B per-annotation runs, so their dpo-gen pairs show the teacher they learned from
    name = f"sft_{split}-old.jsonl" if spec.get("rationales") == "old" else f"sft_{split}.jsonl"
    if not (RATIONALES / dataset / name).exists():
        name = f"sft_{split}.jsonl"
    source = {Path(r["images"][0]).name: json.loads(r["output"])["text_rationale"]
              for r in main.read_jsonl(RATIONALES / dataset / name)}

    def explain(path):
        rationale = source[Path(path).name]
        return CHANNEL.sub(r"\1", strip_label(rationale, dataset) if spec.get("strip") else rationale)

    out, explained = [], set()
    if spec.get("questions"):  # a dataset model: the answer lines as they are, one explain line per window
        for r in rows:
            out.append(r)
            name = Path(r["images"][0]).name
            if split == "train" and name in source and name not in explained and r.get("explain", True):
                explained.add(name)
                out.append({**r, "question": "explain", "instruction": explain_prompt(arm),
                            "output": json.dumps({"text_rationale": explain(r["images"][0])})})
        return out
    text = rationale_prompt(arm)
    for r in rows:
        path, present = r["images"][0], json.loads(r["output"])["present"]
        if order == "multitask":
            out.append(r)
            if split == "train" and Path(path).name in source:  # val keeps the answer question only, so its metric is clean
                out.append({**r, "question": "explain", "instruction": explain_prompt(arm),
                            "output": json.dumps({"text_rationale": explain(path)})})
        elif Path(path).name not in source:
            out.append(r)
        else:
            fields = {"present": present, "text_rationale": explain(path)} if order == "answer-first" else \
                     {"text_rationale": explain(path), "present": present}
            out.append({**r, "instruction": text, "output": json.dumps(fields)})
    return out


def image_root(arm):
    # the TUSZ rerender sits at the top of images/, the other corpora in images/<DS>/ (window names repeat across them)
    images = ARMS[arm]["images"]
    if images in ("shipped", "core") or dataset_of(arm) != "TUSZ":
        return IMAGE_ROOTS[images] / dataset_of(arm)
    return IMAGE_ROOTS[images]


def missing_images(arm):
    # every training, val and test image this arm reads, checked before the job is submitted
    paths = {r["images"][0] for split in ("train", "val") for r in lines(split, False, arm)} | set(test_windows(arm))
    return sorted(p for p in paths if not (image_root(arm) / p).exists())


def prompt_text(arm="freeze"):
    if ARMS[arm]["lines"] in ("dataset", "core"):
        texts = {r["instruction"] for r in dataset_lines(arm, "train")}
    else:
        texts = {r["instruction"] for r in main.read_jsonl(SOURCE / "perclass-train.jsonl") if r["question"] == QUESTION}
    assert len(texts) == 1, texts
    return texts.pop()


def arm_config(cfg, arm, smoke):
    settings = main.load_config(cfg, model_of(arm), dataset_of(arm))
    settings.update({"experiment": f"vision-freeze-{arm}", "target": "labels", "output-dir": str(out_dir(arm, smoke))})
    return settings


# the image side under each family's name: Qwen's "visual" (blocks + merger), and the vision tower plus projector of
# Gemma 3/4, LLaVA and Granite (LLaVA-NeXT)
VISION_PARTS = ("visual", "vision_tower", "multi_modal_projector", "embed_vision",
                "vision_model", "connector")  # the last two: SmolVLM2 / Idefics3 (2026-10-21)


def vision_modules(model):
    # the outermost module of each image part (a part nested in another one is trained with it)
    names = [n for n, _ in model.named_modules() if n.split(".")[-1] in VISION_PARTS]
    names = [n for n in names if not any(n.startswith(other + ".") for other in names)]
    assert names, "no image-side module found"
    return names


def drop_softcap(model):
    # Gemma 4 and Gemma 3n cap their output logits at 30: in bf16 the true/false logits both saturate, the gradient is
    # exactly 0 and P(true) is 0.5 (known-issues.md, "Gemma4's dead gradient"); no other family sets it
    text = model.config.get_text_config()
    if getattr(text, "final_logit_softcapping", None):
        print(f"final_logit_softcapping {text.final_logit_softcapping} -> None", flush=True)
        try:
            text.final_logit_softcapping = None
        except Exception:  # Gemma 3n's strict config takes only a float; tanh(x / 1e6) * 1e6 leaves the logits as they are
            text.final_logit_softcapping = 1e6


def hopper_linear_attention(model):
    # fla's gated delta rule raises in its backward on Hopper GPUs with Triton 3.4-3.7.1 (wrong gradients, fla #640):
    # there Qwen3.5's linear-attention layers train on transformers' own torch kernel, as on Fir (no fla)
    import sys

    import torch

    if torch.cuda.get_device_capability()[0] != 9:
        return
    layers = [m for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")]
    for layer in layers:
        layer.chunk_gated_delta_rule = sys.modules[type(layer).__module__].torch_chunk_gated_delta_rule
    if layers:
        print(f"Hopper: {len(layers)} linear-attention layers on the torch kernel", flush=True)


def init_frozen_language(settings):
    # every weight frozen, then the vision tower (patch embedding, blocks, merger, deepstack mergers) unfrozen in fp32:
    # a 1e-5 step is below bf16's resolution for most weights, so bf16 weights would barely move
    import torch
    from transformers import AutoModelForImageTextToText

    base = settings["base"]
    source = main.resolve_model(base["repo"])
    processor = main.runner.load_processor(source, base, False)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.config.keys_to_ignore_at_inference = main.EVAL_IGNORE
    model.requires_grad_(False)
    drop_softcap(model)

    names = vision_modules(model)
    for name in names:
        vision = model.get_submodule(name)
        vision.float()
        vision.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable / 1e6:.0f}M of {total / 1e6:.0f}M parameters ({', '.join(names)}, fp32); "
          f"language model frozen", flush=True)
    return model, processor


def keyword_input():
    # PEFT's modules_to_save wrapper takes its input positionally (forward(self, x, ...)), and Gemma 3/4 call their
    # vision tower with keywords only (vision_tower(pixel_values=...)): a keyword-only call passes its first keyword
    # on as that input when it is the module's first parameter, else None goes there and the keyword stays one. Gemma
    # 3n calls embed_vision(inputs_embeds=...), whose first parameter is input_ids: passed positionally, the soft tokens
    # became embedding indices (2026-10-25). Qwen, LLaVA and Granite pass the input positionally and are unaffected
    import inspect

    from peft.utils.other import AuxiliaryTrainingWrapper

    forward = AuxiliaryTrainingWrapper.forward
    if getattr(forward, "keyword_input", False):
        return

    def patched(self, *args, **kwargs):
        if not args:
            key = next(k for k in kwargs if k != "adapter_names")
            first = next(iter(inspect.signature(self.original_module.forward).parameters))
            args = (kwargs.pop(key),) if key == first else (None,)
        return forward(self, *args, **kwargs)

    patched.keyword_input = True
    AuxiliaryTrainingWrapper.forward = patched


def hybridize(model, lora):
    # LoRA on every language Linear (the labels round's targets minus the vision ones), and the whole vision tower
    # as a PEFT modules_to_save copy: trained fully, saved in the adapter file, merged back at scoring. The trained
    # copy is fp32 (1e-5 steps are below bf16 resolution); the frozen original stays bf16
    from peft import LoraConfig, get_peft_model

    keyword_input()
    language, _ = main.runner.lora_targets(model)
    names = vision_modules(model)
    # a Linear inside a fully trained image part gets no LoRA (Idefics3's connector is not in lora_targets' vision hints)
    language = [n for n in language if not any(n.startswith(v + ".") for v in names)]
    model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             bias="none", task_type="CAUSAL_LM", target_modules=language,
                                             modules_to_save=sorted({n.split(".")[-1] for n in names})))
    copies = [m for n, m in model.named_modules() if n.endswith("modules_to_save.default")]
    assert len(copies) == len(names), (len(copies), names)
    for copy in copies:
        copy.float()
    print(f"image side trained in full: {', '.join(names)}", flush=True)
    return model


def split_memory():
    # a base split over several GPUs (round 8's 2-GPU bases): GPU 0 also holds the image side's fp32 copy with its
    # gradients and Adam states, the image activations and (tied embeddings) the output head, so it takes ~20% of the
    # weights and the last GPU the rest. Measured on Qwen3-VL 32B's 2-GPU smoke (Trillium debug job 1184144): 30.3 GB
    # allocated on GPU 0, 59.6 GB on GPU 1. One GPU: no change
    import torch

    n = torch.cuda.device_count()
    if n < 2:
        return {}
    total = [torch.cuda.get_device_properties(d).total_memory for d in range(n)]
    return {"max_memory": {d: int(t * (0.2 if d == 0 else 0.9)) for d, t in enumerate(total)}}


def init_hybrid(settings):
    import torch
    from transformers import AutoModelForImageTextToText

    base = settings["base"]
    source = main.resolve_model(base["repo"])
    processor = main.runner.load_processor(source, base, False)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="auto", **split_memory())
    if len(set(getattr(model, "hf_device_map", {}).values())) > 1:
        print(f"split over GPUs: {sorted(set(model.hf_device_map.values()), key=str)}, "
              f"{sum(d == 0 for d in model.hf_device_map.values())} of {len(model.hf_device_map)} modules on GPU 0",
              flush=True)
    model.config.use_cache = False
    model.config.keys_to_ignore_at_inference = main.EVAL_IGNORE
    model.enable_input_require_grads()
    drop_softcap(model)
    hopper_linear_attention(model)
    model = hybridize(model, settings["lora"])
    vision = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "modules_to_save" in n)
    lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "modules_to_save" not in n)
    print(f"trainable: vision tower {vision / 1e6:.0f}M (fp32, lr {VISION_LR}), language LoRA {lora / 1e6:.0f}M", flush=True)
    return model, processor


def hybrid_optimizer(model, training):
    import torch

    vision = [p for n, p in model.named_parameters() if p.requires_grad and "modules_to_save" in n]
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "modules_to_save" not in n]
    return torch.optim.AdamW([{"params": vision, "lr": VISION_LR}, {"params": lora, "lr": training["learning-rate"]}],
                             weight_decay=training["weight-decay"])


def start_wandb(settings, data, resume, smoke):
    # same project as the main runs, group vision-freeze; offline under the side dir
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    out = Path(settings["output-dir"])
    id_file = out / "wandb-id"
    run_id = id_file.read_text().strip() if resume and id_file.exists() else None
    logged = {k: settings[k] for k in ("model", "dataset", "experiment", "target", "base", "lora", "training")}
    logged.update({"train_rows": len(data["train"]), "val_rows": len(data["validation"]), "vision_lr": VISION_LR})
    logs = base_dir(smoke) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    run = wandb.init(entity=settings["wandb"]["entity"], project=settings["wandb"]["project"],
                     name=main.wandb_name(settings), group="vision-freeze", tags=[settings["dataset"], settings["experiment"]],
                     dir=logs, id=run_id, resume="allow" if run_id else None, config=logged)

    out.mkdir(parents=True, exist_ok=True)
    id_file.write_text(run.id + "\n")
    return run


def train(settings, arm, smoke):
    import torch

    if ARMS[arm].get("view"):
        sys.exit(f"{arm} is a scoring view of {ARMS[arm]['view']}: train that arm")
    if ARMS[arm].get("dpo_from"):
        sys.exit(f"{arm} is trained with --dpo")
    if ARMS[arm].get("questions"):  # a dataset arm trains only on the new teacher's rationales (teacher.py assemble)
        folder = RATIONALES / dataset_of(arm)
        digest = hashlib.md5((folder / "sft_train.jsonl").read_bytes()).hexdigest()
        expected = (folder / "sft_train.md5").read_text().strip() if (folder / "sft_train.md5").exists() else None
        if not (folder / "sft_train-old.jsonl").exists() or digest != expected:
            sys.exit(f"{arm}: {folder / 'sft_train.jsonl'} is not the assembled new-teacher file (md5 {digest}, expected "
                     f"{expected}): copy the new file and its sft_train.md5 here first")
        print(f"teacher rationales: {folder / 'sft_train.jsonl'} md5 {digest}", flush=True)
    out = Path(settings["output-dir"])
    if (out / "manifest.json").exists():  # a resubmitted job whose scoring was cut off: go straight to scoring
        print(f"{out} is already trained (manifest.json exists)", flush=True)
        return
    resume = main.resume_point(out)
    data = {"train": lines("train", smoke, arm), "validation": lines("val", smoke, arm)}
    if ARMS[arm].get("questions") and not smoke:  # epochs assume these exact files (data/ and sft_*.jsonl are not in git)
        answers = sum(r["question"] != "explain" for r in data["train"])
        counts, expected = (answers, len(data["train"]) - answers), DATASET_LINES[dataset_of(arm)][:2]
        if counts != expected:
            sys.exit(f"{arm}: {counts[0]} answer and {counts[1]} explain lines, expected {expected[0]} and {expected[1]}: "
                     f"copy the current data/core/{dataset_of(arm)}-all-train.jsonl and sft_train.jsonl here first")
    training, epochs = settings["training"], ARMS[arm]["epochs"]
    steps = epochs * math.ceil(len(data["train"]) / (training["batch-size"] * training["gradient-accumulation"]))
    every = 1 if smoke else math.ceil(steps / EVALS)  # save-steps must equal eval-steps for the best-checkpoint reload
    settings["training"] = {**training, "epochs": epochs, "eval-steps": every, "save-steps": every}

    print(f"{model_of(arm)} on {dataset_of(arm)} '{question_of(arm)}', arm {arm} {ARMS[arm]}: {len(data['train'])} train / "
          f"{len(data['validation'])} val lines, {steps} steps, eval every {every}, images {image_root(arm)} -> {out}",
          flush=True)
    if resume:
        print(f"resuming from {resume}", flush=True)

    seed = ARMS[arm].get("seed")
    if seed is not None:
        from transformers import set_seed
        set_seed(seed)
    recipe = ARMS[arm]["recipe"]
    freeze = recipe == "freeze"
    if recipe == "hybrid":
        model, processor = init_hybrid(settings)
    else:
        model, processor = init_frozen_language(settings) if freeze else main.init_model(settings)
    run = start_wandb(settings, data, resume, smoke)
    trainer = main.init_trainer(settings, model, processor, data, image_root(arm))
    if seed is not None:  # Trainer.__init__ reseeded with args.seed (42); the data order comes from accelerate's config
        set_seed(seed)
        trainer.args.seed = trainer.args.data_seed = seed
        trainer.accelerator.dataloader_config.data_seed = seed
    if freeze:
        trainer.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=VISION_LR,
                                              weight_decay=training["weight-decay"])
        trainer.args.learning_rate = VISION_LR
    if recipe == "hybrid":
        trainer.optimizer = hybrid_optimizer(model, training)
    if order_of(arm) == "rationale-first" or ARMS[arm].get("select") == "last":
        # rationale-first: the teacher-forced val metric reads the boolean after the gold rationale, which names the
        # label; "last": a val set too small to choose on
        trainer.args.load_best_model_at_end = False
    if ARMS[arm].get("select") == "auroc":
        select_by_auroc(trainer, processor)
    if ARMS[arm].get("weight", 1) > 1:
        trainer = weighted(trainer, processor, data, ARMS[arm]["weight"], first=order_of(arm) != "rationale-first")
    if smoke:
        trainer.args.max_steps = 2

    trainer.train(resume_from_checkpoint=resume)
    print(f"peak GPU memory {torch.cuda.max_memory_reserved() / 1e9:.1f} GB", flush=True)
    for device in range(torch.cuda.device_count()):  # reserved counts the allocator's cache; allocated is the need
        print(f"cuda:{device} peak {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB allocated, "
              f"{torch.cuda.max_memory_reserved(device) / 1e9:.1f} GB reserved", flush=True)
    trainer.save_model()
    processor.save_pretrained(out)

    main.write_manifest(settings, data, run.id)
    manifest = json.loads((out / "manifest.json").read_text())
    root = image_root(arm)  # under /project when VF_RENDER_ROOT / VF_CORE_ROOT move it out of the repo
    manifest.update({"arm": arm, **ARMS[arm], "image_root": str(root.relative_to(ROOT) if root.is_relative_to(ROOT) else root),
                     "question": question_of(arm), "smoke": smoke, "vision_lr": VISION_LR if recipe in ("freeze", "hybrid") else None})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run.finish()


def select_by_auroc(trainer, processor):
    # keep the checkpoint with the best val AUROC of the answer's P(true). Argmax balanced accuracy on TUAR's and
    # TUEV's 31 / 16 val lines tied at <= 0.5 across every eval of some runs, and the Trainer kept the first (~20% of
    # training; review, 2026-10-17). AUROC rarely ties and is the reported metric. The balanced accuracy logged beside
    # it compares P(true) with P(false) only, where the default one takes the argmax over the vocabulary
    import torch

    words = main.boolean_token_ids(processor.tokenizer)
    true, false, both = sorted(words["true"]), sorted(words["false"]), sorted(words["true"] | words["false"])

    def margins(logits, labels):
        logits = logits[0] if isinstance(logits, tuple) else logits
        return torch.logsumexp(logits[..., true].float(), -1) - torch.logsumexp(logits[..., false].float(), -1)

    def compute(prediction):
        scores, labels = prediction.predictions[:, :-1], prediction.label_ids[:, 1:]
        found = [(row, np.isin(labels[row], both).nonzero()[0]) for row in range(len(labels))]
        rows = [(row, positions[0]) for row, positions in found if len(positions)]  # the answer: the first boolean
        y = np.array([labels[row, k] in words["true"] for row, k in rows], bool)
        margin = np.array([scores[row, k] for row, k in rows], float)
        if not y.any() or y.all():
            return {"label_auroc": 0.0, "label_balanced_accuracy": 0.0, "label_positions": len(rows)}
        return {"label_auroc": auroc(margin, y), "label_positions": len(rows),
                "label_balanced_accuracy": float(((margin[y] > 0).mean() + (margin[~y] <= 0).mean()) / 2)}

    trainer.compute_metrics, trainer.preprocess_logits_for_metrics = compute, margins
    trainer.args.metric_for_best_model = "label_auroc"


def weighted(base, processor, data, weight, first=False):
    # side/rationale-w10's recipe: weight `weight` on the answer's boolean token, 1 on every other target token,
    # loss = sum(w * CE) / target tokens. The boolean is the first true/false target token when the answer comes
    # first, the last when the rationale does; a multitask "explain" line has none and keeps weight 1
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    words = main.boolean_token_ids(processor.tokenizer)
    ids = torch.tensor(sorted(words["true"] | words["false"]))
    collate_base = base.data_collator

    def collate(examples):
        batch = collate_base(examples)
        labels = batch["labels"]
        weights = (labels != -100).float()
        for row in range(labels.shape[0]):
            positions = torch.isin(labels[row], ids).nonzero().flatten()
            if not len(positions):
                if not first:
                    raise ValueError("no true/false token in a target")
                continue
            weights[row, positions[0] if first else positions[-1]] = weight
        batch["label_weights"] = weights
        return batch

    class WeightedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = True  # compute_loss divides by num_items_in_batch itself

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            labels, weights = inputs.pop("labels"), inputs.pop("label_weights")
            outputs = model(**inputs)
            targets, weights = labels[:, 1:], weights[:, 1:]
            mask = targets != -100
            losses = F.cross_entropy(outputs.logits[:, :-1][mask].float(), targets[mask], reduction="none")
            count = num_items_in_batch if num_items_in_batch is not None else mask.sum()
            loss = (losses * weights[mask]).sum() / count
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer(model=base.model, args=base.args, train_dataset=data["train"], eval_dataset=data["validation"],
                           data_collator=collate, processing_class=processor, optimizers=(base.optimizer, None),
                           compute_metrics=base.compute_metrics,
                           preprocess_logits_for_metrics=base.preprocess_logits_for_metrics)


def load_for_scoring(arm, smoke):
    import torch
    from transformers import AutoModelForImageTextToText

    base = base_spec(config(), model_of(arm))
    if ARMS[arm].get("dpo_from"):  # the trained dataset model, then the DPO adapter on its language model
        from peft import PeftModel

        model, processor, _ = load_for_scoring(ARMS[arm]["dpo_from"], smoke)
        if not (out_dir(arm, smoke) / "manifest.json").exists():
            sys.exit(f"{out_dir(arm, smoke)} has no manifest.json: DPO has not finished")
        model = PeftModel.from_pretrained(model, str(out_dir(arm, smoke))).merge_and_unload()
        model.eval()
        return model, processor, base
    if ARMS[arm]["recipe"] in ("lora", "hybrid") and (LAST or STEP):
        # a checkpoint-N dir has the adapter and the processor but no manifest.json, which runner.load requires
        # (score-last-gm-2b-tuar/tuev failed on it, 2026-10-17)
        from peft import PeftModel

        checkpoint = model_dir(arm, smoke)
        model, _, _ = main.runner.load({"base": model_of(arm)})
        model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload()
        model.eval()
        drop_softcap(model)
        return model, main.runner.load_processor(str(checkpoint), base, base.get("trust-remote-code", False)), base
    if ARMS[arm]["recipe"] in ("lora", "hybrid"):
        model, processor, _ = main.runner.load({"base": model_of(arm), "checkpoint": str(model_dir(arm, smoke).relative_to(ROOT))})
        drop_softcap(model)  # trained without the cap, so scored without it
        return model, processor, base

    source = main.resolve_model(base["repo"]) if ARMS[arm]["recipe"] is None else str(model_dir(arm, smoke))
    if ARMS[arm]["recipe"] and not (out_dir(arm, smoke) / "manifest.json").exists():
        sys.exit(f"{out_dir(arm, smoke)} has no manifest.json: training has not finished")
    print(f"loading {source}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if ARMS[arm]["recipe"]:
        drop_softcap(model)
    processor_source = source if ARMS[arm]["recipe"] is None else str(out_dir(arm, smoke))  # checkpoints may lack it
    return model, main.runner.load_processor(processor_source, base, False), base


def score_batch(model):
    # Qwen's processor takes one flat image list for a batch of prompts; Gemma's reads a flat list as all the images of
    # one prompt ("inconsistently sized batches of images (1) and text (4)"), so the other families score one window
    # per forward
    return SCORE_BATCH if "qwen" in model.config.model_type else 1


def p_true(model, collator, ids, text, paths, answer=ANSWER):
    # the collator builds exactly the training input, answer included, so the boolean token's position is known
    # and only the logits before it are kept
    import torch

    every = torch.cat([ids["true"], ids["false"]]).cpu()
    batch = collator([{"instruction": text, "input": "", "output": answer, "images": [p]} for p in paths])
    targets = batch.pop("labels")
    positions = [int(torch.isin(row, every).nonzero()[0]) for row in targets]
    keep = targets.shape[1] - min(positions) + 1

    with torch.inference_mode():
        logits = model(**batch.to(model.device), logits_to_keep=keep).logits.float()

    offset = targets.shape[1] - keep
    probabilities = []
    for row, position in enumerate(positions):
        logp = logits[row, position - 1 - offset].log_softmax(-1)
        probabilities.append(float(torch.sigmoid(logp[ids["true"]].logsumexp(0) - logp[ids["false"]].logsumexp(0))))
    return probabilities


def rationale_structure():
    # maxLength makes the enforcer close the rationale and reach the boolean: the round-3 smoke model (2 steps)
    # rambled to the 512-token cap at 24 s/window, which would not fit 1,500 windows in the walltime. The teacher's
    # rationales are at most 1,112 characters (median 628)
    from pydantic import Field, create_model

    return create_model("SeizureRationale",
                        text_rationale=(str, Field(max_length=RATIONALE_CHARS, description="Text rationale for the prediction.")),
                        present=(bool, Field(description="Whether a seizure of any type is present.")))


def generate_p_true(model, processor, runner, prefix_fn, ids, text, image_path):
    # the deployed path: constrained decoding writes the rationale, then the boolean; P(true) from that step's raw
    # logits. None when the reply never reached a boolean (cut at max_new_tokens)
    import torch
    from PIL import Image

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = runner.render(processor, messages, True)
    with Image.open(image_path) as image:
        inputs = processor(text=[chat], images=[image.convert("RGB")], return_tensors="pt",
                           **runner.encode_kwargs(processor, chat)).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=runner.MAX_NEW_TOKENS, prefix_allowed_tokens_fn=prefix_fn.fresh(),
                             use_cache=True, do_sample=False, output_logits=True, return_dict_in_generate=True)
    generated = out.sequences[0, inputs["input_ids"].shape[1]:].cpu()
    every = torch.cat([ids["true"], ids["false"]]).cpu()
    steps = torch.isin(generated, every).nonzero().flatten()
    reply = processor.batch_decode([generated], skip_special_tokens=True)[0].strip()
    if not len(steps):
        return None, reply
    logp = out.logits[int(steps[-1])][0].float().log_softmax(-1)
    return float(torch.sigmoid(logp[ids["true"]].logsumexp(0) - logp[ids["false"]].logsumexp(0))), reply


def probe_windows():
    return sorted(r["rel"] for r in read_csv(PROBE) if r["ds"] == DATASET and r["split"] == "test")


def test_windows(arm="freeze"):
    if ARMS[arm]["lines"] in ("dataset", "core"):
        return sorted(r["images"][0] for r in dataset_lines(arm, "test"))
    rows = db().execute("SELECT DISTINCT path FROM pipeline WHERE dataset = ? AND scope = 'full'", (DATASET,))
    return sorted(p.split("/", 1)[1] for (p,) in rows)


def truth(arm="freeze"):
    if ARMS[arm].get("questions"):  # a dataset model: positive when any of its questions is
        labels = {}
        for r in dataset_lines(arm, "test"):
            labels[r["images"][0]] = labels.get(r["images"][0], False) or json.loads(r["output"])["present"]
        return labels
    if ARMS[arm]["lines"] in ("dataset", "core"):
        return {r["images"][0]: json.loads(r["output"])["present"] for r in dataset_lines(arm, "test")}
    return {r["path"]: any(r[c] == "true" for c in SEIZURES)
            for r in read_csv(ROOT / "datasets" / DATASET / "labels.csv")}


def score(arm, smoke):
    import torch

    if ARMS[arm].get("questions"):
        sys.exit(f"{arm} answers several questions: score its views ({', '.join(f'{arm}-{q}' for q in ARMS[arm]['questions'])})")
    results = base_dir(smoke) / "results"
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"scores-{tag(arm)}.csv"
    done = {r["path"] for r in read_csv(out)}
    rationale = order_of(arm) == "rationale-first"  # generation scoring; answer-first and multitask are teacher-forced
    todo = [p for p in (probe_windows() if rationale else test_windows(arm)) if p not in done]
    if smoke:
        todo = todo[::max(len(todo) // SMOKE_WINDOWS, 1)][:SMOKE_WINDOWS]
    print(f"arm {arm}: {len(todo)} windows to score, {len(done)} already done -> {out}", flush=True)

    if todo:
        model, processor, base = load_for_scoring(arm, smoke)
        collator = main.DataCollator(processor, image_root(arm), base)
        words = main.boolean_token_ids(processor.tokenizer)
        ids = {w: torch.tensor(sorted(words[w]), device=model.device) for w in ("true", "false")}
        text, labels = (rationale_prompt(arm) if order_of(arm) == "answer-first" else prompt_text(arm)), truth(arm)
        answer = AF_ANSWER if order_of(arm) == "answer-first" else ANSWER
        started = time.time()

        if rationale:
            text = rationale_prompt()
            prefix_fn = main.runner.enforcer(processor, rationale_structure())
            for n, path in enumerate(todo, 1):
                p, reply = generate_p_true(model, processor, main.runner, prefix_fn, ids, text, image_root(arm) / path)
                append_row(out, ["path", "p_true", "seizure", "reply"],
                           [path, "" if p is None else f"{p:.6f}", str(labels[path]).lower(), reply])
                if n % 50 == 0 or n == len(todo) or smoke and n <= 2:
                    print(f"{n}/{len(todo)} ({(time.time() - started) / n:.2f} s/window)", flush=True)
            todo = []

        batch = score_batch(model)
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            for path, p in zip(chunk, p_true(model, collator, ids, text, chunk, answer)):
                append_row(out, ["path", "p_true", "seizure"], [path, f"{p:.6f}", str(labels[path]).lower()])
            n = start + len(chunk)
            if n % 200 < batch or n == len(todo) or smoke and start == 0:
                print(f"{n}/{len(todo)} ({(time.time() - started) / n:.2f} s/window)", flush=True)

    if not smoke:
        summarise(tag(arm), out)


def rationale_eval_windows(arm="freeze"):
    # TUSZ: the probe's test windows (its test paths are the same in the core lists); other corpora: their test lines
    labels, rng, half = truth(arm), random.Random(0), RATIONALE_EVAL_WINDOWS // 2
    if dataset_of(arm) == "TUSZ":
        paths = [r["rel"] for r in read_csv(PROBE) if r["ds"] == DATASET and r["split"] == "test"]
    else:
        paths = list(labels)
    seizure = sorted(p for p in paths if labels[p])
    calm = sorted(p for p in paths if not labels[p])
    return rng.sample(seizure, min(half, len(seizure))) + rng.sample(calm, min(half, len(calm)))


def eval_formats(arm):
    # an untrained arm is evaluated in both trained formats, so each trained arm has a baseline in its own format; a
    # zero-shot grid arm only in its corpus's trained format
    if ARMS[arm].get("eval-format"):
        return [ARMS[arm]["eval-format"]]
    if ARMS[arm]["recipe"] is None:
        return ["answer-first", "explain"]
    return [{"multitask": "explain", "rationale-first": "rationale-first"}.get(order_of(arm), "answer-first")]


def eval_structure(fmt):
    from pydantic import Field, create_model

    text = (str, Field(max_length=RATIONALE_CHARS, description="Text rationale for the prediction."))
    present = (bool, Field(description="Whether a seizure of any type is present."))
    if fmt == "explain":
        return create_model("Explanation", text_rationale=text)
    if fmt == "answer-first":
        return create_model("AnswerRationale", present=present, text_rationale=text)
    return create_model("RationaleAnswer", text_rationale=text, present=present)


def reference_nll(model, collator, ids, prompt, output, path):
    # mean NLL per token of a gold answer, the boolean token left out: with the reference rationale as the
    # output, how well the model has learned the teacher's text for this window
    import torch

    batch = collator([{"instruction": prompt, "input": "", "output": output, "images": [path]}])
    targets = batch.pop("labels")[0]
    positions = (targets != -100).nonzero().flatten()
    positions = positions[~torch.isin(targets[positions], torch.cat([ids["true"], ids["false"]]).cpu())]
    keep = len(targets) - int(positions[0]) + 1
    with torch.inference_mode():
        logits = model(**batch.to(model.device), logits_to_keep=keep).logits[0].float()
    offset = len(targets) - keep
    logp = logits[positions - 1 - offset].log_softmax(-1)
    nll = -logp.gather(1, targets[positions].to(logp.device)[:, None]).squeeze(1)
    return float(nll.mean()), len(positions)


def generate_reply(model, processor, runner, prefix_fn, text, image_path):
    import torch
    from PIL import Image

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = runner.render(processor, messages, True)
    with Image.open(image_path) as image:
        inputs = processor(text=[chat], images=[image.convert("RGB")], return_tensors="pt",
                           **runner.encode_kwargs(processor, chat)).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=runner.MAX_NEW_TOKENS, prefix_allowed_tokens_fn=prefix_fn.fresh(),
                             use_cache=True, do_sample=False)
    return processor.batch_decode([out[0, inputs["input_ids"].shape[1]:]], skip_special_tokens=True)[0].strip()


def rationale_eval(arm, smoke):
    # per window: the reference rationale's NLL under this model, and the model's own generated rationale
    import torch

    windows = rationale_eval_windows(arm)
    windows = windows[:3] + windows[-3:] if smoke else windows
    wanted = set(windows)
    # a dataset arm is graded against the new teacher's test-side references (teacher.py assemble-refs), and requires
    # them; every other arm keeps labels.csv's gemma3:12b text
    refs = RATIONALES / dataset_of(arm) / "references-test.jsonl"
    if ARMS[arm].get("questions") and not refs.exists():
        sys.exit(f"{arm}: no {refs} yet (teacher.py assemble-refs)")
    source = {r["path"]: r["text_rationale"] for r in main.read_jsonl(refs)} if ARMS[arm].get("questions") else \
        {r["path"]: r["ground_truth_rationale"] for r in read_csv(ROOT / "datasets" / dataset_of(arm) / "labels.csv")}
    references = {path: CHANNEL.sub(r"\1", strip_label(text, dataset_of(arm))) for path, text in source.items() if path in wanted}
    missing = [w for w in windows if not references.get(w)]
    if missing:
        sys.exit(f"{len(missing)} rationale-eval windows have no reference rationale in labels.csv, e.g. {missing[:3]}")
    labels = truth(arm)
    model, processor, base = load_for_scoring(arm, smoke)
    model.eval()
    collator = main.DataCollator(processor, image_root(arm), base)
    words = main.boolean_token_ids(processor.tokenizer)
    ids = {w: torch.tensor(sorted(words[w]), device=model.device) for w in ("true", "false")}
    results = base_dir(smoke) / "results"
    results.mkdir(parents=True, exist_ok=True)
    header = ["path", "seizure", "reference_nll", "reference_tokens", "generated_present", "rationale_chars", "reply"]

    for fmt in eval_formats(arm):
        out = results / f"rationale-{tag(arm)}-{fmt}.csv"
        done = {r["path"] for r in read_csv(out)}
        prompt = explain_prompt(arm) if fmt == "explain" else rationale_prompt(arm)
        prefix_fn = main.runner.enforcer(processor, eval_structure(fmt))
        started, todo = time.time(), [w for w in windows if w not in done]
        print(f"arm {arm} ({fmt}): rationale evaluation on {len(todo)} windows -> {out}", flush=True)
        for n, path in enumerate(todo, 1):
            gold, reference = labels[path], references[path]
            target = {"explain": {"text_rationale": reference},
                      "answer-first": {"present": gold, "text_rationale": reference},
                      "rationale-first": {"text_rationale": reference, "present": gold}}[fmt]
            nll, tokens = reference_nll(model, collator, ids, prompt, json.dumps(target), path)
            reply = generate_reply(model, processor, main.runner, prefix_fn, prompt, image_root(arm) / path)
            try:
                parsed = json.loads(reply)
            except json.JSONDecodeError:
                parsed = {}
            append_row(out, header, [path, str(gold).lower(), f"{nll:.4f}", tokens,
                                     str(parsed.get("present", "")).lower(), len(parsed.get("text_rationale", "")), reply])
            if n % 25 == 0 or n == len(todo) or smoke and n <= 2:
                print(f"{n}/{len(todo)} ({(time.time() - started) / n:.1f} s/window)", flush=True)
        if not smoke:
            summarise_rationale(tag(arm), fmt, out)


def dpo_windows(arm):
    # training windows with a teacher rationale (the review sets the model's rationale against the teacher's), half
    # with the arm's class and half without, as (path, present, teacher rationale as trained). A dataset model: per
    # question 15 with and 15 without, de-duplicated, present = {question: answer} for every question of the window
    rows = lines("train", False, arm)
    if ARMS[arm].get("questions"):
        answers, teacher = defaultdict(dict), {}
        for r in rows:
            if r["question"] == "explain":
                teacher[r["images"][0]] = json.loads(r["output"])["text_rationale"]
            else:
                answers[r["images"][0]][r["question"]] = json.loads(r["output"])["present"]
        # a window can be in one question's training lines and not another's: every question's answer comes from the
        # window's TUH labels in the rationale file (has_<class> / is_abnormal), so a record shows them all
        flags = {Path(r["images"][0]).name: json.loads(r["output"])
                 for r in main.read_jsonl(RATIONALES / dataset_of(arm) / "sft_train.jsonl")}
        for path in teacher:
            for question in ARMS[arm]["questions"]:
                key = "is_abnormal" if question == "abnormal" else f"has_{question}"
                answers[path].setdefault(question, flags[Path(path).name][key])
        rng, half, picks = random.Random(0), DPO_PER_QUESTION.get(dataset_of(arm), DPO_WINDOWS) // 2, []
        for question in ARMS[arm]["questions"]:
            for value in (True, False):
                pool = sorted(p for p in teacher if answers[p].get(question) is value and p not in picks)
                picks += rng.sample(pool, min(half, len(pool)))
        return [(p, answers[p], teacher[p]) for p in picks]
    present = {r["images"][0]: json.loads(r["output"])["present"] for r in rows if r["question"] != "explain"}
    teacher = {r["images"][0]: json.loads(r["output"])["text_rationale"] for r in rows if r["question"] == "explain"}
    rng, half = random.Random(0), DPO_WINDOWS // 2
    picks = []
    for value in (True, False):
        pool = sorted(p for p in teacher if present.get(p) is value)
        picks += rng.sample(pool, min(half, len(pool)))
    return [(p, present[p], teacher[p]) for p in picks]


def dpo_pairs(arm, smoke):
    # (path, chosen, rejected) for this model from the neurologist's submitted answers: the rationale he preferred
    # against the other. Ties and "both wrong" give no pair; a correction is kept for the paper, not trained on. Only
    # the dataset's training windows: the test-side review must never reach DPO. A smoke run pairs the teacher's
    # rationale (chosen) with the model's from the smoke dpo-gen file, to run the code before any answer exists
    sft = ARMS[arm]["dpo_from"]
    if smoke:
        rows = read_csv(base_dir(True) / "results" / f"dpo-gen-{sft}.csv")
        pairs = [(r["path"], r["teacher"], reply_text(r["reply"])) for r in rows]
    else:
        pairs = [(r["path"], *((r["rationale_a"], r["rationale_b"]) if r["preference"] == "A" else
                               (r["rationale_b"], r["rationale_a"])))
                 for r in main.read_jsonl(DPO_ANSWERS) if r["model"] == sft and r.get("preference") in ("A", "B")]
    training = {r["images"][0] for r in lines("train", False, sft)}
    outside = [path for path, _, _ in pairs if path not in training]
    if outside:
        sys.exit(f"{arm}: {len(outside)} pairs are not {dataset_of(arm)} training windows, e.g. {outside[:3]}")
    return pairs


def reply_text(reply):
    try:
        return json.loads(reply)["text_rationale"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return reply


def target_logp(model, collator, prompt, output, path):
    # the summed log-probability of the target's tokens and their count, with gradients when they are on
    batch = collator([{"instruction": prompt, "input": "", "output": output, "images": [path]}])
    targets = batch.pop("labels")[0]
    positions = (targets != -100).nonzero().flatten()
    keep = len(targets) - int(positions[0]) + 1
    logits = model(**batch.to(model.device), logits_to_keep=keep, use_cache=False).logits[0].float()
    logp = logits[positions - 1 - (len(targets) - keep)].log_softmax(-1)
    return logp.gather(1, targets[positions].to(logp.device)[:, None]).sum(), len(positions)


def dpo_train(arm, smoke):
    import torch
    from peft import LoraConfig, get_peft_model

    out = out_dir(arm, smoke)
    if (out / "manifest.json").exists():
        print(f"{out} is already trained (manifest.json exists)", flush=True)
        return
    pairs = dpo_pairs(arm, smoke)
    if not pairs:
        sys.exit(f"{arm}: no preference pairs for {ARMS[arm]['dpo_from']} in {DPO_ANSWERS}")

    settings, sft = DPO_SETTINGS, ARMS[arm]["dpo_from"]
    model, processor, base = load_for_scoring(sft, smoke)
    language, _ = main.runner.lora_targets(model)
    model = get_peft_model(model, LoraConfig(r=settings["rank"], lora_alpha=settings["lora-alpha"], lora_dropout=0.0,
                                             bias="none", task_type="CAUSAL_LM", target_modules=language))
    model.eval()  # no dropout anywhere, so the policy equals the reference at step 0
    collator, prompt = main.DataCollator(processor, image_root(arm), base), explain_prompt(arm)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings["learning-rate"],
                                  weight_decay=0.0)
    rng, log = random.Random(ARMS[arm].get("seed", 0)), []
    print(f"{arm}: DPO on {len(pairs)} pairs of {sft} -> {out}, {settings}", flush=True)

    for epoch in range(1 if smoke else settings["epochs"]):
        order = rng.sample(pairs, len(pairs))
        wins, margins = 0, []
        for n, (path, chosen, rejected) in enumerate(order, 1):
            targets = [json.dumps({"text_rationale": CHANNEL.sub(r"\1", text)}) for text in (chosen, rejected)]
            with torch.no_grad():
                with model.disable_adapter():
                    reference = [target_logp(model, collator, prompt, t, path) for t in targets]
                policy = [target_logp(model, collator, prompt, t, path) for t in targets]
            margin = sum(sign * float(p - r) / tokens for sign, (p, tokens), (r, _) in zip((1, -1), policy, reference))
            # the gradient of -log sigmoid(beta * margin) is -beta * sigmoid(-beta * margin) * d(margin): with that weight
            # fixed, the chosen and rejected terms back-propagate one at a time, so only one graph is held (both at
            # once peaked at 81.9 GB of an 80 GB H100 in the smoke, 2026-10-25)
            weight = settings["beta"] * float(torch.sigmoid(torch.tensor(-settings["beta"] * margin)))
            for sign, target in zip((1, -1), targets):
                logp, tokens = target_logp(model, collator, prompt, target, path)
                loss = -sign * weight * logp / tokens - (settings["alpha"] * logp / tokens if sign == 1 else 0)
                (loss / settings["accumulation"]).backward()
            if n % settings["accumulation"] == 0 or n == len(order):
                optimizer.step()
                optimizer.zero_grad()
            wins += float(margin) > 0
            margins.append(float(margin))
        log.append({"epoch": epoch + 1, "reward_accuracy": round(wins / len(order), 4),
                    "margin": round(float(np.mean(margins)), 5)})
        print(json.dumps(log[-1]), flush=True)

    print(f"peak GPU memory {torch.cuda.max_memory_reserved() / 1e9:.1f} GB", flush=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    manifest = {"arm": arm, **ARMS[arm], "pairs": len(pairs), "settings": settings, "log": log, "smoke": smoke}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def dpo_gen(arm, smoke):
    # the trained model's explain-format rationale on DPO_WINDOWS training windows: with the teacher's, the pairs the
    # expert ranks for DPO (argilla/rationales.py build-dpo). Training windows only, so test windows never reach DPO
    windows = dpo_windows(arm)
    windows = windows[:2] if smoke else windows
    model, processor, base = load_for_scoring(arm, smoke)
    model.eval()
    prompt, prefix_fn = explain_prompt(arm), main.runner.enforcer(processor, eval_structure("explain"))
    out = base_dir(smoke) / "results" / f"dpo-gen-{tag(arm)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["path"] for r in read_csv(out)}
    started, todo = time.time(), [w for w in windows if w[0] not in done]
    print(f"arm {arm}: DPO generation on {len(todo)} training windows -> {out}", flush=True)
    for n, (path, present, teacher) in enumerate(todo, 1):
        reply = generate_reply(model, processor, main.runner, prefix_fn, prompt, image_root(arm) / path)
        present = json.dumps(present) if isinstance(present, dict) else str(present).lower()
        append_row(out, ["path", "present", "teacher", "reply"], [path, present, teacher, reply])
        if n % 10 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} ({(time.time() - started) / n:.1f} s/window)", flush=True)


def review_gen(arm, smoke):
    # the dataset model's explain-format rationale on its corpus's review windows (rerender.py review: training-patient
    # windows in no core list, so no model saw a rationale for them), for the expert review of the top models (Yousif,
    # 2026-10-27). Same prompt and constrained decoding as dpo_gen, one row per window so a stopped run resumes; on a Mac
    # it runs on the GPU (MPS), and while the file VF_PAUSE names exists it waits between windows
    import torch

    windows = [(r["path"], r["present"]) for r in read_csv(SIDE / "data" / "core" / f"{dataset_of(arm)}-review.csv")]
    windows = windows[:2] if smoke else windows
    out = base_dir(smoke) / "results" / f"review-gen-{tag(arm)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["path"] for r in read_csv(out)}
    todo = [w for w in windows if w[0] not in done]
    if not todo:
        print(f"arm {arm}: all {len(windows)} review windows done -> {out}", flush=True)
        return

    model, processor, base = load_for_scoring(arm, smoke)
    if not torch.cuda.is_available() and torch.backends.mps.is_available() and model.device.type != "mps":
        model = model.to("mps")
    model.eval()
    prompt, prefix_fn = explain_prompt(arm), main.runner.enforcer(processor, eval_structure("explain"))
    pause = os.environ.get("VF_PAUSE")
    started = time.time()
    print(f"arm {arm}: review generation on {len(todo)} windows ({model.device}) -> {out}", flush=True)
    for n, (path, present) in enumerate(todo, 1):
        while pause and Path(pause).exists():
            time.sleep(30)
        reply = generate_reply(model, processor, main.runner, prefix_fn, prompt, image_root(arm) / path)
        append_row(out, ["path", "present", "reply"], [path, present, reply])
        if n % 10 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} ({(time.time() - started) / n:.1f} s/window)", flush=True)


def summarise_rationale(arm, fmt, path):
    rows = list({r["path"]: r for r in read_csv(path)}.values())
    nll = np.array([float(r["reference_nll"]) for r in rows])
    chars = np.array([int(r["rationale_chars"]) for r in rows])
    summary = {"arm": arm, "format": fmt, "windows": len(rows), "reference_nll": round(float(nll.mean()), 4),
               "reference_nll_seizure": round(float(nll[[r["seizure"] == "true" for r in rows]].mean()), 4),
               "reference_nll_background": round(float(nll[[r["seizure"] != "true" for r in rows]].mean()), 4),
               "rationale_chars_median": int(np.median(chars)), "rationale_at_cap": int((chars >= RATIONALE_CHARS).sum())}
    answered = [r for r in rows if r["generated_present"] in ("true", "false")]
    if answered:
        y = np.array([r["seizure"] == "true" for r in answered])
        g = np.array([r["generated_present"] == "true" for r in answered])
        summary["generated_answer_ba"] = round(float((g[y].mean() + (~g[~y]).mean()) / 2), 4)
    with (path.parent / f"summary-rationale-{arm}-{fmt}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary), flush=True)


def tuep_scan(arm, smoke):
    # a TUEV epileptiform model scores every scanned TUEP test window (same scorer and prompt as its own test set)
    import torch

    assert ARMS[arm]["images"] == "core" and dataset_of(arm) == "TUEV", f"{arm} is not a core TUEV model"
    results = base_dir(smoke) / "results"
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"tuep-scan-{tag(arm)}.csv"
    rows = main.read_jsonl(TUEP_SCAN)
    labels = {r["images"][0]: r["epilepsy"] for r in rows}
    done = {r["path"] for r in read_csv(out)}
    todo = [r["images"][0] for r in rows if r["images"][0] not in done]
    if smoke:
        todo = todo[:SMOKE_WINDOWS]
    print(f"arm {arm}: {len(todo)} TUEP scan windows to score, {len(done)} already done -> {out}", flush=True)

    if todo:
        model, processor, base = load_for_scoring(arm, smoke)
        collator = main.DataCollator(processor, IMAGE_ROOTS["core"] / "TUEP", base)
        words = main.boolean_token_ids(processor.tokenizer)
        ids = {w: torch.tensor(sorted(words[w]), device=model.device) for w in ("true", "false")}
        answer_first = order_of(arm) == "answer-first"
        text, answer = (rationale_prompt(arm), AF_ANSWER) if answer_first else (prompt_text(arm), ANSWER)
        started = time.time()
        batch = score_batch(model)
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            for path, p in zip(chunk, p_true(model, collator, ids, text, chunk, answer)):
                append_row(out, ["path", "p_true", "epilepsy"], [path, f"{p:.6f}", str(labels[path]).lower()])
            n = start + len(chunk)
            if n % 400 < batch or n == len(todo):
                print(f"{n}/{len(todo)} ({(time.time() - started) / n:.2f} s/window)", flush=True)

    if not smoke:
        summarise_tuep_scan(tag(arm), out)


def summarise_tuep_scan(arm, path):
    # each recording is scored by the mean of its top-k window spike probabilities over exactly SCAN_WINDOWS evenly
    # spaced windows, and a patient by the mean over their recordings. The top of more draws is higher, and window
    # counts differ by class (epileptic patients have more recordings, non-epileptic recordings are more often short):
    # pooling a patient's windows gave random scores a patient AUROC of 0.61, unequal counts per recording 0.55
    if not TUEP_EXCLUDE.exists():
        sys.exit(f"{TUEP_EXCLUDE} is missing: the TUEP patients that share recordings with TUEV training must be left out")
    listed = {r["images"][0] for r in main.read_jsonl(TUEP_SCAN)}
    rows = [r for r in {r["path"]: r for r in read_csv(path)}.values() if r["path"] in listed]
    excluded = {r["patient"] for r in read_csv(TUEP_EXCLUDE)}
    scanned = {Path(r["path"]).stem.rsplit("_", 2)[0] for r in rows}
    if excluded - scanned:
        sys.exit(f"excluded patients not in the scan: {sorted(excluded - scanned)}")
    windows, label = {}, {}
    for r in rows:
        recording = Path(r["path"]).stem.rsplit("_", 1)[0]
        patient = recording.rsplit("_", 1)[0]
        if patient in excluded:
            continue
        windows.setdefault(recording, []).append((int(Path(r["path"]).stem.rsplit("_", 1)[1]), float(r["p_true"])))
        label[recording] = label[patient] = r["epilepsy"] == "true"
    for r in windows:  # a recording scanned in full (it has window 0) loses its first and last window, like the rest
        if min(i for i, _ in windows[r]) == 0:
            last = max(i for i, _ in windows[r])
            windows[r] = [(i, p) for i, p in windows[r] if 0 < i < last]
    short = [r for r in windows if len(windows[r]) < SCAN_WINDOWS]
    for r in windows:  # SCAN_WINDOWS spread through the recording, in time order
        ordered = [p for _, p in sorted(windows[r])]
        keep = np.linspace(0, len(ordered) - 1, SCAN_WINDOWS).round().astype(int)
        windows[r] = [ordered[i] for i in keep] if len(ordered) >= SCAN_WINDOWS else []
    recordings = sorted(r for r in windows if windows[r])
    y_recording = np.array([label[r] for r in recordings])
    all_p = np.array([p for r in recordings for p in windows[r]])
    all_y = np.array([label[r] for r in recordings for _ in windows[r]])
    summary = {"arm": arm, "excluded_patients": len(excluded & scanned), "short_recordings_left_out": len(short),
               "windows": len(all_p), "recordings": len(recordings),
               "auroc_window": round(auroc(all_p, all_y), 4)}
    for k in SCAN_TOP:
        score = np.array([np.mean(sorted(windows[r])[-k:]) for r in recordings])
        summary[f"auroc_recording_top{k}"] = round(auroc(score, y_recording), 4)
    top5 = {r: np.mean(sorted(windows[r])[-5:]) for r in recordings}
    patients = sorted({r.rsplit("_", 1)[0] for r in recordings})
    score = np.array([np.mean([top5[r] for r in recordings if r.rsplit("_", 1)[0] == q]) for q in patients])
    y = np.array([label[q] for q in patients])
    rng = np.random.default_rng(0)
    values = [auroc(score[idx], y[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAP))]
    predicted = score >= 0.5
    summary.update({"patients": len(patients), "epilepsy_patients": int(y.sum()),
                    "auroc_patient": round(auroc(score, y), 4),
                    "auroc_patient_ci_low": round(float(np.nanpercentile(values, 2.5)), 4),
                    "auroc_patient_ci_high": round(float(np.nanpercentile(values, 97.5)), 4),
                    "balanced_accuracy_patient": round(float((predicted[y].mean() + (~predicted[~y]).mean()) / 2), 4),
                    "sensitivity_patient": round(float(predicted[y].mean()), 4),
                    "specificity_patient": round(float((~predicted[~y]).mean()), 4)})
    with (path.parent / f"summary-tuep-scan-{arm}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary), flush=True)


def score_windows(arm, smoke, todo, labels, out):
    # score()'s teacher-forced P(true), on any list of windows, appended to out
    import torch

    model, processor, base = load_for_scoring(arm, smoke)
    collator = main.DataCollator(processor, image_root(arm), base)
    words = main.boolean_token_ids(processor.tokenizer)
    ids = {w: torch.tensor(sorted(words[w]), device=model.device) for w in ("true", "false")}
    text = rationale_prompt(arm) if order_of(arm) == "answer-first" else prompt_text(arm)
    answer = AF_ANSWER if order_of(arm) == "answer-first" else ANSWER
    started, batch = time.time(), score_batch(model)
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        for path, p in zip(chunk, p_true(model, collator, ids, text, chunk, answer)):
            append_row(out, ["path", "p_true", "seizure"], [path, f"{p:.6f}", str(labels[path]).lower()])
        n = start + len(chunk)
        if n % 400 < batch or n == len(todo):
            print(f"{n}/{len(todo)} ({(time.time() - started) / n:.2f} s/window)", flush=True)


def image_check(arm):
    # the untrained base on 8 of the arm's test windows (4 with the class, 4 without): a P(true) that is the same for all
    # of them means the image never reaches the model (MiniCPM-V 4.6, 2026-10-19), so the job stops before training
    import torch

    labels = truth(arm)
    windows = sorted(p for p in labels if labels[p])[:4] + sorted(p for p in labels if not labels[p])[:4]
    model, processor, _ = main.runner.load({"base": model_of(arm)})
    base = base_spec(config(), model_of(arm))
    collator = main.DataCollator(processor, image_root(arm), base)
    words = main.boolean_token_ids(processor.tokenizer)
    ids = {w: torch.tensor(sorted(words[w]), device=model.device) for w in ("true", "false")}
    text = rationale_prompt(arm) if order_of(arm) == "answer-first" else prompt_text(arm)
    answer = AF_ANSWER if order_of(arm) == "answer-first" else ANSWER
    p = [q for path in windows for q in p_true(model, collator, ids, text, [path], answer)]
    print(f"image check, {model_of(arm)}: P(true) " + " ".join(f"{q:.6f}" for q in p), flush=True)
    if len({round(q, 6) for q in p}) == 1:
        sys.exit(f"{arm}: the same P(true) for every window, the image does not reach the model: stop")


def heldout_test(arm, smoke):
    # the final TUSZ test: every shipped window of the 62 patients in no core split (rerender.py heldout)
    assert dataset_of(arm) == "TUSZ" and ARMS[arm]["images"] == "core", f"{arm} is not a core TUSZ arm"
    results = base_dir(smoke) / "results"
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"heldout-{tag(arm)}.csv"
    labels = {r["images"][0]: json.loads(r["output"])["present"] for r in main.read_jsonl(HELDOUT)}
    done = {r["path"] for r in read_csv(out)}
    todo = [p for p in labels if p not in done]
    if smoke:
        todo = todo[::max(len(todo) // SMOKE_WINDOWS, 1)][:SMOKE_WINDOWS]
    print(f"arm {arm}: {len(todo)} held-out windows to score, {len(done)} already done -> {out}", flush=True)
    if todo:
        score_windows(arm, smoke, todo, labels, out)
    if not smoke:
        summarise(f"heldout-{tag(arm)}", out)


def background_test(arm, smoke):
    # the same teacher-forced scorer as score(), on the interior background windows
    results = base_dir(smoke) / "results"
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"background-{arm}.csv"
    done = {r["path"] for r in read_csv(out)}
    todo = [r["path"] for r in read_csv(BACKGROUND) if r["path"] not in done]
    if smoke:
        todo = todo[:SMOKE_WINDOWS]
    print(f"arm {arm}: {len(todo)} interior background windows to score, {len(done)} already done -> {out}", flush=True)

    if todo:
        score_windows(arm, smoke, todo, dict.fromkeys(todo, False), out)

    if not smoke:
        summarise_background(arm, out)


def summarise_background(arm, path):
    # seizure windows from the arm's full test scores; background: the old (97.8% edge) set and the new interior set
    scored = read_csv(base_dir(False) / "results" / f"scores-{arm}.csv")
    seizure = [r for r in scored if r["seizure"] == "true"]
    old = [r for r in scored if r["seizure"] != "true"]
    new = list({r["path"]: r for r in read_csv(path)}.values())
    meta = {r["path"]: r for r in read_csv(BACKGROUND)}
    p_seizure = np.array([float(r["p_true"]) for r in seizure])
    p_new = np.array([float(r["p_true"]) for r in new])
    p_old = np.array([float(r["p_true"]) for r in old])

    def specificity(rows):
        return round(float(np.mean([float(r["p_true"]) < 0.5 for r in rows])), 4) if rows else ""

    combined = np.concatenate([p_seizure, p_new])
    positive = np.concatenate([np.ones(len(p_seizure), bool), np.zeros(len(p_new), bool)])
    patients = [Path(r["path"]).stem.split("_")[0] for r in seizure + new]
    low, high = patient_ci(combined, positive, patients)
    sensitivity = float((p_seizure >= 0.5).mean())
    near = lambda r, lo, hi: meta[r["path"]]["windows_to_seizure"] != "" and lo <= int(meta[r["path"]]["windows_to_seizure"]) <= hi
    summary = {
        "arm": arm, "seizure_windows": len(p_seizure), "interior_background": len(p_new),
        "auroc_interior_background": round(auroc(combined, positive), 4),
        "auroc_ci_low": round(low, 4), "auroc_ci_high": round(high, 4),
        "auroc_old_background": round(auroc(np.concatenate([p_seizure, p_old]),
                                            np.concatenate([np.ones(len(p_seizure), bool), np.zeros(len(p_old), bool)])), 4),
        "sensitivity": round(sensitivity, 4),
        "balanced_accuracy_interior": round((sensitivity + float((p_new < 0.5).mean())) / 2, 4),
        "specificity_old_background": specificity(old),
        "specificity_interior": specificity(new),
        "specificity_seizure_free_recordings": specificity([r for r in new if meta[r["path"]]["seizure_recording"] != "True"]),
        "specificity_1_2_windows_from_seizure": specificity([r for r in new if near(r, 1, 2)]),
        "specificity_3_10_windows_from_seizure": specificity([r for r in new if near(r, 3, 10)]),
        "specificity_over_10_windows_from_seizure": specificity([r for r in new if near(r, 11, 10 ** 6)]),
    }
    with (path.parent / f"summary-background-{arm}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary), flush=True)


def auroc(scores, positive):
    # Mann-Whitney, ties at their average rank
    scores, positive = np.asarray(scores, float), np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return float("nan")
    order = scores.argsort()
    _, inverse, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = (np.bincount(inverse, np.arange(1, len(scores) + 1)) / counts)[inverse]
    n = positive.sum()
    return float((ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(scores) - n)))


def patient_ci(scores, positive, patients):
    # 95% interval over 1,000 resamples of patients, windows of a patient kept together
    rng = np.random.default_rng(0)
    groups = {}
    for i, patient in enumerate(patients):
        groups.setdefault(patient, []).append(i)
    keys = list(groups)
    values = []
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        values.append(auroc(scores[idx], positive[idx]))
    return np.nanpercentile(values, [2.5, 97.5])


def summarise(arm, path):
    rows = list({r["path"]: r for r in read_csv(path)}.values())
    unscored = [r for r in rows if r["p_true"] == ""]
    if unscored:
        print(f"{len(unscored)} windows whose reply never reached the boolean are left out", flush=True)
    rows = [r for r in rows if r["p_true"] != ""]
    p = np.array([float(r["p_true"]) for r in rows])
    y = np.array([r["seizure"] == "true" for r in rows])
    names = [Path(r["path"]).stem for r in rows]
    patients = [n.split("_")[0] for n in names]
    recordings = ["_".join(n.split("_")[:2]) for n in names]
    with_seizure = {rec for rec, positive in zip(recordings, y) if positive}
    within = np.array([rec in with_seizure for rec in recordings])
    probe_paths = {r["rel"] for r in read_csv(PROBE) if r["ds"] == DATASET and r["split"] == "test"}
    in_probe = np.array([r["path"] in probe_paths for r in rows])
    low, high = patient_ci(p, y, patients)
    predicted = p >= 0.5

    summary = {
        "arm": arm, "windows": len(rows), "seizure_windows": int(y.sum()), "patients": len(set(patients)),
        "auroc": round(auroc(p, y), 4), "auroc_ci_low": round(low, 4), "auroc_ci_high": round(high, 4),
        "auroc_within_seizure_recordings": round(auroc(p[within], y[within]), 4),
        "auroc_probe_windows": round(auroc(p[in_probe], y[in_probe]), 4) if in_probe.any() else "",
        "probe_windows": int(in_probe.sum()),
        "balanced_accuracy_at_0.5": round(float((predicted[y].mean() + (~predicted[~y]).mean()) / 2), 4),
        "predicted_seizure_fraction": round(float(predicted.mean()), 4),
    }
    with (path.parent / f"summary-{arm}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary), flush=True)


def submit_job(cfg, arm, dry_run, arms=()):
    name = f"eeg-vlm-vision-freeze-{arm}" + ("-" + "-".join(arms) if arm in PSEUDO else "")
    if name in queued():
        sys.exit(f"{name} is already queued or running")
    if not (SOURCE / "perclass-train.jsonl").exists():
        sys.exit("no training lines: python side/multilabel-pilot/build.py")
    if arm in SINGLE and len(arms) != 1 and not (arm == "score-step" and len({out_dir(a) for a in arms}) == 1):
        sys.exit(f"{arm} takes one arm per job (the job asks for that arm's GPU), or for score-step one model's views")
    if arm == "score-step" and not (STEP and (out_dir(arms[0]) / f"checkpoint-{STEP}").is_dir()):
        sys.exit(f"score-step needs --step N with an existing {out_dir(arms[0])}/checkpoint-N")
    after = os.environ.get("VF_AFTER")  # a job id this one waits for (afterok): dpo-gen queued behind its arm's training
    if arm in ("score-last", "score-step", "dpo-gen") and not (out_dir(arms[0]) / "manifest.json").exists() and not after:
        sys.exit(f"{arms[0]} has not finished training")
    for a in SMOKE_ARMS if arm == "smoke" else (arms if arm in PSEUDO else (arm,)):
        missing = [] if arm in ("background-test", "tuep-scan", "final-test") else missing_images(a)
        if arm == "background-test" or arm == "final-test" or ARMS.get(a, {}).get("final-test"):
            missing += [r["path"] for r in read_csv(BACKGROUND) if not (image_root(a) / r["path"]).exists()]
        if arm == "final-test" or ARMS.get(a, {}).get("final-test"):
            missing += [r["images"][0] for r in main.read_jsonl(HELDOUT) if not (image_root(a) / r["images"][0]).exists()]
        if arm == "final-test" and not (base_dir(False) / "results" / f"scores-{a}.csv").exists():
            sys.exit(f"{a} has no test scores (the background summary reads them)")
        if arm == "tuep-scan":
            if ARMS[a]["images"] != "core" or dataset_of(a) != "TUEV":
                sys.exit(f"{a} is not a TUEV model on the core images")
            missing += [r["images"][0] for r in main.read_jsonl(TUEP_SCAN)
                        if not (IMAGE_ROOTS["core"] / "TUEP" / r["images"][0]).exists()]
            if not (out_dir(a) / "manifest.json").exists():
                sys.exit(f"{a} has not finished training")
        if missing:
            sys.exit(f"{a}: {len(missing)} images missing under {image_root(a)}, e.g. {missing[:3]}")

    me = f"python {SIDE}/run.py"
    modes = lambda a: ("score", "heldout", "background", "rationale-eval") if ARMS[a].get("final-test") else \
        ("score", "rationale-eval") if ARMS[a].get("eval-format") else \
        ("score",) if ARMS[a]["recipe"] is None else \
        ("train", "score", "rationale-eval") if ARMS[a]["recipe"] == "hybrid" else ("train", "score")
    steps = {a: [f"{me} {a} --image-check"] * bool(ARMS[a].get("image-check")) +
                [f"{me} {a} --train --smoke"] * bool(ARMS[a].get("smoke-first")) + [f"{me} {a} --{m}" for m in modes(a)]
             for a in ARMS}
    for a in ARMS:
        if ARMS[a].get("dpo_from") and ARMS[a].get("questions"):
            views = [f"{a}-{q}" for q in ARMS[a]["questions"]]
            steps[a] = [f"{me} {a} --dpo"] + [f"{me} {v} --score" for v in views] + [f"{me} {a} --rationale-eval"]
        elif ARMS[a].get("questions"):
            views = [f"{a}-{q}" for q in ARMS[a]["questions"]]
            # the 8B dataset models feed DPO: their review rationales are written right after training, in the same job
            steps[a] = ([f"{me} {views[0]} --image-check"] * bool(ARMS[a].get("image-check")) +
                        [f"{me} {a} --train --smoke"] * bool(ARMS[a].get("smoke-first")) + [f"{me} {a} --train"] +
                        [f"{me} {a} --dpo-gen"] * (model_of(a) == "qwen3-vl:8b-instruct" and not ARMS[a].get("seed")) +
                        [f"{me} {v} --score" for v in views] + [f"{me} {a} --rationale-eval"])
    steps["smoke"] = [f"{me} {a} --{m} --smoke" for a in SMOKE_ARMS for m in modes(a)]
    steps["rationale-eval"] = [f"{me} {a} --rationale-eval" for a in arms]
    steps["background-test"] = [f"{me} {a} --background" for a in arms]
    steps["tuep-scan"] = [f"{me} {a} --scan" for a in arms]
    steps["score-last"] = [f"{me} {a} --score --last" for a in arms]
    steps["score-step"] = [f"{me} {a} --score --step {STEP}" for a in arms]
    steps["dpo-gen"] = [f"{me} {a} --dpo-gen" for a in arms]
    steps["final-test"] = [f"{me} {a} --{m}" for a in arms for m in ("heldout", "background")]
    logs = base_dir(arm == "smoke") / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    resources = ARMS.get(arms[0] if arm in SINGLE else arm, {})
    gpus, ram = resources.get("gpus", GPUS), resources.get("ram", RAM)
    scoring = arm in PSEUDO or ARMS.get(arm, {}).get("recipe", "train") is None
    big = any(ARMS.get(a, {}).get("big") for a in (arms or (arm,)))  # a base over 12B (round 8)
    if os.environ.get("CC_CLUSTER") in ("fir", "nibi", "rorqual"):  # Nibi and Rorqual have Fir's H100s and slice names
        # scoring holds no optimizer state (the largest grid base below 13B is ~24 GB in bf16), so it takes a 40 GB
        # slice, which starts within minutes where a whole H100 waits behind ~950 jobs (2026-10-18); a big base's needs
        # a whole H100
        slice_ = "h100:1" if big else "nvidia_h100_80gb_hbm3_3g.40gb:1"
        gpus = slice_ if scoring else resources.get("fir-gpus", FIR_GPUS[gpus])
        ram = "128G" if big and scoring else ram
    if os.environ.get("CC_CLUSTER") == "killarney":  # PAICE (2026-10-28): per-GPU jobs on L40S 48 GB or H100, no slices
        # every base below 13B trained within a 40 GB A100 or slice, so L40S; the round-8 bases take H100s as elsewhere
        gpus = f"h100:{1 if scoring else resources.get('h100s', 1)}" if big else "l40s:1"
        if resources.get("killarney-gpus") and not scoring:  # Qwen3.5: L40S, where fla's backward is right
            gpus = resources["killarney-gpus"]
        ram = "128G" if big and scoring else ram
    if os.environ.get("VF_GPUS"):  # a different GPU for this submission (e.g. a whole H100 to fit a short window)
        gpus = os.environ["VF_GPUS"]
    limit = TIME[arm]
    if os.environ.get("VF_LIMIT"):  # a shorter limit (e.g. to fit before a maintenance window); training resumes after
        limit = os.environ["VF_LIMIT"]
    hours = (int(limit.split("-")[0]) * 24 if "-" in limit else 0) + int(limit.split("-")[-1].split(":")[0])
    if os.environ.get("CC_CLUSTER") == "trillium":  # whole H100s only, 24 h at most (jobs resume when resubmitted)
        if resources.get("h100s", 1) > 1 and not scoring:
            sys.exit(f"{arm} trains on {resources['h100s']} H100s and Trillium takes 1 GPU or a whole node: submit it on Fir")
        gpus, limit = "h100:1", limit if hours < 24 else "1-00:00:00"
    elif big and hours >= 24:  # a big base's run goes in 24 h chunks (more partitions; the watchers resubmit on timeout)
        limit = "1-00:00:00"
    text = script(name, limit, ram, cfg["train"]["cpus"], gpus, " && ".join(steps[arm]))
    if big:  # less fragmentation where the weights leave ~15 GB of a card
        text = text.replace("export PYTHONUNBUFFERED=1", "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\nexport PYTHONUNBUFFERED=1")
    if os.environ.get("CC_CLUSTER") in ("killarney", "tamia", "vulcan"):  # PAICE runs on its own allocation
        text = text.replace("#SBATCH --account=def-milad777", "#SBATCH --account=aip-milad777")
    if os.environ.get("CC_CLUSTER") == "trillium":
        text = text.replace("#SBATCH --gres=gpu:", "#SBATCH --gpus-per-node=")
        text = "\n".join(line for line in text.split("\n") if not line.startswith("#SBATCH --mem="))
    if after:
        text = text.replace("#SBATCH", f"#SBATCH --dependency=afterok:{after}\n#SBATCH", 1)
    text = text.replace(str(ROOT / "logs"), str(logs))
    print(f"{'dry run' if dry_run else submit(text)} - {name}, {limit}/{ram}/gpu:{gpus}, log in {logs}")
    if dry_run:
        print(text)


def main_cli():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=[*ARMS, "smoke", *PSEUDO])
    parser.add_argument("arms", nargs="*", help="with a pseudo-arm (PSEUDO): the finished arms")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="train this arm in this process")
    mode.add_argument("--score", action="store_true", help="score this arm on the TUSZ test windows")
    mode.add_argument("--rationale-eval", action="store_true", help="reference NLL and generated rationales, 300 windows")
    mode.add_argument("--background", action="store_true", help="score data/background.csv's interior background windows")
    mode.add_argument("--heldout", action="store_true", help="score the 62 held-out TUSZ patients (final test)")
    mode.add_argument("--image-check", action="store_true", help="stop if the untrained base ignores the image")
    mode.add_argument("--scan", action="store_true", help="a TUEV model scores the TUEP scan windows; patient summary")
    mode.add_argument("--dpo-gen", action="store_true", help="the model's rationales on training windows, for DPO review")
    mode.add_argument("--review-gen", action="store_true", help="the model's rationales on its corpus's review windows")
    mode.add_argument("--dpo", action="store_true", help="expert DPO on a trained 8B dataset model (gm-8b-<ds>-all-dpo)")
    mode.add_argument("--dry-run", action="store_true", help="print the job, submit nothing")
    parser.add_argument("--smoke", action="store_true", help="with --train/--score: 2 steps, 24 windows, under smoke/")
    parser.add_argument("--last", action="store_true", help="with --score/--rationale-eval/--scan: the final checkpoint")
    parser.add_argument("--step", type=int, help="with --score or score-step: checkpoint-N instead of the kept one")
    args = parser.parse_args()
    global LAST, STEP
    LAST, STEP = args.last, args.step

    if (args.train or args.score or args.rationale_eval or args.background or args.heldout or args.scan or args.image_check
            or args.dpo_gen or args.dpo or args.review_gen):
        if args.arm in ("smoke", *PSEUDO) or args.train and ARMS[args.arm]["recipe"] is None:
            parser.error(f"that mode does not apply to {args.arm}")
        os.environ.setdefault("WANDB_MODE", "offline")
        if args.train:
            train(arm_config(cfg, args.arm, args.smoke), args.arm, args.smoke)
        elif args.score:
            score(args.arm, args.smoke)
        elif args.background:
            background_test(args.arm, args.smoke)
        elif args.heldout:
            heldout_test(args.arm, args.smoke)
        elif args.image_check:
            image_check(args.arm)
        elif args.scan:
            tuep_scan(args.arm, args.smoke)
        elif args.dpo_gen:
            dpo_gen(args.arm, args.smoke)
        elif args.review_gen:
            review_gen(args.arm, args.smoke)
        elif args.dpo:
            dpo_train(args.arm, args.smoke)
        else:
            rationale_eval(args.arm, args.smoke)
        return

    if args.arm in PSEUDO and not args.arms:
        parser.error(f"{args.arm} needs the arms to evaluate")
    submit_job(cfg, args.arm, args.dry_run, tuple(args.arms))


if __name__ == "__main__":
    main_cli()
