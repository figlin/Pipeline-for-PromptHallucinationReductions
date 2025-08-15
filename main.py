from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Callable
import difflib
import time
import uuid
import os
import requests
import json

from dataset import dataset  # Assuming dataset.py is in the same directory
from models import models
from pipeline import pipeline
from stages import stages

# -------------------------
# Minimal runnable demo
# -------------------------

if __name__ == "__main__":
    # Read env
    SCALEDOWN_API_KEY = os.environ["SCALEDOWN_API_KEY"]
    SCALEDOWN_CHAT_URL = os.environ["SCALEDOWN_CHAT_URL"]  # your ScaleDown generation endpoint

    models: Dict[str, Model] = {
        "target": ScaledownDirectModel(
            api_key=SCALEDOWN_API_KEY,
            endpoint=SCALEDOWN_CHAT_URL,
            model=os.getenv("SD_MODEL_TARGET", "gemini/gemini-pro"),
            rate=float(os.getenv("SD_RATE_TARGET", "0.7")),
        ),
        "helper": ScaledownDirectModel(
            api_key=SCALEDOWN_API_KEY,
            endpoint=SCALEDOWN_CHAT_URL,
            model=os.getenv("SD_MODEL_HELPER", "gemini/gemini-pro"),
            rate=float(os.getenv("SD_RATE_HELPER", "0.6")),
        ),
        "judge": ScaledownDirectModel(
            api_key=SCALEDOWN_API_KEY,
            endpoint=SCALEDOWN_CHAT_URL,
            model=os.getenv("SD_MODEL_JUDGE", "gemini/gemini-pro"),
            rate=float(os.getenv("SD_RATE_JUDGE", "0.0")),
            default_params={"temperature": 0.0},
        ),
        # You could wrap any model with compression like:
        # "target": ScaleDownWrappedModel(base=SomeOtherModel(...), api_key=SCALEDOWN_API_KEY, rate=0.7)
    }

    config = {
        "stages": [
            {"type":"baseline","id":"s0","model":"target"},
            {"type":"apo","id":"s1","helper":"helper","target":"target"},
            {"type":"cove","id":"s2","model":"target"},
            {"type":"self_correct","id":"s3","model":"target"},
            {"type":"judge","id":"s4","judge":"judge","exit_on_pass": True, "threshold": 0.8}
        ],
        "gate": {"mode": "judge", "judge": "judge", "threshold": 0.8, "template": DEFAULT_TEMPLATES["gate_judge"]},
        "token_diffs": True
    }

    DEBUG = os.getenv("PIPE_DEBUG", "0") not in ("0", "", "false", "False", "FALSE")
    pipe = build_pipeline(config, models)
    # re-wrap with debug (or pass debug into a factory if you prefer)
    pipe.debug = DEBUG
    pipe.debug_maxlen = int(os.getenv("PIPE_DEBUG_MAXLEN", "220"))

    
    for ex in dataset:
        trace = pipe.run_one(ex)
        print(f"[{ex.qid}] Q: {ex.question}")
        print("  Final:", trace.final_answer)
        print("  Exit at:", trace.early_exit_at)
        print("  Tokens:", trace.total_tokens)
        print("  Correct:", trace.final_answer and trace.final_answer.strip().lower() == ex.y_true.lower())
        print("-" * 50)
    print("Final:", trace.final_answer, "Exit at:", trace.early_exit_at, "Tokens:", trace.total_tokens)

DEFAULT_TEMPLATES = {
    "baseline": "Q: {question}\nA:",
    "apo_rewrite": (
        "Rewrite the user question into a concise, specific prompt that reduces ambiguity "
        "and includes constraints to avoid hallucinations. Output only the rewritten prompt.\n\nQ: {question}"
    ),
    "apo_target": "Use this optimized prompt:\n\n{optimized_prompt}\n\nAnswer succinctly and cite key facts.",
    "cove": (
        "You are verifying an answer’s factuality via Chain-of-Verification.\n"
        "Question: {question}\n"
        "Prior answer: {prior_answer}\n"
        "1) List claims.\n2) Verify each claim with independent checks.\n3) Give a corrected final answer only."
    ),
    "self_correct": (
        "Revise only factual errors at temperature 0.\n"
        "Question: {question}\n"
        "Current answer: {prior_answer}\n"
        "Return a corrected final answer only."
    ),
    "judge": (
        "You are a strict fact-checking judge.\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Return exactly: PASS <p=0.80> or FAIL <p=0.20>."
    ),
    "gate_judge": (
        "Judge correctness for early exit.\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Return PASS <p=...> or FAIL <p=...>."
    )
}
