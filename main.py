from __future__ import annotations
from typing import Any, Dict
import os

from dotenv import load_dotenv

from dataset import dataset  # Assuming dataset.py is in the same directory
from models import Model,ScaledownDirectModel
from pipeline import DEFAULT_TEMPLATES, build_pipeline

load_dotenv()

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

