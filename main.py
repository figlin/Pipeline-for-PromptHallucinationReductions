from __future__ import annotations
from typing import Any, Dict
import os
import sys

from dotenv import load_dotenv

from dataset import dataset, load_from_csv  # Assuming dataset.py is in the same directory
from models import Model,ScaledownDirectModel, OllamaModel
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
            model=os.getenv("SD_MODEL_TARGET") or "gemini/gemini-pro",
            rate=float(os.getenv("SD_RATE_TARGET") or "0.7"),
        ),
        "helper": ScaledownDirectModel(
            api_key=SCALEDOWN_API_KEY,
            endpoint=SCALEDOWN_CHAT_URL,
            model=os.getenv("SD_MODEL_HELPER") or "gemini/gemini-pro",
            rate=float(os.getenv("SD_RATE_HELPER") or "0.6"),
        ),
        "judge": ScaledownDirectModel(
            api_key=SCALEDOWN_API_KEY,
            endpoint=SCALEDOWN_CHAT_URL,
            model=os.getenv("SD_MODEL_JUDGE") or "gemini/gemini-pro",
            rate=float(os.getenv("SD_RATE_JUDGE") or "0.0"),
            default_params={"temperature": 0.0},
        ),
        # You could wrap any model with compression like:
        # "target": ScaleDownWrappedModel(base=SomeOtherModel(...), api_key=SCALEDOWN_API_KEY, rate=0.7)
        
        # --- Add your Ollama model here ---
        "my_ollama_model": OllamaModel(
            model="gemma3:27b",  # The model name your Ollama server is using
            # endpoint="http://localhost:11434/api/generate", # Optional: if not default
             timeout=240.0, # Optional: increase timeout for slow models
        ),
    }

    config = {
        "stages": [
            {"type":"baseline","id":"s0","model":"my_ollama_model"}, # <-- Example using the ollama model
            # {"type":"baseline","id":"s0","model":"target"},
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
    pipe.debug_maxlen = int(os.getenv("PIPE_DEBUG_MAXLEN") or "220")

    
    # Load dataset based on environment variable
    dataset_path = os.getenv("DATASET_PATH", "internal")
    if dataset_path.lower() == "internal":
        dataset_to_run = dataset
        print("Running with internal dataset.")
    else:
        print(f"Running with dataset from: {dataset_path}")
        dataset_to_run = load_from_csv(dataset_path)

    if not dataset_to_run:
        print(f"Error: No data loaded from '{dataset_path}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    for ex in dataset_to_run:
        trace = pipe.run_one(ex)
        print(f"[{ex.qid}] Q: {ex.question}")
        print("  Final:", trace.final_answer)
        print("  Exit at:", trace.early_exit_at)
        print("  Tokens:", trace.total_tokens)
        if ex.y_true:
            print("  Correct:", trace.final_answer and trace.final_answer.strip().lower() == ex.y_true.lower())
        print("-" * 50)
    
    if 'trace' in locals() and trace:
        print("Final:", trace.final_answer, "Exit at:", trace.early_exit_at, "Tokens:", trace.total_tokens)

