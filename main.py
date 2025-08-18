from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Callable
import difflib
import time
import uuid
import os
import requests
import json

from src.dataset_loader import Example, dataset
from src.models import Model, ScaledownDirectModel, EchoModel
from src.pipeline import build_pipeline
from src.templates import DEFAULT_TEMPLATES
import logging
import sys
from dotenv import load_dotenv
load_dotenv()

# -------------------------
# Minimal runnable demo
# -------------------------

if __name__ == "__main__":
    # Logging setup
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("main")
    # Read env
    use_echo = os.getenv("USE_ECHO", "0") not in ("0", "", "false", "False", "FALSE")
    SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY")
    SCALEDOWN_CHAT_URL = os.getenv("SCALEDOWN_CHAT_URL")  # your ScaleDown generation endpoint

    if use_echo:
        echo = EchoModel()
        models: Dict[str, Model] = {"target": echo, "helper": echo, "judge": echo}
    else:
        if not SCALEDOWN_API_KEY or not SCALEDOWN_CHAT_URL:
            raise RuntimeError("Missing SCALEDOWN_API_KEY or SCALEDOWN_CHAT_URL. Set USE_ECHO=1 for local runs.")
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
        }

    config = {
        "stages": [
            {"type":"baseline","id":"s0","model":"target"},
            {"type":"apo","id":"s1","helper":"helper","target":"target"},
            {"type":"cove","id":"s2","model":"target"},
            {"type":"self_correct","id":"s3","model":"target"},
            {"type":"judge","id":"s4","judge":"judge","exit_on_pass": True, "threshold": 0.8}
        ],
        "gate": ({"mode": "judge", "judge": "judge", "threshold": 0.8, "template": DEFAULT_TEMPLATES["gate_judge"]}
                 if not use_echo else
                 {"mode": "oracle"}),
        "token_diffs": True
    }

    DEBUG = os.getenv("PIPE_DEBUG", "0") not in ("0", "", "false", "False", "FALSE")
    pipe = build_pipeline(config, models)
    # re-wrap with debug (or pass debug into a factory if you prefer)
    pipe.debug = DEBUG
    pipe.debug_maxlen = int(os.getenv("PIPE_DEBUG_MAXLEN", "220"))

    
    for ex in dataset:
        trace = pipe.run_one(ex)
        logger.info("result qid=%s final=%r exit=%s tokens=%s correct=%s",
                    ex.qid,
                    trace.final_answer,
                    trace.early_exit_at,
                    trace.total_tokens,
                    bool(trace.final_answer and trace.final_answer.strip().lower() == (ex.y_true or "").lower()))
    # templates now live in src/templates.py
