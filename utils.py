from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from dataset import QAResponse
@dataclass
class Example:
    """Use y_true only in offline eval. In production, omit it."""
    qid: str
    question: str
    y_true: Optional[str] = None

# Assuming dataset.py is in the same directory
# -------------------------
# Core data structures
# -------------------------

@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StageResult:
    stage_id: str
    answer: Optional[str]
    should_exit: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    model_usage: Dict[str, Any] = field(default_factory=dict)  # tokens, cost, etc.

@dataclass
class RunTrace:
    qid: str
    question: str
    stage_results: List[StageResult] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_metrics: Optional[Dict[str, float]] = None
    early_exit_at: Optional[str] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    timing_sec: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)  # e.g., diffs, prompts, raw json

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def safe_list(val: Union[str, List[str]]) -> List[str]:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        # handle semicolon-separated fields
        parts = [v.strip() for v in val.split(";") if v.strip()]
        return parts
    return []

# metrics
def f1_score(pred: str, truth: str) -> float:
    """Token overlap F1 for SimpleQA"""
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def parse_llm_response(llm_output: str) -> Optional[QAResponse]:
    try:
        from dataset import QAResponse  # Import here to avoid circular import
        data = json.loads(llm_output)
        return QAResponse(**data)
    except Exception as e:
        print("Parsing failed:", e)
        return None
    
SCHEMA_REGISTRY = {
    "simpleqa": {
        "keys": {"answer", "problem"},
        "parser": "_parse_simpleqa",
        "id_keys": ["problem", "metadata"],   # ← use both for unique ID
    },
    "truthfulqa": {
        "keys": {"Best Answer", "Question"},
        "parser": "_parse_truthfulqa",
        "id_keys": ["Question", "Category"],  # ← combine these
    },
}