from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
class Example:
    """Use y_true only in offline eval. In production, omit it."""
    qid: str
    question: str
    y_true: Optional[str] = None

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
    early_exit_at: Optional[str] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    timing_sec: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)  # e.g., diffs, prompts, raw json