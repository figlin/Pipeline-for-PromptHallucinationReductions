from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    confidence: float = 0.0  # Confidence score (0-1)
    evidence: Dict[str, Any] = field(default_factory=dict)
    model_usage: Dict[str, Any] = field(default_factory=dict)


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
    artifacts: Dict[str, Any] = field(default_factory=dict)


