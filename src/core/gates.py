from __future__ import annotations
from typing import Protocol, Optional, Callable
import logging

from src.core.dataset_loader import Example
from src.core.models import Model


class GatePolicy(Protocol):
	def should_exit(self, example: Example, candidate_answer: str, judge: Optional[Model]) -> bool: ...


class OracleGate:
	def __init__(self, normalize: Callable[[str], str] = lambda s: s.strip().lower()):
		self.norm = normalize

	def should_exit(self, example: Example, candidate_answer: str, judge: Optional[Model]) -> bool:
		if example.y_true is None:
			return False
		return self.norm(candidate_answer) == self.norm(example.y_true)


class JudgeGate:
	def __init__(self, judge_prompt_template: str, threshold: float = 0.5):
		self.tpl = judge_prompt_template
		self.threshold = threshold
		self.logger = logging.getLogger("gates.judge")

	def should_exit(self, example: Example, candidate_answer: str, judge: Optional[Model]) -> bool:
		if judge is None:
			return False
		prompt = self.tpl.format(question=example.question, answer=candidate_answer)
		try:
			r = judge.generate(prompt, temperature=0.0)
		except Exception as e:
			try:
				self.logger.error("judge.call_failed error=%s", e)
			except Exception:
				pass
			return False
		txt = (r.text or "").lower()
		p = 0.5
		if "p=" in txt:
			try:
				p = float(txt.split("p=")[-1].split(">")[0])
			except Exception:
				p = 0.5
		return ("pass" in txt) and (p >= self.threshold)


