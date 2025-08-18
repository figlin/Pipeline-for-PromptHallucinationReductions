# stages.py  — cleaned imports + safe type hints (no circular imports)

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Callable, List, TYPE_CHECKING

# <<< ADDED: import only the data structures we need at runtime
from utils import StageResult

# <<< ADDED: import types only for type checking (no runtime import => avoids cycles)
if TYPE_CHECKING:
    from models import Model
    from dataset import Example

# -------------------------
# Gate policies (early exit)
# -------------------------

class GatePolicy(Protocol):
    def should_exit(self, example: "Example", candidate_answer: str, judge: Optional["Model"]) -> bool: ...


class OracleGate:
    """Offline eval with ground truth (used when y_true is available)."""
    def __init__(self, normalize: Callable[[str], str] = lambda s: s.strip().lower()):
        self.norm = normalize

    def should_exit(self, example: "Example", candidate_answer: str, judge: Optional["Model"]) -> bool:
        if example.y_true is None:
            return False
        return self.norm(candidate_answer) == self.norm(example.y_true)


class JudgeGate:
    """Online early-exit: ask a judge model 'is this correct?' and exit if PASS≥threshold."""
    def __init__(self, judge_prompt_template: str, threshold: float = 0.5):
        self.tpl = judge_prompt_template
        self.threshold = threshold

    def should_exit(self, example: "Example", candidate_answer: str, judge: Optional["Model"]) -> bool:
        if judge is None:
            return False
        prompt = self.tpl.format(question=example.question, answer=candidate_answer)
        r = judge.generate(prompt, temperature=0.0)
        txt = (r.text or "").lower()
        p = 0.5
        if "p=" in txt:
            try:
                p = float(txt.split("p=")[-1].split(">")[0])
            except Exception:
                p = 0.5
        return ("pass" in txt) and (p >= self.threshold)


# -------------------------
# Stage protocol + registry
# -------------------------

class Stage(Protocol):
    id: str
    def run(self, ex: "Example", ctx: Dict[str, Any]) -> StageResult: ...


_STAGE_REGISTRY: Dict[str, Callable[..., "Stage"]] = {}


def register_stage(name: str):
    def deco(factory: Callable[..., "Stage"]):
        _STAGE_REGISTRY[name] = factory
        return factory
    return deco


def make_stage(name: str, **kwargs) -> "Stage":
    if name not in _STAGE_REGISTRY:
        raise KeyError(f"Unknown stage '{name}'. Registered: {list(_STAGE_REGISTRY)}")
    return _STAGE_REGISTRY[name](**kwargs)


# -------------------------
# Built-in stages
# -------------------------

@register_stage("baseline")
class BaselineAsk:
    def __init__(self, id: str, model: "Model", prompt_template: str):
        self.id, self.model, self.tpl = id, model, prompt_template

    def run(self, ex: "Example", ctx: Dict[str, Any]) -> StageResult:
        prompt = self.tpl.format(question=ex.question)
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = (r.text or "").strip() or None
            return StageResult(
                stage_id=self.id,
                answer=ans,
                should_exit=False,
                evidence={"prompt": prompt},
                model_usage={"model": getattr(self.model, "name", "unknown"), **r.__dict__},
            )
        except Exception as e:
            return StageResult(
                stage_id=self.id,
                answer=None,
                should_exit=False,
                evidence={"prompt": prompt, "error": str(e)},
                model_usage={"model": getattr(self.model, "name", "unknown")},
            )


@register_stage("apo")
class APOStage:
    """
    Uses a lightweight helper LLM to rewrite the prompt before calling the target model.
    """
    def __init__(
        self,
        id: str,
        helper: "Model",
        target: "Model",
        rewrite_template: str,
        target_prompt_template: str,
    ):
        self.id, self.helper, self.target = id, helper, target
        self.rewrite_template = rewrite_template
        self.target_prompt_template = target_prompt_template

    def run(self, ex: "Example", ctx: Dict[str, Any]) -> StageResult:
        evidence: Dict[str, Any] = {}
        usage: Dict[str, Any] = {}

        # 1) helper rewrite
        helper_in = self.rewrite_template.format(question=ex.question)
        evidence["helper_in"] = helper_in
        try:
            h = self.helper.generate(helper_in, temperature=0.2)
            optimized = (h.text or "").strip()
            usage["helper"] = {"model": getattr(self.helper, "name", "unknown"), **h.__dict__}
            evidence["optimized"] = optimized
        except Exception as e:
            evidence["helper_error"] = str(e)
            optimized = ex.question  # fallback

        # 2) target answer
        target_in = self.target_prompt_template.format(optimized_prompt=optimized)
        evidence["target_in"] = target_in
        try:
            t = self.target.generate(target_in, temperature=0.0)
            ans = (t.text or "").strip() or None
            usage["target"] = {"model": getattr(self.target, "name", "unknown"), **t.__dict__}
            return StageResult(self.id, ans, False, evidence=evidence, model_usage=usage)
        except Exception as e:
            evidence["target_error"] = str(e)
            return StageResult(
                self.id,
                None,
                False,
                evidence=evidence,
                model_usage=usage if usage else {"model": getattr(self.target, "name", "unknown")},
            )


@register_stage("cove")
class CoVeStage:
    """Chain-of-Verification around prior candidate answer (or the question if none)."""
    def __init__(self, id: str, model: "Model", cove_template: str):
        self.id, self.model, self.tpl = id, model, cove_template

    def run(self, ex: "Example", ctx: Dict[str, Any]) -> StageResult:
        prev_ans = ctx.get("last_answer", "")
        prompt = self.tpl.format(question=ex.question, prior_answer=prev_ans)
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = (r.text or "").strip() or None
            return StageResult(
                self.id,
                ans,
                False,
                evidence={"prompt": prompt},
                model_usage={"model": getattr(self.model, "name", "unknown"), **r.__dict__},
            )
        except Exception as e:
            return StageResult(
                self.id,
                None,
                False,
                evidence={"prompt": prompt, "error": str(e)},
                model_usage={"model": getattr(self.model, "name", "unknown")},
            )


@register_stage("self_correct")
class SelfCorrectStage:
    def __init__(self, id: str, model: "Model", template: str):
        self.id, self.model, self.tpl = id, model, template

    def run(self, ex: "Example", ctx: Dict[str, Any]) -> StageResult:
        prev_ans = ctx.get("last_answer", "")
        prompt = self.tpl.format(question=ex.question, prior_answer=prev_ans)
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = (r.text or "").strip() or None
            return StageResult(
                self.id,
                ans,
                False,
                evidence={"prompt": prompt},
                model_usage={"model": getattr(self.model, "name", "unknown"), **r.__dict__},
            )
        except Exception as e:
            return StageResult(
                self.id,
                None,
                False,
                evidence={"prompt": prompt, "error": str(e)},
                model_usage={"model": getattr(self.model, "name", "unknown")},
            )


@register_stage("judge")
class ExternalJudgeStage:
    """
    Optionally emits should_exit based on judge verdict or returns a verdict
    for the runner to use.
    """
    def __init__(self, id: str, judge: "Model", judge_template: str, exit_on_pass: bool = True, threshold: float = 0.5):
        self.id = id
        self.judge = judge
        self.tpl = judge_template
        self.exit_on_pass = exit_on_pass
        self.threshold = threshold

    def run(self, ex: "Example", ctx: Dict[str, Any]) -> StageResult:
        cand = ctx.get("last_answer", "")
        prompt = self.tpl.format(question=ex.question, answer=cand)
        evidence = {"prompt": prompt}
        try:
            r = self.judge.generate(prompt, temperature=0.0)
            verdict = (r.text or "").strip().lower()
            evidence["verdict"] = verdict
            p = 0.5
            if "p=" in verdict:
                try:
                    p = float(verdict.split("p=")[-1].split(">")[0])
                except Exception:
                    p = 0.5
            evidence["p"] = p
            is_pass = ("pass" in verdict)
            exit_flag = bool(self.exit_on_pass and is_pass and (p >= self.threshold))
            return StageResult(
                stage_id=self.id,
                answer=cand or None,
                should_exit=exit_flag,
                evidence=evidence,
                model_usage={"model": getattr(self.judge, "name", "unknown"), **r.__dict__},
            )
        except Exception as e:
            evidence["error"] = str(e)
            return StageResult(
                stage_id=self.id,
                answer=cand or None,
                should_exit=False,
                evidence=evidence,
                model_usage={"model": getattr(self.judge, "name", "unknown")},
            )
