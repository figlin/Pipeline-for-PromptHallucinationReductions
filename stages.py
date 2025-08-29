from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Callable

from utils import StageResult, Example

from models import Model
# from pipeline import pipeline

# -------------------------
# Gate policies (early exit)
# -------------------------

class GatePolicy(Protocol):
    def should_exit(self, example: Example, candidate_answer: str, judge: Optional[Model]) -> bool: ...

class OracleGate:  # offline eval with ground truth
    def __init__(self, normalize: Callable[[str], str] = lambda s: s.strip().lower()):
        self.norm = normalize
    def should_exit(self, example: Example, candidate_answer: str, judge: Optional[Model]) -> bool:
        if example.y_true is None:
            return False
        return self.norm(candidate_answer) == self.norm(example.y_true)

class JudgeGate:   # online: ask a judge model "is this correct?"
    def __init__(self, judge_prompt_template: str, threshold: float = 0.5):
        self.tpl = judge_prompt_template
        self.threshold = threshold
    def should_exit(self, example: Example, candidate_answer: str, judge: Optional[Model]) -> bool:
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

class MetricGate:  # exit based on evaluation metrics
    def __init__(self, evaluator, criteria: str, dataset_schema: str = None):
        """
        Initialize MetricGate with evaluation criteria
        
        Args:
            evaluator: The evaluator instance to compute metrics
            criteria: String expression like "mc_accuracy>0.7" or "mc_accuracy>0.6||rouge1>0.5"
            dataset_schema: Dataset schema to apply dataset-specific thresholds
        """
        self.evaluator = evaluator
        self.criteria = criteria
        self.dataset_schema = dataset_schema
        
        # Parse criteria into evaluable conditions
        self.parsed_criteria = self._parse_criteria(criteria)
    
    def _parse_criteria(self, criteria: str) -> dict:
        """Parse criteria string into structured conditions"""
        # Split on || (OR) and && (AND) operators
        or_groups = criteria.split("||")
        parsed = {"or_groups": []}
        
        for or_group in or_groups:
            and_conditions = or_group.split("&&")
            group_conditions = []
            
            for condition in and_conditions:
                condition = condition.strip()
                # Parse metric>threshold or metric>=threshold
                if ">=" in condition:
                    metric, threshold = condition.split(">=")
                    operator = ">="
                elif ">" in condition:
                    metric, threshold = condition.split(">")
                    operator = ">"
                else:
                    raise ValueError(f"Invalid condition format: {condition}")
                
                group_conditions.append({
                    "metric": metric.strip(),
                    "operator": operator,
                    "threshold": float(threshold.strip())
                })
            
            parsed["or_groups"].append(group_conditions)
        
        return parsed
    
    def should_exit(self, example: Example, candidate_answer: str, judge: Optional[Model]) -> bool:
        """Evaluate if gate should trigger based on metrics"""
        # Get gold answers for evaluation
        if hasattr(example, 'correct_answers') and example.correct_answers:
            golds = example.correct_answers
        elif example.y_true:
            golds = [example.y_true]
        else:
            return False
        
        # Compute metrics
        try:
            result = self.evaluator._evaluate_example(
                pred=candidate_answer,
                golds=golds,
                bads=example.incorrect_answers
            )
        except Exception:
            return False
        
        # Evaluate parsed criteria
        for or_group in self.parsed_criteria["or_groups"]:
            # All conditions in an AND group must be true
            and_result = True
            for condition in or_group:
                metric_name = condition["metric"]
                operator = condition["operator"]
                threshold = condition["threshold"]
                
                # Get metric value
                metric_value = getattr(result, metric_name, 0.0)
                
                # Check condition
                if operator == ">=":
                    condition_met = metric_value >= threshold
                elif operator == ">":
                    condition_met = metric_value > threshold
                else:
                    condition_met = False
                
                and_result = and_result and condition_met
            
            # If any OR group is satisfied, gate should trigger
            if and_result:
                return True
        
        return False

# -------------------------
# Stage protocol + registry
# -------------------------

class Stage(Protocol):
    id: str
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult: ...

_STAGE_REGISTRY: Dict[str, Callable[..., Stage]] = {}

def register_stage(name: str):
    def deco(factory: Callable[..., Stage]):
        _STAGE_REGISTRY[name] = factory
        return factory
    return deco

def make_stage(name: str, **kwargs) -> Stage:
    if name not in _STAGE_REGISTRY:
        raise KeyError(f"Unknown stage '{name}'. Registered: {list(_STAGE_REGISTRY)}")
    return _STAGE_REGISTRY[name](**kwargs)

# -------------------------
# Built-in stages
# -------------------------

@register_stage("baseline")
class BaselineAsk:
    def __init__(self, id: str, model: Model, prompt_template: str):
        self.id, self.model, self.tpl = id, model, prompt_template
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
        prompt = self.tpl.format(question=ex.question)
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = (r.text or "").strip() or None
            return StageResult(
                stage_id=self.id, answer=ans, should_exit=False,
                evidence={"prompt": prompt},
                model_usage={"model": getattr(self.model, "name", "unknown"), **r.__dict__}
            )
        except Exception as e:
            return StageResult(
                stage_id=self.id, answer=None, should_exit=False,
                evidence={"prompt": prompt, "error": str(e)},
                model_usage={"model": getattr(self.model, "name", "unknown")}
            )

@register_stage("apo")
class APOStage:
    """Uses a lightweight helper LLM to rewrite the prompt before calling the target model."""
    def __init__(self, id: str, helper: Model, target: Model,
                 rewrite_template: str, target_prompt_template: str):
        self.id, self.helper, self.target = id, helper, target
        self.rewrite_template = rewrite_template
        self.target_prompt_template = target_prompt_template
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
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
            return StageResult(self.id, None, False, evidence=evidence,
                               model_usage=usage if usage else {"model": getattr(self.target, "name", "unknown")})

@register_stage("cove")
class CoVeStage:
    """Standardized Chain-of-Verification prompt around a candidate answer (or the question if none)."""
    def __init__(self, id: str, model: Model, cove_template: str):
        self.id, self.model, self.tpl = id, model, cove_template
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
        prev_ans = ctx.get("last_answer", "")
        prompt = self.tpl.format(question=ex.question, prior_answer=prev_ans)
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = (r.text or "").strip() or None
            return StageResult(self.id, ans, False, evidence={"prompt": prompt},
                               model_usage={"model": getattr(self.model, "name", "unknown"), **r.__dict__})
        except Exception as e:
            return StageResult(self.id, None, False, evidence={"prompt": prompt, "error": str(e)},
                               model_usage={"model": getattr(self.model, "name", "unknown")})

@register_stage("self_correct")
class SelfCorrectStage:
    def __init__(self, id: str, model: Model, template: str):
        self.id, self.model, self.tpl = id, model, template
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
        prev_ans = ctx.get("last_answer", "")
        prompt = self.tpl.format(question=ex.question, prior_answer=prev_ans)
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = (r.text or "").strip() or None
            return StageResult(self.id, ans, False, evidence={"prompt": prompt},
                               model_usage={"model": getattr(self.model, "name", "unknown"), **r.__dict__})
        except Exception as e:
            return StageResult(self.id, None, False, evidence={"prompt": prompt, "error": str(e)},
                               model_usage={"model": getattr(self.model, "name", "unknown")})

@register_stage("judge")
class ExternalJudgeStage:
    """Optionally emits should_exit based on judge verdict or returns a verdict for the runner to use."""
    def __init__(self, id: str, judge: Model, judge_template: str, exit_on_pass: bool = True, threshold: float = 0.5):
        self.id = id
        self.judge = judge
        self.tpl = judge_template
        self.exit_on_pass = exit_on_pass
        self.threshold = threshold
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
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
                model_usage={"model": getattr(self.judge, "name", "unknown"), **r.__dict__}
            )
        except Exception as e:
            evidence["error"] = str(e)
            return StageResult(
                stage_id=self.id,
                answer=cand or None,
                should_exit=False,
                evidence=evidence,
                model_usage={"model": getattr(self.judge, "name", "unknown")}
            )

@register_stage("confidence_check")
class ConfidenceCheckStage:
    """Asks the model to evaluate its confidence in its previous answer."""
    def __init__(self, id: str, model: Model, template: str):
        self.id, self.model, self.tpl = id, model, template
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
        prev_ans = ctx.get("last_answer", "")
        prompt = self.tpl.format(question=ex.question, previous_answer=prev_ans, correct_answer=ex.y_true or "unknown")
        try:
            r = self.model.generate(prompt, temperature=0.0)
            response = (r.text or "").strip()
            evidence = {"prompt": prompt, "confidence_response": response}
            
            # Check if the model is confident or admits uncertainty
            response_lower = response.lower()
            is_confident = "my last answer was correct" in response_lower and "confident" in response_lower
            admits_wrong = "i was wrong" in response_lower and "do not know" in response_lower
            
            evidence["is_confident"] = is_confident
            evidence["admits_wrong"] = admits_wrong
            
            # Return the previous answer, not the confidence response (confidence is in evidence)
            return StageResult(
                stage_id=self.id, 
                answer=prev_ans,  # Keep the previous answer as the final answer
                should_exit=False,
                evidence=evidence,
                model_usage={"model": getattr(self.model, "name", "unknown"), **r.__dict__}
            )
        except Exception as e:
            return StageResult(
                stage_id=self.id, 
                answer=prev_ans,  # Keep previous answer on error
                should_exit=False,
                evidence={"prompt": prompt, "error": str(e)},
                model_usage={"model": getattr(self.model, "name", "unknown")}
            )
            
