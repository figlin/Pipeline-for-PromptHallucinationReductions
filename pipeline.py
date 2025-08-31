from __future__ import annotations
from typing import Any, Dict, List, Optional
import difflib
import time

from dataset import EvalResult, Example, Dataset, AggregatedMetrics
from evaluate import Evaluator
from models import Model
from stages import make_stage, JudgeGate, Stage, GatePolicy, OracleGate, MetricGate
from utils import RunTrace

# -------------------------
# Pipeline runner
# -------------------------

class Pipeline:
    def __init__(
        self,
        stages: List[Stage],
        gate: GatePolicy,
        dataset: Dataset,
        judge_for_gate: Optional[Model] = None,
        do_token_diffs: bool = True,
        debug: bool = False,            # <-- new
        debug_maxlen: int = 220,        # <-- new
        debug_context: bool = False,    # <-- new: debug all context values
        keep_context: bool = True,      # <-- new: keep context throughout singular question
        evaluator: Optional[object] = None,  # <-- new: evaluator for per-stage evaluation
        evaluate_all_stages: bool = True    # <-- new: enable per-stage evaluation
    ):
        self.stages = stages
        self.gate = gate
        self.judge_for_gate = judge_for_gate
        self.do_token_diffs = do_token_diffs
        self.debug = debug
        self.debug_maxlen = debug_maxlen
        self.debug_context = debug_context
        self.keep_context = keep_context
        self.evaluator = evaluator
        self.evaluate_all_stages = evaluate_all_stages
        self.aggregated_metrics = AggregatedMetrics()

    def _dbg(self, *parts):
        if self.debug:
            print(*parts)

    def run_one(self, ex: Example) -> RunTrace:
        t0 = time.time()
        trace = RunTrace(qid=ex.qid, question=ex.question)
        ctx: Dict[str, Any] = {}
        last_answer: Optional[str] = None

        self._dbg(f"\n=== Dataset Question {ex.qid} :: {ex.question!r} - \"{ex.y_true or 'No Answer'}\" ===")

        for stage in self.stages:
            if last_answer is not None:
                ctx["last_answer"] = last_answer

            self._dbg(f"\n-- Stage {getattr(stage,'id','?')} ({stage.__class__.__name__}) --")
            
            # Debug: show context flowing into this stage
            if self.debug and ctx:
                self._dbg("Context keys:", list(ctx.keys()))
                if "last_answer" in ctx:
                    last_ans_snip = ctx["last_answer"][:self.debug_maxlen] + ("..." if len(ctx["last_answer"]) > self.debug_maxlen else "")
                    self._dbg("Last answer:", last_ans_snip)
                # Show other stage contexts if available
                stage_contexts = [k for k in ctx.keys() if k.startswith("stage_")]
                if stage_contexts:
                    self._dbg("Available stage contexts:", stage_contexts)
            
            # Debug: show full context values if debug_context is enabled
            if self.debug_context and ctx:
                self._dbg("\n--- FULL CONTEXT DEBUG ---")
                for key, value in ctx.items():
                    if isinstance(value, dict):
                        self._dbg(f"{key}: {dict}")
                        for subkey, subvalue in value.items():
                            subvalue_str = str(subvalue)
                            if len(subvalue_str) > self.debug_maxlen:
                                subvalue_str = subvalue_str[:self.debug_maxlen] + "..."
                            self._dbg(f"  {subkey}: {subvalue_str}")
                    else:
                        value_str = str(value)
                        if len(value_str) > self.debug_maxlen:
                            value_str = value_str[:self.debug_maxlen] + "..."
                        self._dbg(f"{key}: {value_str}")
                self._dbg("--- END CONTEXT DEBUG ---\n")

            res = stage.run(ex, ctx)
            
            # Persist stage context if keep_context is enabled
            if self.keep_context and res.evidence:
                stage_ctx_key = f"stage_{getattr(stage,'id','?')}"
                ctx[stage_ctx_key] = res.evidence
                if self.debug:
                    self._dbg(f"Stored context key: {stage_ctx_key} (evidence keys: {list(res.evidence.keys())})")
            trace.stage_results.append(res)
            metrics_=self.evaluator._evaluate_example(
                    pred=res.answer,
                    bads=ex.incorrect_answers,
                    golds=[ex.y_true] if self.evaluator.dataset.schema == 'simpleqa' else ex.correct_answers,
                )
            if stage.__class__.__name__ != "ConfidenceCheckStage": # skipped the ConfidenceCheckStage as metrics not applicable
                self.aggregated_metrics.add_stage_result(stage.__class__.__name__, metrics_)
                trace.final_metrics = metrics_.model_dump()
                self._dbg(f" {stage.__class__.__name__} Result : {metrics_.model_dump()}")

            # Debug: show prompt/evidence/errors
            ev = res.evidence or {}
            prompt_snip = (ev.get("prompt") or ev.get("helper_in") or ev.get("target_in") or "")
            if prompt_snip:
                self._dbg("Prompt:", (prompt_snip[:self.debug_maxlen] + ("..." if len(prompt_snip) > self.debug_maxlen else "")))

            if "optimized" in ev:
                self._dbg("Optimized:", (ev["optimized"][:self.debug_maxlen] + ("..." if len(ev["optimized"]) > self.debug_maxlen else "")))

            if "error" in ev or "helper_error" in ev or "target_error" in ev:
                for k in ("error","helper_error","target_error"):
                    if k in ev:
                        self._dbg(f"ERROR ({k}):", ev[k])

            # usage accounting helper
            # inside Pipeline.run_one()

            def add_usage(u: Dict[str, Any]):
                if not isinstance(u, dict):
                    return
                if "prompt_tokens" in u: trace.total_tokens += int(u["prompt_tokens"])
                if "completion_tokens" in u: trace.total_tokens += int(u["completion_tokens"])
                if "cost" in u: trace.total_cost += float(u["cost"])

            def dbg_usage(u: Any, label="usage"):
                if not self.debug or not isinstance(u, dict):
                    return
                model = u.get("model", "?")
                status = (u.get("meta") or {}).get("status")
                pt = u.get("prompt_tokens"); ct = u.get("completion_tokens")
                self._dbg(f"{label}: model={model} status={status} ptok={pt} ctok={ct}")

            # …

            if "prompt_tokens" in res.model_usage:
                # flat usage dict (BaselineAsk etc.)
                add_usage(res.model_usage)
                dbg_usage(res.model_usage)
            else:
                # nested usage dicts (APO helper/target)
                for sub_label, sub in res.model_usage.items():
                    add_usage(sub)
                    dbg_usage(sub, label=f"usage.{sub_label}")


            # candidate answer
            if res.answer is not None:
                ans_snip = res.answer[:self.debug_maxlen] + ("..." if len(res.answer) > self.debug_maxlen else "")
                self._dbg("Answer:", ans_snip)
                
                # Per-stage evaluation (skip confidence check stages)
                if self.evaluate_all_stages and self.evaluator and stage.__class__.__name__ != "ConfidenceCheckStage":
                    golds = ex.correct_answers if ex.correct_answers else [ex.y_true] if ex.y_true else []
                    if golds:
                        stage_eval = self.evaluator._evaluate_example(pred=res.answer, golds=golds, bads=ex.incorrect_answers)
                        res.evidence = res.evidence or {}
                        res.evidence["stage_eval"] = stage_eval.model_dump()
                        self._dbg(f"Stage Eval: EM={stage_eval.em:.3f}, F1={stage_eval.f1:.3f}, ROUGE1={stage_eval.rouge1:.3f}, LLM_Judge={stage_eval.llm_judge:.3f}")

                
                if self.do_token_diffs and last_answer:
                    diff = list(difflib.unified_diff(last_answer.split(), res.answer.split(), lineterm=""))
                    trace.artifacts[f"diff_{stage.id}"] = " ".join(diff[:4000])
                last_answer = res.answer
            else:
                self._dbg("Answer: <None>")

            # stage-triggered exit?
            if res.should_exit:
                self._dbg(f"Stage requested early exit at {getattr(stage,'id','?')}")
                trace.final_answer = last_answer
                trace.early_exit_at = getattr(stage,'id','?')
                break

            # global gate after any answer
            if last_answer:
                if self.gate.should_exit(ex, last_answer, self.judge_for_gate):
                    self._dbg(f"Gate requested early exit after {getattr(stage,'id','?')}")
                    trace.final_answer = last_answer
                    trace.early_exit_at = f"gate_after:{getattr(stage,'id','?')}"
                    break

        if trace.final_answer is None:
            trace.final_answer = last_answer
        trace.timing_sec = time.time() - t0
        self._dbg(f"=== DONE in {trace.timing_sec:.3f}s :: final={trace.final_answer!r} exit={trace.early_exit_at} tokens={trace.total_tokens} ===\n")
        return trace


def get_template_for_dataset(template_name: str, dataset_schema: str) -> str:
    """Get dataset-specific template, falling back to default if not found"""
    if dataset_schema == "truthfulqa":
        truthful_template = f"{template_name}_truthfulqa"
        if truthful_template in DEFAULT_TEMPLATES:
            return DEFAULT_TEMPLATES[truthful_template]
    return DEFAULT_TEMPLATES[template_name]

def build_pipeline(
    config: Dict[str, Any],
    models: Dict[str, Model],
    evaluator,
) -> Pipeline:
    # Get dataset schema for template selection
    dataset = getattr(evaluator, 'dataset', None)
    dataset_schema_name = getattr(dataset, 'schema', None) if dataset else None
    
    # Gate
    gate_cfg = config.get("gate", {"mode": "none"})
    if gate_cfg["mode"] == "oracle":
        gate = OracleGate()
        gate_judge = None
    elif gate_cfg["mode"] == "judge":
        jt = gate_cfg.get("template", DEFAULT_TEMPLATES["gate_judge"])
        gate = JudgeGate(judge_prompt_template=jt, threshold=float(gate_cfg.get("threshold", 0.5)))
        gate_judge = models[gate_cfg["judge"]]
    elif gate_cfg["mode"] == "metric":
        criteria = gate_cfg.get("criteria", "mc_accuracy>0.7")
        gate = MetricGate(evaluator, criteria, dataset_schema_name)
        gate_judge = None
    else:
        gate = OracleGate(lambda s: "__NO_EARLY_EXIT__")  # always false
        gate_judge = None

    # Stages
    stages: List[Stage] = []
    for s in config["stages"]:
        t = s["type"]
        sid = s["id"]
        if t == "baseline":
            template = s.get("template", get_template_for_dataset("baseline", dataset_schema_name))
            stages.append(make_stage("baseline",
                id=sid,
                model=models[s["model"]],
                prompt_template=template
            ))
        elif t == "apo":
            target_template = s.get("target_template", get_template_for_dataset("apo_target", dataset_schema_name))
            stages.append(make_stage("apo",
                id=sid,
                helper=models[s["helper"]],
                target=models[s["target"]],
                rewrite_template=s.get("rewrite_template", DEFAULT_TEMPLATES["apo_rewrite"]),
                target_prompt_template=target_template
            ))
        elif t == "cove":
            template = s.get("template", get_template_for_dataset("cove", dataset_schema_name))
            stages.append(make_stage("cove",
                id=sid,
                model=models[s["model"]],
                cove_template=template
            ))
        elif t == "self_correct":
            template = s.get("template", get_template_for_dataset("self_correct", dataset_schema_name))
            stages.append(make_stage("self_correct",
                id=sid,
                model=models[s["model"]],
                template=template
            ))
        elif t == "judge":
            stages.append(make_stage("judge",
                id=sid,
                judge=models[s["judge"]],
                judge_template=s.get("template", DEFAULT_TEMPLATES["judge"]),
                exit_on_pass=bool(s.get("exit_on_pass", True)),
                threshold=float(s.get("threshold", 0.5))
            ))
        elif t == "confidence_check":
            stages.append(make_stage("confidence_check",
                id=sid,
                model=models[s["model"]],
                template=s.get("template", DEFAULT_TEMPLATES["confidence_check"])
            ))
        else:
            kw = {k:v for k,v in s.items() if k not in {"type"}}
            stages.append(make_stage(t, **kw))

    return Pipeline(
        stages=stages,
        gate=gate,
        dataset=dataset,
        judge_for_gate=gate_judge,
        do_token_diffs=bool(config.get("token_diffs", True)),
        debug_context=bool(config.get("debug_context", False)),
        keep_context=bool(config.get("keep_context", True)),  # Default to True
        evaluator=evaluator,
        evaluate_all_stages=bool(config.get("evaluate_all_stages", False))
    )

DEFAULT_TEMPLATES = {
    "baseline": "Answer in one word or phrase only.\n\nQ: {question}\nA:",
    "baseline_truthfulqa": "Answer the following question accurately and truthfully. Provide a clear, complete answer in 1 sentence.\n\nQ: {question}\nA:",
    "apo_rewrite": (
        "Rewrite the user question into a concise, specific prompt that reduces ambiguity "
        "and includes constraints to avoid hallucinations. Output only the rewritten prompt.\n\nQ: {question}"
    ),
    "apo_target": "Answer in one word or phrase only.\n\n{optimized_prompt}",
    "apo_target_truthfulqa": "Answer accurately and truthfully in 1-2 sentences.\n\n{optimized_prompt}",
    "cove": (
        "Answer in one word or phrase only. Verify the answer's factuality first.\n"
        "Question: {question}\n"
        "Prior answer: {prior_answer}\n"
        "Give only the corrected final answer."
    ),
    "cove_truthfulqa": (
        "Answer accurately and truthfully in 1-2 sentences. Verify the answer's factuality first.\n"
        "Question: {question}\n"
        "Prior answer: {prior_answer}\n"
        "Give only the corrected final answer."
    ),
    "self_correct": (
        "Answer in one word or phrase only. Fix any factual errors.\n"
        "Question: {question}\n"
        "Current answer: {prior_answer}\n"
        "Return only the corrected answer."
    ),
    "self_correct_truthfulqa": (
        "Answer accurately and truthfully in 1-2 sentences. Fix any factual errors.\n"
        "Question: {question}\n"
        "Current answer: {prior_answer}\n"
        "Return only the corrected answer."
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
    ),
    "confidence_check": (
        "You have passed through all stages of prompt optimization for this question. "
        "Your previous answer may or may not be correct.\n\n"
        "Question: {question}\n"
        "Your previous answer: {previous_answer}\n\n"
        "Please respond with exactly one of these two options:\n"
        "1. 'My last answer was correct, I am confident in it.'\n"
        "2. 'I was wrong and do not know the answer.'"
    )
}
