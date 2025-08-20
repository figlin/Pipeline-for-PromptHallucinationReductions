from __future__ import annotations
from typing import Any, Dict, List, Optional
import difflib
import time
import logging

from .dataset_loader import Example
from .models import Model
from .gates import GatePolicy, OracleGate, JudgeGate
from .stages import Stage, make_stage
from .core_types import RunTrace
from .templates import DEFAULT_TEMPLATES


# -------------------------
# Pipeline runner
# -------------------------

class Pipeline:
    def __init__(
        self,
        stages: List[Stage],
        gate: GatePolicy,
        judge_for_gate: Optional[Model] = None,
        do_token_diffs: bool = True,
        debug: bool = False,            
        debug_maxlen: int = 220         
    ):
        self.stages = stages
        self.gate = gate
        self.judge_for_gate = judge_for_gate
        self.do_token_diffs = do_token_diffs
        self.debug = debug
        self.debug_maxlen = debug_maxlen
        self.logger = logging.getLogger("pipeline")

    def _dbg(self, *parts):
        if self.debug:
            try:
                self.logger.debug(" ".join(str(p) for p in parts))
            except Exception:
                print(*parts)

    def run_one(self, ex: Example) -> RunTrace:
        t0 = time.time()
        trace = RunTrace(qid=ex.qid, question=ex.question)
        ctx: Dict[str, Any] = {}
        last_answer: Optional[str] = None

        self.logger.info("run.start qid=%s question=%r", ex.qid, ex.question)

        for stage in self.stages:
            if last_answer is not None:
                ctx["last_answer"] = last_answer

            self._dbg(f"\n-- Stage {getattr(stage,'id','?')} ({stage.__class__.__name__}) --")

            res = stage.run(ex, ctx)
            trace.stage_results.append(res)

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
                add_usage(res.model_usage)
                dbg_usage(res.model_usage)
            else:
                for sub_label, sub in res.model_usage.items():
                    add_usage(sub)
                    dbg_usage(sub, label=f"usage.{sub_label}")


            # candidate answer
            if res.answer is not None:
                ans_snip = res.answer[:self.debug_maxlen] + ("..." if len(res.answer) > self.debug_maxlen else "")
                self._dbg("Answer:", ans_snip)
                if self.do_token_diffs and last_answer:
                    diff = list(difflib.unified_diff(last_answer.split(), res.answer.split(), lineterm=""))
                    trace.artifacts[f"diff_{stage.id}"] = " ".join(diff[:4000])
                last_answer = res.answer
            else:
                self._dbg("Answer: <None>")

            if res.should_exit:
                self._dbg(f"Stage requested early exit at {getattr(stage,'id','?')}")
                trace.final_answer = last_answer
                trace.early_exit_at = getattr(stage,'id','?')
                self.logger.info("run.early_exit qid=%s at=%s final=%r", ex.qid, trace.early_exit_at, trace.final_answer)
                break

            if last_answer:
                if self.gate.should_exit(ex, last_answer, self.judge_for_gate):
                    self._dbg(f"Gate requested early exit after {getattr(stage,'id','?')}")
                    trace.final_answer = last_answer
                    trace.early_exit_at = f"gate_after:{getattr(stage,'id','?')}"
                    self.logger.info("run.gate_exit qid=%s at=%s final=%r", ex.qid, trace.early_exit_at, trace.final_answer)
                    break

        if trace.final_answer is None:
            trace.final_answer = last_answer
        trace.timing_sec = time.time() - t0
        self._dbg(f"=== DONE in {trace.timing_sec:.3f}s :: final={trace.final_answer!r} exit={trace.early_exit_at} tokens={trace.total_tokens} ===\n")
        self.logger.info(
            "run.done qid=%s final=%r exit=%s tokens=%s cost=%.6f sec=%.3f",
            ex.qid,
            trace.final_answer,
            trace.early_exit_at,
            trace.total_tokens,
            trace.total_cost,
            trace.timing_sec,
        )
        return trace


def build_pipeline(
    config: Dict[str, Any],
    models: Dict[str, Model],
) -> Pipeline:
    # Gate
    gate_cfg = config.get("gate", {"mode": "none"})
    if gate_cfg["mode"] == "oracle":
        gate = OracleGate()
        gate_judge = None
    elif gate_cfg["mode"] == "judge":
        jt = gate_cfg.get("template", DEFAULT_TEMPLATES["gate_judge"])
        gate = JudgeGate(judge_prompt_template=jt, threshold=float(gate_cfg.get("threshold", 0.5)))
        gate_judge = models[gate_cfg["judge"]]
    else:
        gate = OracleGate(lambda s: "__NO_EARLY_EXIT__")  # always false
        gate_judge = None

    # Stages
    stages: List[Stage] = []
    for s in config["stages"]:
        t = s["type"]
        sid = s["id"]
        if t == "baseline":
            stages.append(make_stage("baseline",
                id=sid,
                model=models[s["model"]],
                prompt_template=s.get("template", DEFAULT_TEMPLATES["baseline"])
            ))
        elif t == "apo":
            stages.append(make_stage("apo",
                id=sid,
                helper=models[s["helper"]],
                target=models[s["target"]],
                rewrite_template=s.get("rewrite_template", DEFAULT_TEMPLATES["apo_rewrite"]),
                target_prompt_template=s.get("target_template", DEFAULT_TEMPLATES["apo_target"])
            ))
        elif t == "cove":
            stages.append(make_stage("cove",
                id=sid,
                model=models[s["model"]],
                cove_template=s.get("template", DEFAULT_TEMPLATES["cove"])
            ))
        elif t == "self_correct":
            stages.append(make_stage("self_correct",
                id=sid,
                model=models[s["model"]],
                template=s.get("template", DEFAULT_TEMPLATES["self_correct"])
            ))
        elif t == "judge":
            stages.append(make_stage("judge",
                id=sid,
                judge=models[s["judge"]],
                judge_template=s.get("template", DEFAULT_TEMPLATES["judge"]),
                exit_on_pass=bool(s.get("exit_on_pass", True)),
                threshold=float(s.get("threshold", 0.5))
            ))
        else:
            kw = {k:v for k,v in s.items() if k not in {"type"}}
            stages.append(make_stage(t, **kw))

    return Pipeline(
        stages=stages,
        gate=gate,
        judge_for_gate=gate_judge,
        do_token_diffs=bool(config.get("token_diffs", True))
    )
