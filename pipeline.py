from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Callable, Tuple
import difflib
import time
import uuid
import os
import requests
import json


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

# -------------------------
# Model adapter protocol
# -------------------------

class Model(Protocol):
    name: str
    def generate(self, prompt: str, **kwargs) -> ModelResponse: ...

# Example dumb adapter (replace with real OpenAI/Anthropic/etc.)
class EchoModel:
    name = "echo-model"
    def generate(self, prompt: str, temperature: float = 0.0, **kwargs) -> ModelResponse:
        # Echoes last line as "answer"
        last = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        txt = f"{last}"
        # Pretend some token & cost math for instrumentation
        ptok = max(1, len(prompt)//4)
        ctok = max(1, len(txt)//4)
        return ModelResponse(text=txt, prompt_tokens=ptok, completion_tokens=ctok, cost=(ptok+ctok)*1e-6)

class ScaledownDirectModel:
    """
    Calls a ScaleDown endpoint that BOTH compresses and generates with the
    provider model specified in payload['model'] (e.g., 'gemini/gemini-pro').
    """
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str = "gemini/gemini-pro",
        rate: float = 0.7,
        timeout: float = 30.0,
        default_params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.rate = rate
        self.timeout = timeout
        self.default_params = default_params or {"temperature": 0.0}
        self.name = f"scaledown:{model}"

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        params = {**self.default_params, **kwargs}

        payload = {
            "prompt": prompt,
            "model": self.model,
            "scaledown": {"rate": self.rate},
            # REQUIRED by your deployment:
            "context": params.pop("context", {}),  # always include; default to {}
        }
        if "temperature" in params and params["temperature"] is not None:
            payload["temperature"] = params["temperature"]

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)

        # If non-2xx, raise so stages can record the failure instead of using error JSON as an answer
        if r.status_code >= 400:
            snip = (r.text or "")[:300].replace("\n", "\\n")
            raise RuntimeError(f"ScaleDown HTTP {r.status_code}; body_snip={snip}")

        ctype = (r.headers.get("Content-Type", "") or "").lower()
        raw = r.text or ""
        text = ""

        if "application/json" in ctype:
            try:
                data = r.json()
            except Exception:
                data = None
            if isinstance(data, dict):
                text = (
                    data.get("output")
                    or data.get("text")
                    or data.get("response")
                    or data.get("output_text")
                    or (data.get("choices", [{}])[0].get("text", "")
                        if isinstance(data.get("choices"), list) else "")
                    or (data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if isinstance(data.get("choices"), list) else "")
                    or ""
                )

        if not text:
            text = raw.strip()

        if not text:
            raise RuntimeError("ScaleDown call returned no usable text")

        # best-effort accounting
        ptok = max(1, len(prompt)//4)
        ctok = max(1, len(text)//4)
        cost = (ptok + ctok) * 1e-6

        return ModelResponse(
            text=text.strip(),
            prompt_tokens=ptok,
            completion_tokens=ctok,
            cost=cost,
            meta={"status": r.status_code}
        )



# -------------------------
# Scaledown Wrapper for API usage
# -------------------------
class ScaleDownWrappedModel:
    """
    Wraps a base Model and compresses the prompt via ScaleDown before calling base.generate().
    If ScaleDown fails, it falls back to the original prompt.
    """
    name = "scaledown-wrapper"

    def __init__(
        self,
        base: Model,
        api_key: str,
        rate: float = 0.7,
        sd_model: str = "gemini/gemini-pro",
        endpoint: str = "https://api.scaledown.xyz/compress/raw",
        only_on_min_chars: int = 0,         # set >0 to skip compression for short prompts
        timeout_sec: float = 15.0
    ):
        self.base = base
        self.api_key = api_key
        self.rate = rate
        self.sd_model = sd_model
        self.endpoint = endpoint
        self.only_on_min_chars = only_on_min_chars
        self.timeout_sec = timeout_sec

    def _compress(self, prompt: str) -> str:
        if self.only_on_min_chars and len(prompt) < self.only_on_min_chars:
            return prompt

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "model": self.sd_model,
            "scaledown": {"rate": self.rate},
            "context": ""  # string, not {}
        }
        try:
            r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_sec)

            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")

            ctype = (r.headers.get("Content-Type","") or "").lower()
            body = r.text or ""

            if "application/json" in ctype:
                try:
                    data = r.json()
                except Exception:
                    raise RuntimeError("Invalid JSON")
                if isinstance(data, dict) and ("detail" in data or "error" in data):
                    raise RuntimeError("ScaleDown error JSON")
                for key in ("compressed_prompt","compressed","text","output"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        return val

                raise RuntimeError("Unexpected JSON response")

            if body.strip() and not body.strip().startswith('{"detail"'):
                return body

            raise RuntimeError("Empty/invalid text")

        except Exception:
            return prompt  # graceful fallback



    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        params = {**self.default_params, **kwargs}
        payload = {
            "prompt": prompt,
            "model": self.model,
            "scaledown": {"rate": self.rate},
            # REQUIRED by your deployment: context must be a STRING
            "context": params.pop("context", ""),  # "" not {}
        }
        if "temperature" in params and params["temperature"] is not None:
            payload["temperature"] = params["temperature"]

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)

        # Treat non-2xx as error
        if r.status_code >= 400:
            snip = (r.text or "")[:300].replace("\n", "\\n")
            raise RuntimeError(f"ScaleDown HTTP {r.status_code}; body_snip={snip}")

        ctype = (r.headers.get("Content-Type", "") or "").lower()
        raw = r.text or ""
        text = ""

        data = None
        if "application/json" in ctype:
            try:
                data = r.json()
            except Exception:
                data = None

            # IMPORTANT: treat {"detail": ...} or {"error": ...} as failure
            if isinstance(data, dict) and ("detail" in data or "error" in data):
                snip = json.dumps(data)[:300]
                raise RuntimeError(f"ScaleDown error JSON; body_snip={snip}")

            if isinstance(data, dict):
                text = (
                    data.get("output") or data.get("text") or data.get("response")
                    or data.get("output_text")
                    or (data.get("choices",[{}])[0].get("text","")
                        if isinstance(data.get("choices"), list) else "")
                    or (data.get("choices",[{}])[0].get("message",{}).get("content","")
                        if isinstance(data.get("choices"), list) else "")
                    or ""
                )

        if not text:
            text = raw.strip()
        if not text:
            raise RuntimeError("ScaleDown call returned no usable text")

        ptok = max(1, len(prompt)//4)
        ctok = max(1, len(text)//4)
        cost = (ptok + ctok) * 1e-6

        return ModelResponse(
            text=text.strip(),
            prompt_tokens=ptok,
            completion_tokens=ctok,
            cost=cost,
            meta={"status": r.status_code}
        )


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

class JudgeGate:   # online: ask a judge model “is this correct?”
    def __init__(self, judge_prompt_template: str, threshold: float = 0.5):
        self.tpl = judge_prompt_template
        self.threshold = threshold
    def should_exit(self, example: Example, candidate_answer: str, judge: Optional[Model]) -> bool:
        if judge is None:
            return False
        prompt = self.tpl.format(question=example.question, answer=candidate_answer)
        r = judge.generate(prompt, temperature=0.0)
        # Expect judge to emit "PASS <p=0.87>" or "FAIL <p=0.12>" or a JSON line; adjust to your judge format.
        txt = r.text.lower()
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
        prompt = self.tpl.format(question=ex.question, prior_answer=ctx.get("last_answer",""))
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = r.text.strip()
            usage = {"model": self.model.name, **r.__dict__}
            return StageResult(self.id, ans or None, False,
                            evidence={"prompt": prompt},
                            model_usage=usage)
        except Exception as e:
            return StageResult(self.id, None, False,
                            evidence={"prompt": prompt, "error": str(e)},
                            model_usage={"model": self.model.name})


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

        # 1) Ask helper to produce an optimized prompt
        helper_in = self.rewrite_template.format(question=ex.question)
        evidence["helper_in"] = helper_in
        try:
            h = self.helper.generate(helper_in, temperature=0.2)
            optimized = (h.text or "").strip()
            usage["helper"] = {"model": getattr(self.helper, "name", "unknown"), **h.__dict__}
            evidence["optimized"] = optimized
        except Exception as e:
            evidence["helper_error"] = str(e)
            # fall back to original question if helper fails
            optimized = ex.question

        # 2) Ask target using the optimized prompt
        target_in = self.target_prompt_template.format(optimized_prompt=optimized)
        evidence["target_in"] = target_in
        try:
            t = self.target.generate(target_in, temperature=0.0)
            ans = (t.text or "").strip() or None
            usage["target"] = {"model": getattr(self.target, "name", "unknown"), **t.__dict__}
            return StageResult(
                stage_id=self.id,
                answer=ans,
                should_exit=False,
                evidence=evidence,
                model_usage=usage
            )
        except Exception as e:
            evidence["target_error"] = str(e)
            return StageResult(
                stage_id=self.id,
                answer=None,
                should_exit=False,
                evidence=evidence,
                model_usage=usage if usage else {"model": getattr(self.target, "name", "unknown")}
            )



@register_stage("cove")
class CoVeStage:
    """Standardized Chain-of-Verification prompt around a candidate answer (or the question if none)."""
    def __init__(self, id: str, model: Model, cove_template: str):
        self.id, self.model, self.tpl = id, model, cove_template
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
        prompt = self.tpl.format(question=ex.question, prior_answer=ctx.get("last_answer",""))
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = r.text.strip()
            usage = {"model": self.model.name, **r.__dict__}
            return StageResult(self.id, ans or None, False,
                            evidence={"prompt": prompt},
                            model_usage=usage)
        except Exception as e:
            return StageResult(self.id, None, False,
                            evidence={"prompt": prompt, "error": str(e)},
                            model_usage={"model": self.model.name})


@register_stage("self_correct")
class SelfCorrectStage:
    def __init__(self, id: str, model: Model, template: str):
        self.id, self.model, self.tpl = id, model, template
    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
        prompt = self.tpl.format(question=ex.question, prior_answer=ctx.get("last_answer",""))
        try:
            r = self.model.generate(prompt, temperature=0.0)
            ans = r.text.strip()
            usage = {"model": self.model.name, **r.__dict__}
            return StageResult(self.id, ans or None, False,
                            evidence={"prompt": prompt},
                            model_usage=usage)
        except Exception as e:
            return StageResult(self.id, None, False,
                            evidence={"prompt": prompt, "error": str(e)},
                            model_usage={"model": self.model.name})


@register_stage("judge")
class ExternalJudgeStage:
    """Optionally emits should_exit based on judge verdict or returns a verdict for the runner to use."""
    def __init__(self, id: str, judge: Model, judge_template: str,
                 exit_on_pass: bool = True, threshold: float = 0.5):
        self.id = id
        self.judge = judge
        self.tpl = judge_template
        self.exit_on_pass = exit_on_pass
        self.threshold = threshold

    def run(self, ex: Example, ctx: Dict[str, Any]) -> StageResult:
        cand = ctx.get("last_answer", "")  # candidate answer from previous stage(s)
        prompt = self.tpl.format(question=ex.question, answer=cand)
        evidence = {"prompt": prompt}
        try:
            r = self.judge.generate(prompt, temperature=0.0)
            verdict = (r.text or "").strip().lower()
            evidence["verdict"] = verdict

            # Parse optional probability like "PASS <p=0.82>"
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
                answer=cand,  # judge doesn't change content, only validates
                should_exit=exit_flag,
                evidence=evidence,
                model_usage={"model": getattr(self.judge, "name", "unknown"),
                             **r.__dict__}
            )
        except Exception as e:
            evidence["error"] = str(e)
            return StageResult(
                stage_id=self.id,
                answer=cand or None,
                should_exit=False,          # don't early-exit on judge failure
                evidence=evidence,
                model_usage={"model": getattr(self.judge, "name", "unknown")}
            )



# -------------------------
# Pipeline runner
# -------------------------

class Pipeline:
    def __init__(
        self,
        stages: List[Stage],
        gate: GatePolicy,
        judge_for_gate: Optional[Model] = None,
        do_token_diffs: bool = True
    ):
        self.stages = stages
        self.gate = gate
        self.judge_for_gate = judge_for_gate
        self.do_token_diffs = do_token_diffs

    def run_one(self, ex: Example) -> RunTrace:
        t0 = time.time()
        trace = RunTrace(qid=ex.qid, question=ex.question)
        ctx: Dict[str, Any] = {}
        last_answer: Optional[str] = None

        for stage in self.stages:
            if last_answer is not None:
                ctx["last_answer"] = last_answer

            res = stage.run(ex, ctx)
            trace.stage_results.append(res)

            # accounting
            def add_usage(u):
                if not u: return
                if "prompt_tokens" in u: trace.total_tokens += int(u["prompt_tokens"])
                if "completion_tokens" in u: trace.total_tokens += int(u["completion_tokens"])
                if "cost" in u: trace.total_cost += float(u["cost"])

            # per-stage usage can be nested (APO helper/target)
            if "prompt_tokens" in res.model_usage:
                add_usage(res.model_usage)
            else:
                for _, sub in res.model_usage.items():
                    add_usage(sub)

            # candidate answer
            if res.answer is not None:
                if self.do_token_diffs and last_answer:
                    diff = list(difflib.unified_diff(
                        last_answer.split(), res.answer.split(), lineterm=""
                    ))
                    trace.artifacts[f"diff_{stage.id}"] = " ".join(diff[:4000])  # cap
                last_answer = res.answer

            # stage can signal exit itself (e.g., judge)
            if res.should_exit:
                trace.final_answer = last_answer
                trace.early_exit_at = stage.id
                break

            # global gate after any stage that produced an answer
            if last_answer:
                if self.gate.should_exit(ex, last_answer, self.judge_for_gate):
                    trace.final_answer = last_answer
                    trace.early_exit_at = f"gate_after:{stage.id}"
                    break

        if trace.final_answer is None:
            trace.final_answer = last_answer
        trace.timing_sec = time.time() - t0
        return trace

# -------------------------
# Example config & factory
# -------------------------



DEFAULT_TEMPLATES = {
    "baseline": "Q: {question}\nA:",
    "apo_rewrite": (
        "Rewrite the user question into a concise, specific prompt that reduces ambiguity "
        "and includes constraints to avoid hallucinations. Output only the rewritten prompt.\n\nQ: {question}"
    ),
    "apo_target": "Use this optimized prompt:\n\n{optimized_prompt}\n\nAnswer succinctly and cite key facts.",
    "cove": (
        "You are verifying an answer’s factuality via Chain-of-Verification.\n"
        "Question: {question}\n"
        "Prior answer: {prior_answer}\n"
        "1) List claims.\n2) Verify each claim with independent checks.\n3) Give a corrected final answer only."
    ),
    "self_correct": (
        "Revise only factual errors at temperature 0.\n"
        "Question: {question}\n"
        "Current answer: {prior_answer}\n"
        "Return a corrected final answer only."
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
    )
}

def build_pipeline(
    config: Dict[str, Any],
    models: Dict[str, Model],
) -> Pipeline:
    """
    config example:
    {
      "stages": [
        {"type": "baseline", "id": "s0", "model": "target"},
        {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
        {"type": "cove", "id": "s2", "model": "target"},
        {"type": "self_correct", "id": "s3", "model": "target"},
        {"type": "judge", "id": "s4", "judge": "judge", "exit_on_pass": True, "threshold": 0.7}
      ],
      "gate": {"mode": "judge", "judge": "judge", "threshold": 0.7}
    }
    """
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
        # no early exit gate (stages can still exit themselves)
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
            # any custom/plug-in stage name is supported if registered
            kw = {k:v for k,v in s.items() if k not in {"type"}}
            stages.append(make_stage(t, **kw))

    return Pipeline(
        stages=stages,
        gate=gate,
        judge_for_gate=gate_judge,
        do_token_diffs=bool(config.get("token_diffs", True))
    )

# -------------------------
# Minimal runnable demo
# -------------------------

if __name__ == "__main__":
    # Plug your real adapters here (OpenAI/Anthropic/vLLM). EchoModel is just a stub.

    # Prefer env var; fall back to literal if you must.
    SCALEDOWN_API_KEY = os.environ.get("SCALEDOWN_API_KEY", "YOUR_API_KEY")

    SCALEDOWN_API_KEY = os.environ["SCALEDOWN_API_KEY"]
    SCALEDOWN_CHAT_URL = os.environ["SCALEDOWN_CHAT_URL"]  # set this to your ScaleDown “generate” endpoint

    models = {
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
        # keep judge deterministic; you can set rate=0.0 if you prefer no compression there
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
        "gate": {"mode": "judge", "judge": "judge", "threshold": 0.8, "template": DEFAULT_TEMPLATES["gate_judge"]},
        "token_diffs": True
    }

    pipe = build_pipeline(config, models)
    ex = Example(qid=str(uuid.uuid4()), question="What is the capital of France?", y_true="Paris")
    trace = pipe.run_one(ex)
    print("Final:", trace.final_answer, "Exit at:", trace.early_exit_at, "Tokens:", trace.total_tokens)


    
