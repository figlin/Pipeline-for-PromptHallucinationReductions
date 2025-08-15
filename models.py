from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Callable
import difflib
import time
import uuid
import os
import requests
import json

from dataset import dataset  # Assuming dataset.py is in the same directory
from main import main
from pipeline import pipeline
from stages import stages

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
        last = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        txt = f"{last}"
        ptok = max(1, len(prompt)//4)
        ctok = max(1, len(txt)//4)
        return ModelResponse(text=txt, prompt_tokens=ptok, completion_tokens=ctok, cost=(ptok+ctok)*1e-6)

# -------------------------
# ScaleDown direct generator
# -------------------------

class ScaledownDirectModel:
    """
    Calls a ScaleDown endpoint that BOTH compresses and generates with the
    provider model specified in payload['model'] (e.g., 'gemini/gemini-pro').
    """
    def __init__(
        self,
        api_key: str,
        endpoint: str,                  # e.g., https://api.scaledown.xyz/<your-generate-endpoint>
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
            # REQUIRED by your deployment: 'context' must be a STRING
            "context": params.pop("context", ""),
        }
        if "temperature" in params and params["temperature"] is not None:
            payload["temperature"] = params["temperature"]

        # final guard to force string context
        if not isinstance(payload["context"], str):
            payload["context"] = ""

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        # right before requests.post(...) in ScaledownDirectModel.generate:
        if os.getenv("PIPE_DEBUG_HTTP", "0") != "0":
            print("POST", self.endpoint, "payload:", json.dumps(payload)[:500])

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

            # Treat {"detail": ...} or {"error": ...} as failure
            if isinstance(data, dict) and ("detail" in data or "error" in data):
                snip = json.dumps(data)[:300]
                raise RuntimeError(f"ScaleDown error JSON; body_snip={snip}")

            if isinstance(data, dict):
                text = (
                    data.get("output")
                    or data.get("text")
                    or data.get("response")
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

        # Best-effort usage accounting
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

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "model": self.sd_model,
            "scaledown": {"rate": self.rate},
            # REQUIRED: context must be a string
            "context": ""
        }

        # final guard
        if not isinstance(payload["context"], str):
            payload["context"] = ""

        try:
            r = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_sec,
            )

            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")

            ctype = (r.headers.get("Content-Type", "") or "").lower()
            body = r.text or ""

            if "application/json" in ctype:
                try:
                    data = r.json()
                except Exception:
                    raise RuntimeError("Invalid JSON")

                if isinstance(data, dict) and ("detail" in data or "error" in data):
                    raise RuntimeError("ScaleDown error JSON")

                for key in ("compressed_prompt", "compressed", "text", "output"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        return val

                raise RuntimeError("Unexpected JSON response")

            if body.strip() and not body.strip().startswith('{"detail"'):
                return body

            raise RuntimeError("Empty/invalid text")

        except Exception:
            # Any problem: use the original prompt
            return prompt

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        compressed = self._compress(prompt)
        resp = self.base.generate(compressed, **kwargs)
        resp.meta = {
            **resp.meta,
            "scaledown": {
                "enabled": True,
                "rate": self.rate,
                "sd_model": self.sd_model,
                "original_len": len(prompt),
                "compressed_len": len(compressed),
                "saved_chars": max(0, len(prompt) - len(compressed)),
            }
        }
        resp.meta["base_model_name"] = getattr(self.base, "name", "unknown")
        self.name = f"scaledown({resp.meta['base_model_name']})"
        return resp

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
        txt = (r.text or "").lower()
        p = 0.5
        if "p=" in txt:
            try:
                p = float(txt.split("p=")[-1].split(">")[0])
            except Exception:
                p = 0.5
        return ("pass" in txt) and (p >= self.threshold)
