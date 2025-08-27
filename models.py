#!/usr/bin/env python
from __future__ import annotations
from typing import Any, Dict, Optional, Protocol
import os
import json
import requests

from utils import ModelResponse

# -------------------------
# Model adapter protocol
# -------------------------

class Model(Protocol):
    name: str
    def generate(self, prompt: str, **kwargs) -> ModelResponse: ...

# -------------------------
# Echo model (for smoke tests)
# -------------------------

class EchoModel:
    name = "echo-model"
    def generate(self, prompt: str, temperature: float = 0.0, **kwargs) -> ModelResponse:
        last = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        txt = f"{last}"
        ptok = max(1, len(prompt)//4)
        ctok = max(1, len(txt)//4)
        return ModelResponse(text=txt, prompt_tokens=ptok, completion_tokens=ctok, cost=(ptok+ctok)*1e-6)

# -------------------------
# ScaleDown direct generator (compress + generate in one call)
# -------------------------

class ScaledownDirectModel:
    """
    Calls a ScaleDown endpoint that BOTH compresses and generates with the
    provider model specified in payload['model'] (e.g., 'gemini/gemini-pro').
    Use env SCALEDOWN_CHAT_URL for endpoint.
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
            "context": params.pop("context", ""),
        }
        if "temperature" in params and params["temperature"] is not None:
            payload["temperature"] = params["temperature"]

        if not isinstance(payload["context"], str):
            payload["context"] = ""

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}

        if os.getenv("PIPE_DEBUG_HTTP", "0") != "0":
            print("POST", self.endpoint, "payload:", json.dumps(payload)[:500])

        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)

        if r.status_code >= 400:
            snip = (r.text or "")[:300].replace("\n", "\\n")
            raise RuntimeError(f"ScaleDown HTTP {r.status_code}; body_snip={snip}")

        ctype = (r.headers.get("Content-Type", "") or "").lower()
        raw = r.text or ""
        text = ""
        data: Optional[Dict[str, Any]] = None
        scaledown_usage: Dict[str, Any] = {}

        if "application/json" in ctype:
            try:
                data = r.json()
            except Exception:
                data = None

            if isinstance(data, dict) and ("detail" in data or "error" in data):
                snip = json.dumps(data)[:300]
                raise RuntimeError(f"ScaleDown error JSON; body_snip={snip}")

            if isinstance(data, dict):
                # Many ScaleDown deployments use these fields:
                text = (
                    data.get("output")
                    or data.get("text")
                    or data.get("response")
                    or data.get("output_text")
                    or data.get("full_response", "")
                    or (data.get("choices", [{}])[0].get("text", "")
                        if isinstance(data.get("choices"), list) else "")
                    or (data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if isinstance(data.get("choices"), list) else "")
                    or ""
                )
                # Capture usage
                for k in ("full_usage", "compressed_usage", "comparison", "compressed_prompt"):
                    if k in data:
                        scaledown_usage[k] = data[k]

        if not text:
            text = raw.strip()
        if not text:
            raise RuntimeError("ScaleDown call returned no usable text")

        ptok = max(1, len(prompt)//4)
        ctok = max(1, len(text)//4)
        cost = (ptok + ctok) * 1e-6

        meta: Dict[str, Any] = {"status": r.status_code}
        if scaledown_usage:
            meta["scaledown_usage"] = scaledown_usage

        return ModelResponse(
            text=text.strip(),
            prompt_tokens=ptok,
            completion_tokens=ctok,
            cost=cost,
            meta=meta
        )

# -------------------------
# ScaleDown Compression Wrapper (compress then call base model)
# -------------------------

class ScaleDownCompressionWrapper:
    """
    Wrap a base Model and compress the prompt via ScaleDown before calling base.generate().
    Falls back to original prompt on any error.
    """
    name = "scaledown-compression-wrapper"

    def __init__(
        self,
        base: Model,
        api_key: str,
        rate: float = 0.7,
        sd_model: str = "gemini/gemini-pro",
        endpoint: str = "https://api.scaledown.xyz/compress/raw",
        only_on_min_chars: int = 0,
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
            "context": ""
        }

        if not isinstance(payload["context"], str):
            payload["context"] = ""

        try:
            r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
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
        self.name = f"scaledown-compression({resp.meta['base_model_name']})"
        return resp

# -------------------------
# ScaleDown LLM Wrapper (direct prompting through SD /compress)
# -------------------------

class ScaleDownLLMWrapper:
    """
    Direct ScaleDown LLM wrapper that prompts LLMs through the ScaleDown API (/compress).
    """
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        rate: float = 0.7,
        endpoint: str = "https://api.scaledown.xyz/compress",
        timeout: float = 30.0,
        default_params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.rate = rate
        self.endpoint = endpoint
        self.timeout = timeout
        self.default_params = default_params or {}
        self.name = f"scaledown-llm({model})"

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        params = {**self.default_params, **kwargs}
        context = params.get("context", "")

        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }

        rate_value = float(self.rate) if isinstance(self.rate, (int, float, str)) else 0.0

        payload = {
            "context": context,
            "prompt": prompt,
            "model": self.model,
            "scaledown": {"rate": rate_value}
        }

        print(f"[DEBUG] ScaleDown LLM Request:")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Payload: {json.dumps(payload, indent=2)}")

        r = requests.post(self.endpoint, headers=headers, data=json.dumps(payload), timeout=self.timeout)

        print(f"[DEBUG] ScaleDown Response Status: {r.status_code}")
        print(f"[DEBUG] ScaleDown Response: {r.text[:500]}...")

        if r.status_code != 200:
            raise RuntimeError(f"ScaleDown LLM HTTP {r.status_code}: {r.text}")

        result = r.json()

        if "detail" in result and ("error" in str(result["detail"]).lower() or "failed" in str(result["detail"]).lower()):
            raise RuntimeError(f"ScaleDown LLM error: {result['detail']}")

        text = ""
        for key in ["full_response", "compressed_response", "response", "text", "output", "completion", "generated_text"]:
            if key in result and isinstance(result[key], str) and result[key].strip():
                text = result[key].strip()
                break
        if not text:
            raise RuntimeError(f"ScaleDown LLM returned no usable text: {result}")

        usage_info = result.get("full_usage", {})
        prompt_tokens = usage_info.get("tokens", max(1, len(prompt)//4))
        completion_tokens = max(1, len(text)//4)
        cost = usage_info.get("cost", (prompt_tokens + completion_tokens) * 1e-6)

        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            meta={
                "status": r.status_code,
                "scaledown_model": self.model,
                "scaledown_rate": rate_value,
                "full_response": result
            }
        )

# -------------------------
# Gemini API model
# -------------------------

class GeminiModel:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        endpoint: str = "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
        timeout: float = 30.0,
        default_params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint.format(model=model)
        self.timeout = timeout
        self.default_params = default_params or {"temperature": 0.0}
        self.name = f"gemini:{model}"

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        params = {**self.default_params, **kwargs}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": params.get("temperature", 0.0),
                "maxOutputTokens": params.get("max_tokens", 2048),
            }
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.endpoint}?key={self.api_key}"
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if r.status_code >= 400:
            snip = (r.text or "")[:300].replace("\n", "\\n")
            raise RuntimeError(f"Gemini HTTP {r.status_code}; body_snip={snip}")

        try:
            data = r.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"Gemini returned invalid JSON: {r.text}")

        text = ""
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            if parts and "text" in parts[0]:
                text = parts[0]["text"]

        if not text:
            raise RuntimeError(f"Gemini call returned no usable text: {data}")

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", max(1, len(prompt)//4))
        completion_tokens = usage.get("candidatesTokenCount", max(1, len(text)//4))

        return ModelResponse(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=(prompt_tokens + completion_tokens) * 1e-6,
            meta={"status": r.status_code}
        )

# -------------------------
# Ollama local model
# -------------------------

class OllamaModel:
    def __init__(
        self,
        model: str,
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 60.0,
        default_params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.name = f"ollama:{model}"
        self.endpoint = endpoint
        self.timeout = timeout
        self.default_params = default_params or {}

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        params = {**self.default_params, **kwargs}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": params,
        }

        headers = {"Content-Type": "application/json"}
        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)

        if r.status_code >= 400:
            snip = (r.text or "")[:300].replace("\n", "\\n")
            raise RuntimeError(f"Ollama HTTP {r.status_code}; body_snip={snip}")

        try:
            data = r.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"Ollama returned invalid JSON: {r.text}")

        text = data.get("response", "")
        if not text:
            raise RuntimeError("Ollama call returned no usable text")

        return ModelResponse(
            text=text.strip(),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            cost=0.0,  # local model
            meta={"status": r.status_code}
        )
