#!/usr/bin/env python
from __future__ import annotations
from typing import Any, Dict, Optional, Protocol, Callable
import os
import requests
import json
from dataclasses import dataclass

from utils import ModelResponse 
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
# ScaleDown Compression Wrapper for API usage
# -------------------------

class ScaleDownCompressionWrapper:
    """
    Wraps a base Model and compresses the prompt via ScaleDown before calling base.generate().
    If ScaleDown fails, it falls back to the original prompt.
    """
    name = "scaledown-compression-wrapper"

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
        self.name = f"scaledown-compression({resp.meta['base_model_name']})"
        return resp

# -------------------------
# ScaleDown LLM Wrapper (Direct LLM prompting via ScaleDown)
# -------------------------

class ScaleDownLLMWrapper:
    """
    Direct ScaleDown LLM wrapper that prompts LLMs through the ScaleDown API.
    Uses the /compress endpoint for actual LLM prompting, not just compression.
    Based on the testAPI.py framework.
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
        """Generate response using ScaleDown's LLM endpoint"""
        params = {**self.default_params, **kwargs}
        context = params.get("context", "")
        
        # Headers
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        # Convert rate to float if it's a string
        rate_value = self.rate
        if isinstance(rate_value, str) and rate_value.replace('.', '').isdigit():
            rate_value = float(rate_value)
        
        # Payload format that works with ScaleDown
        payload = {
            "prompt": prompt,
            "model": self.model,
            "rate": rate_value,
            "context": context  # Always include context, even if empty
        }
        
        try:
            # Debug logging
            print(f"[DEBUG] ScaleDown LLM Request:")
            print(f"  Endpoint: {self.endpoint}")
            print(f"  Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                self.endpoint, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            print(f"[DEBUG] ScaleDown Response Status: {response.status_code}")
            print(f"[DEBUG] ScaleDown Response: {response.text[:500]}...")
            
            if response.status_code != 200:
                raise RuntimeError(f"ScaleDown LLM HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Check for errors in response
            if "detail" in result and ("error" in str(result["detail"]).lower() or "failed" in str(result["detail"]).lower()):
                raise RuntimeError(f"ScaleDown LLM error: {result['detail']}")
            
            # Extract the response text - try different possible keys
            text = ""
            for key in ["response", "text", "output", "completion", "generated_text"]:
                if key in result and isinstance(result[key], str) and result[key].strip():
                    text = result[key].strip()
                    break
            
            if not text:
                raise RuntimeError(f"ScaleDown LLM returned no usable text: {result}")
            
            # Estimate token counts (ScaleDown might not provide these)
            prompt_tokens = result.get("prompt_tokens", max(1, len(prompt)//4))
            completion_tokens = result.get("completion_tokens", max(1, len(text)//4))
            
            return ModelResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=(prompt_tokens + completion_tokens) * 1e-6,  # rough estimate
                meta={
                    "status": response.status_code,
                    "scaledown_model": self.model,
                    "scaledown_rate": rate_value,
                    "full_response": result
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"ScaleDown LLM generation failed: {str(e)}")

# -------------------------
# Gemini API model
# -------------------------

class GeminiModel:
    """
    Direct Gemini API client without ScaleDown compression.
    """
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
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": params.get("temperature", 0.0),
                "maxOutputTokens": params.get("max_tokens", 2048),
            }
        }

        headers = {
            "Content-Type": "application/json",
        }
        
        # Add API key to URL for Gemini
        url = f"{self.endpoint}?key={self.api_key}"
        
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if r.status_code >= 400:
            snip = (r.text or "")[:300].replace("\n", "\\n")
            raise RuntimeError(f"Gemini HTTP {r.status_code}; body_snip={snip}")

        try:
            data = r.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"Gemini returned invalid JSON: {r.text}")

        # Extract text from Gemini response format
        text = ""
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    text = parts[0]["text"]

        if not text:
            raise RuntimeError(f"Gemini call returned no usable text: {data}")

        # Extract usage info if available
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", max(1, len(prompt)//4))
        completion_tokens = usage.get("candidatesTokenCount", max(1, len(text)//4))
        
        return ModelResponse(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=(prompt_tokens + completion_tokens) * 1e-6,  # rough estimate
            meta={"status": r.status_code}
        )

# -------------------------
# Ollama local model (for local LLM testing)
# -------------------------

class OllamaModel:
    """
    Adapter for local Ollama models.
    See https://github.com/ollama/ollama/blob/main/docs/api.md
    """
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
            cost=0.0,  # Local model, no cost
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
        txt = (r.text or "").lower()
        p = 0.5
        if "p=" in txt:
            try:
                p = float(txt.split("p=")[-1].split(">")[0])
            except Exception:
                p = 0.5
        return ("pass" in txt) and (p >= self.threshold)
