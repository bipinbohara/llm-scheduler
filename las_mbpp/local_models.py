"""OpenAI-compatible provider for the independently hosted study models."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from las_mbpp.telemetry import HTTP_HOOKS


@dataclass(frozen=True)
class LocalModel:
    tier: int
    host: str
    env_prefix: str

    def endpoint(self) -> str:
        url = os.environ.get(f"{self.env_prefix}_BASE_URL") or f"http://{self.host}:80/v1"
        parts = urlsplit(url)
        if (
            parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.query or parts.fragment
        ):
            raise ValueError(f"Invalid endpoint in {self.env_prefix}_BASE_URL. Use an HTTP(S) API base URL without credentials.")
        return url.rstrip("/")

    def key(self) -> str:
        return os.environ.get(f"{self.env_prefix}_API_KEY", "").strip()


LOCAL_MODELS = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": LocalModel(0, "192.168.0.211", "NEMOTRON_NANO"),
    "openai/gpt-oss-20b": LocalModel(0, "192.168.0.215", "GPT_OSS_20B"),
    "Qwen/Qwen3.5-35B-A3B-FP8": LocalModel(0, "192.168.0.217", "QWEN_35B"),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": LocalModel(1, "192.168.0.209", "NEMOTRON_SUPER"),
    "openai/gpt-oss-120b": LocalModel(1, "192.168.0.210", "GPT_OSS_120B"),
    "Qwen/Qwen3.5-122B-A10B-FP8": LocalModel(1, "192.168.0.205", "QWEN_122B"),
}


def get_local_model(name: str) -> LocalModel:
    try:
        return LOCAL_MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown local model {name!r}. Choose one of: {', '.join(LOCAL_MODELS)}") from None


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


class LocalChatClient:
    def __init__(self, *, max_tokens: int, temperature: float) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError('Local inference requires httpx. Install with: pip install -e ".[local]"') from exc
        self.httpx = httpx
        self.max_tokens = max_tokens
        self.temperature = temperature

    def call(self, *, system_prompt: str, user_prompt: str, model: str) -> dict:
        spec = get_local_model(model)
        key = spec.key()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        start = time.perf_counter()
        status = None
        finish_reason = ""
        counts = {}
        text = ""
        error = ""
        payload = {}
        reasoning_only = False
        failure_category = None
        try:
            with self.httpx.Client(
                timeout=self.httpx.Timeout(300.0, connect=10.0), trust_env=False, event_hooks=HTTP_HOOKS,
            ) as client:
                response = client.post(f"{spec.endpoint()}/chat/completions", json=request, headers=headers)
            status = response.status_code
            response.raise_for_status()
            payload = response.json()
            counts = payload.get("usage") or {}
            choice = payload["choices"][0]
            finish_reason = choice.get("finish_reason") or ""
            message = choice["message"]
            content = message.get("content")
            text = content.strip() if isinstance(content, str) else ""
            reasoning_only = not text and bool(message.get("reasoning_content") or message.get("reasoning"))
            if finish_reason in {"length", "content_filter"}:
                error = f"Local response finish reason: {finish_reason}."
            elif not text:
                error = "Local response contained no answer text."
        except self.httpx.HTTPStatusError as exc:
            error = f"Local endpoint returned HTTP {status}."
            if status in {400, 413, 422}:
                try:
                    details = exc.response.json()
                except ValueError:
                    details = None
                clue = str(details).lower()
                if "context" in clue and ("length" in clue or "window" in clue):
                    failure_category = "context_limit"
        except (self.httpx.RequestError, KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            error = f"Local request or response failed ({type(exc).__name__})."

        def token_count(name: str) -> int:
            value = counts.get(name) if isinstance(counts, dict) else None
            return value if type(value) is int and value >= 0 else 0

        input_tokens = token_count("prompt_tokens")
        output_tokens = token_count("completion_tokens")
        provider_usage = None
        if isinstance(counts, dict):
            provider_usage = {
                canonical: counts[native]
                for canonical, native in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                                          ("total_tokens", "total_tokens"))
                if type(counts.get(native)) is int and counts[native] >= 0
            } or None
        return {
            "ok": not error,
            "error": error,
            "output_text": text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": token_count("total_tokens") or input_tokens + output_tokens,
            },
            "latency_sec": time.perf_counter() - start,
            "finish_reason": finish_reason,
            "http_status": status,
            "attempts": 1,
            "usage_reported": bool(counts),
            "reasoning_only": reasoning_only,
            "provider_usage": provider_usage,
            "failure_category": failure_category,
            "reported_model": payload.get("model") if isinstance(payload, dict) else None,
            "response_id": payload.get("id") if isinstance(payload, dict) else None,
        }
