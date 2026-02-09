"""OpenAI runtime client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from runtimes.base import ChatResult, RuntimeClient, RuntimeConfig, TokenUsage
from runtimes.pricing import estimate_cost_usd


@dataclass
class OpenAIClient(RuntimeClient):
    config: RuntimeConfig
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None

    async def chat(self, messages: List[Dict[str, Any]], seed: Optional[int] = None) -> ChatResult:
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {"Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        usage_data = data.get("usage", {})
        prompt_tokens = int(usage_data.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage_data.get("completion_tokens", 0) or 0)
        total_tokens = int(usage_data.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        cached_prompt_tokens = int(
            (
                usage_data.get("prompt_tokens_details", {}) or {}
            ).get("cached_tokens", 0)
            or 0
        )
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        return ChatResult(
            content=data["choices"][0]["message"]["content"],
            usage=usage,
            cost_usd=estimate_cost_usd("openai", self.config.model, usage),
        )
