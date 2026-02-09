"""Anthropic runtime client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from runtimes.base import ChatResult, RuntimeClient, RuntimeConfig, TokenUsage
from runtimes.pricing import estimate_cost_usd


@dataclass
class AnthropicClient(RuntimeClient):
    config: RuntimeConfig
    base_url: str = "https://api.anthropic.com/v1"
    api_key: Optional[str] = None
    anthropic_version: str = "2023-06-01"

    async def chat(self, messages: List[Dict[str, Any]], seed: Optional[int] = None) -> ChatResult:
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        system_msg = None
        rest_messages = []
        for msg in messages:
            if msg.get("role") == "system" and system_msg is None:
                system_msg = msg.get("content")
            else:
                rest_messages.append(msg)

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": rest_messages,
        }
        if system_msg:
            payload["system"] = system_msg
        headers = {
            "x-api-key": key,
            "anthropic-version": self.anthropic_version,
        }
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            response = await client.post(f"{self.base_url}/messages", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        usage_data = data.get("usage", {}) or {}
        prompt_tokens = int(usage_data.get("input_tokens", 0) or 0)
        completion_tokens = int(usage_data.get("output_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_creation_input_tokens=int(usage_data.get("cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(usage_data.get("cache_read_input_tokens", 0) or 0),
        )
        content = data.get("content", [])
        if content:
            text = content[0].get("text", "")
        else:
            text = ""
        return ChatResult(
            content=text,
            usage=usage,
            cost_usd=estimate_cost_usd("anthropic", self.config.model, usage),
        )
