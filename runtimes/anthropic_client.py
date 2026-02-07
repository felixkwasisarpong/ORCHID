"""Anthropic runtime client (cloud)."""
from __future__ import annotations

from typing import Any

import httpx

from runtimes.base import LLMResponse, Message, RuntimeConfig, normalize_model, resolve_api_key, resolve_base_url


class AnthropicClient:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.base_url = resolve_base_url(config) or "https://api.anthropic.com"
        self.model = normalize_model(config.runtime, config.model)
        api_key = resolve_api_key(config)
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic runtime")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=config.timeout_s,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )

    async def complete(self, messages: list[Message], system_prompt: str | None = None) -> LLMResponse:
        system = system_prompt or ""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if system:
            payload["system"] = system
        payload.update(self.config.extra)

        response = await self._client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()
        content_blocks = data.get("content", [])
        text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
        content = "".join(text_parts)
        usage = data.get("usage")
        return LLMResponse(content=content, model=self.model, usage=usage)

    async def close(self) -> None:
        await self._client.aclose()
