"""OpenAI runtime client (cloud)."""
from __future__ import annotations

from typing import Any

import httpx

from runtimes.base import LLMResponse, Message, RuntimeConfig, normalize_model, resolve_api_key, resolve_base_url


class OpenAIClient:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.base_url = resolve_base_url(config) or "https://api.openai.com"
        self.model = normalize_model(config.runtime, config.model)
        api_key = resolve_api_key(config)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI runtime")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=config.timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def complete(self, messages: list[Message], system_prompt: str | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if system_prompt:
            payload["messages"] = [{"role": "system", "content": system_prompt}, *messages]
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        payload.update(self.config.extra)

        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        usage = data.get("usage")
        return LLMResponse(content=content, model=self.model, usage=usage)

    async def close(self) -> None:
        await self._client.aclose()
