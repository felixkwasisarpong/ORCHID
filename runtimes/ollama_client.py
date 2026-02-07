"""Ollama runtime client (local)."""
from __future__ import annotations

from typing import Any

import httpx

from runtimes.base import LLMResponse, Message, RuntimeConfig, normalize_model, resolve_base_url


class OllamaClient:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.base_url = resolve_base_url(config) or "http://localhost:11434"
        self.model = normalize_model(config.runtime, config.model).split("/", 1)[-1]
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=config.timeout_s)

    async def complete(self, messages: list[Message], system_prompt: str | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }
        if self.config.max_tokens is not None:
            payload["options"]["num_predict"] = self.config.max_tokens
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed
        if self.config.json_mode:
            payload["format"] = "json"
        if system_prompt:
            payload["messages"] = [{"role": "system", "content": system_prompt}, *messages]

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content") or ""
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }
        return LLMResponse(content=content, model=self.model, usage=usage)

    async def close(self) -> None:
        await self._client.aclose()
