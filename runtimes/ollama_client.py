"""Ollama runtime client."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional

import httpx

from runtimes.base import ChatResult, RuntimeClient, RuntimeConfig, TokenUsage


@dataclass
class OllamaClient(RuntimeClient):
    config: RuntimeConfig
    base_url: str = "http://localhost:11434"

    async def chat(self, messages: List[Dict[str, Any]], seed: Optional[int] = None) -> ChatResult:
        base_url = os.getenv("OLLAMA_BASE_URL", self.base_url)
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ChatResult(
            content=data.get("message", {}).get("content", ""),
            usage=usage,
            cost_usd=0.0,
        )
