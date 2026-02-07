"""Ollama runtime client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from runtimes.base import RuntimeClient, RuntimeConfig


@dataclass
class OllamaClient(RuntimeClient):
    config: RuntimeConfig
    base_url: str = "http://localhost:11434"

    async def chat(self, messages: List[Dict[str, Any]], seed: Optional[int] = None) -> str:
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
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("message", {}).get("content", "")
