"""Gemini runtime client."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional

import httpx

from runtimes.base import ChatResult, RuntimeClient, RuntimeConfig, TokenUsage
from runtimes.pricing import estimate_cost_usd


def _to_gemini_messages(messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
    system_msg: Optional[str] = None
    contents: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        text = str(msg.get("content", ""))
        if role == "system" and system_msg is None:
            system_msg = text
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    return system_msg, contents


@dataclass
class GeminiClient(RuntimeClient):
    config: RuntimeConfig
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key: Optional[str] = None

    async def chat(self, messages: List[Dict[str, Any]], seed: Optional[int] = None) -> ChatResult:
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

        system_msg, contents = _to_gemini_messages(messages)
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system_msg:
            payload["systemInstruction"] = {"parts": [{"text": system_msg}]}
        if seed is not None:
            payload["generationConfig"]["seed"] = seed

        url = f"{self.base_url}/models/{self.config.model}:generateContent"
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            response = await client.post(url, params={"key": key}, json=payload)
            response.raise_for_status()
            data = response.json()

        text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = ((candidates[0].get("content") or {}).get("parts") or [])
            text_parts = [str(part.get("text", "")) for part in parts if isinstance(part, dict)]
            text = "".join(text_parts)

        usage_data = data.get("usageMetadata", {}) or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_data.get("promptTokenCount", 0) or 0),
            completion_tokens=int(usage_data.get("candidatesTokenCount", 0) or 0),
            total_tokens=int(usage_data.get("totalTokenCount", 0) or 0),
            cached_prompt_tokens=int(usage_data.get("cachedContentTokenCount", 0) or 0),
        )
        if usage.total_tokens <= 0:
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return ChatResult(
            content=text,
            usage=usage,
            cost_usd=estimate_cost_usd("gemini", self.config.model, usage),
        )
