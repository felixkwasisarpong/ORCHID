"""LLM runtime configuration for worker plane."""
from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel


class LLMRuntime(str, Enum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLMConfig(BaseModel):
    runtime: LLMRuntime = LLMRuntime.OLLAMA
    model: str = "llama3"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None


DEFAULT_BASE_URL: dict[LLMRuntime, str | None] = {
    LLMRuntime.OLLAMA: "http://localhost:11434",
    LLMRuntime.OPENAI: None,
    LLMRuntime.ANTHROPIC: None,
}

DEFAULT_API_ENV: dict[LLMRuntime, str | None] = {
    LLMRuntime.OLLAMA: None,
    LLMRuntime.OPENAI: "OPENAI_API_KEY",
    LLMRuntime.ANTHROPIC: "ANTHROPIC_API_KEY",
}

DEFAULT_BASE_URL_ENV: dict[LLMRuntime, str | None] = {
    LLMRuntime.OLLAMA: "OLLAMA_BASE_URL",
    LLMRuntime.OPENAI: "OPENAI_BASE_URL",
    LLMRuntime.ANTHROPIC: "ANTHROPIC_BASE_URL",
}


def normalize_model(runtime: LLMRuntime, model: str) -> str:
    if "/" in model:
        return model
    prefix = runtime.value
    return f"{prefix}/{model}"


def build_llm_config_payload(config: LLMConfig) -> dict[str, Any]:
    model = normalize_model(config.runtime, config.model)
    base_url = config.base_url or os.getenv(DEFAULT_BASE_URL_ENV.get(config.runtime, "") or "") or DEFAULT_BASE_URL.get(
        config.runtime
    )
    api_env = config.api_key_env or DEFAULT_API_ENV.get(config.runtime)
    api_key = os.getenv(api_env) if api_env else None
    payload: dict[str, Any] = {
        "model": model,
    }
    if base_url:
        payload["base_url"] = base_url
    if api_key:
        payload["api_key"] = api_key
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if config.seed is not None:
        payload["seed"] = config.seed
    return payload
