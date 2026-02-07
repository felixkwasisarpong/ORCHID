"""Runtime clients for LLM providers."""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RuntimeName(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RuntimeConfig(BaseModel):
    runtime: RuntimeName = RuntimeName.OLLAMA
    model: str = "llama3"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 60.0
    json_mode: bool = True
    seed: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str
    model: str | None = None
    usage: dict[str, Any] | None = None


DEFAULT_BASE_URL: dict[RuntimeName, str | None] = {
    RuntimeName.OLLAMA: "http://localhost:11434",
    RuntimeName.OPENAI: "https://api.openai.com",
    RuntimeName.ANTHROPIC: "https://api.anthropic.com",
}

DEFAULT_API_ENV: dict[RuntimeName, str | None] = {
    RuntimeName.OLLAMA: None,
    RuntimeName.OPENAI: "OPENAI_API_KEY",
    RuntimeName.ANTHROPIC: "ANTHROPIC_API_KEY",
}

DEFAULT_BASE_URL_ENV: dict[RuntimeName, str | None] = {
    RuntimeName.OLLAMA: "OLLAMA_BASE_URL",
    RuntimeName.OPENAI: "OPENAI_BASE_URL",
    RuntimeName.ANTHROPIC: "ANTHROPIC_BASE_URL",
}


def resolve_base_url(config: RuntimeConfig) -> str | None:
    env_key = DEFAULT_BASE_URL_ENV.get(config.runtime)
    env_value = os.getenv(env_key or "") if env_key else None
    return config.base_url or env_value or DEFAULT_BASE_URL.get(config.runtime)


def resolve_api_key(config: RuntimeConfig) -> str | None:
    env_key = config.api_key_env or DEFAULT_API_ENV.get(config.runtime)
    return os.getenv(env_key) if env_key else None


def normalize_model(runtime: RuntimeName, model: str) -> str:
    if "/" in model:
        return model
    return f"{runtime.value}/{model}"


Message = dict[Literal["role", "content"], str]
