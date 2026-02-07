"""Runtime client factory."""
from __future__ import annotations

from runtimes.anthropic_client import AnthropicClient
from runtimes.base import RuntimeConfig, RuntimeName
from runtimes.ollama_client import OllamaClient
from runtimes.openai_client import OpenAIClient


def build_client(config: RuntimeConfig):
    if config.runtime == RuntimeName.OLLAMA:
        return OllamaClient(config)
    if config.runtime == RuntimeName.OPENAI:
        return OpenAIClient(config)
    if config.runtime == RuntimeName.ANTHROPIC:
        return AnthropicClient(config)
    raise ValueError(f"Unsupported runtime: {config.runtime}")
