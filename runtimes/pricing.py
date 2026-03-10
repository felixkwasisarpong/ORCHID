"""Token pricing helpers for cloud runtimes.

Reference pricing sources checked on 2026-02-10:
- OpenAI API pricing: https://openai.com/api/pricing/
- Anthropic model overview: https://platform.claude.com/docs/en/about-claude/models/overview
- Anthropic prompt caching pricing: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Mistral pricing: https://docs.mistral.ai/getting-started/models/pricing/
- xAI API pricing: https://docs.x.ai/api
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from runtimes.base import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: Optional[float] = None
    cache_write_per_million: Optional[float] = None
    cache_read_per_million: Optional[float] = None
    long_context_threshold_tokens: Optional[int] = None
    input_per_million_long_context: Optional[float] = None
    output_per_million_long_context: Optional[float] = None
    cached_input_per_million_long_context: Optional[float] = None


OPENAI_PRICING: Dict[str, ModelPricing] = {
    "gpt-5.2": ModelPricing(input_per_million=1.75, output_per_million=14.0, cached_input_per_million=0.175),
    "gpt-5.2-thinking": ModelPricing(
        input_per_million=1.75, output_per_million=14.0, cached_input_per_million=0.175
    ),
    "gpt-5.2-chat-latest": ModelPricing(
        input_per_million=1.75, output_per_million=14.0, cached_input_per_million=0.175
    ),
    "gpt-5.2-pro": ModelPricing(input_per_million=21.0, output_per_million=168.0),
    "gpt-4.1": ModelPricing(input_per_million=2.0, output_per_million=8.0, cached_input_per_million=0.5),
    "gpt-4.1-mini": ModelPricing(
        input_per_million=0.4, output_per_million=1.6, cached_input_per_million=0.1
    ),
    "gpt-4.1-nano": ModelPricing(
        input_per_million=0.1, output_per_million=0.4, cached_input_per_million=0.025
    ),
}

ANTHROPIC_PRICING: Dict[str, ModelPricing] = {
    "claude-opus-4-6": ModelPricing(
        input_per_million=5.0,
        output_per_million=25.0,
        cache_write_per_million=6.25,
        cache_read_per_million=0.5,
    ),
    "claude-opus-4.6": ModelPricing(
        input_per_million=5.0,
        output_per_million=25.0,
        cache_write_per_million=6.25,
        cache_read_per_million=0.5,
    ),
    "claude-opus-4.1": ModelPricing(
        input_per_million=15.0,
        output_per_million=75.0,
        cache_write_per_million=18.75,
        cache_read_per_million=1.5,
    ),
    "claude-opus-4": ModelPricing(
        input_per_million=15.0,
        output_per_million=75.0,
        cache_write_per_million=18.75,
        cache_read_per_million=1.5,
    ),
    "claude-sonnet-4.5": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_write_per_million=3.75,
        cache_read_per_million=0.3,
    ),
    "claude-sonnet-4": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_write_per_million=3.75,
        cache_read_per_million=0.3,
    ),
    "claude-3-7-sonnet": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_write_per_million=3.75,
        cache_read_per_million=0.3,
    ),
    "claude-3-5-sonnet": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_write_per_million=3.75,
        cache_read_per_million=0.3,
    ),
    "claude-3-5-haiku": ModelPricing(
        input_per_million=0.8,
        output_per_million=4.0,
        cache_write_per_million=1.0,
        cache_read_per_million=0.08,
    ),
}

GOOGLE_PRICING: Dict[str, ModelPricing] = {
    # Gemini 3 Pro Preview rates are tiered by prompt length (<=200K vs >200K tokens).
    "gemini-3-pro-preview": ModelPricing(
        input_per_million=2.0,
        output_per_million=12.0,
        cached_input_per_million=0.2,
        long_context_threshold_tokens=200_000,
        input_per_million_long_context=4.0,
        output_per_million_long_context=18.0,
        cached_input_per_million_long_context=0.4,
    ),
    "gemini 3 pro (preview)": ModelPricing(
        input_per_million=2.0,
        output_per_million=12.0,
        cached_input_per_million=0.2,
        long_context_threshold_tokens=200_000,
        input_per_million_long_context=4.0,
        output_per_million_long_context=18.0,
        cached_input_per_million_long_context=0.4,
    ),
}

MISTRAL_PRICING: Dict[str, ModelPricing] = {
    # Mistral Large 3 (mistral-large-2512+1)
    "mistral-large-2512+1": ModelPricing(input_per_million=0.5, output_per_million=1.5),
}

XAI_PRICING: Dict[str, ModelPricing] = {
    # Grok Fast pricing from xAI API docs.
    "grok-4.1-fast": ModelPricing(input_per_million=0.2, output_per_million=0.5, cached_input_per_million=0.05),
    "grok-4-1-fast": ModelPricing(
        input_per_million=0.2, output_per_million=0.5, cached_input_per_million=0.05
    ),
    "grok-4-fast-non-reasoning": ModelPricing(
        input_per_million=0.2, output_per_million=0.5, cached_input_per_million=0.05
    ),
}


OLLAMA_PRICING: Dict[str, ModelPricing] = {
    # Ollama runs locally — no API cost. Entries exist so pricing lookup never silently fails.
    "qwen2.5:14b": ModelPricing(input_per_million=0.0, output_per_million=0.0),
    "qwen2.5:7b": ModelPricing(input_per_million=0.0, output_per_million=0.0),
    "qwen2.5": ModelPricing(input_per_million=0.0, output_per_million=0.0),
    "llama3.1": ModelPricing(input_per_million=0.0, output_per_million=0.0),
    "llama3.3": ModelPricing(input_per_million=0.0, output_per_million=0.0),
}

RUNTIME_PRICING_TABLES: Dict[str, Dict[str, ModelPricing]] = {
    "openai": OPENAI_PRICING,
    "anthropic": ANTHROPIC_PRICING,
    "gemini": GOOGLE_PRICING,
    "google": GOOGLE_PRICING,
    "mistral": MISTRAL_PRICING,
    "grok": XAI_PRICING,
    "xai": XAI_PRICING,
    "ollama": OLLAMA_PRICING,
}


MODEL_ALIASES: Dict[str, str] = {
    "gemini 3 pro (preview)": "gemini-3-pro-preview",
    "gemini 3 pro": "gemini-3-pro-preview",
    "claude opus 4.6": "claude-opus-4-6",
    "gpt 5.2": "gpt-5.2",
    "gpt-5.2 thinking": "gpt-5.2-thinking",
    "mistral large 3": "mistral-large-2512+1",
    "mistral large 3 (mistral-large-2512+1)": "mistral-large-2512+1",
    "grok 4.1 fast": "grok-4.1-fast",
    # Ollama Qwen aliases
    "qwen2.5": "qwen2.5:14b",
    "qwen 2.5": "qwen2.5:14b",
    "qwen2.5 14b": "qwen2.5:14b",
    "qwen2.5 7b": "qwen2.5:7b",
}


def _normalize_model_name(model: str) -> str:
    normalized = model.lower().strip()
    return MODEL_ALIASES.get(normalized, normalized)


def _lookup_pricing(runtime_name: str, model: str) -> Optional[ModelPricing]:
    runtime_key = runtime_name.lower()
    pricing_map = RUNTIME_PRICING_TABLES.get(runtime_key)
    if pricing_map is None:
        return None

    model_key = _normalize_model_name(model)

    if model_key in pricing_map:
        return pricing_map[model_key]

    for prefix in sorted(pricing_map.keys(), key=len, reverse=True):
        if model_key.startswith(prefix):
            return pricing_map[prefix]
    return None


def estimate_cost_usd(runtime_name: str, model: str, usage: TokenUsage) -> float:
    """Return estimated API cost in USD for one completion."""
    runtime_key = runtime_name.lower()
    if runtime_key == "ollama":
        return 0.0

    pricing = _lookup_pricing(runtime_key, model)
    if pricing is None:
        return 0.0

    prompt_tokens = max(usage.prompt_tokens, 0)
    completion_tokens = max(usage.completion_tokens, 0)
    cached_prompt_tokens = max(usage.cached_prompt_tokens, 0)
    cached_prompt_tokens = min(cached_prompt_tokens, prompt_tokens)

    input_rate = pricing.input_per_million
    output_rate = pricing.output_per_million
    cached_input_rate = pricing.cached_input_per_million
    if (
        pricing.long_context_threshold_tokens is not None
        and prompt_tokens > pricing.long_context_threshold_tokens
    ):
        input_rate = pricing.input_per_million_long_context or input_rate
        output_rate = pricing.output_per_million_long_context or output_rate
        cached_input_rate = pricing.cached_input_per_million_long_context or cached_input_rate

    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
    input_cost = uncached_prompt_tokens * input_rate
    if cached_prompt_tokens > 0 and cached_input_rate is not None:
        input_cost += cached_prompt_tokens * cached_input_rate
    else:
        input_cost += cached_prompt_tokens * input_rate

    output_cost = completion_tokens * output_rate

    cache_write_cost = 0.0
    if pricing.cache_write_per_million is not None:
        cache_write_cost = max(usage.cache_creation_input_tokens, 0) * pricing.cache_write_per_million

    cache_read_cost = 0.0
    if pricing.cache_read_per_million is not None:
        cache_read_cost = max(usage.cache_read_input_tokens, 0) * pricing.cache_read_per_million

    total_per_million = input_cost + output_cost + cache_write_cost + cache_read_cost
    return total_per_million / 1_000_000.0
