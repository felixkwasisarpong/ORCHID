"""Token pricing helpers for cloud runtimes.

Reference pricing sources checked on 2026-02-09:
- OpenAI API pricing: https://openai.com/api/pricing/
- Anthropic pricing: https://www.anthropic.com/pricing
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


OPENAI_PRICING: Dict[str, ModelPricing] = {
    "gpt-4.1": ModelPricing(input_per_million=2.0, output_per_million=8.0, cached_input_per_million=0.5),
    "gpt-4.1-mini": ModelPricing(
        input_per_million=0.4, output_per_million=1.6, cached_input_per_million=0.1
    ),
    "gpt-4.1-nano": ModelPricing(
        input_per_million=0.1, output_per_million=0.4, cached_input_per_million=0.025
    ),
}

ANTHROPIC_PRICING: Dict[str, ModelPricing] = {
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


def _lookup_pricing(runtime_name: str, model: str) -> Optional[ModelPricing]:
    model_key = model.lower()
    pricing_map = OPENAI_PRICING if runtime_name == "openai" else ANTHROPIC_PRICING

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

    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
    input_cost = uncached_prompt_tokens * pricing.input_per_million
    if cached_prompt_tokens > 0 and pricing.cached_input_per_million is not None:
        input_cost += cached_prompt_tokens * pricing.cached_input_per_million
    else:
        input_cost += cached_prompt_tokens * pricing.input_per_million

    output_cost = completion_tokens * pricing.output_per_million

    cache_write_cost = 0.0
    if pricing.cache_write_per_million is not None:
        cache_write_cost = max(usage.cache_creation_input_tokens, 0) * pricing.cache_write_per_million

    cache_read_cost = 0.0
    if pricing.cache_read_per_million is not None:
        cache_read_cost = max(usage.cache_read_input_tokens, 0) * pricing.cache_read_per_million

    total_per_million = input_cost + output_cost + cache_write_cost + cache_read_cost
    return total_per_million / 1_000_000.0
