"""Shared orchestrator types."""
from __future__ import annotations

from pydantic import BaseModel


class EpisodeConfig(BaseModel):
    max_steps: int = 6
    max_llm_retries: int = 2
    max_tool_retries: int = 1
    tool_timeout_s: float = 10.0
    seed: int | None = None
