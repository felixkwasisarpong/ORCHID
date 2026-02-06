"""Retry and failure policies for node execution."""
from __future__ import annotations

import random
from enum import Enum

from pydantic import BaseModel, Field


class FailureAction(str, Enum):
    HALT = "halt"
    CONTINUE = "continue"


class FailurePolicy(BaseModel):
    """What to do when a node ultimately fails."""

    on_failure: FailureAction = FailureAction.HALT


class RetryPolicy(BaseModel):
    """Retry policy with exponential backoff and jitter."""

    max_attempts: int = 1
    backoff_base_s: float = 0.5
    backoff_max_s: float = 10.0
    backoff_jitter_s: float = 0.1
    retry_on: list[str] = Field(default_factory=lambda: ["Exception"])

    def should_retry(self, exc: Exception) -> bool:
        if self.max_attempts <= 1:
            return False
        exc_name = exc.__class__.__name__
        return "Exception" in self.retry_on or exc_name in self.retry_on

    def backoff_s(self, attempt: int) -> float:
        exp = min(self.backoff_base_s * (2 ** (attempt - 1)), self.backoff_max_s)
        jitter = random.uniform(0, self.backoff_jitter_s)
        return exp + jitter
