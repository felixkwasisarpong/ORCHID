"""Runtime client interfaces and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class RuntimeConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 20.0


class RuntimeClient(Protocol):
    config: RuntimeConfig

    async def chat(self, messages: List[Dict[str, Any]], seed: int | None = None) -> str:
        ...
