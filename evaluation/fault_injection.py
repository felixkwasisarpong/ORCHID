"""Fault injection helpers for evaluation."""
from __future__ import annotations

from typing import Any

from tools.mcp_client import FaultConfig, FaultMode


def fault_from_dict(data: dict[str, Any] | None) -> FaultConfig | None:
    if not data:
        return None
    mode = FaultMode(data.get("mode", FaultMode.SUCCESS))
    return FaultConfig(
        mode=mode,
        delay_ms=int(data.get("delay_ms", 0)),
        random_weights=data.get("random_weights", {}),
    )
