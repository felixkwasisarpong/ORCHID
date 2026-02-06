"""Metric calculations for experiment runs."""
from __future__ import annotations

from typing import Any, Iterable

from orchestrator.state import TraceEvent


def count_retries(trace: Iterable[TraceEvent]) -> int:
    attempts_by_node: dict[str, int] = {}
    for event in trace:
        attempts_by_node[event.node] = max(event.attempt, attempts_by_node.get(event.node, 0))
    return sum(max(0, attempts - 1) for attempts in attempts_by_node.values())


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    total = len(results)
    success = sum(1 for result in results if result["success"])
    avg_latency = sum(result["latency_ms"] for result in results) / total
    retries = sum(result.get("retries", 0) for result in results)
    tool_calls = sum(result.get("tool_calls", 0) for result in results)
    token_usage = sum(result.get("token_usage", 0) for result in results)
    return {
        "runs": total,
        "success_rate": success / total,
        "avg_latency_ms": avg_latency,
        "total_retries": retries,
        "total_tool_calls": tool_calls,
        "total_token_usage": token_usage,
    }
