"""JSONL logger for run traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from observability.trace_schema import RunTrace


@dataclass
class JSONLLogger:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_trace(self, trace: RunTrace) -> None:
        payload = trace.model_dump()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def default_trace_path(results_dir: Path, run_id: str) -> Path:
    traces_dir = results_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    return traces_dir / f"{run_id}.jsonl"
