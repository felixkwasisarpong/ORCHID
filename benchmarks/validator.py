"""Task validation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.tasks import BenchmarkTask


def validate_task(task: BenchmarkTask, sandbox: Path) -> dict[str, Any]:
    return task.validate(sandbox)
