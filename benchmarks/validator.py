"""Validation utilities for benchmark tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from benchmarks.tasks import get_task


def validate_task(task_id: str, sandbox_root: Path) -> Tuple[bool, str]:
    task = get_task(task_id)
    return task.validate(sandbox_root)
