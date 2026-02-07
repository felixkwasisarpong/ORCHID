"""Filesystem benchmark task definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from observability.trace_schema import TaskSpec


ALLOWED_FS_TOOLS = [
    "read_file",
    "write_file",
    "list_directory",
    "create_directory",
    "delete_file",
    "move_file",
    "stat",
    "search",
]


@dataclass
class TaskDefinition:
    spec: TaskSpec
    setup: Callable[[Path], None]
    validate: Callable[[Path], Tuple[bool, str]]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, indent=2))


def _write_csv(path: Path, rows: List[List[str]]) -> None:
    content = "\n".join([",".join(row) for row in rows]) + "\n"
    _write_text(path, content)


def _validate_text(path: Path, expected: str) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"Missing file: {path}"
    actual = path.read_text(encoding="utf-8").strip()
    if actual != expected:
        return False, f"Expected '{expected}' but got '{actual}'"
    return True, ""


def _validate_contains(path: Path, expected: str) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"Missing file: {path}"
    actual = path.read_text(encoding="utf-8")
    if expected not in actual:
        return False, f"Expected to contain '{expected}'"
    return True, ""


def task_01_count_lines() -> TaskDefinition:
    spec = TaskSpec(
        id="task_01_count_lines",
        name="Count Lines",
        description=(
            "Read data/input.txt and write the number of lines as an integer to output/line_count.txt."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/input.txt", "alpha\nbeta\ngamma")

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_text(root / "output/line_count.txt", "3")

    return TaskDefinition(spec, setup, validate)


def task_02_extract_json() -> TaskDefinition:
    spec = TaskSpec(
        id="task_02_extract_json",
        name="Extract JSON Value",
        description=(
            "Read data/config.json and write the value of key 'threshold' to output/threshold.txt."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_json(root / "data/config.json", {"mode": "fast", "threshold": 7})

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_text(root / "output/threshold.txt", "7")

    return TaskDefinition(spec, setup, validate)


def task_03_merge_files() -> TaskDefinition:
    spec = TaskSpec(
        id="task_03_merge_files",
        name="Merge Files",
        description=(
            "Read data/a.txt and data/b.txt, then write output/merged.txt with a.txt lines followed by b.txt lines."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/a.txt", "alpha\nbeta\n")
        _write_text(root / "data/b.txt", "gamma\ndelta\n")

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_text(root / "output/merged.txt", "alpha\nbeta\ngamma\ndelta")

    return TaskDefinition(spec, setup, validate)


def task_04_rename_move() -> TaskDefinition:
    spec = TaskSpec(
        id="task_04_rename_move",
        name="Rename and Move",
        description=(
            "Move data/report.txt to archive/2026/report.txt. Create directories as needed."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/report.txt", "quarterly results")

    def validate(root: Path) -> Tuple[bool, str]:
        path = root / "archive/2026/report.txt"
        return _validate_text(path, "quarterly results")

    return TaskDefinition(spec, setup, validate)


def task_05_list_index() -> TaskDefinition:
    spec = TaskSpec(
        id="task_05_list_index",
        name="Directory Index",
        description=(
            "List files in data/files and write sorted names to output/index.txt, one per line."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/files/one.txt", "1")
        _write_text(root / "data/files/two.txt", "2")
        _write_text(root / "data/files/three.txt", "3")

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_text(root / "output/index.txt", "one.txt\nthree.txt\ntwo.txt")

    return TaskDefinition(spec, setup, validate)


def task_06_replace_text() -> TaskDefinition:
    spec = TaskSpec(
        id="task_06_replace_text",
        name="Replace Text",
        description=(
            "Replace 'foo' with 'baz' in data/notes.txt and write result to output/notes.txt."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/notes.txt", "foo bar foo")

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_text(root / "output/notes.txt", "baz bar baz")

    return TaskDefinition(spec, setup, validate)


def task_07_delete_temp() -> TaskDefinition:
    spec = TaskSpec(
        id="task_07_delete_temp",
        name="Delete Temp Files",
        description=(
            "Delete all .tmp files under data/tmp. Keep non-.tmp files."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/tmp/a.tmp", "temp")
        _write_text(root / "data/tmp/b.tmp", "temp")
        _write_text(root / "data/tmp/keep.txt", "keep")

    def validate(root: Path) -> Tuple[bool, str]:
        if (root / "data/tmp/a.tmp").exists() or (root / "data/tmp/b.tmp").exists():
            return False, "Temp files still present"
        if not (root / "data/tmp/keep.txt").exists():
            return False, "keep.txt missing"
        return True, ""

    return TaskDefinition(spec, setup, validate)


def task_08_sum_csv() -> TaskDefinition:
    spec = TaskSpec(
        id="task_08_sum_csv",
        name="Sum CSV",
        description=(
            "Read data/metrics.csv (header: value). Sum the values and write output/total.txt."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_csv(root / "data/metrics.csv", [["value"], ["1"], ["2"], ["3"]])

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_text(root / "output/total.txt", "6")

    return TaskDefinition(spec, setup, validate)


def task_09_copy_if_contains() -> TaskDefinition:
    spec = TaskSpec(
        id="task_09_copy_if_contains",
        name="Copy If Contains",
        description=(
            "If data/msg.txt contains the word ALERT, copy it to output/alert.txt."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/msg.txt", "status: ALERT\n")

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_contains(root / "output/alert.txt", "ALERT")

    return TaskDefinition(spec, setup, validate)


def task_10_append_log() -> TaskDefinition:
    spec = TaskSpec(
        id="task_10_append_log",
        name="Append Log",
        description=(
            "Append the line 'finish' to logs/run.log, preserving existing content."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "logs/run.log", "start\n")

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_text(root / "logs/run.log", "start\nfinish")

    return TaskDefinition(spec, setup, validate)


def task_11_create_manifest() -> TaskDefinition:
    spec = TaskSpec(
        id="task_11_create_manifest",
        name="Create Manifest",
        description=(
            "List all files under data/pkg and write sorted relative paths to output/manifest.txt."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/pkg/a.txt", "a")
        _write_text(root / "data/pkg/b.txt", "b")
        _write_text(root / "data/pkg/sub/c.txt", "c")

    def validate(root: Path) -> Tuple[bool, str]:
        expected = "a.txt\nb.txt\nsub/c.txt"
        return _validate_text(root / "output/manifest.txt", expected)

    return TaskDefinition(spec, setup, validate)


def task_12_normalize_whitespace() -> TaskDefinition:
    spec = TaskSpec(
        id="task_12_normalize_whitespace",
        name="Normalize Whitespace",
        description=(
            "Replace multiple spaces with single spaces in data/raw.txt and write output/normalized.txt."
        ),
        allowed_tools=ALLOWED_FS_TOOLS,
    )

    def setup(root: Path) -> None:
        _write_text(root / "data/raw.txt", "alpha   beta    gamma")

    def validate(root: Path) -> Tuple[bool, str]:
        return _validate_text(root / "output/normalized.txt", "alpha beta gamma")

    return TaskDefinition(spec, setup, validate)


TASKS: Dict[str, TaskDefinition] = {
    task.spec.id: task
    for task in [
        task_01_count_lines(),
        task_02_extract_json(),
        task_03_merge_files(),
        task_04_rename_move(),
        task_05_list_index(),
        task_06_replace_text(),
        task_07_delete_temp(),
        task_08_sum_csv(),
        task_09_copy_if_contains(),
        task_10_append_log(),
        task_11_create_manifest(),
        task_12_normalize_whitespace(),
    ]
}


def list_task_specs() -> List[TaskSpec]:
    return [task.spec for task in TASKS.values()]


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]
