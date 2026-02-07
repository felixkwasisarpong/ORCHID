"""Filesystem benchmark task definitions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from observability.trace_schema import TaskSpec


@dataclass
class BenchmarkTask:
    spec: TaskSpec
    setup: Callable[[Path, int | None], None]
    validate: Callable[[Path], dict[str, Any]]
    allowed_tools: list[str]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _success(**details: Any) -> dict[str, Any]:
    return {"success": True, **details}


def _failure(message: str, **details: Any) -> dict[str, Any]:
    return {"success": False, "message": message, **details}


def task_create_hello() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="create_hello",
        description="Create hello.txt in the sandbox root with content 'hello world'.",
        max_steps=4,
        allowed_tools=["write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        base.mkdir(parents=True, exist_ok=True)

    def validate(base: Path) -> dict[str, Any]:
        path = base / "hello.txt"
        if not path.exists():
            return _failure("hello.txt missing")
        if _read(path).strip() != "hello world":
            return _failure("hello.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_sum_numbers() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="sum_numbers",
        description="Read numbers.txt in the sandbox root, sum the integers, and write the sum to sum.txt.",
        max_steps=6,
        allowed_tools=["read_file", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "numbers.txt", "3\n7\n10\n")

    def validate(base: Path) -> dict[str, Any]:
        path = base / "sum.txt"
        if not path.exists():
            return _failure("sum.txt missing")
        if _read(path).strip() != "20":
            return _failure("sum.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_sort_names() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="sort_names",
        description="Sort the lines in names.txt alphabetically and write to sorted.txt.",
        max_steps=6,
        allowed_tools=["read_file", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "names.txt", "zoe\nalex\nmae\n")

    def validate(base: Path) -> dict[str, Any]:
        expected = "alex\nmae\nzoe"
        path = base / "sorted.txt"
        if not path.exists():
            return _failure("sorted.txt missing")
        if _read(path).strip() != expected:
            return _failure("sorted.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_json_toggle() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="json_toggle",
        description="Update config.json so enabled is true while keeping other fields unchanged.",
        max_steps=6,
        allowed_tools=["read_file", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "config.json", '{"enabled": false, "threshold": 3}')

    def validate(base: Path) -> dict[str, Any]:
        path = base / "config.json"
        if not path.exists():
            return _failure("config.json missing")
        text = _read(path).strip()
        if "\"enabled\": true" not in text:
            return _failure("enabled not true", content=text)
        if "\"threshold\": 3" not in text:
            return _failure("threshold changed", content=text)
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_merge_files() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="merge_files",
        description="Combine a.txt and b.txt into combined.txt with a.txt content first, then b.txt.",
        max_steps=6,
        allowed_tools=["read_file", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "a.txt", "alpha")
        _write(base / "b.txt", "beta")

    def validate(base: Path) -> dict[str, Any]:
        path = base / "combined.txt"
        if not path.exists():
            return _failure("combined.txt missing")
        if _read(path).strip() != "alpha\nbeta":
            return _failure("combined.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_rename_file() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="rename_file",
        description="Rename draft.txt to final.txt in the sandbox root.",
        max_steps=4,
        allowed_tools=["move_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "draft.txt", "v1")

    def validate(base: Path) -> dict[str, Any]:
        if (base / "draft.txt").exists():
            return _failure("draft.txt still exists")
        path = base / "final.txt"
        if not path.exists():
            return _failure("final.txt missing")
        if _read(path).strip() != "v1":
            return _failure("final.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_delete_temp() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="delete_temp",
        description="Delete temp.log from the sandbox root.",
        max_steps=4,
        allowed_tools=["delete_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "temp.log", "discard")

    def validate(base: Path) -> dict[str, Any]:
        if (base / "temp.log").exists():
            return _failure("temp.log still exists")
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_count_lines() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="count_lines",
        description="Count the number of lines in notes.txt and write the count to line_count.txt.",
        max_steps=6,
        allowed_tools=["read_file", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "notes.txt", "a\nb\nc\n")

    def validate(base: Path) -> dict[str, Any]:
        path = base / "line_count.txt"
        if not path.exists():
            return _failure("line_count.txt missing")
        if _read(path).strip() != "3":
            return _failure("line_count.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_list_files() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="list_files",
        description="List files in the sandbox root in alphabetical order and write to index.txt (one per line).",
        max_steps=6,
        allowed_tools=["list_directory", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "alpha.txt", "a")
        _write(base / "beta.txt", "b")
        _write(base / "gamma.txt", "c")

    def validate(base: Path) -> dict[str, Any]:
        path = base / "index.txt"
        if not path.exists():
            return _failure("index.txt missing")
        expected = "alpha.txt\nbeta.txt\ngamma.txt"
        if _read(path).strip() != expected:
            return _failure("index.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_create_dir_and_file() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="create_dir_and_file",
        description="Create directory data and write data/info.txt with content 'ok'.",
        max_steps=6,
        allowed_tools=["create_directory", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        base.mkdir(parents=True, exist_ok=True)

    def validate(base: Path) -> dict[str, Any]:
        path = base / "data" / "info.txt"
        if not path.exists():
            return _failure("data/info.txt missing")
        if _read(path).strip() != "ok":
            return _failure("info.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_replace_word() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="replace_word",
        description="Replace the word 'red' with 'blue' in story.txt.",
        max_steps=6,
        allowed_tools=["read_file", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "story.txt", "the red fox")

    def validate(base: Path) -> dict[str, Any]:
        path = base / "story.txt"
        if not path.exists():
            return _failure("story.txt missing")
        if _read(path).strip() != "the blue fox":
            return _failure("story.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def task_extract_json_field() -> BenchmarkTask:
    spec = TaskSpec(
        task_id="extract_json_field",
        description="Read record.json and write the value field to value.txt.",
        max_steps=6,
        allowed_tools=["read_file", "write_file"],
    )

    def setup(base: Path, seed: int | None) -> None:
        _write(base / "record.json", '{"id": "abc", "value": 42}')

    def validate(base: Path) -> dict[str, Any]:
        path = base / "value.txt"
        if not path.exists():
            return _failure("value.txt missing")
        if _read(path).strip() != "42":
            return _failure("value.txt content mismatch", content=_read(path))
        return _success()

    return BenchmarkTask(spec=spec, setup=setup, validate=validate, allowed_tools=spec.allowed_tools)


def all_tasks() -> list[BenchmarkTask]:
    return [
        task_create_hello(),
        task_sum_numbers(),
        task_sort_names(),
        task_json_toggle(),
        task_merge_files(),
        task_rename_file(),
        task_delete_temp(),
        task_count_lines(),
        task_list_files(),
        task_create_dir_and_file(),
        task_replace_word(),
        task_extract_json_field(),
    ]
