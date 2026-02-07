"""Shared episode helpers for orchestrators."""
from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter

from observability.trace_schema import StepAction, StepResult
from tools.mcp_gateway_client import ToolSpec


SYSTEM_PROMPT = (
    "You are a tool-using agent for filesystem tasks. "
    "You MUST respond with a single JSON object that matches the StepAction schema. "
    "Do not include any extra text, markdown, or code fences."
)

STEP_ACTION_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "type": {"const": "tool_call"},
                "tool_call": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
            "required": ["type", "tool_call"],
        },
        {
            "type": "object",
            "properties": {
                "type": {"const": "finalize"},
                "answer": {"type": "string"},
            },
            "required": ["type", "answer"],
        },
    ]
}


def format_tools(tools: list[ToolSpec]) -> str:
    payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]
    return json.dumps(payload, ensure_ascii=True, indent=2)


def format_history(steps: list[StepResult]) -> str:
    if not steps:
        return "(none)"
    items: list[str] = []
    for step in steps:
        summary = {
            "step_index": step.step_index,
            "action": step.action.model_dump(),
            "errors": step.errors,
        }
        if step.tool_result:
            summary["tool_result"] = step.tool_result
        if step.validation:
            summary["validation"] = step.validation
        items.append(json.dumps(summary, ensure_ascii=True))
    return "\n".join(items)


def build_messages(
    task_description: str,
    sandbox_path: str,
    tools: list[ToolSpec],
    steps: list[StepResult],
    max_steps: int,
) -> list[dict[str, str]]:
    tool_block = format_tools(tools)
    history_block = format_history(steps)
    prompt = (
        f"Task: {task_description}\n"
        f"Sandbox root: {sandbox_path}\n"
        f"Step: {len(steps) + 1} of {max_steps}\n"
        "Available tools (use only these):\n"
        f"{tool_block}\n"
        "Previous steps:\n"
        f"{history_block}\n"
        "Return ONLY a JSON object matching this schema:\n"
        f"{json.dumps(STEP_ACTION_SCHEMA, ensure_ascii=True)}"
    )
    return [{"role": "user", "content": prompt}]


def extract_json(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("```"):
        trimmed = trimmed.strip("`\n")
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return trimmed
    return trimmed[start : end + 1]


def parse_step_action(text: str) -> StepAction:
    adapter = TypeAdapter(StepAction)
    payload = extract_json(text)
    return adapter.validate_json(payload)


def build_repair_messages(
    original_messages: list[dict[str, str]],
    last_output: str,
    error: str,
) -> list[dict[str, str]]:
    repair_prompt = (
        "Your previous response was invalid. "
        f"Error: {error}. "
        "Return ONLY valid JSON that matches the StepAction schema."
    )
    return [
        *original_messages,
        {"role": "assistant", "content": last_output},
        {"role": "user", "content": repair_prompt},
    ]
