"""Load experiment results from JSONL traces and CSV summaries.

Returns two DataFrames:
- df_summary: one row per run, derived from JSONL traces (includes fault columns)
- df_steps:   one row per step, exploded from JSONL traces
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


_DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "evaluation" / "results"


def _classify_error_text(error: str | None) -> str | None:
    if not error:
        return None
    lowered = error.lower()
    if "api_key" in lowered and "not set" in lowered:
        return "auth_missing"
    if "unauthorized" in lowered or "401" in lowered:
        return "auth_failed"
    if "forbidden" in lowered or "403" in lowered:
        return "permission_denied"
    if "rate limit" in lowered or "429" in lowered or "quota" in lowered:
        return "rate_limited"
    if "timeout" in lowered or "timed out" in lowered or "timeouterror" in lowered:
        return "timeout"
    if "llm failed to produce valid stepaction json" in lowered:
        return "llm_invalid_json"
    if "tool_call missing" in lowered:
        return "tool_call_missing"
    if "tool" in lowered and "not allowed" in lowered:
        return "tool_not_allowed"
    if "permission denied" in lowered or "operation not permitted" in lowered or "read-only" in lowered:
        return "permission_denied"
    if "missing file" in lowered or "no such file" in lowered or "not found" in lowered:
        return "missing_file"
    if "tool call failed" in lowered or "iserror=true" in lowered:
        return "tool_execution_error"
    if "mcp subprocess ended" in lowered or "connection" in lowered:
        return "infrastructure_error"
    return "unknown_error"


def _infer_terminal_and_failure(data: dict) -> tuple[str, str | None]:
    terminal_reason = data.get("terminal_reason")
    failure_mode = data.get("failure_mode")
    if terminal_reason:
        return terminal_reason, failure_mode

    if bool(data.get("success")):
        return "validated", None

    run_error = data.get("error")
    if run_error:
        return "run_exception", _classify_error_text(run_error)

    steps = data.get("steps") or []
    if not steps:
        return "no_steps", "no_steps"
    last = steps[-1] or {}
    if bool(last.get("validated")):
        return "validated", None
    step_error = last.get("error")
    if step_error:
        return "step_error", last.get("error_category") or _classify_error_text(step_error)
    action = last.get("action") or {}
    if action.get("action_type") == "finalize":
        return "finalized_without_validation", "premature_finalize"
    return "ended_unknown", "unknown_error"


def _classify_fault(fault: dict) -> tuple[str, str]:
    """Return (fault_type, severity) from a fault_config dict.

    severity is inferred from quantitative thresholds.
    baseline → ('none', 'none')
    """
    permission_path = fault.get("permission_path")
    if permission_path:
        path = str(permission_path)
        if path == "data/notes.txt":
            return ("permission", "low")
        if path == "output":
            return ("permission", "med")
        if path == "data":
            return ("permission", "high")
        return ("permission", "active")

    missing_path = fault.get("missing_path")
    if missing_path:
        path = str(missing_path)
        if path == "data/notes.txt":
            return ("missing", "low")
        if path == "data/input.txt":
            return ("missing", "med")
        if path == "data":
            return ("missing", "high")
        return ("missing", "active")
    latency = fault.get("latency_ms", 0.0) or 0.0
    if latency > 0:
        if latency <= 100:
            sev = "low"
        elif latency <= 400:
            sev = "med"
        else:
            sev = "high"
        return ("latency", sev)
    timeout = fault.get("tool_timeout_s")
    if timeout is not None and timeout < 20.0:
        if timeout >= 5.0:
            sev = "low"
        elif timeout >= 2.0:
            sev = "med"
        else:
            sev = "high"
        return ("timeout", sev)
    return ("none", "none")


def load_traces(results_dir: Optional[Path] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all JSONL traces from *results_dir/traces/*.

    Returns
    -------
    df_summary : pd.DataFrame
        One row per run. Columns include orchestrator, runtime, task_id, seed,
        started_at, ended_at, success, llm_calls, tool_calls, retries, token counts, cost, latency,
        fault_type, fault_severity, fault_latency_ms, fault_timeout_s,
        terminal_reason, failure_mode.
    df_steps : pd.DataFrame
        One row per step. Includes run-level keys plus step_index, tool_name,
        action_type, llm_latency_ms, tool_latency_ms, step_latency_ms,
        step_prompt_tokens, step_completion_tokens, validated, retries,
        step_error_category.
    """
    results_dir = Path(results_dir) if results_dir else _DEFAULT_RESULTS_DIR
    trace_dir = results_dir / "traces"

    summary_rows: list[dict] = []
    step_rows: list[dict] = []

    for path in sorted(trace_dir.glob("*.jsonl")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        fault = data.get("fault_config") or {}
        fault_type, fault_sev = _classify_fault(fault)
        terminal_reason, failure_mode = _infer_terminal_and_failure(data)

        row = {
            "run_id": data.get("run_id", path.stem),
            "orchestrator": data.get("orchestrator"),
            "runtime": data.get("runtime"),
            "task_id": data.get("task_id"),
            "seed": data.get("seed"),
            "started_at": data.get("started_at"),
            "ended_at": data.get("ended_at"),
            "success": bool(data.get("success")),
            "llm_calls": data.get("llm_calls", 0),
            "tool_calls": data.get("tool_calls", 0),
            "retries": data.get("retries", 0),
            "llm_prompt_tokens": data.get("llm_prompt_tokens", 0),
            "llm_completion_tokens": data.get("llm_completion_tokens", 0),
            "llm_total_tokens": data.get("llm_total_tokens", 0),
            "llm_cost_usd": data.get("llm_cost_usd", 0.0),
            "total_latency_ms": data.get("total_latency_ms", 0.0),
            "fault_type": fault_type,
            "fault_severity": fault_sev,
            "fault_latency_ms": fault.get("latency_ms", 0.0) or 0.0,
            "fault_timeout_s": fault.get("tool_timeout_s"),
            "terminal_reason": terminal_reason,
            "failure_mode": failure_mode,
            "error": data.get("error"),
        }
        summary_rows.append(row)

        run_keys = {
            "run_id": row["run_id"],
            "orchestrator": row["orchestrator"],
            "runtime": row["runtime"],
            "task_id": row["task_id"],
            "seed": row["seed"],
            "fault_type": fault_type,
            "fault_severity": fault_sev,
            "terminal_reason": terminal_reason,
            "failure_mode": failure_mode,
        }
        for step in data.get("steps") or []:
            action = step.get("action") or {}
            tool_call = action.get("tool_call") or {}
            step_error = step.get("error")
            step_error_category = step.get("error_category") or _classify_error_text(step_error)
            step_rows.append({
                **run_keys,
                "step_index": step.get("step_index", 0),
                "action_type": action.get("action_type"),
                "tool_name": tool_call.get("name"),
                "validated": step.get("validated", False),
                "validation_error": step.get("validation_error"),
                "llm_latency_ms": step.get("llm_latency_ms", 0.0),
                "tool_latency_ms": step.get("tool_latency_ms") or 0.0,
                "step_latency_ms": step.get("step_latency_ms", 0.0),
                "step_prompt_tokens": step.get("llm_prompt_tokens", 0),
                "step_completion_tokens": step.get("llm_completion_tokens", 0),
                "step_retries": step.get("retries", 0),
                "step_error": step_error,
                "step_error_category": step_error_category,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_steps = pd.DataFrame(step_rows)
    return df_summary, df_steps


def load_csv_summaries(results_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all summary CSV files and concatenate them.

    NOTE: CSVs do not carry fault_config columns. Use load_traces() for full
    fault-aware analysis.
    """
    results_dir = Path(results_dir) if results_dir else _DEFAULT_RESULTS_DIR
    frames = [pd.read_csv(p) for p in sorted(results_dir.glob("summary_*.csv"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["run_id"])
