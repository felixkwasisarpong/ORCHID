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


def _classify_fault(fault: dict) -> tuple[str, str]:
    """Return (fault_type, severity) from a fault_config dict.

    severity is inferred from quantitative thresholds.
    baseline → ('none', 'none')
    """
    if fault.get("permission_path"):
        return ("permission", "active")
    if fault.get("missing_path"):
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
        if timeout >= 10.0:
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
        success, llm_calls, tool_calls, retries, token counts, cost, latency,
        fault_type, fault_severity, fault_latency_ms, fault_timeout_s.
    df_steps : pd.DataFrame
        One row per step. Includes run-level keys plus step_index, tool_name,
        action_type, llm_latency_ms, tool_latency_ms, step_latency_ms,
        step_prompt_tokens, step_completion_tokens, validated, retries.
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

        row = {
            "run_id": data.get("run_id", path.stem),
            "orchestrator": data.get("orchestrator"),
            "runtime": data.get("runtime"),
            "task_id": data.get("task_id"),
            "seed": data.get("seed"),
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
        }
        for step in data.get("steps") or []:
            action = step.get("action") or {}
            tool_call = action.get("tool_call") or {}
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
                "step_error": step.get("error"),
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
