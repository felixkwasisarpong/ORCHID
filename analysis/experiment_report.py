"""Generate paper-ready experiment details and reproducibility metadata.

This script writes:
- experiment_details.json
- experiment_details.md

It combines planned design cardinality from matrix configs with observed run artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.load_results import load_traces
from benchmarks.tasks import list_task_specs
from harness.config import load_config


@dataclass
class ConditionRow:
    config: str
    fault_type: str
    fault_severity: str
    orchestrators: int
    runtimes: int
    tasks: int
    seeds: int
    runs_planned: int


def _run_cmd(cmd: list[str], cwd: Path | None = None, timeout_s: float = 5.0) -> str | None:
    try:
        out = subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _classify_fault(permission_path: str | None, missing_path: str | None, latency_ms: float, timeout_s: float | None) -> tuple[str, str]:
    if permission_path:
        if permission_path == "data/notes.txt":
            return "permission", "low"
        if permission_path == "output":
            return "permission", "med"
        if permission_path == "data":
            return "permission", "high"
        return "permission", "active"
    if missing_path:
        if missing_path == "data/notes.txt":
            return "missing", "low"
        if missing_path == "data/input.txt":
            return "missing", "med"
        if missing_path == "data":
            return "missing", "high"
        return "missing", "active"
    if latency_ms > 0:
        if latency_ms <= 100:
            return "latency", "low"
        if latency_ms <= 400:
            return "latency", "med"
        return "latency", "high"
    if timeout_s is not None and timeout_s < 20.0:
        if timeout_s >= 5.0:
            return "timeout", "low"
        if timeout_s >= 2.0:
            return "timeout", "med"
        return "timeout", "high"
    return "none", "none"


def _phys_mem_gib() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages <= 0 or page_size <= 0:
            return None
        return round((pages * page_size) / (1024 ** 3), 2)
    except Exception:
        return None


def _package_versions() -> dict[str, str | None]:
    pkgs = [
        "pydantic",
        "httpx",
        "pyyaml",
        "langgraph",
        "crewai",
        "pyautogen",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
    ]
    versions: dict[str, str | None] = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = None
    return versions


def _condition_rows(matrix_dir: Path) -> tuple[list[ConditionRow], dict[str, Any]]:
    all_tasks_n = len(list_task_specs())
    rows: list[ConditionRow] = []

    for cfg_path in sorted(matrix_dir.glob("*.yaml")):
        cfg = load_config(cfg_path)
        tasks_n = all_tasks_n if (not cfg.tasks or "all" in cfg.tasks) else len(cfg.tasks)
        orch_n = len(cfg.orchestrators)
        rt_n = len(cfg.runtimes)
        seed_n = len(cfg.seeds)
        fault_type, fault_sev = _classify_fault(
            cfg.faults.permission_path,
            cfg.faults.missing_path,
            float(cfg.faults.latency_ms or 0.0),
            cfg.faults.tool_timeout_s,
        )
        rows.append(
            ConditionRow(
                config=str(cfg_path.as_posix()),
                fault_type=fault_type,
                fault_severity=fault_sev,
                orchestrators=orch_n,
                runtimes=rt_n,
                tasks=tasks_n,
                seeds=seed_n,
                runs_planned=orch_n * rt_n * tasks_n * seed_n,
            )
        )

    if not rows:
        return rows, {
            "conditions": 0,
            "planned_total_runs": 0,
            "orchestrators": 0,
            "runtimes": 0,
            "tasks": all_tasks_n,
            "seeds": 0,
            "per_cell_n": 0,
        }

    sample = rows[0]
    summary = {
        "conditions": len(rows),
        "planned_total_runs": int(sum(r.runs_planned for r in rows)),
        "orchestrators": sample.orchestrators,
        "runtimes": sample.runtimes,
        "tasks": sample.tasks,
        "seeds": sample.seeds,
        "per_cell_n": sample.seeds,
    }
    return rows, summary


def _rerun_stats(df_summary: pd.DataFrame) -> dict[str, Any]:
    if df_summary.empty:
        return {
            "duplicate_groups": 0,
            "extra_runs_from_duplicates": 0,
            "max_repeats_in_group": 0,
        }

    group_cols = [
        "orchestrator",
        "runtime",
        "task_id",
        "seed",
        "fault_type",
        "fault_severity",
    ]
    counts = df_summary.groupby(group_cols).size().rename("n").reset_index()
    dup = counts[counts["n"] > 1]
    return {
        "duplicate_groups": int(len(dup)),
        "extra_runs_from_duplicates": int((dup["n"] - 1).sum()) if not dup.empty else 0,
        "max_repeats_in_group": int(dup["n"].max()) if not dup.empty else 0,
    }


def _date_window(df_summary: pd.DataFrame) -> tuple[str | None, str | None]:
    if df_summary.empty or "started_at" not in df_summary.columns:
        return None, None
    started = pd.to_datetime(df_summary["started_at"], errors="coerce", utc=True)
    ended = pd.to_datetime(df_summary.get("ended_at"), errors="coerce", utc=True)
    min_start = started.min()
    max_end = ended.max()
    min_s = min_start.isoformat() if pd.notna(min_start) else None
    max_s = max_end.isoformat() if pd.notna(max_end) else None
    return min_s, max_s


def _to_markdown(report: dict[str, Any]) -> str:
    card = report["cardinality"]
    repro = report["reproducibility"]
    seed = report["seed_protocol"]

    return f"""# Experimental Details Report

Generated at: {report['generated_at_utc']}

## Exact Run Cardinality

- Conditions: {card['conditions']}
- Orchestrators per condition: {card['orchestrators']}
- Runtimes per condition: {card['runtimes']}
- Tasks per condition: {card['tasks']}
- Seeds per cell (`n`): {card['per_cell_n']}
- Planned total runs: {card['planned_total_runs']}
- Formula: `{card['conditions']} x {card['orchestrators']} x {card['runtimes']} x {card['tasks']} x {card['seeds']} = {card['planned_total_runs']}`
- Observed runs in traces: {card['observed_runs']}

## Reproducibility Metadata

- Git commit: {repro.get('git_commit')}
- Git dirty: {repro.get('git_dirty')}
- Run date window (UTC): {repro.get('run_window_start_utc')} to {repro.get('run_window_end_utc')}
- Host platform: {repro.get('platform')}
- Machine / processor: {repro.get('machine')} / {repro.get('processor')}
- CPU count: {repro.get('cpu_count')}
- Physical memory (GiB): {repro.get('physical_memory_gib')}
- Python: {repro.get('python_version')}
- Docker: {repro.get('docker_version')}
- Docker Compose: {repro.get('docker_compose_version')}
- mcp/filesystem image id: {repro.get('mcp_filesystem_image_id')}

## Seed Protocol

- Seed set: {seed['seed_values']}
- Assignment: fixed, explicit list from config files
- Run ordering: deterministic nested loop (`orchestrator -> runtime -> task -> seed`) within each condition file
- Condition ordering: deterministic list from `scripts/run_matrix.sh`
- Randomized ordering: no
- Rerun diagnostics: duplicate groups={seed['rerun_duplicate_groups']}, extra runs from duplicates={seed['rerun_extra_runs']}, max repeats per group={seed['rerun_max_repeats']}

## Statistical Plan

- Confidence intervals: 95% bootstrap CIs for group-level success rate, mean latency, and mean retries
- Hypothesis tests: pairwise orchestrator comparisons (within runtime/fault cell) using permutation tests on success, latency, and retries
- Multiple-comparison correction: Benjamini-Hochberg FDR correction per metric family
- Script: `python -m analysis.stats_plan --results-dir evaluation/results --out-dir evaluation/results/stats`

## Data Handling Rules

- Timeout/error handling: timeout and execution failures are retained as unsuccessful runs with structured terminal metadata (`terminal_reason`, `failure_mode`)
- Exclusion criteria: none by default; no run is dropped solely due to failure
- Incomplete runs: included as failures (`success=false`) and categorized by terminal outcome
- Duplicate runs: retained; duplicates are reported in seed protocol diagnostics
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment details + reproducibility metadata")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory containing trace artifacts",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=Path("configs/matrix"),
        help="Directory containing matrix YAML configs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation/results/report"),
        help="Output directory for experiment_details.{json,md}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_rows, card_summary = _condition_rows(args.matrix_dir)
    df_summary, _ = load_traces(args.results_dir)
    observed_runs = int(len(df_summary))

    run_start, run_end = _date_window(df_summary)
    reruns = _rerun_stats(df_summary)

    git_commit = _run_cmd(["git", "rev-parse", "HEAD"])
    git_status = _run_cmd(["git", "status", "--porcelain"])
    docker_version = _run_cmd(["docker", "--version"])
    docker_compose_version = _run_cmd(["docker", "compose", "version"])
    mcp_img_id = _run_cmd(["docker", "image", "inspect", "mcp/filesystem", "--format", "{{.Id}}"])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cardinality": {
            **card_summary,
            "observed_runs": observed_runs,
            "conditions_rows": [asdict(r) for r in cond_rows],
        },
        "reproducibility": {
            "git_commit": git_commit,
            "git_dirty": bool(git_status.strip()) if git_status is not None else None,
            "run_window_start_utc": run_start,
            "run_window_end_utc": run_end,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "physical_memory_gib": _phys_mem_gib(),
            "python_version": platform.python_version(),
            "docker_version": docker_version,
            "docker_compose_version": docker_compose_version,
            "mcp_filesystem_image_id": mcp_img_id,
            "package_versions": _package_versions(),
        },
        "seed_protocol": {
            "seed_values": sorted(df_summary["seed"].dropna().unique().tolist()) if not df_summary.empty else [],
            "ordering_randomized": False,
            "ordering_statement": "orchestrator -> runtime -> task -> seed",
            "rerun_duplicate_groups": reruns["duplicate_groups"],
            "rerun_extra_runs": reruns["extra_runs_from_duplicates"],
            "rerun_max_repeats": reruns["max_repeats_in_group"],
        },
        "data_handling": {
            "timeouts_and_errors_included": True,
            "exclusion_criteria": "none",
            "incomplete_runs_treated_as_failures": True,
            "duplicates_retained": True,
        },
    }

    json_path = out_dir / "experiment_details.json"
    md_path = out_dir / "experiment_details.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
