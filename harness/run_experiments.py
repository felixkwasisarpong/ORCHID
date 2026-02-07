"""Experiment harness for orchestrator benchmarks."""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from benchmarks.tasks import BenchmarkTask, all_tasks
from observability.logger import JsonlWriter
from observability.trace_schema import RunTrace
from orchestrators.crewai_engine import CrewAIEngine
from orchestrators.langgraph_engine import LangGraphEngine
from orchestrators.temporal_engine import (
    FaultSettingsModel,
    MCPSettings,
    TemporalEngine,
    TemporalSettings,
)
from orchestrators.types import EpisodeConfig
from runtimes.base import RuntimeConfig, RuntimeName
from tools.mcp_gateway_client import (
    FaultSettings,
    MCPGatewayClient,
    MCPGatewayTransport,
    stdio_gateway_from_env,
)


class FaultConfig(BaseModel):
    permission_path: str | None = None
    missing_path: str | None = None
    latency_ms: int = 0
    jitter_ms: int = 0
    timeout_s: float | None = None


class MCPConfig(BaseModel):
    transport: MCPGatewayTransport = MCPGatewayTransport.STDIO
    base_url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    reuse_session: bool = True


class ExperimentConfig(BaseModel):
    orchestrator: str = "langgraph"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    runs: int = 1
    seeds: list[int] | None = None
    tasks: list[str] | None = None
    sandbox_root: str = "evaluation/sandboxes"
    output_dir: str = "evaluation/results"
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    faults: FaultConfig = Field(default_factory=FaultConfig)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(data)


def apply_permission_fault(base: Path, relative_path: str | None) -> None:
    target = base / relative_path if relative_path else base
    if not target.exists():
        return
    if target.is_dir():
        os.chmod(target, 0o555)
    else:
        os.chmod(target, 0o444)


def apply_missing_fault(base: Path, relative_path: str | None) -> None:
    if not relative_path:
        return
    target = base / relative_path
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def build_mcp_client(config: MCPConfig, faults: FaultConfig) -> MCPGatewayClient:
    transport = config.transport
    if transport == MCPGatewayTransport.STDIO:
        command = config.command
        args = config.args
        if not command and not args:
            command, args, env = stdio_gateway_from_env()
        else:
            env = None
        return MCPGatewayClient(
            transport=transport,
            command=command,
            args=args,
            env=env,
            reuse_session=config.reuse_session,
            faults=FaultSettings(
                latency_ms=faults.latency_ms,
                jitter_ms=faults.jitter_ms,
                timeout_s=faults.timeout_s,
            ),
        )

    return MCPGatewayClient(
        transport=transport,
        base_url=config.base_url,
        reuse_session=config.reuse_session,
        faults=FaultSettings(
            latency_ms=faults.latency_ms,
            jitter_ms=faults.jitter_ms,
            timeout_s=faults.timeout_s,
        ),
    )


def resolve_tasks(task_ids: list[str] | None) -> list[BenchmarkTask]:
    tasks = {task.spec.task_id: task for task in all_tasks()}
    if not task_ids:
        return list(tasks.values())
    missing = [task_id for task_id in task_ids if task_id not in tasks]
    if missing:
        raise ValueError(f"Unknown tasks: {missing}")
    return [tasks[task_id] for task_id in task_ids]


def export_csv(traces: list[RunTrace], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        rows.append(
            {
                "run_id": trace.run_id,
                "orchestrator": trace.orchestrator,
                "runtime": trace.runtime,
                "model": trace.model,
                "task_id": trace.task_id,
                "seed": trace.seed,
                "success": trace.success,
                "llm_calls": trace.counters.llm_calls,
                "tool_calls": trace.counters.tool_calls,
                "retries": trace.counters.retries,
                "total_latency_ms": trace.counters.total_latency_ms,
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


async def run_experiments(config: ExperimentConfig) -> list[RunTrace]:
    tasks = resolve_tasks(config.tasks)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(output_dir / "traces.jsonl")
    traces: list[RunTrace] = []

    seeds = config.seeds or list(range(config.runs))
    if config.seeds and config.runs != len(config.seeds):
        raise ValueError("runs must match number of seeds when seeds are provided")

    for task in tasks:
        for run_index, seed in enumerate(seeds):
            run_id = str(uuid.uuid4())
            sandbox = Path(config.sandbox_root) / task.spec.task_id / f"run-{run_index}-{run_id}"
            sandbox.mkdir(parents=True, exist_ok=True)
            task.setup(sandbox, seed)

            apply_missing_fault(sandbox, config.faults.missing_path)
            apply_permission_fault(sandbox, config.faults.permission_path)

            runtime = config.runtime.model_copy(deep=True)
            runtime.seed = seed

            if config.orchestrator == "langgraph":
                mcp_client = build_mcp_client(config.mcp, config.faults)
                engine = LangGraphEngine(runtime, mcp_client, config.episode)
                trace = await engine.run(
                    run_id=run_id,
                    task_id=task.spec.task_id,
                    description=task.spec.description,
                    sandbox=sandbox,
                    validate=task.validate,
                    allowed_tool_names=task.allowed_tools,
                    seed=seed,
                )
                await mcp_client.close()
            elif config.orchestrator == "crewai":
                mcp_client = build_mcp_client(config.mcp, config.faults)
                engine = CrewAIEngine(runtime, mcp_client, config.episode)
                trace = await engine.run(
                    run_id=run_id,
                    task_id=task.spec.task_id,
                    description=task.spec.description,
                    sandbox=sandbox,
                    validate=task.validate,
                    allowed_tool_names=task.allowed_tools,
                    seed=seed,
                )
                await mcp_client.close()
            elif config.orchestrator == "temporal":
                mcp_settings = MCPSettings.model_validate(config.mcp.model_dump())
                fault_settings = FaultSettingsModel(
                    latency_ms=config.faults.latency_ms,
                    jitter_ms=config.faults.jitter_ms,
                    timeout_s=config.faults.timeout_s,
                )
                engine = TemporalEngine(runtime, config.episode, config.temporal, mcp_settings, fault_settings)
                trace = await engine.run(
                    run_id=run_id,
                    task_id=task.spec.task_id,
                    description=task.spec.description,
                    sandbox=sandbox,
                    allowed_tool_names=task.allowed_tools,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown orchestrator: {config.orchestrator}")

            writer.write(trace.model_dump())
            traces.append(trace)

    export_csv(traces, output_dir / "summary.csv")
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ORCHID benchmarks")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--orchestrator", choices=["langgraph", "crewai", "temporal"], default=None)
    parser.add_argument("--runtime", choices=[runtime.value for runtime in RuntimeName], default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.orchestrator:
        config.orchestrator = args.orchestrator
    if args.runtime:
        config.runtime.runtime = RuntimeName(args.runtime)
    if args.model:
        config.runtime.model = args.model

    asyncio.run(run_experiments(config))


if __name__ == "__main__":
    main()
