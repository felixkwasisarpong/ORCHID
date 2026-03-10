"""Run experiment suites across orchestrators and runtimes."""

from __future__ import annotations

import argparse
import asyncio
import csv
import time
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Dict, List
from uuid import uuid4

from benchmarks.tasks import get_task, list_task_specs
from harness.config import ExperimentConfig, FaultConfig, load_config
from observability.logger import JSONLLogger, default_trace_path
from observability.trace_schema import RunTrace
from orchestrators.autogen_engine import AutoGenEngine
from orchestrators.common import EpisodeConfig
from orchestrators.crewai_engine import CrewAIEngine
from orchestrators.langgraph_engine import LangGraphEngine
from runtimes.anthropic_client import AnthropicClient
from runtimes.base import RuntimeConfig
from runtimes.gemini_client import GeminiClient
from runtimes.ollama_client import OllamaClient
from runtimes.openai_client import OpenAIClient
from tools.mcp_gateway_client import MCPClientConfig, MCPGatewayClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run orchestrator experiments")
    parser.add_argument("--config", type=str, help="Path to YAML config", default=None)
    parser.add_argument("--orchestrator", type=str, help="Override orchestrator")
    parser.add_argument("--runtime", type=str, help="Override runtime")
    parser.add_argument("--task", type=str, help="Task ID or 'all'")
    parser.add_argument("--seeds", type=int, nargs="+", help="Seeds list")
    parser.add_argument("--fault-permission", type=str, help="Relative path to chmod read-only")
    parser.add_argument("--fault-missing", type=str, help="Relative path to delete")
    parser.add_argument("--fault-latency-ms", type=float, help="Client-side latency in ms")
    parser.add_argument("--fault-jitter-ms", type=float, help="Client-side jitter in ms")
    parser.add_argument("--fault-timeout-s", type=float, help="Client-side tool timeout in seconds")
    return parser.parse_args()


def apply_faults(sandbox_root: Path, faults: FaultConfig) -> None:
    if faults.permission_path:
        target = sandbox_root / faults.permission_path
        if target.exists():
            target.chmod(0o444)
    if faults.missing_path:
        target = sandbox_root / faults.missing_path
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()


def build_runtime(runtime_name: str, config: ExperimentConfig) -> tuple[str, object]:
    model = config.runtime_models.get(runtime_name)
    if not model:
        default_models = {
            "ollama": "qwen2.5:14b",
            "openai": "gpt-5.2",
            "anthropic": "claude-opus-4-6",
            "gemini": "gemini-3-pro-preview",
            "mistral": "mistral-large-2512+1",
            "grok": "grok-4.1-fast",
            "xai": "grok-4.1-fast",
        }
        model = default_models.get(runtime_name, "")
    if not model:
        raise ValueError(f"No model configured for runtime {runtime_name}")
    runtime_config = RuntimeConfig(model=model, temperature=0.0, max_tokens=512, timeout_s=config.timeout_s)

    if runtime_name == "ollama":
        return runtime_name, OllamaClient(runtime_config)
    if runtime_name == "openai":
        return runtime_name, OpenAIClient(runtime_config)
    if runtime_name == "anthropic":
        return runtime_name, AnthropicClient(runtime_config)
    if runtime_name == "gemini":
        return runtime_name, GeminiClient(runtime_config)
    if runtime_name == "mistral":
        return runtime_name, OpenAIClient(
            runtime_config,
            base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
            api_key=os.getenv("MISTRAL_API_KEY"),
            enforce_json_response=False,
        )
    if runtime_name in {"grok", "xai"}:
        return runtime_name, OpenAIClient(
            runtime_config,
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
            api_key=os.getenv("XAI_API_KEY"),
            enforce_json_response=False,
        )
    raise ValueError(f"Unknown runtime: {runtime_name}")


def build_orchestrator(
    name: str,
    runtime: object,
    tool_client: MCPGatewayClient,
    validator,
    episode_config: EpisodeConfig,
):
    if name == "langgraph":
        return LangGraphEngine(runtime, tool_client, validator, episode_config)
    if name == "crewai":
        return CrewAIEngine(runtime, tool_client, validator, episode_config)
    if name == "autogen":
        return AutoGenEngine(runtime, tool_client, validator, episode_config)
    raise ValueError(f"Unknown orchestrator: {name}")


def task_ids_from_config(config: ExperimentConfig) -> List[str]:
    if not config.tasks or "all" in config.tasks:
        return [spec.id for spec in list_task_specs()]
    return config.tasks


async def run_once(
    orchestrator_name: str,
    runtime_name: str,
    runtime,
    task_id: str,
    seed: int,
    config: ExperimentConfig,
    results_dir: Path,
) -> RunTrace:
    task_def = get_task(task_id)
    run_id = f"{task_id}-{orchestrator_name}-{runtime_name}-{uuid4().hex[:8]}"
    sandbox_root = (Path(config.sandbox_dir).resolve() / run_id).resolve()
    sandbox_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()

    task_def.setup(sandbox_root)
    # Keep output writes deterministic across tasks by ensuring output root exists.
    (sandbox_root / "output").mkdir(parents=True, exist_ok=True)
    apply_faults(sandbox_root, config.faults)

    episode_config = EpisodeConfig(
        max_steps=config.max_steps,
        max_llm_retries=config.max_llm_retries,
        max_tool_retries=config.max_tool_retries,
        timeout_s=config.timeout_s,
        tool_timeout_s=config.faults.tool_timeout_s or config.tool_timeout_s,
    )

    gateway_cmd = None
    path_rewrite_from = None
    path_rewrite_to = None
    if config.gateway_cmd:
        gateway_cmd = [arg.format(sandbox_root=str(sandbox_root)) for arg in config.gateway_cmd]
        if any("/local-directory" in arg for arg in gateway_cmd):
            path_rewrite_from = str(sandbox_root)
            path_rewrite_to = "/local-directory"

    mcp_config = MCPClientConfig(
        transport=config.transport,
        gateway_cmd=gateway_cmd,
        http_url=config.http_url,
        request_timeout_s=episode_config.tool_timeout_s,
        latency_ms=config.faults.latency_ms,
        jitter_ms=config.faults.jitter_ms,
        path_rewrite_from=path_rewrite_from,
        path_rewrite_to=path_rewrite_to,
    )

    trace_path = default_trace_path(results_dir, run_id)
    logger = JSONLLogger(trace_path)
    fault_config = {
        "permission_path": config.faults.permission_path,
        "missing_path": config.faults.missing_path,
        "latency_ms": config.faults.latency_ms,
        "jitter_ms": config.faults.jitter_ms,
        "tool_timeout_s": episode_config.tool_timeout_s,
    }

    try:
        async with MCPGatewayClient(mcp_config) as tool_client:
            orchestrator = build_orchestrator(
                orchestrator_name,
                runtime,
                tool_client,
                task_def.validate,
                episode_config,
            )
            trace = await orchestrator.run(task_def.spec, sandbox_root, seed, run_id, runtime_name)
        trace = trace.model_copy(update={"fault_config": fault_config})
    except Exception as exc:  # noqa: BLE001
        ended_perf = time.perf_counter()
        trace = RunTrace(
            run_id=run_id,
            orchestrator=orchestrator_name,
            runtime=runtime_name,
            task_id=task_id,
            seed=seed,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            total_latency_ms=(ended_perf - started_perf) * 1000.0,
            llm_calls=0,
            tool_calls=0,
            retries=0,
            steps=[],
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            fault_config=fault_config,
        )

    logger.log_trace(trace)
    return trace


async def run_suite(config: ExperimentConfig) -> List[RunTrace]:
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    traces: List[RunTrace] = []
    for orchestrator_name in config.orchestrators:
        for runtime_name in config.runtimes:
            runtime_name, runtime = build_runtime(runtime_name, config)
            for task_id in task_ids_from_config(config):
                for seed in config.seeds:
                    trace = await run_once(
                        orchestrator_name,
                        runtime_name,
                        runtime,
                        task_id,
                        seed,
                        config,
                        results_dir,
                    )
                    traces.append(trace)
    return traces


def write_summary(traces: List[RunTrace], results_dir: Path) -> Path:
    path = results_dir / f"summary_{int(time.time())}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "orchestrator",
                "runtime",
                "task_id",
                "seed",
                "success",
                "llm_calls",
                "tool_calls",
                "retries",
                "llm_prompt_tokens",
                "llm_completion_tokens",
                "llm_total_tokens",
                "llm_cost_usd",
                "total_latency_ms",
            ]
        )
        for trace in traces:
            writer.writerow(
                [
                    trace.run_id,
                    trace.orchestrator,
                    trace.runtime,
                    trace.task_id,
                    trace.seed,
                    trace.success,
                    trace.llm_calls,
                    trace.tool_calls,
                    trace.retries,
                    trace.llm_prompt_tokens,
                    trace.llm_completion_tokens,
                    trace.llm_total_tokens,
                    f"{trace.llm_cost_usd:.8f}",
                    f"{trace.total_latency_ms:.2f}",
                ]
            )
    return path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    if args.orchestrator:
        config.orchestrators = [args.orchestrator]
    if args.runtime:
        config.runtimes = [args.runtime]
    if args.task:
        config.tasks = [args.task]
    if args.seeds:
        config.seeds = args.seeds

    if args.fault_permission:
        config.faults.permission_path = args.fault_permission
    if args.fault_missing:
        config.faults.missing_path = args.fault_missing
    if args.fault_latency_ms is not None:
        config.faults.latency_ms = args.fault_latency_ms
    if args.fault_jitter_ms is not None:
        config.faults.jitter_ms = args.fault_jitter_ms
    if args.fault_timeout_s is not None:
        config.faults.tool_timeout_s = args.fault_timeout_s

    traces = asyncio.run(run_suite(config))
    summary_path = write_summary(traces, Path(config.results_dir))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
