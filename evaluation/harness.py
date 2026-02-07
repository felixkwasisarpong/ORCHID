"""Experiment harness for running orchestration scenarios."""
from __future__ import annotations

import asyncio
import csv
import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evaluation.fault_injection import fault_from_dict
from evaluation.metrics import compute_metrics, count_retries
from orchestrator.graph import build_example_graph
from orchestrator.state import GlobalState
from tools.mcp_client import MCPClient, MCPTransport
from workers.crew.agents import CrewRunner
from workers.crew.llm import LLMConfig
from workers.crew.tasks import default_crew_config


class Scenario(BaseModel):
    name: str
    user_input: str
    runs: int = 1
    mcp_base_url: str = "http://localhost:9000"
    mode: str = "hybrid"
    fault: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    mcp_transport: str = "http"
    mcp_server_command: str | None = None
    mcp_server_args: list[str] | None = None
    mcp_fs_paths: list[str] | None = None
    force_tool: bool = False
    force_llm: bool = False
    tags: list[str] = Field(default_factory=list)


def load_scenarios(path: str | Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for file_path in Path(path).glob("*.json"):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            scenarios.extend(Scenario(**item) for item in data)
        else:
            scenarios.append(Scenario(**data))
    return scenarios


async def run_scenario(scenario: Scenario) -> dict[str, Any]:
    fault = fault_from_dict(scenario.fault)
    llm_config = LLMConfig(**scenario.llm) if scenario.llm else LLMConfig()
    transport = MCPTransport(scenario.mcp_transport)
    if transport == MCPTransport.STDIO:
        paths = scenario.mcp_fs_paths or ["/app"]
        args = scenario.mcp_server_args or ["-y", "@modelcontextprotocol/server-filesystem", *paths]
        command = scenario.mcp_server_command or "npx"
        client = MCPClient(transport=transport, command=command, args=args)
    else:
        client = MCPClient(transport=transport, base_url=scenario.mcp_base_url)
    crew_runner = CrewRunner(default_crew_config(), llm_config=llm_config)
    if scenario.mode == "hybrid":
        graph = build_example_graph(client, crew_runner, fault_config=fault)
    elif scenario.mode == "crew":
        from orchestrator.graph import Graph, PlanningNode, ValidationNode
        from orchestrator.nodes.crew_node import CrewNode
        from orchestrator.nodes.llm_node import LLMNode
        from orchestrator.nodes.tool_node import ToolNode
        from orchestrator.policies import FailureAction, FailurePolicy, RetryPolicy
        from orchestrator.graph import resolve_tool_config

        tool_name, payload_override = resolve_tool_config(client)
        graph = Graph(
            [
                PlanningNode(name="planner", retry_policy=RetryPolicy(max_attempts=1)),
                CrewNode(
                    name="crew_exec",
                    crew_runner=crew_runner,
                    retry_policy=RetryPolicy(max_attempts=2),
                    timeout_s=300.0,
                ),
                *(
                    [
                        ToolNode(
                            name="tool_call",
                            mcp_client=client,
                            tool_name=tool_name,
                            retry_policy=RetryPolicy(max_attempts=2),
                            timeout_s=10.0,
                            fault_config=fault,
                            payload_override=payload_override,
                        )
                    ]
                    if scenario.force_tool
                    else []
                ),
                ValidationNode(
                    name="validator",
                    failure_policy=FailurePolicy(on_failure=FailureAction.CONTINUE),
                ),
            ]
        )
    elif scenario.mode == "langgraph":
        from orchestrator.graph import Graph, PlanningNode, ValidationNode
        from orchestrator.nodes.llm_node import LLMNode
        from orchestrator.nodes.tool_node import ToolNode
        from orchestrator.policies import FailureAction, FailurePolicy, RetryPolicy
        from orchestrator.graph import resolve_tool_config

        tool_name, payload_override = resolve_tool_config(client)
        nodes = [PlanningNode(name="planner", retry_policy=RetryPolicy(max_attempts=1))]
        if scenario.force_llm:
            nodes.append(
                LLMNode(
                    name="llm_exec",
                    llm_config=llm_config,
                    retry_policy=RetryPolicy(max_attempts=2),
                    timeout_s=300.0,
                )
            )
        if scenario.force_tool:
            nodes.append(
                ToolNode(
                    name="tool_call",
                    mcp_client=client,
                    tool_name=tool_name,
                    retry_policy=RetryPolicy(max_attempts=2),
                    timeout_s=10.0,
                    fault_config=fault,
                    payload_override=payload_override,
                )
            )
        nodes.append(
            ValidationNode(
                name="validator",
                failure_policy=FailurePolicy(on_failure=FailureAction.CONTINUE),
            )
        )
        graph = Graph(
            nodes
        )
    else:
        raise ValueError(f"Unknown scenario mode: {scenario.mode}")

    results: list[dict[str, Any]] = []
    try:
        for _ in range(scenario.runs):
            request_id = str(uuid.uuid4())
            state = GlobalState(request_id=request_id, user_input=scenario.user_input)
            start = time.perf_counter()
            final_state = await graph.run(state)
            latency_ms = (time.perf_counter() - start) * 1000.0
            retries = count_retries(final_state.trace)
            tool_calls = 1 if final_state.tool_result else 0
            token_usage = final_state.metadata.get("token_usage", 0)
            crew_meta = final_state.metadata.get("crew", {}) if isinstance(final_state.metadata, dict) else {}
            llm_runtime = crew_meta.get("llm_runtime")
            llm_model = crew_meta.get("llm_model")
            results.append(
                {
                    "request_id": request_id,
                    "success": len(final_state.errors) == 0,
                    "latency_ms": latency_ms,
                    "retries": retries,
                    "tool_calls": tool_calls,
                    "token_usage": token_usage,
                    "llm_runtime": llm_runtime,
                    "llm_model": llm_model,
                    "errors": final_state.errors,
                }
            )
    finally:
        await client.close()

    summary = compute_metrics(results)
    return {"scenario": scenario.name, "results": results, "summary": summary}


async def run_all(scenarios_path: str | Path, output_dir: str | Path) -> Path:
    scenarios = load_scenarios(scenarios_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"results-{timestamp}.json"

    runs = [await run_scenario(scenario) for scenario in scenarios]
    output_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return output_path


def export_csv(json_path: str | Path, csv_path: str | Path) -> None:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in data:
        scenario = entry["scenario"]
        for result in entry["results"]:
            rows.append({"scenario": scenario, **result})

    fieldnames = [
        "scenario",
        "request_id",
        "success",
        "latency_ms",
        "retries",
        "tool_calls",
        "token_usage",
        "llm_runtime",
        "llm_model",
        "errors",
    ]
    with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run orchestration scenarios")
    parser.add_argument("--scenarios", default="evaluation/scenarios")
    parser.add_argument("--output", default="evaluation/results")
    parser.add_argument("--export-csv", action="store_true")
    args = parser.parse_args()

    output_path = asyncio.run(run_all(args.scenarios, args.output))
    if args.export_csv:
        export_csv(output_path, Path(args.output) / "results.csv")


if __name__ == "__main__":
    main()
