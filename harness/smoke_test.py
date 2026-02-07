"""Run a smoke test across all orchestrators."""
from __future__ import annotations

import asyncio

from harness.run_experiments import load_config, run_experiments


async def run_all() -> None:
    config = load_config("configs/smoke.yaml")
    for orchestrator in ["langgraph", "crewai", "temporal"]:
        config.orchestrator = orchestrator
        await run_experiments(config)


def main() -> None:
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
