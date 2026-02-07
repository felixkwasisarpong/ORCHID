"""Run a single-task smoke test across all orchestrators using Ollama."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness.config import load_config
from harness.run_experiments import run_suite, write_summary


def main() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    traces = asyncio.run(run_suite(config))
    summary_path = write_summary(traces, Path(config.results_dir))
    print(f"Smoke test summary written to {summary_path}")


if __name__ == "__main__":
    main()
