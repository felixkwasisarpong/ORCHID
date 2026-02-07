"""Start a Temporal worker then run experiments."""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time

from harness.run_experiments import load_config, run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Temporal worker and experiments")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--worker-startup-s", type=float, default=3.0)
    args = parser.parse_args()

    worker = subprocess.Popen([sys.executable, "-m", "workers.temporal_worker"])
    try:
        time.sleep(args.worker_startup_s)
        config = load_config(args.config)
        config.orchestrator = "temporal"
        asyncio.run(run_experiments(config))
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()


if __name__ == "__main__":
    main()
