"""Temporal worker for ORCHID workflows."""
from __future__ import annotations

import argparse
import asyncio

from temporalio import client, worker

from orchestrators.temporal_engine import (
    OrchidWorkflow,
    list_tools_activity,
    llm_decide_activity,
    tool_call_activity,
    validate_activity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ORCHID Temporal worker")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--task-queue", default="orchid-task-queue")
    return parser.parse_args()


async def run_worker() -> None:
    args = parse_args()
    temporal_client = await client.Client.connect(args.address, namespace=args.namespace)
    worker_instance = worker.Worker(
        temporal_client,
        task_queue=args.task_queue,
        workflows=[OrchidWorkflow],
        activities=[
            list_tools_activity,
            llm_decide_activity,
            tool_call_activity,
            validate_activity,
        ],
    )
    await worker_instance.run()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
