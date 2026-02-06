"""CrewAI execution wrappers."""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from workers.crew.crew_config import CrewConfig


class CrewResult(BaseModel):
    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrewRunner:
    """Async wrapper around CrewAI execution."""

    def __init__(self, config: CrewConfig) -> None:
        self.config = config

    async def run(self, input_text: str) -> CrewResult:
        return await asyncio.to_thread(self._run_sync, input_text)

    def _run_sync(self, input_text: str) -> CrewResult:
        try:
            from crewai import Agent, Crew, Process, Task
        except Exception as exc:  # noqa: BLE001
            return CrewResult(
                output=f"CrewAI unavailable: {exc}. Input: {input_text}",
                metadata={"fallback": True},
            )

        agents = [
            Agent(
                role=agent.role,
                goal=agent.goal,
                backstory=agent.backstory,
                verbose=False,
                allow_delegation=False,
            )
            for agent in self.config.agents
        ]

        tasks = [
            Task(
                description=task.description,
                expected_output=task.expected_output,
                agent=agents[min(idx, len(agents) - 1)],
            )
            for idx, task in enumerate(self.config.tasks)
        ]

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential if self.config.process == "sequential" else Process.hierarchical,
            verbose=False,
        )

        output = crew.kickoff(inputs={"input": input_text})
        return CrewResult(
            output=str(output),
            metadata={"agents": len(agents), "tasks": len(tasks)},
        )
