"""CrewAI execution wrappers."""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from workers.crew.crew_config import CrewConfig
from workers.crew.llm import LLMConfig, build_llm_config_payload, normalize_model


class CrewResult(BaseModel):
    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrewRunner:
    """Async wrapper around CrewAI execution."""

    def __init__(self, config: CrewConfig, llm_config: LLMConfig | None = None) -> None:
        self.config = config
        self.llm_config = llm_config or LLMConfig()

    async def run(self, input_text: str) -> CrewResult:
        return await asyncio.to_thread(self._run_sync, input_text)

    def _run_sync(self, input_text: str) -> CrewResult:
        try:
            from crewai import Agent, Crew, Process, Task
            try:
                from crewai import LLM as CrewLLM
            except Exception:  # noqa: BLE001
                from crewai.llm import LLM as CrewLLM
        except Exception as exc:  # noqa: BLE001
            return CrewResult(
                output=f"CrewAI unavailable: {exc}. Input: {input_text}",
                metadata={
                    "fallback": True,
                    "llm_runtime": self.llm_config.runtime.value,
                    "llm_model": normalize_model(self.llm_config.runtime, self.llm_config.model),
                },
            )

        llm_payload = build_llm_config_payload(self.llm_config)
        llm = CrewLLM(**llm_payload)

        agents = [
            Agent(
                role=agent.role,
                goal=agent.goal,
                backstory=agent.backstory,
                llm=llm,
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
            manager_llm=llm,
            verbose=False,
        )

        output = crew.kickoff(inputs={"input": input_text})
        return CrewResult(
            output=str(output),
            metadata={
                "agents": len(agents),
                "tasks": len(tasks),
                "llm_runtime": self.llm_config.runtime.value,
                "llm_model": normalize_model(self.llm_config.runtime, self.llm_config.model),
            },
        )
