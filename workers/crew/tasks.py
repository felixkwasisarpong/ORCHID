"""Default CrewAI task definitions."""
from __future__ import annotations

from workers.crew.crew_config import AgentConfig, CrewConfig, TaskConfig


def default_crew_config() -> CrewConfig:
    agents = [
        AgentConfig(
            name="Planner",
            role="Planner",
            goal="Translate user requests into actionable steps.",
            backstory="Specialist in decomposing complex tasks into plans.",
        ),
        AgentConfig(
            name="Analyst",
            role="Analyst",
            goal="Produce concise, structured analysis from plans.",
            backstory="Expert in turning plans into research output.",
        ),
    ]
    tasks = [
        TaskConfig(
            name="PlanExpansion",
            description="Expand the plan into actionable steps.",
            expected_output="Bulleted list of steps.",
        ),
        TaskConfig(
            name="Synthesis",
            description="Synthesize steps into a concise summary.",
            expected_output="Short summary paragraph.",
        ),
    ]
    return CrewConfig(agents=agents, tasks=tasks, process="sequential")
