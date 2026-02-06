"""Pydantic schemas for CrewAI configuration."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    name: str
    role: str
    goal: str
    backstory: str


class TaskConfig(BaseModel):
    name: str
    description: str
    expected_output: str


class CrewConfig(BaseModel):
    agents: list[AgentConfig] = Field(default_factory=list)
    tasks: list[TaskConfig] = Field(default_factory=list)
    process: Literal["sequential", "hierarchical"] = "sequential"
