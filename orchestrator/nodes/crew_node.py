"""CrewAI node wrapper."""
from __future__ import annotations

from typing import Any

from orchestrator.nodes.base_node import BaseNode
from orchestrator.state import GlobalState
from workers.crew.agents import CrewRunner, CrewResult


class CrewNode(BaseNode):
    """Run a CrewAI team for structured input/output."""

    def __init__(
        self,
        name: str,
        crew_runner: CrewRunner,
        input_field: str = "plan",
        output_field: str = "crew_output",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.crew_runner = crew_runner
        self.input_field = input_field
        self.output_field = output_field

    async def run(self, state: GlobalState) -> GlobalState:
        payload = getattr(state, self.input_field)
        if payload is None:
            return state.add_error(f"{self.name} missing input '{self.input_field}'")
        self.logger.info(
            "crew_input",
            extra={
                "request_id": state.request_id,
                "node": self.name,
                "input_field": self.input_field,
            },
        )
        result: CrewResult = await self.crew_runner.run(payload)
        self.logger.info(
            "crew_output",
            extra={
                "request_id": state.request_id,
                "node": self.name,
                "output_len": len(result.output),
                "metadata": result.metadata,
                "llm_runtime": result.metadata.get("llm_runtime"),
                "llm_model": result.metadata.get("llm_model"),
            },
        )
        metadata = {**state.metadata, "crew": result.metadata}
        return state.model_copy(
            update={
                self.output_field: result.output,
                "metadata": metadata,
            }
        )
