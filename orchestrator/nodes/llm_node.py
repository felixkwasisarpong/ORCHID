"""LLM node for non-Crew runs (e.g., langgraph baseline)."""
from __future__ import annotations

import asyncio
from typing import Any

from orchestrator.nodes.base_node import BaseNode
from orchestrator.state import GlobalState
from workers.crew.llm import LLMConfig, build_llm_config_payload, normalize_model


class LLMNode(BaseNode):
    """Direct LLM call without CrewAI."""

    def __init__(
        self,
        name: str,
        llm_config: LLMConfig,
        input_field: str = "plan",
        output_field: str = "crew_output",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.llm_config = llm_config
        self.input_field = input_field
        self.output_field = output_field

    async def run(self, state: GlobalState) -> GlobalState:
        payload = getattr(state, self.input_field)
        if payload is None:
            return state.add_error(f"{self.name} missing input '{self.input_field}'")
        try:
            import litellm
        except Exception as exc:  # noqa: BLE001
            return state.add_error(f"{self.name} missing litellm: {exc}")

        llm_payload = build_llm_config_payload(self.llm_config)
        messages = [
            {
                "role": "user",
                "content": payload,
            }
        ]
        response = await asyncio.to_thread(litellm.completion, messages=messages, **llm_payload)

        content = None
        usage = None
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content")
            usage = response.get("usage")
        else:
            choices = getattr(response, "choices", None)
            if choices:
                message = getattr(choices[0], "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
            usage = getattr(response, "usage", None)

        if content is None:
            content = str(response)

        token_usage = 0
        if isinstance(usage, dict):
            token_usage = int(usage.get("total_tokens", 0) or 0)

        llm_meta = {
            "runtime": self.llm_config.runtime.value,
            "model": normalize_model(self.llm_config.runtime, self.llm_config.model),
            "usage": usage,
        }
        metadata = {**state.metadata, "llm": llm_meta, "token_usage": token_usage}
        return state.model_copy(
            update={
                self.output_field: content,
                "metadata": metadata,
            }
        )
