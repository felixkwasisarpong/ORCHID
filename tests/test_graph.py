import pytest

from orchestrator.graph import build_example_graph
from orchestrator.state import GlobalState
from tools.mcp_client import MCPResponse
from workers.crew.agents import CrewRunner, CrewResult
from workers.crew.tasks import default_crew_config


class FakeMCPClient:
    async def call_tool(self, tool_name, payload, request_id, fault=None):
        return MCPResponse(status="ok", output={"echo": payload}, metadata={})

    async def close(self):
        return None


class FakeCrewRunner(CrewRunner):
    async def run(self, input_text: str) -> CrewResult:
        return CrewResult(output=f"crew:{input_text}", metadata={"fake": True})


@pytest.mark.asyncio
async def test_graph_runs_successfully():
    client = FakeMCPClient()
    crew_runner = FakeCrewRunner(default_crew_config())
    graph = build_example_graph(client, crew_runner)
    state = GlobalState(request_id="test", user_input="hello")
    final_state = await graph.run(state)
    assert final_state.final_output is not None
    assert final_state.errors == []
