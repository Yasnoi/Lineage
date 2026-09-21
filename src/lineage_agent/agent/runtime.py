"""Assemble dependencies and run one agent execution."""

from functools import partial

from lineage_agent.agent.graph import build_graph
from lineage_agent.agent.state import AgentState
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.responses import create_client, generate_response


async def run_agent(user_input: str, settings: ModelSettings) -> AgentState:
    """Run once with fresh state and close the client on success or failure."""
    if not user_input.strip():
        raise ValueError("用户输入不能为空。")

    initial_state: AgentState = {"user_input": user_input}
    async with create_client(settings) as client:
        call_model = partial(generate_response, client, settings.model_name)
        graph = build_graph(call_model)
        return await graph.ainvoke(initial_state)
