"""Build the minimal START -> model -> END execution graph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lineage_agent.agent.state import AgentState
from lineage_agent.models.types import ModelCall, ModelResult


def build_graph(call_model: ModelCall) -> CompiledStateGraph:
    """Inject model access so the graph only handles state and execution edges."""

    async def model_node(state: AgentState) -> dict[str, ModelResult]:
        result = await call_model(state["user_input"])
        return {"result": result}

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    return builder.compile()
