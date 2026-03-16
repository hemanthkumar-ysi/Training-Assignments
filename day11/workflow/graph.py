from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.planner_agent import planner_agent
from agents.research_agent import research_agent
from agents.knowledge_agent import knowledge_agent
from agents.router_agent import router_agent
from agents.writer_agent import writer_agent


class AgentState(TypedDict):

    question: str
    action: str
    topic: str
    research: str
    knowledge: str
    final: str


def create_graph(tools):

    builder = StateGraph(AgentState)

    async def research_node(state):
        return await research_agent(state, tools)

    async def knowledge_node(state):
        return await knowledge_agent(state, tools)

    builder.add_node("planner", planner_agent)
    builder.add_node("research", research_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("router", router_agent)
    builder.add_node("writer", writer_agent)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "knowledge")
    builder.add_edge("knowledge", "router")

    def route_after_knowledge(state):
        action = state.get("action")
        if action == "web_search":
            return "research"
        return "writer"

    builder.add_conditional_edges(
        "router",
        route_after_knowledge,
        {
            "research": "research",
            "writer": "writer"
        }
    )

    builder.add_edge("research", "writer")

    builder.add_edge("writer", END)

    return builder.compile()