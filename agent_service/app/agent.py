from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .config import settings
from .guard import OUT_OF_SCOPE_REPLY, classify
from .llm import llm
from .tools import TOOLS

EMPTY_RETRIES = 2

SYSTEM_PROMPT = (
    "You are the ReserveX booking assistant. "
    "You can ONLY search offerings and list free seats. Holding or booking "
    "seats is not available in this version yet: say so if asked. "
    "Always use tools to check real availability and prices; never guess. "
    "Keep answers short."
)

GREETING_PROMPT = (
    "You are the ReserveX assistant. Reply briefly and warmly to greetings, "
    "thanks and goodbyes. If asked what you can do, say you can search bus "
    "trips and appointments for the user. Never invent offerings or prices."
)

GREETING_FALLBACK = "Hi! I can help you search bus trips and appointments."

NUDGE = (
    "Your last reply was empty. Either call a tool, or answer the user in "
    "plain text. If the request is outside ReserveX, say so politely."
)

FALLBACK = "Sorry, I couldn't process that. Could you rephrase your request?"

llm_with_tools = llm.bind_tools(TOOLS)


class AgentState(MessagesState):
    category: str
    empty_retries: int


def _today() -> str:
    return datetime.now(ZoneInfo(settings.app_timezone)).date().isoformat()


# ---------- nodes: each returns a partial update to the state ----------

async def router_node(state: AgentState) -> dict:
    category = await classify(state["messages"][-1].content)
    return {"category": category, "empty_retries": 0}


async def refuse_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=OUT_OF_SCOPE_REPLY)]}


async def greet_node(state: AgentState) -> dict:
    # No tools are bound here: a greeting can never trigger a search.
    response = await llm.ainvoke(
        [SystemMessage(content=GREETING_PROMPT), state["messages"][-1]]
    )
    content = (response.content or "").strip() or GREETING_FALLBACK
    return {"messages": [AIMessage(content=content)]}


async def agent_node(state: AgentState) -> dict:
    # The system prompt is added on every call and never stored in state.
    system = SystemMessage(content=SYSTEM_PROMPT + f" Today's date is {_today()}.")
    response = await llm_with_tools.ainvoke([system, *state["messages"]])
    return {"messages": [response]}


async def nudge_node(state: AgentState) -> dict:
    empty = state["messages"][-1]
    return {
        # delete the empty reply, then ask again
        "messages": [RemoveMessage(id=empty.id), HumanMessage(content=NUDGE)],
        "empty_retries": state.get("empty_retries", 0) + 1,
    }


async def fallback_node(state: AgentState) -> dict:
    empty = state["messages"][-1]
    return {"messages": [RemoveMessage(id=empty.id), AIMessage(content=FALLBACK)]}


# ---------- routing functions: return a label, the path map picks the node ----------

def route_by_category(state: AgentState) -> str:
    return state["category"]


def after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    if not (last.content or "").strip():
        if state.get("empty_retries", 0) < EMPTY_RETRIES:
            return "nudge"
        return "fallback"
    return "end"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("refuse", refuse_node)
    builder.add_node("greet", greet_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_node("nudge", nudge_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_by_category,
        {"other": "refuse", "greeting": "greet", "reservation": "agent"},
    )
    builder.add_edge("refuse", END)
    builder.add_edge("greet", END)

    builder.add_conditional_edges(
        "agent",
        after_agent,
        {"tools": "tools", "nudge": "nudge", "fallback": "fallback", "end": END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("nudge", "agent")
    builder.add_edge("fallback", END)

    return builder.compile()


graph = build_graph()