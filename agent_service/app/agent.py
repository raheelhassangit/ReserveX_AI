from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import settings
from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are the ReserveX booking assistant. "
    "You can ONLY search offerings and list free seats. Holding or booking "
    "seats is not available in this version yet: say so if asked. "
    "Always use tools to check real availability and prices; never guess. "
    "Keep answers short."
)

llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    num_ctx=settings.ollama_num_ctx,
    temperature=0,
    # Ignored by local Ollama; used if we switch back to the cloud API.
    client_kwargs={"headers": {"Authorization": f"Bearer {settings.ollama_api_key}"}},
)
llm_with_tools = llm.bind_tools(TOOLS)


def _today() -> str:
    return datetime.now(ZoneInfo(settings.app_timezone)).date().isoformat()


async def agent_node(state: MessagesState) -> dict:
    # The system prompt is added on every call and never stored in state.
    system = SystemMessage(content=SYSTEM_PROMPT + f" Today's date is {_today()}.")
    response = await llm_with_tools.ainvoke([system, *state["messages"]])
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # tool calls -> "tools", else END
    builder.add_edge("tools", "agent")  # after tools, the model reads the results

    return builder.compile()


graph = build_graph()