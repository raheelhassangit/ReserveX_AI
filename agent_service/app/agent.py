from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import re

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from .config import settings
from .django_client import django_post
from .guard import OUT_OF_SCOPE_REPLY, classify, classify_decision
from .llm import llm
from .tools import TOOLS, current_token, pending_active

EMPTY_RETRIES = 2
MAX_CONTEXT_MESSAGES = 20

SYSTEM_PROMPT = (
    "You are the ReserveX booking assistant. "
    "You can ONLY do what your tools allow: search offerings, list free seats, "
    "and hold seats for the user. Holding is not booking: the system asks the "
    "user to confirm afterwards, so never say a booking is confirmed. "
    "If the user asks for anything else (cancelling an existing booking, "
    "viewing bookings), say you can't do that yet. "
    "Always use tools to check real availability and prices; never guess. "
    "Offering ids come from search results. Keep answers short."
)

GREETING_PROMPT = (
    "You are the ReserveX assistant. Reply briefly and warmly to greetings, "
    "thanks and goodbyes. If asked what you can do, say you can search bus "
    "trips and appointments and hold seats for the user. Never invent "
    "offerings or prices."
)

GREETING_FALLBACK = "Hi! I can help you search bus trips and appointments."
UNSUPPORTED_RE = re.compile(r"\b(cancel\w*|refund\w*|reschedul\w*|modify)\b", re.IGNORECASE)

UNSUPPORTED_REPLY = (
    "I can't cancel, change or refund bookings from chat yet. "
    "I can search trips and hold seats for you."
)

NUDGE = (
    "Your last reply was empty. Either call a tool, or answer the user in "
    "plain text. If the request is outside ReserveX, say so politely."
)

FALLBACK = "Sorry, I couldn't process that. Could you rephrase your request?"

llm_with_tools = llm.bind_tools(TOOLS)
logger = logging.getLogger("agent")

class AgentState(MessagesState):
    category: str
    empty_retries: int
    pending: dict | None  # a hold waiting for the user's confirmation
    decision: str


def _today() -> str:
    return datetime.now(ZoneInfo(settings.app_timezone)).date().isoformat()


def _confirm_prompt(p: dict) -> str:
    return (
        f"Seats {', '.join(p['seat_numbers'])} on {p['offering_name']} are held for you "
        f"until {p['held_until_local']} (total Rs. {p['total_price']:,.0f}). "
        'Reply "yes" to confirm the booking or "cancel" to release them.'
    )


# ---------- nodes ----------

async def router_node(state: AgentState) -> dict:
    # Short replies like "2 seats please" have no keywords, so once a
    # conversation is in the reservation flow it stays there.
    if state.get("category") == "reservation":
        category = "reservation"
    else:
        category = await classify(state["messages"][-1].content)
    return {"category": category, "empty_retries": 0}


async def refuse_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=OUT_OF_SCOPE_REPLY)]}

async def unsupported_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=UNSUPPORTED_REPLY)]}


async def greet_node(state: AgentState) -> dict:
    # No tools are bound here: a greeting can never trigger a search.
    response = await llm.ainvoke(
        [SystemMessage(content=GREETING_PROMPT), state["messages"][-1]]
    )
    content = (response.content or "").strip() or GREETING_FALLBACK
    return {"messages": [AIMessage(content=content)]}


async def agent_node(state: AgentState) -> dict:
    system_text = SYSTEM_PROMPT + f" Today's date is {_today()}."
    pending = state.get("pending")
    if pending_active(pending):
        system_text += (
            f" The user has seats {', '.join(pending['seat_numbers'])} on hold for "
            f"{pending['offering_name']}, waiting for their confirmation."
        )
    # The full history stays in the database; the model sees only the last 20 messages.
    recent = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,  # count messages instead of tokens
        max_tokens=MAX_CONTEXT_MESSAGES,
        start_on="human",   # never begin in the middle of a tool exchange
        end_on=("human", "tool"),
        include_system=False,
    )
    response = await llm_with_tools.ainvoke(
        [SystemMessage(content=system_text), *recent]
    )
    if response.tool_calls:
        logger.info("TOOL CALL %s", [(c["name"], c["args"]) for c in response.tool_calls])
    return {"messages": [response]}


async def track_hold_node(state: AgentState) -> dict:
    """After the tools ran: log every result, and remember a successful hold."""
    update: dict = {}
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage):
            logger.info("TOOL RESULT %s -> %s", message.name, str(message.content)[:300])
            if message.name == "hold_seats" and message.artifact and not update:
                update = {"pending": message.artifact}
    return update


async def nudge_node(state: AgentState) -> dict:
    empty = state["messages"][-1]
    return {
        "messages": [RemoveMessage(id=empty.id), HumanMessage(content=NUDGE)],
        "empty_retries": state.get("empty_retries", 0) + 1,
    }


async def fallback_node(state: AgentState) -> dict:
    empty = state["messages"][-1]
    return {"messages": [RemoveMessage(id=empty.id), AIMessage(content=FALLBACK)]}


async def approval_node(state: AgentState) -> dict:
    pending = state["pending"]

    # The graph PAUSES here. When resumed, the node restarts from its first
    # line and interrupt() returns the user's reply instead of pausing, so
    # nothing above this line may have side effects.
    answer = interrupt({"type": "confirm_booking", "prompt": _confirm_prompt(pending)})

    text = answer.get("text", "")
    decision = answer.get("action") or await classify_decision(text)

    if decision == "other":
        # A question or unrelated message: let the agent answer, hold stays pending.
        return {
            "messages": [HumanMessage(content=text)],
            "decision": "other",
            "empty_retries": 0,
        }

    token = current_token.get()
    body = {"offering": pending["offering_id"], "seat_ids": pending["seat_ids"]}
    seats = ", ".join(pending["seat_numbers"])

    if decision == "confirm":
        result = await django_post("/api/bookings/confirm/", token, body)
        if "error" in result:
            reply = f"I couldn't confirm the booking: {result['error']}"
        else:
            reply = (
                f"Booked! Booking #{result['id']}: {pending['offering_name']}, "
                f"seats {seats}, total Rs. {float(result['total_price']):,.0f}."
            )
    else:
        await django_post("/api/bookings/release/", token, body)
        reply = f"Okay, I've released seats {seats}. Anything else I can help with?"

    return {
        "messages": [HumanMessage(content=text or decision), AIMessage(content=reply)],
        "pending": None,
        "decision": decision,
    }


# ---------- routing functions ----------

def route_by_category(state: AgentState) -> str:
    # Checked in code, so the model can never claim an action we can't do.
    if UNSUPPORTED_RE.search(state["messages"][-1].content):
        return "unsupported"
    return state["category"]


def after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    if not (last.content or "").strip():
        return "nudge" if state.get("empty_retries", 0) < EMPTY_RETRIES else "fallback"
    if pending_active(state.get("pending")):
        return "approval"
    return "end"


def after_approval(state: AgentState) -> str:
    return "agent" if state["decision"] == "other" else "end"


def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("refuse", refuse_node)
    builder.add_node("greet", greet_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_node("track_hold", track_hold_node)
    builder.add_node("nudge", nudge_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("unsupported", unsupported_node)
    builder.add_node("approval", approval_node)

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
        {"tools": "tools", "nudge": "nudge", "fallback": "fallback",
         "approval": "approval", "end": END},
    )
    builder.add_edge("tools", "track_hold")
    builder.add_edge("track_hold", "agent")
    builder.add_edge("nudge", "agent")
    builder.add_edge("fallback", END)

    builder.add_conditional_edges(
        "approval", after_approval, {"agent": "agent", "end": END}
    )
    builder.add_conditional_edges(
        "router",
        route_by_category,
        {"other": "refuse", "greeting": "greet", "reservation": "agent",
         "unsupported": "unsupported"},
    )
    builder.add_edge("unsupported", END)
    # In-memory for now: state is lost when the container restarts.
    return builder.compile(checkpointer=checkpointer)
