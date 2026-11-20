import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import llm, session
from .config import settings
from .django_client import django_post
from .guard import OUT_OF_SCOPE_REPLY, classify, classify_decision
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, ToolContext

logger = logging.getLogger("agent")

MAX_STEPS = 6
EMPTY_RETRIES = 2

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

NUDGE = (
    "Your last reply was empty. Either call a tool, or answer the user in "
    "plain text. If the request is outside ReserveX, say so politely."
)

FALLBACK = "Sorry, I couldn't process that. Could you rephrase your request?"


def _today() -> str:
    return datetime.now(ZoneInfo(settings.app_timezone)).date().isoformat()


def _expired(pending: dict) -> bool:
    until = datetime.fromisoformat(pending["held_until"].replace("Z", "+00:00"))
    return datetime.now(timezone.utc) >= until


async def _persist(user_key: str, session_id: str, messages: list[dict],
                   category: str, pending: dict | None) -> None:
    history = [m for m in messages[1:] if m.get("content") != NUDGE]
    await session.save(user_key, session_id, history, category, pending)


async def execute_tool(name: str, args: dict, ctx: ToolContext) -> dict:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        result = {"error": f"Unknown tool '{name}'"}
    else:
        try:
            result = await func(ctx, **args)
        except TypeError as e:
            result = {"error": f"Bad arguments: {e}"}
        except Exception as e:
            logger.exception("Tool %s failed", name)
            result = {"error": f"Tool failed: {e}"}
    logger.info("TOOL %s args=%s -> %s", name, args, str(result)[:300])
    return result


async def _resolve_pending(ctx: ToolContext, pending: dict, decision: str) -> str:
    body = {"offering": pending["offering_id"], "seat_ids": pending["seat_ids"]}
    seats = ", ".join(pending["seat_numbers"])

    if decision == "confirm":
        result = await django_post("/api/bookings/confirm/", ctx.token, body)
        if "error" in result:
            return f"I couldn't confirm the booking: {result['error']}"
        return (
            f"Booked! Booking #{result['id']}: {pending['offering_name']}, "
            f"seats {seats}, total Rs. {result['total_price']}."
        )

    await django_post("/api/bookings/release/", ctx.token, body)
    return f"Okay, I've released seats {seats}. Anything else I can help with?"


async def run_agent(
    user_message: str,
    token: str | None,
    user_key: str,
    session_id: str,
    action: str | None = None,
) -> dict:
    ctx = ToolContext(token=token)
    state = await session.load(user_key, session_id)

    pending = state.get("pending")
    if pending and _expired(pending):
        pending = None
    ctx.pending = pending
    pending_before = pending

    system = SYSTEM_PROMPT + f" Today's date is {_today()}."
    if pending:
        system += (
            f" The user has seats {', '.join(pending['seat_numbers'])} on hold for "
            f"{pending['offering_name']}, waiting for their confirmation."
        )

    messages = [
        {"role": "system", "content": system},
        *state["messages"],
        {"role": "user", "content": user_message},
    ]
    start = len(messages) - 1

    # 1) A hold is waiting: CODE decides, the model never confirms.
    if action and not pending:
        text = "There is no active hold. It may have expired. Search again to start a new booking."
        messages.append({"role": "assistant", "content": text})
        await _persist(user_key, session_id, messages, "reservation", None)
        return {"reply": text, "trace": messages[start:]}

    if pending:
        decision = action or await classify_decision(user_message)
        if decision in ("confirm", "cancel"):
            text = await _resolve_pending(ctx, pending, decision)
            messages.append({"role": "assistant", "content": text})
            await _persist(user_key, session_id, messages, "reservation", None)
            return {"reply": text, "trace": messages[start:]}

    # 2) Routing
    if state["last_category"] == "reservation":
        category = "reservation"
    else:
        category = await classify(user_message)

    if category == "other":
        messages.append({"role": "assistant", "content": OUT_OF_SCOPE_REPLY})
        await _persist(user_key, session_id, messages, category, ctx.pending)
        return {"reply": OUT_OF_SCOPE_REPLY, "trace": messages[start:]}

    if category == "greeting":
        reply = await llm.chat(
            [
                {"role": "system", "content": GREETING_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
        content = (reply.get("content") or "").strip() or GREETING_FALLBACK
        messages.append({"role": "assistant", "content": content})
        await _persist(user_key, session_id, messages, category, ctx.pending)
        return {"reply": content, "trace": messages[start:]}

    # 3) Tool loop
    empty_retries = 0

    for _ in range(MAX_STEPS):
        reply = await llm.chat(messages, tools=TOOL_SCHEMAS)
        tool_calls = reply.get("tool_calls")
        content = (reply.get("content") or "").strip()
        new_hold = ctx.pending is not None and ctx.pending is not pending_before

        if not tool_calls and not content and not new_hold:
            if empty_retries < EMPTY_RETRIES:
                empty_retries += 1
                logger.warning("Empty model reply, nudging (%d)", empty_retries)
                messages.append({"role": "user", "content": NUDGE})
                continue
            break

        if not tool_calls and new_hold:
            # Fixed text, written by code: the user must always be asked.
            p = ctx.pending
            content = (content + "\n\n" if content else "") + (
                f"Seats {', '.join(p['seat_numbers'])} are held for you until "
                f"{p['held_until_local']} (total Rs. {p['total_price']:.0f}). "
                f"Reply \"yes\" to confirm the booking or \"cancel\" to release them."
            )

        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            await _persist(user_key, session_id, messages, "reservation", ctx.pending)
            return {"reply": content, "trace": messages[start:]}

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = await execute_tool(name, args, ctx)
            messages.append(
                {"role": "tool", "tool_name": name, "content": json.dumps(result)}
            )

    messages.append({"role": "assistant", "content": FALLBACK})
    await _persist(user_key, session_id, messages, "reservation", ctx.pending)
    return {"reply": FALLBACK, "trace": messages[start:]}