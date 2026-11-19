import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from . import llm, session
from .config import settings
from .guard import OUT_OF_SCOPE_REPLY, classify
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, ToolContext

logger = logging.getLogger("agent")

MAX_STEPS = 6
EMPTY_RETRIES = 2

SYSTEM_PROMPT = (
    "You are the ReserveX booking assistant. "
    "You can ONLY do what your tools allow. Right now that is: search offerings. "
    "If the user asks for anything else (holding seats, cancelling, viewing bookings), "
    "say that you can't do that yet and offer to search offerings instead. "
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


def _today() -> str:
    return datetime.now(ZoneInfo(settings.app_timezone)).date().isoformat()


async def _persist(user_key: str, session_id: str, messages: list[dict], category: str) -> None:
    history = [m for m in messages[1:] if m.get("content") != NUDGE]
    await session.save(user_key, session_id, history, category)


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


async def run_agent(
    user_message: str, token: str | None, user_key: str, session_id: str
) -> dict:
    ctx = ToolContext(token=token)
    state = await session.load(user_key, session_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f" Today's date is {_today()}."},
        *state["messages"],
        {"role": "user", "content": user_message},
    ]
    start = len(messages) - 1  # this turn begins at the new user message

    # Replies like "yes" or "the first one" have no keywords, so once a
    # conversation is in the reservation flow we stay in it.
    if state["last_category"] == "reservation":
        category = "reservation"
    else:
        category = await classify(user_message)

    if category == "other":
        messages.append({"role": "assistant", "content": OUT_OF_SCOPE_REPLY})
        await _persist(user_key, session_id, messages, category)
        return {"reply": OUT_OF_SCOPE_REPLY, "trace": messages[start:]}

    if category == "greeting":
        reply = await llm.chat(
            [
                {"role": "system", "content": GREETING_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )  # no tools offered at all
        content = (reply.get("content") or "").strip() or GREETING_FALLBACK
        messages.append({"role": "assistant", "content": content})
        await _persist(user_key, session_id, messages, category)
        return {"reply": content, "trace": messages[start:]}

    empty_retries = 0

    for _ in range(MAX_STEPS):
        reply = await llm.chat(messages, tools=TOOL_SCHEMAS)
        tool_calls = reply.get("tool_calls")
        content = (reply.get("content") or "").strip()

        if not tool_calls and not content:
            if empty_retries < EMPTY_RETRIES:
                empty_retries += 1
                logger.warning("Empty model reply, nudging (%d)", empty_retries)
                messages.append({"role": "user", "content": NUDGE})
                continue
            break

        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            await _persist(user_key, session_id, messages, "reservation")
            return {"reply": content, "trace": messages[start:]}

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = await execute_tool(name, args, ctx)
            messages.append(
                {"role": "tool", "tool_name": name, "content": json.dumps(result)}
            )

    messages.append({"role": "assistant", "content": FALLBACK})
    await _persist(user_key, session_id, messages, "reservation")
    return {"reply": FALLBACK, "trace": messages[start:]}