import json
import logging
from datetime import date

from . import llm
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


async def run_agent(user_message: str, token: str | None = None) -> dict:
    ctx = ToolContext(token=token)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + f" Today's date is {date.today().isoformat()}.",
        },
        {"role": "user", "content": user_message},
    ]

    category = await classify(user_message)

    if category == "other":
        return {"reply": OUT_OF_SCOPE_REPLY, "trace": messages}

    if category == "greeting":
        greet_messages = [
            {"role": "system", "content": GREETING_PROMPT},
            {"role": "user", "content": user_message},
        ]
        reply = await llm.chat(greet_messages)  # no tools offered at all
        content = (reply.get("content") or "").strip() or GREETING_FALLBACK
        return {"reply": content, "trace": greet_messages}

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
            return {"reply": FALLBACK, "trace": messages}

        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            return {"reply": content, "trace": messages}

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = await execute_tool(name, args, ctx)
            messages.append(
                {"role": "tool", "tool_name": name, "content": json.dumps(result)}
            )

    return {"reply": FALLBACK, "trace": messages}