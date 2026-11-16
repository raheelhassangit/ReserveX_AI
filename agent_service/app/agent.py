import json
import logging

from . import llm
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

logger = logging.getLogger("agent")

MAX_STEPS = 6
EMPTY_RETRIES = 2

SYSTEM_PROMPT = (
    "You are the ReserveX booking assistant. ReserveX handles reservations "
    "for bus trips and appointments only. "
    "Always use tools to check real availability and prices; never guess. "
    "Do not recommend offerings with 0 seats left. "
    "If a request is outside ReserveX (for example flights or hotels), "
    "say so politely and explain what you can help with. "
    "Keep answers short."
)

NUDGE = (
    "Your last reply was empty. Either call a tool, or answer the user in "
    "plain text. If the request is outside ReserveX, say so politely."
)

FALLBACK = "Sorry, I couldn't process that. Could you rephrase your request?"


async def execute_tool(name: str, args: dict) -> dict:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return await func(**args)
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return {"error": f"Tool failed: {e}"}


async def run_agent(user_message: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    empty_retries = 0

    for _ in range(MAX_STEPS):
        reply = await llm.chat(messages, tools=TOOL_SCHEMAS)
        tool_calls = reply.get("tool_calls")
        content = (reply.get("content") or "").strip()

        # Case 1: model returned nothing at all
        if not tool_calls and not content:
            if empty_retries < EMPTY_RETRIES:
                empty_retries += 1
                logger.warning("Empty model reply, nudging (%d)", empty_retries)
                messages.append({"role": "user", "content": NUDGE})
                continue
            return {"reply": FALLBACK, "trace": messages}

        # Normalise the assistant message so the history is always well-formed
        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # Case 2: plain text answer, we're done
        if not tool_calls:
            return {"reply": content, "trace": messages}

        # Case 3: tool calls, run them and loop again
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = await execute_tool(name, args)
            messages.append(
                {"role": "tool", "tool_name": name, "content": json.dumps(result)}
            )

    return {"reply": FALLBACK, "trace": messages}