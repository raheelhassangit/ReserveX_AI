import json
import logging

from . import llm
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

logger = logging.getLogger("agent")

MAX_STEPS = 5

SYSTEM_PROMPT = (
    "You are the ReserveX booking assistant. "
    "Always use tools to check real availability and prices; never guess. "
    "Keep answers short."
)


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

    for _ in range(MAX_STEPS):
        reply = await llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(reply)

        tool_calls = reply.get("tool_calls")
        if not tool_calls:
            return {"reply": reply["content"], "trace": messages}

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = await execute_tool(name, args)
            messages.append(
                {"role": "tool", "tool_name": name, "content": json.dumps(result)}
            )

    return {"reply": "Sorry, I couldn't complete that request.", "trace": messages}