import logging
import re
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from . import session
from .agent import graph
from .django_client import get_me

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

app = FastAPI(title="ReserveX Agent Service")


class AgentIn(BaseModel):
    message: str
    session_id: str | None = None  # used from step 4 (memory)
    action: Literal["confirm", "cancel"] | None = None  # used from step 3 (approval)


@app.exception_handler(httpx.HTTPStatusError)
async def upstream_status_error(request: Request, exc: httpx.HTTPStatusError):
    logger.error("Upstream returned %s: %s", exc.response.status_code, exc.response.text)
    return JSONResponse(
        status_code=502,
        content={"detail": f"Upstream error {exc.response.status_code}: {exc.response.text[:200]}"},
    )


@app.exception_handler(httpx.RequestError)
async def upstream_unreachable(request: Request, exc: httpx.RequestError):
    logger.error("Upstream unreachable: %r", exc)
    return JSONResponse(status_code=502, content={"detail": f"Upstream unreachable: {exc!r}"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/agent")
async def agent(body: AgentIn, authorization: str | None = Header(default=None)):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    user_key = "anon"
    if token:
        me = await get_me(token)
        if "error" in me:
            raise HTTPException(status_code=401, detail=me["error"])
        user_key = str(me.get("id") or me.get("username") or me.get("email") or "")
        if not user_key:
            raise HTTPException(status_code=502, detail="Could not identify the user")

    session_id = body.session_id
    if not session_id or not re.fullmatch(r"[0-9a-f]{32}", session_id):
        session_id = session.new_session_id()

    logger.info("agent call user=%s session=%s", user_key, session_id)

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config={"configurable": {"token": token}, "recursion_limit": 12},
        )
    except GraphRecursionError:
        return {
            "session_id": session_id,
            "reply": "Sorry, I couldn't complete that request.",
            "trace": [],
        }

    messages = result["messages"]
    reply = (messages[-1].content or "").strip() or (
        "Sorry, I couldn't process that. Could you rephrase your request?"
    )
    trace = [
        {"type": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", None)}
        for m in messages
    ]
    return {"session_id": session_id, "reply": reply, "trace": trace}