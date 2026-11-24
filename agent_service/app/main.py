import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from .agent import FALLBACK, build_graph
from .config import settings
from .django_client import get_me
from .tools import current_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: open the pool, create the tables, build the graph.
    async with AsyncConnectionPool(
        settings.agent_db_url,
        max_size=5,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        app.state.graph = build_graph(saver)
        yield
    # leaving the "async with" closes the pool on shutdown


app = FastAPI(title="ReserveX Agent Service", lifespan=lifespan)


class AgentIn(BaseModel):
    message: str
    session_id: str | None = None
    action: Literal["confirm", "cancel"] | None = None  # for a future Confirm button


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
async def agent(
    body: AgentIn,
    request: Request,
    authorization: str | None = Header(default=None),
):
    graph = request.app.state.graph

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
        session_id = uuid.uuid4().hex

    # One graph thread per user+session: another user can't resume this one.
    config = {
        "configurable": {"thread_id": f"{user_key}:{session_id}"},
        "recursion_limit": 12,
    }
    current_token.set(token)  # visible to the tools in this request only

    snapshot = await graph.aget_state(config)
    prior = (snapshot.values or {}).get("messages", [])
    seen_ids = {m.id for m in prior}
    paused = bool(snapshot.next)  # a hold is waiting for the user's answer

    if paused:
        graph_input = Command(resume={"action": body.action, "text": body.message})
    elif body.action:
        return {
            "session_id": session_id,
            "reply": "There is no active hold. It may have expired. Search again to start a new booking.",
            "awaiting_confirmation": False,
            "trace": [],
        }
    else:
        graph_input = {"messages": [HumanMessage(content=body.message)]}

    logger.info("agent call user=%s session=%s paused=%s", user_key, session_id, paused)

    try:
        result = await graph.ainvoke(graph_input, config=config)
    except GraphRecursionError:
        return {
            "session_id": session_id,
            "reply": "Sorry, I couldn't complete that request.",
            "awaiting_confirmation": False,
            "trace": [],
        }

    messages = result["messages"]
    reply = (messages[-1].content or "").strip()

    # Graph paused again: add the fixed confirmation prompt after the model's text.
    interrupts = result.get("__interrupt__")
    if interrupts:
        prompt = interrupts[0].value["prompt"]
        reply = f"{reply}\n\n{prompt}" if reply else prompt
    reply = reply or FALLBACK

    trace = [
        {"type": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", None)}
        for m in messages
        if m.id not in seen_ids
    ]
    return {
        "session_id": session_id,
        "reply": reply,
        "awaiting_confirmation": bool(interrupts),
        "trace": trace,
    }