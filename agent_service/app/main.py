import logging
import re
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import llm, session
from .agent import SYSTEM_PROMPT, run_agent
from .django_client import get_me

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

app = FastAPI(title="ReserveX Agent Service")


class ChatIn(BaseModel):
    message: str


class AgentIn(BaseModel):
    message: str
    session_id: str | None = None
    action: Literal["confirm", "cancel"] | None = None


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


@app.post("/chat")
async def chat(body: ChatIn):
    reply = await llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body.message},
        ]
    )
    return {"reply": reply["content"]}


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

    result = await run_agent(body.message, token, user_key, session_id, body.action)
    return {"session_id": session_id, **result}