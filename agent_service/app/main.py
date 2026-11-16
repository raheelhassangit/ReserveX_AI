import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import llm
from .agent import SYSTEM_PROMPT, run_agent

logger = logging.getLogger("agent")

app = FastAPI(title="ReserveX Agent Service")


class ChatIn(BaseModel):
    message: str


@app.exception_handler(httpx.HTTPStatusError)
async def llm_status_error(request: Request, exc: httpx.HTTPStatusError):
    logger.error("LLM returned %s: %s", exc.response.status_code, exc.response.text)
    return JSONResponse(
        status_code=502,
        content={"detail": f"LLM error {exc.response.status_code}: {exc.response.text[:200]}"},
    )


@app.exception_handler(httpx.RequestError)
async def llm_unreachable(request: Request, exc: httpx.RequestError):
    logger.error("LLM unreachable: %r", exc)
    return JSONResponse(status_code=502, content={"detail": f"LLM unreachable: {exc!r}"})


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
async def agent(body: ChatIn):
    return await run_agent(body.message)