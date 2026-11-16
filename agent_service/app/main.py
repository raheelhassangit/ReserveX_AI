import logging

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import llm

logger = logging.getLogger("agent")

app = FastAPI(title="ReserveX Agent Service")

SYSTEM_PROMPT = (
    "You are the ReserveX booking assistant. "
    "Help users find offerings, hold seats and manage reservations."
)


class ChatIn(BaseModel):
    message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(body: ChatIn):
    try:
        reply = await llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": body.message},
            ]
        )
    except httpx.HTTPStatusError as e:
        logger.error("LLM returned %s: %s", e.response.status_code, e.response.text)
        raise HTTPException(
            status_code=502,
            detail=f"LLM error {e.response.status_code}: {e.response.text[:200]}",
        )
    except httpx.RequestError as e:
        logger.error("LLM unreachable: %r", e)
        raise HTTPException(status_code=502, detail=f"LLM unreachable: {e!r}")
    return {"reply": reply["content"]}