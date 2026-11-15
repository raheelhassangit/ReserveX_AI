from fastapi import FastAPI
from pydantic import BaseModel

from . import llm

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
    reply = await llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body.message},
        ]
    )
    return {"reply": reply["content"]}