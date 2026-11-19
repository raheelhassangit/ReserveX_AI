import json
import uuid

import redis.asyncio as redis

from .config import settings

_redis = redis.from_url(settings.redis_url, decode_responses=True)

SESSION_TTL_SECONDS = 6 * 60 * 60
MAX_MESSAGES = 24


def new_session_id() -> str:
    return uuid.uuid4().hex


def _key(user_key: str, session_id: str) -> str:
    return f"agent:session:{user_key}:{session_id}"


async def load(user_key: str, session_id: str) -> dict:
    raw = await _redis.get(_key(user_key, session_id))
    if not raw:
        return {"messages": [], "last_category": None}
    return json.loads(raw)


def _trim(messages: list[dict]) -> list[dict]:
    messages = messages[-MAX_MESSAGES:]
    # never start in the middle of a tool exchange
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]
    return messages


async def save(user_key: str, session_id: str, messages: list[dict], last_category: str) -> None:
    data = {"messages": _trim(messages), "last_category": last_category}
    await _redis.set(_key(user_key, session_id), json.dumps(data), ex=SESSION_TTL_SECONDS)