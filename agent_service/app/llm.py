import httpx

from .config import settings


async def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": settings.ollama_num_ctx},
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()["message"]