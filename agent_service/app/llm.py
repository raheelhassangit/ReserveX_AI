import httpx

from .config import settings


async def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    format: dict | str | None = None,
    temperature: float | None = None,
) -> dict:
    options = {"num_ctx": settings.ollama_num_ctx}
    if temperature is not None:
        options["temperature"] = temperature

    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if tools:
        payload["tools"] = tools
    if format:
        payload["format"] = format

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()["message"]