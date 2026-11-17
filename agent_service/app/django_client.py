import httpx

from .config import settings


async def django_get(path: str, token: str | None = None, params: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(
        base_url=settings.django_api_url, timeout=15
    ) as client:
        response = await client.get(path, headers=headers, params=params)

    if response.status_code == 401:
        return {"error": "The user's session is invalid or expired. Ask them to log in again."}
    response.raise_for_status()
    return response.json()