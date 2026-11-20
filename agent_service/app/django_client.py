import httpx

from .config import settings


def _error_message(response: httpx.Response) -> str:
    status = response.status_code
    if status == 401:
        return "The user's session is invalid or expired. Ask them to log in again."
    if status == 403:
        return "This action is only available to logged-in customer accounts."
    if status == 429:
        return "Too many attempts. Ask the user to wait a minute and try again."
    if status >= 500:
        return "The booking system had a problem. Ask the user to try again shortly."
    try:
        body = response.json()
    except ValueError:
        return f"Request failed ({status})."
    if isinstance(body, list):  # DRF ValidationError("...") comes back as a list
        return " ".join(str(x) for x in body)
    if isinstance(body, dict):
        if "detail" in body:
            return str(body["detail"])
        return "; ".join(f"{k}: {v}" for k, v in body.items())
    return str(body)


async def _request(method: str, path: str, token: str | None = None,
                   params: dict | None = None, json: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(base_url=settings.django_api_url, timeout=15) as client:
        response = await client.request(method, path, headers=headers, params=params, json=json)
    if response.is_success:
        return response.json()
    return {"error": _error_message(response)}


async def django_get(path: str, token: str | None = None, params: dict | None = None) -> dict:
    return await _request("GET", path, token, params=params)


async def django_post(path: str, token: str | None = None, json: dict | None = None) -> dict:
    return await _request("POST", path, token, json=json)


async def get_me(token: str) -> dict:
    return await django_get("/api/accounts/me/", token)