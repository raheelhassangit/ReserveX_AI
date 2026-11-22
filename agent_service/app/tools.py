import logging
from datetime import date as date_cls
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .config import settings
from .django_client import django_get

logger = logging.getLogger("agent")


def _token(config: RunnableConfig) -> str | None:
    # The JWT travels in the run config, never in the messages,
    # so the model can't see it or repeat it.
    return config.get("configurable", {}).get("token")


def _local(iso: str) -> str:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.astimezone(ZoneInfo(settings.app_timezone)).strftime("%a %d %b %Y, %I:%M %p")


def _shape(o: dict) -> dict:
    return {
        "id": o["id"],
        "name": o["name"],
        "organization": o["organization_name"],
        "origin": o["origin"],
        "destination": o["destination"],
        "departs": _local(o["starts_at"]),
        "price": float(o["price"]) if o.get("price") is not None else None,
        "available_seats": o["available_seats"],
        "max_seats_per_booking": o["max_seats_per_booking"],
    }


@tool(parse_docstring=True)
async def search_offerings(
    config: RunnableConfig,
    origin: str | None = None,
    destination: str | None = None,
    max_price: float | None = None,
    date: str | None = None,
    sort_by: str | None = None,
    query: str | None = None,
) -> dict:
    """Search bookable offerings (bus trips, appointments) that still have free seats.

    Use it for any question about availability, prices or times. All parameters are optional.

    Args:
        origin: Departure city, e.g. Lahore.
        destination: Arrival city, e.g. Islamabad.
        max_price: Maximum price per seat.
        date: Only set this if the user names a specific day, as YYYY-MM-DD. Otherwise leave it out.
        sort_by: Either "price" (cheapest first) or "starts_at" (earliest first).
        query: Free-text match on offering or company name.
    """
    token = _token(config)
    params = {"min_seats": 1, "date_from": date_cls.today().isoformat()}

    if origin and origin.strip():
        params["origin"] = origin.strip()
    if destination and destination.strip():
        params["destination"] = destination.strip()
    if max_price is not None:
        params["max_price"] = max_price
    if query and query.strip():
        params["search"] = query.strip()
    if sort_by in ("price", "starts_at"):
        params["ordering"] = sort_by

    exact_params = None
    if date:
        try:
            date_cls.fromisoformat(date)
        except ValueError:
            return {"error": "date must look like 2026-11-20"}
        exact_params = {**params, "date_from": date, "date_to": date}

    data = await django_get("/api/offerings/", token, params=exact_params or params)
    if "error" in data:
        return data

    offerings = [_shape(o) for o in data.get("results", [])]
    if offerings or not date:
        result = {"count": data.get("count", len(offerings)), "offerings": offerings}
    else:
        # Nothing on that exact day: widen to all upcoming dates and say so.
        data = await django_get("/api/offerings/", token, params=params)
        if "error" in data:
            return data
        result = {
            "count": 0,
            "note": f"Nothing is available on {date}. 'other_dates' lists upcoming trips on other days.",
            "other_dates": [_shape(o) for o in data.get("results", [])],
        }

    logger.info("TOOL search_offerings -> %s", str(result)[:300])
    return result


@tool(parse_docstring=True)
async def list_available_seats(config: RunnableConfig, offering_id: int) -> dict:
    """List the seat numbers that are free on one offering.

    Args:
        offering_id: The offering's id from a search result.
    """
    data = await django_get(f"/api/offerings/{int(offering_id)}/", _token(config))
    if "error" in data:
        return data
    free = [s["seat_number"] for s in data["seats"] if s["status"] == "available"]
    result = {
        "offering": data["name"],
        "available_seat_numbers": free,
        "max_seats_per_booking": data["max_seats_per_booking"],
    }
    logger.info("TOOL list_available_seats -> %s", str(result)[:300])
    return result


TOOLS = [search_offerings, list_available_seats]