from dataclasses import dataclass
from datetime import date as date_cls

from .django_client import django_get


@dataclass
class ToolContext:
    token: str | None = None


def _shape(o: dict) -> dict:
    return {
        "id": o["id"],
        "name": o["name"],
        "organization": o["organization_name"],
        "origin": o["origin"],
        "destination": o["destination"],
        "starts_at": o["starts_at"],
        "price": float(o["price"]) if o.get("price") is not None else None,
        "available_seats": o["available_seats"],
        "max_seats_per_booking": o["max_seats_per_booking"],
    }


async def search_offerings(
    ctx: ToolContext,
    origin: str | None = None,
    destination: str | None = None,
    max_price: float | None = None,
    date: str | None = None,
    sort_by: str | None = None,
    query: str | None = None,
) -> dict:
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

    data = await django_get("/api/offerings/", ctx.token, params=exact_params or params)
    if "error" in data:
        return data

    offerings = [_shape(o) for o in data.get("results", [])]
    if offerings or not date:
        return {"count": data.get("count", len(offerings)), "offerings": offerings}

    # Nothing on that exact day: widen to all upcoming dates and say so.
    data = await django_get("/api/offerings/", ctx.token, params=params)
    if "error" in data:
        return data
    return {
        "count": 0,
        "note": f"Nothing is available on {date}. 'other_dates' lists upcoming trips on other days.",
        "other_dates": [_shape(o) for o in data.get("results", [])],
    }

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_offerings",
            "description": (
                "Search bookable offerings (bus trips, appointments) that still have "
                "free seats. Use it for any question about availability, prices or times. "
                "All parameters are optional."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city, e.g. Lahore"},
                    "destination": {"type": "string", "description": "Arrival city, e.g. Islamabad"},
                    "max_price": {"type": "number", "description": "Maximum price per seat"},
                    "date": {
                        "type": "string",
                        "description": (
                            "Only set this if the user names a specific day "
                            "(like 'tomorrow' or 'on Friday'), as YYYY-MM-DD. "
                            "If the user gave no day, leave it out."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Free-text match on offering or company name",
                    },
                },
                "required": [],
            },
        },
    }
]

TOOL_FUNCTIONS = {"search_offerings": search_offerings}