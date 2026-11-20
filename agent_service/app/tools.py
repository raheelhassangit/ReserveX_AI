from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import settings
from .django_client import django_get, django_post


@dataclass
class ToolContext:
    token: str | None = None
    pending: dict | None = None  # a hold waiting for the user's confirmation


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

    data = await django_get("/api/offerings/", ctx.token, params=params)
    if "error" in data:
        return data
    return {
        "count": 0,
        "note": f"Nothing is available on {date}. 'other_dates' lists upcoming trips on other days.",
        "other_dates": [_shape(o) for o in data.get("results", [])],
    }


async def list_available_seats(ctx: ToolContext, offering_id: int) -> dict:
    data = await django_get(f"/api/offerings/{int(offering_id)}/", ctx.token)
    if "error" in data:
        return data
    free = [s["seat_number"] for s in data["seats"] if s["status"] == "available"]
    return {
        "offering": data["name"],
        "available_seat_numbers": free,
        "max_seats_per_booking": data["max_seats_per_booking"],
    }


async def hold_seats(
    ctx: ToolContext,
    offering_id: int,
    seat_numbers: list[str] | None = None,
    seat_count: int | None = None,
) -> dict:
    if not ctx.token:
        return {"error": "The user is not logged in. Ask them to log in on the website first."}
    if ctx.pending:
        return {"error": "A hold is already waiting for the user's confirmation. Ask them to confirm or cancel it first."}
    if bool(seat_numbers) == bool(seat_count):
        return {"error": "Provide either seat_numbers or seat_count (exactly one)."}

    detail = await django_get(f"/api/offerings/{int(offering_id)}/", ctx.token)
    if "error" in detail:
        return detail

    free = {str(s["seat_number"]): s["id"] for s in detail["seats"] if s["status"] == "available"}

    if seat_numbers:
        chosen = list(dict.fromkeys(str(n) for n in seat_numbers))
        missing = [n for n in chosen if n not in free]
        if missing:
            return {
                "error": f"Seats not available: {', '.join(missing)}",
                "available_seat_numbers": list(free),
            }
    else:
        if int(seat_count) > len(free):
            return {"error": f"Only {len(free)} seats are available."}
        chosen = list(free)[: int(seat_count)]

    if len(chosen) > detail["max_seats_per_booking"]:
        return {"error": f"At most {detail['max_seats_per_booking']} seats per booking."}

    result = await django_post(
        "/api/bookings/hold/",
        ctx.token,
        {"offering": detail["id"], "seat_ids": [free[n] for n in chosen]},
    )
    if "error" in result:
        return result

    total = float(detail["price"]) * len(chosen)
    ctx.pending = {
        "offering_id": detail["id"],
        "offering_name": detail["name"],
        "seat_ids": [free[n] for n in chosen],
        "seat_numbers": chosen,
        "held_until": result["held_until"],
        "held_until_local": _local(result["held_until"]),
        "total_price": total,
    }
    return {
        "status": "held",
        "offering": detail["name"],
        "seats": chosen,
        "total_price": total,
        "held_until": ctx.pending["held_until_local"],
        "note": "Seats are held but NOT booked yet. The system will ask the user to confirm.",
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
                    "sort_by": {
                        "type": "string",
                        "enum": ["price", "starts_at"],
                        "description": "price = cheapest first, starts_at = earliest first",
                    },
                    "query": {
                        "type": "string",
                        "description": "Free-text match on offering or company name",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_seats",
            "description": "List the seat numbers that are free on one offering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offering_id": {"type": "integer", "description": "The offering's id from a search result"}
                },
                "required": ["offering_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hold_seats",
            "description": (
                "Temporarily hold seats for the logged-in user on one offering. "
                "This does NOT confirm a booking. Give seat_numbers if the user named "
                "seats, or seat_count if they only said how many."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "offering_id": {"type": "integer", "description": "The offering's id from a search result"},
                    "seat_numbers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific seat numbers the user asked for, e.g. ['5','6']",
                    },
                    "seat_count": {"type": "integer", "description": "How many seats, if no numbers were named"},
                },
                "required": ["offering_id"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_offerings": search_offerings,
    "list_available_seats": list_available_seats,
    "hold_seats": hold_seats,
}