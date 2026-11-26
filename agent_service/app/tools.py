import logging
from contextvars import ContextVar
from datetime import date as date_cls
from datetime import datetime, timezone
from typing import Annotated
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from .config import settings
from .django_client import django_get, django_post

logger = logging.getLogger("agent")

# The user's JWT for the current request. A ContextVar holds one value per
# request, so concurrent users never see each other's token, and it never
# enters the graph state or the checkpoints.
current_token: ContextVar[str | None] = ContextVar("current_token", default=None)


def _token() -> str | None:
    return current_token.get()


def _local(iso: str) -> str:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.astimezone(ZoneInfo(settings.app_timezone)).strftime("%a %d %b %Y, %I:%M %p")


def pending_active(pending: dict | None) -> bool:
    """True if a hold exists and hasn't expired yet."""
    if not pending:
        return False
    until = datetime.fromisoformat(pending["held_until"].replace("Z", "+00:00"))
    return datetime.now(timezone.utc) < until


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
    token = _token()
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
async def list_available_seats(offering_id: int) -> dict:
    """List the seat numbers that are free on one offering.

    Args:
        offering_id: The offering's id from a search result.
    """
    data = await django_get(f"/api/offerings/{int(offering_id)}/", _token())
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


@tool(parse_docstring=True, response_format="content_and_artifact")
async def hold_seats(
    offering_id: int,
    state: Annotated[dict, InjectedState],
    seat_numbers: list[str] | None = None,
    seat_count: int | None = None,
) -> tuple[dict, dict | None]:
    """Temporarily hold seats for the logged-in user on one offering.

    This does NOT confirm a booking: the system asks the user to confirm afterwards.
    Give seat_numbers if the user named seats, or seat_count if they only said how many.

    Args:
        offering_id: The offering's id from a search result.
        seat_numbers: Only when the user names specific seats by number, e.g. "seats 5 and 6" gives ["5", "6"]. Never use this for "hold 2 seats".
        seat_count: How many seats the user wants, e.g. "hold 1 seat" gives 1 and "hold 2 seats" gives 2. Use this when no seat numbers were named.
    """
    token = _token()
    if not token:
        return {"error": "The user is not logged in. Ask them to log in on the website first."}, None
    if pending_active(state.get("pending")):
        return {"error": "A hold is already waiting for the user's confirmation."}, None
    if bool(seat_numbers) == bool(seat_count):
        return {"error": "Provide either seat_numbers or seat_count (exactly one)."}, None

    detail = await django_get(f"/api/offerings/{int(offering_id)}/", token)
    if "error" in detail:
        return detail, None

    free = {str(s["seat_number"]): s["id"] for s in detail["seats"] if s["status"] == "available"}

    if seat_numbers:
        chosen = list(dict.fromkeys(str(n) for n in seat_numbers))
        missing = [n for n in chosen if n not in free]
        if missing:
            return {
                "error": f"Seats not available: {', '.join(missing)}",
                "available_seat_numbers": list(free),
            }, None
    else:
        if int(seat_count) > len(free):
            return {"error": f"Only {len(free)} seats are available."}, None
        chosen = list(free)[: int(seat_count)]

    if len(chosen) > detail["max_seats_per_booking"]:
        return {"error": f"At most {detail['max_seats_per_booking']} seats per booking."}, None

    result = await django_post(
        "/api/bookings/hold/",
        token,
        {"offering": detail["id"], "seat_ids": [free[n] for n in chosen]},
    )
    if "error" in result:
        return result, None

    total = float(detail["price"]) * len(chosen)
    held_local = _local(result["held_until"])

    # The model sees only this short content...
    content = {
        "status": "held",
        "offering": detail["name"],
        "seats": chosen,
        "total_price": total,
        "held_until": held_local,
        "note": "Seats are held but NOT booked yet. The system will ask the user to confirm.",
    }
    # ...while the graph keeps this artifact (ids, exact expiry) for the approval step.
    artifact = {
        "offering_id": detail["id"],
        "offering_name": detail["name"],
        "seat_ids": [free[n] for n in chosen],
        "seat_numbers": chosen,
        "held_until": result["held_until"],
        "held_until_local": held_local,
        "total_price": total,
    }
    logger.info("TOOL hold_seats -> %s", content)
    return content, artifact


TOOLS = [search_offerings, list_available_seats, hold_seats]