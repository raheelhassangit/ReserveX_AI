import json
import logging

from . import llm

logger = logging.getLogger("agent")

SCOPE_PROMPT = """Classify the user's message for a reservation app that only handles bus trips and appointments.
Categories:
- reservation: searching bus trips or appointments, their prices and availability, holding seats, bookings, cancellations, waitlists, "my bookings".
- greeting: hello, hi, hey, thanks, goodbye, "who are you", "what can you do".
- other: everything else, including flights, airlines, hotels, trains, taxis, weather, poems, coding, news, general knowledge. A price or "cheapest" question about anything that is not a bus trip or appointment is other.

Examples:
"Hey" -> greeting
"thanks!" -> greeting
"what can you do?" -> greeting
"find a bus to Islamabad" -> reservation
"cancel my booking" -> reservation
"is seat 5 free?" -> reservation
"cheapest flight to Karachi" -> other
"Find me the cheapest Karachi flight" -> other
"book a hotel in Murree" -> other
"write a poem" -> other
"who won the match?" -> other

Answer only with JSON."""

SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["reservation", "greeting", "other"]}
    },
    "required": ["category"],
}

OUT_OF_SCOPE_REPLY = (
    "I can only help with ReserveX reservations: searching bus trips and "
    "appointments, holding seats, and managing your bookings."
)


async def classify(message: str) -> str:
    try:
        reply = await llm.chat(
            [
                {"role": "system", "content": SCOPE_PROMPT},
                {"role": "user", "content": message},
            ],
            format=SCOPE_SCHEMA,
            temperature=0,
        )
        category = json.loads(reply["content"])["category"]
        logger.info("Scope category: %s", category)
        return category
    except Exception:
        logger.warning("Classification failed, defaulting to reservation", exc_info=True)
        return "reservation"
    
DECISION_PROMPT = """The user was just asked to confirm a seat booking. Classify their reply.
- confirm: a clear yes ("yes", "confirm", "go ahead", "book it", "ok do it")
- cancel: a clear no or change of mind ("no", "cancel", "never mind", "release them")
- other: anything else (questions, unrelated, unclear, "maybe", "which seats?")

Answer only with JSON."""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["confirm", "cancel", "other"]}},
    "required": ["decision"],
}


async def classify_decision(message: str) -> str:
    try:
        reply = await llm.chat(
            [
                {"role": "system", "content": DECISION_PROMPT},
                {"role": "user", "content": message},
            ],
            format=DECISION_SCHEMA,
            temperature=0,
        )
        decision = json.loads(reply["content"])["decision"]
        logger.info("Decision: %s", decision)
        return decision
    except Exception:
        logger.warning("Decision check failed", exc_info=True)
        return "other"  # when unsure, never confirm    