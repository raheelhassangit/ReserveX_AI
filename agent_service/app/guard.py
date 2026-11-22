import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .llm import llm

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
"who won the match?" -> other"""

OUT_OF_SCOPE_REPLY = (
    "I can only help with ReserveX reservations: searching bus trips and "
    "appointments, holding seats, and managing your bookings."
)


class Scope(BaseModel):
    category: Literal["reservation", "greeting", "other"]


_classifier = llm.with_structured_output(Scope)


async def classify(message: str) -> str:
    try:
        result = await _classifier.ainvoke(
            [SystemMessage(content=SCOPE_PROMPT), HumanMessage(content=message)]
        )
        if result is None:
            return "reservation"
        logger.info("Scope category: %s", result.category)
        return result.category
    except Exception:
        logger.warning("Classification failed, defaulting to reservation", exc_info=True)
        return "reservation"