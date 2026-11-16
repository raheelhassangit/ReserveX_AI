FAKE_OFFERINGS = [
    {"id": 1, "title": "Daewoo Express Lahore to Islamabad 9:00 AM", "price": 1800, "seats_left": 12},
    {"id": 2, "title": "Faisal Movers Lahore to Islamabad 11:30 AM", "price": 2200, "seats_left": 4},
    {"id": 3, "title": "Skyways Lahore to Islamabad 6:00 PM", "price": 1500, "seats_left": 0},
]


async def search_offerings(query: str, max_price: float | None = None) -> dict:
    results = [
        o for o in FAKE_OFFERINGS
        if all(word in o["title"].lower() for word in query.lower().split())
        and (max_price is None or o["price"] <= max_price)
    ]
    return {"count": len(results), "offerings": results}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_offerings",
            "description": (
                "Search bookable offerings (bus trips, appointments) by keywords. "
                "Use this before answering any question about availability or prices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords, e.g. 'Lahore Islamabad'",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price per seat, optional",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_FUNCTIONS = {"search_offerings": search_offerings}