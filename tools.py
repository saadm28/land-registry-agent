"""
LangGraph tools.

The three read tools are bound to the execute_node only. write_research_note
is bound only inside the write_node, which is itself reachable only via the
conditional edge from the approval_gate. The approval gate cannot be bypassed
by graph topology.

Tools take and return primitive types / JSON strings — that is what the LLM
sees. Underneath, every call routes through the typed data_layer.
"""

from __future__ import annotations

import json
from datetime import datetime

from langchain_core.tools import tool

import data_layer
import mock_sheet


@tool
def fetch_area_transactions(postcode_district: str) -> str:
    """Fetch Price Paid transactions for a UK postcode district (e.g. "GU1").

    Returns a JSON string summarising the transaction set:
      - count
      - date range (min, max)
      - average price
      - up to 3 sample transactions

    Uses the HPI-aligned window (2013-04 to 2016-03) so figures line up with
    the regional HPI series. Sparse-data warnings are logged but not raised.
    """
    transactions = data_layer.get_transactions(postcode_district)
    if not transactions:
        return json.dumps({
            "postcode_district": postcode_district,
            "count": 0,
            "warning": "No transactions returned. Confidence low.",
        })

    prices = [t.price for t in transactions]
    dates = sorted(t.date for t in transactions)
    payload = {
        "postcode_district": postcode_district,
        "count": len(transactions),
        "date_range": {"min": dates[0], "max": dates[-1]},
        "average_price": round(sum(prices) / len(prices), 2),
        "sample": [t.model_dump() for t in transactions[:3]],
        "sparse_data": len(transactions) < 10,
    }
    return json.dumps(payload)


@tool
def fetch_top_streets(postcode_district: str, top_n: int = 5) -> str:
    """Rank streets by average sale price within a postcode district.

    Returns a JSON string with a list of {street, avg_price, transaction_count}.
    Aggregation runs in Python — never in SPARQL — to avoid endpoint 503s.
    """
    streets = data_layer.get_top_streets(postcode_district, top_n=top_n)
    payload = {
        "postcode_district": postcode_district,
        "top_n": top_n,
        "streets": [s.model_dump() for s in streets],
    }
    return json.dumps(payload)


@tool
def fetch_regional_hpi(region: str) -> str:
    """Fetch House Price Index records for a region (e.g. "south-east").

    Returns a JSON string with up to 36 monthly records plus an explicit note
    about the 2016-03 data ceiling so the LLM does not assume current-day data.
    """
    records = data_layer.get_hpi(region)
    payload = {
        "region": region,
        "count": len(records),
        "data_ceiling_note": (
            "HPI endpoint data currently ceilings at 2016-03. Records are the "
            "most recent available, not current-day."
        ),
        "records": [r.model_dump() for r in records],
    }
    return json.dumps(payload)


@tool
def write_research_note(note: str, area: str) -> str:
    """Write the research note to the user's tracking sheet (mock).

    DANGER: Only call after explicit user approval has been confirmed.
    In this agent, approval is enforced by graph topology — this tool is only
    bound on the write_node, which is unreachable except via the approval_gate
    conditional edge. Do not bind this tool on any other node.
    """
    return mock_sheet.write({
        "area": area,
        "note": note,
        "generated_at": datetime.now().isoformat(),
    })


# Grouped for convenient binding in the agent layer.
READ_TOOLS = [fetch_area_transactions, fetch_top_streets, fetch_regional_hpi]
WRITE_TOOLS = [write_research_note]
