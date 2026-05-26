"""
Unit tests for agent routing and tool isolation.

These prove the two structural claims made in the README:
  1. The approval gate is enforced by graph topology — _approval_router
     routes to write_node only when state["approved"] is True, and to END
     in all other cases. The LLM has no say.
  2. write_research_note is physically absent from the read-tool set bound
     to execute_node. There is no path from execute_node to write_node.

Run from project root:
    pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

import agent
import data_layer
from agent import _READ_TOOLS_BY_NAME, _approval_router, _data_check_router
from models import HPIRecord, StreetSummary, Transaction
from tools import write_research_note


def test_approval_router_approved_routes_to_write():
    """Approved state must route to write_node."""
    assert _approval_router({"approved": True}) == "write"


def test_approval_router_declined_routes_to_end():
    """Declined or missing approval must route to END, never to write_node."""
    assert _approval_router({"approved": False}) == END
    assert _approval_router({}) == END


def test_data_check_router_routes_correctly():
    """Data present routes to analyse; missing data routes to END."""
    assert _data_check_router({"data_ok": True}) == "analyse"
    assert _data_check_router({"data_ok": False}) == END
    assert _data_check_router({}) == "analyse"  # default — safe to proceed if flag absent


class _ScriptedLLM:
    """Stand-in for ChatOpenAI used to drive execute_node's tool loop.

    Not an AI model — it just returns a pre-scripted list of responses so the
    loop is deterministic and runs with no API key, no network, and no cost.
    bind_tools is a no-op that returns self; invoke walks the script in order.
    """

    def __init__(self, scripted_responses: list[AIMessage]):
        self._responses = scripted_responses
        self._i = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self._responses[self._i]
        self._i += 1
        return response


def test_execute_node_captures_all_payloads(monkeypatch):
    """A scripted LLM that requests all three tools then stops should make the
    loop run them, capture every payload, build a per-tool trace, and set
    data_ok=True. The data layer is stubbed so the real tools return canned
    data with no network call, isolating the agent loop itself.
    """
    monkeypatch.setattr(
        data_layer, "get_transactions",
        lambda *a, **k: [
            Transaction(price=300_000, date="2015-06-01", street="HIGH STREET",
                        postcode="GU1 1AA", property_type="detached")
            for _ in range(20)
        ],
    )
    monkeypatch.setattr(
        data_layer, "get_top_streets",
        lambda *a, **k: [
            StreetSummary(street="HIGH STREET", avg_price=300000.0, transaction_count=20)
        ],
    )
    monkeypatch.setattr(
        data_layer, "get_hpi",
        lambda *a, **k: [
            HPIRecord(period="2016-03", avg_price=266729.0, annual_change=10.3, monthly_change=0.5)
        ],
    )

    fetch_then_stop = [
        AIMessage(content="", tool_calls=[
            {"name": "fetch_area_transactions", "args": {"postcode_district": "GU1"}, "id": "1", "type": "tool_call"},
            {"name": "fetch_top_streets", "args": {"postcode_district": "GU1"}, "id": "2", "type": "tool_call"},
            {"name": "fetch_regional_hpi", "args": {"region": "south-east"}, "id": "3", "type": "tool_call"},
        ]),
        AIMessage(content="All three datasets gathered.", tool_calls=[]),
    ]
    monkeypatch.setattr(agent, "_llm", lambda: _ScriptedLLM(fetch_then_stop))

    state = {"messages": [HumanMessage(content="analyse GU1")], "trace": [], "plan": "test plan"}
    result = agent.execute_node(state)

    assert result["data_ok"] is True
    assert result["transactions"]["count"] == 20
    assert result["top_streets"]["streets"][0]["street"] == "HIGH STREET"
    assert result["hpi_data"]["records"][0]["period"] == "2016-03"
    assert len(result["trace"]) == 3, "expected one trace entry per tool call"


def test_execute_node_aborts_when_llm_fetches_nothing(monkeypatch):
    """If the LLM ends the loop without fetching the required data, execute_node
    must set data_ok=False and record an abort trace entry, rather than passing
    empty data downstream to the analyse node.
    """
    no_tools = [AIMessage(content="I won't fetch anything.", tool_calls=[])]
    monkeypatch.setattr(agent, "_llm", lambda: _ScriptedLLM(no_tools))

    state = {"messages": [HumanMessage(content="analyse GU1")], "trace": [], "plan": "p"}
    result = agent.execute_node(state)

    assert result["data_ok"] is False
    assert result["transactions"] == {}
    assert any("aborting" in e["action"].lower() for e in result["trace"])


def test_write_tool_absent_from_execute_node():
    """write_research_note must not appear in the read-tool dict bound to execute_node.

    If this test ever fails it means the write tool has been accidentally added
    to READ_TOOLS — which would let the LLM call it without user approval.
    """
    assert write_research_note.name not in _READ_TOOLS_BY_NAME, (
        "write_research_note must be bound only on write_node, "
        "never on execute_node"
    )
