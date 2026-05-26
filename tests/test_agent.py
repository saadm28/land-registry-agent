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

from langgraph.graph import END

from agent import _READ_TOOLS_BY_NAME, _approval_router, _data_check_router
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


def test_write_tool_absent_from_execute_node():
    """write_research_note must not appear in the read-tool dict bound to execute_node.

    If this test ever fails it means the write tool has been accidentally added
    to READ_TOOLS — which would let the LLM call it without user approval.
    """
    assert write_research_note.name not in _READ_TOOLS_BY_NAME, (
        "write_research_note must be bound only on write_node, "
        "never on execute_node"
    )
