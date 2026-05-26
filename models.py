"""
Pydantic models and LangGraph state types.

Every cross-layer data structure is a typed model. Raw dicts are only permitted
at the LLM-tool JSON boundary inside tools.py.
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    price: int
    date: str
    street: str
    postcode: str
    property_type: str


class StreetSummary(BaseModel):
    street: str
    avg_price: float
    transaction_count: int


class HPIRecord(BaseModel):
    period: str
    avg_price: float | None = None
    annual_change: float | None = None
    monthly_change: float | None = None


class ResearchNote(BaseModel):
    area: str
    summary: str
    generated_at: str


class TraceEntry(BaseModel):
    step: int
    action: str
    result: str
    timestamp: str


class AgentOutput(BaseModel):
    research_note: ResearchNote
    chart_data: dict[str, Any] = Field(default_factory=dict)
    trace: list[TraceEntry] = Field(default_factory=list)
    approved: bool = False
    write_status: str | None = None


class AgentState(TypedDict, total=False):
    """LangGraph state. TypedDict (not BaseModel) is what StateGraph expects."""

    messages: list[Any]
    plan: str
    transactions: dict[str, Any]
    top_streets: dict[str, Any]
    hpi_data: dict[str, Any]
    research_note: str
    chart_data: dict[str, Any]
    trace: list[dict]
    approved: bool
    write_status: str
    auto_approve: bool
    data_ok: bool
