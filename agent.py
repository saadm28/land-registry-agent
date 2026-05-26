"""
LangGraph agent.

Graph topology (both gates are enforced here, not in tool guards):

    START -> plan -> execute --(data ok?)--> analyse -> approval_gate
                                  |                           |
                             data missing       approved ----- + ----- declined
                                  |                 |                    |
                                 END             write                  END
                                                    |
                                                   END

Design notes:
  - Nodes are explicit (StateGraph, not create_react_agent) because Petex's
    own platform mirrors this shape — plan, execute, interpret, gate, write —
    and the readability is the point of the exercise.
  - The execute_node runs a bounded tool-calling loop with the read tools.
  - The write tool is bound only on the write_node. There is no other path
    to it in the graph.
  - chart_data is built deterministically in Python from the fetched data —
    we do not ask the LLM to echo numbers.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # No display; PNGs only.
import matplotlib.pyplot as plt  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from models import AgentOutput, AgentState, ResearchNote, TraceEntry  # noqa: E402
from tools import READ_TOOLS, write_research_note  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"
EXECUTE_LOOP_MAX_ITERS = 6
CHARTS_DIR = Path("charts")

_READ_TOOLS_BY_NAME = {t.name: t for t in READ_TOOLS}


# ---------------------------------------------------------------------------
# Trace helpers — designed for a non-technical reader.
# ---------------------------------------------------------------------------

def _trace(state: AgentState, action: str, result: str) -> None:
    trace = state.setdefault("trace", [])
    trace.append({
        "step": len(trace) + 1,
        "action": action,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    })


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


# ---------------------------------------------------------------------------
# plan_node — LLM produces a structured plan before any tool call.
# ---------------------------------------------------------------------------

PLAN_SYSTEM = (
    "You are a senior data analyst. Given a user request about UK property "
    "data, produce a short numbered plan (4-6 steps) of how you will fulfil "
    "it using these read tools: fetch_area_transactions, fetch_top_streets, "
    "fetch_regional_hpi. End with a step that says you will pause for "
    "user approval before writing anything. Plain text, no preamble."
)


def plan_node(state: AgentState) -> dict:
    user_msg = state["messages"][-1].content if state.get("messages") else ""
    response = _llm().invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    plan_text = response.content if isinstance(response.content, str) else str(response.content)
    _trace(
        state,
        "Produced analysis plan",
        "LLM produced a numbered plan covering data fetch, analysis, and approval gate.",
    )
    return {"plan": plan_text, "trace": state["trace"]}


# ---------------------------------------------------------------------------
# execute_node — bounded tool loop with read tools only.
# ---------------------------------------------------------------------------

EXECUTE_SYSTEM = (
    "You are gathering data to answer the user. Use the available tools to "
    "fetch: (1) GU1 area transactions, (2) top streets in GU1, (3) regional "
    "HPI for south-east. Call tools in parallel where useful. Stop calling "
    "tools once you have all three datasets. Do not summarise or analyse "
    "yet; that comes later."
)


def _summarise_transactions(payload: dict) -> str:
    if payload.get("count", 0) == 0:
        return "No transactions returned. Sparse-data warning logged."
    dr = payload.get("date_range", {})
    return (
        f"Retrieved {payload['count']} GU1 transactions "
        f"({dr.get('min')} to {dr.get('max')}). "
        f"Average price £{payload['average_price']:,.0f}. "
        f"{'Sparse data; confidence low.' if payload.get('sparse_data') else 'Data cached for session.'}"
    )


def _summarise_top_streets(payload: dict) -> str:
    streets = payload.get("streets", [])
    if not streets:
        return "No streets ranked (no transactions in window)."
    top = streets[0]
    return (
        f"Top street: {top['street']}, average "
        f"£{top['avg_price']:,.0f} across {top['transaction_count']} transactions. "
        f"{len(streets)} streets ranked."
    )


def _summarise_hpi(payload: dict) -> str:
    recs = payload.get("records", [])
    if not recs:
        return "HPI endpoint returned no records."
    latest = recs[0]
    return (
        f"Retrieved {len(recs)} monthly HPI records for {payload['region']}. "
        f"Most recent period: {latest['period']}, average price £{latest['avg_price']}. "
        f"Note: HPI data ceilings at 2016-03; analysis uses most recent available window."
    )


_TOOL_TRACE_SUMMARISERS = {
    "fetch_area_transactions": ("Fetched GU1 property transactions", _summarise_transactions),
    "fetch_top_streets": ("Identified highest-value streets", _summarise_top_streets),
    "fetch_regional_hpi": ("Fetched South East regional house price index", _summarise_hpi),
}


def execute_node(state: AgentState) -> dict:
    llm_with_tools = _llm().bind_tools(READ_TOOLS)
    user_msg = state["messages"][-1].content if state.get("messages") else ""

    messages: list = [
        SystemMessage(content=EXECUTE_SYSTEM),
        HumanMessage(content=f"User request: {user_msg}\n\nAgent plan:\n{state.get('plan', '')}"),
    ]

    transactions_payload: dict | None = None
    top_streets_payload: dict | None = None
    hpi_payload: dict | None = None

    for _ in range(EXECUTE_LOOP_MAX_ITERS):
        ai_msg: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = ai_msg.tool_calls or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = _READ_TOOLS_BY_NAME.get(name)
            if tool_fn is None:
                tool_result = json.dumps({"error": f"Unknown tool {name}"})
            else:
                tool_result = tool_fn.invoke(args)

            messages.append(
                ToolMessage(content=tool_result, tool_call_id=call["id"])
            )

            try:
                parsed = json.loads(tool_result)
            except json.JSONDecodeError:
                parsed = {}

            if name == "fetch_area_transactions":
                transactions_payload = parsed
            elif name == "fetch_top_streets":
                top_streets_payload = parsed
            elif name == "fetch_regional_hpi":
                hpi_payload = parsed

            label, summariser = _TOOL_TRACE_SUMMARISERS.get(
                name, (f"Called {name}", lambda p: json.dumps(p)[:120])
            )
            _trace(state, label, summariser(parsed))

    missing = [
        name for name, payload in [
            ("transactions", transactions_payload),
            ("hpi", hpi_payload),
        ]
        if payload is None
    ]
    if missing:
        _trace(
            state,
            "Data fetch incomplete, aborting",
            f"Required payload(s) missing after execute loop: {', '.join(missing)}. "
            "Cannot analyse on empty data. Check endpoint availability and retry.",
        )
        return {
            "transactions": {},
            "top_streets": {},
            "hpi_data": {},
            "trace": state["trace"],
            "data_ok": False,
        }

    return {
        "transactions": transactions_payload or {},
        "top_streets": top_streets_payload or {},
        "hpi_data": hpi_payload or {},
        "trace": state["trace"],
        "data_ok": True,
    }


def _data_check_router(state: AgentState) -> str:
    """Route to analyse only when all required data payloads are present."""
    return "analyse" if state.get("data_ok", True) else END


# ---------------------------------------------------------------------------
# analyse_node — research note + deterministic chart_data + PNGs.
# ---------------------------------------------------------------------------

ANALYSE_SYSTEM = (
    "You are writing a single-paragraph research note for a property analyst. "
    "Base every figure on the data provided. Cover: (1) GU1 price trend over "
    "the available window, (2) comparison with the South East regional HPI, "
    "(3) the highest-value streets, (4) an explicit note that HPI data "
    "ceilings at 2016-03 so the comparison uses the most recent available "
    "window, not the present day. Be concise: one paragraph, ~120 words. "
    "Do not invent numbers; only use figures present in the provided context. "
    "If the context includes 'sparse_data': true, explicitly state that "
    "confidence is low due to limited transaction volume."
)


def _monthly_gu1_trend() -> list[dict]:
    """Aggregate transactions into a monthly avg price series in Python.

    Re-fetches the full transaction set from data_layer (a cache hit after
    execute_node) because the tool payload only carries a summary, not every row.
    """
    import data_layer

    txns = data_layer.get_transactions("GU1")
    if not txns:
        return []
    by_month: dict[str, list[int]] = defaultdict(list)
    for t in txns:
        month = t.date[:7]
        by_month[month].append(t.price)
    return [
        {"period": m, "avg_price": round(sum(p) / len(p), 2)}
        for m, p in sorted(by_month.items())
    ]


def _se_trend(hpi_payload: dict) -> list[dict]:
    records = hpi_payload.get("records", [])
    series = [
        {"period": r["period"], "avg_price": r["avg_price"]}
        for r in records
        if r.get("period") and r.get("avg_price") is not None
    ]
    return sorted(series, key=lambda r: r["period"])


def _top_streets_chart(top_streets_payload: dict) -> list[dict]:
    return [
        {"street": s["street"], "avg_price": s["avg_price"]}
        for s in top_streets_payload.get("streets", [])
    ]


def _save_charts(chart_data: dict) -> list[str]:
    CHARTS_DIR.mkdir(exist_ok=True)
    saved: list[str] = []

    if chart_data["gu1_trend"] and chart_data["south_east_trend"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        gu1 = chart_data["gu1_trend"]
        se = chart_data["south_east_trend"]
        ax.plot([p["period"] for p in gu1], [p["avg_price"] for p in gu1],
                label="GU1 monthly avg", marker="o", markersize=3)
        ax.plot([p["period"] for p in se], [p["avg_price"] for p in se],
                label="South East HPI", marker="s", markersize=3)
        ax.set_title("GU1 vs South East monthly average price")
        ax.set_xlabel("Period")
        ax.set_ylabel("Avg price (£)")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        # Trim x-tick density.
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[::max(1, len(ticks) // 12)])
        fig.tight_layout()
        out = CHARTS_DIR / "gu1_vs_se_trend.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(str(out))

    if chart_data["top_streets"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        streets = chart_data["top_streets"]
        ax.barh(
            [s["street"] for s in streets][::-1],
            [s["avg_price"] for s in streets][::-1],
        )
        ax.set_title("Highest-value streets in GU1 (avg sale price)")
        ax.set_xlabel("Avg price (£)")
        fig.tight_layout()
        out = CHARTS_DIR / "top_streets.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(str(out))

    return saved


def analyse_node(state: AgentState) -> dict:
    txns = state.get("transactions") or {}
    streets = state.get("top_streets") or {}
    hpi = state.get("hpi_data") or {}

    context_payload = {
        "transactions_summary": {
            "count": txns.get("count"),
            "date_range": txns.get("date_range"),
            "average_price": txns.get("average_price"),
            "sparse_data": txns.get("sparse_data"),
        },
        "top_streets": streets.get("streets", []),
        "hpi_recent": (hpi.get("records") or [])[:6],
        "hpi_data_ceiling_note": hpi.get("data_ceiling_note"),
    }

    response = _llm().invoke([
        SystemMessage(content=ANALYSE_SYSTEM),
        HumanMessage(content=(
            "Data context (only use these figures):\n"
            + json.dumps(context_payload, indent=2)
        )),
    ])
    note = response.content if isinstance(response.content, str) else str(response.content)

    chart_data: dict[str, Any] = {
        "gu1_trend": _monthly_gu1_trend(),
        "south_east_trend": _se_trend(hpi),
        "top_streets": _top_streets_chart(streets),
    }
    saved = _save_charts(chart_data)
    chart_data["saved_png_paths"] = saved

    _trace(
        state,
        "Generated research note",
        f"One-paragraph summary produced. {len(saved)} chart PNG(s) saved to ./charts/.",
    )

    return {
        "research_note": note,
        "chart_data": chart_data,
        "trace": state["trace"],
    }


# ---------------------------------------------------------------------------
# approval_gate — the critical node. Unmissable in the terminal.
# ---------------------------------------------------------------------------

def approval_gate(state: AgentState) -> dict:
    note = state.get("research_note", "")
    auto = bool(state.get("auto_approve"))

    print("\n" + "=" * 60)
    print("RESEARCH NOTE READY - REVIEW BEFORE WRITING")
    print("=" * 60)
    print(f"\n{note}\n")
    print("=" * 60)
    print("This note will be written to your tracking sheet.")
    print("=" * 60)

    if auto:
        print("\n[--auto-approve set] Auto-approving for non-interactive run.")
        approved = True
    else:
        try:
            response = input("\nApprove? (y/n): ").strip().lower()
        except EOFError:
            response = "n"
        approved = response == "y"

    if approved:
        print("\nApproved. Proceeding to write...")
        _trace(state, "User approved write action", "Approval recorded; proceeding to write_node.")
    else:
        print("\nDeclined. Note will not be written.")
        _trace(state, "User declined write action", "Approval refused; write_node will not execute.")

    return {"approved": approved, "trace": state["trace"]}


def _approval_router(state: AgentState) -> str:
    return "write" if state.get("approved") else END


# ---------------------------------------------------------------------------
# write_node — only reachable via the approval-gate edge.
# ---------------------------------------------------------------------------

def write_node(state: AgentState) -> dict:
    note = state.get("research_note", "")
    result = write_research_note.invoke({"note": note, "area": "GU1"})
    _trace(
        state,
        "Research note written to tracking sheet",
        f"write_research_note tool invoked. Sheet response: {result}",
    )
    return {"write_status": result, "trace": state["trace"]}


# ---------------------------------------------------------------------------
# Graph wiring.
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("analyse", analyse_node)
    g.add_node("approval_gate", approval_gate)
    g.add_node("write", write_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_conditional_edges("execute", _data_check_router, {"analyse": "analyse", END: END})
    g.add_edge("analyse", "approval_gate")
    g.add_conditional_edges("approval_gate", _approval_router, {"write": "write", END: END})
    g.add_edge("write", END)

    return g.compile()


def run_agent(
    user_prompt: str, auto_approve: bool = False, area: str = "GU1"
) -> AgentOutput:
    """Run the agent end-to-end and return a validated AgentOutput.

    The internal LangGraph state is a TypedDict (required by StateGraph);
    we wrap it here so the public boundary returns fully typed Pydantic
    models: ResearchNote, TraceEntry[], and the AgentOutput envelope.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env.")

    graph = build_graph()
    initial: AgentState = {
        "messages": [HumanMessage(content=user_prompt)],
        "trace": [],
        "auto_approve": auto_approve,
    }
    final: AgentState = graph.invoke(initial)

    return AgentOutput(
        research_note=ResearchNote(
            area=area,
            summary=final.get("research_note", ""),
            generated_at=datetime.now().isoformat(),
        ),
        chart_data=final.get("chart_data", {}),
        trace=[TraceEntry(**entry) for entry in final.get("trace", [])],
        approved=bool(final.get("approved", False)),
        write_status=final.get("write_status"),
    )
