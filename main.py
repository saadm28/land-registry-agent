"""
Entry point.

    python main.py                 # interactive — prompts at the approval gate
    python main.py --auto-approve  # bypass the prompt for demos / CI
"""

from __future__ import annotations

import argparse
import json
import logging

from dotenv import load_dotenv

from agent import run_agent
from models import AgentOutput, TraceEntry

USER_PROMPT = (
    "Analyse property price trends in GU1 over the last 3 years. Compare with the "
    "South East regional average. Identify the highest-value streets. Then prepare "
    "a one-paragraph research note and add it to my tracking sheet."
)


def _print_trace(trace: list[TraceEntry]) -> None:
    print("\n" + "=" * 60)
    print("EXECUTION TRACE")
    print("=" * 60)
    for entry in trace:
        print(f"\nStep {entry.step}: {entry.action}")
        print(f"        {entry.result}")


def _print_research_note(note: str) -> None:
    print("\n" + "=" * 60)
    print("RESEARCH NOTE")
    print("=" * 60)
    print(f"\n{note}\n")


def _print_chart_data(chart_data: dict) -> None:
    print("=" * 60)
    print("CHART DATA (JSON)")
    print("=" * 60)
    preview = {
        "gu1_trend_points": len(chart_data.get("gu1_trend", [])),
        "south_east_trend_points": len(chart_data.get("south_east_trend", [])),
        "top_streets": chart_data.get("top_streets", []),
        "saved_png_paths": chart_data.get("saved_png_paths", []),
    }
    print(json.dumps(preview, indent=2))


def _print_write_status(output: AgentOutput) -> None:
    print("\n" + "=" * 60)
    print("WRITE STATUS")
    print("=" * 60)
    if output.approved:
        print(f"Approved. Mock sheet response: {output.write_status}")
    else:
        print("Declined. Nothing written.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Land Registry research agent")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Bypass the interactive approval gate (for demos / CI).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable INFO logging from data_layer."
    )
    args = parser.parse_args()

    load_dotenv()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    print(f"\nUser prompt:\n  {USER_PROMPT}\n")

    output: AgentOutput = run_agent(USER_PROMPT, auto_approve=args.auto_approve)

    _print_trace(output.trace)
    _print_research_note(output.research_note.summary or "(no note produced)")
    _print_chart_data(output.chart_data)
    _print_write_status(output)


if __name__ == "__main__":
    main()
