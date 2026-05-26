"""
Mock write target. Stands in for a real Google Sheets / DOF model write.

The approval gate pattern is what matters for the exercise — the actual
write surface is a one-liner anyone can swap.
"""

from __future__ import annotations

import json
from datetime import datetime


def write(row_data: dict) -> str:
    timestamp = datetime.now().isoformat()
    print("\n[MOCK SHEET] Would write to tracking sheet:")
    print(f"  Timestamp: {timestamp}")
    print(f"  Data: {json.dumps(row_data, indent=2)}")
    return f"Write logged successfully at {timestamp} (mock)"
