# Land Registry Research Agent

An agentic pipeline that retrieves UK Land Registry data for **GU1 (Guildford)**,
reasons over it to produce a one-paragraph research note, and only after an
**explicit user approval gate**, writes that note to a mock tracking sheet.

The exercise uses UK Land Registry data as a structural proxy for Petex's real
simulation tools (PROSPER, GAP, MBAL). The architecture is what matters: the
data layer is swappable, the approval-gate pattern is enforced by graph
topology, and the agent's trace is designed to be read by a non-technical
engineer.

---

## Quick start

**Requires Python 3.10+ and an OpenAI API key.**

```bash
# 1. Create venv + install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Create your .env from the template, then edit it with your real key
cp .env.example .env
# open .env and replace sk-your-key-here with your OpenAI key

# 3. Run the agent
.venv/bin/python main.py                 # interactive: prompts at the approval gate
.venv/bin/python main.py --auto-approve  # bypass the prompt (demos / CI)
.venv/bin/python main.py --verbose       # add INFO logging (LLM calls, cache hits)
```

The `--auto-approve` and `--verbose` flags can be combined.

Other useful commands:

```bash
.venv/bin/pytest -v             # unit tests (data layer + agent routing)
.venv/bin/python data_layer.py  # smoke-test live endpoints (no LLM cost)
```

The first SPARQL call takes ~20-30 s (live endpoint is slow). Subsequent calls
hit the in-memory cache. PNG charts are written to `./charts/`.

---

## Architecture

```
START -> plan -> execute --(data ok?)--> analyse -> approval_gate
                    |                                     |
               data missing               approved -------+------- declined
                    |                         |                       |
                   END                      write                    END
                                              |
                                             END
```

| Layer | File | Responsibility |
|---|---|---|
| Models | [models.py](models.py) | Pydantic + `AgentState` TypedDict; every cross-layer value is typed |
| Data access | [data_layer.py](data_layer.py) | API wrappers, in-memory cache, 503 retry, sparse-data warnings |
| Tools | [tools.py](tools.py) | LangGraph `@tool` functions wrapping the data layer |
| Agent | [agent.py](agent.py) | `StateGraph`: plan → execute → analyse → approval → write |
| Write target | [mock_sheet.py](mock_sheet.py) | `print("would write: {row}")` stand-in |
| Entry point | [main.py](main.py) | CLI with `--auto-approve` flag |

### Why LangGraph

Explicit node separation is the whole point of this exercise. Petex's platform
mirrors this shape (plan, execute, interpret, gate, write) and `StateGraph`
with named nodes maps onto it cleanly. A `create_react_agent` would have done
the job in fewer lines but hidden the approval gate inside the agent's tool
choice instead of making it a first-class node with a conditional edge.

### How the approval gate is enforced

Two reinforcing mechanisms, in order of importance:

1. **Graph topology.** `write_research_note` is bound **only** on `write_node`.
   `write_node` is only reachable from `approval_gate` via a conditional edge
   that returns `END` unless `state["approved"]` is `True`. There is no path
   from `execute_node` to `write_node`. The LLM cannot decide to write; only
   the user can.
2. **Tool docstring.** `write_research_note`'s docstring explicitly says
   "Only call after explicit user approval has been confirmed." This is a
   belt-and-braces guard for any future graph change that wires the tool
   elsewhere by accident.

The approval gate prints a 60-char banner, the full note, and a second banner.
Interactive `input("Approve? (y/n): ")` by default; `--auto-approve` is provided
for non-interactive demos and CI runs.

### Why aggregate in Python, not SPARQL

The endpoint 503s under load on `GROUP BY` / `ORDER BY` / `AVG` over large
result sets. The data layer
fetches flat rows with `LIMIT 500` and aggregates in Python using
`collections.defaultdict`. Slower in theory, far more reliable in practice,
and gives us full control over filters (e.g. the `min_transactions` filter
on `get_top_streets`).

### Why a session-scoped in-memory cache

A cold SPARQL call to GU1 takes 20-30 s. The agent fetches the same dataset
twice in a single run (once for `fetch_area_transactions`, once for
`fetch_top_streets`). A `dict` keyed by `(district, limit, from_date, to_date)`
takes the second call to ~1 ms. Persistent caching would be the next
investment but is out of scope at the 3-4 h budget.

### Why a typed data layer with JSON tool boundaries

Inside the data layer everything is `Transaction`, `StreetSummary`,
`HPIRecord`. Tools take primitive args and return JSON strings, which is
what the LLM consumes. The Pydantic discipline is preserved everywhere it
provides safety, and the LLM gets a familiar JSON surface. This split is
what would let you swap Land Registry for PROSPER without touching `agent.py`.

### Date-window handling: the asymmetry that almost bites you

Price Paid transactions are current; the HPI endpoint ceilings at **2016-03**.
The agent queries Price Paid for **2013-04 to 2016-03** to match the HPI window
so that the "compare GU1 to South East regional average" part of the prompt
is an apples-to-apples comparison rather than fresh GU1 transactions vs stale
HPI. The trace and the research note both explicitly call this out, so the
agent does not claim to be reporting on "the present day."

This is encoded as the **default** on `get_transactions(from_date, to_date)`:
a naive call returns the HPI-aligned window, which is the only window in which
the prompt's comparison is meaningful. Callers can pass `from_date` / `to_date`
to override. The "last 3 years" phrasing in the user prompt is interpreted
relative to the latest available comparable data, not the present day.

### Operational instincts encoded

| Failure mode | Where handled | How |
|---|---|---|
| SPARQL 503 under load | `_sparql_with_retry` | 3 attempts, 2 s linear wait |
| SPARQL network timeout | `_sparql_with_retry` | Same retry path |
| `ppi:` vs `common:` namespace trap | `data_layer.get_transactions` | All address fields under `common:` |
| `propertyType` URI ugliness | Same | `.split("/")[-1]` |
| HPI ceiling at 2016-03 | `data_layer.get_hpi`, tool description | Surfaced in payload + trace + note |
| Slow endpoint | Module-level dict cache | Keyed by (district, limit, window) |
| Sparse data (<10 txns) | `data_layer.get_transactions` | Warn + return, never raise |
| Single-sale street outliers (£14 m commercial sales) | `get_top_streets(min_transactions=3)` | Avg over too few sales is misleading; floor at 3 sales |
| Empty HPI response | `data_layer.get_hpi` | Warn if <6 months; return what's there |
| All retries exhausted, payload never set | `_data_check_router` conditional edge | Graph routes to END before `analyse_node` runs; trace entry names the missing payload |

---

## What I deliberately left out

- **Real Google Sheets / Drive integration.** A mock writer prints what it
  would have written. The approval-gate **pattern** is what the exercise is
  about; the actual write surface is a swap.
- **Persistent cache.** A session-scoped `dict` is enough for a single-run
  agent. The next step is a sqlite cache with a TTL keyed on `refPeriod`.
- **HPI pagination.** 36 months covers the requirement. Adding `next`-link
  walking is a 10-minute addition when needed.
- **Streaming output.** I'd add it for production; for a CLI demo the bulk
  print at the end is more legible.
- **Backoff with jitter.** Linear 2 s retry is plenty when 503 is the only
  expected error and the call cadence is low.
- **Broader test coverage.** Nine unit tests cover the cache, sparse-data
  branch, 503 retry, both conditional routers, write tool isolation, and the
  execute-node tool loop driven by a scripted stand-in LLM. A full-graph
  end-to-end test and an eval harness are the next investment.
- **React chart frontend (bonus).** Matplotlib PNGs are saved to `./charts/`.
  React + Recharts is the natural next step.

---

## What I would invest in next

1. **Structured logging on every agent decision.** OpenTelemetry spans per
   node, with `tool_name`, `args`, `result_summary`, `duration_ms`. Hook into
   LangSmith or whatever Petex picks for traces.
2. **Real write target behind the same tool contract.** A Sheets / DOF
   adapter swapped in where `mock_sheet.write` lives; the rest of the agent
   doesn't move.
3. **Persistent cache with `refPeriod`-aware invalidation.** Sqlite keyed by
   `(district, window)` and HPI `(region, refPeriod_max)`. Cold reads only
   when the upstream actually has new data.
4. **MCP `subscribe_resource` pattern.** When the HPI endpoint publishes a
   new `refPeriod`, automatically trigger a refresh + re-analysis. Matches
   the live-data orientation of the real Petex platform.
5. **Eval harness.** Replay recorded traces against the agent and assert on
   shape (research note structure, chart_data keys, trace order) so a
   refactor doesn't silently change what the agent reports.
6. **Frontend chart rendering.** A small React/Recharts component reading
   `chart_data` JSON; straightforward, just out of scope here.
7. **Confidence scoring.** Where data is sparse or where top-street rankings
   are dominated by a few transactions, attach a numeric confidence score
   that the research-note prompt is required to surface.
8. **Per-node model selection.** The agent uses `gpt-4o-mini` for every LLM
   call. The plan and execute nodes are plumbing; model capability there is
   not the bottleneck. The analyse node is where reasoning would actually
   benefit from a larger model. If input variety grew (sparser postcodes,
   contradictory signals, multi-region comparisons) I would upgrade only
   `analyse_node` to `gpt-4o`, keeping mini elsewhere. One constant per node;
   five-minute change. Not justified by current output quality (`gpt-4o-mini`
   produces a publishable note on this data), so I'm noting the pattern
   rather than implementing it speculatively.

---

## File map

```
.
├── README.md                  this file
├── data_sources.py            provided starter, not modified
├── models.py                  Pydantic models + AgentState TypedDict
├── data_layer.py              data access + cache + retries
├── mock_sheet.py              mock write target
├── tools.py                   4 LangGraph tools (3 read, 1 write)
├── agent.py                   StateGraph definition + node functions
├── main.py                    CLI entry point
├── tests/
│   ├── test_data_layer.py     cache, sparse-data, and 503 retry tests
│   └── test_agent.py          routing, tool isolation, and execute-loop tests
├── requirements.txt
└── .gitignore
```

---

## Sample execution trace

The trace is designed for a non-technical engineer to read top-to-bottom and
understand exactly what the agent did:

```
Step 1: Produced analysis plan
        LLM produced a numbered plan covering data fetch, analysis, and
        approval gate.

Step 2: Fetched GU1 property transactions
        Retrieved 500 GU1 transactions (2013-04-04 to 2016-03-31). Average
        price £568,761. Data cached for session.

Step 3: Identified highest-value streets
        Top street: HIGH STREET, average £2,010,445 across 24 transactions.
        5 streets ranked.

Step 4: Fetched South East regional house price index
        Retrieved 36 monthly HPI records for south-east. Most recent period:
        2016-03, average price £266,729. Note: HPI data ceilings at 2016-03;
        analysis uses most recent available window.

Step 5: Generated research note
        One-paragraph summary produced. 2 chart PNG(s) saved to ./charts/.

Step 6: User approved write action
        Approval recorded; proceeding to write_node.

Step 7: Research note written to tracking sheet
        write_research_note tool invoked. Sheet response: Write logged
        successfully at 2026-05-25T... (mock)
```
