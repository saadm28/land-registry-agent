"""
Data access layer for UK Land Registry.

Sits on top of the urllib-based primitives in data_sources.py. Three concerns:
  1. Typed return values (Pydantic models, never raw dicts upward).
  2. Operational safety: in-memory cache + 503 retry + sparse-data warnings.
  3. Endpoint quirks documented and handled here so the agent layer stays clean.

Endpoint behaviours encoded:
  - SPARQL: 10-20s cold; aggregate in Python, never in SPARQL (causes 503).
  - SPARQL: address fields under common: namespace, not ppi:.
  - HPI REST: data ceiling currently 2016-03; we surface that to callers.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

from models import HPIRecord, StreetSummary, Transaction

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://landregistry.data.gov.uk/landregistry/query"
HPI_BASE = "https://landregistry.data.gov.uk/data/hpi/region"

# HPI data ceilings here. Anchoring transaction queries to the same window
# gives like-for-like comparison rather than fresh transactions vs stale HPI.
HPI_CEILING = "2016-03"
DEFAULT_FROM_DATE = "2013-04-01"
DEFAULT_TO_DATE = "2016-03-31"

SPARSE_DATA_THRESHOLD = 10
RETRY_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 2

_cache: dict[str, object] = {}


def _sparql_query(query: str, timeout: int = 60) -> dict:
    """Single SPARQL request. Caller handles retries."""
    params = urllib.parse.urlencode({"query": query, "output": "json"})
    req = urllib.request.Request(
        f"{SPARQL_ENDPOINT}?{params}",
        headers={"Accept": "application/sparql-results+json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _sparql_with_retry(query: str) -> dict:
    """SPARQL with linear retry on 503. The endpoint 503s under load."""
    last_err: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return _sparql_query(query)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 503 and attempt < RETRY_ATTEMPTS:
                logger.warning(
                    "SPARQL 503 on attempt %s/%s; retrying in %ss",
                    attempt, RETRY_ATTEMPTS, RETRY_WAIT_SECONDS,
                )
                time.sleep(RETRY_WAIT_SECONDS)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < RETRY_ATTEMPTS:
                logger.warning(
                    "SPARQL network error on attempt %s/%s: %s; retrying",
                    attempt, RETRY_ATTEMPTS, e,
                )
                time.sleep(RETRY_WAIT_SECONDS)
                continue
            raise
    # Defensive: should be unreachable.
    raise RuntimeError(f"SPARQL retries exhausted: {last_err}")


def _hpi_get(region: str, params: dict | None = None, timeout: int = 20) -> dict:
    url = f"{HPI_BASE}/{region}.json"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "lr-agent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get_transactions(
    postcode_district: str,
    limit: int = 500,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = DEFAULT_TO_DATE,
) -> list[Transaction]:
    """Fetch Price Paid transactions for a postcode district.

    Defaults to the HPI-aligned window (2013-04 to 2016-03) so that downstream
    comparisons with HPI are like-for-like. Override via from_date / to_date.

    - Cached in-memory by (district, limit, window).
    - Retries up to 3 times on 503.
    - Warns (does not raise) if fewer than 10 rows return.
    """
    cache_key = f"transactions_{postcode_district}_{limit}_{from_date}_{to_date}"
    if cache_key in _cache:
        logger.info("Cache hit: %s", cache_key)
        return _cache[cache_key]  # type: ignore[return-value]

    # Flat-row SELECT only; no GROUP BY / ORDER BY / AVG (those 503).
    # Address fields live under common:, NOT ppi:.
    query = f"""
PREFIX ppi:    <http://landregistry.data.gov.uk/def/ppi/>
PREFIX common: <http://landregistry.data.gov.uk/def/common/>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

SELECT ?price ?date ?street ?postcode ?propertyType WHERE {{
  ?tx a ppi:TransactionRecord ;
      ppi:pricePaid       ?price ;
      ppi:transactionDate ?date ;
      ppi:propertyType    ?propertyType ;
      ppi:propertyAddress ?addr .
  ?addr common:postcode ?postcode ;
        common:street   ?street .
  FILTER(STRSTARTS(STR(?postcode), "{postcode_district} "))
  FILTER(?date >= "{from_date}"^^xsd:date)
  FILTER(?date <= "{to_date}"^^xsd:date)
}} LIMIT {limit}
"""

    data = _sparql_with_retry(query)
    bindings = data["results"]["bindings"]

    transactions: list[Transaction] = []
    for b in bindings:
        transactions.append(
            Transaction(
                price=int(b["price"]["value"]),
                date=b["date"]["value"],
                street=b["street"]["value"],
                postcode=b["postcode"]["value"],
                # propertyType is a full URI; the label is the last path segment.
                property_type=b["propertyType"]["value"].split("/")[-1],
            )
        )

    if len(transactions) < SPARSE_DATA_THRESHOLD:
        logger.warning(
            "Sparse data: only %s transactions returned for %s in window %s..%s. "
            "Downstream reasoning should reflect low confidence.",
            len(transactions), postcode_district, from_date, to_date,
        )

    _cache[cache_key] = transactions
    return transactions


def get_top_streets(
    postcode_district: str,
    top_n: int = 5,
    min_transactions: int = 3,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = DEFAULT_TO_DATE,
) -> list[StreetSummary]:
    """Rank streets by average sale price for a postcode district.

    Reuses get_transactions() so we only ever hit SPARQL once per window —
    cache makes the second call effectively free.

    min_transactions filters out streets with too few sales to support an
    average — single £14m commercial sales otherwise dominate the ranking
    and mislead the chart. Default of 3 is a conservative floor.
    """
    transactions = get_transactions(
        postcode_district, limit=500, from_date=from_date, to_date=to_date
    )

    by_street: dict[str, list[int]] = defaultdict(list)
    for tx in transactions:
        by_street[tx.street].append(tx.price)

    eligible = {s: p for s, p in by_street.items() if len(p) >= min_transactions}
    ranked = sorted(
        eligible.items(),
        key=lambda kv: sum(kv[1]) / len(kv[1]),
        reverse=True,
    )[:top_n]

    return [
        StreetSummary(
            street=street,
            avg_price=sum(prices) / len(prices),
            transaction_count=len(prices),
        )
        for street, prices in ranked
    ]


def get_hpi(region: str, months: int = 36) -> list[HPIRecord]:
    """Fetch House Price Index records for a region.

    Endpoint data currently ceilings at 2016-03 — we return what's there and
    warn if fewer than 6 months come back. Cached by (region, months).
    """
    cache_key = f"hpi_{region}_{months}"
    if cache_key in _cache:
        logger.info("Cache hit: %s", cache_key)
        return _cache[cache_key]  # type: ignore[return-value]

    data = _hpi_get(region, {"_pageSize": months, "_sort": "-refPeriod"})
    items = data.get("result", {}).get("items", [])

    records: list[HPIRecord] = []
    for item in items:
        records.append(
            HPIRecord(
                period=item.get("refPeriod"),
                avg_price=item.get("averagePricesSASM"),
                annual_change=item.get("annualChange"),
                monthly_change=item.get("monthlyChange"),
            )
        )

    if len(records) < 6:
        logger.warning(
            "HPI returned only %s months for region %s — below confidence threshold.",
            len(records), region,
        )

    _cache[cache_key] = records
    return records


def clear_cache() -> None:
    """Test helper. Not used in production paths."""
    _cache.clear()


if __name__ == "__main__":
    # Smoke test against live endpoints.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    print("\n── GU1 transactions (HPI-aligned window) ──")
    t0 = time.time()
    txns = get_transactions("GU1")
    print(f"   fetched {len(txns)} in {round((time.time() - t0) * 1000)} ms")
    for t in txns[:3]:
        print(f"   £{t.price:>10,}  {t.date}  {t.street:<35} {t.property_type}")

    print("\n── Top streets (from cache) ──")
    t0 = time.time()
    streets = get_top_streets("GU1")
    print(f"   computed in {round((time.time() - t0) * 1000)} ms (cache hit expected)")
    for s in streets:
        print(f"   {s.street:<40} £{s.avg_price:>10,.0f}  ({s.transaction_count} tx)")

    print("\n── South East HPI ──")
    t0 = time.time()
    hpi = get_hpi("south-east")
    print(f"   fetched {len(hpi)} records in {round((time.time() - t0) * 1000)} ms")
    for h in hpi[:3]:
        print(f"   {h.period}  avg £{h.avg_price}  annual {h.annual_change}%")
