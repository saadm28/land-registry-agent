"""
Minimal unit tests on the data layer — the part that needs operational
confidence (cache + sparse-data handling). Broader coverage is on the
"invest next" list in the README.

Run from project root:
    pytest -v
"""

from __future__ import annotations

import logging
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_layer  # noqa: E402


def _fake_sparql_response(rows: list[dict]) -> dict:
    bindings = []
    for r in rows:
        bindings.append({
            "price": {"value": str(r["price"])},
            "date": {"value": r["date"]},
            "street": {"value": r["street"]},
            "postcode": {"value": r["postcode"]},
            "propertyType": {"value": f"http://landregistry.data.gov.uk/def/ppi/{r['ptype']}"},
        })
    return {"results": {"bindings": bindings}}


@pytest.fixture(autouse=True)
def _clear_cache():
    data_layer.clear_cache()
    yield
    data_layer.clear_cache()


def test_cache_hit_skips_network(monkeypatch):
    """Second call for the same (district, window, limit) must not hit the network."""
    rows = [
        {"price": 300_000 + i * 1000, "date": "2015-06-01", "street": f"S{i}",
         "postcode": "GU1 1AA", "ptype": "detachedType"}
        for i in range(20)
    ]
    call_count = {"n": 0}

    def fake_query(q):
        call_count["n"] += 1
        return _fake_sparql_response(rows)

    monkeypatch.setattr(data_layer, "_sparql_with_retry", fake_query)

    a = data_layer.get_transactions("GU1")
    b = data_layer.get_transactions("GU1")

    assert call_count["n"] == 1, "second call should be served from cache"
    assert len(a) == 20
    assert a == b


def test_sparse_data_warns_does_not_raise(monkeypatch, caplog):
    """Fewer than 10 rows should log a warning and return what's available."""
    rows = [
        {"price": 250_000, "date": "2015-01-01", "street": "ONLY ST",
         "postcode": "ZZ9 9ZZ", "ptype": "flatType"},
    ]
    monkeypatch.setattr(
        data_layer, "_sparql_with_retry", lambda q: _fake_sparql_response(rows)
    )

    with caplog.at_level(logging.WARNING, logger="data_layer"):
        result = data_layer.get_transactions("ZZ9")

    assert len(result) == 1
    assert any("Sparse data" in rec.message for rec in caplog.records), (
        "expected a sparse-data warning to be emitted"
    )


def test_503_retry_succeeds_on_third_attempt(monkeypatch):
    """Two consecutive 503s must be retried; the third attempt must succeed."""
    rows = [
        {"price": 300_000 + i * 1_000, "date": "2015-06-01", "street": f"RETRY ST {i}",
         "postcode": "GU1 1AA", "ptype": "detachedType"}
        for i in range(15)
    ]
    call_count = {"n": 0}

    def fake_query_503_twice(query):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise urllib.error.HTTPError(
                url=None, code=503, msg="Service Unavailable", hdrs=None, fp=None
            )
        return _fake_sparql_response(rows)

    monkeypatch.setattr(data_layer, "_sparql_query", fake_query_503_twice)
    monkeypatch.setattr(data_layer, "RETRY_WAIT_SECONDS", 0)

    result = data_layer.get_transactions("GU1")

    assert call_count["n"] == 3, "expected exactly 3 attempts (2 failures + 1 success)"
    assert len(result) == 15
