"""Injury pilot access check: stopped before collection, no injury data cached or inferred, nothing old touched."""
import inspect
import json

import pytest

from src.analysis import injury_pilot_access as ipa

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


@pytest.fixture(scope="module")
def N():
    if not ipa.NUMBERS_JSON.exists():
        pytest.skip("access check not run")
    return json.loads(ipa.NUMBERS_JSON.read_text())


def test_collection_stopped_and_nothing_collected(N):
    assert N["injury_pages_requested"] == 0 and N["pages_retrieved"] == 0 and N["rows_parsed"] == 0
    cached = [p for p in ipa.INJURY_CACHE.rglob("*") if p.is_file()] if ipa.INJURY_CACHE.exists() else []
    assert not cached
    assert N["decision"].startswith("STOP")


def test_module_cannot_fetch_player_pages():
    src = inspect.getsource(ipa)
    for token in ("urlopen", "import requests", "httpx", "playwright", "verletzungen/spieler/{"):
        assert token not in src


def test_policy_snapshot_is_verified_and_clause_found():
    meta = ipa.load_snapshot()                                   # raises if a file no longer matches its hash
    clauses = ipa.terms_clauses(meta)
    assert all(clauses.values())
    en = next(v for k, v in clauses.items() if ".com/" in k)
    assert "automated processes" in en and "44b" in en


def test_rules_never_count_non_selection_or_suspension():
    rules = {c: r for c, r, _ in ipa.CATEGORIES}
    assert rules["non-selection / absent from the squad"] == "never counts"
    assert rules["suspension"] == "does not count"
    assert any(case.startswith("open injury") and "Do not infer a return date" in rule for case, rule in ipa.EDGE_RULES)


def test_stage1c_and_playing_time_outputs_unchanged(N):
    assert _sha(ipa.OUT / "stage1c_canonical_transfers.csv") == STAGE1C_SHA256
    for name, h in N["old_output_sha256"].items():
        assert _sha(ipa.OUT / name) == h, name
