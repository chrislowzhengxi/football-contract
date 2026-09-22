"""Stage 2 pipeline tests.

These validate everything except the network: query construction from Stage 1C
identity, source classification, event matching, the extraction gate, and the
guarantee that Stage 1 facts are never re-researched or overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.stage2 import run_pilot
from src.stage2.queries import build_queries
from src.stage2.sources import can_establish_alone, classify, official_domain, tier_of

PILOT = Path("data/outputs/rebuild/stage2_pilot_selection.csv")
QUEUE = Path("data/outputs/rebuild/stage2_candidate_queue.csv")

WEGHORST = {
    "event_id": "tm_test", "player_id": 228645, "player_name": "Wout Weghorst",
    "from_club_name": "Besiktas", "to_club_name": "Burnley",
    "transfer_date": "2023-01-12", "candidate_category": "fee_bearing_loan_return",
}


# --- source classification -------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://www.burnleyfootballclub.com/news/x", "official_club_buying"),
    ("https://www.bjk.com.tr/tr/haber/x", "official_club_selling"),
    ("https://www.kap.org.tr/tr/Bildirim/1", "regulatory_filing"),
    ("https://www.bbc.co.uk/sport/football/1", "national_media_major"),
    ("https://www.sportskeeda.com/x", "aggregator_low_quality"),
    ("https://www.transfermarkt.co.uk/x", "transfermarkt_backbone"),
])
def test_sources_are_classified(url, expected):
    assert classify(url, "Besiktas", "Burnley") == expected


def test_unknown_domains_are_readable_but_not_self_sufficient():
    """Reclassified to tier 2 after the smoke test: every event surfaced an
    uncatalogued domain, and tier 3 meant they were never even retrieved."""
    assert tier_of("unknown") == 2
    assert not can_establish_alone("unknown")


def test_only_tier1_can_establish_a_term_alone():
    for cls in ["official_club_buying", "official_club_selling", "regulatory_filing",
                "financial_disclosure", "league_or_federation"]:
        assert can_establish_alone(cls), cls
    for cls in ["national_media_major", "transfer_specialist", "local_club_media",
                "aggregator_low_quality"]:
        assert not can_establish_alone(cls), cls


def test_transfermarkt_can_never_establish_a_contract_term():
    """It is the deterministic backbone, not independent evidence."""
    assert tier_of("transfermarkt_backbone") == 3
    assert not can_establish_alone("transfermarkt_backbone")


def test_listed_turkish_clubs_route_to_kap():
    from src.stage2.sources import disclosure_venue
    for club in ["Galatasaray", "Besiktas", "Trabzonspor"]:
        assert disclosure_venue(club)[0] == "kap.org.tr", club


# --- query construction ----------------------------------------------------

def test_queries_are_built_from_stage1_identity():
    queries = build_queries(WEGHORST)
    assert 1 <= len(queries) <= 6
    joined = " ".join(q.query for q in queries)
    assert "Wout Weghorst" in joined
    assert "Besiktas" in joined and "Burnley" in joined


def test_query_volume_is_bounded():
    for n in (1, 2, 3, 4, 5, 6):
        assert len(build_queries(WEGHORST, max_queries=n)) <= n


def test_a_listed_club_gets_a_regulated_disclosure_query():
    classes = {q.expected_source_class for q in build_queries(WEGHORST)}
    assert "regulatory_filing" in classes


def test_category_drives_which_mechanism_is_searched():
    ret = {q.target_field for q in build_queries({**WEGHORST,
           "candidate_category": "fee_bearing_loan_return"})}
    ctl = {q.target_field for q in build_queries({**WEGHORST,
           "candidate_category": "control_ordinary_permanent"})}
    assert ret != ctl
    assert "sell_on" in ctl or "add_ons" in ctl


def test_local_language_term_is_paired_with_the_club_that_speaks_it():
    queries = build_queries(WEGHORST)
    turkish = [q for q in queries if q.language == "tr"]
    if turkish:
        # Turkish vocabulary must go with Besiktas, not Burnley
        assert "Besiktas" in turkish[0].query


# --- event matching and the extraction gate -------------------------------

def test_event_matching_accepts_the_right_transfer():
    text = "Burnley have signed Wout Weghorst from Besiktas with an obligation to buy."
    assert run_pilot.event_matches(text, WEGHORST)


def test_event_matching_rejects_a_different_transfer():
    text = "Manchester United have signed Wout Weghorst on loan from Besiktas."
    # right player, but neither club of THIS event beyond the shared selling club
    other = {**WEGHORST, "from_club_name": "Ajax", "to_club_name": "Sunderland"}
    assert not run_pilot.event_matches(text, other)


def test_event_matching_rejects_a_page_about_another_player():
    assert not run_pilot.event_matches("Burnley sign a striker from Besiktas", WEGHORST)


def test_gate_requires_real_contract_language():
    ok, hits = run_pilot.passes_gate(
        "The deal includes an obligation to buy and a 10% sell-on clause.")
    assert ok and len(hits) >= 2


def test_gate_rejects_a_page_with_no_contract_content():
    ok, hits = run_pilot.passes_gate("Weghorst scored twice on his debut for Burnley.")
    assert not ok


def test_gate_saves_an_llm_call_when_evidence_is_empty():
    metrics = run_pilot.Metrics()
    record = run_pilot.research_event(WEGHORST, None, None, None, metrics, dry_run=True)
    assert record["extraction_status"] == "insufficient_evidence"
    assert metrics.llm_extraction_calls == 0
    assert metrics.llm_skipped_insufficient_evidence == 1


# --- the pipeline reaches extraction when evidence is real ----------------

def test_pipeline_reaches_extraction_with_seeded_evidence(tmp_path, monkeypatch):
    """Seed the caches with a realistic official-club page and confirm the
    event is classified, matched, gated and routed to extraction."""
    monkeypatch.setattr(run_pilot, "CACHE_DIR", tmp_path / "cache")
    url = "https://www.burnleyfootballclub.com/news/weghorst-signing"
    page = ("Burnley Football Club can confirm the return of Wout Weghorst from "
            "Besiktas. The agreement includes an obligation to buy triggered by "
            "appearances, and a 10% sell-on clause for Besiktas.")
    for query in build_queries(WEGHORST):
        run_pilot._cache_path("search", query.query).write_text(json.dumps(
            {"query": query.query,
             "results": [{"url": url, "title": "Weghorst signing", "content": page[:180]}]}))
    run_pilot._cache_path("page", url).write_text(json.dumps({"url": url, "text": page}))

    metrics = run_pilot.Metrics()
    record = run_pilot.research_event(WEGHORST, None, None, None, metrics, dry_run=True)
    assert record["extraction_status"] == "would_extract"
    assert record["pages_usable"] >= 1
    assert record["tier1_usable"] >= 1              # official club page
    assert "official_club_buying" in record["source_classes_seen"]
    assert metrics.pages_passing_gate >= 1
    assert metrics.llm_extraction_calls == 0        # dry run never calls out


def test_a_low_quality_page_is_never_retrieved(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pilot, "CACHE_DIR", tmp_path / "cache2")
    url = "https://www.sportskeeda.com/rumour"
    for query in build_queries(WEGHORST):
        run_pilot._cache_path("search", query.query).write_text(json.dumps(
            {"query": query.query, "results": [{"url": url, "title": "r", "content": "obligation to buy"}]}))
    record = run_pilot.research_event(WEGHORST, None, None, None, run_pilot.Metrics(), dry_run=True)
    assert record["pages_retrieved"] == 0
    assert record["extraction_status"] == "insufficient_evidence"


# --- Stage 1 / Stage 2 separation -----------------------------------------

STAGE1_OWNED = {
    "player_id", "player_name", "transfer_date", "from_club_name", "to_club_name",
    "transfer_type_normalized", "permanent_transfer_fee_eur", "loan_fee_eur",
    "market_value_eur", "fee_on_return_eur",
}


def test_stage2_target_fields_do_not_overlap_stage1_facts():
    assert not (set(run_pilot.TARGET_FIELDS) & STAGE1_OWNED)


def test_fee_on_return_is_interpreted_not_overwritten():
    """Stage 2 may add an interpretation; the Stage 1 amount stays its own field."""
    assert "fee_on_return_interpretation" in run_pilot.TARGET_FIELDS
    assert "fee_on_return_eur" not in run_pilot.TARGET_FIELDS


# --- the pilot inputs ------------------------------------------------------

@pytest.fixture(scope="module")
def pilot():
    if not PILOT.exists():
        pytest.skip("pilot not built")
    return pd.read_csv(PILOT)


def test_pilot_shape(pilot):
    assert len(pilot) == 26
    assert pilot.candidate_category.value_counts().to_dict() == {
        "loan_with_fee": 10, "fee_bearing_loan_return": 8,
        "high_value_undisclosed": 4, "control_ordinary_permanent": 4}


def test_pilot_is_identified_by_player_id(pilot):
    assert pilot.player_id.notna().all()
    assert pilot.event_id.is_unique
    # Ladislav Krejci is in the pilot and shares his name with another player.
    krejci = pilot[pilot.player_name.str.contains("Krejci", na=False)]
    if len(krejci):
        assert krejci.player_id.iloc[0] == 345911


def test_every_pilot_row_states_the_rule_that_chose_it(pilot):
    assert pilot.pilot_rule.notna().all()
    assert (pilot.pilot_rule.str.len() > 10).all()


def test_pilot_events_have_already_happened(pilot):
    assert (pd.to_datetime(pilot.transfer_date) <= pd.Timestamp("2026-09-21")).all()


def test_every_pilot_event_produces_queries(pilot):
    for _, row in pilot.iterrows():
        queries = build_queries(row.to_dict())
        assert queries, row.event_id
        assert all(q.query.strip() for q in queries)


def test_credential_check_reports_what_is_missing():
    missing = run_pilot.missing_credentials()
    assert isinstance(missing, list)
    assert set(missing) <= {"TAVILY_API_KEY", "PARLEY_API_KEY"}
