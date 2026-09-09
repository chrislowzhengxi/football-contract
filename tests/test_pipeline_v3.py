from src.pipeline_v3 import (
    CORE_FIELDS,
    OPPORTUNISTIC_FIELDS,
    SPECIALIST_CANDIDATES,
    V3Budget,
    classify_source,
    configured_domains_for_source_class,
    field_class,
    v3_event_plan,
)
from src.source_discovery import SourceCandidate, source_tier


def event(**overrides):
    values = {
        "event_id": "tm_v3",
        "player_name": "Player",
        "from_club_name": "Benfica",
        "to_club_name": "Porto",
        "transfer_date": "2025-07-01",
    }
    values.update(overrides)
    return values


def test_core_field_prioritization_before_opportunistic_fields():
    plan = v3_event_plan(event(), set(), V3Budget(event_total_max_queries=20))
    fields = [query.field for query in plan if query.field != "event_resolution"]

    assert "transfer_fee" in fields
    assert "purchase_option" in fields
    assert "sell_on" not in fields
    assert "buy_back" not in fields


def test_opportunistic_fields_do_not_consume_early_budget():
    plan = v3_event_plan(event(), set(), V3Budget(event_total_max_queries=10))
    opportunistic_positions = [idx for idx, query in enumerate(plan) if query.field in OPPORTUNISTIC_FIELDS]

    assert not opportunistic_positions


def test_source_class_targeting_uses_official_buying_and_selling_domains():
    plan = v3_event_plan(event(), set(), V3Budget(event_total_max_queries=20))

    assert any(query.target_source_class == "official_buying_club" and query.target_domain == "fcporto.pt" for query in plan)
    assert any(query.target_source_class == "official_selling_club" and query.target_domain == "slbenfica.pt" for query in plan)


def test_regulatory_query_only_when_configured():
    configured = configured_domains_for_source_class(event(), "financial_filing")
    unconfigured = configured_domains_for_source_class(event(from_club_name="Basel", to_club_name="Al-Ain"), "financial_filing")

    assert configured
    assert unconfigured == []


def test_fair_allocation_among_core_fields():
    plan = v3_event_plan(event(), set(), V3Budget(event_total_max_queries=15))
    planned = {query.field for query in plan}

    assert set(CORE_FIELDS) <= planned


def test_early_stopping_is_represented_by_established_field_omission():
    plan = v3_event_plan(event(), {"transfer_fee", "purchase_option"}, V3Budget(event_total_max_queries=20))
    fields = {query.field for query in plan}

    assert "transfer_fee" not in fields
    assert "purchase_option" not in fields


def test_specialist_candidates_remain_non_admissible_without_review():
    assert SPECIALIST_CANDIDATES["footballtransfers.com"]["reviewed"] is False
    assert SPECIALIST_CANDIDATES["footballtransfers.com"]["tier"] == 3


def test_tier3_never_supports_exact_financial_claims():
    source = SourceCandidate(
        "https://footballtransfers.com/player",
        "Player fee",
        "footballtransfers.com",
        None,
        "aggregator",
        "Player fee EUR 10m",
        "en",
    )

    assert source_tier(source) == 3


def test_field_class_configuration_is_reviewable():
    assert field_class("transfer_fee") == "core"
    assert field_class("add_ons") == "secondary"
    assert field_class("sell_on") == "opportunistic"


def test_source_classification_distinguishes_buying_and_selling_club():
    buying = SourceCandidate("https://fcporto.pt/news", "x", "fcporto.pt", None, "official", "", "en")
    selling = SourceCandidate("https://slbenfica.pt/news", "x", "slbenfica.pt", None, "official", "", "en")

    assert classify_source(buying, event()) == "official_buying_club"
    assert classify_source(selling, event()) == "official_selling_club"
