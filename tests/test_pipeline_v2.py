from src.pipeline_v2 import (
    EventResolution,
    FIELD_RESEARCH_FIELDS,
    FieldEvidenceAssessment,
    SearchBudget,
    assess_field_evidence,
    applicable_research_fields,
    generate_event_query_plan,
    generate_field_queries,
    identify_related_events,
    later_fee_supports_prior_option,
    link_evidence_to_event,
    retrospective_supports_prior_term,
)
from src.source_discovery import SourceCandidate, assess_event_match, source_tier


def event(**overrides):
    values = {
        "event_id": "tm_example",
        "player_name": "Player",
        "from_club_name": "Benfica",
        "to_club_name": "Porto",
        "transfer_date": "2025-07-01",
    }
    values.update(overrides)
    return values


def test_field_level_sufficiency_is_independent():
    sources = [{"evidence_id": "s1", "source_tier": 2}]
    fee = assess_field_evidence("transfer_fee", {"status": "disclosed_yes", "amount": 10_000_000, "evidence_ids": ["s1"]}, sources)
    sell_on = assess_field_evidence("sell_on", {"status": "not_found", "evidence_ids": []}, sources)

    assert fee.status == "sufficient"
    assert sell_on.status == "insufficient"


def test_fee_sufficient_while_sell_on_insufficient():
    fee = FieldEvidenceAssessment("tm_1", "transfer_fee", "sufficient", ["s1"], [2], "fee reported")
    sell_on = FieldEvidenceAssessment("tm_1", "sell_on", "insufficient", [], [], "not reported")

    assert fee.status == "sufficient"
    assert sell_on.status == "insufficient"


def test_related_loan_and_permanent_events_remain_distinct():
    events = [
        event(event_id="tm_loan", from_club_name="Benfica", to_club_name="Porto"),
        event(event_id="tm_return", from_club_name="Porto", to_club_name="Benfica"),
    ]

    relations = identify_related_events(events)

    assert {relation.event_id for relation in relations} == {"tm_loan", "tm_return"}
    assert all(relation.event_id != relation.related_event_id for relation in relations)
    assert {relation.relation_type for relation in relations} == {"reverse_transfer"}


def test_retrospective_source_supports_prior_term_only_when_explicit():
    explicit = "After joining on loan last summer with an option to buy worth EUR 20m, the player moved permanently."
    vague = "The player moved permanently one year after a loan spell."

    assert retrospective_supports_prior_term(explicit, "purchase_option")
    assert not retrospective_supports_prior_term(vague, "purchase_option")


def test_later_fee_cannot_automatically_become_earlier_option_value():
    later_fee_only = "The player completed a permanent transfer for EUR 20m after last season's loan."
    explicit_option = "The original loan deal included an option to buy for EUR 20m."

    assert not later_fee_supports_prior_option(later_fee_only)
    assert later_fee_supports_prior_option(explicit_option)


def test_query_budgets_remain_bounded():
    plans = generate_event_query_plan(event(), budget=SearchBudget(event_total_max_queries=12))

    assert len(plans) <= 12
    assert plans[0].stage == "event_resolution"
    assert {plan.field for plan in plans} <= set(FIELD_RESEARCH_FIELDS) | {"event_resolution"}


def test_fair_query_allocation_prevents_late_field_starvation():
    plans = generate_event_query_plan(event(), budget=SearchBudget(event_total_max_queries=18))
    fields = {plan.field for plan in plans}

    assert "sell_on" in fields
    assert "buy_back" in fields
    assert "release_or_purchase_clause" in fields


def test_applicable_field_prioritization_skips_clear_permanent_loan_fee():
    fields = applicable_research_fields(event(transfer_type="permanent"))

    assert "loan_fee" not in fields
    assert fields.index("transfer_fee") < fields.index("sell_on")


def test_already_sufficient_fields_receive_no_event_queries():
    plans = generate_event_query_plan(event(), established_fields={"transfer_fee", "sell_on"})
    planned_fields = {plan.field for plan in plans}

    assert "transfer_fee" not in planned_fields
    assert "sell_on" not in planned_fields


def test_multilingual_field_queries_generated_correctly():
    plans = generate_field_queries(event(player_name="João Mário", from_club_name="Besiktas", to_club_name="Benfica"), "purchase_option")

    assert any(plan.language == "pt" and "opção de compra" in plan.query for plan in plans)
    assert len(plans) <= 4


def test_field_search_stops_once_sufficient():
    plans = generate_field_queries(event(), "transfer_fee", already_sufficient=True)

    assert plans == []


def test_deterministic_transfer_fee_is_not_a_field_assessment_source():
    assessment = assess_field_evidence("transfer_fee", {"status": "not_found", "amount": None, "evidence_ids": []}, [])

    assert assessment.status == "insufficient"
    assert assessment.evidence_ids == []


def test_direct_event_match_is_accepted_as_direct_link():
    source = SourceCandidate(
        source_url="https://fcporto.pt/news",
        source_title="Player joins Porto",
        publisher="fcporto.pt",
        publication_date="2025-07-01",
        source_type="official",
        evidence_text="Player joined Porto from Benfica in 2025.",
        language="en",
        evidence_id="s1",
    )
    source.source_tier = source_tier(source)
    source.event_match_status, _, _ = assess_event_match(source, event())
    resolution = EventResolution("tm_example", "confirmed", 0.9, ["anchor"])

    assert link_evidence_to_event(source, resolution, event()).status == "direct_match"


def test_safe_anchored_field_evidence_is_accepted():
    source = SourceCandidate(
        source_url="https://fcporto.pt/news/player-option",
        source_title="Player option to buy",
        publisher="fcporto.pt",
        publication_date="2025-07-01",
        source_type="official",
        evidence_text="Player signed for Porto in 2025 with an option to buy.",
        language="en",
        evidence_id="s2",
    )
    source.event_match_status = None
    resolution = EventResolution("tm_example", "confirmed", 0.9, ["anchor"])

    assert link_evidence_to_event(source, resolution, event()).status == "anchored_match"


def test_reverse_transfer_evidence_is_rejected_for_anchor():
    source = SourceCandidate(
        source_url="https://fcporto.pt/news/reverse",
        source_title="Player reverse transfer",
        publisher="fcporto.pt",
        publication_date="2025-07-01",
        source_type="official",
        evidence_text="Player joined Benfica from Porto in 2025 with an option.",
        language="en",
        evidence_id="s3",
    )
    source.event_match_status = "mismatch"
    resolution = EventResolution("tm_example", "confirmed", 0.9, ["anchor"])

    assert link_evidence_to_event(source, resolution, event()).status == "mismatch"
