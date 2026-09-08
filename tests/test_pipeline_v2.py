from src.pipeline_v2 import (
    FIELD_RESEARCH_FIELDS,
    FieldEvidenceAssessment,
    SearchBudget,
    assess_field_evidence,
    generate_event_query_plan,
    generate_field_queries,
    identify_related_events,
    later_fee_supports_prior_option,
    retrospective_supports_prior_term,
)


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


def test_multilingual_field_queries_generated_correctly():
    plans = generate_field_queries(event(player_name="João Mário", from_club_name="Besiktas", to_club_name="Benfica"), "purchase_option")

    assert any(plan.language == "pt" and "opção de compra" in plan.query for plan in plans)


def test_field_search_stops_once_sufficient():
    plans = generate_field_queries(event(), "transfer_fee", already_sufficient=True)

    assert plans == []


def test_deterministic_transfer_fee_is_not_a_field_assessment_source():
    assessment = assess_field_evidence("transfer_fee", {"status": "not_found", "amount": None, "evidence_ids": []}, [])

    assert assessment.status == "insufficient"
    assert assessment.evidence_ids == []
