"""Adversarial checks over an assembled dataset.

Each test feeds the checker a row carrying a defect that reached the first
demonstration artifact, and asserts the checker names it.
"""
from src.stage2 import redteam

TERMS = ["purchase_option", "buy_back", "sell_on", "add_ons",
         "termination_compensation", "third_party_sale_share"]


def _row(**kw):
    base = {"player": "Test Player", "event_id": "e1", "transfer_type": "loan",
            "anchor_event_role": "original_loan", "family_classification": "loan_family",
            "family_notes": "", "family_event_ids": "e1;e2",
            "event_family_summary": "", "quality_grade": "C",
            "n_researched_fields": 0, "economic_mechanism": "",
            "review_reasons": "", "field_statuses": "", "reported_values": "",
            "deal_summary": "", "evidence_summary": ""}
    base.update(kw)
    return base


def test_permanent_transfer_classified_as_a_return_is_caught():
    v = redteam.check_transfer_type_matches_family_roles(
        _row(transfer_type="permanent_transfer",
             family_classification="likely_negotiated_return"))
    assert any(x.check == "permanent_transfer_classified_as_return" for x in v)


def test_permanent_transfer_with_a_return_role_is_caught():
    v = redteam.check_transfer_type_matches_family_roles(
        _row(transfer_type="permanent_transfer", anchor_event_role="loan_return"))
    assert any(x.check == "permanent_transfer_with_return_role" for x in v)


def test_permanent_transfer_called_a_return_in_the_notes_is_caught():
    v = redteam.check_transfer_type_matches_family_roles(
        _row(transfer_type="free_transfer", family_classification="standalone_permanent_transfer",
             family_notes="Standalone fee-bearing return with no follow-on movement"))
    assert any(x.check == "permanent_transfer_described_as_return" for x in v)


def test_a_clean_permanent_transfer_passes():
    assert redteam.check_transfer_type_matches_family_roles(
        _row(transfer_type="permanent_transfer", anchor_event_role="permanent_transfer",
             family_classification="standalone_permanent_transfer",
             family_notes="Standalone permanent transfer from A to B")) == []


def test_amount_scaling_error_is_caught():
    v = redteam.check_amount_scaling(
        _row(purchase_option="amount=8; EUR",
             purchase_option_evidence_span="a buy option of €8 million"), TERMS)
    assert any(x.check == "amount_scaling_error" for x in v)


def test_a_correctly_scaled_amount_passes():
    assert redteam.check_amount_scaling(
        _row(purchase_option="amount=8000000; EUR",
             purchase_option_evidence_span="a buy option of €8 million"), TERMS) == []


def test_mechanism_without_a_supporting_field_is_caught():
    v = redteam.check_mechanism_supported(
        _row(economic_mechanism="buy_back_exercise"), TERMS)
    assert any(x.check == "mechanism_without_field" for x in v)


def test_mechanism_with_its_field_passes():
    assert redteam.check_mechanism_supported(
        _row(economic_mechanism="buy_back_exercise", buy_back="3,000,000 EUR"),
        TERMS) == []


def test_scope_outside_the_family_is_caught():
    v = redteam.check_scope_in_family(
        _row(sell_on="percentage=40", sell_on_scope_event_id="e_other"), TERMS)
    assert any(x.check == "scope_outside_family" for x in v)


def test_grade_a_with_no_finding_is_caught():
    v = redteam.check_grade_has_a_finding(_row(quality_grade="A"), TERMS)
    assert any(x.check == "grade_a_without_finding" for x in v)


def test_grade_a_with_a_finding_passes():
    assert redteam.check_grade_has_a_finding(
        _row(quality_grade="A", n_researched_fields=1,
             buy_back="3,000,000 EUR"), TERMS) == []


def test_source_about_another_player_is_caught():
    v = redteam.check_sources_belong_to_the_player(
        _row(player="Santiago Gimenez"),
        [{"source_title": "Spain striker Alvaro Morata joins Galatasaray",
          "source_url": "https://example.com/morata"}])
    assert any(x.check == "source_about_another_player" for x in v)


def test_source_about_our_player_passes():
    assert redteam.check_sources_belong_to_the_player(
        _row(player="Riccardo Sottil"),
        [{"source_title": "Fiorentina, deciso il contro-riscatto di Riccardo Sottil",
          "source_url": "https://example.com/sottil"}]) == []


def test_undeclared_conflict_is_caught():
    v = redteam.check_conflicts_are_declared(
        _row(buy_back="3,000,000 EUR", buy_back_status="disclosed_yes",
             buy_back_evidence_span="Milan activated the buyback; €3m or €3.5m"), TERMS)
    assert any(x.check == "undeclared_conflict" for x in v)


def test_additive_components_are_not_reported_as_an_undeclared_conflict():
    assert redteam.check_conflicts_are_declared(
        _row(termination_compensation="5,000,000 EUR",
             termination_compensation_status="disclosed_yes",
             termination_compensation_evidence_span=(
                 "a termination fee of 5,000,000 EUR. Additionally, the footballer "
                 "has waived his receivables amounting to 651,562 EUR")), TERMS) == []


def test_a_review_note_about_figures_we_did_not_publish_is_not_a_contradiction():
    assert redteam.check_review_reasons_match_output(
        _row(review_reasons="Conflicting sources on buyback fee: €3.5m vs €3m",
             buy_back="partially_disclosed", quality_grade="A")) == []


def test_a_date_is_not_a_published_figure():
    assert redteam.check_review_reasons_match_output(
        _row(review_reasons="Conflicting sources on the option fee",
             buy_back="2023-06-19 exercised=True", quality_grade="A")) == []


def test_declared_conflict_passes():
    assert redteam.check_conflicts_are_declared(
        _row(buy_back="3,000,000 EUR", buy_back_status="conflicting_sources",
             buy_back_evidence_span="Milan activated the buyback; €3m or €3.5m"), TERMS) == []


def test_review_reason_contradicting_the_output_is_caught():
    v = redteam.check_review_reasons_match_output(
        _row(review_reasons="Conflicting sources on buyback fee: €3.5m vs €3m",
             buy_back="3,000,000 EUR", quality_grade="A"))
    assert any(x.check == "review_reason_contradicts_output" for x in v)


def test_a_reported_conflict_is_acceptable_once_the_grade_is_capped():
    assert redteam.check_review_reasons_match_output(
        _row(review_reasons="Conflicting sources on buyback fee: €3.5m vs €3m",
             buy_back="3,000,000 EUR", quality_grade="B")) == []


def test_summary_about_another_player_is_caught():
    v = redteam.check_summary_subject(
        _row(player="Santiago Gimenez",
             deal_summary="Alvaro Morata joined Galatasaray on loan from AC Milan."))
    assert any(x.check == "summary_about_another_player" for x in v)


def test_run_all_on_a_clean_row_returns_nothing():
    clean = _row(transfer_type="permanent_transfer", anchor_event_role="permanent_transfer",
                 family_classification="standalone_permanent_transfer",
                 family_notes="Standalone permanent transfer from A to B",
                 quality_grade="A", n_researched_fields=1,
                 buy_back="3,000,000 EUR", buy_back_status="disclosed_yes",
                 buy_back_scope_event_id="e1",
                 buy_back_evidence_span="Milan activated the buy-back for €3,000,000",
                 economic_mechanism="buy_back_exercise",
                 deal_summary="Test Player moved for a fee.")
    assert redteam.run_all([clean], {"e1": []}, TERMS) == []
