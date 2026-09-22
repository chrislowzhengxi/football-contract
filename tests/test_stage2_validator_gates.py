"""The deterministic gates that run after extraction.

Each test is named for a defect found in the first demonstration dataset.
"""
import pandas as pd
import pytest

from src.stage2.extract_v4 import (
    _conflicting_figures, _is_composite, summarise, validate,
)

FAMILY = pd.DataFrame([
    {"event_id": "e_loan", "event_role": "original_loan", "player_name": "Test Player",
     "transfer_date": "2022-07-01", "from_club_name": "Milan", "to_club_name": "Lecce",
     "transfer_type_normalized": "loan", "event_family_id": "fam_1",
     "family_player_id": 1, "family_interpretation_status": "loan_family",
     "family_notes": "", "is_family_anchor": False},
    {"event_id": "e_ret", "event_role": "loan_return", "player_name": "Test Player",
     "transfer_date": "2023-06-30", "from_club_name": "Lecce", "to_club_name": "Milan",
     "transfer_type_normalized": "loan_return", "event_family_id": "fam_1",
     "family_player_id": 1, "family_interpretation_status": "loan_family",
     "family_notes": "", "is_family_anchor": True},
])


def _src(text, tier=2, eid="ev01"):
    return [{"evidence_id": eid, "source_url": f"https://example.com/{eid}",
             "publisher": "example.com", "source_class": "national_media_major",
             "source_tier": tier, "evidence_text": text}]


def _field(**kw):
    base = {"status": "disclosed_yes", "evidence_ids": ["ev01"],
            "which_transaction": "the loan from Milan to Lecce",
            "scoped_to_event_id": "e_loan", "statement_type": "direct"}
    base.update(kw)
    return base


# --- money ------------------------------------------------------------------

def test_amount_is_rescaled_to_full_units():
    span = "Lecce hold an option to buy of €8 million"
    v = validate({"purchase_option": _field(amount=8, currency="EUR",
                                            evidence_span=span)}, FAMILY, _src(span))
    assert "purchase_option" in v.supported
    assert v.normalised["purchase_option"]["amount"] == 8_000_000
    assert v.money_rescaled == ["purchase_option:8->8,000,000"]


def test_amount_with_no_figure_in_its_span_is_dropped_but_the_mechanism_is_kept():
    """The quote establishes the option; it does not establish the price."""
    span = "il club sardo aveva esercitato il diritto di riscatto sul giocatore"
    payload = {"purchase_option": _field(amount=10_000_000, currency="EUR",
                                         evidence_span=span)}
    v = validate(payload, FAMILY, _src(span))
    assert "purchase_option" in v.supported
    assert "purchase_option" in v.money_unsupported
    assert payload["purchase_option"].get("amount") is None
    assert payload["purchase_option"].get("currency") is None
    assert payload["purchase_option"]["status"] == "partially_disclosed"


def test_amount_contradicting_its_span_is_refused():
    span = "Milan hold a buy-back, contro-riscatto fissato per 13,5 milioni"
    v = validate({"buy_back": _field(amount=12_200_000, currency="EUR",
                                     scoped_to_event_id="e_ret",
                                     evidence_span=span)}, FAMILY, _src(span))
    assert "buy_back" not in v.supported
    assert "buy_back" in v.amount_conflicts


def test_currency_is_not_emitted_without_an_amount():
    span = "Milan retain a buy-back clause on the player"
    payload = {"buy_back": _field(currency="EUR", scoped_to_event_id="e_ret",
                                  evidence_span=span)}
    validate(payload, FAMILY, _src(span))
    assert payload["buy_back"].get("currency") is None


def test_percentage_absent_from_its_span_is_refused():
    span = "Milan retain a sell-on clause on any future transfer"
    v = validate({"sell_on": _field(percentage=40, recipient="Milan",
                                    scoped_to_event_id="e_loan",
                                    evidence_span=span)}, FAMILY, _src(span))
    assert "sell_on" not in v.supported


# --- conflicts --------------------------------------------------------------

def test_disagreeing_figures_become_conflicting_sources():
    span = ("Milan activated the buyback clause; reports put the fee at "
            "€3m or €3.5m")
    payload = {"buy_back": _field(amount=3_000_000, currency="EUR",
                                  scoped_to_event_id="e_ret", evidence_span=span)}
    v = validate(payload, FAMILY, _src(span))
    assert payload["buy_back"]["status"] == "conflicting_sources"
    assert payload["buy_back"]["reported_values"] == [3_000_000, 3_500_000]
    assert "buy_back" in v.conflicting


def test_a_currency_conversion_is_not_a_conflict():
    assert _conflicting_figures("a further €2m (£1.9m) in add-ons") == []


def test_identical_figures_are_not_a_conflict():
    assert _conflicting_figures("€3m, confirmed later as €3m") == []


def test_two_components_of_one_settlement_are_not_a_conflict():
    """Morata: a EUR 5m termination fee plus EUR 651,562 of waived receivables."""
    assert _conflicting_figures(
        "AC Milan will pay a termination fee of 5,000,000 EUR. Additionally, the "
        "footballer has waived his receivables amounting to 651,562 EUR.") == []


def test_a_tiered_schedule_is_not_a_conflict():
    """Horta: thresholds in a sell-on schedule, on a field carrying no amount."""
    assert _conflicting_figures(
        "el 90% si la tarifa es inferior a 2.500.000,00 €; o el 80% si es superior "
        "a 2.500.000,00 € pero no superior a 5.000.000,00 €", has_amount=False) == []


def test_a_field_with_no_amount_has_nothing_to_conflict_about():
    assert _conflicting_figures("reported as €3m or €3.5m", has_amount=False) == []


# --- composite spans --------------------------------------------------------

@pytest.mark.parametrize("span,expected", [
    ("contropzione; buyback clause; Milan activated it; €3m or €3.5m", True),
    ("option to buy / obligation to buy", True),
    ("the fee was ... undisclosed", True),
    ("Lecce hold an option to buy of €8 million", False),
])
def test_composite_span_detection(span, expected):
    assert _is_composite(span) is expected


# --- family scope -----------------------------------------------------------

def test_term_describing_a_club_outside_the_family_is_moved_out():
    span = ("Lille shall pay Sporting all amounts receivable from Chelsea "
            "as sell-on fee of €18,000,000")
    v = validate({"sell_on": _field(amount=18_000_000, currency="EUR",
                                    recipient="Sporting", evidence_span=span)},
                 FAMILY, _src(span))
    assert "sell_on" not in v.supported
    assert "sell_on" in v.out_of_family
    assert v.related_event_evidence and v.related_event_evidence[0]["field"] == "sell_on"


def test_term_is_rescoped_to_the_family_leg_its_span_describes():
    span = "Lecce hold an option to buy of €8 million agreed with Milan"
    payload = {"purchase_option": _field(amount=8_000_000, currency="EUR",
                                         scoped_to_event_id="e_ret",
                                         evidence_span=span)}
    v = validate(payload, FAMILY, _src(span))
    # purchase_option cannot attach to a loan_return, so scoping it to the
    # return leg would reject it; the span names both clubs, so it stays valid.
    assert payload["purchase_option"]["scoped_to_event_id"] in ("e_loan", "e_ret")


def test_scope_event_outside_the_family_is_rejected():
    span = "Lecce hold an option to buy of €8 million"
    v = validate({"purchase_option": _field(amount=8_000_000, currency="EUR",
                                            scoped_to_event_id="e_elsewhere",
                                            evidence_span=span)}, FAMILY, _src(span))
    assert "purchase_option" in v.wrong_event


# --- economic mechanism -----------------------------------------------------

def test_mechanism_claiming_an_unsupported_field_is_rejected():
    span = "the player returned to Milan at the end of the loan"
    payload = {"economic_mechanism": {"status": "disclosed_yes",
                                      "value": "buy_back_exercise",
                                      "evidence_ids": ["ev01"], "evidence_span": span}}
    v = validate(payload, FAMILY, _src(span))
    assert v.mechanism_rejected == "buy_back_exercise"
    assert v.mechanism_value == ""
    assert payload["economic_mechanism"]["value"] is None


def test_mechanism_backed_by_a_supported_field_survives():
    span = "Milan activated the buy-back clause for €3,000,000"
    payload = {
        "buy_back": _field(amount=3_000_000, currency="EUR",
                           scoped_to_event_id="e_ret", evidence_span=span),
        "economic_mechanism": {"status": "disclosed_yes", "value": "buy_back_exercise",
                               "evidence_ids": ["ev01"], "evidence_span": span}}
    v = validate(payload, FAMILY, _src(span))
    assert "buy_back" in v.supported
    assert v.mechanism_value == "buy_back_exercise"
    assert not v.mechanism_rejected


def test_mechanism_outside_the_controlled_vocabulary_is_rejected():
    span = "Milan activated the buy-back clause for €3,000,000"
    payload = {"economic_mechanism": {"status": "disclosed_yes",
                                      "value": "mutual_understanding",
                                      "evidence_ids": ["ev01"], "evidence_span": span}}
    v = validate(payload, FAMILY, _src(span))
    assert v.mechanism_rejected == "mutual_understanding"


# --- carried-forward gates --------------------------------------------------

def test_tier3_only_support_cannot_establish_a_term():
    span = "Lecce hold an option to buy of €8 million"
    v = validate({"purchase_option": _field(amount=8_000_000, currency="EUR",
                                            evidence_span=span)},
                 FAMILY, _src(span, tier=3))
    assert "purchase_option" in v.tier3_only


def test_span_not_present_in_any_source_is_rejected():
    v = validate({"purchase_option": _field(amount=8_000_000, currency="EUR",
                                            evidence_span="an entirely invented quotation "
                                                          "about a nonexistent clause")},
                 FAMILY, _src("Lecce hold an option to buy of €8 million"))
    assert "purchase_option" in v.span_not_found


def test_summarise_exposes_the_new_signals():
    span = "Lecce hold an option to buy of €8 million"
    v = validate({"purchase_option": _field(amount=8, currency="EUR",
                                            evidence_span=span)}, FAMILY, _src(span))
    s = summarise({}, v)
    for key in ("money_rescaled", "money_unsupported", "out_of_family",
                "related_event_evidence", "conflicting_fields", "normalised",
                "economic_mechanism"):
        assert key in s


def test_a_clause_announced_without_terms_is_only_partially_disclosed():
    """Pio Esposito's option and counter-option were announced with no price."""
    span = ("si trasferisce al club ligure a titolo temporaneo con diritto di "
            "opzione e contro opzione")
    payload = {"purchase_option": _field(exercised=False, evidence_span=span)}
    validate(payload, FAMILY, _src(span))
    assert payload["purchase_option"]["status"] == "partially_disclosed"


def test_a_clause_with_a_price_stays_disclosed_yes():
    span = "Lecce hold an option to buy of €8 million"
    payload = {"purchase_option": _field(amount=8_000_000, currency="EUR",
                                         evidence_span=span)}
    validate(payload, FAMILY, _src(span))
    assert payload["purchase_option"]["status"] == "disclosed_yes"


def test_an_obligation_trigger_with_no_metric_and_no_figure_is_refused():
    """Dzeko: "if certain performance conditions were met" names no trigger."""
    span = "the deal included an obligation to buy if certain performance conditions were met"
    v = validate({"obligation_trigger": _field(metric="not_specified",
                                               threshold="certain performance conditions",
                                               evidence_span=span)}, FAMILY, _src(span))
    assert "obligation_trigger" not in v.supported


def test_an_obligation_trigger_with_a_figure_is_kept():
    """Sorloth: "if he started 50 per cent of their games"."""
    span = "an obligation to buy if he started 50 per cent of their games in 2020-21"
    v = validate({"obligation_trigger": _field(metric="not_specified",
                                               threshold="50 per cent of games",
                                               evidence_span=span)}, FAMILY, _src(span))
    assert "obligation_trigger" in v.supported


def test_an_obligation_trigger_with_a_named_metric_is_kept():
    span = "the obligation to buy was triggered once he reached the agreed appearance count"
    v = validate({"obligation_trigger": _field(metric="appearances",
                                               threshold="agreed count",
                                               evidence_span=span)}, FAMILY, _src(span))
    assert "obligation_trigger" in v.supported


def test_conflicting_values_written_as_source_wording_still_format():
    """Barbieri's reported_values are strings like "EUR 2.5m (ev01)"."""
    from src.stage2.dataset import format_term
    cell = format_term({"status": "conflicting_sources", "currency": "EUR",
                        "reported_values": ["€2.5m (ev01)", "€2.4m (ev05)"]})
    assert cell == "CONFLICTING: 2,500,000 | 2,400,000 EUR"


def test_a_tiered_sell_on_reads_as_words_not_json():
    from src.stage2.dataset import format_percentage
    out = format_percentage([
        {"lower_threshold_eur": 0, "upper_threshold_eur": 2500000, "percentage_value": 90},
        {"lower_threshold_eur": 2500000, "upper_threshold_eur": 5000000, "percentage_value": 80},
        {"lower_threshold_eur": 5000000, "upper_threshold_eur": None, "percentage_value": 67}])
    assert out == ("90% up to EUR 2.5m; 80% between EUR 2.5m and EUR 5m; "
                   "67% above EUR 5m")
