"""Monetary normalisation.

Named for the two cases that motivated it: Morata's purchase option was
recorded as `amount=8` from the span "a buy option of EUR 8 million", and
Sorloth's add-ons as `amount=2` from "a further EUR 2m payable in add-ons".
Both were six orders of magnitude out and both looked fine in a spreadsheet.
"""
import pytest

from src.stage2.money import (
    find_all, normalise_amount, normalise_percentage, parse_number,
)


@pytest.mark.parametrize("raw,expected", [
    ("8", 8.0), ("13,5", 13.5), ("13.5", 13.5),
    ("12,200,000", 12200000.0), ("1.200.000", 1200000.0),
    ("1.200", 1200.0), ("0.5", 0.5), ("2.825", 2825.0),
    ("46,500,000", 46500000.0), ("1 000", None),
])
def test_number_conventions(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize("text,value,currency", [
    ("a buy option of €8 million", 8_000_000, "EUR"),
    ("€8m", 8_000_000, "EUR"),
    ("€2m payable in add-ons", 2_000_000, "EUR"),
    ("will pay €12m to AC Milan", 12_000_000, "EUR"),
    ("£1.9m", 1_900_000, "GBP"),
    ("$4.5m", 4_500_000, "USD"),
    ("13,5 milioni", 13_500_000, None),
    ("20 millones de euros", 20_000_000, None),
    ("5 mln", 5_000_000, None),
    ("2,5 Mio", 2_500_000, None),
    ("500k", 500_000, None),
    ("200 thousand", 200_000, None),
    ("€16,000,000", 16_000_000, "EUR"),
    ("EUR 2,825,000", 2_825_000, "EUR"),
])
def test_money_expressions(text, value, currency):
    found = find_all(text)
    assert found, text
    assert found[0].value == pytest.approx(value)
    if currency:
        assert found[0].currency == currency


def test_morata_purchase_option_is_rescaled_to_full_units():
    n = normalise_amount(8, "EUR",
                         "his contract had a buy option of €8 million")
    assert n.ok and n.status == "rescaled"
    assert n.amount == 8_000_000
    assert n.currency == "EUR"
    assert n.raw_text == "€8 million"


def test_morata_purchase_obligation_is_rescaled_to_full_units():
    n = normalise_amount(12, "EUR", "The Lombardy club will pay €12m to AC Milan")
    assert n.amount == 12_000_000 and n.currency == "EUR"


def test_sorloth_add_ons_are_rescaled_to_full_units():
    n = normalise_amount(
        2, "EUR",
        "with a further €2m (£1.9m) payable in add-ons to be split between "
        "Palace and Trabzonspor")
    assert n.amount == 2_000_000 and n.currency == "EUR"


def test_an_amount_already_in_full_units_is_left_alone():
    n = normalise_amount(2_825_000, "EUR", "compensation of €2,825,000 was agreed")
    assert n.status == "ok" and n.amount == 2_825_000 and n.scale_applied == 1.0


def test_amount_absent_from_its_own_span_is_refused():
    """Sottil's option was recorded as EUR 10m from a span with no figure."""
    n = normalise_amount(10_000_000, "EUR",
                         "il club sardo aveva esercitato il diritto di riscatto sul giocatore")
    assert not n.ok and n.status == "no_figure_in_span" and n.amount is None


def test_amount_contradicting_its_own_span_is_refused():
    """Sottil's buy-back: 12.2m recorded against a span quoting 13,5 milioni."""
    n = normalise_amount(12_200_000, "EUR", "contro-riscatto fissato per 13,5 milioni")
    assert not n.ok and n.status == "contradicts_span" and n.amount is None


def test_never_emits_a_currency_without_an_amount():
    n = normalise_amount(None, "EUR", "some text")
    assert n.amount is None and n.raw_text is None


def test_percentage_must_appear_in_its_span():
    assert normalise_percentage(50, "Malaga retain 50% of any future sale")[0] == 50
    assert normalise_percentage(50, "Malaga retain a share of any future sale")[0] is None
    assert normalise_percentage(90, "entitled to 40% of the sell-on")[0] is None


def test_multiple_incompatible_figures_are_all_found():
    """Leao's award quotes both the FIFA DRC and the CAS sum."""
    found = find_all("LOSC ordered to pay Sporting €16,000,000 (FIFA DRC) or "
                     "€46,500,000 (CAS, comprising €16m + €30.5m additional)")
    assert {m.value for m in found} >= {16_000_000, 46_500_000, 30_500_000}


def test_a_dot_before_a_multiplier_is_a_decimal_point():
    """Weghorst: "EUR 2.825m" is 2,825,000, not 2.8 billion."""
    assert find_all("a termination fee of €2.825m")[0].value == 2_825_000


def test_a_dot_group_without_a_multiplier_is_still_a_thousands_separator():
    assert parse_number("1.200") == 1200
    assert parse_number("1.200.000") == 1_200_000


def test_other_scaled_decimals_are_unaffected():
    assert find_all("€1.5m")[0].value == 1_500_000
    assert find_all("€13.75m")[0].value == 13_750_000
    assert find_all("€2.8bn")[0].value == 2_800_000_000
