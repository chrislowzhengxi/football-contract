"""Stage 1B parser tests.

The one thing this parser must never do is what the upstream dbt model does:
collapse a labelled fee into a bare number. These tests pin that separation,
plus the exact label vocabulary observed across all 135,065 raw rows.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.stage1b.parse import (
    classify_label,
    club_id_from_href,
    parse_amount_eur,
    parse_transfer_row,
    transfer_id_from_url,
)

LOAN_FEE_500K = 'Loan fee:<br /><i class="normaler-text">€500k</i>'
END_OF_LOAN_FEE = 'End of loan<br /><i class="normaler-text">€3.20m</i>'


# --- the label vocabulary --------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("-", "no_fee_shown"),
    ("?", "undisclosed"),
    ("free transfer", "free_transfer"),
    ("loan transfer", "loan_transfer"),
    ("End of loan", "end_of_loan"),
    ("draft", "draft"),
    ("€23.00m", "paid_transfer"),
    ("€500k", "paid_transfer"),
    ("€1", "paid_transfer"),
    (LOAN_FEE_500K, "loan_fee"),
    (END_OF_LOAN_FEE, "end_of_loan_with_fee"),
])
def test_known_labels_classify(text, expected):
    assert classify_label(text) == expected


@pytest.mark.parametrize("text", [None, "", "   ", "something new from transfermarkt"])
def test_unrecognised_text_is_unknown_not_guessed(text):
    assert classify_label(text) == "unknown"


def test_loan_fee_is_never_confused_with_a_paid_transfer():
    assert classify_label(LOAN_FEE_500K) != classify_label("€500k")


# --- amount parsing --------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("€500k", 500_000.0),
    ("€23.00m", 23_000_000.0),
    ("€1.50m", 1_500_000.0),
    ("€3.02m", 3_020_000.0),
    (LOAN_FEE_500K, 500_000.0),
    (END_OF_LOAN_FEE, 3_200_000.0),
])
def test_amounts_parse(text, expected):
    assert parse_amount_eur(text) == expected


@pytest.mark.parametrize("text", ["-", "?", "loan transfer", "End of loan", "draft", None])
def test_labels_without_an_amount_return_none_not_zero(text):
    # None and 0.0 mean different things. Returning 0 here is exactly the
    # upstream bug.
    assert parse_amount_eur(text) is None


# --- the separation that matters ------------------------------------------

def row(fee, market_value="€10.00m"):
    return parse_transfer_row(1, "P", {
        "fee": fee, "marketValue": market_value, "season": "25/26",
        "dateUnformatted": "2025-08-01", "date": "01/08/2025",
        "from": {"clubName": "A", "href": "/a/transfers/verein/10/saison_id/2025"},
        "to": {"clubName": "B", "href": "/b/transfers/verein/20/saison_id/2025"},
        "url": "/p/transfers/spieler/1/transfer_id/999", "futureTransfer": 0, "upcoming": False,
    })


def test_loan_fee_keeps_both_the_label_and_the_amount():
    parsed = row(LOAN_FEE_500K)
    assert parsed["transfer_type_raw"] == "loan_fee"
    assert parsed["fee_numeric_eur"] == 500_000.0
    assert parsed["fee_amount_role"] == "loan_fee"
    assert parsed["fee_display_raw"] == LOAN_FEE_500K   # byte-exact original
    assert parsed["is_loan_related"] is True


def test_a_permanent_fee_of_the_same_size_is_a_different_role():
    loan, permanent = row(LOAN_FEE_500K), row("€500k")
    assert loan["fee_numeric_eur"] == permanent["fee_numeric_eur"] == 500_000.0
    assert loan["fee_amount_role"] == "loan_fee"
    assert permanent["fee_amount_role"] == "permanent_fee"


def test_free_transfer_is_the_only_label_that_asserts_zero():
    assert row("free transfer")["fee_numeric_eur"] == 0.0
    for text in ["loan transfer", "End of loan", "-", "?", "draft"]:
        assert row(text)["fee_numeric_eur"] is None, text


def test_undisclosed_and_no_fee_shown_stay_distinct():
    # The DuckDB stores NULL for both. Stage 1B must not.
    assert row("?")["transfer_type_raw"] != row("-")["transfer_type_raw"]


def test_end_of_loan_with_a_fee_is_not_an_ordinary_end_of_loan():
    parsed = row(END_OF_LOAN_FEE)
    assert parsed["transfer_type_raw"] == "end_of_loan_with_fee"
    assert parsed["fee_amount_role"] == "fee_on_return"
    assert parsed["fee_numeric_eur"] == 3_200_000.0


def test_loan_related_flag_covers_every_loan_label():
    for text in ["loan transfer", "End of loan", LOAN_FEE_500K, END_OF_LOAN_FEE]:
        assert row(text)["is_loan_related"] is True, text
    for text in ["free transfer", "€5.00m", "-", "?", "draft"]:
        assert row(text)["is_loan_related"] is False, text


# --- identifiers -----------------------------------------------------------

def test_club_and_transfer_ids_are_extracted_from_hrefs():
    assert club_id_from_href("/al-orooba/transfers/verein/51771/saison_id/2025") == 51771
    assert club_id_from_href(None) is None
    assert transfer_id_from_url("/witi/transfers/spieler/352425/transfer_id/5811259") == 5811259
    assert transfer_id_from_url(None) is None


def test_row_carries_the_fields_upstream_drops():
    parsed = row("€5.00m")
    assert parsed["transfermarkt_transfer_id"] == 999
    assert parsed["from_club_id"] == 10 and parsed["to_club_id"] == 20
    assert parsed["future_transfer"] is False


# --- against the produced artifacts ---------------------------------------

ARTIFACT = "data/outputs/rebuild/stage1b_transfermarkt_page_rows.csv"
COMPARISON = "data/outputs/rebuild/stage1b_comparison.csv"


@pytest.fixture(scope="module")
def artifacts():
    from pathlib import Path
    if not (Path(ARTIFACT).exists() and Path(COMPARISON).exists()):
        pytest.skip("Stage 1B artifacts not built")
    return pd.read_csv(ARTIFACT), pd.read_csv(COMPARISON)


def test_manually_audited_loan_fees_were_recovered(artifacts):
    """The four loan fees read off the Transfermarkt website by hand."""
    rows, _ = artifacts
    expected = {
        ("Silas", "2024-09-03"): 500_000.0,
        ("Idrissa Gueye", "2025-09-01"): 4_000_000.0,
        ("Donyell Malen", "2026-01-16"): 2_000_000.0,
        ("Kazeem Olaigbe", "2026-02-04"): 3_020_000.0,
    }
    for (player, date), amount in expected.items():
        match = rows[(rows.player_name == player) & (rows.transfer_date == date)]
        assert len(match) == 1, f"{player} {date}"
        assert match.iloc[0]["transfer_type_raw"] == "loan_fee"
        assert match.iloc[0]["fee_numeric_eur"] == amount


def test_backbone_agrees_with_raw_on_clubs_dates_and_market_value(artifacts):
    _, comparison = artifacts
    matched = comparison[comparison.matched_in_raw_api]
    for column in ["field_match_from_club", "field_match_to_club",
                   "field_match_date", "field_match_market_value"]:
        assert matched[column].all(), column


def test_every_fee_disagreement_is_a_collapsed_label_not_a_wrong_number(artifacts):
    _, comparison = artifacts
    matched = comparison[comparison.matched_in_raw_api]
    disagree = matched[~matched.field_match_fee.astype(bool)]
    # If a permanent fee ever disagreed, the DuckDB would be untrustworthy for
    # the field we rely on most.
    assert not (disagree.website_raw_label == "paid_transfer").any()
    assert set(disagree.website_raw_label) <= {
        "loan_transfer", "end_of_loan", "loan_fee", "end_of_loan_with_fee", "draft"}
