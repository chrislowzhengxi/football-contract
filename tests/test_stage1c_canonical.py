"""Stage 1C tests: the canonical backbone's guarantees.

Unit tests pin the policy rules. Integration tests run against the built
canonical artifact and assert both the named regression cases and that nothing
Stage 1 got right has regressed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.stage1c.policy import (
    FEE_DISCLOSURE,
    LABEL_TO_TYPE,
    classify_type,
    is_youth_or_reserve,
    normalize_club_name,
    research_decision,
    split_fees,
)

CANONICAL = Path("data/outputs/rebuild/stage1c_canonical_transfers.csv")
BACKBONE = Path("data/outputs/rebuild/stage1_normalized_transfers.csv")


# ===========================================================================
# Unit: raw label beats the heuristic
# ===========================================================================

def test_raw_label_overrides_a_contradicting_heuristic():
    # Stage 1's timeline rule said "loan"; Transfermarkt says "free transfer".
    kind, source = classify_type("free_transfer", "loan_out", internal_move=False)
    assert kind == "free_transfer"
    assert source == "transfermarkt_raw_label"


def test_heuristic_is_used_only_when_no_raw_label_exists():
    kind, source = classify_type(None, "loan_out", internal_move=False)
    assert kind == "loan"
    assert source == "stage1_heuristic"


def test_internal_move_refines_only_an_uninformative_label():
    # "-" says nothing about the movement, so the internal rule may speak.
    assert classify_type("no_fee_shown", "no_fee_recorded", internal_move=True) == (
        "youth_or_internal", "club_name_pattern")
    # An explicit loan between a club and its own B team stays a loan.
    assert classify_type("loan_transfer", "loan_out", internal_move=True) == (
        "loan", "transfermarkt_raw_label")
    # So does an explicit paid transfer.
    assert classify_type("paid_transfer", "permanent_with_fee", internal_move=True) == (
        "permanent_transfer", "transfermarkt_raw_label")


def test_every_known_label_maps_to_a_type():
    for label in LABEL_TO_TYPE:
        kind, source = classify_type(label, "zero_fee_move", internal_move=False)
        assert kind and source == "transfermarkt_raw_label"


# ===========================================================================
# Unit: fees never share a column
# ===========================================================================

def test_loan_fee_goes_only_to_loan_fee_eur():
    fees = split_fees("loan_fee", 500_000)
    assert fees["loan_fee_eur"] == 500_000
    assert fees["permanent_transfer_fee_eur"] is None
    assert fees["fee_on_return_eur"] is None
    assert fees["fee_disclosure_status"] == "disclosed_amount"


def test_permanent_fee_goes_only_to_permanent_transfer_fee_eur():
    fees = split_fees("paid_transfer", 32_000_000)
    assert fees["permanent_transfer_fee_eur"] == 32_000_000
    assert fees["loan_fee_eur"] is None
    assert fees["fee_on_return_eur"] is None


def test_a_fee_on_a_return_leg_gets_its_own_field():
    fees = split_fees("end_of_loan_with_fee", 3_200_000)
    assert fees["fee_on_return_eur"] == 3_200_000
    assert fees["permanent_transfer_fee_eur"] is None
    assert fees["loan_fee_eur"] is None


def test_free_transfer_is_the_only_asserted_zero():
    fees = split_fees("free_transfer", 0.0)
    assert fees["permanent_transfer_fee_eur"] == 0.0
    assert fees["fee_disclosure_status"] == "disclosed_free"


@pytest.mark.parametrize("label", ["undisclosed", "no_fee_shown", "loan_transfer", "end_of_loan"])
def test_labels_without_an_amount_never_produce_a_zero(label):
    fees = split_fees(label, None)
    assert fees["permanent_transfer_fee_eur"] is None
    assert fees["loan_fee_eur"] is None
    assert fees["fee_on_return_eur"] is None


def test_undisclosed_and_no_fee_shown_have_different_disclosure_status():
    assert FEE_DISCLOSURE["undisclosed"] == "undisclosed"
    assert FEE_DISCLOSURE["no_fee_shown"] == "no_fee_shown"
    assert FEE_DISCLOSURE["undisclosed"] != FEE_DISCLOSURE["no_fee_shown"]


def test_a_bare_loan_transfer_does_not_assert_a_free_loan():
    # We cannot tell a free loan from a loan whose fee was not published.
    assert FEE_DISCLOSURE["loan_transfer"] == "no_fee_shown"
    assert FEE_DISCLOSURE["loan_transfer"] != "disclosed_free"


# ===========================================================================
# Unit: club shape
# ===========================================================================

@pytest.mark.parametrize("name,expected", [
    ("Barcelona B", True), ("Man City U18", True), ("RM Castilla", True),
    ("Hoffenheim Yth.", True), ("Instituto II", True), ("Olimpia Res.", True),
    ("Barcelona", False), ("B. Banja Luka", False), ("Red Star", False),
    ("Al-Ittihad", False), ("Konyaspor", False), ("Aston Villa", False),
])
def test_youth_or_reserve_pattern(name, expected):
    assert is_youth_or_reserve(name) is expected


def test_sides_of_one_organisation_normalize_to_the_same_base():
    assert normalize_club_name("Barcelona B") == normalize_club_name("Barcelona")
    assert normalize_club_name("Newcastle U21") == normalize_club_name("Newcastle")
    assert normalize_club_name("Braga B") != normalize_club_name("Benfica")


# ===========================================================================
# Unit: research-target policy
# ===========================================================================

def target_row(**overrides):
    row = {
        "transfer_type_normalized": "permanent_transfer",
        "involves_placeholder_club": False, "is_internal_move": False,
        "is_senior_move": True, "has_monetary_amount": True, "return_leg_has_fee": False,
    }
    row.update(overrides)
    return row


def test_a_senior_permanent_transfer_is_a_target():
    assert research_decision(target_row())[0] is True


def test_a_loan_return_without_a_fee_is_excluded():
    included, reason = research_decision(
        target_row(transfer_type_normalized="loan_return", has_monetary_amount=False))
    assert included is False
    assert reason == "loan_return_bookkeeping_row_no_deal_terms"


def test_a_loan_return_WITH_a_fee_is_kept():
    included, _ = research_decision(target_row(
        transfer_type_normalized="loan_return", return_leg_has_fee=True))
    assert included is True


def test_placeholder_and_internal_rows_are_never_targets():
    assert research_decision(target_row(involves_placeholder_club=True))[0] is False
    assert research_decision(target_row(is_internal_move=True))[0] is False


def test_a_published_amount_qualifies_even_from_an_academy_side():
    # Sancho: Man City U18 -> Dortmund, EUR 20.6m. A real transfer.
    included, _ = research_decision(target_row(is_senior_move=False, has_monetary_amount=True))
    assert included is True


def test_a_youth_move_with_no_amount_is_excluded():
    included, reason = research_decision(target_row(
        transfer_type_normalized="free_transfer", is_senior_move=False, has_monetary_amount=False))
    assert included is False
    assert reason == "youth_or_reserve_side_and_no_disclosed_amount"


def test_no_fee_shown_is_excluded_by_default_with_a_stated_reason():
    included, reason = research_decision(target_row(
        transfer_type_normalized="no_fee_shown", has_monetary_amount=False))
    assert included is False
    assert reason == "no_fee_shown_no_evidence_of_a_negotiated_deal"


def test_an_undisclosed_senior_move_is_a_target():
    included, _ = research_decision(target_row(
        transfer_type_normalized="undisclosed_transfer", has_monetary_amount=False))
    assert included is True


def test_every_excluded_row_carries_a_reason():
    for kind in ["loan_return", "youth_or_internal", "other", "unknown", "no_fee_shown"]:
        included, reason = research_decision(
            target_row(transfer_type_normalized=kind, has_monetary_amount=False))
        assert included is False and reason, kind


# ===========================================================================
# Integration: the built canonical artifact
# ===========================================================================

@pytest.fixture(scope="module")
def canonical():
    if not CANONICAL.exists():
        pytest.skip("canonical dataset not built")
    return pd.read_csv(CANONICAL, low_memory=False)


def test_named_regression_checks_all_pass(canonical):
    from src.stage1c.regression import run_checks
    failures = [c["check"] for c in run_checks(canonical) if not c["passed"]]
    assert not failures, failures


@pytest.mark.parametrize("player,date,field,value", [
    ("Silas", "2024-09-03", "loan_fee_eur", 500_000),
    ("Idrissa Gueye", "2025-09-01", "loan_fee_eur", 4_000_000),
    ("Idrissa Gueye", "2026-07-01", "permanent_transfer_fee_eur", 6_000_000),
    ("Donyell Malen", "2026-01-16", "loan_fee_eur", 2_000_000),
    ("Donyell Malen", "2026-07-01", "permanent_transfer_fee_eur", 25_000_000),
    ("Kazeem Olaigbe", "2026-02-04", "loan_fee_eur", 3_020_000),
    ("Roger Fernandes", "2025-09-05", "permanent_transfer_fee_eur", 32_000_000),
    ("Kylian Mbappé", "2024-07-01", "permanent_transfer_fee_eur", 0),
])
def test_hand_verified_amounts(canonical, player, date, field, value):
    rows = canonical[(canonical.player_name == player) & (canonical.transfer_date == date)]
    assert len(rows) == 1
    assert rows.iloc[0][field] == value


def test_future_transfer_flag_survives(canonical):
    assert canonical.transfermarkt_future_transfer.any()
    assert canonical.transfermarkt_future_transfer.dtype == bool


def test_loan_and_loan_return_never_share_a_row(canonical):
    loans = canonical.transfer_type_normalized == "loan"
    returns = canonical.transfer_type_normalized == "loan_return"
    assert not (loans & returns).any()
    assert canonical.loc[returns, "loan_fee_eur"].isna().all()


def test_youth_and_internal_rows_are_identifiable(canonical):
    assert canonical.is_internal_move.any()
    assert canonical.is_youth_or_reserve_side.any()
    assert not canonical.loc[canonical.is_internal_move, "is_research_target"].any()


def test_every_row_has_a_type_and_a_source(canonical):
    assert canonical.transfer_type_normalized.notna().all()
    assert canonical.transfer_type_source.notna().all()
    assert set(canonical.transfer_type_source) <= {
        "transfermarkt_raw_label", "club_name_pattern", "stage1_heuristic"}


def test_every_non_target_has_an_exclusion_reason(canonical):
    excluded = canonical[~canonical.is_research_target]
    assert (excluded.research_exclusion_reason.fillna("") != "").all()
    included = canonical[canonical.is_research_target]
    assert (included.research_exclusion_reason.fillna("") == "").all()


# --- no regression against Stage 1 ----------------------------------------

@pytest.fixture(scope="module")
def backbone():
    if not BACKBONE.exists():
        pytest.skip("Stage 1 backbone not built")
    return pd.read_csv(BACKBONE, low_memory=False)


def test_no_events_gained_or_lost(canonical, backbone):
    assert len(canonical) == len(backbone)
    assert set(canonical.event_id) == set(backbone.event_id)
    assert not canonical.event_id.duplicated().any()


def test_player_clubs_dates_and_market_value_are_unchanged(canonical, backbone):
    joined = backbone.merge(canonical, on="event_id", validate="one_to_one")
    assert len(joined) == len(backbone)
    assert (joined.raw_player_id == joined.player_id).all()
    assert (joined.raw_player_name.fillna("") == joined.player_name.fillna("")).all()
    assert (joined.raw_from_club_id == joined.from_club_id).all()
    assert (joined.raw_to_club_id == joined.to_club_id).all()
    assert (joined.raw_from_club_name.fillna("") == joined.from_club_name.fillna("")).all()
    assert (joined.raw_to_club_name.fillna("") == joined.to_club_name.fillna("")).all()
    assert (pd.to_datetime(joined.raw_transfer_date).dt.strftime("%Y-%m-%d")
            == joined.transfer_date).all()
    assert (joined.raw_market_value_in_eur.fillna(-1) == joined.market_value_eur.fillna(-1)).all()
    assert (joined.derived_age_at_transfer.fillna(-1)
            == joined.age_at_transfer.fillna(-1)).all()


def test_the_original_duckdb_fee_is_preserved_untouched(canonical, backbone):
    joined = backbone.merge(canonical, on="event_id", validate="one_to_one")
    assert (joined.raw_transfer_fee.fillna(-1) == joined.duckdb_transfer_fee.fillna(-1)).all()


def test_permanent_fees_agree_with_the_duckdb_where_comparable(canonical):
    comparable = canonical[canonical.fee_matches_duckdb.notna()]
    assert len(comparable) > 0
    assert comparable.fee_matches_duckdb.astype(bool).all()
