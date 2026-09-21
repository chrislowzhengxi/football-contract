"""Stage 1 tests.

These pin the properties the backbone must not lose: stable ids, correct player
identity, correct club direction, preserved dates and fees, honest market-value
matching, and above all that nothing is invented - no transfer type, no fee, no
date precision that the raw data does not support.

Tests that need the real snapshot are skipped when it is absent.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import DEFAULT_DATABASE
from src.enrich_structured import stable_event_id
from src.stage1.backbone import add_event_chains, build_backbone, filter_events
from src.stage1.movement import (
    MAX_LOAN_DAYS,
    classify_movements,
    fee_semantics,
    order_player_timeline,
)

HISTORICAL = Path("data/processed/outputs_previous_pipeline/structured_transfers.csv")


def transfers(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"])
    return frame


def row(player, date, frm, to, fee, frm_name=None, to_name=None) -> dict:
    return {
        "player_id": player, "transfer_date": date, "transfer_season": "24/25",
        "from_club_id": frm, "to_club_id": to,
        "from_club_name": frm_name or f"club{frm}", "to_club_name": to_name or f"club{to}",
        "transfer_fee": fee, "market_value_in_eur": 1_000_000.0, "player_name": f"P{player}",
    }


def classified(rows: list[dict]) -> pd.DataFrame:
    return classify_movements(order_player_timeline(transfers(rows)))


# --- event identity -------------------------------------------------------

def test_event_id_recipe_is_pinned_to_the_timestamp_string_form():
    # The hash input stringifies a pandas Timestamp. Passing a plain date string
    # produces a different id, which would silently orphan every earlier result.
    assert stable_event_id(29241, pd.Timestamp("2026-01-01"), 2462, 720) == "tm_78e3733d67be84096fe7"
    assert stable_event_id(29241, "2026-01-01", 2462, 720) != "tm_78e3733d67be84096fe7"


def test_event_id_is_stable_across_calls():
    assert stable_event_id(1, pd.Timestamp("2024-07-01"), 10, 20) == stable_event_id(
        1, pd.Timestamp("2024-07-01"), 10, 20)


def test_event_id_distinguishes_direction():
    assert stable_event_id(1, pd.Timestamp("2024-07-01"), 10, 20) != stable_event_id(
        1, pd.Timestamp("2024-07-01"), 20, 10)


# --- ordering -------------------------------------------------------------

def test_same_day_rows_are_chained_by_club_continuity():
    # Loan return on the same day as the next departure. Date alone cannot order
    # these; club continuity can.
    ordered = order_player_timeline(transfers([
        row(1, "2024-01-01", 10, 20, 0.0),
        row(1, "2024-06-01", 20, 10, 0.0),
        row(1, "2024-06-01", 10, 30, 0.0),
    ]))
    assert list(ordered.to_club_id) == [20, 10, 30]
    assert not ordered.derived_same_date_order_ambiguous.any()


def test_unchainable_same_day_rows_are_flagged_not_guessed():
    ordered = order_player_timeline(transfers([
        row(1, "2024-06-01", 40, 50, 0.0),
        row(1, "2024-06-01", 60, 70, 0.0),
    ]))
    assert ordered.derived_same_date_order_ambiguous.all()


# --- loan classification --------------------------------------------------

def test_reverse_direction_fee_free_pair_is_a_loan():
    out = classified([
        row(1, "2024-08-01", 10, 20, 0.0),
        row(1, "2025-06-30", 20, 10, 0.0),
    ])
    assert list(out.derived_movement_class) == ["loan_out", "loan_return"]
    assert out.derived_movement_basis.tolist() == ["derived_heuristic"] * 2
    assert out.derived_loan_duration_days.tolist() == [333.0, 333.0]


def test_a_fee_on_either_leg_blocks_loan_classification():
    # Ajax -> Man Utd for money, then Man Utd -> Ajax for money, is two permanent
    # deals, not a loan. Both legs must stay permanent_with_fee.
    out = classified([
        row(1, "2014-09-01", 10, 20, 17_500_000.0),
        row(1, "2015-06-30", 20, 10, 16_000_000.0),
    ])
    assert list(out.derived_movement_class) == ["permanent_with_fee", "permanent_with_fee"]


def test_return_leg_outside_the_window_is_not_a_loan():
    out = classified([
        row(1, "2020-01-01", 10, 20, 0.0),
        row(1, "2024-01-01", 20, 10, 0.0),
    ])
    assert "loan_out" not in set(out.derived_movement_class)


def test_loan_window_boundary_is_inclusive():
    start = pd.Timestamp("2024-01-01")
    out = classified([
        row(1, start, 10, 20, 0.0),
        row(1, start + pd.Timedelta(days=MAX_LOAN_DAYS), 20, 10, 0.0),
    ])
    assert out.derived_movement_class.iloc[0] == "loan_out"


def test_loan_out_and_loan_return_counts_always_match():
    out = classified([
        row(1, "2023-08-01", 10, 20, 0.0),
        row(1, "2024-06-30", 20, 10, 0.0),
        row(1, "2024-08-01", 10, 30, 0.0),
        row(1, "2025-06-30", 30, 10, 0.0),
    ])
    counts = out.derived_movement_class.value_counts()
    assert counts["loan_out"] == counts["loan_return"] == 2


def test_loans_do_not_pair_across_different_players():
    out = classified([
        row(1, "2024-08-01", 10, 20, 0.0),
        row(2, "2025-06-30", 20, 10, 0.0),
    ])
    assert set(out.derived_movement_class) == {"zero_fee_move"}


def test_permanent_after_loan_is_a_flag_not_a_class():
    out = classified([
        row(1, "2017-08-31", 10, 20, 0.0),
        row(1, "2018-06-30", 20, 10, 0.0),
        row(1, "2018-07-01", 10, 20, 180_000_000.0),
    ])
    assert list(out.derived_movement_class) == ["loan_out", "loan_return", "permanent_with_fee"]
    assert list(out.derived_follows_loan_return) == [False, False, True]
    assert out.derived_days_after_loan_return.iloc[2] == 1


# --- pseudo clubs ---------------------------------------------------------

def test_pseudo_club_markers_win_over_the_loan_rules():
    # Without Club -> club -> Without Club is free agency, never a loan.
    out = classified([
        row(1, "2024-01-01", 10, 20, None, frm_name="Without Club"),
        row(1, "2024-08-01", 20, 10, None, to_name="Without Club"),
    ])
    assert list(out.derived_movement_class) == ["arrival_from_no_club", "exit_to_no_club"]
    assert set(out.derived_movement_basis) == {"explicit_raw_marker"}


def test_retirement_is_read_from_the_placeholder_club():
    out = classified([row(1, "2024-01-01", 10, 123, None, to_name="Retired")])
    assert out.derived_movement_class.iloc[0] == "retirement"


# --- nothing is invented --------------------------------------------------

def test_no_transfer_type_column_is_produced():
    out = classified([row(1, "2024-01-01", 10, 20, 5_000_000.0)])
    assert "transfer_type" not in out.columns


def test_fee_is_never_invented_or_rewritten():
    out = classified([
        row(1, "2024-01-01", 10, 20, None),
        row(2, "2024-01-01", 10, 20, 0.0),
        row(3, "2024-01-01", 10, 20, 7_500_000.0),
    ])
    # classify_movements works on the raw frame, so the column is still transfer_fee here.
    assert pd.isna(out.transfer_fee.iloc[0])
    assert out.transfer_fee.iloc[1] == 0.0
    assert out.transfer_fee.iloc[2] == 7_500_000.0


def test_fee_semantics_keeps_zero_and_null_apart():
    assert fee_semantics(None) == "no_fee_recorded"
    assert fee_semantics(0.0) == "zero_recorded"
    assert fee_semantics(1.0) == "fee_reported"


def test_zero_fee_without_a_return_leg_is_not_called_a_free_transfer():
    out = classified([row(1, "2024-07-01", 10, 20, 0.0)])
    assert out.derived_movement_class.iloc[0] == "zero_fee_move"
    assert out.derived_movement_basis.iloc[0] == "raw_fee_value"


# --- chains ---------------------------------------------------------------

def test_event_chain_links_an_immediate_onward_move():
    events = pd.DataFrame({
        "raw_player_id": [1, 1, 1],
        "raw_transfer_date": pd.to_datetime(["2024-06-30", "2024-07-01", "2025-07-01"]),
        "raw_from_club_id": [20, 10, 10],
        "raw_to_club_id": [10, 30, 40],
        "event_id": ["tm_a", "tm_b", "tm_c"],
    })
    chains = add_event_chains(events).derived_event_chain_id.tolist()
    assert chains[0] == chains[1] != chains[2]


# --- against the real snapshot -------------------------------------------

@pytest.fixture(scope="module")
def backbone():
    if not DEFAULT_DATABASE.exists():
        pytest.skip("Transfermarkt snapshot not present")
    from src.load_data import connect_database
    return build_backbone(connect_database(DEFAULT_DATABASE, read_only=True))


def test_snapshot_row_count_is_preserved_exactly(backbone):
    # The legacy parser inflates 175,165 raw rows to 175,401 through a
    # many-to-many market-value merge. Stage 1 must not.
    from src.load_data import connect_database
    connection = connect_database(DEFAULT_DATABASE, read_only=True)
    raw_rows = connection.execute("SELECT count(*) FROM transfers").fetchone()[0]
    assert len(backbone) == raw_rows


def test_snapshot_event_ids_are_unique(backbone):
    assert not backbone.event_id.duplicated().any()


def test_snapshot_reproduces_every_historical_event_id(backbone):
    if not HISTORICAL.exists():
        pytest.skip("historical output not present")
    historical = pd.read_csv(HISTORICAL)
    assert historical.event_id.isin(set(backbone.event_id)).all()


def test_snapshot_preserves_club_direction_and_fee(backbone):
    if not HISTORICAL.exists():
        pytest.skip("historical output not present")
    historical = pd.read_csv(HISTORICAL)
    joined = historical.merge(backbone, on="event_id")
    assert len(joined) == len(historical)
    assert (joined.from_club_id == joined.raw_from_club_id).all()
    assert (joined.to_club_id == joined.raw_to_club_id).all()
    assert (joined.transfer_fee.fillna(-1) == joined.raw_transfer_fee.fillna(-1)).all()
    assert (joined.age_at_transfer.fillna(-1) == joined.derived_age_at_transfer.fillna(-1)).all()


def test_snapshot_dates_are_copied_not_reconstructed(backbone):
    from src.load_data import connect_database
    connection = connect_database(DEFAULT_DATABASE, read_only=True)
    raw = connection.execute(
        "SELECT player_id, transfer_date, from_club_id, to_club_id FROM transfers"
    ).fetchdf()
    raw["transfer_date"] = pd.to_datetime(raw["transfer_date"])
    key = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
    left = backbone.rename(columns={
        "raw_player_id": "player_id", "raw_transfer_date": "transfer_date",
        "raw_from_club_id": "from_club_id", "raw_to_club_id": "to_club_id",
    })[key]
    assert len(left.merge(raw, on=key)) == len(raw)


def test_snapshot_market_value_is_not_backfilled(backbone):
    from src.load_data import connect_database
    connection = connect_database(DEFAULT_DATABASE, read_only=True)
    raw_nulls = connection.execute(
        "SELECT count(*) FROM transfers WHERE market_value_in_eur IS NULL"
    ).fetchone()[0]
    assert int(backbone.raw_market_value_in_eur.isna().sum()) == raw_nulls


def test_snapshot_asof_market_value_never_looks_ahead(backbone):
    lookup = backbone.dropna(subset=["derived_mv_asof_date"])
    assert (pd.to_datetime(lookup.derived_mv_asof_date) <= lookup.raw_transfer_date).all()


def test_snapshot_loan_legs_never_carry_a_fee(backbone):
    legs = backbone[backbone.derived_movement_class.isin(["loan_out", "loan_return"])]
    assert (legs.raw_transfer_fee.fillna(0) == 0).all()


def test_snapshot_loan_pairs_are_symmetric(backbone):
    counts = backbone.derived_movement_class.value_counts()
    assert counts["loan_out"] == counts["loan_return"]


def test_snapshot_every_row_has_a_basis(backbone):
    assert backbone.derived_movement_basis.isin(
        {"explicit_raw_marker", "raw_fee_value", "derived_heuristic"}).all()
    assert backbone.derived_movement_class.notna().all()


def test_filter_events_matches_club_names_case_insensitively(backbone):
    selected = filter_events(backbone, ["porto", "BENFICA"], ["24/25", "25/26"])
    assert len(selected) > 0
    assert selected.raw_transfer_season.isin({"24/25", "25/26"}).all()
    sides = selected.raw_from_club_name.str.casefold().isin({"porto", "benfica"}) | \
        selected.raw_to_club_name.str.casefold().isin({"porto", "benfica"})
    assert sides.all()
