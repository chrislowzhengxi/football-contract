"""Exploratory loan-timing run: the unwindowed next move, the parameterised
re-classification, the A-F regrouping, and the calendar-date helpers."""
import pandas as pd
import pytest

from src.analysis import loan_timing_exploration as lte
from src.analysis.loan_episodes import build_universe

from tests.test_loan_episodes import history, real  # noqa: F401  (fixture)

NO_GEO = pd.DataFrame(columns=["club_id", "country", "confederation"])


def table(*moves):
    u = build_universe(history(*moves), NO_GEO)
    return u, lte.episode_table(u, lte.next_moves(u))


def test_next_move_is_found_past_the_window_and_skips_internal_rows():
    u, T = table(("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"),
                 ("2024-07-05", 4, 1, "youth_or_internal"), ("2024-09-15", 1, 2, "permanent_transfer"))
    r = T.iloc[0]
    assert r.next_move_gap_days == 77 and r.next_move_is_permanent_to_borrower
    assert pd.isna(r.current_follow_on_event_id)                 # beyond the 60-day production window
    assert lte.check_follow_on(u.loans, lte.next_moves(u)) == 0
    assert r.economic_ending == "ordinary_end_of_loan" and r.sequence_category == "A"
    assert lte.reclassify(T).iloc[0] == "ordinary_end_of_loan"
    assert lte.reclassify(T, window=None).iloc[0] == "purchase_option_or_permanent_conversion"


def test_a_release_is_recorded_as_the_next_move_but_is_never_a_follow_on():
    u, T = table(("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"),
                 ("2024-07-01", 1, 9, "free_transfer"))
    r = T.iloc[0]
    assert r.next_move_involves_placeholder and r.next_move_kind == "to_placeholder"
    assert lte.reclassify(T, window=None, immediate=None).iloc[0] == "ordinary_end_of_loan"


def test_windows_reproduce_the_production_rules_on_small_histories():
    cases = [
        [("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"), ("2024-07-01", 1, 2, "permanent_transfer")],
        [("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"), ("2024-07-01", 1, 3, "permanent_transfer")],
        [("2023-09-01", 1, 2, "loan"), ("2024-01-31", 2, 1, "loan_return"), ("2024-02-01", 1, 5, "free_transfer")],
        [("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"), ("2024-08-10", 1, 5, "loan")],
        [("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return", "End of loan €1.00m", 1e6)],
    ]
    for moves in cases:
        _, T = table(*moves)
        assert (lte.reclassify(T) == T.economic_ending).all(), moves


def test_real_data_reproduces_production_and_partitions_every_loan(real):  # noqa: F811
    u = real["u"]
    nm = lte.next_moves(u)
    assert lte.check_follow_on(u.loans, nm) == 0
    T = lte.episode_table(u, nm)
    assert (lte.reclassify(T) == T.economic_ending).all()
    assert T.sequence_category.value_counts().sum() == len(u.loans)
    conv = T.economic_ending.eq("purchase_option_or_permanent_conversion")
    assert conv.sum() == (T.sequence_category.eq("B").sum()
                          + (conv & T.sequence_category.eq("E")).sum())
    assert (T.next_move_gap_days.dropna() >= 0).all()
    B = T[T.sequence_category.eq("B")]
    assert (B.current_follow_on_gap_days <= lte.FOLLOW_ON_WINDOW_DAYS).all()
    assert B.next_move_is_permanent_to_borrower.all()


def test_calendar_date_helpers():
    dates = pd.to_datetime(["2020-06-30"] * 6 + ["2021-06-30"] * 2 + ["2020-12-31"] * 1 + ["2020-05-31"] * 1)
    fr = pd.DataFrame({"ending_date": dates, "ending_mmdd": dates.strftime("%m-%d")})
    d = lte.mmdd_distribution(fr)
    assert d.mmdd.tolist()[0] == "06-30" and d["count"].iloc[0] == 8 and d.years_observed.iloc[0] == 2
    assert lte.dates_needed(d, 0.8) == 1
    assert lte.dates_needed(d, 0.9) == 2 and lte.dates_needed(d, 1.0) == 3


def test_lag_stats_uses_closed_windows():
    s = lte.lag_stats(pd.Series([0, 1, 1, 2, 60, 61]))
    assert s["within_0"] == 1 and s["within_1"] == 3 and s["within_60"] == 5
    assert s["median"] == 1 and s["negative"] == 0
