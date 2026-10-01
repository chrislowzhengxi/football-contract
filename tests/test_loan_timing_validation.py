"""Strict A -> B loan, B -> A return, A -> B permanent validation."""
import pandas as pd

from src.analysis import loan_timing_validation as v
from src.analysis.loan_episodes import build_universe

from tests.test_loan_episodes import history, real  # noqa: F401  (fixture)

NO_GEO = pd.DataFrame(columns=["club_id", "country", "confederation"])


def strict(*moves):
    return v.strict_sequence(build_universe(history(*moves), NO_GEO))


def test_exact_sequence_is_origin_a_with_no_intervening_rows():
    S = strict(("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"),
               ("2024-07-01", 1, 2, "permanent_transfer"))
    r = S.iloc[0]
    assert r.origin == "A" and r.return_exact_b_to_a and r.intervening_rows == 0
    assert r.calendar_immediate_1d and not r.may31_to_jul1_same_year


def test_other_side_of_the_lender_is_origin_b_and_the_skipped_row_is_counted():
    """Loan out of Atalanta U19; the permanent move comes from Atalanta after an internal registration."""
    S = strict(("2023-09-01", 4, 2, "loan"), ("2024-06-30", 2, 4, "loan_return"),
               ("2024-06-30", 4, 1, "youth_or_internal"), ("2024-07-01", 1, 2, "permanent_transfer"))
    r = S.iloc[0]
    assert r.origin == "B" and r.intervening_rows == 1
    assert r.intervening_row_types == "youth or internal registration"


def test_british_31_may_to_1_july_is_flagged_only_in_the_same_year():
    same = strict(("2023-09-01", 1, 2, "loan"), ("2024-05-31", 2, 1, "loan_return"),
                  ("2024-07-01", 1, 2, "permanent_transfer")).iloc[0]
    later = strict(("2023-09-01", 1, 2, "loan"), ("2024-05-31", 2, 1, "loan_return"),
                   ("2025-07-01", 1, 2, "permanent_transfer")).iloc[0]
    assert same.next_move_gap_days == 31 and same.calendar_immediate_with_may31 and not same.calendar_immediate_1d
    assert not later.calendar_immediate_with_may31


def test_real_data_origins_partition_the_cases(real):  # noqa: F811
    N, S = v.compute(real["u"])
    for r in N["split"]:
        assert r["A"] + r["B"] + r["C"] == r["N"]
    assert N["split"][1]["N"] == N["current_3816"]
    assert (S.next_move_gap_days >= 0).all()
    assert (S.loc[S.origin == "A", "next_move_from_club_id"] == S.loc[S.origin == "A", "lender_club_id"]).all()
