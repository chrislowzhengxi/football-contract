"""Category C split, country/date reconciliation, and the playing-time join.

Synthetic cases pin the rules; real-data checks reconcile counts with the
existing loan universe and guard against reading missing coverage as zero.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis import loan_category_c_split as csplit
from src.analysis import loan_country_dates as cdates
from src.analysis import playing_time_join as ptj
from src.analysis.loan_episodes import build_universe
from src.analysis.loan_timing_exploration import episode_table, next_moves

from tests.test_loan_episodes import history, real  # noqa: F401  (fixture)

NO_GEO = pd.DataFrame(columns=["club_id", "country", "confederation"])


def c_subgroup(*moves):
    u = build_universe(history(*moves), NO_GEO)
    C = csplit.split(episode_table(u, next_moves(u)))
    return C.subgroup.tolist()


# ---------------------------------------------------------------------------
# Category C
# ---------------------------------------------------------------------------

def test_reloan_to_the_same_borrower_is_c1():
    assert c_subgroup(("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"),
                      ("2024-07-15", 1, 2, "loan")) == ["C1"]


def test_loan_to_a_different_club_is_c2():
    assert c_subgroup(("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"),
                      ("2024-07-01", 1, 5, "loan")) == ["C2"]


def test_free_and_no_fee_moves_to_a_third_club_are_c3_and_c4():
    assert c_subgroup(("2023-09-01", 1, 2, "loan"), ("2024-01-31", 2, 1, "loan_return"),
                      ("2024-02-01", 1, 5, "free_transfer")) == ["C3"]
    assert c_subgroup(("2023-09-01", 1, 2, "loan"), ("2024-01-31", 2, 1, "loan_return"),
                      ("2024-02-01", 1, 5, "no_fee_shown")) == ["C4"]


def test_a_conversion_is_not_in_category_c():
    assert c_subgroup(("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"),
                      ("2024-07-01", 1, 2, "permanent_transfer")) == []


def test_category_c_split_is_exhaustive_and_exclusive(real):  # noqa: F811
    N, C = csplit.compute(real["u"])
    T = episode_table(real["u"], next_moves(real["u"]))
    production_c = T.economic_ending.isin(["immediate_follow_on_transfer", "early_termination"]).sum()
    assert N["category_c"] == len(C) == production_c
    assert sum(s["N"] for s in N["subgroups"].values()) == production_c
    assert C.loan_event_id.is_unique and C.subgroup.isin(list(csplit.SUBGROUPS)).all()
    assert (C.lag_days >= 0).all()


# ---------------------------------------------------------------------------
# Country / date
# ---------------------------------------------------------------------------

def test_country_date_counts_reconcile_with_the_loan_universe(real):  # noqa: F811
    N, tb = cdates.compute(real["u"])
    assert N["loans"] == len(real["u"].loans) == 29699
    assert N["realised"] + N["scheduled"] + N["no_ending"] == N["loans"]
    assert sum(N["contexts"].values()) == N["realised"]
    dom = tb["country"]
    dom = dom[(dom["sample"] == "all realised endings") & (dom.attribution == "domestic")]
    assert dom.N.sum() == N["contexts"]["domestic"]
    for m in N["major_dates"]:
        assert m["from_domestic"] + m["from_cross-border"] + m["from_one or both clubs unmapped"] == m["total"]
    rules = tb["rules"]
    for _, g in rules.groupby("rule"):
        assert g.returns.sum() == N["rules_n_returns"]


# ---------------------------------------------------------------------------
# Playing-time join: synthetic
# ---------------------------------------------------------------------------

def _synthetic():
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "competition_id": "XX1",
                          "competition_type": "domestic_league", "season": 2023,
                          "date": pd.to_datetime(["2023-08-10", "2023-09-10", "2023-10-10"]),
                          "club_id": 2, "rows": [14, 14, 5]})
    games["complete"] = games.rows >= ptj.COMPLETE_ROWS
    app = pd.DataFrame({"game_id": ["g1"], "club_id": [2], "player_id": [7], "minutes_played": [90],
                        "date": pd.to_datetime(["2023-08-10"]), "national_team": [False]})
    E = pd.DataFrame({"player_id": [7, 8, 9], "b_club_id": [2, 2, 5],
                      "move_date": pd.to_datetime(["2023-07-01", "2023-09-01", "2023-07-01"]),
                      "departure_date": pd.to_datetime([None, "2023-10-01", None]),
                      "arrival_season": pd.array([2023, 2023, None], dtype="Int64"),
                      "scheduled": False, "history_captured_through": pd.Timestamp("2026-06-11")})
    return ptj.classify(ptj.measure(E, games, app), players={7, 8, 9})


def test_incomplete_window_gets_no_value_and_a_reason():
    E = _synthetic()
    x = E.iloc[0]
    assert x.exclusion_reason == "incomplete_club_games" and not x.reliable
    assert pd.isna(x.minutes) and not x.true_zero


def test_true_zero_needs_a_complete_window():
    y = _synthetic().iloc[1]
    assert y.reliable and y.departed_in_window and y.available_games == 1
    assert y.minutes == 0 and y.true_zero and y.minutes_share == 0


def test_uncovered_club_is_missing_not_zero():
    z = _synthetic().iloc[2]
    assert z.exclusion_reason == "not_covered" and pd.isna(z.minutes) and not z.true_zero


# ---------------------------------------------------------------------------
# Playing-time join: real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def joined(real):  # noqa: F811
    try:
        return ptj.compute(real["u"])
    except Exception as e:                       # the DuckDB snapshot is not present
        pytest.skip(f"playing-time tables unavailable: {e}")


def test_join_keeps_one_row_per_episode_and_reconciles(joined, real):  # noqa: F811
    N, E, _ = joined
    eps = real["u"].episodes
    assert E.event_id.is_unique
    assert N["candidates"]["loan"] == len(real["u"].loans)
    assert N["candidates"]["permanent"] == eps.episode_type.isin(
        ["permanent_paid", "free_transfer", "undisclosed_fee", "no_fee_shown"]).sum()


def test_missing_coverage_is_never_zero_minutes(joined):
    _, E, _ = joined
    assert E.loc[~E.reliable, "minutes"].isna().all()
    assert (E.reliable == (E.exclusion_reason == "")).all()
    assert set(E.exclusion_reason) - {""} <= set(ptj.REASONS)
    assert not (E.true_zero & ~E.reliable).any()
    assert (E.loc[E.reliable, "complete_games"] == E.loc[E.reliable, "available_games"]).all()


def test_waterfall_is_monotone_and_primary_is_reliable(joined):
    N, E, _ = joined
    for k, w in N["waterfall"].items():
        steps = [w[s] for s in ptj.STEPS]
        assert steps == sorted(steps, reverse=True)
        assert w["candidate"] - w["reliable"] == sum(w["excluded"].values())
        assert w["reliable_with_appearances"] + w["reliable_true_zero"] == w["reliable"]
    assert not (E.primary_sample & ~E.reliable).any()


def test_new_outputs_do_not_overwrite_existing_ones():
    from src.analysis import loan_timing_exploration as lte, daniel_scope_final as dsf
    old = {p.name for p in (lte.REPORT_MD, lte.LAGS_CSV, lte.EPISODES_CSV, lte.DATES_CSV, lte.COUNTRY_CSV,
                            lte.CATEGORY_CSV, lte.NEXT_KIND_CSV, lte.WINDOW_CSV, lte.CYCLE_CSV, dsf.FINAL_MD)}
    new = {p.name for p in (csplit.SPLIT_CSV, csplit.REPORT_MD, cdates.REPORT_MD, cdates.COUNTRY_CSV, cdates.RULES_CSV,
                            ptj.AUDIT_CSV, ptj.COVERAGE_CSV, ptj.EPISODES_CSV, ptj.REPORT_MD)}
    assert not old & new


# ---------------------------------------------------------------------------
# Playing-time descriptives
# ---------------------------------------------------------------------------

from src.analysis import playing_time_descriptives as ptd  # noqa: E402


def test_describe_and_diff_on_toy_data():
    d = ptd.describe(pd.Series([0.0, 0.05, 0.3, 0.6, 0.8]))
    assert d["N"] == 5 and d["zero"] == 0.2 and d["below_10"] == 0.4 and d["above_50"] == 0.4 and d["above_75"] == 0.2
    x = ptd.diff(pd.Series([0.5, 0.7]), pd.Series([0.2, 0.4]))
    assert np.isclose(x["mean_diff"], 0.3) and x["mean_diff_lo"] < 0.3 < x["mean_diff_hi"]


def test_primary_sample_is_the_base_sample_with_the_arrival_rule(joined):
    _, E, _ = joined
    B = ptd.base_sample(E)
    rebuilt = B[(B.games_remaining_at_arrival / B.season_games) >= ptj.FULL_SEASON_SHARE]
    assert set(rebuilt.event_id) == set(E.loc[E.primary_sample, "event_id"])


def test_primary_shares_are_valid_and_subtotals_reconcile(joined):
    _, E, _ = joined
    P = E[E.primary_sample]
    assert P.minutes_share.between(0, 1).all() and (P.minutes <= P.available_minutes).all()
    assert P.minutes.notna().all() and (P.complete_games == P.available_games).all()
    for col in ("receiving_league", "arrival_season"):
        t = ptd.by_group(P, col)
        for k in ptd.KINDS:
            assert t[f"{k}_n"].sum() == (P.kind == k).sum()


# ---------------------------------------------------------------------------
# Controlled regression: valuation rules and estimator
# ---------------------------------------------------------------------------

from src.analysis import playing_time_regression as ptr  # noqa: E402


def _pv(rows):
    pv = pd.DataFrame(rows, columns=["player_id", "vdate", "value", "vclub"])
    pv["vdate"] = pd.to_datetime(pv.vdate).astype("datetime64[ns]")
    return pv.astype({"player_id": "int64", "vclub": "int64", "value": float}).sort_values("vdate").reset_index(drop=True)


def test_player_value_uses_only_valuations_strictly_before_the_move():
    pv = _pv([(1, "2023-06-01", 1e6, 10), (1, "2023-07-01", 5e6, 20), (2, "2023-08-01", 3e6, 20)])
    E = pd.DataFrame({"event_id": ["a", "b"], "player_id": [1, 2],
                      "move_date": pd.to_datetime(["2023-07-01", "2023-07-15"])})
    v = ptr.player_value(E, pv).set_index("event_id")
    assert v.loc["a", "player_value"] == 1e6 and v.loc["a", "player_value_gap_days"] == 30   # same-day value ignored
    assert pd.isna(v.loc["b", "player_value"]) and pd.notna(v.loc["b", "next_vdate"])        # missing, not zero


def test_squad_value_excludes_the_focal_player_and_needs_a_full_squad():
    rows = [(p, "2023-06-01", 1e6, 50) for p in range(100, 112)] + [(7, "2023-06-01", 9e6, 50)]
    pv = _pv(rows)
    E = pd.DataFrame({"event_id": ["x", "y"], "player_id": [7, 8], "b_club_id": [50, 60],
                      "move_date": pd.to_datetime(["2023-07-01", "2023-07-01"])})
    s = ptr.squad_value(E, pv).set_index("event_id")
    assert s.loc["x", "squad_value"] == 12e6 and s.loc["x", "squad_players"] == 12 and s.loc["x", "focal_in_squad"]
    assert pd.isna(s.loc["y", "squad_value"])                                                 # no squad: missing


def test_absorbed_fixed_effects_match_dummies():
    rng = np.random.default_rng(0)
    n = 600
    d = pd.DataFrame({"g": rng.integers(0, 30, n), "b_club_id": rng.integers(0, 60, n),
                      "loan": rng.integers(0, 2, n), "x": rng.normal(size=n)})
    d["g_eff"] = d.g.map(dict(zip(range(30), rng.normal(size=30))))
    d["share"] = 0.3 + 0.05 * d.loan + 0.1 * d.x + d.g_eff + rng.normal(scale=0.05, size=n)
    a = ptr.fit(d, ["loan", "x"], absorb="g")
    b = ptr.fit(d, ["loan", "x"], fe="g")
    assert np.isclose(a["res"].params["loan"], b["res"].params["loan"])
    assert abs(a["res"].params["loan"] - 0.05) < 0.02


def test_valuation_joins_on_real_data_keep_episodes_unique(joined):
    import duckdb
    from src.config import DEFAULT_DATABASE
    _, E, _ = joined
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        pv = ptr.valuations(con)
    finally:
        con.close()
    B = ptr.build(E, pv)
    assert B.event_id.is_unique
    assert not (B.player_value_date >= B.move_date).any()
    assert (B.player_value.isna() == B.log_player_value.isna()).all()
    assert (B.player_value.dropna() > 0).all() and B.share.between(0, 1).all()
