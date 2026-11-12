"""November 2026 professor feedback: unit tests on synthetic cases and checks of the rebuilt master."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis import data_lineage as dl
from src.analysis import feedback_nov2026 as fb
from src.analysis import feedback_nov2026_specs as fs

from tests.test_loan_episodes import STAGE1C_SHA256, _sha

ARCH_SHA = dl.OUT / "professor_feedback_nov_2026" / "archive_pre_nov2026" / "master_column_sha256_150col.csv"


# ---------------------------------------------------------------------------
# Unit tests: minutes classification
# ---------------------------------------------------------------------------

def _rows(**cols):
    n = len(next(iter(cols.values())))
    base = {"event_id": ["e"] * n, "game_id": list(range(n)), "player_id": [1] * n, "club_id": [9] * n,
            "game_max_minutes": [90] * n, "club_lineup_rows": [22] * n, "m_in": [np.nan] * n, "m_out": [np.nan] * n,
            "m_red": [np.nan] * n, "lineup_type": ["starting_lineup"] * n}
    base.update(cols)
    return fb.classify_minutes(pd.DataFrame(base))


def test_minutes_conventions_and_errors():
    A = _rows(lineup_type=["starting_lineup", "substitutes", "substitutes", "starting_lineup", "substitutes",
                           "starting_lineup", "substitutes", "starting_lineup", "starting_lineup", None, None,
                           "starting_lineup", "substitutes"],
              m_in=[np.nan, 46, 90, np.nan, 60, np.nan, 85, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              m_out=[np.nan, np.nan, np.nan, 70, 80, np.nan, 85, -1, 46, np.nan, np.nan, np.nan, np.nan],
              m_red=[np.nan, np.nan, np.nan, np.nan, np.nan, 30, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              minutes_played=[90, 45, 1, 70, 20, 30, 0, 1, 90, 90, 80, 135, 25],
              club_lineup_rows=[22] * 9 + [np.nan, 22, 22, 22])
    assert list(A.minutes_status) == [
        "verified_exact",                      # starter, whole game
        "verified_convention",                 # 46' half-time change counts 45
        "verified_convention",                 # stoppage-time substitute counts 1
        "verified_exact",                      # starter taken off at 70
        "verified_exact",                      # came on at 60, off at 80
        "verified_exact",                      # sent off at 30
        "verified_exact",                      # on and off in the same minute
        "unverifiable_event_minute_unknown",   # event minute -1
        "contradicted",                        # credited 90, substituted at 46
        "unverifiable_no_lineup",
        "unverifiable_not_in_lineup",
        "impossible_over_90",
        "unverifiable_missing_entry_event"]


def test_shortened_game_is_consistent_not_an_error():
    A = _rows(minutes_played=[30, 30], game_max_minutes=[30, 30], lineup_type=["starting_lineup", "starting_lineup"])
    assert A.minutes_status.eq("verified_shortened_game").all()


def test_minutes_validation_status_priority():
    A = pd.DataFrame({"event_id": ["a", "a", "b", "c"],
                      "minutes_status": ["verified_exact", "contradicted", "unverifiable_no_lineup", "verified_exact"]})
    V = fb.minutes_validation(A, pd.Series(["a", "b", "c", "d"])).set_index("event_id")
    assert V.pt_minutes_validation_status.to_dict() == {
        "a": "contradicted_record", "b": "verified_except_unverifiable_rows", "c": "all_rows_verified",
        "d": "no_appearances_to_check"}


# ---------------------------------------------------------------------------
# Unit tests: quarantine rule
# ---------------------------------------------------------------------------

def test_quarantine_counts_but_not_fitness_or_rest():
    INJ = pd.DataFrame({"injury_reason_clean": ["quarantine", "fitness", "rest", "hamstring injury"],
                        "counts_primary": [False, False, False, True], "counts_strict": [False, False, False, True],
                        "category": ["x"] * 4})
    N = fb.injury_rows_v2(INJ)
    assert list(N.counts_primary) == [True, False, False, True]
    assert list(N.counts_strict) == [False, False, False, True]            # strict rule unchanged
    assert list(INJ.counts_primary) == [False, False, False, True]         # input not modified


# ---------------------------------------------------------------------------
# Unit tests: prior season (registration spells, pooling, no leakage, unregistered appearances)
# ---------------------------------------------------------------------------

def _universe():
    rows = pd.DataFrame({
        "event_id": ["r1", "r2", "r3", "r4"], "transfermarkt_transfer_id": [1, 2, 3, 4], "player_id": [7] * 4,
        "_date": pd.to_datetime(["2018-07-01", "2020-01-15", "2020-06-30", "2020-08-01"]),
        "transfer_type_normalized": ["free_transfer", "loan", "loan_return", "permanent_transfer"],
        "transfermarkt_future_transfer": [False] * 4, "from_club_id": [100, 1, 2, 1], "to_club_id": [1, 2, 1, 3]})
    return SimpleNamespace(rows=rows)


def _sides():
    d1 = pd.date_range("2019-08-10", "2020-05-20", freq="14D")
    d2 = pd.date_range("2019-08-11", "2020-05-21", freq="14D")
    S = pd.concat([pd.DataFrame({"game_id": range(len(d1)), "club_id": 1, "date": d1}),
                   pd.DataFrame({"game_id": range(100, 100 + len(d2)), "club_id": 2, "date": d2})])
    return S.assign(season=2019, competition_id=np.where(S.club_id.eq(1), "L1", "L2"), rows=14, complete=True)


def test_prior_season_pools_spells_without_leakage():
    sp = fb.registration_spells(_universe())
    assert list(sp.club_id) == [100, 1, 2, 1, 3]
    S = _sides()
    E = pd.DataFrame({"event_id": ["ep"], "player_id": [7], "move_date": [pd.Timestamp("2020-08-01")],
                      "ref_season": [2020]})
    at1 = S[(S.club_id == 1) & ((S.date < "2020-01-15") | (S.date >= "2020-06-30"))]
    at2 = S[(S.club_id == 2) & (S.date >= "2020-01-15") & (S.date < "2020-06-30")]
    apps = pd.concat([at1.head(5), at2.head(4)])[["game_id", "club_id", "date"]].assign(player_id=7, minutes_played=90)
    O, G = fb.prior_season(E, sp, S, apps)
    o = O.iloc[0]
    assert o.prior_reliable and o.prior_covered_clubs == 2
    assert o.prior_league_games == len(at1) + len(at2)                 # pooled over both spells
    assert o.prior_minutes == 9 * 90
    assert abs(o.prior_minutes_share - 9 * 90 / (90 * (len(at1) + len(at2)))) < 1e-12
    assert (G.date < pd.Timestamp("2020-08-01")).all()
    # an appearance for club 2 while registered at club 1 makes the measure unreliable (a move is missing)
    bad = pd.concat([apps, S[(S.club_id == 2) & (S.date < "2020-01-15")].head(1)[["game_id", "club_id", "date"]]
                     .assign(player_id=7, minutes_played=90)])
    O2, _ = fb.prior_season(E, sp, S, bad)
    assert O2.iloc[0].prior_exclusion_reason == "appearances_for_unregistered_club"
    assert np.isnan(O2.iloc[0].prior_minutes_share)                    # missing, never zero
    # an incomplete club-game makes it unreliable too
    O3, _ = fb.prior_season(E, sp, S.assign(complete=S.game_id.ne(0)), apps)
    assert O3.iloc[0].prior_exclusion_reason == "incomplete_club_games"


def test_prior_season_without_covered_club_is_blank():
    sp = fb.registration_spells(_universe())
    E = pd.DataFrame({"event_id": ["ep"], "player_id": [7], "move_date": [pd.Timestamp("2020-08-01")],
                      "ref_season": [2020]})
    S = _sides().assign(club_id=55)
    O, _ = fb.prior_season(E, sp, S, pd.DataFrame(columns=["game_id", "player_id", "club_id", "date", "minutes_played"])
                           .astype({"game_id": "int64", "player_id": "int64", "club_id": "int64",
                                    "date": "datetime64[ns]"}))
    assert O.iloc[0].prior_exclusion_reason == "no_covered_club" and np.isnan(O.iloc[0].prior_minutes_share)


# ---------------------------------------------------------------------------
# The rebuilt master
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def M():
    M = pd.read_csv(dl.MASTER_CSV, low_memory=False)
    if "perm_moved_on_before_first_league_game" not in M.columns:
        pytest.skip("master not rebuilt")
    return M


def test_stage1c_unchanged():
    assert _sha(dl.OUT / "stage1c_canonical_transfers.csv") == STAGE1C_SHA256


def test_new_columns_appended_and_old_columns_identical(M):
    assert list(M.columns[-len(fs.NEW_COLUMNS):]) == fs.NEW_COLUMNS
    if not ARCH_SHA.exists():
        pytest.skip("archive not present")
    H = pd.read_csv(ARCH_SHA)
    assert list(M.columns[:len(H)]) == list(H.column) and len(H) == 150
    Mr = pd.read_csv(dl.MASTER_CSV, dtype=str, keep_default_na=False, usecols=list(H.column))
    for c, h in zip(H.column, H.sha256):
        assert hashlib.sha256("\n".join(Mr[c]).encode()).hexdigest() == h, c


def test_every_new_column_documented(M):
    D = pd.read_csv(dl.VARDICT_CSV, keep_default_na=False).set_index("variable")
    for c in fs.NEW_COLUMNS:
        d = D.loc[c]
        assert d.plain_english_definition and d.formula_or_rule and d.valid_when and d.usefulness
        assert d.n_non_blank == int(M[c].notna().sum())


def test_brunner(M):
    b = M[(M.player_id == 918414) & M.to_club_name.eq("Monaco") & M.kind.eq("permanent")].iloc[0]
    assert b.next_departure_type == "loan_out" and b.days_at_receiving_club == 1
    assert b.perm_moved_on_before_first_league_game == True  # noqa: E712
    assert b.receiving_first_league_game_date == b.next_departure_date == "2024-08-17"
    assert b.pt_exclusion_reason == "no_league_games_in_window"


def test_departure_flags_are_consistent(M):
    C = M[M.in_playing_time_candidates]
    lb = C.left_before_first_receiving_league_game == True  # noqa: E712
    # left before the first game <=> no league game in the window (covered episodes; at most one unknown)
    cov = C.pt_covered_league_season
    assert (C[lb & cov].available_games == 0).all()
    nl = C.pt_exclusion_reason.eq("no_league_games_in_window")
    assert (nl & ~lb).sum() <= 1
    assert not (C.next_departure_is_loan.eq(True) & C.next_departure_is_permanent.eq(True)).any()
    assert C.days_at_receiving_club.dropna().ge(0).all()
    assert M.loc[~M.kind.eq("permanent"), "perm_moved_on_before_first_league_game"].isna().all()
    assert set(C.arrival_season_presence) <= set(fb.PRESENCE)
    assert set(C.next_departure_type) <= set(fb.DEPARTURE_TYPES)
    assert not (M.in_cf_main_model_sample & lb.reindex(M.index, fill_value=False)).any()
    st = C.stayed_through_arrival_season_end == False  # noqa: E712
    assert (C[st].next_departure_date <= C[st].receiving_last_league_game_date).all()


def test_minutes_validation_columns(M):
    R = M[M.pt_reliable]
    assert (R.pt_minutes_rows_verified + R.pt_minutes_rows_unverifiable + R.pt_minutes_rows_contradicted
            == R.appearances).all()
    bad = M.pt_minutes_validation_status.eq("contradicted_record")
    assert M.minutes_share_validated.notna().equals(M.pt_reliable & ~bad)
    assert (M.in_cf_main_model_sample_v2 == (M.in_cf_main_model_sample & ~bad)).all()
    assert (M.in_age_model_sample_v2 == (M.in_age_model_sample & ~bad)).all()


def test_combined_validation(M):
    """8 Nov correction: the line-up check and the recorded-minute check are applied together."""
    full = M.pt_reliable_v2 & M.pt_minutes_validated
    assert M.pt_fully_validated.equals(full)
    assert not (M.pt_lineup_contradicted & M.pt_reliable & ~M.pt_minutes_validated).any()   # disjoint problems
    for old, new in (("in_age_model_sample_v2", "in_age_model_sample_validated"),
                     ("in_cf_main_model_sample_v2", "in_cf_main_model_sample_validated"),
                     ("in_injury_model_sample_v2", "in_injury_model_sample_validated"),
                     ("in_cf_injury_model_sample_v2", "in_cf_injury_model_sample_validated")):
        assert M[new].equals(M[old] & full)
    assert int(M.in_cf_main_model_sample_validated.sum()) == 18_006
    assert int(M.in_cf_injury_model_sample_validated.sum()) == 13_871
    # the existing samples are not redefined
    assert int(M.in_cf_main_model_sample_v2.sum()) == 18_080 and int(M.in_cf_injury_model_sample_v2.sum()) == 13_941


def test_injury_v2(M):
    R = M[M.in_revised_sample]
    assert (R.confirmed_injury_games_v2 - R.confirmed_injury_games == R.quarantine_unavailable_games).all()
    assert (R.confirmed_injury_games_v2 >= R.confirmed_injury_games).all()
    ok = R.fit_adjusted_share_v2.notna() & R.fit_adjusted_share.notna()
    assert (R.fit_adjusted_share_v2[ok] >= R.fit_adjusted_share[ok] - 1e-12).all()
    assert not (M.in_injury_usable_v2 & ~M.in_revised_sample).any()
    assert not (M.in_cf_injury_model_sample_v2 & ~(M.in_injury_usable_v2 & M.in_cf_main_model_sample_v2)).any()
    assert (M.in_injury_model_sample == (M.in_injury_usable & M.in_age_model_sample)).all()


def test_tiers(M):
    for side in ("from", "to"):
        t, st = M[f"{side}_league_tier_at_move"], M[f"{side}_league_tier_status"]
        assert t.dropna().ge(1).all() and (t.dropna() % 1 == 0).all()
        assert t.notna().equals(st.eq("verified")) and M[f"{side}_league_name_at_move"].notna().equals(st.eq("verified"))
    assert M.loc[M.in_cf_main_model_sample, "to_league_tier_at_move"].eq(1).all()
    assert (M.from_league_tier_at_move.notna() & M.from_league_tier_prior_season.notna()
            & (M.from_league_tier_at_move != M.from_league_tier_prior_season)).any()       # season-specific
    # outside a covered top flight is not evidence of a lower tier
    assert M.loc[M.from_in_covered_top_flight_at_move.eq(False) & M.from_league_tier_status.ne("verified"),
                 "from_league_tier_at_move"].isna().all()


def test_prior_season_columns(M):
    R = M[M.prior_reliable == True]  # noqa: E712
    assert np.allclose(R.prior_minutes_share, R.prior_minutes / (90 * R.prior_league_games))
    assert M.loc[M.prior_reliable != True, "prior_minutes_share"].isna().all()  # noqa: E712
    assert (M.prior_season.dropna() == M.tier_reference_season[M.prior_season.notna()] - 1).all()
    f = M.prior_fit_adjusted_share.notna()
    assert (M.prior_fit_adjusted_share[f] >= M.prior_minutes_share[f] - 1e-12).all()
    assert M.loc[f, "prior_injury_data_quality_flag"].eq("usable").all()


def test_origin_squad_value(M):
    v = M.from_squad_value.notna()
    assert M.loc[v, "from_squad_players"].ge(11).all()
    assert (M.from_squad_value_incl_player[v] >= M.from_squad_value[v]).all()
    assert M.from_squad_value_incl_player.notna().equals(v)
    s = M.squad_value.notna()
    assert np.allclose(M.to_squad_value[s], M.squad_value[s])
    r = M.from_to_squad_value_ratio.notna()
    assert np.allclose(M.from_to_squad_value_ratio[r], M.from_squad_value[r] / M.to_squad_value[r])


def test_new_columns_do_not_change_stored_samples(M):
    N = json.loads(dl.NUMBERS_JSON.read_text())
    cf = N["calendar_first"]["variants"]["classifiable"]["totals"]
    assert int(M.in_cf_main_model_sample.sum()) == cf["main"]["total"]
    assert int(M.in_cf_injury_model_sample.sum()) == cf["injury_model"]["total"]
    from src.analysis.feedback_nov2026_report import m3
    r = m3(M, M.in_cf_main_model_sample, "minutes_share")
    est = N["calendar_first"]["variants"]["classifiable"]["estimates"]["m3"]
    assert r["N"] == est["N"] and abs(r["coef"] - est["coef"]) < 1e-9
