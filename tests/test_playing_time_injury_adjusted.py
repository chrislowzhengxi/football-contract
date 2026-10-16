"""Injury-adjusted playing time: parsing, classification, spell logic, exact-id join, same-sample models."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import playing_time_injury_adjusted as ia

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


def raw(rows):
    return pd.DataFrame(rows, columns=ia.RAW_COLS).astype(str)


# ---------------------------------------------------------------------------
# Pure logic (synthetic)
# ---------------------------------------------------------------------------

def test_normalize_dates_duplicates_and_flags():
    R = raw([["1", "20/21", "Hamstring injury", "2020-10-01", "2020-10-10", "10.0", "2"],
             ["1", "20/21", "Hamstring injury", "2020-10-01", "2020-10-10", "10.0", "2"],      # exact duplicate
             ["1", "20/21", "Knee injury", "2020-10-01", "2020-10-20", "20.0", "3"],           # same start, differs
             ["2", "21/22", "Ankle injury", "2021-03-05", "", "", "0"],                         # open
             ["3", "21/22", "Ill", "", "2021-01-02", "", "1"],                                   # no start
             ["4", "21/22", "Knock", "2021-02-10", "2021-02-01", "1.0", "0"]])                  # end before start
    N, info = ia.normalize(R)
    assert info["exact_duplicates_dropped"] == 1 and len(N) == 5
    p1 = N[N.player_id == 1]
    assert p1.conflicting_duplicate.all() and p1.from_date.eq(pd.Timestamp("2020-10-01")).all()
    assert N.loc[N.player_id == 2, "open_injury"].item() and pd.isna(N.loc[N.player_id == 2, "end_date"].item())
    assert N.loc[N.player_id == 3, "missing_start"].item()
    assert N.loc[N.player_id == 4, "end_before_start"].item()
    assert N.from_date_raw.tolist()[0] == "2020-10-01"                        # raw strings preserved
    S = ia.spells(N, "primary")
    assert set(S.player_id) == {1}                                            # open, no-start, reversed rows excluded


def test_classification_never_counts_non_medical():
    for lab in ("Rest", "Fitness", "Quarantine"):
        c = ia.classify_reason(lab)
        assert not c["primary"] and not c["strict"], lab
    assert ia.classify_reason("unknown injury") == {"category": "injury, type unspecified (primary only)",
                                                    "primary": True, "strict": False}
    assert ia.classify_reason("Ill")["category"] == "illness"
    assert ia.classify_reason("Cruciate ligament tear")["strict"]
    assert ia.classify_reason("Muscle fatigue")["primary"] and not ia.classify_reason("Muscle fatigue")["strict"]


def test_union_and_appearance_override():
    d = pd.Timestamp
    W = pd.DataFrame({"event_id": ["e"] * 4, "player_id": [1] * 4, "game_id": ["g1", "g2", "g3", "g4"],
                      "date": [d("2021-01-02"), d("2021-01-09"), d("2021-01-16"), d("2021-01-23")],
                      "played": [False, True, False, False]})
    Sp = pd.DataFrame({"player_id": [1, 1], "spell_id": [10, 11], "from_date": [d("2021-01-01"), d("2021-01-08")],
                       "end_date": [d("2021-01-10"), d("2021-01-16")]})
    G, M = ia.game_flags(W, Sp)
    assert G.covering_spells.tolist() == [1, 2, 1, 0]
    assert G.medical_unavailable.tolist() == [True, False, True, False]       # g2 played: appearance wins; g3 ends on day
    assert G.appearance_conflict.tolist() == [False, True, False, False]
    assert ia.union_days([(d("2021-01-01"), d("2021-01-10")), (d("2021-01-08"), d("2021-01-16"))]) == 16
    assert int(G.medical_unavailable.sum()) == 2                              # overlapping spells counted once per game


def test_usability_open_and_truncated_and_snapshot():
    d = pd.Timestamp
    N = pd.DataFrame({"player_id": [1, 2], "counts_primary": [True, True], "open_injury": [True, False],
                      "missing_start": [False, False], "from_date": [d("2020-01-10"), d("2019-06-01")],
                      "player_history_truncated": [False, True], "player_history_floor": [pd.NaT, d("2019-06-01")]})
    E = pd.DataFrame({"player_id": [1, 2, 2, 3, 1], "available_games_fit": [5, 5, 5, 5, 5],
                      "window_start": [d("2019-08-01"), d("2018-08-01"), d("2019-08-01"), d("2019-08-01"), d("2025-08-01")],
                      "window_end": [d("2020-05-01"), d("2019-05-01"), d("2020-05-01"), d("2020-05-01"), d("2026-05-01")]})
    status = pd.Series({1: "A", 2: "A", 3: "C"})
    U = ia.usability(E, N, status).tolist()
    assert U == ["open injury (no end date) could cover the window", "injury history truncated before the window",
                 "usable", "player not in the external datalake", "window reaches the 2025-09-15 snapshot"]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def out():
    if not ia.NUMBERS_JSON.exists():
        pytest.skip("injury-adjusted analysis not run")
    return json.loads(ia.NUMBERS_JSON.read_text()), pd.read_csv(ia.EPISODES_CSV)


def test_raw_files_verified_and_unchanged():
    for rel, (sha, size) in ia.EXPECTED.items():
        b = (ia.RAW / rel).read_bytes()
        assert hashlib.sha256(b).hexdigest() == sha and len(b) == size


def test_exact_player_id_join(out):
    N, E = out
    inj = set(pd.read_csv(ia.INJ_FILE, usecols=["player_id"]).player_id.astype(int))
    prof = set(pd.read_csv(ia.PROF_FILE, usecols=["player_id"]).player_id.astype(int))
    expect = np.where(E.player_id.isin(inj), "A", np.where(E.player_id.isin(prof), "B", "C"))
    assert (E.status.values == expect).all()
    assert sum(N["join_players"].values()) == N["revised_players"]


def test_episode_invariants(out):
    N, E = out
    assert (E.available_games_fit <= E.available_games_current).all()
    ok = E.fit_adjusted_share.notna()
    assert (E.loc[ok, "fit_adjusted_share"] >= E.loc[ok, "current_share"] - 1e-12).all()
    z = ok & E.confirmed_injury_games.eq(0)
    assert np.allclose(E.loc[z, "fit_adjusted_share"], E.loc[z, "current_share"])
    U = E[E.usable]
    assert (U.available_games_fit >= 1).all() and U.data_quality_flag.eq("usable").all()
    assert (E.loc[E.data_quality_flag.eq("window reaches the 2025-09-15 snapshot"), "window_end"] >= "2025-09-15").all()
    for k in ("strict", "severe3", "long5"):
        assert (E[f"injury_games_{k}"] <= E.confirmed_injury_games).all()


def test_same_sample_models_and_revised_sample(out):
    N, E = out
    M = pd.read_csv(ia.MODELS_CSV)
    pref = M[M.model == "preferred (M3)"]
    assert pref.N.nunique() == 1 and set(pref.outcome) == {"current", "fit-adjusted"}
    assert len(N["model_event_ids"]) == N["preferred"]["current"]["N"] == N["preferred"]["fit"]["N"]
    assert set(N["model_event_ids"]) <= set(E.loc[E.usable, "event_id"])
    red = json.loads((ia.OUT / "playing_time_sample_redesign_numbers.json").read_text())
    assert N["revised"] == red["revised"] and len(E) == sum(red["revised"].values())


def test_stage1c_and_old_outputs_unchanged(out):
    N, _ = out
    assert _sha(ia.OUT / "stage1c_canonical_transfers.csv") == STAGE1C_SHA256
    for name, h in N["old_output_sha256"].items():
        assert _sha(ia.OUT / name) == h, name
    assert not {p.name for p in ia.WRITES} & set(N["old_output_sha256"])
