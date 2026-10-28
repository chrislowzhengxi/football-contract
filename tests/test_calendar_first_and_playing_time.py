"""Calendar status on the master, the calendar-first samples, and an independent check of the playing-time outcome."""
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import data_lineage as dl
from src.analysis import playing_time_qa as qa


@pytest.fixture(scope="module")
def out():
    if not dl.NUMBERS_JSON.exists():
        pytest.skip("data lineage not run")
    N = json.loads(dl.NUMBERS_JSON.read_text())
    M = pd.read_csv(dl.MASTER_CSV, low_memory=False)
    return N, M


def test_loan_calendar_status_relabels_the_existing_classification(out):
    _, M = out
    st = M.loan_calendar_status
    assert set(st) == {"normal", "off_calendar", "unclassified", "ending_not_realised", "not_applicable"}
    assert st[M.kind.ne("loan")].eq("not_applicable").all()                 # permanents never get a loan-ending status
    assert not st[M.kind.eq("loan")].eq("not_applicable").any()
    loan = M[M.kind.eq("loan")]
    expect = loan.calendar_state.map({"calendar-normal": "normal", "off-calendar": "off_calendar"})
    expect = expect.fillna(loan.calendar_state.str.startswith("unclassified").map({True: "unclassified"}))
    expect = expect.fillna("ending_not_realised")
    assert loan.loan_calendar_status.equals(expect)
    assert loan.loc[st[M.kind.eq("loan")].eq("ending_not_realised"), "in_calendar_population"].eq(False).all()


def test_off_calendar_flag(out):
    _, M = out
    f = M.off_calendar_flag
    assert set(f.dropna().astype(int)) == {0, 1}
    assert f.notna().eq(M.loan_calendar_status.isin(["normal", "off_calendar"])).all()
    assert (f.eq(1) == M.off_calendar.eq(True)).all() and (f.eq(0) == M.off_calendar.eq(False)).all()


def test_calendar_first_flags(out):
    N, M = out
    el = M.calendar_first_eligible
    assert el.eq(M.kind.eq("permanent") | M.loan_calendar_status.isin(["normal", "off_calendar"])).all()
    for cf, base in (("in_cf_revised_sample", "in_revised_sample"), ("in_cf_injury_usable", "in_injury_usable"),
                     ("in_cf_main_model_sample", "in_age_model_sample"),
                     ("in_cf_injury_model_sample", "in_injury_model_sample")):
        assert M[cf].eq(M[base] & el).all()
        assert M.loc[M[cf] & M.kind.eq("permanent")].shape[0] == M.loc[M[base] & M.kind.eq("permanent")].shape[0]
    T = N["calendar_first"]["variants"]["classifiable"]["totals"]
    assert M.in_cf_revised_sample.sum() == T["revised"]["total"]
    assert M.in_cf_main_model_sample.sum() == T["main"]["total"]
    assert M.in_cf_injury_usable.sum() == T["injury_usable"]["total"]
    assert M.in_cf_injury_model_sample.sum() == T["injury_model"]["total"]
    assert M.loc[M.in_cf_revised_sample & M.kind.eq("loan"), "off_calendar_flag"].eq(1).sum() > 0   # off-calendar kept


def test_calendar_first_waterfalls_add_up(out):
    N, _ = out
    CF = N["calendar_first"]
    assert sum(CF["loan_status"].values()) == CF["loans"]
    cur = CF["variants"]["current"]["totals"]
    mem = N["master"]["membership"]
    assert (cur["revised"]["total"], cur["main"]["total"], cur["injury_usable"]["total"], cur["injury_model"]["total"]) == \
        (mem["in_revised_sample"], mem["in_age_model_sample"], mem["in_injury_usable"], mem["in_injury_model_sample"])
    for key, V in CF["variants"].items():
        n = CF["loans"]
        for s in V["gate_steps"]:
            n -= s["removed"]
            assert s["n"] == n
        assert V["after_gate"] == n
        for br, B in (("loan", V["loan_branch"]), ("permanent", CF["permanent_branch"])):
            x = B["start"]
            for steps, end in (("pt_steps", "reliable"), ("revised_steps", "revised"), ("control_steps", "main")):
                x -= sum(s["removed"] for s in B[steps])
                assert x == B[end], (key, br, end)
            y = B["revised"] - sum(s["removed"] for s in B["injury_steps"])
            assert y == B["injury_usable"]
            assert y - sum(s["removed"] for s in B["injury_control_steps"]) == B["injury_model"]
        assert V["loan_branch"]["start"] == V["after_gate"]
    v = CF["variants"]
    assert v["normal_only"]["totals"]["main"]["total"] <= v["classifiable"]["totals"]["main"]["total"] \
        <= v["realised"]["totals"]["main"]["total"] <= v["current"]["totals"]["main"]["total"]


def test_playing_time_recomputation_agrees_everywhere():
    if not qa.NUMBERS_JSON.exists():
        pytest.skip("playing-time QA not run")
    Q = json.loads(qa.NUMBERS_JSON.read_text())
    for k, v in Q["checks"].items():
        if isinstance(v, dict):
            assert v["agree"] == v["checked"], k
    assert Q["checks"]["agree_all_loan"] == Q["checks"]["n_loan"] > 0
    assert Q["checks"]["agree_all_permanent"] == Q["checks"]["n_permanent"] > 0


def test_example_games_reproduce_the_master(out):
    if not qa.GAMES_CSV.exists():
        pytest.skip("playing-time QA not run")
    _, M = out
    G = pd.read_csv(qa.GAMES_CSV)
    Mk = M.set_index("event_id")
    kinds = set()
    for eid, g in G.groupby("event_id"):
        x = Mk.loc[eid]
        w = g[g.in_window]
        kinds.add(x.kind)
        assert len(w) == x.available_games
        assert (w.b_appearance_rows >= qa.COMPLETE_ROWS).all()
        assert int(w.player_minutes.fillna(0).sum()) == x.minutes
        assert np.isclose(x.minutes / (90 * x.available_games), x.minutes_share)
        assert w.date.min() >= x.transfer_date and w.date.max() <= x.window_end
        if x.true_zero:
            assert w.player_minutes.isna().all()
    assert kinds == {"loan", "permanent"}


def test_live_recompute_on_a_sample(out):
    duckdb = pytest.importorskip("duckdb")
    from src.config import DEFAULT_DATABASE
    if not DEFAULT_DATABASE.exists():
        pytest.skip("DuckDB snapshot missing")
    _, M = out
    R = M[M.pt_reliable]
    S = pd.concat([R[R.kind.eq(k)].sample(150, random_state=7) for k in ("loan", "permanent")])
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        Q = qa.recompute(con, S)
    finally:
        con.close()
    X = S.set_index("event_id").join(Q, rsuffix="_qa")
    assert (X.available_games == X.available_games_qa).all()
    assert (X.minutes == X.minutes_qa).all()
    assert (X.complete_games_qa == X.available_games_qa).all()
    assert np.allclose(X.minutes_share, X.share)
