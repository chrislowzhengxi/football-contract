"""Empirical summary: its numbers agree with the lineage pipeline and the master, and its text has no loose ends."""
import json
import re

import pandas as pd
import pytest

from src.analysis import data_lineage as dl
from src.analysis import professor_empirical as pe


@pytest.fixture(scope="module")
def out():
    if not (pe.NUMBERS_JSON.exists() and pe.SUMMARY_MD.exists()):
        pytest.skip("empirical summary not run")
    E = json.loads(pe.NUMBERS_JSON.read_text())
    N = json.loads(dl.NUMBERS_JSON.read_text())
    M = pd.read_csv(dl.MASTER_CSV, low_memory=False)
    return E, N, M


def test_preferred_sample_and_coefficient_match_the_pipeline(out):
    E, N, M = out
    cf = N["calendar_first"]["variants"]["classifiable"]
    assert E["N"] == {"loan": cf["totals"]["main"]["loan"], "permanent": cf["totals"]["main"]["permanent"]}
    assert sum(E["N"].values()) == int(M[pe.MAIN].sum()) == 18_090
    L = list(E["ladder"].values())
    assert L[-1]["coef"] == pytest.approx(cf["estimates"]["m3"]["coef"], abs=1e-12)
    assert L[0]["coef"] == pytest.approx(cf["estimates"]["raw_gap_main"], abs=1e-12)
    assert all(r["N"] == 18_090 for r in L)


def test_decomposition_adds_up(out):
    E, _, _ = out
    for key in ("gelbach", "gelbach_alt"):
        rows = {r["term"]: r["estimate"] for r in E[key]}
        parts = sum(v for k, v in rows.items() if k not in ("raw", "full"))
        assert rows["raw"] - rows["full"] == pytest.approx(parts, abs=1e-9)
    a = {r["term"]: r["estimate"] for r in E["gelbach"]}
    b = {r["term"]: r["estimate"] for r in E["gelbach_alt"]}
    assert a["log_relative_value"] + a["log_squad_value"] == pytest.approx(b["log_player_value"] + b["log_squad_value"])
    assert a["full"] == pytest.approx(E["ladder"][list(E["ladder"])[-1]]["coef"], abs=1e-8)


def test_robustness_rows_match_saved_estimates(out):
    E, N, _ = out
    R = {r["specification"]: r for r in E["robustness"]}
    cfv = N["calendar_first"]["variants"]
    pairs = [("Week 5/6 sample (no loan-calendar step)", N["regression"]["revised"]["m3"]),
             ("Calendar-first sample (preferred)", cfv["classifiable"]["estimates"]["m3"]),
             ("Calendar-normal loans only", cfv["normal_only"]["estimates"]["m3"]),
             ("Calendar-first, without high-confidence same-organisation moves",
              N["academy"]["variants"]["B"]["estimates"]["m3"]),
             ("Injury sample, injury-adjusted (fit) share", cfv["classifiable"]["estimates"]["injury_m3_fit"])]
    for name, s in pairs:
        assert R[name]["N"] == s["N"] and R[name]["coef"] == pytest.approx(s["coef"], abs=1e-12), name
    assert "NOT preferred" in R["Calendar-normal loans only"]["note"]


def test_tables_add_up(out):
    E, _, M = out
    B = pd.read_csv(pe.EMP / "B1_playing_time_distribution.csv").set_index("kind")
    assert B.N.to_dict() == E["N"]
    for csv in ("C1_age_groups.csv", "D1_relative_value_quintiles.csv", "E1_arrival_window.csv", "E2_season_remaining.csv"):
        T = pd.read_csv(pe.EMP / csv)
        assert T["N loans"].sum() == E["N"]["loan"] and T["N permanent"].sum() == E["N"]["permanent"], csv
    F = pd.read_csv(pe.EMP / "F1_off_calendar_loans.csv")
    assert F.N.sum() == E["N"]["loan"]


def test_examples_follow_their_rules(out):
    _, _, M = out
    X = pd.read_csv(pe.EMP / "G1_examples.csv")
    Q = pd.read_csv(pe.EMP / "Q1_zero_minutes_contradicted.csv")
    Mk = M.set_index("event_id")
    for _, r in X.iterrows():
        m = Mk.loc[r.event_id]
        assert m[pe.MAIN]
        if "zero" in r.case:
            assert m.true_zero and r.event_id not in set(Q.event_id)
        if "loan" in r.case:
            assert m.kind == "loan"
    assert set(X.case) == {c for c, _, _ in pe.EXAMPLE_RULES}


def test_appearance_gap_list(out):
    E, _, M = out
    Q = pd.read_csv(pe.EMP / "Q1_zero_minutes_contradicted.csv")
    assert Q.games_played_without_row.gt(0).all()
    assert set(Q.event_id) <= set(M.event_id[M.pt_reliable])
    assert E["lineup"]["reliable_contradicted"] == len(Q)
    assert E["lineup"]["main_contradicted"] == int(Q[pe.MAIN].sum())
    pepe = Q[Q.player_name.eq("Nicolas Pépé") & Q.to_club_name.eq("Arsenal")]
    assert len(pepe) == 1 and pepe.started_no_row.iloc[0] > 0 and pepe.appearances.iloc[0] == 0


def test_summary_text_is_clean(out):
    E, _, _ = out
    text = pe.SUMMARY_MD.read_text()
    assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE)
    assert "-0.0 pp" not in text and "+0.0 pp" not in text and "-0.00 " not in text
    assert "<br" not in text and "<i " not in text
    assert len(E["facts"]) == 5
    for f in E["files"]:
        assert (pe.OUT / f).exists(), f
    for fig in ("fig1_playing_time_distribution", "fig2_ladder_and_decomposition", "fig3_relative_value", "fig4_age",
                "fig5_arrival_timing", "fig6_loan_pathway_by_calendar_status"):
        assert (pe.FIG / f"{fig}.png").exists() and (pe.FIG / f"{fig}.pdf").exists()
