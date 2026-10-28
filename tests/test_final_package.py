"""Final pass: appearance-gap flags, the calendar-gate comparison, the value stress test and the final package."""
import hashlib
import json
import re

import pandas as pd
import pytest

from src.analysis import appearance_audit as aud
from src.analysis import data_lineage as dl
from src.analysis import final_package as fp


@pytest.fixture(scope="module")
def M():
    if not dl.MASTER_CSV.exists():
        pytest.skip("master not built")
    return pd.read_csv(dl.MASTER_CSV, low_memory=False)


@pytest.fixture(scope="module")
def F():
    if not fp.NUMBERS_JSON.exists():
        pytest.skip("final package not run")
    return json.loads(fp.NUMBERS_JSON.read_text())


# --- appearance-gap flags (non-destructive) ---------------------------------------------------------------------

def test_flags_are_nested_in_pt_reliable(M):
    rel = M.pt_reliable
    assert M.pt_reliable_v2.eq(rel & ~M.pt_lineup_contradicted).all()
    assert not (M.pt_lineup_contradicted & ~rel).any()
    assert M.loc[~rel, ["pt_lineup_games_checked", "pt_games_played_without_row", "pt_contradiction_type"]].isna().all().all()
    t = M.loc[rel, "pt_contradiction_type"]
    assert set(t) == {"none", "false_zero", "partial_undercount"}
    assert M.pt_lineup_contradicted.eq(M.pt_contradiction_type.isin(["false_zero", "partial_undercount"])).all()
    fz = M.pt_contradiction_type.eq("false_zero")
    assert M.loc[fz, "true_zero"].astype(bool).all() and M.loc[fz, "minutes"].eq(0).all()
    assert M.loc[M.pt_contradiction_type.eq("partial_undercount"), "minutes"].gt(0).all()


def test_original_samples_untouched(M):
    # the new flags never feed a sample flag: contradicted episodes are still in the preferred samples
    assert (M[fp.MAIN] & M.pt_lineup_contradicted).sum() > 0
    assert M[fp.MAIN].sum() == 18_090 and M[fp.INJ].sum() == 13_951


def test_unchecked_episodes_are_2012(M):
    R = M[M.pt_reliable]
    unchecked = R[R.pt_lineup_games_checked.eq(0)]
    assert unchecked.arrival_season.eq(2012).mean() > 0.95


def test_audit_tables_add_up():
    if not aud.NUMBERS_JSON.exists():
        pytest.skip("appearance audit not run")
    A = json.loads(aud.NUMBERS_JSON.read_text())
    assert A["undetected"] + A["in_incomplete_club_games"] == A["on_pitch_no_row"]
    C = pd.read_csv(aud.CELLS_CSV)
    s = C[C.level.eq("season")]
    assert s.on_pitch_no_row.sum() == A["on_pitch_no_row"] and s.no_row_in_complete.sum() == A["undetected"]
    assert C[C.level.eq("league_season")].no_row_in_complete.sum() == A["undetected"]
    K = pd.read_csv(aud.CLUBS_CSV)
    assert K.no_row_in_complete.sum() == A["undetected"]


# --- calendar gate ----------------------------------------------------------------------------------------------

def test_gate_table_reconciles(M):
    T = pd.read_csv(fp.GATE_CSV)
    assert T["loans before the calendar gate (no-gate main sample)"].sum() == int((M.in_age_model_sample
                                                                                   & M.kind.eq("loan")).sum())
    assert T["final main-sample loans (calendar-first)"].sum() == int((M[fp.MAIN] & M.kind.eq("loan")).sum())


def test_gate_variants_match_flags(M, F):
    g = {r["variant"]: r for r in F["gate"]}
    done = M.arrival_season.le(fp.LAST_COMPLETE)
    for k, mask in (("A", M.in_age_model_sample), ("B", M[fp.MAIN]), ("C", M.in_age_model_sample & done),
                    ("D", M[fp.MAIN] & done)):
        assert g[k]["N loans"] == int((mask & M.kind.eq("loan")).sum())
        assert g[k]["N permanent"] == int((mask & M.kind.eq("permanent")).sum())


# --- value stress test and headline -----------------------------------------------------------------------------

def test_value_parametrisations_are_identical(F):
    e = F["equiv"]
    assert e["max_abs_identity_gap"] < 1e-9 and e["coef_difference"] < 1e-9 and e["se_difference"] < 1e-9


def test_headline_matches_pipeline(F):
    N = json.loads(dl.NUMBERS_JSON.read_text())
    cf = N["calendar_first"]["variants"]["classifiable"]["estimates"]
    assert F["headline"]["m3"]["coef"] == pytest.approx(cf["m3"]["coef"], abs=1e-12)
    assert F["headline"]["injury_fit"]["coef"] == pytest.approx(cf["injury_m3_fit"]["coef"], abs=1e-12)


def test_robustness_rows_reconcile(M, F):
    R = {r["specification"]: r for r in F["robust"]}
    assert R["Calendar-first sample (preferred)"]["N"] == int(M[fp.MAIN].sum())
    assert R["Calendar-first, without line-up-contradicted episodes (pt_reliable_v2)"]["N"] == \
        int((M[fp.MAIN] & M.pt_reliable_v2).sum())
    for r in F["robust"]:
        assert r["N loans"] + r["N permanent"] == r["N"] or r["group"] == "functional form" or \
            r["specification"].startswith("Receiving club")       # absorbed FE drop singleton club-seasons


# --- package ----------------------------------------------------------------------------------------------------

def test_package_files_and_text(F):
    for p in (fp.README, fp.QNOTES, fp.FLOW_CSV, fp.REG_CSV, fp.ROB_CSV, fp.HET_CSV, fp.GATE_CSV, fp.OFFCAL_CSV,
              fp.FILES_CSV, fp.COLHASH_CSV):
        assert p.exists(), p
    for name in ("fig1_playing_time_distribution", "fig2_ladder_and_decomposition", "fig3_relative_value",
                 "fig4_age", "final_fig5_robustness"):
        assert (fp.FFIG / f"{name}.png").exists() and (fp.FFIG / f"{name}.pdf").exists()
    for p in (fp.README, fp.QNOTES):
        t = p.read_text()
        assert not re.search(r"\bnan\b", t, flags=re.IGNORECASE) and "-0.0 pp" not in t
    assert "early termination or breach" in fp.README.read_text()


def test_column_hashes_match_master():
    if not fp.COLHASH_CSV.exists():
        pytest.skip("final package not run")
    H = pd.read_csv(fp.COLHASH_CSV)
    Mr = pd.read_csv(dl.MASTER_CSV, dtype=str, keep_default_na=False)
    assert list(H.column) == list(Mr.columns)
    for c in ("event_id", "minutes_share", "in_cf_main_model_sample", "pt_reliable_v2"):
        assert H.set_index("column").loc[c, "sha256"] == hashlib.sha256("\n".join(Mr[c]).encode()).hexdigest()


def test_sample_flow_reconciles(M):
    S = pd.read_csv(fp.FLOW_CSV)
    for br in ("loan", "permanent"):
        b = S[S.branch.eq(br)]
        assert b.loc[b.stage.eq("= main"), "remaining"].iloc[0] == int((M[fp.MAIN] & M.kind.eq(br)).sum())
        assert b.loc[b.stage.eq("= injury_model"), "remaining"].iloc[0] == int((M[fp.INJ] & M.kind.eq(br)).sum())
