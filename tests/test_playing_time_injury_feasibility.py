"""Injury feasibility audit: no injury is inferred, coverage reconciles, nothing old is touched."""
import json

import pandas as pd
import pytest

from src.analysis import playing_time_injury_feasibility as inj
from src.analysis.daniel_scope_final import f0

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


@pytest.fixture(scope="module")
def N():
    if not inj.NUMBERS_JSON.exists():
        pytest.skip("injury feasibility not run")
    return json.loads(inj.NUMBERS_JSON.read_text())


def test_no_injury_spell_source_and_no_fit_adjusted_outcome(N):
    assert all(not s["availability_columns"] for s in N["schema"])
    for k in inj.KINDS:
        assert N["waterfall"][k]["with_injury_spell_data"] == 0
        assert N["waterfall"][k]["fit_adjusted_computable"] == 0
    for name in ("playing_time_fit_adjusted_sample.csv", "playing_time_fit_adjusted_results.csv"):
        assert name not in {p.name for p in inj.WRITES}


def test_revised_sample_and_coverage_table_reconcile(N):
    red = json.loads((inj.OUT / "playing_time_sample_redesign_numbers.json").read_text())
    assert N["revised"] == red["revised"]
    C = pd.read_csv(inj.COVERAGE_CSV)
    for k in inj.KINDS:
        top = C[(C.dimension == "all") & (C.kind == k)].iloc[0]
        assert top.episodes == N["revised"][k]
        for dim in ("league", "season", "league_season"):
            sub = C[(C.dimension == dim) & (C.kind == k)]
            assert sub.episodes.sum() == top.episodes
            assert sub.squad_proxy_computable.sum() == top.squad_proxy_computable
        w = N["waterfall"][k]
        assert w["squad_proxy_computable"] + w["squad_proxy_lost"] == w["episodes"]
    assert (C.with_injury_spell_data == 0).all() and (C.fit_adjusted_computable == 0).all()
    assert (C.squad_proxy_computable <= C.episodes).all()


def test_bounds_are_ordered(N):
    b = N["bounds"]
    assert b["loan_range"][0] <= b["loan_range"][1] and b["perm_range"][0] <= b["perm_range"][1]
    assert b["gap_worst_low"] <= b["current_gap"] <= b["gap_worst_high"]
    for k in inj.KINDS:
        lv = N["levels"][k]
        assert lv["played"] <= lv["named"] + N["consistency"]["played_not_named"] and lv["named"] <= lv["games"]


def test_old_outputs_unchanged_stage1c_and_generated_numbers(N):
    assert _sha(inj.OUT / "stage1c_canonical_transfers.csv") == STAGE1C_SHA256
    for name, h in N["old_output_sha256"].items():
        assert _sha(inj.OUT / name) == h, name
    md = inj.REPORT_MD.read_text()
    for k in inj.KINDS:
        assert f0(N["revised"][k]) in md and f0(N["waterfall"][k]["squad_proxy_computable"]) in md
