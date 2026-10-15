"""Playing-time sample redesign: waterfall reconciles, follow-on audit partitions, no season-fraction rule."""
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import playing_time_join as ptj
from src.analysis import playing_time_sample_redesign as rd
from src.analysis.daniel_scope_final import f0

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


# ---------------------------------------------------------------------------
# Classification logic (synthetic)
# ---------------------------------------------------------------------------

def test_follow_on_and_arrival_group():
    d = pd.Timestamp
    E = pd.DataFrame({
        "move_date": [d("2020-07-01")] * 6 + [d("2021-01-20")],
        "next_date": [pd.NaT, d("2021-07-01"), d("2020-07-02"), d("2020-09-01"), d("2020-08-01"), d("2020-08-01"),
                      d("2021-03-01")],
        "next_type": [None, "permanent_transfer", "loan", "free_transfer", "free_transfer", "loan_return", "loan"],
        "next_to_club": [None, "X", "Y", "Without Club", "Z", "A", "W"],
        "season_last_game": [d("2021-05-20")] * 7,
        "games_remaining_at_arrival": [34] * 6 + [15], "season_games": [34] * 7})
    F = rd.follow_on(E)
    assert F.follow_on.tolist() == [rd.FOLLOW[4], rd.FOLLOW[0], rd.FOLLOW[2], rd.FOLLOW[3], rd.FOLLOW[1], rd.FOLLOW[5],
                                    rd.FOLLOW[2]]
    assert F.onward_lag.tolist()[2] == "1–3 days" and F.days_to_next.tolist()[2] == 1
    assert pd.isna(F.onward_lag.iloc[1])                       # after season end: no onward lag
    assert F.arrival_group.tolist() == [rd.GROUPS[0]] * 6 + ["Dec–Jan"]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def N():
    if not rd.NUMBERS_JSON.exists():
        pytest.skip("redesign not run")
    return json.loads(rd.NUMBERS_JSON.read_text())


def test_starting_counts_and_old_sample_reproduce(N):
    saved = json.loads(ptj.NUMBERS_JSON.read_text())
    assert N["reliable"] == {k: saved["waterfall"][k]["reliable"] for k in rd.KINDS}
    assert N["old_primary"] == {k: saved["primary"][k] for k in rd.KINDS}
    for k in rd.KINDS:
        assert [s["remaining"] for s in N["old_steps"][k] if s["step"] != "OLD PRIMARY SAMPLE"] == \
            [s["n"] for s in saved["design"][k][1:]] + ([N["old_primary"][k]] if k == "permanent" else [])


def test_waterfall_reconciles_every_exclusion():
    W = pd.read_csv(rd.WATERFALL_CSV)
    for k in rd.KINDS:
        up = W[(W.block == "upstream") & (W.kind == k)]
        cand = int(up.loc[up.step == "candidate episodes", "remaining"].iloc[0])
        rel = int(up.loc[up.step == "reliable playing-time outcome", "remaining"].iloc[0])
        assert cand - up[up.step.str.startswith("excluded")].removed.sum() == rel
        for block, last in (("revised", "REVISED PRIMARY SAMPLE"), ("old", "OLD PRIMARY SAMPLE")):
            b = W[(W.block == block) & (W.kind == k)]
            assert rel - b.removed.sum() == int(b.loc[b.step == last, "remaining"].iloc[0])
        rv = W[(W.block == "revised") & (W.kind == k)]
        assert int(rv.loc[rv.step.str.contains("other reason"), "removed"].iloc[0]) == 0
        assert int(rv.loc[rv.step.str.contains("season remaining"), "removed"].iloc[0]) == 0


def test_revised_sample_has_no_season_fraction_rule(N):
    sv = N["sensitivity"]
    rev = sv["REVISED PRIMARY: reliable + loan return + first spell"]
    assert rev["loan_N"] == N["revised"]["loan"] and rev["permanent_N"] == N["revised"]["permanent"]
    assert sv["revised + two-thirds rule (= old primary)"]["loan_N"] == N["old_primary"]["loan"]
    assert N["restored_n"] == rev["loan_N"] + rev["permanent_N"] - N["old_primary"]["loan"] - N["old_primary"]["permanent"]
    groups = [N["arrival"][f"group: {g}"] for g in rd.GROUPS]
    for k in rd.KINDS:
        assert sum(g[f"{k}_N"] for g in groups) == N["revised"][k]       # every arrival kept, January included
    assert N["arrival"]["January only: January"]["loan_N"] > 0


def test_followon_audit_partitions_each_scope():
    F = pd.read_csv(rd.FOLLOWON_CSV, keep_default_na=False)
    for sc, g in F.groupby("scope"):
        cats = g[g.lag == ""]
        assert cats.n.sum() == cats.scope_n.iloc[0]
        lag = g[g.lag != ""]
        assert lag.n.sum() == int(cats.loc[cats.follow_on == rd.FOLLOW[2], "n"].iloc[0])
        cum = lag.cumulative_n.astype(float).astype(int)
        assert (cum.diff().fillna(lag.n.iloc[0]) == lag.n).all() and cum.iloc[-1] == lag.n.sum()


def test_zero_minute_decomposition(N):
    z = N["zeros"]
    w = [z["onward_loan_within"][str(x)] for x in rd.ONWARD_DAYS]
    assert w == sorted(w) and w[-1] <= z["onward_loan_any"] <= z["permanent_zero"]
    assert sum(z["by_follow"].values()) == z["permanent_zero"]


def test_outputs_new_old_unchanged_and_generated_numbers(N):
    assert _sha(ptj.ROOT / "data" / "outputs" / "rebuild" / "stage1c_canonical_transfers.csv") == STAGE1C_SHA256
    for name, h in N["old_output_sha256"].items():
        assert _sha(rd.OUT / name) == h, name
    assert not {p.name for p in rd.WRITES} & set(N["old_output_sha256"])
    for p in rd.WRITES:
        assert p.exists(), p
    md = rd.REPORT_MD.read_text()
    for v in (N["revised"]["loan"], N["revised"]["permanent"], N["reliable"]["loan"], N["reliable"]["permanent"],
              N["old_primary"]["loan"], N["old_primary"]["permanent"], N["zeros"]["permanent_zero"]):
        assert f0(v) in md
