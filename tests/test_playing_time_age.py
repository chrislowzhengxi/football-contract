"""Age analysis: fixed cutoffs, groups partition the revised sample, contrasts are internally consistent."""
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import playing_time_age as pa
from src.analysis.daniel_scope_final import f0

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


def test_age_model_uses_groups_not_raw_age():
    rng = np.random.default_rng(1)
    n = 600
    C = pd.DataFrame({"loan": rng.integers(0, 2, n), "age_years": rng.integers(17, 33, n),
                      "remain": rng.random(n), "log_relative_value": rng.normal(-3, 1, n),
                      "log_squad_value": rng.normal(18, 1, n), "position_clean": rng.choice(["Defender", "Attack"], n),
                      "league_season": rng.choice(["A_2019", "B_2019", "A_2020"], n), "b_club_id": rng.integers(0, 40, n),
                      "age": rng.uniform(17, 33, n)})
    C["kind"] = np.where(C.loan == 1, "loan", "permanent")
    C["share"] = (0.4 + 0.05 * C.loan - 0.04 * C.loan * (C.age_years > 23) + rng.normal(0, .1, n)).clip(0, 1)
    G = pa.add_groups(C, 23)
    assert ((G.older == 1) == (G.age_years >= 24)).all()
    m = pa.age_model(G)
    assert abs(m["older"]["coef"] - (m["young"]["coef"] + m["interaction"]["coef"])) < 1e-12
    names = pa.reg.fit(G, ["loan", "older", "loan_x_older"] + pa.CONTROLS, fe="league_season",
                       cats=["position_clean"])["res"].params.index
    assert not any(x in names for x in ("age", "age_c", "age_c2"))


@pytest.fixture(scope="module")
def N():
    if not pa.NUMBERS_JSON.exists():
        pytest.skip("age analysis not run")
    return json.loads(pa.NUMBERS_JSON.read_text())


def test_cutoffs_fixed_and_revised_sample_reproduces(N):
    assert N["cutoff"] == 23 and N["sensitivity_cutoffs"] == [22, 24]
    assert set(N["cuts"]) == {"22", "23", "24"}
    red = json.loads((pa.OUT / "playing_time_sample_redesign_numbers.json").read_text())
    assert N["revised"] == red["revised"]


def test_groups_partition_and_contrasts_are_consistent(N):
    for cut, rec in N["cuts"].items():
        cells = rec["cells"]
        for k in pa.KINDS:
            n = cells[f"revised sample | {k}, young"]["N"] + cells[f"revised sample | {k}, older"]["N"]
            assert n == N["age_distribution"][k]["N"]                      # everyone with a birth date, once
            assert N["age_distribution"][k]["N"] + N["age_distribution"][k]["missing"] == N["revised"][k]
        for lab in ("young", "older"):
            raw = rec[f"raw | revised sample | {lab}"]["coef"]
            assert abs(raw - (cells[f"revised sample | loan, {lab}"]["mean"]
                              - cells[f"revised sample | permanent, {lab}"]["mean"])) < 1e-12
        a = rec["adjusted"]
        assert abs(a["older"]["coef"] - a["young"]["coef"] - a["interaction"]["coef"]) < 1e-12


def test_outputs_new_old_unchanged_and_numbers_generated(N):
    assert _sha(pa.OUT / "stage1c_canonical_transfers.csv") == STAGE1C_SHA256
    for name, h in N["old_output_sha256"].items():
        assert _sha(pa.OUT / name) == h, name
    assert not {p.name for p in pa.WRITES} & set(N["old_output_sha256"])
    for p in pa.WRITES:
        assert p.exists(), p
    md = pa.REPORT_MD.read_text()
    rec = N["cuts"]["23"]
    for lab in ("young", "older"):
        assert f"{100 * rec['adjusted'][lab]['coef']:+.1f} pp" in md
        assert f0(rec["cells"][f"revised sample | loan, {lab}"]["N"]) in md
    M = pd.read_csv(pa.MODELS_CSV)
    assert set(M.cutoff_young_max_age.dropna().astype(int)) == {22, 23, 24}
