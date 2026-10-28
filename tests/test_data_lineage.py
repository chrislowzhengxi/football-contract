"""Data-lineage documentation: outputs agree with the saved samples, every variable is documented, nothing else changed."""
import json

import pandas as pd
import pytest

from src.analysis import data_lineage as dl
from src.analysis.data_lineage_specs import STAGE1C, VARS
from src.stage1c.build import CANONICAL_COLUMNS

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


@pytest.fixture(scope="module")
def out():
    if not dl.NUMBERS_JSON.exists():
        pytest.skip("data lineage not run")
    N = json.loads(dl.NUMBERS_JSON.read_text())
    M = pd.read_csv(dl.MASTER_CSV, low_memory=False)
    D = pd.read_csv(dl.VARDICT_CSV, keep_default_na=False)
    return N, M, D


def test_specs_cover_every_stage1c_column_once():
    assert set(STAGE1C) == set(CANONICAL_COLUMNS)
    assert len({v.variable for v in VARS}) == len(VARS)


def test_stage1c_dictionary_has_every_column(out):
    D1 = pd.read_csv(dl.STAGE1C_DICT_CSV, keep_default_na=False)
    assert list(D1.column) == CANONICAL_COLUMNS
    assert (D1.n_rows == 175_165).all()
    assert D1.plain_english_meaning.ne("").all() and D1.missing_value_meaning.ne("").all()


def test_master_grain_and_dictionary(out):
    N, M, D = out
    assert M.event_id.is_unique and len(M) == N["episodes"]["episodes"]
    assert list(D.variable) == list(M.columns)
    der = D[D.raw_or_derived.str.startswith("derived")]
    assert der.source_columns.ne("").all() and der.formula_or_rule.ne("").all()


def test_waterfalls_add_up(out):
    N, _, _ = out
    ep = N["episodes"]
    assert ep["steps"][0]["before"] == 175_165 and ep["steps"][-1]["after"] == ep["episodes"]
    assert sum(s["removed"] for s in ep["steps"]) + ep["episodes"] == 175_165
    v2 = N["calendar"]["v2"]
    assert v2["normal"] + v2["off"] + v2["country_unknown"] + v2["below_threshold"] == v2["realised"]
    for k in ("loan", "permanent"):
        w = N["playing_time"]["waterfall"][k]
        assert w["steps"][0]["n"] - sum(s["removed"] for s in w["steps"][1:]) == w["reliable"]
        assert w["reliable"] - sum(s["removed"] for s in w["revised_steps"]) == w["revised"]


def test_master_flags_match_saved_samples(out):
    _, M, _ = out
    red = json.loads((dl.OUT / "playing_time_sample_redesign_numbers.json").read_text())
    inj = json.loads((dl.OUT / "playing_time_injury_adjusted_numbers.json").read_text())
    assert M.in_revised_sample.sum() == sum(red["revised"].values())
    assert M.in_injury_usable.sum() == sum(inj["usable"].values())
    assert set(M.loc[M.in_injury_model_sample, "event_id"]) == set(inj["model_event_ids"])
    assert not (M.in_injury_model_sample & ~M.in_injury_usable).any()
    assert not (M.in_injury_usable & ~M.in_revised_sample).any()
    assert not (M.in_revised_sample & ~M.pt_reliable).any()
    assert M.minutes_share.notna().eq(M.pt_reliable).all()          # an outcome exists only where it is reliable


def test_off_calendar_is_not_a_filter(out):
    N, M, _ = out
    assert not any(N["code_check"].values())
    rv = M[M.in_revised_sample & M.kind.eq("loan")]
    assert rv.off_calendar.eq(True).sum() > 0                       # off-calendar loans stay in the samples


def test_stage1c_and_other_outputs_unchanged(out):
    N, _, _ = out
    assert _sha(dl.OUT / "stage1c_canonical_transfers.csv") == STAGE1C_SHA256
    from src.analysis import (appearance_audit, data_guide_professor, playing_time_qa, professor_data_overview,
                              professor_empirical, variable_provenance)
    downstream = {p.name for m in (variable_provenance, data_guide_professor, professor_data_overview, playing_time_qa,
                                   professor_empirical, appearance_audit) for p in m.WRITES}                       # regenerated after the lineage run, from its outputs
    latex_build = (".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz")   # editor auto-builds
    for name, h in N["old_output_sha256"].items():
        if name not in downstream and not name.endswith(latex_build):
            assert _sha(dl.OUT / name) == h, name
    assert not {p.name for p in dl.WRITES} & set(N["old_output_sha256"])
