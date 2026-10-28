"""Variable provenance: every column classified once, every input real, and no LLM on any value's execution path."""
import pandas as pd
import pytest

from src.analysis import data_lineage as dl
from src.analysis import variable_provenance as vp
from src.stage1c.build import CANONICAL_COLUMNS


@pytest.fixture(scope="module")
def M():
    if not vp.MATRIX_CSV.exists():
        pytest.skip("variable provenance not generated")
    return pd.read_csv(vp.MATRIX_CSV, keep_default_na=False)


def test_every_column_exactly_once(M):
    assert list(M.loc[M.scope == "stage1c", "variable"]) == CANONICAL_COLUMNS
    assert list(M.loc[M.scope == "master", "variable"]) == list(pd.read_csv(dl.MASTER_CSV, nrows=0).columns)
    assert M.loc[M.scope == "report", "variable"].is_unique
    assert set(vp.REQUIRED) <= set(M.loc[M.scope == "report", "variable"])


def test_categories_and_inputs(M):
    assert set(M.provenance_category) <= set(vp.CATEGORIES)
    assert M.input_tokens.ne("").all()
    assert M.loc[M.scope.isin(["stage1c", "master"]), "provenance_category"].ne(vp.STAT).all()


def test_llm_marks(M):
    assert set(M.llm_inference_used_for_value) <= {"YES", "NO", "UNKNOWN"}
    assert not M.llm_inference_used_for_value.eq("YES").any()
    unknown = M[M.llm_inference_used_for_value.eq("UNKNOWN")]
    assert unknown.provenance_category.eq(vp.EXT).all()


def test_no_llm_module_reachable_from_producing_code():
    A = vp.llm_audit()
    assert "src.research_contract" in A["llm_call_modules"]          # the scan does find the real LLM code
    assert not A["llm_modules_reached"]
    assert not A["cache_reader_callers_among_producers"]
