"""Professor data overview: every count comes from the lineage outputs; the LaTeX is paste-safe."""
import json

import pytest

from src.analysis import data_lineage as dl
from src.analysis import professor_data_overview as pdo
from src.analysis.daniel_scope_final import f0


@pytest.fixture(scope="module")
def docs():
    if not (pdo.MD.exists() and pdo.TEX.exists()):
        pytest.skip("professor data overview not generated")
    return pdo.MD.read_text(), pdo.TEX.read_text(), json.loads(dl.NUMBERS_JSON.read_text())


def test_counts_match_lineage(docs):
    md, tex, N = docs
    ep = N["episodes"]
    must = [ep["rows"], ep["episodes"], ep["loans"], ep["permanent_pooled"], N["master"]["rows"], N["loans"]["realised"],
            N["loans"]["early_termination"], N["master"]["membership"]["in_revised_sample"]]
    must += [s["removed"] for s in ep["steps"]]
    for n in must:
        assert f0(n) in md and f0(n) in tex, n
    assert f"{N['master']['columns']} columns" in md


def test_regenerates_identically(docs):
    md, tex, _ = docs
    c = pdo.content(pdo.load())
    assert pdo.render_md(c) == md and pdo.render_tex(c) == tex


def test_latex_is_ascii_and_balanced(docs):
    _, tex, _ = docs
    assert all(ord(ch) < 128 for ch in tex)
    assert tex.count(r"\begin{") == tex.count(r"\end{")
    assert r"\section*{Data overview}" in tex
