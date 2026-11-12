"""Week 10: Wikipedia league lists, tier bands, movement direction and the 29 x 29 flow matrices."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import league_tiers_wiki as w
from src.analysis import week10_followup as wk


S = w.SERIES_BY_CODE


@pytest.mark.parametrize("title,code,expect", [
    ("2015–16 Belgian Second Division", "be.2", (2015, 2)),
    ("2017–18 Belgian First Division B", "be.2", (2017, 2)),
    ("2015–16 Belgian First Division B", "be.2", None),          # outside that name's season window
    ("2016–17 Belgian First Division B", "be.1", None),          # fullmatch: not the first division
    ("2019–20 Football League (Greece)", "gr.2", None),          # tier 3 from 2019/20
    ("2019–20 Super League Greece 2", "gr.2", (2019, 2)),
    ("2019 Campeonato Brasileiro Série A", "br.1", (2018, 1)),   # calendar year Y -> season Y-1 (as Week 9)
    ("Apertura 2016 Liga MX", "mx.1", (2016, 1)),
    ("Clausura 2017 Liga MX", "mx.1", (2016, 1)),
    ("2010 Russian Premier League", "ru.1", None),               # calendar-year era skipped
    ("2011–12 Russian Premier League", "ru.1", (2011, 1)),
    ("2015–17 Serie A", "it.1", None),                           # not consecutive years
])
def test_title_classification(title, code, expect):
    r = w.classify_title(title, S[code])
    assert (None if r is None else (r["season"], r["tier"])) == expect


def _table(names, extra=""):
    rows = "".join(f'<tr><td>{i + 1}</td><td><a href="/wiki/{n}" title="{n} FC">{n}</a>{extra}</td><td>30</td></tr>'
                   for i, n in enumerate(names))
    return f'<table class="wikitable"><tr><th>Pos</th><th>Team</th><th>Pld</th></tr>{rows}</table>'


def test_division_teams_groups_and_rounds():
    a = [f"A{i}" for i in range(10)]
    b = [f"B{i}" for i in range(9)]
    html = _table(a, " (C)") + _table(b) + _table(a[:6]) + _table(["A1", "X1", "X2", "X3"])
    teams, status = w.division_teams(html)
    assert status == "ok_2_groups"
    assert sorted(t["text"] for t in teams) == sorted(a + b)       # rounds and play-off tables ignored
    assert all(t["title"].endswith(" FC") for t in teams)


def test_direction_rules():
    nan = np.nan
    f_lb = np.array([1, 2, 1, 3, 1, 3, nan, 2])
    f_ex = np.array([True, True, True, False, True, False, False, False])
    t_lb = np.array([2, 1, 1, 1, 3, 3, 1, 1])
    t_ex = np.array([True, True, True, True, False, False, True, True])
    got = wk.direction(f_lb, f_ex, t_lb, t_ex)
    assert list(got) == ["down", "up", "same level", "up", "down", "indeterminate", "unknown", "up"]


def test_flow_matrix_shape_and_order():
    D = pd.DataFrame({"from_flow_category": ["England T1", "Other", None, "Turkey T2"],
                      "to_flow_category": ["Germany T1", "England T2", "Other", "Other"]})
    m = wk.flow_matrix(D)
    assert m.shape == (29, 29) and list(m.index) == wk.CATS and list(m.columns) == wk.CATS
    assert m.to_numpy().sum() == 3                                 # the row with an unclassified side is omitted
    assert m.loc["England T1", "Germany T1"] == 1 and m.loc["Turkey T2", "Other"] == 1
    assert wk.CATS[-1] == "Other" and len(set(wk.CATS)) == 29


# ---- audit corrections (12 Nov 2026) ----
def test_subset_fallback_disabled_and_initials_count():
    assert w.USE_SUBSET_FALLBACK is False
    assert w._word("y") and not w._word("sp") and w._word("malatyaspor") and not w._word("1903")


def test_reject_after_end_year():
    tm = {"names": pd.DataFrame({"club_id": [204, 28107], "name": ["Lierse SK (- 2018)", "Lierse Kempenzonen"]})}
    a = pd.DataFrame({"season": [2017, 2024], "club_id": pd.array([204, 204], dtype="Int64"),
                      "match_method": ["alias_learned", "alias_learned"], "club_name_tm": ["x", "x"],
                      "unmatched_reason": [None, None]})
    r = w.reject_after_end(a, tm)
    assert r.club_id.iloc[0] == 204 and pd.isna(r.club_id.iloc[1])
    assert r.unmatched_reason.iloc[1].startswith("matched club ended in 2018")


def test_clash_catches_transliteration_and_shared_words():
    unm = {("Ukraine", 2023, 2): [({"metalurh", "zaporizhzhia"}, {"metalurh zaporizhzhia"}, False)]}
    ctok = {1: ({"mfk", "metalurg", "zaporizhya"}, {"mfk metalurg zaporizhya"}, False),
            2: ({"metalurh"}, {"metalurh"}, False),
            3: ({"vilkhivtsi"}, {"sk vilkhivtsi"}, False)}
    clash = wk.make_clash(unm, ctok)
    assert clash(1, "Ukraine", 2023, (2,))          # similarity (no shared word)
    assert clash(2, "Ukraine", 2023, (2,))          # shared word
    assert not clash(3, "Ukraine", 2023, (2,))      # unrelated club stays classifiable


def test_final_lookup_has_no_malatyaspor_error():
    from pathlib import Path
    p = Path(wk.V2) / "club_season_league_v2.csv"
    if not p.exists():
        pytest.skip("lookup not built")
    L = pd.read_csv(p)
    y = L[L.club_id == 19789]
    assert y.season.min() >= 2015, y          # Yeni Malatyaspor rows only from its own seasons


# ---- Week 10 LaTeX report generator ----
def test_report_fill_and_number_scan():
    from src.analysis import week10_latex as wl
    with pytest.raises(KeyError):
        wl.fill("coverage <<missing_key>>", {})
    assert wl.fill("N = <<n>>", {"n": "18,090"}) == "N = 18,090"
    scan = wl.number_scan({"t": "Loans are 41.3\\% of <<x>> arrivals (Week 9, 2012/13, Question 2, 365 days)."})
    assert list(scan[~scan.allowed].number) == ["41.3"]          # a hand-typed result is caught
    assert wl.number_scan({"MAIN": wl.MAIN, "APPENDIX": wl.APPENDIX}).allowed.all()


def test_report_formatting():
    from src.analysis import week10_latex as wl
    assert wl.c(-0.0049) == "0.00" and wl.c(-0.811) == "$-0.81$" and wl.c(1.28) == "+1.28"
    assert wl.esc("a_b & 5% -> c") == r"a\_b \& 5\% $\rightarrow$ c"
