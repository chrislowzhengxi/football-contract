"""Country-specific loan-ending calendar: clustering, guardrails, application, reconciliation."""
import json

import numpy as np
import pandas as pd

from src.analysis import loan_calendar_rule as lc

from tests.test_loan_episodes import real  # noqa: F401  (fixture)


def days_of(spec):
    """spec: {mmdd: (count, years)} -> (doy series, year series)."""
    d, y = [], []
    for md, (n, years) in spec.items():
        for i in range(n):
            d.append(md)
            y.append(2010 + i % years)
    return lc.doy(pd.Series(d)), pd.Series(y)


def test_clusters_wrap_the_year_boundary_and_centre_on_the_mode():
    d, y = days_of({"12-30": (5, 3), "12-31": (10, 5), "01-01": (8, 4), "01-03": (2, 2), "01-05": (1, 1)})
    cl = lc.make_clusters(d, y, tol=3)
    top = cl.iloc[0]
    assert top.center_mmdd == "12-31" and top["count"] == 25          # 12-30, 12-31, 01-01, 01-03
    assert top.from_mmdd == "12-30" and top.to_mmdd == "01-03"
    assert lc.circ_dist(lc.doy(pd.Series(["12-31"]))[0], lc.doy(pd.Series(["01-01"]))[0]) == 1


def test_selection_guardrails():
    d, y = days_of({"06-30": (80, 10), "12-31": (8, 1), "01-30": (6, 5), "03-15": (1, 3)})
    cl = lc.make_clusters(d, y, tol=3)
    kept, cov, why = lc.select(cl, d, target=0.99, tol=3, y_min=3, s_min=0.02, k_max=5)
    assert [lc.mmdd_of(c) for c in kept] == ["06-30", "01-30"]         # 12-31 skipped: one year only
    assert why == "next cluster below the minimum share" and np.isclose(cov, 86 / 95)
    kept, _, why = lc.select(cl, d, target=0.99, tol=3, y_min=1, s_min=0.0, k_max=2)
    assert len(kept) == 2 and why == "cluster cap reached"
    kept, cov, why = lc.select(cl, d, target=0.8, tol=3)
    assert [lc.mmdd_of(c) for c in kept] == ["06-30"] and why == "target reached"
    assert np.isclose(cov, lc.coverage(d, kept, 3))


def test_apply_never_relearns_and_leaves_unclassified_without_a_calendar():
    cal = {"domestic: X": {"centers": [int(lc.doy(pd.Series(["06-30"]))[0])], "coverage": 1.0, "N": 100,
                           "stopped": "target reached"}}
    frozen = json.dumps(cal, sort_keys=True)
    R = pd.DataFrame({"calendar_key": ["domestic: X", "domestic: X", "", "domestic: Y"],
                      "context": ["domestic", "domestic", "one or both clubs unmapped", "domestic"],
                      "ending_mmdd": ["07-02", "07-04", "06-30", "06-30"]})
    R["doy"] = lc.doy(R.ending_mmdd)
    out = lc.apply(R, cal)
    assert json.dumps(cal, sort_keys=True) == frozen
    assert out.calendar_class.tolist() == ["calendar_normal_domestic", "off_calendar_domestic",
                                           "geography_unclassified", "geography_unclassified"]
    assert out.matched_standard_mmdd.iloc[2:].isna().all()
    assert out.unclassified_reason.tolist()[2:] == ["club country unknown", "country below the history threshold"]


def test_real_data_classification_reconciles(real):  # noqa: F811
    N, tb = lc.compute(real["u"])
    R = tb["R"]
    assert R.loan_event_id.is_unique and len(R) == N["realised"]
    assert R.calendar_class.isin(lc.CLASSES).all()
    assert sum(N["by_class"].values()) == N["realised"]
    assert N["population"]["domestic"] + N["population"]["cross_border"] + N["population"]["unmapped"] == len(R)
    assert not (R.context.eq("domestic") & R.calendar_class.str.endswith("cross_border")).any()
    assert not (R.context.eq("one or both clubs unmapped") & (R.calendar_class != "geography_unclassified")).any()
    dom = R[R.context == "domestic"]
    by_key = dom.groupby("calendar_key").size()
    learned = [k for k in by_key.index if k in N["calendars"]]
    assert by_key[learned].sum() + N["unclassified"].get("country below the history threshold", 0) == len(dom)
    for k, v in N["calendars"].items():                     # learned from non-COVID endings only
        g = R[(R.calendar_key == k) & ~R.covid]
        assert v["N"] == len(g)
    S = tb["sens"]
    assert (S.N == len(R)).all() and (S.classifiable + S.unclassified == S.N).all()


def test_outputs_are_new_files():
    from src.analysis import loan_country_dates as cd, loan_timing_exploration as lte
    old = {p.name for p in (cd.REPORT_MD, cd.COUNTRY_CSV, cd.RULES_CSV, lte.REPORT_MD, lte.EPISODES_CSV, lte.COUNTRY_CSV)}
    new = {p.name for p in (lc.SUMMARY_MD, lc.RULES_CSV, lc.CLASSIFIED_CSV, lc.EDGE_CSV, lc.SENS_CSV, lc.XB_CSV)}
    assert not old & new
