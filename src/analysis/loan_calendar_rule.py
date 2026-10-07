"""A country-specific loan-ending calendar: learn normal ending dates, isolate off-calendar edge cases.

    python -m src.analysis.loan_calendar_rule

Proposed rule, not a production classification. Uses the existing loan
episodes (one row per loan) and the snapshot's deterministic club countries.
The calendar is learned from realised endings outside the COVID window and then
applied, unchanged, to every realised ending, COVID ones included. The dates are
empirically common loan-ending dates in our data; the repository holds no
official transfer-window calendar, so none is called one.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from .daniel_scope_final import f0, md_table
from .loan_country_dates import COVID_END, COVID_START, context
from .loan_episodes import ROOT
from .loan_timing_exploration import episode_table, next_moves, share
from .stage1_scope_report import display_country as dc

OUT = ROOT / "data" / "outputs" / "rebuild"
SUMMARY_MD = OUT / "loan_calendar_rule_summary.md"
RULES_CSV = OUT / "loan_calendar_country_rules.csv"
CLASSIFIED_CSV = OUT / "loan_calendar_classified_loans.csv"
EDGE_CSV = OUT / "loan_calendar_edge_cases.csv"
SENS_CSV = OUT / "loan_calendar_sensitivity.csv"
XB_CSV = OUT / "loan_calendar_cross_border_analysis.csv"
NUMBERS_JSON = OUT / "loan_calendar_rule_numbers.json"
FIGS = {"clusters": "figures/loan_calendar_fig1_country_clusters.png",
        "edge": "figures/loan_calendar_fig2_normal_vs_off_calendar.png",
        "country": "figures/loan_calendar_fig3_off_calendar_by_country.png"}

# Primary specification (chosen from the data; see the report)
TOL = 3              # days either side of a cluster centre
TARGET = 0.90        # stop adding clusters once this share of the calendar's training endings is covered
N_MIN = 100          # a domestic country calendar needs this many realised non-COVID training endings
Y_MIN = 3            # a cluster is standard only if it recurs in at least this many calendar years
S_MIN = 0.02         # ... and holds at least this share of the calendar's training endings
K_MAX = 5            # ... and at most this many clusters per calendar
XB_SEGMENTS = "pooled"   # one cross-border calendar; segmentations tested in the report add nothing

CLASSES = ("calendar_normal_domestic", "calendar_normal_cross_border", "off_calendar_domestic",
           "off_calendar_cross_border", "geography_unclassified")
CIRCLE = 366


# ---------------------------------------------------------------------------
# Calendar arithmetic
# ---------------------------------------------------------------------------

def doy(mmdd: pd.Series) -> pd.Series:
    """Month-day as a day index 0..365 on a leap-year circle (29 February has its own slot)."""
    return pd.to_datetime("2000-" + mmdd.astype(str)).dt.dayofyear - 1


def mmdd_of(d: int) -> str:
    return (pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(d))).strftime("%m-%d")


def circ_dist(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, CIRCLE - d)


def make_clusters(days: pd.Series, years: pd.Series, tol: int) -> pd.DataFrame:
    """Greedy circular clustering. The most frequent unassigned day becomes a centre; every unassigned day
    within +/- tol joins it; repeat. Ties go to the earlier day. 31 December and 1 January are 1 day apart."""
    cnt = days.value_counts()
    left = set(cnt.index)
    rows = []
    n = len(days)
    while left:
        c = max(left, key=lambda d: (cnt[d], -d))
        mem = sorted(d for d in left if circ_dist(d, c) <= tol)
        m = days.isin(mem)
        y = years[m].value_counts()
        offs = sorted(((d - c + CIRCLE // 2) % CIRCLE) - CIRCLE // 2 for d in mem)
        rows.append({"center": int(c), "center_mmdd": mmdd_of(c), "from_mmdd": mmdd_of((c + offs[0]) % CIRCLE),
                     "to_mmdd": mmdd_of((c + offs[-1]) % CIRCLE), "count": int(m.sum()), "share": m.sum() / n,
                     "years": int(y.size), "top_year": int(y.index[0]), "top_year_share": float(y.iloc[0] / m.sum())})
        left -= set(mem)
    out = pd.DataFrame(rows).sort_values(["count", "center"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["cumulative_share"] = out["share"].cumsum()
    return out


def coverage(days: pd.Series, centers: list[int], tol: int) -> float:
    if not centers or not len(days):
        return 0.0
    d = np.min(np.stack([circ_dist(days.to_numpy(), c) for c in centers]), axis=0)
    return float((d <= tol).mean())


def select(cl: pd.DataFrame, days: pd.Series, target: float, tol: int, y_min: int = Y_MIN,
           s_min: float = S_MIN, k_max: int = K_MAX) -> tuple[list[int], float, str]:
    """Add clusters in frequency order until the target is met; skip non-recurring clusters; stop at the
    first cluster below s_min or when k_max clusters are held. Returns centres, coverage and why it stopped."""
    kept: list[int] = []
    for r in cl.itertuples():
        if coverage(days, kept, tol) >= target:
            return kept, coverage(days, kept, tol), "target reached"
        if len(kept) >= k_max:
            return kept, coverage(days, kept, tol), "cluster cap reached"
        if r.share < s_min:
            return kept, coverage(days, kept, tol), "next cluster below the minimum share"
        if r.years < y_min:
            continue
        kept.append(int(r.center))
    cov = coverage(days, kept, tol)
    return kept, cov, "target reached" if cov >= target else "no clusters left"


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def segment_cross_border(R: pd.DataFrame, how: str = XB_SEGMENTS) -> pd.Series:
    """Cross-border calendar key. 'pooled': one calendar. 'confederation pair': both UEFA / one UEFA /
    neither UEFA, which the analysis shows separates the calendar-year clubs from the European ones."""
    if how == "pooled":
        return pd.Series("cross-border", index=R.index)
    lu, bu = R.lender_confederation.eq("UEFA"), R.borrower_confederation.eq("UEFA")
    return pd.Series(np.select([lu & bu, lu ^ bu], ["cross-border: both UEFA", "cross-border: one UEFA"],
                               default="cross-border: neither UEFA"), index=R.index)


def population(u) -> pd.DataFrame:
    T = episode_table(u, next_moves(u))
    R = T[T.ending_realised].copy()
    R["context"] = context(R)
    R["covid"] = R.ending_date.between(COVID_START, COVID_END)
    R["doy"] = doy(R.ending_mmdd)
    R["year"] = R.ending_date.dt.year
    R["calendar_key"] = np.select(
        [R.context.eq("domestic"), R.context.eq("cross-border")],
        ["domestic: " + R.lender_country.fillna(""), segment_cross_border(R)], default="")
    return R


# ---------------------------------------------------------------------------
# Learn and apply
# ---------------------------------------------------------------------------

def learn(train: pd.DataFrame, tol: int = TOL, target: float = TARGET, n_min: int = N_MIN, y_min: int = Y_MIN,
          s_min: float = S_MIN, k_max: int = K_MAX) -> tuple[dict, pd.DataFrame]:
    """Calendars from the training endings only: {key: {"centers": [...], "coverage": .., "N": ..}}."""
    cal, tables = {}, []
    for key, g in train[train.calendar_key != ""].groupby("calendar_key"):
        if key.startswith("domestic") and len(g) < n_min:
            continue
        cl = make_clusters(g.doy, g.year, tol)
        kept, cov, why = select(cl, g.doy, target, tol, y_min, s_min, k_max)
        cal[key] = {"centers": kept, "coverage": cov, "N": len(g), "stopped": why}
        tables.append(cl.assign(calendar_key=key, N=len(g), retained=cl.center.isin(kept),
                                recurring=cl.years >= y_min))
    return cal, (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame())


def apply(R: pd.DataFrame, cal: dict, tol: int = TOL) -> pd.DataFrame:
    """Classify every ending against a fixed calendar. Never re-learns."""
    out = pd.DataFrame(index=R.index)
    has = R.calendar_key.isin(cal.keys())
    centers = R.calendar_key.map(lambda k: cal[k]["centers"] if k in cal else [])
    dist = [int(np.min(circ_dist(d, c))) if c else np.nan for d, c in zip(R.doy, centers)]
    near = [mmdd_of(c[int(np.argmin(circ_dist(d, c)))]) if c else None for d, c in zip(R.doy, centers)]
    out["days_to_nearest_standard"] = dist
    out["nearest_standard_mmdd"] = near
    normal = has & (out.days_to_nearest_standard <= tol)
    dom = R.context.eq("domestic")
    out["calendar_class"] = np.select(
        [~has, normal & dom, normal, dom], ["geography_unclassified", "calendar_normal_domestic",
                                            "calendar_normal_cross_border", "off_calendar_domestic"],
        default="off_calendar_cross_border")
    out["matched_standard_mmdd"] = out.nearest_standard_mmdd.where(normal)
    out["unclassified_reason"] = np.select(
        [has, R.context.eq("one or both clubs unmapped"), dom],
        ["", "club country unknown", "country below the history threshold"], default="no calendar for segment")
    return out


def shares(cls: pd.Series) -> dict:
    n = len(cls)
    classifiable = cls != "geography_unclassified"
    normal = cls.str.startswith("calendar_normal")
    return {"N": n, "classifiable": int(classifiable.sum()), "normal": int(normal.sum()),
            "off": int((classifiable & ~normal).sum()), "unclassified": int((~classifiable).sum()),
            "normal_share": float(normal.sum() / classifiable.sum()) if classifiable.any() else None,
            "off_share": float((classifiable & ~normal).sum() / classifiable.sum()) if classifiable.any() else None}


# ---------------------------------------------------------------------------
# Characterisation helpers
# ---------------------------------------------------------------------------

NEXT_CATS = ("permanent move to the borrower", "re-loan to the same borrower", "loan to a different club",
             "paid or undisclosed move to a third club", "free or no-fee move to a third club",
             "release (Without Club, Retired, ...)", "no observed next move", "ended without a return row", "other")
WITHIN = (1, 3, 7, 21, 60)


def next_category(R: pd.DataFrame) -> pd.Series:
    k = R.next_move_kind.fillna("")
    to_b = R.next_move_to_club_id == R.borrower_club_id
    return pd.Series(np.select(
        [~R.ending_is_return, k.eq("permanent_to_borrower"), k.eq("loan") & to_b, k.eq("loan"),
         k.eq("sale_to_third_club"), k.eq("free_move_to_third_club"), k.eq("to_placeholder"), k.eq("")],
        list(NEXT_CATS[7:8]) + list(NEXT_CATS[:7]), default="other"), index=R.index)


def profile(g: pd.DataFrame) -> dict:
    dur = g.loan_duration_days.dropna()
    gap = g.next_move_gap_days
    rec = {"N": len(g), "duration_median": float(dur.median()), "duration_p25": float(dur.quantile(.25)),
           "duration_p75": float(dur.quantile(.75)), "fee_bearing": float(g.ending_fee_on_return_eur.notna().mean()),
           "early_termination_label": float((g.economic_ending == "early_termination").mean()),
           "months": {int(m): float(v) for m, v in g.ending_date.dt.month.value_counts(normalize=True).items()}}
    for x in WITHIN:
        rec[f"next_within_{x}"] = float((gap <= x).mean())
    rec["next"] = {k: float(v) for k, v in g.next_category.value_counts(normalize=True).items()}
    return rec


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

DETAIL = ("England", "Scotland", "Japan", "Italy", "Spain", "Germany", "France", "Portugal", "Netherlands",
          "Belgium", "Brazil", "Argentina")


def compute(u=None) -> tuple[dict, dict]:
    u = u or loan_scope.universe()
    R = population(u)
    T_loans = len(u.loans)
    N: dict = {"loans": T_loans, "realised": len(R)}
    ctx = R.context.value_counts().to_dict()
    N["population"] = {"domestic": int(ctx.get("domestic", 0)), "cross_border": int(ctx.get("cross-border", 0)),
                       "unmapped": int(ctx.get("one or both clubs unmapped", 0)),
                       "covid": int(R.covid.sum()), "non_covid": int((~R.covid).sum()),
                       "by_context_covid": {f"{c} | {'COVID' if v else 'non-COVID'}": int(n)
                                            for (c, v), n in R.groupby(["context", "covid"]).size().items()}}
    N["covid_window"] = [str(COVID_START.date()), str(COVID_END.date())]
    assert sum(N["population"][k] for k in ("domestic", "cross_border", "unmapped")) == len(R)
    assert R.loan_event_id.is_unique

    # ---- learn (non-COVID) and apply (everything)
    train = R[~R.covid]
    assert not train.covid.any()
    cal, CL = learn(train)
    cal_frozen = json.dumps(cal, sort_keys=True)
    R = R.join(apply(R, cal))
    assert json.dumps(cal, sort_keys=True) == cal_frozen, "applying the rule changed it"
    R["next_category"] = next_category(R)
    R["old_followon_60d"] = R.current_follow_on_event_id.notna()
    R["covid_period"] = R.covid
    assert R.calendar_class.isin(CLASSES).all()
    assert R.loc[R.calendar_class == "geography_unclassified", "matched_standard_mmdd"].isna().all()
    assert not (R.context.eq("domestic") & R.calendar_class.str.endswith("cross_border")).any()
    assert not (R.context.eq("cross-border") & R.calendar_class.str.endswith("domestic")).any()
    N["calendars"] = {k: {**v, "centers_mmdd": [mmdd_of(c) for c in v["centers"]]} for k, v in cal.items()}
    N["primary"] = shares(R.calendar_class)
    N["by_class"] = R.calendar_class.value_counts().to_dict()
    N["unclassified"] = R.loc[R.calendar_class == "geography_unclassified", "unclassified_reason"].value_counts().to_dict()
    um = R[R.context == "one or both clubs unmapped"]
    N["unmapped_sides"] = {"lender only missing": int((um.lender_country.isna() & um.borrower_country.notna()).sum()),
                           "borrower only missing": int((um.lender_country.notna() & um.borrower_country.isna()).sum()),
                           "both missing": int((um.lender_country.isna() & um.borrower_country.isna()).sum())}

    # ---- COVID window check
    cls_ = R[R.calendar_class != "geography_unclassified"].copy()
    cls_["off"] = cls_.calendar_class.str.startswith("off")
    cls_["ym"] = cls_.ending_date.dt.to_period("Q").astype(str)
    q = cls_[(cls_.ending_date >= "2018-07-01") & (cls_.ending_date < "2022-07-01")].groupby("ym").agg(
        n=("off", "size"), off=("off", "mean"))
    N["quarterly_off"] = {k: {"n": int(v.n), "off": float(v.off)} for k, v in q.iterrows()}
    D = R[R.context == "domestic"]
    N["covid_examples"] = {
        "Italy 2020-08-31": int(((D.lender_country == "Italy") & (D.ending_date == "2020-08-31")).sum()),
        "Italy endings in 2020": int(((D.lender_country == "Italy") & (D.ending_date.dt.year == 2020)).sum()),
        "Italy 08-31 in other years": int(((D.lender_country == "Italy") & (D.ending_mmdd == "08-31")
                                            & (D.ending_date.dt.year != 2020)).sum()),
        "Brazil 2021-02-28": int(((D.lender_country == "Brazil") & (D.ending_date == "2021-02-28")).sum()),
        "Brazil 02-28 in other years": int(((D.lender_country == "Brazil") & (D.ending_mmdd == "02-28")
                                             & (D.ending_date != "2021-02-28")).sum())}

    # ---- guardrail evidence
    gr = []
    for key, g in train[train.calendar_key.str.startswith("domestic")].groupby("calendar_key"):
        cl = make_clusters(g.doy, g.year, TOL)
        need = {t: int((cl.cumulative_share < t - 1e-12).sum() + 1) for t in (0.85, 0.9, 0.95)}
        gr.append({"calendar_key": key, "N": len(g), **{f"clusters_for_{int(t * 100)}": v for t, v in need.items()},
                   "top_share": float(cl.share.iloc[0]),
                   "nonrecurring_in_top5": int((cl.head(5).years < Y_MIN).sum()),
                   "adequate": len(g) >= N_MIN})
    GR = pd.DataFrame(gr).sort_values("N", ascending=False)
    N["guardrail_evidence"] = GR.to_dict("records")
    dn = GR.set_index("calendar_key").N
    N["n_min_gap"] = {"smallest_adequate": int(dn[dn >= N_MIN].min()), "largest_inadequate": int(dn[dn < N_MIN].max())}
    rel = CL[CL.calendar_key.str.startswith("domestic") & (CL.N >= N_MIN)]
    N["cluster_share_years"] = {
        "clusters_share_ge_2pct": int((rel.share >= .02).sum()),
        "of_which_recurring": int(((rel.share >= .02) & (rel.years >= Y_MIN)).sum()),
        "clusters_1_to_2pct": int(rel.share.between(.01, .02, inclusive="left").sum()),
        "share_ge_2pct_years_min": int(rel.loc[rel.share >= .02, "years"].min())}
    variants = {"no guardrails (target only)": dict(y_min=1, s_min=0.0, k_max=99),
                f"recurring >= {Y_MIN} years only": dict(y_min=Y_MIN, s_min=0.0, k_max=99),
                f"+ minimum share {S_MIN:.0%}": dict(y_min=Y_MIN, s_min=S_MIN, k_max=99),
                f"+ at most {K_MAX} clusters (primary)": dict(y_min=Y_MIN, s_min=S_MIN, k_max=K_MAX),
                "at most 3 clusters": dict(y_min=Y_MIN, s_min=S_MIN, k_max=3)}
    gv = []
    for name, kw in variants.items():
        c2, _ = learn(train, **kw)
        s = shares(apply(R, c2).calendar_class)
        dom = {k: v for k, v in c2.items() if k.startswith("domestic")}
        gv.append({"variant": name, **s, "max_clusters": max(len(v["centers"]) for v in dom.values()),
                   "mean_clusters": float(np.mean([len(v["centers"]) for v in dom.values()]))})
    N["guardrail_variants"] = gv

    # ---- cross-border
    X = R[R.context == "cross-border"].copy()
    X["normal"] = X.calendar_class.str.startswith("calendar_normal")
    lu, bu = X.lender_confederation.eq("UEFA"), X.borrower_confederation.eq("UEFA")
    X["pair"] = np.select([lu & bu, lu & ~bu, ~lu & bu], ["both UEFA", "UEFA lender to non-UEFA borrower",
                                                         "non-UEFA lender to UEFA borrower"], default="neither UEFA")
    xb = []
    for how in ("pooled", "confederation pair"):
        Xh = X.drop(columns=[c for c in X.columns if c in ("calendar_key",)]).assign(
            calendar_key=segment_cross_border(X, how))
        ch, _ = learn(Xh[~Xh.covid])
        cls = apply(Xh, ch).calendar_class
        xb.append({"analysis": "segmentation", "segment": how, "n": len(Xh),
                   "normal_share": float(cls.str.startswith("calendar_normal").mean()),
                   "normal_share_non_covid": float(cls[~Xh.covid].str.startswith("calendar_normal").mean()),
                   "calendars": "; ".join(f"{k}: {', '.join(mmdd_of(c) for c in v['centers'])}" for k, v in ch.items())})
    brit = X.lender_country.isin(["England", "Scotland"]) & X.borrower_country.isin(["England", "Scotland"])
    Xb = X.assign(calendar_key=np.where(brit, "cross-border: England-Scotland", "cross-border: other"))
    cb, _ = learn(Xb[~Xb.covid])
    clsb = apply(Xb, cb).calendar_class
    xb.append({"analysis": "segmentation", "segment": "pooled + separate England-Scotland", "n": len(Xb),
               "normal_share": float(clsb.str.startswith("calendar_normal").mean()),
               "normal_share_non_covid": float(clsb[~Xb.covid].str.startswith("calendar_normal").mean()),
               "calendars": "; ".join(f"{k}: {', '.join(mmdd_of(c) for c in v['centers'])}" for k, v in cb.items())})
    for col, lab in (("pair", "confederation pair"), ("lender_country", "lender country"),
                     ("borrower_country", "borrower country")):
        for v, g in X.groupby(col):
            if len(g) < 50:
                continue
            top = g.ending_mmdd.value_counts(normalize=True).head(2)
            xb.append({"analysis": lab, "segment": v, "n": len(g), "normal_share": float(g.normal.mean()),
                       "normal_share_non_covid": float(g.loc[~g.covid, "normal"].mean()),
                       "calendars": ", ".join(f"{m} {100 * s:.0f}%" for m, s in top.items())})
    XB = pd.DataFrame(xb)
    N["cross_border"] = XB.to_dict("records")
    N["xb_brit_n"] = int(brit.sum())

    # ---- characterisation
    C = R[R.calendar_class != "geography_unclassified"].copy()
    C["state"] = np.where(C.calendar_class.str.startswith("calendar_normal"), "calendar-normal", "off-calendar")
    N["edge"] = {s: profile(g) for s, g in C.groupby("state")}
    N["edge_non_covid"] = {s: profile(g) for s, g in C[~C.covid].groupby("state")}
    off = C[C.state == "off-calendar"]
    N["off_by_calendar"] = {k: int(v) for k, v in off.calendar_key.value_counts().items()}
    N["off_top_dates"] = [{"calendar_key": k, "mmdd": m, "n": int(n)} for (k, m), n in
                          off.groupby(["calendar_key", "ending_mmdd"]).size().sort_values(ascending=False).head(12).items()]
    N["off_months"] = {int(m): int(v) for m, v in off.ending_date.dt.month.value_counts().items()}

    # ---- tolerance and date-rank evidence (non-COVID)
    Cn = C[~C.covid].copy()
    bands = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2–3"), (4, 7, "4–7"), (8, 14, "8–14"), (15, 30, "15–30"), (31, 400, "> 30")]
    N["distance_profile"] = []
    for lo, hi, lab in bands:
        g = Cn[Cn.days_to_nearest_standard.between(lo, hi)]
        N["distance_profile"].append({"band": lab, "n": len(g), "duration_median": float(g.loan_duration_days.median()),
                                      "next_within_1": float((g.next_move_gap_days <= 1).mean()),
                                      "january": float((g.ending_date.dt.month == 1).mean())})
    nm = Cn[Cn.days_to_nearest_standard.between(1, TOL)]
    N["near_miss"] = {"n": len(nm), "midseason": int(nm.nearest_standard_mmdd.str[:2].isin(["01", "02"]).sum()),
                      "january": float((nm.ending_date.dt.month == 1).mean())}
    N["return_share"] = float(R.ending_is_return.mean())
    rank_of = {}
    for key, v in cal.items():
        for i, c in enumerate(v["centers"], 1):
            rank_of[(key, mmdd_of(c))] = i
    C["matched_rank"] = [rank_of.get((k, m)) for k, m in zip(C.calendar_key, C.matched_standard_mmdd)]
    R["matched_rank"] = R.loan_event_id.map(C.set_index("loan_event_id").matched_rank)
    Cn = C[~C.covid]
    groups = {"main date (largest cluster)": Cn[Cn.matched_rank == 1],
              "secondary standard dates": Cn[Cn.matched_rank > 1], "off-calendar": Cn[Cn.state == "off-calendar"]}
    N["rank_profile"] = {k: profile(g) for k, g in groups.items()}
    capped = [k for k, v in cal.items() if v["stopped"] == "cluster cap reached"]
    N["capped_calendars"] = capped
    rest = Cn[~Cn.calendar_key.isin(capped)]
    N["off_share_without_capped"] = {"n": len(rest), "off": int((rest.state == "off-calendar").sum()),
                                     "share": float((rest.state == "off-calendar").mean())}
    capd = Cn[Cn.calendar_key.isin(capped)]
    N["off_share_capped"] = {"n": len(capd), "off": int((capd.state == "off-calendar").sum()),
                             "share": float((capd.state == "off-calendar").mean())}

    # ---- COVID
    cv = C.covid
    N["covid"] = {"off_cases": int(len(off)), "off_in_covid": int(off.covid.sum()),
                  "covid_classifiable": int(cv.sum()), "covid_off": int((cv & (C.state == "off-calendar")).sum()),
                  "noncovid_classifiable": int((~cv).sum()), "noncovid_off": int((~cv & (C.state == "off-calendar")).sum())}
    N["covid_anomalies"] = [{"calendar_key": k, "mmdd": m, "n": int(n)} for (k, m), n in
                            off[off.covid].groupby(["calendar_key", "ending_mmdd"]).size()
                            .sort_values(ascending=False).head(10).items()]
    N["shares_all"] = shares(R.calendar_class)
    N["shares_non_covid"] = shares(R.loc[~R.covid, "calendar_class"])
    c_all, _ = learn(R)
    N["learned_all_years"] = {**shares(apply(R, c_all).calendar_class),
                              "changed_calendars": [k for k in cal if k in c_all and cal[k]["centers"] != c_all[k]["centers"]]}

    # ---- sensitivity
    sens = []
    for target in (0.85, 0.90, 0.95):
        for tol in (1, 3, 7):
            c2, _ = learn(train, tol=tol, target=target)
            s = shares(apply(R, c2, tol=tol).calendar_class)
            app = apply(R, c2, tol=tol)
            dom = {k: v for k, v in c2.items() if k.startswith("domestic")}
            xbn = app.calendar_class[R.context == "cross-border"]
            sens.append({"target": target, "tolerance_days": tol, "primary": target == TARGET and tol == TOL, **s,
                         "countries_with_rule": len(dom),
                         "capped_calendars": ", ".join(sorted(label_of(k) for k, v in c2.items()
                                                              if v["stopped"] == "cluster cap reached")),
                         "calendars_below_target": ", ".join(sorted(label_of(k) for k, v in c2.items()
                                                                    if v["coverage"] < target)),
                         "mean_clusters_per_country": float(np.mean([len(v["centers"]) for v in dom.values()])),
                         "cross_border_normal_share": float(xbn.str.startswith("calendar_normal").mean())})
    S = pd.DataFrame(sens)
    assert (S.N == len(R)).all() and (S.classifiable + S.unclassified == S.N).all()
    N["sensitivity"] = S.to_dict("records")

    # ---- country detail
    det = []
    for key, v in cal.items():
        g = R[R.calendar_key == key]
        gtr = g[~g.covid]
        clusters = []
        if v["centers"]:
            dm = np.stack([circ_dist(gtr.doy.to_numpy(), c) for c in v["centers"]])
            nearest = dm.argmin(axis=0)                     # ties go to the larger (earlier-ranked) cluster
            within = dm.min(axis=0) <= TOL
        for i, c in enumerate(v["centers"]):
            m = within & (nearest == i)
            clusters.append({"center": mmdd_of(c), "from": mmdd_of((c - TOL) % CIRCLE), "to": mmdd_of((c + TOL) % CIRCLE),
                             "share_training": float(m.mean())})
        assert abs(sum(c["share_training"] for c in clusters) - v["coverage"]) < 1e-9
        off_all = g.calendar_class.str.startswith("off").mean()
        gr_row = GR[GR.calendar_key == key]
        det.append({"calendar_key": key, "N_training": v["N"], "N_all": len(g), "clusters": clusters,
                    "coverage_training": v["coverage"], "stopped": v["stopped"],
                    "off_share_all": float(off_all), "off_share_non_covid": float(g.loc[~g.covid].calendar_class.str.startswith("off").mean()),
                    "off_share_covid": float(g.loc[g.covid].calendar_class.str.startswith("off").mean()) if g.covid.any() else None,
                    "n_covid": int(g.covid.sum()),
                    "clusters_for_90_unguarded": int(gr_row.clusters_for_90.iloc[0]) if len(gr_row) else None})
    N["detail"] = det

    # ---- old 60-day follow-on rule vs calendar rule
    Rr = C[C.ending_is_return].copy()
    Rr["cell"] = np.select(
        [(Rr.state == "calendar-normal") & ~Rr.old_followon_60d, (Rr.state == "calendar-normal") & Rr.old_followon_60d,
         (Rr.state == "off-calendar") & ~Rr.old_followon_60d], ["A", "B", "C"], default="D")
    cells = {}
    for cell, g in Rr.groupby("cell"):
        cells[cell] = {"N": len(g), "share": len(g) / len(Rr), "duration_median": float(g.loan_duration_days.median()),
                       "january": float((g.ending_date.dt.month == 1).mean()),
                       "next": {k: float(v) for k, v in g.next_category.value_counts(normalize=True).head(4).items()},
                       "nonstandard_old": float(g.economic_ending.isin(loan_scope.NONSTANDARD).mean())}
    N["two_way"] = cells
    N["two_way_n"] = len(Rr)
    N["old_vs_new"] = {"old_followon": int(Rr.old_followon_60d.sum()), "new_off": int((Rr.state == "off-calendar").sum()),
                       "both": int(((Rr.state == "off-calendar") & Rr.old_followon_60d).sum()),
                       "old_nonstandard": int(Rr.economic_ending.isin(loan_scope.NONSTANDARD).sum()),
                       "old_nonstandard_and_off": int((Rr.economic_ending.isin(loan_scope.NONSTANDARD)
                                                       & (Rr.state == "off-calendar")).sum())}
    ex = []
    for cell in ("B", "C"):
        g = Rr[(Rr.cell == cell) & (Rr.loan_season >= "2018/19")]
        for r in g.sample(min(3, len(g)), random_state=7).itertuples():
            ex.append({"cell": cell, "player": r.player_name, "loan": f"{r.lender_club} → {r.borrower_club}",
                       "loan_start": str(r.loan_start_date.date()), "return": str(r.ending_date.date()),
                       "calendar": r.calendar_key, "nearest_standard": r.nearest_standard_mmdd,
                       "days_off": r.days_to_nearest_standard,
                       "next": r.next_category, "next_gap": None if pd.isna(r.next_move_gap_days) else int(r.next_move_gap_days)})
    N["examples"] = ex
    R["two_way_cell"] = R.loan_event_id.map(Rr.set_index("loan_event_id").cell)

    tables = {"R": R, "clusters": CL, "sens": S, "xb": XB, "guard": GR}
    return N, tables


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2b6cb0", "#dd6b20", "#a0aec0", "#1a202c", "#555", "#e2e8f0"
RANK_BLUES = ("#1e4e8c", "#3b78c3", "#6aa0db", "#9cc2ea", "#c9def3")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def label_of(key: str) -> str:
    return dc(key.split(": ", 1)[1]) if key.startswith("domestic") else "Cross-border (pooled)"


def figures(N: dict, R: pd.DataFrame) -> None:
    plt = _plt()
    (OUT / "figures").mkdir(exist_ok=True)

    # 1. retained clusters per calendar (training shares), sorted by off-calendar share
    det = sorted(N["detail"], key=lambda d: d["coverage_training"])
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(det) + 1.6))
    for y, d in enumerate(det):
        left = 0.0
        for i, c in enumerate(d["clusters"]):
            w = 100 * c["share_training"]
            ax.barh(y, w, left=left, color=RANK_BLUES[i], edgecolor="white", lw=1)
            if w >= 4.5:
                ax.text(left + w / 2, y, f"{int(c['center'][3:])} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(c['center'][:2]) - 1]}",
                        ha="center", va="center", fontsize=7.5, color="white" if i < 2 else INK)
            left += w
        ax.barh(y, 100 - left, left=left, color="#edf2f7", edgecolor="white", lw=1)
        ax.text(101, y, f"{100 - left:.0f}% off", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(det)))
    ax.set_yticklabels([f"{label_of(d['calendar_key'])} ({d['N_training']:,})" for d in det], fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of the calendar's realised non-COVID loan endings (%): retained standard dates (± 3 days), then off-calendar")
    ax.set_title("Learned standard loan-ending dates by calendar (empirical, not official window dates)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["clusters"], dpi=160)
    plt.close(fig)

    # 2. normal vs off-calendar
    C = R[(R.calendar_class != "geography_unclassified") & ~R.covid]
    off = C.calendar_class.str.startswith("off")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    bins = np.arange(0, 760, 30)
    for m, col, lab in ((~off, BLUE, "calendar-normal"), (off, ORANGE, "off-calendar")):
        d = C.loc[m, "loan_duration_days"].clip(upper=749)
        a1.hist(d, bins=bins, weights=np.full(len(d), 100 / len(d)), histtype="step", lw=2, color=col,
                label=f"{lab} (N = {m.sum():,}; median {C.loc[m, 'loan_duration_days'].median():.0f} days)")
    a1.set_xlabel("Loan length (days; 30-day bins, last bin ≥ 720)")
    a1.set_ylabel("Share of group (%)")
    a1.set_title("Loan length", loc="left")
    a1.legend(frameon=False, fontsize=8.5, loc="upper left")
    a1.yaxis.grid(True, color=GRID, lw=0.8)
    cats = [c for c in NEXT_CATS if c not in ("ended without a return row", "other")]
    x = np.arange(len(cats))
    for k, (m, col, lab) in enumerate(((~off, BLUE, "calendar-normal"), (off, ORANGE, "off-calendar"))):
        v = [100 * (C.loc[m, "next_category"] == c).mean() for c in cats]
        a2.barh(x + (k - 0.5) * 0.38, v, height=0.36, color=col, label=lab)
    a2.set_yticks(x)
    a2.set_yticklabels(cats, fontsize=8.5)
    a2.invert_yaxis()
    a2.set_xlabel("Share of group (%)")
    a2.set_title("What the player did next (any lag)", loc="left")
    a2.legend(frameon=False, fontsize=8.5, loc="lower right")
    a2.xaxis.grid(True, color=GRID, lw=0.8)
    fig.suptitle("Calendar-normal vs off-calendar loan endings, non-COVID, classifiable loans", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["edge"], dpi=160)
    plt.close(fig)

    # 3. off-calendar share by calendar, non-COVID vs COVID period
    det = sorted(N["detail"], key=lambda d: d["off_share_non_covid"])
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(det) + 1.6))
    for y, d in enumerate(det):
        ax.plot([100 * d["off_share_non_covid"]], [y], "o", color=BLUE, ms=6, label="outside the COVID window" if y == 0 else None)
        if d["off_share_covid"] is not None and d["n_covid"] >= 10:
            ax.plot([100 * d["off_share_covid"]], [y], "o", mfc="white", mec=ORANGE, mew=1.6, ms=6,
                    label="in the COVID window (N ≥ 10)" if y == 0 or not any(
                        x["off_share_covid"] is not None and x["n_covid"] >= 10 for x in det[:y]) else None)
    ax.axvspan(5, 10, color="#f0fff4", zorder=0)
    ax.text(7.5, len(det) - 0.3, "5–10%", ha="center", fontsize=8, color="#2f855a")
    ax.set_yticks(range(len(det)))
    ax.set_yticklabels([f"{label_of(d['calendar_key'])} ({d['N_all']:,})" for d in det], fontsize=9)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Off-calendar share of realised loan endings (%)")
    ax.set_title("Off-calendar share by calendar", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / FIGS["country"], dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

MON = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def nice(md: str) -> str:
    return f"{int(md[3:])} {MON[int(md[:2]) - 1]}"


def render(N: dict) -> str:
    W: list[str] = []
    w = W.append
    p, pn = N["shares_all"], N["shares_non_covid"]
    pop = N["population"]
    cv = N["covid"]
    e = N["edge"]
    nrm, off = e["calendar-normal"], e["off-calendar"]
    det = {d["calendar_key"]: d for d in N["detail"]}
    S = pd.DataFrame(N["sensitivity"])
    unc = N["unclassified"]
    wc, cp = N["off_share_without_capped"], N["off_share_capped"]
    capped = [label_of(k) for k in N["capped_calendars"]]
    cs = lambda k, n: f"{f0(k)} ({share(k / n) if n else '–'})"

    def dates(key):
        return ", ".join(f"{nice(c['center'])} ({share(c['share_training'])})" for c in det[key]["clusters"])

    w("# A country-specific loan-ending calendar: proposed rule")
    w("")
    w("*Generated by `python -m src.analysis.loan_calendar_rule`. A proposal: no production classification is "
      "changed. The dates are empirically common loan-ending dates in our data; the repository has no official "
      "transfer-window calendar, so none of them is described as one. Club countries come only from the snapshot's "
      "deterministic mapping; nothing is guessed.*")
    w("")
    w("## The proposal in brief")
    w("")
    w(f"1. **Rule.** Each country with at least {N_MIN} realised domestic loan endings outside the COVID window gets "
      f"its own calendar of standard ending dates, learned from those endings. Cross-border loans share one calendar. "
      f"A loan whose ending falls within ±{TOL} days of one of its calendar's dates is **calendar-normal**; otherwise "
      "it is **off-calendar**. Loans with a club of unknown country are **unclassified**. The full statement is at "
      "the end.")
    w(f"2. **Calendar-normal:** {cs(p['normal'], p['classifiable'])} of the {f0(p['classifiable'])} classifiable "
      f"realised endings.")
    w(f"3. **Off-calendar (edge cases):** {cs(p['off'], p['classifiable'])}; {share(pn['off_share'])} outside the "
      f"COVID window. Leaving out the three calendars whose dates are too dispersed to reach the target "
      f"({', '.join(capped)}), the off-calendar share outside the COVID window is "
      f"**{share(wc['share'])}** ({f0(wc['off'])} of {f0(wc['n'])}), within the 5–10% Daniel suggested. Those three "
      f"calendars are off-calendar {share(cp['share'])} of the time.")
    w(f"4. **Cannot be classified:** {cs(p['unclassified'], p['N'])} of all {f0(p['N'])} realised endings. Of these, "
      f"{f0(unc.get('club country unknown', 0))} have a club of unknown country and "
      f"{f0(unc.get('country below the history threshold', 0))} are domestic loans in countries with too few endings.")
    w("5. **Major dates:** England "
      + f"{dates('domestic: England')}; Scotland {dates('domestic: Scotland')}; Japan {dates('domestic: Japan')}; "
      f"Brazil {dates('domestic: Brazil')}; Argentina {dates('domestic: Argentina')}; cross-border "
      f"{dates('cross-border')}. Most European countries: 30 June, plus 31 December or late January.")
    w(f"6. **COVID explains part of the edge cases, not most of them.** {cs(cv['off_in_covid'], cv['off_cases'])} of "
      f"the off-calendar cases end in the COVID window. Inside it, {share(cv['covid_off'] / cv['covid_classifiable'])} "
      f"of classifiable endings are off-calendar, against {share(cv['noncovid_off'] / cv['noncovid_classifiable'])} "
      "outside it.")
    w(f"7. **Edge cases look different.** Off-calendar loans are shorter (median {off['duration_median']:.0f} days vs "
      f"{nrm['duration_median']:.0f}). They end in January far more often ({share(off['months'].get(1, 0))} vs "
      f"{share(nrm['months'].get(1, 0))}), carry the existing `early_termination` label far more often "
      f"({share(off['early_termination_label'])} vs {share(nrm['early_termination_label'])}), and are followed more "
      f"often by a loan to a different club ({share(off['next'].get('loan to a different club', 0))} vs "
      f"{share(nrm['next'].get('loan to a different club', 0))}).")
    same_countries = S.countries_with_rule.nunique() == 1
    same_capped = S.capped_calendars.nunique() == 1
    w(f"8. **Sensitivity.** Across coverage targets of 85/90/95% and tolerances of ±1/3/7 days, the off-calendar "
      f"share ranges from {share(S.off_share.min())} to {share(S.off_share.max())}. The tolerance moves it more than "
      "the target does. "
      + (f"Every combination gives the same {S.countries_with_rule.iloc[0]} countries a rule" if same_countries
         else "The number of countries with a rule varies")
      + (", and the same calendars hit the cluster cap." if same_capped else
         "; which calendars hit the cluster cap varies with the tolerance (section 10)."))
    w("9. **Relation to the 60-day follow-on idea: complement, not replace.** The calendar rule asks whether the loan "
      "ended on a normal date; the 60-day rule asks whether something happened soon after. Crossing them gives four "
      "groups (section 9). Off-calendar endings followed quickly by another move are the sharpest edge cases.")
    w("")

    w("## 1. Population")
    w("")
    pc_ = pop["by_context_covid"]
    W += md_table(["Realised loan endings", "Outside COVID window", "In COVID window", "Total"], [
        ["Domestic (both clubs in the same mapped country)", f0(pc_.get("domestic | non-COVID", 0)),
         f0(pc_.get("domestic | COVID", 0)), f0(pop["domestic"])],
        ["Cross-border (both mapped, different countries)", f0(pc_.get("cross-border | non-COVID", 0)),
         f0(pc_.get("cross-border | COVID", 0)), f0(pop["cross_border"])],
        ["One or both clubs of unknown country", f0(pc_.get("one or both clubs unmapped | non-COVID", 0)),
         f0(pc_.get("one or both clubs unmapped | COVID", 0)), f0(pop["unmapped"])],
        ["**Total**", f"**{f0(pop['non_covid'])}**", f"**{f0(pop['covid'])}**", f"**{f0(N['realised'])}**"]])
    w("")
    w(f"One row per loan (the existing loan episodes; the return row is not counted again). Only realised endings are "
      f"used: of all {f0(N['loans'])} loans, the others have a scheduled ending or none. The ending is the row that "
      f"closed the loan: the `End of loan` return in {share(N['return_share'])} of cases.")
    w("")

    w("## 2. The COVID window")
    w("")
    qo = N["quarterly_off"]
    w(f"The window is realised endings dated {N['covid_window'][0]} to {N['covid_window'][1]}. Checked against the "
      "data, using the calendar learned outside the window (endings per quarter and share off-calendar):")
    w("")
    qs = sorted(qo)
    W += md_table(["Quarter"] + qs, [["Endings"] + [f0(qo[k]["n"]) for k in qs],
                                      ["Off-calendar"] + [share(qo[k]["off"], 0) for k in qs]])
    w("")
    ex = N["covid_examples"]
    w("- Every year, July–September and January–March have a high off-calendar share, because few loans end then "
      "and those that do mostly fall away from the standard dates. The COVID signature is the **volume**: "
      f"{f0(qo['2020Q3']['n'])} endings in 2020 Q3 against {f0(qo['2019Q3']['n'])} in 2019 Q3 and "
      f"{f0(qo['2021Q3']['n'])} in 2021 Q3, almost all off-calendar. Also fewer endings than usual in 2020 Q2, and a "
      f"higher off-calendar share in 2020 Q4 ({share(qo['2020Q4']['off'])} against {share(qo['2019Q4']['off'])} and "
      f"{share(qo['2021Q4']['off'])} a year either side).")
    w(f"- **Verified examples.** Italy: {ex['Italy 2020-08-31']} of its {ex['Italy endings in 2020']} domestic endings "
      f"in 2020 fall on 31 August 2020 ({ex['Italy 08-31 in other years']} on 31 August in all other years). Brazil: "
      f"{ex['Brazil 2021-02-28']} domestic endings on 28 February 2021 ({ex['Brazil 02-28 in other years']} on 28 "
      "February in any other year).")
    w("- **Conclusion.** The window still fits: it covers the shifted endings from spring 2020 to February 2021. "
      "Endings from 2021 Q2 onward look like earlier years.")
    w("")

    w("## 3. How a calendar is learned, and the guardrails")
    w("")
    w(f"1. **Clusters.** Each month-day is placed on a 366-day circle, so 31 December and 1 January are one day apart. "
      f"The most frequent date becomes a cluster centre, and every date within ±{TOL} days joins it. Then the next "
      "most frequent remaining date, and so on. The centre is always the most frequent observed date, never a "
      "chosen one.")
    w("2. **Selection.** Clusters are taken in order of frequency until they cover the target share of the "
      "calendar's training endings, subject to the guardrails below.")
    w("3. **Classification.** An ending is calendar-normal if it lies within the tolerance of a retained centre.")
    w("")
    ge = N["guardrail_evidence"]
    big = [g for g in ge if g["adequate"]]
    hard = sorted([g for g in big if g["clusters_for_90"] > K_MAX], key=lambda g: -g["clusters_for_90"])
    easy = [g for g in big if g["clusters_for_90"] <= 3]
    cy = N["cluster_share_years"]
    w(f"- **Country minimum: {N_MIN} training endings.** Countries fall into two groups with a gap: the smallest "
      f"included country has {N['n_min_gap']['smallest_adequate']}, the largest excluded "
      f"{N['n_min_gap']['largest_inadequate']}. The excluded ones' leading dates often occur in one or two years only.")
    w(f"- **Recurrence: a cluster must occur in at least {Y_MIN} calendar years.** Of the {cy['clusters_share_ge_2pct']} "
      f"clusters holding at least 2% of an included country's endings, {cy['of_which_recurring']} recur in "
      f"{Y_MIN}+ years. So the rule rarely binds, but it stops one abnormal year from creating a standard date. COVID "
      "years are excluded from learning anyway.")
    w(f"- **Minimum share: {S_MIN:.0%} of the calendar's endings.** Below 2% the clusters are scattered single dates "
      f"({cy['clusters_1_to_2pct']} clusters hold 1–2%). Adding them inflates the calendar without capturing a "
      "recurring pattern.")
    w(f"- **Maximum: {K_MAX} clusters.** {len(easy)} of {len(big)} countries reach 90% with 3 or fewer clusters. "
      "Reaching 90% would take "
      + ", ".join(f"{label_of(g['calendar_key'])} {g['clusters_for_90']}" for g in hard)
      + " clusters. That defeats the purpose, so these calendars stop at the cap and their remaining endings stay "
      "edge cases.")
    w("")
    W += md_table(["Guardrails", "Calendar-normal", "Off-calendar", "Most clusters in a calendar",
                   "Mean clusters per country"],
                  [[g["variant"], cs(g["normal"], g["classifiable"]), cs(g["off"], g["classifiable"]),
                    g["max_clusters"], f"{g['mean_clusters']:.1f}"] for g in N["guardrail_variants"]])
    w("")
    w(f"![Standard dates by calendar]({FIGS['clusters']})")
    w("")

    w("## 4. The calendars")
    w("")
    order = [f"domestic: {c}" for c in DETAIL] + sorted(k for k in det if k.startswith("domestic") and
                                                         k[10:] not in DETAIL) + ["cross-border"]
    rows = []
    for k in order:
        if k not in det:
            continue
        d = det[k]
        rows.append([label_of(k), f0(d["N_training"]),
                     "; ".join(f"{nice(c['center'])} ±{TOL} ({share(c['share_training'])})" for c in d["clusters"]),
                     share(d["coverage_training"]), share(d["off_share_non_covid"]),
                     share(d["off_share_covid"]) + f" ({d['n_covid']})" if d["off_share_covid"] is not None else "–",
                     d["stopped"]])
    W += md_table(["Calendar", "Training endings", "Standard dates (share of training endings)", "Normal coverage",
                   "Off-calendar outside COVID", "Off-calendar in COVID window (N)", "Selection stopped because"],
                  rows, right={1, 3, 4, 5})
    w("")
    w("Reading the table, for example Scotland: "
      + f"{f0(det['domestic: Scotland']['N_training'])} training endings; standard dates "
      + "; ".join(f"{nice(c['center'])} ±{TOL}: {share(c['share_training'])}" for c in det["domestic: Scotland"]["clusters"])
      + f"; normal coverage {share(det['domestic: Scotland']['coverage_training'])}; the other "
      f"{share(1 - det['domestic: Scotland']['coverage_training'])} are off-calendar. Scottish domestic loans end on "
      "several different May dates, and which one varies from year to year. Each forms its own cluster, so the "
      "calendar reaches the cluster cap before reaching 90%. England is similar: 31 May plus several early-January "
      "dates.")
    w("")

    w("## 5. Cross-border loans")
    w("")
    XB = pd.DataFrame(N["cross_border"])
    seg = XB[XB.analysis == "segmentation"]
    W += md_table(["Cross-border calendar", "Normal share", "Outside COVID", "Learned dates"],
                  [[r.segment, share(r.normal_share), share(r.normal_share_non_covid), r.calendars] for r in seg.itertuples()],
                  right={1, 2})
    w("")
    w("One pooled calendar does as well as splitting by confederation pair or giving England–Scotland loans their own "
      f"calendar ({f0(N['xb_brit_n'])} loans), so the rule uses one pooled calendar. Normal share under that calendar "
      "by segment (N ≥ 50):")
    w("")
    cp_ = XB[XB.analysis == "confederation pair"]
    W += md_table(["Confederation pair", "N", "Normal share", "Most common dates"],
                  [[r.segment, f0(r.n), share(r.normal_share), r.calendars] for r in cp_.itertuples()], right={1, 2})
    w("")
    bc = XB[XB.analysis == "borrower country"].sort_values("normal_share")
    w("Lowest normal shares by borrower country: "
      + ", ".join(f"{dc(r.segment)} {share(r.normal_share)} (N {r.n})" for r in bc.head(3).itertuples())
      + ". Highest: " + ", ".join(f"{dc(r.segment)} {share(r.normal_share)}" for r in bc.tail(3).itertuples())
      + ". Their most common dates: "
      + "; ".join(f"{dc(r.segment)} {r.calendars}" for r in bc.head(2).itertuples())
      + ". These segments are too small for calendars of their own. All segments are in "
      "`loan_calendar_cross_border_analysis.csv`.")
    w("")

    w("## 6. Geography that cannot be classified")
    w("")
    us = N["unmapped_sides"]
    w(f"{f0(p['unclassified'])} realised endings ({share(p['unclassified'] / p['N'])}) get no calendar:")
    w(f"- {f0(unc.get('club country unknown', 0))} have a club whose country the snapshot does not record: lender "
      f"only {f0(us['lender only missing'])}, borrower only {f0(us['borrower only missing'])}, both "
      f"{f0(us['both missing'])}.")
    w(f"- {f0(unc.get('country below the history threshold', 0))} are domestic loans in countries with fewer than "
      f"{N_MIN} training endings.")
    w("")
    w("This is the rule's main limitation. The country mapping covers clubs that appear in the snapshot's "
      "competitions, so most lower-division and youth clubs have no country.")
    w("")

    w("## 7. Do the edge cases look different?")
    w("")
    rows = [["N (share of classifiable)", cs(nrm["N"], p["classifiable"]), cs(off["N"], p["classifiable"])],
            ["Loan length, median (p25–p75), days", f"{nrm['duration_median']:.0f} ({nrm['duration_p25']:.0f}–{nrm['duration_p75']:.0f})",
             f"{off['duration_median']:.0f} ({off['duration_p25']:.0f}–{off['duration_p75']:.0f})"],
            ["Ends in January", share(nrm["months"].get(1, 0)), share(off["months"].get(1, 0))],
            ["Ends in July or August", share(nrm["months"].get(7, 0) + nrm["months"].get(8, 0)),
             share(off["months"].get(7, 0) + off["months"].get(8, 0))],
            ["Fee-bearing return", share(nrm["fee_bearing"], 2), share(off["fee_bearing"], 2)],
            ["Existing `early_termination` label", share(nrm["early_termination_label"]), share(off["early_termination_label"])]]
    for x in WITHIN:
        rows.append([f"Next move within {x} day{'s' if x > 1 else ''}", share(nrm[f"next_within_{x}"]), share(off[f"next_within_{x}"])])
    for c in NEXT_CATS:
        if c in nrm["next"] or c in off["next"]:
            rows.append([f"Next: {c}", share(nrm["next"].get(c, 0)), share(off["next"].get(c, 0))])
    W += md_table(["All realised classifiable endings", "Calendar-normal", "Off-calendar"], rows)
    w("")
    w(f"![Normal vs off-calendar]({FIGS['edge']})")
    w("")
    w(f"Within 60 days the two groups move again equally often ({share(nrm['next_within_60'])} and "
      f"{share(off['next_within_60'])}). What differs is how soon, and to where: off-calendar loans are more often "
      "followed within days by another loan elsewhere. Off-calendar endings concentrate on "
      + ", ".join(f"{label_of(x['calendar_key'])} {nice(x['mmdd'])} ({x['n']})" for x in N["off_top_dates"][:6])
      + ", several of them COVID dates (section 8).")
    w("")
    rp = N["rank_profile"]
    w("**A finer reading of \"normal\" (optional refinement).** Outside COVID, endings on a calendar's main date "
      "(its largest cluster) differ from endings on its secondary dates, such as late January or 31 December in "
      "European countries:")
    w("")
    W += md_table(["", "N", "Median loan length", "Next move within 1 day", "Ends in January", "Next: loan elsewhere"],
                  [[k, f0(v["N"]), f"{v['duration_median']:.0f}", share(v["next_within_1"]), share(v["months"].get(1, 0)),
                    share(v["next"].get("loan to a different club", 0))] for k, v in rp.items()])
    w("")
    w("Secondary-date endings are calendar events (mostly mid-season), but their loans are as short as off-calendar "
      "ones. A three-level label (main date / secondary date / off-calendar) would keep that distinction; the "
      "binary rule does not need it.")
    w("")
    dp = N["distance_profile"]
    w(f"**Why ±{TOL} days.** Outside COVID, by distance from the nearest standard date:")
    w("")
    W += md_table(["Days from the nearest standard date", "Endings", "Median loan length", "Next move within 1 day",
                   "Ends in January"],
                  [[d["band"], f0(d["n"]), f"{d['duration_median']:.0f}", share(d["next_within_1"]), share(d["january"])]
                   for d in dp])
    w("")
    nmx = N["near_miss"]
    w("The sharp break is between endings exactly on a standard date and all others; endings 1 to 14 days away look "
      f"alike (short loans, mostly in January). The {f0(nmx['n'])} endings 1–{TOL} days from a standard date are "
      f"mostly ({f0(nmx['midseason'])}) around late-January dates, whose exact day varies from year to year. Near "
      "misses around the season-end dates are rare. So the tolerance mainly decides how a mid-season date that "
      "moves by a day or two is treated. "
      f"±{TOL} counts it as one date, which is what a calendar should do, at the cost of labelling some short "
      "mid-season loans normal. The three-level reading above separates those anyway. ±1 and ±7 are in section 10.")
    w("")

    w("## 8. Do the edge cases mostly come from 2020?")
    w("")
    w(f"**No.** {cs(cv['off_in_covid'], cv['off_cases'])} of the {f0(cv['off_cases'])} off-calendar endings are in the "
      f"COVID window, which holds {share(cv['covid_classifiable'] / p['classifiable'])} of classifiable endings. "
      f"COVID endings are much more often off-calendar ({cs(cv['covid_off'], cv['covid_classifiable'])}) than others "
      f"({cs(cv['noncovid_off'], cv['noncovid_classifiable'])}), but most edge cases are outside it.")
    w("")
    W += md_table(["Endings", "Classifiable", "Calendar-normal", "Off-calendar"],
                  [["All realised (calendar learned outside COVID)", f0(p["classifiable"]), share(p["normal_share"]), share(p["off_share"])],
                   ["Excluding COVID-window endings", f0(pn["classifiable"]), share(pn["normal_share"]), share(pn["off_share"])],
                   ["All realised, calendar learned from all years", f0(N["learned_all_years"]["classifiable"]),
                    share(N["learned_all_years"]["normal_share"]), share(N["learned_all_years"]["off_share"])]])
    w("")
    w("Learning from all years changes "
      + f"{len(N['learned_all_years']['changed_calendars'])} calendars ("
      + ", ".join(label_of(k) for k in N["learned_all_years"]["changed_calendars"])
      + "). It lets COVID dates such as Italy's 31 August count as standard and makes those endings look normal, "
      "which is why the rule learns outside the window and applies inside it. Largest COVID anomalies: "
      + ", ".join(f"{label_of(x['calendar_key'])} {nice(x['mmdd'])} ({x['n']})" for x in N["covid_anomalies"][:6]) + ".")
    w("")
    w(f"![Off-calendar share by calendar]({FIGS['country']})")
    w("")

    w("## 9. The old 60-day follow-on rule and the calendar rule")
    w("")
    tw, ov = N["two_way"], N["old_vs_new"]
    n2 = N["two_way_n"]
    w(f"They answer different questions. The old rule (the production follow-on: another move within 60 days of the "
      f"return) flags {cs(ov['old_followon'], n2)} of the {f0(n2)} classifiable realised returns. The calendar rule "
      f"flags {cs(ov['new_off'], n2)}. Both flag {f0(ov['both'])}. The old \"non-ordinary sequence\" class covers "
      f"{f0(ov['old_nonstandard'])} of these returns, {f0(ov['old_nonstandard_and_off'])} of them off-calendar.")
    w("")
    lab = {"A": "A. Normal date, no follow-on within 60 days", "B": "B. Normal date, follow-on within 60 days",
           "C": "C. Off-calendar, no follow-on within 60 days", "D": "D. Off-calendar, follow-on within 60 days"}
    W += md_table(["Group", "Returns", "Share", "Median loan length", "Ends in January", "Most common next moves"],
                  [[lab[k], f0(v["N"]), share(v["share"]), f"{v['duration_median']:.0f}", share(v["january"]),
                    ", ".join(f"{c} {share(s)}" for c, s in list(v["next"].items())[:2])] for k, v in sorted(tw.items())])
    w("")
    w("- **B** is what the old rule calls interesting but the calendar calls normal: a loan that ends on time and is "
      "followed by a planned move (a conversion, a sale, a new loan) at the usual date.")
    w("- **C** is what the calendar flags with no quick follow-on: a loan ending at an unusual date, then nothing for "
      "a while.")
    w("- **D** combines both, an off-date ending followed quickly by another move. It is the clearest candidate for "
      "a loan cut short.")
    w("")
    w("Examples (seasons from 2018/19; a reproducible random draw):")
    w("")
    W += md_table(["Group", "Player", "Loan", "Started", "Returned", "Nearest standard date", "Days off", "Next move (days later)"],
                  [[x["cell"], x["player"], x["loan"], x["loan_start"],
                    x["return"] + (" (COVID window)" if COVID_START <= pd.Timestamp(x["return"]) <= COVID_END else ""),
                    nice(x["nearest_standard"]), int(x["days_off"]),
                    f"{x['next']} ({x['next_gap'] if x['next_gap'] is not None else '–'})"]
                   for x in N["examples"]], right={6})
    w("")
    w("**Recommendation: complement, not replace.** Use the calendar rule to say whether the loan itself ended "
      "normally, and keep the follow-on window to describe what came next. The A–D grid is a useful way to choose "
      "Stage 2 cases: D first.")
    w("")

    w("## 10. Sensitivity")
    w("")
    W += md_table(["Target", "Tolerance", "Calendar-normal", "Off-calendar", "Countries with a rule",
                   "Mean dates per country", "Cross-border normal", "Unclassified"],
                  [[f"{r.target:.0%}" + (" (primary)" if r.primary else ""), f"±{r.tolerance_days}",
                    share(r.normal_share), share(r.off_share), r.countries_with_rule,
                    f"{r.mean_clusters_per_country:.2f}", share(r.cross_border_normal_share), f0(r.unclassified)]
                   for r in S.itertuples()])
    w("")
    w(f"The conclusions do not change: "
      + (f"every combination gives the same {S.countries_with_rule.iloc[0]} countries a rule, "
         if S.countries_with_rule.nunique() == 1 else "")
      + f"the same unclassified count, and an off-calendar share between {share(S.off_share.min())} and "
      f"{share(S.off_share.max())}. Calendars that hit the cluster cap: "
      + "; ".join(f"{caps} ({', '.join(f'±{r.tolerance_days}/{r.target:.0%}' for r in g.itertuples())})"
                  for caps, g in S.groupby("capped_calendars", sort=False))
      + f". Holding the tolerance at ±{TOL}, the target moves the off-calendar share only "
      f"from {share(S[(S.tolerance_days == TOL)].off_share.max())} to {share(S[(S.tolerance_days == TOL)].off_share.min())}. "
      f"The recommended specification is the {TARGET:.0%} target with ±{TOL} days.")
    w("")

    w("## 11. The proposed rule")
    w("")
    w("**In plain English.** Learn each calendar from realised loan endings outside the COVID window. For each "
      f"country with at least {N_MIN} such domestic endings, group ending dates into clusters of ±{TOL} days around "
      f"their most frequent date. Keep clusters in order of size, skipping any seen in fewer than {Y_MIN} years, until "
      f"they cover {TARGET:.0%} of the country's endings, the next cluster holds under {S_MIN:.0%}, or {K_MAX} are "
      "kept. Do the same once for all cross-border loans together. Then classify every realised loan, COVID ones "
      "included, against its calendar, and flag COVID-window endings separately.")
    w("")
    w("```")
    w("calendars = learn(realised endings outside 2020-03-01..2021-02-28)   # once; never re-learned on application")
    w("")
    w("for each realised loan ending:")
    w("    if lender country and borrower country are both mapped and equal:")
    w(f"        if that country has >= {N_MIN} training endings:  calendar = calendars[country]")
    w("        else:                                          class = geography_unclassified; stop")
    w("    elif both countries are mapped (and differ):        calendar = calendars['cross-border']")
    w("    else:                                              class = geography_unclassified; stop")
    w(f"    if ending date within +/- {TOL} days of a date in calendar:   class = calendar_normal_(domestic|cross_border)")
    w("    else:                                                   class = off_calendar_(domestic|cross_border)")
    w("    covid_flag = ending date in 2020-03-01..2021-02-28                 # flagged, not dropped")
    w("```")
    w("")
    W += md_table(["Parameter", "Value", "Why"], [
        ["Coverage target", f"{TARGET:.0%}", "Supported by the sensitivity grid; 85% and 95% change little"],
        ["Date tolerance", f"±{TOL} days", "Treats a mid-season date that moves by a day or two between years as one "
                                            "date; ±1 and ±7 change the off-calendar share by a few points"],
        ["Country minimum", f"{N_MIN} realised non-COVID domestic endings",
         f"Natural gap between {N['n_min_gap']['largest_inadequate']} and {N['n_min_gap']['smallest_adequate']}"],
        ["Recurrence", f"cluster seen in ≥ {Y_MIN} calendar years", "Stops a single abnormal year from making a date standard"],
        ["Minimum cluster share", f"{S_MIN:.0%}", "Below it, clusters are scattered single dates"],
        ["Maximum clusters", f"{K_MAX}", "Most countries need 1–3; the few that need more stay with edge cases"],
        ["Cross-border", "one pooled calendar", "Segmenting by confederation pair or country pair does not improve it"],
        ["Missing geography", "geography_unclassified", "No country is guessed"],
        ["COVID", "excluded from learning, classified and flagged", "Learning on COVID makes shifted dates look normal"],
    ], right=set())
    w("")
    return "\n".join(W) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _json(x):
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


CLASSIFIED_COLS = ["loan_event_id", "player_id", "player_name", "loan_season", "lender_club_id", "lender_club",
                   "borrower_club_id", "borrower_club", "lender_country", "borrower_country", "lender_confederation",
                   "borrower_confederation", "loan_start_date", "ending_event_id", "ending_date", "ending_mmdd",
                   "ending_is_return", "context", "calendar_key", "covid_period", "calendar_class",
                   "unclassified_reason", "matched_standard_mmdd", "matched_rank", "nearest_standard_mmdd",
                   "days_to_nearest_standard", "loan_duration_days", "next_category", "next_move_type",
                   "next_move_gap_days", "economic_ending", "ending_label_class", "ending_fee_on_return_eur",
                   "old_followon_60d", "two_way_cell"]


def main() -> dict:
    N, tb = compute()
    R = tb["R"]
    out = R[CLASSIFIED_COLS].copy()
    for c in ("loan_start_date", "ending_date"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(CLASSIFIED_CSV, index=False)
    out[out.calendar_class.str.startswith("off_calendar")].to_csv(EDGE_CSV, index=False)
    tb["clusters"].to_csv(RULES_CSV, index=False)
    tb["sens"].to_csv(SENS_CSV, index=False)
    tb["xb"].to_csv(XB_CSV, index=False)
    figures(N, R)
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    SUMMARY_MD.write_text(render(N))
    p = N["shares_all"]
    print(f"wrote {SUMMARY_MD.name}")
    print(f"realised {p['N']:,}  classifiable {p['classifiable']:,}  normal {p['normal']:,} ({100 * p['normal_share']:.1f}%)  "
          f"off {p['off']:,} ({100 * p['off_share']:.1f}%)  unclassified {p['unclassified']:,}")
    return N


if __name__ == "__main__":
    main()
