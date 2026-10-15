"""Calendar v2: the country-calendar rule re-learned from scratch on geography v2, compared with v1.

    python -m src.analysis.loan_calendar_rule_v2

The rule is `loan_calendar_rule.compute` unchanged: same code, same parameters (±3-day clusters, 90% target,
>=100 non-COVID domestic training endings, >=3 years, >=2% share, <=5 clusters, one pooled cross-border
calendar, COVID window excluded from learning). Only the club countries differ: `club_geography_v2.csv`.
v1 is recomputed in memory for the comparison and checked against its saved outputs, which are never written.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_calendar_rule as lc
from . import loan_country_mapping_audit as audit
from . import loan_scope
from .club_geography_v2 import CANONICAL_CSV, GEO_V2_CSV, STAGE1C_SHA256, geography_frame, sha256
from .daniel_scope_final import f0, md_table, pc
from .loan_episodes import ROOT, build_universe, load_canonical
from .stage1_scope_report import display_country as dc
from ..stage1c.policy import YOUTH_OR_RESERVE

OUT = ROOT / "data" / "outputs" / "rebuild"
SUMMARY_MD = OUT / "loan_calendar_rule_v2_summary.md"
RULES_CSV = OUT / "loan_calendar_rules_v2.csv"
CLASSIFIED_CSV = OUT / "loan_calendar_classified_v2.csv"
EDGE_CSV = OUT / "loan_calendar_edge_cases_v2.csv"
SENS_CSV = OUT / "loan_calendar_sensitivity_v2.csv"
NUMBERS_JSON = OUT / "loan_calendar_numbers_v2.json"
WRITES = (SUMMARY_MD, RULES_CSV, CLASSIFIED_CSV, EDGE_CSV, SENS_CSV, NUMBERS_JSON)
OLD_OUTPUTS = (lc.SUMMARY_MD, lc.RULES_CSV, lc.CLASSIFIED_CSV, lc.EDGE_CSV, lc.SENS_CSV, lc.XB_CSV, lc.NUMBERS_JSON,
               audit.AUDIT_MD, audit.CLUBS_CSV, audit.SAMPLE_CSV, audit.CANDIDATES_CSV, audit.COVERAGE_CSV,
               audit.PROBE_JSON, audit.NUMBERS_JSON)
PARAMS = {"TOL": 3, "TARGET": 0.90, "N_MIN": 100, "Y_MIN": 3, "S_MIN": 0.02, "K_MAX": 5, "XB_SEGMENTS": "pooled"}
STATES = ("calendar-normal", "off-calendar", "unclassified: club country unknown",
          "unclassified: below history threshold")


def state(R: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select(
        [R.calendar_class.str.startswith("calendar_normal"), R.calendar_class.str.startswith("off_calendar"),
         R.unclassified_reason.eq("club country unknown")], STATES[:3], default=STATES[3]), index=R.index)


def old_hashes() -> dict:
    return {p.name: sha256(p) for p in OLD_OUTPUTS if p.exists()}


# ---------------------------------------------------------------------------
# Comparison pieces
# ---------------------------------------------------------------------------

def overview(N: dict, R: pd.DataFrame) -> dict:
    s = state(R)
    dom = [k for k in N["calendars"] if k.startswith("domestic")]
    n = len(R)
    return {"Realised endings": n, "Geography-mapped": int(R.context.ne(audit.UNMAPPED).sum()),
            "Classifiable": int(s.isin(STATES[:2]).sum()), "Calendar-normal": int(s.eq(STATES[0]).sum()),
            "Off-calendar": int(s.eq(STATES[1]).sum()), "Geography-unclassified": int(s.eq(STATES[2]).sum()),
            "Below-history-threshold": int(s.eq(STATES[3]).sum()), "Countries with calendar": len(dom),
            "Cross-border N": int(R.context.eq("cross-border").sum()), "Domestic N": int(R.context.eq("domestic").sum())}


def transitions(R1: pd.DataFrame, R2: pd.DataFrame, cal1: dict, cal2: dict) -> tuple[dict, pd.DataFrame]:
    a = R1.set_index("loan_event_id")
    b = R2.set_index("loan_event_id").loc[a.index]
    s1, s2 = state(a.reset_index()).values, state(b.reset_index()).values
    T = pd.DataFrame({"v1": s1, "v2": s2}, index=a.index)
    ct = pd.crosstab(T.v1, T.v2)
    geo_new = a.context.eq(audit.UNMAPPED) & b.context.ne(audit.UNMAPPED)
    cls1 = T.v1.isin(STATES[:2])
    changed = cls1 & T.v1.ne(T.v2)
    same_ctx = (a.context == b.context) & (a.calendar_key == b.calendar_key)

    def centres(cal, k):
        return cal.get(k, {}).get("centers")

    relearned = pd.Series([centres(cal1, k) != centres(cal2, k) for k in a.calendar_key], index=a.index)
    out = {
        "crosstab": {f"{i} -> {j}": int(ct.loc[i, j]) for i in ct.index for j in ct.columns if ct.loc[i, j]},
        "geography_newly_available": int(geo_new.sum()),
        "geography_newly_available_by_v2_state": {k: int(v) for k, v in T.v2[geo_new].value_counts().items()},
        "unclassified_to_normal": int((~cls1 & T.v2.eq(STATES[0])).sum()),
        "unclassified_to_off": int((~cls1 & T.v2.eq(STATES[1])).sum()),
        "unclassified_still": int((~cls1 & ~T.v2.isin(STATES[:2])).sum()),
        "below_threshold_to_classified": int((T.v1.eq(STATES[3]) & T.v2.isin(STATES[:2])).sum()),
        "classified_v1": int(cls1.sum()),
        "classified_changed": int(changed.sum()),
        "classified_changed_direction": {k: int(v) for k, v in (T.v1[changed] + " -> " + T.v2[changed]).value_counts().items()},
        "classified_changed_geography_changed": int((changed & ~same_ctx).sum()),
        "classified_changed_calendar_relearned": int((changed & same_ctx & relearned).sum()),
        "classified_changed_unexplained": int((changed & same_ctx & ~relearned).sum()),
        "classified_changed_by_key": {k: int(v) for k, v in a.calendar_key[changed].value_counts().items()},
        "classified_same_state_new_matched_date": int((cls1 & T.v1.eq(T.v2) & T.v1.eq(STATES[0])
                                                       & (a.matched_standard_mmdd != b.matched_standard_mmdd)).sum()),
        "classified_context_changed": int((cls1 & ~same_ctx).sum())}
    return out, T


def calendar_changes(cal1: dict, cal2: dict) -> dict:
    d1 = {k for k in cal1 if k.startswith("domestic")}
    d2 = {k for k in cal2 if k.startswith("domestic")}
    fmt = lambda v: [lc.mmdd_of(c) for c in v["centers"]]
    return {"new": sorted(d2 - d1), "dropped": sorted(d1 - d2),
            "new_detail": {k: {"N": cal2[k]["N"], "centers": fmt(cal2[k])} for k in sorted(d2 - d1)},
            "changed": {k: {"v1": fmt(cal1[k]), "v2": fmt(cal2[k]), "N1": cal1[k]["N"], "N2": cal2[k]["N"]}
                        for k in sorted(set(cal1) & set(cal2)) if cal1[k]["centers"] != cal2[k]["centers"]},
            "unchanged": sorted(k for k in set(cal1) & set(cal2) if cal1[k]["centers"] == cal2[k]["centers"]),
            "n_growth": {k: [cal1[k]["N"], cal2[k]["N"]] for k in sorted(set(cal1) & set(cal2))}}


def findings(N: dict) -> dict:
    """The Part 8 checks, read from `lc.compute`'s own profiles."""
    e = N["edge"]
    out = {}
    for s in ("calendar-normal", "off-calendar"):
        p = e[s]
        m = p["months"]
        out[s] = {"N": p["N"], "median duration": p["duration_median"], "January": m.get(1, 0.0),
                  "July/August": m.get(7, 0.0) + m.get(8, 0.0), "early_termination label": p["early_termination_label"],
                  "next move <= 1 day": p["next_within_1"], "next move <= 21 days": p["next_within_21"],
                  "next move <= 60 days": p["next_within_60"],
                  "permanent to borrower": p["next"].get("permanent move to the borrower", 0.0),
                  "loan to different club": p["next"].get("loan to a different club", 0.0)}
    out["by_country"] = {d["calendar_key"]: {"N": d["N_all"], "off": d["off_share_all"],
                                             "off_non_covid": d["off_share_non_covid"]} for d in N["detail"]}
    c = N["covid"]
    out["covid"] = {"COVID classifiable": c["covid_classifiable"], "COVID off": c["covid_off"],
                    "non-COVID classifiable": c["noncovid_classifiable"], "non-COVID off": c["noncovid_off"]}
    out["two_way"] = {k: {"N": v["N"], "share": v["share"], "duration_median": v["duration_median"],
                          "january": v["january"]} for k, v in sorted(N["two_way"].items())}
    out["two_way_n"] = N["two_way_n"]
    return out


def selection(R1: pd.DataFrame, R2: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    L = loans.set_index("loan_event_id")
    a = R1.set_index("loan_event_id")
    b = R2.set_index("loan_event_id").loc[a.index]
    c1 = state(a.reset_index()).isin(STATES[:2]).values
    c2 = state(b.reset_index()).isin(STATES[:2]).values
    B = b.copy()
    B["age"] = B.index.map(L.age_at_transfer)
    B["mv"] = B.index.map(L.market_value_eur)
    B["fee"] = B.index.map(L.outbound_loan_fee_eur).fillna(0) > 0
    B["yr"] = (B.lender_club.fillna("").map(lambda s: bool(YOUTH_OR_RESERVE.search(s)))
               | B.borrower_club.fillna("").map(lambda s: bool(YOUTH_OR_RESERVE.search(s))))
    B["start"] = B.loan_season.str[:4].astype(int)
    B["next"] = lc.next_category(B.reset_index()).values
    groups = {"v1 classifiable": c1, "newly classifiable in v2": c2 & ~c1, "v2 classifiable": c2,
              "v2 still unclassified": ~c2, "all realised endings": np.ones(len(B), bool)}
    rows = []
    for name, m in groups.items():
        g = B[m]
        rec = {"group": name, "endings": len(g), "median age": float(g.age.median()), "mean age": float(g.age.mean()),
               "market value recorded": float(g.mv.notna().mean()), "median market value (EUR)": float(g.mv.median()),
               "median duration (days)": float(g.loan_duration_days.median()),
               "youth/reserve side involved": float(g.yr.mean()), "fee-bearing loan": float(g.fee.mean()),
               "median loan start year": float(g.start.median()), "loan starts 2020 or later": float(g.start.ge(2020).mean())}
        for k in lc.NEXT_CATS[:5] + lc.NEXT_CATS[6:7]:
            rec[f"next: {k}"] = float(g.next.eq(k).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(u1=None) -> tuple[dict, dict]:
    assert {k: getattr(lc, k) for k in PARAMS} == PARAMS, "calendar parameters differ from the frozen rule"
    u1 = u1 or loan_scope.universe()
    G = pd.read_csv(GEO_V2_CSV, keep_default_na=False)
    u2 = build_universe(load_canonical(), geography_frame(G))
    N1, T1 = lc.compute(u1)
    N2, T2 = lc.compute(u2)
    R1, R2 = T1["R"], T2["R"]
    assert set(R1.loan_event_id) == set(R2.loan_event_id) and len(R1) == len(R2)
    a, b = R1.set_index("loan_event_id"), R2.set_index("loan_event_id").loc[R1.loan_event_id]
    for side in ("lender", "borrower"):                     # a mapped v1 country is never overwritten
        m = a[f"{side}_country"].notna()
        assert (a.loc[m, f"{side}_country"] == b.loc[m, f"{side}_country"]).all()
    same = ["ending_date", "ending_mmdd", "loan_duration_days", "next_move_kind", "covid", "doy"]
    assert a[same].equals(b[same]), "geography changed something other than the countries"

    N = {"params": PARAMS, "v1": overview(N1, R1), "v2": overview(N2, R2)}
    for k in ("v1", "v2"):
        o = N[k]
        assert o["Calendar-normal"] + o["Off-calendar"] + o["Geography-unclassified"] + o["Below-history-threshold"] \
            == o["Realised endings"]
    for NN, RR in ((N1, R1), (N2, R2)):
        for key, v in NN["calendars"].items():           # learned from non-COVID endings only
            assert v["N"] == int(((RR.calendar_key == key) & ~RR.covid).sum())
    N["calendars"] = calendar_changes(N1["calendars"], N2["calendars"])
    N["calendars_v2"] = {k: {"N": v["N"], "centers": v["centers_mmdd"], "coverage": v["coverage"],
                             "stopped": v["stopped"]} for k, v in N2["calendars"].items()}
    N["transitions"], T = transitions(R1, R2, N1["calendars"], N2["calendars"])
    N["findings"] = {"v1": findings(N1), "v2": findings(N2)}
    f1, f2 = N["findings"]["v1"], N["findings"]["v2"]
    N["direction_check"] = {k: bool((f1["off-calendar"][k] > f1["calendar-normal"][k])
                                    == (f2["off-calendar"][k] > f2["calendar-normal"][k]))
                            for k in f1["calendar-normal"] if k != "N"}
    N["shares"] = {"v1": N1["shares_all"], "v2": N2["shares_all"],
                   "v1_non_covid": N1["shares_non_covid"], "v2_non_covid": N2["shares_non_covid"]}
    N["old_vs_new_rule"] = {"v1": N1["old_vs_new"], "v2": N2["old_vs_new"]}
    N["capped"] = {"v1": N1["capped_calendars"], "v2": N2["capped_calendars"]}
    N["off_share_without_capped"] = {"v1": N1["off_share_without_capped"], "v2": N2["off_share_without_capped"]}
    SEL = selection(R1, R2, u1.loans)
    N["selection"] = SEL.set_index("group").to_dict(orient="index")
    gs = G.set_index("club_id")
    R2 = R2.copy()
    for side in ("lender", "borrower"):
        ids = R2[f"{side}_club_id"].astype("int64")
        R2[f"{side}_geography_source"] = ids.map(gs.source).fillna("")
        R2[f"{side}_geography_status"] = ids.map(gs.status).fillna("")
    R2["v1_calendar_class"] = R2.loan_event_id.map(a.calendar_class)
    R2["v1_state"] = R2.loan_event_id.map(T.v1)
    R2["v2_state"] = R2.loan_event_id.map(T.v2)
    R2["status_changed"] = R2.v1_state.ne(R2.v2_state)
    N["n_v2_detail"] = N2["detail"]
    for k, RR in (("v1", R1), ("v2", R2)):
        bt = RR.loc[state(RR).eq(STATES[3]).values, "lender_country"]
        N[f"below_threshold_{k}"] = {"countries": int(bt.nunique()),
                                     "top": {c: int(v) for c, v in bt.value_counts().head(10).items()},
                                     "unnamed_id_endings": int(bt.str.startswith("Transfermarkt country").sum())}
    N["v1_reproduces_saved"] = v1_matches_saved(R1)
    return N, {"R1": R1, "R2": R2, "T1": T1, "T2": T2, "SEL": SEL, "N1": N1, "N2": N2}


def v1_matches_saved(R1: pd.DataFrame) -> dict:
    if not lc.CLASSIFIED_CSV.exists():
        return {"checked": False}
    old = pd.read_csv(lc.CLASSIFIED_CSV, usecols=["loan_event_id", "calendar_class"])
    m = old.set_index("loan_event_id").calendar_class.reindex(R1.loan_event_id)
    return {"checked": True, "rows": len(old), "identical": bool((m.values == R1.calendar_class.values).all())}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def pct(x, d=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def label(k: str) -> str:
    return dc(k.split(": ", 1)[1]) if k.startswith("domestic: ") else k


def render(N: dict) -> str:
    o1, o2 = N["v1"], N["v2"]
    n = o1["Realised endings"]
    tr, cc = N["transitions"], N["calendars"]
    f1, f2 = N["findings"]["v1"], N["findings"]["v2"]
    S = N["selection"]
    W = ["# Loan calendar rule v2: re-learned on geography v2", "",
         "*Generated by `python -m src.analysis.loan_calendar_rule_v2`. Same code and parameters as the v1 rule "
         f"(±{PARAMS['TOL']}-day clusters, {PARAMS['TARGET']:.0%} target, ≥{PARAMS['N_MIN']} non-COVID domestic "
         f"training endings, ≥{PARAMS['Y_MIN']} years, ≥{PARAMS['S_MIN']:.0%} share, ≤{PARAMS['K_MAX']} clusters, "
         "one pooled cross-border calendar, COVID window excluded from learning). Only the club countries changed "
         "(`club_geography_v2.csv`). v1 outputs are not overwritten. The dates are empirically common loan-ending "
         "dates, not official transfer windows.*", ""]

    W += ["## 1. v1 vs v2", ""]
    rows = []
    for k in o1:
        if k in ("Countries with calendar",):
            rows.append([k, f0(o1[k]), "", f0(o2[k]), ""])
        else:
            rows.append([k, f0(o1[k]), pc(o1[k], n, 2), f0(o2[k]), pc(o2[k], n, 2)])
    W += md_table(["", "v1 N", "v1 %", "v2 N", "v2 %"], rows)
    cl1, cl2 = o1["Classifiable"], o2["Classifiable"]
    nu = tr["unclassified_to_normal"] + tr["unclassified_to_off"]
    off_v2_of_v1 = sum(v for k, v in tr["crosstab"].items()
                       if k.split(" -> ")[0] in STATES[:2] and k.split(" -> ")[1] == STATES[1])
    bt = N["below_threshold_v2"]
    W += ["", f"Of classifiable endings, the off-calendar share is {pc(o1['Off-calendar'], cl1)} in v1 and "
          f"{pc(o2['Off-calendar'], cl2)} in v2 (outside the COVID window: "
          f"{pct(N['shares']['v1_non_covid']['off_share'])} and {pct(N['shares']['v2_non_covid']['off_share'])}). "
          f"The rise comes from the newly classified endings, which are off-calendar {pc(tr['unclassified_to_off'], nu)} "
          f"of the time. The {f0(tr['classified_v1'])} endings classified in v1 are off-calendar "
          f"{pc(o1['Off-calendar'], cl1)} under v1 and {pc(off_v2_of_v1, tr['classified_v1'])} under v2. "
          f"Excluding the calendars that hit the {PARAMS['K_MAX']}-cluster cap (v1: "
          + ", ".join(label(k) for k in N["capped"]["v1"]) + "; v2: " + ", ".join(label(k) for k in N["capped"]["v2"])
          + f"), the off-calendar share is {pct(N['off_share_without_capped']['v1']['share'])} in v1 and "
          f"{pct(N['off_share_without_capped']['v2']['share'])} in v2 (non-COVID endings).",
          "", f"The {f0(o2['Below-history-threshold'])} below-threshold endings are domestic loans in "
          f"{bt['countries']} countries that still have fewer than {PARAMS['N_MIN']} non-COVID training endings, led by "
          + ", ".join(f"{label('domestic: ' + k)} ({v})" for k, v in bt["top"].items())
          + f". {f0(bt['unnamed_id_endings'])} of them are in countries the snapshot does not name.", ""]

    W += ["## 2. Calendars", "",
          "**New countries with their own calendar** (now ≥100 non-COVID domestic training endings): "
          + (", ".join(f"{label(k)} (N = {v['N']}; " + ", ".join(v["centers"]) + ")" for k, v in cc["new_detail"].items())
             or "none") + ".", "",
          f"**Dropped:** {', '.join(label(k) for k in cc['dropped']) or 'none'}. "
          f"**Unchanged centres:** {len(cc['unchanged'])} calendars. **Changed centres:** {len(cc['changed'])}.", ""]
    if cc["changed"]:
        W += md_table(["Calendar", "v1 N", "v2 N", "v1 dates", "v2 dates"],
                      [[label(k), f0(v["N1"]), f0(v["N2"]), ", ".join(v["v1"]), ", ".join(v["v2"])]
                       for k, v in cc["changed"].items()], right={1, 2})
        W += [""]
    W += ["Full v2 calendars:", ""]
    W += md_table(["Calendar", "Training N", "Dates", "Coverage", "Stopped because"],
                  [[label(k), f0(v["N"]), ", ".join(v["centers"]), pct(v["coverage"]), v["stopped"]]
                   for k, v in sorted(N["calendars_v2"].items(), key=lambda kv: -kv[1]["N"])], right={1, 3})
    W += [""]

    W += ["## 3. What moved between v1 and v2", ""]
    W += md_table(["v1 state → v2 state", "Endings"], [[k, f0(v)] for k, v in tr["crosstab"].items()])
    W += ["", f"- **Geography newly available:** {f0(tr['geography_newly_available'])} endings had a club without a "
          f"country in v1 and have both in v2. In v2 they are: "
          + "; ".join(f"{k} {f0(v)}" for k, v in tr["geography_newly_available_by_v2_state"].items()) + ".",
          f"- **Previously unclassified, now calendar-normal:** {f0(tr['unclassified_to_normal'])}; **now "
          f"off-calendar:** {f0(tr['unclassified_to_off'])}; still unclassified: {f0(tr['unclassified_still'])}. "
          f"Of the v1 below-threshold endings, {f0(tr['below_threshold_to_classified'])} are now classified.",
          f"- **Previously classified, status changed:** {f0(tr['classified_changed'])} of {f0(tr['classified_v1'])} "
          f"({pc(tr['classified_changed'], tr['classified_v1'], 2)}). Their club countries are unchanged "
          f"({tr['classified_context_changed']} classified endings changed context). "
          f"{f0(tr['classified_changed_calendar_relearned'])} changed because their calendar was re-learned with "
          f"more training endings, {tr['classified_changed_unexplained']} for any other reason. Directions: "
          + ("; ".join(f"{k} {v}" for k, v in tr["classified_changed_direction"].items()) or "none") + ". By calendar: "
          + ("; ".join(f"{label(k)} {v}" for k, v in tr["classified_changed_by_key"].items()) or "none") + ".",
          f"- {f0(tr['classified_same_state_new_matched_date'])} calendar-normal endings stay normal but match a "
          "different standard date.", ""]

    W += ["## 4. Do the calendar findings survive?", "", "**Calendar-normal vs off-calendar (all realised, classifiable)**", ""]
    keys = list(f1["calendar-normal"])
    rows = []
    for k in keys:
        fmt = (lambda x: f0(x)) if k in ("N",) else (lambda x: f"{x:.0f}") if k == "median duration" else pct
        rows.append([k, fmt(f1["calendar-normal"][k]), fmt(f1["off-calendar"][k]),
                     fmt(f2["calendar-normal"][k]), fmt(f2["off-calendar"][k])])
    W += md_table(["", "v1 normal", "v1 off", "v2 normal", "v2 off"], rows)
    W += ["", "Direction of each off-minus-normal difference, v1 → v2: "
          + "; ".join(f"{k}: {'same' if ok else 'REVERSED'}" for k, ok in N["direction_check"].items()) + ".", ""]
    W += ["**Off-calendar rate by calendar (v2, all realised endings with that calendar)**", ""]
    bc2, bc1 = f2["by_country"], f1["by_country"]
    W += md_table(["Calendar", "v2 N", "v2 off", "v2 off (non-COVID)", "v1 N", "v1 off"],
                  [[label(k), f0(v["N"]), pct(v["off"]), pct(v["off_non_covid"]),
                    f0(bc1[k]["N"]) if k in bc1 else "—", pct(bc1[k]["off"]) if k in bc1 else "—"]
                   for k, v in sorted(bc2.items(), key=lambda kv: -kv[1]["N"])])
    c1, c2 = f1["covid"], f2["covid"]
    W += ["", "**COVID vs non-COVID**", ""]
    W += md_table(["", "v1", "v2"],
                  [["COVID-window classifiable", f0(c1["COVID classifiable"]), f0(c2["COVID classifiable"])],
                   ["COVID-window off-calendar", f"{f0(c1['COVID off'])} ({pc(c1['COVID off'], c1['COVID classifiable'])})",
                    f"{f0(c2['COVID off'])} ({pc(c2['COVID off'], c2['COVID classifiable'])})"],
                   ["non-COVID classifiable", f0(c1["non-COVID classifiable"]), f0(c2["non-COVID classifiable"])],
                   ["non-COVID off-calendar", f"{f0(c1['non-COVID off'])} ({pc(c1['non-COVID off'], c1['non-COVID classifiable'])})",
                    f"{f0(c2['non-COVID off'])} ({pc(c2['non-COVID off'], c2['non-COVID classifiable'])})"]])
    W += ["", "**Calendar rule × 60-day follow-on rule** (classifiable endings that are returns)", ""]
    lab = {"A": "A normal / no quick move", "B": "B normal / quick move", "C": "C off-calendar / no quick move",
           "D": "D off-calendar / quick move"}
    W += md_table(["Cell", "v1 N", "v1 share", "v1 median days", "v1 January", "v2 N", "v2 share", "v2 median days",
                   "v2 January"],
                  [[lab[k], f0(f1["two_way"][k]["N"]), pct(f1["two_way"][k]["share"]),
                    f"{f1['two_way'][k]['duration_median']:.0f}", pct(f1["two_way"][k]["january"]),
                    f0(f2["two_way"][k]["N"]), pct(f2["two_way"][k]["share"]),
                    f"{f2['two_way'][k]['duration_median']:.0f}", pct(f2["two_way"][k]["january"])]
                   for k in "ABCD"])
    W += [""]

    W += ["## 5. Selection: did the mapping fix reduce it?", ""]
    cols = list(S)
    vars_ = [k for k in S[cols[0]] if k != "endings"]
    rows = [["endings"] + [f0(S[c]["endings"]) for c in cols]]
    for v in vars_:
        fmt = ((lambda x: f"€{x / 1e3:,.0f}k") if "EUR" in v else (lambda x: f"{x:.0f}") if "year" in v
               else (lambda x: f"{x:.1f}") if "age" in v else (lambda x: f"{x:.0f}") if "days" in v else pct)
        rows.append([v] + [fmt(S[c][v]) for c in cols])
    W += md_table([""] + cols, rows)
    pop = S["all realised endings"]
    gaps = []
    for v, scale in (("median age", 1), ("median market value (EUR)", 1e3), ("fee-bearing loan", 0.01),
                     ("youth/reserve side involved", 0.01), ("median duration (days)", 1)):
        g1, g2 = S["v1 classifiable"][v] - pop[v], S["v2 classifiable"][v] - pop[v]
        gaps.append((v, g1 / scale, g2 / scale))
    unit = {"median age": "years", "median market value (EUR)": "€k", "fee-bearing loan": "pp",
            "youth/reserve side involved": "pp", "median duration (days)": "days"}
    W += ["", "Gap between the classified sample and all realised endings, v1 → v2: "
          + "; ".join(f"{v} {g1:+.1f} → {g2:+.1f} {unit[v]}" for v, g1, g2 in gaps) + "."]
    shrunk = all(abs(g2) <= abs(g1) / 2 for _, g1, g2 in gaps if abs(g1) > 1e-9)
    W += ["", ("**Yes.** Every gap is at least halved, because the v2 classified sample is "
               f"{pc(o2['Classifiable'], n)} of all realised endings instead of {pc(o1['Classifiable'], n)}.")
          if shrunk else "**Partly.** Not every gap is halved; see the table.", ""]
    v1s = N["v1_reproduces_saved"]
    W += ["## 6. QA", "",
          f"- v1 recomputed in memory reproduces the saved v1 classifications: "
          f"{'yes' if v1s.get('identical') else 'NO'} ({f0(v1s.get('rows', 0))} rows).",
          "- Mapped v1 countries are identical in v2 for every loan; dates, durations and next moves are identical.",
          "- Calendars learned from non-COVID endings only (training N re-counted for every calendar).",
          "- normal + off-calendar + unclassified = realised endings, in v1 and v2.",
          f"- Old outputs hashed before and after this run: {len(N.get('old_output_sha256', {}))} files, unchanged.", ""]
    return "\n".join(W)


def main(u1=None) -> dict:
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    before = old_hashes()
    N, tb = compute(u1)
    R2, T2 = tb["R2"], tb["T2"]
    extra = ["lender_geography_source", "lender_geography_status", "borrower_geography_source",
             "borrower_geography_status", "v1_calendar_class", "v1_state", "v2_state", "status_changed"]
    out = R2[lc.CLASSIFIED_COLS + extra].copy()
    for c in ("loan_start_date", "ending_date"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(CLASSIFIED_CSV, index=False)
    out[out.calendar_class.str.startswith("off_calendar")].to_csv(EDGE_CSV, index=False)
    T2["clusters"].to_csv(RULES_CSV, index=False)
    T2["sens"].to_csv(SENS_CSV, index=False)
    N["old_output_sha256"] = before
    SUMMARY_MD.write_text(render(N))
    NUMBERS_JSON.write_text(json.dumps(lc._json(N), indent=1, default=str))
    assert old_hashes() == before, "an old output changed"
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print_summary(N)
    return N


def print_summary(N: dict) -> None:
    G = json.loads((OUT / "club_geography_v2_numbers.json").read_text())
    ua, cl, e1, e2 = G["unmapped_after"], G["clubs"], G["endings_v1"], G["endings_v2"]
    o1, o2, tr = N["v1"], N["v2"], N["transitions"]
    n = o1["Realised endings"]
    S = N["selection"]
    print("GEOGRAPHY RECOVERY")
    print(f"- unmapped club IDs before: {G['unmapped_before']['clubs']:,}")
    print(f"- recovered by raw countryFlag: {ua['recovered_raw']:,} ({ua['raw_flag_named']:,} named country, "
          f"{ua['raw_flag_id_only']} country id not named in the snapshot)")
    print(f"- recovered by deterministic fallback: {ua['recovered_fallback']} ("
          + ", ".join(f"{k.split(':')[0]} {v}" for k, v in ua["fallback"].items()) + ")")
    print(f"- real clubs still unresolved: {ua['unresolved']} of the {G['unmapped_before']['clubs']:,} "
          f"({ua['unresolved_endings']} endings); {cl['real_missing']} across all Stage 1C clubs")
    print(f"- special/non-club entries: {cl['special']} in Stage 1C ({ua['special']} among the loan clubs)")
    print(f"- club mapping coverage: {cl['with_country']:,} / {cl['real']:,} real Stage 1C clubs = "
          f"{pc(cl['with_country'], cl['real'], 2)}")
    print("LOAN ENDINGS")
    print(f"- realised endings: {n:,}")
    print(f"- both clubs mapped before: {e1['both_mapped']:,} ({pc(e1['both_mapped'], n)})")
    print(f"- both clubs mapped after: {e2['both_mapped']:,} ({pc(e2['both_mapped'], n, 2)})")
    print(f"- geography coverage after: {e2['both_mapped']:,} / {n:,} = {pc(e2['both_mapped'], n, 2)}")
    print("CALENDAR V2")
    print(f"- classifiable before: {o1['Classifiable']:,} ({pc(o1['Classifiable'], n)})")
    print(f"- classifiable after: {o2['Classifiable']:,} ({pc(o2['Classifiable'], n)})")
    print(f"- normal: {o2['Calendar-normal']:,} ({pc(o2['Calendar-normal'], o2['Classifiable'])} of classifiable)")
    print(f"- off-calendar: {o2['Off-calendar']:,} ({pc(o2['Off-calendar'], o2['Classifiable'])} of classifiable)")
    print(f"- unclassified: {o2['Geography-unclassified'] + o2['Below-history-threshold']:,} "
          f"({o2['Geography-unclassified']} country unknown, {o2['Below-history-threshold']:,} below the history threshold)")
    print(f"- countries with their own calendar: {o2['Countries with calendar']} (v1: {o1['Countries with calendar']})")
    print("NEW COUNTRIES")
    print("- " + (", ".join(f"{label(k)} (N={v['N']})" for k, v in N["calendars"]["new_detail"].items()) or "none"))
    print("STABILITY")
    print(f"- previously classified rows whose status changes: {tr['classified_changed']:,} of {tr['classified_v1']:,} "
          f"({pc(tr['classified_changed'], tr['classified_v1'], 2)})")
    print(f"- reason: {tr['classified_changed_calendar_relearned']} re-learned calendar dates ("
          + ", ".join(f"{label(k)} {v}" for k, v in tr["classified_changed_by_key"].items()) + f"); "
          f"{tr['classified_changed_geography_changed']} geography; {tr['classified_changed_unexplained']} other")
    print("SELECTION")
    v1c, v2c, pop = S["v1 classifiable"], S["v2 classifiable"], S["all realised endings"]
    print(f"- did coverage bias materially improve?: classified share {pc(o1['Classifiable'], n)} -> "
          f"{pc(o2['Classifiable'], n)}; median age {v1c['median age']:.0f}/{v2c['median age']:.0f} vs population "
          f"{pop['median age']:.0f}; fee-bearing {pct(v1c['fee-bearing loan'])}/{pct(v2c['fee-bearing loan'])} vs "
          f"{pct(pop['fee-bearing loan'])}; youth/reserve {pct(v1c['youth/reserve side involved'])}/"
          f"{pct(v2c['youth/reserve side involved'])} vs {pct(pop['youth/reserve side involved'])}")


if __name__ == "__main__":
    main()
