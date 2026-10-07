"""Audit of the playing-time regression: selection vs controls, prior playing time, England, zeros, valuations.

    python -m src.analysis.playing_time_regression_audit

Read-only robustness checks of `playing_time_regression`. Re-estimates its
existing specifications; the one new variable, prior-season playing time at
the origin club, uses only league games dated before the move. Observational,
not causal.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from . import playing_time_join as ptj
from . import playing_time_regression as ptr
from .daniel_scope_final import f0, md_table
from .loan_episodes import ROOT
from .loan_timing_exploration import share

OUT = ROOT / "data" / "outputs" / "rebuild"
AUDIT_MD = OUT / "playing_time_regression_audit.md"
PRIOR_CSV = OUT / "playing_time_prior_playing_time.csv"
MODELS_CSV = OUT / "playing_time_regression_audit_models.csv"
SUBGROUPS_CSV = OUT / "playing_time_regression_audit_subgroups.csv"
NUMBERS_JSON = OUT / "playing_time_regression_audit_numbers.json"

M1, M2, M3, M4 = ptr.M1, ptr.M2, ptr.M3, ptr.M4
FE = "league_season"
CATS = ["position_clean"]
SHARE_BINS = (("exactly 0", 0, 0), ("0–10%", 1e-12, .10), ("10–25%", .10 + 1e-12, .25), ("25–50%", .25 + 1e-12, .50),
              ("50–75%", .50 + 1e-12, .75), ("over 75%", .75 + 1e-12, 1.0))
GAP_BINS = (("0–30 days", 0, 30), ("31–90", 31, 90), ("91–180", 91, 180), ("181–365", 181, 365), ("over 365", 366, 10 ** 6))


# ---------------------------------------------------------------------------
# Prior playing time (pre-move only)
# ---------------------------------------------------------------------------

def prior_playing_time(C: pd.DataFrame, rows: pd.DataFrame, sides: pd.DataFrame, app: pd.DataFrame,
                       covered: list[str]) -> pd.DataFrame:
    """Origin club A's league season before the arrival season (s - 1). The window runs from the player's most
    recent recorded move into A (any movement type) to the day before the move; with no recorded move into A it
    is the whole season (flagged). Measurable only if A played in a covered league in s - 1, the window holds at
    least one A league game, and every such game has a complete appearance record."""
    lg = sides[(sides.competition_type == "domestic_league") & sides.competition_id.isin(covered)]
    by_cs = {k: g.sort_values("date") for k, g in lg.groupby(["club_id", "season"])}
    la = app[app.game_id.isin(set(lg.game_id))]
    by_pc = {k: g for k, g in la.groupby(["player_id", "club_id"])}
    entries = rows[["player_id", "to_club_id", "_date", "transfer_type_normalized"]].dropna(
        subset=["player_id", "to_club_id", "_date"]).sort_values("_date")
    by_pt = {k: (g._date.values, g.transfer_type_normalized.values) for k, g in entries.groupby(["player_id", "to_club_id"])}
    out = []
    for r in C.itertuples():
        a, t, p, s = int(r.a_club_id), r.move_date, int(r.player_id), int(r.arrival_season) - 1
        rec = {"event_id": r.event_id, "prior_season": s, "origin_covered_prior_season": (a, s) in by_cs,
               "origin_entry_date": pd.NaT, "origin_entry_type": None, "origin_entry_unobserved": False,
               "prior_games_available": np.nan,
               "prior_games_complete": np.nan, "prior_minutes": np.nan, "prior_window_max_game_date": pd.NaT}
        if rec["origin_covered_prior_season"]:
            ds = by_pt.get((p, a))
            keep = ds[0] < np.datetime64(t) if ds is not None else np.array([], dtype=bool)
            if keep.any():
                rec["origin_entry_date"] = pd.Timestamp(ds[0][keep][-1])
                rec["origin_entry_type"] = ds[1][keep][-1]
            else:
                rec["origin_entry_unobserved"] = True
            g = by_cs[(a, s)]
            w = g[(g.date < t) & ((g.date >= rec["origin_entry_date"]) if pd.notna(rec["origin_entry_date"]) else True)]
            ap = by_pc.get((p, a))
            mins = 0 if ap is None else int(ap[ap.game_id.isin(set(w.game_id))].minutes_played.sum())
            rec.update({"prior_games_available": len(w), "prior_games_complete": int(w.complete.sum()),
                        "prior_minutes": mins, "prior_window_max_game_date": w.date.max() if len(w) else pd.NaT})
        out.append(rec)
    P = pd.DataFrame(out)
    P["prior_measurable"] = (P.origin_covered_prior_season & (P.prior_games_available > 0)
                             & (P.prior_games_complete == P.prior_games_available))
    P["prior_minutes"] = P.prior_minutes.where(P.prior_measurable)               # never a number otherwise
    P["prior_share"] = P.prior_minutes / (90 * P.prior_games_available)
    P["prior_share"] = P.prior_share.where(P.prior_measurable)
    P["prior_zero"] = (P.prior_minutes == 0).where(P.prior_measurable)
    P["prior_reason"] = np.select(
        [P.prior_measurable, ~P.origin_covered_prior_season, P.prior_games_available.fillna(0) == 0],
        ["measurable", "origin club not in a covered league the season before",
         "no origin-club league game between his arrival there and the move"],
        default="incomplete appearance record in the window")
    return P


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def est(df, x, label, sample, **kw) -> dict:
    m = ptr.fit(df, x, **kw)
    c = ptr.coef(m)
    return {"model": label, "sample": sample, "loans": int(df.loan.sum()), "permanents": int((1 - df.loan).sum()),
            "N_used": m["N"], **{k: c[k] for k in ("coef", "se", "lo", "hi", "p")}}


def means(df) -> dict:
    return {"loans": int(df.loan.sum()), "permanents": int((1 - df.loan).sum()),
            "loan_mean": float(df.loc[df.loan == 1, "share"].mean()),
            "perm_mean": float(df.loc[df.loan == 0, "share"].mean())}


def quintile_effects(D: pd.DataFrame, cuts: np.ndarray, extra: list[str]) -> list[dict]:
    D = D.copy()
    D["q"] = pd.cut(D.log_relative_value, cuts, labels=[f"Q{i}" for i in range(1, 6)], include_lowest=True)
    qd = pd.get_dummies(D.q, dtype=float)
    for q in qd.columns:
        D[f"loan_{q}"] = D.loan * qd[q]
    m = ptr.fit(D, [f"loan_{q}" for q in qd.columns] + ["remain", "age_c", "age_c2", "log_squad_value"] + extra,
                fe=FE, cats=CATS + ["q"])
    return [{"q": q, "loans": int(((D.q == q) & (D.loan == 1)).sum()), "permanents": int(((D.q == q) & (D.loan == 0)).sum()),
             **{k: v for k, v in ptr.coef(m, f"loan_{q}").items() if k in ("coef", "se", "lo", "hi")}} for q in qd.columns]


def league_effect(D: pd.DataFrame, league: str = "GB1", extra: list[str] | None = None, absorb: str | None = None,
                  x: list[str] | None = None) -> dict:
    """Adjusted loan − permanent difference for one league from the common loan × league model."""
    D = D.copy()
    leagues = sorted(D.receiving_league.unique())
    for lg_ in leagues:
        D[f"loan_{lg_}"] = D.loan * (D.receiving_league == lg_)
    base = (x or M3)[1:] + (extra or [])
    m = ptr.fit(D, [f"loan_{x_}" for x_ in leagues] + base, fe=None if absorb else FE, absorb=absorb, cats=CATS)
    c = ptr.coef(m, f"loan_{league}")
    g = D[D.receiving_league == league]
    return {"loans": int(g.loan.sum()), "permanents": int((1 - g.loan).sum()), **{k: c[k] for k in ("coef", "se", "lo", "hi")}}


def bins(df) -> dict:
    out = {}
    for k, g in df.groupby("kind"):
        out[k] = {lab: float(g.share.between(lo, hi).mean()) for lab, lo, hi in SHARE_BINS}
    return out


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute() -> tuple[dict, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    R0, tb = ptr.compute()                       # rebuilds the sample and every original estimate
    C = tb["C"].copy()
    S = tb["sample"]
    P = S[S.primary_sample].copy()
    for F in (P,):
        F["age_c"] = F.age - R0["age_center"]
        F["age_c2"] = F.age_c ** 2
    N: dict = {"original": {k: R0["models"][k] for k in ("M0 full", "M0", "M1", "M2", "M3", "M4")},
               "original_quintiles": R0["quintiles"],
               "original_england": next(r for r in R0["league"] if r["league"] == "GB1")}
    assert len(P) == 15158 or len(P) == sum(R0["primary"].values())
    assert C.event_id.is_unique and P.event_id.is_unique

    # ---- Part 1: selection vs controls
    steps = [("Full primary sample, raw", P, None), ("Market-value sample, raw", C, None),
             ("+ league × season FE, arrival timing", C, M1), ("+ age, age², position", C, M2),
             ("+ log relative value, log squad value (preferred)", C, M3)]
    dec = []
    for lab, D, x in steps:
        m = ptr.fit(D, ["loan"]) if x is None else ptr.fit(D, x, fe=FE, cats=CATS if x is not M1 else None)
        c = ptr.coef(m)
        dec.append({"step": lab, **means(D), "difference": c["coef"], "lo": c["lo"], "hi": c["hi"], "N": m["N"]})
    assert abs(dec[0]["difference"] - R0["models"]["M0 full"]["coef"]) < 1e-12
    N["decomposition"] = dec
    N["selection_effect"] = dec[1]["difference"] - dec[0]["difference"]
    N["control_effect"] = dec[4]["difference"] - dec[1]["difference"]
    drop = P[~P.event_id.isin(C.event_id)]
    N["dropped"] = {k: {"N": len(g), "mean": float(g.share.mean()), "zero": float((g.share == 0).mean())}
                    for k, g in drop.groupby("kind")}

    # ---- prior playing time
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        mt = ptj.match_tables(con)
        _, covered = ptj.league_coverage(mt["sides"])
    finally:
        con.close()
    u = loan_scope.universe()
    PR = prior_playing_time(C, u.rows, mt["sides"], mt["app"], covered)
    assert PR.event_id.is_unique and len(PR) == len(C)
    C = C.merge(PR, on="event_id", how="left", validate="one_to_one")
    win_ok = C.prior_window_max_game_date.isna() | (C.prior_window_max_game_date < C.move_date)
    assert win_ok.all(), "a prior-window game is not before the move"
    assert C.loc[~C.prior_measurable, "prior_share"].isna().all()
    N["prior_coverage"] = {}
    for k, g in C.groupby("kind"):
        N["prior_coverage"][k] = {"N": len(g), "origin_covered": int(g.origin_covered_prior_season.sum()),
                                  "measurable": int(g.prior_measurable.sum()),
                                  "entry_unobserved_measurable": int((g.prior_measurable & g.origin_entry_unobserved).sum()),
                                  "reasons": {r: int(v) for r, v in g.prior_reason.value_counts().items()},
                                  "prior_share_mean": float(g.prior_share.mean()),
                                  "prior_zero": float(g.prior_zero.mean()),
                                  "prior_games_median": float(g.loc[g.prior_measurable, "prior_games_available"].median())}
    ng = C[C.prior_reason == "no origin-club league game between his arrival there and the move"]
    N["no_game_entry"] = {k: {"N": len(g), "returned_from_loan": int((g.origin_entry_type == "loan_return").sum())}
                          for k, g in ng.groupby("kind")}
    Cp = C[C.prior_measurable].copy()
    N["prior_sample"] = means(Cp)
    a = ptr.fit(Cp, M3, fe=FE, cats=CATS)
    b = ptr.fit(Cp, M3 + ["prior_share"], fe=FE, cats=CATS)
    assert a["N"] == b["N"] == len(Cp), "prior comparison does not use identical observations"
    assert a["res"].model.endog.shape == b["res"].model.endog.shape
    N["prior_models"] = {"A": ptr.coef(a), "B": ptr.coef(b),
                         "prior_share_coef": {"coef": float(b["res"].params["prior_share"]),
                                              "se": float(b["res"].bse["prior_share"])},
                         "A_full_sample_same_spec": R0["models"]["M3"]}
    Cp["prior_zero_f"] = Cp.prior_zero.astype(float)
    b2 = ptr.fit(Cp, M3 + ["prior_share", "prior_zero_f"], fe=FE, cats=CATS)
    N["prior_models"]["B2"] = ptr.coef(b2)
    N["prior_balance"] = {k: {"prior_share_mean": float(g.prior_share.mean()), "prior_zero": float(g.prior_zero.mean())}
                          for k, g in Cp.groupby("kind")}

    # ---- Part 3: relative-value fifths
    cuts = np.quantile(C.log_relative_value, [0, .2, .4, .6, .8, 1])
    N["quintiles"] = {"full": quintile_effects(C, cuts, []),
                      "prior sample, without prior PT": quintile_effects(Cp, cuts, []),
                      "prior sample, with prior PT": quintile_effects(Cp, cuts, ["prior_share"])}
    assert abs(N["quintiles"]["full"][0]["coef"] - R0["quintiles"][0]["coef"]) < 1e-9, "quintiles do not reproduce"

    # ---- Part 4: England
    E = {}
    g = C[C.receiving_league == "GB1"]
    gp = P[P.receiving_league == "GB1"]
    E["raw_full"] = {**means(gp), "diff": float(gp.loc[gp.loan == 1, "share"].mean() - gp.loc[gp.loan == 0, "share"].mean())}
    E["raw_common"] = {**means(g), "diff": float(g.loc[g.loan == 1, "share"].mean() - g.loc[g.loan == 0, "share"].mean())}
    E["preferred"] = league_effect(C)
    assert abs(E["preferred"]["coef"] - N["original_england"]["coef"]) < 1e-9
    E["club_season_fe"] = league_effect(C, absorb="club_season", x=M4)
    E["prior_sample_without"] = league_effect(Cp)
    E["prior_sample_with"] = league_effect(Cp, extra=["prior_share"])
    gs = []
    for s in sorted(g.arrival_season.unique()):
        r = league_effect(C[~((C.receiving_league == "GB1") & (C.arrival_season == s))])
        gs.append({"dropped": f"season {int(s)}/{str(int(s) + 1)[2:]}", "n_dropped": int((g.arrival_season == s).sum()), **r})
    clubs = g.groupby(["b_club_id", "b_club"]).size().sort_values(ascending=False)
    gc = []
    for (cid, cname), n in clubs[clubs >= 5].items():
        r = league_effect(C[~((C.receiving_league == "GB1") & (C.b_club_id == cid))])
        gc.append({"dropped": f"club {cname}", "n_dropped": int(n), **r})
    gt = [{"dropped": "only arrivals before the club's first league game",
           **league_effect(C[C.arrived_before_first_game])},
          {"dropped": "only arrivals after the season started", **league_effect(C[~C.arrived_before_first_game])}]
    plural = {"Goalkeeper": "goalkeepers", "Defender": "defenders", "Midfield": "midfielders", "Attack": "forwards"}
    gpz = [{"dropped": f"excluding {plural[p]}", **league_effect(C[C.position_clean != p])} for p in ptr.POSITIONS]
    E["influence"] = {"season": gs, "club": gc, "timing": gt, "position": gpz}
    E["club_counts"] = {"clubs": int(g.b_club_id.nunique()), "clubs_with_loans": int(g[g.loan == 1].b_club_id.nunique()),
                        "top3_loan_share": float(g[g.loan == 1].b_club.value_counts().head(3).sum() / g.loan.sum())}
    N["england"] = E

    # ---- Part 5: zero composition
    N["bins"] = {"full primary sample": bins(P), "market-value sample": bins(C)}

    # ---- Part 6: valuation timing
    N["gap_bins"] = {}
    for k, gk in C.groupby("kind"):
        gap = gk.player_value_gap_days
        N["gap_bins"][k] = {lab: float(gap.between(lo, hi).mean()) for lab, lo, hi in GAP_BINS}
    N["stale"] = {"within 180 days": est(C[C.player_value_gap_days <= 180], M3, "M3", "valuation ≤ 180 days old", fe=FE, cats=CATS),
                  "within 365 days": est(C[C.player_value_gap_days <= 365], M3, "M3", "valuation ≤ 365 days old", fe=FE, cats=CATS)}

    # ---- Part 7: sanity table
    rows = [est(P, ["loan"], "1. Raw OLS", "full primary sample"),
            est(C, ["loan"], "1. Raw OLS", "market-value sample"),
            est(C, M1, "2. + league × season FE, arrival timing", "market-value sample", fe=FE),
            est(C, M2, "3. + age, age², position", "market-value sample", fe=FE, cats=CATS),
            est(C, M3, "4. Preferred (+ market values)", "market-value sample", fe=FE, cats=CATS),
            est(C, M4, "5. Club × season FE", "market-value sample", absorb="club_season", cats=CATS),
            est(C[C.share > 0], M3, "6. Preferred, nonzero-minute players", "market-value sample, share > 0", fe=FE, cats=CATS),
            est(Cp, M3, "7a. Preferred, prior-PT subsample", "prior playing time measurable", fe=FE, cats=CATS),
            est(Cp, M3 + ["prior_share"], "7b. Preferred + prior playing-time share", "prior playing time measurable", fe=FE, cats=CATS)]
    for r, k in zip(rows[:6], ("M0 full", "M0", "M1", "M2", "M3", "M4")):
        assert abs(r["coef"] - R0["models"][k]["coef"]) < 1e-9, f"{k} does not reproduce"
    MOD = pd.DataFrame(rows)
    N["sanity"] = rows

    subgroups = []
    for lab, qs in N["quintiles"].items():
        for q in qs:
            subgroups.append({"analysis": "relative-value fifth", "variant": lab, "group": q["q"], **q})
    for lab, r in (("raw (full sample)", E["raw_full"]), ("raw (market-value sample)", E["raw_common"])):
        subgroups.append({"analysis": "England", "variant": lab, "group": "GB1", "loans": r["loans"],
                          "permanents": r["permanents"], "coef": r["diff"]})
    for lab in ("preferred", "club_season_fe", "prior_sample_without", "prior_sample_with"):
        subgroups.append({"analysis": "England", "variant": lab, "group": "GB1", **E[lab]})
    for fam, lst in E["influence"].items():
        for r in lst:
            subgroups.append({"analysis": f"England influence: {fam}", "variant": r["dropped"], "group": "GB1", **r})
    for samp, bk in N["bins"].items():
        for k, v in bk.items():
            for lab, x in v.items():
                subgroups.append({"analysis": "share distribution", "variant": samp, "group": f"{k}: {lab}", "coef": x})
    tables = {"models": MOD, "subgroups": pd.DataFrame(subgroups), "prior": C}
    return N, tables


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pp(x) -> str:
    return f"{100 * x:+.1f} pp"


def ci(r) -> str:
    return f"[{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"


def render(N: dict) -> str:
    W: list[str] = []
    w = W.append
    dec = N["decomposition"]
    pm = N["prior_models"]
    pc = N["prior_coverage"]
    Q = N["quintiles"]
    E = N["england"]
    infl = E["influence"]
    st = N["stale"]
    sane = N["sanity"]
    cs = lambda k, n: f"{f0(k)} ({share(k / n) if n else '–'})"

    w("# Playing-time regression: audit before presenting")
    w("")
    w("*Generated by `python -m src.analysis.playing_time_regression_audit`. Re-estimates the existing "
      "specifications; adds one variable (prior-season playing time at the origin club) built only from games "
      "before the move. Observational, not causal. Intervals are 95%, clustered by receiving club.*")
    w("")
    d0, d1, d2, d3, d4 = dec
    sel, ctl = N["selection_effect"], N["control_effect"]
    total = d4["difference"] - d0["difference"]
    q0f, q0a, q0b = Q["full"][0], Q["prior sample, without prior PT"][0], Q["prior sample, with prior PT"][0]
    lo_l, hi_l = pc["loan"], pc["permanent"]
    ctrl_models = [r for r in sane if r["model"][0] in "4567"]
    eng_all = [E["preferred"], E["club_season_fe"], E["prior_sample_without"], E["prior_sample_with"]]
    eng_loo = [r["coef"] for r in infl["season"] + infl["club"]]

    w("## Final judgment")
    w("")
    w(f"1. **Is the raw gap real in the observed sample?** Yes: {pp(d0['difference'])} {ci(d0)} on "
      f"{f0(d0['N'])} episodes. It is a descriptive fact about these episodes, not an effect of loans.")
    w(f"2. **Sample restriction or controls?** Of the {abs(100 * total):.1f}-point move from {pp(d0['difference'])} "
      f"to {pp(d4['difference'])}, {abs(100 * sel):.1f} points ({share(sel / total)}) come from dropping episodes "
      f"without a pre-move valuation, and {abs(100 * ctl):.1f} points ({share(ctl / total)}) from controls. Almost "
      f"all of the control effect is market value ({pp(d4['difference'] - d3['difference'])}); age and position push "
      f"the other way ({pp(d3['difference'] - d2['difference'])}).")
    w(f"3. **Does prior playing time change the loan coefficient?** On the same {f0(pm['A']['N'])} episodes it "
      f"moves from {pp(pm['A']['coef'])} {ci(pm['A'])} to {pp(pm['B']['coef'])} {ci(pm['B'])}. That is closer to "
      f"zero, and close to the full-sample preferred estimate ({pp(pm['A_full_sample_same_spec']['coef'])}). It does "
      "not overturn the conclusion.")
    w(f"4. **Does the low-relative-value result survive?** In sign and rough size, yes: lowest fifth "
      f"{pp(q0f['coef'])} on the full sample; on the prior-playing-time subsample {pp(q0a['coef'])} without and "
      f"**{pp(q0b['coef'])} {ci(q0b)}** with the prior control. But that rests on {q0b['loans']} loans, so it stays an "
      "exploratory finding.")
    w(f"5. **Is England still unusually negative?** Yes. Every check gives between {pp(min(r['coef'] for r in eng_all))} "
      f"and {pp(max(r['coef'] for r in eng_all))}. Dropping any one season or any one receiving club keeps it between "
      f"{pp(min(eng_loo))} and {pp(max(eng_loo))}.")
    w(f"6. **Are stale valuations a concern?** No. Keeping only valuations at most 180 days old gives "
      f"{pp(st['within 180 days']['coef'])} {ci(st['within 180 days'])}; at most 365 days, "
      f"{pp(st['within 365 days']['coef'])} {ci(st['within 365 days'])}.")
    excl = [r for r in ctrl_models if r["hi"] < 0]
    w(f"7. **Is \"adjusted difference approximately zero\" defensible?** Yes, stated as \"no positive loan association "
      f"after controls; at most a small negative one\". The controlled estimates range from "
      f"{pp(min(r['coef'] for r in ctrl_models))} to {pp(max(r['coef'] for r in ctrl_models))}; none is positive. "
      f"Intervals entirely below zero: "
      + "; ".join(f"{r['model'][3:].strip()} ({pp(r['coef'])}, upper bound {100 * r['hi']:+.2f} pp)" for r in excl)
      + f". The preferred model's interval includes zero ({ci(sane[4])}).")
    w("8. **What to show Daniel.**")
    w(f"   - **Main result:** raw {pp(d0['difference'])}, falling to {pp(d4['difference'])} {ci(d4)} once relative "
      "value and squad value are controlled, with the decomposition (section 1) to show what moves it.")
    w(f"   - **Robustness:** club × season FE {pp(N['original']['M4']['coef'])}, nonzero-minute players "
      f"{pp(sane[6]['coef'])}, prior playing time {pp(pm['B']['coef'])} (same-sample comparison), and the valuation "
      "age restrictions.")
    w(f"   - **Exploratory findings:** the lowest relative-value fifth ({pp(q0f['coef'])}; {pp(q0b['coef'])} with "
      f"prior playing time on {q0b['loans']} loans) and England ({pp(E['preferred']['coef'])}).")
    w("")

    w("## 1. Sample selection vs controls")
    w("")
    W += md_table(["Step", "Loans", "Permanent", "Loan mean", "Permanent mean", "Difference", "95% CI"],
                  [[d["step"], f0(d["loans"]), f0(d["permanents"]), f"{d['loan_mean']:.3f}", f"{d['perm_mean']:.3f}",
                    pp(d["difference"]), ci(d)] for d in dec])
    w("")
    dr = N["dropped"]
    w(f"The first step is pure selection: the {f0(dr['permanent']['N'])} permanent signings dropped for lack of a "
      f"pre-move valuation have mean share {dr['permanent']['mean']:.2f} ({share(dr['permanent']['zero'])} at zero), "
      f"against {dr['loan']['mean']:.2f} for the {f0(dr['loan']['N'])} dropped loans. Dropping them raises the "
      f"permanent mean from {d0['perm_mean']:.3f} to {d1['perm_mean']:.3f}. The later steps keep the sample fixed, "
      "so their changes are pure covariate adjustment.")
    w("")

    w("## 2. Prior playing time")
    w("")
    w("**Construction (pre-move only).** It uses the origin club's league games in the season before the arrival "
      "season. The window runs from the player's most recent recorded move into the origin club to the day before "
      "the transfer; every game in it is checked to be dated before the move. It counts only if every game in the "
      "window has a complete appearance record. Prior share = prior league minutes ÷ (90 × the club's league games "
      "in the window). Where it cannot be measured it is missing, never zero.")
    w("")
    rows = []
    for k, lab in (("loan", "Loans"), ("permanent", "Permanent")):
        c = pc[k]
        rows.append([lab, f0(c["N"]), cs(c["origin_covered"], c["N"]), cs(c["measurable"], c["N"]),
                     f0(c["reasons"].get("origin club not in a covered league the season before", 0)),
                     f0(c["reasons"].get("no origin-club league game between his arrival there and the move", 0)),
                     f0(c["reasons"].get("incomplete appearance record in the window", 0)),
                     f"{c['prior_share_mean']:.2f}", share(c["prior_zero"])])
    W += md_table(["Market-value sample", "N", "Origin club covered the season before", "Prior playing time measurable",
                   "Not measurable: origin not covered", "… no origin-club game since he (re)joined",
                   "… incomplete record", "Mean prior share", "Prior zero minutes"], rows)
    w("")
    ng = N["no_game_entry"]
    w("\"No origin-club game since he (re)joined\" means his last recorded move into the origin club came after "
      "that club's last league game of the previous season. For "
      + "; ".join(f"{share(v['returned_from_loan'] / v['N'])} of the {f0(v['N'])} {k}s" for k, v in ng.items())
      + " that move was a return from a loan elsewhere. The measurable subsample is therefore selected toward players "
      "who spent the previous season at their origin club, in a covered league.")
    w("")
    W += md_table(["Same observations", "N", "Loan coefficient", "95% CI", "Prior-share coefficient"], [
        ["A. Preferred controls", f0(pm["A"]["N"]), pp(pm["A"]["coef"]), ci(pm["A"]), "–"],
        ["B. Preferred controls + prior share", f0(pm["B"]["N"]), pp(pm["B"]["coef"]), ci(pm["B"]),
         f"{100 * pm['prior_share_coef']['coef']:+.1f} pp per 100% (SE {100 * pm['prior_share_coef']['se']:.1f})"],
        ["B'. + prior share and a prior-zero indicator", f0(pm["B2"]["N"]), pp(pm["B2"]["coef"]), ci(pm["B2"]), "–"]])
    w("")
    pb = N["prior_balance"]
    w(f"On this subsample, loanees arrive having played far less the season before (mean prior share "
      f"{pb['loan']['prior_share_mean']:.2f} vs {pb['permanent']['prior_share_mean']:.2f}). Controlling for it "
      f"moves the loan coefficient by {pp(pm['B']['coef'] - pm['A']['coef'])}. This is a robustness check, not a new "
      "primary model, because the subsample is selected.")
    w("")

    w("## 3. The low-relative-value result")
    w("")
    w("Fifths use the cut-points of the full market-value sample, so they are comparable across rows.")
    w("")
    rows = []
    for q in range(5):
        r = [f"Q{q + 1}"]
        for lab in ("full", "prior sample, without prior PT", "prior sample, with prior PT"):
            x = Q[lab][q]
            r.append(f"{pp(x['coef'])} {ci(x)} ({x['loans']}/{x['permanents']})")
        rows.append(r)
    W += md_table(["Fifth", "Full market-value sample", "Prior-PT subsample, no prior control",
                   "Prior-PT subsample, + prior share"], rows)
    w("")
    w("Cells: adjusted loan − permanent difference, 95% CI, (loans / permanents).")
    w("")

    w("## 4. England")
    w("")
    rows = [["Raw (full primary sample)", f0(E["raw_full"]["loans"]), f0(E["raw_full"]["permanents"]), pp(E["raw_full"]["diff"]), "–"],
            ["Raw (market-value sample)", f0(E["raw_common"]["loans"]), f0(E["raw_common"]["permanents"]), pp(E["raw_common"]["diff"]), "–"],
            ["Preferred (loan × league)", f0(E["preferred"]["loans"]), f0(E["preferred"]["permanents"]), pp(E["preferred"]["coef"]), ci(E["preferred"])],
            ["Club × season FE (loan × league)", f0(E["club_season_fe"]["loans"]), f0(E["club_season_fe"]["permanents"]),
             pp(E["club_season_fe"]["coef"]), ci(E["club_season_fe"])],
            ["Prior-PT subsample, no prior control", f0(E["prior_sample_without"]["loans"]), f0(E["prior_sample_without"]["permanents"]),
             pp(E["prior_sample_without"]["coef"]), ci(E["prior_sample_without"])],
            ["Prior-PT subsample, + prior share", f0(E["prior_sample_with"]["loans"]), f0(E["prior_sample_with"]["permanents"]),
             pp(E["prior_sample_with"]["coef"]), ci(E["prior_sample_with"])]]
    W += md_table(["England (GB1)", "Loans", "Permanent", "Difference", "95% CI"], rows)
    w("")
    cc = E["club_counts"]
    ss = infl["season"]
    cl = infl["club"]
    w(f"- **Season.** Dropping one arrival season at a time keeps England between "
      f"{pp(min(r['coef'] for r in ss))} and {pp(max(r['coef'] for r in ss))}.")
    w(f"- **Receiving club.** {cc['clubs_with_loans']} English clubs received loans, and the top three hold "
      f"{share(cc['top3_loan_share'])} of them. Dropping any one of the {len(cl)} clubs with at least 5 episodes keeps "
      f"England between {pp(min(r['coef'] for r in cl))} and {pp(max(r['coef'] for r in cl))}.")
    tm = infl["timing"]
    w(f"- **Arrival timing** (model re-estimated on each subset): {tm[0]['dropped']} {pp(tm[0]['coef'])} {ci(tm[0])}; "
      f"{tm[1]['dropped']} {pp(tm[1]['coef'])} {ci(tm[1])}.")
    w("- **Position** (dropping one at a time): "
      + "; ".join(f"{r['dropped']} {pp(r['coef'])}" for r in infl["position"]) + ".")
    w("- England remains unusually negative under every check; no single season, club or position drives it.")
    w("")

    w("## 5. Where the zero-minute difference sits")
    w("")
    for samp, bk in N["bins"].items():
        rows = []
        for lab, _, _ in SHARE_BINS:
            a_, b_ = bk["loan"][lab], bk["permanent"][lab]
            rows.append([lab, share(a_), share(b_), f"{100 * (a_ - b_):+.1f} pp"])
        w(f"**{samp[0].upper() + samp[1:]}:**")
        w("")
        W += md_table(["Playing-time share", "Loans", "Permanent", "Loan − permanent"], rows)
        w("")
    fb = N["bins"]["full primary sample"]
    low = sum(fb["permanent"][l] - fb["loan"][l] for l in ("exactly 0", "0–10%"))
    mid = sum(fb["loan"][l] - fb["permanent"][l] for l in ("25–50%", "50–75%"))
    w(f"It is neither only true zeros nor a whole-distribution shift. Permanent signings are over-represented at "
      f"zero and below 10% ({100 * low:.1f} points together), and loanees in the 25–75% range ({100 * mid:.1f} "
      f"points). Above 75% the two groups are close ({share(fb['loan']['over 75%'])} vs "
      f"{share(fb['permanent']['over 75%'])}).")
    w("")

    w("## 6. Valuation timing")
    w("")
    gb = N["gap_bins"]
    W += md_table(["Valuation before the move", "Loans", "Permanent"],
                  [[lab, share(gb["loan"][lab]), share(gb["permanent"][lab])] for lab, _, _ in GAP_BINS])
    w("")
    W += md_table(["Preferred model", "N", "Loan coefficient", "95% CI"],
                  [["All valuations", f0(sane[4]["N_used"]), pp(sane[4]["coef"]), ci(sane[4])]]
                  + [[f"Valuation {k}", f0(v["N_used"]), pp(v["coef"]), ci(v)] for k, v in st.items()])
    w("")

    w("## 7. The core specifications, re-estimated")
    w("")
    W += md_table(["Model", "Sample", "N", "Loan coefficient", "SE", "95% CI"],
                  [[r["model"], r["sample"], f0(r["N_used"]), pp(r["coef"]), f"{100 * r['se']:.2f}", ci(r)] for r in sane])
    w("")
    w(f"Models 1–5 reproduce the regression report exactly. Every controlled estimate lies between "
      f"{pp(min(r['coef'] for r in ctrl_models))} and {pp(max(r['coef'] for r in ctrl_models))}, so the conclusion "
      "is stable: the raw loan advantage does not survive the controls, and what remains is zero to slightly "
      "negative.")
    w("")
    return "\n".join(W) + "\n"


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


PRIOR_COLS = ["event_id", "kind", "player_id", "player_name", "move_date", "arrival_season", "receiving_league",
              "a_club_id", "a_club", "b_club_id", "b_club", "share", "prior_season", "origin_covered_prior_season",
              "origin_entry_date", "origin_entry_unobserved", "prior_games_available", "prior_games_complete",
              "prior_minutes", "prior_share", "prior_zero", "prior_measurable", "prior_reason",
              "prior_window_max_game_date"]


def main() -> dict:
    N, tb = compute()
    pr = tb["prior"][PRIOR_COLS].copy()
    for c in ("move_date", "origin_entry_date", "prior_window_max_game_date"):
        pr[c] = pd.to_datetime(pr[c]).dt.date
    pr.to_csv(PRIOR_CSV, index=False)
    extra = [{**v, "model": f"Preferred, {k}"} for k, v in N["stale"].items()]
    extra += [{"model": f"Prior-PT comparison {k}", "sample": "prior playing time measurable", **v}
              for k, v in N["prior_models"].items() if k in ("A", "B", "B2")]
    pd.concat([tb["models"], pd.DataFrame(extra)], ignore_index=True).to_csv(MODELS_CSV, index=False)
    tb["subgroups"].to_csv(SUBGROUPS_CSV, index=False)
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    AUDIT_MD.write_text(render(N))
    print(f"wrote {AUDIT_MD.name}")
    return N


if __name__ == "__main__":
    main()
