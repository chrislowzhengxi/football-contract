"""Professor-ready empirical summary: loans vs permanent transfers on the calendar-first sample.

    python -m src.analysis.professor_empirical

Reads the master (`analysis_episode_master.csv`) and the lineage numbers only. Every preferred estimate is recomputed
here from the master and asserted equal to the value the lineage pipeline saved, so the two can never drift apart.
Writes `professor_empirical_summary.md` and, under `empirical/`, one CSV per table, the figures (PNG and PDF) and
`empirical_numbers.json`. Nothing here is causal: every coefficient is a conditional association.
"""
from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from . import appearance_audit as aud  # noqa: E402
from . import data_lineage as dl  # noqa: E402
from . import playing_time_age as pta  # noqa: E402
from . import playing_time_regression as reg  # noqa: E402
from .daniel_scope_final import f0, md_table  # noqa: E402

OUT = dl.OUT
EMP = OUT / "empirical"
FIG = EMP / "figures"
SUMMARY_MD = OUT / "professor_empirical_summary.md"
NUMBERS_JSON = EMP / "empirical_numbers.json"
WRITES = (SUMMARY_MD,)                     # plus everything under empirical/
KINDS = ("loan", "permanent")
MAIN, INJ = "in_cf_main_model_sample", "in_cf_injury_model_sample"
SUMMER, WINTER = (6, 7, 8, 9), (12, 1, 2, 3)
BOOT = 200                                 # cluster-bootstrap replications for the decomposition
SEED = 20261028
TOL = 1e-10

# chart roles (validated pair: categorical slots 1-2 of the reference palette, light surface)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COL = {"loan": "#2a78d6", "permanent": "#eb6834"}
COL_CAL = {"normal": "#1baf7a", "off_calendar": "#6250d6"}
LABEL = {"loan": "Loans", "permanent": "Permanent transfers"}


# ---------------------------------------------------------------------------
# Frames and models
# ---------------------------------------------------------------------------

def frame(M: pd.DataFrame, mask: pd.Series, y: str = "minutes_share") -> pd.DataFrame:
    """Regression frame for the rows in `mask`, built from master columns only; age centred on these rows."""
    C = M[mask].copy()
    C["share"] = C[y].astype(float)
    C["loan"] = C.kind.eq("loan").astype(int)
    C["age"] = C.age_exact
    C["age_c"] = C.age - C.age.mean()
    C["age_c2"] = C.age_c ** 2
    C["b_club_id"] = C.to_club_id
    C["month"] = pd.to_datetime(C.transfer_date).dt.month
    C["window"] = np.select([C.month.isin(SUMMER), C.month.isin(WINTER)], ["summer", "winter"], "outside windows")
    C["zero"] = C.true_zero.astype(bool).astype(int)
    return C


def m3(C: pd.DataFrame, extra: list[str] | None = None, cats: list[str] | None = None) -> dict:
    return reg.fit(C, reg.M3 + (extra or []), fe="league_season", cats=["position_clean"] + (cats or []))


def co(m: dict, name: str = "loan") -> dict:
    return reg.coef(m, name)


def ladder(C: pd.DataFrame) -> dict:
    specs = {
        "L0 raw difference": dict(x=["loan"], fe=None, cats=None),
        "L1 + league×season FE + position FE": dict(x=["loan"], fe="league_season", cats=["position_clean"]),
        "L2 + arrival timing": dict(x=["loan", "remain"], fe="league_season", cats=["position_clean"]),
        "L3 + age, age²": dict(x=reg.M2, fe="league_season", cats=["position_clean"]),
        "L4 + log relative value, log squad value (M3, preferred)": dict(x=reg.M3, fe="league_season",
                                                                         cats=["position_clean"]),
    }
    return {k: co(reg.fit(C, **v)) for k, v in specs.items()}


def raw_diff(C: pd.DataFrame, y: str = "share") -> dict:
    """Loan minus permanent mean, with the receiving-club clustered SE (OLS on the loan dummy alone)."""
    return co(reg.fit(C, ["loan"], y=y))


# ---------------------------------------------------------------------------
# Decomposition (Gelbach 2016): raw gap -> M3, split by covariate group, order-invariant
# ---------------------------------------------------------------------------

def design(C: pd.DataFrame, value_cols: list[str]) -> tuple[np.ndarray, dict]:
    parts = {"const": pd.DataFrame({"const": np.ones(len(C))}, index=C.index), "loan": C[["loan"]].astype(float)}
    groups = {
        "league × season": pd.get_dummies(C.league_season, prefix="fe", drop_first=True, dtype=float),
        "position": pd.get_dummies(C.position_clean, prefix="pos", drop_first=True, dtype=float),
        "arrival timing": C[["remain"]].astype(float),
        "age": C[["age_c", "age_c2"]].astype(float),
    }
    for c in value_cols:
        groups[c] = C[[c]].astype(float)
    cols, idx = [], {}
    for name, df in list(parts.items()) + list(groups.items()):
        idx[name] = list(range(sum(len(x) for x in cols), sum(len(x) for x in cols) + df.shape[1]))
        cols.append(df.columns)
    X = pd.concat(list(parts.values()) + list(groups.values()), axis=1).to_numpy()
    return X, {k: v for k, v in idx.items()}


def gelbach_point(X: np.ndarray, y: np.ndarray, loan: np.ndarray, idx: dict, w: np.ndarray | None = None) -> dict:
    w = np.ones(len(y)) if w is None else w
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    l1, l0 = loan == 1, loan == 0
    mean = lambda v, m: float((v[m] * w[m]).sum() / w[m].sum())
    raw = mean(y, l1) - mean(y, l0)
    out = {"raw": raw, "full": float(beta[idx["loan"][0]])}
    for g, cols in idx.items():
        if g in ("const", "loan"):
            continue
        comp = X[:, cols] @ beta[cols]
        out[g] = mean(comp, l1) - mean(comp, l0)
    return out


def gelbach(C: pd.DataFrame, value_cols: list[str]) -> dict:
    X, idx = design(C, value_cols)
    y, loan = C.share.to_numpy(float), C.loan.to_numpy()
    point = gelbach_point(X, y, loan, idx)
    groups = [g for g in idx if g not in ("const", "loan")]
    assert abs(point["raw"] - point["full"] - sum(point[g] for g in groups)) < 1e-9
    rng = np.random.default_rng(SEED)
    clubs = C.b_club_id.to_numpy()
    uc, inv = np.unique(clubs, return_inverse=True)
    draws = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(uc), len(uc))
        w = np.bincount(pick, minlength=len(uc))[inv].astype(float)
        draws.append(gelbach_point(X, y, loan, idx, w))
    D = pd.DataFrame(draws)
    rows = []
    for g in ["raw", *groups, "full"]:
        rows.append({"term": g, "estimate": point[g], "boot_se": float(D[g].std(ddof=1)),
                     "lo": float(D[g].quantile(0.025)), "hi": float(D[g].quantile(0.975))})
    return {"rows": rows, "groups": groups, "point": point, "boot_reps": BOOT}


def one_at_a_time(C: pd.DataFrame) -> list[dict]:
    """Loan coefficient when ONE control group is added to the FE + position base (order-dependent view)."""
    base = dict(fe="league_season", cats=["position_clean"])
    add = {"(base: FE + position only)": [], "arrival timing": ["remain"], "age, age²": ["age_c", "age_c2"],
           "log player value": ["log_player_value"], "log squad value": ["log_squad_value"],
           "log relative value": ["log_relative_value"],
           "log relative value + log squad value": ["log_relative_value", "log_squad_value"]}
    return [{"added": k, **co(reg.fit(C, ["loan"] + v, **base))} for k, v in add.items()]


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------

def dist(s: pd.Series) -> dict:
    s = s.dropna()
    return {"N": len(s), "mean": float(s.mean()), "p10": float(s.quantile(.1)), "p25": float(s.quantile(.25)),
            "median": float(s.median()), "p75": float(s.quantile(.75)), "p90": float(s.quantile(.9))}


CONT = [("age at the move (years)", "age", 1), ("player value (EUR m)", "player_value", 1e-6),
        ("receiving squad value (EUR m)", "squad_value", 1e-6),
        ("relative value (% of receiving squad)", "relative_value", 100),
        ("season remaining at arrival (share of league games)", "remain", 1),
        ("league games in the window", "available_games", 1), ("minutes share", "share", 1)]


def overview(C: pd.DataFrame) -> dict:
    out = {"N": {k: int(C.kind.eq(k).sum()) for k in KINDS}, "players": int(C.player_id.nunique()),
           "receiving_clubs": int(C.to_club_id.nunique()), "leagues": sorted(C.receiving_league.unique()),
           "seasons": [int(C.arrival_season.min()), int(C.arrival_season.max())]}
    out["continuous"] = {lab: {k: {kk: (vv * sc if kk != "N" else vv) for kk, vv in dist(C.loc[C.kind.eq(k), col]).items()}
                               for k in KINDS} for lab, col, sc in CONT}
    out["position"] = {k: {p: float(v) for p, v in C.loc[C.kind.eq(k), "position_clean"].value_counts(normalize=True).items()}
                       for k in KINDS}
    out["window"] = {k: {p: float(v) for p, v in C.loc[C.kind.eq(k), "window"].value_counts(normalize=True).items()}
                     for k in KINDS}
    out["by_season"] = pd.crosstab(C.arrival_season.astype(int), C.kind)
    out["by_league"] = pd.crosstab(C.receiving_league, C.kind)
    out["by_month"] = pd.crosstab(C.month, C.kind)
    return out


def pt_distribution(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in KINDS:
        s = C.loc[C.kind.eq(k), "share"]
        rows.append({"kind": k, "N": len(s), "mean": s.mean(), "median": s.median(),
                     "true zero (0 minutes)": C.loc[C.kind.eq(k), "zero"].mean(),
                     **{f"below {int(t * 100)}%": (s < t).mean() for t in (0.10, 0.25, 0.50, 0.75)},
                     "75% or more": (s >= 0.75).mean()})
    return pd.DataFrame(rows)


def balance(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, col in [("log player value", "log_player_value"), ("log relative value", "log_relative_value"),
                     ("log squad value", "log_squad_value"), ("age (years)", "age"),
                     ("season remaining at arrival", "remain"), ("summer-window arrival", None),
                     ("goalkeeper", "Goalkeeper"), ("defender", "Defender"), ("midfield", "Midfield"),
                     ("attack", "Attack")]:
        if col is None:
            v = C.window.eq("summer").astype(float)
        elif col in ("Goalkeeper", "Defender", "Midfield", "Attack"):
            v = C.position_clean.eq(col).astype(float)
        else:
            v = C[col].astype(float)
        a, b = v[C.loan.eq(1)], v[C.loan.eq(0)]
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"characteristic": lab, "loans": a.mean(), "permanent": b.mean(), "difference": a.mean() - b.mean(),
                     "standardised difference": (a.mean() - b.mean()) / sd if sd > 0 else np.nan})
    return pd.DataFrame(rows)


def league_season_balance(C: pd.DataFrame) -> dict:
    sh = pd.crosstab(C.league_season, C.kind, normalize="columns")
    d = (sh["loan"] - sh["permanent"]).abs()
    lg = pd.crosstab(C.receiving_league, C.kind, normalize="columns")
    return {"cells": int(sh.shape[0]), "dissimilarity_index": float(0.5 * d.sum()),
            "league_share": {i: {"loan": float(r.loan), "permanent": float(r.permanent)} for i, r in lg.iterrows()}}


# ---------------------------------------------------------------------------
# Heterogeneity
# ---------------------------------------------------------------------------

def age_section(C: pd.DataFrame) -> dict:
    G = pta.add_groups(C, pta.CUTOFF)
    rows = []
    for o, lab in ((0.0, f"young (≤{pta.CUTOFF})"), (1.0, f"older (≥{pta.CUTOFF + 1})")):
        g = G[G.older.eq(o)]
        rd = raw_diff(g)
        rows.append({"group": lab, "N loans": int(g.loan.sum()), "N permanent": int((1 - g.loan).sum()),
                     "mean share, loans": g.loc[g.loan.eq(1), "share"].mean(),
                     "mean share, permanent": g.loc[g.loan.eq(0), "share"].mean(),
                     "raw difference": rd["coef"], "raw SE": rd["se"], "raw p": rd["p"]})
    am = pta.age_model(G)
    by_year = G.assign(a=G.age_years.clip(pta.AGE_LO, pta.AGE_HI)).groupby(["a", "kind"]).share.agg(["mean", "sem", "size"])
    return {"table": pd.DataFrame(rows), "model": am, "by_year": by_year.reset_index()}


def value_section(C: pd.DataFrame) -> dict:
    C = C.copy()
    C["q"] = pd.qcut(C.log_relative_value, 5, labels=[f"Q{i}" for i in range(1, 6)])
    qd = pd.get_dummies(C.q, dtype=float)
    for q in qd.columns:
        C[f"loan_{q}"] = C.loan * qd[q]
    mq = reg.fit(C, [f"loan_{q}" for q in qd.columns] + ["remain", "age_c", "age_c2", "log_squad_value"],
                 fe="league_season", cats=["position_clean", "q"])
    rows = []
    for q in qd.columns:
        g = C[C.q.eq(q)]
        rd = raw_diff(g)
        a = co(mq, f"loan_{q}")
        rows.append({"bin": q, "relative value min (% of squad)": 100 * g.relative_value.min(),
                     "relative value median": 100 * g.relative_value.median(),
                     "relative value max": 100 * g.relative_value.max(),
                     "N loans": int(g.loan.sum()), "N permanent": int((1 - g.loan).sum()),
                     "loan share of bin": g.loan.mean(),
                     "mean share, loans": g.loc[g.loan.eq(1), "share"].mean(),
                     "mean share, permanent": g.loc[g.loan.eq(0), "share"].mean(),
                     "raw difference": rd["coef"], "raw SE": rd["se"],
                     "adjusted loan coef": a["coef"], "adjusted SE": a["se"], "adjusted lo": a["lo"], "adjusted hi": a["hi"],
                     "adjusted p": a["p"]})
    # continuous interaction, as in the Week 5 regression module
    C["lrv_c"] = C.log_relative_value - C.log_relative_value.mean()
    C["loan_x_lrv"] = C.loan * C.lrv_c
    mi = reg.fit(C, ["loan", "remain", "age_c", "age_c2", "lrv_c", "loan_x_lrv", "log_squad_value"],
                 fe="league_season", cats=["position_clean"])
    r = mi["res"]
    inter = {v: {"coef": float(r.params[v]), "se": float(r.bse[v]), "p": float(r.pvalues[v])}
             for v in ("loan", "lrv_c", "loan_x_lrv")}
    inter["mean_log_relative_value"] = float(C.log_relative_value.mean())
    C["d"] = pd.qcut(C.log_relative_value, 10, labels=False) + 1
    dec = C.groupby(["d", "kind"]).agg(mean=("share", "mean"), sem=("share", "sem"), n=("share", "size"),
                                       rel_median=("relative_value", "median")).reset_index()
    return {"table": pd.DataFrame(rows), "interaction": inter, "deciles": dec}


def timing_section(C: pd.DataFrame) -> dict:
    C = C.copy()
    rows = []
    for w in ("summer", "winter", "outside windows"):
        g = C[C.window.eq(w)]
        rd = raw_diff(g) if g.loan.nunique() == 2 else {"coef": np.nan, "se": np.nan, "p": np.nan}
        rows.append({"window": w, "N loans": int(g.loan.sum()), "N permanent": int((1 - g.loan).sum()),
                     "mean remain": g.remain.mean(), "mean share, loans": g.loc[g.loan.eq(1), "share"].mean(),
                     "mean share, permanent": g.loc[g.loan.eq(0), "share"].mean(), "raw difference": rd["coef"],
                     "raw SE": rd["se"]})
    C["winter"] = C.window.eq("winter").astype(float)
    C["loan_x_winter"] = C.loan * C.winter
    mw = reg.fit(C, reg.M3 + ["winter", "loan_x_winter"], fe="league_season", cats=["position_clean"])
    C["remain_c"] = C.remain - C.remain.mean()
    C["loan_x_remain"] = C.loan * C.remain_c
    mr = reg.fit(C, ["loan", "remain_c", "loan_x_remain", "age_c", "age_c2", "log_relative_value", "log_squad_value"],
                 fe="league_season", cats=["position_clean"])
    bins = pd.cut(C.remain, [0, 0.5, 0.75, 0.999999, 1.0], labels=["under half", "half to three-quarters",
                                                                     "three-quarters to almost all",
                                                                     "all (arrived before the first league game)"],
                  include_lowest=True)
    rb = []
    for b in bins.cat.categories:
        g = C[bins.eq(b)]
        rd = raw_diff(g)
        rb.append({"season remaining at arrival": b, "N loans": int(g.loan.sum()), "N permanent": int((1 - g.loan).sum()),
                   "mean share, loans": g.loc[g.loan.eq(1), "share"].mean(),
                   "mean share, permanent": g.loc[g.loan.eq(0), "share"].mean(), "raw difference": rd["coef"],
                   "raw SE": rd["se"]})
    return {"window": pd.DataFrame(rows), "remain_bins": pd.DataFrame(rb),
            "winter_model": {"loan (summer)": co(mw, "loan"), "loan × winter": co(mw, "loan_x_winter")},
            "remain_model": {"loan (at mean remain)": co(mr, "loan"), "loan × remain": co(mr, "loan_x_remain"),
                             "mean_remain": float(C.remain.mean())}}


def calendar_section(C: pd.DataFrame) -> dict:
    L = C[C.kind.eq("loan")].copy()
    rows = []
    for st in ("normal", "off_calendar"):
        g = L[L.loan_calendar_status.eq(st)]
        rows.append({"status": st, "N": len(g), "mean share": g.share.mean(), "median share": g.share.median(),
                     "true zero": g.zero.mean(), "share below 25%": (g.share < .25).mean(), "mean age": g.age.mean(),
                     "median relative value (% of squad)": 100 * g.relative_value.median(),
                     "mean log relative value": g.log_relative_value.mean(), "mean remain at arrival": g.remain.mean(),
                     "winter-window arrival": g.window.eq("winter").mean(),
                     "cross-border loan": g.calendar_context.eq("cross-border").mean(),
                     "median loan duration (days)": g.duration_days.median(),
                     "median days from nearest calendar date": g.days_to_nearest_standard.median()})
    path = pd.crosstab(L.next_category, L.loan_calendar_status, normalize="columns")
    pathn = pd.crosstab(L.next_category, L.loan_calendar_status)
    L["off"] = L.loan_calendar_status.eq("off_calendar").astype(int)
    diff = co(reg.fit(L, ["off"]), "off")
    return {"table": pd.DataFrame(rows), "pathway": path, "pathway_n": pathn, "share_diff": diff,
            "contexts": sorted(L.calendar_context.dropna().unique())}


# ---------------------------------------------------------------------------
# Appearance-row gaps: minutes the appearances table does not have
# ---------------------------------------------------------------------------

def lineup_check(M: pd.DataFrame) -> pd.DataFrame:
    """The appearance-row check of `appearance_audit.episode_check`, for every reliable episode; must agree with the
    master's `pt_lineup_contradicted`."""
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        Q = aud.episode_check(con, M[M.pt_reliable])
    finally:
        con.close()
    Q["contradicted"] = Q.games_played_without_row.gt(0)
    assert set(Q.index[Q.contradicted]) == set(M.event_id[M.pt_lineup_contradicted])
    return Q


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def frac_logit_ame(C: pd.DataFrame) -> dict:
    """The Week 5 functional-form check: fractional logit with league and season dummies, average marginal effect."""
    Xg = C[reg.M3].astype(float).join(pd.get_dummies(C.position_clean, prefix="pos", drop_first=True, dtype=float))
    Xg = Xg.join(pd.get_dummies(C.receiving_league, prefix="lg", drop_first=True, dtype=float))
    Xg = Xg.join(pd.get_dummies(C.arrival_season.astype(int), prefix="s", drop_first=True, dtype=float))
    Xg = sm.add_constant(Xg)
    grp = pd.factorize(C.b_club_id)[0]
    fl = sm.GLM(C.share.astype(float), Xg, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                                              cov_kwds={"groups": grp})
    out = reg.ame_logit(fl, Xg)
    from scipy import stats
    out["p"] = float(2 * stats.norm.sf(abs(out["coef"] / out["se"])))
    return out


def robustness(M: pd.DataFrame, N: dict, Q: pd.DataFrame) -> list[dict]:
    cf = M[MAIN]
    bad = M.event_id.isin(Q.index[Q.contradicted])
    normal_only = cf & (M.kind.eq("permanent") | M.loan_calendar_status.eq("normal"))
    spec = [
        ("Week 5/6 sample (no loan-calendar step)", M.in_age_model_sample, "minutes_share", "earlier sequence"),
        ("Calendar-first sample (preferred)", cf, "minutes_share", "preferred"),
        ("Calendar-normal loans only", normal_only, "minutes_share",
         "NOT preferred: off-calendar endings can follow low playing time, so this selects on the outcome"),
        ("Calendar-first, without high-confidence same-organisation moves", cf & ~M.same_organisation_move_v2,
         "minutes_share", "academy sensitivity B"),
        ("Calendar-first, without any academy-promotion candidate", cf & ~M.academy_promotion_candidate,
         "minutes_share", "academy sensitivity C"),
        ("Calendar-first, without episodes whose minutes are contradicted by line-ups / substitutions",
         cf & ~bad, "minutes_share", "data check (section 11)"),
        ("Injury sample, current share", M[INJ], "current_share", "like-for-like base for the next row"),
        ("Injury sample, injury-adjusted (fit) share", M[INJ], "fit_adjusted_share", "injury robustness"),
    ]
    rows = []
    for lab, mask, y, note in spec:
        C = frame(M, mask, y)
        r = co(m3(C))
        rows.append({"specification": lab, "note": note, "N": r["N"], "N loans": int(C.loan.sum()),
                     "N permanent": int((1 - C.loan).sum()), "raw gap": raw_diff(C)["coef"], **r})
    C = frame(M, cf)
    rows.append({"specification": "Calendar-first, fractional logit (average marginal effect)",
                 "note": "functional form (Week 5 check)", "N": len(C), "N loans": int(C.loan.sum()),
                 "N permanent": int((1 - C.loan).sum()), "raw gap": raw_diff(C)["coef"], **frac_logit_ame(C)})
    # every row that the lineage pipeline also estimates must agree with it exactly
    cfv = N["calendar_first"]["variants"]
    saved = {0: N["regression"]["revised"]["m3"], 1: cfv["classifiable"]["estimates"]["m3"],
             2: cfv["normal_only"]["estimates"]["m3"], 3: N["academy"]["variants"]["B"]["estimates"]["m3"],
             4: N["academy"]["variants"]["C"]["estimates"]["m3"],
             6: cfv["classifiable"]["estimates"]["injury_m3_current"],
             7: cfv["classifiable"]["estimates"]["injury_m3_fit"]}
    for i, s in saved.items():
        assert rows[i]["N"] == s["N"] and abs(rows[i]["coef"] - s["coef"]) < TOL, (rows[i]["specification"], s)
    return rows


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

EX_COLS = ["event_id", "player_name", "player_id", "transfer_date", "season", "kind", "from_club_name", "to_club_name",
           "receiving_league", "transfer_type_raw", "fee_display_raw", "position_clean", "age_years", "player_value",
           "squad_value", "relative_value", "remain", "available_games", "appearances", "minutes", "minutes_share",
           "true_zero", "terminal_date", "duration_days", "loan_calendar_status", "next_category",
           "confirmed_injury_games", "fit_adjusted_share", "same_org_confidence"]
EXAMPLE_RULES = [
    ("high-playing-time loan", "loan; share ≥ 0.85 and at least 30 window games; highest player value",
     lambda C: C.kind.eq("loan") & C.share.ge(.85) & C.available_games.ge(30)),
    ("zero-playing-time loan", "loan; true zero with at least 15 window games, not contradicted by line-ups or "
     "substitutions; highest player value",
     lambda C: C.kind.eq("loan") & C.true_zero.astype(bool) & C.available_games.ge(15) & ~C.contradicted),
    ("high-playing-time permanent transfer",
     "permanent, not an academy candidate; share ≥ 0.85 and at least 30 window games; highest player value",
     lambda C: C.kind.eq("permanent") & ~C.academy_promotion_candidate & C.share.ge(.85) & C.available_games.ge(30)),
    ("zero-playing-time permanent transfer",
     "permanent, not an academy candidate; true zero with at least 15 window games, not contradicted by line-ups or "
     "substitutions; highest player value",
     lambda C: C.kind.eq("permanent") & ~C.academy_promotion_candidate & C.true_zero.astype(bool)
     & C.available_games.ge(15) & ~C.contradicted),
    ("off-calendar loan", "loan with an off-calendar ending; highest player value",
     lambda C: C.kind.eq("loan") & C.loan_calendar_status.eq("off_calendar")),
    ("academy-promotion sensitivity case",
     "high-confidence same-organisation move with at least 15 window games; highest player value",
     lambda C: C.same_organisation_move_v2 & C.available_games.ge(15)),
]


def examples(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, rule, f in EXAMPLE_RULES:
        g = C[f(C)].sort_values(["player_value", "event_id"], ascending=[False, True])
        assert len(g), case
        x = g.iloc[0][EX_COLS].to_dict()
        x["fee_display_raw"] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(x["fee_display_raw"]))).strip()
        rows.append({"case": case, "selection rule": rule, "eligible episodes": len(g), **x})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK, fontsize=11)


def _save(fig, name: str) -> list[str]:
    fig.patch.set_facecolor(SURFACE)
    paths = []
    for ext in ("png", "pdf"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        paths.append(str(p.relative_to(OUT)))
    plt.close(fig)
    return paths


def fig_distribution(C: pd.DataFrame) -> list[str]:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    edges = np.r_[0, 1e-9, np.arange(0.05, 1.0001, 0.05)]
    w = 0.018
    centers = np.r_[-0.04, (edges[1:-1] + edges[2:]) / 2]
    for i, k in enumerate(KINDS):
        s = C.loc[C.kind.eq(k), "share"]
        h = np.r_[(s == 0).mean(), np.histogram(s[s > 0], bins=edges[1:])[0] / len(s)]
        a.bar(centers + (i - 0.5) * (w + 0.002), 100 * h, width=w, color=COL[k], label=LABEL[k])
    a.set_xticks([-0.04, .25, .5, .75, 1], ["exactly 0", "0.25", "0.5", "0.75", "1"])
    _style(a, "minutes share (bars of 0.05; first bar = no minutes at all)", "% of the group")
    a.set_title("Distribution of playing time", color=INK, fontsize=12, loc="left")
    a.legend(frameon=False, fontsize=10)
    for k in KINDS:
        s = np.sort(C.loc[C.kind.eq(k), "share"].to_numpy())
        b.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=COL[k], linewidth=2, label=LABEL[k])
        x0 = 0.45
        y0 = np.searchsorted(s, x0, side="right") / len(s)
        if k == "permanent":
            b.text(x0 - 0.02, y0 + 0.06, LABEL[k], color=INK2, fontsize=10, ha="right", va="bottom")
        else:
            b.text(x0 + 0.03, y0 - 0.06, LABEL[k], color=INK2, fontsize=10, ha="left", va="top")
    _style(b, "minutes share", "share of the group at or below")
    b.set_title("Cumulative distribution", color=INK, fontsize=12, loc="left")
    b.grid(axis="x", color=GRID, linewidth=0.8)
    return _save(fig, "fig1_playing_time_distribution")


def fig_ladder(L: dict, G: dict) -> list[str]:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.05, 1]}, constrained_layout=True)
    labs = ["Raw", "+ FE &\nposition", "+ arrival\ntiming", "+ age,\nage²", "+ value\n(M3)"]
    v = list(L.values())
    x = np.arange(len(v))
    a.axhline(0, color=INK2, linewidth=1)
    a.errorbar(x, [100 * r["coef"] for r in v], yerr=[[100 * (r["coef"] - r["lo"]) for r in v],
                                                       [100 * (r["hi"] - r["coef"]) for r in v]],
               fmt="o", color=COL["loan"], ecolor=COL["loan"], elinewidth=2, capsize=0, markersize=8)
    for xi, r in zip(x, v):
        a.text(xi + 0.1, 100 * r["coef"] + 0.12, pp(r["coef"]).replace(" pp", ""), color=INK, fontsize=10, va="bottom")
    a.set_xticks(x, labs)
    _style(a, "", "loan coefficient (percentage points of share)")
    a.set_title("Loan coefficient as controls are added (95% CI)", color=INK, fontsize=12, loc="left")
    terms = [r for r in G["rows"] if r["term"] not in ("raw", "full")]
    y = np.arange(len(terms))[::-1]
    b.axvline(0, color=INK2, linewidth=1)
    b.barh(y, [100 * r["estimate"] for r in terms], height=0.55, color=COL["loan"])
    b.errorbar([100 * r["estimate"] for r in terms], y, xerr=[[100 * (r["estimate"] - r["lo"]) for r in terms],
                                                             [100 * (r["hi"] - r["estimate"]) for r in terms]],
               fmt="none", ecolor=INK2, elinewidth=1.2)
    nice = {"log_relative_value": "relative value", "log_squad_value": "squad value"}
    b.set_yticks(y, [nice.get(r["term"], r["term"]) for r in terms])
    xr = max(100 * r["hi"] for r in terms) + 0.4
    for yi, r in zip(y, terms):
        b.text(xr, yi, pp(r["estimate"]).replace(" pp", ""), color=INK, fontsize=10, va="center", ha="left")
    b.set_xlim(min(100 * r["lo"] for r in terms) - 0.4, xr + 1.0)
    _style(b, "share of the raw gap it accounts for (pp)", "")
    b.grid(axis="x", color=GRID, linewidth=0.8)
    b.grid(axis="y", visible=False)
    b.set_title("What explains the raw gap (Gelbach decomposition)", color=INK, fontsize=12, loc="left")
    return _save(fig, "fig2_ladder_and_decomposition")


def fig_value(V: dict, C: pd.DataFrame) -> list[str]:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    d = V["deciles"]
    for i, k in enumerate(KINDS):
        g = d[d.kind.eq(k)]
        xs = g.d + (i - 0.5) * 0.18
        a.errorbar(xs, g["mean"], yerr=1.96 * g["sem"], fmt="o-", color=COL[k], linewidth=2, markersize=7,
                   elinewidth=1.5, label=LABEL[k])
    rel = d.groupby("d").rel_median.median()
    a.set_xticks(rel.index, [f"{100 * r:.1f}%" if r < 0.1 else f"{100 * r:.0f}%" for r in rel], fontsize=9)
    _style(a, "relative-value decile (median % of receiving squad)", "mean minutes share")
    a.set_title("Playing time rises with relative value in both groups", color=INK, fontsize=12, loc="left")
    a.legend(frameon=False, fontsize=10, loc="upper left")
    bins = np.linspace(C.log_relative_value.quantile(.005), C.log_relative_value.quantile(.995), 40)
    for k in KINDS:
        b.hist(C.loc[C.kind.eq(k), "log_relative_value"], bins=bins, density=True, histtype="step", linewidth=2,
               color=COL[k], label=LABEL[k])
    ticks = np.log([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    b.set_xticks(ticks, ["0.1%", "0.3%", "1%", "3%", "10%", "30%"])
    _style(b, "relative value (% of receiving squad, log scale)", "density")
    b.set_title("Loans are larger relative investments", color=INK, fontsize=12, loc="left")
    b.legend(frameon=False, fontsize=10, loc="upper left")
    return _save(fig, "fig3_relative_value")


def fig_age(A: dict) -> list[str]:
    fig, a = plt.subplots(figsize=(8, 4.6))
    d = A["by_year"]
    for i, k in enumerate(KINDS):
        g = d[d.kind.eq(k)]
        a.errorbar(g.a + (i - 0.5) * 0.2, g["mean"], yerr=1.96 * g["sem"], fmt="o-", color=COL[k], linewidth=2,
                   markersize=6, elinewidth=1.2, label=LABEL[k])
    a.axvline(pta.CUTOFF + 0.5, color=INK2, linewidth=1, linestyle=":")
    a.text(pta.CUTOFF + 0.6, a.get_ylim()[1] * 0.97, f"young ≤ {pta.CUTOFF} | older ≥ {pta.CUTOFF + 1}", color=INK2,
           fontsize=9, va="top")
    _style(a, f"age at the move (years; ≤{pta.AGE_LO} and ≥{pta.AGE_HI} pooled)", "mean minutes share")
    a.set_title("Playing time by age", color=INK, fontsize=12, loc="left")
    a.legend(frameon=False, fontsize=10, loc="lower right")
    return _save(fig, "fig4_age")


def fig_timing(T: dict) -> list[str]:
    fig, a = plt.subplots(figsize=(9, 5), constrained_layout=True)
    d = T["remain_bins"]
    x = np.arange(len(d))
    for i, k in enumerate(KINDS):
        col = "mean share, loans" if k == "loan" else "mean share, permanent"
        a.bar(x + (i - 0.5) * 0.36, d[col], width=0.34, color=COL[k], label=LABEL[k])
    wrap = {"under half": "under half", "half to three-quarters": "half to\nthree-quarters",
            "three-quarters to almost all": "three-quarters\nto almost all",
            "all (arrived before the first league game)": "all (arrived before\nthe first league game)"}
    a.set_xticks(x, [wrap[s] for s in d["season remaining at arrival"]], fontsize=9)
    a.set_ylim(0, max(d["mean share, loans"].max(), d["mean share, permanent"].max()) * 1.25)
    _style(a, "share of the receiving club's league season left at arrival", "mean minutes share")
    a.set_title("Playing time by arrival timing", color=INK, fontsize=12, loc="left")
    a.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    return _save(fig, "fig5_arrival_timing")


def fig_calendar(K: dict) -> list[str]:
    fig, a = plt.subplots(figsize=(10, 5), constrained_layout=True)
    p = K["pathway"]
    p = p.loc[p.sum(axis=1).sort_values().index]
    y = np.arange(len(p))
    for i, st in enumerate(("normal", "off_calendar")):
        yy = y + (0.5 - i) * 0.38                      # normal on top, matching the legend order
        a.barh(yy, 100 * p[st], height=0.36, color=COL_CAL[st],
               label="calendar-normal ending" if st == "normal" else "off-calendar ending")
        for yi, v in zip(yy, p[st]):
            a.text(100 * v + 0.4, yi, f"{100 * v:.0f}%", color=INK2, fontsize=8.5, va="center")
    a.set_yticks(y, p.index, fontsize=9)
    _style(a, "% of loans in that calendar group", "")
    a.grid(axis="x", color=GRID, linewidth=0.8)
    a.grid(axis="y", visible=False)
    a.set_title("What the player did after the loan (main-sample loans)", color=INK, fontsize=12, loc="left")
    a.legend(frameon=False, fontsize=10, loc="lower right")
    return _save(fig, "fig6_loan_pathway_by_calendar_status")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def pp(x, sign=True) -> str:
    v = round(100 * x, 1) + 0.0                      # + 0.0 turns -0.0 into 0.0
    return (f"{v:+.1f} pp" if sign and v != 0 else f"{v:.1f} pp")


def pct(x) -> str:
    return f"{100 * x:.1f}%"


def cs(r: dict, scale=100) -> str:
    """coef (SE) [CI], p, in percentage points."""
    v = round(scale * r["coef"], 2) + 0.0
    c = f"{v:+.2f}" if v != 0 else "0.00"
    return f"{c} ({scale * r['se']:.2f}) [{scale * r['lo']:+.2f}, {scale * r['hi']:+.2f}], p = {r['p']:.3f}"


def render(R: dict) -> str:
    O, B, L, G = R["overview"], R["distribution"], R["ladder"], R["gelbach"]
    A, V, T, K, RB, X = R["age"], R["value"], R["timing"], R["calendar"], R["robustness"], R["examples"]
    Nl, Np = O["N"]["loan"], O["N"]["permanent"]
    lv = list(L.values())
    raw, fe, timing, age, full = lv
    Bk = B.set_index("kind")
    gp = G["point"]
    val = gp["log_relative_value"] + gp["log_squad_value"]
    bal = R["balance"].set_index("characteristic")
    cont = O["continuous"]
    med = lambda lab, k: cont[lab][k]["median"]
    W = ["# Loans vs permanent transfers: what the calendar-first data show", "",
         "Generated by `python -m src.analysis.professor_empirical` from `analysis_episode_master.csv`. Every number "
         "below is computed by that code; the preferred estimates are asserted equal to the lineage pipeline's saved "
         "values. **Nothing here is causal**: loans and permanent transfers are not randomly assigned, and every "
         "coefficient is a conditional association. Outcome: `minutes_share` = the player's league minutes for the "
         "receiving club ÷ (90 × the club's league games in the observation window), 0 to 1.", "",
         f"Sample: the calendar-first main regression sample (`{MAIN}`), {f0(Nl + Np)} episodes = {f0(Nl)} loans + "
         f"{f0(Np)} permanent transfers, {f0(O['players'])} players, {f0(O['receiving_clubs'])} receiving clubs, "
         f"{len(O['leagues'])} leagues, arrival seasons {O['seasons'][0]}/{str(O['seasons'][0] + 1)[2:]}–"
         f"{O['seasons'][1]}/{str(O['seasons'][1] + 1)[2:]}. Standard errors are clustered by receiving club "
         "throughout. Tables: `empirical/*.csv`; figures: `empirical/figures/` (PNG and PDF).", "",
         "## Five facts to show", ""]
    for i, f in enumerate(R["facts"], 1):
        W.append(f"{i}. {f}")
    # ---- A
    W += ["", "## 1. Sample overview", ""]
    W += md_table(["Characteristic", "Group", "Mean", "p10", "p25", "Median", "p75", "p90"],
                  [[lab, LABEL[k], *(f"{cont[lab][k][q]:,.2f}" for q in ("mean", "p10", "p25", "median", "p75", "p90"))]
                   for lab, _, _ in CONT for k in KINDS])
    W += ["", "Position and arrival window (% of each group):", ""]
    cats = ["Goalkeeper", "Defender", "Midfield", "Attack"]
    W += md_table(["", *cats, "summer window (Jun–Sep)", "winter window (Dec–Mar)", "outside windows"],
                  [[LABEL[k], *(pct(O["position"][k].get(c, 0)) for c in cats),
                    *(pct(O["window"][k].get(w, 0)) for w in ("summer", "winter", "outside windows"))] for k in KINDS])
    W += ["", f"Arrival months are bimodal (see `empirical/A3_arrivals_by_month.csv`): the windows above are read off "
          "that distribution, not imposed. By season and by league: `A4_by_season.csv`, `A5_by_league.csv`.", ""]
    # ---- B
    W += ["## 2. Playing-time distribution", "", "![Distribution](empirical/figures/fig1_playing_time_distribution.png)", ""]
    W += md_table(["", "N", "Mean", "Median", "True zero", "< 10%", "< 25%", "< 50%", "< 75%", "≥ 75%"],
                  [[LABEL[k], f0(Bk.loc[k, "N"]), f"{Bk.loc[k, 'mean']:.3f}", f"{Bk.loc[k, 'median']:.3f}",
                    *(pct(Bk.loc[k, c]) for c in ("true zero (0 minutes)", "below 10%", "below 25%", "below 50%",
                                                  "below 75%", "75% or more"))] for k in KINDS])
    W += ["", f"Loans have a higher mean share ({pp(raw['coef'])} raw difference, clustered SE {pp(raw['se'], False)}) "
          f"and fewer complete zeros ({pct(Bk.loc['loan', 'true zero (0 minutes)'])} vs "
          f"{pct(Bk.loc['permanent', 'true zero (0 minutes)'])}). The difference sits in the lower half of the "
          f"distribution: below 25% of the minutes, {pct(Bk.loc['loan', 'below 25%'])} of loans vs "
          f"{pct(Bk.loc['permanent', 'below 25%'])} of permanent arrivals; at 75% or more, "
          f"{pct(Bk.loc['loan', '75% or more'])} vs {pct(Bk.loc['permanent', '75% or more'])}.", ""]
    # ---- 3 ladder
    W += ["## 3. Regression ladder (main sample)", "", "![Ladder](empirical/figures/fig2_ladder_and_decomposition.png)", ""]
    W += md_table(["Model", "Loan coef, pp (SE) [95% CI], p", "N", "R²"],
                  [[k, cs(r), f0(r["N"]), f"{r['r2']:.3f}"] for k, r in L.items()], right={2, 3})
    W += ["", "Read across: the raw loan advantage "
          f"({pp(raw['coef'])}) changes to {pp(fe['coef'])} once loans and permanent moves are compared inside the same "
          f"league-season and position, {pp(timing['coef'])} after arrival timing, {pp(age['coef'])} after age, and "
          f"{pp(full['coef'])} once player and squad value enter (M3). Why each step moves it is in section 4.", ""]
    # ---- 5 diagnosis
    W += ["## 4. Why the value controls erase the raw loan advantage", "",
          "How loans and permanent arrivals differ before any model (means; standardised difference = difference ÷ "
          "pooled SD):", ""]
    W += md_table(["Characteristic", "Loans", "Permanent", "Difference", "Std. difference"],
                  [[i, f"{r.loans:.3f}", f"{r.permanent:.3f}", f"{r.difference:+.3f}",
                    f"{r['standardised difference']:+.2f}"] for i, r in bal.iterrows()])
    lb = R["league_balance"]
    W += ["", f"Medians: player value €{med('player value (EUR m)', 'loan'):.2f}m for loans vs "
          f"€{med('player value (EUR m)', 'permanent'):.2f}m; receiving squad €{med('receiving squad value (EUR m)', 'loan'):.1f}m "
          f"vs €{med('receiving squad value (EUR m)', 'permanent'):.1f}m; relative value "
          f"{med('relative value (% of receiving squad)', 'loan'):.2f}% vs "
          f"{med('relative value (% of receiving squad)', 'permanent'):.2f}% of the squad. League-season mix: a "
          f"dissimilarity index of {lb['dissimilarity_index']:.3f} over {lb['cells']} cells (the share of loans that "
          "would have to move cell to match the permanent distribution).", "",
          f"**Decomposition** (Gelbach 2016; order-invariant): the raw gap of {pp(gp['raw'])} minus the M3 coefficient "
          f"of {pp(gp['full'])} is split exactly into the part each covariate group accounts for. Each part = (how "
          "much loans differ from permanent arrivals in that covariate) × (how that covariate relates to playing time "
          f"in M3). 95% intervals from {G['boot_reps']} receiving-club bootstrap draws.", ""]
    nice = {"log_relative_value": "relative value (log)", "log_squad_value": "squad value (log)"}
    W += md_table(["Covariate group", "Accounts for (pp)", "Bootstrap SE", "95% interval"],
                  [[nice.get(r["term"], r["term"]), f"{100 * r['estimate']:+.2f}", f"{100 * r['boot_se']:.2f}",
                    f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]"] for r in G["rows"] if r["term"] not in ("raw", "full")])
    alt = R["gelbach_alt"]["point"]
    W += ["", f"Because log relative value = log player value − log squad value, the split between the two value terms "
          f"depends on how they are written; together they account for {pp(val)} in both parametrisations "
          f"(player value {pp(alt['log_player_value'])} and squad value {pp(alt['log_squad_value'])} when written "
          "that way).", "", "One control group at a time, added to the FE + position model (order-dependent view):", ""]
    W += md_table(["Added to FE + position", "Loan coef, pp (SE) [95% CI], p"],
                  [[r["added"], cs(r)] for r in R["one_at_a_time"]])
    W += ["", "**Plain English.** " + R["story"], ""]
    # ---- C age
    W += ["## 5. Age", "", "![Age](empirical/figures/fig4_age.png)", ""]
    at = A["table"]
    W += md_table(["Group", "N loans", "N permanent", "Mean share, loans", "Mean share, permanent", "Raw difference (SE)"],
                  [[r.group, f0(r["N loans"]), f0(r["N permanent"]), f"{r['mean share, loans']:.3f}",
                    f"{r['mean share, permanent']:.3f}", f"{pp(r['raw difference'])} ({pp(r['raw SE'], False)})"]
                   for _, r in at.iterrows()])
    am = A["model"]
    W += ["", "Adjusted (one model: loan, older, loan × older, arrival timing, log relative value, log squad value, "
          "position, league-season FE; the specification of the Week 6 age analysis):", ""]
    W += md_table(["Term", "Coef, pp (SE) [95% CI], p"],
                  [[f"loan, young (≤{pta.CUTOFF})", cs(am["young"])], [f"loan, older (≥{pta.CUTOFF + 1})", cs(am["older"])],
                   ["loan × older (difference)", cs(am["interaction"])]])
    W += ["", R["age_text"], ""]
    # ---- D value
    W += ["## 6. Relative value", "", "![Relative value](empirical/figures/fig3_relative_value.png)", ""]
    vt = V["table"]
    W += md_table(["Quintile", "Relative value (median, range)", "N loans / perm.", "Mean share L / P",
                   "Raw diff (SE)", "Adjusted loan coef (SE) [95% CI]"],
                  [[r.bin, f"{r['relative value median']:.2f}% ({r['relative value min (% of squad)']:.2f}–"
                    f"{r['relative value max']:.1f}%)", f"{f0(r['N loans'])} / {f0(r['N permanent'])}",
                    f"{r['mean share, loans']:.3f} / {r['mean share, permanent']:.3f}",
                    f"{pp(r['raw difference'])} ({pp(r['raw SE'], False)})",
                    f"{pp(r['adjusted loan coef'])} ({pp(r['adjusted SE'], False)}) [{pp(r['adjusted lo'])}, "
                    f"{pp(r['adjusted hi'])}]"] for _, r in vt.iterrows()])
    it = V["interaction"]
    W += ["", "Quintiles of log relative value on the main sample; adjusted coefficients from one pooled model with "
          "a loan dummy per quintile, quintile main effects, arrival timing, age, age², log squad value, position and "
          f"league-season FE. Continuous version (M3 with loan × centred log relative value): loan at the mean "
          f"{pp(it['loan']['coef'])} (SE {pp(it['loan']['se'], False)}), interaction {it['loan_x_lrv']['coef']:+.4f} per "
          f"log point (SE {it['loan_x_lrv']['se']:.4f}, p = {it['loan_x_lrv']['p']:.3f}).", "", R["value_text"], ""]
    # ---- E timing
    W += ["## 7. Arrival timing", "", "![Timing](empirical/figures/fig5_arrival_timing.png)", ""]
    W += md_table(["Window", "N loans", "N permanent", "Mean remain", "Mean share L / P", "Raw diff (SE)"],
                  [[r.window, f0(r["N loans"]), f0(r["N permanent"]), f"{r['mean remain']:.2f}",
                    f"{r['mean share, loans']:.3f} / {r['mean share, permanent']:.3f}",
                    f"{pp(r['raw difference'])} ({pp(r['raw SE'], False)})"] for _, r in T["window"].iterrows()])
    W += [""]
    W += md_table(["Season left at arrival", "N loans", "N permanent", "Mean share L / P", "Raw diff (SE)"],
                  [[r["season remaining at arrival"], f0(r["N loans"]), f0(r["N permanent"]),
                    f"{r['mean share, loans']:.3f} / {r['mean share, permanent']:.3f}",
                    f"{pp(r['raw difference'])} ({pp(r['raw SE'], False)})"] for _, r in T["remain_bins"].iterrows()])
    wm, rm = T["winter_model"], T["remain_model"]
    W += ["", f"Adjusted (M3 + winter + loan × winter): loan for summer-window arrivals {cs(wm['loan (summer)'])}; "
          f"loan × winter {cs(wm['loan × winter'])}. Continuous (M3 with loan × centred remain): loan × remain "
          f"{cs(rm['loan × remain'])} per full season.", "", R["timing_text"], ""]
    # ---- F calendar
    W += ["## 8. Off-calendar loans (descriptive)", "", "![Pathway](empirical/figures/fig6_loan_pathway_by_calendar_status.png)",
          "", "Main-sample loans by the calendar status of their ending. **Off-calendar means only that the return "
          "date falls away from the dates on which loans usually end in that country; it is not evidence of early "
          "termination or a contract breach.**", ""]
    kt = K["table"].set_index("status").T
    W += md_table(["", "Calendar-normal", "Off-calendar"],
                  [[i, *(f0(v) if i in ("N",) else (pct(v) if i in ("true zero", "share below 25%",
                                                                       "winter-window arrival", "cross-border loan")
                                                    else f"{v:,.2f}") for v in r)] for i, r in kt.iterrows()])
    W += ["", f"Difference in mean share, off-calendar minus normal: {cs(K['share_diff'])} (raw; clustered SE).", ""]
    pn = K["pathway"]
    W += md_table(["Next step after the loan", "Calendar-normal", "Off-calendar"],
                  [[i, pct(r["normal"]), pct(r["off_calendar"])] for i, r in pn.iterrows()])
    W += ["", R["calendar_text"], ""]
    # ---- robustness
    W += ["## 9. Robustness of the preferred coefficient (M3)", ""]
    W += md_table(["Specification", "N (loans / perm.)", "Raw gap", "Loan coef, pp (SE) [95% CI], p", "Note"],
                  [[r["specification"], f"{f0(r['N'])} ({f0(r['N loans'])} / {f0(r['N permanent'])})", pp(r["raw gap"]),
                    cs(r), r["note"]] for r in RB], right={1, 2, 3})
    W += ["", R["robust_text"], ""]
    # ---- examples
    W += ["## 10. Real examples (selected by rule, not by hand)", "",
          "Each is the highest-valued player meeting the stated rule in the main sample. Fields are those already in "
          "the master; contract terms are not researched.", ""]
    for _, r in X.iterrows():
        z = lambda v, f: "" if pd.isna(v) else f(v)
        W += [f"**{r.case}**: {r.player_name}, {r.from_club_name} → {r.to_club_name} ({r.receiving_league}, "
              f"{r.transfer_date}; {r.kind}, fee label '{r.fee_display_raw}'). Rule: {r['selection rule']} "
              f"({f0(r['eligible episodes'])} eligible).",
              f"Age {int(r.age_years)}, {r.position_clean}; player value €{r.player_value / 1e6:.2f}m, squad €"
              f"{r.squad_value / 1e6:.0f}m (relative value {100 * r.relative_value:.2f}%); arrived with "
              f"{pct(r.remain)} of the season left; {int(r.available_games)} league games in the window, "
              f"{int(r.appearances)} appearances, {int(r.minutes):,} minutes, share {r.minutes_share:.3f}"
              + z(r.confirmed_injury_games, lambda v: f"; {int(v)} game{'' if int(v) == 1 else 's'} lost to confirmed "
                                                       "injury")
              + z(r.terminal_date, lambda v: f"; loan ended {v} ({r.loan_calendar_status.replace('_', '-')}), next: "
                                             f"{r.next_category}")
              + (f"; same-organisation status `{r.same_org_confidence}`"
                 if r.same_org_confidence != "no_youth_or_reserve_side" else "") + ".", ""]
    # ---- data check
    Ls = R["lineup_stats"]
    top = R["lineup_rows"][R["lineup_rows"][MAIN]].iloc[0]
    W += ["## 11. Data check: zero minutes that other tables contradict", "",
          "The outcome counts a player's minutes from the snapshot's `appearances` table and treats a window game "
          "with no row as zero minutes once the club's own rows for that game are complete (≥ 11). Cross-checking "
          "against two other tables of the same snapshot (`game_lineups`: starting line-ups; `game_events`: "
          "substitutions) shows that `appearances` sometimes lacks a player's rows entirely.", "",
          f"- Reliable episodes checked: {f0(Ls['reliable'])}; contradicted (the player started or came on for club B "
          f"in a window game with no appearance row): {f0(Ls['reliable_contradicted'])}, of which "
          f"{f0(Ls['reliable_contradicted_zero'])} are recorded as true zeros.",
          f"- Main sample: {f0(Ls['main_contradicted'])} contradicted episodes ({f0(Ls['main_by_kind']['loan'])} loans, "
          f"{f0(Ls['main_by_kind']['permanent'])} permanent; {f0(Ls['main_started'])} with at least one start, "
          f"{f0(Ls['main_subbed_only'])} with substitute appearances only). {f0(Ls['main_contradicted_zero'])} of them "
          f"are among the {f0(Ls['main_true_zeros'])} true zeros ({pct(Ls['main_contradicted_zero'] / Ls['main_true_zeros'])}"
          f" of all zeros): whole-spell gaps. The other {f0(Ls['main_contradicted_nonzero'])} have minutes but miss "
          f"{f0(Ls['main_missing_games_nonzero'])} substitute or starting appearances between them (against "
          f"{f0(Ls['main_games_nonzero_episodes'])} recorded appearances): small undercounts. By arrival season: "
          + ", ".join(f"{k}/{str(k + 1)[2:]}: {v}" for k, v in Ls["main_by_season"].items()) + ".",
          f"- Highest-valued case: {top.player_name}, {top.from_club_name} → {top.to_club_name} ({top.transfer_date}): "
          f"recorded with {int(top.appearances)} appearances in {int(top.available_games)} league games, but started "
          f"{int(top.started_no_row)} and came on in {int(top.subbed_on_no_row)} more of them according to the "
          "line-ups and substitution events.",
          f"- Effect: true-zero rates without these episodes are {pct(Ls['main_true_zero_by_kind_clean']['loan'])} "
          f"(loans) and {pct(Ls['main_true_zero_by_kind_clean']['permanent'])} (permanent), against "
          f"{pct(Ls['main_true_zero_by_kind']['loan'])} and {pct(Ls['main_true_zero_by_kind']['permanent'])}; the M3 "
          "coefficient without them is in the robustness table (section 9).", "",
          "This is a gap in the upstream appearance data, not in our rule, but our ≥ 11-row completeness test cannot "
          "see it because it checks the club's rows, not the player's. The samples are **not** changed; the master "
          "marks these episodes non-destructively (`pt_lineup_contradicted`, `pt_reliable_v2`), and the full audit by "
          "season, league and club is `appearance_audit.md`. Every "
          "contradicted episode is listed in `empirical/Q1_zero_minutes_contradicted.csv`; the zero-minute examples "
          "in section 10 exclude them.", ""]
    # ---- next
    W += ["## 12. Caveats", "",
          "- Associations, not effects: clubs choose loans and transfers knowing things we do not observe.",
          "- Market values are Transfermarkt's estimates, dated strictly before the move.",
          "- The decomposition attributes the gap to observed covariates only; it cannot say whether value causes "
          "playing time or merely marks players clubs already intend to use.",
          "- Off-calendar is a calendar classification of the return date, not a reason for the return.", "",
          "## 13. Files", ""]
    W += [f"- `{p}`" for p in R["files"]]
    return "\n".join(W) + "\n"


# ---------------------------------------------------------------------------
# Plain-language text (generated from the numbers; the wording follows the signs)
# ---------------------------------------------------------------------------

def texts(R: dict) -> None:
    L, G, O = R["ladder"], R["gelbach"], R["overview"]
    raw, fe, timing, age, full = L.values()
    gp = G["point"]
    val = gp["log_relative_value"] + gp["log_squad_value"]
    others = {g: gp[g] for g in G["groups"] if g not in ("log_relative_value", "log_squad_value")}
    bal = R["balance"].set_index("characteristic")
    cont = O["continuous"]
    med = lambda lab, k: cont[lab][k]["median"]
    value_dominates = abs(val) > max(abs(v) for v in others.values()) and abs(full["coef"]) < abs(raw["coef"]) / 3
    if value_dominates:
        story = (f"Loans initially appear to play {pp(raw['coef'])[1:]} more of the available minutes, but loaned players "
                 f"are systematically larger investments relative to the club that receives them: median relative value "
                 f"{med('relative value (% of receiving squad)', 'loan'):.2f}% vs "
                 f"{med('relative value (% of receiving squad)', 'permanent'):.2f}% of the squad (higher player value, "
                 f"€{med('player value (EUR m)', 'loan'):.2f}m vs €{med('player value (EUR m)', 'permanent'):.2f}m, at "
                 f"smaller squads, €{med('receiving squad value (EUR m)', 'loan'):.0f}m vs "
                 f"€{med('receiving squad value (EUR m)', 'permanent'):.0f}m). Relative value is strongly associated with "
                 f"playing time for both groups, so the value terms account for {pp(val)} of the gap. Once we compare "
                 f"players of similar relative value and destination-squad value, the difference falls to "
                 f"{pp(full['coef'])} (95% CI {pp(full['lo'])} to {pp(full['hi'])}), statistically indistinguishable "
                 "from zero.")
        a = gp["age"]
        if a < 0 and bal.loc["age (years)", "difference"] < 0:
            story += (f" Age works the other way: loaned players are younger (by {abs(bal.loc['age (years)', 'difference']):.1f} "
                      f"years on average) and younger players play less, so holding age fixed widens the loan "
                      f"advantage ({pp(a)}); this is why the coefficient rises at the age step of the ladder before the "
                      "value step removes it.")
        R["story"] = story
    else:
        R["story"] = (f"The raw gap is {pp(raw['coef'])} and M3 gives {pp(full['coef'])}; the largest single contribution "
                      f"is {max(G['groups'], key=lambda g: abs(gp[g]))} ({pp(max(gp[g] for g in G['groups']))}).")
    am = R["age"]["model"]
    sig = lambda r: r["p"] < 0.05
    R["age_text"] = (f"Conditional on value and timing, the loan coefficient is {pp(am['young']['coef'])} for young players "
                     f"({'distinguishable' if sig(am['young']) else 'not distinguishable'} from zero at 5%) and "
                     f"{pp(am['older']['coef'])} for older players "
                     f"({'distinguishable' if sig(am['older']) else 'not distinguishable'} from zero); the difference "
                     f"between them is {pp(am['interaction']['coef'])} (p = {am['interaction']['p']:.3f}). These are "
                     "conditional associations, not effects of loaning a young or an older player.")
    vt = R["value"]["table"]
    it = R["value"]["interaction"]["loan_x_lrv"]
    lo, hi = vt.iloc[0], vt.iloc[-1]
    mono = all(vt[c].is_monotonic_increasing for c in ("mean share, loans", "mean share, permanent"))
    R["value_rise"] = (f"mean share rises {'at every step ' if mono else ''}from the lowest to the highest quintile of "
                       f"relative value for both groups (loans {lo['mean share, loans']:.3f} → "
                       f"{hi['mean share, loans']:.3f}; permanent {lo['mean share, permanent']:.3f} → "
                       f"{hi['mean share, permanent']:.3f})")
    R["value_text"] = (f"Playing time tracks relative value: {R['value_rise']}. Raw loan-minus-permanent "
                       f"differences run from {pp(lo['raw difference'])} in the lowest quintile to "
                       f"{pp(hi['raw difference'])} in the highest; adjusted ones from {pp(lo['adjusted loan coef'])} to "
                       f"{pp(hi['adjusted loan coef'])}. Loans make up {pct(lo['loan share of bin'])} of the lowest "
                       f"quintile and {pct(hi['loan share of bin'])} of the highest. The continuous interaction is "
                       + ("statistically distinguishable from zero" if it["p"] < 0.05 else
                          "not statistically distinguishable from zero")
                       + f" (p = {it['p']:.3f}): "
                       + ("the loan association " + ("rises" if it["coef"] > 0 else "falls")
                          + " with the player's relative value." if it["p"] < 0.05 else
                          "there is no clear evidence that loans behave differently for small versus large relative "
                          "investments once the other controls are held fixed."))
    T = R["timing"]
    wm = T["winter_model"]["loan × winter"]
    ws = T["winter_model"]["loan (summer)"]
    rm = T["remain_model"]["loan × remain"]
    tw = T["window"].set_index("window")
    cmp = lambda k: ("more" if tw.loc["winter", k] > tw.loc["summer", k] else "less")
    R["timing_text"] = (f"Winter-window permanent arrivals play {cmp('mean share, permanent')} than summer ones "
                        f"({tw.loc['winter', 'mean share, permanent']:.3f} vs {tw.loc['summer', 'mean share, permanent']:.3f}),"
                        f" while winter-window loans play {cmp('mean share, loans')} than summer loans "
                        f"({tw.loc['winter', 'mean share, loans']:.3f} vs {tw.loc['summer', 'mean share, loans']:.3f}). "
                        f"So the raw loan advantage is {pp(tw.loc['winter', 'raw difference'])} for winter arrivals and "
                        f"{pp(tw.loc['summer', 'raw difference'])} for summer arrivals. With the M3 controls the loan "
                        f"coefficient for summer arrivals is {pp(ws['coef'])} (p = {ws['p']:.3f}) and the winter "
                        f"interaction {pp(wm['coef'])} (p = {wm['p']:.3f}); the continuous version gives "
                        f"{pp(rm['coef'])} per full season of remaining games (p = {rm['p']:.3f}). "
                        + ("Descriptively, loans compare better with permanent transfers when the player arrives "
                           "mid-season with less of the season left." if wm["coef"] > 0 and wm["p"] < 0.05 and rm["coef"] < 0
                           else "There is no clear timing pattern once controls are held fixed.")
                        + " This is a comparison of arrivals, not an effect of choosing the winter window.")
    K = R["calendar"]
    kt = K["table"].set_index("status")
    d = K["share_diff"]
    R["calendar_text"] = (f"Off-calendar loans ({f0(kt.loc['off_calendar', 'N'])} of "
                          f"{f0(kt['N'].sum())} main-sample loans) have a mean share of "
                          f"{kt.loc['off_calendar', 'mean share']:.3f} against {kt.loc['normal', 'mean share']:.3f}; "
                          f"{pct(kt.loc['off_calendar', 'winter-window arrival'])} of them arrived in the winter window "
                          f"(vs {pct(kt.loc['normal', 'winter-window arrival'])}) and their median duration is "
                          f"{kt.loc['off_calendar', 'median loan duration (days)']:.0f} days (vs "
                          f"{kt.loc['normal', 'median loan duration (days)']:.0f}). The share difference is "
                          + ("distinguishable" if d["p"] < 0.05 else "not distinguishable") + " from zero at 5%. "
                          "This describes which loans end off-calendar; it says nothing about why they ended.")
    RB = R["robustness"]
    by = {r["specification"]: r for r in RB}
    rng = [r["coef"] for r in RB if "NOT preferred" not in r["note"]]
    no = next(r for r in RB if "NOT preferred" in r["note"])
    R["robust_text"] = (f"Across the comparable specifications the loan coefficient stays between {pp(min(rng))} and "
                        f"{pp(max(rng))}, with every 95% interval including zero"
                        + ("" if all(r["lo"] <= 0 <= r["hi"] for r in RB if "NOT preferred" not in r["note"])
                           else " except where noted")
                        + f". The calendar-normal-only sample gives {pp(no['coef'])} (p = {no['p']:.3f}); it drops the "
                        "off-calendar loans, whose ending may itself reflect low playing time, so it is reported as a "
                        "selected sample and not as an estimate to prefer.")
    B = R["distribution"].set_index("kind")
    inj_c, inj_f = by["Injury sample, current share"], by["Injury sample, injury-adjusted (fit) share"]
    acad = by["Calendar-first, without high-confidence same-organisation moves"]
    R["facts"] = [
        f"Raw comparison: loans play {pp(raw['coef'])} more of the available league minutes "
        f"(mean {B.loc['loan', 'mean']:.3f} vs {B.loc['permanent', 'mean']:.3f}) and are less often shut out entirely "
        f"({pct(B.loc['loan', 'true zero (0 minutes)'])} vs {pct(B.loc['permanent', 'true zero (0 minutes)'])} true zeros).",
        f"Loans are bigger relative investments: median relative value "
        f"{med('relative value (% of receiving squad)', 'loan'):.2f}% vs "
        f"{med('relative value (% of receiving squad)', 'permanent'):.2f}% of the receiving squad, and "
        f"{R['value_rise']}.",
        f"After league-season, position, timing, age and value controls (M3) the loan coefficient is {pp(full['coef'])} "
        f"(95% CI {pp(full['lo'])} to {pp(full['hi'])}); the value terms account for {pp(val)} of the "
        f"{pp(raw['coef'])} raw gap.",
        f"Injury adjustment does not change this: {pp(inj_c['coef'])} on the current share and {pp(inj_f['coef'])} on "
        f"the injury-adjusted share of the same {f0(inj_f['N'])} episodes. Removing the high-confidence academy "
        f"promotions gives {pp(acad['coef'])}.",
        f"By age, the adjusted loan coefficient is {pp(am['young']['coef'])} for players ≤{pta.CUTOFF} and "
        f"{pp(am['older']['coef'])} for players ≥{pta.CUTOFF + 1} (difference p = {am['interaction']['p']:.3f}).",
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _csv(df: pd.DataFrame, name: str, files: list[str], index=False) -> None:
    p = EMP / name
    df.to_csv(p, index=index)
    files.append(str(p.relative_to(OUT)))


def compute() -> dict:
    assert dl.g2.sha256(dl.CANONICAL_CSV) == dl.g2.STAGE1C_SHA256
    N = json.loads(dl.NUMBERS_JSON.read_text())
    M = pd.read_csv(dl.MASTER_CSV, low_memory=False)
    C = frame(M, M[MAIN])
    cf = N["calendar_first"]["variants"]["classifiable"]
    assert len(C) == cf["totals"]["main"]["total"] and int(C.loan.sum()) == cf["totals"]["main"]["loan"]
    R = {"overview": overview(C), "distribution": pt_distribution(C), "ladder": ladder(C), "balance": balance(C),
         "league_balance": league_season_balance(C), "one_at_a_time": one_at_a_time(C)}
    full = R["ladder"]["L4 + log relative value, log squad value (M3, preferred)"]
    assert abs(full["coef"] - cf["estimates"]["m3"]["coef"]) < TOL and full["N"] == cf["estimates"]["m3"]["N"]
    assert abs(R["ladder"]["L0 raw difference"]["coef"] - cf["estimates"]["raw_gap_main"]) < TOL
    R["gelbach"] = gelbach(C, ["log_relative_value", "log_squad_value"])
    R["gelbach_alt"] = gelbach(C.assign(log_player_value=C.log_relative_value + C.log_squad_value),
                               ["log_player_value", "log_squad_value"])
    assert abs(R["gelbach"]["point"]["full"] - full["coef"]) < 1e-8
    R["age"] = age_section(C)
    for t in ("young", "older", "interaction"):
        assert abs(R["age"]["model"][t]["coef"] - cf["estimates"]["age"][t]["coef"]) < TOL
    R["value"] = value_section(C)
    R["timing"] = timing_section(C)
    R["calendar"] = calendar_section(C)
    Q = lineup_check(M)
    C["contradicted"] = C.event_id.map(Q.contradicted).fillna(False).astype(bool)
    Rl = M[M.pt_reliable].join(Q, on="event_id")
    assert Rl.contradicted.notna().all()
    bad = Rl[Rl.contradicted]
    main = bad[bad[MAIN]]
    R["lineup_rows"] = bad[["event_id", "player_name", "player_id", "transfer_date", "arrival_season", "kind",
                            "from_club_name", "to_club_name", "receiving_league", "player_value", "available_games",
                            "appearances", "minutes", "true_zero", "window_games", "started_no_row", "subbed_on_no_row",
                            "games_played_without_row", MAIN, "in_cf_revised_sample", INJ]] \
        .sort_values(["player_value", "event_id"], ascending=[False, True])
    R["lineup_stats"] = {
        "reliable": len(Rl), "reliable_contradicted": len(bad),
        "all_contradicted_are_true_zeros": bool(bad.true_zero.astype(bool).all()),
        "main_true_zeros": int(C.zero.sum()), "main_contradicted": len(main),
        "main_contradicted_zero": int(main.true_zero.astype(bool).sum()),
        "main_contradicted_nonzero": int((~main.true_zero.astype(bool)).sum()),
        "main_missing_games_nonzero": int(main.loc[~main.true_zero.astype(bool), "games_played_without_row"].sum()),
        "main_games_nonzero_episodes": int(main.loc[~main.true_zero.astype(bool), "appearances"].sum()),
        "reliable_contradicted_zero": int(bad.true_zero.astype(bool).sum()),
        "main_by_kind": {k: int(main.kind.eq(k).sum()) for k in KINDS},
        "main_started": int(main.started_no_row.gt(0).sum()),
        "main_subbed_only": int((main.started_no_row.eq(0) & main.subbed_on_no_row.gt(0)).sum()),
        "main_by_season": {int(k): int(v) for k, v in main.arrival_season.value_counts().sort_index().items()},
        "main_true_zero_by_kind": {k: float(C.loc[C.kind.eq(k), "zero"].mean()) for k in KINDS},
        "main_true_zero_by_kind_clean": {k: float(C.loc[C.kind.eq(k) & ~C.contradicted, "zero"].mean()) for k in KINDS},
    }
    R["robustness"] = robustness(M, N, Q)
    R["examples"] = examples(C)
    texts(R)
    R["C"] = C
    return R


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    R = compute()
    files: list[str] = []
    O = R["overview"]
    ov = pd.DataFrame([{"characteristic": lab, "group": k, **O["continuous"][lab][k]} for lab, _, _ in CONT for k in KINDS])
    _csv(ov, "A1_sample_overview.csv", files)
    _csv(pd.DataFrame([{"group": k, "kind": "position", "category": c, "share": v} for k in KINDS
                       for c, v in O["position"][k].items()]
                      + [{"group": k, "kind": "window", "category": c, "share": v} for k in KINDS
                         for c, v in O["window"][k].items()]), "A2_position_and_window.csv", files)
    _csv(O["by_month"], "A3_arrivals_by_month.csv", files, index=True)
    _csv(O["by_season"], "A4_by_season.csv", files, index=True)
    _csv(O["by_league"], "A5_by_league.csv", files, index=True)
    _csv(R["distribution"], "B1_playing_time_distribution.csv", files)
    _csv(pd.DataFrame([{"model": k, **v} for k, v in R["ladder"].items()]), "R1_regression_ladder.csv", files)
    _csv(R["balance"], "R2_loan_permanent_balance.csv", files)
    _csv(pd.DataFrame(R["gelbach"]["rows"]), "R3_decomposition.csv", files)
    _csv(pd.DataFrame(R["gelbach_alt"]["rows"]), "R3b_decomposition_player_value_parametrisation.csv", files)
    _csv(pd.DataFrame(R["one_at_a_time"]), "R4_one_control_at_a_time.csv", files)
    _csv(R["age"]["table"], "C1_age_groups.csv", files)
    _csv(pd.DataFrame([{"term": k, **v} for k, v in R["age"]["model"].items() if isinstance(v, dict)]),
         "C2_age_model.csv", files)
    _csv(R["age"]["by_year"], "C3_share_by_age_year.csv", files)
    _csv(R["value"]["table"], "D1_relative_value_quintiles.csv", files)
    _csv(pd.DataFrame([{"term": k, **v} for k, v in R["value"]["interaction"].items() if isinstance(v, dict)]),
         "D2_relative_value_interaction.csv", files)
    _csv(R["value"]["deciles"], "D3_share_by_relative_value_decile.csv", files)
    _csv(R["timing"]["window"], "E1_arrival_window.csv", files)
    _csv(R["timing"]["remain_bins"], "E2_season_remaining.csv", files)
    _csv(pd.DataFrame([{"term": k, **v} for d in (R["timing"]["winter_model"], R["timing"]["remain_model"])
                       for k, v in d.items() if isinstance(v, dict)]), "E3_timing_interactions.csv", files)
    _csv(R["calendar"]["table"], "F1_off_calendar_loans.csv", files)
    _csv(R["calendar"]["pathway_n"], "F2_loan_pathway_by_calendar_status.csv", files, index=True)
    _csv(pd.DataFrame(R["robustness"]), "R5_robustness.csv", files)
    _csv(R["examples"], "G1_examples.csv", files)
    _csv(R["lineup_rows"], "Q1_zero_minutes_contradicted.csv", files)
    files += fig_distribution(R["C"])
    files += fig_ladder(R["ladder"], R["gelbach"])
    files += fig_value(R["value"], R["C"])
    files += fig_age(R["age"])
    files += fig_timing(R["timing"])
    files += fig_calendar(R["calendar"])
    R["files"] = files
    SUMMARY_MD.write_text(render(R))
    js = {"ladder": R["ladder"], "gelbach": R["gelbach"]["rows"], "gelbach_alt": R["gelbach_alt"]["rows"],
          "robustness": R["robustness"], "age_model": {k: v for k, v in R["age"]["model"].items() if isinstance(v, dict)},
          "value_interaction": R["value"]["interaction"], "timing": {"winter": R["timing"]["winter_model"],
                                                                     "remain": R["timing"]["remain_model"]},
          "calendar_share_diff": R["calendar"]["share_diff"], "facts": R["facts"], "story": R["story"],
          "N": R["overview"]["N"], "lineup": R["lineup_stats"],
          "lineup_top": R["lineup_rows"][R["lineup_rows"][MAIN]].iloc[0][["player_name", "from_club_name", "to_club_name",
                                                                         "transfer_date", "available_games",
                                                                         "started_no_row", "subbed_on_no_row"]].to_dict(),
          "files": files}
    NUMBERS_JSON.write_text(json.dumps(js, indent=1, default=float))
    print(f"wrote {SUMMARY_MD.name} and {len(files)} files under {EMP.name}/")
    return R


if __name__ == "__main__":
    main()
