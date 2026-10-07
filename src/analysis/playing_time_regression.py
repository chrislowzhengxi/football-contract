"""Controlled association between loan status and playing time. Observational, not causal.

    python -m src.analysis.playing_time_regression

Starts from the clean primary sample (rebuilt by `playing_time_join`, checked
against the descriptive run), adds pre-move player and club characteristics,
and estimates a sequence of OLS models of the playing-time share. Market
values come only from valuations dated strictly before the move; missing
values stay missing (those episodes leave the controlled models, counted).
No post-move movement variable enters any primary model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import loan_scope
from . import playing_time_join as ptj
from .daniel_scope_final import f0, md_table
from .loan_episodes import ROOT
from .loan_timing_exploration import share

OUT = ROOT / "data" / "outputs" / "rebuild"
SUMMARY_MD = OUT / "playing_time_regression_summary.md"
SAMPLE_CSV = OUT / "playing_time_regression_sample.csv"
MODELS_CSV = OUT / "playing_time_regression_models.csv"
MV_AUDIT_CSV = OUT / "playing_time_market_value_audit.csv"
BALANCE_CSV = OUT / "playing_time_covariate_balance.csv"
ROBUST_CSV = OUT / "playing_time_regression_robustness.csv"
LEAGUE_CSV = OUT / "playing_time_regression_by_league.csv"
NUMBERS_JSON = OUT / "playing_time_regression_numbers.json"
DESCRIPTIVE_JSON = OUT / "playing_time_descriptive_numbers.json"
FIGS = {"models": "figures/playing_time_reg_fig1_loan_coefficient_by_model.png",
        "relative": "figures/playing_time_reg_fig2_relative_value.png",
        "league": "figures/playing_time_reg_fig3_by_league.png"}

LOOKBACK_DAYS = 365         # a valuation older than this does not place a player in a squad
MIN_SQUAD_PLAYERS = 11      # fewer valued players than a starting XI: squad value left missing
STALE_DAYS = 365            # player valuation older than this is "stale" (robustness exclusion)
POSITIONS = ("Goalkeeper", "Defender", "Midfield", "Attack")
COVID = (2019, 2020)
COLOR = {"loan": "#dd6b20", "permanent": "#2b6cb0"}
INK, MUTED, GRID = "#1a202c", "#555", "#e2e8f0"


# ---------------------------------------------------------------------------
# Market values
# ---------------------------------------------------------------------------

def valuations(con) -> pd.DataFrame:
    pv = con.sql("""select player_id, date as vdate, market_value_in_eur as "value", current_club_id as vclub
                    from player_valuations""").df()
    pv["vdate"] = pd.to_datetime(pv.vdate).astype("datetime64[ns]")
    pv["player_id"] = pv.player_id.astype("int64")
    pv["vclub"] = pv.vclub.astype("int64")
    pv["value"] = pv["value"].astype(float)
    return pv.sort_values("vdate").reset_index(drop=True)


def player_value(E: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """Latest valuation strictly before the move date (a valuation on the move day may already reflect it).
    (player, date) is unique in the table, so the choice is deterministic. Also the first valuation on/after."""
    left = E[["event_id", "player_id", "move_date"]].astype({"player_id": "int64", "move_date": "datetime64[ns]"})
    left = left.sort_values("move_date")
    before = pd.merge_asof(left, pv, left_on="move_date", right_on="vdate", by="player_id",
                           direction="backward", allow_exact_matches=False)
    after = pd.merge_asof(left, pv[["player_id", "vdate"]].rename(columns={"vdate": "next_vdate"}),
                          left_on="move_date", right_on="next_vdate", by="player_id", direction="forward")
    out = before.merge(after[["event_id", "next_vdate"]], on="event_id")
    out = out.rename(columns={"value": "player_value", "vdate": "player_value_date", "vclub": "player_value_club_id"})
    out["player_value_gap_days"] = (out.move_date - out.player_value_date).dt.days
    return out[["event_id", "player_value", "player_value_date", "player_value_gap_days", "player_value_club_id",
                "next_vdate"]]


def squad_value(E: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """B's squad value just before the move: the sum of the latest pre-move valuation of every player whose
    latest valuation strictly before the move (and within LOOKBACK_DAYS) lists B as his club, excluding the
    focal player. Left missing when fewer than MIN_SQUAD_PLAYERS players qualify."""
    E = E.astype({"player_id": "int64", "b_club_id": "int64", "move_date": "datetime64[ns]"})
    pairs = E[["b_club_id", "move_date"]].drop_duplicates().reset_index(drop=True)
    pairs["pair"] = np.arange(len(pairs))
    lb = pd.Timedelta(days=LOOKBACK_DAYS)
    cand = pairs.merge(pv[["player_id", "vdate", "vclub"]], left_on="b_club_id", right_on="vclub")
    cand = cand[(cand.vdate < cand.move_date) & (cand.vdate >= cand.move_date - lb)]
    cand = cand[["pair", "b_club_id", "move_date", "player_id"]].drop_duplicates().sort_values("move_date")
    latest = pd.merge_asof(cand, pv, left_on="move_date", right_on="vdate", by="player_id",
                           direction="backward", allow_exact_matches=False)
    squad = latest[(latest.vclub == latest.b_club_id) & (latest.vdate >= latest.move_date - lb)]
    agg = squad.groupby("pair").agg(squad_value_incl=("value", "sum"), squad_players_incl=("player_id", "nunique"),
                                    squad_latest_vdate=("vdate", "max"), squad_oldest_vdate=("vdate", "min"))
    ep = E[["event_id", "player_id", "b_club_id", "move_date"]].merge(pairs, on=["b_club_id", "move_date"])
    ep = ep.join(agg, on="pair")
    focal = squad[["pair", "player_id", "value"]].rename(columns={"value": "focal_in_squad_value"})
    ep = ep.merge(focal, on=["pair", "player_id"], how="left")
    ep["focal_in_squad"] = ep.focal_in_squad_value.notna()
    ep["squad_value"] = ep.squad_value_incl - ep.focal_in_squad_value.fillna(0)
    ep["squad_players"] = ep.squad_players_incl - ep.focal_in_squad.astype(int)
    ok = ep.squad_players >= MIN_SQUAD_PLAYERS
    for c in ("squad_value", "squad_players"):
        ep[c] = ep[c].where(ok)
    ep["squad_value_gap_days"] = (ep.move_date - ep.squad_latest_vdate).dt.days.where(ok)
    return ep[["event_id", "squad_value", "squad_players", "focal_in_squad", "squad_value_gap_days"]]


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

def build(E: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """Base sample (primary without the arrival rule) with every pre-move covariate."""
    B = E[E.reliable & ~E.prior_spell_at_b & ((E.kind == "permanent") | (E.loan_ending == "return to lender"))].copy()
    n0 = len(B)
    B = B.merge(player_value(B, pv), on="event_id", how="left", validate="one_to_one")
    B = B.merge(squad_value(B, pv), on="event_id", how="left", validate="one_to_one")
    assert len(B) == n0 and B.event_id.is_unique, "valuation joins duplicated episodes"
    B["loan"] = (B.kind == "loan").astype(int)
    B["remain"] = B.games_remaining_at_arrival / B.season_games
    B["arrived_before_first_game"] = B.games_remaining_at_arrival == B.season_games
    dob = pd.to_datetime(B.date_of_birth, errors="coerce")
    B["age"] = (B.move_date - dob).dt.days / 365.25
    B["position_clean"] = B.position.where(B.position.isin(POSITIONS))
    pos_ok = B.player_value > 0
    B["log_player_value"] = np.log(B.player_value.where(pos_ok))
    B["log_squad_value"] = np.log(B.squad_value.where(B.squad_value > 0))
    B["relative_value"] = B.player_value / B.squad_value
    B["log_relative_value"] = B.log_player_value - B.log_squad_value
    B["league_season"] = B.receiving_league + "_" + B.arrival_season.astype(int).astype(str)
    B["club_season"] = B.b_club_id.astype(int).astype(str) + "_" + B.arrival_season.astype(int).astype(str)
    B["stale_player_value"] = B.player_value_gap_days > STALE_DAYS
    B["zero"] = (B.minutes == 0).astype(int)
    B["share"] = B.minutes_share
    return B


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def fit(df: pd.DataFrame, x: list[str], fe: str | None = None, absorb: str | None = None,
        y: str = "share", cluster: str | tuple = "b_club_id", cats: list[str] | None = None) -> dict:
    """OLS of y on x. `fe`: fixed effects as dummies (exact degrees of freedom). `absorb`: fixed effects by
    within-demeaning (for many groups nested in the clusters); singleton groups are dropped. `cats`: categorical
    controls entered as dummies. Cluster-robust covariance by `cluster` (one name, or a pair for two-way)."""
    d = df.copy()
    X = d[x].astype(float)
    for c in (cats or []):
        X = X.join(pd.get_dummies(d[c], prefix=c, drop_first=True, dtype=float))
    if fe:
        X = X.join(pd.get_dummies(d[fe], prefix="fe", drop_first=True, dtype=float))
    Y = d[y].astype(float)
    info = {"N": len(d)}
    if absorb:
        size = d.groupby(absorb)[absorb].transform("size")
        keep = size > 1
        d, X, Y = d[keep], X[keep], Y[keep]
        g = d[absorb]
        X = X - X.groupby(g).transform("mean")
        Y = Y - Y.groupby(g).transform("mean")
        X = X.loc[:, X.abs().sum() > 1e-9]
        info.update({"N": len(d), "absorbed_groups": int(g.nunique()), "singletons_dropped": int((~keep).sum())})
    else:
        X = sm.add_constant(X, has_constant="add")
    if isinstance(cluster, tuple):
        groups = np.column_stack([pd.factorize(d[c])[0] for c in cluster])
    else:
        groups = pd.factorize(d[cluster])[0]
    res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    info.update({"res": res, "clusters": int(len(np.unique(groups if groups.ndim == 1 else groups[:, 1])))})
    return info


def coef(m: dict, name: str = "loan") -> dict:
    r = m["res"]
    b, se = float(r.params[name]), float(r.bse[name])
    return {"coef": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": float(r.pvalues[name]),
            "N": m["N"], "r2": float(r.rsquared)}


def vif(df: pd.DataFrame, cols: list[str]) -> dict:
    X = sm.add_constant(df[cols].astype(float))
    out = {}
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        others = X.drop(columns=c)
        r2 = sm.OLS(X[c], others).fit().rsquared
        out[c] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    return out


def ame_logit(res, X: pd.DataFrame, name: str = "loan") -> dict:
    """Average marginal effect of a 0/1 regressor in a logit-link model, delta-method SE."""
    b, V = res.params.to_numpy(), res.cov_params().to_numpy()
    X1, X0 = X.copy(), X.copy()
    X1[name], X0[name] = 1.0, 0.0
    X1, X0 = X1.to_numpy(), X0.to_numpy()
    lam = lambda z: 1 / (1 + np.exp(-z))
    p1, p0 = lam(X1 @ b), lam(X0 @ b)
    ame = float((p1 - p0).mean())
    grad = ((p1 * (1 - p1))[:, None] * X1 - (p0 * (1 - p0))[:, None] * X0).mean(axis=0)
    se = float(np.sqrt(grad @ V @ grad))
    return {"coef": ame, "se": se, "lo": ame - 1.96 * se, "hi": ame + 1.96 * se, "N": len(X)}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

M1 = ["loan", "remain"]
M2 = M1 + ["age_c", "age_c2"]
M3 = M2 + ["log_relative_value", "log_squad_value"]
M4 = M2 + ["log_player_value"]


def compute() -> tuple[dict, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    desc = json.loads(DESCRIPTIVE_JSON.read_text())
    u = loan_scope.universe()
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        J, E, _ = ptj.compute(u, con)
        pv = valuations(con)
        lg_ids = ", ".join(f"'{x}'" for x in J["covered_leagues"])
        club_seasons = con.sql(f"""select distinct home_club_id club, cast(season as int) season from games
                                   where competition_type = 'domestic_league' and competition_id in ({lg_ids})
                                   union select distinct away_club_id, cast(season as int) from games
                                   where competition_type = 'domestic_league' and competition_id in ({lg_ids})""").df()
        league_apps = con.sql("""select a.player_id, a.date from appearances a
                                 join games g on g.game_id = cast(a.game_id as varchar)
                                 where g.competition_type = 'domestic_league'""").df()
    finally:
        con.close()
    B = build(E, pv)
    P = B[B.event_id.isin(E.loc[E.primary_sample, "event_id"])].copy()
    N: dict = {"primary": P.kind.value_counts().to_dict(), "base": B.kind.value_counts().to_dict()}
    assert N["primary"] == {k: int(v) for k, v in desc["primary"].items()}, "primary sample does not reproduce"
    assert P.share.between(0, 1).all()
    assert (P.player_value_date < P.move_date).all() | P.player_value_date.isna().any()
    assert not (P.player_value_date >= P.move_date).any(), "a valuation on or after the move entered"

    # ---- audit: market values
    def cov(g):
        return {"N": len(g), "player_value": int(g.player_value.notna().sum()),
                "squad_value": int(g.squad_value.notna().sum()),
                "both": int((g.player_value.notna() & g.squad_value.notna()).sum()),
                "only_post_move_valuation": int((g.player_value.isna() & g.next_vdate.notna()).sum()),
                "no_valuation_at_all": int((g.player_value.isna() & g.next_vdate.isna()).sum()),
                "gap_median": float(g.player_value_gap_days.median()), "gap_p75": float(g.player_value_gap_days.quantile(.75)),
                "gap_p90": float(g.player_value_gap_days.quantile(.9)), "gap_p95": float(g.player_value_gap_days.quantile(.95)),
                "stale": int(g.stale_player_value.sum()),
                "squad_gap_median": float(g.squad_value_gap_days.median()),
                "squad_players_median": float(g.squad_players.median())}
    audit = []
    for dim, col in (("all", None), ("league", "receiving_league"), ("arrival_season", "arrival_season")):
        for k, g in P.groupby("kind"):
            if col is None:
                audit.append({"kind": k, "dimension": dim, "value": "all", **cov(g)})
            else:
                for v, h in g.groupby(col):
                    audit.append({"kind": k, "dimension": dim, "value": v, **cov(h)})
    A = pd.DataFrame(audit)
    N["mv_coverage"] = {r.kind: {c: getattr(r, c) for c in A.columns if c not in ("kind", "dimension", "value")}
                        for r in A[A.dimension == "all"].itertuples()}
    gaps = P.player_value_gap_days.dropna()
    N["gap_hist"] = {lab: int(gaps.between(lo, hi).sum()) for lab, lo, hi in
                     (("1–30 days", 1, 30), ("31–90", 31, 90), ("91–180", 91, 180), ("181–365", 181, 365),
                      ("366–730", 366, 730), ("over 730", 731, 10 ** 6))}
    s1c = P[P.player_value.notna() & P.market_value_eur.notna()]
    N["stage1c_vs_rule"] = {"compared": len(s1c), "equal": int((s1c.market_value_eur == s1c.player_value).sum())}
    N["focal_in_squad"] = int(P.focal_in_squad.sum())

    # ---- covariates
    age_mean = float(P.age.mean())
    for F in (B, P):
        F["age_c"] = F.age - age_mean
        F["age_c2"] = F.age_c ** 2
    N["age_center"] = age_mean
    N["missing"] = {"age": int(P.age.isna().sum()), "position": int(P.position_clean.isna().sum()),
                    "player_value": int(P.player_value.isna().sum()), "squad_value": int(P.squad_value.isna().sum()),
                    "nonpositive_value": int((P.player_value <= 0).sum())}
    need = ["share", "loan", "remain", "age_c", "position_clean", "log_relative_value", "log_squad_value"]
    C = P.dropna(subset=need).copy()
    N["common"] = C.kind.value_counts().to_dict()
    N["common_exclusions"] = {
        "missing position": int(P.position_clean.isna().sum()),
        "no player valuation before the move": int(P.player_value.isna().sum()),
        f"squad value missing (fewer than {MIN_SQUAD_PLAYERS} valued players)": int(P.squad_value.isna().sum()),
        "any of these (episodes dropped)": int(len(P) - len(C))}

    X_ = P[~P.event_id.isin(C.event_id)]
    N["excluded_profile"] = {k: {"N": len(g), "share_mean": float(g.share.mean()), "zero": float(g.zero.mean()),
                                 "age_mean": float(g.age.mean()), "no_player_value": int(g.player_value.isna().sum())}
                             for k, g in X_.groupby("kind")}
    N["kept_profile"] = {k: {"N": len(g), "share_mean": float(g.share.mean()), "zero": float(g.zero.mean())}
                         for k, g in C.groupby("kind")}
    ls = C.groupby("league_season").loan.agg(["size", "sum"])
    N["league_seasons_with_both"] = int(((ls["sum"] > 0) & (ls["sum"] < ls["size"])).sum())
    N["raw_positive"] = {}
    for lab, d in (("primary", P), ("common", C)):
        a = d[d.share > 0]
        N["raw_positive"][lab] = {"all": float(d.loc[d.loan == 1, "share"].mean() - d.loc[d.loan == 0, "share"].mean()),
                                  "positive_only": float(a.loc[a.loan == 1, "share"].mean() - a.loc[a.loan == 0, "share"].mean())}
    N["league_seasons"] = len(ls)

    # ---- prior playing time: coverage audit only (not used in any model)
    cs_set = set(zip(club_seasons.club.astype("int64"), club_seasons.season.astype(int)))
    league_apps["date"] = pd.to_datetime(league_apps.date)
    by_player = {p: np.sort(g.date.values) for p, g in league_apps.groupby("player_id")}
    origin_cov = pd.Series([(int(a), int(s) - 1) in cs_set for a, s in zip(C.a_club_id, C.arrival_season)], index=C.index)
    def played_before(p, t):
        ds = by_player.get(p)
        if ds is None:
            return False
        lo, hi = np.searchsorted(ds, np.datetime64(t - pd.Timedelta(days=365))), np.searchsorted(ds, np.datetime64(t))
        return hi > lo
    prior_app = pd.Series([played_before(p, t) for p, t in zip(C.player_id, C.move_date)], index=C.index)
    N["prior_audit"] = {k: {"N": int((C.kind == k).sum()),
                            "origin_club_covered_previous_season": int(origin_cov[C.kind == k].sum()),
                            "covered_league_appearance_in_previous_365_days": int(prior_app[C.kind == k].sum())}
                        for k in ("loan", "permanent")}

    # ---- balance
    bal = []
    for k, g in P.groupby("kind"):
        rec = {"kind": k, "N": len(g), "age_mean": g.age.mean(), "age_median": g.age.median(),
               "player_value_median_eur": g.player_value.median(), "player_value_mean_eur": g.player_value.mean(),
               "squad_value_median_eur": g.squad_value.median(), "relative_value_median": g.relative_value.median(),
               "log_relative_value_mean": g.log_relative_value.mean(), "remain_mean": g.remain.mean(),
               "arrived_before_first_game": g.arrived_before_first_game.mean(),
               "share_mean": g.share.mean(), "share_median": g.share.median(), "zero_share": g.zero.mean()}
        for p in POSITIONS:
            rec[f"position_{p}"] = (g.position_clean == p).mean()
        rec["position_missing"] = g.position_clean.isna().mean()
        bal.append(rec)
    BAL = pd.DataFrame(bal)
    N["balance"] = BAL.to_dict("records")
    sd = {}
    for c in ("age", "log_player_value", "log_squad_value", "log_relative_value", "remain"):
        a, b = P.loc[P.loan == 1, c].dropna(), P.loc[P.loan == 0, c].dropna()
        sd[c] = float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
    N["std_diff"] = sd

    # ---- collinearity
    cols = ["log_player_value", "log_squad_value", "log_relative_value", "age", "remain"]
    N["corr"] = C[cols].corr().round(3).to_dict()
    N["vif_preferred"] = vif(C, ["remain", "age_c", "age_c2", "log_relative_value", "log_squad_value"])
    N["vif_all_three"] = vif(C.assign(noise=0), ["log_player_value", "log_squad_value", "log_relative_value"]) \
        if False else {"note": "log_relative_value = log_player_value - log_squad_value exactly, so the three are "
                               "perfectly collinear and cannot enter together"}

    # ---- model sequence
    models = {}
    models["M0 full"] = fit(P, ["loan"])
    models["M0"] = fit(C, ["loan"])
    models["M1"] = fit(C, M1, fe="league_season")
    models["M2"] = fit(C, M2, fe="league_season", cats=["position_clean"])
    models["M3"] = fit(C, M3, fe="league_season", cats=["position_clean"])
    models["M3 alt"] = fit(C, M2 + ["log_player_value", "log_squad_value"], fe="league_season", cats=["position_clean"])
    models["M4"] = fit(C, M4, absorb="club_season", cats=["position_clean"])
    models["M1 full"] = fit(P, M1, fe="league_season")
    models["M2 full"] = fit(P.dropna(subset=["age_c", "position_clean"]), M2, fe="league_season", cats=["position_clean"])
    rows = []
    for name, m in models.items():
        r = m["res"]
        for v in r.params.index:
            if v.startswith("fe_") or v == "const" or v.startswith("position_clean_"):
                continue
            rows.append({"model": name, "variable": v, "coef": r.params[v], "se": r.bse[v],
                         "lo": r.params[v] - 1.96 * r.bse[v], "hi": r.params[v] + 1.96 * r.bse[v],
                         "p": r.pvalues[v], "N": m["N"], "r2": r.rsquared, "clusters": m["clusters"],
                         "absorbed_groups": m.get("absorbed_groups"), "singletons_dropped": m.get("singletons_dropped")})
    MOD = pd.DataFrame(rows)
    N["models"] = {k: coef(m) for k, m in models.items()}
    raw = desc["diff"]["mean_diff"]
    N["raw_descriptive"] = raw
    assert abs(N["models"]["M0 full"]["coef"] - raw) < 1e-9, "Model 0 does not reproduce the raw difference"
    m3 = models["M3"]["res"]
    N["m3_controls"] = {v: {"coef": float(m3.params[v]), "se": float(m3.bse[v])}
                        for v in ("remain", "age_c", "age_c2", "log_relative_value", "log_squad_value")}
    m3a = models["M3 alt"]["res"]
    N["m3_alt_controls"] = {v: {"coef": float(m3a.params[v]), "se": float(m3a.bse[v])}
                            for v in ("log_player_value", "log_squad_value")}

    # club-season FE variation
    cs = C.groupby("club_season").loan.agg(["size", "sum"])
    mixed = cs[(cs["sum"] > 0) & (cs["sum"] < cs["size"])]
    N["club_season"] = {"groups": len(cs), "singletons": int((cs["size"] == 1).sum()), "with_both": len(mixed),
                        "obs_in_mixed": int(mixed["size"].sum()), "loans_in_mixed": int(mixed["sum"].sum()),
                        "N_used": models["M4"]["N"]}
    assert len(mixed) > 0

    # ---- dependence
    N["dependence"] = {"episodes": len(C), "players": int(C.player_id.nunique()),
                       "players_multiple": int((C.player_id.value_counts() > 1).sum()),
                       "episodes_of_repeat_players": int(C.player_id.map(C.player_id.value_counts()).gt(1).sum()),
                       "clubs": int(C.b_club_id.nunique()), "club_seasons": int(C.club_season.nunique()),
                       "league_seasons": int(C.league_season.nunique())}
    se_variants = {"cluster: receiving club (primary)": models["M3"],
                   "cluster: player": fit(C, M3, fe="league_season", cats=["position_clean"], cluster="player_id"),
                   "two-way cluster: player and receiving club": fit(C, M3, fe="league_season", cats=["position_clean"],
                                                                     cluster=("player_id", "b_club_id")),
                   "cluster: club-season": fit(C, M3, fe="league_season", cats=["position_clean"], cluster="club_season")}
    N["se_variants"] = {k: coef(m) for k, m in se_variants.items()}

    # ---- interaction with relative value
    C["lrv_c"] = C.log_relative_value - C.log_relative_value.mean()
    C["loan_x_lrv"] = C.loan * C.lrv_c
    mi = fit(C, ["loan", "remain", "age_c", "age_c2", "lrv_c", "loan_x_lrv", "log_squad_value"],
             fe="league_season", cats=["position_clean"])
    ri = mi["res"]
    N["interaction"] = {v: {"coef": float(ri.params[v]), "se": float(ri.bse[v]), "p": float(ri.pvalues[v])}
                        for v in ("loan", "lrv_c", "loan_x_lrv")}
    N["lrv_mean"] = float(C.log_relative_value.mean())
    lo_q, hi_q = C.log_relative_value.quantile([.05, .95])
    grid = np.linspace(lo_q, hi_q, 40)
    V = ri.cov_params().loc[["loan", "loan_x_lrv"], ["loan", "loan_x_lrv"]].to_numpy()
    fitted = ri.fittedvalues
    pred = []
    for gv in grid:
        gc = gv - N["lrv_mean"]
        base = fitted - ri.params["loan"] * C.loan - ri.params["lrv_c"] * C.lrv_c - ri.params["loan_x_lrv"] * C.loan_x_lrv
        p0 = float((base + ri.params["lrv_c"] * gc).mean())
        p1 = float((base + ri.params["lrv_c"] * gc + ri.params["loan"] + ri.params["loan_x_lrv"] * gc).mean())
        dvec = np.array([1, gc])
        d, dse = ri.params["loan"] + ri.params["loan_x_lrv"] * gc, float(np.sqrt(dvec @ V @ dvec))
        pred.append({"log_relative_value": gv, "relative_value": float(np.exp(gv)), "pred_loan": p1, "pred_perm": p0,
                     "diff": float(d), "diff_lo": float(d - 1.96 * dse), "diff_hi": float(d + 1.96 * dse)})
    N["pred"] = pred
    bl, bi = ri.params["loan"], ri.params["loan_x_lrv"]
    N["crossover_relative_value"] = float(np.exp(N["lrv_mean"] - bl / bi)) if bi != 0 else None
    N["doubling_effect"] = float(models["M3"]["res"].params["log_relative_value"] * np.log(2))
    C["lrv_q"] = pd.qcut(C.log_relative_value, 5, labels=[f"Q{i}" for i in range(1, 6)])
    qd = pd.get_dummies(C.lrv_q, dtype=float)
    for q in qd.columns:
        C[f"loan_{q}"] = C.loan * qd[q]
    mq = fit(C, [f"loan_{q}" for q in qd.columns] + ["remain", "age_c", "age_c2", "log_squad_value"],
             fe="league_season", cats=["position_clean", "lrv_q"])
    N["quintiles"] = [{"q": q, "rel_median": float(C.loc[C.lrv_q == q, "relative_value"].median()),
                       "n_loan": int(((C.lrv_q == q) & (C.loan == 1)).sum()), "n_perm": int(((C.lrv_q == q) & (C.loan == 0)).sum()),
                       **{k: v for k, v in coef(mq, f"loan_{q}").items() if k in ("coef", "se", "lo", "hi")}}
                      for q in qd.columns]

    # ---- zeros and bounded outcome
    z = {}
    z["OLS excluding exact zeros"] = coef(fit(C[C.share > 0], M3, fe="league_season", cats=["position_clean"]))
    z["LPM: P(zero minutes)"] = coef(fit(C, M3, fe="league_season", cats=["position_clean"], y="zero"))
    Xg = C[M3].astype(float).join(pd.get_dummies(C.position_clean, prefix="pos", drop_first=True, dtype=float))
    Xg = Xg.join(pd.get_dummies(C.receiving_league, prefix="lg", drop_first=True, dtype=float))
    Xg = Xg.join(pd.get_dummies(C.arrival_season.astype(int), prefix="s", drop_first=True, dtype=float))
    Xg = sm.add_constant(Xg)
    grp = pd.factorize(C.b_club_id)[0]
    try:
        lg = sm.Logit(C.zero.astype(float), Xg).fit(disp=0, cov_type="cluster", cov_kwds={"groups": grp}, maxiter=200)
        z["Logit: P(zero minutes), average marginal effect"] = ame_logit(lg, Xg)
    except Exception as e:  # noqa: BLE001
        z["Logit: P(zero minutes), average marginal effect"] = {"error": str(e)}
    try:
        fl = sm.GLM(C.share.astype(float), Xg, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": grp})
        z["Fractional logit (share), average marginal effect"] = ame_logit(fl, Xg)
    except Exception as e:  # noqa: BLE001
        z["Fractional logit (share), average marginal effect"] = {"error": str(e)}
    N["zeros"] = z
    N["zero_rate"] = {k: float(g.zero.mean()) for k, g in C.groupby("kind")}

    # ---- sample / timing robustness (preferred specification M3)
    Bc = B.dropna(subset=need).copy()
    rob = {"primary arrival rule (at least two-thirds left)": C,
           "at least half of the season left": Bc[Bc.remain >= 0.5],
           "arrived before B's first league game": Bc[Bc.arrived_before_first_game],
           "excluding 2019/20 and 2020/21 arrivals": C[~C.arrival_season.isin(COVID)],
           f"excluding stale player values (older than {STALE_DAYS} days)": C[~C.stale_player_value],
           "excluding goalkeepers": C[C.position_clean != "Goalkeeper"]}
    RB = []
    for name, frame in rob.items():
        c = coef(fit(frame, M3, fe="league_season", cats=["position_clean"]))
        RB.append({"variant": name, "loans": int(frame.loan.sum()), "permanents": int((1 - frame.loan).sum()), **c})
    for name, c in N["se_variants"].items():
        RB.append({"variant": f"standard errors: {name}", "loans": int(C.loan.sum()), "permanents": int((1 - C.loan).sum()), **c})
    RB = pd.DataFrame(RB)
    N["robustness"] = RB.to_dict("records")

    # ---- league heterogeneity (common framework)
    for lgname in sorted(C.receiving_league.unique()):
        C[f"loan_{lgname}"] = C.loan * (C.receiving_league == lgname)
    lnames = [f"loan_{x}" for x in sorted(C.receiving_league.unique())]
    ml = fit(C, lnames + M3[1:], fe="league_season", cats=["position_clean"])
    LG = []
    for v in lnames:
        lgname = v[5:]
        c = coef(ml, v)
        g = C[C.receiving_league == lgname]
        LG.append({"league": lgname, "loans": int(g.loan.sum()), "permanents": int((1 - g.loan).sum()),
                   "raw_diff": float(g.loc[g.loan == 1, "share"].mean() - g.loc[g.loan == 0, "share"].mean()),
                   **{k: c[k] for k in ("coef", "se", "lo", "hi", "p")}, "small": int(g.loan.sum()) < 100})
    LGd = pd.DataFrame(LG)
    R = ml["res"]
    idx = [list(R.params.index).index(v) for v in lnames]
    Rm = np.zeros((len(lnames) - 1, len(R.params)))
    for i in range(len(lnames) - 1):
        Rm[i, idx[i]], Rm[i, idx[-1]] = 1, -1
    wt = R.wald_test(Rm, scalar=True)
    N["league_equality_p"] = float(wt.pvalue)
    N["league"] = LGd.to_dict("records")

    C["predicted_share_m3"] = models["M3"]["res"].fittedvalues
    tables = {"sample": B.assign(primary_sample=B.event_id.isin(P.event_id), in_common_sample=B.event_id.isin(C.event_id)),
              "models": MOD, "audit": A, "balance": BAL, "robust": RB, "league": LGd, "C": C}
    return N, tables


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def figures(N: dict) -> None:
    plt = _plt()
    (OUT / "figures").mkdir(exist_ok=True)
    M = N["models"]

    # 1. loan coefficient by model
    seq = [("M0 full", "Raw (full sample)"), ("M0", "Raw (market-value sample)"),
           ("M1", "+ league × season FE, arrival timing"), ("M2", "+ age, age², position"),
           ("M3", "+ log relative value, log squad value  [preferred]"), ("M4", "Club × season FE (instead of league × season)")]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ys = np.arange(len(seq))[::-1]
    for y, (k, lab) in zip(ys, seq):
        c = M[k]
        col = "#2b6cb0" if k == "M3" else "#4a5568"
        ax.errorbar(100 * c["coef"], y, xerr=[[100 * (c["coef"] - c["lo"])], [100 * (c["hi"] - c["coef"])]], fmt="o",
                    color=col, ms=6, capsize=3, lw=1.6)
        ax.text(100 * c["hi"] + 0.25, y, f"{100 * c['coef']:+.1f} pp  (N = {c['N']:,})", va="center", fontsize=8.5,
                color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in seq], fontsize=9)
    ax.set_xlim(-5, 9.5)
    ax.set_xlabel("Loan − permanent difference (pp of available league minutes, 95% CI)")
    ax.set_title("Loan − permanent playing-time difference by model (association, not causal)", loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / FIGS["models"], dpi=160)
    plt.close(fig)

    # 2. relative value
    pr = pd.DataFrame(N["pred"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    x = pr.relative_value * 100
    a1.plot(x, pr.pred_loan, color=COLOR["loan"], lw=2.2, label="Loan")
    a1.plot(x, pr.pred_perm, color=COLOR["permanent"], lw=2.2, label="Permanent")
    a1.set_ylabel("Predicted share of\navailable league minutes")
    a1.set_title("Adjusted playing time by relative player value (association, not causal)", loc="left")
    a1.legend(frameon=False, loc="upper left")
    a1.yaxis.grid(True, color=GRID, lw=0.8)
    a2.fill_between(x, 100 * pr.diff_lo, 100 * pr.diff_hi, color="#cbd5e0", alpha=0.7, lw=0)
    a2.plot(x, 100 * pr["diff"], color=INK, lw=2, label="Loan − permanent (interaction model, 95% CI)")
    q = pd.DataFrame(N["quintiles"])
    a2.errorbar(q.rel_median * 100, 100 * q.coef, yerr=[100 * (q.coef - q.lo), 100 * (q.hi - q.coef)], fmt="s",
                color="#805ad5", ms=5, capsize=3, label="By relative-value quintile (95% CI)")
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("Loan − permanent (pp)")
    a2.set_xscale("log")
    ticks = [0.3, 0.5, 1, 2, 5, 10, 20]
    a2.set_xticks([t for t in ticks if x.min() * 0.9 <= t <= x.max() * 1.1])
    a2.set_xticklabels([f"{t:g}%" for t in ticks if x.min() * 0.9 <= t <= x.max() * 1.1])
    a2.set_xlabel("Player market value as % of the receiving club's squad value (log scale; 5th–95th percentile)")
    a2.legend(frameon=False, loc="upper right", fontsize=8.5)
    a2.yaxis.grid(True, color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / FIGS["relative"], dpi=160)
    plt.close(fig)

    # 3. by league
    L = pd.DataFrame(N["league"]).sort_values("coef")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ys = np.arange(len(L))
    for y, r in zip(ys, L.itertuples()):
        col = "#c53030" if r.hi < 0 else ("#2f855a" if r.lo > 0 else "#4a5568")
        ax.errorbar(100 * r.coef, y, xerr=[[100 * (r.coef - r.lo)], [100 * (r.hi - r.coef)]], fmt="o",
                    color=col, mfc="white" if r.small else col, ms=6, capsize=3, lw=1.5)
    pref = 100 * N["models"]["M3"]["coef"]
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(pref, color="#2b6cb0", lw=1, ls="--")
    ax.text(pref + 0.3, len(L) - 0.4, f"pooled {pref:+.1f} pp", color="#2b6cb0", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.league}  (loans {r.loans} / perm. {r.permanents})" + ("  *" if r.small else "")
                        for r in L.itertuples()], fontsize=9)
    ax.set_xlabel("Adjusted loan − permanent difference (pp, 95% CI), preferred specification")
    ax.set_title("Adjusted loan association by receiving league (not causal)", loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.01, "* hollow: fewer than 100 loans. Red: interval entirely below zero; green: entirely above.",
             fontsize=8, color=MUTED)
    fig.savefig(OUT / FIGS["league"], dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pp(x, signed=True) -> str:
    return f"{100 * x:+.1f} pp" if signed else f"{100 * x:.1f} pp"


def ci(c) -> str:
    return f"[{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]"


def render(N: dict) -> str:
    W: list[str] = []
    w = W.append
    M = N["models"]
    m0f, m1f, m2f = M["M0 full"], M["M1 full"], M["M2 full"]
    m0, m1, m2, m3, m4 = (M[k] for k in ("M0", "M1", "M2", "M3", "M4"))
    it = N["interaction"]
    q = N["quintiles"]
    z = N["zeros"]
    R = pd.DataFrame(N["robustness"])
    samp = R[~R.variant.str.startswith("standard errors")]
    L = pd.DataFrame(N["league"])
    eng = L[L.league == "GB1"].iloc[0]
    neg = L[L.hi < 0].sort_values("coef")
    pos = L[L.lo > 0]
    cs = N["club_season"]
    ex, kp = N["excluded_profile"], N["kept_profile"]
    covd = N["mv_coverage"]
    inc0 = lambda c: c["lo"] <= 0 <= c["hi"]

    w("# Playing time after a loan vs a permanent transfer: controlled association")
    w("")
    w("*Generated by `python -m src.analysis.playing_time_regression`. Observational, not causal: loan status is not "
      "randomly assigned. The \"adjusted\" numbers compare loans with permanent moves that look alike on the observed "
      "characteristics; differences in what clubs and players know but the data do not record remain. Outcome: the "
      "share of the receiving club's available league minutes the player played in the window. Coefficients are in "
      "percentage points of available minutes; intervals are 95% and clustered by receiving club.*")
    w("")
    w("## Answers")
    w("")
    w(f"1. **Raw difference.** Loan − permanent: {pp(m0f['coef'])} of available minutes {ci(m0f)}, on "
      f"{f0(m0f['N'])} episodes. This reproduces the descriptive result.")
    w(f"2. **After league-season and timing controls:** {pp(m1f['coef'])} {ci(m1f)} on the full sample. On the "
      f"{f0(m0['N'])} episodes with market values, the raw gap is already smaller ({pp(m0['coef'])}). The "
      f"{f0(ex['permanent']['N'])} permanent signings that drop out ({share(ex['permanent']['no_player_value'] / ex['permanent']['N'])} "
      "for lack of a pre-move valuation) "
      f"play little: mean share {ex['permanent']['share_mean']:.2f}, {share(ex['permanent']['zero'])} at zero. With "
      "league-season FE and "
      f"timing on that sample: {pp(m1['coef'])} {ci(m1)}.")
    w(f"3. **After player and market-value controls:** adding age and position gives {pp(m2['coef'])} {ci(m2)}. "
      f"Adding market values (preferred model) gives **{pp(m3['coef'])} {ci(m3)}**. Conditional on observed player "
      "value relative to the receiving squad, squad value, age, position, arrival timing and league-season, loan "
      f"arrivals are associated with {abs(100 * m3['coef']):.1f} percentage points "
      f"{'less' if m3['coef'] < 0 else 'more'} playing time than permanent arrivals"
      + (", which is statistically indistinguishable from zero." if inc0(m3) else "."))
    w(f"4. **With receiving-club × season fixed effects:** {pp(m4['coef'])} {ci(m4)}, on {f0(m4['N'])} episodes. It "
      f"compares loans and permanent signings joining the same club in the same season; {f0(cs['with_both'])} of "
      f"{f0(cs['groups'])} club-seasons have both kinds, covering {f0(cs['obs_in_mixed'])} episodes.")
    w(f"5. **Relative player value matters a lot.** Each doubling of a player's value relative to the receiving "
      f"squad is associated with {pp(N['doubling_effect'], signed=False)} more playing time (preferred model). It is the control "
      "that removes the raw loan gap: loanees are more valuable relative to the club they join (below).")
    lo_q, hi_q = q[0], q[-1]
    w(f"6. **The loan association changes with relative value.** Interaction coefficient "
      f"{100 * it['loan_x_lrv']['coef']:+.1f} pp per log unit (SE {100 * it['loan_x_lrv']['se']:.1f}). For players who "
      f"are small relative to the squad (lowest fifth, median {100 * lo_q['rel_median']:.1f}% of squad value), the "
      f"adjusted loan − permanent difference is {pp(lo_q['coef'])} {ci(lo_q)}. For the most valuable fifth (median {100 * hi_q['rel_median']:.1f}%) "
      f"the difference is {pp(hi_q['coef'])} {ci(hi_q)}. The lines cross near "
      f"{100 * N['crossover_relative_value']:.1f}% of squad value.")
    zl, ze, zf = z["LPM: P(zero minutes)"], z["OLS excluding exact zeros"], z["Fractional logit (share), average marginal effect"]
    rp = N["raw_positive"]["primary"]
    w(f"7. **Zero-minute players.** Without controls, dropping zero-minute players cuts the raw gap from "
      f"{pp(rp['all'])} to {pp(rp['positive_only'])}, so the zeros, which are far more common among permanent "
      f"signings, account for {share(1 - rp['positive_only'] / rp['all'])} of it. After controls, loanees are "
      f"{abs(100 * zl['coef']):.1f} pp "
      f"{'less' if zl['coef'] < 0 else 'more'} likely to play no league minutes at all {ci(zl)}. Among players with any "
      f"minutes, the loan association is {pp(ze['coef'])} {ci(ze)}. A fractional-logit model of the whole share gives "
      f"{pp(zf['coef'])} {ci(zf)}. So the zeros drive much of the raw gap, and once characteristics are controlled "
      + ("no positive loan association remains among those who play." if ze["hi"] <= 0.005 else
         "the association among those who play is small."))
    w(f"8. **England remains an exception, in the same direction as before.** Adjusted loan − permanent in GB1: "
      f"{pp(eng['coef'])} {ci(eng)} ({eng['loans']} loans). Intervals lie entirely below zero in "
      + (", ".join(f"{r.league} ({pp(r.coef)})" for r in neg.itertuples()) or "no league")
      + (f" and entirely above zero in " + ", ".join(f"{r.league} ({pp(r.coef)})" for r in pos.itertuples())
         if len(pos) else "; in no league does it lie entirely above zero")
      + f". A test that all league differences are equal gives p = {N['league_equality_p']:.4f}.")
    w(f"9. **Robustness.** Across the sample and timing variants of the preferred model the loan coefficient ranges "
      f"from {pp(samp.coef.min())} to {pp(samp.coef.max())}; "
      f"{int(samp.apply(lambda r: r['lo'] <= 0 <= r['hi'], axis=1).sum())} of {len(samp)} intervals include zero. "
      + ("Clustering by player, by player and club (two-way), or by club-season leaves the conclusion unchanged."
         if all((v["lo"] <= 0 <= v["hi"]) == inc0(m3) for v in N["se_variants"].values())
         else "The conclusion depends on the clustering choice (section 6)."))
    w("10. **What can be concluded.** The raw advantage of loans is explained by observable differences, chiefly "
      "player value relative to the receiving club: loanees tend to be better relative to where they go. Once those "
      "are held fixed, loan and permanent arrivals play about the same on average; loanees play more when they are "
      "small relative to the squad and slightly less when they are among its most valuable players, and less in "
      "England. None of this is causal: clubs choose loans for players with characteristics we do not observe "
      "(expected development, contract situation, the lender's plans), and those may drive playing time too.")
    w("")

    w("## 1. Sample and market-value audit")
    w("")
    w(f"The primary sample is rebuilt and reproduces the descriptive run: {f0(N['primary']['loan'])} loans and "
      f"{f0(N['primary']['permanent'])} permanent transfers. Joining valuations duplicates no episode.")
    w("")
    w(f"**Player value.** The latest `player_valuations` entry dated **strictly before** the move date (a valuation "
      "on the move day itself may already reflect the move). (player, date) is unique in the table, so the choice "
      "is deterministic. No later valuation is used, and episodes with none before the move have no value (not "
      f"zero). This rule agrees with Stage 1C's market value at the transfer in "
      f"{share(N['stage1c_vs_rule']['equal'] / N['stage1c_vs_rule']['compared'])} of "
      f"{f0(N['stage1c_vs_rule']['compared'])} comparable cases.")
    w("")
    w(f"**Squad value.** The sum of the latest pre-move valuations of every player whose latest valuation before the "
      f"move (within {LOOKBACK_DAYS} days) lists the receiving club, **excluding the focal player** "
      f"({N['focal_in_squad']} focal players were in it). Left missing if fewer than {MIN_SQUAD_PLAYERS} players "
      "qualify. The `clubs` table's value is a present-day snapshot and is not used.")
    w("")
    rows = []
    for k in ("loan", "permanent"):
        c = covd[k]
        rows.append([k, f0(c["N"]), f"{f0(c['player_value'])} ({share(c['player_value'] / c['N'])})",
                     f"{f0(c['only_post_move_valuation'])} / {f0(c['no_valuation_at_all'])}",
                     f"{f0(c['squad_value'])} ({share(c['squad_value'] / c['N'])})",
                     f"{f0(c['both'])} ({share(c['both'] / c['N'])})",
                     f"{c['gap_median']:.0f} / {c['gap_p90']:.0f} / {c['gap_p95']:.0f}",
                     f0(c["stale"]), f"{c['squad_players_median']:.0f}"])
    W += md_table(["", "N", "Player value before move", "Only a later valuation / none", "Squad value", "Both",
                   "Valuation gap, days (median / p90 / p95)", f"Stale (> {STALE_DAYS} days)",
                   "Valued squad players (median)"], rows)
    w("")
    w("Valuation-to-move gap: " + ", ".join(f"{k} {f0(v)}" for k, v in N["gap_hist"].items())
      + ". Coverage by league and season is in `playing_time_market_value_audit.csv`.")
    w("")
    w("**Episodes leaving the controlled models** (the common sample used for every model below): "
      + "; ".join(f"{k}: {f0(v)}" for k, v in N["common_exclusions"].items())
      + f". Common sample: {f0(N['common']['loan'])} loans and {f0(N['common']['permanent'])} permanent transfers. "
      f"The dropped episodes differ: dropped permanents have mean share {ex['permanent']['share_mean']:.2f} "
      f"({share(ex['permanent']['zero'])} zero) against {kp['permanent']['share_mean']:.2f} for kept ones, "
      f"which is why the raw gap is smaller on the common sample.")
    w("")

    w("## 2. Variables")
    w("")
    w("- **Relative player value** = player value ÷ squad value excluding the player. Used as "
      "`log_relative_value` = log(player value) − log(squad value). All values are positive, so no constant is "
      "added to any log.")
    w("- **Squad value** enters as `log_squad_value`. Log relative value and log squad value together carry exactly "
      "the information of log player value and log squad value (one is a linear transformation of the other; the "
      "fit is identical). The three cannot enter together. The relative form is used because it states Daniel's "
      "point directly: the same €10m is a bigger share of a small club's squad.")
    w(f"- **Age** at the move (from date of birth), centred at the sample mean ({N['age_center']:.1f} years), and its "
      "square. **Position** (goalkeeper / defender / midfield / attack) as dummies. Position is Transfermarkt's "
      "current main position, not the position at the move; a player's broad position rarely changes.")
    w("- **Arrival timing**: share of the receiving club's league games still to play at arrival.")
    w("- **Not used**, because they are decided after the move: early departure, later sale, re-loan, permanent "
      "conversion, later market values, injuries.")
    pa = N["prior_audit"]
    w(f"- **Prior playing time (audited, not used).** The origin club played in a covered league the season before "
      f"for {share(pa['loan']['origin_club_covered_previous_season'] / pa['loan']['N'])} of loans and "
      f"{share(pa['permanent']['origin_club_covered_previous_season'] / pa['permanent']['N'])} of permanents. The "
      f"player had a covered league appearance in the previous year for "
      f"{share(pa['loan']['covered_league_appearance_in_previous_365_days'] / pa['loan']['N'])} and "
      f"{share(pa['permanent']['covered_league_appearance_in_previous_365_days'] / pa['permanent']['N'])}. A prior-minutes "
      "control would only be measurable for moves out of covered leagues, a selected subsample, so it is left for a "
      "separate check.")
    w("")

    w("## 3. How loans and permanent signings differ before controls")
    w("")
    b = {r["kind"]: r for r in N["balance"]}
    sd = N["std_diff"]
    eur = lambda x: f"€{x / 1e6:.1f}m"
    rows = [["N", f0(b["loan"]["N"]), f0(b["permanent"]["N"]), ""],
            ["Age (mean / median)", f"{b['loan']['age_mean']:.1f} / {b['loan']['age_median']:.1f}",
             f"{b['permanent']['age_mean']:.1f} / {b['permanent']['age_median']:.1f}", f"{sd['age']:+.2f}"],
            ["Player value (median / mean)", f"{eur(b['loan']['player_value_median_eur'])} / {eur(b['loan']['player_value_mean_eur'])}",
             f"{eur(b['permanent']['player_value_median_eur'])} / {eur(b['permanent']['player_value_mean_eur'])}",
             f"{sd['log_player_value']:+.2f} (log)"],
            ["Receiving squad value (median)", eur(b["loan"]["squad_value_median_eur"]),
             eur(b["permanent"]["squad_value_median_eur"]), f"{sd['log_squad_value']:+.2f} (log)"],
            ["Relative value (median, % of squad)", f"{100 * b['loan']['relative_value_median']:.1f}%",
             f"{100 * b['permanent']['relative_value_median']:.1f}%", f"{sd['log_relative_value']:+.2f} (log)"],
            ["League games left at arrival (mean share)", f"{b['loan']['remain_mean']:.3f}",
             f"{b['permanent']['remain_mean']:.3f}", f"{sd['remain']:+.2f}"],
            ["Arrived before the club's first league game", share(b["loan"]["arrived_before_first_game"]),
             share(b["permanent"]["arrived_before_first_game"]), ""]]
    for p in POSITIONS:
        rows.append([f"Position: {p}", share(b["loan"][f"position_{p}"]), share(b["permanent"][f"position_{p}"]), ""])
    rows += [["Playing-time share (mean / median)", f"{b['loan']['share_mean']:.3f} / {b['loan']['share_median']:.3f}",
              f"{b['permanent']['share_mean']:.3f} / {b['permanent']['share_median']:.3f}", ""],
             ["Zero minutes", share(b["loan"]["zero_share"]), share(b["permanent"]["zero_share"]), ""]]
    W += md_table(["Primary sample", "Loans", "Permanent", "Standardised difference"], rows)
    w("")
    who = ("their own higher value" if abs(sd["log_player_value"]) > abs(sd["log_squad_value"])
           else "the lower value of the squads they join")
    w(f"The largest difference is relative value (standardised difference {sd['log_relative_value']:+.2f}): loanees "
      f"are worth more relative to the squad they join. Both parts contribute: they are worth more themselves "
      f"({sd['log_player_value']:+.2f}) and join less valuable squads ({sd['log_squad_value']:+.2f}), with "
      f"{who} the larger part. They are also "
      + ("younger" if b["loan"]["age_mean"] < b["permanent"]["age_mean"] else "older")
      + (", less often goalkeepers" if b["loan"]["position_Goalkeeper"] < b["permanent"]["position_Goalkeeper"] else "")
      + (", and more often arrive after the season has started." if b["loan"]["arrived_before_first_game"]
         < b["permanent"]["arrived_before_first_game"] else "."))
    w("")

    w("## 4. The model sequence")
    w("")
    seq = [("M0 full", "M0. Raw", "full primary sample"), ("M1 full", "M1. + league × season FE, arrival timing", "full"),
           ("M2 full", "M2. + age, age², position", "full"), ("M0", "M0. Raw", "market-value sample"),
           ("M1", "M1. + league × season FE, arrival timing", "market-value sample"),
           ("M2", "M2. + age, age², position", "market-value sample"),
           ("M3", "**M3. + log relative value, log squad value (preferred)**", "market-value sample"),
           ("M4", "M4. Club × season FE instead of league × season; + log player value", "market-value sample")]
    W += md_table(["Model", "Sample", "Loan − permanent", "95% CI", "N", "R²"],
                  [[lab, s, pp(M[k]["coef"]), ci(M[k]), f0(M[k]["N"]), f"{M[k]['r2']:.3f}"] for k, lab, s in seq])
    w("")
    c3 = N["m3_controls"]
    w(f"In the preferred model the other coefficients are: log relative value {100 * c3['log_relative_value']['coef']:+.1f} "
      f"pp per log unit; log squad value {100 * c3['log_squad_value']['coef']:+.1f} pp; share of season left "
      f"{100 * c3['remain']['coef']:+.1f} pp (whole season vs none); age {100 * c3['age_c']['coef']:+.1f} pp per year at "
      f"the mean age, with a negative square. Adding age and position raises the loan coefficient (loanees are "
      "younger, and younger players play less); adding relative value removes it.")
    w("")
    w(f"**Collinearity.** Correlations: log player value with log squad value "
      f"{N['corr']['log_player_value']['log_squad_value']:.2f}, with log relative value "
      f"{N['corr']['log_player_value']['log_relative_value']:.2f}; log relative value with log squad value "
      f"{N['corr']['log_relative_value']['log_squad_value']:.2f}. Variance inflation factors in the preferred model: "
      + ", ".join(f"{k} {v:.2f}" for k, v in N["vif_preferred"].items()) + ". No problem.")
    w("")
    w(f"![Loan coefficient by model]({FIGS['models']})")
    w("")
    w(f"**Club × season fixed effects (M4).** {f0(cs['groups'])} club-seasons; {f0(cs['singletons'])} have a single "
      f"episode and drop out; {f0(cs['with_both'])} contain both loans and permanents ({f0(cs['obs_in_mixed'])} episodes, "
      f"{f0(cs['loans_in_mixed'])} of them loans). These are the comparisons that identify the loan coefficient, so "
      "there is ample within-club-season variation. Squad value is nearly constant within a club-season and is "
      "absorbed by the fixed effects; relative value then differs from player value by almost a constant, so the "
      f"model uses log player value. For M1–M3, {N['league_seasons_with_both']} of {N['league_seasons']} "
      "league-seasons contain both loans and permanent moves.")
    w("")

    w("## 5. Does the loan association vary with relative value?")
    w("")
    w("Preferred model plus loan × (log relative value, centred). Average adjusted predictions: each episode's "
      "prediction with loan status and relative value set, averaged over the sample. Plotted over the 5th–95th "
      "percentile of relative value.")
    w("")
    W += md_table(["Relative-value fifth", "Median relative value", "Loans", "Permanent", "Adjusted loan − permanent", "95% CI"],
                  [[x["q"], f"{100 * x['rel_median']:.1f}%", f0(x["n_loan"]), f0(x["n_perm"]), pp(x["coef"]), ci(x)]
                   for x in q])
    w("")
    rest = q[1:]
    w(f"The fifths show the shape better than the straight line: the positive loan difference is concentrated in "
      f"the lowest fifth ({pp(q[0]['coef'])}), while from the second fifth upward the differences range from "
      f"{pp(min(x['coef'] for x in rest))} to {pp(max(x['coef'] for x in rest))}.")
    w("")
    w(f"![Relative value]({FIGS['relative']})")
    w("")

    w("## 6. Repeated observations and standard errors")
    w("")
    dp = N["dependence"]
    w(f"{f0(dp['episodes'])} episodes, {f0(dp['players'])} players ({f0(dp['players_multiple'])} appear more than "
      f"once, accounting for {f0(dp['episodes_of_repeat_players'])} episodes, for example a loan to one club and a "
      f"later permanent move to another), {f0(dp['clubs'])} receiving clubs, {f0(dp['club_seasons'])} club-seasons. "
      "Primary clustering is by receiving club"
      + ("; the alternatives below give the same conclusion." if all(
          (v["lo"] <= 0 <= v["hi"]) == (M["M3"]["lo"] <= 0 <= M["M3"]["hi"]) for v in N["se_variants"].values())
         else "; the alternatives below do not all agree."))
    w("")
    W += md_table(["Clustering", "Loan coefficient", "SE", "95% CI"],
                  [[k, pp(v["coef"]), f"{100 * v['se']:.2f}", ci(v)] for k, v in N["se_variants"].items()])
    w("")

    w("## 7. Zero minutes and the bounded outcome")
    w("")
    w(f"Zero minutes: {share(N['zero_rate']['loan'])} of loans and {share(N['zero_rate']['permanent'])} of permanent "
      "signings in the common sample. The OLS share model stays primary; these checks ask whether the zeros drive it.")
    w("")
    W += md_table(["Model (preferred controls)", "Loan association", "95% CI", "N"],
                  [[k, pp(v["coef"]), ci(v), f0(v["N"])] for k, v in z.items() if "error" not in v])
    w("")
    w("The logit models use league and season effects additively (not their interaction), to avoid groups with no "
      "zeros; the effect is the average change in predicted probability or share.")
    w("")

    w("## 8. Sample and timing robustness (preferred model)")
    w("")
    W += md_table(["Variant", "Loans", "Permanent", "Loan coefficient", "SE", "95% CI", "Interval includes zero?"],
                  [[r.variant, f0(r.loans), f0(r.permanents), pp(r.coef), f"{100 * r.se:.2f}",
                    f"[{100 * r.lo:+.1f}, {100 * r.hi:+.1f}]", "yes" if r.lo <= 0 <= r.hi else "no"]
                   for r in R.itertuples()])
    w("")
    w("All variants are reported; none was chosen for its estimate.")
    w("")

    w("## 9. By league (one model with loan × league)")
    w("")
    W += md_table(["League", "Loans", "Permanent", "Raw difference", "Adjusted difference", "95% CI"],
                  [[r.league + (" *" if r.small else ""), f0(r.loans), f0(r.permanents), pp(r.raw_diff), pp(r.coef),
                    f"[{100 * r.lo:+.1f}, {100 * r.hi:+.1f}]"] for r in L.sort_values("coef").itertuples()])
    w("")
    w("\\* fewer than 100 loans: interpret with care. The leagues share every other coefficient; only the loan "
      "difference varies by league.")
    w("")
    w(f"![By league]({FIGS['league']})")
    w("")

    w("## 10. What can and cannot be said")
    w("")
    w("- Say: *conditional on observed player value relative to the receiving squad, squad value, age, position, "
      f"arrival timing and league-season, loan arrivals are associated with {abs(100 * m3['coef']):.1f} percentage "
      f"points {'less' if m3['coef'] < 0 else 'more'} playing time than permanent arrivals "
      f"({ci(m3)}).*")
    w("- Do not say that loans increase or decrease playing time. Loan status is chosen by clubs and players who "
      "know things the data do not: the lender's plans for the player, contract length, wage arrangements, "
      "guarantees of minutes written into some loan agreements, injuries. Any of these can move both loan status "
      "and playing time.")
    w("- The comparison covers first-division receiving clubs in 14 European leagues, players of the 2023/24–2025/26 "
      "cohort, and league minutes only.")
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


SAMPLE_COLS = ["event_id", "kind", "loan", "player_id", "player_name", "move_date", "arrival_season",
               "receiving_league", "league_season", "club_season", "a_club_id", "a_club", "b_club_id", "b_club",
               "share", "minutes", "available_minutes", "zero", "remain", "arrived_before_first_game", "age",
               "position_clean", "player_value", "player_value_date", "player_value_gap_days", "stale_player_value",
               "squad_value", "squad_players", "squad_value_gap_days", "focal_in_squad", "relative_value",
               "log_player_value", "log_squad_value", "log_relative_value", "primary_sample", "in_common_sample"]


def main() -> dict:
    N, tb = compute()
    S = tb["sample"]
    out = S[S.primary_sample | S.in_common_sample | True][SAMPLE_COLS].copy()
    for c in ("move_date", "player_value_date"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(SAMPLE_CSV, index=False)
    tb["models"].to_csv(MODELS_CSV, index=False)
    tb["audit"].to_csv(MV_AUDIT_CSV, index=False)
    tb["balance"].to_csv(BALANCE_CSV, index=False)
    tb["robust"].to_csv(ROBUST_CSV, index=False)
    tb["league"].to_csv(LEAGUE_CSV, index=False)
    figures(N)
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    SUMMARY_MD.write_text(render(N))
    M = N["models"]
    print(f"wrote {SUMMARY_MD.name}")
    for k in ("M0 full", "M1", "M2", "M3", "M4"):
        print(f"{k:8} {100 * M[k]['coef']:+.2f} pp [{100 * M[k]['lo']:+.2f}, {100 * M[k]['hi']:+.2f}] N={M[k]['N']:,}")
    return N


if __name__ == "__main__":
    main()
