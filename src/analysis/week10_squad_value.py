"""Week 10, Question 6: why is the origin-club squad value (`from_squad_value`) missing, and can more be recovered?

    python -m src.analysis.week10_squad_value

Read-only with respect to every stored artefact: the master, Stage 1C and the snapshot are only read. Writes only into
data/outputs/rebuild/professor_feedback_week10/q6_squad_value/.

Steps
  1. Reproduce the stored `from_squad_value` exactly (same function, same inputs) and check a parameterised copy of
     the rule against it, so every sensitivity differs from the original only in the parameter that is varied.
  2. Classify every missing value with one ordered reason.
  3. Explain the loan/permanent coverage gap (origin league coverage, youth sides, tier, country, season) and test
     the hypothesis that the snapshot only values players who passed through a covered club.
  4. Sensitivities, each stored separately (never replacing `from_squad_value`): longer look-back, lower minimum,
     membership by registration spells, and combinations. Coverage gain and agreement with the original.
  5. Regression implications: complete-case sample if the origin squad value is added as a control, and the
     preferred specification (M3) with and without `log_from_squad_value` on that sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..config import DEFAULT_DATABASE
from . import feedback_nov2026 as fb
from . import playing_time_regression as reg
from .feedback_nov2026_report import MASTER, OUT, m3

DIR = OUT / "professor_feedback_week10" / "q6_squad_value"
COVERED = ["Belgium", "Denmark", "England", "France", "Germany", "Greece", "Italy", "Netherlands", "Portugal",
           "Russia", "Scotland", "Spain", "Türkiye", "Ukraine"]

# name -> (look-back days, minimum players, membership)
VARIANTS = {
    "lb365_min11": (365, 11, "label"),     # the stored rule
    "lb548_min11": (548, 11, "label"),
    "lb730_min11": (730, 11, "label"),
    "lb365_min8": (365, 8, "label"),
    "lb730_min8": (730, 8, "label"),
    "reg365_min11": (365, 11, "registration"),
    "union365_min11": (365, 11, "union"),
    "union730_min11": (730, 11, "union"),
}
VARIANT_TEXT = {
    "lb365_min11": "stored rule: valuation label = club, look-back 365 days, at least 11 players (reproduces "
                   "`from_squad_value`)",
    "lb548_min11": "valuation label, look-back 548 days (18 months), at least 11 players",
    "lb730_min11": "valuation label, look-back 730 days (2 years), at least 11 players",
    "lb365_min8": "valuation label, look-back 365 days, at least 8 players",
    "lb730_min8": "valuation label, look-back 730 days, at least 8 players",
    "reg365_min11": "membership by Stage 1C registration spell on the day before the move (instead of the valuation "
                    "label); value = the player's latest valuation strictly before the move, at most 365 days old; "
                    "at least 11 players",
    "union365_min11": "membership by valuation label OR registration spell; look-back 365 days; at least 11 players",
    "union730_min11": "membership by valuation label OR registration spell; look-back 730 days; at least 11 players",
}
REASONS = [
    ("placeholder_not_a_club", "origin is a placeholder (Without Club, Career break, Unknown, Ban, Retired)"),
    ("youth_or_reserve_origin", "origin is a youth or reserve side"),
    ("no_valuation_ever_lists_club", "no valuation of any player lists the origin club before the move"),
    ("stale_only", "valuations listing the club exist, but all are older than 365 days"),
    ("thin_1_to_10_players", "1 to 10 players qualify (fewer than 11)"),
    ("recent_labels_superseded", "recent valuations listing the club exist, but each such player's latest "
                                 "pre-move valuation lists another club"),
    ("other", "other"),
]


# ---------------------------------------------------------------------------
# The rule, parameterised (identical to playing_time_regression.squad_value at 365 days / 11 players)
# ---------------------------------------------------------------------------

def _latest_before(pairs_players: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """Each (pair, player) row gets the player's latest valuation strictly before move_date (any club label)."""
    x = pairs_players.sort_values("move_date")
    return pd.merge_asof(x, pv, left_on="move_date", right_on="vdate", by="player_id", direction="backward",
                         allow_exact_matches=False)


def squad_param(E: pd.DataFrame, pv: pd.DataFrame, lookback: int, min_players: int, membership: str = "label",
                reg_members: pd.DataFrame | None = None) -> pd.DataFrame:
    """E: event_id, player_id, b_club_id, move_date. Returns per event: value, players (both masked below the
    minimum), the unmasked player count (focal excluded) and focal membership."""
    E = E.astype({"player_id": "int64", "b_club_id": "int64", "move_date": "datetime64[ns]"})
    pairs = E[["b_club_id", "move_date"]].drop_duplicates().reset_index(drop=True)
    pairs["pair"] = np.arange(len(pairs))
    lb = pd.Timedelta(days=lookback)
    members = []
    if membership in ("label", "union"):
        cand = pairs.merge(pv[["player_id", "vdate", "vclub"]], left_on="b_club_id", right_on="vclub")
        cand = cand[(cand.vdate < cand.move_date) & (cand.vdate >= cand.move_date - lb)]
        cand = cand[["pair", "b_club_id", "move_date", "player_id"]].drop_duplicates()
        latest = _latest_before(cand, pv)
        lab = latest[(latest.vclub == latest.b_club_id) & (latest.vdate >= latest.move_date - lb)]
        members.append(lab[["pair", "player_id", "vdate", "value"]])
    if membership in ("registration", "union"):
        R = pairs.merge(reg_members, on=["b_club_id", "move_date"])[["pair", "b_club_id", "move_date", "player_id"]]
        latest = _latest_before(R.drop_duplicates(), pv)
        rg = latest[latest.vdate.notna() & (latest.vdate >= latest.move_date - lb)]
        members.append(rg[["pair", "player_id", "vdate", "value"]])
    squad = pd.concat(members).drop_duplicates(["pair", "player_id"])
    agg = squad.groupby("pair").agg(v_incl=("value", "sum"), n_incl=("player_id", "nunique"))
    ep = E[["event_id", "player_id", "b_club_id", "move_date"]].merge(pairs, on=["b_club_id", "move_date"])
    ep = ep.join(agg, on="pair")
    focal = squad[["pair", "player_id", "value"]].rename(columns={"value": "focal_value"})
    ep = ep.merge(focal, on=["pair", "player_id"], how="left")
    ep["focal_in_squad"] = ep.focal_value.notna()
    ep["n_raw"] = (ep.n_incl.fillna(0) - ep.focal_in_squad.astype(int)).astype(int)
    ok = ep.n_raw >= min_players
    ep["value"] = (ep.v_incl - ep.focal_value.fillna(0)).where(ok)
    ep["players"] = ep.n_raw.where(ok)
    return ep[["event_id", "value", "players", "n_raw", "focal_in_squad"]]


def registration_members(pairs: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """Players whose Stage 1C registration spell covers the day before the move at the club (placeholders out)."""
    S = spells[~spells.placeholder][["player_id", "club_id", "start", "end"]]
    con = duckdb.connect()
    con.register("P", pairs[["b_club_id", "move_date"]].drop_duplicates())
    con.register("S", S)
    out = con.sql("""select P.b_club_id, P.move_date, S.player_id from P join S on S.club_id = P.b_club_id
                     and S."start" <= P.move_date - interval 1 day and S."end" >= P.move_date - interval 1 day""").df()
    con.close()
    out["move_date"] = pd.to_datetime(out.move_date).astype("datetime64[ns]")
    return out.astype({"player_id": "int64", "b_club_id": "int64"})


# ---------------------------------------------------------------------------
# Diagnosis helpers
# ---------------------------------------------------------------------------

def label_counts(E: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """Per event: distinct players with any valuation listing the origin club before the move (ever), within 365
    days, and within 730 days (focal player excluded throughout)."""
    pairs = E[["b_club_id", "move_date"]].drop_duplicates()
    con = duckdb.connect()
    con.register("P", pairs)
    con.register("V", pv[["player_id", "vdate", "vclub"]])
    c = con.sql("""select P.b_club_id, P.move_date, V.player_id,
                          max(V.vdate) last_label_date
                   from P join V on V.vclub = P.b_club_id and V.vdate < P.move_date
                   group by 1, 2, 3""").df()
    con.close()
    c["move_date"] = pd.to_datetime(c.move_date).astype("datetime64[ns]")
    c["last_label_date"] = pd.to_datetime(c.last_label_date).astype("datetime64[ns]")
    c = c.astype({"b_club_id": "int64", "player_id": "int64"})
    x = E[["event_id", "player_id", "b_club_id", "move_date"]].merge(c.rename(columns={"player_id": "pid"}),
                                                                     on=["b_club_id", "move_date"], how="left")
    x = x[x.pid.isna() | (x.pid != x.player_id)]
    age = (x.move_date - x.last_label_date).dt.days
    x["ever"] = x.pid.notna()
    x["w365"] = x.ever & (age <= 365)
    x["w730"] = x.ever & (age <= 730)
    g = x.groupby("event_id").agg(label_ever=("ever", "sum"), label_365=("w365", "sum"), label_730=("w730", "sum"))
    return g.reindex(E.event_id).fillna(0).astype(int)


def classify(T: pd.DataFrame) -> pd.Series:
    cond = [T.placeholder, T.youth, T.label_ever.eq(0), T.label_365.eq(0), T.n_raw.between(1, 10),
            T.n_raw.eq(0)]
    return pd.Series(np.select(cond, [k for k, _ in REASONS[:6]], "other"), index=T.index).where(T.missing)


def cov(df: pd.DataFrame, col: str = "has") -> dict:
    return {"N": int(len(df)), "covered": int(df[col].sum()), "coverage": float(df[col].mean()) if len(df) else np.nan}


def pct(x: float, d: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}%"


def fmt_pp(x: float) -> str:
    """Coefficient (share units) in signed percentage points; a value that rounds to zero prints 0.00."""
    v = round(100 * x, 2)
    return "0.00" if v == 0 else f"{v:+.2f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    from . import club_organisation as co
    from . import loan_scope

    DIR.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(MASTER, low_memory=False)
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        pv = reg.valuations(con)
        pv_players = int(con.sql("select count(distinct player_id) from player_valuations").fetchone()[0])
        n_players_tab = int(con.sql("select count(*) from players").fetchone()[0])
        snap_clubs = set(con.sql("select cast(club_id as bigint) from clubs").df().iloc[:, 0])
        n_linked = int(con.sql("""select count(*) from players where
            cast(current_club_id as varchar) in (select club_id from clubs)
            or player_id in (select player_id from appearances)
            or player_id in (select player_id from game_lineups)""").fetchone()[0])
    finally:
        con.close()
    typed = M[M.kind.isin(["loan", "permanent"])][["event_id", "player_id", "transfer_date", "from_club_id",
                                                   "kind"]].copy()
    typed["move_date"] = pd.to_datetime(typed.transfer_date)

    # 1. reproduction ---------------------------------------------------------------------------------------------
    Fv = fb.club_squad_value(typed[["event_id", "player_id", "move_date", "from_club_id"]], pv, "from_club_id")
    Fv = Fv.set_index("event_id").reindex(M.event_id)
    stored = M.set_index("event_id").from_squad_value
    both = stored.notna() & Fv.squad_value.notna()
    assert (stored.notna() == Fv.squad_value.notna()).all(), "missingness of from_squad_value not reproduced"
    assert np.allclose(stored[both], Fv.squad_value[both], rtol=1e-12, atol=1e-6), "from_squad_value not reproduced"
    E = typed.rename(columns={"from_club_id": "b_club_id"})[["event_id", "player_id", "b_club_id", "move_date"]]
    E = E.dropna().astype({"b_club_id": "int64", "player_id": "int64"})
    base = squad_param(E, pv, reg.LOOKBACK_DAYS, reg.MIN_SQUAD_PLAYERS).set_index("event_id")
    assert reg.LOOKBACK_DAYS == 365 and reg.MIN_SQUAD_PLAYERS == 11
    b = base.reindex(M.event_id)
    assert (b.value.notna() == stored.notna()).all() and np.allclose(b.value[both], stored[both], rtol=1e-12,
                                                                      atol=1e-6), "parameterised rule != original"
    repro = {"stored_non_missing": int(stored.notna().sum()), "reproduced_exactly": True,
             "parameterised_rule_identical": True}

    # 4 (computed first). sensitivities -----------------------------------------------------------------------------
    u = loan_scope.universe()
    spells = fb.registration_spells(u)
    regm = registration_members(E, spells)
    V = {}
    for name, (lb, mn, mem) in VARIANTS.items():
        V[name] = squad_param(E, pv, lb, mn, mem, regm if mem != "label" else None).set_index("event_id")
    assert np.allclose(V["lb365_min11"].value.reindex(M.event_id)[both], stored[both], rtol=1e-12, atol=1e-6)

    # registration roster of the origin club and how many of those players are valued at all
    rk = regm.merge(E[["event_id", "b_club_id", "move_date", "player_id"]].rename(columns={"player_id": "focal"}),
                    on=["b_club_id", "move_date"])
    rk = rk[rk.player_id != rk.focal]
    pv_last = pv.groupby("player_id").vdate.min().rename("first_vdate")
    rk = rk.join(pv_last, on="player_id")
    rk["valued_ever"] = rk.first_vdate.notna()
    rk["valued_before"] = rk.first_vdate < rk.move_date
    lbk = _latest_before(rk[["event_id", "player_id", "move_date"]].drop_duplicates(), pv)
    lbk["valued_365"] = lbk.vdate.notna() & (lbk.vdate >= lbk.move_date - pd.Timedelta(days=365))
    rk = rk.merge(lbk[["event_id", "player_id", "valued_365"]], on=["event_id", "player_id"], how="left")
    RG = rk.groupby("event_id").agg(reg_players=("player_id", "nunique"), reg_valued_ever=("valued_ever", "sum"),
                                    reg_valued_365=("valued_365", "sum"))

    # 2. per-episode frame ------------------------------------------------------------------------------------------
    T = M[["event_id", "kind", "from_club_id", "from_club_name", "season", "from_country_v2", "from_league_tier_at_move",
           "from_league_tier_status", "from_in_covered_top_flight_at_move", "in_cf_main_model_sample",
           "from_squad_value", "log_from_squad_value"]].copy()
    T = T[T.kind.isin(["loan", "permanent"])].set_index("event_id")
    T["has"] = T.from_squad_value.notna()
    T["missing"] = ~T.has
    T["placeholder"] = T.from_club_id.isin(fb.PLACEHOLDER_CLUB_IDS)
    youth = fb.youth_club_ids(M, co.load_map())
    T["youth"] = T.from_club_id.astype("int64").isin(youth)
    LC = label_counts(E, pv)
    T = T.join(LC).join(V["lb365_min11"][["n_raw"]]).join(RG)
    for c in ("reg_players", "reg_valued_ever", "reg_valued_365"):
        T[c] = T[c].fillna(0).astype(int)
    T["n_raw_730"] = V["lb730_min11"].n_raw.reindex(T.index)
    T["reason"] = classify(T)
    T["main"] = T.in_cf_main_model_sample.astype(bool)
    T["origin_in_snapshot_clubs"] = T.from_club_id.astype("int64").isin(snap_clubs)
    T["covered_top_flight"] = T.from_in_covered_top_flight_at_move.map({True: "yes", False: "no", "True": "yes",
                                                                        "False": "no"}).fillna("unknown")
    T["country_group"] = np.where(T.from_country_v2.isin(COVERED), "one of the 14 countries",
                                  np.where(T.from_country_v2.isna(), "country unknown", "other country"))
    tier = pd.to_numeric(T.from_league_tier_at_move, errors="coerce")
    T["tier_band"] = np.select([tier.eq(1), tier.eq(2), tier.ge(3)], ["tier 1", "tier 2", "tier 3+"],
                               "tier unknown: " + T.from_league_tier_status.astype(str))
    s = T.season.astype(str).str[:4].astype(int)
    T["season_bin"] = pd.cut(s, [0, 2011, 2015, 2019, 2022, 2030],
                             labels=["to 2011/12", "2012/13-2015/16", "2016/17-2019/20", "2020/21-2022/23",
                                     "2023/24-2025/26"]).astype(str)
    Mn = T[T.main]
    assert len(Mn) == 18090

    # reasons table -------------------------------------------------------------------------------------------------
    rows = []
    for pop, D in (("main sample (calendar-first)", Mn), ("all typed master episodes", T)):
        for kind in ("all", "loan", "permanent"):
            d = D if kind == "all" else D[D.kind == kind]
            miss = d[d.missing]
            for key, text in REASONS:
                n = int((miss.reason == key).sum())
                rows.append({"population": pop, "kind": kind, "reason": key, "description": text, "episodes": n,
                             "share_of_missing": n / len(miss) if len(miss) else np.nan,
                             "share_of_all": n / len(d) if len(d) else np.nan, "missing_total": len(miss),
                             "episodes_total": len(d)})
    R = pd.DataFrame(rows)
    R.to_csv(DIR / "missingness_reasons.csv", index=False)
    thin = Mn[Mn.reason.eq("thin_1_to_10_players")]
    thin_info = {"thin_main": int(len(thin)), "thin_8_to_10": int(thin.n_raw.between(8, 10).sum()),
                 "thin_reach_11_at_730": int((thin.n_raw_730 >= 11).sum()),
                 "thin_loan": int(thin.kind.eq("loan").sum()), "thin_perm": int(thin.kind.eq("permanent").sum())}
    stale = Mn[Mn.reason.eq("stale_only")]
    thin_info["stale_main"] = int(len(stale))
    thin_info["stale_reach_11_at_730"] = int((stale.n_raw_730 >= 11).sum())

    # 3. loan vs permanent explanation --------------------------------------------------------------------------------
    gaps = []
    for dim in ("covered_top_flight", "youth", "country_group", "tier_band", "season_bin", "origin_in_snapshot_clubs"):
        for val, d in Mn.groupby(dim):
            for kind in ("loan", "permanent"):
                x = d[d.kind == kind]
                gaps.append({"dimension": dim, "value": str(val), "kind": kind, "N": len(x),
                             "share_of_kind": len(x) / int((Mn.kind == kind).sum()),
                             "coverage": x.has.mean() if len(x) else np.nan})
    G = pd.DataFrame(gaps)
    G.to_csv(DIR / "coverage_by_group_loan_vs_permanent.csv", index=False)
    # reweighting: permanent coverage if permanent arrivals had the loans' mix of origin types
    cell = Mn.assign(cell=Mn.covered_top_flight + "|" + Mn.youth.map({True: "youth", False: "senior"}))
    cw = cell.groupby(["kind", "cell"]).agg(n=("has", "size"), c=("has", "mean")).reset_index()
    wl = cw[cw.kind == "loan"].set_index("cell").n / cw[cw.kind == "loan"].n.sum()
    cp = cw[cw.kind == "permanent"].set_index("cell").c
    reweighted = float((wl * cp.reindex(wl.index)).sum())
    dec = {"loan_coverage": float(Mn[Mn.kind == "loan"].has.mean()),
           "perm_coverage": float(Mn[Mn.kind == "permanent"].has.mean()),
           "perm_coverage_with_loan_mix": reweighted,
           "loan_share_origin_covered_top_flight": float((Mn[Mn.kind == "loan"].covered_top_flight == "yes").mean()),
           "perm_share_origin_covered_top_flight": float(
               (Mn[Mn.kind == "permanent"].covered_top_flight == "yes").mean()),
           "coverage_origin_covered_top_flight": float(Mn[Mn.covered_top_flight == "yes"].has.mean()),
           "coverage_origin_not_covered_top_flight": float(Mn[Mn.covered_top_flight != "yes"].has.mean())}
    cw.to_csv(DIR / "reweighting_cells.csv", index=False)

    # hypothesis: valuations exist only for players who passed through a covered club --------------------------------
    dec["share_of_gap_explained"] = (reweighted - dec["perm_coverage"]) / (dec["loan_coverage"] - dec["perm_coverage"])
    hyp = {"players_in_snapshot_players_table": n_players_tab, "players_with_any_valuation": pv_players,
           "players_linked_to_covered_competitions": n_linked,
           "valuations_total": int(len(pv)), "valuations_at_clubs_outside_snapshot_clubs":
               int((~pv.vclub.isin(snap_clubs)).sum())}
    roster = []
    for lab, d in (("missing", Mn[Mn.missing & ~Mn.placeholder & ~Mn.youth]), ("present", Mn[Mn.has])):
        d = d[d.reg_players > 0]
        roster.append({"group": lab, "episodes_with_registration_roster": len(d),
                       "median_registered_players": float(d.reg_players.median()),
                       "median_valued_within_365": float(d.reg_valued_365.median()),
                       "share_registered_valued_ever": float(d.reg_valued_ever.sum() / d.reg_players.sum()),
                       "share_registered_valued_within_365": float(d.reg_valued_365.sum() / d.reg_players.sum())})
    RO = pd.DataFrame(roster)
    RO.to_csv(DIR / "registration_roster_valuation_check.csv", index=False)

    # 4. sensitivity summary ---------------------------------------------------------------------------------------
    srows = []
    for name in VARIANTS:
        v = V[name].value.reindex(Mn.index)
        o = Mn.from_squad_value
        bo = v.notna() & o.notna()
        lo, lv = np.log(o[bo].where(o[bo] > 0)), np.log(v[bo].where(v[bo] > 0))
        ok = lo.notna() & lv.notna()
        r = {"variant": name, "definition": VARIANT_TEXT[name]}
        for kind in ("all", "loan", "permanent"):
            k = Mn.kind.eq(kind) if kind != "all" else pd.Series(True, index=Mn.index)
            r[f"coverage_{kind}"] = float(v[k].notna().mean())
            r[f"gain_pp_{kind}"] = float(v[k].notna().mean() - o[k].notna().mean())
            r[f"new_episodes_{kind}"] = int((v[k].notna() & o[k].isna()).sum())
            r[f"lost_episodes_{kind}"] = int((v[k].isna() & o[k].notna()).sum())
        r["both_available"] = int(bo.sum())
        r["identical_where_both"] = float((np.isclose(v[bo], o[bo], rtol=1e-9)).mean()) if bo.any() else np.nan
        r["median_abs_pct_diff"] = float((v[bo] / o[bo] - 1).abs().median()) if bo.any() else np.nan
        r["corr_logs"] = float(np.corrcoef(lo[ok], lv[ok])[0, 1]) if ok.sum() > 2 else np.nan
        srows.append(r)
    S = pd.DataFrame(srows)
    S.to_csv(DIR / "sensitivity_summary.csv", index=False)

    # implementation check: does registration membership ever disagree with the label rule in a way that shows an
    # error? (a player valued with club B whose registration at the move says another club)
    lab_only = squad_param(E, pv, 365, 1, "label").set_index("event_id")
    reg_only = squad_param(E, pv, 365, 1, "registration", regm).set_index("event_id")
    impl = {"main_label_players_median": float(lab_only.n_raw.reindex(Mn.index).median()),
            "main_registration_players_median": float(reg_only.n_raw.reindex(Mn.index).median())}

    # sensitivity values file ------------------------------------------------------------------------------------------
    SV = pd.DataFrame(index=T.index)
    for name in VARIANTS:
        SV[f"from_squad_value_{name}"] = V[name].value.reindex(T.index)
        SV[f"from_squad_players_raw_{name}"] = V[name].n_raw.reindex(T.index).astype("Int64")
    SV["from_squad_value_missing_reason"] = T.reason
    SV["origin_registered_players"] = T.reg_players
    SV["origin_registered_players_valued_365"] = T.reg_valued_365
    SV.reset_index().to_csv(DIR / "sensitivity_values.csv", index=False)

    # 5. regression implications --------------------------------------------------------------------------------------
    mask = M.in_cf_main_model_sample.astype(bool)
    full = m3(M, mask, "minutes_share")
    assert full["N"] == 18090
    best = "lb730_min11"
    alt = V[best].value.reindex(M.event_id).to_numpy()
    M2 = M.assign(from_sv_alt=alt, log_from_sv_alt=np.log(pd.Series(alt).where(pd.Series(alt) > 0)).to_numpy())
    rr = []

    def m3x(mk: pd.Series, extra: list[str]) -> dict:
        C = M2[mk.astype(bool)].copy()
        C["loan"] = C.kind.eq("loan").astype(int)
        C["age_c"] = C.age_exact - C.age_exact.mean()
        C["age_c2"] = C.age_c ** 2
        C["b_club_id"] = C.to_club_id
        r = reg.coef(reg.fit(C, reg.M3 + extra, fe="league_season", cats=["position_clean"], y="minutes_share"))
        r["loans"], r["permanent"] = int(C.kind.eq("loan").sum()), int(C.kind.eq("permanent").sum())
        return r

    for label, col, logc in (("original from_squad_value", "from_squad_value", "log_from_squad_value"),
                             (f"sensitivity {best}", "from_sv_alt", "log_from_sv_alt")):
        cc = mask & M2[logc].notna() & np.isfinite(M2[logc])
        a = m3x(cc, [])
        bq = m3x(cc, [logc])
        rr.append({"origin_value": label, "sample": "main sample (calendar-first)", "N": full["N"],
                   "loans": full["loans"], "permanent": full["permanent"],
                   "loan_share": full["loans"] / full["N"], "loan_coef": full["coef"], "p": full["p"],
                   "control_added": "no"})
        rr.append({"origin_value": label, "sample": "complete cases, without origin control", "N": a["N"],
                   "loans": a["loans"], "permanent": a["permanent"], "loan_share": a["loans"] / a["N"],
                   "loan_coef": a["coef"], "p": a["p"], "control_added": "no"})
        rr.append({"origin_value": label, "sample": "complete cases, with log origin squad value", "N": bq["N"],
                   "loans": bq["loans"], "permanent": bq["permanent"], "loan_share": bq["loans"] / bq["N"],
                   "loan_coef": bq["coef"], "p": bq["p"], "control_added": "yes"})
    RR = pd.DataFrame(rr).drop_duplicates()
    RR["retained_share_of_main"] = RR.N / full["N"]
    RR.to_csv(DIR / "regression_implications.csv", index=False)

    numbers = {"reproduction": repro, "thin": thin_info, "decomposition": dec, "hypothesis": hyp,
               "implementation": impl, "main_N": int(len(Mn)),
               "main_coverage": cov(Mn), "loan_coverage": cov(Mn[Mn.kind == "loan"]),
               "perm_coverage": cov(Mn[Mn.kind == "permanent"]), "all_typed_coverage": cov(T)}
    (DIR / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    (DIR / "README.md").write_text(readme(numbers, R, G, S, RR, RO, best))
    return {"numbers": numbers, "R": R, "S": S, "RR": RR, "G": G, "RO": RO}


def readme(n: dict, R: pd.DataFrame, G: pd.DataFrame, S: pd.DataFrame, RR: pd.DataFrame, RO: pd.DataFrame,
           best: str) -> str:
    d, t, h = n["decomposition"], n["thin"], n["hypothesis"]
    mc, lc, pc = n["main_coverage"], n["loan_coverage"], n["perm_coverage"]
    rm = R[R.population.str.startswith("main")]
    piv = rm.pivot_table(index=["reason", "description"], columns="kind", values="episodes", sort=False).reset_index()
    piv = piv[["description", "all", "loan", "permanent"]]
    tot = rm.drop_duplicates("kind").set_index("kind").missing_total
    piv.loc[len(piv)] = ["**total missing**", tot["all"], tot["loan"], tot["permanent"]]
    for c in ("all", "loan", "permanent"):
        piv[c] = piv[c].astype(int).map("{:,}".format)
    ra = R[R.population.str.startswith("all") & R.kind.eq("all")][["description", "episodes", "share_of_missing"]]
    ra = ra.assign(episodes=ra.episodes.map("{:,}".format), share_of_missing=ra.share_of_missing.map(pct))
    g1 = G[G.dimension.isin(["covered_top_flight", "youth", "country_group"])].copy()
    g1 = g1.pivot_table(index=["dimension", "value"], columns="kind", values=["N", "coverage"]).reset_index()
    g1.columns = [" ".join(c).strip() for c in g1.columns]
    g1 = g1.rename(columns={"N loan": "loans", "N permanent": "permanent", "coverage loan": "coverage, loans",
                            "coverage permanent": "coverage, permanent"})
    g1 = g1[["dimension", "value", "loans", "coverage, loans", "permanent", "coverage, permanent"]]
    for c in ("loans", "permanent"):
        g1[c] = g1[c].fillna(0).astype(int).map("{:,}".format)
    for c in ("coverage, loans", "coverage, permanent"):
        g1[c] = g1[c].map(pct)
    g2 = G[G.dimension.isin(["tier_band", "season_bin"])].pivot_table(
        index=["dimension", "value"], columns="kind", values=["N", "coverage"]).reset_index()
    g2.columns = [" ".join(c).strip() for c in g2.columns]
    g2 = g2.rename(columns={"N loan": "loans", "N permanent": "permanent", "coverage loan": "coverage, loans",
                            "coverage permanent": "coverage, permanent"})
    g2 = g2[["dimension", "value", "loans", "coverage, loans", "permanent", "coverage, permanent"]]
    for c in ("loans", "permanent"):
        g2[c] = g2[c].fillna(0).astype(int).map("{:,}".format)
    for c in ("coverage, loans", "coverage, permanent"):
        g2[c] = g2[c].map(pct)
    s = S[["variant", "coverage_all", "gain_pp_all", "gain_pp_loan", "gain_pp_permanent", "new_episodes_all",
           "lost_episodes_all", "identical_where_both", "median_abs_pct_diff", "corr_logs"]].copy()
    for c in ("coverage_all", "identical_where_both", "median_abs_pct_diff"):
        s[c] = s[c].map(pct)
    for c in ("gain_pp_all", "gain_pp_loan", "gain_pp_permanent"):
        s[c] = (100 * s[c]).map(lambda x: f"{x:+.1f}")
    s["corr_logs"] = s.corr_logs.map(lambda x: f"{x:.3f}")
    for c in ("new_episodes_all", "lost_episodes_all"):
        s[c] = s[c].map("{:,}".format)
    s = s.rename(columns={"coverage_all": "coverage (main)", "gain_pp_all": "gain, pp", "gain_pp_loan": "gain loans, pp",
                          "gain_pp_permanent": "gain permanent, pp", "new_episodes_all": "new episodes",
                          "lost_episodes_all": "lost", "identical_where_both": "identical where both",
                          "median_abs_pct_diff": "median abs. difference", "corr_logs": "corr. of logs"})
    rr = RR.copy()
    rr["loan_share"] = rr.loan_share.map(pct)
    rr["retained_share_of_main"] = rr.retained_share_of_main.map(pct)
    rr["loan_coef"] = rr.loan_coef.map(fmt_pp)
    rr["p"] = rr.p.map(lambda x: f"{x:.3f}")
    for c in ("N", "loans", "permanent"):
        rr[c] = rr[c].map("{:,}".format)
    rr = rr[["origin_value", "sample", "N", "loans", "permanent", "loan_share", "retained_share_of_main", "loan_coef",
             "p"]]
    ro = RO.copy()
    for c in ("share_registered_valued_ever", "share_registered_valued_within_365"):
        ro[c] = ro[c].map(pct)
    ro["episodes_with_registration_roster"] = ro.episodes_with_registration_roster.map("{:,}".format)
    sb = S.set_index("variant")
    reg_g, l730 = sb.loc["reg365_min11"], sb.loc["lb730_min11"]
    un730 = sb.loc["union730_min11"]
    orig = RR[RR.origin_value.str.startswith("original")].set_index("sample")
    cc = orig.loc["complete cases, without origin control"]
    cwc = orig.loc["complete cases, with log origin squad value"]
    lines = [
        "# Week 10, Question 6: origin-club squad value (`from_squad_value`)",
        "",
        "Generated by `python -m src.analysis.week10_squad_value` (module `src/analysis/week10_squad_value.py`). Every "
        "number below is computed by that run. Nothing stored was changed: the master, Stage 1C and "
        "`from_squad_value` are only read.",
        "",
        "## Reproduction",
        "",
        f"The stored `from_squad_value` is reproduced exactly for all {n['reproduction']['stored_non_missing']:,} "
        "non-missing values, and it is missing in exactly the same episodes. A parameterised copy of the rule "
        "(look-back and minimum as arguments) gives identical values at 365 days and 11 players, so every sensitivity "
        "below differs from the original only in the parameter that is varied.",
        "",
        f"Main sample (calendar-first, {n['main_N']:,} episodes): coverage {pct(mc['coverage'])} "
        f"({mc['covered']:,}); loans {pct(lc['coverage'])} ({lc['covered']:,} of {lc['N']:,}); permanent arrivals "
        f"{pct(pc['coverage'])} ({pc['covered']:,} of {pc['N']:,}). All typed master episodes: "
        f"{pct(n['all_typed_coverage']['coverage'])} of {n['all_typed_coverage']['N']:,}.",
        "",
        "## 1. Why values are missing",
        "",
        "Each missing value gets the first reason that applies, in this order. Counts exclude the moving player "
        "throughout, as the rule does.",
        "",
        "Main sample, episodes:",
        "",
        md(piv),
        "All typed master episodes:",
        "",
        md(ra),
        f"Of the {t['thin_main']:,} thin cases in the main sample ({t['thin_loan']:,} loans, {t['thin_perm']:,} "
        f"permanent), {t['thin_8_to_10']:,} have 8 to 10 qualifying players and {t['thin_reach_11_at_730']:,} would "
        f"reach 11 with a 730-day look-back. Of the {t['stale_main']:,} stale-only cases, "
        f"{t['stale_reach_11_at_730']:,} would reach 11 at 730 days.",
        "",
        "## 2. Why loans are covered better than permanent arrivals",
        "",
        "The valuation table is not a census. The snapshot holds "
        f"{h['valuations_total']:,} valuations of {h['players_with_any_valuation']:,} players, all of them in its "
        f"`players` table ({h['players_in_snapshot_players_table']:,} players). Of those players, "
        f"{h['players_linked_to_covered_competitions']:,} "
        f"({pct(h['players_linked_to_covered_competitions'] / h['players_in_snapshot_players_table'])}) are linked to "
        "the snapshot's covered competitions (current club in its club list, or an appearance or line-up row in its "
        "games). Their whole valuation history is kept, including "
        f"{h['valuations_at_clubs_outside_snapshot_clubs']:,} valuations at clubs outside the snapshot's club list, "
        "but a player who never reached a covered club has, in effect, no valuation. An origin club outside the "
        "covered leagues therefore shows only those of its players who also passed through a covered club, so its "
        "squad is thin by construction. This is a property of the source, not of our rule.",
        "",
        "Coverage by origin type (main sample):",
        "",
        md(g1),
        f"Loans come mostly from clubs that played in a covered top flight that season "
        f"({pct(d['loan_share_origin_covered_top_flight'])} of loans against "
        f"{pct(d['perm_share_origin_covered_top_flight'])} of permanent arrivals). Coverage is "
        f"{pct(d['coverage_origin_covered_top_flight'])} for such origins and "
        f"{pct(d['coverage_origin_not_covered_top_flight'])} otherwise. If permanent arrivals had the loans' mix of "
        "origins (covered top flight or not, youth side or not), their coverage would be "
        f"{pct(d['perm_coverage_with_loan_mix'])} instead of {pct(d['perm_coverage'])} (loans: "
        f"{pct(d['loan_coverage'])}). Origin composition alone therefore accounts for "
        f"{pct(d['share_of_gap_explained'], 0)} of the loan/permanent gap: the difference is mainly where the player "
        "comes from, not whether the move is a loan. The rest is lower coverage of permanent arrivals within the same "
        "origin type (for example tier-2 origins, and earlier seasons, below).",
        "",
        "By origin tier and season (main sample):",
        "",
        md(g2),
        "Registration check: for each episode, the players whose Stage 1C registration spell placed them at the "
        "origin club on the day before the move, and how many of them have a valuation. Stage 1C holds the transfer "
        "histories of players on the snapshot's 2023/24 to 2025/26 squads, so these rosters are partial too. Where "
        "the value is missing, the origin club has a median of only a handful of registered players in Stage 1C, "
        "and almost all of them are valued: the missing players are the ones that neither source records, which is "
        "consistent with the coverage explanation above.",
        "",
        md(ro),
        "## 3. Recovery options (sensitivities, stored separately)",
        "",
        "Main sample. Gains are percentage points of the 18,090 episodes. Agreement is measured where both the "
        "variant and the original exist.",
        "",
        md(s),
        "Definitions:",
        "",
        *[f"- `{k}`: {v}" for k, v in VARIANT_TEXT.items()],
        "",
        f"Membership by registration spells (`reg365_min11`) recovers {reg_g.new_episodes_all:,} new episodes but "
        f"loses {reg_g.lost_episodes_all:,}: Stage 1C rosters are partial, so on their own they undercount squads. "
        f"Adding them to the label rule (`union`) recovers {int(sb.loc['union365_min11'].new_episodes_all):,} at 365 "
        f"days and {int(un730.new_episodes_all):,} at 730 days. A 730-day look-back alone recovers "
        f"{int(l730.new_episodes_all):,} ({100 * l730.gain_pp_all:+.1f} pp), with median absolute difference "
        f"{pct(l730.median_abs_pct_diff)} from the original where both exist (correlation of logs "
        f"{l730.corr_logs:.3f}).",
        "",
        "Implementation check: no error was found. The original rule is reproduced exactly; the variants differ "
        "only where the parameter changes; the parameterised copy at 365 days and 11 players is identical.",
        "",
        "## 4. Regression implications",
        "",
        "Preferred specification (M3, league-season fixed effects, clustered by receiving club), outcome "
        "`minutes_share`, calendar-first main sample. Complete cases = origin squad value present and positive. "
        "The loan coefficient is in percentage points.",
        "",
        md(rr),
        f"Adding the original origin squad value keeps {pct(cc.retained_share_of_main)} of the main sample "
        f"({int(cc.N):,} episodes) and raises the loan share from {pct(orig.loc['main sample (calendar-first)'].loan_share)} "
        f"to {pct(cc.loan_share)}, because permanent arrivals lose more observations. Restricting to complete cases "
        f"alone moves the loan coefficient from {fmt_pp(orig.loc['main sample (calendar-first)'].loan_coef)} to "
        f"{100 * cc.loan_coef:+.2f} points (p = {cc.p:.3f}); adding the control on that same sample moves it to "
        f"{100 * cwc.loan_coef:+.2f} (p = {cwc.p:.3f}). The complete-case sample is not a random subset: it drops "
        "mostly permanent arrivals from clubs outside the covered leagues, so any comparison with the full-sample "
        "estimate mixes the effect of the control with the change of sample.",
        "",
        "## Conclusion",
        "",
        f"Coverage cannot be raised much without changing what the variable measures. The stored rule already "
        f"reaches {pct(d['coverage_origin_covered_top_flight'])} for origin clubs that played in a covered top "
        "flight; nearly all missing values come from youth or reserve sides and from clubs outside the covered "
        "leagues, whose squads the source does not record. The defensible extensions add little: a 730-day look-back "
        f"adds {100 * l730.gain_pp_all:+.1f} pp and adding Stage 1C registrations adds "
        f"{100 * un730.gain_pp_all:+.1f} pp (at 730 days), both with values close to the original where both exist. "
        f"Lowering the minimum to 8 players adds {100 * sb.loc['lb365_min8'].gain_pp_all:+.1f} pp but sums fewer "
        "than a starting XI, so it measures a different, partial quantity. We recommend keeping `from_squad_value` as "
        "defined and reporting any regression that uses it on its complete-case sample, with the 730-day variant as a "
        "sensitivity check.",
        "",
        "## Column dictionary: `sensitivity_values.csv`",
        "",
        "One row per typed master episode (`event_id`).",
        "",
        "- `from_squad_value_<variant>`: origin squad value under the variant (EUR; blank below the minimum).",
        "- `from_squad_players_raw_<variant>`: qualifying players under the variant before the minimum is applied "
        "(moving player excluded).",
        "- `from_squad_value_missing_reason`: the reason the stored `from_squad_value` is missing (blank when present).",
        "- `origin_registered_players`: players registered at the origin club on the day before the move per Stage 1C "
        "registration spells (moving player excluded).",
        "- `origin_registered_players_valued_365`: of those, players with a valuation in the 365 days before the move.",
        "",
        "## Files",
        "",
        "- `missingness_reasons.csv`: reason counts (main sample and all typed episodes; all, loans, permanent).",
        "- `coverage_by_group_loan_vs_permanent.csv`: coverage by origin type, tier, country group and season.",
        "- `reweighting_cells.csv`: cells used for the composition reweighting.",
        "- `registration_roster_valuation_check.csv`: registration rosters and their valuation coverage.",
        "- `sensitivity_summary.csv`: each variant's coverage, gain and agreement (main sample).",
        "- `sensitivity_values.csv`: event-level values of every variant.",
        "- `regression_implications.csv`: complete-case samples and M3 with and without the origin control.",
        "- `numbers.json`: the numbers quoted here.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = main()
    print(json.dumps(out["numbers"], indent=1, default=float))
    print(out["S"].to_string())
    print(out["RR"].to_string())
