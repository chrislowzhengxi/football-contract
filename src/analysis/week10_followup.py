"""Week 10 follow-up (meeting with Professors Freund and Fernandes): tier-2 playing-time feasibility, origin-tier
recovery, tier-movement statistics and the two 29 x 29 transfer-flow heatmaps.

    python -m src.analysis.week10_followup

Inputs (read-only): the 227-column master, the Week 10 club-season lookup built by league_tiers_wiki
(data/processed/league_tiers_v2/), the Week 9 lookup and conflicts, the Transfermarkt snapshot.
Writes only data/outputs/rebuild/professor_feedback_week10/ (the master, Stage 1C and every stored output are
untouched). Every number in the generated markdown comes from this run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import club_organisation as co
from . import feedback_nov2026 as fb

ROOT = Path(__file__).resolve().parents[2]
REB = ROOT / "data" / "outputs" / "rebuild"
MASTER = REB / "analysis_episode_master.csv"
V2 = ROOT / "data" / "processed" / "league_tiers_v2"
V1 = ROOT / "data" / "processed" / "league_tiers"
OUT = REB / "professor_feedback_week10"
STAGE1C_SHA = "92f383d72a3ed59925c922f0d1b5f98528bc66ad3a9f49bb990867ed1d1d50dd"

FOURTEEN = ["Belgium", "Denmark", "England", "France", "Germany", "Greece", "Italy", "Netherlands", "Portugal",
            "Russia", "Scotland", "Spain", "Türkiye", "Ukraine"]
DISPLAY = {"Türkiye": "Turkey"}
CATS = [f"{DISPLAY.get(c, c)} T{t}" for c in FOURTEEN for t in (1, 2)] + ["Other"]
PERIOD = (2012, 2025)     # reference seasons of the main tables and heatmaps (2012/13 to 2025/26)
PLACEHOLDER = set(fb.PLACEHOLDER_CLUB_IDS)


# --------------------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------------------
def load():
    import duckdb
    from ..config import DEFAULT_DATABASE
    M = pd.read_csv(MASTER, low_memory=False)
    assert M.shape == (115044, 227) and M.event_id.is_unique
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        sides = fb.league_sides(con)
    finally:
        con.close()
    L1 = fb.tier_lookup()
    L2 = pd.read_csv(V2 / "club_season_league_v2.csv")
    L2 = L2.astype({"club_id": "int64", "season": "int64", "tier": "int64"})
    C = pd.read_csv(V2 / "league_list_completeness.csv")
    conf = set()
    for p, cols in ((V1 / "club_season_conflicts.csv", ("club_id", "season")),
                    (V2 / "wiki_internal_conflicts.csv", ("club_id", "season"))):
        d = pd.read_csv(p)
        if len(d):
            conf |= set(map(tuple, d[list(cols)].dropna().astype("int64").to_numpy()))
    ov = pd.read_csv(V2 / "wiki_vs_week9_overlap.csv")
    conf |= set(map(tuple, ov.loc[~ov.agrees.astype(bool), ["club_id", "season"]].astype("int64").to_numpy()))
    return M, sides, L1, L2, C, conf


# --------------------------------------------------------------------------------------------------------------------
# Tiers v2
# --------------------------------------------------------------------------------------------------------------------
def complete_sets(C: pd.DataFrame) -> tuple[set, set]:
    """(country, season) pairs with a full division list (at least 8 teams) for tier 1 and for tier 2."""
    c = C[C.teams >= 8]
    t1 = set(map(tuple, c[c.tier == 1][["country", "season"]].to_numpy()))
    t2 = set(map(tuple, c[c.tier == 2][["country", "season"]].to_numpy()))
    return t1, t2


def _distinct(name: str) -> set:
    from . import league_tiers as lt
    return {t for t in lt.norm_keys(name)[1].split(" #")[0].split() if len(t) >= 3 and not t.isdigit()}


SIM_MAX = 0.72     # string similarity at or above which an unmatched list name may be the club (transliterations)


def _norm(name: str) -> str:
    from . import league_tiers as lt
    return " ".join(lt.norm_keys(name)[1].split(" #")[0].split())


def unmatched_tokens() -> dict:
    """(country, season, tier) -> list of (token set, normalised names, is_reserve), one per listed team that was NOT
    matched to a club id (all name variants pooled). Wikipedia lists and the Week 9 openfootball lists."""
    from . import league_tiers as lt
    from . import league_tiers_wiki as w
    out: dict = {}
    A = pd.read_csv(V2 / "wiki_name_match_audit.csv")
    # teams whose match was dropped as a conflict (club in two divisions; disagreement with Week 9) are treated as
    # unmatched: their identity is uncertain (e.g. a reserve side linked to its parent club's article)
    W = pd.read_csv(V2 / "wiki_internal_conflicts.csv")
    ov = pd.read_csv(V2 / "wiki_vs_week9_overlap.csv")
    drop = set(map(tuple, W[["code", "season", "text"]].to_numpy())) | \
        set(map(tuple, ov.loc[~ov.agrees.astype(bool), ["code", "season", "text"]].to_numpy()))
    keyed = pd.Series([(c, s_, t) in drop for c, s_, t in zip(A.code, A.season, A.text)], index=A.index)
    A = A[A.club_id.isna() | keyed].drop_duplicates(["code", "season", "text", "title"])
    for r in A.itertuples(index=False):
        v = w.variants(r)
        toks = set().union(*(_distinct(x) for x in v))
        out.setdefault((r.country, int(r.season), int(r.tier)), []).append(
            (toks, {_norm(x) for x in v}, any(lt.is_reserve_name(x) for x in v)))
    O = pd.read_csv(V1 / "league_season_coverage.csv")
    O = O[O.status.eq("ok") & O.unmatched_names.notna()]
    for r in O.itertuples(index=False):
        for n in str(r.unmatched_names).split("; "):
            out.setdefault((r.country, int(r.season), int(r.tier)), []).append(
                (_distinct(n), {_norm(n)}, lt.is_reserve_name(n)))
    return out


def club_tokens() -> dict:
    """club id -> (distinctive tokens, normalised names, is_reserve)."""
    from . import league_tiers as lt
    con = lt.connect()
    try:
        tm = lt.tm_universe(con, lt.load_master())
    finally:
        con.close()
    toks: dict = {}
    names: dict = {}
    for cid, nm, loose in zip(tm["names"].club_id, tm["names"].name, tm["names"].loose):
        toks.setdefault(int(cid), set()).update(
            t for t in loose.split(" #")[0].split() if len(t) >= 3 and not t.isdigit())
        names.setdefault(int(cid), set()).add(_norm(nm))
    return {i: (toks[i], names[i], tm["reserve"].get(i, False)) for i in toks}


def make_clash(unm: dict, ctok: dict):
    """clash(club, country, season, tiers): could the club be one of the unmatched teams of those lists? Yes if it
    shares a distinctive name token with one, or (same reserve status) a name similarity >= SIM_MAX."""
    import difflib
    cache: dict = {}

    def clash(c, k, x, tiers):
        key = (int(c), k, int(x), tiers)
        if key in cache:
            return cache[key]
        ct, cn, cres = ctok.get(int(c), (set(), set(), False))
        hit = False
        for t in tiers:
            for ts, un, ures in unm.get((k, int(x), t), []):
                if ct & ts:
                    hit = True
                elif ures == cres:
                    for a in cn:
                        for b in un:
                            sm = difflib.SequenceMatcher(None, a, b)
                            if sm.real_quick_ratio() >= SIM_MAX and sm.quick_ratio() >= SIM_MAX and \
                                    sm.ratio() >= SIM_MAX:
                                hit = True
                                break
                        if hit:
                            break
                if hit:
                    break
            if hit:
                break
        cache[key] = hit
        return hit
    return clash


def tiers_v2(M, sides, L1, L2, C, conf, rule: str = "per_club") -> pd.DataFrame:
    """rule 'per_club' (primary): "not tier 1 (or 2)" when the club's country has the season's tier-1 (and tier-2)
    list, the club is not in it, and it cannot be one of the lists' unmatched teams (make_clash).
    rule 'complete_lists' (strict sensitivity): additionally every list involved must be complete (every listed team
    matched, or unmatched teams sharing no name token with any club of that country in our data)."""
    youth = fb.youth_club_ids(M, co.load_map())
    covered = fb.ptj_covered(sides)
    ref = M.tier_reference_season.astype("Int64")
    # reproduce the stored Week 9 columns exactly with the Week 9 lookup (guards against drift)
    T1 = fb.league_tiers(M, ref, sides, covered, youth, L1)
    for side in ("from", "to"):
        for c in ("league_tier_at_move", "league_tier_status", "league_name_at_move"):
            a, b = T1[f"{side}_{c}"].astype(object), M[f"{side}_{c}"].astype(object)
            same = (a == b) | (pd.isna(a) & pd.isna(b))
            assert same.all(), f"Week 9 {side}_{c} not reproduced ({(~same).sum()} rows)"
    T = fb.league_tiers(M, ref, sides, covered, youth, L2)
    t1, t2 = complete_sets(C)
    if rule == "complete_lists":
        cc = C[C.complete.astype(bool)]
        t1 &= set(map(tuple, cc[cc.tier == 1][["country", "season"]].to_numpy()))
        t2 &= set(map(tuple, cc[cc.tier == 2][["country", "season"]].to_numpy()))
    else:
        assert rule == "per_club", rule
    unm, ctok = unmatched_tokens(), club_tokens()
    clash = make_clash(unm, ctok)
    out = pd.DataFrame({"event_id": M.event_id.to_numpy(), "tier_reference_season": ref.to_numpy()})
    for side in ("from", "to"):
        club = M[f"{side}_club_id"].astype("int64").to_numpy()
        ctry = M[f"{side}_country_v2"].to_numpy()
        st = T[f"{side}_league_tier_status"].to_numpy().astype(object)
        src = T[f"{side}_league_tier_source"].to_numpy()
        tier = T[f"{side}_league_tier_at_move"]
        old = M[f"{side}_league_tier_status"].to_numpy()
        # Week 9 verified rows must be unchanged in v2 (old rows always win)
        wv = old == "verified"
        assert (tier[wv].to_numpy() == M.loc[wv, f"{side}_league_tier_at_move"].to_numpy()).all()
        s = ref.fillna(-1).astype("int64").to_numpy()
        status = np.where(st == "verified", np.where(src == "wikipedia", "verified_week10_wikipedia",
                                                     "verified_week9"), st)
        unk = np.isin(st, ["unknown_division_not_in_sources", "youth_or_reserve_side_division_unknown",
                           "unknown_season_outside_sources"])
        inconf = np.array([(c, x) in conf for c, x in zip(club, s)])

        k12 = np.array([(k, x) in t1 and (k, x) in t2 and not clash(c, k, x, (1, 2))
                        for c, k, x in zip(club, ctry, s)])
        k1 = np.array([(k, x) in t1 and not clash(c, k, x, (1,)) for c, k, x in zip(club, ctry, s)])
        status = np.where(unk & inconf, "unknown_conflicting_sources", status)
        status = np.where(unk & ~inconf & k12, "not_tier_1_or_2_absent_from_lists", status)
        status = np.where(unk & ~inconf & ~k12 & k1, "not_tier_1_absent_from_list", status)
        out[f"{side}_league_tier_v2"] = pd.array(tier, dtype="Int64")
        out[f"{side}_league_name_v2"] = T[f"{side}_league_name_at_move"].to_numpy()
        out[f"{side}_league_country_v2"] = T[f"{side}_league_country_at_move"].to_numpy()
        out[f"{side}_league_tier_source_v2"] = src
        out[f"{side}_league_tier_status_v2"] = status
        out[f"{side}_week9_status"] = old
        # tier band and a lower bound for comparisons
        tv = tier.to_numpy(dtype="float", na_value=np.nan)
        isyouth = np.isin(club, list(youth))
        band = np.select([tv == 1, tv == 2, tv >= 3,
                          (status == "not_tier_1_or_2_absent_from_lists") & isyouth,
                          status == "not_tier_1_or_2_absent_from_lists",
                          status == "not_tier_1_absent_from_list"],
                         ["1", "2", "3+", "youth/reserve", "3+", "not 1"], "unknown")
        out[f"{side}_tier_band_v2"] = band
        lb = np.select([~np.isnan(tv), band == "3+", band == "not 1"], [tv, 3.0, 2.0], np.nan)
        out[f"{side}_tier_exact_v2"] = ~np.isnan(tv)
        out[f"{side}_tier_lower_bound_v2"] = lb
        # country of the side: league country when verified, else the club's country
        lc = out[f"{side}_league_country_v2"].to_numpy()
        country = np.where(pd.notna(lc), lc, ctry)
        out[f"{side}_country_for_flows"] = country
        # heatmap category
        placeholder = np.isin(club, list(PLACEHOLDER))
        in14 = np.isin(country, FOURTEEN)
        cat = np.full(len(out), None, dtype=object)
        why = np.full(len(out), None, dtype=object)
        for i in range(len(out)):
            if placeholder[i]:
                why[i] = "excluded: no club (without club, career break, retired, unknown, ban)"
            elif band[i] in ("1", "2") and in14[i]:
                cat[i] = f"{DISPLAY.get(country[i], country[i])} T{band[i]}"
                why[i] = "fourteen-country tier 1 or 2, verified"
            elif band[i] == "youth/reserve" and in14[i]:
                cat[i] = "Other"
                why[i] = "Other: fourteen-country youth or reserve side absent from that season's tier-1 and tier-2 lists"
            elif band[i] == "3+" and in14[i]:
                cat[i] = "Other"
                why[i] = ("Other: fourteen-country club verified in tier 3 or lower" if not np.isnan(tv[i]) else
                          "Other: fourteen-country club absent from that season's tier-1 and tier-2 lists")
            elif pd.notna(country[i]) and not in14[i]:
                cat[i] = "Other"
                why[i] = "Other: club outside the fourteen countries"
            elif in14[i]:
                why[i] = "unclassifiable: fourteen-country club, tier 1 or 2 cannot be ruled out"
            else:
                why[i] = "unclassifiable: club country unknown"
        out[f"{side}_flow_category"] = cat
        out[f"{side}_flow_category_reason"] = why
    return out


# --------------------------------------------------------------------------------------------------------------------
# Q2: origin-tier missingness and recovery (main sample)
# --------------------------------------------------------------------------------------------------------------------
def q2_origin(M: pd.DataFrame, X: pd.DataFrame, C: pd.DataFrame) -> dict:
    main = M.in_cf_main_model_sample.astype(bool)
    D = X[main.to_numpy()].copy()
    D["kind"] = M.loc[main, "kind"].to_numpy()
    D["from_country_v2"] = M.loc[main, "from_country_v2"].to_numpy()
    D["from_club_name"] = M.loc[main, "from_club_name"].to_numpy()
    n_main = len(D)
    miss = D[D.from_week9_status != "verified"].copy()
    in14 = miss.from_country_v2.isin(FOURTEEN)
    youth = miss.from_week9_status.eq("youth_or_reserve_side_division_unknown")
    # is there any list at all for the club's country in that season (tier 1 or 2), and was it fully matched?
    Cs = C.groupby(["country", "season", "tier"]).agg(complete=("complete", "max"), any=("teams", "size"))
    anylist = {(c, s) for c, s, t in Cs.index}
    has_list = np.array([(k, s) in anylist for k, s in zip(miss.from_country_v2, miss.tier_reference_season)])
    miss["week9_reason"] = np.select(
        [miss.from_country_v2.isna(), ~in14 & youth, ~in14, in14 & youth],
        ["club country unknown", "outside the 14 countries: youth or reserve side",
         "outside the 14 countries: senior club", "within the 14 countries: youth or reserve side"],
        "within the 14 countries: senior club, lower-division list missing")
    miss["country_had_any_list_week10"] = has_list
    miss["outcome_v2"] = np.where(
        miss.from_tier_band_v2.eq("youth/reserve"),
        "recovered: youth or reserve side, not in tier 1 or 2 (absent from the lists)", None)
    miss["outcome_v2"] = miss.outcome_v2.fillna(miss.from_league_tier_status_v2.map({
        "verified_week10_wikipedia": "recovered: exact tier verified (Wikipedia list)",
        "not_tier_1_or_2_absent_from_lists": "recovered: not tier 1 or 2 (absent from the lists)",
        "not_tier_1_absent_from_list": "partly recovered: not tier 1 (absent from the top-flight list)",
    })).fillna("still unknown")
    miss["still_unknown_reason"] = np.where(
        miss.outcome_v2.eq("still unknown"),
        np.select([miss.from_league_tier_status_v2.eq("unknown_conflicting_sources"), miss.from_country_v2.isna(),
                   ~miss.country_had_any_list_week10],
                  ["sources disagree for this club-season", "club country unknown",
                   "no league list for the club's country in that season"],
                  "lists exist, but the club may be one of their unmatched names (shared name token)"), "")
    reason_tab = pd.crosstab(miss.week9_reason, miss.outcome_v2, margins=True, margins_name="Total")
    kind_tab = pd.crosstab([miss.kind], miss.outcome_v2, margins=True, margins_name="Total")
    ctry = miss.groupby("from_country_v2").agg(
        missing_week9=("event_id", "size"),
        recovered_exact=("outcome_v2", lambda s: s.str.startswith("recovered: exact").sum()),
        recovered_not_1_2=("outcome_v2", lambda s: s.str.startswith("recovered: not").sum()),
        recovered_youth_reserve=("outcome_v2", lambda s: s.str.startswith("recovered: youth").sum()),
        partly_not_1=("outcome_v2", lambda s: s.str.startswith("partly").sum()),
        still_unknown=("outcome_v2", lambda s: s.eq("still unknown").sum())).sort_values("missing_week9",
                                                                                         ascending=False)
    still = miss[miss.outcome_v2.eq("still unknown")].groupby(["still_unknown_reason"]).size()
    top_still = (miss[miss.outcome_v2.eq("still unknown")].groupby(["from_country_v2", "from_club_name"]).size()
                 .sort_values(ascending=False).head(25).rename("episodes").reset_index())
    # tier distribution after recovery
    exact = D.from_tier_exact_v2.astype(bool)
    after = D.from_tier_band_v2.value_counts()
    N = {"n_main": n_main, "missing_week9": len(miss),
         "verified_week9": int((D.from_week9_status == "verified").sum()),
         "recovered_exact": int(miss.outcome_v2.str.startswith("recovered: exact").sum()),
         "recovered_not_1_2": int(miss.outcome_v2.str.startswith("recovered: not").sum()),
         "recovered_youth_reserve": int(miss.outcome_v2.str.startswith("recovered: youth").sum()),
         "partly_not_1": int(miss.outcome_v2.str.startswith("partly").sum()),
         "still_unknown": int(miss.outcome_v2.eq("still unknown").sum()),
         "exact_tier_v2": int(exact.sum()),
         "band_known_v2": int(D.from_tier_band_v2.isin(["1", "2", "3+", "youth/reserve"]).sum()),
         "band_counts_v2": {k: int(v) for k, v in after.items()},
         "exact_tier_counts_v2": {int(k): int(v) for k, v in D.from_league_tier_v2.value_counts().sort_index().items()},
         "week9_reason": miss.week9_reason.value_counts().to_dict(),
         "still_unknown_reason": {k: int(v) for k, v in still.items()},
         "by_kind": {k: {"main": int((D.kind == k).sum()),
                         "verified_week9": int(((D.kind == k) & (D.from_week9_status == "verified")).sum()),
                         "exact_v2": int(((D.kind == k) & exact).sum()),
                         "band_v2": int(((D.kind == k) & D.from_tier_band_v2.isin(["1", "2", "3+", "youth/reserve"])).sum())}
                     for k in ("loan", "permanent")}}
    return {"N": N, "miss": miss, "reason_tab": reason_tab, "kind_tab": kind_tab, "country": ctry,
            "top_still": top_still}


# --------------------------------------------------------------------------------------------------------------------
# Q3: movement between tiers (full master, reference seasons in PERIOD)
# --------------------------------------------------------------------------------------------------------------------
def direction(f_lb, f_ex, t_lb, t_ex) -> np.ndarray:
    """down = destination at a lower level (larger tier number); up = higher; same; indeterminate."""
    out = np.full(len(f_lb), "unknown", dtype=object)
    both = ~np.isnan(f_lb) & ~np.isnan(t_lb)
    ex = both & f_ex & t_ex
    out[ex & (t_lb > f_lb)] = "down"
    out[ex & (t_lb < f_lb)] = "up"
    out[ex & (t_lb == f_lb)] = "same level"
    # one side only bounded below (e.g. "3+"): decidable only when the exact side is above the bound
    fb_ = both & f_ex & ~t_ex
    out[fb_ & (f_lb < t_lb)] = "down"
    out[fb_ & (f_lb >= t_lb)] = "indeterminate"
    tb_ = both & ~f_ex & t_ex
    out[tb_ & (t_lb < f_lb)] = "up"
    out[tb_ & (t_lb >= f_lb)] = "indeterminate"
    out[both & ~f_ex & ~t_ex] = "indeterminate"
    return out


def q3_movement(M: pd.DataFrame, X: pd.DataFrame) -> dict:
    D = X.assign(kind=M.kind.to_numpy(), season=M.season.to_numpy())
    D = D[D.kind.isin(["loan", "permanent"])]
    s = D.tier_reference_season.astype("float")
    D = D[(s >= PERIOD[0]) & (s <= PERIOD[1])].copy()
    D["dir"] = direction(D.from_tier_lower_bound_v2.to_numpy(float), D.from_tier_exact_v2.to_numpy(bool),
                         D.to_tier_lower_bound_v2.to_numpy(float), D.to_tier_exact_v2.to_numpy(bool))
    fc, tc = D.from_country_for_flows, D.to_country_for_flows
    D["border"] = np.select([fc.isna() | tc.isna(), fc == tc], ["country unknown", "domestic"], "cross-border")
    D["pair"] = D.from_tier_band_v2 + " -> " + D.to_tier_band_v2
    both_known = D.from_tier_band_v2.isin(["1", "2", "3+"]) & D.to_tier_band_v2.isin(["1", "2", "3+"])
    D["both_known"] = both_known
    tabs = {}
    rows = []
    for k in ("loan", "permanent"):
        d = D[D.kind == k]
        kk = d[d.both_known]
        n_all, n_k = len(d), len(kk)
        def cnt(mask):
            return int(mask.sum())
        spec = [("Tier 1 -> Tier 1", (kk.from_tier_band_v2 == "1") & (kk.to_tier_band_v2 == "1")),
                ("Tier 1 -> Tier 2", (kk.from_tier_band_v2 == "1") & (kk.to_tier_band_v2 == "2")),
                ("Tier 2 -> Tier 1", (kk.from_tier_band_v2 == "2") & (kk.to_tier_band_v2 == "1")),
                ("Tier 2 -> Tier 2", (kk.from_tier_band_v2 == "2") & (kk.to_tier_band_v2 == "2")),
                ("Tier 1 -> Tier 3 or below", (kk.from_tier_band_v2 == "1") & (kk.to_tier_band_v2 == "3+")),
                ("Tier 2 -> Tier 3 or below", (kk.from_tier_band_v2 == "2") & (kk.to_tier_band_v2 == "3+")),
                ("Tier 3 or below -> Tier 1", (kk.from_tier_band_v2 == "3+") & (kk.to_tier_band_v2 == "1")),
                ("Tier 3 or below -> Tier 2", (kk.from_tier_band_v2 == "3+") & (kk.to_tier_band_v2 == "2")),
                ("Tier 3 or below -> Tier 3 or below", (kk.from_tier_band_v2 == "3+") & (kk.to_tier_band_v2 == "3+"))]
        for lab, m in spec:
            rows.append({"kind": k, "group": "tier pair (both tiers known)", "category": lab, "count": cnt(m),
                         "denominator": n_k, "pct": 100 * cnt(m) / n_k if n_k else np.nan})
        for lab, m in (("Youth/reserve side -> Tier 1", (d.from_tier_band_v2 == "youth/reserve") & (d.to_tier_band_v2 == "1")),
                       ("Youth/reserve side -> Tier 2", (d.from_tier_band_v2 == "youth/reserve") & (d.to_tier_band_v2 == "2")),
                       ("Youth/reserve side -> Tier 3 or below", (d.from_tier_band_v2 == "youth/reserve") & (d.to_tier_band_v2 == "3+")),
                       ("Tier 1 -> youth/reserve side", (d.from_tier_band_v2 == "1") & (d.to_tier_band_v2 == "youth/reserve"))):
            rows.append({"kind": k, "group": "youth or reserve sides (all episodes)", "category": lab, "count": cnt(m),
                         "denominator": n_all, "pct": 100 * cnt(m) / n_all})
        dd = d[d.dir.isin(["up", "down", "same level"])]
        for lab in ("down", "same level", "up"):
            m = dd.dir == lab
            rows.append({"kind": k, "group": "direction (decidable)", "category": {
                "down": "Higher division -> lower division", "up": "Lower division -> higher division",
                "same level": "Same level"}[lab], "count": cnt(m), "denominator": len(dd),
                "pct": 100 * cnt(m) / len(dd) if len(dd) else np.nan})
        db = d[d.border != "country unknown"]
        for lab in ("domestic", "cross-border"):
            m = db.border == lab
            rows.append({"kind": k, "group": "country (both known)", "category": lab, "count": cnt(m),
                         "denominator": len(db), "pct": 100 * cnt(m) / len(db) if len(db) else np.nan})
        for lab, m in (("both tiers known", d.both_known), ("direction decidable", d.dir.isin(["up", "down", "same level"])),
                       ("both countries known", d.border != "country unknown")):
            rows.append({"kind": k, "group": "coverage", "category": lab, "count": cnt(m), "denominator": n_all,
                         "pct": 100 * cnt(m) / n_all})
        tabs[k] = pd.crosstab(kk.from_tier_band_v2, kk.to_tier_band_v2)
    S = pd.DataFrame(rows)
    # direction by domestic / cross-border
    dec = D[D.dir.isin(["up", "down", "same level"]) & D.border.isin(["domestic", "cross-border"])]
    by_border = (dec.groupby(["kind", "border", "dir"]).size().unstack("dir").fillna(0).astype(int))
    by_border["n"] = by_border.sum(axis=1)
    for c in ("down", "same level", "up"):
        by_border[f"{c}_pct"] = 100 * by_border[c] / by_border.n
    # loans and permanent moves leaving a fourteen-country tier-1 club: where do they go?
    o1 = D[(D.from_tier_band_v2 == "1") & D.from_country_for_flows.isin(FOURTEEN)]
    dest = (o1.groupby(["kind", "to_tier_band_v2"]).size().unstack("kind").fillna(0).astype(int))
    dest_pct = 100 * dest / dest.sum()
    # by origin country: share of decidable moves from tier 1 that go down
    oc = o1[o1.dir.isin(["up", "down", "same level"])]
    by_ctry = oc.groupby(["from_country_for_flows", "kind"]).dir.agg(n="size", down=lambda x: (x == "down").sum())
    by_ctry["down_pct"] = 100 * by_ctry.down / by_ctry.n
    by_ctry = by_ctry.unstack("kind")
    # by season period
    D["period"] = pd.cut(D.tier_reference_season.astype(float), [2011, 2015, 2019, 2022, 2025],
                         labels=["2012-15", "2016-19", "2020-22", "2023-25"])
    dp = D[D.dir.isin(["up", "down", "same level"])].groupby(["kind", "period"], observed=True).dir.agg(
        n="size", down=lambda x: (x == "down").sum(), up=lambda x: (x == "up").sum())
    dp["down_pct"], dp["up_pct"] = 100 * dp.down / dp.n, 100 * dp.up / dp.n
    return {"summary": S, "pair_tabs": tabs, "by_border": by_border, "dest_from_t1": dest,
            "dest_from_t1_pct": dest_pct, "by_country": by_ctry, "by_period": dp, "D": D}


# --------------------------------------------------------------------------------------------------------------------
# Q4: 29 x 29 flow matrices and heatmaps
# --------------------------------------------------------------------------------------------------------------------
def flow_matrix(D: pd.DataFrame) -> pd.DataFrame:
    ok = D.from_flow_category.notna() & D.to_flow_category.notna()
    m = pd.crosstab(D.loc[ok, "from_flow_category"], D.loc[ok, "to_flow_category"])
    return m.reindex(index=CATS, columns=CATS, fill_value=0).astype(int)


def category_coverage(C: pd.DataFrame) -> pd.DataFrame:
    """Seasons of PERIOD with a division list for each fourteen-country tier (needed for its row/column)."""
    t1, t2 = complete_sets(C)
    rows = []
    for c in FOURTEEN:
        for t, ss in ((1, t1), (2, t2)):
            seasons = [s for s in range(PERIOD[0], PERIOD[1] + 1) if (c, s) in ss]
            rows.append({"category": f"{DISPLAY.get(c, c)} T{t}", "country": c, "tier": t,
                         "seasons_with_list": len(seasons), "seasons_in_period": PERIOD[1] - PERIOD[0] + 1,
                         "seasons": " ".join(map(str, seasons))})
    return pd.DataFrame(rows)


def plot_heatmap(mat: pd.DataFrame, title: str, subtitle: str, cov: pd.DataFrame, vmax: float, path_base: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue_seq", ramp)
    ink, muted, other_fill, zero_fill, na_fill = "#1f1f1e", "#6b6a66", "#e8e7e3", "#ffffff", "#f4f3f0"
    n = len(CATS)
    nodata = set(cov.loc[cov.seasons_with_list == 0, "category"])
    partial = set(cov.loc[(cov.seasons_with_list > 0) & (cov.seasons_with_list < cov.seasons_in_period), "category"])
    norm = Normalize(vmin=0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(17, 16.5))
    for i, r in enumerate(CATS):
        for j, c in enumerate(CATS):
            v = int(mat.loc[r, c])
            is_other = r == "Other" or c == "Other"
            is_na = r in nodata or c in nodata
            if is_na:
                fc = na_fill
            elif is_other:
                fc = other_fill
            elif v == 0:
                fc = zero_fill
            else:
                fc = cmap(norm(np.log1p(v)))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="#ffffff", linewidth=1.2,
                                   hatch="////" if is_na else None))
            if is_na and v == 0:
                ax.text(j + .5, i + .5, "n/a", ha="center", va="center", fontsize=6.5, color=muted)
                continue
            lum = None
            if not is_other and v > 0:
                rgb = matplotlib.colors.to_rgb(fc)
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            col = "#ffffff" if (lum is not None and lum < 0.45) else (muted if v == 0 else ink)
            ax.text(j + .5, i + .5, f"{v:,}", ha="center", va="center", fontsize=6.6 if v < 1000 else 5.9,
                    color=col, fontweight="bold" if is_other else "normal")
    # country separators
    for k in range(0, 29, 2):
        ax.axhline(k, color="#9a9893", lw=0.8)
        ax.axvline(k, color="#9a9893", lw=0.8)
    ax.axhline(28, color=ink, lw=1.6)
    ax.axvline(28, color=ink, lw=1.6)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    lab = [c + (" *" if c in partial else "") + (" \u2020" if c in nodata else "") for c in CATS]
    ax.set_xticks(np.arange(n) + .5)
    ax.set_yticks(np.arange(n) + .5)
    ax.xaxis.tick_top()
    ax.set_xticklabels(lab, rotation=55, ha="left", rotation_mode="anchor", fontsize=8.5, color=ink)
    ax.set_yticklabels(lab, fontsize=8.5, color=ink)
    ax.tick_params(length=0)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("TO (receiving club)", fontsize=10.5, color=ink, labelpad=10)
    ax.set_ylabel("FROM (origin club)", fontsize=10.5, color=ink, labelpad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.suptitle(title, x=0.06, ha="left", fontsize=15, color=ink, y=0.985)
    fig.text(0.06, 0.957, subtitle, ha="left", fontsize=9.2, color=muted, va="top", linespacing=1.45)
    # legend row under the subtitle: colour bar in log(1 + count) with raw-count tick labels, then swatches
    ly = 0.885
    cax = fig.add_axes([0.06, ly, 0.16, 0.011])
    grad = np.linspace(0, vmax, 256)[None, :]
    cax.imshow(grad, aspect="auto", cmap=cmap, extent=[0, vmax, 0, 1])
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000, 3000) if np.log1p(t) <= vmax]
    cax.set_xticks([np.log1p(t) for t in ticks])
    cax.set_xticklabels([f"{t:,}" for t in ticks], fontsize=7.5, color=ink)
    cax.set_yticks([])
    cax.set_title("Colour: log(1 + count). Numbers in cells: raw counts.", fontsize=7.8, color=ink, loc="left")
    for sp in cax.spines.values():
        sp.set_visible(False)
    items = [(zero_fill, None, "0 = no transfers"),
             (other_fill, None, "Other row/column: raw counts, not on the colour scale"),
             (na_fill, "////", "\u2020 no division list in any season: counts are lower bounds; n/a = none verified")]
    if not nodata:
        items = items[:2]
    if "Other" not in CATS:
        items = [it for it in items if it[0] != other_fill]
    xs = [0.27, 0.41, 0.66]
    for x, (fc, h, txt) in zip(xs, items):
        bx = fig.add_axes([x, ly, 0.011, 0.011])
        bx.add_patch(Rectangle((0, 0), 1, 1, facecolor=fc, edgecolor="#9a9893", lw=0.6, hatch=h))
        bx.set_axis_off()
        fig.text(x + 0.015, ly + 0.0055, txt, fontsize=7.6, color=ink, va="center")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.795, bottom=0.015)
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=220 if ext == "png" else None)
    plt.close(fig)


def q4_flows(M: pd.DataFrame, X: pd.DataFrame, C: pd.DataFrame) -> dict:
    D = X.assign(kind=M.kind.to_numpy())
    D = D[D.kind.isin(["loan", "permanent"])]
    s = D.tier_reference_season.astype("float")
    D = D[(s >= PERIOD[0]) & (s <= PERIOD[1])]
    cov = category_coverage(C)
    out = {"coverage": cov, "matrices": {}, "accounting": {}}
    for k in ("loan", "permanent"):
        d = D[D.kind == k]
        mat = flow_matrix(d)
        excl = d[d.from_flow_category.isna() | d.to_flow_category.isna()]
        why = []
        for side in ("from", "to"):
            w = excl[excl[f"{side}_flow_category"].isna()][f"{side}_flow_category_reason"].value_counts()
            for r, v in w.items():
                why.append({"side": side, "reason": r, "episodes": int(v)})
        out["matrices"][k] = mat
        out["accounting"][k] = {"episodes_in_period": len(d), "in_matrix": int(mat.to_numpy().sum()),
                                "omitted": len(excl), "omitted_reasons": why,
                                "core_28x28": int(mat.iloc[:28, :28].to_numpy().sum()),
                                "other_row_or_col": int(mat.to_numpy().sum() - mat.iloc[:28, :28].to_numpy().sum()),
                                "other_to_other": int(mat.loc["Other", "Other"])}
    vmax = max(np.log1p(m.iloc[:28, :28].to_numpy().max()) for m in out["matrices"].values())
    out["vmax"] = float(vmax)
    return out


# --------------------------------------------------------------------------------------------------------------------
# Q1: tier-2 playing-time feasibility
# --------------------------------------------------------------------------------------------------------------------
def q1_tier2(M: pd.DataFrame, X: pd.DataFrame, C: pd.DataFrame, sides: pd.DataFrame) -> dict:
    import duckdb
    from ..config import DEFAULT_DATABASE
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        comps = con.sql("select competition_id, sub_type, country_name from competitions "
                        "where type = 'domestic_league'").df()
        lg = con.sql("select competition_id, count(*) games, min(cast(season as int)) s0, "
                     "max(cast(season as int)) s1 from games "
                     "where competition_type = 'domestic_league' group by 1").df()
    finally:
        con.close()
    snap14 = comps[comps.country_name.isin(FOURTEEN)].merge(lg, on="competition_id", how="left")
    t1, t2 = complete_sets(C)
    grid = []
    for c in FOURTEEN:
        for s in range(PERIOD[0], PERIOD[1] + 1):
            grid.append({"country": c, "season": s, "tier2_list_available": (c, s) in t2,
                         "tier2_match_level_appearances_in_snapshot": False,
                         "tier1_match_level_appearances_in_snapshot": bool(
                             ((snap14.country_name == c) & (snap14.s0 <= s) & (snap14.s1 >= s)).any())})
    G = pd.DataFrame(grid)
    D = X.assign(kind=M.kind.to_numpy(), in_revised=M.in_revised_sample.astype(bool).to_numpy(),
                 main=M.in_cf_main_model_sample.astype(bool).to_numpy())
    D = D[D.kind.isin(["loan", "permanent"])]
    s = D.tier_reference_season.astype(float)
    D = D[(s >= PERIOD[0]) & (s <= PERIOD[1])]
    to14 = D.to_country_for_flows.isin(FOURTEEN)
    t2 = D[to14 & (D.to_tier_band_v2 == "2")]
    t1 = D[to14 & (D.to_tier_band_v2 == "1")]
    pot = t2.groupby(["to_country_for_flows", "kind"]).size().unstack("kind").fillna(0).astype(int)
    pot["total"] = pot.sum(axis=1)
    pass_rate = t1.groupby("kind").in_revised.mean()
    cur = M[M.in_revised_sample.astype(bool)].kind.value_counts()
    proj = (t2.kind.value_counts() * pass_rate).round()
    N = {"snapshot_domestic_league_competitions": int(len(comps)),
         "snapshot_sub_types": comps.sub_type.value_counts().to_dict(),
         "tier2_episodes_14": {k: int(v) for k, v in t2.kind.value_counts().items()},
         "tier1_episodes_14": {k: int(v) for k, v in t1.kind.value_counts().items()},
         "tier1_pass_rate_to_revised": {k: float(v) for k, v in pass_rate.items()},
         "revised_sample": {k: int(v) for k, v in cur.items()},
         "projected_tier2_additions": {k: int(v) for k, v in proj.items()},
         "tier2_list_available_country_seasons": int(G.tier2_list_available.sum()), "country_seasons": len(G)}
    rs = N["revised_sample"]
    N["loan_share_revised"] = rs["loan"] / (rs["loan"] + rs["permanent"])
    pr = N["projected_tier2_additions"]
    N["loan_share_projected_additions"] = pr.get("loan", 0) / max(1, pr.get("loan", 0) + pr.get("permanent", 0))
    N["loan_share_projected_combined"] = (rs["loan"] + pr.get("loan", 0)) / (
        rs["loan"] + rs["permanent"] + pr.get("loan", 0) + pr.get("permanent", 0))
    t2n = N["tier2_episodes_14"]
    N["loan_share_tier2_episodes"] = t2n.get("loan", 0) / max(1, sum(t2n.values()))
    t1n = N["tier1_episodes_14"]
    N["loan_share_tier1_episodes"] = t1n.get("loan", 0) / max(1, sum(t1n.values()))
    return {"N": N, "grid": G, "potential": pot, "snapshot_leagues_14": snap14}


# --------------------------------------------------------------------------------------------------------------------
# Integrity checks
# --------------------------------------------------------------------------------------------------------------------
def checks(M: pd.DataFrame, X: pd.DataFrame, L2: pd.DataFrame) -> dict:
    import hashlib
    from .loan_episodes import CANONICAL_CSV
    sha = hashlib.sha256(Path(CANONICAL_CSV).read_bytes()).hexdigest()
    dup_key = M.duplicated(["player_id", "transfer_date", "from_club_id", "to_club_id"], keep=False)
    # verified league country vs the club's own country (families allowed)
    fam = {"England": {"Wales"}, "France": {"Monaco"}, "Spain": {"Andorra"}, "Switzerland": {"Liechtenstein"},
           "Italy": {"San Marino"}, "United States": {"Canada"}, "Australia": {"New Zealand"}}
    bad = []
    for side in ("from", "to"):
        lc, cc = X[f"{side}_league_country_v2"], M[f"{side}_country_v2"]
        m = lc.notna() & cc.notna() & (lc != cc) & ~np.array([c in fam.get(l, set()) for l, c in zip(lc, cc)])
        if m.any():
            bad.append(pd.DataFrame({"side": side, "event_id": M.event_id[m], "club_id": M[f"{side}_club_id"][m],
                                     "club_name": M[f"{side}_club_name"][m], "club_country_v2": cc[m],
                                     "league_country": lc[m], "league": X[f"{side}_league_name_v2"][m]}))
    bad = pd.concat(bad) if bad else pd.DataFrame()
    # promotions / relegations: tier changes between consecutive seasons in the lookup
    Ls = L2.sort_values(["club_id", "season"])
    prev = Ls.groupby("club_id")[["season", "tier", "country"]].shift()
    cons = Ls[prev.season.eq(Ls.season - 1)].assign(prev_tier=prev.tier, prev_country=prev.country)
    jumps = cons[(cons.tier - cons.prev_tier).abs() > 1]
    ctry_change = cons[cons.country != cons.prev_country]
    N = {"stage1c_sha256_ok": sha == STAGE1C_SHA, "master_shape": list(M.shape),
         "event_id_unique": bool(M.event_id.is_unique), "duplicate_player_date_clubs_rows": int(dup_key.sum()),
         "main_sample": int(M.in_cf_main_model_sample.sum()),
         "main_validated": int(M.in_cf_main_model_sample_validated.sum()),
         "injury_sample": int(M.in_cf_injury_model_sample.sum()),
         "injury_validated": int(M.in_cf_injury_model_sample_validated.sum()),
         "league_country_vs_club_country_mismatch_rows": int(len(bad)),
         "consecutive_club_seasons": int(len(cons)),
         "tier_unchanged": int((cons.tier == cons.prev_tier).sum()),
         "tier_moved_one_level": int(((cons.tier - cons.prev_tier).abs() == 1).sum()),
         "tier_jumps_over_one_level": int(len(jumps)), "league_country_changes": int(len(ctry_change))}
    assert N["stage1c_sha256_ok"] and N["event_id_unique"]
    assert (N["main_sample"], N["main_validated"], N["injury_sample"], N["injury_validated"]) == \
        (18090, 18006, 13951, 13871), N
    return {"N": N, "country_mismatch": bad, "jumps": jumps, "dups": M[dup_key]}


# --------------------------------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------------------------------
SUPPLEMENT_COLS = {
    "event_id": "episode key (joins to the master)",
    "tier_reference_season": "season of the move used for tiers (as in the master; 2024 = 2024/25)",
}
for _s, _w in (("from", "origin"), ("to", "receiving")):
    SUPPLEMENT_COLS.update({
        f"{_s}_league_tier_v2": f"{_w} club's tier in the reference season, Week 9 lookup plus Wikipedia lists "
                                 "(blank unless verified; Week 9 values never changed)",
        f"{_s}_league_name_v2": f"{_w} club's league name (verified rows only)",
        f"{_s}_league_country_v2": f"country of that league (verified rows only)",
        f"{_s}_league_tier_source_v2": "transfermarkt_snapshot_games / openfootball / wikipedia",
        f"{_s}_league_tier_status_v2": "verified_week9, verified_week10_wikipedia, not_tier_1_or_2_absent_from_lists "
                                       "(club of that country absent from that season's tier-1 and tier-2 lists, and "
                                       "sharing no name token with any unmatched team of those lists), not_tier_1_absent_from_list, unknown_conflicting_sources, or the Week 9 "
                                       "unknown statuses",
        f"{_s}_week9_status": "the stored Week 9 status, for comparison",
        f"{_s}_tier_band_v2": "1, 2, 3+ (verified tier 3 or lower, or a senior club absent from the tier-1/2 lists), "
                              "youth/reserve (youth or reserve side absent from the tier-1/2 lists), "
                              "'not 1', unknown",
        f"{_s}_tier_exact_v2": "true when the tier number itself is verified",
        f"{_s}_tier_lower_bound_v2": "tier number, or its lower bound (3 for band 3+, 2 for 'not 1')",
        f"{_s}_country_for_flows": "league country when verified, otherwise the club's country (from/to_country_v2)",
        f"{_s}_flow_category": "heatmap category (one of the 28 country-tiers or Other); blank = omitted",
        f"{_s}_flow_category_reason": "why the category was assigned or why the side was omitted",
    })


def _md(df: pd.DataFrame, floatfmt: str = "{:.1f}") -> str:
    d = df.copy()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in d.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pct(a, b) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    M, sides, L1, L2, C, conf = load()
    X = tiers_v2(M, sides, L1, L2, C, conf)
    Q1 = q1_tier2(M, X, C, sides)
    Q2 = q2_origin(M, X, C)
    Q3 = q3_movement(M, X)
    Q4 = q4_flows(M, X, C)
    K = checks(M, X, L2)
    q6 = json.loads((OUT / "q6_squad_value" / "numbers.json").read_text())
    lt2 = json.loads((V2 / "league_tiers_v2_numbers.json").read_text())

    # supplement and dictionary
    X[list(SUPPLEMENT_COLS)].to_csv(OUT / "week10_episode_supplement.csv", index=False)
    pd.DataFrame({"column": list(SUPPLEMENT_COLS), "meaning": list(SUPPLEMENT_COLS.values())}).to_csv(
        OUT / "week10_episode_supplement_dictionary.csv", index=False)
    # Q1
    d1 = OUT / "q1_tier2_playing_time"
    d1.mkdir(exist_ok=True)
    Q1["grid"].to_csv(d1 / "tier2_coverage_by_country_season.csv", index=False)
    Q1["potential"].to_csv(d1 / "potential_tier2_episodes_by_country.csv")
    Q1["snapshot_leagues_14"].to_csv(d1 / "snapshot_league_competitions_14_countries.csv", index=False)
    # Q2
    d2 = OUT / "q2_origin_tiers"
    d2.mkdir(exist_ok=True)
    Q2["miss"][["event_id", "kind", "from_club_name", "from_country_v2", "tier_reference_season", "week9_reason",
                "from_league_tier_status_v2", "from_league_tier_v2", "from_league_name_v2", "from_tier_band_v2",
                "outcome_v2", "still_unknown_reason"]].to_csv(d2 / "origin_tier_missing_5311_audit.csv", index=False)
    Q2["reason_tab"].to_csv(d2 / "reason_by_outcome.csv")
    Q2["kind_tab"].to_csv(d2 / "kind_by_outcome.csv")
    Q2["country"].to_csv(d2 / "recovery_by_country.csv")
    Q2["top_still"].to_csv(d2 / "still_unknown_top_clubs.csv", index=False)
    # Q3
    d3 = OUT / "q3_tier_movement"
    d3.mkdir(exist_ok=True)
    Q3["summary"].to_csv(d3 / "movement_summary.csv", index=False)
    for k, t in Q3["pair_tabs"].items():
        t.to_csv(d3 / f"tier_band_crosstab_{k}.csv")
    Q3["by_border"].to_csv(d3 / "direction_by_domestic_crossborder.csv")
    Q3["dest_from_t1"].to_csv(d3 / "destinations_from_14country_tier1_counts.csv")
    Q3["dest_from_t1_pct"].to_csv(d3 / "destinations_from_14country_tier1_pct.csv")
    Q3["by_country"].to_csv(d3 / "down_share_from_tier1_by_origin_country.csv")
    Q3["by_period"].to_csv(d3 / "direction_by_period.csv")
    # Q4
    d4 = OUT / "q4_heatmaps"
    d4.mkdir(exist_ok=True)
    Q4["coverage"].to_csv(d4 / "category_list_coverage.csv", index=False)
    acc_rows = []
    for k, lab, v in (("loan", "V1", "Loans"), ("permanent", "V2", "Permanent transfers")):
        mat = Q4["matrices"][k]
        mat.to_csv(d4 / f"{lab}_{k}_flow_matrix_29x29.csv")
        a = Q4["accounting"][k]
        for w in a["omitted_reasons"]:
            acc_rows.append({"kind": k, **w})
        sub = (f"{v}: transfer episodes in the master with reference season {PERIOD[0]}/{str(PERIOD[0] + 1)[-2:]} "
               f"to {PERIOD[1]}/{str(PERIOD[1] + 1)[-2:]}. Rows: origin club (FROM); columns: receiving club (TO).\n"
               f"{a['in_matrix']:,} of {a['episodes_in_period']:,} episodes shown; {a['omitted']:,} omitted "
               f"(tier 1 or 2 cannot be ruled out, country unknown, or no club). Colour scale shared by V1 and V2.\n"
               f"Tiers: Transfermarkt snapshot, openfootball and Wikipedia league tables. * = division lists for only "
               f"some seasons; episodes of the other seasons are omitted, not counted as Other "
               f"(category_list_coverage.csv).")
        plot_heatmap(mat, f"{lab}: {v} between country-tier categories", sub, Q4["coverage"], Q4["vmax"],
                     d4 / f"{lab}_{k}_heatmap")
    pd.DataFrame(acc_rows).to_csv(d4 / "omitted_episodes_by_reason.csv", index=False)
    # checks
    dc = OUT / "checks"
    dc.mkdir(exist_ok=True)
    K["country_mismatch"].to_csv(dc / "league_country_vs_club_country_mismatches.csv", index=False)
    K["jumps"].to_csv(dc / "tier_jumps_over_one_level.csv", index=False)
    K["dups"].to_csv(dc / "duplicate_player_date_club_rows.csv", index=False)
    numbers = {"q1": Q1["N"], "q2": Q2["N"], "q4": {k: {kk: vv for kk, vv in a.items() if kk != "omitted_reasons"}
                                                     for k, a in Q4["accounting"].items()},
               "checks": K["N"], "league_tiers_v2": lt2}
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=int))
    return {"M": M, "X": X, "Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4, "K": K, "q6": q6, "lt2": lt2, "C": C}


if __name__ == "__main__":
    from .week10_report import write_all
    r = run()
    write_all(r)
    print(json.dumps({"q1": r["Q1"]["N"], "q2": r["Q2"]["N"], "checks": r["K"]["N"]}, indent=1, default=int))
