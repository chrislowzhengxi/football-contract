"""What happens around the end of a loan: an exploratory, descriptive run.

    python -m src.analysis.loan_timing_exploration

Read-only. It takes the current loan-episode universe (`loan_scope.universe()`)
as it is, and nothing here feeds back into production classifications. It adds
one thing the production code does not keep: the player's next substantive
movement after a return at ANY distance, so timing can be seen without the 60-
day follow-on window. Restricted to that window it reproduces the production
follow-on exactly, and `reclassify` with the production windows reproduces
every production `economic_ending` (both checked on every run).

Every number in the report is interpolated from this computation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from .loan_episodes import LOAN, PERMANENT_TYPES, RETURN, ROOT, _boundary_days, _sorted
from .stage1_scope_report import europe_end_dates, latest_completed_season
from ..stage1c.policy import normalize_club_name
from ..stage2.event_family import (FOLLOW_ON_IMMEDIATE_DAYS, FOLLOW_ON_WINDOW_DAYS,
                                   SEASON_END_TOLERANCE_DAYS)

OUT = ROOT / "data" / "outputs" / "rebuild"
FIG = OUT / "figures"
REPORT_MD = OUT / "loan_timing_exploration.md"
LAGS_CSV = OUT / "loan_return_to_permanent_lags.csv"
EPISODES_CSV = OUT / "loan_follow_on_timing.csv"
DATES_CSV = OUT / "loan_ending_date_distribution.csv"
COUNTRY_CSV = OUT / "loan_ending_dates_by_country.csv"
CATEGORY_CSV = OUT / "loan_timing_category_summary.csv"
NEXT_KIND_CSV = OUT / "loan_timing_next_move_by_kind.csv"
WINDOW_CSV = OUT / "loan_timing_window_sensitivity.csv"
CYCLE_CSV = OUT / "loan_timing_on_off_cycle.csv"
NUMBERS_JSON = OUT / "loan_timing_exploration_numbers.json"

SKIP_TYPES = ("youth_or_internal", RETURN, "other")    # as in loan_episodes.classify_spells
SALE_TYPES = ("permanent_transfer", "undisclosed_transfer")
LAG_MARKS = (0, 1, 3, 7, 14, 21, 30, 60, 90)
CATEGORY_MARKS = (1, 3, 7, 14, 30, 60)
LAG_BANDS = ((0, 0), (1, 1), (2, 7), (8, 14), (15, 21), (22, 30), (31, 60), (61, 90), (91, 180),
             (181, 365), (366, None))
WINDOW_GRID = (0, 1, 3, 7, 14, 21, 30, 60, 90, 180, 365, None)
MIN_COUNTRY_ENDINGS = 100
CYCLE_FAST_DAYS = 1

CATEGORIES = {
    "A": "Return, no nearby subsequent move",
    "B": "Return, then permanent move to the borrower",
    "C": "Return, then another move or re-loan",
    "D": "Return, then sale to a third club",
    "E": "Ended by another movement, no End-of-loan row",
    "F": "No observed ending",
}
NEXT_KINDS = {
    "permanent_to_borrower": "Permanent move to the borrower",
    "sale_to_third_club": "Paid or undisclosed-fee move to a third club",
    "free_move_to_third_club": "Free / no-fee-shown move to a third club",
    "loan": "A new loan (any club)",
    "to_placeholder": "Release (Without Club, Retired, ...)",
    "other_type": "Other movement type",
}


# ---------------------------------------------------------------------------
# The unwindowed next move after a return
# ---------------------------------------------------------------------------

def next_moves(u) -> pd.DataFrame:
    """For every loan ended by a return row: the player's next substantive
    movement after that return, at any distance, or none."""
    d = _sorted(u.rows)
    pid = d.player_id.to_numpy()
    typ = d.transfer_type_normalized.to_numpy()
    skip = d.is_internal_move.to_numpy().astype(bool) | np.isin(typ, SKIP_TYPES)
    pos = pd.Series(np.arange(len(d)), index=d.event_id.to_numpy())
    L = u.loans[u.loans.terminal_is_return]
    out = []
    for loan_id, ret_id, player in zip(L.loan_event_id, L.terminal_event_id, L.player_id):
        j = pos[ret_id] + 1
        while j < len(d) and pid[j] == player and skip[j]:
            j += 1
        out.append((loan_id, pos[ret_id], j if j < len(d) and pid[j] == player else -1))
    k = pd.DataFrame(out, columns=["loan_event_id", "ret_pos", "next_pos"])
    has = k.next_pos >= 0
    n = d.iloc[k.loc[has, "next_pos"]]
    r = d.iloc[k.loc[has, "ret_pos"]]
    res = pd.DataFrame({"loan_event_id": k.loan_event_id})
    cols = {"next_move_event_id": n.event_id, "next_move_date": n._date,
            "next_move_type": n.transfer_type_normalized, "next_move_raw_label": n.fee_display_raw,
            "next_move_from_club_id": n.from_club_id, "next_move_from_club": n.from_club_name,
            "next_move_to_club_id": n.to_club_id, "next_move_to_club": n.to_club_name,
            "next_move_involves_placeholder": n.involves_placeholder_club.astype(bool),
            "next_move_scheduled": n.transfermarkt_future_transfer.astype(bool),
            "next_move_permanent_fee_eur": n.permanent_transfer_fee_eur,
            "_next_from_org": n._from_org}
    for c, v in cols.items():
        res.loc[has, c] = v.to_numpy()
    res.loc[has, "next_move_gap_days"] = (n._date.to_numpy() - r._date.to_numpy()).astype("timedelta64[D]").astype(int)
    res["next_move_date"] = pd.to_datetime(res.next_move_date)
    for c in ("next_move_from_club_id", "next_move_to_club_id", "next_move_permanent_fee_eur", "next_move_gap_days"):
        res[c] = pd.to_numeric(res[c])
    res["next_move_involves_placeholder"] = res.next_move_involves_placeholder.astype("boolean").fillna(False).astype(bool)
    res["next_move_scheduled"] = res.next_move_scheduled.astype("boolean").fillna(False).astype(bool)
    return res


def check_follow_on(loans: pd.DataFrame, nm: pd.DataFrame) -> int:
    """Rows where the windowed next move differs from the production follow-on."""
    m = loans[["loan_event_id", "follow_on_event_id"]].merge(nm, how="left", on="loan_event_id")
    win = m.next_move_event_id.where((m.next_move_gap_days <= FOLLOW_ON_WINDOW_DAYS)
                                     & ~m.next_move_involves_placeholder.fillna(False).astype(bool))
    return int((win.fillna("") != m.follow_on_event_id.fillna("")).sum())


def reclassify(T: pd.DataFrame, window: int | None = FOLLOW_ON_WINDOW_DAYS,
               immediate: int | None = FOLLOW_ON_IMMEDIATE_DAYS,
               tolerance: int = SEASON_END_TOLERANCE_DAYS) -> pd.Series:
    """`loan_episodes._economic`, vectorised, with the windows as parameters.
    None means no limit. Loans not ended by a return keep their class: it does
    not depend on any window."""
    inf = 10 ** 9
    window = inf if window is None else window
    immediate = inf if immediate is None else min(immediate, window)
    ret = T.ending_is_return.to_numpy(bool)
    gap = T.next_move_gap_days.to_numpy(float)
    has = ret & ~np.isnan(gap) & (np.nan_to_num(gap, nan=inf) <= window) & ~T.next_move_involves_placeholder.to_numpy(bool)
    typ = T.next_move_type.fillna("").to_numpy(str)
    perm = np.isin(typ, PERMANENT_TYPES)
    to_b = (T.next_move_to_club_id == T.borrower_club_id).to_numpy(bool)
    imm = np.nan_to_num(gap, nan=inf) <= immediate
    off = T.return_boundary_days.fillna(-1).to_numpy(float) > tolerance
    fallback = np.where(T.ending_fee_on_return_eur.notna(), "fee_bearing_return", "ordinary_end_of_loan")
    return pd.Series(np.select(
        [~ret, has & perm & to_b, has & perm & np.isin(typ, SALE_TYPES), has & imm & off,
         has & (imm | (typ == LOAN))],
        [T.economic_ending.to_numpy(str), "purchase_option_or_permanent_conversion",
         "third_party_sale_related", "early_termination", "immediate_follow_on_transfer"],
        default=fallback), index=T.index)


def sequence_category(econ: pd.Series, is_return: pd.Series) -> pd.Series:
    return pd.Series(np.select(
        [econ.eq("unresolved"), ~is_return.astype(bool),
         econ.eq("purchase_option_or_permanent_conversion"),
         econ.isin(["immediate_follow_on_transfer", "early_termination"]),
         econ.eq("third_party_sale_related")],
        ["F", "E", "B", "C", "D"], default="A"), index=econ.index)


def next_kind(T: pd.DataFrame) -> pd.Series:
    typ = T.next_move_type.fillna("")
    to_b = T.next_move_to_club_id == T.borrower_club_id
    perm = typ.isin(PERMANENT_TYPES)
    return pd.Series(np.select(
        [T.next_move_event_id.isna(), T.next_move_involves_placeholder, perm & to_b,
         perm & typ.isin(SALE_TYPES), perm, typ.eq(LOAN)],
        [None, "to_placeholder", "permanent_to_borrower", "sale_to_third_club",
         "free_move_to_third_club", "loan"], default="other_type"), index=T.index)


# ---------------------------------------------------------------------------
# Episode-level table
# ---------------------------------------------------------------------------

def episode_table(u, nm: pd.DataFrame) -> pd.DataFrame:
    L = u.loans.reset_index(drop=True)
    T = pd.DataFrame({
        "player_id": L.player_id, "player_name": L.player_name, "loan_event_id": L.loan_event_id,
        "loan_season": L.loan_season,
        "lender_club_id": L.lender_club_id, "lender_club": L.lender_club,
        "borrower_club_id": L.borrower_club_id, "borrower_club": L.borrower_club,
        "lender_country": L.from_country, "borrower_country": L.to_country,
        "lender_confederation": L.from_confederation, "borrower_confederation": L.to_confederation,
        "loan_start_date": L.loan_date, "loan_start_scheduled": L.transfermarkt_future_transfer.astype(bool),
        "ending_event_id": L.terminal_event_id, "ending_date": L.terminal_date,
        "ending_is_return": L.terminal_is_return.astype(bool),
        "ending_scheduled": L.terminal_scheduled_future.astype(bool),
        "ending_label_class": L.raw_label_class, "ending_raw_label": L.terminal_raw_label,
        "ending_fee_on_return_eur": L.ending_fee_on_return_eur,
        "loan_duration_days": L.duration_days, "match_reason": L.match_reason,
        "economic_ending": L.economic_ending,
        "current_follow_on_event_id": L.follow_on_event_id,
        "current_follow_on_gap_days": L.follow_on_gap_days,
        "return_boundary_days": L.return_boundary_days,
    }).merge(nm, how="left", on="loan_event_id", validate="one_to_one")
    T["next_move_involves_placeholder"] = T.next_move_involves_placeholder.astype("boolean").fillna(False).astype(bool)
    T["next_move_scheduled"] = T.next_move_scheduled.astype("boolean").fillna(False).astype(bool)
    T["ending_observed"] = T.ending_date.notna()
    T["ending_realised"] = T.ending_observed & ~T.ending_scheduled
    T["ending_mmdd"] = T.ending_date.dt.strftime("%m-%d")
    T["sequence_category"] = sequence_category(T.economic_ending, T.ending_is_return)
    T["next_move_kind"] = next_kind(T)
    T["next_move_is_permanent_to_borrower"] = T.next_move_kind.eq("permanent_to_borrower")
    T["next_move_from_lender_organisation"] = (
        T._next_from_org == T.lender_club.map(normalize_club_name)).where(T.next_move_event_id.notna())
    return T.drop(columns="_next_from_org")


# ---------------------------------------------------------------------------
# Descriptive helpers
# ---------------------------------------------------------------------------

def q(g: pd.Series, p: float) -> float:
    """Nearest-rank percentile: the smallest lag that covers at least share p."""
    return float(np.quantile(g, p, method="inverted_cdf"))


def lag_stats(g: pd.Series, marks=LAG_MARKS) -> dict:
    g = g.dropna().astype(int)
    n = len(g)
    out = {"N": n}
    if not n:
        return out
    out.update({"min": int(g.min()), "max": int(g.max()), "median": q(g, .5), "p75": q(g, .75),
                "p90": q(g, .9), "p95": q(g, .95), "negative": int((g < 0).sum())})
    for x in marks:
        k = int(((g >= 0) & (g <= x)).sum())
        out[f"within_{x}"] = k
        out[f"within_{x}_share"] = k / n
    return out


def band_table(g: pd.Series) -> list[dict]:
    g = g.dropna().astype(int)
    rows = []
    for lo, hi in LAG_BANDS:
        m = (g >= lo) if hi is None else g.between(lo, hi)
        k = int(m.sum())
        rows.append({"band": f"{lo}" if lo == hi else (f"> {lo - 1}" if hi is None else f"{lo}–{hi}"),
                     "lo": lo, "hi": hi, "n": k, "share": k / len(g),
                     "per_day": (k / (hi - lo + 1)) if hi is not None else None})
    return rows


def return_date_group(md: pd.Series) -> pd.Series:
    return pd.Series(np.select([md.isin(["06-30", "12-31"]), md.eq("05-31")],
                               ["return on 30 Jun or 31 Dec", "return on 31 May"],
                               default="other return date"), index=md.index)


def mmdd_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    md = frame.ending_mmdd
    years = frame.ending_date.dt.year.groupby(md).nunique()
    t = md.value_counts().rename("count").to_frame()
    t["mmdd"] = t.index
    t = t.sort_values(["count", "mmdd"], ascending=[False, True]).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    t["share"] = t["count"] / len(md)
    t["cumulative_share"] = t["count"].cumsum() / len(md)
    t["years_observed"] = t.mmdd.map(years).astype(int)
    return t[["rank", "mmdd", "count", "share", "cumulative_share", "years_observed"]]


def dates_needed(dist: pd.DataFrame, p: float) -> int:
    return int(np.searchsorted(dist.cumulative_share.to_numpy() + 1e-12, p) + 1)


def date_share(dist: pd.DataFrame, md: str) -> tuple[int, float, int | None]:
    r = dist[dist.mmdd == md]
    return (0, 0.0, None) if r.empty else (int(r["count"].iloc[0]), float(r.share.iloc[0]), int(r["rank"].iloc[0]))


def country_dates(R: pd.DataFrame, side: str, global_top3: list[str]) -> pd.DataFrame:
    col = f"{side}_country"
    rows = []
    for c, g in R[R[col].notna()].groupby(col):
        dist = mmdd_distribution(g)
        top3 = dist.head(3)
        rec = {"country_side": side, "country": c, "N": len(g),
               "included": len(g) >= MIN_COUNTRY_ENDINGS,
               "most_common_mmdd": dist.mmdd.iloc[0], "most_common_share": float(dist.share.iloc[0]),
               "top3_mmdd": ", ".join(top3.mmdd), "top3_share": float(top3.share.sum()),
               "share_on_global_top3": float(g.ending_mmdd.isin(global_top3).mean()),
               "dates_for_80pct": dates_needed(dist, .8), "dates_for_90pct": dates_needed(dist, .9),
               "dates_for_95pct": dates_needed(dist, .95)}
        for md in ("05-31", "06-30", "12-31"):
            rec[f"share_{md}"] = date_share(dist, md)[1]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["N"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part 7: appearances join diagnostics (coverage only, no playing-time measure)
# ---------------------------------------------------------------------------

def appearance_diagnostics(T: pd.DataFrame) -> tuple[dict, pd.Series]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        schema = {t: con.sql(f"describe {t}").df()[["column_name", "column_type"]].values.tolist()
                  for t in ("appearances", "game_lineups", "games")}
        a = con.sql("""select min(date), max(date), count(*), count(minutes_played),
                              count(*) - count(minutes_played), min(minutes_played), max(minutes_played),
                              sum(case when minutes_played = 0 then 1 else 0 end),
                              count(distinct player_id), count(distinct player_club_id)
                       from appearances""").fetchone()
        lu = con.sql("select min(date), max(date), count(*) from game_lineups").fetchone()
        seasons = con.sql("""select min(g.season), max(g.season) from appearances a
                             join games g on g.game_id = cast(a.game_id as varchar)""").fetchone()
        unmatched_games = con.sql("""select count(*) from appearances a left join games g
                                     on g.game_id = cast(a.game_id as varchar) where g.game_id is null""").fetchone()[0]
        comp = con.sql("""select g.competition_id, coalesce(g.competition_type, '(not in competitions)') t,
                                 min(g.season) s0, count(distinct g.game_id) games,
                                 count(distinct a.game_id) games_with_appearances
                          from games g left join appearances a on g.game_id = cast(a.game_id as varchar)
                          group by 1, 2""").df()
        no_comp_type = con.sql("""select count(*) from appearances a join games g
                                  on g.game_id = cast(a.game_id as varchar) where g.competition_type is null""").fetchone()[0]
        ap = con.sql("""select a.player_id, a.player_club_id club_id, a.date from appearances a
                        join games g on g.game_id = cast(a.game_id as varchar)
                        where coalesce(g.competition_type, '') <> 'national_team_competition'""").df()
    finally:
        con.close()
    ap["date"] = pd.to_datetime(ap.date)
    first, last = ap.date.min(), ap.date.max()
    first_season = f"{first.year}/{str(first.year + 1)[2:]}" if first.month >= 7 else f"{first.year - 1}/{str(first.year)[2:]}"

    W = T[T.loan_season >= first_season].copy()
    W["end"] = W.ending_date.fillna(last).clip(upper=last)
    club_dates = {c: np.sort(g.date.unique()) for c, g in ap.groupby("club_id")}

    def observed(r) -> bool:
        ds = club_dates.get(r.borrower_club_id)
        if ds is None:
            return False
        i = np.searchsorted(ds, np.datetime64(r.loan_start_date))
        return i < len(ds) and ds[i] <= np.datetime64(r.end)
    W["borrower_observed"] = [observed(r) for r in W[["borrower_club_id", "loan_start_date", "end"]].itertuples()]
    k = ap[ap.player_id.isin(set(W.player_id))].merge(
        W[["loan_event_id", "player_id", "borrower_club_id", "loan_start_date", "end"]],
        left_on=["player_id", "club_id"], right_on=["player_id", "borrower_club_id"])
    played = set(k.loc[(k.date >= k.loan_start_date) & (k.date <= k.end), "loan_event_id"])
    W["player_played"] = W.loan_event_id.isin(played)
    by_period = []
    for name, m in (("2012/13–2023/24", W.loan_season <= "2023/24"), ("2024/25 onwards", W.loan_season >= "2024/25")):
        g = W[m]
        by_period.append({"period": name, "loans": len(g), "borrower_observed": int(g.borrower_observed.sum()),
                          "player_played": int(g.player_played.sum()),
                          "played_given_observed": int((g.player_played & g.borrower_observed).sum())})
    new_leagues = comp[(comp.t == "domestic_league") & (comp.games_with_appearances == 0)]
    zero_other = comp[(comp.t != "domestic_league") & (comp.games_with_appearances == 0)]
    D = {"schema": schema,
         "appearances_first_date": str(first.date()), "appearances_last_date": str(pd.Timestamp(a[1]).date()),
         "appearances_rows": int(a[2]), "minutes_nonnull": int(a[3]), "minutes_null": int(a[4]),
         "minutes_min": int(a[5]), "minutes_max": int(a[6]), "minutes_zero": int(a[7]),
         "appearance_players": int(a[8]), "appearance_clubs": int(a[9]),
         "appearance_seasons": [seasons[0], seasons[1]],
         "appearances_without_game_row": int(unmatched_games),
         "appearances_in_competitions_missing_from_competitions_table": int(no_comp_type),
         "lineups_first_date": str(pd.Timestamp(lu[0]).date()), "lineups_last_date": str(pd.Timestamp(lu[1]).date()),
         "lineup_rows": int(lu[2]),
         "leagues_with_appearances": sorted(comp[(comp.t == "domestic_league") & (comp.games_with_appearances > 0)].competition_id),
         "leagues_without_appearances": sorted(new_leagues.competition_id),
         "leagues_without_appearances_first_season": sorted(set(new_leagues.s0)),
         "other_competitions_without_appearances": sorted(zero_other.competition_id),
         "first_loan_season_checked": first_season,
         "loans_checked": len(W),
         "loans_player_in_appearances": int(W.player_id.isin(set(ap.player_id)).sum()),
         "loans_borrower_observed": int(W.borrower_observed.sum()),
         "loans_player_played": int(W.player_played.sum()),
         "loans_played_given_observed": int((W.player_played & W.borrower_observed).sum()),
         "loans_borrower_observed_mapped": int((W.borrower_observed & W.borrower_country.notna()).sum()),
         "loans_borrower_mapped": int(W.borrower_country.notna().sum()),
         "by_period": by_period}
    flag = pd.Series(pd.NA, index=T.index, dtype="boolean")
    flag.loc[W.index] = W.borrower_observed.to_numpy()
    return D, flag


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute() -> tuple[dict, dict]:
    u = loan_scope.universe()
    nm = next_moves(u)
    N: dict = {"follow_on_mismatches": check_follow_on(u.loans, nm)}
    assert N["follow_on_mismatches"] == 0, "next-move table does not reproduce the production follow-on"
    T = episode_table(u, nm)
    re = reclassify(T)
    N["reclassify_mismatches"] = int((re != T.economic_ending).sum())
    assert N["reclassify_mismatches"] == 0, "reclassify() does not reproduce production economic_ending"
    N["loans"] = len(T)
    N["windows"] = {"immediate": FOLLOW_ON_IMMEDIATE_DAYS, "window": FOLLOW_ON_WINDOW_DAYS,
                    "tolerance": SEASON_END_TOLERANCE_DAYS}
    realised = T.ending_realised
    N["realised_endings"] = int(realised.sum())
    N["observed_endings"] = int(T.ending_observed.sum())

    # ---- Part 1: return -> same-borrower permanent
    conv = T.economic_ending.eq("purchase_option_or_permanent_conversion")
    N["p1_current"] = int(conv.sum())
    N["p1_current_with_return"] = int((conv & T.ending_is_return).sum())
    N["p1_current_no_return"] = int((conv & ~T.ending_is_return).sum())
    SB = T[T.ending_is_return & T.next_move_is_permanent_to_borrower].copy()
    SB["in_current_3818"] = conv.loc[SB.index]
    SB["return_date_group"] = return_date_group(SB.ending_mmdd)
    SB["permanent_mmdd"] = pd.to_datetime(SB.next_move_date).dt.strftime("%m-%d")
    assert SB.in_current_3818.sum() == N["p1_current_with_return"]
    assert (SB.loc[SB.in_current_3818, "next_move_gap_days"] <= FOLLOW_ON_WINDOW_DAYS).all()
    N["p1_any_lag"] = len(SB)
    N["p1_any_lag_stats"] = lag_stats(SB.next_move_gap_days)
    N["p1_current_stats"] = lag_stats(SB.loc[SB.in_current_3818, "next_move_gap_days"])
    N["p1_beyond_window"] = int((SB.next_move_gap_days > FOLLOW_ON_WINDOW_DAYS).sum())
    N["p1_bands"] = band_table(SB.next_move_gap_days)
    daily = SB.next_move_gap_days.astype(int).value_counts().sort_index()
    N["p1_daily_first"] = {int(k): int(v) for k, v in daily.loc[:120].items()}
    busiest = daily.drop(index=[1], errors="ignore").sort_values(ascending=False).head(5)
    N["p1_busiest_other_days"] = []
    for lag, cnt in busiest.items():
        g = SB[SB.next_move_gap_days == lag]
        top = g.ending_mmdd.value_counts()
        N["p1_busiest_other_days"].append({
            "lag": int(lag), "n": int(cnt), "top_return_mmdd": top.index[0], "top_return_n": int(top.iloc[0]),
            "top_move_mmdd": g.permanent_mmdd.value_counts().index[0],
            "top_move_n": int(g.permanent_mmdd.value_counts().iloc[0]),
            "english_or_scottish_lender": int(g.lender_country.isin(["England", "Scotland"]).sum())})
    d1 = SB[SB.next_move_gap_days == 1]
    N["p1_day1_return_mmdd"] = {k: int(v) for k, v in d1.ending_mmdd.value_counts().head(4).items()}
    N["p1_day1_seasonend"] = int(d1.ending_mmdd.isin(["06-30", "12-31"]).sum())
    N["p1_may31_jul1"] = int(((SB.ending_mmdd == "05-31") & (SB.permanent_mmdd == "07-01")
                              & (SB.next_move_gap_days == 31)).sum())
    other1 = d1[~d1.ending_mmdd.isin(["06-30", "12-31"])]
    N["p1_day1_other"] = len(other1)
    bump = SB[SB.next_move_gap_days.between(62, 63)]
    near = SB.next_move_gap_days.between(55, 61) | SB.next_move_gap_days.between(64, 70)
    N["p1_deadline_bump"] = {"n": len(bump), "neighbour_per_day": float(near.sum() / 14),
                             "jun30_to_aug31_or_sep1": int(((bump.ending_mmdd == "06-30")
                                                            & bump.permanent_mmdd.isin(["08-31", "09-01"])).sum())}
    grp = {}
    for gname, g in SB.groupby("return_date_group"):
        grp[gname] = lag_stats(g.next_move_gap_days)
    N["p1_by_return_group"] = grp
    tail = SB[SB.next_move_gap_days > 365].sort_values("next_move_permanent_fee_eur", ascending=False)
    N["p1_tail_n"] = len(tail)
    N["p1_tail_example"] = (tail.iloc[0][["player_name", "borrower_club", "ending_date", "next_move_date",
                                          "next_move_gap_days"]].astype(str).to_dict() if len(tail) else None)
    N["p1_by_scheduled"] = {f"return_{'scheduled' if a else 'realised'}__move_{'scheduled' if b else 'realised'}": int(v)
                            for (a, b), v in SB.groupby(["ending_scheduled", "next_move_scheduled"]).size().items()}
    both_real = SB[~SB.ending_scheduled & ~SB.next_move_scheduled]
    N["p1_realised_stats"] = lag_stats(both_real.next_move_gap_days)
    other_from = SB[SB.next_move_from_club_id != SB.lender_club_id]
    N["p1_from_not_lender"] = len(other_from)
    N["p1_from_not_lender_same_org"] = int(other_from.next_move_from_lender_organisation.fillna(False).astype(bool).sum())
    N["p1_from_not_lender_in_current"] = int(other_from.in_current_3818.sum())
    N["p1_same_day_cases"] = SB.loc[SB.next_move_gap_days == 0, ["player_name", "lender_club", "borrower_club",
                                                              "ending_date", "next_move_date"]].astype(str).to_dict("records")
    N["p1_no_return_cases"] = T.loc[conv & ~T.ending_is_return, ["player_name", "lender_club", "borrower_club",
                                                                 "loan_start_date", "ending_date"]].astype(str).to_dict("records")

    lags = SB.assign(population="return then permanent to borrower (any lag)")
    noret = T[conv & ~T.ending_is_return].assign(population="permanent move lender->borrower with no return row",
                                                 in_current_3818=True)
    lags = pd.concat([lags, noret], ignore_index=True)
    lags_out = pd.DataFrame({
        "population": lags.population, "in_current_3818": lags.in_current_3818.astype(bool),
        "player_id": lags.player_id, "player_name": lags.player_name, "loan_event_id": lags.loan_event_id,
        "loan_season": lags.loan_season,
        "lender_club_id": lags.lender_club_id, "lender_club": lags.lender_club,
        "borrower_club_id": lags.borrower_club_id, "borrower_club": lags.borrower_club,
        "lender_country": lags.lender_country, "borrower_country": lags.borrower_country,
        "loan_date": lags.loan_start_date.dt.date, "loan_scheduled": lags.loan_start_scheduled,
        "return_event_id": lags.ending_event_id.where(lags.ending_is_return),
        "return_date": lags.ending_date.where(lags.ending_is_return).dt.date,
        "return_mmdd": lags.ending_mmdd.where(lags.ending_is_return),
        "return_scheduled": lags.ending_scheduled.where(lags.ending_is_return),
        "permanent_event_id": lags.next_move_event_id.fillna(lags.ending_event_id),
        "permanent_date": pd.to_datetime(lags.next_move_date.fillna(lags.ending_date)).dt.date,
        "permanent_type": lags.next_move_type,
        "permanent_raw_label": lags.next_move_raw_label,
        "permanent_fee_eur": lags.next_move_permanent_fee_eur,
        "permanent_from_club": lags.next_move_from_club,
        "permanent_from_is_lender": (lags.next_move_from_club_id == lags.lender_club_id).where(lags.next_move_event_id.notna()),
        "permanent_scheduled": lags.next_move_scheduled.where(lags.next_move_event_id.notna()),
        "lag_days": lags.next_move_gap_days,
    })

    # ---- Part 2: categories
    cat_rows = []
    for c, label in CATEGORIES.items():
        g = T[T.sequence_category == c]
        lag = g.current_follow_on_gap_days if c in "BCD" else pd.Series(dtype=float)
        st = lag_stats(lag, CATEGORY_MARKS) if c in "BCD" else {}
        rec = {"category": c, "label": label, "count": len(g), "share_of_loans": len(g) / len(T),
               "realised_ending": int(g.ending_realised.sum()),
               "share_of_realised_endings": g.ending_realised.sum() / N["realised_endings"],
               "follow_on_realised": int((g.ending_realised & ~g.next_move_scheduled).sum()) if c in "BCD" else None}
        for k in ("median", "p75", "p90", "p95"):
            rec[f"lag_{k}"] = st.get(k)
        for x in CATEGORY_MARKS:
            rec[f"within_{x}_share"] = st.get(f"within_{x}_share")
        rec["existing_classes"] = "; ".join(f"{k} {v}" for k, v in g.economic_ending.value_counts().items())
        cat_rows.append(rec)
    cats = pd.DataFrame(cat_rows)
    assert cats["count"].sum() == len(T)
    N["p2_categories"] = cats.to_dict("records")
    A = T[T.sequence_category == "A"]
    N["p2_A_why"] = {
        "no_later_movement": int(A.next_move_event_id.isna().sum()),
        "next_is_release": int(A.next_move_involves_placeholder.sum()),
        "next_after_window": int((~A.next_move_involves_placeholder & (A.next_move_gap_days > FOLLOW_ON_WINDOW_DAYS)).sum()),
        "next_in_window_free_third_club": int((~A.next_move_involves_placeholder & (A.next_move_gap_days <= FOLLOW_ON_WINDOW_DAYS)).sum()),
        "fee_bearing": int(A.economic_ending.eq("fee_bearing_return").sum()),
    }
    assert sum(v for k, v in N["p2_A_why"].items() if k != "fee_bearing") == len(A)
    inwin = A[~A.next_move_involves_placeholder & (A.next_move_gap_days <= FOLLOW_ON_WINDOW_DAYS)]
    N["p2_A_in_window_kinds"] = {k: int(v) for k, v in inwin.next_move_kind.value_counts().items()}
    N["p2_A_in_window_gap_min"] = int(inwin.next_move_gap_days.min()) if len(inwin) else None
    C = T[T.sequence_category == "C"]
    N["p2_C_classes"] = {k: int(v) for k, v in C.economic_ending.value_counts().items()}
    N["p2_C_kinds"] = {k: int(v) for k, v in C.next_move_kind.value_counts().items()}
    N["p2_C_loan_after_immediate"] = int((C.next_move_kind.eq("loan") & (C.current_follow_on_gap_days > FOLLOW_ON_IMMEDIATE_DAYS)).sum())
    E = T[T.sequence_category == "E"]
    N["p2_E_reasons"] = {k: int(v) for k, v in E.match_reason.value_counts().items()}

    rets = T[T.ending_is_return & T.next_move_event_id.notna()]
    nk = []
    for k, label in NEXT_KINDS.items():
        g = rets[rets.next_move_kind == k]
        st = lag_stats(g.next_move_gap_days, (0, 1, 3, 7, 14, 30, 60, 90, 365))
        nk.append({"next_move_kind": k, "label": label, "n": len(g), **{x: st.get(x) for x in
                   ("median", "p75", "p90", "p95")},
                   **{f"within_{x}_share": st.get(f"within_{x}_share") for x in (0, 1, 3, 7, 14, 30, 60, 90, 365)}})
    N["p2_next_kinds"] = nk
    N["p2_returns"] = int(T.ending_is_return.sum())
    N["p2_returns_with_next"] = len(rets)

    # ---- Part 3: calendar dates of realised endings
    R = T[realised].copy()
    last = latest_completed_season(u)
    N["latest_completed_season"] = last
    eu = R[(R.lender_confederation == "UEFA") & (R.borrower_confederation == "UEFA") & (R.loan_season <= last)]
    q4 = europe_end_dates(u)[0]
    q4n = int(q4[(q4.population == "all matched endings") & (q4.slice == f"realised, loans through {last}")].N.iloc[0])
    assert len(eu) == q4n, (len(eu), q4n)
    samples = {"all_realised": R, "uefa_q4_main": eu, "all_realised_through_latest_completed": R[R.loan_season <= last],
               "returns_only_realised": R[R.ending_is_return]}
    dists, P3 = [], {}
    for name, fr in samples.items():
        dist = mmdd_distribution(fr)
        dists.append(dist.assign(sample=name))
        P3[name] = {"N": len(fr), "distinct": len(dist),
                    "single_year_dates": int((dist.years_observed == 1).sum()),
                    "top": dist.head(15).to_dict("records"),
                    **{f"top{k}_cover": float(dist.cumulative_share.iloc[k - 1]) for k in (1, 2, 3, 5, 10)},
                    **{f"dates_for_{int(p * 100)}": dates_needed(dist, p) for p in (.8, .9, .95)},
                    "named": {md: date_share(dist, md) for md in ("05-31", "06-30", "12-31")}}
    N["p3"] = P3
    date_dist = pd.concat(dists, ignore_index=True)[["sample", "rank", "mmdd", "count", "share", "cumulative_share",
                                                     "years_observed"]]
    N["p3_2025_26_realised"] = int((R.loan_season == "2025/26").sum())

    # ---- Part 4: by country
    gtop3 = list(mmdd_distribution(R).mmdd.head(3))
    N["p4_global_top3"] = gtop3
    cty = pd.concat([country_dates(R, "lender", gtop3), country_dates(R, "borrower", gtop3)], ignore_index=True)
    lend = cty[cty.country_side == "lender"]
    inc = lend[lend.included]
    N["p4_countries_included"] = len(inc)
    N["p4_min_included_N"] = int(inc.N.min())
    N["p4_max_excluded_N"] = int(lend.loc[~lend.included, "N"].max()) if (~lend.included).any() else None
    N["p4_excluded"] = lend.loc[~lend.included, ["country", "N"]].to_dict("records")
    N["p4_unmapped_lender"] = int(R.lender_country.isna().sum())
    N["p4_lender"] = inc.to_dict("records")
    N["p4_not_jun30_top"] = inc.loc[inc.most_common_mmdd != "06-30", ["country", "most_common_mmdd",
                                                                      "most_common_share"]].to_dict("records")
    N["p4_top3_range"] = [float(inc.top3_share.min()), float(inc.top3_share.max())]
    N["p4_dates90_range"] = [int(inc.dates_for_90pct.min()), int(inc.dates_for_90pct.max())]
    N["p4_global_top3_below_70"] = inc.loc[inc.share_on_global_top3 < .7, ["country", "share_on_global_top3"]
                                           ].sort_values("share_on_global_top3").to_dict("records")
    N["p4_global_top3_at_least_80"] = int((inc.share_on_global_top3 >= .8).sum())
    bor = cty[(cty.country_side == "borrower") & cty.included]
    N["p4_borrower_not_jun30_top"] = bor.loc[bor.most_common_mmdd != "06-30", ["country", "most_common_mmdd",
                                                                               "most_common_share"]].to_dict("records")
    both = inc.merge(bor, on="country", suffixes=("_l", "_b"))
    N["p4_lender_vs_borrower_top_differs"] = both.loc[both.most_common_mmdd_l != both.most_common_mmdd_b,
                                                      ["country", "most_common_mmdd_l", "most_common_mmdd_b"]].to_dict("records")

    # ---- Part 5: on-cycle vs off-cycle
    gdist = mmdd_distribution(R)
    csets = {r.country: set(mmdd_distribution(g).mmdd.head(3)) for r in inc.itertuples()
             for g in [R[R.lender_country == r.country]]}
    cset90 = {r.country: set(mmdd_distribution(g).pipe(lambda d: d.head(dates_needed(d, .9))).mmdd)
              for r in inc.itertuples() for g in [R[R.lender_country == r.country]]}
    cset95 = {r.country: set(mmdd_distribution(g).pipe(lambda d: d.head(dates_needed(d, .95))).mmdd)
              for r in inc.itertuples() for g in [R[R.lender_country == r.country]]}
    n90 = dates_needed(gdist, .9)
    boundary = R.ending_date.map(_boundary_days)

    def by_country(sets):
        known = R.lender_country.isin(sets.keys())
        on = pd.Series([md in sets.get(c, ()) for c, md in zip(R.lender_country, R.ending_mmdd)], index=R.index)
        return on.where(known)

    defs = {
        "G2": ("Global top 2 dates (" + ", ".join(gdist.mmdd.head(2)) + ")", R.ending_mmdd.isin(gdist.mmdd.head(2))),
        "G3": ("Global top 3 dates (" + ", ".join(gdist.mmdd.head(3)) + ")", R.ending_mmdd.isin(gdist.mmdd.head(3))),
        "G5": ("Global top 5 dates (" + ", ".join(gdist.mmdd.head(5)) + ")", R.ending_mmdd.isin(gdist.mmdd.head(5))),
        "G90": (f"Global dates covering 90% ({n90} dates)", R.ending_mmdd.isin(gdist.mmdd.head(n90))),
        "K3": ("Lender-country top 3 dates", by_country(csets)),
        "K90": ("Lender-country dates covering 90%", by_country(cset90)),
        "K95": ("Lender-country dates covering 95%", by_country(cset95)),
        "T10": (f"Existing rule: within {SEASON_END_TOLERANCE_DAYS} days of 30 Jun or 31 Dec",
                boundary <= SEASON_END_TOLERANCE_DAYS),
    }
    et = T[T.economic_ending.eq("early_termination")]
    N["p5_et_on_own_country_top3"] = int(sum(md in csets.get(c, ()) for c, md in zip(et.lender_country, et.ending_mmdd)))
    N["p5_et_lender_not_classifiable"] = int((~et.lender_country.isin(csets.keys())).sum())
    own = et[[md in csets.get(c, ()) for c, md in zip(et.lender_country, et.ending_mmdd)]]
    N["p5_et_on_own_top_country"] = {k: int(v) for k, v in own.lender_country.value_counts().head(1).items()}
    nonplain = R.ending_label_class != "End of loan"
    fastperm = R.next_move_kind.isin(["permanent_to_borrower", "sale_to_third_club", "free_move_to_third_club"]) & \
        (R.next_move_gap_days <= CYCLE_FAST_DAYS)
    fastsb = R.next_move_is_permanent_to_borrower & (R.next_move_gap_days <= CYCLE_FAST_DAYS)
    catB = R.sequence_category == "B"
    cyc = []
    for key, (label, on) in defs.items():
        known = on.notna()
        onb = on.fillna(False).astype(bool)
        for state, m in (("on-cycle", known & onb), ("off-cycle", known & ~onb), ("not classifiable", ~known)):
            k = int(m.sum())
            if state == "not classifiable" and k == 0:
                continue
            cyc.append({"definition": key, "label": label, "state": state, "n": k,
                        "share_of_classified": k / int(known.sum()) if state != "not classifiable" else None,
                        "share_of_realised": k / len(R),
                        "non_plain_n": int((m & nonplain).sum()), "non_plain_share": (m & nonplain).sum() / k if k else None,
                        "fast_permanent_n": int((m & fastperm).sum()),
                        "fast_permanent_share": (m & fastperm).sum() / k if k else None,
                        "fast_permanent_to_borrower_n": int((m & fastsb).sum()),
                        "fast_permanent_to_borrower_share": (m & fastsb).sum() / k if k else None,
                        "category_B_n": int((m & catB).sum()), "category_B_share": (m & catB).sum() / k if k else None})
    cycle = pd.DataFrame(cyc)
    N["p5"] = cycle.to_dict("records")
    N["p5_nonplain_total"] = int(nonplain.sum())

    # ---- Part 6: the 48.6%
    nonord = T.economic_ending.isin(loan_scope.NONSTANDARD)
    N["p6_nonordinary"] = int(nonord.sum())
    X = T[nonord]
    gap = X.current_follow_on_gap_days
    N["p6_parts"] = {
        "B": int(X.sequence_category.eq("B").sum()), "C": int(X.sequence_category.eq("C").sum()),
        "D": int(X.sequence_category.eq("D").sum()), "E": int(X.sequence_category.eq("E").sum()),
        "A_fee_bearing": int(X.sequence_category.eq("A").sum())}
    assert sum(N["p6_parts"].values()) == len(X)
    N["p6_no_follow_on"] = int(gap.isna().sum())
    rows6 = []
    for c in ("B", "C", "D"):
        g = X.loc[X.sequence_category == c, "current_follow_on_gap_days"]
        rows6.append({"category": c, "n": len(g), **{f"b_{lo}_{hi}": int(g.between(lo, hi).sum())
                                                     for lo, hi in ((0, 0), (1, 1), (2, 7), (8, 21), (22, 60))}})
    N["p6_bands"] = rows6
    N["p6_within_1"] = int((gap <= 1).sum())
    N["p6_within_7"] = int((gap <= 7).sum())
    N["p6_22_60"] = int((gap > FOLLOW_ON_IMMEDIATE_DAYS).sum())
    N["p6_8_60"] = int((gap > 7).sum())
    beyond = T[T.ending_is_return & T.next_move_event_id.notna() & ~T.next_move_involves_placeholder
               & (T.next_move_gap_days > FOLLOW_ON_WINDOW_DAYS)]
    N["p6_beyond_window_kinds"] = {k: int(v) for k, v in beyond.next_move_kind.value_counts().items()}
    grid = []
    for imm in WINDOW_GRID:
        for win in WINDOW_GRID:
            if imm is not None and win is not None and imm > win:
                continue
            if imm is None and win is not None:
                continue
            e = reclassify(T, window=win, immediate=imm)
            cat = sequence_category(e, T.ending_is_return)
            no = e.isin(loan_scope.NONSTANDARD)
            grid.append({"immediate_days": "none" if imm is None else imm, "window_days": "none" if win is None else win,
                         "nonordinary": int(no.sum()), "share": no.mean(),
                         **{f"cat_{c}": int(cat.eq(c).sum()) for c in "ABCDEF"},
                         "is_current": imm == FOLLOW_ON_IMMEDIATE_DAYS and win == FOLLOW_ON_WINDOW_DAYS,
                         "is_single_window": imm == win})
    G = pd.DataFrame(grid)
    assert int(G.loc[G.is_current, "nonordinary"].iloc[0]) == N["p6_nonordinary"]
    N["p6_grid_single"] = G[G.is_single_window].to_dict("records")
    N["p6_grid_current"] = G[G.is_current].to_dict("records")[0]
    N["p6_grid_1_60"] = G[(G.immediate_days == 1) & (G.window_days == FOLLOW_ON_WINDOW_DAYS)].to_dict("records")[0]
    N["p6_grid_21_none"] = G[(G.immediate_days == FOLLOW_ON_IMMEDIATE_DAYS) & (G.window_days == "none")].to_dict("records")[0]
    N["p6_fee_bearing_returns_total"] = int(T.ending_fee_on_return_eur.notna().sum())
    N["p6_B_day1"] = int((T.sequence_category.eq("B") & T.current_follow_on_gap_days.le(1)).sum())
    N["p6_B_2_60"] = int((T.sequence_category.eq("B") & T.current_follow_on_gap_days.gt(1)).sum())
    N["p6_C_offcycle_label"] = int(T.economic_ending.eq("early_termination").sum())
    N["p6_D"] = int(T.sequence_category.eq("D").sum())
    N["p6_D_day1"] = int((T.sequence_category.eq("D") & T.current_follow_on_gap_days.le(1)).sum())

    # ---- Part 7: join readiness
    P7, observed = appearance_diagnostics(T)
    T["borrower_club_has_appearance_rows_during_loan"] = observed
    N["p7"] = P7

    tables = {"T": T, "lags": lags_out, "dates": date_dist, "country": cty, "cats": cats,
              "next_kinds": pd.DataFrame(nk), "grid": G, "cycle": cycle, "SB": SB, "R": R, "eu": eu}
    return N, tables


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {"return on 30 Jun or 31 Dec": "#2b6cb0", "return on 31 May": "#dd6b20",
          "other return date": "#a0aec0"}
KIND_COLORS = {"permanent_to_borrower": "#2b6cb0", "sale_to_third_club": "#c53030",
               "free_move_to_third_club": "#718096", "loan": "#38a169"}
LOG_TICKS = [0, 1, 3, 7, 14, 21, 30, 60, 90, 180, 365, 730]


def _ecdf(ax, g, **kw):
    x = np.sort(g.dropna().astype(int).to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, 100 * y, where="post", **kw)


def _log_axis(ax, right=730, drop=()):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, right)
    ticks = [t for t in LOG_TICKS if t <= right and t not in drop]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def figures(N: dict, tb: dict) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    SB = tb["SB"]
    paths = {}

    # 1. histogram, long tail folded into one bar
    cap, clip = 120, 120
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottom = np.zeros(cap + 2)
    for grp in COLORS:
        g = SB.loc[SB.return_date_group == grp, "next_move_gap_days"].astype(int)
        counts = np.bincount(g.clip(upper=cap + 1), minlength=cap + 2)[:cap + 2]
        xs = np.r_[np.arange(cap + 1), cap + 6]
        ax.bar(xs, counts, bottom=bottom, width=0.9, color=COLORS[grp], label=grp)
        bottom += counts
    day1 = int(bottom[1])
    ax.set_ylim(0, clip)
    ax.annotate(f"day 1: {day1:,} (bar clipped)", xy=(1, clip), xytext=(8, clip * 0.9),
                arrowprops={"arrowstyle": "->", "color": "#333"}, fontsize=9)
    over = int(bottom[-1])
    ax.text(cap + 6, min(over, clip) + 2, f"{over}" + (" (clipped)" if over > clip else ""), ha="center", fontsize=8)
    for x in (FOLLOW_ON_IMMEDIATE_DAYS, FOLLOW_ON_WINDOW_DAYS):
        ax.axvline(x + 0.5, color="#555", ls=":", lw=1)
    ax.text(FOLLOW_ON_WINDOW_DAYS + 1.5, clip * 0.75, "dotted lines: current 21- and 60-day\nwindows (for reference only)",
            fontsize=8, color="#555")
    ax.set_xticks(list(range(0, cap + 1, 10)) + [cap + 6])
    ax.set_xticklabels([str(x) for x in range(0, cap + 1, 10)] + [f">{cap}"])
    ax.set_xlabel("Days from the loan return to the permanent move back to the borrower")
    ax.set_ylabel("Loans")
    ax.set_title(f"Return → permanent move back to the borrower, any lag (N = {len(SB):,})", loc="left")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    paths["hist"] = "figures/loan_timing_fig1_return_to_permanent_lag_histogram.png"
    fig.savefig(OUT / paths["hist"], dpi=160)
    plt.close(fig)

    # 2. ECDF
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _ecdf(ax, SB.next_move_gap_days, color="#1a202c", lw=2.2, label=f"all ({len(SB):,})")
    for grp in ("return on 30 Jun or 31 Dec", "return on 31 May"):
        g = SB.loc[SB.return_date_group == grp, "next_move_gap_days"]
        _ecdf(ax, g, color=COLORS[grp], lw=1.2, label=f"{grp} ({len(g):,})")
    st = N["p1_any_lag_stats"]
    for x in (FOLLOW_ON_IMMEDIATE_DAYS, FOLLOW_ON_WINDOW_DAYS):
        ax.axvline(x, color="#555", ls=":", lw=1)
        ax.text(x * 1.04, 8, f"{x} d: {100 * st[f'within_{x}_share']:.1f}%", fontsize=8, color="#555")
    _log_axis(ax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Days from the loan return (log scale above 1 day)")
    ax.set_ylabel("Cumulative share of cases (%)")
    ax.set_title("Cumulative share of permanent moves back to the borrower, by lag", loc="left")
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout()
    paths["ecdf"] = "figures/loan_timing_fig2_return_to_permanent_ecdf.png"
    fig.savefig(OUT / paths["ecdf"], dpi=160)
    plt.close(fig)

    # 3. calendar dates
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    top = pd.DataFrame(N["p3"]["all_realised"]["top"])
    a1.bar(top.mmdd, 100 * top.share, color="#2b6cb0")
    a1.set_xticks(range(len(top)))
    a1.set_xticklabels(top.mmdd, rotation=60, ha="right")
    a1.set_ylabel("Share of realised loan endings (%)")
    a1.set_title(f"15 most common ending dates (N = {N['p3']['all_realised']['N']:,})", loc="left")
    for label, key, color in (("all realised endings", "all_realised", "#2b6cb0"),
                              ("mapped-UEFA sample (Q4)", "uefa_q4_main", "#dd6b20")):
        d = tb["dates"][tb["dates"]["sample"] == key]
        a2.step(d["rank"], 100 * d.cumulative_share, where="post", color=color, lw=1.8, label=label)
    for p in (80, 90, 95):
        a2.axhline(p, color="#999", ls=":", lw=0.8)
    a2.set_xscale("log")
    a2.set_xlim(1, 366)
    a2.set_xticks([1, 2, 3, 5, 10, 30, 100, 366])
    a2.set_xticklabels(["1", "2", "3", "5", "10", "30", "100", "366"])
    a2.minorticks_off()
    a2.set_ylim(40, 100)
    a2.set_xlabel("Number of calendar dates (most common first)")
    a2.set_ylabel("Cumulative share covered (%)")
    a2.set_title("Coverage by the top-K dates", loc="left")
    a2.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    paths["dates"] = "figures/loan_timing_fig3_ending_dates.png"
    fig.savefig(OUT / paths["dates"], dpi=160)
    plt.close(fig)

    # 4. B / C / D and the unwindowed next move by kind
    T = tb["T"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for c, color in (("B", "#2b6cb0"), ("C", "#38a169"), ("D", "#c53030")):
        g = T.loc[T.sequence_category == c, "current_follow_on_gap_days"]
        _ecdf(a1, g, color=color, lw=1.8, label=f"{c}: {CATEGORIES[c].split(', then ')[1]} ({len(g):,})")
    _log_axis(a1, 60)
    a1.set_ylim(0, 100)
    a1.set_xlabel("Days from return to the follow-on move")
    a1.set_ylabel("Cumulative share (%)")
    a1.set_title("Categories B, C, D (current 60-day window)", loc="left")
    a1.legend(frameon=False, loc="lower right", fontsize=8)
    rets = T[T.ending_is_return & T.next_move_event_id.notna()]
    for k, color in KIND_COLORS.items():
        g = rets.loc[rets.next_move_kind == k, "next_move_gap_days"]
        _ecdf(a2, g, color=color, lw=1.8, label=f"{NEXT_KINDS[k]} ({len(g):,})")
    _log_axis(a2, drop=(21, 90))
    a2.set_ylim(0, 100)
    a2.set_xlabel("Days from return to the player's next move (no window)")
    a2.set_title("Next move after any return, by kind", loc="left")
    a2.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    paths["kinds"] = "figures/loan_timing_fig4_follow_on_by_category.png"
    fig.savefig(OUT / paths["kinds"], dpi=160)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")


def md_words(md: str) -> str:
    return f"{int(md[3:])} {MONTHS[int(md[:2]) - 1]}"


def cs(k, n, digits=1) -> str:
    from .daniel_scope_final import f0, pc
    return f"{f0(k)} ({pc(k, n, digits)})"


def days(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{int(x)}" if float(x).is_integer() else f"{x:.1f}"


def share(x, digits=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def render(N: dict, tb: dict, paths: dict) -> str:
    from .daniel_scope_final import f0, frac, md_table, pc
    from .stage1_scope_report import display_country as dc

    W: list[str] = []
    w = W.append
    s1 = N["p1_any_lag_stats"]
    s3816 = N["p1_current_stats"]
    sreal = N["p1_realised_stats"]
    n_sb = N["p1_any_lag"]
    daily = N["p1_daily_first"]
    bands = N["p1_bands"]
    busy = N["p1_busiest_other_days"]
    grp = N["p1_by_return_group"]
    kinds = {r["next_move_kind"]: r for r in N["p2_next_kinds"]}
    cats = {r["category"]: r for r in N["p2_categories"]}
    P3 = N["p3"]
    A3 = P3["all_realised"]
    U3 = P3["uefa_q4_main"]
    grid = {str(r["window_days"]): r for r in N["p6_grid_single"]}
    cur = N["p6_grid_current"]
    p7 = N["p7"]
    nono = N["p6_nonordinary"]
    loans = N["loans"]
    win, imm, tol = FOLLOW_ON_WINDOW_DAYS, FOLLOW_ON_IMMEDIATE_DAYS, SEASON_END_TOLERANCE_DAYS
    per_day = [b["per_day"] for b in bands if b["lo"] >= 2 and b["per_day"] is not None]
    smooth = all(a > b for a, b in zip(per_day, per_day[1:]))
    lend = pd.DataFrame(N["p4_lender"])
    eng = lend[lend.country == "England"].iloc[0]
    by_md: dict[str, list[str]] = {}
    for r in N["p4_not_jun30_top"]:
        by_md.setdefault(r["most_common_mmdd"], []).append(dc(r["country"]))
    not_jun30 = "; ".join(f"{md_words(md)}: {', '.join(cs_)}" for md, cs_ in
                          sorted(by_md.items(), key=lambda kv: -len(kv[1])))
    cyc = pd.DataFrame(N["p5"])

    def cyc_get(d, state, col):
        r = cyc[(cyc.definition == d) & (cyc.state == state)]
        return r[col].iloc[0]

    classified = cyc[cyc.state != "not classifiable"]
    piv = classified.pivot(index="definition", columns="state", values="non_plain_share")
    fast = classified.pivot(index="definition", columns="state", values="fast_permanent_share")
    off_more_nonplain = int((piv["off-cycle"] > piv["on-cycle"]).sum())
    off_less_fast = int((fast["off-cycle"] < fast["on-cycle"]).sum())
    ndefs = len(piv)

    w("# What happens around the end of a loan? (exploratory)")
    w("")
    w("*Descriptive and exploratory, for deciding how to segment the data. Generated by "
      "`python -m src.analysis.loan_timing_exploration` from the frozen Stage 1C table through the current "
      f"loan-episode universe ({f0(loans)} loans); production classifications are not changed. Two checks run every "
      "time: the next-move table below reproduces the production follow-on move for every loan "
      f"({N['follow_on_mismatches']} mismatches), and re-running the production ending rules with their current windows "
      f"reproduces every loan's `economic_ending` ({N['reclassify_mismatches']} mismatches). No network or API calls.*")
    w("")
    w("## Headline findings")
    w("")
    w(f"1. **One sharp break, at one day.** Of the {f0(n_sb)} loan returns whose next move is a permanent move back to "
      f"the borrower (at any lag), {cs(daily.get(1, 0), n_sb)} are dated exactly one day after the return and "
      f"{f0(s1['within_0'])} on the same day. Day 2 has {f0(daily.get(2, 0))}. After that the daily count declines "
      + ("gradually" if smooth else "unevenly") + ", apart from calendar spikes (finding 2); neither 21 nor 60 days "
      "marks a change.")
    b31 = busy[0]
    w(f"2. **The next-biggest spike is a date convention.** Day {b31['lag']} has {f0(b31['n'])} cases, "
      f"{f0(b31['top_return_n'])} of them a {md_words(b31['top_return_mmdd'])} return followed by a "
      f"{md_words(b31['top_move_mmdd'])} move: the British way of dating loan ends (Part 3). Returns dated 31 May have a "
      f"median lag of {days(grp['return on 31 May']['median'])} days, against "
      f"{days(grp['return on 30 Jun or 31 Dec']['median'])} for returns dated 30 June or 31 December.")
    pb, sa, fr, lo = (kinds[k] for k in ("permanent_to_borrower", "sale_to_third_club", "free_move_to_third_club", "loan"))
    w(f"3. **Moves back to the borrower are much more concentrated at day 1 than other moves.** "
      f"{share(pb['within_1_share'])} of permanent moves back to the borrower come within a day of the return, against "
      f"{share(sa['within_1_share'])} of sales to a third club, {share(fr['within_1_share'])} of free moves and "
      f"{share(lo['within_1_share'])} of new loans (medians {days(pb['median'])}, {days(sa['median'])}, "
      f"{days(fr['median'])} and {days(lo['median'])} days).")
    w(f"4. **The current windows, as diagnostics only.** {imm} days covers {share(s1['within_21_share'])} of these "
      f"permanent moves back to the borrower and {win} days covers {share(s1['within_60_share'])}. "
      f"{cs(N['p1_beyond_window'], n_sb)} happen later and are currently counted as ordinary returns.")
    w(f"5. **Loan endings cluster on three dates, then spread thin.** 30 June "
      f"({share(A3['named']['06-30'][1])}), 31 December ({share(A3['named']['12-31'][1])}) and 31 May "
      f"({share(A3['named']['05-31'][1])}) cover {share(A3['top3_cover'])} of the {f0(A3['N'])} realised endings. "
      f"{A3['dates_for_80']} dates cover 80%, but 90% needs {A3['dates_for_90']} and 95% needs {A3['dates_for_95']}.")
    w(f"6. **Standard dates differ by country.** {len(N['p4_not_jun30_top'])} of the {N['p4_countries_included']} "
      f"lender countries with enough endings have a most common date other than 30 June ({not_jun30}). "
      f"England splits between 30 June ({share(eng['share_06-30'])}) and 31 May ({share(eng['share_05-31'])}). "
      f"The global top 3 dates cover at least 80% of endings in {N['p4_global_top3_at_least_80']} of the "
      f"{N['p4_countries_included']} countries.")
    w(f"7. **The 48.6% figure mostly measures the window.** {cs(N['p6_within_1'], nono)} of the {f0(nono)} "
      f"non-ordinary loans have their follow-on move within one day, and {f0(N['p6_22_60'])} count only because of a "
      f"move {imm + 1}–{win} days after the return. With a single window of X days for every follow-on move the share "
      f"would be {share(grid['1']['share'])} at 1 day, {share(grid['21']['share'])} at 21, "
      f"{share(grid['60']['share'])} at 60, {share(grid['365']['share'])} at 365 and {share(grid['none']['share'])} "
      "with no limit, because most players move again eventually.")
    w(f"8. **Playing time can be joined, but only where appearances exist.** `appearances` has player id, club id, "
      f"date and `minutes_played` (no missing values), but only for the {len(p7['leagues_with_appearances'])} "
      "long-covered European leagues plus their cups and the UEFA Champions and Europa Leagues. In "
      f"{cs(p7['loans_borrower_observed'], p7['loans_checked'])} of the loans since "
      f"{p7['first_loan_season_checked']}, the borrower has any appearance rows during the loan. For the rest, "
      "an absence of appearances is missing data, not zero minutes.")
    w("")

    w("## Definitions used throughout")
    w("")
    w(f"- **Loan**: one of the {f0(loans)} loan episodes of the current definition (`loan_episodes.py`), unchanged.")
    w("- **Return**: the `End of loan` row that closed the loan. **Ending**: the row that closed the loan, whether a "
      "return or another movement. **Realised**: that row had already happened when the player's history was "
      "captured. **Scheduled**: Transfermarkt showed it as a future move.")
    w("- **Next move**: after a return, the player's next *substantive* movement at any distance. It skips internal "
      "registration changes, other return rows and draft rows, as the production code does. The production "
      f"*follow-on* is that move only if it is within {win} days and is not a release to Without Club or Retired.")
    w("- **lag_days** = date of the next move − date of the return, both as Transfermarkt records them. **Within X "
      "days** means 0 ≤ lag_days ≤ X. Percentiles are nearest-rank: the smallest lag that covers at least that "
      "share of cases.")
    w("- Transfermarkt's dates follow conventions: a return is usually dated on the last day of a season (30 June, "
      "or 31 May for British loans) and a new registration on the first day of the next. A lag measures those "
      "recorded dates, not when anything was agreed.")
    w("")

    # ---- Part 1
    w("## Part 1. Loan return → permanent move back to the borrower")
    w("")
    w("### Verifying the 3,818")
    w("")
    nr = N["p1_no_return_cases"]
    w(f"The current count is **{f0(N['p1_current'])}** loans (`economic_ending` = "
      f"`purchase_option_or_permanent_conversion`):")
    w("")
    W += md_table(["Pattern", "Loans"], [
        [f"A → B loan, B → A `End of loan`, A → B permanent move within {win} days", f0(N["p1_current_with_return"])],
        ["A → B loan, then A → B permanent move with **no** return row in between ("
         + "; ".join(f"{r['player_name']}, {r['lender_club']} → {r['borrower_club']}" for r in nr) + ")",
         f0(N["p1_current_no_return"])],
        ["**Total (current)**", f"**{f0(N['p1_current'])}**"]])
    w("")
    w(f"All {f0(N['p1_current_with_return'])} with a return are within {win} days **by construction**, because the "
      f"production follow-on search stops at {win} days. To look at the timing without that truncation, this run takes "
      f"every return whose next move is a permanent move to the borrower, at any lag: **{f0(n_sb)}** loans. That is "
      f"the {f0(N['p1_current_with_return'])} plus {f0(N['p1_beyond_window'])} whose move came later. The 2 without "
      "a return row have no lag; they are kept in the CSV and not used below.")
    w("")
    w("### Distribution of lag_days")
    w("")

    def col(s, key):
        if key in ("min", "max", "median", "p75", "p90", "p95"):
            return days(s[key])
        return cs(s[key], s["N"])
    rowspec = [("N", "N"), ("Minimum", "min"), ("Same day (0)", "within_0")] + \
        [(f"Within {x} day{'s' if x > 1 else ''}", f"within_{x}") for x in LAG_MARKS if x > 0] + \
        [("Median", "median"), ("p75", "p75"), ("p90", "p90"), ("p95", "p95"), ("Maximum", "max")]
    W += md_table(["", f"Any lag (N = {f0(s1['N'])})", f"Current, ≤ {win} days by construction (N = {f0(s3816['N'])})",
                   f"Any lag, both dates realised (N = {f0(sreal['N'])})"],
                  [[lab, f0(s1["N"]) if k == "N" else col(s1, k), f0(s3816["N"]) if k == "N" else col(s3816, k),
                    f0(sreal["N"]) if k == "N" else col(sreal, k)] for lab, k in rowspec])
    w("")
    w(f"![Histogram of lag days]({paths['hist']})")
    w("")
    w(f"![Cumulative share by lag]({paths['ecdf']})")
    w("")
    w("### Is there a natural break?")
    w("")
    w(f"**Yes, one: between one day and two.** {f0(daily.get(1, 0))} cases fall on day 1 and {f0(daily.get(2, 0))} on "
      f"day 2. Beyond day 1 the cases per day " + ("fall in every successive band" if smooth else "do not fall evenly")
      + ", and no band after day 1 stands apart from its neighbours:")
    w("")
    W += md_table(["Lag (days)", "Cases", "Share", "Cases per day"],
                  [[b["band"], f0(b["n"]), share(b["share"]), "–" if b["per_day"] is None else f"{b['per_day']:,.1f}"]
                   for b in bands])
    w("")
    w("The busiest single days after day 1 are all calendar effects: the return is dated on a season-end date and "
      "the move on a fixed later date.")
    w("")
    W += md_table(["Lag", "Cases", "Most common return date", "Most common move date", "English or Scottish lender"],
                  [[b["lag"], f0(b["n"]), f"{md_words(b['top_return_mmdd'])} ({f0(b['top_return_n'])})",
                    f"{md_words(b['top_move_mmdd'])} ({f0(b['top_move_n'])})", f0(b["english_or_scottish_lender"])]
                   for b in busy])
    w("")
    w("By the date of the return:")
    w("")
    W += md_table(["Return dated", "Cases", "Median lag", "Within 1 day", "Within 30 days", f"Within {win} days"],
                  [[g, f0(grp[g]["N"]), days(grp[g]["median"]), share(grp[g]["within_1_share"]),
                    share(grp[g]["within_30_share"]), share(grp[g]["within_60_share"])]
                   for g in ("return on 30 Jun or 31 Dec", "return on 31 May", "other return date")])
    w("")
    d1 = N["p1_day1_return_mmdd"]
    db = N["p1_deadline_bump"]
    w(f"One smaller bump sits just past the current window: days 62–63 have {f0(db['n'])} cases (neighbouring days "
      f"average {db['neighbour_per_day']:.1f} per day), {f0(db['jun30_to_aug31_or_sep1'])} of them a 30 June return "
      "with the move dated 31 August or 1 September, the summer registration deadline.")
    w("")
    w(f"**Reading.** Most day-1 cases ({f0(N['p1_day1_seasonend'])} of {f0(daily.get(1, 0))}) are a return dated 30 "
      "June or 31 December and a permanent move dated the next day. The other "
      f"{f0(N['p1_day1_other'])} are spread over other return dates, led by the window deadlines 31 January "
      f"({f0(d1.get('01-31', 0))}) and 31 August ({f0(d1.get('08-31', 0))}). This is the pattern Transfermarkt shows "
      "when a permanent move takes effect as the loan ends. The data cannot say whether a clause (an option or "
      f"obligation) or a new negotiation lay behind it. A 31 May return followed by a 1 July move "
      f"({f0(N['p1_may31_jul1'])} cases) "
      "looks the same but is recorded 31 days apart. A single day-count cutoff therefore treats British loans "
      "differently from the rest. A calendar-aware rule, such as \"the move is dated at the first registration date "
      "after the return\", is one alternative; the data do not choose between the two.")
    w("")
    ex = N["p1_tail_example"]
    if ex:
        w(f"The long tail mixes in later, separate deals: {f0(N['p1_tail_n'])} cases come more than a year after the "
          f"return. The largest by fee is {ex['player_name']}, who returned from {ex['borrower_club']} on "
          f"{ex['ending_date'][:10]} and joined them permanently on {ex['next_move_date'][:10]}, "
          f"{days(float(ex['next_move_gap_days']))} days later.")
        w("")
    w("### Negative, same-day and inconsistent cases (kept, not dropped)")
    w("")
    sch = N["p1_by_scheduled"]
    w(f"- **Negative lags: {s1['negative']}.** Rows are ordered by date, so a move dated before the return cannot be "
      "its next move.")
    w(f"- **Same day: {s1['within_0']}.** " + "; ".join(
        f"{r['player_name']} ({r['lender_club']} → {r['borrower_club']}, return and move on {r['ending_date'][:10]})"
        for r in N["p1_same_day_cases"]) + ".")
    w(f"- **Permanent move not from the lender: {N['p1_from_not_lender']}** ({N['p1_from_not_lender_in_current']} of "
      f"them in the current 3,818). {N['p1_from_not_lender_same_org']} come from another side of the lender's "
      "organisation (the loan left from the U19 side, the sale from the first team); the rest from a different club. "
      "They are flagged in the CSV (`permanent_from_is_lender`).")
    w(f"- **Scheduled dates.** Both dates realised: {f0(sch.get('return_realised__move_realised', 0))}; return realised, "
      f"move scheduled: {f0(sch.get('return_realised__move_scheduled', 0))}; both scheduled: "
      f"{f0(sch.get('return_scheduled__move_scheduled', 0))}. No outbound loan is scheduled. The third column of the "
      "table above keeps only cases with both dates realised; the shares barely move.")
    w("")

    # ---- Part 2
    w("## Part 2. What happens after the other loan returns?")
    w("")
    w("The six categories are the existing classes regrouped, so they are mutually exclusive and cover every loan:")
    w("")
    W += md_table(["Category", "Existing classes it contains"],
                  [[f"{c}. {CATEGORIES[c]}", cats[c]["existing_classes"]] for c in CATEGORIES], right=set())
    w("")
    w("The class names are engineering labels, not findings. For example, `early_termination` means a return dated "
      f"more than {tol} days from 30 June or 31 December followed by another move within {imm} days.")
    w("")
    rows = []
    for c in CATEGORIES:
        r = cats[c]
        lagcols = [days(r["lag_median"]), days(r["lag_p75"]), days(r["lag_p90"]), days(r["lag_p95"])] + \
            [share(r[f"within_{x}_share"]) for x in CATEGORY_MARKS]
        rows.append([f"{c}. {CATEGORIES[c]}", f0(r["count"]), share(r["share_of_loans"]),
                     f"{f0(r['realised_ending'])} ({share(r['share_of_realised_endings'])})"] + lagcols)
    W += md_table(["Category", "Loans", f"Share of {f0(loans)}", f"Realised endings (share of {f0(N['realised_endings'])})",
                   "Median lag", "p75", "p90", "p95"] + [f"≤ {x} d" for x in CATEGORY_MARKS], rows)
    w("")
    aw = N["p2_A_why"]
    ck = N["p2_C_kinds"]
    w(f"- Lags for B, C and D are the production follow-on gap, so they are at most {win} days by construction. A, E "
      "and F have no follow-on move.")
    w(f"- **A is not \"nothing happened\".** Of its {f0(cats['A']['count'])} loans: {f0(aw['no_later_movement'])} have "
      f"no later movement, {f0(aw['next_is_release'])} are next released (Without Club, Retired, ...), "
      f"{f0(aw['next_after_window'])} move again after {win} days, and {f0(aw['next_in_window_free_third_club'])} have "
      f"a free or no-fee move to a third club {N['p2_A_in_window_gap_min']}–{win} days later (the rules count such a "
      f"move only within {imm} days). {aw['fee_bearing']} are fee-bearing returns.")
    w(f"- **C** is {f0(ck.get('loan', 0))} new loans and {f0(ck.get('free_move_to_third_club', 0))} free or no-fee moves "
      f"to a third club. {f0(N['p2_C_loan_after_immediate'])} of the new loans come {imm + 1}–{win} days after the "
      f"return: the rules accept a new loan at any gap up to {win} days, other moves only up to {imm}.")
    w(f"- **E**: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in N["p2_E_reasons"].items()) + ".")
    w("")
    w("**Do the categories behave differently?** Within the current window (left panel below), B is concentrated at "
      f"day 1 ({share(cats['B']['within_1_share'])}), while C and D are spread across the {win} days (medians "
      f"{days(cats['C']['lag_median'])} and {days(cats['D']['lag_median'])}). Without any window (right panel, every "
      f"return with a later move: {f0(N['p2_returns_with_next'])} of {f0(N['p2_returns'])}), the next move's timing "
      "depends strongly on what kind of move it is:")
    w("")
    W += md_table(["Next move after the return", "Returns", "Median lag", "p75", "p90", "≤ 1 d", "≤ 7 d", "≤ 30 d",
                   f"≤ {win} d", "≤ 365 d"],
                  [[r["label"], f0(r["n"]), days(r["median"]), days(r["p75"]), days(r["p90"]),
                    share(r["within_1_share"]), share(r["within_7_share"]), share(r["within_30_share"]),
                    share(r["within_60_share"]), share(r["within_365_share"])]
                   for r in N["p2_next_kinds"] if r["n"] > 1])
    w("")
    w(f"![Lag distributions by category]({paths['kinds']})")
    w("")

    # ---- Part 3
    w("## Part 3. Standard loan-ending dates")
    w("")
    w(f"All **{f0(A3['N'])} realised endings** with an observed date (every ending row, return or not; "
      f"{f0(N['observed_endings'] - N['realised_endings'])} scheduled endings are excluded), by calendar date "
      f"(MM-DD, year ignored). All {A3['distinct']} possible dates occur, and only {A3['single_year_dates']} of them in "
      "a single year.")
    w("")
    W += md_table(["Rank", "MM-DD", "Endings", "Share", "Cumulative"],
                  [[r["rank"], r["mmdd"], f0(r["count"]), share(r["share"], 2), share(r["cumulative_share"])]
                   for r in A3["top"]])
    w("")
    samples = [("all_realised", "All realised"), ("uefa_q4_main", "Mapped-UEFA (Q4 main sample)"),
               ("all_realised_through_latest_completed", f"All realised, loans through {N['latest_completed_season']}")]
    rows = [["Endings"] + [f0(P3[k]["N"]) for k, _ in samples]]
    rows += [[f"Top {k} date{'s' if k > 1 else ''} cover"] + [share(P3[s][f"top{k}_cover"]) for s, _ in samples]
             for k in (1, 2, 3, 5, 10)]
    rows += [[f"**Dates needed for {p}%**"] + [f"**{P3[s][f'dates_for_{p}']}**" for s, _ in samples] for p in (80, 90, 95)]
    rows += [[f"{md_words(md)} ({md})"] + [f"{share(P3[s]['named'][md][1])} (rank {P3[s]['named'][md][2]})"
                                           for s, _ in samples] for md in ("05-31", "06-30", "12-31")]
    W += md_table([""] + [lab for _, lab in samples], rows)
    w("")
    w(f"**Answer.** Covering 80% of realised endings takes **{A3['dates_for_80']}** recurring dates, 90% takes "
      f"**{A3['dates_for_90']}** and 95% takes **{A3['dates_for_95']}**. In the mapped-UEFA sample it is "
      f"{U3['dates_for_80']}, {U3['dates_for_90']} and {U3['dates_for_95']}. Beyond the top three, each additional "
      "date adds little, so 90% and 95% coverage cannot be reached with a short list.")
    w("")
    w("Mapped-UEFA sample (both clubs in UEFA countries, realised endings, loans through "
      f"{N['latest_completed_season']}; the same {f0(U3['N'])} as Q4), top 10:")
    w("")
    W += md_table(["Rank", "MM-DD", "Endings", "Share", "Cumulative"],
                  [[r["rank"], r["mmdd"], f0(r["count"]), share(r["share"], 2), share(r["cumulative_share"])]
                   for r in U3["top"][:10]])
    w("")
    w(f"![Loan-ending calendar dates]({paths['dates']})")
    w("")
    w(f"Caveat: the {f0(N['p3_2025_26_realised'])} realised endings of 2025/26 loans are early exits only (the season "
      "had not ended at capture). The third column leaves them out, and the ranking hardly changes.")
    w("")

    # ---- Part 4
    w("## Part 4. Country heterogeneity")
    w("")
    ex = N["p4_excluded"]
    w(f"By lending-club country, for the {N['p4_countries_included']} countries with at least {MIN_COUNTRY_ENDINGS} "
      f"realised endings. Every mapped lender country qualifies except "
      + ", ".join(f"{dc(r['country'])} ({r['N']})" for r in ex)
      + f"; any threshold between {N['p4_max_excluded_N'] + 1} and {N['p4_min_included_N']} gives the same set. "
      f"{f0(N['p4_unmapped_lender'])} realised endings have a lender of unknown country and are left out, not pooled.")
    w("")
    W += md_table(["Lender country", "N", "Most common (share)", "Top 3 dates", "Top-3 share", "On global top 3",
                   "05-31", "06-30", "12-31", "Dates for 90%"],
                  [[dc(r.country), f0(r.N), f"{r.most_common_mmdd} ({share(r.most_common_share)})", r.top3_mmdd,
                    share(r.top3_share), share(r.share_on_global_top3), share(r["share_05-31"]),
                    share(r["share_06-30"]), share(r["share_12-31"]), r.dates_for_90pct]
                   for _, r in lend.iterrows()], right={1, 4, 5, 6, 7, 8, 9})
    w("")
    gt3 = ", ".join(N["p4_global_top3"])
    low = N["p4_global_top3_below_70"]
    w(f"- **Most common date other than 30 June:** {not_jun30}.")
    w(f"- **The global top 3 ({gt3}) cover less than 70% of endings in** "
      + ", ".join(f"{dc(r['country'])} ({share(r['share_on_global_top3'])})" for r in low) + ".")
    w(f"- **Top-3 share by country** ranges from {share(N['p4_top3_range'][0])} to {share(N['p4_top3_range'][1])}, and "
      f"the dates needed for 90% from {N['p4_dates90_range'][0]} to {N['p4_dates90_range'][1]}.")
    diff = N["p4_lender_vs_borrower_top_differs"]
    w("- **Lender or borrower country?** Grouping by the borrower's country instead changes the most common date "
      "for " + "; ".join(f"{dc(r['country'])} ({r['most_common_mmdd_l']} as lender, {r['most_common_mmdd_b']} as "
                         "borrower)" for r in diff)
      + ". Both versions are in `loan_ending_dates_by_country.csv`.")
    w("")
    brit = sorted(dc(r["country"]) for r in low if r["country"] in ("England", "Scotland"))
    cal = [dc(r["country"]) for r in low if r["country"] not in ("England", "Scotland")]
    w(f"**What this says about one global definition.** The global top 3 cover at least 70% of endings in "
      f"{N['p4_countries_included'] - len(low)} of the {N['p4_countries_included']} countries. The {len(low)} where "
      f"they cover less are the British countries ({', '.join(brit)}), where many loans end on 31 May, and countries "
      f"whose league plays a calendar-year season ({', '.join(cal)}), where they end in November, December or "
      "January. A single global list labels those countries' normal endings as off-cycle; a country-specific list "
      "does not.")
    w("")

    # ---- Part 5
    w("## Part 5. On-cycle and off-cycle, descriptively")
    w("")
    w(f"Each definition splits the {f0(N['realised_endings'])} realised endings into on-cycle (the ending falls on a "
      "standard date) and off-cycle (it doesn't). Country-specific definitions cannot classify endings whose lender's "
      "country is unknown or has too few endings; those are shown separately. \"Next-day permanent\" means the next "
      f"move is a permanent move (to any club) within {CYCLE_FAST_DAYS} day of the return, the break seen in Part 1. "
      f"Its share among moves back to the borrower is given separately. **Off-cycle does not mean early termination.**")
    w("")
    rows = []
    for d in cyc.definition.unique():
        g = cyc[cyc.definition == d]
        on, off = g[g.state == "on-cycle"].iloc[0], g[g.state == "off-cycle"].iloc[0]
        nc = g[g.state == "not classifiable"]
        rows.append([f"{d}: {on.label}", cs(on.n, N["realised_endings"]), cs(off.n, N["realised_endings"]),
                     f0(nc.n.iloc[0]) if len(nc) else "0",
                     f"{f0(on.non_plain_n)} ({share(on.non_plain_share, 2)}) / {f0(off.non_plain_n)} ({share(off.non_plain_share, 2)})",
                     f"{share(on.fast_permanent_share)} / {share(off.fast_permanent_share)}",
                     f"{share(on.fast_permanent_to_borrower_share)} / {share(off.fast_permanent_to_borrower_share)}"])
    W += md_table(["Definition", "On-cycle", "Off-cycle", "Not classifiable", "Non-plain ending, on / off",
                   "Next-day permanent, on / off", "… to the borrower, on / off"], rows)
    w("")
    w(f"- In {off_more_nonplain} of {ndefs} definitions, off-cycle endings have a higher share of non-plain endings "
      f"than on-cycle ones. But non-plain endings are rare ({f0(N['p5_nonplain_total'])} of the "
      f"{f0(N['realised_endings'])} realised endings), so each group's share stays around 1% or lower.")
    w(f"- In {off_less_fast} of {ndefs} definitions, off-cycle endings are **less** often followed by a next-day "
      "permanent move. Quick moves cluster at the season-end dates, not away from them.")
    w("- K90 and K95 are ≈90% and ≈95% on-cycle within each country **by construction**. What they add is how many "
      "dates each country needs (Part 4).")
    w(f"- T10 is the rule already inside the production code: an ending within {tol} days of 30 June or 31 December. "
      "It treats British 31 May endings as off-cycle.")
    w("- None of these definitions separates the two groups sharply, so the data do not single one out. The "
      "country-specific versions (K3, K90) at least stop calling British and calendar-year countries' normal "
      f"endings off-cycle, at the cost of leaving the {f0(int(cyc_get('K3', 'not classifiable', 'n')))} endings "
      "whose lender country is unknown or too small unclassified.")
    w("")

    # ---- Part 6
    w("## Part 6. What this means for the 48.6% figure")
    w("")
    parts = N["p6_parts"]
    w(f"The official secondary statistic is {frac(nono, loans)} of loans in a non-ordinary surrounding registration "
      f"sequence. It is left unchanged; this part only explores how sensitive it is. Its {f0(nono)} loans are B "
      f"{f0(parts['B'])} + C {f0(parts['C'])} + D {f0(parts['D'])} + E {f0(parts['E'])} + fee-bearing returns with "
      f"no follow-on move {f0(parts['A_fee_bearing'])}.")
    w("")
    bt = {r["category"]: r for r in N["p6_bands"]}
    W += md_table(["Category", "Loans", "0 days", "1 day", "2–7", f"8–{imm}", f"{imm + 1}–{win}"],
                  [[f"{c}. {CATEGORIES[c]}", f0(bt[c]["n"]), f0(bt[c]["b_0_0"]), f0(bt[c]["b_1_1"]), f0(bt[c]["b_2_7"]),
                    f0(bt[c]["b_8_21"]), f0(bt[c]["b_22_60"])] for c in "BCD"]
                  + [["E + fee-bearing returns (no follow-on move)", f0(N["p6_no_follow_on"]), "–", "–", "–", "–", "–"]])
    w("")
    w(f"- **Followed very quickly:** {cs(N['p6_within_1'], nono)} within 1 day and {cs(N['p6_within_7'], nono)} within "
      "7 days.")
    w(f"- **Non-ordinary only because of a later move:** {cs(N['p6_22_60'], nono)} count only because of a move "
      f"{imm + 1}–{win} days after the return, and {cs(N['p6_8_60'], nono)} if the line is drawn at 7 days. No loan "
      f"counts because of a move after {win} days: the search stops there. In the other direction, "
      f"{f0(sum(N['p6_beyond_window_kinds'].values()))} returns followed by a move after {win} days are counted as "
      f"ordinary, {f0(N['p6_beyond_window_kinds'].get('permanent_to_borrower', 0))} of them permanent moves back to "
      "the borrower.")
    w(f"- **Sensitivity to the windows.** The production rules use two windows: {imm} days for most follow-on moves "
      f"and {win} days for moves back to the borrower, sales and new loans. Setting both to one value X:")
    w("")
    order = ["0", "1", "3", "7", "14", "21", "30", "60", "90", "180", "365", "none"]
    W += md_table(["Window X (days)", "Non-ordinary", "Share", "B", "C", "D"],
                  [[x if x != "none" else "no limit", f0(grid[x]["nonordinary"]), share(grid[x]["share"]),
                    f0(grid[x]["cat_B"]), f0(grid[x]["cat_C"]), f0(grid[x]["cat_D"])] for x in order]
                  + [[f"**current ({imm} / {win})**", f"**{f0(cur['nonordinary'])}**", f"**{share(cur['share'])}**",
                      f0(cur["cat_B"]), f0(cur["cat_C"]), f0(cur["cat_D"])]])
    w("")
    g160, g21n = N["p6_grid_1_60"], N["p6_grid_21_none"]
    w(f"The two windows can also be varied separately: `loan_timing_window_sensitivity.csv` has every combination. "
      f"For example, 1 day / {win} days gives {share(g160['share'])}, and {imm} days / no limit gives "
      f"{share(g21n['share'])}. The {tol}-day season-end tolerance does not affect the total, because it only decides "
      "which of two C labels a loan gets. **So 48.6% is mainly a statement about the windows**: widen them and it "
      "approaches the share of players who ever move again.")
    w("")
    w("**Subgroups that look most useful for later contract research** (suggestions only; the data describe "
      "sequences, not clauses):")
    w("")
    w(f"1. **B, next day** ({f0(N['p6_B_day1'])} loans, including {s1['within_0']} same-day): the tightest pattern, "
      "and the group where a purchase option or obligation is most plausible. Stage 2 could measure how often one "
      "existed.")
    w(f"2. **Fee-bearing returns** ({f0(N['p6_fee_bearing_returns_total'])}): the only endings where Transfermarkt "
      "records money on the return row.")
    w(f"3. **Off-cycle returns followed quickly by another move** (for example the {f0(N['p6_C_offcycle_label'])} "
      f"returns dated more than {tol} days from 30 June or 31 December with a next move within {imm} days): loans "
      "that appear to be cut short. Country-specific dates change this group little: "
      f"{f0(N['p5_et_on_own_country_top3'])} of them fall on one of their lender country's top 3 dates (most in "
      + ", ".join(f"{dc(k)}, {v}" for k, v in N["p5_et_on_own_top_country"].items())
      + f"), and {f0(N['p5_et_lender_not_classifiable'])} have a lender whose country is unknown or too small.")
    w(f"4. **D, next day** ({f0(N['p6_D_day1'])} of {f0(N['p6_D'])}): a sale to a third club dated the day after the "
      "return.")
    w("")
    w(f"Lower priority: B at 2–{win} days ({f0(N['p6_B_2_60'])}) and new loans {imm + 1}–{win} days after a return, "
      "whose timing looks like ordinary later dealing.")
    w("")

    # ---- Part 7
    w("## Part 7. Ready for next week's playing-time join")
    w("")
    w(f"`loan_follow_on_timing.csv` has one row per loan ({f0(loans)} rows). It carries: player_id, loan_event_id, "
      "lender and borrower club_id and name, loan start date and whether it was scheduled, ending event_id and date "
      "and whether it is a return and scheduled, the next move's event_id, date, type, from and to club_id and "
      "whether it was scheduled, lag_days (`next_move_gap_days`), season, lender and borrower country and "
      "confederation, the ending-label class, the current `economic_ending`, the A–F `sequence_category`, and "
      "`borrower_club_has_appearance_rows_during_loan`, a coverage flag, not a playing-time measure.")
    w("")
    w("**How it joins to the snapshot's match tables** (inspected only; no playing-time measure was built):")
    w("")
    W += md_table(["Need", "Available", "Note"], [
        ["Player", "`appearances.player_id`", "Same Transfermarkt id as Stage 1C `player_id`."],
        ["Club", "`appearances.player_club_id`", "Same Transfermarkt id as Stage 1C club ids; the club the player "
                                                 "appeared for in that game."],
        ["Date", "`appearances.date` (DATE)", "Compare with `loan_start_date` and `ending_date`."],
        ["Game / competition", "`appearances.game_id` (INTEGER), `competition_id`; `games.game_id` (VARCHAR), "
                               "`games.season`", "`game_id` must be cast to join `games`."],
        ["Minutes", "`appearances.minutes_played`",
         f"Present on every row ({f0(p7['minutes_null'])} missing; range {p7['minutes_min']}–{p7['minutes_max']}). "
         "No calculation needed."],
        ["Unused substitutes", "`game_lineups` (type = starting_lineup / substitutes)",
         f"From {p7['lineups_first_date']}; no minutes. `appearances` holds only players who got on the pitch."],
    ], right=set())
    w("")
    w(f"**Seasons covered:** appearances run {p7['appearances_first_date']} to {p7['appearances_last_date']} (season "
      f"labels {p7['appearance_seasons'][0]}–{p7['appearance_seasons'][1]}); lineups start {p7['lineups_first_date']}.")
    w("")
    w(f"**Coverage for the {f0(p7['loans_checked'])} loans since {p7['first_loan_season_checked']}:**")
    w("")
    rows = [["All", f0(p7["loans_checked"]), cs(p7["loans_borrower_observed"], p7["loans_checked"]),
             cs(p7["loans_player_played"], p7["loans_checked"]),
             pc(p7["loans_played_given_observed"], p7["loans_borrower_observed"])]]
    for r in p7["by_period"]:
        rows.append([r["period"], f0(r["loans"]), cs(r["borrower_observed"], r["loans"]), cs(r["player_played"], r["loans"]),
                     pc(r["played_given_observed"], r["borrower_observed"])])
    W += md_table(["Loans", "N", "Borrower has any appearance rows during the loan", "Player appears for the borrower "
                   "at least once", "… as share of observed borrowers"], rows)
    w("")
    w("**Join caveats:**")
    w("")
    w(f"1. **Appearances exist only for the {len(p7['leagues_with_appearances'])} long-covered European leagues** "
      f"({', '.join(p7['leagues_with_appearances'])}), part of their domestic cups, and "
      f"the UEFA Champions and Europa Leagues. The {len(p7['leagues_without_appearances'])} leagues added in "
      f"{'/'.join(p7['leagues_without_appearances_first_season'])} ({', '.join(p7['leagues_without_appearances'])}) "
      f"have games but no appearances, and neither do {', '.join(p7['other_competitions_without_appearances'])}. A "
      "loan to any other club, which covers most lower-division and youth loans, has at most a few rows (cup games "
      "against covered clubs). Missing rows there are missing data, not zero minutes.")
    w(f"2. Even for borrowers mapped to a country, only {pc(p7['loans_borrower_observed_mapped'], p7['loans_borrower_mapped'])} "
      "are observed during the loan. Next week's measure needs an explicit \"borrower observed\" condition (the flag "
      "in the CSV) before a zero can be read as no playing time.")
    w(f"3. {f0(p7['appearances_in_competitions_missing_from_competitions_table'])} appearance rows belong to "
      "competitions missing from `competitions` (e.g. KLUB, POCP). National-team games must be excluded, since there "
      "`player_club_id` is the national team.")
    w("4. Loan windows use Transfermarkt's recorded dates, so a British loan \"ends\" on 31 May. Scheduled and open "
      "loans were capped here at the last appearance date.")
    w("5. The players are the 2023/24–2025/26 cohort (see `dataset_provenance_and_coverage.md`), so earlier seasons' "
      "loans are those of players who were still active later.")
    w("")

    # ---- Questions
    w("## Questions for Professor Freund")
    w("")
    w(f"1. **What lag should count as an immediate conversion?** The data show one break, between one day and two: "
      f"next-day moves are {share(s1['within_1_share'])} of permanent moves back to the borrower. After that the "
      f"distribution declines smoothly ({imm} days would cover {share(s1['within_21_share'])} and {win} days "
      f"{share(s1['within_60_share'])}). Should \"immediate\" mean next-day, next-day plus a calendar-aware allowance "
      "for 31 May returns, or a wider day window?")
    w(f"2. **Should standard ending dates be global or country-specific?** The global top 3 cover "
      f"{share(A3['top3_cover'])} overall. But {len(N['p4_not_jun30_top'])} of the {N['p4_countries_included']} "
      "countries have a most common date other than 30 June, and the global list covers less than 70% of endings in "
      f"{len(N['p4_global_top3_below_70'])} of the {N['p4_countries_included']}.")
    w("3. **Which off-cycle or follow-on subgroups should Stage 2 prioritise?** Candidates (Part 6): next-day "
      f"conversions ({f0(N['p6_B_day1'])}), fee-bearing returns ({f0(N['p6_fee_bearing_returns_total'])}), off-cycle "
      f"returns followed quickly by another move ({f0(N['p6_C_offcycle_label'])} under the current rule), and next-day "
      f"sales ({f0(N['p6_D_day1'])}).")
    w(f"4. **For the playing-time join, should the sample be limited to loans whose borrower is observed** "
      f"({share(p7['loans_borrower_observed'] / p7['loans_checked'])} of loans since {p7['first_loan_season_checked']}, "
      "almost all in the 14 European leagues), with the rest treated as missing?")
    w("")
    w("## Files")
    w("")
    for f in (LAGS_CSV, EPISODES_CSV, DATES_CSV, COUNTRY_CSV, CATEGORY_CSV, NEXT_KIND_CSV, WINDOW_CSV, CYCLE_CSV,
              NUMBERS_JSON):
        w(f"- `{f.name}`")
    for p in paths.values():
        w(f"- `{p}`")
    w("")
    return "\n".join(W) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


ID_COLS = ("player_id", "lender_club_id", "borrower_club_id", "next_move_from_club_id", "next_move_to_club_id")


def write_tables(tb: dict) -> None:
    T = tb["T"].copy()
    for frame in (T, tb["lags"]):
        for c in ID_COLS:
            if c in frame:
                frame[c] = pd.to_numeric(frame[c]).astype("Int64")
    for c in ("loan_start_date", "ending_date", "next_move_date"):
        T[c] = pd.to_datetime(T[c]).dt.date
    T.to_csv(EPISODES_CSV, index=False)
    tb["lags"].to_csv(LAGS_CSV, index=False)
    tb["dates"].to_csv(DATES_CSV, index=False)
    tb["country"].to_csv(COUNTRY_CSV, index=False)
    tb["cats"].to_csv(CATEGORY_CSV, index=False)
    tb["next_kinds"].to_csv(NEXT_KIND_CSV, index=False)
    tb["grid"].to_csv(WINDOW_CSV, index=False)
    tb["cycle"].to_csv(CYCLE_CSV, index=False)


def summary(N: dict) -> str:
    s = N["p1_any_lag_stats"]
    A3 = N["p3"]["all_realised"]
    p7 = N["p7"]
    lines = [
        f"SAME-BORROWER PERMANENT CASES:  {N['p1_current']:,} current (= {N['p1_current_with_return']:,} with a return, "
        f"all <= {FOLLOW_ON_WINDOW_DAYS} d by construction, + {N['p1_current_no_return']} with no return row); "
        f"{N['p1_any_lag']:,} at any lag (shares below are of these)",
    ]
    for lab, k in (("SAME DAY", 0), ("WITHIN 1 DAY", 1), ("WITHIN 3 DAYS", 3), ("WITHIN 7 DAYS", 7),
                   ("WITHIN 14 DAYS", 14), ("WITHIN 21 DAYS", 21), ("WITHIN 30 DAYS", 30), ("WITHIN 60 DAYS", 60)):
        lines.append(f"{lab + ':':32}{s[f'within_{k}']:,} ({100 * s[f'within_{k}_share']:.1f}%)")
    lines += [f"{'MEDIAN LAG:':32}{days(s['median'])} days", f"{'P90 LAG:':32}{days(s['p90'])} days",
              f"{'P95 LAG:':32}{days(s['p95'])} days", ""]
    lines.append(f"{'TOP 10 ENDING MM-DD VALUES:':32}" + ", ".join(
        f"{r['mmdd']} {100 * r['share']:.1f}%" for r in A3["top"][:10]))
    lines += [f"{'TOP 3 DATE COVERAGE:':32}{100 * A3['top3_cover']:.1f}%",
              f"{'TOP 5 DATE COVERAGE:':32}{100 * A3['top5_cover']:.1f}%",
              f"{'DATES NEEDED FOR 80%:':32}{A3['dates_for_80']}",
              f"{'DATES NEEDED FOR 90%:':32}{A3['dates_for_90']}",
              f"{'DATES NEEDED FOR 95%:':32}{A3['dates_for_95']}", ""]
    parts = N["p6_parts"]
    lines += [f"{'BROAD NON-ORDINARY CASES:':32}{N['p6_nonordinary']:,} / {N['loans']:,} = "
              f"{100 * N['p6_nonordinary'] / N['loans']:.1f}% (unchanged)",
              f"{'BREAKDOWN BY FOLLOW-ON CATEGORY:':32}B {parts['B']:,}; C {parts['C']:,}; D {parts['D']:,}; "
              f"E {parts['E']}; fee-bearing return, no follow-on {parts['A_fee_bearing']}. "
              f"Follow-on within 1 d: {N['p6_within_1']:,}; {FOLLOW_ON_IMMEDIATE_DAYS + 1}-{FOLLOW_ON_WINDOW_DAYS} d: "
              f"{N['p6_22_60']:,}", ""]
    lines += [f"{'CURRENT 21-DAY WINDOW COVERAGE:':32}{100 * s['within_21_share']:.1f}% of same-borrower permanent moves",
              f"{'CURRENT 60-DAY WINDOW COVERAGE:':32}{100 * s['within_60_share']:.1f}%", ""]
    lines += [f"{'PLAYING-TIME JOIN FEASIBLE:':32}YES, on player_id + club_id + date, where appearances exist",
              f"{'MINUTES FIELD AVAILABLE:':32}YES, appearances.minutes_played ({p7['minutes_null']} missing)",
              f"{'KEY JOIN CAVEATS:':32}appearances only for {len(p7['leagues_with_appearances'])} European leagues + "
              f"cups/UEFA; borrower observed for {100 * p7['loans_borrower_observed'] / p7['loans_checked']:.1f}% of "
              f"loans since {p7['first_loan_season_checked']}; game_id type cast; no unused-sub minutes", ""]
    return "\n".join(lines)


def main() -> dict:
    N, tb = compute()
    paths = figures(N, tb)
    write_tables(tb)
    NUMBERS_JSON.write_text(json.dumps(_jsonable(N), indent=1, default=str))
    REPORT_MD.write_text(render(N, tb, paths))
    print(f"wrote {REPORT_MD.name}")
    print(summary(N))
    return N


if __name__ == "__main__":
    main()
