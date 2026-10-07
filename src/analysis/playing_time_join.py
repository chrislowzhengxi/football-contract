"""Can transfer episodes be joined cleanly to playing time? Join audit and sample design.

    python -m src.analysis.playing_time_join

Read-only. Takes the real transfer episodes of the current universe (loans and
permanent moves, unchanged definitions) and joins each to the snapshot's league
appearances at the receiving club B, in B's league season of arrival. It
measures coverage only: no loan-vs-permanent comparison of playing time is made.

Missing coverage is never read as zero minutes. An episode gets a playing-time
value only if every league game of B in its window has a complete appearance
record; then, and only then, a player with no appearance row truly did not play.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from .daniel_scope_final import f0, md_table
from .loan_episodes import ROOT, _sorted
from .loan_timing_exploration import share
from .provenance_audit import player_source_file, source_file_capture

OUT = ROOT / "data" / "outputs" / "rebuild"
AUDIT_CSV = OUT / "playing_time_join_audit.csv"
COVERAGE_CSV = OUT / "playing_time_join_coverage_by_league_season.csv"
EPISODES_CSV = OUT / "playing_time_join_episodes.csv"
REPORT_MD = OUT / "playing_time_join_feasibility.md"
NUMBERS_JSON = OUT / "playing_time_join_numbers.json"

KINDS = {"loan": "loan", "permanent_paid": "permanent", "free_transfer": "permanent",
         "undisclosed_fee": "permanent", "no_fee_shown": "permanent"}
COMPLETE_ROWS = 11              # a club-game with fewer appearance rows than a starting XI is incomplete
FULL_SEASON_SHARE = 2 / 3       # sample design: arrived with at least two-thirds of B's league games left
SKIP_TYPES = ("youth_or_internal", "other")
NOT_COVERED = "(not in a covered league)"

STEPS = ["candidate", "realised", "covered_league_season", "matched", "reliable"]
STEP_LABELS = {
    "candidate": "Candidate episodes",
    "realised": "Move had happened at capture (not scheduled)",
    "covered_league_season": "Receiving club B plays in a covered league in its arrival season",
    "matched": "Player in the playing-time universe and B has appearance rows that season",
    "reliable": "Playing-time outcome reliably calculable",
}
REASONS = {
    "scheduled_move": "Move still scheduled at capture",
    "not_covered": "B not in a covered league in its arrival season (lower division, other league, youth side, "
                   "or before 2012/13)",
    "player_not_in_universe": "Player not in the players table",
    "club_season_without_rows": "B has no appearance rows that season",
    "no_league_games_in_window": "No league game of B between arrival and departure (the player left B again, "
                                 "typically bought and immediately loaned or sold on)",
    "window_after_capture": "Window ends after the player's history was captured (a departure could be missing)",
    "incomplete_club_games": "Some league game of B in the window lacks a complete appearance record",
    "appearances_for_other_club": "The player appears for another club during the window (a move is missing "
                                  "from his history)",
}


# ---------------------------------------------------------------------------
# Match tables
# ---------------------------------------------------------------------------

def match_tables(con) -> dict:
    games = con.sql("""select game_id, competition_id, competition_type, cast(season as int) season, date,
                              home_club_id, away_club_id from games""").df()
    games["date"] = pd.to_datetime(games.date)
    app = con.sql("""select cast(a.game_id as varchar) game_id, a.player_club_id club_id, a.player_id,
                            a.minutes_played, a.date,
                            coalesce(g.competition_type, '') = 'national_team_competition' national_team
                     from appearances a left join games g on g.game_id = cast(a.game_id as varchar)""").df()
    app["date"] = pd.to_datetime(app.date)
    rows = app.groupby(["game_id", "club_id"]).size().rename("rows")
    sides = pd.concat([games.rename(columns={"home_club_id": "club_id"}).drop(columns="away_club_id"),
                       games.rename(columns={"away_club_id": "club_id"}).drop(columns="home_club_id")])
    sides = sides.join(rows, on=["game_id", "club_id"]).fillna({"rows": 0})
    sides["complete"] = sides.rows >= COMPLETE_ROWS
    checks = con.sql("""select count(*) n, count(distinct a.appearance_id) ids,
                               count(distinct (a.player_id, a.game_id)) player_games,
                               count(*) - count(a.minutes_played) null_minutes,
                               sum(case when a.player_club_id not in (g.home_club_id, g.away_club_id) then 1 else 0 end) club_not_in_game,
                               sum(case when a.date <> g.date then 1 else 0 end) date_mismatch,
                               sum(case when a.competition_id <> g.competition_id then 1 else 0 end) competition_mismatch,
                               sum(case when g.game_id is null then 1 else 0 end) no_game_row
                        from appearances a left join games g on g.game_id = cast(a.game_id as varchar)""").df().iloc[0]
    lineups = con.sql(f"""with cg as (select cast(game_id as varchar) game_id, player_club_id club_id from appearances
                                       group by 1, 2 having count(*) >= {COMPLETE_ROWS}),
                               lg as (select game_id from games where competition_type = 'domestic_league'),
                               l as (select distinct cast(game_id as varchar) game_id, club_id, player_id, type from game_lineups),
                               a as (select distinct cast(game_id as varchar) game_id, player_id from appearances)
                          select l.type, count(*) n, sum(case when a.player_id is null then 1 else 0 end) not_in_appearances
                          from l join cg on cg.game_id = l.game_id and cg.club_id = l.club_id
                                 join lg on lg.game_id = l.game_id
                                 left join a on a.game_id = l.game_id and a.player_id = l.player_id
                          group by 1""").df()
    players = set(con.sql("select player_id from players").df().player_id)
    return {"games": games, "sides": sides, "app": app, "checks": checks, "lineups": lineups, "players": players}


def league_coverage(sides: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Per domestic league and season: club-games and the share with a complete appearance record.
    A league counts as long-run covered if it has appearance rows in every season of the snapshot."""
    lg = sides[sides.competition_type == "domestic_league"]
    cov = lg.groupby(["competition_id", "season"]).agg(
        games=("game_id", "nunique"), club_games=("game_id", "size"), club_games_with_rows=("rows", lambda r: int((r > 0).sum())),
        club_games_complete=("complete", "sum"), first_game=("date", "min"), last_game=("date", "max")).reset_index()
    cov["share_complete"] = cov.club_games_complete / cov.club_games
    seasons = sorted(lg.season.unique())
    has = cov[cov.club_games_with_rows > 0].groupby("competition_id").season.nunique()
    covered = sorted(has[has == len(seasons)].index)
    return cov, covered


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

def candidate_episodes(u) -> pd.DataFrame:
    ep = u.episodes
    E = ep[ep.episode_type.isin(KINDS)].copy()
    E["kind"] = E.episode_type.map(KINDS)
    E = E.drop(columns="transfer_season").rename(columns={"_date": "move_date", "from_club_id": "a_club_id", "to_club_id": "b_club_id",
                          "from_club_name": "a_club", "to_club_name": "b_club", "season": "transfer_season"})
    E["scheduled"] = E.transfermarkt_future_transfer.astype(bool)

    # Departure from B: the player's next substantive movement after the episode row (any type),
    # skipping internal registrations and draft rows, as the loan code does.
    d = _sorted(u.rows)
    pid = d.player_id.to_numpy()
    skip = d.is_internal_move.to_numpy().astype(bool) | np.isin(d.transfer_type_normalized.to_numpy(), SKIP_TYPES)
    dates = d._date.to_numpy()
    sched = d.transfermarkt_future_transfer.to_numpy().astype(bool)
    pos = pd.Series(np.arange(len(d)), index=d.event_id.to_numpy())
    dep, dep_sched = [], []
    for eid, p in zip(E.event_id, E.player_id):
        j = pos[eid] + 1
        while j < len(d) and pid[j] == p and skip[j]:
            j += 1
        ok = j < len(d) and pid[j] == p
        dep.append(dates[j] if ok else np.datetime64("NaT"))
        dep_sched.append(bool(sched[j]) if ok else False)
    E["departure_date"] = pd.to_datetime(dep)
    E["departure_scheduled"] = dep_sched

    src = E.player_id.map(player_source_file())
    cap = source_file_capture(u.rows).set_index("source_season_file").last_realised_date
    E["history_file"] = src.to_numpy()
    E["history_captured_through"] = pd.to_datetime(src.map(cap.astype(str)))

    E = E.sort_values(["player_id", "move_date", "event_id"])
    E["prior_spell_at_b"] = E.groupby(["player_id", "b_club_id"]).cumcount() > 0
    prev = E.groupby("player_id")[["kind", "b_club_id"]].shift()
    E["preceded_by_loan_to_b"] = (prev.kind == "loan") & (prev.b_club_id == E.b_club_id)

    L = u.loans.set_index("loan_event_id")
    ending = pd.Series(np.select([L.terminal_date.isna(), L.terminal_is_return.astype(bool)],
                                 ["no ending recorded", "return to lender"], default="other movement"), index=L.index)
    E["loan_ending"] = E.event_id.map(ending)
    E["loan_return_date"] = E.event_id.map(L.terminal_date.where(L.terminal_is_return))
    E["loan_return_scheduled"] = E.event_id.map(L.terminal_scheduled_future.where(L.terminal_is_return))
    return E.reset_index(drop=True)


def arrival_season(E: pd.DataFrame, sides: pd.DataFrame, covered: list[str]) -> pd.DataFrame:
    """B's league season of arrival: the covered season s of B with prev_end(L, s) < move date <= end(L, s),
    trying s = year - 1 then year. For a league's first season the previous season is taken to end on
    the same calendar day one year before its end."""
    lg = sides[(sides.competition_type == "domestic_league") & sides.competition_id.isin(covered)]
    league_end = lg.groupby(["competition_id", "season"]).date.max()
    club = lg.groupby(["club_id", "season"]).competition_id.first()

    def prev_end(league, s):
        if (league, s - 1) in league_end.index:
            return league_end[(league, s - 1)]
        return league_end[(league, s)] - pd.DateOffset(years=1)

    out_s, out_l = [], []
    for b, t in zip(E.b_club_id, E.move_date):
        found = (np.nan, NOT_COVERED)
        for s in (t.year - 1, t.year):
            league = club.get((b, s))
            if league is not None and prev_end(league, s) < t <= league_end[(league, s)]:
                found = (s, league)
                break
        out_s.append(found[0])
        out_l.append(found[1])
    E = E.copy()
    E["arrival_season"] = pd.array(out_s, dtype="Int64")
    E["receiving_league"] = out_l
    return E


def measure(E: pd.DataFrame, sides: pd.DataFrame, app: pd.DataFrame) -> pd.DataFrame:
    """Window, available league games and league minutes at B for episodes in a covered league season."""
    lg = sides[sides.competition_type == "domestic_league"]
    by_cs = {k: g.sort_values("date") for k, g in lg.groupby(["club_id", "season"])}
    other = sides[sides.competition_type != "domestic_league"]
    other_by_club = {k: g for k, g in other.groupby("club_id")}
    la = app[app.game_id.isin(set(lg.game_id))]
    by_pc = {k: g for k, g in la.groupby(["player_id", "club_id"])}
    club_app = app[~app.national_team]
    by_p = {k: g for k, g in club_app[club_app.player_id.isin(set(E.player_id))].groupby("player_id")}
    rows = []
    for r in E.itertuples():
        rec = {"window_start": pd.NaT, "window_end": pd.NaT, "season_games": np.nan, "games_remaining_at_arrival": np.nan,
               "available_games": np.nan, "complete_games": np.nan, "appearances": np.nan, "minutes": np.nan,
               "nonleague_games_in_window": np.nan, "nonleague_games_without_rows": np.nan, "departed_in_window": False,
               "club_season_rows": np.nan, "other_club_appearances": np.nan}
        if pd.notna(r.arrival_season):
            g = by_cs[(r.b_club_id, int(r.arrival_season))]
            season_end = g.date.max()
            end = season_end
            if pd.notna(r.departure_date) and r.departure_date <= season_end:
                end = r.departure_date - pd.Timedelta(days=1)
                rec["departed_in_window"] = True
            w = g[(g.date >= r.move_date) & (g.date <= end)]
            a = by_pc.get((r.player_id, r.b_club_id))
            a = a[a.game_id.isin(set(w.game_id))] if a is not None else a
            o = other_by_club.get(r.b_club_id)
            o = o[(o.date >= r.move_date) & (o.date <= end)] if o is not None else None
            rec.update({"window_start": r.move_date, "window_end": end, "season_games": len(g),
                        "club_season_rows": int(g.rows.sum()),
                        "other_club_appearances": 0 if r.player_id not in by_p else int(
                            ((by_p[r.player_id].club_id != r.b_club_id) & (by_p[r.player_id].date >= r.move_date)
                             & (by_p[r.player_id].date <= end)).sum()),
                        "games_remaining_at_arrival": int((g.date >= r.move_date).sum()),
                        "available_games": len(w), "complete_games": int(w.complete.sum()),
                        "appearances": 0 if a is None else len(a),
                        "minutes": 0 if a is None else int(a.minutes_played.sum()),
                        "nonleague_games_in_window": 0 if o is None else len(o),
                        "nonleague_games_without_rows": 0 if o is None else int((o.rows == 0).sum())})
        rows.append(rec)
    M = pd.DataFrame(rows, index=E.index)
    return pd.concat([E, M], axis=1)


def classify(E: pd.DataFrame, players: set) -> pd.DataFrame:
    E = E.copy()
    covered = E.arrival_season.notna()
    in_players = E.player_id.isin(players)
    reason = np.select(
        [E.scheduled, ~covered, ~in_players, E.club_season_rows == 0,
         E.available_games.fillna(0) == 0, E.window_end > E.history_captured_through,
         E.complete_games < E.available_games, E.other_club_appearances > 0],
        ["scheduled_move", "not_covered", "player_not_in_universe", "club_season_without_rows",
         "no_league_games_in_window", "window_after_capture", "incomplete_club_games",
         "appearances_for_other_club"], default="")
    E["exclusion_reason"] = reason
    E["realised"] = ~E.scheduled
    E["covered_league_season"] = E.realised & covered
    E["matched"] = E.covered_league_season & in_players & (E.club_season_rows > 0)
    E["reliable"] = E.exclusion_reason == ""
    for c in ("minutes", "appearances"):
        E[c] = E[c].where(E.reliable)                       # never a number unless the window is complete
    E["available_minutes"] = (90 * E.available_games).where(E.reliable)
    E["minutes_share"] = E.minutes / E.available_minutes
    E["true_zero"] = E.reliable & (E.minutes == 0)
    E["arrived_full_season"] = (E.games_remaining_at_arrival / E.season_games) >= FULL_SEASON_SHARE
    return E


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def waterfall(E: pd.DataFrame) -> dict:
    out = {}
    for k, g in E.groupby("kind"):
        rec = {s: int(g[s].sum()) if s != "candidate" else len(g) for s in STEPS}
        rec["matched_with_appearance_record"] = int((g.matched & (g.appearances_any > 0)).sum())
        rec["reliable_with_appearances"] = int((g.reliable & (g.minutes > 0)).sum())
        rec["reliable_true_zero"] = int(g.true_zero.sum())
        rec["excluded"] = {r: int((g.exclusion_reason == r).sum()) for r in REASONS}
        out[k] = rec
    return out


def audit_table(E: pd.DataFrame) -> pd.DataFrame:
    dims = {"all": pd.Series("all", index=E.index), "transfer_season": E.transfer_season,
            "arrival_season": E.arrival_season.astype("string").fillna("(none)"),
            "receiving_league": E.receiving_league, "receiving_country": E.to_country.fillna("(unmapped)"),
            "transfer_year": E.move_date.dt.year.astype(str)}
    rows = []
    for dim, val in dims.items():
        for (k, v), g in E.groupby([E.kind, val]):
            rec = {"kind": k, "dimension": dim, "value": v}
            for s in STEPS:
                rec[s] = len(g) if s == "candidate" else int(g[s].sum())
            rec["reliable_with_appearances"] = int((g.reliable & (g.minutes > 0)).sum())
            rec["reliable_true_zero"] = int(g.true_zero.sum())
            for r in REASONS:
                rec[f"excluded_{r}"] = int((g.exclusion_reason == r).sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def coverage_table(cov: pd.DataFrame, E: pd.DataFrame, covered: list[str]) -> pd.DataFrame:
    c = cov[cov.competition_id.isin(covered)].copy()
    arr = E[E.arrival_season.notna()].assign(season=lambda x: x.arrival_season.astype(int))
    key = ["receiving_league", "season", "kind"]
    n_all = arr.groupby(key).size().unstack(fill_value=0).add_suffix("_arrivals")
    n_rel = arr[arr.reliable].groupby(key).size().unstack(fill_value=0).add_suffix("_reliable")
    n_pri = arr[arr.primary_sample].groupby(key).size().unstack(fill_value=0).add_suffix("_primary")
    for t in (n_all, n_rel, n_pri):
        c = c.merge(t, how="left", left_on=["competition_id", "season"], right_index=True)
    cols = [x for x in c.columns if x.endswith(("_arrivals", "_reliable", "_primary"))]
    c[cols] = c[cols].fillna(0).astype(int)
    modal = c.groupby("competition_id").games.agg(lambda s: s.mode().iloc[0])
    c["games_vs_league_mode"] = c.games - c.competition_id.map(modal)
    return c


def design(E: pd.DataFrame) -> dict:
    """Sequential sample-design restrictions applied to the reliable episodes, by kind."""
    out = {}
    for k, g in E[E.reliable].groupby("kind"):
        steps = [("reliable", g)]
        g1 = g[g.arrived_full_season]
        steps.append((f"arrived with at least {FULL_SEASON_SHARE:.0%} of B's league games left", g1))
        g2 = g1[~g1.prior_spell_at_b]
        steps.append(("first spell at B (no earlier move into B)", g2))
        if k == "loan":
            g3 = g2[g2.loan_ending == "return to lender"]
            steps.append(("loan ends (or is scheduled to end) with a return to the lender", g3))
        out[k] = [{"step": s, "n": len(x)} for s, x in steps]
    return out


def timing_sensitivity(E: pd.DataFrame) -> list[dict]:
    R = E[E.reliable].copy()
    R["remain"] = R.games_remaining_at_arrival / R.season_games
    rows = []
    for th in (0.5, 0.6, FULL_SEASON_SHARE, 0.75, 0.9):
        rows.append({"threshold": th, **{k: int(((R.kind == k) & (R.remain >= th)).sum()) for k in ("loan", "permanent")}})
    hist = np.histogram(R.remain, bins=[0, .2, .4, .5, .6, .65, .7, .75, .8, .9, .95, 1.0001])
    return rows, [{"lo": float(lo), "hi": float(min(hi, 1)), **{k: int(np.histogram(R.remain[R.kind == k], bins=hist[1])[0][i])
                                                                  for k in ("loan", "permanent")}}
                  for i, (lo, hi) in enumerate(zip(hist[1][:-1], hist[1][1:]))]


def market_value_coverage(E: pd.DataFrame, con) -> dict:
    P = E[E.primary_sample].copy()
    pv = con.sql("select player_id, date, market_value_in_eur, current_club_id from player_valuations").df()
    pv["date"] = pd.to_datetime(pv.date)
    out = {"primary_n": len(P), "stage1c_market_value": int(P.market_value_eur.notna().sum())}
    pv_p = pv.groupby("player_id").date.apply(lambda s: np.sort(s.values))
    has_player = []
    for p, t in zip(P.player_id, P.move_date):
        ds = pv_p.get(p)
        has_player.append(ds is not None and bool(((ds <= np.datetime64(t)) & (ds >= np.datetime64(t - pd.Timedelta(days=365)))).any()))
    out["player_valuation_within_365d_before"] = int(sum(has_player))
    club = pv.groupby("current_club_id")
    sq = []
    for b, t in zip(P.b_club_id, P.move_date):
        g = club.get_group(b) if b in club.groups else None
        if g is None:
            sq.append(0)
            continue
        w = g[(g.date <= t) & (g.date >= t - pd.Timedelta(days=365))]
        sq.append(w.player_id.nunique())
    sq = pd.Series(sq)
    out["squad_valued_players_median"] = float(sq.median())
    out["squad_with_at_least_11_valued_players"] = int((sq >= 11).sum())
    return out


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(u=None, con=None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    u = u or loan_scope.universe()
    own = con is None
    con = con or duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        mt = match_tables(con)
        cov, covered = league_coverage(mt["sides"])
        E = candidate_episodes(u)
        E = arrival_season(E, mt["sides"], covered)
        E = measure(E, mt["sides"], mt["app"])
        E["appearances_any"] = E.appearances
        E = classify(E, mt["players"])
        E["primary_sample"] = (E.reliable & E.arrived_full_season & ~E.prior_spell_at_b
                               & ((E.kind == "permanent") | (E.loan_ending == "return to lender")))
        N: dict = {}
        N["candidates"] = E.kind.value_counts().to_dict()
        N["loans_in_universe"] = len(u.loans)
        N["unknown_type_episodes"] = int((u.episodes.episode_type == "unknown_type").sum())
        assert N["candidates"]["loan"] == len(u.loans)
        assert E.event_id.is_unique
        assert not (E.minutes.notna() & ~E.reliable).any()
        assert E.true_zero.sum() == (E.reliable & (E.minutes == 0)).sum()
        lg = mt["sides"][mt["sides"].competition_type == "domestic_league"]
        cov14 = cov[cov.competition_id.isin(covered)]
        N["covered_leagues"] = covered
        N["covered_seasons"] = [int(cov14.season.min()), int(cov14.season.max())]
        N["leagues_with_some_rows_not_covered"] = sorted(set(cov[cov.club_games_with_rows > 0].competition_id) - set(covered))
        N["leagues_without_rows"] = sorted(set(lg.competition_id) - set(cov[cov.club_games_with_rows > 0].competition_id))
        N["club_games"] = int(cov14.club_games.sum())
        N["club_games_complete"] = int(cov14.club_games_complete.sum())
        N["club_games_zero_rows"] = int((cov14.club_games - cov14.club_games_with_rows).sum())
        N["worst_league_seasons"] = cov14.sort_values("share_complete").head(6)[
            ["competition_id", "season", "club_games", "club_games_complete", "share_complete"]].to_dict("records")
        N["season_start_months"] = sorted({int(m) for m in cov14.first_game.dt.month})
        odd = cov14[~cov14.last_game.dt.month.isin([4, 5, 6])]
        N["season_end_exceptions"] = [{"league": r.competition_id, "season": int(r.season),
                                       "last_game": str(r.last_game.date())} for r in odd.itertuples()]
        N["short_seasons"] = cov14[cov14.games < cov14.groupby("competition_id").games.transform(lambda s: s.mode().iloc[0])][
            ["competition_id", "season", "games"]].to_dict("records")
        N["checks"] = {k: int(v) for k, v in mt["checks"].items()}
        N["lineups"] = mt["lineups"].to_dict("records")
        N["players_in_universe"] = len(mt["players"])
        N["episode_players_not_in_universe"] = int((~E.player_id.isin(mt["players"])).sum())
        N["waterfall"] = waterfall(E)
        z = E[E.exclusion_reason == "no_league_games_in_window"]
        N["no_games_departed"] = int(z.departed_in_window.sum())
        N["no_games_median_days_to_departure"] = float((z.departure_date - z.move_date).dt.days.median())
        N["design"] = design(E)
        N["timing_thresholds"], N["timing_hist"] = timing_sensitivity(E)
        R = E[E.reliable]
        N["reliable_departed_in_window"] = R.groupby("kind").departed_in_window.sum().astype(int).to_dict()
        N["reliable_mid_season"] = R.groupby("kind").arrived_full_season.apply(lambda s: int((~s).sum())).to_dict()
        N["reliable_prior_spell"] = R.groupby("kind").prior_spell_at_b.sum().astype(int).to_dict()
        N["reliable_conversions"] = int((R.kind.eq("permanent") & R.preceded_by_loan_to_b).sum())
        N["reliable_loan_endings"] = R[R.kind == "loan"].loan_ending.value_counts().to_dict()
        N["reliable_nonleague_missing"] = R.groupby("kind").apply(
            lambda g: int((g.nonleague_games_without_rows > 0).sum())).to_dict()
        N["reliable_covid_season"] = R.groupby("kind").apply(lambda g: int((g.arrival_season == 2019).sum())).to_dict()
        P = E[E.primary_sample]
        N["primary"] = P.kind.value_counts().to_dict()
        N["primary_by_league"] = P.groupby(["receiving_league", "kind"]).size().unstack(fill_value=0).to_dict("index")
        N["primary_by_season"] = P.groupby(["arrival_season", "kind"]).size().unstack(fill_value=0).to_dict("index")
        N["primary_true_zero"] = P.groupby("kind").true_zero.sum().astype(int).to_dict()
        N["primary_departed"] = P.groupby("kind").departed_in_window.sum().astype(int).to_dict()
        N["primary_window_games_median"] = P.groupby("kind").available_games.median().to_dict()
        N["primary_next_season_at_b"] = next_season_feasibility(P, lg)
        N["market_value"] = market_value_coverage(E, con)
        coverage = coverage_table(cov, E, covered)
    finally:
        if own:
            con.close()
    return N, E, coverage


def next_season_feasibility(P: pd.DataFrame, lg: pd.DataFrame) -> dict:
    """Can 'minutes in the following season at B' be measured? Needs B covered in s+1 and the player still at B."""
    ends = lg.groupby(["club_id", "season"]).date.max()
    out = {}
    for k, g in P.groupby("kind"):
        ok = [((b, s + 1) in ends.index) and (pd.isna(d) or d > ends[(b, s + 1)])
              for b, s, d in zip(g.b_club_id, g.arrival_season.astype(int), g.departure_date)]
        out[k] = int(sum(ok))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render(N: dict) -> str:
    W: list[str] = []
    w = W.append
    wf = N["waterfall"]
    lo, pe = wf["loan"], wf["permanent"]
    pr = N["primary"]
    cs = lambda k, n: f"{f0(k)} ({share(k / n) if n else '–'})"
    seasons = f"{N['covered_seasons'][0]}/{str(N['covered_seasons'][0] + 1)[2:]}–{N['covered_seasons'][1]}/{str(N['covered_seasons'][1] + 1)[2:]}"

    w("# Can we join transfers to playing time? Feasibility and the clean sample")
    w("")
    w("*Generated by `python -m src.analysis.playing_time_join` from the frozen Stage 1C table and the snapshot's "
      "match tables. Coverage only: no playing-time comparison between loans and permanent moves is made here.*")
    w("")
    w("## Answers")
    w("")
    w(f"1. **Can we do the playing-time comparison?** Yes, on a clean subset. League appearances are complete and "
      f"consistent for {len(N['covered_leagues'])} leagues over {seasons}, and a transfer can be matched to them "
      "exactly by player id, receiving-club id and date.")
    w(f"2. **Which leagues and seasons?** {', '.join(N['covered_leagues'])}, seasons {seasons}: the leagues with "
      f"appearance rows in every season. {len(N['leagues_without_rows'])} other leagues in the snapshot "
      f"({', '.join(N['leagues_without_rows'])}) have games but no appearance rows.")
    w(f"3. **Clean permanent-transfer observations:** {f0(pe['reliable'])} with a reliable playing-time outcome, of "
      f"{f0(pe['candidate'])} candidates; **{f0(pr.get('permanent', 0))}** in the recommended primary sample.")
    w(f"4. **Clean loan observations:** {f0(lo['reliable'])} reliable, of {f0(lo['candidate'])} candidates; "
      f"**{f0(pr.get('loan', 0))}** in the recommended primary sample.")
    w(f"5. **Join rate:** {share(lo['reliable'] / lo['candidate'])} of loan and {share(pe['reliable'] / pe['candidate'])} "
      "of permanent candidates. Among episodes whose receiving club plays in a covered league that season, "
      f"{share(lo['reliable'] / lo['covered_league_season'])} and {share(pe['reliable'] / pe['covered_league_season'])}. "
      "Almost all losses come from the receiving club being outside the covered leagues, not from failed matches.")
    w("6. **What \"playing time\" should mean:** the share of the receiving club's league minutes the player played, "
      "from the move date to the end of that league season (or to his earlier departure): league minutes ÷ (90 × "
      "B's league games in that window). See the outcome section.")
    w("7. **Exclusions:** see the waterfall. Each excluded episode has exactly one reason, the first condition it "
      "fails.")
    w("8. **Remaining limitations:** listed at the end.")
    w(f"9. **Sample for the next analysis:** reliable episodes where the player arrived with at least "
      f"{FULL_SEASON_SHARE:.0%} of B's league season left, in his first spell at B, and (for loans) with a return to the "
      f"lender: {f0(pr.get('loan', 0))} loans and {f0(pr.get('permanent', 0))} permanent moves "
      f"(`playing_time_join_episodes.csv`, column `primary_sample`).")
    w("")

    w("## The playing-time data, verified")
    w("")
    ck = N["checks"]
    lu = {r["type"]: r for r in N["lineups"]}
    w(f"- **Identifiers.** `appearances` has {f0(ck['n'])} rows. `appearance_id` is unique and (player, game) is "
      f"unique ({f0(ck['player_games'])} pairs). Each row's club is one of the two clubs in the game "
      f"({ck['club_not_in_game']} exceptions), and its date and competition match the game ({ck['date_mismatch']} "
      f"and {ck['competition_mismatch']} mismatches). Player and club ids are the same Transfermarkt ids as in Stage 1C; "
      f"{N['episode_players_not_in_universe']} candidate episodes have a player missing from the `players` table.")
    w(f"- **Minutes** are present on every appearance row ({ck['null_minutes']} missing).")
    w(f"- **Completeness.** Of the {f0(N['club_games'])} league club-games in the {len(N['covered_leagues'])} leagues, "
      f"{cs(N['club_games_complete'], N['club_games'])} have at least {COMPLETE_ROWS} appearance rows (a full starting "
      f"XI); {f0(N['club_games_zero_rows'])} have none. The gaps cluster in a few league-seasons: "
      + ", ".join(f"{r['competition_id']} {r['season']}/{str(r['season'] + 1)[2:]} "
                  f"({share(r['share_complete'])} complete)" for r in N["worst_league_seasons"][:4]) + ".")
    w(f"- **Unused substitutes are absent from `appearances`.** In league club-games with a complete record, "
      f"{cs(lu['substitutes']['not_in_appearances'], lu['substitutes']['n'])} of substitute lineup entries have no "
      f"appearance row (the unused substitutes), against "
      f"{cs(lu['starting_lineup']['not_in_appearances'], lu['starting_lineup']['n'])} of starters. So `appearances` "
      "lists the players who got on the pitch, and `game_lineups` (which has no minutes) is the only record of the "
      "bench.")
    w("- **What absence means.** If B's appearance record for a game is complete, a player registered at B with no row "
      "did not play in that game: a true zero. If B's game is not covered, or its record is incomplete, absence says "
      "nothing, and the episode gets no value. Cup and UEFA games are only partly covered, so only league games are used.")
    s19 = [r for r in N["short_seasons"] if r["season"] == 2019]
    w("- **Season lengths vary within a league**, through format changes, play-off rounds missing from some seasons, "
      f"and the 2019/20 interruption. In 2019/20, {len(s19)} leagues have fewer league games than their most common "
      "season length: " + ", ".join(f"{r['competition_id']} ({r['games']})" for r in s19) + ". A share of "
      "available minutes is unaffected, because its denominator counts B's own games.")
    mo = lambda ms: "/".join(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1] for m in ms)
    ex = N["season_end_exceptions"]
    covid = [x for x in ex if x["season"] == 2019]
    march = [x for x in ex if x["season"] != 2019 and x["last_game"][5:7] == "03"]
    other = [x for x in ex if x not in covid and x not in march]
    by_l: dict[str, int] = {}
    for x in march:
        by_l[x["league"]] = by_l.get(x["league"], 0) + 1
    w(f"- **One calendar.** Every season of the {len(N['covered_leagues'])} leagues starts in "
      f"{mo(N['season_start_months'])}, so a season label identifies the same autumn-to-spring period in each league. "
      f"Most end in April–June. The exceptions are of three kinds:")
    w(f"  - In {len(march)} league-seasons the snapshot's league games stop in March ("
      + ", ".join(f"{k} {v}" for k, v in sorted(by_l.items(), key=lambda kv: -kv[1]))
      + " seasons). Their game counts match a regular season alone, so the second-phase (play-off) rounds appear "
      "to be missing, and the window ends with the regular season (inference).")
    w(f"  - In 2019/20, {len(covid)} leagues end early or late (between "
      f"{min(x['last_game'] for x in covid)} and {max(x['last_game'] for x in covid)}).")
    if other:
        w("  - " + "; ".join(f"{x['league']} {x['season']}/{str(x['season'] + 1)[2:]} stops on {x['last_game']}"
                             for x in other) + ".")
    w("")

    w("## Join waterfall")
    w("")
    rows = []
    for s in STEPS:
        rows.append([STEP_LABELS[s], cs(lo[s], lo["candidate"]), cs(pe[s], pe["candidate"])])
    rows.append(["… of which the player has at least one league appearance for B",
                 cs(lo["reliable_with_appearances"], lo["candidate"]), cs(pe["reliable_with_appearances"], pe["candidate"])])
    rows.append(["… of which a true zero (B fully observed, the player never played)",
                 cs(lo["reliable_true_zero"], lo["candidate"]), cs(pe["reliable_true_zero"], pe["candidate"])])
    W += md_table(["Step", "Loans", "Permanent transfers"], rows)
    w("")
    w(f"Before the reliability check, {f0(lo['matched_with_appearance_record'])} loans and "
      f"{f0(pe['matched_with_appearance_record'])} permanent moves have at least one league appearance record at B. "
      "A true zero has no appearance record but is a valid outcome, which is why the record count comes after the "
      "reliability check.")
    w("")
    W += md_table(["Exclusion reason (first failed condition)", "Loans", "Permanent transfers"],
                  [[REASONS[r], f0(lo["excluded"][r]), f0(pe["excluded"][r])] for r in REASONS])
    w("")
    w(f"All {f0(N['no_games_departed'])} episodes without a league game in the window left B before B's next "
      f"league game, a median of {N['no_games_median_days_to_departure']:.0f} day(s) after arriving.")
    w("")
    w("Counts by transfer season, arrival season, receiving league, receiving country and transfer year are in "
      "`playing_time_join_audit.csv`; per league-season coverage is in `playing_time_join_coverage_by_league_season.csv`.")
    w("")

    w("## The outcome: what \"playing time\" should mean")
    w("")
    nsa = N["primary_next_season_at_b"]
    nl = N["reliable_nonleague_missing"]
    W += md_table(["Candidate", "Constructible?", "Coverage", "Advantages", "Problems"], [
        ["Total minutes at B in all competitions", "No",
         f"{f0(nl.get('loan', 0))} reliable loans and {f0(nl.get('permanent', 0))} permanents have a cup or UEFA game of B "
         "with no appearance rows in the window", "Uses every match", "Cup and UEFA games only partly covered; the "
         "total would mix real zeros with missing games"],
        ["Total league minutes at B, arrival to season end (or departure)", "Yes", "Every reliable episode",
         "Simple and exact", "Depends on window length: a January arrival has half a season, a 34-game league "
         "fewer minutes than a 38-game one"],
        ["Minutes in the season after the transfer (s + 1)", "Only for players still at B",
         f"In the primary sample: {f0(nsa.get('loan', 0))} loans and {f0(nsa.get('permanent', 0))} permanents",
         "A full season for permanent moves", "A season-long loan ends before s + 1, so loans and permanents are not "
         "comparable"],
        ["**Share of available league minutes at B, arrival to season end (or departure)**", "**Yes**",
         "**Every reliable episode**", "Comparable across window lengths, league sizes and the 2019/20 interruption",
         "Truncating at departure keeps players who left early, but their shorter window is itself informative"],
        ["League minutes per available match", "Yes", "Every reliable episode",
         "Easy to read", "Equals the share × 90, so it adds no information"],
    ], right=set())
    w("")
    w("**Recommended primary measure:** the share of B's available league minutes, in B's league season of arrival.")
    w("")
    w("- **Window.** It starts on the Transfermarkt move date. It ends at B's last league game of that season, or the "
      "day before the player's next recorded move if that comes first (a sale, a recall, a sub-loan). Only B's league "
      "games in the window count, in both the numerator and the denominator.")
    w("- **Permanent A → B:** the season of arrival is the first league season of B ending after the move. For a "
      "summer move this is the whole next season.")
    w("- **Loan A → B:** the same window. For a season-long loan it coincides with the loan spell, because the return "
      "is dated after the season's last game. A loan that ends early is cut at the return date. A multi-season loan "
      "is measured in its first season only, as a permanent move is.")
    w("- This makes the two windows economically comparable: both measure the player's first season at the new club.")
    w("")
    tt = N["timing_thresholds"]
    th = {round(r["threshold"], 3): r for r in tt}
    w("**Complications, and how the sample handles them:**")
    w("")
    w(f"- **Mid-season arrivals.** Arrival timing is bimodal: most moves come before or early in B's season, the "
      "others in the winter. Few come with 60–75% of the season left, so the split hardly depends on the exact "
      f"threshold. Reliable episodes arriving with at least 60% / two-thirds / 75% of B's league games left: loans "
      f"{f0(th[0.6]['loan'])} / {f0(th[round(FULL_SEASON_SHARE, 3)]['loan'])} / {f0(th[0.75]['loan'])}, permanents "
      f"{f0(th[0.6]['permanent'])} / {f0(th[round(FULL_SEASON_SHARE, 3)]['permanent'])} / {f0(th[0.75]['permanent'])}. "
      f"The primary sample uses two-thirds; {f0(N['reliable_mid_season'].get('loan', 0))} loans and "
      f"{f0(N['reliable_mid_season'].get('permanent', 0))} permanents arriving later form a separate mid-season sample.")
    w(f"- **Onward moves during the window.** {f0(N['reliable_departed_in_window'].get('loan', 0))} reliable loans and "
      f"{f0(N['reliable_departed_in_window'].get('permanent', 0))} permanents leave B before the season ends. The window "
      "is cut at departure and the episode is flagged (`departed_in_window`). An early departure may itself reflect "
      "little playing time, so dropping these cases would bias the sample.")
    w(f"- **Repeat spells.** {f0(N['reliable_prior_spell'].get('loan', 0))} reliable loans and "
      f"{f0(N['reliable_prior_spell'].get('permanent', 0))} permanents are not the player's first move into B: re-loans, "
      f"returns, and {f0(N['reliable_conversions'])} permanent moves that convert a loan at B. They are excluded from the "
      "primary sample, because the player has already played at B.")
    w(f"- **Loans without a clean ending.** Reliable loans by ending: "
      + ", ".join(f"{k} {f0(v)}" for k, v in N["reliable_loan_endings"].items())
      + ". The primary sample keeps loans that end, or are scheduled to end, with a return to the lender.")
    w(f"- **Different calendars.** All {len(N['covered_leagues'])} covered leagues play autumn-to-spring seasons (see "
      "above), and the window is defined by B's own league games, so no further calendar assumption is needed. The "
      "calendar-year leagues have no appearance data, so they cannot enter the sample.")
    w(f"- **2019/20.** {f0(N['reliable_covid_season'].get('loan', 0))} reliable loans and "
      f"{f0(N['reliable_covid_season'].get('permanent', 0))} permanents arrive in 2019/20, when several leagues stopped "
      "early. They are kept and flagged; a sensitivity without them is easy.")
    w("")

    w("## The recommended sample, step by step")
    w("")
    for k, lab in (("loan", "Loans"), ("permanent", "Permanent transfers")):
        w(f"**{lab}:** " + " → ".join(f"{r['step']} {f0(r['n'])}" for r in N["design"][k]))
        w("")
    pl = N["primary_by_league"]
    W += md_table(["Receiving league", "Loans", "Permanent transfers"],
                  [[lg, f0(v.get("loan", 0)), f0(v.get("permanent", 0))] for lg, v in sorted(pl.items())]
                  + [["**Total**", f"**{f0(pr.get('loan', 0))}**", f"**{f0(pr.get('permanent', 0))}**"]])
    w("")
    ps = N["primary_by_season"]
    W += md_table(["Arrival season", "Loans", "Permanent transfers"],
                  [[f"{s}/{str(int(s) + 1)[2:]}", f0(v.get("loan", 0)), f0(v.get("permanent", 0))]
                   for s, v in sorted(ps.items(), key=lambda kv: int(kv[0]))])
    w("")
    w(f"In the primary sample, {f0(N['primary_true_zero'].get('loan', 0))} loans and "
      f"{f0(N['primary_true_zero'].get('permanent', 0))} permanents are true zeros (B fully observed, no minutes).")
    w("")
    mv = N["market_value"]
    w(f"**Market values (coverage only; no model built).** In the primary sample ({f0(mv['primary_n'])} episodes):")
    w(f"- Stage 1C market value at the transfer: {cs(mv['stage1c_market_value'], mv['primary_n'])}.")
    w(f"- A `player_valuations` entry in the year before the move: {cs(mv['player_valuation_within_365d_before'], mv['primary_n'])}.")
    w(f"- B's squad value could be built from `player_valuations` (club at valuation date). The median episode has "
      f"{mv['squad_valued_players_median']:.0f} valued B players in the year before the move, and "
      f"{cs(mv['squad_with_at_least_11_valued_players'], mv['primary_n'])} have at least 11.")
    w("")
    w("## Remaining limitations")
    w("")
    w(f"- **Leagues.** Only the {len(N['covered_leagues'])} long-covered European first divisions. Loans to second "
      "divisions, which are a large share of all loans, and moves to the leagues added in 2024 cannot be measured.")
    w("- **Selection.** The players are the 2023/24–2025/26 cohort, so earlier seasons hold only players who were still "
      "active later, and the history-capture rule drops the most recent seasons of players captured early.")
    w("- **Scope of the measure.** League minutes only: cup, European and national-team minutes are left out, and "
      "unused-substitute status cannot be told apart from absence from the squad.")
    w("- **Recorded dates.** Windows use Transfermarkt's recorded dates. A move dated on a matchday may count one "
      "game the player could not have played.")
    w("")
    return "\n".join(W) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


EPISODE_COLS = ["event_id", "kind", "episode_type", "player_id", "player_name", "transfer_season", "move_date",
                "a_club_id", "a_club", "b_club_id", "b_club", "from_country", "to_country", "scheduled",
                "arrival_season", "receiving_league", "season_games", "games_remaining_at_arrival",
                "arrived_full_season", "departure_date", "departed_in_window", "window_start", "window_end",
                "available_games", "complete_games", "history_file", "history_captured_through",
                "prior_spell_at_b", "preceded_by_loan_to_b", "loan_ending", "loan_return_date", "loan_return_scheduled",
                "exclusion_reason", "realised", "covered_league_season", "matched", "reliable",
                "appearances", "minutes", "available_minutes", "minutes_share", "true_zero",
                "nonleague_games_in_window", "nonleague_games_without_rows", "market_value_eur", "primary_sample"]


def main() -> dict:
    N, E, coverage = compute()
    out = E[EPISODE_COLS].copy()
    for c in ("move_date", "departure_date", "window_start", "window_end", "history_captured_through", "loan_return_date"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(EPISODES_CSV, index=False)
    audit_table(E).to_csv(AUDIT_CSV, index=False)
    coverage.to_csv(COVERAGE_CSV, index=False)
    NUMBERS_JSON.write_text(json.dumps(_jsonable(N), indent=1, default=str))
    REPORT_MD.write_text(render(N))
    wf = N["waterfall"]
    print(f"wrote {REPORT_MD.name}, {AUDIT_CSV.name}, {COVERAGE_CSV.name}, {EPISODES_CSV.name}")
    for k in ("loan", "permanent"):
        print(f"{k:10} candidates {wf[k]['candidate']:,}  covered {wf[k]['covered_league_season']:,}  "
              f"reliable {wf[k]['reliable']:,}  primary {N['primary'].get(k, 0):,}")
    return N


if __name__ == "__main__":
    main()
