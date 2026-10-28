"""Do the snapshot's appearance rows miss players who played? A read-only audit against line-ups and substitutions.

    python -m src.analysis.appearance_audit

`appearances` is the only source of minutes. Two other tables of the same DuckDB snapshot record who was on the pitch:
`game_lineups` (the starting XI) and `game_events` (type 'Substitutions', `player_in_id` = the player coming on). A
player-game that either table puts on the pitch for a club, with no `appearances` row for that player and club, is a
contradiction. Neither table gives minutes, so they are used only as checks, never to rebuild minutes.

Two parts:
  1. cell audit: every covered-league club-game, with contradiction rates by season, league, league × season and
     club × season (`appearance_audit_cells.csv`, `appearance_audit.md`);
  2. episode check: for each reliable playing-time episode, how many of its window games contradict the appearance
     rows (used by `data_lineage` for the master columns `pt_lineup_*` and `pt_reliable_v2`).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .daniel_scope_final import f0, md_table
from .loan_episodes import ROOT
from .playing_time_join import COMPLETE_ROWS

OUT = ROOT / "data" / "outputs" / "rebuild"
CELLS_CSV = OUT / "appearance_audit_cells.csv"
CLUBS_CSV = OUT / "appearance_audit_club_seasons.csv"
REPORT_MD = OUT / "appearance_audit.md"
NUMBERS_JSON = OUT / "appearance_audit_numbers.json"
WRITES = (CELLS_CSV, CLUBS_CSV, REPORT_MD, NUMBERS_JSON)
LINEAGE_JSON = OUT / "data_lineage_numbers.json"

ON_PITCH = """
    select l.game_id, l.player_id, l.club_id, 'start' as how from game_lineups l where l.type = 'starting_lineup'
    union
    select v.game_id, v.player_in_id, v.club_id, 'sub' from game_events v
     where v.type = 'Substitutions' and v.player_in_id is not null"""


# ---------------------------------------------------------------------------
# 1. Cell audit
# ---------------------------------------------------------------------------

def cell_audit(con, leagues: list[str]) -> pd.DataFrame:
    """One row per club-game of the covered leagues: players on the pitch per line-ups/substitutions, and how many of
    them have no appearance row for that club."""
    lg = ", ".join(f"'{x}'" for x in leagues)
    return con.execute(f"""
        with g as (
          select game_id, competition_id league, cast(season as integer) season, home_club_id club_id from games
           where competition_type = 'domestic_league' and competition_id in ({lg})
          union all
          select game_id, competition_id, cast(season as integer), away_club_id from games
           where competition_type = 'domestic_league' and competition_id in ({lg})),
        lu as (select distinct game_id, club_id from game_lineups),
        op as (select game_id, player_id, club_id, max(case when how = 'start' then 1 else 0 end) started
                 from ({ON_PITCH}) group by 1, 2, 3),
        ap as (select game_id, player_club_id club_id, player_id from appearances)
        select g.game_id, g.league, g.season, g.club_id,
               (lu.game_id is not null) as has_lineup,
               (select count(*) from ap where ap.game_id = g.game_id and ap.club_id = g.club_id) as app_rows,
               count(op.player_id) as on_pitch,
               coalesce(sum(case when op.player_id is not null and not exists (
                   select 1 from ap where ap.game_id = op.game_id and ap.player_id = op.player_id
                     and ap.club_id = op.club_id) then 1 else 0 end), 0) as on_pitch_no_row
        from g left join lu on lu.game_id = g.game_id and lu.club_id = g.club_id
               left join op on op.game_id = g.game_id and op.club_id = g.club_id
        group by 1, 2, 3, 4, 5""").fetchdf()


def player_season_gaps(con, leagues: list[str]) -> pd.DataFrame:
    """Player × club × season: games on the pitch per line-ups/substitutions vs games with an appearance row."""
    lg = ", ".join(f"'{x}'" for x in leagues)
    return con.execute(f"""
        with gm as (select game_id, competition_id league, cast(season as integer) season from games
                     where competition_type = 'domestic_league' and competition_id in ({lg})),
        op as (select distinct o.game_id, o.player_id, o.club_id from ({ON_PITCH}) o join gm using (game_id)),
        ap as (select distinct a.game_id, a.player_id, a.player_club_id club_id from appearances a join gm using (game_id)),
        cg as (select game_id, club_id, count(*) n from ap group by 1, 2)
        select gm.league, gm.season, op.club_id, op.player_id, count(*) as games_on_pitch,
               sum(case when ap.player_id is null then 1 else 0 end) as games_no_row,
               sum(case when coalesce(cg.n, 0) >= {COMPLETE_ROWS} then 1 else 0 end) as games_in_complete,
               sum(case when ap.player_id is null and coalesce(cg.n, 0) >= {COMPLETE_ROWS} then 1 else 0 end)
                   as games_no_row_in_complete
        from op join gm using (game_id)
        left join ap on ap.game_id = op.game_id and ap.player_id = op.player_id and ap.club_id = op.club_id
        left join cg on cg.game_id = op.game_id and cg.club_id = op.club_id
        group by 1, 2, 3, 4""").fetchdf()


def rates(G: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """All checked club-games, and separately those that pass the ≥ 11-row completeness test: a contradiction there
    is the one the playing-time rule cannot see (in an incomplete club-game the episode is already unreliable)."""
    C = G[G.has_lineup].assign(complete=lambda d: d.app_rows >= COMPLETE_ROWS)
    C = C.assign(on_pitch_c=C.on_pitch.where(C.complete, 0), no_row_c=C.on_pitch_no_row.where(C.complete, 0))
    out = C.groupby(by).agg(club_games=("game_id", "size"), on_pitch=("on_pitch", "sum"),
                            on_pitch_no_row=("on_pitch_no_row", "sum"),
                            club_games_with_gap=("on_pitch_no_row", lambda s: int((s > 0).sum())),
                            complete_club_games=("complete", "sum"), on_pitch_complete=("on_pitch_c", "sum"),
                            no_row_in_complete=("no_row_c", "sum"))
    out["contradiction_rate"] = out.on_pitch_no_row / out.on_pitch
    out["undetected_rate"] = out.no_row_in_complete / out.on_pitch_complete
    allg = G.groupby(by).size().rename("club_games_all")
    out = out.join(allg, how="right").fillna({"club_games": 0, "on_pitch": 0, "on_pitch_no_row": 0,
                                              "club_games_with_gap": 0, "complete_club_games": 0,
                                              "on_pitch_complete": 0, "no_row_in_complete": 0})
    out["lineup_coverage"] = out.club_games / out.club_games_all
    return out.reset_index()


# ---------------------------------------------------------------------------
# 2. Episode check (used for the master)
# ---------------------------------------------------------------------------

def episode_check(con, E: pd.DataFrame) -> pd.DataFrame:
    """For episodes with an observation window (event_id, player_id, to_club_id, window_start, window_end,
    receiving_league): window league games of club B, those with B line-ups, and games where the player started or came
    on for B with no appearance row."""
    E = E[["event_id", "player_id", "to_club_id", "window_start", "window_end", "receiving_league"]].copy()
    con.register("ep_audit", E)
    try:
        D = con.execute(f"""
            with wg as (
              select e.event_id, e.player_id, e.to_club_id b, g.game_id from ep_audit e join games g
                on g.competition_id = e.receiving_league and g.competition_type = 'domestic_league'
               and g.date between cast(e.window_start as date) and cast(e.window_end as date)
               and (g.home_club_id = e.to_club_id or g.away_club_id = e.to_club_id)),
            x as (
              select wg.event_id, wg.game_id,
                exists (select 1 from game_lineups l where l.game_id = wg.game_id and l.club_id = wg.b) as has_lineup,
                exists (select 1 from game_lineups l where l.game_id = wg.game_id and l.player_id = wg.player_id
                        and l.club_id = wg.b and l.type = 'starting_lineup') as started,
                exists (select 1 from game_events v where v.game_id = wg.game_id and v.player_in_id = wg.player_id
                        and v.club_id = wg.b and v.type = 'Substitutions') as subbed_on,
                exists (select 1 from appearances a where a.game_id = wg.game_id and a.player_id = wg.player_id
                        and a.player_club_id = wg.b) as has_row
              from wg)
            select event_id, count(*) as window_games,
              sum(case when has_lineup then 1 else 0 end) as window_games_with_lineups,
              sum(case when started and not has_row then 1 else 0 end) as started_no_row,
              sum(case when subbed_on and not started and not has_row then 1 else 0 end) as subbed_on_no_row
            from x group by 1""").fetchdf()
    finally:
        con.unregister("ep_audit")
    Q = E[["event_id"]].merge(D, on="event_id", how="left")
    for c in ("window_games", "window_games_with_lineups", "started_no_row", "subbed_on_no_row"):
        Q[c] = Q[c].fillna(0).astype(int)
    Q["games_played_without_row"] = Q.started_no_row + Q.subbed_on_no_row
    return Q.set_index("event_id")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def compute(con=None) -> tuple[dict, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    L = json.loads(LINEAGE_JSON.read_text())
    leagues = L["playing_time"]["covered_leagues"]
    own = con is None
    con = con or duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        G = cell_audit(con, leagues)
        P = player_season_gaps(con, leagues)
    finally:
        if own:
            con.close()
    tb = {"season": rates(G, ["season"]), "league": rates(G, ["league"]),
          "league_season": rates(G, ["league", "season"]), "club_season": rates(G, ["league", "season", "club_id"])}
    C = G[G.has_lineup]
    cm = C.app_rows >= COMPLETE_ROWS
    P["kind"] = np.select([P.games_no_row.eq(0), P.games_no_row.eq(P.games_on_pitch)], ["none", "whole"], "partial")
    Pc = P[P.games_no_row_in_complete > 0]
    ls = tb["league_season"]
    und = int(C.loc[cm, "on_pitch_no_row"].sum())
    top = ls.sort_values("no_row_in_complete", ascending=False)
    N = {"leagues": leagues, "complete_rows": COMPLETE_ROWS, "club_games": len(G),
         "club_games_with_lineups": int(G.has_lineup.sum()),
         "on_pitch": int(C.on_pitch.sum()), "on_pitch_no_row": int(C.on_pitch_no_row.sum()),
         "rate": float(C.on_pitch_no_row.sum() / C.on_pitch.sum()),
         "in_incomplete_club_games": int(C.loc[~cm, "on_pitch_no_row"].sum()),
         "complete_club_games": int(cm.sum()), "on_pitch_complete": int(C.loc[cm, "on_pitch"].sum()),
         "undetected": und, "undetected_rate": float(und / C.loc[cm, "on_pitch"].sum()),
         "undetected_spells": len(Pc),
         "undetected_spells_whole": int((Pc.games_no_row_in_complete == Pc.games_in_complete).sum()),
         "undetected_spells_partial": int((Pc.games_no_row_in_complete < Pc.games_in_complete).sum()),
         "top_cells": top.head(8)[["league", "season", "no_row_in_complete", "on_pitch_complete", "undetected_rate"]]
         .to_dict("records"),
         "share_in_top_cell": float(top.no_row_in_complete.iloc[0] / und) if und else 0.0,
         "share_in_top3_cells": float(top.no_row_in_complete.head(3).sum() / und) if und else 0.0,
         "share_2021": float(tb["season"].set_index("season").loc[2021, "no_row_in_complete"] / und) if und else 0.0,
         "seasons_without_lineups": [int(x) for x in tb["season"].loc[tb["season"].lineup_coverage.lt(0.5), "season"]]}
    return N, {"G": G, "P": P, **tb}


def render(N: dict, tb: dict) -> str:
    pc = lambda x: f"{100 * x:.3f}%"
    sea = lambda x: f"{int(x)}/{str(int(x) + 1)[2:]}"
    W = ["# Appearance-row audit: line-ups and substitutions vs `appearances`", "",
         "Generated by `python -m src.analysis.appearance_audit` (read-only). A *contradiction* is a player-game that "
         "`game_lineups` (starting XI) or `game_events` (substitution, player coming on) puts on the pitch for a club, "
         "with no `appearances` row for that player and club. Line-ups and substitutions carry no minutes, so they are "
         "used only as checks.", "",
         f"Covered leagues ({len(N['leagues'])}): {', '.join(N['leagues'])}. Club-games: {f0(N['club_games'])}, of "
         f"which {f0(N['club_games_with_lineups'])} have line-ups (season(s) without line-ups, which cannot be checked: "
         f"{', '.join(sea(x) for x in N['seasons_without_lineups']) or 'none'}).", "",
         "## What matters for the playing-time samples", "",
         f"The playing-time rule already rejects any club-game with fewer than {N['complete_rows']} appearance rows. "
         f"Of the {f0(N['on_pitch_no_row'])} contradicted player-games ({pc(N['rate'])} of {f0(N['on_pitch'])} checked), "
         f"{f0(N['in_incomplete_club_games'])} sit in club-games that fail that test, so the episodes involved are "
         f"already unreliable and outside every sample. The rest, **{f0(N['undetected'])} player-games in "
         f"{f0(N['complete_club_games'])} complete club-games ({pc(N['undetected_rate'])} of {f0(N['on_pitch_complete'])} "
         "checked), are invisible to the rule**: one player is missing from an otherwise complete record.", "",
         f"They come from {f0(N['undetected_spells'])} player × club × season spells: "
         f"{f0(N['undetected_spells_whole'])} miss every complete-game row of the spell (a whole spell recorded as no "
         f"appearances, which becomes a false zero) and {f0(N['undetected_spells_partial'])} miss only some games (minutes "
         f"undercounted). Concentration: 2021/22 holds {100 * N['share_2021']:.0f}% of them; the largest league-season "
         f"holds {100 * N['share_in_top_cell']:.0f}%, the largest three {100 * N['share_in_top3_cells']:.0f}%.", ""]
    W += md_table(["League", "Season", "Undetected player-games", "Checked (complete club-games)", "Rate"],
                  [[r["league"], sea(r["season"]), f0(r["no_row_in_complete"]), f0(r["on_pitch_complete"]),
                    pc(r["undetected_rate"])] for r in N["top_cells"]], right={2, 3, 4})
    W += ["", "## By season", ""]
    W += md_table(["Season", "Club-games with line-ups", "Contradicted (all)", "Rate (all)",
                   "Undetected (complete club-games)", "Rate (undetected)"],
                  [[sea(r.season), f0(r.club_games), f0(r.on_pitch_no_row), pc(r.contradiction_rate) if r.on_pitch else "—",
                    f0(r.no_row_in_complete), pc(r.undetected_rate) if r.on_pitch_complete else "—"]
                   for r in tb["season"].itertuples()])
    W += ["", "## By league", ""]
    W += md_table(["League", "Contradicted (all)", "Rate (all)", "Undetected", "Rate (undetected)"],
                  [[r.league, f0(r.on_pitch_no_row), pc(r.contradiction_rate), f0(r.no_row_in_complete),
                    pc(r.undetected_rate)] for r in tb["league"].itertuples()])
    cs = tb["club_season"].sort_values("no_row_in_complete", ascending=False).head(10)
    W += ["", "## Club-seasons with the most undetected gaps", ""]
    W += md_table(["League", "Season", "Club id", "Undetected", "Checked (complete)", "Rate"],
                  [[r.league, sea(r.season), str(int(r.club_id)), f0(r.no_row_in_complete), f0(r.on_pitch_complete),
                    pc(r.undetected_rate)] for r in cs.itertuples()])
    W += ["", "Club-seasons with near-total gaps (e.g. whole 2014/15 squads) fail the completeness test and are already "
          "outside the samples; they appear in the 'all' columns only.", "",
          f"Full tables: `{CELLS_CSV.name}` (season, league and league × season rows), `{CLUBS_CSV.name}` "
          "(club × season).", ""]
    return "\n".join(W) + "\n"


def main() -> dict:
    N, tb = compute()
    cells = pd.concat([tb["season"].assign(level="season"), tb["league"].assign(level="league"),
                       tb["league_season"].assign(level="league_season")], ignore_index=True)
    cells.to_csv(CELLS_CSV, index=False)
    tb["club_season"].to_csv(CLUBS_CSV, index=False)
    REPORT_MD.write_text(render(N, tb))
    NUMBERS_JSON.write_text(json.dumps(N, indent=1, default=float))
    print(f"wrote {REPORT_MD.name}: {N['on_pitch_no_row']:,} of {N['on_pitch']:,} player-games contradicted")
    return N


if __name__ == "__main__":
    main()
