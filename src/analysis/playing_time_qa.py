"""Independent check of the playing-time outcome, for loans and permanent moves alike.

    python -m src.analysis.playing_time_qa

Recomputes, with plain SQL on the DuckDB snapshot and the Stage 1C file (none of the playing-time code is reused):
the departure date, the observation window, club B's domestic-league games in it, the ≥ 11-row completeness test, the
player's minutes for B, and the share. Compares every reliable episode of the master with the recomputation, and
writes game-level rows for a few real examples that can be checked by hand. Read-only apart from its own outputs.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data_lineage as dl
from .club_organisation import rough_stem_tokens as stem_tokens
from .daniel_scope_final import f0, md_table
from .loan_episodes import CANONICAL_CSV
from ..stage1c.policy import YOUTH_OR_RESERVE

OUT = dl.OUT
REPORT_MD = OUT / "playing_time_qa.md"
GAMES_CSV = OUT / "playing_time_qa_example_games.csv"
NUMBERS_JSON = OUT / "playing_time_qa_numbers.json"
WRITES = (REPORT_MD, GAMES_CSV, NUMBERS_JSON)
COMPLETE_ROWS = 11
SKIP_TYPES = ("youth_or_internal", "other")


def departures(M: pd.DataFrame) -> pd.Series:
    """Next substantive Stage 1C row of the player dated strictly after the move (internal and draft rows skipped)."""
    S = pd.read_csv(CANONICAL_CSV, usecols=["player_id", "transfer_date", "is_internal_move", "transfer_type_normalized"])
    S = S[~S.is_internal_move.astype(bool) & ~S.transfer_type_normalized.isin(SKIP_TYPES)]
    S["d"] = pd.to_datetime(S.transfer_date)
    S = S[["player_id", "d"]].drop_duplicates().sort_values(["player_id", "d"])
    E = M[["event_id", "player_id", "transfer_date"]].assign(t=pd.to_datetime(M.transfer_date)).sort_values("t")
    out = pd.merge_asof(E, S.rename(columns={"d": "dep"}).assign(dep_key=lambda x: x.dep).sort_values("dep_key"),
                        left_on="t", right_on="dep_key", by="player_id", direction="forward", allow_exact_matches=False)
    return out.set_index("event_id").dep


def recompute(con, M: pd.DataFrame) -> pd.DataFrame:
    """Window, games, completeness and minutes for the episodes in M (needs receiving_league, arrival_season)."""
    ep = M[["event_id", "player_id", "to_club_id", "receiving_league", "arrival_season", "transfer_date"]].copy()
    ep["dep"] = ep.event_id.map(departures(M))
    ep = ep.rename(columns={"to_club_id": "b", "receiving_league": "league", "arrival_season": "season"})
    ep["start"] = pd.to_datetime(ep.transfer_date)
    ep["season"] = ep.season.astype(int)
    con.register("ep", ep[["event_id", "player_id", "b", "league", "season", "start", "dep"]])
    R = con.sql(f"""
        with g as (select game_id, date, competition_id, cast(season as int) season, home_club_id, away_club_id
                   from games where competition_type = 'domestic_league'),
             bg as (select e.event_id, g.game_id, g.date from ep e join g on g.competition_id = e.league
                    and g.season = e.season and e.b in (g.home_club_id, g.away_club_id)),
             se as (select event_id, max(date) season_end, count(*) season_games from bg group by 1),
             wnd as (select e.event_id, e.start window_start,
                            case when e.dep is not null and e.dep <= se.season_end then cast(e.dep - interval 1 day as date)
                                 else se.season_end end window_end,
                            e.dep is not null and e.dep <= se.season_end departed_in_window, se.season_games
                     from ep e join se using (event_id)),
             w as (select bg.* from bg join wnd using (event_id)
                   where bg.date >= wnd.window_start and bg.date <= wnd.window_end),
             r as (select cast(game_id as varchar) game_id, player_club_id club, count(*) n from appearances group by 1, 2),
             pm as (select cast(game_id as varchar) game_id, player_id, player_club_id club, sum(minutes_played) mins,
                           count(*) k from appearances group by 1, 2, 3)
        select e.event_id, any_value(wnd.window_end) window_end, any_value(wnd.departed_in_window) departed_in_window,
               any_value(wnd.season_games) season_games, count(w.game_id) available_games,
               coalesce(sum(case when coalesce(r.n, 0) >= {COMPLETE_ROWS} then 1 else 0 end), 0) complete_games,
               coalesce(sum(pm.mins), 0) "minutes", coalesce(sum(pm.k), 0) "appearances"
        from ep e left join wnd using (event_id) left join w using (event_id)
             left join r on r.game_id = w.game_id and r.club = e.b
             left join pm on pm.game_id = w.game_id and pm.player_id = e.player_id and pm.club = e.b
        group by e.event_id""").df()
    con.unregister("ep")
    R["window_end"] = pd.to_datetime(R.window_end)
    R["share"] = R.minutes / (90 * R.available_games)
    return R.set_index("event_id")


def league_bounds(con) -> pd.DataFrame:
    return con.sql("""select competition_id league, cast(season as int) season, max(date) league_end from games
                      where competition_type = 'domestic_league' group by 1, 2""").df()


def compare(M: pd.DataFrame, R: pd.DataFrame, LB: pd.DataFrame) -> dict:
    X = M.set_index("event_id").join(R, rsuffix="_qa")
    chk = {
        "available_games": X.available_games == X.available_games_qa,
        "complete_games": X.complete_games == X.complete_games_qa,
        "all_games_complete": X.complete_games_qa == X.available_games_qa,
        "minutes": X.minutes == X.minutes_qa,
        "appearances": X.appearances == X.appearances_qa,
        "minutes_share": pd.Series(np.isclose(X.minutes_share, X.share, atol=1e-12), index=X.index),
        "window_start_is_transfer_date": X.window_start == X.transfer_date,
        "window_end": pd.to_datetime(X.window_end) == X.window_end_qa,
        "departed_in_window": X.departed_in_window.astype(bool) == X.departed_in_window_qa.astype(bool),
        "season_games": X.season_games == X.season_games_qa,
        "true_zero_iff_no_appearance_row": X.true_zero.astype(bool) == X.appearances_qa.eq(0),
        "share_within_0_1": X.minutes_share.between(0, 1),
    }
    lb = LB.set_index(["league", "season"]).league_end
    prev = [lb.get((l, int(s) - 1)) for l, s in zip(X.receiving_league, X.arrival_season)]
    end = [lb.get((l, int(s))) for l, s in zip(X.receiving_league, X.arrival_season)]
    t = pd.to_datetime(X.transfer_date)
    has_prev = pd.Series([p is not None for p in prev], index=X.index)
    chk["arrival_season_contains_move"] = pd.Series(
        [(p is None or p < d) and d <= e for p, d, e in zip(prev, t, end)], index=X.index)
    out = {k: {"checked": int(v.size), "agree": int(v.sum()),
               "disagree_ids": list(X.index[~v.to_numpy()][:10])} for k, v in chk.items()}
    out["arrival_season_first_snapshot_season_unchecked"] = int((~has_prev).sum())
    for k in ("loan", "permanent"):
        g = X[X.kind == k]
        out[f"agree_all_{k}"] = int(pd.concat([v[g.index] for v in chk.values()], axis=1).all(axis=1).sum())
        out[f"n_{k}"] = len(g)
    return out


# ---------------------------------------------------------------------------
# Hand-checkable examples
# ---------------------------------------------------------------------------

def pick_examples(M: pd.DataFrame, N: dict) -> list[tuple[str, str]]:
    """Deterministic choice: a fixed rule per slot, latest arrival season first, then the lowest event id."""
    R = M[M.in_revised_sample].sort_values(["arrival_season", "event_id"], ascending=[False, True])
    loan, perm = R[R.kind.eq("loan")], R[R.kind.eq("permanent")]
    picks = []

    def take(df, why):
        df = df[~df.event_id.isin([p for p, _ in picks])]
        picks.append((df.event_id.iloc[0], why))

    take(loan[loan.true_zero & loan.available_games.ge(10)], "loan, true zero (complete record, no appearance row)")
    take(loan[loan.minutes_share.ge(0.85) & loan.available_games.ge(30)], "loan, substantial minutes")
    take(loan[loan.departed_in_window & loan.minutes.gt(0)], "loan, early subsequent move (window cut short)")
    take(perm[perm.true_zero & perm.available_games.ge(10)], "permanent, true zero")
    take(perm[perm.minutes_share.ge(0.85) & perm.available_games.ge(30)], "permanent, substantial minutes")
    take(perm[perm.departed_in_window & perm.minutes.gt(0)], "permanent, early subsequent move")
    anton = M[M.player_id.eq(dl.ANTON) & M.in_revised_sample]
    if len(anton):
        picks.append((anton.event_id.iloc[0], "permanent, the worked example used in the data guide"))
    return picks


def example_games(con, X: pd.Series) -> pd.DataFrame:
    """Every league game of B in the arrival season, flagged in / out of the window, with B's rows and his minutes."""
    return con.sql(f"""
        with g as (select game_id, date, competition_id, home_club_id, away_club_id, home_club_name, away_club_name
                   from games where competition_type = 'domestic_league' and competition_id = '{X.receiving_league}'
                   and cast(season as int) = {int(X.arrival_season)} and {int(X.to_club_id)} in (home_club_id, away_club_id)),
             r as (select cast(game_id as varchar) game_id, count(*) b_rows from appearances
                   where player_club_id = {int(X.to_club_id)} group by 1),
             p as (select cast(game_id as varchar) game_id, sum(minutes_played) player_minutes from appearances
                   where player_id = {int(X.player_id)} and player_club_id = {int(X.to_club_id)} group by 1)
        select '{X.event_id}' event_id, g.game_id, g.date, g.competition_id,
               case when g.home_club_id = {int(X.to_club_id)} then 'home' else 'away' end venue,
               case when g.home_club_id = {int(X.to_club_id)} then g.away_club_name else g.home_club_name end opponent,
               coalesce(r.b_rows, 0) b_appearance_rows, p.player_minutes,
               g.date >= date '{X.window_start}' and g.date <= date '{X.window_end}' in_window
        from g left join r using (game_id) left join p using (game_id) order by g.date""").df()


def other_minutes(con, X: pd.Series) -> dict:
    """The player's minutes in the same season for other clubs, and for B outside the league: all excluded by design."""
    d = con.sql(f"""select a.player_club_id = {int(X.to_club_id)} at_b, coalesce(g.competition_type, '') ctype,
                           a.date >= date '{X.window_start}' and a.date <= date '{X.window_end}' in_window,
                           count(*) n, sum(minutes_played) mins
                    from appearances a left join games g on g.game_id = cast(a.game_id as varchar)
                    where a.player_id = {int(X.player_id)}
                      and cast(g.season as int) = {int(X.arrival_season)}
                    group by 1, 2, 3""").df()
    other = d[~d.at_b]
    cup = d[d.at_b & d.ctype.ne("domestic_league") & d.in_window]
    return {"other_club_games": int(other.n.sum()), "other_club_minutes": int(other.mins.fillna(0).sum()),
            "b_nonleague_in_window_games": int(cup.n.sum()), "b_nonleague_in_window_minutes": int(cup.mins.fillna(0).sum())}


def youth_side_diagnostic(M: pd.DataFrame) -> dict:
    """How many sample episodes come from a youth/reserve side, and how many of those look like a promotion inside one
    organisation that the exact-name internal-move rule missed (a rough name-overlap screen, for discussion only)."""
    out = {}
    for s in ("in_revised_sample", "in_age_model_sample", "in_cf_main_model_sample"):
        g = M[M[s] & M.is_youth_or_reserve_side.astype(bool)]
        a_youth = g.from_club_name.map(lambda x: bool(YOUTH_OR_RESERVE.search(str(x))))
        same = a_youth & pd.Series([bool(stem_tokens(a) & stem_tokens(b)) for a, b in zip(g.from_club_name, g.to_club_name)],
                                   index=g.index)
        out[s] = {k: {"youth_side": int(g.kind.eq(k).sum()), "from_youth_side_same_stem": int((same & g.kind.eq(k)).sum()),
                      "true_zero_share": float(g.loc[g.kind.eq(k), "true_zero"].mean()),
                      "mean_share": float(g.loc[g.kind.eq(k), "minutes_share"].mean()),
                      "examples": [f"{a} → {b}" for a, b in g.loc[same & g.kind.eq(k), ["from_club_name", "to_club_name"]]
                                   .head(8).itertuples(index=False)]}
                  for k in ("loan", "permanent")}
        rest = M[M[s] & ~M.is_youth_or_reserve_side.astype(bool)]
        out[s]["other_mean_share"] = {k: float(rest.loc[rest.kind.eq(k), "minutes_share"].mean()) for k in ("loan", "permanent")}
    wil = M.from_club_name.eq("Willem II") | M.to_club_name.eq("Willem II")
    out["willem_ii_flagged_youth"] = int((wil & M.is_youth_or_reserve_side.astype(bool)).sum())
    return out


def render(Q: dict, EX: list[dict]) -> str:
    W = ["# Playing-time QA: independent recomputation and hand-checkable examples", "",
         "Generated by `python -m src.analysis.playing_time_qa`. The recomputation uses plain SQL on the DuckDB snapshot "
         "and the Stage 1C file; none of the playing-time code is reused.", "",
         "## Rules being checked", "",
         "- **Window start** = the transfer date (Stage 1C `transfer_date`), for loans and permanent moves alike.",
         "- **Window end** = club B's last domestic-league game of its arrival season, or the day before the player's next "
         "substantive move (internal moves and draft rows skipped) if that comes first.",
         "- **Eligible games** = B's domestic-league games (`games.competition_type = 'domestic_league'`, the arrival "
         "league and season) dated inside the window. Cups, continental and national-team games never count.",
         f"- **Complete record** = B has at least {COMPLETE_ROWS} rows in `appearances` for that game "
         "(`player_club_id` = B). One incomplete game makes the whole outcome unreliable: no value is written.",
         "- **Minutes** = sum of `appearances.minutes_played` over rows with `player_id` = the player and "
         "`player_club_id` = B in the eligible games. Minutes for the old club A, or for B outside the league, are "
         "never included.",
         "- **True zero** = every eligible game complete and no appearance row for the player: minutes 0, share 0.",
         "- **Share** = minutes / (90 × eligible games).", "",
         "## Every reliable episode, recomputed", ""]
    rows = [[k, f0(v["checked"]), f0(v["agree"]), f0(v["checked"] - v["agree"])] for k, v in Q["checks"].items()
            if isinstance(v, dict)]
    W += md_table(["Check", "Episodes", "Agree", "Disagree"], rows, right={1, 2, 3}) + [""]
    W += [f"All checks agree for {f0(Q['checks']['agree_all_loan'])} of {f0(Q['checks']['n_loan'])} loans and "
          f"{f0(Q['checks']['agree_all_permanent'])} of {f0(Q['checks']['n_permanent'])} permanent moves. "
          f"The arrival-season test cannot look at the season before the first snapshot season "
          f"({f0(Q['checks']['arrival_season_first_snapshot_season_unchecked'])} episodes; they pass the upper bound).", ""]
    W += ["## Hand-checkable examples", "",
          "Game-level rows for every example: `playing_time_qa_example_games.csv` (one row per league game of B in the "
          "arrival season; `in_window` marks the eligible games; `b_appearance_rows` is the completeness count; "
          "`player_minutes` is blank where he has no row).", ""]
    rows = [[e["why"], f"{e['player']} ({e['player_id']})", f"{e['from']} → {e['to']}", e["date"], e["kind"],
             f"{e['league']} {e['season']}", f"{e['window_start']} – {e['window_end']}", str(e["games"]),
             f0(e["minutes"]), f"{e['share']:.3f}"] for e in EX]
    W += md_table(["Case", "Player (id)", "Transfer", "Date", "Kind", "League season", "Window", "Eligible games",
                   "Minutes", "Share"], rows, right={7, 8, 9}) + [""]
    for e in EX:
        W += [f"**{e['player']}**, {e['from']} → {e['to']} ({e['kind']}, {e['date']}, event `{e['event_id']}`).",
              ""] + [f"- {x}" for x in e["reasons"]] + [""]
    Y = Q["youth"]
    r, m = Y["in_revised_sample"], Y["in_age_model_sample"]
    W += ["## Data issue found: academy promotions counted as permanent arrivals", "",
          "The internal-move rule (Stage 1C) sets aside a move only when the two club names are identical after the "
          "youth / reserve suffix is removed. When the first team is written differently ('B. Dortmund II' → "
          "'Dortmund', 'RM Castilla' → 'Real Madrid'), a promotion inside one club becomes a permanent arrival, and "
          "a player who only trains with the first team becomes a true zero.", ""]
    rows = []
    for lab, d in (("revised sample", r), ("main regression sample", m), ("calendar-first main sample",
                                                                          Y["in_cf_main_model_sample"])):
        for k in ("loan", "permanent"):
            rows.append([lab, k, f0(d[k]["youth_side"]), f0(d[k]["from_youth_side_same_stem"]),
                         f"{d[k]['true_zero_share']:.1%}", f"{d[k]['mean_share']:.3f}", f"{d['other_mean_share'][k]:.3f}"])
    W += md_table(["Sample", "Kind", "A youth/reserve side involved", "…of which A is a youth side sharing a name stem "
                   "with B (rough screen)", "True zeros among them", "Mean share among them", "Mean share, all others"],
                  rows, right={2, 3, 4, 5, 6}) + [""]
    W += ["Examples caught by the screen (revised sample, permanent): " + "; ".join(r["permanent"]["examples"]) + ".",
          "", "The screen is a rough name-overlap check for discussion, not a definition: it also catches some genuine "
          "moves and misses promotions with unrelated names. The youth/reserve name pattern also flags the senior club "
          f"Willem II ({f0(Y['willem_ii_flagged_youth'])} master episodes), because its name ends in 'II'. Stage 1C is not "
          "changed; the follow-up club-organisation map, its master flags and the sensitivity regressions are in "
          f"`{dl.ACADEMY_MD.name}`.", ""]
    return "\n".join(W) + "\n"


def main() -> dict:
    import duckdb
    from ..config import DEFAULT_DATABASE

    M = pd.read_csv(dl.MASTER_CSV, low_memory=False)
    N = json.loads(dl.NUMBERS_JSON.read_text())
    Rl = M[M.pt_reliable].copy()
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        R = recompute(con, Rl)
        Q = {"checks": compare(Rl, R, league_bounds(con))}
        EX, G = [], []
        for eid, why in pick_examples(M, N):
            X = M.set_index("event_id").loc[eid]
            X["event_id"] = eid
            g = example_games(con, X)
            om = other_minutes(con, X)
            w = g[g.in_window]
            assert len(w) == X.available_games and int(w.player_minutes.fillna(0).sum()) == X.minutes
            assert (w.b_appearance_rows >= COMPLETE_ROWS).all()
            reasons = [
                f"Window {X.window_start} (transfer date) to {X.window_end}: "
                + (f"the day before his next move on {X.departure_date}." if X.departed_in_window else
                   f"B's last league game of the {int(X.arrival_season)}/{str(int(X.arrival_season) + 1)[2:]} season."),
                f"{int(X.available_games)} of B's {int(X.season_games)} league games fall in the window; all "
                f"{int(X.available_games)} have at least {COMPLETE_ROWS} B appearance rows (minimum "
                f"{int(w.b_appearance_rows.min())}).",
                f"He has an appearance row for B in {int(w.player_minutes.notna().sum())} of them; the minutes in those rows "
                f"sum to {f0(X.minutes)}, so the share is {f0(X.minutes)} / (90 × {int(X.available_games)}) = "
                f"{X.minutes_share:.3f}." + (" No row in any complete game: a true zero." if X.true_zero else ""),
                f"Not counted, by design: {om['other_club_games']} appearance(s) ({f0(om['other_club_minutes'])} min) for "
                f"other clubs that season, and {om['b_nonleague_in_window_games']} non-league appearance(s) for B in the "
                f"window ({f0(om['b_nonleague_in_window_minutes'])} min).",
                "The independent SQL recomputation gives the same games, completeness and minutes."
                if bool(np.isclose(R.loc[eid, "share"], X.minutes_share)) else "MISMATCH with the recomputation."]
            EX.append({"event_id": eid, "why": why, "player": X.player_name, "player_id": int(X.player_id),
                       "from": X.from_club_name, "to": X.to_club_name, "date": X.transfer_date, "kind": X.kind,
                       "league": X.receiving_league, "season": int(X.arrival_season), "window_start": X.window_start,
                       "window_end": X.window_end, "games": int(X.available_games), "minutes": int(X.minutes),
                       "share": float(X.minutes_share), "reasons": reasons, **om})
            G.append(g.assign(player_name=X.player_name, example=why))
    finally:
        con.close()
    Q["examples"] = [{k: v for k, v in e.items() if k != "reasons"} for e in EX]
    Q["youth"] = youth_side_diagnostic(M)
    pd.concat(G).to_csv(GAMES_CSV, index=False)
    REPORT_MD.write_text(render(Q, EX))
    NUMBERS_JSON.write_text(json.dumps(dl._json(Q), indent=1, default=str))
    bad = {k: v for k, v in Q["checks"].items() if isinstance(v, dict) and v["agree"] != v["checked"]}
    print(f"reliable episodes checked: {len(Rl):,}; disagreements: {bad or 'none'}")
    return Q


if __name__ == "__main__":
    main()
