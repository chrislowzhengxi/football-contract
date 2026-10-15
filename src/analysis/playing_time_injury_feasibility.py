"""Can playing time be adjusted for injury availability? Data audit and coverage on the revised sample.

    python -m src.analysis.playing_time_injury_feasibility

Audit only. The repository has no injury spells (no start or end dates of medical unavailability), so the
fit-adjusted outcome is NOT built. This module measures what the data do contain, on the revised playing-time
sample, without inferring any injury:
  - matchday squads (`game_lineups`: starting XI and named substitutes). Being named is positive evidence the
    player was available for that match; not being named is not evidence of injury;
  - in-match injury substitutions (`game_events`, reason "Injury"): positive evidence of an injury event on a date,
    with no duration or return date, and with the reason missing for part of the substitutions;
  - red cards (`appearances`), which do not give suspension dates.
Because squad matches are a subset of available matches, which are a subset of all matches, the data bound any
fit-adjusted share between the current share and the share over squad matches. New files only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from . import playing_time_join as ptj
from .daniel_scope_final import f0, md_table, pc
from .loan_episodes import ROOT
from .playing_time_descriptives import LABEL, describe, diff

OUT = ROOT / "data" / "outputs" / "rebuild"
REPORT_MD = OUT / "playing_time_injury_feasibility.md"
COVERAGE_CSV = OUT / "playing_time_injury_coverage.csv"
NUMBERS_JSON = OUT / "playing_time_injury_numbers.json"
WRITES = (REPORT_MD, COVERAGE_CSV, NUMBERS_JSON)
KINDS = ("loan", "permanent")
SEARCH_TERMS = ("injur", "suspen", "unavailab", "illness", "sidelin", "absence", "fitness", "medical")


# ---------------------------------------------------------------------------
# Part 1: what the repository holds
# ---------------------------------------------------------------------------

def schema_audit(con) -> list[dict]:
    """Every snapshot table, its row count, and any column whose name mentions availability."""
    out = []
    for (t,) in con.sql("show tables").fetchall():
        cols = [c[0] for c in con.sql(f"describe {t}").fetchall()]
        hits = [c for c in cols if any(s in c.lower() for s in SEARCH_TERMS)]
        out.append({"table": t, "rows": int(con.sql(f"select count(*) from {t}").fetchone()[0]), "columns": len(cols),
                    "availability_columns": hits})
    return out


def file_audit() -> dict:
    """Files under data/ (below 60 MB) that mention injuries, grouped by folder."""
    hits: dict = {}
    own = {p.name for p in WRITES}
    for p in sorted((ROOT / "data").rglob("*")):
        if p.suffix not in (".csv", ".json", ".md") or not p.is_file() or p.stat().st_size > 60e6 or p.name in own:
            continue
        try:
            txt = p.read_text(errors="ignore").lower()
        except OSError:
            continue
        if "injur" in txt:
            hits.setdefault(str(p.parent.relative_to(ROOT)), []).append(p.name)
    return {k: len(v) for k, v in hits.items()}


def event_audit(con, covered: list[str]) -> pd.DataFrame:
    """Per covered league-season: club-games, lineup completeness, substitution reasons recorded."""
    ids = ", ".join(f"'{c}'" for c in covered)
    return con.sql(f"""
        with lg as (select game_id, competition_id, cast(season as int) season, home_club_id, away_club_id
                    from games where competition_type = 'domestic_league' and competition_id in ({ids})),
             cg as (select game_id, competition_id, season, home_club_id club_id from lg
                    union all select game_id, competition_id, season, away_club_id from lg),
             lu as (select cast(game_id as varchar) game_id, club_id,
                           sum(case when type = 'starting_lineup' then 1 else 0 end) starters,
                           sum(case when type = 'substitutes' then 1 else 0 end) bench
                    from game_lineups group by 1, 2),
             ev as (select game_id, club_id, count(*) subs,
                           sum(case when description like '%Injury%' then 1 else 0 end) injury_subs,
                           sum(case when coalesce(trim(description), '') in ('', ', Not reported') then 1 else 0 end) no_reason
                    from game_events where type = 'Substitutions' group by 1, 2)
        select cg.competition_id league, cg.season, count(*) club_games,
               sum(case when lu.starters >= 11 then 1 else 0 end) lineup_complete,
               sum(case when lu.bench > 0 then 1 else 0 end) bench_listed,
               sum(case when ev.subs > 0 then 1 else 0 end) with_substitutions,
               coalesce(sum(ev.subs), 0) substitutions, coalesce(sum(ev.injury_subs), 0) injury_substitutions,
               coalesce(sum(ev.no_reason), 0) substitutions_without_reason
        from cg left join lu on lu.game_id = cg.game_id and lu.club_id = cg.club_id
                left join ev on ev.game_id = cg.game_id and ev.club_id = cg.club_id
        group by 1, 2 order by 1, 2""").df()


def substitution_semantics(con) -> dict:
    """Which player a substitution event names: the one going off (his minutes end at the event minute)."""
    r = con.sql("""select count(*) n,
                          avg(case when a.minutes_played between e.minute - 3 and e.minute + 3 then 1 else 0 end) going_off
                   from game_events e join appearances a
                     on cast(a.game_id as varchar) = e.game_id and a.player_id = e.player_id
                   where e.type = 'Substitutions' and e.description like '%Injury%' and e.minute between 10 and 80""").df().iloc[0]
    return {"checked": int(r.n), "player_id_is_player_going_off": float(r.going_off)}


def substitution_reasons(con) -> dict:
    d = con.sql("""select coalesce(nullif(trim(description), ''), '(blank)') reason, count(*) n from game_events
                   where type = 'Substitutions' group by 1 order by 2 desc""").df()
    return dict(zip(d.reason.str.lstrip(", "), d.n.astype(int)))


# ---------------------------------------------------------------------------
# Part 2-3: per-episode coverage on the revised sample
# ---------------------------------------------------------------------------

def revised_sample(u, con) -> tuple[pd.DataFrame, list[str]]:
    J, E, _ = ptj.compute(u, con)
    rev = E.reliable & ~E.prior_spell_at_b & ((E.kind == "permanent") | (E.loan_ending == "return to lender"))
    S = E[rev].copy()
    saved = OUT / "playing_time_sample_redesign_numbers.json"
    if saved.exists():
        assert S.kind.value_counts().to_dict() == json.loads(saved.read_text())["revised"], "revised sample differs"
    return S, J["covered_leagues"]


def episode_games(S: pd.DataFrame, con) -> pd.DataFrame:
    """One row per (episode, league game of B inside the window), with squad, appearance and injury evidence."""
    g = con.sql("""select game_id, cast(season as int) season, date, home_club_id, away_club_id from games
                   where competition_type = 'domestic_league'""").df()
    g["date"] = pd.to_datetime(g.date)
    cg = pd.concat([g.rename(columns={"home_club_id": "club_id"}).drop(columns="away_club_id"),
                    g.rename(columns={"away_club_id": "club_id"}).drop(columns="home_club_id")])
    left = S[["event_id", "kind", "player_id", "b_club_id", "arrival_season", "window_start", "window_end"]].copy()
    left["arrival_season"] = left.arrival_season.astype(int)
    W = left.merge(cg, left_on=["b_club_id", "arrival_season"], right_on=["club_id", "season"])
    W = W[(W.date >= W.window_start) & (W.date <= W.window_end)].drop(columns=["club_id", "season"])
    games = set(W.game_id)
    players = set(S.player_id.astype(int))
    con.register("gids", pd.DataFrame({"game_id": sorted(games)}))
    con.register("pids", pd.DataFrame({"player_id": sorted(players)}))
    lu = con.sql("""select cast(game_id as varchar) game_id, club_id, player_id, type from game_lineups
                    where cast(game_id as varchar) in (select game_id from gids)""").df()
    starters = lu[lu.type == "starting_lineup"].groupby(["game_id", "club_id"]).size().rename("b_starters")
    named = lu[lu.player_id.isin(players)][["game_id", "club_id", "player_id", "type"]].drop_duplicates(
        ["game_id", "club_id", "player_id"])
    ev = con.sql("""select game_id, club_id, player_id, description from game_events
                    where type = 'Substitutions' and game_id in (select game_id from gids)""").df()
    reason = ev.assign(no_reason=ev.description.fillna("").str.strip().isin(["", ", Not reported"])) \
        .groupby(["game_id", "club_id"]).agg(b_subs=("no_reason", "size"), b_subs_no_reason=("no_reason", "sum"))
    inj = ev[ev.description.fillna("").str.contains("Injury")].groupby(["game_id", "player_id"]).size().rename("injury_off")
    ap = con.sql("""select cast(game_id as varchar) game_id, player_id, player_club_id club_id, minutes_played, red_cards
                    from appearances where cast(game_id as varchar) in (select game_id from gids)
                    and player_id in (select player_id from pids)""").df()
    W = W.merge(starters.reset_index().rename(columns={"club_id": "b_club_id"}), on=["game_id", "b_club_id"], how="left")
    W = W.merge(named.rename(columns={"club_id": "b_club_id", "type": "named_as"}), on=["game_id", "b_club_id", "player_id"],
                how="left")
    W = W.merge(reason.reset_index().rename(columns={"club_id": "b_club_id"}), on=["game_id", "b_club_id"], how="left")
    W = W.merge(inj.reset_index(), on=["game_id", "player_id"], how="left")
    W = W.merge(ap.rename(columns={"club_id": "b_club_id"}), on=["game_id", "player_id", "b_club_id"], how="left")
    W["lineup_complete"] = W.b_starters.fillna(0) >= ptj.COMPLETE_ROWS
    W["named"] = W.named_as.notna()
    W["played"] = W.minutes_played.notna()
    W["injury_off"] = W.injury_off.fillna(0).astype(int)
    return W


def per_episode(S: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    a = W.groupby("event_id").agg(
        games=("game_id", "size"), games_lineup_complete=("lineup_complete", "sum"), games_named=("named", "sum"),
        games_started=("named_as", lambda s: int((s == "starting_lineup").sum())), games_played=("played", "sum"),
        minutes_window=("minutes_played", "sum"),
        injury_subs_off=("injury_off", "sum"), red_cards=("red_cards", "sum"),
        b_subs=("b_subs", "sum"), b_subs_no_reason=("b_subs_no_reason", "sum"))
    a["played_not_named"] = W[W.played & ~W.named].groupby("event_id").size().reindex(a.index).fillna(0).astype(int)
    P = S.set_index("event_id").join(a)
    assert (P.games == P.available_games).all(), "window games do not reproduce the join's available games"
    assert np.allclose(P.minutes_window, P.minutes), "window minutes do not reproduce the join's minutes"
    P["squad_proxy_ok"] = P.games_lineup_complete == P.games
    P["current_share"] = P.minutes_share
    named_min = 90 * P.games_named
    P["share_over_named"] = np.where(P.games_named > 0, P.minutes / named_min.replace(0, np.nan), 0.0)
    P["named_rate"] = P.games_named / P.games
    P["played_rate"] = P.games_played / P.games
    P["b_reason_missing"] = P.b_subs_no_reason / P.b_subs.replace(0, np.nan)
    return P.reset_index()


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(u=None) -> tuple[dict, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    u = u or loan_scope.universe()
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        S, covered = revised_sample(u, con)
        N: dict = {"schema": schema_audit(con), "files": file_audit(), "sub_semantics": substitution_semantics(con),
                   "sub_reasons": substitution_reasons(con)}
        EA = event_audit(con, covered)
        W = episode_games(S, con)
    finally:
        con.close()
    P = per_episode(S, W)
    N["revised"] = {k: int((P.kind == k).sum()) for k in KINDS}
    N["distinct_players"] = int(P.player_id.nunique())
    N["covered"] = covered
    N["injury_spell_tables"] = [s["table"] for s in N["schema"] if s["availability_columns"]]
    by_season = EA.groupby("season")[["club_games", "lineup_complete", "substitutions", "substitutions_without_reason",
                                      "injury_substitutions"]].sum()
    N["events_by_season"] = {int(s): {"lineup_complete": float(r.lineup_complete / r.club_games),
                                      "reason_missing": float(r.substitutions_without_reason / r.substitutions),
                                      "injury_subs": int(r.injury_substitutions)} for s, r in by_season.iterrows()}
    by_league = EA.groupby("league")[["club_games", "lineup_complete", "substitutions", "substitutions_without_reason"]].sum()
    N["events_by_league"] = {lg: {"lineup_complete": float(r.lineup_complete / r.club_games),
                                  "reason_missing": float(r.substitutions_without_reason / r.substitutions)}
                             for lg, r in by_league.iterrows()}

    # ---- Part 3: coverage waterfall
    wf = {}
    for k in KINDS:
        g = P[P.kind == k]
        wf[k] = {"episodes": len(g), "players": int(g.player_id.nunique()),
                 "with_injury_spell_data": 0,
                 "fit_adjusted_computable": 0,
                 "with_any_injury_substitution_in_window": int((g.injury_subs_off > 0).sum()),
                 "players_with_injury_substitution": int(g.loc[g.injury_subs_off > 0, "player_id"].nunique()),
                 "squad_proxy_computable": int(g.squad_proxy_ok.sum()),
                 "squad_proxy_lost": int((~g.squad_proxy_ok).sum()),
                 "squad_proxy_lost_2012": int((~g.squad_proxy_ok & g.arrival_season.eq(2012)).sum()),
                 "with_red_card_in_window": int((g.red_cards > 0).sum()),
                 "b_reason_missing_mean": float(g.b_reason_missing.mean())}
    N["waterfall"] = wf
    N["squad_by_season"] = {k: {int(s): float(g.squad_proxy_ok.mean()) for s, g in P[P.kind == k].groupby("arrival_season")}
                            for k in KINDS}
    N["consistency"] = {"played_not_named": int(P.played_not_named.sum()),
                        "episodes_with_played_not_named": int((P.played_not_named > 0).sum())}

    # ---- what the squad data show (not fit-adjusted)
    Q = P[P.squad_proxy_ok]
    levels = {}
    for k in KINDS:
        g = Q[Q.kind == k]
        levels[k] = {"N": len(g), "games": int(g.games.sum()), "named": int(g.games_named.sum()),
                     "started": int(g.games_started.sum()), "played": int(g.games_played.sum()),
                     "named_rate_mean": float(g.named_rate.mean()), "played_rate_mean": float(g.played_rate.mean()),
                     "never_named": float((g.games_named == 0).mean()),
                     "current_share": describe(g.current_share), "share_over_named": describe(g.share_over_named),
                     "injury_subs_per_100_games_played": float(100 * g.injury_subs_off.sum() / max(g.games_played.sum(), 1))}
    N["levels"] = levels
    lo = diff(Q.loc[Q.kind == "loan", "current_share"], Q.loc[Q.kind == "permanent", "current_share"])
    hi_ = diff(Q.loc[Q.kind == "loan", "share_over_named"], Q.loc[Q.kind == "permanent", "share_over_named"])
    L, Pm = levels["loan"], levels["permanent"]
    N["bounds"] = {"current_gap": lo["mean_diff"], "named_gap": hi_["mean_diff"],
                   "loan_range": [L["current_share"]["mean"], L["share_over_named"]["mean"]],
                   "perm_range": [Pm["current_share"]["mean"], Pm["share_over_named"]["mean"]],
                   "gap_worst_low": L["current_share"]["mean"] - Pm["share_over_named"]["mean"],
                   "gap_worst_high": L["share_over_named"]["mean"] - Pm["current_share"]["mean"]}

    # ---- coverage table by kind, league and season
    rows = []
    for dim, keys in (("all", ["kind"]), ("league", ["kind", "receiving_league"]), ("season", ["kind", "arrival_season"]),
                      ("league_season", ["kind", "receiving_league", "arrival_season"])):
        for key, g in P.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            rec = {"dimension": dim, "kind": key[0], "league": key[1] if dim in ("league", "league_season") else "all",
                   "season": key[-1] if dim in ("season", "league_season") else "all",
                   "episodes": len(g), "with_injury_spell_data": 0, "fit_adjusted_computable": 0,
                   "squad_proxy_computable": int(g.squad_proxy_ok.sum()),
                   "with_injury_substitution": int((g.injury_subs_off > 0).sum()),
                   "club_substitutions_without_reason": float(g.b_subs_no_reason.sum() / max(g.b_subs.sum(), 1)),
                   "mean_named_rate": float(g.named_rate.mean()), "mean_played_rate": float(g.played_rate.mean())}
            rows.append(rec)
    COV = pd.DataFrame(rows)
    return N, {"P": P, "COV": COV, "EA": EA}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def p3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def pp(x) -> str:
    return f"{100 * x:+.1f} pp"


def render(N: dict) -> str:
    wf, lv, bd = N["waterfall"], N["levels"], N["bounds"]
    sr = N["sub_reasons"]
    tot_sub = sum(sr.values())
    es = N["events_by_season"]
    seasons = sorted(es)
    W = ["# Injury-adjusted playing time: feasibility", "",
         "*Generated by `python -m src.analysis.playing_time_injury_feasibility`. Audit only. No injury spell is "
         "invented, no absence from a lineup is read as an injury, and the current outcome is untouched. Revised "
         "playing-time sample. New files only.*", ""]
    W += ["## Answer", "",
          "**The fit-adjusted outcome cannot be built from the repository.** No table or file holds injury spells: "
          "there is no injury start date, end date, return date, or medical-absence status for any player. The closest "
          "evidence is (a) in-match substitutions tagged \"Injury\", which date an injury event but say nothing about how "
          "long the player was out, and (b) matchday squads, which show when a player was available and named, but not "
          f"why he was absent otherwise. Fit-adjusted outcomes computable: **0 of {f0(N['revised']['loan'])} loans and "
          f"0 of {f0(N['revised']['permanent'])} permanent moves.** Parts 4–6 (fit-adjusted outcome, comparison, "
          "regression) were therefore not run.", ""]

    # ---- Part 1
    W += ["## 1. What the repository holds", "", "**Snapshot tables** (`data/raw/transfermarkt-datasets.duckdb`):", ""]
    W += md_table(["Table", "Rows", "Columns about availability / injury"],
                  [[s["table"], f0(s["rows"]), ", ".join(s["availability_columns"]) or "none"] for s in N["schema"]],
                  right={1})
    fl = N["files"]
    cr = {k: v for k, v in fl.items() if "contract_research" in k or "discovered_sources" in k or k.endswith("stage2")}
    other = {k: v for k, v in fl.items() if k not in cr}
    W += ["", "Other sources checked: the cached upstream raw files hold transfer histories and market values only. "
          f"Under `data/`, {f0(sum(fl.values()))} files mention injuries: {f0(sum(cr.values()))} are web pages and text "
          f"extracts cached by the earlier contract research ({len(cr)} folders)"
          + ("; " + ", ".join(f"{v} in `{k}`" for k, v in other.items()) + " (a generated report that mentions "
             "injuries as a limitation of the playing-time analysis)" if other else "")
          + ". None is a structured injury record.", "",
          "**The candidate sources, one by one:**", ""]
    lc = [es[s]["lineup_complete"] for s in seasons]
    rm = [es[s]["reason_missing"] for s in seasons]
    W += md_table(["Source", "What it records", "Seasons / leagues", "Date granularity", "Start / end of absence?",
                   "Injury vs suspension vs other?", "Missingness", "Join keys"],
                  [["`game_lineups` (starting_lineup, substitutes)", "who was named in each matchday squad",
                    f"14 covered leagues; complete from {seasons[1]}/{str(seasons[1] + 1)[2:]} "
                    f"({pct(min(lc[1:]))}–{pct(max(lc[1:]))} of club-games), {pct(lc[0])} in {seasons[0]}/{str(seasons[0] + 1)[2:]}",
                    "match", "no", "no: absence from a squad has no reason", "see the per-season table",
                    "game_id, club_id, player_id"],
                   ["`game_events`, type Substitutions, reason \"Injury\"",
                    f"player taken off injured ({f0(sr.get('Injury', 0))} events; player_id is the player going off in "
                    f"{pct(N['sub_semantics']['player_id_is_player_going_off'])} of {f0(N['sub_semantics']['checked'])} checks)",
                    "14 covered leagues, 2012/13–2025/26", "match minute", "start only (the match); no end or return",
                    "injury only, when a reason is recorded",
                    f"reason missing for {pct(rm[0])} of substitutions in {seasons[0]}/{str(seasons[0] + 1)[2:]} and "
                    f"{pct(rm[-1])} in {seasons[-1]}/{str(seasons[-1] + 1)[2:]}; worst league "
                    f"{max(N['events_by_league'], key=lambda k: N['events_by_league'][k]['reason_missing'])} "
                    f"({pct(max(v['reason_missing'] for v in N['events_by_league'].values()))})",
                    "game_id, club_id, player_id"],
                   ["`game_events`, type Cards; `appearances.red_cards`", "yellow and red cards",
                    "14 covered leagues", "match", "no suspension dates", "cards only; suspension length depends on "
                    "league rules, so it would be inferred", "not audited further", "game_id, player_id"],
                   ["`players`, `player_valuations`, raw transfer files", "profile, values, transfers", "all", "–",
                    "no", "no", "–", "player_id"]], right=set())
    W += ["", "Substitution reasons across all events: "
          + "; ".join(f"{k} {f0(v)} ({pct(v / tot_sub)})" for k, v in sr.items()) + ".", "",
          "Per season, covered leagues:", ""]
    W += md_table(["Season"] + [f"{s}/{str(s + 1)[2:]}" for s in seasons],
                  [["club-games with complete lineup"] + [pct(es[s]["lineup_complete"]) for s in seasons],
                   ["substitutions without a reason"] + [pct(es[s]["reason_missing"]) for s in seasons],
                   ["injury substitutions"] + [f0(es[s]["injury_subs"]) for s in seasons]])
    W += [""]

    # ---- Part 2
    W += ["## 2. Can \"fit minutes available\" be built?", "",
          "No. The denominator needs, for each league game of the receiving club in the window, positive evidence that "
          "the player was medically unavailable. None of the sources gives that:", "",
          "- **Injury spells:** no start, end or return date anywhere. An injury substitution dates the injury but not "
          "the return, so every later game would have to be guessed.",
          "- **Injuries between matches** (training, illness) leave no event at all.",
          "- **Missing reasons:** where a substitution has no reason, an injury cannot be ruled out; the share without a "
          "reason varies by season and league, so the evidence is uneven, not just thin.",
          "- **Absent from the squad:** could be injury, illness, suspension, international duty, personal reasons or "
          "non-selection. It cannot be read as injury.",
          "- **Suspensions:** only cards are recorded; the games served would have to be derived from league rules.",
          "- **Partial availability** (named but not fit to start) is not recorded.", ""]

    # ---- Part 3
    W += ["## 3. Coverage on the revised sample", ""]
    rows = []
    for lab, key in (("episodes in the revised sample", "episodes"), ("distinct players", "players"),
                     ("with injury-spell data", "with_injury_spell_data"),
                     ("**fit-adjusted outcome computable**", "fit_adjusted_computable"),
                     ("with ≥1 in-match injury substitution in the window", "with_any_injury_substitution_in_window"),
                     ("players with such an event", "players_with_injury_substitution"),
                     ("squad proxy computable (every window game has a complete lineup)", "squad_proxy_computable"),
                     ("squad proxy lost", "squad_proxy_lost"), ("… of which arrival season 2012/13", "squad_proxy_lost_2012"),
                     ("with a red card in the window", "with_red_card_in_window")):
        rows.append([lab, f0(wf["loan"][key]), f0(wf["permanent"][key])])
    W += md_table(["", "Loans", "Permanent"], rows)
    W += ["", f"Every episode is lost for the fit-adjusted outcome, for one reason: no injury-spell data. The squad "
          f"proxy loses {f0(wf['loan']['squad_proxy_lost'])} loans and {f0(wf['permanent']['squad_proxy_lost'])} permanent "
          f"moves: {f0(wf['loan']['squad_proxy_lost_2012'])} and {f0(wf['permanent']['squad_proxy_lost_2012'])} arrived in "
          "2012/13, when the snapshot has no lineups; the rest have at least one window game whose lineup is "
          f"incomplete. Per episode, on average {pct(wf['loan']['b_reason_missing_mean'])} (loans) and "
          f"{pct(wf['permanent']['b_reason_missing_mean'])} (permanent) of the receiving club's substitutions in the "
          "window have no reason. Squad-proxy coverage by arrival season:", ""]
    ss = N["squad_by_season"]
    W += md_table(["Season"] + [f"{s}/{str(int(s) + 1)[2:]}" for s in ss["loan"]],
                  [[f"{LABEL[k]} computable"] + [pct(v) for v in ss[k].values()] for k in KINDS])
    W += ["", f"By league and season, with every column, in `{COVERAGE_CSV.name}`.", ""]

    # ---- Part 7
    W += ["## 4. Medically fit, named, played: what the squad data can say", "",
          "These are three different things: medically fit ⊇ available and named in the matchday squad ⊇ played. "
          "Only the last two are observed. On the episodes where the squad proxy is computable:", ""]
    rows = []
    for lab, f in (("episodes", lambda d: f0(d["N"])), ("league games in the windows", lambda d: f0(d["games"])),
                   ("named in the matchday squad", lambda d: f"{f0(d['named'])} ({pc(d['named'], d['games'])})"),
                   ("started", lambda d: f"{f0(d['started'])} ({pc(d['started'], d['games'])})"),
                   ("played", lambda d: f"{f0(d['played'])} ({pc(d['played'], d['games'])})"),
                   ("never named during the window", lambda d: pct(d["never_named"])),
                   ("current share (all games), mean", lambda d: p3(d["current_share"]["mean"])),
                   ("share over named games only, mean", lambda d: p3(d["share_over_named"]["mean"])),
                   ("in-match injury substitutions per 100 games played",
                    lambda d: f"{d['injury_subs_per_100_games_played']:.2f}")):
        rows.append([lab, f(lv["loan"]), f(lv["permanent"])])
    W += md_table(["", "Loans", "Permanent"], rows)
    W += ["", "The only injury comparison the data support is event incidence: in-match injury substitutions occur at "
          f"{lv['loan']['injury_subs_per_100_games_played']:.2f} (loans) and "
          f"{lv['permanent']['injury_subs_per_100_games_played']:.2f} (permanent) per 100 games played. That counts "
          "injuries suffered on the pitch while playing, not time lost, and it depends on reasons being recorded.", "",
          f"**Bounds, not an estimate.** For each player, the fit-adjusted share lies between the current share "
          "(all games in the denominator) and the share over named games (only games in which he was demonstrably "
          "available; a player never named is set to 0, which assumes he was fit for at least one game). The group "
          f"means are bounded by [{p3(bd['loan_range'][0])}, {p3(bd['loan_range'][1])}] for loans and "
          f"[{p3(bd['perm_range'][0])}, {p3(bd['perm_range'][1])}] for permanent moves, so the loan-minus-permanent "
          f"gap is bounded only by [{pp(bd['gap_worst_low'])}, {pp(bd['gap_worst_high'])}]. The current gap on these "
          f"episodes is {pp(bd['current_gap'])}. The share over named games ({pp(bd['named_gap'])} gap) is not a "
          "fit-adjusted outcome: it also removes every game in which a fit player was left out, which is what the "
          "playing-time measure is meant to capture.", "",
          "The squad data therefore cannot stand in for injury data. They can support a separately named "
          "\"conditional on being named\" outcome, if Professor Freund wants one, but not \"conditional on being fit\".", ""]
    W += ["## 5. What would be needed", "",
          "- **Fields:** per player, injury (and illness) spells with start date, end or return date, type, and ideally "
          "the club games missed; separately, suspensions with the games served; keyed by the Transfermarkt player id "
          "(the id we already use).",
          "- **Coverage needed:** every player in the revised sample during the window, i.e. the 14 covered leagues, "
          "2012/13–2025/26; a source covering only some leagues or seasons would give a selected sample.",
          "- **Most compatible source:** Transfermarkt's own per-player injury history, which lists season, injury, "
          "from, until, days and games missed under the same player id. It is not in the snapshot or in the upstream "
          f"raw files; collecting it means one page per player ({f0(N['distinct_players'])} distinct players in the "
          "revised sample), which is scraping and needs approval and a terms-of-use check. Community scrapers that "
          "expose it (for example an "
          "open-source Transfermarkt API with an injuries endpoint) are the same scraping underneath.",
          "- **Paid feeds** (commercial football data APIs with injury or \"sidelined\" endpoints) are excluded by the "
          "no-paid-API rule and use their own player ids, which would need a mapping.",
          "- **Without new data:** keep the current outcome as primary; at most, report the \"named in squad\" view "
          "above as a descriptive bound, labelled as not injury-adjusted.", ""]
    return "\n".join(W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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


def main(u=None) -> dict:
    from .club_geography_v2 import CANONICAL_CSV, STAGE1C_SHA256, sha256

    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    old = [ptj.EPISODES_CSV, ptj.NUMBERS_JSON, OUT / "playing_time_sample_redesign_numbers.json",
           OUT / "playing_time_age_numbers.json", OUT / "playing_time_regression_numbers.json"]
    before = {p.name: sha256(p) for p in old if p.exists()}
    N, tb = compute(u)
    tb["COV"].to_csv(COVERAGE_CSV, index=False)
    md = render(N)
    for k in KINDS:
        assert f0(N["revised"][k]) in md
    REPORT_MD.write_text(md)
    N["old_output_sha256"] = before
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    assert {p.name: sha256(p) for p in old if p.exists()} == before
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print_summary(N)
    return N


def print_summary(N: dict) -> None:
    wf, es = N["waterfall"], N["events_by_season"]
    s0, s1 = min(es), max(es)
    print("INJURY DATA")
    print("- source: none with injury spells. Closest: game_events substitutions tagged 'Injury' (event date only) "
          "and game_lineups (matchday squads); cards for suspensions")
    print(f"- leagues/seasons: the 14 covered leagues, {s0}/{str(s0 + 1)[2:]}–{s1}/{str(s1 + 1)[2:]}; lineups missing in "
          f"{s0}/{str(s0 + 1)[2:]}; substitution reason missing {pct(es[s0]['reason_missing'])} ({s0}) to "
          f"{pct(es[s1]['reason_missing'])} ({s1})")
    print(f"- coverage: injury spells 0 of {wf['loan']['episodes']:,} loans and 0 of {wf['permanent']['episodes']:,} "
          f"permanent; squad proxy computable for {wf['loan']['squad_proxy_computable']:,} / "
          f"{wf['permanent']['squad_proxy_computable']:,}")
    print("FIT-ADJUSTED OUTCOME")
    print("- feasible yes/no: no")
    print("- usable loans: 0")
    print("- usable permanent: 0")
    print("RESULT, IF FEASIBLE")
    print("- raw-outcome loan coefficient: not run (no same-sample fit-adjusted outcome exists)")
    print("- fit-adjusted coefficient: not run")
    b = N["bounds"]
    print(f"- material change?: cannot be determined; the data bound the fit-adjusted gap only to "
          f"[{pp(b['gap_worst_low'])}, {pp(b['gap_worst_high'])}] around the current {pp(b['current_gap'])}")
    print("LIMITATION")
    print("- biggest data concern: no injury start/end dates; absence from a squad cannot be told apart from "
          "non-selection, suspension or international duty")
    print("RECOMMENDATION")
    print("- not currently supportable; keep the current outcome. Supporting it needs Transfermarkt per-player injury "
          "histories (same player id), which requires approval to collect")


if __name__ == "__main__":
    main()
