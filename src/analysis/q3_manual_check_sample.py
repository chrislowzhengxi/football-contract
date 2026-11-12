"""Small random sample of bench appearances for a by-hand check of substitute minutes (Fernandes, Question 3).

Draws, with a fixed seed, appearance rows of the calendar-first main regression sample in which the player came off
the bench, stratified by scenario, and one worked loan episode whose match-by-match minutes add up to the stored
`minutes` and `minutes_share`. Every row carries the Transfermarkt match-report URL from the games table so it can be
opened by hand in a browser (no automated access), plus blank columns for the checker to fill in.

Read-only: uses the same functions as the Question 3 validation (`league_minutes_rows`, `window_rows`,
`classify_minutes`) and writes only into OUT_DIR.

Run: python -m src.analysis.q3_manual_check_sample
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from ..config import DEFAULT_DATABASE
from . import feedback_nov2026 as fb
from .feedback_nov2026_report import MASTER, OUT

SEED = 20261110
OUT_DIR = OUT / "professor_feedback_nov_2026" / "q3_manual_check_sample"

# scenario -> (row filter on bench rows, number drawn)
SCENARIOS = {
    "came on in regular time (47-89'), played to the end": (
        lambda A: A.m_in.between(47, 89) & A.m_out.isna() & A.m_red.isna(), 10),
    "came on at half-time (46')": (lambda A: A.m_in.eq(46) & A.m_out.isna() & A.m_red.isna(), 3),
    "came on in the first half (1-45')": (lambda A: A.m_in.between(1, 45) & A.m_out.isna() & A.m_red.isna(), 2),
    "came on in stoppage time (90')": (lambda A: A.m_in.ge(90) & A.m_out.isna() & A.m_red.isna(), 2),
    "came on and was later taken off": (lambda A: A.m_out.notna() & A.m_out.gt(A.m_in), 2),
    "came on and was sent off": (lambda A: A.m_red.notna() & A.m_red.ge(A.m_in), 1),
}

FILL_IN = ["your_minute_on", "your_minute_off", "your_expected_minutes", "agrees_yes_no", "your_notes"]


def load():
    M = pd.read_csv(MASTER, low_memory=False)
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        Aall = fb.league_minutes_rows(con)
        G = con.sql("select game_id, competition_id, season, home_club_id, home_club_name, away_club_name, "
                    "home_club_goals, away_club_goals, url from games").df()
    finally:
        con.close()
    main = M[M.in_cf_main_model_sample.astype(bool)]
    W = main[["event_id", "player_id", "to_club_id", "window_start", "window_end"]].rename(
        columns={"to_club_id": "club_id"})
    A = fb.window_rows(Aall, W)
    G = G.assign(game_id=G.game_id.astype("int64"))
    A = A.merge(G, on="game_id", how="left")
    home = A.club_id.eq(A.home_club_id)
    A["opponent"] = np.where(home, A.away_club_name, A.home_club_name)
    A["venue"] = np.where(home, "home", "away")
    A["score_club_first"] = np.where(
        home, A.home_club_goals.astype("Int64").astype(str) + "-" + A.away_club_goals.astype("Int64").astype(str),
        A.away_club_goals.astype("Int64").astype(str) + "-" + A.home_club_goals.astype("Int64").astype(str))
    Mk = main.set_index("event_id")
    for c, src in (("player_name", "player_name"), ("club", "to_club_name"), ("kind", "kind")):
        A[c] = A.event_id.map(Mk[src])
    A["date"] = A.date.dt.strftime("%Y-%m-%d")
    return main, A


def draw_bench(A: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    B = A[A.role.eq("substitute")].drop_duplicates(["game_id", "player_id"])
    picks, used = [], set()
    for name, (rule, n) in SCENARIOS.items():
        pool = B[rule(B) & ~B.game_id.isin(used)]
        if name.startswith("came on in regular time"):
            # one row from each of n different leagues, so the sample spans the covered competitions
            comps = rng.choice(sorted(pool.competition_id.unique()), size=n, replace=False)
            rows = [pool[pool.competition_id.eq(c)].sample(1, random_state=int(rng.integers(1 << 31)))
                    for c in comps]
            got = pd.concat(rows)
        else:
            got = pool.sample(min(n, len(pool)), random_state=int(rng.integers(1 << 31)))
        used |= set(got.game_id)
        picks.append(got.assign(scenario=name))
    S = pd.concat(picks).reset_index(drop=True)
    S.insert(0, "check_id", [f"B{i + 1:02d}" for i in range(len(S))])
    return S


def tidy(S: pd.DataFrame) -> pd.DataFrame:
    T = S.rename(columns={"m_in": "minute_on_in_events", "m_out": "minute_off_in_events",
                          "m_red": "minute_sent_off_in_events", "minutes_played": "minutes_recorded",
                          "expected_minutes": "minutes_expected_from_events", "url": "match_report_url",
                          "competition_id": "league", "kind": "episode_kind"})
    for c in ("minute_on_in_events", "minute_off_in_events", "minute_sent_off_in_events", "minutes_recorded",
              "minutes_expected_from_events", "minutes_difference"):
        T[c] = T[c].astype("Int64")
    cols = ["check_id", "scenario", "player_name", "club", "episode_kind", "league", "season", "date", "venue",
            "opponent", "score_club_first", "minute_on_in_events", "minute_off_in_events",
            "minute_sent_off_in_events", "minutes_recorded", "minutes_expected_from_events", "minutes_difference",
            "minutes_status", "match_report_url", "event_id", "game_id", "player_id"]
    T = T[cols]
    for c in FILL_IN:
        T[c] = ""
    return T


def worked_episode(main: pd.DataFrame, A: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """A loan in the main sample, all rows verified, with between 6 and 12 bench appearances among 10-20 rows,
    chosen at random among those. Its match rows must add up to the stored minutes and share."""
    rng = np.random.default_rng(SEED + 1)
    g = A.groupby("event_id").agg(rows=("game_id", "size"), bench=("role", lambda r: r.eq("substitute").sum()),
                                  ok=("minutes_status", lambda s: s.str.startswith("verified").all()))
    loans = set(main.event_id[main.kind.eq("loan")])
    cand = g[g.ok & g.rows.between(10, 20) & g.bench.between(6, 12) & g.index.isin(loans)].index
    ev = str(rng.choice(sorted(cand)))
    m = main.set_index("event_id").loc[ev]
    E = A[A.event_id.eq(ev)].sort_values("date")
    E = tidy(E.assign(scenario=E.role, check_id="")).drop(columns=["check_id"] + FILL_IN)
    E = E.rename(columns={"scenario": "role"})
    E["your_minutes_from_match_report"] = ""
    total = int(E.minutes_recorded.sum())
    info = {"event_id": ev, "player_name": m.player_name, "from": m.from_club_name, "to": m.to_club_name,
            "window_start": m.window_start, "window_end": m.window_end, "rows": len(E),
            "bench_rows": int(E.role.eq("substitute").sum()), "sum_of_rows": total,
            "master_minutes": int(m.minutes), "available_games": int(m.available_games),
            "available_minutes": int(m.available_minutes), "minutes_share": float(m.minutes_share)}
    assert total == info["master_minutes"], (total, info["master_minutes"])
    assert info["available_minutes"] == 90 * info["available_games"]
    assert abs(info["minutes_share"] - total / info["available_minutes"]) < 1e-9
    return E, info


def readme(S: pd.DataFrame, info: dict, n_bench_main: int) -> str:
    sc = S.groupby("scenario", sort=False).size()
    lines = [
        "# Substitute minutes: sample for a manual check",
        "",
        f"Generated by `src/analysis/q3_manual_check_sample.py` (seed {SEED}). Drawn from the "
        f"{n_bench_main:,} bench appearances in the calendar-first main regression sample.",
        "",
        "## What to check",
        "",
        "Open each `match_report_url` in a browser (by hand: no automated access) and read the line-ups and "
        "substitutions. For the player, note the minute they came on (and went off or were sent off, if they did), "
        "work out the minutes, and compare with `minutes_recorded`, the value that enters the playing-time share. "
        "For a check that does not rely on Transfermarkt, use another match report (BBC, ESPN, the club or "
        "league site) for the same game.",
        "",
        "Rule used in the code: expected minutes = (minute off, minute sent off, or 90) minus (minute on).",
        "Transfermarkt conventions that produce a difference of 1 and are not errors:",
        "- a half-time substitution is shown at 46' but credited 45 minutes;",
        "- a stoppage-time substitution is shown at 90' (the +3 is dropped) and credited 1 minute;",
        "- minutes are capped at 90; stoppage time is never counted.",
        "",
        "## Sample 1: bench appearances (`bench_appearances_sample.csv`)",
        "",
        "| Scenario | Rows |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in sc.items()],
        "",
        "Status in the automatic check: " + ", ".join(f"{k} {v}" for k, v in S.minutes_status.value_counts().items())
        + ".",
        "",
        "## Sample 2: one whole loan, match by match (`worked_episode_matches.csv`)",
        "",
        f"{info['player_name']}, on loan from {info['from']} to {info['to']}, window {info['window_start']} to "
        f"{info['window_end']}: {info['rows']} league appearances, {info['bench_rows']} of them off the bench.",
        "",
        f"- Sum of the match rows: {info['sum_of_rows']:,} minutes; stored `minutes` in the master: "
        f"{info['master_minutes']:,}.",
        f"- Club league games available in the window: {info['available_games']}, so available minutes = 90 x "
        f"{info['available_games']} = {info['available_minutes']:,}.",
        f"- `minutes_share` = {info['sum_of_rows']:,} / {info['available_minutes']:,} = "
        f"{info['minutes_share']:.4f}, as stored.",
        "",
        "Checking a few of these matches by hand confirms the inputs; the sum and share are asserted by the script.",
        "",
        "Fill-in columns are left blank in both files: " + ", ".join(f"`{c}`" for c in FILL_IN) + ".",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    main_s, A = load()
    S = tidy(draw_bench(A))
    E, info = worked_episode(main_s, A)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT_DIR / "bench_appearances_sample.csv", index=False)
    E.to_csv(OUT_DIR / "worked_episode_matches.csv", index=False)
    n_bench = int(A.drop_duplicates(["game_id", "player_id"]).role.eq("substitute").sum())
    (OUT_DIR / "README.md").write_text(readme(S, info, n_bench))
    print(S[["check_id", "scenario", "player_name", "club", "date", "opponent", "minute_on_in_events",
             "minute_off_in_events", "minutes_recorded", "minutes_expected_from_events", "minutes_status"]]
          .to_string(index=False))
    print(info)


if __name__ == "__main__":
    main()
