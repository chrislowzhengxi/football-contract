"""November 2026 professor feedback (Freund, Fernandes): new master variables. Computation only, no file writes.

Called by `data_lineage._master_table` (`extend`) so that the master is rebuilt in one pass. Every new column is
appended after the 150 existing columns; no existing column is changed. The QA reports are written separately by
`feedback_nov2026_report.py`.

    1. Departure from the receiving club (Freund): next recorded move, days at the club, whether the player left before
       the club's first league game, stayed through the season, and a permanent-arrival flag.
    2. Quarantine counted as unavailability (Freund): a v2 injury adjustment next to the archived v1 columns.
    3. Substitute minutes (Fernandes): every league appearance in a reliable window checked against line-ups and
       substitution / card events; contradicted records flagged and removed from a validated outcome (never edited).
    4. League tiers (Fernandes): season-specific league and tier of both clubs (`league_tiers.py` lookup).
    5. Prior-season playing time (Fernandes): league minutes share in the season before the move, plain and
       injury-adjusted (v2 rule), aggregated over every covered club spell.
    6. Origin-club squad value (Fernandes): the existing receiving-club rule applied to the origin club.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import playing_time_injury_adjusted as ia
from . import playing_time_join as ptj
from . import playing_time_regression as reg
from .loan_episodes import ROOT, _sorted

PERM_TYPES = ("permanent_transfer", "free_transfer", "undisclosed_transfer", "no_fee_shown")
MIN_TOL = 2                      # |recorded - expected| <= 2 minutes: Transfermarkt labelling conventions
FULL_GAME = 90
QUARANTINE = "quarantine"        # injury_reason_clean value counted as unavailability under rule v2


# ---------------------------------------------------------------------------
# 1. Departure from the receiving club
# ---------------------------------------------------------------------------

DEPARTURE_TYPES = {
    "loan_out": "loaned out by the receiving club",
    "permanent_transfer": "transferred permanently (paid, free, undisclosed or no fee shown)",
    "loan_return": "returned from loan (end-of-loan row)",
    "released_or_retired": "moved to a placeholder (Without Club, Retired, Career break, Ban, Unknown)",
    "unknown_type": "next row has type 'unknown'",
    "inconsistent_history": "the next recorded move does not leave the receiving club's organisation (a move is missing)",
    "scheduled_not_realised": "the next recorded move is still scheduled (Transfermarkt future transfer)",
    "none_recorded": "no later movement in the player's history",
}
PRESENCE = {
    "left_before_first_league_game": "left before the receiving club's first league game after arrival",
    "left_during_season": "left after the first and on or before the last league game of the arrival season",
    "stayed_through_season_end": "still at the club after its last league game of the arrival season",
    "unknown_next_move_inconsistent": "unknown: the next recorded move does not leave this club",
    "unknown_season_not_finished": "unknown: history captured before the decisive league game",
    "unknown_no_league_schedule": "unknown: receiving club not in a league of the snapshot in its arrival season",
    "not_applicable_move_scheduled": "not applicable: the move itself was still scheduled at capture",
}


def next_departures(u, E: pd.DataFrame) -> pd.DataFrame:
    """The first later substantive Stage 1C row of the player (same skip rule as playing_time_join: internal
    registrations and draft/other rows are skipped), classified relative to the receiving club."""
    d = _sorted(u.rows).reset_index(drop=True)
    pid = d.player_id.to_numpy()
    skip = d.is_internal_move.to_numpy().astype(bool) | np.isin(d.transfer_type_normalized.to_numpy(), ptj.SKIP_TYPES)
    pos = pd.Series(np.arange(len(d)), index=d.event_id.to_numpy())
    nxt = []
    for eid, p in zip(E.event_id, E.player_id):
        j = pos[eid] + 1
        while j < len(d) and pid[j] == p and skip[j]:
            j += 1
        nxt.append(j if (j < len(d) and pid[j] == p) else -1)
    nxt = np.array(nxt)
    has = nxt >= 0
    X = d.iloc[np.where(has, nxt, 0)].reset_index(drop=True)
    me = d.iloc[pos[E.event_id].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({"event_id": E.event_id.to_numpy()})
    for c, s in (("next_departure_event_id", "event_id"), ("next_departure_date", "_date"),
                 ("next_departure_to_club_id", "to_club_id"), ("next_departure_to_club_name", "to_club_name")):
        out[c] = X[s].where(has)
    t = X.transfer_type_normalized
    same_org = (X._from_org.astype(str) == me._to_org.astype(str)) | (X.from_club_id == me.to_club_id)
    typ = np.select(
        [~has, X.transfermarkt_future_transfer.astype(bool), ~same_org,
         X.involves_placeholder_club.astype(bool) & ~t.eq("loan_return"), t.eq("loan"), t.isin(PERM_TYPES),
         t.eq("loan_return")],
        ["none_recorded", "scheduled_not_realised", "inconsistent_history", "released_or_retired", "loan_out",
         "permanent_transfer", "loan_return"], default="unknown_type")
    out["next_departure_type"] = typ
    realised = out.next_departure_type.isin(["loan_out", "permanent_transfer", "loan_return", "released_or_retired",
                                             "unknown_type"])
    known = realised & out.next_departure_type.ne("unknown_type")
    out["next_departure_realised"] = realised
    out["days_at_receiving_club"] = ((out.next_departure_date - me._date).dt.days).where(realised).astype("Int64")
    out["next_departure_is_loan"] = out.next_departure_type.eq("loan_out").astype("boolean").where(known)
    out["next_departure_is_permanent"] = out.next_departure_type.eq("permanent_transfer").astype("boolean").where(known)
    out["next_departure_date"] = out.next_departure_date.where(out.next_departure_type.ne("none_recorded"))
    return out


def league_sides(con) -> pd.DataFrame:
    """Every domestic-league club-game of the snapshot (all 31 leagues), with its appearance-row count."""
    S = con.sql("""with g as (select game_id, competition_id, cast(season as int) season, date, home_club_id,
                                     away_club_id from games where competition_type = 'domestic_league'),
                        r as (select cast(game_id as varchar) game_id, player_club_id club_id, count(*) n_rows
                              from appearances group by 1, 2),
                        s as (select game_id, competition_id, season, date, home_club_id club_id from g
                              union all select game_id, competition_id, season, date, away_club_id from g)
                   select s.*, coalesce(r.n_rows, 0) n_rows
                   from s left join r on r.game_id = s.game_id and r.club_id = s.club_id""").df()
    S = S.rename(columns={"n_rows": "rows"})
    S["date"] = pd.to_datetime(S.date)
    S["club_id"] = S.club_id.astype("int64")
    S["complete"] = S.rows >= ptj.COMPLETE_ROWS
    return S


def arrival_schedule(E: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    """Receiving club's league season of arrival, by playing_time_join.arrival_season's rule extended to every league
    of the snapshot; first league game on or after the move and last league game of that season."""
    league_end = sides.groupby(["competition_id", "season"]).date.max()
    club = sides.groupby(["club_id", "season"]).competition_id.first()
    by_cs = {k: g.date.sort_values().to_numpy() for k, g in sides.groupby(["club_id", "season"])}

    def prev_end(league, s):
        if (league, s - 1) in league_end.index:
            return league_end[(league, s - 1)]
        return league_end[(league, s)] - pd.DateOffset(years=1)

    rows = []
    for eid, b, t in zip(E.event_id, E.b_club_id, E.move_date):
        rec = {"event_id": eid, "sched_season": np.nan, "sched_league": None, "first_game": pd.NaT, "last_game": pd.NaT}
        for s in (t.year - 1, t.year):
            league = club.get((b, s))
            if league is not None and prev_end(league, s) < t <= league_end[(league, s)]:
                g = by_cs[(b, s)]
                after = g[g >= np.datetime64(t)]
                rec.update({"sched_season": s, "sched_league": league, "last_game": g.max(),
                            "first_game": after.min() if len(after) else pd.NaT})
                break
        rows.append(rec)
    return pd.DataFrame(rows)


def departure_flags(u, E: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    """One row per playing-time candidate episode (loan or permanent)."""
    D = next_departures(u, E)
    A = arrival_schedule(E, sides)
    D = D.merge(A, on="event_id", validate="one_to_one")
    Ek = E.set_index("event_id")
    cap = D.event_id.map(Ek.history_captured_through)
    sched_move = D.event_id.map(Ek.scheduled).astype(bool)
    dep = D.next_departure_date.where(D.next_departure_realised)
    inconsistent = D.next_departure_type.eq("inconsistent_history")
    has_sched = D.sched_season.notna()
    # left before the first league game: departure on or before that game's date (the window ends the day before)
    first = D.first_game
    lb = pd.Series(pd.NA, index=D.index, dtype="boolean")
    ok = has_sched & first.notna() & ~inconsistent & ~sched_move
    lb[ok & dep.notna()] = (dep <= first)[ok & dep.notna()]
    lb[ok & dep.isna() & (cap >= first)] = False
    st = pd.Series(pd.NA, index=D.index, dtype="boolean")
    ok2 = has_sched & ~inconsistent & ~sched_move
    last = D.last_game
    st[ok2 & dep.notna()] = (dep > last)[ok2 & dep.notna()]
    st[ok2 & dep.isna() & (cap >= last)] = True
    # no league game left after arrival in that season: the first game is not observed in the arrival season
    lb[ok2 & first.isna() & dep.notna() & (dep <= last)] = pd.NA
    pres = np.select(
        [sched_move, ~has_sched, inconsistent, lb.eq(True).fillna(False).astype(bool),
         st.eq(True).fillna(False).astype(bool), st.eq(False).fillna(False).astype(bool)],
        ["not_applicable_move_scheduled", "unknown_no_league_schedule", "unknown_next_move_inconsistent",
         "left_before_first_league_game", "stayed_through_season_end", "left_during_season"],
        default="unknown_season_not_finished")
    kind = D.event_id.map(Ek.kind)
    out = D[["event_id", "next_departure_event_id", "next_departure_date", "next_departure_type",
             "next_departure_to_club_id", "next_departure_to_club_name", "days_at_receiving_club",
             "next_departure_is_loan", "next_departure_is_permanent"]].copy()
    out["receiving_first_league_game_date"] = first
    out["receiving_last_league_game_date"] = last
    out["left_before_first_receiving_league_game"] = lb
    out["stayed_through_arrival_season_end"] = st
    out["arrival_season_presence"] = pres
    out["perm_moved_on_before_first_league_game"] = lb.where(kind.eq("permanent").to_numpy())
    out["schedule_league"] = D.sched_league
    out["schedule_season"] = D.sched_season
    return out


# ---------------------------------------------------------------------------
# 3. Substitute minutes: appearance-level validation against line-ups and events
# ---------------------------------------------------------------------------

ROW_STATUS = {
    "verified_exact": "recorded minutes equal the minutes implied by line-up and events",
    "verified_convention": f"within {MIN_TOL} minutes (Transfermarkt conventions: a 46' half-time change counts 45, a "
                           "stoppage-time change counts 1, minute labels truncate stoppage time)",
    "verified_shortened_game": "game ended early (every player's minutes capped below 90); recorded minutes agree "
                               "with the game's length",
    "unverifiable_no_lineup": "no line-up recorded for this club-game (all 2012/13 games and a few others)",
    "unverifiable_not_in_lineup": "club line-up exists but does not list the player",
    "unverifiable_missing_entry_event": "listed as a substitute with no substitution-in event",
    "unverifiable_event_minute_unknown": "a substitution or card event has minute -1 (unknown)",
    "unverifiable_lineup_event_conflict": "line-up says starter, events say he came on (one source is wrong)",
    "contradicted": f"recorded minutes differ by more than {MIN_TOL} from the line-up and events, not explained by "
                    "a convention or a shortened game",
    "impossible_over_90": "more than 90 minutes in a league game",
}
CONTRADICTED = ("contradicted", "impossible_over_90")


def league_minutes_rows(con) -> pd.DataFrame:
    """Every domestic-league appearance row of the snapshot with its line-up type, substitution and red-card events
    and the game's maximum recorded minutes (any player, either club), classified by `classify_minutes`."""
    A = con.sql("""
    with app as (
      select a.game_id, a.player_id, a.player_club_id club_id, a.date, a.minutes_played
      from appearances a join games g on g.game_id = cast(a.game_id as varchar)
      where g.competition_type = 'domestic_league'),
    gm as (select game_id, max(minutes_played) game_max_minutes from appearances group by 1),
    lu as (select game_id, player_id, string_agg(distinct type, '|') lineup_type from game_lineups group by 1, 2),
    glu as (select game_id, club_id, count(*) club_lineup_rows from game_lineups group by 1, 2),
    ev as (select cast(game_id as int) game_id, type, minute, player_id, player_in_id, description from game_events
           where type in ('Substitutions', 'Cards')),
    sin as (select game_id, player_in_id player_id, min(minute) m_in from ev
            where type = 'Substitutions' and player_in_id is not null group by 1, 2),
    sout as (select game_id, player_id, max(minute) m_out from ev where type = 'Substitutions' group by 1, 2),
    red as (select game_id, player_id, min(minute) m_red from ev
            where type = 'Cards' and (description ilike '%red card%' or description ilike '%second yellow%')
            group by 1, 2)
    select app.*, gm.game_max_minutes, lu.lineup_type, glu.club_lineup_rows, sin.m_in, sout.m_out, red.m_red
    from app left join gm using (game_id) left join lu using (game_id, player_id) left join glu using (game_id, club_id)
         left join sin using (game_id, player_id) left join sout using (game_id, player_id)
         left join red using (game_id, player_id)""").df()
    A["date"] = pd.to_datetime(A.date)
    A = A.astype({"game_id": "int64", "player_id": "int64", "club_id": "int64"})
    return classify_minutes(A)


def window_rows(Aall: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    """Appearance rows of the player for the club inside each window. W: event_id, player_id, club_id, window_start,
    window_end."""
    W = W.astype({"player_id": "int64", "club_id": "int64"})
    X = W.merge(Aall, on=["player_id", "club_id"])
    X = X[(X.date >= pd.to_datetime(X.window_start)) & (X.date <= pd.to_datetime(X.window_end))]
    return X.drop(columns=["window_start", "window_end"]).reset_index(drop=True)


def classify_minutes(A: pd.DataFrame) -> pd.DataFrame:
    """Expected minutes from line-up and events, and a status per appearance row (ROW_STATUS)."""
    A = A.copy()
    for c in ("m_in", "m_out", "m_red", "club_lineup_rows", "game_max_minutes", "minutes_played"):
        A[c] = pd.to_numeric(A[c], errors="coerce").astype(float)
    starter = A.lineup_type.eq("starting_lineup")
    has_in = A.m_in.notna()
    A["role"] = np.select([starter & has_in, starter, has_in, A.lineup_type.eq("substitutes"),
                           A.lineup_type.isna() & A.club_lineup_rows.notna()],
                          ["lineup_event_conflict", "starter", "substitute", "substitute_without_entry_event",
                           "not_in_lineup"], "no_lineup")
    start = np.where(A.role.eq("substitute"), A.m_in, 0.0)
    end = np.full(len(A), float(FULL_GAME))
    for c in ("m_out", "m_red"):
        hit = A[c].notna() & (A[c] >= start) & (A[c] >= 0)
        end = np.where(hit, np.minimum(end, A[c]), end)
    A["expected_minutes"] = np.where(A.role.isin(["starter", "substitute"]), end - start, np.nan)
    capped = np.minimum(end, A.game_max_minutes.fillna(FULL_GAME)) - start
    A["expected_minutes_game_length"] = np.where(A.role.isin(["starter", "substitute"]), capped, np.nan)
    A["minutes_difference"] = A.minutes_played - A.expected_minutes
    unknown_minute = (A.m_in.eq(-1) | A.m_out.eq(-1) | A.m_red.eq(-1))
    d, dg = A.minutes_difference.abs(), (A.minutes_played - A.expected_minutes_game_length).abs()
    A["minutes_status"] = np.select(
        [A.minutes_played > FULL_GAME, A.role.eq("no_lineup"), A.role.eq("not_in_lineup"),
         A.role.eq("substitute_without_entry_event"), A.role.eq("lineup_event_conflict"), unknown_minute,
         d.eq(0), d.le(MIN_TOL), A.game_max_minutes.lt(FULL_GAME) & dg.le(MIN_TOL)],
        ["impossible_over_90", "unverifiable_no_lineup", "unverifiable_not_in_lineup",
         "unverifiable_missing_entry_event", "unverifiable_lineup_event_conflict", "unverifiable_event_minute_unknown",
         "verified_exact", "verified_convention", "verified_shortened_game"], "contradicted")
    return A


def minutes_validation(A: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
    """Episode-level summary over the appearance rows A (one row per event_id in `events`)."""
    st = A.minutes_status
    g = A.assign(ver=st.str.startswith("verified"), unv=st.str.startswith("unverifiable"),
                 con=st.isin(CONTRADICTED)).groupby("event_id")
    out = pd.DataFrame({"event_id": events.to_numpy()}).set_index("event_id")
    out["pt_minutes_rows_verified"] = g.ver.sum()
    out["pt_minutes_rows_unverifiable"] = g.unv.sum()
    out["pt_minutes_rows_contradicted"] = g.con.sum()
    out = out.fillna(0).astype("Int64")
    out["pt_minutes_validation_status"] = np.select(
        [out.pt_minutes_rows_contradicted.gt(0), out.pt_minutes_rows_unverifiable.gt(0),
         out.pt_minutes_rows_verified.gt(0)],
        ["contradicted_record", "verified_except_unverifiable_rows", "all_rows_verified"],
        "no_appearances_to_check")
    return out.reset_index()


# ---------------------------------------------------------------------------
# 2. Quarantine counted as unavailability (injury rule v2)
# ---------------------------------------------------------------------------

def injury_rows_v2(INJ: pd.DataFrame) -> pd.DataFrame:
    """Rule v2: the v1 rule plus 'Quarantine' (isolation: unavailable to the coach, not necessarily infected).
    'Fitness' and 'Rest' still do not count; the strict rule (confirmed injury or illness) is unchanged."""
    N = INJ.copy()
    q = N.injury_reason_clean.eq(QUARANTINE)
    N["counts_primary"] = N.counts_primary.astype(bool) | q
    N.loc[q, "category"] = "quarantine / isolation (counted as unavailability from rule v2)"
    return N


def player_status(INJ: pd.DataFrame, players) -> pd.Series:
    """Exact-id join status of injury_adjusted.compute: A rows, B profile only, C absent, D duplicate profile id."""
    prof = pd.read_csv(ia.PROF_FILE, dtype=str, keep_default_na=False, usecols=["player_id"]).player_id.astype("int64")
    dup, rows, lake = set(prof[prof.duplicated()]), set(INJ.player_id), set(prof)
    return pd.Series({int(p): ("D" if p in dup else "A" if p in rows else "B" if p in lake else "C")
                      for p in pd.Series(players).astype("int64").unique()})


def injury_v2(tb: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the injury adjustment of injury_adjusted.compute on the same window games with rule v2.
    Returns episode outcomes (event_id ...) and the game-level flags."""
    E1, G1, INJ = tb["E"], tb["G"], tb["INJ"]
    N2 = injury_rows_v2(INJ)
    Sp2 = ia.spells(N2, "primary")
    W = G1[["event_id", "player_id", "game_id", "date", "played"]]
    G2, _ = ia.game_flags(W, Sp2)
    S = E1[["event_id", "player_id", "kind", "minutes", "window_start", "window_end"]].assign(
        minutes_share=E1.current_share)
    E2 = ia.episode_outcomes(S, G2, Sp2)
    status = E1.groupby("player_id").status.first()
    E2["data_quality_flag"] = ia.usability(E2, N2, status).values
    E2["usable"] = E2.data_quality_flag.eq("usable")
    # games newly unavailable because of a quarantine spell (and no other medical spell)
    G2 = G2.merge(G1[["event_id", "game_id", "medical_unavailable"]].rename(columns={"medical_unavailable": "unavail_v1"}),
                  on=["event_id", "game_id"], validate="one_to_one")
    G2["quarantine_only"] = G2.medical_unavailable & ~G2.unavail_v1
    assert not (G2.unavail_v1 & ~G2.medical_unavailable).any(), "rule v2 must only add unavailable games"
    E2["quarantine_unavailable_games"] = E2.event_id.map(G2.groupby("event_id").quarantine_only.sum()).fillna(0).astype(int)
    return E2, G2


# ---------------------------------------------------------------------------
# 6. Origin-club squad value
# ---------------------------------------------------------------------------

def club_squad_value(E: pd.DataFrame, pv: pd.DataFrame, club_col: str) -> pd.DataFrame:
    """playing_time_regression.squad_value (unchanged rule) for the club in `club_col`, plus the value including the
    focal player and the oldest valuation used. E: event_id, player_id, move_date, club_col."""
    X = E.rename(columns={club_col: "b_club_id"})[["event_id", "player_id", "b_club_id", "move_date"]].dropna()
    X = X.astype({"player_id": "int64", "b_club_id": "int64"})
    sv = reg.squad_value(X, pv)
    # focal player's own latest pre-move valuation, when it lists this club within the look-back
    pvv = reg.player_value(X, pv).set_index("event_id")
    Xk = X.set_index("event_id")
    lb = pd.Timedelta(days=reg.LOOKBACK_DAYS)
    own = (pvv.player_value_club_id.reindex(Xk.index) == Xk.b_club_id) & \
          (pvv.player_value_date.reindex(Xk.index) >= Xk.move_date - lb)
    sv = sv.set_index("event_id")
    assert (sv.focal_in_squad == own.reindex(sv.index)).all(), "focal membership disagrees with squad_value"
    sv["value_incl_focal"] = (sv.squad_value + pvv.player_value.reindex(sv.index).where(sv.focal_in_squad, 0)).where(
        sv.squad_value.notna())
    return sv.reset_index()


# ---------------------------------------------------------------------------
# 5. Prior-season playing time
# ---------------------------------------------------------------------------

PLACEHOLDER_CLUB_IDS = (515, 2113, 75, 2077, 123)   # Without Club, Career break, Unknown, Ban, Retired
PRIOR_REASONS = {
    "": "reliable",
    "no_reference_season": "move season unknown",
    "prior_season_outside_snapshot": "prior season outside the snapshot's league seasons (2012/13-2025/26)",
    "no_registration_history": "no recorded club for the player in the prior season",
    "no_covered_club": "no club of the player played in a league of the snapshot that season (lower division, "
                       "youth side, other country, or league not yet in the snapshot)",
    "no_league_games_before_move": "covered club, but no league game between the spell start and the move",
    "incomplete_club_games": "some league game of a covered club in the prior-season window lacks a complete "
                             "appearance record",
    "appearances_for_unregistered_club": "the player appears in a league game for a club he was not registered "
                                         "with at that date (a move or promotion is missing from his history)",
    "contradicted_minutes": "a league appearance's minutes are contradicted by line-up and events",
}


def registration_spells(u) -> pd.DataFrame:
    """Club of registration over time from every realised Stage 1C row (internal registrations included: they change
    the registered team; draft/other and scheduled rows skipped). Before a player's first row he is at that row's
    origin club. Placeholder destinations (Without Club, Retired, ...) give no club."""
    d = _sorted(u.rows)
    d = d[~d.transfermarkt_future_transfer.astype(bool) & ~d.transfer_type_normalized.isin(["other"])].copy()
    d["start"] = d._date
    d["end"] = d.groupby("player_id")._date.shift(-1) - pd.Timedelta(days=1)
    first = d.groupby("player_id").head(1)
    pre = pd.DataFrame({"player_id": first.player_id, "club_id": first.from_club_id,
                        "start": pd.Timestamp("1900-01-01"), "end": first._date - pd.Timedelta(days=1)})
    S = pd.concat([pre, d[["player_id", "to_club_id", "start", "end"]].rename(columns={"to_club_id": "club_id"})])
    S["end"] = S.end.fillna(pd.Timestamp("2100-01-01"))
    S = S[S.end >= S.start]
    S["club_id"] = S.club_id.astype("int64")
    S["placeholder"] = S.club_id.isin(PLACEHOLDER_CLUB_IDS)
    return S.reset_index(drop=True)


def reference_season(E: pd.DataFrame) -> pd.Series:
    """Season of the move: the receiving club's league arrival season where known (the current playing-time rule),
    otherwise the start year of the Transfermarkt season label."""
    tm = pd.to_numeric(E.transfer_season.astype(str).str[:4], errors="coerce")
    return E.arrival_season.astype("Float64").fillna(tm).astype("Int64")


def league_appearances(con) -> pd.DataFrame:
    A = con.sql("""select a.game_id, a.player_id, a.player_club_id club_id, a.date, a.minutes_played
                   from appearances a join games g on g.game_id = cast(a.game_id as varchar)
                   where g.competition_type = 'domestic_league'""").df()
    A["date"] = pd.to_datetime(A.date)
    return A.astype({"game_id": "int64", "player_id": "int64", "club_id": "int64"})


def prior_season(E: pd.DataFrame, spells: pd.DataFrame, sides: pd.DataFrame, apps: pd.DataFrame,
                 Arows: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prior-season league playing time for every episode in E (event_id, player_id, move_date, ref_season).
    The window of season p = ref_season - 1 is 1 July p to 30 June p+1, cut at the day before the move. Games are the
    league games of every covered club the player was registered with, while registered; minutes are his league
    minutes for that club in those games. Returns (episode table, game table)."""
    seasons = set(sides.season.unique())
    X = E[["event_id", "player_id", "move_date", "ref_season"]].copy()
    X["p"] = X.ref_season - 1
    X["lo"] = pd.to_datetime(X.p.map(lambda y: f"{int(y)}-07-01" if pd.notna(y) else None))
    X["hi"] = pd.concat([X.lo + pd.DateOffset(years=1) - pd.Timedelta(days=1),
                         X.move_date - pd.Timedelta(days=1)], axis=1).min(axis=1).where(X.lo.notna())
    ok = X.p.notna() & X.p.isin(seasons)
    Y = X[ok].merge(spells, on="player_id")
    Y = Y[(Y.start <= Y.hi) & (Y.end >= Y.lo)].copy()
    Y["a"], Y["b"] = Y[["start", "lo"]].max(axis=1), Y[["end", "hi"]].min(axis=1)
    Y["days"] = (Y.b - Y.a).dt.days + 1
    cs = sides.groupby(["club_id", "season"]).size().rename("n").reset_index()
    Y = Y.merge(cs.rename(columns={"season": "p"}), on=["club_id", "p"], how="left")
    Y["covered"] = Y.n.notna() & ~Y.placeholder
    club_days = Y[~Y.placeholder].groupby("event_id").days.sum()
    cov_days = Y[Y.covered].groupby("event_id").days.sum()
    n_clubs = Y[~Y.placeholder].groupby("event_id").club_id.nunique()
    n_cov = Y[Y.covered].groupby("event_id").club_id.nunique()
    # games of covered clubs inside the registration spells
    C = Y[Y.covered][["event_id", "player_id", "club_id", "p", "a", "b"]]
    G = C.merge(sides[["game_id", "season", "date", "club_id", "complete", "competition_id"]].rename(columns={"season": "p"}),
                on=["club_id", "p"])
    G = G[(G.date >= G.a) & (G.date <= G.b)].drop(columns=["a", "b"])
    G["game_id"] = G.game_id.astype("int64")
    pm = apps.set_index(["game_id", "player_id", "club_id"]).minutes_played
    k = pd.MultiIndex.from_arrays([G.game_id, G.player_id, G.club_id])
    G["minutes"] = pm.reindex(k).to_numpy()
    G["played"] = G.minutes.notna()
    assert not G.duplicated(["event_id", "game_id"]).any(), "a game counted twice in one prior season"
    # league appearances in season p (before the move) for a club he was not registered with at that date
    reg_ = apps[["game_id", "player_id", "club_id", "date"]].merge(
        spells[["player_id", "club_id", "start", "end"]], on=["player_id", "club_id"], how="left")
    reg_["ok"] = (reg_.date >= reg_.start) & (reg_.date <= reg_.end)
    unreg = reg_.groupby(["game_id", "player_id", "date"]).ok.any().reset_index()
    unreg = unreg[~unreg.ok]
    Ap = unreg.merge(X[ok][["event_id", "player_id", "lo", "hi"]], on="player_id")
    other = Ap[(Ap.date >= Ap.lo) & (Ap.date <= Ap.hi)].groupby("event_id").size()
    # minute-level QA on the prior games (same validation as the current window)
    contra = pd.Series(dtype=float)
    if Arows is not None:
        bad = Arows[Arows.minutes_status.isin(CONTRADICTED)][["game_id", "player_id", "club_id"]].drop_duplicates()
        hit = G[G.played].merge(bad.astype("int64"), on=["game_id", "player_id", "club_id"])
        contra = hit.groupby("event_id").size()
    agg = G.groupby("event_id").agg(prior_league_games=("game_id", "size"), prior_complete_games=("complete", "sum"),
                                    prior_appearances=("played", "sum"), prior_minutes=("minutes", "sum"),
                                    prior_first_game=("date", "min"), prior_last_game=("date", "max"),
                                    prior_leagues=("competition_id", lambda s: "|".join(sorted(set(s)))))
    O = X[["event_id", "p"]].rename(columns={"p": "prior_season"}).set_index("event_id").join(agg)
    O["prior_club_days"] = club_days.reindex(O.index).fillna(0).astype(int)
    O["prior_covered_days"] = cov_days.reindex(O.index).fillna(0).astype(int)
    O["prior_clubs"] = n_clubs.reindex(O.index).fillna(0).astype(int)
    O["prior_covered_clubs"] = n_cov.reindex(O.index).fillna(0).astype(int)
    O["prior_unregistered_club_apps"] = other.reindex(O.index).fillna(0).astype(int)
    O["prior_contradicted_rows"] = contra.reindex(O.index).fillna(0).astype(int)
    has_p = X.set_index("event_id").p.notna()
    in_snap = X.set_index("event_id").p.isin(seasons)
    games = O.prior_league_games.fillna(0)
    O["prior_exclusion_reason"] = np.select(
        [~has_p, ~in_snap, O.prior_club_days.eq(0), O.prior_covered_clubs.eq(0), games.eq(0),
         O.prior_complete_games < games, O.prior_unregistered_club_apps.gt(0), O.prior_contradicted_rows.gt(0)],
        ["no_reference_season", "prior_season_outside_snapshot", "no_registration_history", "no_covered_club",
         "no_league_games_before_move", "incomplete_club_games", "appearances_for_unregistered_club",
         "contradicted_minutes"], "")
    O["prior_reliable"] = O.prior_exclusion_reason.eq("")
    for c in ("prior_league_games", "prior_appearances", "prior_minutes"):
        O[c] = O[c].astype("Float64").astype("Int64")
    O["prior_minutes_share"] = (O.prior_minutes / (FULL_GAME * O.prior_league_games)).astype(float).where(O.prior_reliable)
    O["prior_covered_share_of_days"] = (O.prior_covered_days / O.prior_club_days.where(O.prior_club_days > 0))
    O["prior_season_coverage"] = np.select(
        [~O.prior_reliable, O.prior_covered_share_of_days.ge(0.999)],
        ["not_reliable", "whole_season_at_covered_clubs"], "part_of_season_at_covered_clubs")
    return O.reset_index(), G


def prior_injury(O: pd.DataFrame, G: pd.DataFrame, INJ2: pd.DataFrame, status: pd.Series) -> pd.DataFrame:
    """Injury-adjusted prior share with rule v2 and the usability rules of injury_adjusted (window = first to last
    prior-season league game)."""
    R = O[O.prior_reliable][["event_id", "prior_league_games", "prior_minutes", "prior_first_game", "prior_last_game"]]
    Gr = G[G.event_id.isin(R.event_id)][["event_id", "player_id", "game_id", "date", "played"]]
    Gf, _ = ia.game_flags(Gr, ia.spells(INJ2, "primary"))
    inj = Gf.groupby("event_id").medical_unavailable.sum()
    Ek = R.set_index("event_id")
    P = pd.DataFrame({"event_id": R.event_id.to_numpy(),
                      "player_id": R.event_id.map(G.groupby("event_id").player_id.first()).to_numpy(),
                      "window_start": R.prior_first_game.to_numpy(), "window_end": R.prior_last_game.to_numpy()})
    P["prior_injury_unavailable_games"] = P.event_id.map(inj).fillna(0).astype(int).to_numpy()
    P["available_games_fit"] = P.event_id.map(Ek.prior_league_games).astype(int).to_numpy() - P.prior_injury_unavailable_games
    st = status.reindex(P.player_id.unique())
    P["prior_injury_data_quality_flag"] = ia.usability(P, INJ2, st).to_numpy()
    usable = P.prior_injury_data_quality_flag.eq("usable")
    P["prior_fit_games"] = P.available_games_fit
    mins = P.event_id.map(Ek.prior_minutes).astype(float)
    P["prior_fit_adjusted_share"] = (mins / (FULL_GAME * P.prior_fit_games.where(P.prior_fit_games > 0))).where(usable)
    return P[["event_id", "prior_injury_unavailable_games", "prior_fit_games", "prior_injury_data_quality_flag",
              "prior_fit_adjusted_share"]]


# ---------------------------------------------------------------------------
# 4. League tiers of both clubs (lookup built by league_tiers.py)
# ---------------------------------------------------------------------------

TIER_LOOKUP = ROOT / "data" / "processed" / "league_tiers" / "club_season_league.csv"
TIER_STATUS = {
    "verified": "club found in a league table of that season (snapshot league games or an external league file)",
    "placeholder_not_a_club": "Without Club, Retired, Career break, Ban or Unknown",
    "youth_or_reserve_side_division_unknown": "youth or reserve side not found in any league table of that season",
    "unknown_season_outside_sources": "no league source covers that season",
    "unknown_division_not_in_sources": "senior club not found in any league table of that season (lower division "
                                       "without a source, other country, or a name that could not be matched)",
}


def tier_lookup() -> pd.DataFrame:
    L = pd.read_csv(TIER_LOOKUP)
    L = L.astype({"club_id": "int64", "season": "int64", "tier": "int64"})
    assert not L.duplicated(["club_id", "season"]).any(), "tier lookup has duplicate club-seasons"
    return L


def league_tiers(M: pd.DataFrame, ref: pd.Series, sides: pd.DataFrame, covered: list[str], youth: set[int],
                 L: pd.DataFrame) -> pd.DataFrame:
    """Season-specific league and tier of the origin and receiving clubs in the reference season of the move, and of
    the origin club in the season before. Unmatched clubs stay blank with a status; nothing is inferred."""
    Lk = L.set_index(["club_id", "season"])
    lo, hi = int(L.season.min()), int(L.season.max())
    top = set(map(tuple, sides[sides.competition_id.isin(covered)][["club_id", "season"]].drop_duplicates().to_numpy()))
    out = pd.DataFrame({"event_id": M.event_id.to_numpy()})
    out["tier_reference_season"] = ref.to_numpy()
    for side, col in (("from", "from_club_id"), ("to", "to_club_id")):
        for suffix, shift in (("at_move", 0), ("prior_season", 1)):
            if side == "to" and suffix == "prior_season":
                continue
            s = (ref - shift).astype("Int64")
            key = pd.MultiIndex.from_arrays([M[col].astype("int64").to_numpy(), s.fillna(-1).astype("int64").to_numpy()])
            hit = Lk.reindex(key)
            name, tier = hit.league_name.to_numpy(), hit.tier.astype("Int64").to_numpy()
            out[f"{side}_league_name_{suffix}"] = name
            out[f"{side}_league_tier_{suffix}"] = pd.array(tier, dtype="Int64")
            if suffix != "at_move":
                continue
            out[f"{side}_league_country_at_move"] = hit.country.to_numpy()
            out[f"{side}_league_tier_source"] = hit.source.to_numpy()
            club = M[col].astype("int64")
            found = pd.Series(hit.league_name.notna().to_numpy())
            inside = s.notna() & s.between(lo, hi)
            out[f"{side}_league_tier_status"] = np.select(
                [found.to_numpy(), club.isin(PLACEHOLDER_CLUB_IDS).to_numpy(), club.isin(youth).to_numpy(),
                 ~inside.to_numpy()],
                ["verified", "placeholder_not_a_club", "youth_or_reserve_side_division_unknown",
                 "unknown_season_outside_sources"], "unknown_division_not_in_sources")
            out[f"{side}_in_covered_top_flight_at_move"] = pd.array(
                [(int(c), int(x)) in top if pd.notna(x) else pd.NA for c, x in zip(club, s)], dtype="boolean")
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def extend(M: pd.DataFrame, u, E: pd.DataFrame, inj_tb: dict, pv: pd.DataFrame) -> pd.DataFrame:
    """Append every November-2026 column to the master (in place order: NEW_COLUMNS of feedback_nov2026_specs)."""
    import duckdb
    from ..config import DEFAULT_DATABASE
    from . import club_organisation as co
    from .feedback_nov2026_specs import NEW_COLUMNS

    n0, cols0 = len(M), list(M.columns)
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        sides = league_sides(con)
        Aall = league_minutes_rows(con)
        apps = league_appearances(con)
    finally:
        con.close()
    ev = M.event_id
    new = {}

    # 1. departure
    D = departure_flags(u, E, sides).set_index("event_id")
    for c in ("next_departure_event_id", "next_departure_date", "next_departure_type", "next_departure_to_club_id",
              "next_departure_to_club_name", "days_at_receiving_club", "next_departure_is_loan",
              "next_departure_is_permanent", "receiving_first_league_game_date", "receiving_last_league_game_date",
              "left_before_first_receiving_league_game", "stayed_through_arrival_season_end", "arrival_season_presence",
              "perm_moved_on_before_first_league_game"):
        new[c] = ev.map(D[c])
    for c in ("next_departure_date", "receiving_first_league_game_date", "receiving_last_league_game_date"):
        new[c] = pd.to_datetime(new[c]).dt.strftime("%Y-%m-%d")
    for c in ("next_departure_to_club_id", "days_at_receiving_club"):
        new[c] = pd.array(new[c], dtype="Int64")
    for c in ("next_departure_is_loan", "next_departure_is_permanent", "left_before_first_receiving_league_game",
              "stayed_through_arrival_season_end", "perm_moved_on_before_first_league_game"):
        new[c] = pd.array(new[c], dtype="boolean")

    # 3. minutes validation (reliable windows)
    rel = M.pt_reliable.astype(bool)
    W = M.loc[rel, ["event_id", "player_id", "to_club_id", "window_start", "window_end"]].rename(
        columns={"to_club_id": "club_id"})
    A = window_rows(Aall, W)
    chk = A.groupby("event_id").minutes_played.sum().reindex(W.event_id).fillna(0)
    assert np.array_equal(chk.to_numpy(), M.loc[rel, "minutes"].to_numpy()), "window minutes do not reproduce"
    V = minutes_validation(A, W.event_id).set_index("event_id")
    for c in ("pt_minutes_rows_verified", "pt_minutes_rows_unverifiable", "pt_minutes_rows_contradicted",
              "pt_minutes_validation_status"):
        new[c] = ev.map(V[c])
        if c != "pt_minutes_validation_status":
            new[c] = pd.array(new[c], dtype="Int64")
    validated = rel & new["pt_minutes_validation_status"].ne("contradicted_record")
    new["pt_minutes_validated"] = validated.astype(bool)
    new["minutes_share_validated"] = M.minutes_share.where(validated)
    new["in_age_model_sample_v2"] = (M.in_age_model_sample & validated).astype(bool)
    new["in_cf_main_model_sample_v2"] = (M.in_cf_main_model_sample & validated).astype(bool)

    # 2. injury rule v2
    E2, _ = injury_v2(inj_tb)
    Ik = E2.set_index("event_id")
    new["quarantine_unavailable_games"] = pd.array(ev.map(Ik.quarantine_unavailable_games), dtype="Int64")
    new["confirmed_injury_games_v2"] = pd.array(ev.map(Ik.confirmed_injury_games), dtype="Int64")
    new["available_games_fit_v2"] = pd.array(ev.map(Ik.available_games_fit), dtype="Int64")
    new["fit_adjusted_share_v2"] = ev.map(Ik.fit_adjusted_share)
    new["injury_data_quality_flag_v2"] = ev.map(Ik.data_quality_flag)
    usable2 = ev.isin(set(E2.event_id[E2.usable]))
    # the injury model sample is the injury-usable part of the main (age) model sample: checked on v1
    assert (M.in_injury_model_sample == (M.in_injury_usable & M.in_age_model_sample)).all()
    new["in_injury_usable_v2"] = usable2.astype(bool)
    new["in_injury_model_sample_v2"] = (usable2 & M.in_age_model_sample & validated).astype(bool)
    new["in_cf_injury_usable_v2"] = (usable2 & M.calendar_first_eligible).astype(bool)
    new["in_cf_injury_model_sample_v2"] = (new["in_injury_model_sample_v2"] & M.calendar_first_eligible).astype(bool)

    # 4. league tiers
    ref = pd.Series(reference_season(M.assign(transfer_season=M.season)).to_numpy(), index=M.index)
    youth = youth_club_ids(M, co.load_map())
    T = league_tiers(M, ref, sides, ptj_covered(sides), youth, tier_lookup()).set_index("event_id")
    for c in T.columns:
        new[c] = ev.map(T[c])
    for c in ("tier_reference_season", "from_league_tier_at_move", "to_league_tier_at_move",
              "from_league_tier_prior_season"):
        new[c] = pd.array(new[c], dtype="Int64")
    for c in ("from_in_covered_top_flight_at_move", "to_in_covered_top_flight_at_move"):
        new[c] = pd.array(new[c], dtype="boolean")

    # 5. prior season
    X = E[["event_id", "player_id", "move_date"]].assign(ref_season=reference_season(E))
    O, Gp = prior_season(X, registration_spells(u), sides, apps, Aall)
    INJ = inj_tb["INJ"]
    P = prior_injury(O, Gp, injury_rows_v2(INJ), player_status(INJ, E.player_id))
    O = O.merge(P, on="event_id", how="left", validate="one_to_one").set_index("event_id")
    for c in ("prior_season", "prior_league_games", "prior_appearances", "prior_minutes", "prior_minutes_share",
              "prior_injury_unavailable_games", "prior_fit_games", "prior_fit_adjusted_share",
              "prior_injury_data_quality_flag", "prior_reliable", "prior_exclusion_reason", "prior_season_coverage",
              "prior_covered_share_of_days", "prior_clubs", "prior_covered_clubs", "prior_leagues",
              "prior_unregistered_club_apps"):
        new[c] = ev.map(O[c])
    rel_p = new["prior_reliable"].eq(True)
    for c in ("prior_league_games", "prior_appearances", "prior_minutes"):
        new[c] = pd.array(new[c].where(rel_p), dtype="Int64")
    for c in ("prior_season", "prior_injury_unavailable_games", "prior_fit_games", "prior_clubs", "prior_covered_clubs",
              "prior_unregistered_club_apps"):
        new[c] = pd.array(new[c], dtype="Int64")
    new["prior_reliable"] = pd.array(new["prior_reliable"], dtype="boolean")
    new["prior_leagues"] = new["prior_leagues"].where(rel_p)
    new["prior_exclusion_reason"] = new["prior_exclusion_reason"].replace("", "none (reliable)")

    # 6. origin and receiving squad value (same rule as the stored receiving value)
    typed = M[M.kind.isin(["loan", "permanent"])][["event_id", "player_id", "transfer_date", "from_club_id",
                                                   "to_club_id"]].assign(move_date=lambda d: pd.to_datetime(d.transfer_date))
    Fv = club_squad_value(typed, pv, "from_club_id").set_index("event_id")
    Tv = club_squad_value(typed, pv, "to_club_id").set_index("event_id")
    new["from_squad_value"] = ev.map(Fv.squad_value)
    new["from_squad_value_incl_player"] = ev.map(Fv.value_incl_focal)
    new["from_squad_players"] = pd.array(ev.map(Fv.squad_players), dtype="Int64")
    new["from_focal_in_squad"] = pd.array(ev.map(Fv.focal_in_squad), dtype="boolean")
    new["from_squad_value_gap_days"] = pd.array(ev.map(Fv.squad_value_gap_days), dtype="Int64")
    new["log_from_squad_value"] = np.log(new["from_squad_value"].where(new["from_squad_value"] > 0))
    new["to_squad_value"] = ev.map(Tv.squad_value)
    stored = M.squad_value.notna()
    assert np.allclose(new["to_squad_value"][stored], M.squad_value[stored]), "receiving squad value changed"
    assert new["to_squad_value"][M.in_revised_sample & ~stored].isna().all()
    new["from_to_squad_value_ratio"] = new["from_squad_value"] / new["to_squad_value"].where(new["to_squad_value"] > 0)
    new["log_from_to_squad_value_ratio"] = np.log(new["from_to_squad_value_ratio"].where(new["from_to_squad_value_ratio"] > 0))

    # 7. combined validation (8 Nov correction): line-up check (pt_reliable_v2) and recorded-minute check together
    full = (validated & M.pt_reliable_v2.astype(bool)).astype(bool)
    new["pt_fully_validated"] = full
    new["in_age_model_sample_validated"] = (new["in_age_model_sample_v2"] & full).astype(bool)
    new["in_cf_main_model_sample_validated"] = (new["in_cf_main_model_sample_v2"] & full).astype(bool)
    new["in_injury_model_sample_validated"] = (new["in_injury_model_sample_v2"] & full).astype(bool)
    new["in_cf_injury_model_sample_validated"] = (new["in_cf_injury_model_sample_v2"] & full).astype(bool)

    X2 = pd.DataFrame(new, index=M.index)
    assert set(X2.columns) == set(NEW_COLUMNS) and len(NEW_COLUMNS) == len(set(NEW_COLUMNS)), \
        set(X2.columns) ^ set(NEW_COLUMNS)
    out = pd.concat([M, X2[NEW_COLUMNS]], axis=1)
    assert len(out) == n0 and list(out.columns[:len(cols0)]) == cols0 and out.event_id.is_unique
    return out


def youth_club_ids(M: pd.DataFrame, MAP: pd.DataFrame) -> set[int]:
    """Youth / reserve sides on either side of a move: the Stage 1C name rule applied to each club name, plus the
    club-organisation map's youth sides, minus the clubs that map rejects as senior clubs (e.g. Willem II)."""
    from ..stage1c.policy import is_youth_or_reserve
    names = pd.concat([M[["from_club_id", "from_club_name"]].set_axis(["id", "name"], axis=1),
                       M[["to_club_id", "to_club_name"]].set_axis(["id", "name"], axis=1)]).drop_duplicates()
    by_name = set(names.loc[names.name.map(is_youth_or_reserve).astype(bool), "id"].astype(int))
    rejected = set(MAP.loc[MAP.confidence.eq("rejected_senior_club"), "child_club_id"].astype(int))
    from . import club_organisation as co
    return (by_name | co.youth_side_ids(MAP)) - rejected


def ptj_covered(sides: pd.DataFrame) -> list[str]:
    """The 14 long-run covered leagues (playing_time_join.league_coverage: appearance rows in every season)."""
    lg = sides.assign(competition_type="domestic_league")
    return ptj.league_coverage(lg.rename(columns={}))[1]
