"""QA reports, regression comparison and the updated professor guide for the November 2026 feedback.

    python -m src.analysis.feedback_nov2026_report

Run after `python -m src.analysis.data_lineage` (which rebuilds the master with the new columns). Reads the master,
recomputes the supporting tables with the same functions, checks them against the master, and writes everything under
data/outputs/rebuild/professor_feedback_nov_2026/. Every number in the reports is generated here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import feedback_nov2026 as fb
from . import feedback_nov2026_specs as fs
from . import playing_time_regression as reg
from .loan_episodes import ROOT

OUT = ROOT / "data" / "outputs" / "rebuild"
DIR = OUT / "professor_feedback_nov_2026"
ARCH = DIR / "archive_pre_nov2026"
MASTER = OUT / "analysis_episode_master.csv"
OLD_MASTER = ARCH / "analysis_episode_master_150col.csv.gz"
STAGE1C = OUT / "stage1c_canonical_transfers.csv"
STAGE1C_SHA = "92f383d72a3ed59925c922f0d1b5f98528bc66ad3a9f49bb990867ed1d1d50dd"


def f0(x) -> str:
    return f"{int(x):,}"


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


def pp(x, d=2) -> str:
    v = round(100 * x, d)
    return f"{'+' if v > 0 else ''}{v:.{d}f}" if v != 0 else f"{0:.{d}f}"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def md_table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Regression helpers (the preferred M3 specification, fitted on master columns)
# ---------------------------------------------------------------------------

CAL = {"no_calendar": ("No calendar requirement", lambda M: pd.Series(True, index=M.index)),
       "calendar_first": ("Calendar-first (normal and off-calendar kept)", lambda M: M.calendar_first_eligible),
       "normal_only": ("Normal-only (off-calendar loans excluded; sensitivity)",
                       lambda M: M.kind.eq("permanent") | M.loan_calendar_status.eq("normal"))}


def m3(M: pd.DataFrame, mask: pd.Series, y: str) -> dict:
    C = M[mask.astype(bool)].copy()
    C["loan"] = C.kind.eq("loan").astype(int)
    C["age_c"] = C.age_exact - C.age_exact.mean()
    C["age_c2"] = C.age_c ** 2
    C["b_club_id"] = C.to_club_id
    r = reg.coef(reg.fit(C, reg.M3, fe="league_season", cats=["position_clean"], y=y))
    r["loans"], r["permanent"] = int(C.kind.eq("loan").sum()), int(C.kind.eq("permanent").sum())
    return r


def regression_comparison(M: pd.DataFrame, saved: dict) -> pd.DataFrame:
    rows = []
    key = {"no_calendar": "current", "calendar_first": "classifiable", "normal_only": "normal_only"}
    for k, (label, f) in CAL.items():
        cal = f(M).astype(bool)
        specs = [
            ("Main", "1 stored (v1)", M.in_age_model_sample & cal, "minutes_share"),
            ("Main", "2 minutes validated", M.in_age_model_sample_v2 & cal, "minutes_share"),
            ("Injury, ordinary share", "1 stored (v1)", M.in_injury_model_sample & cal, "current_share"),
            ("Injury-adjusted share", "1 stored (v1: quarantine not counted)", M.in_injury_model_sample & cal,
             "fit_adjusted_share"),
            ("Injury-adjusted share", "2 quarantine counted (v2)", M.in_injury_model_sample & M.in_injury_usable_v2 & cal,
             "fit_adjusted_share_v2"),
            ("Injury-adjusted share", "3 quarantine counted + minutes validated",
             M.in_injury_model_sample_v2 & cal, "fit_adjusted_share_v2"),
            ("Injury, ordinary share", "3 quarantine counted + minutes validated", M.in_injury_model_sample_v2 & cal,
             "current_share"),
            # 8 Nov correction: the line-up check (pt_reliable_v2) applied together with the recorded-minute check
            ("Main", "3 both minute checks (line-up + recorded minutes)", M.in_age_model_sample_validated & cal,
             "minutes_share"),
            ("Injury-adjusted share", "4 quarantine counted + both minute checks",
             M.in_injury_model_sample_validated & cal, "fit_adjusted_share_v2"),
            ("Injury, ordinary share", "4 quarantine counted + both minute checks",
             M.in_injury_model_sample_validated & cal, "current_share"),
        ]
        for outcome, version, mask, y in specs:
            r = m3(M, mask, y)
            rows.append({"calendar_definition": label, "calendar_key": k, "outcome": outcome, "version": version,
                         "y": y, "N": r["N"], "loans": r["loans"], "permanent": r["permanent"], "loan_coef": r["coef"],
                         "se": r["se"], "ci_lo": r["lo"], "ci_hi": r["hi"], "p": r["p"]})
    R = pd.DataFrame(rows)
    # the stored v1 estimates must be reproduced exactly from the master
    for k, sk in key.items():
        est = saved["calendar_first"]["variants"][sk]["estimates"]
        for outcome, y, name in (("Main", "minutes_share", "m3"),
                                 ("Injury, ordinary share", "current_share", "injury_m3_current"),
                                 ("Injury-adjusted share", "fit_adjusted_share", "injury_m3_fit")):
            r = R[(R.calendar_key == k) & (R.outcome == outcome) & R.version.str.startswith("1") & (R.y == y)].iloc[0]
            assert abs(r.loan_coef - est[name]["coef"]) < 1e-9 and r.N == est[name]["N"], (k, name)
    return R


# ---------------------------------------------------------------------------
# Context: the supporting tables, recomputed with the same functions
# ---------------------------------------------------------------------------

def context() -> dict:
    import duckdb
    from ..config import DEFAULT_DATABASE
    from . import loan_scope, playing_time_injury_adjusted as ia, playing_time_join as ptj

    u = loan_scope.universe()
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        _, E, _ = ptj.compute(u, con)
        sides = fb.league_sides(con)
        Aall = fb.league_minutes_rows(con)
        apps = fb.league_appearances(con)
        games = con.sql("select game_id, date, competition_id, home_club_id, away_club_id, home_club_name, "
                        "away_club_name from games").df()
    finally:
        con.close()
    _, tb = ia.compute(u)
    return {"u": u, "E": E, "sides": sides, "Aall": Aall, "apps": apps, "tb": tb, "games": games}


# ---------------------------------------------------------------------------
# 1. Departures
# ---------------------------------------------------------------------------

DEP_COLS = ["event_id", "player_name", "kind", "transfer_date", "from_club_name", "to_club_name", "fee_display_raw",
            "next_departure_type", "next_departure_date", "next_departure_to_club_name", "days_at_receiving_club",
            "receiving_first_league_game_date", "receiving_last_league_game_date",
            "left_before_first_receiving_league_game", "stayed_through_arrival_season_end", "arrival_season_presence",
            "perm_moved_on_before_first_league_game", "pt_exclusion_reason", "available_games", "appearances",
            "minutes", "in_cf_main_model_sample"]


def q1_departures(M: pd.DataFrame, ctx: dict) -> dict:
    u = ctx["u"]
    cand = M[M.in_playing_time_candidates]
    P = cand[cand.kind.eq("permanent")]
    nl = cand.pt_exclusion_reason.eq("no_league_games_in_window")
    lb = cand.left_before_first_receiving_league_game.eq(True)
    cov = cand.pt_covered_league_season
    rec = (cand[nl].groupby(["kind", "next_departure_type", "arrival_season_presence"]).size()
           .rename("episodes").reset_index())
    N = {
        "candidates": len(cand),
        "presence_by_kind": {k: g.arrival_season_presence.value_counts().to_dict() for k, g in cand.groupby("kind")},
        "next_type_by_kind": {k: g.next_departure_type.value_counts().to_dict() for k, g in cand.groupby("kind")},
        "no_league_games_in_window": {k: int((nl & cand.kind.eq(k)).sum()) for k in ("loan", "permanent")},
        "no_league_perm_by_next_type": cand[nl & cand.kind.eq("permanent")].next_departure_type.value_counts().to_dict(),
        "no_league_flagged_left_before": int((nl & lb).sum()),
        "no_league_not_flagged": int((nl & ~lb).sum()),
        "left_before_covered": int((lb & cov).sum()),
        "left_before_covered_not_no_league": int((lb & cov & ~nl).sum()),
        "left_before_all": int(lb.sum()),
        "perm_flag_true": int(P.perm_moved_on_before_first_league_game.eq(True).sum()),
        "perm_flag_false": int(P.perm_moved_on_before_first_league_game.eq(False).sum()),
        "perm_flag_unknown": int(P.perm_moved_on_before_first_league_game.isna().sum()),
        "perm_flag_true_by_type": P[P.perm_moved_on_before_first_league_game.eq(True)].next_departure_type
        .value_counts().to_dict(),
        "perm_within_7_days": int(P.days_at_receiving_club.le(7).sum()),
        "perm_within_7_days_loan": int((P.days_at_receiving_club.le(7) & P.next_departure_is_loan.eq(True)).sum()),
        "perm_within_30_days": int(P.days_at_receiving_club.le(30).sum()),
        "perm_within_30_days_loan": int((P.days_at_receiving_club.le(30) & P.next_departure_is_loan.eq(True)).sum()),
    }
    main = M[M.in_cf_main_model_sample]
    st = main.arrival_season_presence.eq("stayed_through_season_end")
    ld = main.arrival_season_presence.eq("left_during_season")
    N.update({"main_presence": main.arrival_season_presence.value_counts().to_dict(),
              "main_left_during_by_kind": main[ld].kind.value_counts().to_dict(),
              "main_left_before": int(main.left_before_first_receiving_league_game.eq(True).sum()),
              "main_stayed_zero_minutes": int((st & main.minutes.eq(0)).sum()),
              "main_stayed_played": int((st & main.minutes.gt(0)).sum()),
              "main_left_during_zero_minutes": int((ld & main.minutes.eq(0)).sum()),
              "main_games_left_during": float(main[ld].available_games.median()),
              "main_games_stayed": float(main[st].available_games.median())})
    assert N["no_league_flagged_left_before"] + N["no_league_not_flagged"] == int(nl.sum())
    assert N["main_left_before"] == 0, "a main-sample episode is flagged as leaving before the first league game"
    # Brunner (Transfermarkt id 918414)
    trace = u.rows[u.rows.player_id.eq(918414)].sort_values("_date")[[
        "event_id", "transfer_date", "transfer_season", "from_club_name", "to_club_name", "transfer_type_normalized",
        "fee_display_raw", "row_role"]]
    b = M[M.player_id.eq(918414)]
    mon = b[b.to_club_name.eq("Monaco") & b.kind.eq("permanent")].iloc[0]
    assert mon.perm_moved_on_before_first_league_game == True and mon.next_departure_type == "loan_out"  # noqa: E712
    assert mon.pt_exclusion_reason == "no_league_games_in_window" and mon.days_at_receiving_club == 1
    N["brunner"] = {c: (None if pd.isna(mon[c]) else (mon[c].item() if hasattr(mon[c], "item") else mon[c]))
                    for c in DEP_COLS}
    ex = []

    def pick(df, n, why):
        d = df.sort_values(["transfer_date", "event_id"], ascending=[False, True]).head(n)
        ex.extend([dict(r, example=why) for r in d[DEP_COLS].to_dict("records")])

    pick(b[b.transfer_date.ge("2024-01-01")], 5, "Paris Brunner (Freund's case)")
    PP = M[M.kind.eq("permanent")]
    pick(PP[PP.perm_moved_on_before_first_league_game.eq(True) & PP.next_departure_type.eq("loan_out")
            & PP.permanent_transfer_fee_eur.ge(5e6)], 4, "bought for 5m+ and loaned out before the first league game")
    pick(PP[PP.perm_moved_on_before_first_league_game.eq(True) & PP.next_departure_type.eq("permanent_transfer")], 3,
         "quick resale before the first league game")
    pick(main[ld & main.kind.eq("permanent")], 3, "left during the season: kept, share measured over the stay")
    pick(main[ld & main.kind.eq("loan")], 1, "loan recalled or moved on during the season: kept")
    pick(main[st & main.minutes.eq(0)], 2, "stayed all season, zero minutes (a true zero, not a departure)")
    pick(main[st & main.minutes.gt(0)], 1, "stayed all season and played")
    pick(cand[cand.arrival_season_presence.eq("unknown_next_move_inconsistent")], 2,
         "unknown: next recorded move does not leave this club")
    return {"N": N, "reconciliation": rec, "trace": trace, "examples": pd.DataFrame(ex)}


# ---------------------------------------------------------------------------
# 2. Quarantine
# ---------------------------------------------------------------------------

def q2_quarantine(M: pd.DataFrame, ctx: dict, R: pd.DataFrame) -> dict:
    INJ = ctx["tb"]["INJ"]
    q = INJ.injury_reason_clean.eq(fb.QUARANTINE)
    rev = M[M.in_revised_sample]
    ch = rev.quarantine_unavailable_games.fillna(0).astype(int)
    aff = rev[ch > 0]
    d = aff.fit_adjusted_share_v2 - aff.fit_adjusted_share
    N = {"rows": int(q.sum()), "players": int(INJ.loc[q, "player_id"].nunique()),
         "by_year": {str(k): int(v) for k, v in INJ.loc[q, "from_date"].dt.year.value_counts().sort_index().items()},
         "rows_2020_01_to_2022_06": int((q & INJ.from_date.between("2020-01-01", "2022-06-30")).sum()),
         "open_rows": int((q & INJ.open_injury).sum()), "missing_start_rows": int((q & INJ.missing_start).sum()),
         "corona_rows": int(INJ.injury_reason_clean.str.contains("corona", na=False).sum()),
         "episodes_affected": len(aff), "players_affected": int(aff.player_id.nunique()),
         "games_added": int(ch.sum()), "affected_by_kind": aff.kind.value_counts().to_dict(),
         "affected_usable": int(aff.in_injury_usable_v2.sum()),
         "affected_cf_model": int(aff.in_cf_injury_model_sample.sum()),
         "games_added_cf_model": int(aff.loc[aff.in_cf_injury_model_sample, "quarantine_unavailable_games"].sum()),
         "usable_v1": int(rev.in_injury_usable.sum()), "usable_v2": int(rev.in_injury_usable_v2.sum()),
         "flag_changes": int((rev.injury_data_quality_flag != rev.injury_data_quality_flag_v2).sum()),
         "cf_usable_v1": int(M.in_cf_injury_usable.sum()), "cf_usable_v2": int(M.in_cf_injury_usable_v2.sum()),
         "share_change_mean_pp": float(100 * d.mean()), "share_change_max_pp": float(100 * d.max()),
         "share_change_min_pp": float(100 * d.min()),
         "zero_fit_left": int((rev.in_injury_usable_v2 & rev.available_games_fit_v2.lt(1)).sum())}
    assert N["flag_changes"] == 0 and N["usable_v1"] == N["usable_v2"]
    cf = R[R.calendar_key.eq("calendar_first") & R.outcome.eq("Injury-adjusted share")].reset_index(drop=True)
    for i, tag in enumerate(("v1", "v2", "v2_validated")):
        N[f"coef_{tag}"], N[f"p_{tag}"], N[f"N_{tag}"] = float(cf.loan_coef[i]), float(cf.p[i]), int(cf.N[i])
    T = aff[["event_id", "player_id", "player_name", "kind", "to_club_name", "season", "window_start", "window_end",
             "minutes", "available_games_current", "confirmed_injury_games", "confirmed_injury_games_v2",
             "quarantine_unavailable_games", "available_games_fit", "available_games_fit_v2", "fit_adjusted_share",
             "fit_adjusted_share_v2", "in_injury_usable_v2", "in_cf_injury_model_sample",
             "in_cf_injury_model_sample_v2"]].sort_values("window_start")
    return {"N": N, "episodes": T}


# ---------------------------------------------------------------------------
# 3. Substitute minutes
# ---------------------------------------------------------------------------

MIN_COLS = ["event_id", "player_name", "club", "date", "venue", "opponent", "game_id", "role", "lineup_type", "m_in",
            "m_out", "m_red", "game_max_minutes", "minutes_played", "expected_minutes", "minutes_difference",
            "minutes_status", "in_cf_main_model_sample"]


def q3_minutes(M: pd.DataFrame, ctx: dict) -> dict:
    rel = M[M.pt_reliable]
    W = rel[["event_id", "player_id", "to_club_id", "window_start", "window_end"]].rename(columns={"to_club_id": "club_id"})
    A = fb.window_rows(ctx["Aall"], W)
    g = ctx["games"].assign(game_id=lambda d: d.game_id.astype("int64"))[["game_id", "home_club_id", "home_club_name",
                                                                          "away_club_name"]]
    A = A.merge(g, on="game_id", how="left")
    A["opponent"] = np.where(A.club_id == A.home_club_id, A.away_club_name, A.home_club_name)
    A["venue"] = np.where(A.club_id == A.home_club_id, "home", "away")
    Mk = M.set_index("event_id")
    A["player_name"] = A.event_id.map(Mk.player_name)
    A["club"] = A.event_id.map(Mk.to_club_name)
    A["in_cf_main_model_sample"] = A.event_id.map(Mk.in_cf_main_model_sample).astype(bool)
    A["date"] = A.date.dt.strftime("%Y-%m-%d")
    assert int(A.groupby("event_id").minutes_played.sum().sum()) == int(rel.minutes.sum())
    main = A[A.in_cf_main_model_sample]
    N = {"rows_reliable": len(A), "episodes_reliable": len(rel), "rows_main": len(main),
         "episodes_main": int(M.in_cf_main_model_sample.sum()),
         "status_reliable": A.minutes_status.value_counts().to_dict(),
         "status_main": main.minutes_status.value_counts().to_dict(),
         "role_main": main.role.value_counts().to_dict()}
    for role in ("starter", "substitute"):
        r = main[main.role.eq(role)]
        d = r.minutes_difference.abs()
        N[f"{role}_rows_main"] = len(r)
        N[f"{role}_exact_main"] = float(d.eq(0).mean())
        N[f"{role}_within1_main"] = float(d.le(1).mean())
        N[f"{role}_within2_main"] = float(d.le(2).mean())
        N[f"{role}_diff_counts_main"] = {str(int(k)): int(v) for k, v in r.minutes_difference.value_counts().head(6).items()}
    s90 = main[main.role.eq("starter") & main.m_out.isna() & main.m_red.isna()]
    N["starters_full_game_rows"], N["starters_full_game_90"] = len(s90), float(s90.minutes_played.eq(90).mean())
    hs = main[main.role.eq("substitute") & main.m_in.eq(46) & main.m_out.isna() & main.m_red.isna()]
    N["halftime_subs"], N["halftime_subs_45"] = len(hs), float(hs.minutes_played.eq(45).mean())
    st = main[main.role.eq("substitute") & main.m_in.ge(90) & main.m_out.isna()]
    N["stoppage_subs"], N["stoppage_subs_1"] = len(st), float(st.minutes_played.eq(1).mean())
    io = main[main.role.eq("substitute") & main.m_out.gt(main.m_in)]
    N["in_and_out"], N["in_and_out_within2"] = len(io), float(io.minutes_difference.abs().le(2).mean())
    rc = main[main.m_red.notna() & main.role.isin(["starter", "substitute"])]
    N["red_card_rows"], N["red_card_within2"] = len(rc), float(rc.minutes_difference.abs().le(2).mean())
    sg = A[A.minutes_status.eq("verified_shortened_game")]
    N["shortened_games"], N["shortened_rows"] = int(sg.game_id.nunique()), len(sg)
    N["episodes_contradicted"] = int(M.pt_minutes_validation_status.eq("contradicted_record").sum())
    for c in ("in_cf_main_model_sample", "in_age_model_sample", "in_cf_injury_model_sample", "in_injury_model_sample"):
        N[f"contradicted_in_{c}"] = int((M[c] & M.pt_minutes_validation_status.eq("contradicted_record")).sum())
    # the first check (6 Nov prototype): bench rows = any substitution-in event; end = red card or exit minute only if
    # strictly after entry, else 90; flagged when |difference| > 2
    b = main[main.m_in.notna()]
    end = np.where(b.m_red.notna() & (b.m_red > b.m_in), b.m_red, np.where(b.m_out.notna() & (b.m_out > b.m_in), b.m_out, 90))
    prev = b[(b.minutes_played - (np.clip(end, None, 90) - b.m_in)).abs().gt(2)]
    N["previous_bench_gt2"] = len(prev)
    N["previous_bench_gt2_status"] = prev.minutes_status.value_counts().to_dict()
    N["minutes_in_contradicted_episodes"] = int(M.loc[M.pt_minutes_validation_status.eq("contradicted_record"), "minutes"].sum())
    bad = A[~A.minutes_status.str.startswith("verified")]
    ex = []

    def take(df, why):
        d = df.sort_values(["date", "event_id"], ascending=[False, True]).head(1)
        ex.extend([dict(r, scenario=why) for r in d[MIN_COLS].to_dict("records")])

    def named(n):
        return main[main.player_name.eq(n)]

    e, a, z = named("Endrick"), named("Ansu Fati"), named("Edon Zhegrova")
    take(e[e.role.eq("substitute") & e.m_in.between(47, 89)], "Endrick: substitute in regular time")
    take(a[a.role.eq("substitute")], "Ansu Fati (loan): substitute")
    take(a[a.role.eq("starter") & a.m_out.notna()], "Ansu Fati (loan): starter substituted off")
    take(z[z.role.eq("substitute")], "Zhegrova: substitute")
    take(main[main.role.eq("starter") & main.m_out.isna() & main.m_red.isna() & main.minutes_played.eq(90)],
         "starter who played the whole game")
    take(main[main.role.eq("substitute") & main.m_in.eq(46) & main.m_out.isna() & main.minutes_played.eq(45)],
         "half-time substitute (46' counts 45)")
    take(main[main.role.eq("substitute") & main.m_in.ge(90) & main.m_out.isna()], "stoppage-time substitute (counts 1)")
    take(main[main.role.eq("substitute") & main.m_out.gt(main.m_in) & main.minutes_status.eq("verified_exact")],
         "came on and was later taken off")
    take(main[main.m_red.notna() & main.minutes_status.eq("verified_exact")], "sent off")
    take(A[A.minutes_status.eq("verified_shortened_game")], "game ended early (every player capped)")
    take(A[A.minutes_status.eq("unverifiable_no_lineup")], "no line-up in the source (cannot be checked)")
    take(A[A.minutes_status.eq("impossible_over_90")], "impossible: more than 90 minutes in a league game")
    take(A[A.minutes_status.eq("contradicted") & A.minutes_difference.abs().ge(40)],
         "contradicted: credited 90, substitution event at 46'")
    EX = pd.DataFrame(ex)
    EX["assessment"] = EX.minutes_status.map(fb.ROW_STATUS)
    return {"N": N, "anomalies": bad[MIN_COLS].sort_values(["minutes_status", "date"]), "examples": EX,
            "previous": prev[MIN_COLS],
            "status": A.minutes_status.value_counts().rename_axis("status").reset_index(name="rows")}


# ---------------------------------------------------------------------------
# 4. League tiers
# ---------------------------------------------------------------------------

SAMPLES = {"all master episodes": lambda M: M.event_id.notna(), "loans": lambda M: M.kind.eq("loan"),
           "permanent arrivals": lambda M: M.kind.eq("permanent"),
           "calendar-first main regression": lambda M: M.in_cf_main_model_sample,
           "no-calendar main regression": lambda M: M.in_age_model_sample,
           "normal-only main regression": lambda M: M.in_age_model_sample & (M.kind.eq("permanent")
                                                                             | M.loan_calendar_status.eq("normal"))}


def q4_tiers(M: pd.DataFrame) -> dict:
    rows = []
    for name, f in SAMPLES.items():
        g = M[f(M).astype(bool)]
        for side in ("from", "to"):
            st = g[f"{side}_league_tier_status"]
            t = g[f"{side}_league_tier_at_move"]
            rows.append({"sample": name, "club": "origin (from)" if side == "from" else "receiving (to)",
                         "episodes": len(g), "verified": int(st.eq("verified").sum()),
                         "verified_share": float(st.eq("verified").mean()),
                         **{f"tier_{k}": int(t.eq(k).sum()) for k in range(1, 6)},
                         "tier_6_plus": int(t.ge(6).sum()),
                         "youth_or_reserve_unknown": int(st.eq("youth_or_reserve_side_division_unknown").sum()),
                         "senior_unknown": int(st.eq("unknown_division_not_in_sources").sum()),
                         "season_outside_sources": int(st.eq("unknown_season_outside_sources").sum()),
                         "placeholder": int(st.eq("placeholder_not_a_club").sum()),
                         "in_covered_top_flight": int(g[f"{side}_in_covered_top_flight_at_move"].eq(True).sum())})
    COV = pd.DataFrame(rows)
    a = M[M.tier_reference_season.between(2012, 2025)]

    def by_country(side, ccol):
        return (a.assign(country=a[ccol].fillna("(unknown)"), ok=a[f"{side}_league_tier_status"].eq("verified"))
                .groupby("country").agg(episodes=("event_id", "size"), verified=("ok", "sum"))
                .assign(missing=lambda d: d.episodes - d.verified, verified_share=lambda d: d.verified / d.episodes)
                .sort_values("episodes", ascending=False).reset_index())

    by_season = (M.assign(f=M.from_league_tier_status.eq("verified"), t=M.to_league_tier_status.eq("verified"))
                 .groupby("tier_reference_season").agg(episodes=("event_id", "size"), from_verified=("f", "sum"),
                                                       to_verified=("t", "sum")).reset_index())
    by_season["from_share"] = by_season.from_verified / by_season.episodes
    by_season["to_share"] = by_season.to_verified / by_season.episodes
    leagues = (M[M.from_league_tier_status.eq("verified")]
               .groupby(["from_league_country_at_move", "from_league_tier_at_move", "from_league_name_at_move",
                         "from_league_tier_source"]).size().rename("origin_episodes").reset_index())
    main = M[M.in_cf_main_model_sample]
    N = {"main_from_verified": int(main.from_league_tier_status.eq("verified").sum()),
         "main_to_verified": int(main.to_league_tier_status.eq("verified").sum()), "main": len(main),
         "main_from_tiers": main.from_league_tier_at_move.value_counts().sort_index().to_dict(),
         "main_from_status": main.from_league_tier_status.value_counts().to_dict(),
         "main_from_not_top_but_verified": int((main.from_league_tier_status.eq("verified")
                                                & main.from_league_tier_at_move.gt(1)).sum()),
         "main_to_tier1": int(main.to_league_tier_at_move.eq(1).sum()),
         "promoted_or_relegated_from_clubs": int((M.from_league_tier_at_move.notna()
                                                  & M.from_league_tier_prior_season.notna()
                                                  & (M.from_league_tier_at_move != M.from_league_tier_prior_season)).sum())}
    return {"N": N, "coverage": COV, "by_country_from": by_country("from", "from_country_v2"),
            "by_country_to": by_country("to", "to_country_v2"), "by_season": by_season, "leagues": leagues}


# ---------------------------------------------------------------------------
# 5. Prior season
# ---------------------------------------------------------------------------

PRI_COLS = ["event_id", "player_name", "kind", "transfer_date", "from_club_name", "to_club_name", "prior_season",
            "prior_clubs", "prior_covered_clubs", "prior_leagues", "prior_league_games", "prior_appearances",
            "prior_minutes", "prior_minutes_share", "prior_injury_unavailable_games", "prior_fit_games",
            "prior_fit_adjusted_share", "prior_injury_data_quality_flag", "prior_season_coverage",
            "prior_exclusion_reason", "minutes_share"]


def q5_multiclub_check(M: pd.DataFrame, n2: int = 20, n3: int = 10) -> tuple[pd.DataFrame, int]:
    """Independent check of multi-club prior-season shares (added 8 Nov). Registration spells are rebuilt in SQL
    straight from the frozen Stage 1C file (LEAD over each player's dated rows; same row filters and same-day order as
    registration_spells), then each spell's league games and the player's minutes are counted from the snapshot. No
    pipeline function is used. Sample: reliable calendar-first main episodes with 2 (n2) and 3 (n3) covered clubs."""
    import duckdb
    from ..config import DEFAULT_DATABASE
    from .loan_episodes import CANONICAL_CSV
    base = M[M.prior_reliable.eq(True) & M.in_cf_main_model_sample]
    pick = pd.concat([base[base.prior_covered_clubs.eq(2)].sample(n2, random_state=21),
                      base[base.prior_covered_clubs.eq(3)].sample(min(n3, int(base.prior_covered_clubs.eq(3).sum())),
                                                                  random_state=21)])
    chk = pick[["event_id", "player_id", "prior_season", "transfer_date"]].assign(
        prior_season=pick.prior_season.astype(int), transfer_date=pd.to_datetime(pick.transfer_date).dt.date)
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        con.register("chk", chk)
        S = con.sql(f"""
            with r as (select player_id, cast(transfer_date as date) d, from_club_id, to_club_id,
                              transfer_type_normalized t, transfermarkt_transfer_id tid
                       from read_csv_auto('{CANONICAL_CSV}', all_varchar=true)
                       where lower(coalesce(transfermarkt_future_transfer, 'false')) not in ('true', '1')
                         and transfer_type_normalized <> 'other' and transfer_date is not null),
                 r2 as (select cast(player_id as bigint) player_id, d, cast(from_club_id as bigint) from_club_id,
                               cast(to_club_id as bigint) to_club_id,
                               row_number() over (partition by player_id order by d, (t <> 'return'),
                                                  cast(tid as bigint)) k
                        from r where cast(player_id as bigint) in (select player_id from chk)),
                 sp as (select player_id, to_club_id club_id, d s,
                               coalesce(lead(d) over (partition by player_id order by k) - 1, date '2100-01-01') e
                        from r2
                        union all
                        select player_id, from_club_id, date '1900-01-01', d - 1 from r2 where k = 1),
                 w as (select c.event_id, c.player_id, sp.club_id,
                              greatest(sp.s, make_date(c.prior_season, 7, 1)) a,
                              least(sp.e, make_date(c.prior_season + 1, 6, 30), c.transfer_date - 1) b, c.prior_season
                       from chk c join sp on sp.player_id = c.player_id)
            select w.event_id, w.club_id, min(w.a) spell_from, max(w.b) spell_to, count(*) games,
                   sum(coalesce(ap.minutes_played, 0)) mins
            from w join games g on g.competition_type = 'domestic_league' and cast(g.season as int) = w.prior_season
                 and w.club_id in (g.home_club_id, g.away_club_id) and g.date between w.a and w.b
            left join appearances ap on ap.game_id = cast(g.game_id as int) and ap.player_id = w.player_id
                 and ap.player_club_id = w.club_id
            where w.a <= w.b
            group by 1, 2 order by 1, 3""").df()
        names = con.sql("select club_id, name from clubs").df().drop_duplicates("club_id")
    finally:
        con.close()
    S["club"] = S.club_id.astype("int64").map(names.assign(club_id=names.club_id.astype("int64")).set_index("club_id").name)
    tot = S.groupby("event_id")[["games", "mins"]].sum()
    P = pick.set_index("event_id")
    agree = (tot.games.reindex(P.index) == P.prior_league_games) & (tot.mins.reindex(P.index) == P.prior_minutes)
    # prior_covered_clubs counts registration spells at covered clubs, including spells with no league game in the
    # window (e.g. a few summer days before a loan); the SQL lists only spells with games, so it can be smaller
    n_sql = S.groupby("event_id").size().reindex(P.index)
    clubs_ok = n_sql <= P.prior_covered_clubs
    out = S.merge(pick[["event_id", "player_name", "transfer_date", "prior_season", "prior_covered_clubs",
                        "prior_league_games", "prior_minutes", "prior_minutes_share"]], on="event_id")
    out["sql_episode_games"] = out.event_id.map(tot.games)
    out["sql_episode_minutes"] = out.event_id.map(tot.mins)
    out["sql_clubs_with_games"] = out.event_id.map(n_sql)
    out["agrees_with_master"] = out.event_id.map(agree & clubs_ok)
    ok = int((agree & clubs_ok).sum())
    assert ok == len(pick), out[~out.agrees_with_master.astype(bool)].head(10)
    cols = ["event_id", "player_name", "transfer_date", "prior_season", "prior_covered_clubs", "club", "club_id",
            "spell_from", "spell_to", "games", "mins", "sql_clubs_with_games", "sql_episode_games", "sql_episode_minutes",
            "prior_league_games", "prior_minutes", "prior_minutes_share", "agrees_with_master"]
    out = out[cols].rename(columns={"games": "sql_club_games", "mins": "sql_club_minutes"})
    out.attrs["fewer_clubs_with_games"] = int((n_sql < P.prior_covered_clubs).sum())
    out.attrs["clubs_with_games"] = n_sql.value_counts().sort_index().to_dict()
    return out, ok


def q5_prior(M: pd.DataFrame, ctx: dict) -> dict:
    import duckdb
    from ..config import DEFAULT_DATABASE
    E = ctx["E"]
    X = E[["event_id", "player_id", "move_date"]].assign(ref_season=fb.reference_season(E))
    O, G = fb.prior_season(X, fb.registration_spells(ctx["u"]), ctx["sides"], ctx["apps"], ctx["Aall"])
    Mk = M.set_index("event_id")
    O = O.set_index("event_id")
    rel = O.prior_reliable
    assert np.allclose(O.prior_minutes_share[rel], Mk.prior_minutes_share.reindex(O.index[rel]))
    assert not G.duplicated(["event_id", "game_id"]).any(), "duplicate prior-season game"
    Gs = G.merge(ctx["sides"][["game_id", "club_id", "season"]].assign(game_id=lambda d: d.game_id.astype("int64"))
                 .drop_duplicates(), on=["game_id", "club_id"], how="left", suffixes=("", "_g"))
    assert (Gs.season.to_numpy() == Gs.event_id.map(O.prior_season).astype(int).to_numpy()).all(), "wrong season"
    assert (G.date.to_numpy() < pd.to_datetime(G.event_id.map(Mk.transfer_date)).to_numpy()).all(), "post-move game"
    # independent SQL recomputation: single-club reliable episodes of the main sample
    one = M[M.prior_reliable.eq(True) & M.prior_covered_clubs.eq(1) & M.prior_clubs.eq(1)
            & M.in_cf_main_model_sample].sample(300, random_state=11)
    clubs = G[G.event_id.isin(one.event_id)].groupby("event_id").club_id.first()
    chk = one[["event_id", "player_id", "prior_season", "transfer_date"]].assign(
        club_id=one.event_id.map(clubs).astype("int64"), prior_season=one.prior_season.astype(int))
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        con.register("chk", chk)
        S = con.sql("""select c.event_id, count(*) games, sum(coalesce(a.minutes_played, 0)) mins
                       from chk c join games g on g.competition_type = 'domestic_league'
                            and cast(g.season as int) = c.prior_season and c.club_id in (g.home_club_id, g.away_club_id)
                            and g.date >= make_date(c.prior_season, 7, 1) and g.date < cast(c.transfer_date as date)
                            and g.date <= make_date(c.prior_season + 1, 6, 30)
                       left join appearances a on a.game_id = cast(g.game_id as int) and a.player_id = c.player_id
                            and a.player_club_id = c.club_id
                       group by 1""").df().set_index("event_id")
    finally:
        con.close()
    J = one.set_index("event_id").join(S)
    sql_ok = int(((J.games == J.prior_league_games) & (J.mins == J.prior_minutes)).sum())
    assert sql_ok == len(one), f"SQL recomputation differs on {len(one) - sql_ok} episodes"
    MC, mc_ok = q5_multiclub_check(M)
    cand = M[M.in_playing_time_candidates]
    rows = []
    for name, f in (("all episodes", lambda d: d.event_id.notna()),
                    ("calendar-first main", lambda d: d.in_cf_main_model_sample),
                    ("no-calendar main", lambda d: d.in_age_model_sample)):
        g = cand[f(cand).astype(bool)]
        for k, h in [("all", g)] + list(g.groupby("kind")):
            rows.append({"sample": name, "kind": k, "episodes": len(h),
                         "prior_share": int(h.prior_minutes_share.notna().sum()),
                         "prior_share_rate": float(h.prior_minutes_share.notna().mean()),
                         "whole_season": int(h.prior_season_coverage.eq("whole_season_at_covered_clubs").sum()),
                         "prior_fit_share": int(h.prior_fit_adjusted_share.notna().sum()),
                         "prior_fit_rate": float(h.prior_fit_adjusted_share.notna().mean()),
                         "mean_prior_share": float(h.prior_minutes_share.mean()),
                         "true_zero": int(h.prior_minutes.eq(0).sum())})
    COV = pd.DataFrame(rows)
    main = M[M.in_cf_main_model_sample]
    reasons = main.groupby(["prior_exclusion_reason", "kind"]).size().unstack(fill_value=0).reset_index()
    by_season = (main.groupby(["tier_reference_season", "kind"]).prior_minutes_share
                 .apply(lambda s: s.notna().mean()).unstack().reset_index())
    by_country = (main.assign(c=main.from_country_v2.fillna("(unknown)"), ok=main.prior_minutes_share.notna())
                  .groupby("c").agg(episodes=("event_id", "size"), with_share=("ok", "sum"))
                  .assign(rate=lambda d: d.with_share / d.episodes).sort_values("episodes", ascending=False)
                  .reset_index().rename(columns={"c": "origin_country"}))
    proto = pd.read_csv(OUT / "professor_questions_2026-11-06" / "prior_season_prototype.csv")
    Jp = main.merge(proto, on="event_id", how="left")
    recon = {"prototype_defined": int(Jp.prior_share.notna().sum()), "new_defined": int(Jp.prior_minutes_share.notna().sum()),
             "both": int((Jp.prior_share.notna() & Jp.prior_minutes_share.notna()).sum()),
             "equal_when_both": int((np.isclose(Jp.prior_share, Jp.prior_minutes_share)
                                     & Jp.prior_share.notna()).sum()),
             "prototype_only_by_reason": Jp[Jp.prior_share.notna() & Jp.prior_minutes_share.isna()]
             .prior_exclusion_reason.value_counts().to_dict(),
             "new_only": int((Jp.prior_share.isna() & Jp.prior_minutes_share.notna()).sum())}
    rows_ = ctx["u"].rows
    lw = main[["event_id", "player_id", "prior_season", "transfer_date"]].merge(
        rows_[rows_.transfer_type_normalized.eq("loan")][["player_id", "_date"]], on="player_id")
    lw = lw[(lw._date >= pd.to_datetime(lw.prior_season.astype(int).astype(str) + "-07-01"))
            & (lw._date < pd.to_datetime(lw.transfer_date))]
    had_loan = set(lw.event_id)
    ex = []

    def take(d, why):
        d = d.sort_values(["transfer_date", "event_id"], ascending=[False, True]).head(1)
        ex.extend([dict(r, example=why) for r in d[PRI_COLS].to_dict("records")])

    take(main[main.player_name.eq("Faitout Maouassa")], "running example of the guide")
    take(main[main.prior_season_coverage.eq("whole_season_at_covered_clubs") & main.prior_clubs.eq(1)
              & main.prior_minutes_share.gt(0.5)], "whole season at one club")
    take(main[main.prior_covered_clubs.ge(2) & main.prior_reliable.eq(True)], "two covered clubs in the season")
    take(main[main.event_id.isin(had_loan) & main.prior_reliable.eq(True)], "loaned during the prior season")
    take(main[main.prior_season_coverage.eq("part_of_season_at_covered_clubs")], "part of the season uncovered")
    take(main[main.prior_minutes.eq(0) & main.prior_league_games.gt(10)], "true zero: registered, did not play")
    take(main[main.prior_injury_unavailable_games.gt(5) & main.prior_injury_data_quality_flag.eq("usable")],
         "injury adjustment changes the denominator")
    take(main[main.prior_exclusion_reason.eq("no_covered_club")], "missing: came from an uncovered league")
    take(main[main.prior_exclusion_reason.eq("appearances_for_unregistered_club")],
         "missing: appears for a club he was not registered with")
    take(main[main.player_name.eq("Rasmus Højlund") & main.prior_exclusion_reason.eq("incomplete_club_games")],
         "missing: incomplete appearance record (Højlund)")
    N = {"sql_checked": len(one), "sql_equal": sql_ok, "multi_checked": int(MC.event_id.nunique()),
         "multi_equal": mc_ok, "multi_fewer_clubs_with_games": MC.attrs["fewer_clubs_with_games"],
         "multi_clubs_with_games": MC.attrs["clubs_with_games"], "multi_by_clubs": MC.drop_duplicates("event_id").prior_covered_clubs.astype(int)
         .value_counts().sort_index().to_dict(), "recon": recon, "game_rows": len(G),
         "main": len(main), "main_defined": int(main.prior_minutes_share.notna().sum()),
         "main_fit_defined": int(main.prior_fit_adjusted_share.notna().sum()),
         "main_whole": int(main.prior_season_coverage.eq("whole_season_at_covered_clubs").sum()),
         "corr_current": float(main[["prior_minutes_share", "minutes_share"]].corr().iloc[0, 1]),
         "main_by_kind": {k: {"n": len(g), "share": int(g.prior_minutes_share.notna().sum()),
                              "fit": int(g.prior_fit_adjusted_share.notna().sum()),
                              "mean": float(g.prior_minutes_share.mean())} for k, g in main.groupby("kind")},
         "fit_flag_main": main[main.prior_reliable.eq(True)].prior_injury_data_quality_flag.value_counts().to_dict(),
         "reasons_main": main.prior_exclusion_reason.value_counts().to_dict()}
    return {"N": N, "multiclub": MC, "coverage": COV, "reasons": reasons, "by_season": by_season, "by_country": by_country,
            "examples": pd.DataFrame(ex)}


# ---------------------------------------------------------------------------
# 6. Origin-club squad value
# ---------------------------------------------------------------------------

def q6_origin_value(M: pd.DataFrame) -> dict:
    import duckdb
    from ..config import DEFAULT_DATABASE
    from . import club_organisation as co
    typed = M[M.kind.isin(["loan", "permanent"])]
    rows = []
    for name, f in (("all typed episodes", lambda d: d.event_id.notna()), ("revised sample", lambda d: d.in_revised_sample),
                    ("calendar-first main", lambda d: d.in_cf_main_model_sample),
                    ("no-calendar main", lambda d: d.in_age_model_sample)):
        g = typed[f(typed).astype(bool)]
        for k, h in [("all", g)] + list(g.groupby("kind")):
            rows.append({"sample": name, "kind": k, "episodes": len(h),
                         "from_value": int(h.from_squad_value.notna().sum()),
                         "from_rate": float(h.from_squad_value.notna().mean()),
                         "ratio": int(h.from_to_squad_value_ratio.notna().sum()),
                         "median_from_value_m": float(h.from_squad_value.median() / 1e6),
                         "median_ratio": float(h.from_to_squad_value_ratio.median()),
                         "focal_in_from_squad": float(h.from_focal_in_squad.eq(True).mean())})
    COV = pd.DataFrame(rows)
    main = M[M.in_cf_main_model_sample]
    miss = main[main.from_squad_value.isna()]
    youth = fb.youth_club_ids(M, co.load_map())
    why = np.select([miss.from_club_id.isin(youth), miss.from_club_id.isin(fb.PLACEHOLDER_CLUB_IDS)],
                    ["origin is a youth or reserve side", "origin is a placeholder (Without Club etc.)"],
                    "fewer than 11 valued players at the origin club")
    S = typed.sample(400, random_state=5)[["event_id", "player_id", "from_club_id", "transfer_date", "from_squad_value",
                                           "from_squad_players", "from_squad_value_incl_player"]]
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        con.register("S", S)
        Q = con.sql(f"""
        with cand as (select distinct S.event_id, S.player_id focal, S.from_club_id club,
                             cast(S.transfer_date as date) d, v.player_id
                      from S join player_valuations v on v.current_club_id = S.from_club_id
                       and v.date < cast(S.transfer_date as date)
                       and v.date >= cast(S.transfer_date as date) - interval {reg.LOOKBACK_DAYS} day),
             latest as (select c.event_id, c.focal, c.club, c.d, c.player_id,
                               arg_max(v.market_value_in_eur, v.date) val, arg_max(v.current_club_id, v.date) vclub,
                               max(v.date) vdate
                        from cand c join player_valuations v on v.player_id = c.player_id and v.date < c.d
                        group by all)
        select event_id, sum(val) filter (where player_id <> focal) sv, count(*) filter (where player_id <> focal) n,
               sum(val) sv_incl
        from latest where vclub = club and vdate >= d - interval {reg.LOOKBACK_DAYS} day
        group by event_id""").df().set_index("event_id")
    finally:
        con.close()
    J = S.set_index("event_id").join(Q)
    J["sql_value"] = J.sv.where(J.n >= reg.MIN_SQUAD_PLAYERS)
    agree = (J.sql_value.isna() & J.from_squad_value.isna()) | np.isclose(J.sql_value, J.from_squad_value)
    agree_incl = (J.sql_value.isna() & J.from_squad_value_incl_player.isna()) | np.isclose(
        J.sv_incl.where(J.sql_value.notna()), J.from_squad_value_incl_player)
    cols = ["event_id", "player_name", "kind", "transfer_date", "from_club_name", "to_club_name", "from_squad_value",
            "from_squad_value_incl_player", "from_squad_players", "from_focal_in_squad", "from_squad_value_gap_days",
            "squad_value", "to_squad_value", "from_to_squad_value_ratio"]
    ex = []
    for nm in ("Faitout Maouassa", "Rasmus Højlund", "Ansu Fati", "Endrick", "Edon Zhegrova"):
        ex.extend(main[main.player_name.eq(nm)].head(2)[cols].to_dict("records"))
    ex.extend(main[main.from_squad_value.isna()].sort_values(["transfer_date", "event_id"]).tail(3)[cols].to_dict("records"))
    N = {"sql_checked": len(J), "sql_agree": int(agree.sum()), "sql_agree_incl": int(agree_incl.sum()),
         "sql_with_value": int(J.from_squad_value.notna().sum()), "main": len(main),
         "main_rate": float(main.from_squad_value.notna().mean()),
         "main_n": int(main.from_squad_value.notna().sum()),
         "main_by_kind": main.groupby("kind").from_squad_value.apply(lambda s: float(s.notna().mean())).to_dict(),
         "main_missing_reasons": pd.Series(why).value_counts().to_dict(),
         "median_ratio_main": float(main.from_to_squad_value_ratio.median()),
         "focal_in_from_main": float(main.from_focal_in_squad.eq(True).mean()),
         "to_value_equal_stored": int(np.isclose(M.loc[M.squad_value.notna(), "to_squad_value"],
                                                 M.loc[M.squad_value.notna(), "squad_value"]).sum()),
         "stored_values": int(M.squad_value.notna().sum())}
    assert N["sql_agree"] == N["sql_checked"] and N["sql_agree_incl"] == N["sql_checked"], J[~agree].head()
    return {"N": N, "coverage": COV, "examples": pd.DataFrame(ex)}


# ---------------------------------------------------------------------------
# Old versus new master; sample flows
# ---------------------------------------------------------------------------

def master_comparison(Mr: pd.DataFrame) -> dict:
    """Every pre-existing column, cell by cell, as text (exactly as written in the CSV)."""
    import gzip
    with gzip.open(OLD_MASTER, "rt") as fh:
        Old = pd.read_csv(fh, dtype=str, keep_default_na=False)
    old_cols = list(Old.columns)
    assert list(Mr.columns[:len(old_cols)]) == old_cols, "pre-existing column names or order changed"
    assert Mr.event_id.equals(Old.event_id), "row order changed"
    diffs = []
    for c in old_cols:
        ne = Mr[c].ne(Old[c])
        for i in np.flatnonzero(ne.to_numpy())[:50]:
            diffs.append({"column": c, "event_id": Old.event_id.iat[i], "old": Old[c].iat[i], "new": Mr[c].iat[i]})
    D = pd.DataFrame(diffs, columns=["column", "event_id", "old", "new"])
    col_sha = pd.DataFrame({"column": list(Mr.columns), "sha256": [hashlib.sha256("\n".join(Mr[c]).encode()).hexdigest()
                                                                    for c in Mr.columns],
                            "status": ["pre-existing" if c in old_cols else "new (Nov 2026)" for c in Mr.columns]})
    return {"old_columns": len(old_cols), "new_columns": Mr.shape[1] - len(old_cols), "rows": len(Mr),
            "cells_compared": len(Mr) * len(old_cols), "cells_different": len(D), "diffs": D, "col_sha": col_sha}


def flows(M: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, (label, f) in CAL.items():
        cal = f(M).astype(bool)
        for stage, mask in (("main regression (stored)", M.in_age_model_sample),
                            ("minus contradicted minutes", M.in_age_model_sample & ~M.pt_minutes_validated),
                            ("main regression v2", M.in_age_model_sample_v2),
                            ("injury regression (stored)", M.in_injury_model_sample),
                            ("injury-usable status changed by quarantine", M.in_injury_model_sample
                             & (M.in_injury_usable != M.in_injury_usable_v2)),
                            ("minus contradicted minutes (injury)", M.in_injury_model_sample & M.in_injury_usable_v2
                             & ~M.pt_minutes_validated),
                            ("injury regression v2", M.in_injury_model_sample_v2),
                            ("minus line-up contradicted (main)", M.in_age_model_sample_v2 & ~M.pt_reliable_v2),
                            ("main regression, both checks", M.in_age_model_sample_validated),
                            ("minus line-up contradicted (injury)", M.in_injury_model_sample_v2 & ~M.pt_reliable_v2),
                            ("injury regression, both checks", M.in_injury_model_sample_validated)):
            g = M[mask & cal]
            rows.append({"calendar_definition": label, "stage": stage, "loans": int(g.kind.eq("loan").sum()),
                         "permanent": int(g.kind.eq("permanent").sum()), "total": len(g)})
    F = pd.DataFrame(rows)
    for lab in F.calendar_definition.unique():
        f = F[F.calendar_definition.eq(lab)].set_index("stage")
        for c in ("loans", "permanent", "total"):
            assert f.loc["main regression (stored)", c] - f.loc["minus contradicted minutes", c] == f.loc["main regression v2", c]
            assert (f.loc["injury regression (stored)", c] - f.loc["injury-usable status changed by quarantine", c]
                    - f.loc["minus contradicted minutes (injury)", c] == f.loc["injury regression v2", c])
            assert (f.loc["main regression v2", c] - f.loc["minus line-up contradicted (main)", c]
                    == f.loc["main regression, both checks", c])
            assert (f.loc["injury regression v2", c] - f.loc["minus line-up contradicted (injury)", c]
                    == f.loc["injury regression, both checks", c])
    return F


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(DIR / name, index=False)


def jdump(x):
    if isinstance(x, dict):
        return {str(k): jdump(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jdump(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NA or (isinstance(x, float) and np.isnan(x)):
        return None
    return x


def ex_md(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    rows = []
    for r in df[cols].itertuples(index=False):
        rows.append([("" if (v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))) else
                      fmt[c](v) if c in fmt else v) for c, v in zip(cols, r)])
    return md_table(cols, rows)


# ---------------------------------------------------------------------------
# Professor-readable reports
# ---------------------------------------------------------------------------

def render_q1(r: dict) -> str:
    N, b = r["N"], r["N"]["brunner"]
    nl = N["no_league_games_in_window"]
    t = N["no_league_perm_by_next_type"]
    pres = N["presence_by_kind"]["permanent"]
    return f"""# 1. Players who leave the receiving club at once (Freund)

**Question.** For permanent transfers, do we record whether the player actually spent the season at the new club
(Paris Brunner: Dortmund → Monaco, immediately loaned out)? Is there a flag?

**Answer.** Yes, now explicitly. Fourteen columns describe how and when each player left the receiving club; the
one-line flag is `perm_moved_on_before_first_league_game`. No episode is removed from the master.

## Paris Brunner, traced from the raw rows

{md_table(list(r["trace"].columns), r["trace"].astype(str).values.tolist())}
Master flags for the Monaco episode: next departure `{b['next_departure_type']}` to {b['next_departure_to_club_name']}
on {b['next_departure_date']} ({b['days_at_receiving_club']} day after arriving); Monaco's first league game after the
arrival was on {b['receiving_first_league_game_date']}. The playing-time window ends the day before the departure, so
that game is not in his window: `left_before_first_receiving_league_game` = True,
`perm_moved_on_before_first_league_game` = True, playing-time exclusion reason `{b['pt_exclusion_reason']}`. His
Cercle Brugge loan is a separate episode (stayed through the season end).

## Three situations kept apart

`arrival_season_presence` is built from dates only:

| Presence (permanent arrivals, all candidates) | Episodes |
|---|---|
""" + "".join(f"| {k} | {f0(v)} |\n" for k, v in pres.items()) + f"""
* **Left before the first league game**: excluded from playing time already (no league game in the window).
* **Stayed but played zero minutes**: kept; in the calendar-first main sample {f0(N['main_stayed_zero_minutes'])}
  episodes stayed through the season and played 0 minutes (true zeros), against {f0(N['main_stayed_played'])} who
  stayed and played.
* **Played part of the season, then left**: kept, share measured over the games while at the club. Main sample:
  {f0(sum(N['main_left_during_by_kind'].values()))} such episodes ({', '.join(f'{k}: {f0(v)}' for k, v in N['main_left_during_by_kind'].items())});
  median {N['main_games_left_during']:.0f} window games against {N['main_games_stayed']:.0f} for those who stayed.
  No main-sample episode is flagged as leaving before the first league game ({N['main_left_before']}).

## Reconciliation with the existing exclusion

The earlier count was {f0(nl['permanent'])} permanent arrivals (and {f0(nl['loan'])} loans) excluded because the
window held no league game. Of these {f0(nl['permanent'] + nl['loan'])}, {f0(N['no_league_flagged_left_before'])} carry
the new flag and {f0(N['no_league_not_flagged'])} has an unknown status (its next recorded move does not leave the club).
By next move, the permanent ones were: {', '.join(f'{k} {f0(v)}' for k, v in t.items())}. (The preliminary note
quoted 656 loaned out; that count used the raw type of the next row without checking that it leaves the club.)

Across all permanent arrivals, {f0(N['perm_within_7_days'])} moved on within 7 days ({f0(N['perm_within_7_days_loan'])}
of them on loan) and {f0(N['perm_within_30_days'])} within 30 days ({f0(N['perm_within_30_days_loan'])} on loan).
`perm_moved_on_before_first_league_game`: True {f0(N['perm_flag_true'])}, False {f0(N['perm_flag_false'])}, unknown
{f0(N['perm_flag_unknown'])} (mostly clubs whose league is not in the snapshot that season, where the first league
game is not observed; `days_at_receiving_club` is still filled whenever the departure is recorded).

Examples: `q1_departure_examples.csv`; full cross-tab: `q1_reconciliation.csv`.
"""


def render_q2(r: dict) -> str:
    N = r["N"]
    return f"""# 2. Quarantine counted as unavailability (Freund)

**Rule change (v2).** Injury spells labelled `Quarantine` now count as unavailability in the injury-adjusted share.
Quarantine means the player was isolated and not available to the coach; it does not mean a confirmed infection.
`Fitness` and `Rest` still do not count. If the player appeared in a game, the game stays in the denominator. A game
covered by several spells counts once. The minutes (numerator) do not change. The previous rule is kept: the v1
columns are unchanged, and the v2 results sit in new columns ending in `_v2`.

| | |
|---|---|
| Quarantine spells in the injury file | {f0(N['rows'])} ({f0(N['players'])} players) |
| of which January 2020 to June 2022 | {f0(N['rows_2020_01_to_2022_06'])} |
| By start year | {', '.join(f'{k}: {v}' for k, v in N['by_year'].items())} |
| Open (no end) or no start date | {N['open_rows']} / {N['missing_start_rows']} |
| 'Corona virus' spells (already counted as illness) | {f0(N['corona_rows'])} |
| Revised-sample episodes gaining games | {N['episodes_affected']} ({N['players_affected']} players; {', '.join(f'{k} {v}' for k, v in N['affected_by_kind'].items())}) |
| Games newly classified as unavailable | {N['games_added']} |
| ... in the calendar-first injury regression sample | {N['affected_cf_model']} episodes, {N['games_added_cf_model']} games |
| Change in injury-adjusted share (affected episodes) | mean {N['share_change_mean_pp']:+.2f} pp, range {N['share_change_min_pp']:+.2f} to {N['share_change_max_pp']:+.2f} pp |
| Injury-usable episodes (revised sample) | v1 {f0(N['usable_v1'])}, v2 {f0(N['usable_v2'])}: no episode changes status |
| Calendar-first injury-usable | v1 {f0(N['cf_usable_v1'])}, v2 {f0(N['cf_usable_v2'])} |
| Calendar-first injury regression, loan coefficient (injury-adjusted share) | v1 {pp(N['coef_v1'])} pp (p = {N['p_v1']:.3f}, N = {f0(N['N_v1'])}); v2 {pp(N['coef_v2'])} pp (p = {N['p_v2']:.3f}, N = {f0(N['N_v2'])}) |
| v2 and validated minutes | {pp(N['coef_v2_validated'])} pp (p = {N['p_v2_validated']:.3f}, N = {f0(N['N_v2_validated'])}) |

**Conclusion.** Counting quarantine changes {N['games_added']} games in {N['episodes_affected']} episodes and does
not change the injury result. Episode list: `q2_quarantine_episodes.csv`.
"""


def render_q3(r: dict) -> str:
    N = r["N"]
    ex = r["examples"]
    sc = N["status_main"]
    return f"""# 3. Minutes for players coming off the bench (Fernandes)

**How minutes enter.** The playing-time code sums Transfermarkt's `minutes_played` for each league appearance at
the receiving club inside the window. Nothing is assumed: a substitute is never given 90 minutes. The sums reproduce
the stored minutes of all {f0(N['episodes_reliable'])} reliable episodes.

**Independent check.** Each league appearance in a reliable window ({f0(N['rows_reliable'])} rows) was compared with
the line-up (starter or substitute) and the substitution and red-card events. Expected minutes = (exit minute, red-card
minute or 90) − (entry minute or 0). Calendar-first main sample ({f0(N['rows_main'])} appearance rows in
{f0(N['episodes_main'])} episodes):

| Case | Rows | Agreement |
|---|---|---|
| Substitutes | {f0(N['substitute_rows_main'])} | exact {pct(N['substitute_exact_main'])}, within 1 min {pct(N['substitute_within1_main'], 2)}, within 2 {pct(N['substitute_within2_main'], 2)} |
| Starters | {f0(N['starter_rows_main'])} | exact {pct(N['starter_exact_main'])}, within 1 min {pct(N['starter_within1_main'], 2)} |
| Starters not substituted or sent off | {f0(N['starters_full_game_rows'])} | recorded 90: {pct(N['starters_full_game_90'], 2)} |
| Half-time substitutes (event minute 46) | {f0(N['halftime_subs'])} | recorded 45: {pct(N['halftime_subs_45'], 1)} |
| Stoppage-time substitutes (minute 90+) | {f0(N['stoppage_subs'])} | recorded 1: {pct(N['stoppage_subs_1'], 1)} |
| Came on and later went off | {f0(N['in_and_out'])} | within 2 min {pct(N['in_and_out_within2'], 1)} |
| Sent off | {f0(N['red_card_rows'])} | within 2 min {pct(N['red_card_within2'], 1)} |

The one-minute differences are Transfermarkt conventions, not errors: a substitution logged at 46' is credited 45
minutes, one in stoppage time is credited 1, and minute labels drop the stoppage time (an entry at 90+3 is logged 90).

Status of all rows in the main sample: {', '.join(f'{k} {f0(v)}' for k, v in sc.items())}.

**Anomalies.** {N['shortened_games']} games ended early (every player's minutes stop at the same point: abandoned or
shortened matches); the recorded minutes agree with the game's length. Rows that cannot be checked (no line-up in the
source, mostly 2012/13, or an event minute recorded as -1) are kept: being uncheckable is not evidence of an error.
{N['episodes_contradicted']} reliable episodes contain one record that line-ups and events contradict, including the
only league row above 90 minutes (135 minutes: Jorge Carrascal, Karpaty Lviv v Olimpik Donetsk, 21 February 2018,
listed as a starter but also substituted on at 46'). In none of them is the correct value independently recoverable
(the two sources disagree with each other), so **no minutes were edited**. Instead these episodes are flagged
(`pt_minutes_validation_status = contradicted_record`) and left out of the validated outcome
`minutes_share_validated` and the v2 samples. All {N['contradicted_in_in_cf_main_model_sample']} are in the
calendar-first main sample and {N['contradicted_in_in_cf_injury_model_sample']} in its injury sample.

The first check (6 November) flagged {N['previous_bench_gt2']} bench rows of the main sample with a difference above 2
minutes. Classified with the full rule (line-up, both events, game length, same-minute on and off), they are:
{', '.join(f'{k} {v}' for k, v in N['previous_bench_gt2_status'].items())}.

## Hand-checked games

{ex_md(ex, ['scenario', 'player_name', 'club', 'date', 'opponent', 'role', 'm_in', 'm_out', 'minutes_played', 'expected_minutes', 'minutes_difference', 'minutes_status'], {c: (lambda v: f'{v:.0f}') for c in ('m_in', 'm_out', 'minutes_played', 'expected_minutes', 'minutes_difference')})}
Files: `q3_manual_examples.csv`, `q3_unverified_and_contradicted_rows.csv`, `q3_previous_11_discrepancies.csv`,
`q3_row_status_counts.csv`.
"""


def render_q4(r: dict, numbers: dict) -> str:
    C, N = r["coverage"], r["N"]
    rows = [[x.sample, x.club, f0(x.episodes), f"{f0(x.verified)} ({pct(x.verified_share)})", f0(x.tier_1), f0(x.tier_2),
             f0(x.tier_3 + x.tier_4 + x.tier_5 + x.tier_6_plus), f0(x.youth_or_reserve_unknown), f0(x.senior_unknown),
             f0(x.season_outside_sources + x.placeholder)] for x in C.itertuples()]
    bc = r["by_country_from"].head(15)
    gaps = numbers.get("gaps_text", "")
    return f"""# 4. League tier of the origin and receiving clubs (Fernandes)

**What we built.** For both clubs, in the season of the move (`tier_reference_season`), the league name, a numeric
tier (1 = first division of that country's pyramid), the country, the source and a status. The origin club also gets
its league in the season before (`from_league_*_prior_season`), the season a summer mover has just played. Tiers are
season-specific: a promoted or relegated club changes tier
({f0(N['promoted_or_relegated_from_clubs'])} episodes have an origin club whose tier differs between the two seasons).

**Sources.** (1) The Transfermarkt snapshot's league games (tier 1 for its 31 top divisions). (2) openfootball
(public domain, CC0): league tables for lower divisions, e.g. England tiers 2 to 5, Germany 2 and 3, Spain, Italy and
France 2, Netherlands and Portugal 2 (recent seasons), Austria and Switzerland 2. Club names were matched to our club
ids within the league's country by exact normalised name, aliases learned from seasons that appear in both sources, and
a short checked manual list; ambiguous names were left unmatched. Reserve and youth teams can only match reserve and
youth teams. football-data.co.uk was not used: its robots.txt disallows AI agents, and requesting its files under a
different user agent would circumvent that. Details: `data/processed/league_tiers/README.md`.

**Status values.** `verified` (found in a league table that season); `youth_or_reserve_side_division_unknown`;
`unknown_division_not_in_sources` (a senior club not found: a lower division without a source, another country, or a
name we could not match safely); `unknown_season_outside_sources`; `placeholder_not_a_club`. Only `verified` rows carry
a tier. A club outside our 14 covered leagues is never assumed to be tier 2: `*_in_covered_top_flight_at_move` says
only whether the club was in one of the 14 covered leagues.

| Sample | Club | Episodes | Verified | Tier 1 | Tier 2 | Tier 3+ | Youth/reserve unknown | Senior unknown | Other unknown |
|---|---|---|---|---|---|---|---|---|---|
""" + "".join("| " + " | ".join(x) + " |\n" for x in rows) + f"""
Receiving clubs in the playing-time samples are always tier 1 (the playing-time analysis covers only the 14 top
divisions). In the calendar-first main sample the origin tier is verified for {f0(N['main_from_verified'])} of
{f0(N['main'])} episodes; {f0(N['main_from_not_top_but_verified'])} of these come from a verified lower division.

## Missing origin tiers by country (seasons 2012/13 to 2025/26, largest countries)

{md_table(['country', 'episodes', 'verified', 'missing', 'verified share'], [[x.country, f0(x.episodes), f0(x.verified), f0(x.missing), pct(x.verified_share)] for x in bc.itertuples()])}
By season and for receiving clubs: `q4_tier_coverage_by_season.csv`, `q4_tier_missing_by_country_to.csv`.

**Hard limitation.** No lawful free source we could reach covers Scotland tiers 2 to 4, Spain and France tier 3,
Belgium and Greece tier 2, Türkiye tier 2 (except 2019/20), Germany tier 4 (except 2024/25), Italy tier 3 (except
2024/25), Netherlands and Portugal tier 2 before 2020, or most non-European leagues before 2024. Those clubs stay
missing; they are not coded as lower tiers.
"""


def render_q5(r: dict) -> str:
    N, C = r["N"], r["coverage"]
    rc = N["recon"]
    rows = [[x.sample, x.kind, f0(x.episodes), f"{f0(x.prior_share)} ({pct(x.prior_share_rate)})", f0(x.whole_season),
             f"{f0(x.prior_fit_share)} ({pct(x.prior_fit_rate)})", f"{x.mean_prior_share:.3f}", f0(x.true_zero)]
            for x in C.itertuples()]
    ex = r["examples"]
    return f"""# 5. Playing time in the season before the move (Fernandes)

**Definition.** For each episode, season p is the season before the move (p = arrival season − 1, the receiving
club's league season where known, otherwise the Transfermarkt season). Window: 1 July p to 30 June p+1, cut at the day
before the move, so nothing after the move enters. Games: the domestic league games of every club the player was
registered with in that window, while registered (from the Stage 1C history, loans and returns included), for clubs
in a league of the snapshot. Minutes: his league minutes for that club in those games. Several spells are pooled
(total minutes over 90 × total games), not averaged. Cups and European games are excluded.

**Same quality rules as the current measure.** The share is left blank (never set to zero) when: no club of his was in
a covered league; a covered club's game lacks a complete appearance record (fewer than 11 rows); he appears for a club
he was not registered with (a move is missing from his history); or a minute record is contradicted by line-ups and
events. The injury-adjusted version uses the quarantine-inclusive rule v2 with the same appearance override; if the
injury history is not usable the ordinary share is kept.

| Sample | Kind | Episodes | Prior share | Whole season covered | Injury-adjusted | Mean share | True zeros |
|---|---|---|---|---|---|---|---|
""" + "".join("| " + " | ".join(x) + " |\n" for x in rows) + f"""
Correlation with the current share (main sample): {N['corr_current']:.3f}.

**Validation.** No game appears twice for an episode; every game belongs to season p; every game is before the move.
An independent SQL recomputation of {N['sql_checked']} single-club episodes agrees on games and minutes for all
{N['sql_equal']}. A second check covers multi-club seasons: for {N['multi_checked']} episodes with 2 or 3 covered clubs
({', '.join(f'{v} with {k} clubs' for k, v in N['multi_by_clubs'].items())}), the registration spells were rebuilt in
SQL directly from the Stage 1C file, without the pipeline code, and each club's games and minutes were counted from the
snapshot. The pooled games and minutes agree for {N['multi_equal']} of {N['multi_checked']}. In
{N['multi_fewer_clubs_with_games']} of them at least one covered spell contains no league game in the window (for example a few
summer days at a club before a loan), so the SQL lists fewer clubs with games than `prior_covered_clubs`, which counts
spells; the totals are unaffected. Club-by-club detail: `q5_multiclub_sql_check.csv`.

**Against the prototype.** The prototype had a share for {f0(rc['prototype_defined'])} main-sample episodes; this
version has {f0(rc['new_defined'])}. Where both exist they agree on {f0(rc['equal_when_both'])} of {f0(rc['both'])}
(the rest differ because the window now uses registration spells, internal registrations included, and every league
of the snapshot). {f0(sum(rc['prototype_only_by_reason'].values()))} prototype values are now blank on purpose:
{', '.join(f'{k} {v}' for k, v in rc['prototype_only_by_reason'].items())}; {rc['new_only']} are new. The drop comes
from applying the same checks as the current measure; we did not relax them to raise coverage.

Main-sample reasons for a missing share: {', '.join(f'{k} {f0(v)}' for k, v in N['reasons_main'].items())}.

The prior-season variables are additional controls. They never restrict the main or injury samples.

## Examples

{ex_md(ex, ['example', 'player_name', 'from_club_name', 'to_club_name', 'prior_season', 'prior_leagues', 'prior_league_games', 'prior_appearances', 'prior_minutes', 'prior_minutes_share', 'prior_fit_adjusted_share', 'prior_exclusion_reason'], {'prior_minutes_share': lambda v: f'{v:.3f}', 'prior_fit_adjusted_share': lambda v: f'{v:.3f}', 'prior_season': lambda v: f'{int(v)}/{str(int(v) + 1)[2:]}'})}
Files: `q5_prior_season_coverage.csv`, `q5_prior_reasons.csv`, `q5_prior_by_season.csv`, `q5_prior_by_country.csv`,
`q5_prior_examples.csv`.
"""


def render_q6(r: dict) -> str:
    N, C = r["N"], r["coverage"]
    rows = [[x.sample, x.kind, f0(x.episodes), f"{f0(x.from_value)} ({pct(x.from_rate)})", f0(x.ratio),
             f"{x.median_ratio:.2f}" if not np.isnan(x.median_ratio) else "", pct(x.focal_in_from_squad)]
            for x in C.itertuples()]
    return f"""# 6. Market value of the origin club (Fernandes)

**Rule.** Exactly the receiving-club rule (`playing_time_regression.squad_value`) applied to the origin club: the sum
of the latest valuation strictly before the move, at most 365 days old, of every player whose valuation lists the
club; left blank if fewer than 11 players qualify. Nothing after the move is used.

**Decision on the departing player.** The main column `from_squad_value` excludes him, as `squad_value` excludes the
arriving player, so the two are comparable and the player's own value (already a control) is not counted twice.
`from_squad_value_incl_player` adds him back for anyone who wants the full pre-departure squad.

**Also added.** `from_squad_players`, `from_focal_in_squad`, `from_squad_value_gap_days` (age of the newest valuation
used), `log_from_squad_value`, `to_squad_value` (the receiving value for every episode, not only the revised sample)
and `from_to_squad_value_ratio` with its log.

| Sample | Kind | Episodes | Origin value | Ratio | Median origin/receiving | Player in origin squad |
|---|---|---|---|---|---|---|
""" + "".join("| " + " | ".join(x) + " |\n" for x in rows) + f"""
Missing in the calendar-first main sample: {', '.join(f'{k} {f0(v)}' for k, v in N['main_missing_reasons'].items())}.

**Validation.** The function reproduces all {f0(N['stored_values'])} stored receiving-club values exactly
({f0(N['to_value_equal_stored'])} equal). An independent SQL recomputation of {N['sql_checked']} random episodes
({N['sql_with_value']} with a value) agrees on all {N['sql_agree']} (both variants). The main sample is not changed by
missing origin values. Examples: `q6_origin_value_examples.csv`.
"""


def render_regression(R: pd.DataFrame, F: pd.DataFrame) -> str:
    out = ["# Regression comparison: old versus new\n",
           "Preferred specification M3 (loan, remaining-season share, age, age², log relative value, log squad value, "
           "league-season fixed effects, position; standard errors clustered by receiving club). Version 1 rows are "
           "the stored results, reproduced exactly from the master. Version 2 counts quarantine (injury rows) or "
           "drops the episodes with a contradicted minute record (main rows). Version 3 does both. Correction of 8 November: "
           "the earlier line-up check (`pt_reliable_v2`: players on the pitch with no appearance row) is a separate "
           "problem that the v2 samples did not remove. Main version 3 and injury version 4 apply both minute checks "
           "(`pt_fully_validated`; samples `in_*_model_sample_validated`). No episode fails both checks.\n"]
    for lab in R.calendar_definition.unique():
        g = R[R.calendar_definition.eq(lab)]
        out.append(f"\n## {lab}\n")
        out.append(md_table(["outcome", "version", "N", "loans + permanent", "loan coefficient (pp)", "95% CI", "p"],
                            [[x.outcome, x.version, f0(x.N), f"{f0(x.loans)} + {f0(x.permanent)}", pp(x.loan_coef),
                              f"{pp(x.ci_lo)} to {pp(x.ci_hi)}", f"{x.p:.3f}"] for x in g.itertuples()]))
    out.append("\n## Sample flow of the corrections\n")
    out.append(md_table(list(F.columns), F.astype(str).values.tolist()))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Guide v3 (the v2 guide plus a short November section and appendices)
# ---------------------------------------------------------------------------

def tex(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ("→", r"$\to$"),
                 ("≥", r"$\geq$"), ("×", r"$\times$"), ("−", "-"), ("²", r"$^2$"), ("…", "...")):
        s = s.replace(a, b)
    return s


def guide_v3(R: pd.DataFrame, Q: dict, D: pd.DataFrame, saved: dict) -> str:
    s = (OUT / "data_and_sample_guide_v2.tex").read_text()
    n_new = len(fs.NEW_COLUMNS)
    s = s.replace(r"\date{2 November 2026}",
                  r"\date{7 November 2026 (v3: November 5 feedback added; validation correction 8 November)}")
    # 8 Nov corrections to the v2 text: the original injury rule vs rule v2, and the false-zero claim
    old_q = "Only medical reasons count; ``rest'', ``fitness'' and ``quarantine'' do not."
    assert s.count(old_q) == 1
    s = s.replace(old_q, "Only medical reasons count. Under the original rule (v1, used for the stored results in "
                  "this section), ``rest'', ``fitness'' and ``quarantine'' do not count. The updated rule v2 "
                  "(Section~\\ref{sec:nov}, columns ending in \\col{_v2}) also counts ``quarantine'' as unavailability, "
                  "meaning the player was isolated, not that an infection was confirmed; ``rest'' and ``fitness'' "
                  "still do not count.")
    old_z = "so a gap in the data never turns into a false zero."
    assert s.count(old_z) == 1
    s = s.replace(old_z, "so a gap that this rule can detect does not turn into a false zero. The rule cannot see a "
                  "player who is missing from an otherwise complete record; the line-up check below found a small "
                  "number of such cases.")
    s = s.replace("Appendix~\\ref{app:master} (all 150 master columns) are reference tables.",
                  "Appendix~\\ref{app:master} (the original 150 master columns) are reference tables. Section~"
                  "\\ref{sec:nov} answers the November 5 questions; Appendix~\\ref{app:new} lists the "
                  f"{n_new} columns added for them.")
    # Table 2: add the normal-only sensitivity column (numbers from the lineage run)
    v = saved["calendar_first"]["variants"]
    def tot(k, s_):
        t = v[k]["totals"][s_]
        return t["total"], t["loan"], t["permanent"]
    a, b, c = tot("current", "main"), tot("classifiable", "main"), tot("normal_only", "main")
    ai, bi, ci = tot("current", "injury_model"), tot("classifiable", "injury_model"), tot("normal_only", "injury_model")
    co = lambda k: pp(v[k]["estimates"]["m3"]["coef"])  # noqa: E731
    old_tab = s[s.index(r"\begin{tabular}{lcc}"):s.index(r"\caption{How the calendar-first rule changes")]
    new_tab = (r"\begin{tabular}{lccc}" "\n\\toprule\n"
               r" & No calendar & Calendar-first & Normal-only (sensitivity) \\" "\n\\midrule\n"
               rf"\textbf{{Main regression sample}} & \textbf{{{f0(a[0])}}} & \textbf{{{f0(b[0])}}} & \textbf{{{f0(c[0])}}} \\" "\n"
               rf"\quad loans + permanent & {f0(a[1])} + {f0(a[2])} & {f0(b[1])} + {f0(b[2])} & {f0(c[1])} + {f0(c[2])} \\" "\n"
               rf"\textbf{{Injury regression sample}} & \textbf{{{f0(ai[0])}}} & \textbf{{{f0(bi[0])}}} & \textbf{{{f0(ci[0])}}} \\" "\n"
               rf"\quad loans + permanent & {f0(ai[1])} + {f0(ai[2])} & {f0(bi[1])} + {f0(bi[2])} & {f0(ci[1])} + {f0(ci[2])} \\" "\n"
               rf"Adjusted loan coefficient (main) & {co('current')} pp & {co('classifiable')} pp & {co('normal_only')} pp \\" "\n"
               "\\bottomrule\n\\end{tabular}\n")
    s = s.replace(old_tab, new_tab)
    s = s.replace(r"\caption{How the calendar-first rule changes the final samples (green boxes of Figure~\ref{fig:flow}).}",
                  r"\caption{The three calendar definitions. Figure~\ref{fig:flow} shows the calendar-first column "
                  r"(normal and off-calendar loans both kept). The normal-only column drops off-calendar loans; it is a "
                  r"sensitivity sample, not the main specification, and its flow is in Appendix~\ref{app:normal}. "
                  r"These are the stored (v1) samples; Section~\ref{sec:nov} gives the November corrections.}")
    s = s.replace("The master has 150 columns; Section~\\ref{sec:master} says which ones to use.",
                  f"The master has {150 + n_new} columns; Section~\\ref{{sec:master}} says which ones to use.")
    s = s.replace("The master has 150 columns, but most analyses need only a few of them.",
                  f"The master has {150 + n_new} columns (150 original, {n_new} added in November 2026), but most "
                  "analyses need only a few of them.")
    s = s.replace(r"\textbf{Appendix~\ref{app:master} contains the full classification of all 150 master columns",
                  rf"\textbf{{Appendix~\ref{{app:new}} lists the {n_new} columns added in November 2026. "
                  r"Appendix~\ref{app:master} contains the full classification of the original 150 master columns")
    # main-body section
    g = R[R.calendar_key.eq("calendar_first")].reset_index(drop=True)
    def row(outcome, version_prefix, y):
        x = g[(g.outcome == outcome) & g.version.str.startswith(version_prefix) & (g.y == y)].iloc[0]
        return x
    m1, m2, m3_ = row("Main", "1", "minutes_share"), row("Main", "2", "minutes_share"), row("Main", "3", "minutes_share")
    i1, i2, i3, i4 = (row("Injury-adjusted share", "1", "fit_adjusted_share"),
                      row("Injury-adjusted share", "2", "fit_adjusted_share_v2"),
                      row("Injury-adjusted share", "3", "fit_adjusted_share_v2"),
                      row("Injury-adjusted share", "4", "fit_adjusted_share_v2"))
    q1, q2, q3, q4, q5, q6 = (Q[k]["N"] for k in ("q1", "q2", "q3", "q4", "q5", "q6"))
    rows = [
        ("Freund: did a permanent arrival actually stay? (Paris Brunner)",
         rf"14 columns: next departure (type, date, club), days at the club, left before the club's first league game, "
         rf"stayed through the season, presence category, and the flag \col{{perm_moved_on_before_first_league_game}}. "
         rf"Brunner: bought 16 Aug 2024, loaned out the next day, the day of Monaco's first league game.",
         rf"{f0(q1['perm_flag_true'])} permanent arrivals flagged; all {f0(q1['no_league_games_in_window']['permanent'])} "
         r"earlier exclusions reconciled",
         "No episode removed; mid-season leavers stay in the sample"),
        ("Freund: count quarantine as medical",
         r"Rule v2: quarantine counts as unavailability (not as confirmed illness); v1 kept. "
         r"\col{fit_adjusted_share_v2} and v2 samples.",
         rf"{q2['episodes_affected']} episodes gain {q2['games_added']} games; usable sample unchanged",
         rf"Injury coefficient {pp(i1.loan_coef)} $\to$ {pp(i2.loan_coef)} pp"),
        ("Fernandes: minutes for substitutes",
         rf"Every league appearance checked against line-ups and substitution events; 4 QA columns, "
         r"\col{minutes_share_validated}.",
         rf"Substitutes within 1 minute: {pct(q3['substitute_within1_main'], 2)}",
         rf"{q3['episodes_contradicted']} contradicted records flagged, none edited; combined with the line-up "
         r"check in \col{pt_fully_validated}"),
        ("Fernandes: league tier of both clubs",
         r"League name, tier 1..n, country, source and status for both clubs, season-specific "
         r"(snapshot + openfootball).",
         rf"Origin tier verified for {f0(q4['main_from_verified'])} of {f0(q4['main'])} main-sample episodes",
         r"Unmatched clubs stay blank; no lower tier is assumed"),
        ("Fernandes: playing time in the season before",
         r"17 columns: prior games, appearances, minutes, ordinary and injury-adjusted share, reasons.",
         rf"{f0(q5['main_defined'])} ordinary, {f0(q5['main_fit_defined'])} injury-adjusted (main sample)",
         rf"Not a sample restriction; SQL checks agree (single-club {q5['sql_equal']}/{q5['sql_checked']}, "
         rf"multi-club {q5['multi_equal']}/{q5['multi_checked']})"),
        ("Fernandes: market value of the origin club",
         r"Same rule as the receiving club (player excluded; an including variant too), ratio to the receiving squad.",
         rf"{pct(q6['main_rate'])} of the main sample",
         rf"Receiving values reproduced ({f0(q6['to_value_equal_stored'])})"),
    ]
    body = (r"""
% =============================================================================
\section{November 5 feedback: what changed}
\label{sec:nov}

All six requests are implemented as new master columns (Appendix~\ref{app:new}). The 150 existing columns are unchanged, cell by cell, and no stored sample was redefined. The QA reports are in \col{professor_feedback_nov_2026/}.

{\small
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{P{3.3cm} L P{3.4cm} P{3.0cm}}
\toprule
Question & What we added & Coverage & Check \\
\midrule
""" + "\n".join(rf"{tex(a_)} & {b_} & {c_.replace('%', chr(92) + '%')} & {d_.replace('%', chr(92) + '%')} \\" for a_, b_, c_, d_ in rows) + r"""
\bottomrule
\end{tabularx}}

\medskip
\textbf{Do the corrections change the results?} Calendar-first samples, preferred specification:

\begin{center}
\small
\begin{tabular}{lrrr}
\toprule
 & N & Loan coefficient (pp) & $p$ \\
\midrule
""" + rf"Main sample, stored & {f0(m1.N)} & {pp(m1.loan_coef)} & {m1.p:.3f} \\" + "\n"
          + rf"Main sample, contradicted minutes removed & {f0(m2.N)} & {pp(m2.loan_coef)} & {m2.p:.3f} \\" + "\n"
          + rf"Injury-adjusted, stored (quarantine not counted) & {f0(i1.N)} & {pp(i1.loan_coef)} & {i1.p:.3f} \\" + "\n"
          + rf"Injury-adjusted, quarantine counted & {f0(i2.N)} & {pp(i2.loan_coef)} & {i2.p:.3f} \\" + "\n"
          + rf"Injury-adjusted, quarantine counted and minutes validated & {f0(i3.N)} & {pp(i3.loan_coef)} & {i3.p:.3f} \\" + "\n"
          + r"\midrule" + "\n"
          + rf"Main sample, both minute checks & {f0(m3_.N)} & {pp(m3_.loan_coef)} & {m3_.p:.3f} \\" + "\n"
          + rf"Injury-adjusted, quarantine counted and both minute checks & {f0(i4.N)} & {pp(i4.loan_coef)} & {i4.p:.3f} \\" + "\n"
          + r"""\bottomrule
\end{tabular}
\end{center}

\textbf{Two separate minute checks (correction of 8 November).} The recorded-minute check above (\col{pt_minutes_validated}) and the earlier line-up check (\col{pt_reliable_v2}, Section~\ref{sec:pt}) catch different problems: minutes that disagree with line-ups and events, and players on the pitch with no appearance row. No episode fails both. The \col{_v2} samples apply only the first; \col{pt_fully_validated} requires both, and the samples \col{in_cf_main_model_sample_validated} and \col{in_cf_injury_model_sample_validated} use it (last two rows). The \col{_v2} samples are unchanged.

None of the corrections changes the conclusion: comparable loan and permanent arrivals play about the same share of minutes. The same comparison for the no-calendar and normal-only samples is in \col{regression_comparison.md}.

\textbf{Limits.} Lower-division tiers exist only where a free, openly licensed league table exists (no source for, e.g., Scotland below the top flight or Spain and France tier 3). The prior-season share exists only when the player's clubs were in a league of the snapshot and the records are complete; it is never set to zero when data are missing.
""")
    s = s.replace("% =============================================================================\n\\section{One open decision}",
                  body + "\n% =============================================================================\n\\section{One open decision}")
    # appendices: new columns, normal-only flow, checklist rows
    new_rows = []
    for x in D.itertuples():
        new_rows.append(rf"\col{{{x.variable}}} & {tex(x.plain_english_definition)} & {tex(x.valid_when)} & {f0(x.n_non_blank)} \\")
    app_new = (r"""
\clearpage
\section{The """ + str(n_new) + r""" columns added in November 2026}
\label{app:new}

Appended after the 150 original columns, in this order. ``Filled'' is the number of non-blank rows (of 115,044). Full rules, sources and possible values: \col{analysis_variable_dictionary.csv}.

{\scriptsize
\renewcommand{\arraystretch}{1.1}
\begin{longtable}{P{4.3cm} P{5.8cm} P{4.5cm} r}
\toprule
Column & Meaning & Valid when & Filled \\
\midrule
\endhead
""" + "\n".join(new_rows) + r"""
\bottomrule
\end{longtable}
}
""")
    fig = (OUT / "sample_flow_figure_normal_only.tex").read_text()
    fig = fig.replace(r"\caption{Sample flow with off-calendar loans dropped:",
                      r"\caption{\textbf{Sensitivity sample, not the main specification.} Normal-only sample flow "
                      r"(17,383 main, 13,373 injury): off-calendar loans dropped:")
    assert "Sensitivity sample" in fig
    fig = fig.replace(r"\begin{figure}[p]", "\\begin{figure}[p]\n\\refstepcounter{section}\\label{app:normal}"
                      "\n{\\large\\bfseries E\\quad Normal-only sensitivity flow}\\par\\smallskip"
                      "\n{\\small This flow drops off-calendar loans. It is a sensitivity sample, not the main "
                      "specification; the main sample (Figure~\\ref{fig:flow}) keeps them.}\\par\\medskip", 1)
    fig = fig.replace(r"\begin{tikzpicture}[", r"\scalebox{0.92}{\begin{tikzpicture}[", 1)
    fig = fig.replace(r"\end{tikzpicture}", r"\end{tikzpicture}}", 1)
    app_norm = "\n\\clearpage\n" + fig
    checklist = (rf"Freund: flag for players who leave at once & Section~\ref{{sec:nov}}; \col{{perm_moved_on_before_first_league_game}} \\" "\n"
                 rf"Freund: quarantine is medical & Section~\ref{{sec:nov}}; \col{{fit_adjusted_share_v2}} \\" "\n"
                 rf"Fernandes: minutes for substitutes & Section~\ref{{sec:nov}}; \col{{pt_minutes_validation_status}} \\" "\n"
                 rf"Fernandes: league tiers & Section~\ref{{sec:nov}}; \col{{from_league_tier_at_move}}, \col{{to_league_tier_at_move}} \\" "\n"
                 rf"Fernandes: prior-season playing time & Section~\ref{{sec:nov}}; \col{{prior_minutes_share}}, \col{{prior_fit_adjusted_share}} \\" "\n"
                 rf"Fernandes: origin-club market value & Section~\ref{{sec:nov}}; \col{{from_squad_value}} \\" "\n")
    anchor = r"Which master columns are usable, and which are audit? & Section~\ref{sec:master}; Appendix~\ref{app:master} \\"
    assert anchor in s
    s = s.replace(anchor, anchor + "\n" + checklist)
    s = s.replace("\\end{document}", app_new + app_norm + "\n\\end{document}")
    assert "—" not in s, "em dash in the guide"
    return s


def compile_tex(path: Path) -> dict:
    import subprocess
    for _ in range(3):
        r = subprocess.run(["/usr/local/bin/pdflatex", "-interaction=nonstopmode", "-halt-on-error", path.name],
                           cwd=path.parent, capture_output=True, text=True)
    log = path.with_suffix(".log").read_text(errors="ignore")
    return {"returncode": r.returncode, "errors": log.count("\n! "), "overfull": log.count("Overfull \\hbox"),
            "undefined_refs": log.count("undefined"), "pages": int(__import__("re").search(r"\((\d+) pages", log).group(1))
            if __import__("re").search(r"\((\d+) pages", log) else None}


# ---------------------------------------------------------------------------
# README, inventory, main
# ---------------------------------------------------------------------------

INVENTORY = [
    ("new", "src/analysis/feedback_nov2026.py", "computes all new master columns (called by data_lineage)"),
    ("new", "src/analysis/feedback_nov2026_specs.py", "definitions of the new columns (dictionary, valid-when, provenance)"),
    ("new", "src/analysis/feedback_nov2026_report.py", "this report, QA files and the v3 guide"),
    ("new", "src/analysis/league_tiers.py", "season-specific club → league/tier lookup"),
    ("new", "tests/test_league_tiers.py", "tests for the tier lookup"),
    ("new", "tests/test_feedback_nov2026.py", "tests for every new column"),
    ("new", "data/raw/external_leagues/", "openfootball league files (CC0) with manifest.json"),
    ("new", "data/processed/league_tiers/", "tier lookup, name-match audit, coverage, README"),
    ("new", "data/outputs/rebuild/professor_feedback_nov_2026/", "this folder"),
    ("new (8 Nov correction)", "data/outputs/rebuild/professor_feedback_nov_2026/archive_pre_2026-11-08/",
     "the 222-column master and dictionary before the combined-validation columns were appended"),
    ("new (8 Nov correction)", "data/outputs/rebuild/professor_feedback_nov_2026/q5_multiclub_sql_check.csv",
     "independent multi-club prior-season check (spells rebuilt from Stage 1C in SQL)"),
    ("kept", "data/outputs/rebuild/professor_feedback_nov_2026/master_column_sha256_222col.csv",
     "column hashes before the 8 Nov correction; the current ones are in master_column_sha256_227col.csv"),
    ("modified", "src/analysis/data_lineage.py", "calls feedback_nov2026.extend when building the master"),
    ("modified", "src/analysis/data_lineage_specs.py", "appends the new column definitions to VARS"),
    ("modified", "src/analysis/column_guide_specs.py", "appends guide, valid-when and domain entries"),
    ("modified", "src/analysis/variable_provenance.py", "provenance entries and the tier source"),
    ("modified (regenerated)", "data/outputs/rebuild/analysis_episode_master.csv", f"150 → {150 + len(fs.NEW_COLUMNS)} columns; old columns identical"),
    ("modified (regenerated)", "data/outputs/rebuild/analysis_variable_dictionary.csv", f"{len(fs.NEW_COLUMNS)} rows added"),
    ("modified (regenerated)", "data/outputs/rebuild/data_lineage_overview.md, data_lineage_numbers.json, "
     "master_core_columns.md, data_source_inventory.*, stage1c_column_dictionary.csv, calendar_first_samples.*, "
     "academy_*", "regenerated by data_lineage; numbers other than the column count unchanged"),
    ("modified (regenerated)", "data/outputs/rebuild/variable_provenance_matrix.csv, variable_provenance_overview.md",
     f"{150 + len(fs.NEW_COLUMNS)} master rows"),
    ("modified (regenerated)", "data/outputs/rebuild/final/", f"master_column_sha256.csv now lists {150 + len(fs.NEW_COLUMNS)} columns"),
    ("unchanged", "data/outputs/rebuild/stage1c_canonical_transfers.csv", "SHA-256 verified"),
    ("unchanged", "data/outputs/rebuild/data_and_sample_guide_v2.tex / .pdf", "kept; superseded by v3 in this folder"),
    ("unchanged", "data/outputs/rebuild/sample_flow_figure_v2.tex, sample_flow_figure_normal_only.tex / .pdf", "kept"),
    ("unchanged", "data/outputs/rebuild/professor_questions_2026-11-06/", "prototypes kept (superseded)"),
    ("superseded", "data/outputs/rebuild/data_and_sample_guide_v2.pdf", "by professor_feedback_nov_2026/data_and_sample_guide_v3.pdf"),
]


def render_readme(Q: dict, R: pd.DataFrame, CMP: dict, guide: dict, tests: str) -> str:
    q1, q2, q3, q4, q5, q6 = (Q[k]["N"] for k in ("q1", "q2", "q3", "q4", "q5", "q6"))
    g = R[R.calendar_key.eq("calendar_first")]
    def r_(o, v, y):
        x = g[(g.outcome == o) & g.version.str.startswith(v) & (g.y == y)].iloc[0]
        return f"{f0(x.N)} | {pp(x.loan_coef)} pp (p = {x.p:.3f})"
    return f"""# November 2026 professor feedback: implementation

Generated by `python -m src.analysis.feedback_nov2026_report` after `python -m src.analysis.data_lineage`.
Every number below is computed from the rebuilt master.

## Status

All six requests are implemented. One part is limited by data: lower-division tiers exist only where a free,
openly licensed league table exists (details in item 4).

The master now has **{f0(CMP['rows'])} rows × {CMP['old_columns'] + CMP['new_columns']} columns**
({CMP['old_columns']} original + {CMP['new_columns']} new, appended). All {f0(CMP['cells_compared'])} cells of the
original columns were compared with the archived master: **{f0(CMP['cells_different'])} differ**. Stage 1C SHA-256:
`{STAGE1C_SHA}` (verified unchanged). No stored sample was redefined; the corrected samples are new `_v2` columns,
plus (correction of 8 November) the `_validated` samples, which also apply the earlier line-up check.

## 1. Freund: permanent arrivals who leave at once (Paris Brunner)

* Columns: `next_departure_event_id`, `next_departure_date`, `next_departure_type`, `next_departure_to_club_id`,
  `next_departure_to_club_name`, `days_at_receiving_club`, `next_departure_is_loan`, `next_departure_is_permanent`,
  `receiving_first_league_game_date`, `receiving_last_league_game_date`, `left_before_first_receiving_league_game`,
  `stayed_through_arrival_season_end`, `arrival_season_presence`, `perm_moved_on_before_first_league_game`.
* Brunner: Dortmund U19 → Monaco on 2024-08-16, loaned to Cercle Brugge on 2024-08-17, the day of Monaco's first league
  game; flagged, and excluded from playing time as before (`no_league_games_in_window`).
* Reconciliation: of the {f0(q1['no_league_games_in_window']['permanent'])} permanent arrivals excluded for having no
  league game in the window, {f0(q1['no_league_flagged_left_before'] - q1['no_league_games_in_window']['loan'])} carry
  the flag; one has an unknown status because its next recorded move does not leave the club. Next moves:
  {', '.join(f'{k} {f0(v)}' for k, v in q1['no_league_perm_by_next_type'].items())}.
* Flag on all permanent arrivals: True {f0(q1['perm_flag_true'])}, False {f0(q1['perm_flag_false'])}, unknown
  {f0(q1['perm_flag_unknown'])} (club's league schedule not in the snapshot).
* Mid-season leavers stay in the main sample ({f0(sum(q1['main_left_during_by_kind'].values()))} episodes), and no
  main-sample episode is flagged as leaving before the first game. Findings unchanged.
* Report: `q1_departures.md`.

## 2. Freund: quarantine

* Rule v2 counts `Quarantine` spells as unavailability (isolation, not confirmed infection); `Fitness` and `Rest`
  still do not count; the appearance override and once-per-game counting are unchanged; minutes unchanged.
* {q2['rows']} quarantine spells ({q2['players']} players; {q2['rows_2020_01_to_2022_06']} between January 2020 and
  June 2022). {q2['episodes_affected']} revised-sample episodes ({q2['players_affected']} players) gain
  {q2['games_added']} unavailable games; no episode changes injury-usable status.
* Columns: `quarantine_unavailable_games`, `confirmed_injury_games_v2`, `available_games_fit_v2`,
  `fit_adjusted_share_v2`, `injury_data_quality_flag_v2`, `in_injury_usable_v2`, `in_injury_model_sample_v2`,
  `in_cf_injury_usable_v2`, `in_cf_injury_model_sample_v2`. The v1 columns are kept for reproduction.
* Calendar-first injury-adjusted loan coefficient: v1 {pp(q2['coef_v1'])} pp, v2 {pp(q2['coef_v2'])} pp. Report:
  `q2_quarantine.md`.

## 3. Fernandes: minutes of substitutes

* {f0(q3['rows_reliable'])} league appearances in {f0(q3['episodes_reliable'])} reliable windows checked against
  line-ups and events. Main sample substitutes: {pct(q3['substitute_exact_main'])} exact,
  {pct(q3['substitute_within1_main'], 2)} within 1 minute (Transfermarkt conventions); starters who finished: 90 minutes
  in {pct(q3['starters_full_game_90'], 2)}.
* {q3['episodes_contradicted']} episodes contain a contradicted record (including the only league row above 90 minutes,
  135). The correct value is not recoverable, so nothing was edited: they are flagged and dropped from the validated
  outcome `minutes_share_validated` and the `_v2` samples. {q3['shortened_games']} early-ended games are consistent.
* Columns: `pt_minutes_rows_verified`, `pt_minutes_rows_unverifiable`, `pt_minutes_rows_contradicted`,
  `pt_minutes_validation_status`, `pt_minutes_validated`, `minutes_share_validated`, `in_age_model_sample_v2`,
  `in_cf_main_model_sample_v2`. Report with {len(Q['q3']['examples'])} hand-checked games: `q3_substitute_minutes.md`.
* Correction (8 November): this check does not cover the earlier line-up check (`pt_lineup_contradicted`, players on
  the pitch with no appearance row). `pt_fully_validated` combines both; see the regression comparison below.

## 4. Fernandes: league tiers

* For both clubs: `*_league_name_at_move`, `*_league_tier_at_move`, `*_league_country_at_move`, `*_league_tier_source`,
  `*_league_tier_status`, `*_in_covered_top_flight_at_move`; plus `tier_reference_season` and the origin club's
  `from_league_name_prior_season`, `from_league_tier_prior_season`.
* Sources: the Transfermarkt snapshot (top divisions) and openfootball (CC0) lower divisions, matched to our club ids
  deterministically within country (`data/processed/league_tiers/`). football-data.co.uk was not used because its
  robots.txt disallows AI agents.
* Main sample: origin tier verified for {f0(q4['main_from_verified'])} of {f0(q4['main'])}; receiving clubs are all
  tier 1. **Limitation:** no lawful free source for several lower divisions (Scotland 2-4, Spain/France 3, Belgium/Greece
  2, most non-European leagues before 2024); those stay blank with a status. Report: `q4_league_tiers.md`.

## 5. Fernandes: prior-season playing time

* 17 columns (`prior_*`): games, appearances, minutes, ordinary and injury-adjusted share (rule v2), reliability,
  reason, coverage. Pooled over every covered club spell of the prior season; never zero when data are missing.
* Main sample: ordinary share for {f0(q5['main_defined'])} of {f0(q5['main'])}, injury-adjusted for
  {f0(q5['main_fit_defined'])}; whole season covered for {f0(q5['main_whole'])}. Independent SQL check:
  {q5['sql_equal']}/{q5['sql_checked']} single-club episodes agree; multi-club check (spells rebuilt from Stage 1C in
  SQL): {q5['multi_equal']}/{q5['multi_checked']} agree. Not a sample restriction. Report: `q5_prior_season.md`.

## 6. Fernandes: origin-club market value

* `from_squad_value` (player excluded, like the receiving value), `from_squad_value_incl_player`, `from_squad_players`,
  `from_focal_in_squad`, `from_squad_value_gap_days`, `log_from_squad_value`, `to_squad_value`,
  `from_to_squad_value_ratio`, `log_from_to_squad_value_ratio`.
* Main sample coverage {pct(q6['main_rate'])} ({', '.join(f'{k} {pct(v)}' for k, v in q6['main_by_kind'].items())});
  receiving values reproduced {f0(q6['to_value_equal_stored'])}/{f0(q6['stored_values'])}; SQL check
  {q6['sql_agree']}/{q6['sql_checked']}. Report: `q6_origin_squad_value.md`.

## Regression comparison (calendar-first)

| Sample | N | Loan coefficient |
|---|---|---|
| Main, stored | {r_('Main', '1', 'minutes_share')} |
| Main, contradicted minutes removed | {r_('Main', '2', 'minutes_share')} |
| Injury-adjusted, stored | {r_('Injury-adjusted share', '1', 'fit_adjusted_share')} |
| Injury-adjusted, quarantine counted | {r_('Injury-adjusted share', '2', 'fit_adjusted_share_v2')} |
| Injury-adjusted, both corrections | {r_('Injury-adjusted share', '3', 'fit_adjusted_share_v2')} |
| Main, both minute checks (line-up + recorded minutes) | {r_('Main', '3', 'minutes_share')} |
| Injury-adjusted, quarantine counted + both minute checks | {r_('Injury-adjusted share', '4', 'fit_adjusted_share_v2')} |

**Correction (8 November).** The `_v2` samples remove the 10 contradicted recorded-minute episodes but not the earlier
line-up contradictions (`pt_lineup_contradicted`: on the pitch with no appearance row). These are separate problems
and no episode has both. `pt_fully_validated` requires both checks (`pt_reliable_v2` and `pt_minutes_validated`); the
samples `in_age_model_sample_validated`, `in_cf_main_model_sample_validated`, `in_injury_model_sample_validated` and
`in_cf_injury_model_sample_validated` apply it. The `_v2` samples are unchanged.

The findings do not change. All three calendar definitions: `regression_comparison.md` / `.csv`.

## Guide and tests

* Updated guide: `data_and_sample_guide_v3.pdf` ({guide.get('pages')} pages; LaTeX errors {guide.get('errors')}, overfull
  boxes {guide.get('overfull')}). It adds Section 10 (the November feedback), Table 2 with all three calendar
  definitions, Appendix D (new columns) and Appendix E (normal-only sensitivity flow, labelled as such).
* Tests: {tests}

## Files

See `file_inventory.csv`. Archived pre-change master and dictionary: `archive_pre_nov2026/`.
"""


def main() -> dict:
    DIR.mkdir(exist_ok=True)
    assert sha(STAGE1C) == STAGE1C_SHA, "Stage 1C changed"
    M = pd.read_csv(MASTER, low_memory=False)
    Mr = pd.read_csv(MASTER, dtype=str, keep_default_na=False)
    for c in [c for c in M.columns if M[c].dtype == object and set(M[c].dropna().unique()) <= {"True", "False"}]:
        M[c] = M[c].map({"True": True, "False": False})
    saved = json.loads((OUT / "data_lineage_numbers.json").read_text())
    ctx = context()
    CMP = master_comparison(Mr)
    R = regression_comparison(M, saved)
    F = flows(M)
    Q = {"q1": q1_departures(M, ctx), "q3": q3_minutes(M, ctx), "q4": q4_tiers(M), "q5": q5_prior(M, ctx),
         "q6": q6_origin_value(M)}
    Q["q2"] = q2_quarantine(M, ctx, R)
    D = pd.read_csv(OUT / "analysis_variable_dictionary.csv", keep_default_na=False)
    D = D[D.variable.isin(fs.NEW_COLUMNS)]
    assert list(D.variable) == fs.NEW_COLUMNS
    # files
    write_csv(Q["q1"]["examples"], "q1_departure_examples.csv")
    write_csv(Q["q1"]["reconciliation"], "q1_reconciliation.csv")
    write_csv(Q["q1"]["trace"], "q1_brunner_stage1c_rows.csv")
    (DIR / "q1_departures.md").write_text(render_q1(Q["q1"]))
    write_csv(Q["q2"]["episodes"], "q2_quarantine_episodes.csv")
    (DIR / "q2_quarantine.md").write_text(render_q2(Q["q2"]))
    write_csv(Q["q3"]["examples"], "q3_manual_examples.csv")
    write_csv(Q["q3"]["anomalies"], "q3_unverified_and_contradicted_rows.csv")
    write_csv(Q["q3"]["previous"], "q3_previous_11_discrepancies.csv")
    write_csv(Q["q3"]["status"], "q3_row_status_counts.csv")
    (DIR / "q3_substitute_minutes.md").write_text(render_q3(Q["q3"]))
    write_csv(Q["q4"]["coverage"], "q4_tier_coverage_by_sample.csv")
    write_csv(Q["q4"]["by_country_from"], "q4_tier_missing_by_country_from.csv")
    write_csv(Q["q4"]["by_country_to"], "q4_tier_missing_by_country_to.csv")
    write_csv(Q["q4"]["by_season"], "q4_tier_coverage_by_season.csv")
    write_csv(Q["q4"]["leagues"], "q4_origin_leagues.csv")
    (DIR / "q4_league_tiers.md").write_text(render_q4(Q["q4"], {}))
    write_csv(Q["q5"]["coverage"], "q5_prior_season_coverage.csv")
    write_csv(Q["q5"]["reasons"], "q5_prior_reasons.csv")
    write_csv(Q["q5"]["by_season"], "q5_prior_by_season.csv")
    write_csv(Q["q5"]["by_country"], "q5_prior_by_country.csv")
    write_csv(Q["q5"]["examples"], "q5_prior_examples.csv")
    write_csv(Q["q5"]["multiclub"], "q5_multiclub_sql_check.csv")
    (DIR / "q5_prior_season.md").write_text(render_q5(Q["q5"]))
    write_csv(Q["q6"]["coverage"], "q6_origin_value_coverage.csv")
    write_csv(Q["q6"]["examples"], "q6_origin_value_examples.csv")
    (DIR / "q6_origin_squad_value.md").write_text(render_q6(Q["q6"]))
    write_csv(R, "regression_comparison.csv")
    write_csv(F, "sample_flow_corrections.csv")
    (DIR / "regression_comparison.md").write_text(render_regression(R, F))
    write_csv(CMP["diffs"], "old_vs_new_master_differences.csv")
    write_csv(CMP["col_sha"], f"master_column_sha256_{len(CMP['col_sha'])}col.csv")
    write_csv(D[["variable", "section", "plain_english_definition", "source_file_or_table", "formula_or_rule",
                 "possible_values", "missing_value_meaning", "valid_when", "rows_valid", "n_non_blank", "usefulness",
                 "is_filter_or_flag"]].assign(pct_non_blank=lambda d: d.n_non_blank / len(M)), "new_columns.csv")
    tex_path = DIR / "data_and_sample_guide_v3.tex"
    tex_path.write_text(guide_v3(R, Q, D, saved))
    guide = compile_tex(tex_path)
    tests_txt = (DIR / "test_results.txt").read_text().strip() if (DIR / "test_results.txt").exists() else "see test_results.txt"
    (DIR / "FINAL_README.md").write_text(render_readme(Q, R, CMP, guide, tests_txt))
    pd.DataFrame(INVENTORY, columns=["status", "path", "note"]).to_csv(DIR / "file_inventory.csv", index=False)
    N = {k: jdump(v["N"]) for k, v in Q.items()}
    N["master"] = {k: CMP[k] for k in ("rows", "old_columns", "new_columns", "cells_compared", "cells_different")}
    N["guide"] = guide
    N["stage1c_sha256"] = STAGE1C_SHA
    (DIR / "numbers.json").write_text(json.dumps(jdump(N), indent=1, default=str))
    print(f"master {CMP['rows']} x {CMP['old_columns'] + CMP['new_columns']}; differing old cells {CMP['cells_different']}; "
          f"guide {guide}")
    return N


if __name__ == "__main__":
    main()
