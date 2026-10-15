"""Playing-time sample redesign after Professor Freund's feedback: no season-fraction rule; onward moves audited.

    python -m src.analysis.playing_time_sample_redesign

Descriptive only: no regression is run. The episodes, windows and outcome come from `playing_time_join.compute`
unchanged (and are checked against its saved reliable and primary counts). The outcome stays the player's league
minutes divided by the receiving club's league minutes while he is there: from arrival to the earlier of the
season's last league game and the day before his next recorded move. Writes new files only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from . import playing_time_join as ptj
from .daniel_scope_final import f0, md_table, pc
from .loan_episodes import ROOT, _sorted
from .playing_time_descriptives import COLOR, GRID, INK, LABEL, MUTED, _style, describe, diff
from ..stage1c.policy import PLACEHOLDER_CLUBS

OUT = ROOT / "data" / "outputs" / "rebuild"
REPORT_MD = OUT / "playing_time_sample_redesign.md"
WATERFALL_CSV = OUT / "playing_time_sample_waterfall.csv"
FOLLOWON_CSV = OUT / "playing_time_permanent_followon_audit.csv"
ARRIVAL_CSV = OUT / "playing_time_arrival_month.csv"
SENS_CSV = OUT / "playing_time_sample_sensitivity.csv"
NUMBERS_JSON = OUT / "playing_time_sample_redesign_numbers.json"
FIGS = {"hist": "figures/playing_time_redesign_fig1_share_distribution.png",
        "arrival": "figures/playing_time_redesign_fig2_arrival_timing.png",
        "zeros": "figures/playing_time_redesign_fig3_permanent_zeros.png"}
WRITES = (REPORT_MD, WATERFALL_CSV, FOLLOWON_CSV, ARRIVAL_CSV, SENS_CSV, NUMBERS_JSON) + tuple(OUT / f for f in FIGS.values())
KINDS = ("loan", "permanent")
ONWARD_DAYS = (0, 3, 7, 14, 30, 60)
MIN_GAMES = (3, 5, 10)
FOLLOW = ("A stays through season end", "B permanent onward before season end", "C loaned onward before season end",
          "D released / without club / retired", "E no next move recorded", "F other")
PERMANENT_TYPES = ("permanent_transfer", "undisclosed_transfer", "free_transfer", "no_fee_shown")
LAGS = ((0, 0, "same day"), (1, 3, "1–3 days"), (4, 7, "4–7 days"), (8, 14, "8–14 days"), (15, 30, "15–30 days"),
        (31, 60, "31–60 days"), (61, 10 ** 6, "> 60 days"))
GROUPS = ("preseason: before B's first league game", "in season, Jun–Sep", "Oct–Nov", "Dec–Jan", "Feb or later")


# ---------------------------------------------------------------------------
# Next move and season bounds
# ---------------------------------------------------------------------------

def attach_next_move(E: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The player's next substantive Stage 1C row after each episode, by the join's own rule (internal and draft
    rows skipped), so its date is exactly the join's `departure_date`."""
    d = _sorted(rows).reset_index(drop=True)
    pid = d.player_id.to_numpy()
    skip = d.is_internal_move.to_numpy().astype(bool) | np.isin(d.transfer_type_normalized.to_numpy(), ptj.SKIP_TYPES)
    pos = pd.Series(np.arange(len(d)), index=d.event_id.to_numpy())
    nxt = []
    for eid, p in zip(E.event_id, E.player_id):
        j = pos[eid] + 1
        while j < len(d) and pid[j] == p and skip[j]:
            j += 1
        nxt.append(j if j < len(d) and pid[j] == p else -1)
    idx = np.array(nxt)
    ok = idx >= 0
    E = E.copy()
    for c, out in (("transfer_type_normalized", "next_type"), ("to_club_name", "next_to_club"),
                   ("to_club_id", "next_to_club_id"), ("_date", "next_date"), ("event_id", "next_event_id")):
        vals = d[c].to_numpy()[np.where(ok, idx, 0)]
        E[out] = pd.Series(vals, index=E.index).where(ok)
    E["next_date"] = pd.to_datetime(E.next_date)
    same = E.next_date.eq(E.departure_date) | (E.next_date.isna() & E.departure_date.isna())
    assert same.all(), "next move does not reproduce the join's departure date"
    return E


def season_bounds(con) -> pd.DataFrame:
    s = con.sql("""with x as (select home_club_id club_id, season, date from games where competition_type = 'domestic_league'
                              union all
                              select away_club_id, season, date from games where competition_type = 'domestic_league')
                   select cast(club_id as bigint) b_club_id, cast(season as int) arrival_season,
                          min(date) season_first_game, max(date) season_last_game from x group by 1, 2""").df()
    s["arrival_season"] = s.arrival_season.astype("Int64")
    for c in ("season_first_game", "season_last_game"):
        s[c] = pd.to_datetime(s[c])
    return s


def follow_on(E: pd.DataFrame) -> pd.DataFrame:
    E = E.copy()
    E["days_to_next"] = (E.next_date - E.move_date).dt.days
    t = E.next_type.fillna("")
    before = E.next_date.notna() & (E.next_date <= E.season_last_game)
    E["follow_on"] = np.select(
        [E.next_date.isna(), ~before, t.eq("loan"), E.next_to_club.isin(PLACEHOLDER_CLUBS), t.isin(PERMANENT_TYPES)],
        [FOLLOW[4], FOLLOW[0], FOLLOW[2], FOLLOW[3], FOLLOW[1]], default=FOLLOW[5])
    E["onward_lag"] = pd.Series(pd.NA, index=E.index, dtype="object")
    for lo, hi, lab in LAGS:
        E.loc[before & E.days_to_next.between(lo, hi), "onward_lag"] = lab
    E["left_before_season_end"] = before
    E["arrival_group"] = arrival_group(E)
    return E


def arrival_group(E: pd.DataFrame) -> pd.Series:
    """Calendar-adjusted: 'preseason' is relative to B's own first league game; in-season arrivals by month."""
    pre = E.games_remaining_at_arrival.eq(E.season_games)
    m = E.move_date.dt.month
    return pd.Series(np.select([pre, m.isin([6, 7, 8, 9]), m.isin([10, 11]), m.isin([12, 1]), m.isin([2, 3, 4, 5])],
                               GROUPS, default=""), index=E.index)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def masks(E: pd.DataFrame) -> dict[str, pd.Series]:
    rel = E.reliable
    ret = (E.kind == "permanent") | (E.loan_ending == "return to lender")
    first = ~E.prior_spell_at_b
    revised = rel & ret & first
    out = {"old primary (two-thirds rule + first spell + loan return)": E.primary_sample,
           "reliable, no restriction": rel,
           "reliable + loan return (all spells)": rel & ret,
           "REVISED PRIMARY: reliable + loan return + first spell": revised}
    for x in ONWARD_DAYS:
        out[f"revised, staying: drop next move within {x} days" if x else
            "revised, staying: drop next move on the arrival day"] = revised & ~(E.left_before_season_end
                                                                                & E.days_to_next.le(x))
    out["revised, drop any onward loan before season end (conditions on the outcome)"] = \
        revised & ~E.follow_on.eq(FOLLOW[2])
    for k in MIN_GAMES:
        out[f"revised, at least {k} league games available"] = revised & E.available_games.ge(k)
    out["revised + two-thirds rule (= old primary)"] = revised & E.arrived_full_season
    assert (out["revised + two-thirds rule (= old primary)"] == E.primary_sample).all()
    return out


def pair(S: pd.DataFrame) -> dict:
    rec = {}
    for k in KINDS:
        g = S[S.kind == k]
        d = describe(g.minutes_share)
        rec.update({f"{k}_{x}": d.get(x) for x in ("N", "mean", "median", "p25", "p75", "zero", "below_10", "above_50")})
        rec[f"{k}_weighted"] = float(g.minutes.sum() / g.available_minutes.sum()) if len(g) else np.nan
    dd = diff(S.loc[S.kind == "loan", "minutes_share"], S.loc[S.kind == "permanent", "minutes_share"])
    rec.update({"gap": dd["mean_diff"], "gap_lo": dd["mean_diff_lo"], "gap_hi": dd["mean_diff_hi"],
                "median_gap": dd["median_diff"], "weighted_gap": rec["loan_weighted"] - rec["permanent_weighted"]})
    return rec


def waterfall(E: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in KINDS:
        g = E[E.kind == k]
        rows.append({"block": "upstream", "kind": k, "step": "candidate episodes", "remaining": len(g), "removed": 0})
        for r in ptj.REASONS:
            n = int((g.exclusion_reason == r).sum())
            rows.append({"block": "upstream", "kind": k, "step": f"excluded: {ptj.REASONS[r]}", "remaining": np.nan,
                         "removed": n})
        R = g[g.reliable]
        rows.append({"block": "upstream", "kind": k, "step": "reliable playing-time outcome", "remaining": len(R),
                     "removed": len(g) - len(R)})
        # revised path, one exclusion category per row
        cur = R
        steps = [("no league match opportunity after arrival", lambda c: c.available_games.lt(1)),
                 ("ambiguous loan ending (not a return to the lender)",
                  lambda c: c.kind.eq("loan") & c.loan_ending.ne("return to lender")),
                 ("repeat spell at the receiving club", lambda c: c.prior_spell_at_b)]
        for lab, f in steps:
            m = f(cur)
            rows.append({"block": "revised", "kind": k, "step": f"excluded: {lab}", "remaining": len(cur) - int(m.sum()),
                         "removed": int(m.sum())})
            cur = cur.loc[~m]
        for lab, n in (("excluded: same-day / near-immediate onward movement (kept; flagged, see staying sample)", 0),
                       ("excluded: arrival timing / share of season remaining (rule removed)", 0),
                       ("excluded: other reason", 0)):
            rows.append({"block": "revised", "kind": k, "step": lab, "remaining": len(cur), "removed": n})
        rows.append({"block": "revised", "kind": k, "step": "REVISED PRIMARY SAMPLE", "remaining": len(cur), "removed": 0})
        cur = R
        for lab, f in (("excluded: fewer than two-thirds of B's league games left at arrival", lambda c: ~c.arrived_full_season),
                       ("excluded: repeat spell at the receiving club", lambda c: c.prior_spell_at_b),
                       ("excluded: ambiguous loan ending", lambda c: c.kind.eq("loan") & c.loan_ending.ne("return to lender"))):
            m = f(cur)
            rows.append({"block": "old", "kind": k, "step": lab, "remaining": len(cur) - int(m.sum()), "removed": int(m.sum())})
            cur = cur.loc[~m]
        assert len(cur) == int((E.primary_sample & E.kind.eq(k)).sum())
        rows.append({"block": "old", "kind": k, "step": "OLD PRIMARY SAMPLE", "remaining": len(cur), "removed": 0})
    W = pd.DataFrame(rows)
    W["remaining"] = W.remaining.astype("Int64")
    return W


def followon_table(E: pd.DataFrame, rev: pd.Series) -> pd.DataFrame:
    P = E[E.kind == "permanent"]
    scopes = {"all permanent arrivals into a covered league season": P[P.matched],
              "excluded upstream: no league game of B before the player left": P[P.exclusion_reason == "no_league_games_in_window"],
              "reliable outcome": P[P.reliable],
              "revised primary sample": P[rev.reindex(P.index, fill_value=False)],
              "revised primary sample, zero playing-time share": P[rev.reindex(P.index, fill_value=False) & P.minutes_share.eq(0)],
              "revised primary sample, positive share": P[rev.reindex(P.index, fill_value=False) & P.minutes_share.gt(0)]}
    rows = []
    for sc, g in scopes.items():
        for f in FOLLOW:
            n = int((g.follow_on == f).sum())
            rows.append({"scope": sc, "scope_n": len(g), "follow_on": f, "lag": "", "n": n, "share": n / len(g) if len(g) else np.nan,
                         "cumulative_n": np.nan})
        c = g[g.follow_on == FOLLOW[2]]
        cum = 0
        for lo, hi, lab in LAGS:
            n = int((c.onward_lag == lab).sum())
            cum += n
            rows.append({"scope": sc, "scope_n": len(g), "follow_on": FOLLOW[2], "lag": lab, "n": n,
                         "share": n / len(g) if len(g) else np.nan, "cumulative_n": cum})
    return pd.DataFrame(rows)


def arrival_table(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, key in (("group", S.arrival_group), ("month", S.move_date.dt.month.astype(str)),
                     ("January only", pd.Series(np.where(S.move_date.dt.month.eq(1), "January", "not January"), index=S.index))):
        for v, g in S.groupby(key):
            rec = {"dimension": dim, "value": v}
            rec.update(pair(g) if g.kind.nunique() == 2 else {f"{k}_N": int((g.kind == k).sum()) for k in KINDS})
            rec["median_games_available_loan"] = float(g.loc[g.kind == "loan", "available_games"].median())
            rec["median_games_available_permanent"] = float(g.loc[g.kind == "permanent", "available_games"].median())
            rows.append(rec)
    T = pd.DataFrame(rows)

    def rank(d, v):
        if d == "group":
            return GROUPS.index(v)
        if d == "month":
            return 100 + int(v)
        return 200 + (v != "January")

    T["_o"] = [rank(d, v) for d, v in zip(T.dimension, T.value)]
    return T.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def repeat_spells(E: pd.DataFrame) -> pd.DataFrame:
    X = E.sort_values(["player_id", "move_date", "event_id"]).copy()
    X["prev_kind_at_b"] = X.groupby(["player_id", "b_club_id"]).kind.shift()
    X["prev_date_at_b"] = X.groupby(["player_id", "b_club_id"]).move_date.shift()
    rep = X[X.reliable & X.prior_spell_at_b & ((X.kind == "permanent") | (X.loan_ending == "return to lender"))].copy()
    rep["repeat_type"] = np.select(
        [rep.preceded_by_loan_to_b & rep.kind.eq("permanent"), rep.preceded_by_loan_to_b & rep.kind.eq("loan"),
         rep.prev_kind_at_b.eq("loan"), rep.prev_kind_at_b.eq("permanent")],
        ["permanent move right after a loan to the same club (conversion)",
         "loan right after a loan to the same club (re-loan)",
         "back after an earlier loan there, other moves in between",
         "back after an earlier permanent spell there"], default="other")
    rep["gap_days"] = (rep.move_date - rep.prev_date_at_b).dt.days
    return rep


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(u=None) -> tuple[dict, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    u = u or loan_scope.universe()
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        J, E, _ = ptj.compute(u, con)
        sb = season_bounds(con)
    finally:
        con.close()
    saved = json.loads(ptj.NUMBERS_JSON.read_text()) if ptj.NUMBERS_JSON.exists() else None
    N: dict = {"reliable": {k: int((E.reliable & E.kind.eq(k)).sum()) for k in KINDS},
               "old_primary": {k: int((E.primary_sample & E.kind.eq(k)).sum()) for k in KINDS}}
    if saved is not None:
        assert N["old_primary"] == {k: int(saved["primary"][k]) for k in KINDS}, "old primary does not reproduce"
        assert N["reliable"] == {k: int(saved["waterfall"][k]["reliable"]) for k in KINDS}
    E = attach_next_move(E, u.rows)
    E = E.merge(sb, on=["b_club_id", "arrival_season"], how="left")
    E = follow_on(E)
    M = masks(E)
    rev = M["REVISED PRIMARY: reliable + loan return + first spell"]
    E["revised_primary"] = rev

    # ---- waterfall
    W = waterfall(E)
    N["upstream_no_games"] = {k: int(((E.exclusion_reason == "no_league_games_in_window") & E.kind.eq(k)).sum()) for k in KINDS}
    R = E[E.reliable]
    N["two_thirds_removed"] = {k: int((R.kind.eq(k) & ~R.arrived_full_season).sum()) for k in KINDS}
    tt = R[~R.arrived_full_season]
    N["two_thirds_removed_by_month"] = {k: {int(m): int(v) for m, v in tt[tt.kind == k].move_date.dt.month.value_counts().sort_index().items()}
                                        for k in KINDS}
    N["two_thirds_removed_jan_feb"] = {k: int((tt.kind.eq(k) & tt.move_date.dt.month.isin([1, 2])).sum()) for k in KINDS}
    N["old_steps"] = {k: W[(W.block == "old") & (W.kind == k)][["step", "removed", "remaining"]].to_dict("records") for k in KINDS}
    N["revised_steps"] = {k: W[(W.block == "revised") & (W.kind == k)][["step", "removed", "remaining"]].to_dict("records")
                          for k in KINDS}
    N["revised"] = {k: int((rev & E.kind.eq(k)).sum()) for k in KINDS}

    # ---- opportunity
    S = E[rev]
    N["available_games"] = {k: {"min": int(S[S.kind == k].available_games.min()),
                                "median": float(S[S.kind == k].available_games.median()),
                                "bins": {lab: int(S[(S.kind == k) & S.available_games.between(lo, hi)].shape[0])
                                         for lo, hi, lab in ((1, 1, "1"), (2, 3, "2–3"), (4, 5, "4–5"), (6, 10, "6–10"),
                                                             (11, 20, "11–20"), (21, 60, "21+"))},
                                "zero_by_bin": {lab: float(S[(S.kind == k) & S.available_games.between(lo, hi)].minutes_share.eq(0).mean())
                                                for lo, hi, lab in ((1, 1, "1"), (2, 3, "2–3"), (4, 5, "4–5"), (6, 10, "6–10"),
                                                                    (11, 20, "11–20"), (21, 60, "21+"))}}
                           for k in KINDS}
    few = S[S.available_games <= 3]
    N["few_games"] = {k: {"n": int((few.kind == k).sum()),
                          "left_before_season_end": int((few.kind.eq(k) & few.left_before_season_end).sum()),
                          "onward_loan": int((few.kind.eq(k) & few.follow_on.eq(FOLLOW[2])).sum()),
                          "summer_months": int((few.kind.eq(k) & few.move_date.dt.month.isin([6, 7, 8])).sum())} for k in KINDS}

    # ---- follow-on (Part 3)
    F = followon_table(E, rev)
    N["followon"] = {sc: {r.follow_on if not r.lag else f"{r.follow_on} | {r.lag}": int(r.n) for r in g.itertuples()}
                     for sc, g in F.groupby("scope", sort=False)}
    N["followon_n"] = {sc: int(g.scope_n.iloc[0]) for sc, g in F.groupby("scope", sort=False)}
    Pz = S[(S.kind == "permanent") & S.minutes_share.eq(0)]
    Pa = S[S.kind == "permanent"]
    N["zeros"] = {"permanent_zero": len(Pz), "permanent_n": len(Pa),
                  "onward_loan_within": {x: int((Pz.follow_on.eq(FOLLOW[2]) & Pz.days_to_next.le(x)).sum()) for x in ONWARD_DAYS},
                  "onward_loan_any": int(Pz.follow_on.eq(FOLLOW[2]).sum()),
                  "by_follow": {k: int(v) for k, v in Pz.follow_on.value_counts().items()},
                  "stayed": int(Pz.follow_on.isin([FOLLOW[0], FOLLOW[4]]).sum()),
                  "loan_zero": int(((S.kind == "loan") & S.minutes_share.eq(0)).sum())}
    N["zeros"]["zero_share_if_quick_loans_dropped"] = {
        x: float((len(Pz) - N["zeros"]["onward_loan_within"][x]) / (len(Pa) - int((Pa.follow_on.eq(FOLLOW[2]) & Pa.days_to_next.le(x)).sum())))
        for x in ONWARD_DAYS}
    N["onward_any_move_within"] = {x: {k: int(((S.kind == k) & S.left_before_season_end & S.days_to_next.le(x)).sum())
                                       for k in KINDS} for x in ONWARD_DAYS}

    # ---- sensitivity (Parts 1, 4, 5)
    sens = [{"variant": name, **pair(E[m])} for name, m in M.items()]
    restored = rev & ~E.arrived_full_season
    sens.append({"variant": "restored by dropping the two-thirds rule (revised, fewer than two-thirds left)",
                 **pair(E[restored])})
    N["restored_jan_feb"] = int((restored & E.move_date.dt.month.isin([1, 2])).sum())
    N["restored_n"] = int(restored.sum())
    SENS = pd.DataFrame(sens)
    N["sensitivity"] = SENS.set_index("variant").to_dict(orient="index")

    # ---- arrival timing (Part 6)
    A = arrival_table(S)
    N["arrival"] = {f"{r.dimension}: {r.value}": {c: getattr(r, c) for c in ("loan_N", "permanent_N", "loan_mean", "permanent_mean",
                                                                            "loan_median", "permanent_median", "gap", "gap_lo", "gap_hi")
                                                  if hasattr(r, c)} for r in A.itertuples()}

    # ---- first spell (Part 7)
    rep = repeat_spells(E)
    N["repeat"] = {k: {t: int(v) for t, v in rep[rep.kind == k].repeat_type.value_counts().items()} for k in KINDS}
    N["repeat_mean"] = {k: {t: float(g.minutes_share.mean()) for t, g in rep[rep.kind == k].groupby("repeat_type")} for k in KINDS}
    N["repeat_gap_days"] = {k: {t: float(g.gap_days.median()) for t, g in rep[rep.kind == k].groupby("repeat_type")} for k in KINDS}
    N["repeat_total"] = {k: int((rep.kind == k).sum()) for k in KINDS}
    return N, {"E": E, "W": W, "F": F, "SENS": SENS, "A": A, "S": S, "rep": rep}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(N: dict, S: pd.DataFrame, A: pd.DataFrame) -> None:
    plt = _style()
    (OUT / "figures").mkdir(exist_ok=True)
    sv = N["sensitivity"]["REVISED PRIMARY: reliable + loan return + first spell"]

    # 1. distribution, same layout as the original figure 1
    edges = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.018
    for off, k in ((-w / 2, "loan"), (w / 2, "permanent")):
        s = S.loc[S.kind == k, "minutes_share"]
        ax.bar(-0.03 + off, (s == 0).mean() * 100, width=w, color=COLOR[k])
        h = np.histogram(s[s > 0], bins=edges)[0] / len(s) * 100
        ax.bar(edges[:-1] + 0.025 + off, h, width=w, color=COLOR[k],
               label=f"{LABEL[k]} (N = {sv[f'{k}_N']:,}; mean {sv[f'{k}_mean']:.3f}, median {sv[f'{k}_median']:.2f})")
    ax.set_xticks([-0.03] + list(np.round(np.linspace(0.1, 1, 10), 1)))
    ax.set_xticklabels(["exactly\n0"] + [f"{x:.1f}" for x in np.linspace(0.1, 1, 10)])
    ax.axvline(-0.008, color=MUTED, lw=0.6)
    ax.set_xlim(-0.06, 1.01)
    ax.set_xlabel("Share of the receiving club's league minutes played (window: arrival to season end or departure)")
    ax.set_ylabel("Share of the group (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Revised sample, no season-fraction rule: loans vs permanent transfers (0.05-wide bins)", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["hist"], dpi=160)
    plt.close(fig)

    # 2. mean share by arrival group, with 95% intervals
    G = A[A.dimension == "group"].set_index("value").loc[list(GROUPS)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(G))
    for off, k in ((-0.12, "loan"), (0.12, "permanent")):
        sub = S[S.kind == k]
        m, se = [], []
        for gname in G.index:
            v = sub.loc[sub.arrival_group == gname, "minutes_share"]
            m.append(v.mean())
            se.append(v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan)
        ax.errorbar(x + off, m, yerr=1.96 * np.array(se), fmt="o", ms=8, color=COLOR[k], ecolor=COLOR[k],
                    elinewidth=2, capsize=0, label=LABEL[k])
        for xi, mi, n in zip(x + off, m, G[f"{k}_N"]):
            ax.text(xi, 0.25, f"n={int(n):,}", ha="center", va="bottom", fontsize=8, color=MUTED, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(": ", ":\n") for g in G.index], fontsize=9)
    ax.set_ylim(0.2, 0.6)
    ax.set_ylabel("Mean playing-time share")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Mean playing-time share by arrival timing (revised sample; bars are 95% intervals)", loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / FIGS["arrival"], dpi=160)
    plt.close(fig)

    # 3. zero-share permanent arrivals: what happened next
    z = N["zeros"]
    Pz = S[(S.kind == "permanent") & S.minutes_share.eq(0)]
    cats = [("stayed through season end (A, E)", int(Pz.follow_on.isin([FOLLOW[0], FOLLOW[4]]).sum())),
            ("loaned onward within 14 days", int((Pz.follow_on.eq(FOLLOW[2]) & Pz.days_to_next.le(14)).sum())),
            ("loaned onward after 15–60 days", int((Pz.follow_on.eq(FOLLOW[2]) & Pz.days_to_next.between(15, 60)).sum())),
            ("loaned onward after > 60 days", int((Pz.follow_on.eq(FOLLOW[2]) & Pz.days_to_next.gt(60)).sum())),
            ("sold / moved permanently onward (B)", int(Pz.follow_on.eq(FOLLOW[1]).sum())),
            ("released / other (D, F)", int(Pz.follow_on.isin([FOLLOW[3], FOLLOW[5]]).sum()))]
    assert sum(c for _, c in cats) == z["permanent_zero"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(cats))[::-1]
    ax.barh(y, [c for _, c in cats], height=0.6, color=COLOR["permanent"])
    for yi, (lab, c) in zip(y, cats):
        ax.text(c + 8, yi, f"{c:,} ({100 * c / z['permanent_zero']:.0f}%)", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in cats])
    ax.set_xlim(0, max(c for _, c in cats) * 1.2)
    ax.set_xlabel("Permanent arrivals with zero playing-time share (revised sample)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"What happened next to the {z['permanent_zero']:,} zero-minute permanent arrivals", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["zeros"], dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def p3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pp(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.1f} pp"


def pct(x, d=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def render(N: dict) -> str:
    sv = N["sensitivity"]
    OLD, REV = "old primary (two-thirds rule + first spell + loan return)", "REVISED PRIMARY: reliable + loan return + first spell"
    o, r = sv[OLD], sv[REV]
    z, rl, op = N["zeros"], N["reliable"], N["old_primary"]
    tt, rv = N["two_thirds_removed"], N["revised"]
    fo, fn = N["followon"], N["followon_n"]
    up = N["upstream_no_games"]
    UPS = "excluded upstream: no league game of B before the player left"
    stay14 = sv["revised, staying: drop next move within 14 days"]
    anyloan = sv["revised, drop any onward loan before season end (conditions on the outcome)"]
    jan = N["arrival"].get("January only: January", {})
    dj = N["arrival"].get("group: Dec–Jan", {})
    W = ["# Playing-time sample redesign: no season-fraction rule", "",
         "*Generated by `python -m src.analysis.playing_time_sample_redesign`. Descriptive only; no regression was "
         "run. The episodes, windows and outcome are `playing_time_join`'s, unchanged and checked against its saved "
         "counts. Old outputs are not overwritten. Raw differences are not causal.*", ""]

    # ---- answers
    W += ["## Answers", "",
          f"1. **Why did {f0(rl['loan'])} loans become {f0(op['loan'])}?** The two-thirds rule removed "
          f"{f0(tt['loan'])} ({f0(N['two_thirds_removed_jan_feb']['loan'])} of them January or February arrivals), "
          f"the first-spell rule {f0(N['old_steps']['loan'][1]['removed'])}, "
          f"and the loan-return rule {N['old_steps']['loan'][2]['removed']}. For permanent moves the two-thirds rule "
          f"removed {f0(tt['permanent'])} and the first-spell rule {f0(N['old_steps']['permanent'][1]['removed'])}.",
          f"2. **Which restrictions should go?** The two-thirds rule. The outcome already scales by the minutes "
          f"available while the player is at the club, so a January arrival is compared on the games actually left. "
          f"The first-spell and loan-return rules are about what kind of move it is, not about exposure; the "
          f"evidence (section 7) supports keeping first spell as the primary definition.",
          f"3. **Removing the two-thirds rule:** the revised sample has {f0(rv['loan'])} loans and "
          f"{f0(rv['permanent'])} permanent moves (was {f0(op['loan'])} and {f0(op['permanent'])}). Mean share "
          f"{p3(r['loan_mean'])} vs {p3(r['permanent_mean'])}: raw gap {pp(r['gap'])} "
          f"[{pp(r['gap_lo'])}, {pp(r['gap_hi'])}] (old: {p3(o['loan_mean'])} vs {p3(o['permanent_mean'])}, "
          f"{pp(o['gap'])}).",
          f"4. **Permanent arrivals immediately loaned onward:** almost none are in any playing-time sample, because B "
          f"played no league game before they left: {f0(up['permanent'])} permanent arrivals were excluded upstream "
          f"for that reason, {f0(fo[UPS].get(FOLLOW[2], 0))} of them loaned onward "
          f"({f0(fo[UPS].get(FOLLOW[2] + ' | same day', 0) + fo[UPS].get(FOLLOW[2] + ' | 1–3 days', 0))} within 3 days). "
          f"Inside the revised sample only {f0(N['onward_any_move_within'][14]['permanent'])} permanent arrivals "
          f"move on within 14 days.",
          f"5. **How much of the zero-minute mass is onward loans?** Of {f0(z['permanent_zero'])} zero-share permanent "
          f"arrivals in the revised sample, {f0(z['onward_loan_within'][14])} were loaned onward within 14 days "
          f"({pc(z['onward_loan_within'][14], z['permanent_zero'])}), {f0(z['onward_loan_within'][60])} within 60 days "
          f"({pc(z['onward_loan_within'][60], z['permanent_zero'])}) and {f0(z['onward_loan_any'])} before season end at "
          f"any lag. {f0(z['stayed'])} ({pc(z['stayed'], z['permanent_zero'])}) stayed at the club through the season "
          f"end and did not play. The zero mass is mostly genuine non-selection, not buy-and-loan.",
          f"6. **Revised primary sample:** every reliable episode (at least one complete covered league game of B "
          f"while the player is there), first spell at B, and for loans a recorded return to the lender. No "
          f"arrival-date or season-fraction rule; onward moves end the window and are flagged, not excluded.",
          f"7. **Does the raw gap change?** No, not materially: {pp(o['gap'])} → {pp(r['gap'])}; "
          f"{pp(stay14['gap'])} after dropping moves within 14 days; minutes-weighted {pp(r['weighted_gap'])}.", ""]

    # ---- Part 1
    W += ["## 1. The outcome and the minimum opportunity", "",
          "Outcome: player league minutes / (90 × receiving club's league games from arrival to the earlier of the "
          "season's last league game and the day before his next recorded move). The window already starts at the "
          "actual arrival and stops at the actual departure, so summer and January arrivals are measured on the "
          "games each could actually play.", "",
          "**Minimum opportunity: at least one covered league game of B, with a complete appearance record, while "
          "the player is at B.** This is the only objective floor: the denominator must be positive. It is already "
          "part of the reliability rule, so it removes nothing from the reliable episodes. Distribution of the "
          "opportunity in the revised sample (league games available):", ""]
    ag = N["available_games"]
    W += md_table(["Games available", *ag["loan"]["bins"].keys()],
                  [[f"{LABEL[k]} N"] + [f0(v) for v in ag[k]["bins"].values()] for k in KINDS]
                  + [[f"{LABEL[k]} zero share"] + [pct(v) for v in ag[k]["zero_by_bin"].values()] for k in KINDS])
    fg = N["few_games"]
    W += ["", f"Episodes with 1–3 games are few ({fg['loan']['n']} loans, {fg['permanent']['n']} permanent). Most "
          f"permanent ones are summer arrivals who left before the season end ({fg['permanent']['left_before_season_end']}; "
          f"{fg['permanent']['onward_loan']} loaned onward), so the high zero share there is the onward-move issue "
          "of section 3, not a measurement problem. A higher games floor is a sensitivity, not a rule: "
          + "; ".join(f"≥{k} games {pp(sv[f'revised, at least {k} league games available']['gap'])}" for k in MIN_GAMES)
          + f". The minutes-weighted gap, which weights each episode by its opportunity without any cutoff, is "
          f"{pp(r['weighted_gap'])}.", ""]

    # ---- Part 2
    W += ["## 2. Waterfall", "", "**From reliable outcomes to the revised primary sample** (each episode removed once, "
          "in the order shown)", ""]
    rows = []
    for i, st in enumerate(N["revised_steps"]["loan"]):
        ps = N["revised_steps"]["permanent"][i]
        rows.append([st["step"], f0(st["removed"]), f0(st["remaining"]), f0(ps["removed"]), f0(ps["remaining"])])
    W += md_table(["Step", "Loans removed", "Loans left", "Permanent removed", "Permanent left"],
                  [["reliable playing-time outcomes", "", f0(rl["loan"]), "", f0(rl["permanent"])]] + rows)
    W += ["", f"No league match opportunity is 0 here because reliability already requires one game; upstream, "
          f"{f0(up['loan'])} loans and {f0(up['permanent'])} permanent moves were excluded for having none. The old "
          "path, for comparison:", ""]
    rows = []
    for i, st in enumerate(N["old_steps"]["loan"]):
        ps = N["old_steps"]["permanent"][i]
        rows.append([st["step"], f0(st["removed"]), f0(st["remaining"]), f0(ps["removed"]), f0(ps["remaining"])])
    W += md_table(["Step", "Loans removed", "Loans left", "Permanent removed", "Permanent left"],
                  [["reliable playing-time outcomes", "", f0(rl["loan"]), "", f0(rl["permanent"])]] + rows)
    W += ["", "Two-thirds-rule exclusions by arrival month (loans / permanent): "
          + "; ".join(f"{m}: {N['two_thirds_removed_by_month']['loan'].get(m, 0)} / "
                      f"{N['two_thirds_removed_by_month']['permanent'].get(m, 0)}"
                      for m in sorted(set(N['two_thirds_removed_by_month']['loan']) | set(N['two_thirds_removed_by_month']['permanent'])))
          + f". Full upstream and revised waterfalls in `{WATERFALL_CSV.name}`.", ""]

    # ---- Part 3
    W += ["## 3. Permanent arrivals who leave again", ""]
    scopes = list(fn)
    W += md_table(["What came next"] + [f"{s} (N = {f0(fn[s])})" for s in scopes],
                  [[f] + [f"{f0(fo[s].get(f, 0))} ({pc(fo[s].get(f, 0), fn[s])})" for s in scopes] for f in FOLLOW])
    W += ["", "A–B–C–D compare the next recorded move with B's last league game of the arrival season. In the "
          "reliable scopes, E (no next move at all) also means the player stayed through the season end, because "
          "every window ends before the player's history was captured.", "",
          "**Days from permanent arrival to the onward loan** (onward loans before season end):", ""]
    W += md_table(["Lag"] + scopes, [[lab] + [f0(fo[s].get(f"{FOLLOW[2]} | {lab}", 0)) for s in scopes] for _, _, lab in LAGS])
    cum = lambda s, x: sum(fo[s].get(f"{FOLLOW[2]} | {lab}", 0) for lo, hi, lab in LAGS if hi <= x)
    W += ["", "Cumulative (same day / ≤3 / ≤7 / ≤14 / ≤30 / ≤60 days), all permanent arrivals into a covered league "
          "season: " + " / ".join(f0(cum(scopes[0], x)) for x in ONWARD_DAYS)
          + f"; revised sample: " + " / ".join(f0(cum(scopes[3], x)) for x in ONWARD_DAYS) + ".", "",
          f"**Zero playing-time share, revised sample.** {f0(z['permanent_zero'])} of {f0(z['permanent_n'])} permanent "
          f"arrivals ({pc(z['permanent_zero'], z['permanent_n'])}) have zero minutes; loans: {f0(z['loan_zero'])}. "
          "Loaned onward within: " + "; ".join(f"{'same day' if x == 0 else f'≤{x} days'} {f0(z['onward_loan_within'][x])}"
                                                 for x in ONWARD_DAYS)
          + f"; at any lag before season end {f0(z['onward_loan_any'])}. Dropping the quick onward loans would move "
          "the permanent zero share only from " + pct(z["permanent_zero"] / z["permanent_n"]) + " to "
          + pct(z["zero_share_if_quick_loans_dropped"][14]) + " (14 days) or "
          + pct(z["zero_share_if_quick_loans_dropped"][60]) + " (60 days).", "",
          f"![Zero-minute permanent arrivals]({FIGS['zeros']})", ""]

    # ---- Part 4 & 5
    W += ["## 4–5. Two playing-time concepts and the raw comparison", "",
          "**Actual time at the receiving club** (the revised primary sample): arrival to season end or actual "
          "departure. **Staying sample**: the same, minus episodes whose next move comes within X days of arrival "
          "(before season end), for X = 0, 3, 7, 14 and, because the evidence points to end-of-window loans, 30 "
          "and 60.", ""]
    keys = [OLD, "reliable, no restriction", "reliable + loan return (all spells)", REV] + \
        [k for k in sv if k.startswith("revised, staying")] + \
        ["revised, drop any onward loan before season end (conditions on the outcome)"] + \
        [f"revised, at least {k} league games available" for k in MIN_GAMES]
    rows = []
    for k in keys:
        v = sv[k]
        rows.append([k.replace("REVISED PRIMARY: ", "**revised primary**: "), f0(v["loan_N"]), f0(v["permanent_N"]),
                     p3(v["loan_mean"]), p3(v["permanent_mean"]), pp(v["gap"]), f"[{pp(v['gap_lo'])}, {pp(v['gap_hi'])}]",
                     pp(v["median_gap"]), pp(v["weighted_gap"])])
    W += md_table(["Sample", "Loans", "Permanent", "Loan mean", "Perm. mean", "Gap", "95% interval", "Median gap",
                   "Minutes-weighted gap"], rows)
    W += ["", "**Full distribution, revised primary vs old primary**", ""]
    stats = (("N", f0, "N"), ("mean", p3, "mean"), ("median", p3, "median"), ("p25", p3, "p25"), ("p75", p3, "p75"),
             ("zero", pct, "zero share"), ("below_10", pct, "share < 10%"), ("above_50", pct, "share > 50%"))
    W += md_table(["", "Old loans", "Old permanent", "Revised loans", "Revised permanent", "Staying ≤14d loans",
                   "Staying ≤14d permanent"],
                  [[name] + [fmt(v[f"{k}_{lab}"]) for v in (o, r, stay14) for k in KINDS] for lab, fmt, name in stats])
    W += ["", f"Dropping every permanent arrival later loaned onward before season end, at any lag, gives "
          f"{pp(anyloan['gap'])}. That variant conditions on what happened after arrival (players who do not play "
          "are the ones loaned out), so it is shown only as a bound, not as a sample.", "",
          f"![Distribution]({FIGS['hist']})", ""]

    # ---- Part 6
    A = N["arrival"]
    W += ["## 6. Arrival timing", "",
          "Preseason is relative to B's own first league game, which adjusts for league calendars; in-season arrivals "
          "are grouped by month. No group is excluded.", ""]
    W += md_table(["Arrival", "Loans", "Permanent", "Loan mean", "Perm. mean", "Loan median", "Perm. median", "Gap",
                   "95% interval"],
                  [[g, f0(A[f"group: {g}"]["loan_N"]), f0(A[f"group: {g}"]["permanent_N"]), p3(A[f"group: {g}"]["loan_mean"]),
                    p3(A[f"group: {g}"]["permanent_mean"]), p3(A[f"group: {g}"]["loan_median"]),
                    p3(A[f"group: {g}"]["permanent_median"]), pp(A[f"group: {g}"]["gap"]),
                    f"[{pp(A[f'group: {g}']['gap_lo'])}, {pp(A[f'group: {g}']['gap_hi'])}]"] for g in GROUPS]
                  + [["**January only**", f0(jan.get("loan_N", 0)), f0(jan.get("permanent_N", 0)), p3(jan.get("loan_mean")),
                      p3(jan.get("permanent_mean")), p3(jan.get("loan_median")), p3(jan.get("permanent_median")),
                      pp(jan.get("gap")), f"[{pp(jan.get('gap_lo'))}, {pp(jan.get('gap_hi'))}]"]])
    pre = A[f"group: {GROUPS[0]}"]
    gaps = {g: A[f"group: {g}"]["gap"] for g in GROUPS}
    hi, lo = max(gaps, key=gaps.get), min(gaps, key=gaps.get)
    rs = sv["restored by dropping the two-thirds rule (revised, fewer than two-thirds left)"]
    W += ["", f"January arrivals: {f0(jan.get('loan_N', 0))} loans and {f0(jan.get('permanent_N', 0))} permanent moves. "
          f"Their mean shares ({p3(jan.get('loan_mean'))} and {p3(jan.get('permanent_mean'))}) are within "
          f"{max(abs(jan.get('loan_mean', 0) - r['loan_mean']), abs(jan.get('permanent_mean', 0) - r['permanent_mean'])):.3f} "
          "of the sample means, so the opportunity-share outcome does put them on the same scale. They do behave "
          f"differently in one respect: the loan-minus-permanent gap is {pp(jan.get('gap'))}, against "
          f"{pp(pre['gap'])} for preseason arrivals and {pp(r['gap'])} overall. Across groups the gap is largest for "
          f"{hi} ({pp(gaps[hi])}) and smallest for {lo} ({pp(gaps[lo])}). The {f0(N['restored_n'])} episodes restored "
          f"by dropping the two-thirds rule ({f0(N['restored_jan_feb'])} of them January–February arrivals) have a "
          f"gap of {pp(rs['gap'])}, which is what lifts the overall gap from {pp(o['gap'])} to {pp(r['gap'])}. Per "
          f"month in `{ARRIVAL_CSV.name}`.", "", f"![Arrival timing]({FIGS['arrival']})", ""]

    # ---- Part 7
    rp, rm, rg = N["repeat"], N["repeat_mean"], N["repeat_gap_days"]
    W += ["## 7. First-spell restriction", "",
          f"It removes {f0(N['repeat_total']['loan'])} loans and {f0(N['repeat_total']['permanent'])} permanent moves "
          "(after the loan-return rule). What they are:", ""]
    types = sorted(set(rp["loan"]) | set(rp["permanent"]))
    W += md_table(["Repeat spell", "Loans", "Loan mean share", "Permanent", "Perm. mean share",
                   "Median days since previous arrival at B (loan / permanent)"],
                  [[t, f0(rp["loan"].get(t, 0)), p3(rm["loan"].get(t)), f0(rp["permanent"].get(t, 0)), p3(rm["permanent"].get(t)),
                    " / ".join(f"{rg[k].get(t, float('nan')):.0f}" if t in rg[k] else "–" for k in KINDS)] for t in types])
    allsp = sv["reliable + loan return (all spells)"]
    W += ["", f"Including them: {f0(allsp['loan_N'])} loans, {f0(allsp['permanent_N'])} permanent, gap {pp(allsp['gap'])} "
          f"(first spell only: {pp(r['gap'])}). Most permanent repeat spells are conversions: the permanent move "
          "that follows a loan to the same club. They are a different event from a new arrival (the player already "
          f"knows the club), repeat spells of both kinds play more than first spells, and including them narrows the "
          f"gap by {abs(r['gap'] - allsp['gap']) * 100:.1f} pp. "
          "**Recommendation: keep first spell as the primary definition and report all spells as a robustness "
          "check**; it is a definition of the event, not an exposure rule, so it does not bias against January "
          "arrivals.", ""]

    # ---- Part 8
    W += ["## 8. Recommended primary sample", "",
          "- **Episodes:** loans and permanent moves with a reliable playing-time outcome: B plays in a covered "
          "league in its arrival season, every league game of B in the window has a complete appearance record, "
          "the window ends before the player's history was captured, and **B plays at least one league game while "
          "he is there** (the only opportunity floor).",
          "- **Event definition:** first spell at B; loans whose recorded ending is a return to the lender.",
          "- **Removed:** the two-thirds-of-season rule. No arrival-date or season-fraction condition.",
          "- **Outcome:** minutes share over the actual window (arrival to season end or departure), unchanged.",
          "- **Onward moves:** not excluded; each episode carries its follow-on category and lag. Report the staying "
          "sample (next move within 14 days dropped) as a robustness check, plus the ≤30 and ≤60 day versions.",
          f"- **Size:** {f0(rv['loan'])} loans and {f0(rv['permanent'])} permanent moves. Raw gap {pp(r['gap'])}; "
          f"old {pp(o['gap'])}.",
          "- **Next:** rerun the market-value regression on this sample once Professor Freund agrees to it.", ""]
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
    old = [ptj.EPISODES_CSV, ptj.NUMBERS_JSON, ptj.REPORT_MD, OUT / "playing_time_descriptive_summary.md",
           OUT / "playing_time_primary_sample.csv", OUT / "playing_time_sensitivity.csv"]
    before = {p.name: sha256(p) for p in old if p.exists()}
    assert not {p.name for p in WRITES} & set(before)
    N, tb = compute(u)
    tb["W"].to_csv(WATERFALL_CSV, index=False)
    tb["F"].to_csv(FOLLOWON_CSV, index=False)
    tb["A"].to_csv(ARRIVAL_CSV, index=False)
    tb["SENS"].to_csv(SENS_CSV, index=False)
    figures(N, tb["S"], tb["A"])
    md = render(N)
    sv = N["sensitivity"]
    for v in (N["revised"]["loan"], N["revised"]["permanent"], N["reliable"]["loan"], N["reliable"]["permanent"],
              N["zeros"]["permanent_zero"]):
        assert f0(v) in md, v
    REPORT_MD.write_text(md)
    N["old_output_sha256"] = before
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    assert {p.name: sha256(p) for p in old if p.exists()} == before, "an old output changed"
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print_summary(N)
    return N


def print_summary(N: dict) -> None:
    sv = N["sensitivity"]
    o = sv["old primary (two-thirds rule + first spell + loan return)"]
    r = sv["REVISED PRIMARY: reliable + loan return + first spell"]
    fo = N["followon"]["revised primary sample"]
    allp = N["followon"]["all permanent arrivals into a covered league season"]
    cum = lambda f, x: sum(f.get(f"{FOLLOW[2]} | {lab}", 0) for lo, hi, lab in LAGS if hi <= x)
    z = N["zeros"]
    jan = N["arrival"]["January only: January"]
    print("OLD SAMPLE")
    print(f"- loans: {o['loan_N']:,} (mean {o['loan_mean']:.3f})")
    print(f"- permanent: {o['permanent_N']:,} (mean {o['permanent_mean']:.3f})")
    print(f"- raw gap: {100 * o['gap']:+.1f} pp [{100 * o['gap_lo']:+.1f}, {100 * o['gap_hi']:+.1f}]")
    print("REVISED SAMPLE")
    print(f"- loans: {r['loan_N']:,} (mean {r['loan_mean']:.3f})")
    print(f"- permanent: {r['permanent_N']:,} (mean {r['permanent_mean']:.3f})")
    print(f"- raw gap: {100 * r['gap']:+.1f} pp [{100 * r['gap_lo']:+.1f}, {100 * r['gap_hi']:+.1f}]; "
          f"staying (drop moves within 14 days) "
          f"{100 * sv['revised, staying: drop next move within 14 days']['gap']:+.1f} pp; minutes-weighted "
          f"{100 * r['weighted_gap']:+.1f} pp")
    print("PERMANENT ONWARD LOANS (all permanent arrivals into a covered league season / revised sample)")
    print(f"- same day: {cum(allp, 0):,} / {cum(fo, 0):,}")
    print(f"- <= 7d: {cum(allp, 7):,} / {cum(fo, 7):,}")
    print(f"- <= 30d: {cum(allp, 30):,} / {cum(fo, 30):,}")
    print(f"- zero-minute permanent arrivals explained by quick onward loan: {z['onward_loan_within'][14]} of "
          f"{z['permanent_zero']:,} within 14 days ({pc(z['onward_loan_within'][14], z['permanent_zero'])}), "
          f"{z['onward_loan_within'][60]} within 60 days ({pc(z['onward_loan_within'][60], z['permanent_zero'])}); "
          f"{N['upstream_no_games']['permanent']} buy-and-move-on arrivals never enter the sample (no league game before leaving)")
    print("JANUARY")
    print(f"- N and raw gap: {jan['loan_N']:,} loans, {jan['permanent_N']:,} permanent; mean {jan['loan_mean']:.3f} vs "
          f"{jan['permanent_mean']:.3f}; gap {100 * jan['gap']:+.1f} pp [{100 * jan['gap_lo']:+.1f}, {100 * jan['gap_hi']:+.1f}]")
    print("RECOMMENDATION")
    print("- Primary sample = reliable outcome (>= 1 complete covered league game of B while the player is there) + "
          "first spell at B + loans ending in a recorded return to the lender; no season-fraction or arrival-date "
          "rule; onward moves end the window and are flagged, with a 14-day staying sample as robustness.")


if __name__ == "__main__":
    main()
