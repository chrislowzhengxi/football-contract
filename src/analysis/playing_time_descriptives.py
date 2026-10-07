"""Raw playing time after a loan vs a permanent move, on the clean primary sample. Descriptive only.

    python -m src.analysis.playing_time_descriptives

Rebuilds the primary sample with `playing_time_join` and stops if it does not
reproduce the saved feasibility run. No regression, no controls, no market
values in the analysis (their coverage is only counted). Raw differences
are descriptive, not causal.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from . import playing_time_join as ptj
from .daniel_scope_final import f0, md_table
from .loan_category_c_split import split as c_split
from .loan_episodes import ROOT
from .loan_timing_exploration import episode_table, next_moves, share

OUT = ROOT / "data" / "outputs" / "rebuild"
SUMMARY_MD = OUT / "playing_time_descriptive_summary.md"
BY_LEAGUE_CSV = OUT / "playing_time_descriptive_by_league.csv"
BY_SEASON_CSV = OUT / "playing_time_descriptive_by_season.csv"
SAMPLE_CSV = OUT / "playing_time_primary_sample.csv"
SENS_CSV = OUT / "playing_time_sensitivity.csv"
NUMBERS_JSON = OUT / "playing_time_descriptive_numbers.json"
FIGS = {"hist": "figures/playing_time_fig1_share_distribution.png",
        "ecdf": "figures/playing_time_fig2_share_ecdf.png",
        "league": "figures/playing_time_fig3_by_league.png",
        "season": "figures/playing_time_fig4_by_season.png"}
KINDS = ("loan", "permanent")
LABEL = {"loan": "Loan", "permanent": "Permanent"}
COLOR = {"loan": "#dd6b20", "permanent": "#2b6cb0"}     # validated pair (dataviz validator, light mode)
INK, MUTED, GRID = "#1a202c", "#555", "#e2e8f0"
SMALL_LEAGUE = 100                                       # fewer loans or permanents than this is flagged
SMALL_SUBGROUP = 30
COVID_SEASONS = (2019, 2020)
REQUIRED = ["player_id", "b_club_id", "move_date", "arrival_season", "receiving_league", "window_start",
            "window_end", "available_minutes", "minutes", "minutes_share"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def describe(s: pd.Series) -> dict:
    s = s.dropna()
    if not len(s):
        return {"N": 0}
    q = lambda p: float(np.quantile(s, p))
    return {"N": len(s), "mean": float(s.mean()), "median": q(.5), "sd": float(s.std()), "p10": q(.1),
            "p25": q(.25), "p75": q(.75), "p90": q(.9), "zero": float((s == 0).mean()),
            "below_10": float((s < .10).mean()), "below_25": float((s < .25).mean()),
            "above_50": float((s > .50).mean()), "above_75": float((s > .75).mean())}


def diff(a: pd.Series, b: pd.Series) -> dict:
    """Loan minus permanent, raw. The interval is a descriptive Welch normal approximation."""
    a, b = a.dropna(), b.dropna()
    d = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var() / len(a) + b.var() / len(b))) if len(a) > 1 and len(b) > 1 else np.nan
    return {"mean_diff": d, "mean_diff_lo": d - 1.96 * se, "mean_diff_hi": d + 1.96 * se,
            "median_diff": float(a.median() - b.median())}


def by_group(P: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for v, g in P.groupby(col):
        rec = {col: v}
        for k in KINDS:
            s = g.loc[g.kind == k, "minutes_share"]
            d = describe(s)
            rec.update({f"{k}_n": d["N"], f"{k}_mean": d.get("mean"), f"{k}_median": d.get("median")})
        ld, pdd = g.loc[g.kind == "loan", "minutes_share"], g.loc[g.kind == "permanent", "minutes_share"]
        if len(ld) and len(pdd):
            rec.update(diff(ld, pdd))
        rec["small_sample"] = min(rec["loan_n"], rec["permanent_n"]) < SMALL_LEAGUE
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

def base_sample(E: pd.DataFrame) -> pd.DataFrame:
    """The primary sample without its arrival-timing condition."""
    return E[E.reliable & ~E.prior_spell_at_b & ((E.kind == "permanent") | (E.loan_ending == "return to lender"))]


def loan_subgroups(u, P: pd.DataFrame) -> pd.Series:
    """What followed each primary-sample loan, from the existing sequence categories (descriptive)."""
    T = episode_table(u, next_moves(u)).set_index("loan_event_id")
    C = c_split(T.reset_index()).set_index("loan_event_id").subgroup
    L = P[P.kind == "loan"]
    cat = L.event_id.map(T.sequence_category)
    sub = L.event_id.map(C)
    scheduled = L.loan_return_scheduled.fillna(False).astype(bool)
    out = pd.Series(np.select(
        [scheduled, cat.eq("A"), cat.eq("B"), sub.eq("C1"), sub.eq("C2"), cat.eq("C"), cat.eq("D")],
        ["return still scheduled at capture",
         "returned to the lender; no nearby next move (A)",
         "returned, then permanent move to the borrower within 60 days (B)",
         "returned, then re-loan to the same borrower (C1)",
         "returned, then loan to a different club (C2)",
         "returned, then free or no-fee move or other (C3–C5)",
         "returned, then sold to a third club (D)"], default="other"), index=L.index)
    any_lag = L.event_id.map(T.next_move_is_permanent_to_borrower).fillna(False).astype(bool) & ~scheduled
    return out, any_lag


def compute() -> tuple[dict, pd.DataFrame, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    saved = json.loads(ptj.NUMBERS_JSON.read_text()) if ptj.NUMBERS_JSON.exists() else None
    u = loan_scope.universe()
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        J, E, _ = ptj.compute(u, con)
        P = E[E.primary_sample].copy()
        N: dict = {"primary": {k: int((P.kind == k).sum()) for k in KINDS}}
        if saved is not None:
            N["saved_primary"] = saved["primary"]
            if N["primary"] != {k: int(v) for k, v in saved["primary"].items()}:
                raise SystemExit(f"primary sample does not reproduce the saved run: {N['primary']} vs {saved['primary']}")
        N["leagues"] = sorted(P.receiving_league.unique())
        N["seasons"] = [int(P.arrival_season.min()), int(P.arrival_season.max())]
        assert N["leagues"] == J["covered_leagues"], "primary sample leagues differ from the covered leagues"

        # ---- QA
        qa = {"episodes_unique": bool(P.event_id.is_unique),
              "required_fields_present": {c: int(P[c].isna().sum()) for c in REQUIRED},
              "all_reliable": bool(P.reliable.all()),
              "complete_windows": bool((P.complete_games == P.available_games).all()),
              "zero_only_when_complete": bool((P.loc[P.minutes == 0, "complete_games"]
                                               == P.loc[P.minutes == 0, "available_games"]).all()),
              "share_min": float(P.minutes_share.min()), "share_max": float(P.minutes_share.max()),
              "minutes_above_available": int((P.minutes > P.available_minutes).sum()),
              "no_coverage_rows_in_sample": int((~E.reliable & E.primary_sample).sum())}
        over = P[P.minutes > P.available_minutes]
        qa["minutes_above_available_cases"] = over[["player_name", "b_club", "arrival_season", "minutes",
                                                    "available_minutes"]].astype(str).to_dict("records")
        assert qa["episodes_unique"] and qa["all_reliable"] and qa["complete_windows"] and qa["zero_only_when_complete"]
        assert all(v == 0 for v in qa["required_fields_present"].values())
        N["qa"] = qa

        # ---- descriptives
        N["share"] = {k: describe(P.loc[P.kind == k, "minutes_share"]) for k in KINDS}
        N["minutes"] = {k: describe(P.loc[P.kind == k, "minutes"]) for k in KINDS}
        N["available"] = {k: describe(P.loc[P.kind == k, "available_minutes"]) for k in KINDS}
        N["games"] = {k: describe(P.loc[P.kind == k, "available_games"]) for k in KINDS}
        N["diff"] = diff(P.loc[P.kind == "loan", "minutes_share"], P.loc[P.kind == "permanent", "minutes_share"])
        N["window"] = {k: {"departed": float(g.departed_in_window.mean()),
                           "arrived_before_first_game": float((g.games_remaining_at_arrival == g.season_games).mean()),
                           "median_games": float(g.available_games.median())}
                       for k, g in P.groupby("kind")}

        # ---- league / season
        lg = by_group(P, "receiving_league")
        ss = by_group(P, "arrival_season")
        for t, col in ((lg, "receiving_league"), (ss, "arrival_season")):
            for k in KINDS:
                assert t[f"{k}_n"].sum() == N["primary"][k], f"{col} subtotals do not reconcile"
        N["league_loan_lower"] = int((lg.mean_diff < 0).sum())
        N["league_median_loan_lower"] = int((lg.median_diff < 0).sum())
        N["league_ci_below_zero"] = int((lg.mean_diff_hi < 0).sum())
        N["league_ci_above_zero"] = int((lg.mean_diff_lo > 0).sum())
        N["league_opposite"] = lg.loc[np.sign(lg.mean_diff) != np.sign(N["diff"]["mean_diff"]),
                                      ["receiving_league", "mean_diff", "loan_n", "permanent_n"]].to_dict("records")
        cv_d = ss.loc[ss.arrival_season.isin(COVID_SEASONS), "mean_diff"]
        ot_d = ss.loc[~ss.arrival_season.isin(COVID_SEASONS), "mean_diff"]
        N["covid_diffs"] = [float(x) for x in cv_d]
        N["other_season_diff_range"] = [float(ot_d.min()), float(ot_d.max())]
        N["season_loan_lower"] = int((ss.mean_diff < 0).sum())
        N["small_leagues"] = lg.loc[lg.small_sample, ["receiving_league", "loan_n", "permanent_n"]].to_dict("records")

        # ---- sensitivity
        B = base_sample(E).copy()
        B["remain"] = B.games_remaining_at_arrival / B.season_games
        sens = []

        def add(family, label, frame, col="minutes_share"):
            rec = {"family": family, "variant": label}
            for k in KINDS:
                d = describe(frame.loc[frame.kind == k, col])
                rec.update({f"{k}_n": d["N"], f"{k}_mean": d.get("mean"), f"{k}_median": d.get("median")})
            if rec["loan_n"] and rec["permanent_n"]:
                rec.update(diff(frame.loc[frame.kind == "loan", col], frame.loc[frame.kind == "permanent", col]))
            sens.append(rec)

        add("arrival", "any arrival time", B)
        add("arrival", "at least half of B's league games left", B[B.remain >= 0.5])
        add("arrival", "at least two-thirds left (primary)", B[B.remain >= ptj.FULL_SEASON_SHARE])
        add("arrival", "at least 90% left", B[B.remain >= 0.9])
        add("arrival", "arrived before B's first league game", B[B.remain >= 1])
        add("covid", "primary", P)
        add("covid", "excluding 2019/20 arrivals", P[P.arrival_season != 2019])
        add("covid", "excluding 2019/20 and 2020/21 arrivals", P[~P.arrival_season.isin(COVID_SEASONS)])
        add("outcome", "A. share of available league minutes", P)
        add("outcome", "B. raw league minutes in the window", P, "minutes")
        P["minutes_per_match"] = P.minutes / P.available_games
        P["appearance_share"] = P.appearances / P.available_games
        add("outcome", "C. minutes per club league match (= 90 × A)", P, "minutes_per_match")
        add("outcome", "D. share of club league matches played in", P, "appearance_share")
        full = P[(P.games_remaining_at_arrival == P.season_games) & ~P.departed_in_window]
        add("outcome", "A, full-season windows only (arrived before first game, stayed all season)", full)
        add("outcome", "B, full-season windows only", full, "minutes")
        add("window", "stayed the whole window (no departure)", P[~P.departed_in_window])
        add("window", "left B before season end", P[P.departed_in_window])
        S = pd.DataFrame(sens)
        N["sensitivity"] = S.to_dict("records")
        assert np.isclose(P.minutes_per_match, 90 * P.minutes_share).all()

        # ---- loan subgroups
        sub, any_lag = loan_subgroups(u, P)
        P.loc[sub.index, "loan_subgroup"] = sub
        P.loc[any_lag.index, "loan_later_permanent_to_borrower_any_lag"] = any_lag
        L = P[P.kind == "loan"]
        subs = []
        for name, g in L.groupby("loan_subgroup"):
            d = describe(g.minutes_share)
            subs.append({"subgroup": name, "N": d["N"], "mean": d["mean"], "median": d["median"], "p25": d["p25"],
                         "p75": d["p75"], "above_50": d["above_50"], "small": d["N"] < SMALL_SUBGROUP})
        for name, g in (("later permanent move to the borrower, any lag", L[L.loan_later_permanent_to_borrower_any_lag == True]),  # noqa: E712
                        ("all returned loans without a later permanent move to the borrower",
                         L[(L.loan_later_permanent_to_borrower_any_lag == False)  # noqa: E712
                           & (L.loan_subgroup != "return still scheduled at capture")])):
            d = describe(g.minutes_share)
            subs.append({"subgroup": f"[alternative grouping] {name}", "N": d["N"], "mean": d["mean"],
                         "median": d["median"], "p25": d["p25"], "p75": d["p75"], "above_50": d["above_50"],
                         "small": d["N"] < SMALL_SUBGROUP})
        N["loan_subgroups"] = subs
        assert sum(s["N"] for s in subs if not s["subgroup"].startswith("[")) == N["primary"]["loan"]

        # ---- market-value coverage only
        N["market_value"] = {k: ptj.market_value_coverage(E[E.kind == k], con) for k in KINDS}
        N["clubs_table_value_is_current_only"] = True
    finally:
        con.close()
    return N, P, {"league": lg, "season": ss, "sens": S}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def figures(N: dict, P: pd.DataFrame, tb: dict) -> None:
    plt = _style()
    (OUT / "figures").mkdir(exist_ok=True)
    n = N["primary"]

    # 1. distribution: share of each group per 0.05 bin, zero shown as its own bar
    edges = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.018
    for off, k in ((-w / 2, "loan"), (w / 2, "permanent")):
        s = P.loc[P.kind == k, "minutes_share"]
        zero = (s == 0).mean() * 100
        h = np.histogram(s[s > 0], bins=edges)[0] / len(s) * 100
        ax.bar(-0.03 + off, zero, width=w, color=COLOR[k])
        ax.bar(edges[:-1] + 0.025 + off, h, width=w, color=COLOR[k],
               label=f"{LABEL[k]} (N = {n[k]:,}; median {N['share'][k]['median']:.2f})")
    ax.set_xticks([-0.03] + list(np.round(np.linspace(0.1, 1, 10), 1)))
    ax.set_xticklabels(["exactly\n0"] + [f"{x:.1f}" for x in np.linspace(0.1, 1, 10)])
    ax.axvline(-0.008, color=MUTED, lw=0.6)
    ax.set_xlim(-0.06, 1.01)
    ax.set_xlabel("Share of the receiving club's league minutes played (window: arrival to season end or departure)")
    ax.set_ylabel("Share of the group (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Playing time after the move: loans vs permanent transfers (0.05-wide bins)", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["hist"], dpi=160)
    plt.close(fig)

    # 2. ECDF
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k in KINDS:
        x = np.sort(P.loc[P.kind == k, "minutes_share"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x) * 100
        ax.step(np.r_[0, x], np.r_[0, y], where="post", color=COLOR[k], lw=2)
        med = N["share"][k]["median"]
        ax.plot([med], [50], "o", color=COLOR[k], ms=6)
        if k == "loan":
            ax.text(med + 0.02, 50 - 7, f"{LABEL[k]}: median {med:.2f}", color=COLOR[k], fontsize=9, ha="left")
        else:
            ax.text(med - 0.02, 50 + 4, f"{LABEL[k]}: median {med:.2f}", color=COLOR[k], fontsize=9, ha="right")
    ax.axhline(50, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of the receiving club's league minutes played")
    ax.set_ylabel("Cumulative share of episodes (%)")
    ax.set_title(f"Cumulative distribution (loans N = {n['loan']:,}, permanent N = {n['permanent']:,})", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["ecdf"], dpi=160)
    plt.close(fig)

    # 3. by league: means, sorted by the raw difference; small samples hollow
    lg = tb["league"].sort_values("mean_diff")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ys = np.arange(len(lg))
    for y, r in zip(ys, lg.itertuples()):
        ax.plot([r.loan_mean, r.permanent_mean], [y, y], color="#cbd5e0", lw=2, zorder=1)
        for k in KINDS:
            ax.scatter(getattr(r, f"{k}_mean"), y, s=46, zorder=2, color=COLOR[k] if not r.small_sample else "white",
                       edgecolors=COLOR[k], linewidths=1.6, label=LABEL[k] if y == 0 else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.receiving_league}  (loans {r.loan_n:,} / perm. {r.permanent_n:,})"
                        + ("  *" if r.small_sample else "") for r in lg.itertuples()], fontsize=9)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Mean share of the receiving club's league minutes played")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("By receiving league (sorted by loan − permanent difference)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if lg.small_sample.any():
        fig.text(0.01, 0.01, f"* hollow markers: fewer than {SMALL_LEAGUE} loans or permanent moves", fontsize=8,
                 color=MUTED)
    fig.savefig(OUT / FIGS["league"], dpi=160)
    plt.close(fig)

    # 4. by season: mean share, COVID seasons shaded
    ss = tb["season"].sort_values("arrival_season")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = ss.arrival_season.astype(int).to_numpy()
    for s in COVID_SEASONS:
        ax.axvspan(s - 0.5, s + 0.5, color="#edf2f7", zorder=0)
    ax.text(2019.5, 0.06, "2019/20–2020/21", ha="center", fontsize=8, color=MUTED)
    for k in KINDS:
        ax.plot(x, ss[f"{k}_mean"], "-o", color=COLOR[k], lw=2, ms=4, label=f"{LABEL[k]} (mean)")
        ax.plot(x, ss[f"{k}_median"], ":", color=COLOR[k], lw=1.4, label=f"{LABEL[k]} (median)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}/{str(s + 1)[2:]}" for s in x], rotation=45, ha="right")
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Share of the receiving club's league minutes")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("By season of arrival", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / FIGS["season"], dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render(N: dict, tb: dict) -> str:
    W: list[str] = []
    w = W.append
    sh, mi, av, gm = N["share"], N["minutes"], N["available"], N["games"]
    lo, pe = sh["loan"], sh["permanent"]
    d = N["diff"]
    n = N["primary"]
    lg, ss, S = tb["league"], tb["season"], tb["sens"]
    pct = lambda x: share(x)
    num = lambda x: f"{x:.2f}"

    w("# Playing time after a loan vs a permanent transfer: raw distributions")
    w("")
    w("*Generated by `python -m src.analysis.playing_time_descriptives` on the clean primary sample of the "
      "playing-time feasibility audit. Descriptive only: no regression, no controls, and the raw difference is not "
      "a causal effect. Loans and permanent moves differ in many ways (player age, quality, the clubs involved) that "
      "are not accounted for here.*")
    w("")
    w("## Answers")
    w("")
    w(f"- **Raw difference.** Loanees play a mean {num(lo['mean'])} of the receiving club's league minutes in the "
      f"window, permanent signings {num(pe['mean'])}: a difference of {d['mean_diff']:+.3f} (descriptive 95% interval "
      f"{d['mean_diff_lo']:+.3f} to {d['mean_diff_hi']:+.3f}). The medians are {num(lo['median'])} and "
      f"{num(pe['median'])} ({d['median_diff']:+.3f}).")
    sign = "higher" if d["mean_diff"] > 0 else "lower"
    more0, less0 = ("permanent signings", "loanees") if pe["zero"] > lo["zero"] else ("loanees", "permanent signings")
    w(f"- **In the distribution, not just the mean.** The difference sits mostly in the lower half. Zero minutes: "
      f"{pct(lo['zero'])} of loanees against {pct(pe['zero'])} of permanent signings, so {more0} are more often at "
      f"zero. Below 10%: {pct(lo['below_10'])} against {pct(pe['below_10'])}. p25: {num(lo['p25'])} against "
      f"{num(pe['p25'])}. Higher up the two distributions converge: above 50%, {pct(lo['above_50'])} against "
      f"{pct(pe['above_50'])}; p90, {num(lo['p90'])} against {num(pe['p90'])}.")
    opp = N["league_opposite"]
    w(f"- **Across leagues.** The mean is {sign} for loans in "
      f"{N['league_loan_lower'] if sign == 'lower' else len(lg) - N['league_loan_lower']} of {len(lg)} leagues, and "
      f"the median in {N['league_median_loan_lower'] if sign == 'lower' else len(lg) - N['league_median_loan_lower']} "
      f"of {len(lg)}. The descriptive interval excludes zero in {N['league_ci_above_zero'] if sign == 'higher' else N['league_ci_below_zero']} "
      "leagues in the overall direction"
      + (". The exception: " + ", ".join(f"{r['receiving_league']} ({r['mean_diff']:+.3f}; {r['loan_n']} loans)"
                                        for r in opp) + ", where loanees play less." if opp else "."))
    w(f"- **Across seasons.** Loans are {sign} on average in "
      f"{N['season_loan_lower'] if sign == 'lower' else len(ss) - N['season_loan_lower']} of {len(ss)} arrival "
      f"seasons. The COVID seasons' differences ({', '.join(f'{x:+.3f}' for x in N['covid_diffs'])}) lie within or at "
      f"the edge of the other seasons' range ({N['other_season_diff_range'][0]:+.3f} to "
      f"{N['other_season_diff_range'][1]:+.3f}), and dropping them changes nothing (below).")
    arr = S[S.family == "arrival"]
    w(f"- **Arrival-window definition.** The mean difference ranges from {arr.mean_diff.min():+.3f} to "
      f"{arr.mean_diff.max():+.3f} across the five arrival definitions tested, so it is not an artefact of the "
      "two-thirds cutoff.")
    subs = sorted([s for s in N["loan_subgroups"] if not s["subgroup"].startswith("[") and not s["small"]
                   and not s["subgroup"].startswith("return still")], key=lambda s: -s["mean"])
    w(f"- **Loan subtypes differ.** By what followed the loan, mean shares during the loan range from "
      f"{subs[-1]['mean']:.2f} to {subs[0]['mean']:.2f}. Highest: {subs[0]['subgroup']} ({subs[0]['mean']:.2f}) and "
      f"{subs[1]['subgroup']} ({subs[1]['mean']:.2f}). Lowest: {subs[-1]['subgroup']} ({subs[-1]['mean']:.2f}) and "
      f"{subs[-2]['subgroup']} ({subs[-2]['mean']:.2f}). "
      "This conditions on what happened later, so it is descriptive only.")
    mvl = N["market_value"]
    cov_lo = min(m["stage1c_market_value"] / m["primary_n"] for m in mvl.values())
    cov_hi = max(m["stage1c_market_value"] / m["primary_n"] for m in mvl.values())
    w(f"- **Enough data for a controlled analysis?** Yes: {f0(n['loan'])} loans and {f0(n['permanent'])} permanent "
      f"moves, every league with at least {min(lg.loan_n.min(), lg.permanent_n.min())} of each, and player market "
      f"values for {pct(cov_lo)}–{pct(cov_hi)} of episodes (coverage below). The groups differ in composition: "
      f"{pct(pe['zero'])} of permanent signings never play in the window, and loanees more often arrive after the "
      "season has started and leave early (section 1). Those differences are what the controls will need to "
      "address.")
    w("")

    w("## 1. The sample")
    w("")
    q = N["qa"]
    w(f"The primary sample is rebuilt from scratch and reproduces the feasibility audit exactly: **{f0(n['loan'])} "
      f"loans and {f0(n['permanent'])} permanent transfers**, {len(N['leagues'])} leagues "
      f"({', '.join(N['leagues'])}), arrival seasons {N['seasons'][0]}/{str(N['seasons'][0] + 1)[2:]}–"
      f"{N['seasons'][1]}/{str(N['seasons'][1] + 1)[2:]}. Every episode appears once and has a player, receiving "
      "club, move date, season, league, window start and end, available league minutes, player minutes and share. "
      "Every window has a complete appearance record, so each zero is a true zero. Shares lie between "
      f"{q['share_min']:.2f} and {q['share_max']:.2f}; {q['minutes_above_available']} episodes have more minutes "
      "than 90 × games.")
    w("")
    w("What the share is made of (per episode):")
    w("")
    W += md_table(["", "Loans", "Permanent transfers"], [
        ["Player league minutes in the window, mean / median", f"{mi['loan']['mean']:,.0f} / {mi['loan']['median']:,.0f}",
         f"{mi['permanent']['mean']:,.0f} / {mi['permanent']['median']:,.0f}"],
        ["Club league minutes available (90 × games), mean / median",
         f"{av['loan']['mean']:,.0f} / {av['loan']['median']:,.0f}",
         f"{av['permanent']['mean']:,.0f} / {av['permanent']['median']:,.0f}"],
        ["Club league games in the window, median", f"{gm['loan']['median']:.0f}", f"{gm['permanent']['median']:.0f}"],
        ["Arrived before B's first league game", pct(N["window"]["loan"]["arrived_before_first_game"]),
         pct(N["window"]["permanent"]["arrived_before_first_game"])],
        ["Left B before the season ended (window cut at departure)", pct(N["window"]["loan"]["departed"]),
         pct(N["window"]["permanent"]["departed"])],
    ])
    w("")

    w("## 2. Descriptive statistics of the playing-time share")
    w("")
    rows = [["N", f0(lo["N"]), f0(pe["N"]), ""]]
    for key, lab in (("mean", "Mean"), ("median", "Median"), ("sd", "Standard deviation"), ("p10", "p10"),
                     ("p25", "p25"), ("p75", "p75"), ("p90", "p90")):
        rows.append([lab, num(lo[key]), num(pe[key]), f"{lo[key] - pe[key]:+.3f}"])
    for key, lab in (("zero", "Exactly zero"), ("below_10", "Below 10%"), ("below_25", "Below 25%"),
                     ("above_50", "Above 50%"), ("above_75", "Above 75%")):
        rows.append([lab, pct(lo[key]), pct(pe[key]), f"{100 * (lo[key] - pe[key]):+.1f} pp"])
    W += md_table(["Playing-time share", "Loans", "Permanent transfers", "Loan − permanent"], rows)
    w("")
    w(f"![Distribution]({FIGS['hist']})")
    w("")
    w(f"![Cumulative distribution]({FIGS['ecdf']})")
    w("")

    w("## 3. By receiving league")
    w("")
    rows = []
    for r in lg.sort_values("receiving_league").itertuples():
        rows.append([r.receiving_league + (" *" if r.small_sample else ""), f0(r.loan_n), f0(r.permanent_n),
                     num(r.loan_median), num(r.permanent_median), num(r.loan_mean), num(r.permanent_mean),
                     f"{r.mean_diff:+.3f}", f"{r.mean_diff_lo:+.3f} to {r.mean_diff_hi:+.3f}"])
    W += md_table(["League", "Loans", "Permanent", "Median loan", "Median permanent", "Mean loan", "Mean permanent",
                   "Mean difference", "Descriptive 95% interval"], rows)
    w("")
    if N["small_leagues"]:
        w(f"\\* fewer than {SMALL_LEAGUE} loans or permanent moves: "
          + ", ".join(f"{r['receiving_league']} ({r['loan_n']} loans, {r['permanent_n']} permanent)"
                      for r in N["small_leagues"]) + ". Shown separately, not pooled.")
        w("")
    w(f"The descriptive interval lies entirely above zero in {N['league_ci_above_zero']} of {len(lg)} leagues and "
      f"entirely below zero in {N['league_ci_below_zero']}.")
    w("")
    w(f"![By league]({FIGS['league']})")
    w("")

    w("## 4. By season of arrival")
    w("")
    rows = []
    for r in ss.sort_values("arrival_season").itertuples():
        s = int(r.arrival_season)
        rows.append([f"{s}/{str(s + 1)[2:]}" + (" (COVID)" if s in COVID_SEASONS else ""), f0(r.loan_n),
                     f0(r.permanent_n), num(r.loan_mean), num(r.permanent_mean), num(r.loan_median),
                     num(r.permanent_median), f"{r.mean_diff:+.3f}"])
    W += md_table(["Arrival season", "Loans", "Permanent", "Mean loan", "Mean permanent", "Median loan",
                   "Median permanent", "Mean difference"], rows)
    w("")
    cv = S[S.family == "covid"]
    w("COVID sensitivity (2019/20 was interrupted; 2020/21 was played in the pandemic):")
    w("")
    W += md_table(["Sample", "Loans", "Permanent", "Mean loan", "Mean permanent", "Mean difference",
                   "Median difference"],
                  [[r.variant, f0(r.loan_n), f0(r.permanent_n), num(r.loan_mean), num(r.permanent_mean),
                    f"{r.mean_diff:+.3f}", f"{r.median_diff:+.3f}"] for r in cv.itertuples()])
    w("")
    w(f"![By season]({FIGS['season']})")
    w("")

    w("## 5. Arrival-timing sensitivity")
    w("")
    w("All variants use the primary sample's other conditions (reliable outcome, first spell at B, loans returning "
      "to the lender) and change only the arrival condition.")
    w("")
    W += md_table(["Arrival condition", "Loans", "Permanent", "Mean loan", "Mean permanent", "Median loan",
                   "Median permanent", "Mean difference", "Median difference"],
                  [[r.variant, f0(r.loan_n), f0(r.permanent_n), num(r.loan_mean), num(r.permanent_mean),
                    num(r.loan_median), num(r.permanent_median), f"{r.mean_diff:+.3f}", f"{r.median_diff:+.3f}"]
                   for r in arr.itertuples()])
    w("")

    w("## 6. Outcome-definition sensitivity")
    w("")
    oc = S[S.family.isin(["outcome", "window"])]

    def fmt(variant, v, signed=False):
        sgn = "+" if signed else ""
        if variant.startswith("B"):
            return f"{v:{sgn},.0f}"
        if variant.startswith("C"):
            return f"{v:{sgn}.1f}"
        return f"{v:{sgn}.3f}"
    W += md_table(["Measure / subset", "Loans", "Permanent", "Mean loan", "Mean permanent", "Median loan",
                   "Median permanent", "Mean difference"],
                  [[r.variant, f0(r.loan_n), f0(r.permanent_n)] + [fmt(r.variant, v) for v in
                    (r.loan_mean, r.permanent_mean, r.loan_median, r.permanent_median)] + [fmt(r.variant, r.mean_diff, True)]
                   for r in oc.itertuples()])
    w("")
    w("- **A (share) is the most robust.** Its denominator is the club's own league games in the window, so it "
      "adjusts for mid-season arrival, for leagues and seasons of different length, for an onward move that shortens the "
      "window, and for an unequal number of remaining matches.")
    w("- **B (raw minutes)** grows with the window, so it mixes playing time with how long the player was there. "
      "Among full-season windows only, where windows are comparable, B gives the same ordering as A (table).")
    w("- **C (minutes per club match)** is exactly 90 × A, so it adds nothing.")
    w("- **D (share of matches played in)** counts a 5-minute substitute like a full game. It is useful as a second "
      "measure of being used at all, but not as the primary one.")
    w("- **The primary outcome stays A.**")
    w("")

    w("## 7. Loan subgroups (descriptive; what followed the loan)")
    w("")
    w("Grouped by what happened after the loan, using the existing sequence categories. This conditions on the "
      "future, so it describes which loans went where, not why. No contract clause is inferred.")
    w("")
    order = ["returned to the lender; no nearby next move (A)",
             "returned, then permanent move to the borrower within 60 days (B)",
             "returned, then re-loan to the same borrower (C1)", "returned, then loan to a different club (C2)",
             "returned, then free or no-fee move or other (C3–C5)", "returned, then sold to a third club (D)",
             "return still scheduled at capture"]
    ranked = sorted(N["loan_subgroups"], key=lambda s: (s["subgroup"].startswith("["),
                                                        order.index(s["subgroup"]) if s["subgroup"] in order else 99))
    W += md_table(["Loan subgroup", "N", "Mean", "Median", "p25", "p75", "Above 50%"],
                  [[s["subgroup"] + (" (small)" if s["small"] else ""), f0(s["N"]), num(s["mean"]), num(s["median"]),
                    num(s["p25"]), num(s["p75"]), pct(s["above_50"])] for s in ranked])
    w("")
    w(f"\"(small)\" means fewer than {SMALL_SUBGROUP} loans: too few to compare. The two rows in brackets regroup the "
      "same loans by whether a permanent move to the borrower followed at any lag, rather than within 60 days.")
    w("")

    w("## 8. Market values for the next phase (coverage only, not used here)")
    w("")
    mv = N["market_value"]
    W += md_table(["Available for the primary sample", "Loans", "Permanent transfers"], [
        ["Player market value at the move (Stage 1C)",
         f"{f0(mv['loan']['stage1c_market_value'])} ({pct(mv['loan']['stage1c_market_value'] / mv['loan']['primary_n'])})",
         f"{f0(mv['permanent']['stage1c_market_value'])} ({pct(mv['permanent']['stage1c_market_value'] / mv['permanent']['primary_n'])})"],
        ["Player valuation in the year before the move (`player_valuations`)",
         f"{f0(mv['loan']['player_valuation_within_365d_before'])} ({pct(mv['loan']['player_valuation_within_365d_before'] / mv['loan']['primary_n'])})",
         f"{f0(mv['permanent']['player_valuation_within_365d_before'])} ({pct(mv['permanent']['player_valuation_within_365d_before'] / mv['permanent']['primary_n'])})"],
        ["Receiving club with at least 11 valued players in that year (a squad value is buildable)",
         f"{f0(mv['loan']['squad_with_at_least_11_valued_players'])} ({pct(mv['loan']['squad_with_at_least_11_valued_players'] / mv['loan']['primary_n'])})",
         f"{f0(mv['permanent']['squad_with_at_least_11_valued_players'])} ({pct(mv['permanent']['squad_with_at_least_11_valued_players'] / mv['permanent']['primary_n'])})"],
    ])
    w("")
    w("The `clubs` table's `total_market_value` is a current snapshot, not a value at the time of the move, so a "
      "historical squad value has to be built from `player_valuations` (each valuation records the player's club at "
      "that date).")
    w("")
    return "\n".join(W) + "\n"


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


SAMPLE_COLS = ["event_id", "kind", "episode_type", "player_id", "player_name", "move_date", "transfer_season",
               "arrival_season", "receiving_league", "a_club_id", "a_club", "b_club_id", "b_club", "from_country",
               "to_country", "window_start", "window_end", "season_games", "games_remaining_at_arrival",
               "available_games", "available_minutes", "minutes", "appearances", "minutes_share", "true_zero",
               "departed_in_window", "departure_date", "loan_return_date", "loan_return_scheduled", "loan_subgroup",
               "loan_later_permanent_to_borrower_any_lag", "market_value_eur"]


def main() -> dict:
    N, P, tb = compute()
    out = P[SAMPLE_COLS].copy()
    for c in ("move_date", "window_start", "window_end", "departure_date", "loan_return_date"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(SAMPLE_CSV, index=False)
    tb["league"].to_csv(BY_LEAGUE_CSV, index=False)
    tb["season"].to_csv(BY_SEASON_CSV, index=False)
    tb["sens"].to_csv(SENS_CSV, index=False)
    figures(N, P, tb)
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    SUMMARY_MD.write_text(render(N, tb))
    sh, d = N["share"], N["diff"]
    print(f"wrote {SUMMARY_MD.name}")
    for k in ("loan", "permanent"):
        print(f"{k:10} N {sh[k]['N']:,}  mean {sh[k]['mean']:.3f}  median {sh[k]['median']:.3f}")
    print(f"difference  mean {d['mean_diff']:+.3f}  median {d['median_diff']:+.3f}")
    return N


if __name__ == "__main__":
    main()
