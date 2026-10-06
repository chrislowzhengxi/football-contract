"""One-page meeting brief for the loan-timing work, plus pre-meeting QA.

    python -m src.analysis.loan_timing_brief

Reads the exploration's numbers file and recomputes the strict-sequence
validation; adds no new statistic. QA checks run first and the brief is
written only if they pass.
"""
from __future__ import annotations

import hashlib
import json

from . import loan_timing_validation as val
from .daniel_scope_final import f0
from .loan_episodes import CANONICAL_CSV
from .loan_timing_exploration import (NUMBERS_JSON, OUT, days, md_words, plot_lag_ecdf, plot_lag_histogram,
                                      return_date_group, share)

BRIEF_MD = OUT / "loan_timing_meeting_brief.md"
STAGE1C_SHA256 = "92f383d72a3ed59925c922f0d1b5f98528bc66ad3a9f49bb990867ed1d1d50dd"
FIGURES = {
    "bins": "figures/loan_timing_meeting_fig1_lag_bins.png",
    "day1": "figures/loan_timing_meeting_fig2_day1_return_dates.png",
    "fig1": "figures/loan_timing_fig1_strict_lag_histogram.png",
    "fig2": "figures/loan_timing_fig2_strict_ecdf.png",
    "fig3": "figures/loan_timing_fig3_ending_dates.png",
}


LAG_BINS = (("same day", 0, 0), ("1 day", 1, 1), ("2–3", 2, 3), ("4–7", 4, 7), ("8–14", 8, 14),
            ("15–21", 15, 21), ("22–30", 22, 30), ("31–60", 31, 60), ("61–120", 61, 120), (">120", 121, None))
DAY1_NAMED = ("06-30", "12-31", "01-31", "08-31", "05-31")
BLUE, GREY, INK, MUTED = "#2b6cb0", "#a0aec0", "#1a202c", "#555"


def lag_bins(A) -> list[dict]:
    lag = A.next_move_gap_days.astype(int)
    out = []
    for label, lo, hi in LAG_BINS:
        k = int(((lag >= lo) if hi is None else lag.between(lo, hi)).sum())
        out.append({"bin": label, "n": k, "share": k / len(A)})
    return out


def day1_composition(A) -> tuple[list[dict], int, dict]:
    """Return dates of the exactly-one-day conversions. The named dates are shown unless another date
    outnumbers one of them, which is the same as taking the five most common; the rest are "other"."""
    d1 = A[A.next_move_gap_days == 1]
    vc = d1.ending_mmdd.value_counts()
    keep = sorted(vc.index, key=lambda m: (-vc[m], m))[:len(DAY1_NAMED)]
    rows = [{"mmdd": m, "n": int(vc[m]), "share": vc[m] / len(d1)} for m in keep]
    other = int(len(d1) - sum(r["n"] for r in rows))
    rows.append({"mmdd": "other", "n": other, "share": other / len(d1)})
    rows_named = {m: int(vc.get(m, 0)) for m in DAY1_NAMED}
    assert all(m in keep for m in ("06-30", "12-31"))
    return rows, len(d1), rows_named


def meeting_figures(A) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    bins = lag_bins(A)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    xs = range(len(bins))
    ax.bar(xs, [b["n"] for b in bins], width=0.75, color=BLUE)
    top = max(b["n"] for b in bins)
    for x, b in zip(xs, bins):
        pc_ = "<0.1%" if 0 < b["share"] < 0.0005 else f"{100 * b['share']:.1f}%"
        ax.text(x, b["n"] + top * 0.012, f"{b['n']:,}\n{pc_}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b["bin"] for b in bins])
    ax.set_ylim(0, top * 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Days from the loan return to the permanent move (bins have unequal widths)")
    ax.set_ylabel("Loans")
    ax.yaxis.grid(True, color="#e2e8f0", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Strict A → B → A → B sequence (N = {len(A):,}): lag from return to permanent move", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / FIGURES["bins"], dpi=160)
    plt.close(fig)

    rows, n1, named = day1_composition(A)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ys = list(range(len(rows)))[::-1]
    labels = [md_words(r["mmdd"]) if r["mmdd"] != "other" else "other dates" for r in rows]
    ax.barh(ys, [r["n"] for r in rows], height=0.7, color=[GREY if r["mmdd"] == "other" else BLUE for r in rows])
    big = max(r["n"] for r in rows)
    for y, r in zip(ys, rows):
        ax.text(r["n"] + big * 0.01, y, f"{r['n']:,} ({100 * r['share']:.1f}%)", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, big * 1.25)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Next-day conversions")
    ax.xaxis.grid(True, color="#e2e8f0", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Strict next-day conversions (N = {n1:,}), by return date", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / FIGURES["day1"], dpi=160)
    plt.close(fig)
    return {"bins": bins, "day1": rows, "day1_n": n1, "day1_named": named}


def strict_figures(A) -> dict:
    """Figures 1-2 redrawn on the strict A -> B -> A -> B sequences only."""
    A = A.assign(return_date_group=return_date_group(A.ending_mmdd))
    seq = f"Strict A → B → A → B sequence (N = {len(A):,})"
    f1 = plot_lag_histogram(A, f"{seq}: days from return to permanent move", OUT / FIGURES["fig1"])
    f2 = plot_lag_ecdf(A, f"{seq}: cumulative share by lag", OUT / FIGURES["fig2"],
                       marks=(1, 21, 60))
    return {"fig1": f1, "fig2": f2}


def qa(E: dict, V: dict, S, F: dict) -> dict:
    A = S[S.origin == "A"]
    a = V["stats"]["A"]
    sha = hashlib.sha256(CANONICAL_CSV.read_bytes()).hexdigest()
    return {
        "Stage 1C unchanged": sha == STAGE1C_SHA256,
        "strict: permanent move from exact lender id": bool((A.next_move_from_club_id == A.lender_club_id).all()),
        "strict: permanent move to exact borrower id": bool((A.next_move_to_club_id == A.borrower_club_id).all()),
        "strict: return exactly B -> A": bool(A.return_exact_b_to_a.all()),
        "strict: zero intervening rows": bool((A.intervening_rows == 0).all()),
        "strict: no negative lag": bool((A.next_move_gap_days >= 0).all()),
        "any-origin count agrees with exploration": V["split"][0]["N"] == E["p1_any_lag"],
        "<=60-day count agrees with production B": V["split"][1]["N"] == E["p1_current_with_return"],
        "production count = with-return + no-return": E["p1_current"] == E["p1_current_with_return"] + E["p1_current_no_return"],
        "next-move table reproduces production": E["follow_on_mismatches"] == 0 and E["reclassify_mismatches"] == 0,
        "Figure 1 N and day-1 count = strict sample": F["fig1"]["N"] == a["N"] and F["fig1"]["day1"] == V["break"]["day1"],
        "Figure 2 N and <=1/21/60 shares = strict sample": F["fig2"]["N"] == a["N"] and all(
            abs(F["fig2"]["shares"][x] - a[f"within_{x}_share"]) < 1e-12 for x in (1, 21, 60)),
        "binned histogram covers every strict case": sum(b["n"] for b in F["meeting"]["bins"]) == a["N"],
        "binned day-1 bar = daily day-1 count": F["meeting"]["bins"][1]["n"] == V["break"]["day1"],
        "binned same-day + day-1 = within 1 day": F["meeting"]["bins"][0]["n"] + F["meeting"]["bins"][1]["n"] == a["within_1"],
        "day-1 composition sums to day-1 count": sum(r["n"] for r in F["meeting"]["day1"]) == V["break"]["day1"] == F["meeting"]["day1_n"],
    }


def render(E: dict, V: dict) -> str:
    a, r = V["stats"]["A"], V["stats"]["A, both dates realised"]
    br, cal, sp = V["break"], V["calendar"]["A"], V["split"][0]
    A3, U3 = E["p3"]["all_realised"], E["p3"]["uefa_q4_main"]
    p7 = E["p7"]
    n = a["N"]
    pct = lambda k: share(a[f"within_{k}"] / n)
    named = lambda P, md: share(P["named"][md][1])

    W: list[str] = []
    w = W.append
    w("# Loan timing: meeting brief")
    w("")
    w("*All numbers are generated by `python -m src.analysis.loan_timing_brief` from the frozen Stage 1C table. "
      "Details are in `loan_timing_exploration.md` and `loan_strict_sequence_check.md`.*")
    w("")
    w("## This week")
    w("")
    w(f"1. **Timing from loan return to permanent conversion.** There are {f0(n)} strict sequences: an A → B loan, "
      "a B → A return and an A → B permanent move, with exact club ids and no row in between. The lag is the "
      "permanent date minus the return date, as Transfermarkt records them. "
      f"{pct(1)} of conversions come within 1 day; the median lag is {days(a['median'])} day and p90 is "
      f"{days(a['p90'])} days.")
    w(f"2. **Immediate vs later conversions.** There is one sharp break: {f0(br['day1'])} conversions on day 1 against "
      f"{f0(br['day2'])} on day 2. After that the counts decline smoothly, and nothing marks 21 or 60 days. The "
      f"busiest later day is day {br['busiest_other_lag']} ({f0(br['busiest_other_n'])} cases). "
      f"{f0(V['may31_jul1_strict'])} of them are a return dated 31 May and a move dated 1 July, which is how British "
      f"loan ends are recorded. Counting all 31 May → 1 July conversions as immediate adds {f0(cal['added'])} "
      f"({share(cal['le1'] / n)} → {share(cal['le1_or_may31'] / n)}).")
    w(f"3. **Standard loan-ending dates.** Across all {f0(A3['N'])} realised endings, 30 June ({named(A3, '06-30')}), "
      f"31 December ({named(A3, '12-31')}) and 31 May ({named(A3, '05-31')}) cover {share(A3['top3_cover'])}. "
      f"After that the dates spread thin: covering 90% takes {A3['dates_for_90']} dates. "
      f"{len(E['p4_not_jun30_top'])} of {E['p4_countries_included']} lender countries have a most common date other "
      "than 30 June.")
    w("4. **Main interpretation.** Next-day conversion is the one sharp pattern in the data. It is how Transfermarkt "
      "records a move that takes effect as the loan ends (a 30 June return, a 1 July registration). That is "
      "consistent with a conversion arranged in advance, but it does not show an option or obligation. Beyond day "
      "1 there is no natural cutoff, so any wider \"immediate\" window is a choice, not a finding.")
    w("")
    w("## Key numbers")
    w("")
    w("| Measure | Value |")
    w("| --- | ---: |")
    w(f"| Strict sequences (A → B loan, B → A return, A → B permanent) | {f0(n)} |")
    for k in (1, 3, 7, 21, 60):
        w(f"| … permanent move within {k} day{'s' if k > 1 else ''} | {f0(a[f'within_{k}'])} ({pct(k)}) |")
    w(f"| … median / p90 lag | {days(a['median'])} / {days(a['p90'])} days |")
    w(f"| … both dates realised: N, within 1 day | {f0(r['N'])}, {share(r['within_1'] / r['N'])} |")
    w(f"| … within 1 day, or a 31 May return and 1 July move | {f0(cal['le1_or_may31'])} ({share(cal['le1_or_may31'] / n)}) |")
    for lab, P in ((f"All realised endings (N = {f0(A3['N'])})", A3), (f"Mapped-UEFA Q4 sample (N = {f0(U3['N'])})", U3)):
        w(f"| {lab}: share on 30 Jun / 31 Dec / 31 May | {named(P, '06-30')} / {named(P, '12-31')} / "
          f"{named(P, '05-31')} |")
        w(f"| … top 3 / top 5 dates cover | {share(P['top3_cover'])} / {share(P['top5_cover'])} |")
        w(f"| … dates needed for 80% / 90% / 95% | {P['dates_for_80']} / {P['dates_for_90']} / {P['dates_for_95']} |")
    w("")
    w(f"*Which count is which:* **{f0(n)}** is the strict sequence above. **{f0(sp['N'])}** also includes "
      f"{f0(sp['B'] + sp['C'])} conversions whose permanent move leaves from another side of the lender or another "
      f"club (used in the technical exploration report). **{f0(E['p1_current'])}** is last week's "
      f"production count: the move came within 60 days, and {E['p1_current_no_return']} of those loans have no return "
      "row. It is a counting check, not a timing population.")
    w("")
    M = V["figures"]["meeting"]
    bins = {b["bin"]: b for b in M["bins"]}
    nm = M["day1_named"]
    season = nm["06-30"] + nm["12-31"]
    deadline = nm["01-31"] + nm["08-31"]
    w("## Figures to show")
    w("")
    w(f"1. `{FIGURES['bins']}`: all {f0(n)} strict sequences in broad lag bins, at full scale. Exactly 1 day: "
      f"{f0(bins['1 day']['n'])} ({share(bins['1 day']['share'])}); same day: {f0(bins['same day']['n'])}; more "
      f"than 120 days: {f0(bins['>120']['n'])} ({share(bins['>120']['share'])}). The bins have unequal widths.")
    w(f"2. `{FIGURES['day1']}`: the {f0(M['day1_n'])} next-day conversions by return date. "
      f"{share(season / M['day1_n'])} follow a return on 30 June or 31 December (season end) and "
      f"{share(deadline / M['day1_n'])} a return on 31 January or 31 August (window deadline); 31 May accounts "
      f"for {f0(nm['05-31'])}. So most next-day conversions follow a season-end return, with window deadlines a "
      "distant second.")
    w(f"3. `{FIGURES['fig2']}`: the cumulative share of the same {f0(n)} by lag ({pct(1)} within 1 day, {pct(21)} "
      f"within 21, {pct(60)} within 60). The 31 May returns form a curve shifted by a month.")
    w(f"4. `{FIGURES['fig3']}`: the most common ending dates, and how many dates it takes to cover 80/90/95%. It "
      "shows all realised endings and the mapped-UEFA sample.")
    w("")
    w(f"Appendix: `{FIGURES['fig1']}`, the day-by-day histogram of the strict sequences (y-axis clipped to show "
      "the tail), and Figure 4 of the technical report (follow-on categories), which mixes the current 60-day "
      "production categories with unwindowed populations.")
    w("")
    w("## Questions for Daniel")
    w("")
    w(f"1. Should \"immediate conversion\" mean next-day ({pct(1)}), or the next registration date in calendar terms, "
      f"which also counts 31 May → 1 July ({share(cal['le1_or_may31'] / n)})?")
    w(f"2. Should standard ending dates be country-specific? The global top 3 cover less than 70% of endings in "
      f"{len(E['p4_global_top3_below_70'])} of {E['p4_countries_included']} countries: Britain, plus leagues that "
      "play a calendar-year season.")
    w("3. For the playing-time analysis, which outcome and window matter: minutes at the borrower during the loan, "
      "minutes at the lender in the season after the return, or both?")
    w("")
    w("## Next week: playing-time join")
    w("")
    w("- **Already joinable:** `loan_follow_on_timing.csv` has one row per loan. It joins to `appearances` on player "
      f"id, club id and date, and `minutes_played` is present on every row ({f0(p7['minutes_null'])} missing).")
    w(f"- **Expected coverage:** of the {f0(p7['loans_checked'])} loans since {p7['first_loan_season_checked']}, the "
      f"borrower has appearance rows during the loan for {f0(p7['loans_borrower_observed'])} "
      f"({share(p7['loans_borrower_observed'] / p7['loans_checked'])}). The player appears at least once in "
      f"{share(p7['loans_played_given_observed'] / p7['loans_borrower_observed'])} of those.")
    w(f"- **Main caveat:** appearances exist only for the {len(p7['leagues_with_appearances'])} long-covered "
      f"European leagues, part of their cups and the Champions/Europa Leagues; the {len(p7['leagues_without_appearances'])} "
      "leagues added in 2024 have none. Elsewhere, no appearances means missing data, not zero minutes. Unused "
      "substitutes appear only in `game_lineups`, which has no minutes.")
    return "\n".join(W) + "\n"


def summary(E: dict, V: dict, checks: dict, tests: str = "") -> str:
    a, A3 = V["stats"]["A"], E["p3"]["all_realised"]
    n = a["N"]
    lines = [f"{'STRICT N:':24}{n:,}"]
    for k in (1, 3, 7, 21, 60):
        lines.append(f"{f'<={k} DAY' + ('S' if k > 1 else '') + ':':24}{a[f'within_{k}']:,} ({100 * a[f'within_{k}'] / n:.1f}%)")
    lines += ["", f"{'TOP 3 DATE COVERAGE:':24}{100 * A3['top3_cover']:.1f}% (all realised endings)",
              f"{'TOP 5 DATE COVERAGE:':24}{100 * A3['top5_cover']:.1f}%",
              f"{'DATES FOR 90%:':24}{A3['dates_for_90']}", f"{'DATES FOR 95%:':24}{A3['dates_for_95']}", ""]
    lines += ["QA CHECKS:"] + [f"  {'PASS' if ok else 'FAIL'}  {k}" for k, ok in checks.items()]
    return "\n".join(lines)


def main() -> dict:
    E = json.loads(NUMBERS_JSON.read_text())
    V, S = val.compute()
    A = S[S.origin == "A"]
    V["may31_jul1_strict"] = int((A.may31_to_jul1_same_year & (A.next_move_gap_days == V["break"]["busiest_other_lag"])).sum())
    F = strict_figures(A)
    F["meeting"] = meeting_figures(A)
    V["figures"] = F
    checks = qa(E, V, S, F)
    if not all(checks.values()):
        print(summary(E, V, checks))
        raise SystemExit("QA failed; brief not written")
    BRIEF_MD.write_text(render(E, V))
    print(f"wrote {BRIEF_MD.name}")
    print(summary(E, V, checks))
    return {"E": E, "V": V, "checks": checks}


if __name__ == "__main__":
    main()
