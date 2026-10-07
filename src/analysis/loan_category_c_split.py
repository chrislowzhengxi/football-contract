"""Split Category C ("return, then another move or re-loan") by the actual next movement.

    python -m src.analysis.loan_category_c_split

Read-only. Category C is taken exactly as `loan_timing_exploration` defines it
(the production classes `immediate_follow_on_transfer` and `early_termination`,
7,853 loans). Its follow-on move is the production follow-on, which is the
player's next substantive move after the return. Every C loan lands in exactly
one subgroup (checked). The strict A -> B -> A -> B conversions are a different
population (Category B) and are not touched.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from .daniel_scope_final import f0, md_table
from .loan_episodes import ROOT
from .loan_timing_exploration import (FOLLOW_ON_IMMEDIATE_DAYS, FOLLOW_ON_WINDOW_DAYS, episode_table, md_words,
                                      next_moves, share)
from ..stage1c.policy import normalize_club_name

OUT = ROOT / "data" / "outputs" / "rebuild"
SPLIT_CSV = OUT / "loan_category_c_split.csv"
COUNTRY_CSV = OUT / "loan_category_c_split_by_country.csv"
REPORT_MD = OUT / "loan_category_c_split_summary.md"
FIGURE = "figures/loan_category_c_split_timing.png"
NUMBERS_JSON = OUT / "loan_category_c_split_numbers.json"

SUBGROUPS = {
    "C1": "Re-loan to the same borrower (A → B loan, B → A return, A → B loan)",
    "C2": "Loan to a different club (A → B loan, B → A return, A → C loan)",
    "C3": "Free transfer to a third club",
    "C4": "Move with no fee shown to a third club",
    "C5": "Other",
}
MARKS = (1, 3, 7, 14, 21, 30, 60)
BINS = (("same day", 0, 0), ("1 day", 1, 1), ("2–3", 2, 3), ("4–7", 4, 7), ("8–14", 8, 14), ("15–21", 15, 21),
        ("22–30", 22, 30), ("31–60", 31, 60))


def split(T: pd.DataFrame) -> pd.DataFrame:
    """Category C loans with `subgroup` (C1-C5) and, for C5, `other_detail`. Rules apply in order."""
    C = T[T.sequence_category == "C"].copy()
    assert (C.next_move_event_id == C.current_follow_on_event_id).all()
    lorg = C.lender_club.map(normalize_club_name)
    borg = C.borrower_club.map(normalize_club_name)
    to_org = C.next_move_to_club.map(normalize_club_name)
    from_lender_org = (C.next_move_from_club_id == C.lender_club_id) | \
        C.next_move_from_lender_organisation.fillna(False).astype(bool)
    typ = C.next_move_type
    loan = typ.eq("loan")
    to_b = C.next_move_to_club_id == C.borrower_club_id
    to_b_org = ~to_b & (to_org == borg)
    to_a_org = to_org == lorg
    rules = [
        (~typ.isin(["loan", "free_transfer", "no_fee_shown"]), "C5", "next move of an unknown type"),
        (~from_lender_org, "C5", "next move does not start from the lender's organisation (a row is probably missing)"),
        (loan & to_b, "C1", ""),
        (loan & to_b_org, "C5", "loan to another side of the borrower's organisation"),
        (loan, "C2", ""),
        (~loan & (to_b | to_b_org), "C5", "free or no-fee move to another side of the borrower's organisation"),
        (~loan & to_a_org, "C5", "free or no-fee move within the lender's organisation"),
        (typ.eq("free_transfer"), "C3", ""),
        (typ.eq("no_fee_shown"), "C4", ""),
    ]
    C["subgroup"] = np.select([m for m, _, _ in rules], [g for _, g, _ in rules], default="")
    C["other_detail"] = np.select([m for m, _, _ in rules], [d for _, _, d in rules], default="")
    assert (C.subgroup != "").all(), "a Category C loan matched no subgroup"
    C["next_move_from_exact_lender"] = C.next_move_from_club_id == C.lender_club_id
    C["lag_days"] = C.current_follow_on_gap_days.astype(int)
    return C


def stats(g: pd.DataFrame, n_c: int) -> dict:
    lag = g.lag_days
    q = lambda p: float(np.quantile(lag, p, method="inverted_cdf")) if len(lag) else None
    rec = {"N": len(g), "share_of_c": len(g) / n_c, "median": q(.5), "p25": q(.25), "p75": q(.75), "p90": q(.9)}
    for x in MARKS:
        rec[f"within_{x}"] = float((lag <= x).mean()) if len(lag) else None
    vc = g.ending_mmdd.value_counts()
    rec["top_return_dates"] = [{"mmdd": m, "n": int(v), "share": v / len(g)} for m, v in vc.head(3).items()]
    rec["existing_label"] = {k: int(v) for k, v in g.economic_ending.value_counts().items()}
    rec["from_exact_lender"] = float(g.next_move_from_exact_lender.mean()) if len(g) else None
    rec["lender_mapped"] = int(g.lender_country.notna().sum())
    rec["top_lender_countries"] = [{"country": c, "n": int(v)} for c, v in g.lender_country.value_counts().head(5).items()]
    rec["domestic"] = int((g.lender_country.notna() & (g.lender_country == g.borrower_country)).sum())
    rec["bins"] = [{"bin": b, "n": int(lag.between(lo, hi).sum())} for b, lo, hi in BINS]
    return rec


def figure(C: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    groups = list(SUBGROUPS)
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 3.9), sharey=True)
    for ax, k in zip(axes, groups):
        g = C[C.subgroup == k]
        shares = [g.lag_days.between(lo, hi).mean() * 100 if len(g) else 0 for _, lo, hi in BINS]
        ax.bar(range(len(BINS)), shares, width=0.75, color="#2b6cb0" if k != "C5" else "#a0aec0")
        for x, s in enumerate(shares):
            if s >= 0.5:
                ax.text(x, s + 1, f"{s:.0f}%", ha="center", fontsize=7.5, color="#1a202c")
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels([b for b, _, _ in BINS], rotation=60, ha="right", fontsize=7.5)
        title = {"C1": "C1 re-loan to same borrower", "C2": "C2 loan to a different club",
                 "C3": "C3 free transfer, third club", "C4": "C4 no fee shown, third club", "C5": "C5 other"}[k]
        ax.set_title(f"{title}\n(N = {len(g):,})", loc="left", fontsize=9)
        ax.yaxis.grid(True, color="#e2e8f0", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of the subgroup (%)")
    fig.suptitle(f"Category C by next move: days from the loan return to the next move. Loans count up to "
                 f"{FOLLOW_ON_WINDOW_DAYS} days and other moves up to {FOLLOW_ON_IMMEDIATE_DAYS} days, by the "
                 "existing definition, so C3–C4 cannot have lags of 22–60 days. Bins have unequal widths.",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compute(u=None) -> tuple[dict, pd.DataFrame]:
    u = u or loan_scope.universe()
    T = episode_table(u, next_moves(u))
    C = split(T)
    n_c = len(C)
    N = {"category_c": n_c, "loans": len(T),
         "category_counts": {k: int(v) for k, v in T.sequence_category.value_counts().items()},
         "subgroups": {k: stats(C[C.subgroup == k], n_c) for k in SUBGROUPS},
         "other_detail": {k: int(v) for k, v in C.loc[C.subgroup == "C5", "other_detail"].value_counts().items()}}
    assert sum(s["N"] for s in N["subgroups"].values()) == n_c
    assert C.loan_event_id.is_unique
    N["windows"] = {"loan": FOLLOW_ON_WINDOW_DAYS, "other": FOLLOW_ON_IMMEDIATE_DAYS}
    N["c1_lag_beyond_immediate"] = int((C[C.subgroup == "C1"].lag_days > FOLLOW_ON_IMMEDIATE_DAYS).sum())
    N["c2_lag_beyond_immediate"] = int((C[C.subgroup == "C2"].lag_days > FOLLOW_ON_IMMEDIATE_DAYS).sum())
    return N, C


def render(N: dict) -> str:
    W: list[str] = []
    w = W.append
    sg = N["subgroups"]
    n = N["category_c"]
    w("# Category C, split by what actually happens next")
    w("")
    w("*Generated by `python -m src.analysis.loan_category_c_split`. Category C is unchanged: the "
      f"{f0(n)} loans whose return is followed by another move or re-loan (the existing classes "
      "`immediate_follow_on_transfer` and `early_termination`). The strict A → B → A → B conversions are Category B, "
      "a separate population, and are not included.*")
    w("")
    w(f"**By the existing definition, Category C is window-limited.** A new loan counts if it comes within "
      f"{N['windows']['loan']} days of the return, and a free or no-fee move only within {N['windows']['other']} days. "
      "So every C2 and C1 lag is at most 60 days, and every C3 and C4 lag at most 21 days.")
    w("")
    w("## The split")
    w("")
    w("Rules apply in this order, and each loan takes the first that fits: unknown type → C5; next move not starting "
      "from the lender's organisation → C5; loan to the same borrower → C1; loan to another side of the borrower's "
      "organisation → C5; any other loan → C2; free or no-fee move to the borrower's or the lender's own organisation "
      "→ C5; free transfer → C3; no fee shown → C4.")
    w("")
    W += md_table(["Subgroup", "N", "Share of C", "Median lag", "p25", "p75", "p90"] + [f"≤ {x} d" for x in MARKS],
                  [[f"{k}. {SUBGROUPS[k]}", f0(s["N"]), share(s["share_of_c"]), f"{s['median']:.0f}",
                    f"{s['p25']:.0f}", f"{s['p75']:.0f}", f"{s['p90']:.0f}"] + [share(s[f"within_{x}"]) for x in MARKS]
                   for k, s in sg.items()]
                  + [["**Total = Category C**", f"**{f0(n)}**", "**100.0%**"] + [""] * (4 + len(MARKS))])
    w("")
    w("C5 in detail: " + "; ".join(f"{k} ({f0(v)})" for k, v in N["other_detail"].items()) + ".")
    w("")
    w(f"![Category C subgroups]({FIGURE})")
    w("")
    w("## Return dates, existing labels and countries")
    w("")
    rows = []
    for k, s in sg.items():
        rows.append([k, ", ".join(f"{md_words(d['mmdd'])} {share(d['share'])}" for d in s["top_return_dates"]),
                     ", ".join(f"`{lab}` {f0(v)}" for lab, v in s["existing_label"].items()),
                     share(s["from_exact_lender"]),
                     ", ".join(f"{d['country']} {f0(d['n'])}" for d in s["top_lender_countries"]),
                     f"{f0(s['lender_mapped'])} / {f0(s['domestic'])}"])
    W += md_table(["Subgroup", "Most common return dates", "Existing class", "Next move from the exact lender club",
                   "Top lender countries", "Lender mapped / domestic"], rows, right={3, 5})
    w("")
    c1, c2, c3, c4 = sg["C1"], sg["C2"], sg["C3"], sg["C4"]
    w("## What differs")
    w("")
    w(f"- **Re-loans to the same borrower (C1) and loans elsewhere (C2) have different timing.** C1: "
      f"{share(c1['within_1'])} of re-loans come within 1 day and the median lag is {c1['median']:.0f} days; C2: "
      f"{share(c2['within_1'])} and {c2['median']:.0f} days. {f0(N['c1_lag_beyond_immediate'])} C1 and "
      f"{f0(N['c2_lag_beyond_immediate'])} C2 loans are in C only because a new loan counts up to "
      f"{N['windows']['loan']} days.")
    w(f"- **Free (C3) and no-fee (C4) moves:** {share(c3['within_1'])} and {share(c4['within_1'])} within 1 day. Their "
      f"lags cannot exceed {N['windows']['other']} days, so their medians ({c3['median']:.0f} and "
      f"{c4['median']:.0f}) are not comparable with C1 and C2.")
    with_et = [k for k, s in sg.items() if s["existing_label"].get("early_termination", 0) > 0]
    w(f"- The existing labels cut across the subgroups: `early_termination` (a return away from 30 June and 31 "
      f"December followed quickly by another move) appears in {len(with_et)} of {len(sg)} subgroups ("
      + ", ".join(f"{k} {f0(sg[k]['existing_label']['early_termination'])}" for k in with_et)
      + "). So it describes the return date, not the kind of next move.")
    w("")
    w("Case-level file: `loan_category_c_split.csv` (one row per Category C loan, with `subgroup` and "
      "`other_detail`). Subgroup × lender country counts are in `loan_category_c_split_by_country.csv`.")
    w("")
    return "\n".join(W) + "\n"


def main() -> dict:
    N, C = compute()
    cols = ["subgroup", "other_detail", "loan_event_id", "player_id", "player_name", "loan_season", "lender_club_id",
            "lender_club", "borrower_club_id", "borrower_club", "lender_country", "borrower_country",
            "ending_event_id", "ending_date", "ending_mmdd", "ending_scheduled", "economic_ending",
            "next_move_event_id", "next_move_date", "next_move_type", "next_move_from_club_id", "next_move_from_club",
            "next_move_from_exact_lender", "next_move_to_club_id", "next_move_to_club", "next_move_scheduled",
            "lag_days"]
    out = C[cols].copy()
    for c in ("ending_date", "next_move_date"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(SPLIT_CSV, index=False)
    (C.assign(lender_country=C.lender_country.fillna("(unmapped)"))
       .groupby(["subgroup", "lender_country"]).size().rename("loans").reset_index()
       .to_csv(COUNTRY_CSV, index=False))
    (OUT / "figures").mkdir(exist_ok=True)
    figure(C, OUT / FIGURE)
    NUMBERS_JSON.write_text(json.dumps(N, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    REPORT_MD.write_text(render(N))
    print(f"wrote {REPORT_MD.name}, {SPLIT_CSV.name}, {FIGURE}")
    print("Category C", f"{N['category_c']:,}", {k: s["N"] for k, s in N["subgroups"].items()})
    return N


if __name__ == "__main__":
    main()
