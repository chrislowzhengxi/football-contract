"""Validation of the exact sequence A -> B loan, B -> A return, A -> B permanent.

    python -m src.analysis.loan_timing_validation

Takes the "return -> permanent move back to the borrower" cases of
`loan_timing_exploration` (any lag) and checks them four ways: where the
permanent move comes from, realised dates only, literal adjacency of the two
rows, and the British 31 May -> 1 July date convention. Read-only; no
definition is changed. Every number in the report is interpolated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import loan_scope
from .loan_episodes import _sorted
from .loan_timing_exploration import OUT, SKIP_TYPES, days, episode_table, lag_stats, next_moves, share

REPORT_MD = OUT / "loan_strict_sequence_check.md"
CHECK_CSV = OUT / "loan_strict_sequence_check.csv"
MARKS = (0, 1, 3, 7, 21, 30, 60)
LAG_CUTS = (None, 60, 21, 1)
ORIGINS = {"A": "Exact lender club → borrower",
           "B": "Other side of the lender's organisation → borrower",
           "C": "Different club → borrower"}


def skip_label(typ: str) -> str:
    return {"loan_return": "another End-of-loan row", "other": "draft / other row",
            "youth_or_internal": "youth or internal registration"}.get(typ, f"internal move labelled {typ}")


def strict_sequence(u) -> pd.DataFrame:
    """One row per loan whose return is followed, as the next substantive
    move, by a permanent move to the borrower (any lag)."""
    T = episode_table(u, next_moves(u))
    S = T[T.ending_is_return & T.next_move_is_permanent_to_borrower].copy()
    exact = S.next_move_from_club_id == S.lender_club_id
    same_org = ~exact & S.next_move_from_lender_organisation.fillna(False).astype(bool)
    S["origin"] = np.select([exact, same_org], ["A", "B"], default="C")

    rows = u.rows.set_index("event_id")
    ret = rows.loc[S.ending_event_id]
    S["return_from_club_id"] = ret.from_club_id.to_numpy()
    S["return_to_club_id"] = ret.to_club_id.to_numpy()
    S["return_exact_b_to_a"] = ((S.return_from_club_id == S.borrower_club_id)
                               & (S.return_to_club_id == S.lender_club_id))

    d = _sorted(u.rows)
    pos = pd.Series(np.arange(len(d)), index=d.event_id.to_numpy())
    typ = d.transfer_type_normalized.to_numpy()
    internal = d.is_internal_move.to_numpy().astype(bool)
    pid = d.player_id.to_numpy()
    n_skip, kinds = [], []
    for ret_id, nxt_id, player in zip(S.ending_event_id, S.next_move_event_id, S.player_id):
        i, j = pos[ret_id], pos[nxt_id]
        between = range(i + 1, j)
        assert all(pid[k] == player and (internal[k] or typ[k] in SKIP_TYPES) for k in between)
        n_skip.append(len(between))
        kinds.append("; ".join(skip_label(typ[k]) for k in between))
    S["intervening_rows"] = n_skip
    S["intervening_row_types"] = kinds

    S["permanent_mmdd"] = pd.to_datetime(S.next_move_date).dt.strftime("%m-%d")
    S["both_realised"] = ~S.ending_scheduled & ~S.next_move_scheduled
    lag = S.next_move_gap_days
    S["calendar_immediate_1d"] = lag <= 1
    S["may31_to_jul1_same_year"] = (S.ending_mmdd == "05-31") & (S.permanent_mmdd == "07-01") & (lag == 31)
    S["calendar_immediate_with_may31"] = S.calendar_immediate_1d | S.may31_to_jul1_same_year
    return S


def core(g: pd.Series) -> dict:
    return lag_stats(g, MARKS)


def compute(u=None) -> tuple[dict, pd.DataFrame]:
    u = u or loan_scope.universe()
    S = strict_sequence(u)
    lag = S.next_move_gap_days
    N: dict = {"any_lag": len(S), "current_3816": int((lag <= 60).sum())}

    split = []
    for cut in LAG_CUTS:
        g = S if cut is None else S[lag <= cut]
        rec = {"lag": "any" if cut is None else f"<= {cut}", "N": len(g)}
        for o in ORIGINS:
            rec[o] = int((g.origin == o).sum())
            rec[f"{o}_share"] = rec[o] / len(g)
        split.append(rec)
    N["split"] = split

    A = S[S.origin == "A"]
    AB = S[S.origin.isin(["A", "B"])]
    groups = {"A": A, "B": S[S.origin == "B"], "A+B": AB, "C": S[S.origin == "C"],
              "A, return also exactly B → A": A[A.return_exact_b_to_a],
              "A, both dates realised": A[A.both_realised], "A+B, both dates realised": AB[AB.both_realised]}
    N["stats"] = {k: core(g.next_move_gap_days) for k, g in groups.items()}
    N["A_return_not_exact"] = int((~A.return_exact_b_to_a).sum())
    N["A_return_not_exact_reasons"] = {k: int(v) for k, v in A.loc[~A.return_exact_b_to_a, "match_reason"].value_counts().items()}

    daily = A.next_move_gap_days.astype(int).value_counts()
    other = daily.drop(index=[1], errors="ignore")
    N["break"] = {"day0": int(daily.get(0, 0)), "day1": int(daily.get(1, 0)), "day2": int(daily.get(2, 0)),
                  "busiest_other_lag": int(other.idxmax()), "busiest_other_n": int(other.max())}
    N["break"]["ratio"] = N["break"]["day1"] / N["break"]["busiest_other_n"]
    N["break"]["survives"] = N["break"]["ratio"] >= 10

    R = A[A.both_realised]
    N["realised_excluded"] = {"return_scheduled": int(A.ending_scheduled.sum()),
                              "permanent_scheduled_return_realised": int((~A.ending_scheduled & A.next_move_scheduled).sum())}

    adj = []
    for label, m in (("0", A.intervening_rows == 0), ("1", A.intervening_rows == 1), ("2+", A.intervening_rows >= 2)):
        st = core(A.loc[m, "next_move_gap_days"])
        adj.append({"intervening": label, "n": int(m.sum()), "share": m.mean(),
                    "within_1_share": st.get("within_1_share"), "median": st.get("median")})
    N["adjacency"] = adj
    kinds = A.intervening_row_types[A.intervening_rows > 0].str.split("; ").explode()
    N["adjacency_kinds"] = {k: int(v) for k, v in kinds.value_counts().items()}
    N["adjacency_kinds_loans"] = {k: int(A.intervening_row_types.str.contains(k, regex=False).sum())
                                  for k in N["adjacency_kinds"]}

    N["calendar"] = {}
    for name, g in (("A", A), ("A, both dates realised", R)):
        k1, k2 = int(g.calendar_immediate_1d.sum()), int(g.calendar_immediate_with_may31.sum())
        N["calendar"][name] = {"N": len(g), "le1": k1, "le1_or_may31": k2, "added": k2 - k1}
    N["may31_not_same_year"] = int(((A.ending_mmdd == "05-31") & (A.permanent_mmdd == "07-01")
                                    & (A.next_move_gap_days != 31)).sum())
    kinds_all = S.loc[S.intervening_rows > 0, ["origin", "intervening_row_types"]]
    N["adjacency_kinds_by_origin"] = {
        o: {k: int(v) for k, v in g.intervening_row_types.str.split("; ").explode().value_counts().items()}
        for o, g in kinds_all.groupby("origin")}
    N["adjacency_cases_by_origin"] = {o: int(v) for o, v in kinds_all.origin.value_counts().items()}
    N["C_cases"] = S.loc[S.origin == "C", ["player_name", "lender_club", "borrower_club", "next_move_from_club",
                                           "next_move_gap_days"]].to_dict("records")
    N["B_examples"] = (S.loc[S.origin == "B", ["lender_club", "next_move_from_club"]]
                       .value_counts().head(3).reset_index().to_dict("records"))
    return N, S


def render(N: dict) -> str:
    from .daniel_scope_final import f0, md_table

    W: list[str] = []
    w = W.append

    def cs(k, n):
        return f"{f0(k)} ({share(k / n) if n else '–'})"

    st = N["stats"]
    a = st["A"]
    br = N["break"]
    w("# Strict-sequence check: A → B loan, B → A return, A → B permanent")
    w("")
    w("*Validation of the \"return → permanent move back to the borrower\" cases in `loan_timing_exploration.md`. "
      "Generated by `python -m src.analysis.loan_timing_validation`; no definition, classification or earlier "
      "statistic is changed. Case-level flags are in `loan_strict_sequence_check.csv`.*")
    w("")
    w("## 1. Direction: where does the permanent move come from?")
    w("")
    w(f"All {f0(N['any_lag'])} cases have the loan A → B and the permanent move going to B (that is how they were "
      "selected). What varies is the permanent move's origin. **A** means the exact lender club id; **B** means "
      "another side of the lender's organisation, by Stage 1C's club-name rule (e.g. the loan left from Atalanta U19 "
      "and the sale came from Atalanta); **C** means any other club.")
    w("")
    W += md_table(["Lag", "Cases", "A. Exact lender → B", "B. Same lender organisation → B", "C. Different club → B"],
                  [[r["lag"], f0(r["N"])] + [cs(r[o], r["N"]) for o in ORIGINS] for r in N["split"]])
    w("")
    w("The C cases, in full:")
    w("")
    W += md_table(["Player", "Loan", "Permanent move from", "Lag (days)"],
                  [[r["player_name"], f"{r['lender_club']} → {r['borrower_club']}", r["next_move_from_club"],
                    days(r["next_move_gap_days"])] for r in N["C_cases"]], right={3})
    w("")
    rows = []
    for k in ("A", "B", "A+B", "C", "A, return also exactly B → A"):
        s = st[k]
        rows.append([k, f0(s["N"])] + [cs(s[f"within_{x}"], s["N"]) for x in MARKS]
                    + [days(s.get("median")), days(s.get("p90")), days(s.get("p95"))])
    W += md_table(["Group", "N", "Same day"] + [f"≤ {x} d" for x in MARKS if x] + ["Median", "p90", "p95"], rows)
    w("")
    w("Several C origins are youth or reserve sides whose names Stage 1C's club-name rule does not link to the "
      "lender (see the table), so C is an upper bound on genuinely different origins.")
    w("")
    if N["A_return_not_exact"] == 0:
        w(f"All {f0(a['N'])} exact-origin cases also have a return row going exactly B → A, so group A is the full "
          "exact sequence A → B loan, B → A return, A → B permanent; the last row repeats group A.")
    else:
        w(f"{f0(N['A_return_not_exact'])} of the {f0(a['N'])} exact-origin cases have a return row that went to "
          "another side of the lender rather than exactly B → A ("
          + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in N["A_return_not_exact_reasons"].items())
          + "); the last row requires an exact B → A return as well.")
    w("")
    w(f"**The day-1 / day-2 break survives the strict check: {'yes' if br['survives'] else 'no'}.** In group A, "
      f"{f0(br['day1'])} cases fall on day 1 and {f0(br['day2'])} on day 2 ({f0(br['day0'])} on day 0). The busiest "
      f"other day is day {br['busiest_other_lag']} with {f0(br['busiest_other_n'])}, so day 1 is {br['ratio']:.0f} "
      "times the busiest other day. (The yes/no rule used is \"day 1 at least 10 times the busiest other day\".)")
    w("")
    w("## 2. Realised dates only")
    w("")
    ex = N["realised_excluded"]
    w(f"Keeping group A cases whose return and permanent move had both happened at capture. This excludes "
      f"{f0(ex['return_scheduled'])} with a scheduled return and {f0(ex['permanent_scheduled_return_realised'])} with "
      "a realised return but a scheduled move. Those stay in the CSV (`both_realised` = False).")
    w("")
    rows = []
    for k in ("A", "A, both dates realised", "A+B, both dates realised"):
        s = st[k]
        rows.append([k, f0(s["N"])] + [cs(s[f"within_{x}"], s["N"]) for x in MARKS]
                    + [days(s.get("median")), days(s.get("p90")), days(s.get("p95"))])
    W += md_table(["Group", "N", "Same day"] + [f"≤ {x} d" for x in MARKS if x] + ["Median", "p90", "p95"], rows)
    w("")
    w("## 3. Literal adjacency")
    w("")
    w("The next-move routine passes over internal registrations, other End-of-loan rows and rows normalised as "
      "\"other\". Group A, by how many such rows sit between the return and the permanent move in the player's "
      "history:")
    w("")
    W += md_table(["Intervening rows", "Cases", "Share", "≤ 1 day", "Median lag"],
                  [[r["intervening"], f0(r["n"]), share(r["share"]), share(r["within_1_share"]), days(r["median"])]
                   for r in N["adjacency"]])
    w("")
    if N["adjacency_kinds"]:
        w("Types of intervening row (rows, and cases containing at least one):")
        w("")
        W += md_table(["Row type", "Rows", "Cases"],
                      [[k, f0(v), f0(N["adjacency_kinds_loans"][k])] for k, v in N["adjacency_kinds"].items()])
        w("")
    else:
        w("No row of any type sits between the return and the permanent move in any group A case: each is literally "
          "the next row. Intervening rows occur only outside group A: "
          + "; ".join(f"group {o}, {f0(N['adjacency_cases_by_origin'][o])} cases ("
                      + ", ".join(f"{k} {v}" for k, v in ks.items()) + ")"
                      for o, ks in N["adjacency_kinds_by_origin"].items()) + ".")
        w("")
    w("## 4. Calendar-aware sensitivity (descriptive only, not a rule)")
    w("")
    W += md_table(["Group", "N", "A. lag ≤ 1 day", "B. lag ≤ 1 day, or 31 May return and 1 July move", "Added by B"],
                  [[k, f0(c["N"]), cs(c["le1"], c["N"]), cs(c["le1_or_may31"], c["N"]), f0(c["added"])]
                   for k, c in N["calendar"].items()])
    w("")
    k = N["may31_not_same_year"]
    w("\"31 May → 1 July\" requires the same year (a 31-day lag). "
      + ("No case is excluded by that." if k == 0 else
         f"{'One' if k == 1 else f0(k)} 31 May return{'' if k == 1 else 's'} followed by a 1 July move a year or more "
         f"later {'is' if k == 1 else 'are'} not counted."))
    w("")
    return "\n".join(W) + "\n"


def summary(N: dict) -> str:
    a = N["stats"]["A"]
    r = N["stats"]["A, both dates realised"]
    sp = N["split"][0]
    adj = {x["intervening"]: x for x in N["adjacency"]}
    cal = N["calendar"]["A"]
    p = lambda k, n: f"{k:,} ({100 * k / n:.1f}%)"
    return "\n".join([
        f"{'ANY-LAG TO-BORROWER CASES:':42}{sp['N']:,}",
        f"{'STRICT A->B:':42}{p(sp['A'], sp['N'])}",
        f"{'SAME LENDER ORGANISATION -> B:':42}{p(sp['B'], sp['N'])}",
        f"{'DIFFERENT ORIGIN -> B:':42}{p(sp['C'], sp['N'])}",
        "",
        f"{'STRICT WITHIN 1 DAY:':42}{p(a['within_1'], a['N'])}",
        f"{'STRICT WITHIN 21 DAYS:':42}{p(a['within_21'], a['N'])}",
        f"{'STRICT WITHIN 60 DAYS:':42}{p(a['within_60'], a['N'])}",
        f"{'STRICT MEDIAN:':42}{days(a['median'])} days",
        f"{'STRICT P90:':42}{days(a['p90'])} days",
        "",
        f"{'STRICT REALISED-ONLY N:':42}{r['N']:,}",
        f"{'STRICT REALISED WITHIN 1 DAY:':42}{p(r['within_1'], r['N'])}",
        "",
        f"{'ZERO INTERVENING ROWS:':42}{p(adj['0']['n'], a['N'])}",
        f"{'ONE INTERVENING ROW:':42}{p(adj['1']['n'], a['N'])}",
        f"{'TWO+ INTERVENING ROWS:':42}{p(adj['2+']['n'], a['N'])}",
        "",
        f"{'CALENDAR-IMMEDIATE (<=1 DAY):':42}{p(cal['le1'], cal['N'])}",
        f"{'CALENDAR-IMMEDIATE (+ 31 MAY -> 1 JULY):':42}{p(cal['le1_or_may31'], cal['N'])}",
        "",
        f"{'DAY-1 BREAK SURVIVES STRICT CHECK:':42}{'YES' if N['break']['survives'] else 'NO'} "
        f"(day 1 {N['break']['day1']:,} vs day 2 {N['break']['day2']:,}; busiest other day "
        f"{N['break']['busiest_other_lag']} with {N['break']['busiest_other_n']:,})",
    ])


def main() -> dict:
    N, S = compute()
    cols = ["origin", "player_id", "player_name", "loan_event_id", "loan_season", "lender_club_id", "lender_club",
            "borrower_club_id", "borrower_club", "lender_country", "borrower_country", "loan_start_date",
            "ending_event_id", "ending_date", "ending_mmdd", "ending_scheduled", "return_from_club_id",
            "return_to_club_id", "return_exact_b_to_a", "match_reason", "next_move_event_id", "next_move_date",
            "permanent_mmdd", "next_move_type", "next_move_from_club_id", "next_move_from_club", "next_move_scheduled",
            "next_move_gap_days", "both_realised", "intervening_rows", "intervening_row_types",
            "calendar_immediate_1d", "may31_to_jul1_same_year", "calendar_immediate_with_may31"]
    out = S[cols].rename(columns={"ending_event_id": "return_event_id", "ending_date": "return_date",
                                  "ending_mmdd": "return_mmdd", "ending_scheduled": "return_scheduled",
                                  "next_move_event_id": "permanent_event_id", "next_move_date": "permanent_date",
                                  "next_move_type": "permanent_type", "next_move_from_club_id": "permanent_from_club_id",
                                  "next_move_from_club": "permanent_from_club",
                                  "next_move_scheduled": "permanent_scheduled", "next_move_gap_days": "lag_days"})
    for c in ("loan_start_date", "return_date", "permanent_date"):
        out[c] = pd.to_datetime(out[c]).dt.date
    for c in ("player_id", "lender_club_id", "borrower_club_id", "return_from_club_id", "return_to_club_id",
              "permanent_from_club_id", "lag_days"):
        out[c] = pd.to_numeric(out[c]).astype("Int64")
    out.to_csv(CHECK_CSV, index=False)
    REPORT_MD.write_text(render(N))
    print(f"wrote {REPORT_MD.name}, {CHECK_CSV.name}")
    print(summary(N))
    return N


if __name__ == "__main__":
    main()
