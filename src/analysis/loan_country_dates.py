"""Loan-ending dates by country, the COVID period, and candidate rules for normal vs unusual endings.

    python -m src.analysis.loan_country_dates

Read-only and descriptive. Uses the realised loan endings of the current loan
universe. Club countries come only from the snapshot's deterministic mapping.
The dates reported are dates that are common in our data; the repository has no
official transfer-window calendar, so none of them is called a window boundary.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from .daniel_scope_final import f0, md_table
from .loan_category_c_split import split as c_split
from .loan_episodes import ROOT
from .loan_timing_exploration import dates_needed, episode_table, md_words, mmdd_distribution, next_moves, share
from .stage1_scope_report import display_country as dc

OUT = ROOT / "data" / "outputs" / "rebuild"
REPORT_MD = OUT / "loan_country_date_analysis.md"
COUNTRY_CSV = OUT / "loan_country_date_by_country.csv"
RULES_CSV = OUT / "loan_country_date_rules.csv"
FIGURE = "figures/loan_country_date_heatmap.png"
NUMBERS_JSON = OUT / "loan_country_date_numbers.json"

MIN_N = 100
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-02-28")
CONTEXTS = ("domestic", "cross-border", "one or both clubs unmapped")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def context(R: pd.DataFrame) -> pd.Series:
    both = R.lender_country.notna() & R.borrower_country.notna()
    return pd.Series(np.select([both & (R.lender_country == R.borrower_country), both],
                               ["domestic", "cross-border"], default="one or both clubs unmapped"), index=R.index)


def profile(g: pd.DataFrame) -> dict:
    d = mmdd_distribution(g)
    month = g.ending_date.dt.month.value_counts(normalize=True)
    rec = {"N": len(g), "top1_cover": float(d.cumulative_share.iloc[0]),
           "top3_cover": float(d.cumulative_share.iloc[min(2, len(d) - 1)]),
           "top5_cover": float(d.cumulative_share.iloc[min(4, len(d) - 1)]),
           "dates_for_80": dates_needed(d, .8), "dates_for_90": dates_needed(d, .9),
           "top_month": MONTHS[int(month.index[0]) - 1], "top_month_share": float(month.iloc[0])}
    for i in range(5):
        rec[f"date{i + 1}"] = d.mmdd.iloc[i] if i < len(d) else None
        rec[f"date{i + 1}_share"] = float(d.share.iloc[i]) if i < len(d) else None
    return rec


def country_table(R: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows = []
    for ctx, col in (("domestic", "lender_country"), ("by borrower country (all loans)", "borrower_country"),
                     ("by lender country (all loans)", "lender_country")):
        frame = R[R.context == "domestic"] if ctx == "domestic" else R[R[col].notna()]
        for c, g in frame.groupby(col):
            rows.append({"sample": sample, "attribution": ctx, "country": c, "adequate": len(g) >= MIN_N, **profile(g)})
    return pd.DataFrame(rows).sort_values(["sample", "attribution", "N"], ascending=[True, True, False])


def attribution_test(R: pd.DataFrame, modal: dict) -> dict:
    X = R[(R.context == "cross-border") & R.lender_country.isin(modal) & R.borrower_country.isin(modal)].copy()
    X["lm"], X["bm"] = X.lender_country.map(modal), X.borrower_country.map(modal)
    Y = X[X.lm != X.bm]
    pairs = (Y.groupby(["lender_country", "borrower_country"])
              .apply(lambda g: pd.Series({"n": len(g), "on_borrower_modal": int((g.ending_mmdd == g.bm).sum()),
                                          "on_lender_modal": int((g.ending_mmdd == g.lm).sum()),
                                          "borrower_modal": g.bm.iloc[0], "lender_modal": g.lm.iloc[0]}))
              .reset_index().sort_values("n", ascending=False))
    cross = R[R.context == "cross-border"]
    return {"n": len(Y), "on_borrower_modal": int((Y.ending_mmdd == Y.bm).sum()),
            "on_lender_modal": int((Y.ending_mmdd == Y.lm).sum()), "pairs": pairs.head(8).to_dict("records"),
            "cross_border": profile(cross),
            "england_domestic": profile(R[(R.context == "domestic") & (R.lender_country == "England")]),
            "england_one_side": profile(R[(R.context == "cross-border") & ((R.lender_country == "England")
                                                                          | (R.borrower_country == "England"))])}


def major_dates(R: pd.DataFrame, dom_adequate: list[str]) -> list[dict]:
    """Where each major date comes from: domestic loans by country, cross-border, unmapped."""
    glob = mmdd_distribution(R)
    D = R[R.context == "domestic"]
    modal = [mmdd_distribution(D[D.lender_country == c]).mmdd.iloc[0] for c in dom_adequate]
    dates = list(dict.fromkeys(list(glob.mmdd.head(10)) + modal))
    D_ad = D[D.lender_country.isin(dom_adequate)]
    out = []
    for md in dates:
        on = R.ending_mmdd == md
        rec = {"mmdd": md, "total": int(on.sum()), "share_all": float(on.mean()),
               **{f"from_{c}": int((on & (R.context == c)).sum()) for c in CONTEXTS}}
        base = float((D_ad.ending_mmdd == md).mean())
        rec["share_domestic_adequate"] = base
        um = R[on & (R.context == "one or both clubs unmapped")]
        side = um.lender_country.fillna(um.borrower_country)
        rec["unmapped_one_side_mapped"] = int(side.notna().sum())
        rec["unmapped_side_countries"] = [{"country": c, "n": int(v)} for c, v in side.value_counts().head(2).items()]
        countries = []
        for c in dom_adequate:
            g = D[D.lender_country == c]
            k = int((g.ending_mmdd == md).sum())
            if k:
                countries.append({"country": c, "n": k, "share_in_country": k / len(g),
                                  "lift": (k / len(g)) / base if base else None})
        countries.sort(key=lambda x: -x["share_in_country"])
        rec["countries"] = countries
        rec["especially_common_in"] = [x for x in countries if x["share_in_country"] >= 0.10 and (x["lift"] or 0) >= 2]
        out.append(rec)
    return out


def covid_check(R: pd.DataFrame, dom_adequate: list[str]) -> dict:
    yr = R.ending_date.dt.year
    D = R[R.context == "domestic"]
    by_year = []
    for c in dom_adequate:
        g = D[D.lender_country == c]
        m = mmdd_distribution(g[~g.covid]).mmdd.iloc[0]
        row = {"country": c, "usual_modal": m}
        for y in range(2018, 2023):
            h = g[g.ending_date.dt.year == y]
            row[str(y)] = float((h.ending_mmdd == m).mean()) if len(h) else None
            row[f"n{y}"] = len(h)
        h = g[g.covid]
        row["covid_top"] = h.ending_mmdd.value_counts().head(2).to_dict()
        by_year.append(row)
    glob_year = []
    for y in range(2016, 2024):
        h = R[yr == y]
        glob_year.append({"year": y, "N": len(h), **{md: float((h.ending_mmdd == md).mean())
                                                    for md in ("06-30", "12-31", "05-31", "07-31", "08-31")}})
    return {"by_country_year": by_year, "global_year": glob_year, "excluded": int(R.covid.sum())}


def rules(T: pd.DataFrame, R: pd.DataFrame, dom_adequate: list[str], C: pd.DataFrame) -> pd.DataFrame:
    """Candidate definitions of a calendar-normal return, over realised returns with any ending context."""
    ret = T[T.ending_is_return & T.ending_realised].copy()
    ret["context"] = context(ret)
    ret["covid"] = ret.ending_date.between(COVID_START, COVID_END)
    glob = mmdd_distribution(R)
    D = R[R.context == "domestic"]
    cross = mmdd_distribution(R[R.context == "cross-border"])

    def ctx_set(k: int | None, p: float | None):
        sets = {}
        for c in dom_adequate:
            d = mmdd_distribution(D[D.lender_country == c])
            sets[c] = set(d.mmdd.head(k if k else dates_needed(d, p)))
        cset = set(cross.mmdd.head(k if k else dates_needed(cross, p)))
        gset = set(glob.mmdd.head(k if k else dates_needed(glob, p)))
        return sets, cset, gset

    def ctx_rule(k=None, p=None):
        sets, cset, gset = ctx_set(k, p)
        out = []
        for ctxv, lc, md in zip(ret.context, ret.lender_country, ret.ending_mmdd):
            if ctxv == "domestic" and lc in sets:
                out.append(md in sets[lc])
            elif ctxv == "cross-border":
                out.append(md in cset)
            else:
                out.append(md in gset)
        return pd.Series(out, index=ret.index)

    defs = {
        "R0": ("Existing production rule: within 10 days of 30 June or 31 December", ret.return_boundary_days <= 10),
        "R1": ("Global top 3 dates (" + ", ".join(glob.mmdd.head(3)) + ")", ret.ending_mmdd.isin(glob.mmdd.head(3))),
        "R2": ("Global top 5 dates (" + ", ".join(glob.mmdd.head(5)) + ")", ret.ending_mmdd.isin(glob.mmdd.head(5))),
        "R3": ("Context top 3: own country's top 3 for domestic loans, the cross-border top 3 for cross-border loans, "
               "the global top 3 otherwise", ctx_rule(k=3)),
        "R4": ("Context dates covering 80% (same contexts as R3)", ctx_rule(p=.8)),
    }
    ret["calendar_known"] = ((ret.context == "domestic") & ret.lender_country.isin(dom_adequate)) | \
        (ret.context == "cross-border")
    Cset = set(C.loan_event_id)
    csub = C.set_index("loan_event_id").subgroup
    gap = ret.next_move_gap_days
    rows = []
    for key, (label, normal) in defs.items():
        normal = normal.astype(bool)
        for state, m in (("calendar-normal", normal), ("off-cycle", ~normal)):
            g = ret[m]
            rec = {"rule": key, "label": label, "state": state, "returns": len(g), "share": len(g) / len(ret),
                   "in_covid_window": int(g.covid.sum()),
                   "next_move_any": int(g.next_move_event_id.notna().sum())}
            for x in (1, 7, 21, 60):
                rec[f"next_move_within_{x}"] = int((gap[m] <= x).sum())
            for cat in "ABCD":
                rec[f"category_{cat}"] = int((g.sequence_category == cat).sum())
            for sgk in ("C1", "C2", "C3", "C4", "C5"):
                rec[f"subgroup_{sgk}"] = int(g.loan_event_id.map(csub).eq(sgk).sum())
            rec["production_early_termination"] = int((g.economic_ending == "early_termination").sum())
            k = g[g.calendar_known & ~g.covid]
            kg = k.next_move_gap_days
            rec.update({"known_ex_covid": len(k), "known_median_duration": float(k.loan_duration_days.median()),
                        "known_within_1": float((kg <= 1).mean()), "known_within_21": float((kg <= 21).mean()),
                        "known_january": float((k.ending_date.dt.month == 1).mean()),
                        "known_next_is_loan": float((k.next_move_kind == "loan").mean()),
                        "unknown_ex_covid": int((~g.calendar_known & ~g.covid).sum())})
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs["n_returns"] = len(ret)
    out.attrs["category_c_returns"] = int(ret.loan_event_id.isin(Cset).sum())
    return out


def heatmap(N: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    countries = N["domestic_adequate"]
    dates = [m["mmdd"] for m in N["major_dates"]]
    share_of = {(x["country"], m["mmdd"]): x["share_in_country"] for m in N["major_dates"] for x in m["countries"]}
    M = np.array([[share_of.get((c, d), 0) * 100 for d in dates] for c in countries])
    order = np.lexsort((-M.max(axis=1), M.argmax(axis=1)))
    M = M[order]
    cs = [countries[i] for i in order]
    fig, ax = plt.subplots(figsize=(2.6 + 0.62 * len(dates), 1.3 + 0.32 * len(cs)))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 1:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 55 else "#1a202c")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([f"{int(d[3:])} {MONTHS[int(d[:2]) - 1]}" for d in dates], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(cs)))
    ax.set_yticklabels([f"{dc(c)} ({N['domestic_n'][c]:,})" for c in cs])
    ax.set_title("Domestic loan endings by date\n(% of each country's realised domestic endings)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label("% of the country's domestic endings")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compute(u=None) -> tuple[dict, dict]:
    u = u or loan_scope.universe()
    T = episode_table(u, next_moves(u))
    R = T[T.ending_realised].copy()
    R["context"] = context(R)
    R["covid"] = R.ending_date.between(COVID_START, COVID_END)
    N: dict = {"loans": len(T), "realised": len(R), "scheduled": int((T.ending_observed & ~T.ending_realised).sum()),
               "no_ending": int((~T.ending_observed).sum()),
               "contexts": {c: int((R.context == c).sum()) for c in CONTEXTS}}
    assert N["realised"] + N["scheduled"] + N["no_ending"] == N["loans"]
    assert sum(N["contexts"].values()) == N["realised"]
    D = R[R.context == "domestic"]
    dn = D.lender_country.value_counts()
    adequate = list(dn[dn >= MIN_N].index)
    N["domestic_adequate"] = adequate
    N["domestic_n"] = {c: int(v) for c, v in dn.items()}
    N["min_adequate_n"] = int(dn[dn >= MIN_N].min())
    N["max_inadequate_n"] = int(dn[dn < MIN_N].max()) if (dn < MIN_N).any() else None
    N["inadequate_countries"] = [{"country": c, "n": int(v)} for c, v in dn[dn < MIN_N].items()]
    N["domestic_in_adequate"] = int(D.lender_country.isin(adequate).sum())
    modal = {c: mmdd_distribution(D[D.lender_country == c]).mmdd.iloc[0] for c in adequate}
    N["modal"] = modal
    N["global"] = profile(R)
    N["global_top10"] = mmdd_distribution(R).head(10).to_dict("records")
    N["attribution"] = attribution_test(R, modal)
    N["major_dates"] = major_dates(R, adequate)
    N["country_profiles"] = {c: profile(D[D.lender_country == c]) for c in adequate}

    Rx = R[~R.covid]
    N["covid_window"] = [str(COVID_START.date()), str(COVID_END.date())]
    N["covid"] = covid_check(R, adequate)
    N["global_ex_covid"] = profile(Rx)
    Dx = Rx[Rx.context == "domestic"]
    N["country_profiles_ex_covid"] = {c: profile(Dx[Dx.lender_country == c]) for c in adequate}
    N["modal_ex_covid"] = {c: mmdd_distribution(Dx[Dx.lender_country == c]).mmdd.iloc[0] for c in adequate}
    N["modal_changes"] = [c for c in adequate if N["modal"][c] != N["modal_ex_covid"][c]]
    N["top3_change_pp"] = {c: 100 * (N["country_profiles_ex_covid"][c]["top3_cover"] - N["country_profiles"][c]["top3_cover"])
                           for c in adequate}
    import importlib
    val = importlib.import_module("src.analysis.loan_timing_validation")
    _, S = val.compute(u)
    A = S[S.origin == "A"]
    Ax = A[~A.ending_date.between(COVID_START, COVID_END)]
    N["strict"] = {"all": {"N": len(A), "le1": float((A.next_move_gap_days <= 1).mean())},
                   "ex_covid": {"N": len(Ax), "le1": float((Ax.next_move_gap_days <= 1).mean())}}

    C = c_split(T)
    RU = rules(T, R, adequate, C)
    N["rules_n_returns"] = RU.attrs["n_returns"]
    N["rules"] = RU.to_dict("records")
    tables = {"country": pd.concat([country_table(R, "all realised endings"),
                                    country_table(Rx, "excluding the COVID window")], ignore_index=True),
              "rules": RU}
    return N, tables


def render(N: dict) -> str:
    W: list[str] = []
    w = W.append
    G, Gx = N["global"], N["global_ex_covid"]
    at = N["attribution"]
    ctx = N["contexts"]
    n = N["realised"]
    cs = lambda k, d: f"{f0(k)} ({share(k / d) if d else '–'})"
    w("# Loan-ending dates by country")
    w("")
    w("*Generated by `python -m src.analysis.loan_country_date_analysis`. Club countries come only from the snapshot's "
      "deterministic mapping. **Terminology:** the dates below are dates that are common among loan endings in our "
      "data. The repository and the snapshot contain no official transfer-window or registration calendar, so none of "
      "these dates is called a window boundary. That would need an external source.*".replace(
          "loan_country_date_analysis", "loan_country_dates"))
    w("")
    w("## Which country's calendar sets the ending date?")
    w("")
    w(f"Of the {f0(n)} realised loan endings: {cs(ctx['domestic'], n)} are domestic loans (both clubs mapped to the "
      f"same country), {cs(ctx['cross-border'], n)} are cross-border (both mapped, different countries), and "
      f"{cs(ctx['one or both clubs unmapped'], n)} have at least one club of unknown country.")
    w("")
    w(f"**Test.** For cross-border loans between two countries whose usual domestic ending dates differ, "
      f"{cs(at['on_borrower_modal'], at['n'])} end on the borrower country's usual date and "
      f"{cs(at['on_lender_modal'], at['n'])} on the lender country's. The largest country pairs:")
    w("")
    W += md_table(["Lender → borrower", "Loans", "On borrower's usual date", "On lender's usual date"],
                  [[f"{dc(p['lender_country'])} → {dc(p['borrower_country'])}", f0(p["n"]),
                    f"{f0(p['on_borrower_modal'])} ({p['borrower_modal']})", f"{f0(p['on_lender_modal'])} ({p['lender_modal']})"]
                   for p in at["pairs"]])
    w("")
    ed, eo, cb = at["england_domestic"], at["england_one_side"], at["cross_border"]
    w(f"So neither club's country alone sets the date. 31 May is a **domestic** English convention: "
      f"{share(ed['date1_share'])} of loans between two English clubs end on {ed['date1']}, but loans between England "
      f"and another country mostly end on {eo['date1']} ({share(eo['date1_share'])}), whichever direction they go. "
      f"Cross-border loans as a whole end on {cb['date1']} in {share(cb['date1_share'])} of cases. The clean unit for "
      "a country's calendar is therefore its **domestic** loans, and the rest of this document uses them, with "
      "cross-border loans as their own group.")
    w("")

    w("## Domestic loans, country by country")
    w("")
    w(f"Countries with at least {MIN_N} realised domestic loan endings: {len(N['domestic_adequate'])} countries, "
      f"{f0(N['domestic_in_adequate'])} endings. The largest country below the threshold has "
      f"{N['max_inadequate_n']} endings, so any threshold between {N['max_inadequate_n'] + 1} and {N['min_adequate_n']} "
      "selects the same countries. All countries, including domestic, borrower-country and lender-country "
      "attributions and the version without the COVID window, are in `loan_country_date_by_country.csv`.")
    w("")
    rows = []
    for c in N["domestic_adequate"]:
        p = N["country_profiles"][c]
        rows.append([dc(c), f0(p["N"]), f"{p['date1']} ({share(p['date1_share'])})",
                     ", ".join(f"{p[f'date{i}']} {share(p[f'date{i}_share'])}" for i in (2, 3) if p[f"date{i}"]),
                     share(p["top1_cover"]), share(p["top3_cover"]), share(p["top5_cover"]), p["dates_for_80"],
                     f"{p['top_month']} {share(p['top_month_share'])}"])
    rows.append(["*Cross-border loans*", f0(cb["N"]), f"{cb['date1']} ({share(cb['date1_share'])})",
                 ", ".join(f"{cb[f'date{i}']} {share(cb[f'date{i}_share'])}" for i in (2, 3)),
                 share(cb["top1_cover"]), share(cb["top3_cover"]), share(cb["top5_cover"]), cb["dates_for_80"],
                 f"{cb['top_month']} {share(cb['top_month_share'])}"])
    rows.append(["*All realised endings*", f0(G["N"]), f"{G['date1']} ({share(G['date1_share'])})",
                 ", ".join(f"{G[f'date{i}']} {share(G[f'date{i}_share'])}" for i in (2, 3)),
                 share(G["top1_cover"]), share(G["top3_cover"]), share(G["top5_cover"]), G["dates_for_80"],
                 f"{G['top_month']} {share(G['top_month_share'])}"])
    W += md_table(["Country (domestic loans)", "N", "Most common date", "Next two dates", "Top 1", "Top 3", "Top 5",
                   "Dates for 80%", "Most common month"], rows, right={1, 4, 5, 6, 7})
    w("")
    flat = [c for c in N["domestic_adequate"] if N["country_profiles"][c]["top1_cover"] < 0.3
            and N["country_profiles"][c]["top_month_share"] >= 0.5]
    if flat:
        w("In " + ", ".join(dc(c) for c in flat) + ", no single date dominates but one month does (see the last "
          "column): loans end on a date that varies from year to year within that month.")
        w("")
    w(f"![Domestic ending dates by country]({FIGURE})")
    w("")

    w("## Which countries are responsible for the major dates?")
    w("")
    w("Major dates are the 10 most common dates overall plus each country's most common domestic date. For each "
      "date: how many realised endings fall on it, how many of those are domestic, cross-border or unmapped, and in "
      "which countries it is **especially common**. That means at least 10% of the country's domestic endings fall "
      "on the date, and its share there is at least twice its share among all domestic endings in the countries "
      "above.")
    w("")
    rows = []
    for m in N["major_dates"]:
        esp = ", ".join(f"{dc(x['country'])} {share(x['share_in_country'])}" for x in m["especially_common_in"]) or "–"
        side = ", ".join(f"{dc(x['country'])} {f0(x['n'])}" for x in m["unmapped_side_countries"]) or "–"
        rows.append([md_words(m["mmdd"]), f0(m["total"]), share(m["share_all"]), f0(m["from_domestic"]),
                     f0(m["from_cross-border"]), f0(m["from_one or both clubs unmapped"]), side, esp])
    W += md_table(["Date", "Realised endings", "Share of all", "Domestic", "Cross-border", "Unmapped club",
                   "… the mapped club's country, where one is mapped", 
                   "Especially common in (share of that country's domestic endings)"], rows, right={1, 2, 3, 4, 5})
    w("")
    w("Statements the data support, by the rule above:")
    w("")
    for m in N["major_dates"]:
        if m["especially_common_in"]:
            w(f"- **{md_words(m['mmdd'])}** is especially common in "
              + ", ".join(f"{dc(x['country'])} ({share(x['share_in_country'])} of its domestic loan endings)"
                          for x in m["especially_common_in"]) + ".")
    for m in N["major_dates"]:
        if m["especially_common_in"] or m["mmdd"] == "06-30":
            continue
        um, side = m["from_one or both clubs unmapped"], m["unmapped_side_countries"]
        if um >= m["total"] / 2 and m["unmapped_one_side_mapped"] >= um / 2 and side:
            w(f"- **{md_words(m['mmdd'])}** has no domestic home among the countries above. {f0(um)} of its "
              f"{f0(m['total'])} endings involve a club of unknown country; where the other club is mapped "
              f"({f0(m['unmapped_one_side_mapped'])}), it is most often from "
              + " or ".join(f"{dc(x['country'])} ({f0(x['n'])})" for x in side) + ".")
    rest = [md_words(m["mmdd"]) for m in N["major_dates"] if not m["especially_common_in"] and m["mmdd"] != "06-30"
            and not (m["from_one or both clubs unmapped"] >= m["total"] / 2
                     and m["unmapped_one_side_mapped"] >= m["from_one or both clubs unmapped"] / 2)]
    w("- **30 June** is the most common date nearly everywhere, so no single country stands out for it.")
    if rest:
        w(f"- No single country accounts for {', '.join(rest)}.")
    w("")

    w("## Does the COVID period distort the results?")
    w("")
    cv = N["covid"]
    gy = {r["year"]: r for r in cv["global_year"]}
    w("**What the data show.** In loans ending in 2020, 30 June's share drops and late-summer dates appear:")
    w("")
    W += md_table(["Ending year", "Endings", "30 June", "31 December", "31 May", "31 July", "31 August"],
                  [[y, f0(r["N"]), share(r["06-30"]), share(r["12-31"]), share(r["05-31"]), share(r["07-31"]),
                    share(r["08-31"])] for y, r in gy.items()])
    w("")
    rows = []
    for r in cv["by_country_year"]:
        if r["2020"] is not None and r["2019"] is not None and abs(r["2020"] - r["2019"]) >= 0.2:
            rows.append([dc(r["country"]), r["usual_modal"]] + [f"{share(r[str(y)])} ({r[f'n{y}']})" for y in range(2018, 2023)]
                        + [", ".join(f"{k} ({v})" for k, v in r["covid_top"].items())])
    if rows:
        w("Countries whose usual domestic date loses at least 20 percentage points in 2020 (share of that year's "
          "domestic endings on the usual date, with N):")
        w("")
        W += md_table(["Country", "Usual date", "2018", "2019", "2020", "2021", "2022", "Most common dates in the window"],
                      rows, right=set())
        w("")
    w(f"**Exclusion window used for the second version: realised endings dated {N['covid_window'][0]} to "
      f"{N['covid_window'][1]}** ({f0(cv['excluded'])} endings). Why these dates: the window must cover the shifted "
      "dates visible above. Those are 2019/20 seasons ended early (spring 2020) or late (July and August 2020), and "
      "calendar-year leagues' 2020 seasons ending in early 2021 (for example 28 February 2021 in Brazil). Endings from "
      "March 2021 on look normal again.")
    w("")
    rows = [["Endings", f0(G["N"]), f0(Gx["N"])],
            ["Top 1 / top 3 / top 5 cover", f"{share(G['top1_cover'])} / {share(G['top3_cover'])} / {share(G['top5_cover'])}",
             f"{share(Gx['top1_cover'])} / {share(Gx['top3_cover'])} / {share(Gx['top5_cover'])}"],
            ["Dates for 80% / 90%", f"{G['dates_for_80']} / {G['dates_for_90']}", f"{Gx['dates_for_80']} / {Gx['dates_for_90']}"],
            ["Strict A → B → A → B conversions within 1 day",
             f"{share(N['strict']['all']['le1'])} (N = {f0(N['strict']['all']['N'])})",
             f"{share(N['strict']['ex_covid']['le1'])} (N = {f0(N['strict']['ex_covid']['N'])})"],
            ["Countries whose most common domestic date changes", "", str(len(N["modal_changes"]))]]
    W += md_table(["Statistic", "All realised", "Excluding the COVID window"], rows)
    w("")
    big = {c: v for c, v in N["top3_change_pp"].items() if abs(v) >= 2}
    w("**Answer.** Overall, excluding the window changes little (table above). The country rows change by less than "
      "2 percentage points in their top-3 share, except "
      + (", ".join(f"{dc(c)} ({v:+.1f} pp)" for c, v in sorted(big.items(), key=lambda kv: -abs(kv[1]))) or "none")
      + ". COVID matters locally, for 2020 endings in the affected countries, rather than for the overall picture. "
      "It does matter for any rule that labels individual endings: within the window, a 31 July or 31 August return "
      "is the shifted season end, not an unusual ending.")
    w("")

    w("## Can country timing separate normal endings from edge cases?")
    w("")
    w(f"Candidate definitions of a **calendar-normal return**, applied to the {f0(N['rules_n_returns'])} realised loan "
      "returns. Off-cycle does not mean early termination; it means the return falls outside the dates the "
      "definition treats as normal.")
    w("")
    RU = pd.DataFrame(N["rules"])
    rows = []
    for k in RU.rule.unique():
        on = RU[(RU.rule == k) & (RU.state == "calendar-normal")].iloc[0]
        off = RU[(RU.rule == k) & (RU.state == "off-cycle")].iloc[0]
        rows.append([f"{k}: {on.label}", cs(on.returns, N["rules_n_returns"]), cs(off.returns, N["rules_n_returns"]),
                     f0(off.in_covid_window),
                     f"{share(on.next_move_within_1 / on.returns)} / {share(off.next_move_within_1 / off.returns)}",
                     f"{share(on.next_move_within_21 / on.returns)} / {share(off.next_move_within_21 / off.returns)}",
                     f"{f0(off.category_C)} ({f0(off.subgroup_C2)} C2, {f0(off.subgroup_C3 + off.subgroup_C4)} C3–C4)",
                     f0(off.production_early_termination)])
    W += md_table(["Definition", "Calendar-normal", "Off-cycle", "Off-cycle in the COVID window",
                   "Next move within 1 day: normal / off", "Within 21 days: normal / off", "Off-cycle in Category C",
                   "Off-cycle labelled `early_termination` now"], rows)
    w("")
    r1 = RU[(RU.rule == "R1") & (RU.state == "off-cycle")].iloc[0]
    r3 = RU[(RU.rule == "R3") & (RU.state == "off-cycle")].iloc[0]
    r4 = RU[(RU.rule == "R4") & (RU.state == "off-cycle")].iloc[0]
    w(f"Moving from global dates to context-specific ones moves normal endings out of the off-cycle set: "
      f"{f0(r1.returns)} off-cycle returns with the global top 3 (R1), {f0(r3.returns)} with each context's own top 3 "
      f"(R3), {f0(r4.returns)} with the dates covering 80% of each context (R4). **Over all returns, though, off-cycle "
      "and normal returns are followed by another move about equally fast** (the 1-day and 21-day columns). The "
      "reason is that most off-cycle returns involve a club of unknown country, whose calendar we cannot know.")
    w("")
    w("**Where the calendar is known.** These are domestic loans in the countries above, plus cross-border loans, "
      "with the COVID window set aside. Here off-cycle returns form a distinct group:")
    w("")
    rows = []
    for k in RU.rule.unique():
        on = RU[(RU.rule == k) & (RU.state == "calendar-normal")].iloc[0]
        off = RU[(RU.rule == k) & (RU.state == "off-cycle")].iloc[0]
        rows.append([k, f"{f0(on.known_ex_covid)} / {f0(off.known_ex_covid)}",
                     f"{on.known_median_duration:.0f} / {off.known_median_duration:.0f}",
                     f"{share(on.known_within_1)} / {share(off.known_within_1)}",
                     f"{share(on.known_within_21)} / {share(off.known_within_21)}",
                     f"{share(on.known_january)} / {share(off.known_january)}",
                     f"{share(on.known_next_is_loan)} / {share(off.known_next_is_loan)}",
                     f0(off.unknown_ex_covid)])
    W += md_table(["Definition", "Returns: normal / off-cycle", "Median loan length (days)", "Next move within 1 day",
                   "Within 21 days", "Return in January", "Next move is a new loan",
                   "Off-cycle with an unknown calendar (not in these columns)"], rows)
    w("")
    known = RU[RU.state == "off-cycle"].set_index("rule")
    norm = RU[RU.state == "calendar-normal"].set_index("rule")
    shorter = int((known.known_median_duration < norm.known_median_duration).sum())
    quicker = int((known.known_within_21 > norm.known_within_21).sum())
    w(f"Off-cycle loans are shorter in {shorter} of {len(known)} definitions and followed by another move within 21 "
      f"days more often in {quicker} of {len(known)}. Many of them end in January, mid-season, and the next move is "
      "more often a new loan. That is the profile of a loan cut short. The existing rule (R0) dilutes it: "
      f"{share(known.loc['R0', 'known_january'])} of its off-cycle returns are in January, against "
      f"{share(known.drop(index='R0').known_january.min())}–{share(known.drop(index='R0').known_january.max())} "
      "for the other definitions, because it also counts normal endings on other countries' dates (such as English "
      "31 May endings) as off-cycle.")
    w("")
    w("**So a calendar-based rule without a day cutoff looks feasible, but only where the calendar is known.** Call "
      "a return calendar-normal if it falls on one of its context's usual dates: the country's own dates for domestic "
      "loans, the cross-border list for cross-border loans. Set aside returns inside the COVID window. Treat loans "
      "with an unmapped club as unclassifiable rather than off-cycle. The off-cycle returns that remain are the "
      "candidate edge cases. How many usual dates each context should get (top 3, or the dates covering 80%) is "
      "still a choice; the table shows what it does to the counts.")
    w("")
    w("Per-rule counts, including Category C subgroups, are in `loan_country_date_rules.csv`.")
    w("")
    return "\n".join(W) + "\n"


def _json(x):
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if np.isnan(x) else float(x)
    return x


def main() -> dict:
    N, tb = compute()
    tb["country"].to_csv(COUNTRY_CSV, index=False)
    tb["rules"].to_csv(RULES_CSV, index=False)
    (OUT / "figures").mkdir(exist_ok=True)
    heatmap(N, OUT / FIGURE)
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    REPORT_MD.write_text(render(N))
    print(f"wrote {REPORT_MD.name}, {COUNTRY_CSV.name}, {RULES_CSV.name}, {FIGURE}")
    print("domestic adequate:", {c: N["modal"][c] for c in N["domestic_adequate"]})
    print("COVID modal changes:", N["modal_changes"])
    return N


if __name__ == "__main__":
    main()
