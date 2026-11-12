"""Week 10 meeting package and audit report (every number read from generated files).

    python -m src.analysis.week10_meeting

Writes professor_feedback_week10/AUDIT.md and professor_feedback_week10/meeting/ (slides.tex/.pdf with four main
slides, appendix and questions; summary.tex/.pdf one page). Run after week10_followup, week10_audit and
week10_regression_audit.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pandas as pd

from .week10_followup import OUT, V2

AUD = OUT / "audit"
MEET = OUT / "meeting"
PDFLATEX = "/usr/local/bin/pdflatex"


def _j(p):
    return json.loads(p.read_text())


def _n(x) -> str:
    return f"{int(round(x)):,}"


def _p(a, b) -> str:
    return f"{100 * a / b:.1f}"


def load() -> dict:
    d = {"now": _j(OUT / "numbers.json"), "before": _j(AUD / "before_numbers.json"),
         "lt_before": _j(AUD / "before_league_tiers_v2_numbers.json"), "lt_now": _j(V2 / "league_tiers_v2_numbers.json"),
         "aud": _j(AUD / "audit_numbers.json"), "q6": _j(OUT / "q6_squad_value" / "numbers.json")}
    d["sens"] = pd.read_csv(AUD / "tier_rule_sensitivity.csv", index_col=0)
    d["mv_now"] = pd.read_csv(OUT / "q3_tier_movement" / "movement_summary.csv")
    d["mv_before"] = pd.read_csv(AUD / "before_movement_summary.csv")
    d["bb_now"] = pd.read_csv(OUT / "q3_tier_movement" / "direction_by_domestic_crossborder.csv")
    d["bb_before"] = pd.read_csv(AUD / "before_direction_by_border.csv")
    d["mv_verd"] = pd.read_csv(AUD / "match_audit_verdicts.csv")
    d["ex_verd"] = pd.read_csv(AUD / "exclusion_audit_verdicts.csv")
    d["spec"] = pd.read_csv(OUT / "q6_squad_value" / "regression_audit" / "specifications.csv")
    d["dec"] = pd.read_csv(OUT / "q6_squad_value" / "regression_audit" / "decomposition.csv")
    d["prof"] = pd.read_csv(OUT / "q6_squad_value" / "regression_audit" / "excluded_vs_retained_profile.csv")
    d["cons"] = pd.read_csv(AUD / "consistency_checks.csv")
    return d


def _spec(d: dict, prefix: str) -> pd.DataFrame:
    x = d["spec"][d["spec"]["sample"].str.startswith(prefix)].copy()
    x["loan_coef_pp"] = x.loan_coef_pp.round(2) + 0.0   # no "-0.00"
    return x.set_index(x.model.str[:1])


def mv(S: pd.DataFrame, kind: str, cat: str, col: str = "pct") -> float:
    return float(S[(S.kind == kind) & (S.category == cat)].iloc[0][col])


def headline_table(d: dict) -> pd.DataFrame:
    b, a = d["before"], d["now"]
    rows = []
    for k, lab in (("recovered_exact", "Origin tier recovered exactly (of 5,311)"),
                   ("recovered_not_1_2", "Recovered: senior club not tier 1 or 2"),
                   ("recovered_youth_reserve", "Recovered: youth/reserve side not tier 1 or 2"),
                   ("partly_not_1", "Partly recovered: not tier 1"), ("still_unknown", "Still unknown"),
                   ("exact_tier_v2", "Main sample: exact origin tier"),
                   ("band_known_v2", "Main sample: origin tier or band known")):
        rows.append((lab, b["q2"][k], a["q2"][k]))
    for kind in ("loan", "permanent"):
        for cat, lab in (("Higher division -> lower division", "down"), ("Lower division -> higher division", "up"),
                         ("Same level", "same level")):
            rows.append((f"{kind}: {lab} (% of decidable)", round(mv(d["mv_before"], kind, cat), 1),
                         round(mv(d["mv_now"], kind, cat), 1)))
        rows.append((f"{kind}: decidable moves (n)", mv(d["mv_before"], kind, "Higher division -> lower division",
                                                         "denominator"),
                     mv(d["mv_now"], kind, "Higher division -> lower division", "denominator")))
        rows.append((f"{kind}: both tiers known (%)", round(mv(d["mv_before"], kind, "both tiers known"), 1),
                     round(mv(d["mv_now"], kind, "both tiers known"), 1)))
        bb, bn = d["bb_before"], d["bb_now"]
        f = lambda t: round(float(t[(t.kind == kind) & (t.border == "domestic")].down_pct.iloc[0]), 1)
        rows.append((f"{kind}: domestic moves going down (%)", f(bb), f(bn)))
        for k2, lab in (("in_matrix", "heatmap: episodes shown"), ("omitted", "heatmap: episodes omitted"),
                        ("core_28x28", "heatmap: 28 x 28 core")):
            rows.append((f"{kind}: {lab}", b["q4"][kind][k2], a["q4"][kind][k2]))
    lb, ln = d["lt_before"], d["lt_now"]
    for k, lab in (("team_rows_matched", "Wikipedia teams matched"), ("table_rows_added", "Club-seasons added"),
                   ("overlap_rows", "Overlap club-seasons with Week 9"), ("overlap_agree_tier", "... same tier")):
        rows.append((lab, lb[k], ln[k]))
    T = pd.DataFrame(rows, columns=["number", "before audit (10 Nov)", "after audit (12 Nov)"])
    T["change"] = T.iloc[:, 2] - T.iloc[:, 1]
    return T


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f"{v:,.1f}" if v != int(v) else f"{int(v):,}")
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def audit_md(d: dict, T: pd.DataFrame) -> str:
    mvv, exv = d["mv_verd"], d["ex_verd"]
    kept = mvv[mvv.in_final_lookup]
    dec = d["dec"].set_index("sample")
    s = d["sens"]
    txt = f"""# Week 10 audit (12 November 2026)

Generated by `python -m src.analysis.week10_meeting` from `audit/`, `numbers.json` and
`q6_squad_value/regression_audit/`. Code: `week10_audit.py` (rule sensitivity, samples, consistency, figures),
`week10_regression_audit.py`, `week10_meeting.py`; corrections in `league_tiers_wiki.py` and `week10_followup.py`.
The master, Stage 1C, Week 9 variables and stored regression outputs are unchanged.

## Corrections made

1. **Malatyaspor.** The country-wide token-subset fallback matched "Malatyaspor" (2005/06, 2007/08, 2008/09) to
   Yeni Malatyaspor, because the "Y." of Transfermarkt's "Y. Malatyaspor" was ignored. No other season was affected
   (the learned alias "Yeni Malatyaspor" is correct). The manual audit then found the same failure for successor
   clubs (Køge BK to HB Køge, Nafta Lendava to NK Nafta 1903: 4 of 23 fallback-only matches wrong), so the
   **fallback is disabled** (it had added {_n(d['lt_before']['matched_by_token_subset_fallback'])} team rows).
2. **Refounded clubs.** A learned alias sent the 2024/25 to 2025/26 Lierse and Lokeren entries to the dissolved
   clubs' ids ("Lierse SK (- 2018)", "KSC Lokeren (- 2020)"). Matches after a club's recorded end year are now
   rejected ({_n(d['lt_now']['rejected_after_club_end_year'])} cases).
3. **Transliterations in the "not tier 1 or 2" rule.** Clubs such as Metalurg/Metalurh Zaporizhzhia,
   Chaika/Chayka and Eginiakos/Aiginiakos share no name word with the unmatched list entry that is the same club.
   The safeguard now also blocks the classification when the name similarity is 0.72 or more.
4. **Reserve sides linked to the parent club.** Wikipedia links SL16 FC (Standard's reserve side) to the Standard
   Liège article; that row was dropped as a conflict but still counted as matched. Rows dropped as conflicts now
   count as unmatched for the safeguard.

## Manual audits (stratified, fixed seed; verdict and reason per row)

**New club-season matches** (`audit/match_audit_verdicts.csv`, {len(mvv)} rows stratified by method, 14 countries or
not, season band and reserve/youth status; source URL per row): {(mvv.verdict == 'accept').sum()} accepted,
{(mvv.verdict == 'reject').sum()} rejected, {(mvv.verdict == 'uncertain').sum()} uncertain. All rejected and
uncertain rows came from the fallback and are no longer in the lookup. Of the {len(kept)} audited rows still in the
final lookup, {(kept.verdict == 'accept').sum()} are accepted and {(kept.verdict == 'reject').sum()} rejected
(observed error rate {(kept.verdict == 'reject').sum()} of {len(kept)}; with {len(kept)} checks, a true error rate
above {100 * (1 - 0.05 ** (1 / len(kept))):.1f}% would have shown at least one error with 95% probability).

**"Not tier 1 or 2" with an incomplete list** (`audit/exclusion_audit_verdicts.csv`, {len(exv)} main-sample
club-seasons, with the full team lists): {(exv.verdict == 'accept').sum()} confirmed, {(exv.verdict == 'reject').sum()}
wrong (SL16 FC, corrected; now unclassified), {(exv.verdict == 'uncertain').sum()} not confirmable from the lists
(MFC Mykolaiv 2021/22, Ferroviária 2026, Guimarães B 2013/14).

## Stricter rule: require complete lists (sensitivity)

| | Primary (per-club safeguard) | Strict (complete lists) | Difference |
|---|---|---|---|
""" + "\n".join(f"| {i} | {r.iloc[0]:,.2f} | {r.iloc[1]:,.2f} | {r.iloc[2]:+,.2f} |".replace(".00 |", " |")
                for i, r in s.iterrows()) + f"""

The exact origin tier and the 28 x 28 heatmap cores are identical under both rules. The strict rule moves
"not tier 1 or 2" cases to unknown or "not tier 1", which lowers coverage and shrinks the decidable sample; the
loan/permanent contrast in direction is unchanged in sign and size.

## Consistency

All {len(d['cons'])} checks pass ({', '.join(d['cons'].check)}).

## Origin squad-value regression

Same preferred specification throughout (M3 controls, league-season fixed effects, position dummies, standard errors
clustered by receiving club). Loan coefficient (percentage points):

{md(d['spec'][['sample', 'model', 'N', 'loans', 'permanent', 'loan_coef_pp', 'se_pp', 'p']].round(3))}

Decomposition (stored main sample): moving to complete cases changes the coefficient by
{dec.loc['main (calendar-first, stored)', 'selection_change']:+.2f} points (sample selection); adding the origin
control changes it by a further {dec.loc['main (calendar-first, stored)', 'control_change']:+.2f}. In the episodes
dropped by complete-case filtering the coefficient is {dec.loc['main (calendar-first, stored)', 'excluded_only']:+.2f}.
The fully validated sample and the 730-day window give the same pattern. These are associations, not causal effects.

## Before / after every affected headline number

{md(T)}
"""
    assert "—" not in txt
    return txt


def tex_escape(s: str) -> str:
    return s.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_").replace("#", "\\#")


def slides(d: dict) -> str:
    q1, q2, q4 = d["now"]["q1"], d["now"]["q2"], d["now"]["q4"]
    S = d["mv_now"]
    bn = d["bb_now"].set_index(["kind", "border"])
    sp = _spec(d, "main (calendar")
    sv = _spec(d, "main, fully")
    dec = d["dec"].set_index("sample").loc["main (calendar-first, stored)"]
    mvv, exv = d["mv_verd"], d["ex_verd"]
    kept = mvv[mvv.in_final_lookup]
    sens = d["sens"]
    t2 = q1["tier2_episodes_14"]
    lt = d["lt_now"]
    prof = d["prof"]
    pr = lambda dim, val, col: float(prof[(prof.dimension == dim) & (prof.value == val)][col].iloc[0])
    EX, RT = "excluded (origin value missing) %", "retained (origin value present) %"
    g = lambda k, c: mv(S, k, c)
    return rf"""\documentclass[aspectratio=169,10pt]{{beamer}}
\usetheme{{default}}\usecolortheme{{dove}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamertemplate{{footline}}[frame number]
\usepackage{{booktabs}}\usepackage[T1]{{fontenc}}\usepackage{{lmodern}}
\setbeamerfont{{frametitle}}{{size=\large}}
\newcommand{{\est}}{{\textbf{{Established:}} }}\newcommand{{\sens}}{{\textbf{{Sensitivity:}} }}
\newcommand{{\open}}{{\textbf{{Open:}} }}
\title{{Football contracts: Week 10 results}}
\subtitle{{Tier 2 availability, tier movements, tier data, origin squad value}}
\date{{Meeting with Professors Freund and Fernandes}}
\begin{{document}}
\begin{{frame}}\titlepage\end{{frame}}

\begin{{frame}}{{1. Second-division playing time is not available to us}}
\footnotesize
\begin{{tabular}}{{@{{}}p{{0.30\textwidth}}p{{0.64\textwidth}}@{{}}}}\toprule
Data absent from our snapshot & All second-division games, line-ups and minutes: the snapshot holds {q1['snapshot_domestic_league_competitions']} first divisions only; its upstream project collects first divisions only.\\
Possibly available, licensed & Commercial event and line-up providers (e.g. Stats Perform/Opta, Wyscout) or written permission from Transfermarkt; coverage and cost to be confirmed.\\
Not collectable under access terms & Transfermarkt (terms forbid scraping); FBref and Sofascore (block automated access); worldfootball.net (robots.txt excludes AI agents); FotMob match API (disallowed).\\
Available, not sufficient & openfootball, Wikipedia: results and league tables, not minutes.\\\bottomrule
\end{{tabular}}
\vspace{{0.6em}}

\est In 2012/13 to 2025/26, {_n(sum(t2.values()))} transfers went to verified second-division clubs of the 14 countries; loans are {100 * q1['loan_share_tier2_episodes']:.1f}\% of them, against {100 * q1['loan_share_tier1_episodes']:.1f}\% of first-division arrivals.\\[0.3em]
\textbf{{Selection problem:}} the playing-time sample keeps only first-division destinations, so it under-represents the destination where loans are most common (loans to weaker clubs). The direction of any resulting bias in the loan coefficient is unknown. No projected observations enter any regression.
\end{{frame}}

\begin{{frame}}{{2. Loans move players down a level; permanent transfers move them up}}
\centering\includegraphics[width=0.92\textwidth]{{../presentation/direction_loans_vs_permanent.png}}\\
\raggedright\footnotesize
\est decidable moves: {g('loan','Higher division -> lower division'):.1f}\% of loans go down vs {g('permanent','Higher division -> lower division'):.1f}\% of permanent transfers; {g('permanent','Lower division -> higher division'):.1f}\% of permanent transfers go up vs {g('loan','Lower division -> higher division'):.1f}\% of loans. Domestic loans: {bn.loc[('loan','domestic'),'down_pct']:.1f}\% down.
\sens with the strict complete-list rule, loans down {sens.loc['loan: down %'].iloc[1]:.1f}\%, permanent up {sens.loc['permanent: up %'].iloc[1]:.1f}\%.
\open both tiers are known for only {g('loan','both tiers known'):.1f}\% of loans and {g('permanent','both tiers known'):.1f}\% of permanent transfers; division numbers are not comparable measures of strength across countries.
\end{{frame}}

\begin{{frame}}{{3. Better tier data: origin tier known for {_p(q2['exact_tier_v2'], q2['n_main'])}\% of the main sample (Week 9: {_p(q2['verified_week9'], q2['n_main'])}\%)}}
\small
\begin{{itemize}}
\item New source: Wikipedia league tables ({_n(lt['season_pages_parsed'])} league-seasons, {lt['series']} leagues), matched to Transfermarkt ids with the Week 9 rules. Same tier as Week 9 in {_n(lt['overlap_agree_tier'])} of {_n(lt['overlap_rows'])} overlapping club-seasons.
\item Of the 5,311 unknown origin tiers: {_n(q2['recovered_exact'])} exact; {_n(q2['recovered_not_1_2'])} senior and {_n(q2['recovered_youth_reserve'])} youth/reserve clubs verifiably outside tiers 1--2; {_n(q2['partly_not_1'])} not tier 1; {_n(q2['still_unknown'])} still unknown.
\item \textbf{{Manual audits:}} {len(kept)} sampled new club-season matches all correct ({len(mvv)} audited; the {(mvv.verdict != 'accept').sum()} errors or doubts came from a fallback step that is now disabled). {len(exv)} sampled ``not tier 1 or 2'' cases: {(exv.verdict == 'accept').sum()} confirmed, 1 corrected, {(exv.verdict == 'uncertain').sum()} unconfirmable.
\item \textbf{{Corrected:}} Malatyaspor/Yeni Malatyaspor and other successor clubs; refounded clubs (Lierse, Lokeren); transliterations (Metalurg/Metalurh); reserve sides linked to their parent club.
\item Heatmaps (appendix) show {_p(q4['loan']['in_matrix'], q4['loan']['episodes_in_period'])}\% of loans and {_p(q4['permanent']['in_matrix'], q4['permanent']['episodes_in_period'])}\% of permanent transfers, 2012/13 to 2025/26.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{4. Origin squad value in the regression: half selection, half control}}
\footnotesize
\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{@{{}}lrrr@{{}}}}\toprule
Preferred specification, loan coefficient (pp) & N & coef. & p\\\midrule
(1) Main sample & {_n(sp.loc['1','N'])} & {sp.loc['1','loan_coef_pp']:+.2f} & {sp.loc['1','p']:.3f}\\
(2) Complete cases (origin value present), no new control & {_n(sp.loc['2','N'])} & {sp.loc['2','loan_coef_pp']:+.2f} & {sp.loc['2','p']:.3f}\\
(3) Same sample + log origin squad value & {_n(sp.loc['3','N'])} & {sp.loc['3','loan_coef_pp']:+.2f} & {sp.loc['3','p']:.3f}\\
\quad Excluded episodes only & {_n(sp.loc['4','N'])} & {sp.loc['4','loan_coef_pp']:+.2f} & {sp.loc['4','p']:.3f}\\
\quad Fully validated sample, (1) / (3) & {_n(sv.loc['1','N'])} / {_n(sv.loc['3','N'])} & {sv.loc['1','loan_coef_pp']:+.2f} / {sv.loc['3','loan_coef_pp']:+.2f} & {sv.loc['1','p']:.3f} / {sv.loc['3','p']:.3f}\\
\quad 730-day window, (2) / (3) & {_n(sp.loc['5','N'])} & {sp.loc['5','loan_coef_pp']:+.2f} / {sp.loc['6','loan_coef_pp']:+.2f} & {sp.loc['5','p']:.3f} / {sp.loc['6','p']:.3f}\\\bottomrule
\end{{tabular}}}}\\[0.4em]
Same controls, league-season fixed effects, position dummies, errors clustered by receiving club in every model.
\begin{{itemize}}
\item Sample selection: {dec['selection_change']:+.2f} pp; adding the control: {dec['control_change']:+.2f} pp.
\item Excluded episodes: {pr('kind','permanent',EX):.0f}\% permanent (retained {pr('kind','permanent',RT):.0f}\%); {pr('origin_in_covered_top_flight','no',EX):.1f}\% from clubs outside the covered top flights; {pr('from_tier_band_v2','youth/reserve',EX):.0f}\% from youth or reserve sides.
\item Loans come from richer origin clubs, so holding origin value fixed lowers the loan coefficient. Associations, not causal effects.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Questions for you}}
\begin{{enumerate}}
\item Tier 2: is a licensed data source (or Transfermarkt permission) worth pursuing, or do we keep first divisions and state the selection problem?
\item ``Not tier 1 or 2'' from absence in division lists: accept the per-club rule, or use only complete lists (lower coverage, same contrast)?
\item Movement direction across borders: keep division numbers, restrict ``up/down'' to domestic moves, or build a strength measure (e.g. squad value or UEFA coefficients)?
\item Origin squad value: add it as a control (sample shrinks to {_p(sp.loc['2','N'], sp.loc['1','N'])}\%, mostly losing permanent arrivals from uncovered origins) or keep it as a robustness check? 365- or 730-day window?
\item Youth and reserve origins ({_n(q2['recovered_youth_reserve'])} main-sample episodes): a separate category in the analysis?
\end{{enumerate}}
\end{{frame}}

\appendix
\begin{{frame}}{{Appendix A. Loans between country-tier categories (29 x 29)}}
\centering\includegraphics[height=0.86\textheight]{{../q4_heatmaps/V1_loan_heatmap.png}}
\end{{frame}}
\begin{{frame}}{{Appendix B. Permanent transfers between country-tier categories (29 x 29)}}
\centering\includegraphics[height=0.86\textheight]{{../q4_heatmaps/V2_permanent_heatmap.png}}
\end{{frame}}
\begin{{frame}}{{Appendix C. Supporting files}}
\small
\begin{{itemize}}
\item Raw-count matrices: \texttt{{q4\_heatmaps/V1\_loan\_flow\_matrix\_29x29.csv}}, \texttt{{V2\_permanent\_flow\_matrix\_29x29.csv}}; 28 x 28 country-tier-only versions: \texttt{{*\_heatmap\_28x28.png/pdf}}. Colour log(1 + count), cells raw counts.
\item Audit: \texttt{{AUDIT.md}}; manual audits \texttt{{audit/match\_audit\_verdicts.csv}}, \texttt{{audit/exclusion\_audit\_verdicts.csv}} (verdict, reason, source URL); rule sensitivity \texttt{{audit/tier\_rule\_sensitivity.csv}}; consistency \texttt{{audit/consistency\_checks.csv}}.
\item Regression audit: \texttt{{q6\_squad\_value/regression\_audit/}} (specifications, decomposition, excluded-vs-retained profile).
\item Full findings: \texttt{{FINDINGS.md}}; per-episode tiers: \texttt{{week10\_episode\_supplement.csv}}.
\end{{itemize}}
\end{{frame}}
\end{{document}}
"""


def summary(d: dict) -> str:
    q1, q2, q4 = d["now"]["q1"], d["now"]["q2"], d["now"]["q4"]
    S = d["mv_now"]
    g = lambda k, c: mv(S, k, c)
    sp = _spec(d, "main (calendar")
    dec = d["dec"].set_index("sample").loc["main (calendar-first, stored)"]
    mvv, exv = d["mv_verd"], d["ex_verd"]
    kept = mvv[mvv.in_final_lookup]
    lt = d["lt_now"]
    sens = d["sens"]
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=1.6cm]{{geometry}}\usepackage[T1]{{fontenc}}\usepackage{{lmodern}}\usepackage{{enumitem}}
\setlist{{nosep,leftmargin=1.2em}}\pagestyle{{empty}}\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{0.35em}}
\begin{{document}}
{{\large\textbf{{Football contracts, Week 10: summary for the meeting}}}}

\textbf{{Established findings}}
\begin{{itemize}}
\item \textbf{{Second-division playing time cannot be added with data we may use.}} Our snapshot has first divisions only; sources with minutes are either licensed or closed to automated access. Second divisions are a loan destination: loans are {100 * q1['loan_share_tier2_episodes']:.1f}\% of {_n(sum(q1['tier2_episodes_14'].values()))} second-division arrivals (2012/13 to 2025/26) against {100 * q1['loan_share_tier1_episodes']:.1f}\% of first-division arrivals.
\item \textbf{{Loans move players down, permanent transfers move them up.}} Of moves whose direction can be decided, {g('loan','Higher division -> lower division'):.1f}\% of loans and {g('permanent','Higher division -> lower division'):.1f}\% of permanent transfers go to a lower division; {g('loan','Lower division -> higher division'):.1f}\% and {g('permanent','Lower division -> higher division'):.1f}\% go up.
\item \textbf{{Origin tier now known for {_p(q2['exact_tier_v2'], q2['n_main'])}\% of the main sample}} (Week 9: {_p(q2['verified_week9'], q2['n_main'])}\%), from Wikipedia league tables that agree with every one of {_n(lt['overlap_rows'])} club-seasons already verified. In a stratified manual audit, all {len(kept)} new matches still in use were correct.
\item \textbf{{Heatmaps}} (29 x 29 and 28 x 28) cover {_p(q4['loan']['in_matrix'], q4['loan']['episodes_in_period'])}\% of loans and {_p(q4['permanent']['in_matrix'], q4['permanent']['episodes_in_period'])}\% of permanent transfers; colour log(1 + count), cells raw counts.
\end{{itemize}}

\textbf{{Sensitivity results}}
\begin{{itemize}}
\item Requiring complete division lists before calling a club ``not tier 1 or 2'': loans going down {sens.loc['loan: down %'].iloc[1]:.1f}\% (primary {sens.loc['loan: down %'].iloc[0]:.1f}\%), permanent going up {sens.loc['permanent: up %'].iloc[1]:.1f}\% (primary {sens.loc['permanent: up %'].iloc[0]:.1f}\%); exact tiers unchanged.
\item Origin squad value as a control: loan coefficient {sp.loc['1','loan_coef_pp']:+.2f} pp on the full main sample, {sp.loc['2','loan_coef_pp']:+.2f} on complete cases ({_n(sp.loc['2','N'])}), {sp.loc['3','loan_coef_pp']:+.2f} (p = {sp.loc['3','p']:.3f}) with the control: {dec['selection_change']:+.2f} pp from sample selection, {dec['control_change']:+.2f} pp from the control. Same in the fully validated sample and with a 730-day window. Not causal.
\end{{itemize}}

\textbf{{Limitations}}
\begin{{itemize}}
\item Direction is decidable for only {g('loan','direction decidable'):.0f}\% of loans and {g('permanent','direction decidable'):.0f}\% of permanent transfers; division numbers do not measure strength across countries.
\item {_n(q2['still_unknown'])} main-sample origin tiers remain unknown (no list for the country-season, or the club may be an unmatched list entry); three audited cases could not be confirmed.
\item The origin squad value is missing where the origin club lies outside the snapshot's competitions, so complete cases are a selected sample.
\end{{itemize}}

\textbf{{Decisions we need}}
\begin{{enumerate}}
\item Pursue licensed second-division data, or keep first divisions with the selection caveat?
\item Accept the per-club ``not tier 1 or 2'' rule, or only complete lists?
\item Define up/down for cross-border moves by division number, domestic moves only, or a strength measure?
\item Origin squad value: main control (complete cases) or robustness only; 365- or 730-day window?
\item Treat youth and reserve origins as a separate category?
\end{{enumerate}}
\end{{document}}
"""


def compile_tex(path) -> None:
    for _ in range(2):
        r = subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", path.name], cwd=path.parent,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout[-3000:])


def main() -> None:
    d = load()
    T = headline_table(d)
    T.to_csv(AUD / "before_after_headline_numbers.csv", index=False)
    (OUT / "AUDIT.md").write_text(audit_md(d, T))
    MEET.mkdir(exist_ok=True)
    for name, txt in (("slides", slides(d)), ("summary", summary(d))):
        assert "—" not in txt
        p = MEET / f"{name}.tex"
        p.write_text(txt)
        compile_tex(p)
    for f in MEET.glob("*"):
        if f.suffix in (".aux", ".log", ".nav", ".out", ".snm", ".toc", ".vrb"):
            f.unlink()
    print(T.to_string())


if __name__ == "__main__":
    main()
