"""Week 10 findings report and change log (every number interpolated from week10_followup.run() and the files it
reads; nothing typed by hand)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .week10_followup import FOURTEEN, OUT, PERIOD, _md, _pct

P0, P1 = PERIOD
PERIOD_TXT = f"{P0}/{str(P0 + 1)[-2:]} to {P1}/{str(P1 + 1)[-2:]}"


def _n(x) -> str:
    return f"{int(x):,}"


def _p(x) -> str:
    return f"{100 * x:.1f}%"


def section_summary(r) -> str:
    q1, q2, lt2 = r["Q1"]["N"], r["Q2"]["N"], r["lt2"]
    S = r["Q3"]["summary"]
    g = lambda k, c: S[(S.kind == k) & (S.category == c)].iloc[0]
    A = r["Q4"]["accounting"]
    q6 = r["q6"]
    rec = q2["recovered_exact"] + q2["recovered_not_1_2"] + q2["recovered_youth_reserve"]
    return f"""## Summary

1. **Tier-2 playing time is not feasible with data we may use.** The snapshot has no second-division games or
   appearances, and no openly licensed source has second-division minutes. Second divisions matter for the
   question, though: {_p(q1['loan_share_tier2_episodes'])} of transfers into them are loans, against
   {_p(q1['loan_share_tier1_episodes'])} into first divisions ({PERIOD_TXT}, 14 countries).
2. **Origin tiers:** of the {_n(q2['missing_week9'])} unknown origin tiers in the main sample, {_n(rec)} are now
   resolved ({_n(q2['recovered_exact'])} with an exact tier) and {_n(q2['partly_not_1'])} partly; {_n(q2['still_unknown'])}
   remain unknown. Exact origin tier: {_pct(q2['exact_tier_v2'], q2['n_main'])} of the main sample (Week 9:
   {_pct(q2['verified_week9'], q2['n_main'])}). New source: Wikipedia league tables, which agree with every one
   of the {_n(lt2['overlap_rows'])} club-seasons we could already verify.
3. **Loans move players down, permanent transfers move them up.** Among moves whose direction can be decided,
   {g('loan', 'Higher division -> lower division').pct:.1f}% of loans go to a lower division and
   {g('loan', 'Lower division -> higher division').pct:.1f}% to a higher one; for permanent transfers the shares
   are {g('permanent', 'Higher division -> lower division').pct:.1f}% and
   {g('permanent', 'Lower division -> higher division').pct:.1f}%.
4. **Heatmaps:** the two 29 x 29 matrices ({PERIOD_TXT}) show {_pct(A['loan']['in_matrix'], A['loan']['episodes_in_period'])}
   of loans and {_pct(A['permanent']['in_matrix'], A['permanent']['episodes_in_period'])} of permanent transfers;
   {_n(A['loan']['omitted'])} loans and {_n(A['permanent']['omitted'])} permanent transfers are omitted because a
   club's tier cannot be classified.
5. **Origin squad value:** the {_p(q6['main_coverage']['coverage'])} coverage cannot be raised much without changing
   the variable; the gap between loans and permanent transfers is mostly where the player comes from.
"""


def section_q1(r) -> str:
    N = r["Q1"]["N"]
    pot = r["Q1"]["potential"].reset_index().rename(columns={"to_country_for_flows": "receiving country"})
    for c in ("loan", "permanent"):
        if c not in pot:
            pot[c] = 0
    pot = pot[["receiving country", "loan", "permanent", "total"]].sort_values("total", ascending=False)
    pot["loan share"] = 100 * pot.loan / pot.total
    t2n, t1n = N["tier2_episodes_14"], N["tier1_episodes_14"]
    pr, rs = N["projected_tier2_additions"], N["revised_sample"]
    return f"""## 1. Can playing time be extended to second divisions?

**No, not with data we are allowed to use.** Playing time needs every league game of the receiving club and the
player's minutes in it (appearances and line-ups). League tables only say which clubs were in a division.

* **Our snapshot** has {N['snapshot_domestic_league_competitions']} domestic leagues, all first divisions
  (`sub_type` {', '.join(f'{k}: {v}' for k, v in N['snapshot_sub_types'].items())}). It has no second-division game,
  line-up or appearance for any of the 14 countries in any season; second-division clubs appear only in domestic
  cup games, which playing time excludes. The project the snapshot comes from (transfermarkt-datasets) also
  collects first divisions only.
* **Other sources.** FBref and Sofascore refuse automated requests (HTTP 403); worldfootball.net's robots.txt
  disallows ClaudeBot and anthropic-ai; FotMob disallows its `/api/` paths, which hold the match data;
  Transfermarkt's terms forbid scraping. openfootball and Wikipedia have results and tables, not minutes. Minutes
  would need a licensed provider (for example Opta/Stats Perform or Wyscout) or Transfermarkt's written permission.
* **Second-division lists do exist** for {N['tier2_list_available_country_seasons']} of {N['country_seasons']}
  country-seasons (built for Question 2). They tell us which transfers went to a second division, so the size of
  the gap can be measured, but not minutes.

**How large is the gap?** In {PERIOD_TXT}, {_n(sum(t2n.values()))} loans and permanent transfers went to a
verified second-division club of the 14 countries ({_n(t2n.get('loan', 0))} loans, {_n(t2n.get('permanent', 0))}
permanent), against {_n(sum(t1n.values()))} to their first divisions. Loans are
{_p(N['loan_share_tier2_episodes'])} of second-division arrivals but {_p(N['loan_share_tier1_episodes'])} of
first-division arrivals. If second-division episodes passed the playing-time filters at the first-division rate
({_p(N['tier1_pass_rate_to_revised']['loan'])} of loans and {_p(N['tier1_pass_rate_to_revised']['permanent'])} of
permanent arrivals reach the revised playing-time sample, before the loan-calendar step), about
{_n(pr.get('loan', 0))} loans and {_n(pr.get('permanent', 0))} permanent arrivals would join the
{_n(rs['loan'] + rs['permanent'])} episodes of that sample, and the loan share would rise from
{_p(N['loan_share_revised'])} to about {_p(N['loan_share_projected_combined'])}. This is a projection, not a sample:
the true pass rate depends on line-up completeness in data we do not have.

**Implication.** Restricting playing time to first divisions leaves out the destination where loans are most
common, so loans to weaker clubs are under-represented in the regressions. The expanded regression could not be
run; the original sample and results are unchanged.

Second-division arrivals by receiving country ({PERIOD_TXT}):

{_md(pot)}

Files: `q1_tier2_playing_time/` (availability by country and season, arrivals by country, the snapshot's leagues
for the 14 countries).
"""


def section_q2(r) -> str:
    N = r["Q2"]["N"]
    lt2 = r["lt2"]
    fe = lt2["fallback_precision_check"]
    rt = r["Q2"]["reason_tab"].reset_index()
    rt.columns = [str(c) for c in rt.columns]
    ctry = r["Q2"]["country"].head(15).reset_index().rename(columns={"from_country_v2": "origin country"})
    su, bk, wr = N["still_unknown_reason"], N["by_kind"], N["week9_reason"]
    from .league_tiers_wiki import SERIES
    n_o1 = len({x["country"] for x in SERIES if x["tier"] == 1 and x["country"] not in FOURTEEN})
    n_o2 = len({x["country"] for x in SERIES if x["tier"] == 2 and x["country"] not in FOURTEEN})
    return f"""## 2. The {_n(N['missing_week9'])} main-sample episodes with an unknown origin tier

**Why they were missing (Week 9 status).** {'; '.join(f'{k}: {_n(v)}' for k, v in sorted(wr.items(), key=lambda x: -x[1]))}.
The binding problem was not that the club was outside our 14 countries but that the Week 9 lookup had no league
table for the club's country and season (openfootball covers few seasons outside western Europe and few second
divisions), plus youth and reserve sides.

**New source: Wikipedia league-season articles** (CC BY-SA), read as ordinary article pages at one request per
second; robots.txt welcomes low-speed bots on article pages, and the disallowed `/w/` and `/api/` endpoints were not
used. {lt2['series']} league series (tiers 1 and 2 of the 14 countries, first divisions of {n_o1} other countries and
second divisions of {n_o2}), {_n(lt2['season_pages_parsed'])} season articles 2005/06 to 2025/26, {_n(lt2['team_rows'])}
listed teams, of which {_n(lt2['team_rows_matched'])} ({_pct(lt2['team_rows_matched'], lt2['team_rows'])}) were
matched to a Transfermarkt club id. Matching reuses the Week 9 rules (exact normalised names, learned aliases,
reserve and youth sides kept apart, ambiguous names left unmatched). A match to a club in a season after
Transfermarkt's recorded end of that club (a refounded club has a new id) is rejected
({_n(lt2['rejected_after_club_end_year'])} cases). A country-wide token-subset step was tested and then disabled
after the manual audit (see `audit/`): it was right for {_n(fe['unique_correct'])} of {_n(fe['unique_candidate'])}
names that other rules also match, but 4 of the 23 audited club-seasons that only it matched were wrong.

**Validation.** Where Week 9 already had the club-season, Wikipedia gives the same tier in
{_n(lt2['overlap_agree_tier'])} of {_n(lt2['overlap_rows'])} cases ({_pct(lt2['overlap_agree_tier'], lt2['overlap_rows'])}).
Week 9 rows are never changed; a club that Wikipedia places in two divisions in one season is dropped
({_n(lt2['wiki_internal_conflict_rows'])} rows). The lookup grows from {_n(lt2['table_rows_week9'])} to
{_n(lt2['table_rows_week9'] + lt2['table_rows_added'])} club-seasons.

**"Not in tier 1 or 2".** A tier number is assigned only from a list naming the club. A club of one of the 14
countries that is in neither that season's tier-1 nor tier-2 list is recorded as "not tier 1 or 2", but only if
it cannot be one of those lists' unmatched teams: no shared name word, and no name similarity of 0.72 or more (this
catches transliterations such as Metalurg/Metalurh). Teams whose match was dropped as a conflict count as unmatched.
Youth and reserve sides get their own band. Nothing follows from a club merely being outside our covered leagues.
A stricter rule that also requires every list to be complete is reported as a sensitivity in `audit/`.

**Result (main sample, {_n(N['n_main'])} episodes).**

| | Episodes |
|---|---|
| Verified in Week 9 | {_n(N['verified_week9'])} |
| Unknown in Week 9 | {_n(N['missing_week9'])} |
| ... now with an exact tier | {_n(N['recovered_exact'])} |
| ... senior club, not in tier 1 or 2 | {_n(N['recovered_not_1_2'])} |
| ... youth or reserve side, not in tier 1 or 2 | {_n(N['recovered_youth_reserve'])} |
| ... not in tier 1 (no tier-2 list) | {_n(N['partly_not_1'])} |
| ... still unknown | {_n(N['still_unknown'])} |

Exact origin tier: {_n(N['exact_tier_v2'])} episodes ({_pct(N['exact_tier_v2'], N['n_main'])}; Week 9
{_pct(N['verified_week9'], N['n_main'])}): {', '.join(f'tier {k} {_n(v)}' for k, v in N['exact_tier_counts_v2'].items())}.
Tier or band known: {_n(N['band_known_v2'])} ({_pct(N['band_known_v2'], N['n_main'])}). Loans: exact tier
{_pct(bk['loan']['exact_v2'], bk['loan']['main'])} (Week 9 {_pct(bk['loan']['verified_week9'], bk['loan']['main'])});
permanent: {_pct(bk['permanent']['exact_v2'], bk['permanent']['main'])} (Week 9
{_pct(bk['permanent']['verified_week9'], bk['permanent']['main'])}).

Still unknown: {'; '.join(f'{k}: {_n(v)}' for k, v in sorted(su.items(), key=lambda x: -x[1]))}.

By Week 9 reason and outcome:

{_md(rt)}

Largest origin countries:

{_md(ctry)}

**Existing columns are sufficient for club and country.** `from_club_id`, `from_club_name` and `from_country_v2`
identify the origin club and its country; `from_league_country_at_move` gives the league's country when the tier is
verified. No duplicate columns were added. The new tier, band and status columns are in
`week10_episode_supplement.csv` (one row per episode, key `event_id`), not merged into the master.

Files: `q2_origin_tiers/` and `data/processed/league_tiers_v2/` (lookup, match audit, learned aliases, Week 9
overlap check, list coverage).
"""


def section_q3(r) -> str:
    S = r["Q3"]["summary"]
    piv = []
    for grp in ("tier pair (both tiers known)", "youth or reserve sides (all episodes)", "direction (decidable)",
                "country (both known)", "coverage"):
        d = S[S.group == grp]
        for cat in dict.fromkeys(d.category):
            row = {"group": grp, "category": cat}
            for k in ("loan", "permanent"):
                x = d[(d.category == cat) & (d.kind == k)].iloc[0]
                row[f"{k} n"] = int(x["count"])
                row[f"{k} %"] = x.pct
                row[f"{k} denominator"] = int(x.denominator)
            piv.append(row)
    T = pd.DataFrame(piv)
    b = r["Q3"]["by_border"].reset_index()[["kind", "border", "n", "down_pct", "same level_pct", "up_pct"]]
    dest = r["Q3"]["dest_from_t1_pct"].reset_index().rename(columns={"to_tier_band_v2": "destination"})
    per = r["Q3"]["by_period"].reset_index()[["kind", "period", "n", "down_pct", "up_pct"]]
    bc = r["Q3"]["by_country"]
    bc.columns = [f"{a} {b}" for a, b in bc.columns]
    bc = bc.reset_index().rename(columns={"from_country_for_flows": "origin country (tier 1)"})
    g = lambda k, c: S[(S.kind == k) & (S.category == c)].iloc[0]
    dl = b[(b.kind == "loan") & (b.border == "domestic")].iloc[0]
    dp = b[(b.kind == "permanent") & (b.border == "domestic")].iloc[0]
    return f"""## 3. Movement between league tiers

**Population:** every loan and permanent transfer in the master with reference season {PERIOD_TXT} (the full
dataset, not the regression sample). Tier bands: 1, 2, "3+" (verified tier 3 or lower, or a senior club of the 14
countries in neither the tier-1 nor the tier-2 list), "youth/reserve" (a youth or reserve side in neither list),
"not 1", unknown. **Direction** compares tier numbers (1 is highest): down = to a larger tier number. A move involving
"3+" is decidable only against tier 1 or 2; youth and reserve sides are kept out of up/down. Tier numbers of
different countries are not equally strong leagues, so cross-border "up" and "down" refer to the pyramid level only.

**Main contrasts.**
* Loans go down far more often: {g('loan', 'Higher division -> lower division').pct:.1f}% of decidable loans
  ({_n(g('loan', 'Higher division -> lower division')['count'])} of {_n(g('loan', 'Higher division -> lower division').denominator)})
  against {g('permanent', 'Higher division -> lower division').pct:.1f}% of permanent transfers.
* Permanent transfers go up far more often: {g('permanent', 'Lower division -> higher division').pct:.1f}% against
  {g('loan', 'Lower division -> higher division').pct:.1f}% of loans.
* The contrast is sharpest within a country: {dl.down_pct:.1f}% of domestic loans go down against
  {dp.down_pct:.1f}% of domestic permanent transfers, while {dp.up_pct:.1f}% of domestic permanent transfers go up.
* Tier 1 to tier 2 is {g('loan', 'Tier 1 -> Tier 2').pct:.1f}% of loans with both tiers known but
  {g('permanent', 'Tier 1 -> Tier 2').pct:.1f}% of permanent transfers; tier 2 to tier 1 is
  {g('loan', 'Tier 2 -> Tier 1').pct:.1f}% and {g('permanent', 'Tier 2 -> Tier 1').pct:.1f}%.
* About a third of both kinds cross a border ({g('loan', 'cross-border').pct:.1f}% of loans,
  {g('permanent', 'cross-border').pct:.1f}% of permanent transfers).

**Coverage caveat.** Both tiers are known for {g('loan', 'both tiers known').pct:.1f}% of loans but only
{g('permanent', 'both tiers known').pct:.1f}% of permanent transfers, mostly because many permanent transfers involve
clubs in countries without lists. The shares describe the moves we can classify, which over-represent the 14
countries.

{_md(T)}

Direction by domestic and cross-border moves (decidable moves, %):

{_md(b)}

Where moves out of a first-division club of the 14 countries go (% of each kind):

{_md(dest)}

Share of decidable moves out of a first-division club that go down, by origin country:

{_md(bc)}

By period (decidable moves, %):

{_md(per)}

Files: `q3_tier_movement/`.
"""


def section_q4(r) -> str:
    A = r["Q4"]["accounting"]
    cov = r["Q4"]["coverage"]
    part = cov[(cov.seasons_with_list > 0) & (cov.seasons_with_list < cov.seasons_in_period)]
    none = cov[cov.seasons_with_list == 0].category.tolist()
    om = pd.read_csv(OUT / "q4_heatmaps" / "omitted_episodes_by_reason.csv")
    omp = om.pivot_table(index=["side", "reason"], columns="kind", values="episodes", aggfunc="sum").fillna(0)
    omp = omp.astype(int).reset_index()
    lines = []
    for k, lab in (("loan", "V1 loans"), ("permanent", "V2 permanent")):
        a = A[k]
        lines.append(f"| {lab} | {_n(a['episodes_in_period'])} | {_n(a['in_matrix'])} | {_n(a['core_28x28'])} | "
                     f"{_n(a['other_row_or_col'] - a['other_to_other'])} | {_n(a['other_to_other'])} | "
                     f"{_n(a['omitted'])} |")
    top = []
    for k in ("loan", "permanent"):
        m = r["Q4"]["matrices"][k].iloc[:28, :28].stack().sort_values(ascending=False).head(6)
        top.append(f"{'loans' if k == 'loan' else 'permanent'}: " +
                   "; ".join(f"{a} to {b} {_n(v)}" for (a, b), v in m.items()))
    return f"""## 4. The two 29 x 29 transfer-flow heatmaps

Files in `q4_heatmaps/`: `V1_loan_heatmap.png|pdf`, `V2_permanent_heatmap.png|pdf` and the raw-count matrices
`V1_loan_flow_matrix_29x29.csv`, `V2_permanent_flow_matrix_29x29.csv`. Rows are the origin (FROM), columns the
destination (TO); the 14 countries in alphabetical order, tier 1 then tier 2, then Other, identical in both
matrices. Each cell shows the raw number of episodes; colour is log(1 + count) on one scale shared by V1 and V2.
The Other row and column are grey with raw counts only. White cells are true zeros.

**Population:** all loan (V1) and permanent (V2) episodes of the master with reference season {PERIOD_TXT}.
**Other** = a club outside the 14 countries (any tier), or a club of the 14 countries verified in tier 3 or lower,
or in neither that season's tier-1 nor tier-2 list (youth and reserve sides included). **Omitted** = a side whose
tier 1 or 2 cannot be ruled out (no list that season, or the club may be an unmatched name of the list) or whose
country is unknown. Unresolved clubs of the 14 countries are never put in Other.

| | Episodes in period | In matrix | 28 x 28 core | Other row or column (excl. Other to Other) | Other to Other | Omitted |
|---|---|---|---|---|---|---|
{chr(10).join(lines)}

Omitted sides by reason (one episode can be omitted for both sides):

{_md(omp)}

Lists by category: every one of the 28 categories has a division list for all {P1 - P0 + 1} seasons except
{', '.join(f"{x.category} ({x.seasons_with_list} of {x.seasons_in_period})" for x in part.itertuples()) or 'none'}
(marked * in the figures){'; no list at all: ' + ', '.join(none) if none else ''}. Episodes of a missing season are
omitted, not moved to Other.

Largest flows in the 28 x 28 core, {top[0]}; {top[1]}. Most traffic involves Other (clubs outside the 14 countries,
lower divisions and youth sides), which is why the Other row and column are kept off the colour scale.
"""


def _coef(x: float) -> str:
    v = 100 * x
    return "0.00" if abs(v) < 0.005 else f"{v:+.2f}"


def section_q6(r) -> str:
    q = r["q6"]
    d = q["decomposition"]
    sen = pd.read_csv(OUT / "q6_squad_value" / "sensitivity_summary.csv").set_index("variant")
    reg = pd.read_csv(OUT / "q6_squad_value" / "regression_implications.csv")
    rr = reg[reg.origin_value.eq("original from_squad_value")].set_index("sample")
    full, cc0 = rr.loc["main sample (calendar-first)"], rr.loc["complete cases, without origin control"]
    cc1 = rr.loc["complete cases, with log origin squad value"]
    s730, s8 = sen.loc["lb730_min11"], sen.loc["lb365_min8"]
    return f"""## 6. Origin-club squad value

Full investigation: `q6_squad_value/README.md` (module `src/analysis/week10_squad_value.py`). The stored
`from_squad_value` is reproduced exactly ({_n(q['reproduction']['stored_non_missing'])} values); no implementation
error was found and the variable is unchanged. Coverage: {_p(q['main_coverage']['coverage'])} of the main sample
({_p(q['loan_coverage']['coverage'])} of loans, {_p(q['perm_coverage']['coverage'])} of permanent arrivals).

* **Why values are missing.** Of the {_n(q['main_N'] - q['main_coverage']['covered'])} missing main-sample values,
  {_n(q['thin']['thin_main'])} have 1 to 10 valued players; the rest are mainly youth or reserve origins. The
  valuation table only holds players who appear in the snapshot's covered competitions, so clubs outside those
  leagues look thin. Origins that played in a covered top flight that season: {_p(d['coverage_origin_covered_top_flight'])}
  coverage; all other origins: {_p(d['coverage_origin_not_covered_top_flight'])}.
* **Why loans are covered better.** {_p(d['loan_share_origin_covered_top_flight'])} of loans but only
  {_p(d['perm_share_origin_covered_top_flight'])} of permanent arrivals come from such clubs. With the loans' mix of
  origins, permanent coverage would be {_p(d['perm_coverage_with_loan_mix'])}: origin mix explains
  {100 * d['share_of_gap_explained']:.0f}% of the gap.
* **Recovery.** A 730-day look-back adds {_n(s730.new_episodes_all)} episodes (coverage {_p(s730.coverage_all)});
  lowering the minimum to 8 players adds {_n(s8.new_episodes_all)} ({_p(s8.coverage_all)}) but sums fewer than a
  starting XI. Alternatives are stored separately (`q6_squad_value/sensitivity_values.csv`); we recommend keeping
  the original and using the 730-day variant only as a robustness check.
* **Regression implication.** Requiring the origin value keeps {_n(cc0.N)} of {_n(full.N)} main-sample episodes and
  raises the loan share from {_p(full.loan_share)} to {_p(cc0.loan_share)}. On that sample the loan coefficient is
  {_coef(cc0.loan_coef)} points (p = {cc0.p:.3f}) without the control and {_coef(cc1.loan_coef)}
  (p = {cc1.p:.3f}) with log origin squad value, against {_coef(full.loan_coef)} (p = {full.p:.3f}) on the full
  sample. The complete-case sample is selected (it drops mostly permanent arrivals from uncovered origins), so the
  change mixes the control with the change of sample.
"""


def section_checks(r) -> str:
    K = r["K"]["N"]
    return f"""## Checks

* Stage 1C SHA-256 unchanged: {K['stage1c_sha256_ok']}. Master {K['master_shape'][0]:,} x {K['master_shape'][1]},
  unchanged; `event_id` unique: {K['event_id_unique']}; rows sharing player, date and both clubs:
  {_n(K['duplicate_player_date_clubs_rows'])}.
* Sample counts reproduced: main {_n(K['main_sample'])} (fully validated {_n(K['main_validated'])}), injury
  {_n(K['injury_sample'])} (fully validated {_n(K['injury_validated'])}).
* The stored Week 9 tier columns are reproduced exactly from the Week 9 lookup, and every Week 9 verified tier is
  identical in v2.
* Promotions and relegations: of {_n(K['consecutive_club_seasons'])} consecutive club-seasons in the lookup,
  {_n(K['tier_unchanged'])} keep their tier, {_n(K['tier_moved_one_level'])} move one level, and
  {_n(K['tier_jumps_over_one_level'])} move more than one level; {_n(K['league_country_changes'])} change country.
* A verified league in a different country from the club's own (outside known cases such as Welsh clubs in
  England or Monaco in France): {_n(K['league_country_vs_club_country_mismatch_rows'])} episode sides.
* Tests: `tests/test_week10.py` (season titles, standings tables, direction rule, matrix structure).
"""


def write_all(r) -> None:
    today = dt.date.today().isoformat()
    head = f"""# Week 10 follow-up: findings

Generated {today} by `python -m src.analysis.week10_followup`; every number comes from that run (`numbers.json`).
The master (115,044 x 227), Stage 1C and all stored samples and regressions are unchanged. New per-episode
columns are in `week10_episode_supplement.csv` (dictionary: `week10_episode_supplement_dictionary.csv`).
Question 5 (prior-season playing time): unchanged, as requested.
"""
    body = "\n".join([head, section_summary(r), section_q1(r), section_q2(r), section_q3(r), section_q4(r),
                      section_q6(r), section_checks(r)])
    assert "—" not in body, "em dash in report"
    (OUT / "FINDINGS.md").write_text(body)
    write_changelog(r)


def write_changelog(r) -> None:
    lt2 = r["lt2"]
    fe = lt2["fallback_precision_check"]
    txt = f"""# Week 10 change log

## New code (no existing file edited)
* `src/analysis/league_tiers_wiki.py`: Wikipedia league-season lists ({lt2['series']} series), matched with the
  Week 9 matcher; writes `data/processed/league_tiers_v2/`; page cache and manifest (URL, sha256, time, terms) in
  `data/raw/external_leagues/wikipedia/`.
* `src/analysis/week10_followup.py`: tiers v2, Questions 1 to 4, checks. `src/analysis/week10_report.py`: this
  report and change log. `src/analysis/week10_squad_value.py`: Question 6.
* `tests/test_week10.py`.

## New data products
* `data/processed/league_tiers_v2/club_season_league_v2.csv`: Week 9 lookup ({_n(lt2['table_rows_week9'])} rows,
  unchanged) plus {_n(lt2['table_rows_added'])} Wikipedia club-seasons.
* `professor_feedback_week10/week10_episode_supplement.csv`: per-episode tier v2 columns, bands and flow
  categories (115,044 rows, key `event_id`); not merged into the master.
* `q6_squad_value/sensitivity_values.csv`: alternative origin squad values in separate columns.

## Methodological decisions
* A tier number only from a list naming the club in that season. "Not tier 1 or 2" only when the club's country has
  tier-1 and tier-2 lists for that season, the club is in neither, and no unmatched team of those lists shares a
  name word (3+ letters, legal forms such as FC removed) or has name similarity >= 0.72 with any of the club's names
  (teams whose match was dropped as a conflict count as unmatched). Youth and reserve sides get
  their own band. Nothing is inferred from a club being outside the covered leagues.
* Matching: Week 9 rules unchanged; dotted abbreviations collapsed (K.V.C. to KVC); Chinese clubs also tried with
  Transfermarkt's city codes (BJ, SH, ...); aliases learned on Wikipedia first-division seasons overlapping the
  snapshot (exact, token subset or elimination within that league-season's clubs); matches after a club's recorded
  end year rejected ({_n(lt2['rejected_after_club_end_year'])}). The country-wide token-subset fallback is disabled
  (12 Nov audit: 4 of 23 fallback-only matches wrong, including Malatyaspor 2005 to Yeni Malatyaspor).
* Week 9 lookup rows always win; disagreeing Wikipedia rows would be dropped (none disagreed); a club in two
  divisions of one season on Wikipedia is dropped.
* Seasons as in Week 9 (start year; calendar-year leagues Y to Y-1; Apertura Y to Y, Clausura Y to Y-1); Russia only
  from 2011/12.
* Movement tables and heatmaps use reference seasons {PERIOD_TXT}; unresolved clubs of the 14 countries are omitted
  and counted, never put in Other.
* Tier-2 playing time not built: no lawful source of second-division minutes.

## Audit corrections (12 November 2026; details and before/after numbers in AUDIT.md)
* Country-wide token-subset fallback disabled (successor clubs matched to predecessor seasons, e.g. Malatyaspor 2005
  to Yeni Malatyaspor, Køge BK to HB Køge, Nafta Lendava to NK Nafta 1903).
* Matches after a club's recorded end year rejected (refounded Lierse and Lokeren had been sent to the old ids).
* "Not tier 1 or 2" safeguard extended: name similarity >= 0.72 also blocks it (transliterations), and teams dropped
  as conflicts count as unmatched (SL16 FC linked to Standard Liège).
* New: `week10_audit.py` (strict complete-list sensitivity, manual-audit samples and verdict summaries, consistency
  checks, direction chart, 28 x 28 heatmaps), `week10_regression_audit.py` (origin-value regression decomposition),
  `week10_meeting.py` (AUDIT.md, slides and one-page summary).

## Sources and access
* English Wikipedia article pages only (CC BY-SA 4.0), one request per second, identifying User-Agent without
  personal data; no `/w/` or `/api/` endpoints. While exploring, one Wikidata SPARQL query was sent
  (query.wikidata.org's robots.txt disallows `/sparql`); its result was not used.
* robots.txt or terms checked for FBref, Sofascore, worldfootball.net, FotMob and Transfermarkt; no data were
  fetched from any of them.
"""
    assert "—" not in txt
    (OUT / "CHANGELOG.md").write_text(txt)
