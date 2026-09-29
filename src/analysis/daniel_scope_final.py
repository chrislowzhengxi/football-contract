"""Final dataset-scope answers for Professor Freund.

    python -m src.analysis.provenance_audit --upstream   # once; needs the public R2 bucket
    python -m src.analysis.daniel_scope_final            # offline

Deterministic. Reads the frozen Stage 1C table, the upstream DuckDB snapshot,
the provenance-audit outputs and the Stage 2 extraction cache, all read-only.
Writes to data/outputs/rebuild/:

  daniel_dataset_scope_final.md            the professor-facing answers
  daniel_dataset_scope_final_numbers.json  every number the document states
  loan_scope_cube.csv                      loans by lender country x borrower country x season
  token_workload_by_scope_size.csv         LOW / BASE / HIGH extraction tokens for a range of N
  daniel_dataset_scope_crosscheck.csv      earlier documents re-checked against these numbers

Every statistic in the Markdown is interpolated from a computed value.
"""
from __future__ import annotations

import json
import math
import re

import duckdb
import pandas as pd

from ..config import DEFAULT_DATABASE
from . import loan_scope
from .loan_episodes import CANONICAL_CSV, EXPECTED_ROWS, build_universe, club_geography, load_canonical
from .provenance_audit import club_country_counts, player_source_file, source_file_capture, upstream_tables
from .stage1_scope_report import (OUT, date_buckets, europe_end_dates, latest_completed_season,
                                  loan_outcomes, sensitivities, sha256, waterfall)

SQUAD_CSV = OUT / "dataset_provenance_squad_coverage.csv"
LEAGUE_CSV = OUT / "dataset_provenance_league_coverage.csv"
FACTS_JSON = OUT / "dataset_provenance_upstream_facts.json"
ANSWERS_MD = OUT / "daniel_stage1_scope_answers.md"
PROVENANCE_MD = OUT / "dataset_provenance_and_coverage.md"
FINAL_MD = OUT / "daniel_dataset_scope_final.md"

EXAMPLE = dict(countries=["Italy", "England", "Spain"], seasons=["2021/22", "2022/23", "2023/24"])
SHARES = (0.1, 0.5, 0.9)
SCOPE_SIZES = (100, 500, 1_000, 5_000, 10_000, 29_699)
UNMAPPED = "(unmapped)"
LABEL_DISPLAY = {"End of loan + fee": "`End of loan` + a fee (a return row)",
                 "- (no fee shown)": "`-` (no fee shown; no return row)",
                 "free transfer": "`free transfer` (no return row)",
                 "? (undisclosed)": "`?` (undisclosed fee; no return row)",
                 "explicit fee amount": "an explicit fee amount (no return row)",
                 "loan transfer": "`loan transfer` (a new loan; no return row)"}


def f0(n) -> str:
    return f"{round(n):,}"


def pc(n, d, digits=1) -> str:
    return f"{100 * n / d:.{digits}f}%" if d else "n/a"


def frac(n, d, digits=1) -> str:
    return f"{f0(n)} / {f0(d)} = {pc(n, d, digits)}"


def eur(x) -> str:
    return f"€{x / 1e6:,.2f}m" if x >= 1e6 else f"€{x / 1e3:,.0f}k"


def tok(n) -> str:
    return f"{n / 1e9:,.2f}B" if n >= 1e9 else (f"{n / 1e6:,.1f}M" if n >= 1e6 else f"{n / 1e3:,.0f}k")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def season(y: int | str) -> str:
    y = int(y)
    return f"{y}/{str(y + 1)[2:]}"


# ---------------------------------------------------------------------------
# Scope cube: every loan, by lender country x borrower country x season
# ---------------------------------------------------------------------------

def scope_cube(L: pd.DataFrame) -> pd.DataFrame:
    """Summing rows where both countries are in S reproduces
    `scope(countries=S, country_mode="both")`; where either is, `either`."""
    ended = L.terminal_event_id.notna()
    f = L.assign(
        lender_country=L.from_country.fillna(UNMAPPED), borrower_country=L.to_country.fillna(UNMAPPED),
        lender_confederation=L.from_confederation.fillna(UNMAPPED),
        borrower_confederation=L.to_confederation.fillna(UNMAPPED),
        loans=1, loans_with_recorded_ending=ended.astype(int),
        plain_end_of_loan=(ended & (L.raw_label_class == "End of loan")).astype(int),
        not_plain_end_of_loan=(ended & (L.raw_label_class != "End of loan")).astype(int),
        fee_bearing_returns=L.ending_fee_on_return_eur.notna().astype(int),
        nonordinary_sequence=L.economic_ending.isin(loan_scope.NONSTANDARD).astype(int),
        scheduled_ending=L.terminal_scheduled_future.astype(int),
        no_recorded_ending=(~ended).astype(int))
    keys = ["lender_country", "borrower_country", "lender_confederation", "borrower_confederation",
            "loan_season"]
    vals = ["loans", "loans_with_recorded_ending", "plain_end_of_loan", "not_plain_end_of_loan",
            "fee_bearing_returns", "nonordinary_sequence", "scheduled_ending", "no_recorded_ending"]
    return f.groupby(keys, as_index=False)[vals].sum().sort_values(keys).reset_index(drop=True)


def cube_count(cube: pd.DataFrame, countries, seasons, mode: str, col: str = "loans") -> int:
    cs = set(countries)
    lend, borr = cube.lender_country.isin(cs), cube.borrower_country.isin(cs)
    m = (lend & borr) if mode == "both" else (lend | borr)
    return int(cube.loc[m & cube.loan_season.isin(seasons), col].sum())


# ---------------------------------------------------------------------------

def compute() -> dict:
    sha_before = sha256(CANONICAL_CSV)
    df = load_canonical()
    assert len(df) == EXPECTED_ROWS, f"Stage 1C has {len(df):,} rows, expected {EXPECTED_ROWS:,}"
    geo = club_geography()
    u = build_universe(df, geo)
    loan_scope.universe = lambda: u        # scope() runs on this complete universe
    rows, ep, L = u.rows, u.episodes, u.loans
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    N: dict = {"stage1c_sha256": sha_before}

    # ---- Q1
    wf = waterfall(u)
    role = rows.row_role
    internal = rows[role == "internal_registration"]
    sens = sensitivities(u)
    ret = u.returns.dropna(subset=["loan_idx"])
    ret_role = rows.loc[ret.index, "row_role"]
    loan_role = rows.loc[ret.loan_idx.astype(int), "row_role"].to_numpy()
    q1 = {
        "raw_rows": len(rows),
        "internal_registration": int((role == "internal_registration").sum()),
        "internal_unlabelled_youth_or_reserve": int((internal.transfer_type_normalized == "youth_or_internal").sum()),
        "internal_loans": int((internal.transfer_type_normalized == "loan").sum()),
        "internal_returns": int((internal.transfer_type_normalized == "loan_return").sum()),
        "internal_other": int((~internal.transfer_type_normalized.isin(["youth_or_internal", "loan", "loan_return"])).sum()),
        "no_counterpart_placeholder": int((role == "placeholder_non_club_movement").sum()),
        "draft": int((role == "non_transfer_other").sum()),
        "returns_folded_into_loan": int((role == "loan_return_collapsed").sum()),
        "returns_unmatched_set_aside": int((role == "loan_return_unmatched").sum()),
        "real_transfer_episodes": len(ep),
        "episodes_involving_youth_or_reserve_side": sens["episodes_involving_youth_or_reserve_side"],
        "senior_only_episodes": sens["episodes_senior_only"],
        "permanent_episodes_after_same_borrower_conversion": sens["permanent_episodes_that_follow_a_loan_conversion"],
        "matched_returns_of_real_loans_removed_as_internal": int(((ret_role == "internal_registration").to_numpy()
                                                                 & (loan_role == "episode")).sum()),
        "episode_types": ep.episode_type.value_counts().to_dict(),
    }
    q1["broad_episodes_loan_plus_conversion_as_one"] = (q1["real_transfer_episodes"]
                                                        - q1["permanent_episodes_after_same_borrower_conversion"])
    assert int(wf.rows.iloc[-1]) == q1["real_transfer_episodes"]
    N["q1"] = q1

    # ---- Q2
    ended = L.terminal_event_id.notna()
    senior = ~ep.is_youth_or_reserve_side.fillna(False)
    q2 = {"loans": len(L), "real_transfer_episodes": len(ep),
          "senior_only_loans": int((senior & (ep.episode_type == "loan")).sum()),
          "senior_only_episodes": int(senior.sum()),
          "matched_return": int(L.has_matched_return.sum()),
          "ended_by_another_movement": int((ended & ~L.has_matched_return).sum()),
          "no_observed_ending": int((~ended).sum()),
          "ended_by_another_movement_reasons": L.loc[ended & ~L.has_matched_return, "match_reason"]
          .value_counts().to_dict(),
          "scheduled_endings": int(L.terminal_scheduled_future.sum())}
    assert q2["matched_return"] == q1["returns_folded_into_loan"] + q1["matched_returns_of_real_loans_removed_as_internal"]
    N["q2"] = q2

    # ---- Q3
    a, b, fee = loan_outcomes(u)
    labels = a.set_index("raw_label_class").loans.astype(int).to_dict()
    n_end = int(ended.sum())
    plain = labels.get("End of loan", 0)
    fees = L.ending_fee_on_return_eur.dropna()
    econ = L.economic_ending.value_counts().to_dict()
    nonord = L.economic_ending.isin(loan_scope.NONSTANDARD)
    q3 = {"loans_with_observed_ending": n_end, "plain_end_of_loan": plain, "not_plain": n_end - plain,
          "labels": labels,
          "fee_bearing_returns": int(len(fees)),
          "fee_median": float(fees.median()), "fee_p25": float(fees.quantile(.25)),
          "fee_p75": float(fees.quantile(.75)), "fee_max": float(fees.max()), "fee_min": float(fees.min()),
          "nonordinary_sequence": int(nonord.sum()),
          "nonordinary_sequence_ended_by_plain_end_of_loan": int((nonord & (L.raw_label_class == "End of loan")).sum()),
          "economic": {k: int(v) for k, v in econ.items()},
          "loans_2025_26": int((L.loan_season == "2025/26").sum()),
          "loans_2025_26_scheduled_end": int(((L.loan_season == "2025/26") & L.terminal_scheduled_future).sum())}
    assert q3["fee_bearing_returns"] == labels.get("End of loan + fee", 0)
    N["q3"] = q3

    # ---- Q4
    eu, cov, _ = europe_end_dates(u)
    last = latest_completed_season(u)
    through = cov["through_slice"]

    def cell(pop, sl):
        r = eu[(eu.population == pop) & (eu.slice == sl)].iloc[0]
        return {k: (int(v) if k.endswith("_n") or k == "N" else float(v)) for k, v in r.items()
                if k not in ("population", "slice")}

    pops = {"all": "all matched endings", "plain": "raw label: plain End of loan",
            "not_plain": "raw label: not plain End of loan", "fee": "fee-bearing returns"}
    EU = L[(L.from_confederation == "UEFA") & (L.to_confederation == "UEFA")]
    E = EU[EU.terminal_date.notna()]
    realised = E[~E.terminal_scheduled_future]
    main = realised[realised.loan_season <= last]
    src = player_source_file()
    main_src = main.player_id.map(src)
    s2324 = realised[realised.loan_season <= "2023/24"]
    lend = main.from_country
    uk = lend.isin(["England", "Scotland"])
    md = main.terminal_date.dt.strftime("%m-%d")
    s1920 = main[main.loan_season == "2019/20"]
    s1920_late = s1920.terminal_date.dt.strftime("%Y-%m-%d").isin(["2020-07-31", "2020-08-31"])
    ns = main[main.raw_label_class != "End of loan"]
    ns_either = int(ns.terminal_date.dt.strftime("%m-%d").isin(["06-30", "12-31"]).sum())
    lo, hi = wilson(ns_either, len(ns))

    def lender_row(mask):
        g = main[mask]
        k = int(g.terminal_date.dt.strftime("%m-%d").isin(["06-30", "12-31"]).sum())
        return {"N": len(g), "jun30_or_dec31_n": k}

    q4 = {
        "all_loans": len(L),
        "loans_with_unmapped_club": cov["loans_with_a_club_of_unknown_country"],
        "loans_both_clubs_mapped": cov["loans_both_clubs_mapped"],
        "loans_both_uefa": cov["loans_both_clubs_uefa"],
        "uefa_with_recorded_ending": cov["uefa_loans_with_matched_ending"],
        "uefa_without_ending": cov["uefa_loans_without_ending"],
        "uefa_scheduled_endings": cov["uefa_loans_ending_scheduled_future"],
        "uefa_scheduled_on_jun30_or_dec31": int(E[E.terminal_scheduled_future].terminal_date.dt.strftime("%m-%d")
                                                .isin(["06-30", "12-31"]).sum()),
        "uefa_realised": len(realised),
        "latest_completed_season": last,
        "main": {k: cell(v, through) for k, v in pops.items()},
        "all_recorded_incl_scheduled": {k: cell(v, "all") for k, v in pops.items()},
        "realised_all_seasons": {k: cell(v, "realised") for k, v in pops.items()},
        "realised_through_2023_24": {
            "all": date_buckets(s2324), "not_plain": date_buckets(s2324[s2324.raw_label_class != "End of loan"]),
            "fee": date_buckets(s2324[s2324.ending_fee_on_return_eur.notna()])},
        "main_by_source_file": main_src.value_counts().sort_index().to_dict(),
        "main_2024_25_by_source_file": main_src[main.loan_season == "2024/25"].value_counts().sort_index().to_dict(),
        "main_2024_25": int((main.loan_season == "2024/25").sum()),
        "realised_2025_26": int((realised.loan_season == "2025/26").sum()),
        "realised_2025_26_on_jun30": int((realised[realised.loan_season == "2025/26"].terminal_date
                                          .dt.strftime("%m-%d") == "06-30").sum()),
        "not_plain_wilson95": [lo, hi],
        "not_plain_fee_bearing": int(ns.ending_fee_on_return_eur.notna().sum()),
        "may31_n": int((md == "05-31").sum()),
        "may31_from_england_or_scotland": int(((md == "05-31") & uk).sum()),
        "lender_england": lender_row(lend == "England"), "lender_scotland": lender_row(lend == "Scotland"),
        "lender_italy": lender_row(lend == "Italy"), "lender_all_others": lender_row(~uk),
        "season_2019_20_N": len(s1920), "season_2019_20_ending_31jul_or_31aug_2020": int(s1920_late.sum()),
    }
    N["q4"] = q4

    # ---- Q5a
    cube = scope_cube(L)
    ex_both = loan_scope.scope(**EXAMPLE, country_mode="both")
    ex_either = loan_scope.scope(**EXAMPLE, country_mode="either")
    ex_both_r = loan_scope.scope(**EXAMPLE, country_mode="both", realised_only=True)
    ex_either_r = loan_scope.scope(**EXAMPLE, country_mode="either", realised_only=True)
    full = loan_scope.scope()
    checks = {
        "scope_default_loans_equals_all_loans": full["loan_episodes"] == len(L),
        "cube_total_equals_all_loans": int(cube.loans.sum()) == len(L),
        "cube_reproduces_example_both": cube_count(cube, EXAMPLE["countries"], EXAMPLE["seasons"], "both")
        == ex_both["loan_episodes"],
        "cube_reproduces_example_either": cube_count(cube, EXAMPLE["countries"], EXAMPLE["seasons"], "either")
        == ex_either["loan_episodes"],
        "cube_reproduces_example_fee_bearing": cube_count(cube, EXAMPLE["countries"], EXAMPLE["seasons"], "both",
                                                          "fee_bearing_returns") == ex_both["fee_bearing_returns"],
    }
    assert all(checks.values()), checks
    keep = ("real_transfer_episodes", "loan_episodes", "loans_with_recorded_ending",
            "raw_label_not_plain_end_of_loan", "fee_bearing_returns", "ordinary_loan_endings",
            "nonstandard_loan_endings", "unresolved_or_open_loans")
    N["q5a"] = {"example": EXAMPLE, "both": {k: ex_both[k] for k in keep},
                "either": {k: ex_either[k] for k in keep},
                "both_realised_only_loans": ex_both_r["loan_episodes"],
                "either_realised_only_loans": ex_either_r["loan_episodes"],
                "all_loans_in_scope_default": full["loan_episodes"], "checks": checks,
                "cube_rows": len(cube),
                "loans_lender_mapped": int(L.from_country.notna().sum()),
                "loans_borrower_mapped": int(L.to_country.notna().sum())}

    # ---- Q5b
    squads = pd.read_csv(SQUAD_CSV)
    facts = json.loads(FACTS_JSON.read_text())
    leagues = pd.read_csv(LEAGUE_CSV)
    assert facts["db_transfer_players"] == rows.player_id.nunique(), "provenance outputs are stale"
    up = upstream_tables(con)
    clubs = club_country_counts(u, geo, con)
    cap = source_file_capture(rows)
    fut = rows.transfermarkt_future_transfer.astype(bool)
    gt = up["games_by_type"].set_index("type")
    club_types = [t for t in gt.index if t != "national_team_competition"]
    lbs = up["leagues_by_season"].assign(season=lambda d: d.season.astype(int))
    sel = squads[squads.transfers_json_exists]
    pls = up["players_by_last_season"]
    pls["ls"] = pls.last_season.astype(int)
    q5b = {
        "commit": con.sql("select commit_hash from version").fetchall()[0][0],
        "transfer_players": int(rows.player_id.nunique()),
        "episode_players": int(ep.player_id.nunique()),
        "raw_clubs": clubs["raw_transfer_clubs"], "episode_clubs": clubs["episode_clubs"],
        "clubs_table": clubs["clubs_table"], "clubs_with_country_all": clubs["clubs_with_country"],
        "episode_clubs_with_country": clubs["episode_clubs_with_country"],
        "loans_both_countries_known": clubs["loans_both_known"],
        "episodes_both_countries_known": int((ep.from_country.notna() & ep.to_country.notna()).sum()),
        "players_table": int(up["players"]),
        "players_last_season_2012_2021": int(pls[pls.ls <= 2021].players.sum()),
        "players_last_season_2012_2021_with_transfers": int(pls[pls.ls <= 2021].with_transfers.sum()),
        "earliest_row": str(rows._date.min().date()),
        "latest_realised_row": str(rows.loc[~fut, "_date"].max().date()),
        "latest_scheduled_row": str(rows.loc[fut, "_date"].max().date()),
        "scheduled_rows": int(fut.sum()),
        "raw_seasons": sorted(set(rows.season))[0] + " to " + sorted(set(rows.season))[-1],
        "episode_seasons_n": int(ep.season.nunique()),
        "episode_seasons": min(ep.season) + " to " + max(ep.season),
        "club_games_first_season": season(min(gt.loc[t, "first_season"] for t in club_types)),
        "club_games_last_season": season(max(gt.loc[t, "last_season"] for t in club_types)),
        "club_games_first_date": str(pd.Timestamp(min(gt.loc[t, "first_date"] for t in club_types)).date()),
        "club_games_last_date": str(pd.Timestamp(max(gt.loc[t, "last_date"] for t in club_types)).date()),
        "national_team_games_first_date": str(pd.Timestamp(gt.loc["national_team_competition", "first_date"]).date()),
        "leagues_per_season": {season(r.season): int(r.leagues) for r in lbs.itertuples()},
        "selection_squad_seasons": sel.season.tolist(),
        "transfer_files": facts["transfer_files"],
        "squad_seasons_audited": squads.season.tolist(),
        "squad_rows": squads.to_dict("records"),
        "cohort_union_players": facts["squad_players_2023_2025_union"],
        "cohort_union_with_history": facts["squad_players_2023_2025_union_with_history"],
        "transfer_players_on_2023_2025_squad": facts["db_players_on_a_2023_2025_squad_file"],
        "transfer_players_not_on_2023_2025_squad": facts["db_players_not_on_a_2023_2025_squad_file"],
        "transfer_players_only_via_2025_national_team": facts["db_players_only_via_2025_national_team_squad"],
        "capture": cap.assign(last_realised_date=cap.last_realised_date.astype(str),
                              first_scheduled_date=cap.first_scheduled_date.astype(str)).to_dict("records"),
        "earliest_rows": up["earliest_transfers"].astype(str).to_dict("records"),
        "league_2024_new_leagues_in_2024_file": int(leagues[(leagues.squad_season == 2024)
                                                            & ~leagues.league.isin(lbs[lbs.season == 2023].ids.iloc[0].split(","))
                                                            & (leagues.league != "national_team")]
                                                    .in_this_seasons_transfers_json.sum()),
        "league_2024_new_leagues_squad_players": int(leagues[(leagues.squad_season == 2024)
                                                             & ~leagues.league.isin(lbs[lbs.season == 2023].ids.iloc[0].split(","))
                                                             & (leagues.league != "national_team")]
                                                     .squad_players.sum()),
    }
    early = lbs[lbs.season == lbs.season.min()].ids.iloc[0].split(",")
    q5b["leagues_from_2012"] = early
    l24 = leagues[(leagues.squad_season == 2024) & leagues.league.isin(early)]
    q5b["league_2024_old_leagues_in_2024_file"] = int(l24.in_this_seasons_transfers_json.sum())
    q5b["league_2024_old_leagues_squad_players"] = int(l24.squad_players.sum())
    r25 = squads[squads.squad_season == 2025].iloc[0]
    q5b["club_squad_2025_without_history"] = int(r25.club_squad_players - r25.club_squad_with_history)
    q5b["nt_only_2025_with_history"] = int(r25.national_team_only_with_history)
    q5b["pre_2012_players"] = int(rows.loc[rows._date < "2012-07-01", "player_id"].nunique())
    N["q5b"] = q5b

    # ---- Q5c
    tp = loan_scope.token_profile()
    old = loan_scope.token_profile(loan_scope.EARLIER_PROMPT_CACHE)
    scen = loan_scope.scenarios()
    grid = [{"scope_size_N": n, "scenario": s, "tokens_per_call": scen[s][0], "calls_per_case": scen[s][1],
             "share_reaching_extraction": sh, "extraction_tokens": loan_scope.extraction_tokens(n, sh, s)}
            for n in SCOPE_SIZES + (ex_both["loan_episodes"],) for s in scen for sh in SHARES]
    N["q5c"] = {"observed": {k: tp[k] for k in ("researched_cases", "researched_cases_reaching_extraction",
                                                "calls", "families", "calls_per_family", "total_tokens_median",
                                                "total_tokens_mean", "total_tokens_p90", "total_tokens_p25",
                                                "cost_usd_mean_per_call")},
                "earlier_prompt_calls": old["calls"], "earlier_prompt_families": old["families"],
                "scenarios": {k: {"tokens_per_call": v[0], "calls_per_case": v[1],
                                  "tokens_per_case_reaching_extraction": v[0] * v[1]} for k, v in scen.items()},
                "shares": list(SHARES)}
    con.close()
    N["stage1c_unchanged"] = sha256(CANONICAL_CSV) == sha_before
    return {"N": N, "cube": cube, "grid": pd.DataFrame(grid), "squads": squads, "leagues": leagues, "u": u}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def md_table(head: list[str], rows: list[list], right: set[int] | None = None) -> list[str]:
    right = right if right is not None else set(range(1, len(head)))
    out = ["| " + " | ".join(head) + " |",
           "| " + " | ".join("---:" if i in right else "---" for i in range(len(head))) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def dates_row(label, b) -> list:
    n = b["N"]
    return [label, f0(n), f"{f0(b['jun30_n'])} ({pc(b['jun30_n'], n)})", f"{f0(b['dec31_n'])} ({pc(b['dec31_n'], n)})",
            f"{f0(b['jun30_or_dec31_n'])} ({pc(b['jun30_or_dec31_n'], n)})",
            f"{f0(b['other_dates_n'])} ({pc(b['other_dates_n'], n)})"]


DATE_HEAD = ["Ending", "N", "30 June", "31 December", "Either date", "Other date"]


def render(N: dict) -> str:
    q1, q2, q3, q4, q5a, q5b, q5c = (N[k] for k in ("q1", "q2", "q3", "q4", "q5a", "q5b", "q5c"))
    W: list[str] = []
    w = W.append
    sel = q5b["selection_squad_seasons"]
    sel_txt = ", ".join(sel[:-1]) + " and " + sel[-1]
    sel_or = ", ".join(sel[:-1]) + " or " + sel[-1]
    cap = {c["source_season_file"]: c for c in q5b["capture"]}
    last = q4["latest_completed_season"]

    w("# What is in the Transfermarkt dataset, and how large would different analyses be?")
    w("")
    w(f"*Answers to Professor Freund's questions, in the order of his email. Every number below is computed by "
      f"`python -m src.analysis.daniel_scope_final` from the frozen Stage 1C table ({f0(q1['raw_rows'])} rows, "
      f"SHA-256 `{N['stage1c_sha256'][:8]}…{N['stage1c_sha256'][-6:]}`, unchanged), the upstream DuckDB snapshot, "
      "the provenance audit and the Stage 2 extraction cache. Nothing was scraped and no new transfer histories were "
      "acquired. This document supersedes `daniel_stage1_scope_answers.md`: its counts are unchanged, and its "
      "description of how players were selected is corrected.*")
    w("")
    w("## Headline numbers")
    w("")
    fr = q5b["capture"]
    W += md_table(["Item", "Value"], [
        ["Raw rows (Transfermarkt transfer-history entries)", f0(q1["raw_rows"])],
        ["Real transfer episodes (primary count, Q1)", f0(q1["real_transfer_episodes"])],
        ["Senior-only sensitivity (Q1)", f0(q1["senior_only_episodes"])],
        ["Loans (Q2)", frac(q2["loans"], q2["real_transfer_episodes"]) + " of real transfer episodes"],
        ["Loans with a recorded ending (Q3)", f0(q3["loans_with_observed_ending"])],
        ["… whose ending is not a plain `End of loan` (Q3)", frac(q3["not_plain"], q3["loans_with_observed_ending"])],
        ["Fee-bearing loan returns (`End of loan` + fee, Q3)", f0(q3["fee_bearing_returns"])],
        ["Players with transfer histories", f0(q5b["transfer_players"])],
        ["Players in real transfer episodes", f0(q5b["episode_players"])],
        ["Clubs in raw transfer rows", f0(q5b["raw_clubs"])],
        ["Clubs in real transfer episodes", f0(q5b["episode_clubs"])],
        ["Observed transfer-history range", f"{q5b['earliest_row']} to {q5b['latest_realised_row']} for moves that "
                                            f"had happened; scheduled moves run to {q5b['latest_scheduled_row']}"],
        ["Transfer-selection squad seasons", f"{sel_txt} (players on those squads had their histories fetched)"],
    ], right=set())
    w("")
    w("**Three dates that mean different things.** "
      f"(1) **{q5b['earliest_row'][:4]}/{str(int(q5b['earliest_row'][:4]) + 1)[2:]}** is the oldest move in any "
      "selected player's history, not a cutoff anyone chose. "
      f"(2) **{q5b['club_games_first_season']}** is where the upstream project's separate league, squad and match "
      "data begins; it plays no part in which transfers are included. "
      f"(3) **{sel_txt}** are the squad seasons whose players actually had their transfer histories fetched.")
    w("")

    # ---- Q1
    w("## Q1. How many transfers are in the dataset?")
    w("")
    w("**Question.** How many rows are there, and how many are *real* transfers, i.e. not counting youth, B-team "
      "or internal progressions, and not counting a loan and its return as two transfers?")
    w("")
    w(f"**Short answer.** **{f0(q1['raw_rows'])} raw rows** reduce to **{f0(q1['real_transfer_episodes'])} real "
      f"transfer episodes** ({pc(q1['real_transfer_episodes'], q1['raw_rows'])} of rows). Counting only moves "
      f"between two senior sides gives **{f0(q1['senior_only_episodes'])}** as a sensitivity.")
    w("")
    w("A *real transfer episode* is our term for one move of a player's registration from one club organisation "
      "to a different one, counted once. A loan and its matching `End of loan` return are one episode (the loan). "
      "Moves inside one organisation (U19 → first team, Porto B → Porto) and moves to or from \"Without Club\" or "
      "\"Retired\" are not episodes.")
    w("")
    w("**How we calculated it.** Each Stage 1C row is given exactly one role, in this order:")
    w("")
    W += md_table(["Step", "Rows"], [
        ["Raw rows (registration movements)", f0(q1["raw_rows"])],
        [f"− Internal moves within one club organisation ({f0(q1['internal_unlabelled_youth_or_reserve'])} youth/"
         f"reserve progressions, {f0(q1['internal_loans'])} internal loans, {f0(q1['internal_returns'])} internal "
         f"returns, {f0(q1['internal_other'])} other)", "−" + f0(q1["internal_registration"])],
        ["− Movements with no counterpart club (Without Club, Retired, Career break, Ban, Unknown)",
         "−" + f0(q1["no_counterpart_placeholder"])],
        ["− Draft rows (e.g. MLS SuperDraft): an allocation, not a club-to-club transfer", "−" + f0(q1["draft"])],
        ["− Loan returns folded into the outbound loan they close", "−" + f0(q1["returns_folded_into_loan"])],
        ["− Loan returns with no outbound loan to close, set aside (not counted, not forced onto a loan)",
         "−" + f0(q1["returns_unmatched_set_aside"])],
        ["**= Real transfer episodes (primary count)**", f"**{f0(q1['real_transfer_episodes'])}**"],
    ])
    w("")
    w("Two sensitivities answer *different definitions*; neither is automatically more correct:")
    w("")
    w(f"- **Senior-only: {f0(q1['senior_only_episodes'])}.** Also drops the "
      f"{f0(q1['episodes_involving_youth_or_reserve_side'])} episodes in which either club is a youth or reserve side "
      "of a *different* organisation (e.g. Man City U18 → Dortmund). Those are genuine moves between clubs, so the "
      "primary count keeps them; drop them if only senior-football transfers matter.")
    w(f"- **Loan + conversion as one episode: {f0(q1['broad_episodes_loan_plus_conversion_as_one'])}.** "
      f"{f0(q1['permanent_episodes_after_same_borrower_conversion'])} permanent episodes are the second step of a "
      "loan followed by a permanent move to the same borrowing club. Counting each pair once subtracts them. This "
      "is a counting choice; it does not show that a purchase option existed.")
    w("")
    w("**Important caveat.** Same-organisation moves are detected from club names, so reserve sides with different "
      "names (Barça Atlètic → Barcelona, Milan Futuro → AC Milan) slip through as real transfers in both counts. "
      "The count is also only as complete as the players selected (Q5b).")
    w("")

    # ---- Q2
    w("## Q2. How many real transfers are loans?")
    w("")
    w("**Question.** Of the real transfers, how many are loans?")
    w("")
    w(f"**Short answer.** **{frac(q2['loans'], q2['real_transfer_episodes'])}** of real transfer episodes. "
      f"Senior-only sensitivity: {frac(q2['senior_only_loans'], q2['senior_only_episodes'])}.")
    w("")
    w("**How we calculated it.** A loan is counted once, on its outbound row; its return is folded into it. Each "
      "player's moves are walked in date order and every return closes at most one open loan (tested over the "
      "whole table). How the loans ended:")
    w("")
    W += md_table(["Ending", "Loans", "Share of loans"], [
        ["Matched `End of loan` return row", f0(q2["matched_return"]), pc(q2["matched_return"], q2["loans"])],
        ["Ended by another movement (permanent move, re-loan, left the borrower, sold by the lender)",
         f0(q2["ended_by_another_movement"]), pc(q2["ended_by_another_movement"], q2["loans"])],
        ["No observed ending (still open, or next move involves neither club)", f0(q2["no_observed_ending"]),
         pc(q2["no_observed_ending"], q2["loans"])],
    ])
    w("")
    w(f"The {f0(q2['matched_return'])} matched returns are the {f0(q1['returns_folded_into_loan'])} returns folded "
      f"in Q1 plus {q1['matched_returns_of_real_loans_removed_as_internal']} whose return row was already removed as an "
      "internal move.")
    w("")
    w(f"**Important caveat.** {f0(q2['scheduled_endings'])} of the recorded endings are *scheduled* returns — loans "
      "still running when the player's history was captured, shown with the date they are due to end.")
    w("")

    # ---- Q3
    lab = q3["labels"]
    other = [(k, v) for k, v in sorted(lab.items(), key=lambda kv: -kv[1]) if k not in ("End of loan",)]
    w("## Q3. How many loans end with something other than a plain `End of loan`?")
    w("")
    w("**Question.** How many loans end with something other than a plain `End of loan`, for example where a fee "
      "is paid?")
    w("")
    w(f"**Short answer (literal, by the ending label).** **{frac(q3['not_plain'], q3['loans_with_observed_ending'])}** "
      f"of loans with an observed ending. **{q3['fee_bearing_returns']}** of them are `End of loan` + a fee.")
    w("")
    w(f"**How we calculated it.** We read Transfermarkt's own fee cell on the row that ended each of the "
      f"{f0(q3['loans_with_observed_ending'])} loans with an observed ending (the {q2['no_observed_ending']} open "
      "loans have no ending row and are left out of the denominator). `?`, `-`, `free transfer` and fee amounts are "
      "kept separate; none is turned into zero.")
    w("")
    W += md_table(["Label on the ending row", "Loans", f"Share of {f0(q3['loans_with_observed_ending'])}"],
                  [["Plain `End of loan`", f0(q3["plain_end_of_loan"]), pc(q3["plain_end_of_loan"], q3["loans_with_observed_ending"], 2)],
                   ["**Not plain `End of loan`**", f"**{f0(q3['not_plain'])}**",
                    f"**{pc(q3['not_plain'], q3['loans_with_observed_ending'], 2)}**"]]
                  + [["… " + LABEL_DISPLAY.get(k, f"`{k}`"), f0(v), pc(v, q3["loans_with_observed_ending"], 2)]
                     for k, v in other])
    w("")
    w(f"Only the `End of loan` + fee endings are return rows. The other "
      f"{f0(q3['not_plain'] - q3['fee_bearing_returns'])} are loans that ended with a different kind of movement "
      "(no return row), whose label is that movement's fee cell. Transfermarkt return rows are only ever labelled "
      "`End of loan`, with or without a fee.")
    w("")
    w(f"**Fee on the {q3['fee_bearing_returns']} fee-bearing returns:**")
    w("")
    W += md_table(["Count", "Median", "p25", "p75", "Maximum"],
                  [[q3["fee_bearing_returns"], eur(q3["fee_median"]), eur(q3["fee_p25"]), eur(q3["fee_p75"]),
                    eur(q3["fee_max"])]], right={0, 1, 2, 3, 4})
    w("")
    w("The fee is reported as Transfermarkt shows it; what it paid for is not known.")
    w("")
    e = q3["economic"]
    w(f"**Secondary statistic, not the answer to this question.** {frac(q3['nonordinary_sequence'], q2['loans'])} of "
      "loans sit in a *non-ordinary surrounding registration sequence*: something other than a return with nothing "
      "after it. It counts, for example, a plain `End of loan` followed shortly by a permanent transfer "
      f"({f0(e.get('purchase_option_or_permanent_conversion', 0))} loans are followed by a permanent move to the "
      f"borrower), a re-loan or another move within weeks ({f0(e.get('immediate_follow_on_transfer', 0))}), and a "
      f"sale to a third club ({f0(e.get('third_party_sale_related', 0))}). "
      f"{f0(q3['nonordinary_sequence_ended_by_plain_end_of_loan'])} of these "
      f"{f0(q3['nonordinary_sequence'])} loans ended with a plain `End of loan` row. The measure describes what the "
      "sequence of moves looks like; it does **not** prove that any contractual clause (option, obligation, "
      "buy-back) existed.")
    w("")
    w(f"**Important caveat.** Recent loans cannot yet show their ending: {f0(q3['loans_2025_26_scheduled_end'])} of "
      f"the {f0(q3['loans_2025_26'])} loans from 2025/26 have only a scheduled end.")
    w("")

    # ---- Q4
    m = q4["main"]
    w("## Q4. European loan end dates")
    w("")
    w("**Question.** Within UEFA, with both clubs European: what fraction of loans end on 30 June, on 31 December, "
      "or on either, and how do those fractions change for loans whose ending is not a plain `End of loan`?")
    w("")
    a = m["all"]
    ns = m["not_plain"]
    w(f"**Short answer.** **{pc(a['jun30_or_dec31_n'], a['N'])}** of the European loan endings observed in the data "
      f"fall on 30 June or 31 December: **{pc(a['jun30_n'], a['N'])}** on 30 June and **{pc(a['dec31_n'], a['N'])}** "
      f"on 31 December (N = {f0(a['N'])}). For endings that are not a plain `End of loan` the share is "
      f"**{pc(ns['jun30_or_dec31_n'], ns['N'])}** ({pc(ns['jun30_n'], ns['N'])} on 30 June, "
      f"{pc(ns['dec31_n'], ns['N'])} on 31 December; N = {ns['N']}), too few to tell apart from the overall share.")
    w("")
    w("**How we calculated it.**")
    w("")
    w(f"- **Sample:** loans where both the lending and the borrowing club are mapped to a UEFA country. Countries "
      "come only from the upstream snapshot; no club's country is guessed.")
    w(f"- **Denominator:** of all {f0(q4['all_loans'])} loans, {frac(q4['loans_with_unmapped_club'], q4['all_loans'])} "
      f"have a club of unknown country and are excluded, not assumed European. {f0(q4['loans_both_uefa'])} have both "
      "clubs in UEFA countries, and " + ("all of them" if q4["uefa_with_recorded_ending"] == q4["loans_both_uefa"]
                                        else f"{f0(q4['uefa_with_recorded_ending'])} of those") +
      " have a recorded ending.")
    w(f"- **Main table:** endings that had already happened when the player's history was captured "
      f"({f0(q4['uefa_realised'])}), for loans starting in {last} or earlier (**N = {f0(a['N'])}**). Scheduled "
      f"endings are left out because they are the date a still-running loan is *due* to end: "
      f"{f0(q4['uefa_scheduled_on_jun30_or_dec31'])} of the {f0(q4['uefa_scheduled_endings'])} scheduled European "
      f"endings ({pc(q4['uefa_scheduled_on_jun30_or_dec31'], q4['uefa_scheduled_endings'])}) fall on 30 June or 31 "
      "December.")
    w("")
    W += md_table(DATE_HEAD, [dates_row("All endings", m["all"]), dates_row("Plain `End of loan`", m["plain"]),
                              dates_row("**Not plain `End of loan`**", m["not_plain"]),
                              dates_row("… of which fee-bearing returns", m["fee"])])
    w("")
    lo, hi = q4["not_plain_wilson95"]
    w(f"{q4['not_plain_fee_bearing']} of the {ns['N']} non-plain European endings are `End of loan` + a fee. With "
      f"N = {ns['N']}, the 95% interval on {pc(ns['jun30_or_dec31_n'], ns['N'])} is {100 * lo:.0f}–{100 * hi:.0f}%, "
      f"which contains the overall {pc(a['jun30_or_dec31_n'], a['N'])}.")
    w("")
    w("Sensitivities:")
    w("")
    r23 = q4["realised_through_2023_24"]
    ai, ri = q4["all_recorded_incl_scheduled"], q4["realised_all_seasons"]
    W += md_table(["Sample", "All endings: either date", "Not plain: either date"], [
        [f"Main table (realised, loans through {last})", frac(a["jun30_or_dec31_n"], a["N"]),
         frac(ns["jun30_or_dec31_n"], ns["N"])],
        ["Realised, loans through 2023/24 (every history captured after that season ended)",
         frac(r23["all"]["jun30_or_dec31_n"], r23["all"]["N"]),
         frac(r23["not_plain"]["jun30_or_dec31_n"], r23["not_plain"]["N"])],
        ["Realised, all seasons including 2025/26", frac(ri["all"]["jun30_or_dec31_n"], ri["all"]["N"]),
         frac(ri["not_plain"]["jun30_or_dec31_n"], ri["not_plain"]["N"])],
        ["All recorded endings, including scheduled", frac(ai["all"]["jun30_or_dec31_n"], ai["all"]["N"]),
         frac(ai["not_plain"]["jun30_or_dec31_n"], ai["not_plain"]["N"])],
    ], right={1, 2})
    w("")
    e_, s_, i_, o_ = (q4[k] for k in ("lender_england", "lender_scotland", "lender_italy", "lender_all_others"))
    w("**Important caveats.**")
    w("")
    w(f"- **30 June / 31 December is not a universal season end.** Transfermarkt records British loans as ending on "
      f"31 May, the second most common end date ({frac(q4['may31_n'], a['N'])} of main-table endings, "
      f"{f0(q4['may31_from_england_or_scotland'])} of them from English or Scottish lenders). The share on 30 June or "
      f"31 December is {frac(e_['jun30_or_dec31_n'], e_['N'])} for English lenders and "
      f"{frac(s_['jun30_or_dec31_n'], s_['N'])} for Scottish lenders, against "
      f"{frac(i_['jun30_or_dec31_n'], i_['N'])} for Italian lenders and "
      f"{frac(o_['jun30_or_dec31_n'], o_['N'])} for all lenders outside England and Scotland. 2019/20 is also unusual: "
      f"{f0(q4['season_2019_20_ending_31jul_or_31aug_2020'])} of its {f0(q4['season_2019_20_N'])} endings fall on "
      "31 July or 31 August 2020, when the COVID-extended season finished.")
    bysrc = q4["main_by_source_file"]
    b2425 = q4["main_2024_25_by_source_file"]
    w(f"- **These are descriptive statistics of the observed dataset, not a census of European loans.** Histories "
      f"were captured at three different times (Q5b). {last} is complete only for players captured in 2026: the "
      f"{f0(q4['main_2024_25'])} main-table loans from {last} come from the spring-2026 file "
      f"({f0(b2425.get('2024', 0))}) and the June-2026 file ({f0(b2425.get('2025', 0))}), and "
      f"{f0(b2425.get('2023', 0))} from the {f0(cap['2023']['players'])} players captured in July 2024, whose later "
      "moves are absent from the data. The 2023/24 sensitivity above avoids that. The only 2025/26 European loans "
      f"with an ending that had happened are early exits ({f0(q4['realised_2025_26'])}, "
      f"{f0(q4['realised_2025_26_on_jun30'])} on 30 June).")
    w(f"- The sample depends on club countries being known, which is true for "
      f"{frac(q5b['loans_both_countries_known'], q4['all_loans'])} of loans (Q5b).")
    w("")

    # ---- Q5a
    b, ei = q5a["both"], q5a["either"]
    ex = q5a["example"]
    w("## Q5a. For countries {X, Y, Z} and seasons {A, B, C}, how many loans are involved?")
    w("")
    w("**Question.** For an arbitrary set of countries and seasons, how many loans would a study cover?")
    w("")
    w(f"**Short answer.** Any combination can be counted with one call, over all {f0(q5a['all_loans_in_scope_default'])} "
      f"loans. *Worked example only:* {', '.join(ex['countries'])} × {', '.join(ex['seasons'])} gives "
      f"**{f0(b['loan_episodes'])} loans with both clubs in those countries**, or **{f0(ei['loan_episodes'])} with at "
      "least one**.")
    w("")
    w("**How we calculated it.**")
    w("")
    w("```")
    w("python -m src.analysis.loan_scope --countries X,Y,Z --seasons A,B,C --country-mode both")
    w("python -m src.analysis.loan_scope --countries X,Y,Z --seasons A,B,C --country-mode either")
    w("```")
    w("")
    w("or from Python: `from src.analysis.loan_scope import scope; scope(countries=[X, Y, Z], seasons=[A, B, C], "
      "country_mode=\"both\")`. Seasons are the season of the outbound loan and can be written `2021/22`, `21/22` "
      "or `2021-22`.")
    w("")
    w("- `both`: both clubs are mapped to a country, and both countries are in the selected set.")
    w("- `either`: at least one club is mapped to a selected country. This also admits cross-border loans.")
    w("- Clubs whose country is unknown are left out of the country test, never guessed.")
    w("")
    W += md_table([f"Worked example: {', '.join(ex['countries'])} × {ex['seasons'][0]}–{ex['seasons'][-1]}", "`both`", "`either`"], [
        ["Real transfer episodes", f0(b["real_transfer_episodes"]), f0(ei["real_transfer_episodes"])],
        ["Loans", f"**{f0(b['loan_episodes'])}**", f"**{f0(ei['loan_episodes'])}**"],
        ["… with a recorded ending", f0(b["loans_with_recorded_ending"]), f0(ei["loans_with_recorded_ending"])],
        ["… ending not plain `End of loan` (Q3 literal)", f0(b["raw_label_not_plain_end_of_loan"]),
         f0(ei["raw_label_not_plain_end_of_loan"])],
        ["… fee-bearing returns", f0(b["fee_bearing_returns"]), f0(ei["fee_bearing_returns"])],
        ["… non-ordinary surrounding sequence (Q3 secondary)", f0(b["nonstandard_loan_endings"]),
         f0(ei["nonstandard_loan_endings"])],
        ["… no observed ending", f0(b["unresolved_or_open_loans"]), f0(ei["unresolved_or_open_loans"])],
    ])
    w("")
    w(f"`--realised-only` drops scheduled moves and scheduled endings ({f0(q5a['both_realised_only_loans'])} and "
      f"{f0(q5a['either_realised_only_loans'])} loans here); `--through-season 2024/25` stops at a season; "
      "`--confederation UEFA` filters by confederation.")
    w("")
    w(f"**Counts without running code:** `loan_scope_cube.csv` has one row per lender country × borrower country × "
      f"loan season ({f0(q5a['cube_rows'])} rows covering all {f0(q5a['all_loans_in_scope_default'])} loans; unmapped "
      "clubs appear as `(unmapped)`). Summing it gives counts by country (either side), by season, by country × "
      "season and by lender → borrower country; summing the rows where both countries are in the set reproduces "
      "`both`, and rows where either is reproduces `either`. Both reproductions are checked each time the document is "
      "generated.")
    w("")
    w(f"**Important caveat.** Neither mode is a complete count: the lending club's country is known for "
      f"{frac(q5a['loans_lender_mapped'], q2['loans'])} of loans and the borrowing club's for "
      f"{frac(q5a['loans_borrower_mapped'], q2['loans'])}. `both` misses loans to unmapped lower-league and youth "
      "clubs, and a loan between two unmapped clubs is missed by both, so neither is an exact lower or upper bound. "
      "The counts are also bounded by the player selection (Q5b).")
    w("")

    # ---- Q5b
    rowsq = q5b["squad_rows"]
    w("## Q5b. What does the dataset cover?")
    w("")
    w("**Question.** What does the dataset cover: seasons, players, clubs, countries?")
    w("")
    w(f"**Short answer.** Transfermarkt's full transfer-history lists for a recent player cohort, selected from "
      f"covered squads in {sel[0]}–{sel[-1]}; the historical moves in those players' careers extend back to "
      f"{q5b['earliest_row'][:4]}. It is a sample of players, not a census of any league.")
    w("")
    W += md_table(["Item", "Value"], [
        ["Source", f"`dcaribou/transfermarkt-datasets`, a community dataset scraped from Transfermarkt (DuckDB "
                   f"snapshot, upstream commit `{q5b['commit'][:10]}`). Not an official Transfermarkt product."],
        ["Unit", "One entry on a player's Transfermarkt transfer-history list"],
        ["Sampling", "Player-centred: leagues decide which squads are read, squads decide which players are selected, "
                     "and every move on a selected player's list is kept"],
        ["How a player enters", f"On the squad of a club in a covered league in {sel_or}, or on a 2025/26 national-team "
                                "squad, and that player's history was fetched"],
        ["Transfer acquisition files", ", ".join(f"`{f}`" for f in q5b["transfer_files"]) + " only"],
        ["What is fetched", "The whole transfer history, in one request per player; no transfer-date cutoff"],
        ["Realised transfer dates", f"{q5b['earliest_row']} to {q5b['latest_realised_row']}"],
        ["Competition / squad data (separate)", f"{q5b['club_games_first_season']}–{q5b['club_games_last_season']}: "
                                                f"{q5b['leagues_per_season'][q5b['club_games_first_season']]} European "
                                                f"first-tier leagues until 2023/24, "
                                                f"{q5b['leagues_per_season'].get('2024/25')} leagues worldwide in 2024/25 "
                                                f"and {q5b['leagues_per_season'].get('2025/26')} in 2025/26"],
        ["Transfer-selection window", f"{sel[0]}–{sel[-1]} squads"],
        ["Players", f"{f0(q5b['transfer_players'])} with transfer histories; {f0(q5b['episode_players'])} in real "
                    "episodes"],
        ["Clubs", f"{f0(q5b['raw_clubs'])} in raw rows; {f0(q5b['episode_clubs'])} in real episodes; "
                  f"{f0(q5b['clubs_table'])} in the upstream `clubs` table"],
        ["Transfer episodes / loans", f"{f0(q1['real_transfer_episodes'])} / {f0(q2['loans'])}"],
        ["Club country known", f"{frac(q5b['episode_clubs_with_country'], q5b['episode_clubs'])} of episode clubs; both "
                               f"clubs for {frac(q5b['loans_both_countries_known'], q2['loans'])} of loans and "
                               f"{frac(q5b['episodes_both_countries_known'], q1['real_transfer_episodes'])} of episodes"],
        ["Capture dates", "; ".join(f"{c['source_season_file']} file: {f0(c['players'])} players, captured between "
                                    f"{c['last_realised_date']} and {c['first_scheduled_date']}"
                                    for c in q5b["capture"])],
    ], right=set())
    w("")
    er = q5b["earliest_rows"][:2]
    w(f"**Why 1993/94 appears.** It is simply the oldest youth registration among the selected recent players. The "
      f"two earliest rows are both dated {er[0]['transfer_date'][:10]}: {er[0]['name']} (born "
      f"{er[0]['date_of_birth'][:4]}) moved {er[0]['from_club_name']} → {er[0]['to_club_name']}, and {er[1]['name']} "
      f"(born {er[1]['date_of_birth'][:4]}) moved {er[1]['from_club_name']} → {er[1]['to_club_name']}; there is no "
      "date filter in the upstream acquisition code or its data model.")
    w("")
    w(f"**Why 2012/13 appears.** It is where the upstream project's separate competition, squad and match scraping "
      f"begins (first club game {q5b['club_games_first_date']}). Transfers are not selected by season, so 2012/13 is "
      f"no boundary in the transfer data: of the {f0(rowsq[0]['squad_players'])} players on covered squads in "
      f"{rowsq[0]['season']}, only {frac(rowsq[0]['with_transfer_history'], rowsq[0]['squad_players'])} have any "
      f"transfer row, all because a later season's acquisition fetched them. The {f0(q5b['players_last_season_2012_2021'])} "
      "players whose last covered season was 2012/13–2021/22 have "
      + ("no" if q5b["players_last_season_2012_2021_with_transfers"] == 0
         else f0(q5b["players_last_season_2012_2021_with_transfers"])) + " transfer rows.")
    w("")
    nt = q5b["transfer_players_not_on_2023_2025_squad"]
    w(f"**Why {sel[0]}–{sel[-1]} matters.** Those are the only squad seasons with a transfer acquisition file, so they "
      f"define the cohort. {f0(q5b['transfer_players_on_2023_2025_squad'])} of the {f0(q5b['transfer_players'])} "
      f"transfer players are on one of those squad files ({f0(q5b['transfer_players_only_via_2025_national_team'])} only through a "
      f"2025/26 national-team squad). The {'one exception' if len(nt) == 1 else f'{len(nt)} exceptions'}, player "
      f"{', '.join(str(x['player_id']) for x in nt)}, is in "
      f"the {nt[0]['history_from_file'] if nt else ''} transfer file but on no squad file after "
      f"{nt[0]['last_squad_season_file'] if nt else ''}; the squad list most likely changed after the transfer run. Even "
      "the selection seasons are incomplete:")
    w("")
    head = ["Squad season", "Squad players", "… national-team only", "transfers.json exists",
            "In that season's transfers.json", "History because fetched that season",
            "History only via another season's fetch", "No transfer history", "% with history"]
    W += md_table(head, [[r["season"], f0(r["squad_players"]), f0(r["national_team_only_players"]),
                          "yes" if r["transfers_json_exists"] else "no",
                          f0(r["squad_players_in_this_seasons_transfers_json"]),
                          f0(r["history_fetched_in_this_season"]),
                          f0(r["history_only_via_other_season_fetch"]), f0(r["without_transfer_history"]),
                          f"{r['pct_with_transfer_history']:.1f}%"] for r in rowsq], right=set(range(1, 9)) - {3})
    w("")
    w(f"The 2024/25 transfer file covers mainly the {len(q5b['leagues_from_2012'])} long-covered European leagues "
      f"({f0(q5b['league_2024_old_leagues_in_2024_file'])} of their {f0(q5b['league_2024_old_leagues_squad_players'])} "
      f"squad players); the leagues added in 2024/25 have {f0(q5b['league_2024_new_leagues_in_2024_file'])} of their "
      f"{f0(q5b['league_2024_new_leagues_squad_players'])} (`dataset_provenance_league_coverage.csv`). The code does "
      "not record why. In 2025/26 the counts coincide by chance: "
      f"{f0(q5b['club_squad_2025_without_history'])} club-squad players lack a history and "
      f"{f0(q5b['nt_only_2025_with_history'])} national-team-only players have one.")
    w("")
    w("**Important caveat.** Histories were captured once per player, at different times: " + "; ".join(
        f"{f0(c['players'])} players ({pc(c['rows'], q1['raw_rows'])} of rows) between {c['last_realised_date']} and "
        f"{c['first_scheduled_date']}" for c in q5b["capture"]) +
      f". For the {f0(cap['2023']['players'])} players captured in July 2024, nothing after "
      f"{cap['2023']['last_realised_date']} has happened in the data, so {last} is the latest complete season only for "
      "the others. Every season before 2023/24 holds only the earlier moves of players who were still active in "
      "2023–2026.")
    w("")

    # ---- Q5c
    o = q5c["observed"]
    sc = q5c["scenarios"]
    w("## Q5c. What would the approximate Stage 2 token workload be?")
    w("")
    w("**Question.** Roughly how many LLM tokens would researching a given scope take?")
    w("")
    w(f"**Short answer.** Observed: **{o['calls']} extraction calls** for **{o['researched_cases_reaching_extraction']} "
      f"cases** that reached extraction ({o['researched_cases']} researched), **{o['calls_per_family']:.2f} calls per "
      f"case**; per call, median **{f0(o['total_tokens_median'])}**, mean **{f0(o['total_tokens_mean'])}** and p90 "
      f"**{f0(o['total_tokens_p90'])}** total tokens. Everything beyond that is a scenario.")
    w("")
    w("**How we calculated it.** Observations come from the cached Stage 2 extraction runs with the current prompt. "
      "Input tokens are taken as total minus completion, because the provider's prompt-token field is unreliable. "
      "The estimate for a scope of N loans is")
    w("")
    w("`tokens = N × share reaching extraction × calls per case × tokens per call`")
    w("")
    w("and in Python `loan_scope.extraction_tokens(N, share, scenario)`. The three scenarios are *assumptions* built "
      "from the observed percentiles, not forecasts:")
    w("")
    W += md_table(["Scenario (assumption)", "Tokens per call", "Calls per case", "Tokens per case reaching extraction"], [
        ["LOW: 25th-percentile call, one call per case", f0(sc["LOW"]["tokens_per_call"]),
         f"{sc['LOW']['calls_per_case']:.2f}", f0(sc["LOW"]["tokens_per_case_reaching_extraction"])],
        ["BASE: mean call, observed calls per case", f0(sc["BASE"]["tokens_per_call"]),
         f"{sc['BASE']['calls_per_case']:.2f}", f0(sc["BASE"]["tokens_per_case_reaching_extraction"])],
        [f"HIGH: 90th-percentile call, calls per case with the earlier prompt ({q5c['earlier_prompt_calls']} calls, "
         f"{q5c['earlier_prompt_families']} cases, incl. development re-runs)", f0(sc["HIGH"]["tokens_per_call"]),
         f"{sc['HIGH']['calls_per_case']:.2f}", f0(sc["HIGH"]["tokens_per_case_reaching_extraction"])],
    ])
    w("")
    w("Extraction tokens by scope size N and the share of loans assumed to reach extraction (also an assumption):")
    w("")
    ns_ = list(SCOPE_SIZES) + [b["loan_episodes"]]
    trows = []
    for n in ns_:
        for s in ("LOW", "BASE", "HIGH"):
            trows.append([f0(n) + (" (worked example, Q5a)" if n == b["loan_episodes"] else "") if s == "LOW" else "",
                          s] + [tok(loan_scope.extraction_tokens(n, sh, s)) for sh in SHARES])
    W += md_table(["N loans", "Scenario"] + [f"{int(sh * 100)}% reach extraction" for sh in SHARES], trows,
                  right={2, 3, 4})
    w("")
    w("`token_workload_by_scope_size.csv` has the same grid.")
    w("")
    w("**Important caveats.**")
    w("")
    w("- These are **extraction tokens only**. Search and page-retrieval workload was not measured on an equivalent "
      "basis, because the researched cases ran from cached evidence.")
    w(f"- The researched cases were **selected**, because evidence was likely to exist; they are not a random sample "
      f"of loans. {o['researched_cases_reaching_extraction']} of {o['researched_cases']} reached extraction, but for "
      "ordinary loans the share is unobserved and probably much lower.")
    w("")

    # ---- closing
    w("## What this dataset is")
    w("")
    w(f"- Transfermarkt's full transfer-history lists for {f0(q5b['transfer_players'])} players selected from covered "
      f"squads in {sel[0]}–{sel[-1]}, with every move in their careers back to {q5b['earliest_row'][:4]}.")
    w(f"- {f0(q1['raw_rows'])} registration movements, which reduce to {f0(q1['real_transfer_episodes'])} real "
      f"transfer episodes, {f0(q2['loans'])} of them loans.")
    w("- A descriptive base for sizing and choosing a Stage 2 sample, with club countries only where the snapshot "
      "records them.")
    w("")
    w("## What this dataset is not")
    w("")
    w("- Not a census of any league or season, and not \"European transfers since 1993\".")
    w(f"- Not all transfers made by clubs in the covered competitions since {q5b['club_games_first_season']}: before "
      "2023/24 it holds only the earlier moves of players still active in 2023–2026.")
    w("- Not captured at a single date, and not a record of contract terms: labels and sequences describe "
      "registrations, not clauses.")
    w("")
    return "\n".join(W) + "\n"


# ---------------------------------------------------------------------------
# Cross-check earlier documents
# ---------------------------------------------------------------------------

def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^## (\d)\. ", text, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def expected_in_answer_sheet(N: dict) -> dict[str, list[tuple[str, str]]]:
    q1, q2, q3, q4, q5a, q5b, q5c = (N[k] for k in ("q1", "q2", "q3", "q4", "q5a", "q5b", "q5c"))
    m, ai, ri = q4["main"], q4["all_recorded_incl_scheduled"], q4["realised_all_seasons"]
    lab = q3["labels"]
    cell = lambda b, k: f"{f0(b[k + '_n'])} ({pc(b[k + '_n'], b['N'])})"
    o, sc = q5c["observed"], q5c["scenarios"]
    lo, hi = q4["not_plain_wilson95"]
    ex = [("both loans", f0(q5a["both"]["loan_episodes"])), ("either loans", f0(q5a["either"]["loan_episodes"]))]
    return {
        "1": [("raw rows", f0(q1["raw_rows"])), ("episodes", f0(q1["real_transfer_episodes"])),
              ("episodes % of rows", pc(q1["real_transfer_episodes"], q1["raw_rows"])),
              ("internal", f0(q1["internal_registration"])), ("placeholder", f0(q1["no_counterpart_placeholder"])),
              ("draft", f0(q1["draft"])), ("folded", f0(q1["returns_folded_into_loan"])),
              ("unmatched", f0(q1["returns_unmatched_set_aside"])),
              ("senior-only", frac(q1["senior_only_episodes"], q1["real_transfer_episodes"])),
              ("youth episodes", f0(q1["episodes_involving_youth_or_reserve_side"])),
              ("conversions", f0(q1["permanent_episodes_after_same_borrower_conversion"])),
              ("broad", f0(q1["broad_episodes_loan_plus_conversion_as_one"]))],
        "2": [("loans", frac(q2["loans"], q2["real_transfer_episodes"])),
              ("senior loans", frac(q2["senior_only_loans"], q2["senior_only_episodes"])),
              ("matched", f"{f0(q2['matched_return'])} ({pc(q2['matched_return'], q2['loans'])})"),
              ("other movement", f"{q2['ended_by_another_movement']} were ended"),
              ("no ending", f"{q2['no_observed_ending']} have no ending")],
        "3": [("non-plain", frac(q3["not_plain"], q3["loans_with_observed_ending"])),
              ("plain", f0(q3["plain_end_of_loan"])), ("fee-bearing", f"{q3['fee_bearing_returns']}"),
              ("dash", f"`-` ({lab.get('- (no fee shown)', 0)})"),
              ("free", f"`free transfer` ({lab.get('free transfer', 0)})"),
              ("?", f"`?` ({lab.get('? (undisclosed)', 0)})"),
              ("amount", f"a fee amount ({lab.get('explicit fee amount', 0)})"),
              ("loan transfer", f"`loan transfer` ({lab.get('loan transfer', 0)})"),
              ("median", eur(q3["fee_median"])), ("p25", eur(q3["fee_p25"])), ("p75", eur(q3["fee_p75"])),
              ("max", eur(q3["fee_max"])),
              ("secondary", frac(q3["nonordinary_sequence"], q2["loans"])),
              ("secondary plain", f0(q3["nonordinary_sequence_ended_by_plain_end_of_loan"])),
              ("2025/26 scheduled", f"{f0(q3['loans_2025_26_scheduled_end'])} of the {f0(q3['loans_2025_26'])}")],
        "4": [("headline", pc(m["all"]["jun30_or_dec31_n"], m["all"]["N"])), ("N", f0(m["all"]["N"])),
              ("uefa", f0(q4["loans_both_uefa"])),
              ("unmapped", frac(q4["loans_with_unmapped_club"], q4["all_loans"]).split(" = ")[0]),
              ("jun30", cell(m["all"], "jun30")), ("dec31", cell(m["all"], "dec31")),
              ("either", cell(m["all"], "jun30_or_dec31")), ("other", cell(m["all"], "other_dates")),
              ("plain N", f0(m["plain"]["N"])), ("not plain either", cell(m["not_plain"], "jun30_or_dec31")),
              ("fee either", cell(m["fee"], "jun30_or_dec31")),
              ("scheduled", f0(q4["uefa_scheduled_endings"])),
              ("scheduled share", pc(q4["uefa_scheduled_on_jun30_or_dec31"], q4["uefa_scheduled_endings"])),
              ("2025/26 realised", f"early exits ({f0(q4['realised_2025_26'])}"),
              ("incl scheduled either", cell(ai["all"], "jun30_or_dec31")),
              ("realised all seasons", frac(ri["all"]["jun30_or_dec31_n"], ri["all"]["N"])),
              ("realised all non-plain", frac(ri["not_plain"]["jun30_or_dec31_n"], ri["not_plain"]["N"])),
              ("may31", frac(q4["may31_n"], m["all"]["N"])),
              ("may31 uk", f0(q4["may31_from_england_or_scotland"])),
              ("england", f"{f0(q4['lender_england']['jun30_or_dec31_n'])} / {f0(q4['lender_england']['N'])}"),
              ("scotland", f"{f0(q4['lender_scotland']['jun30_or_dec31_n'])} / {f0(q4['lender_scotland']['N'])}"),
              ("italy", f"{f0(q4['lender_italy']['jun30_or_dec31_n'])} / {f0(q4['lender_italy']['N'])}"),
              ("others", f"{f0(q4['lender_all_others']['jun30_or_dec31_n'])} / {f0(q4['lender_all_others']['N'])}"),
              ("interval", f"{100 * lo:.0f}–{100 * hi:.0f}%"),
              ("through 2023/24", frac(q4["realised_through_2023_24"]["all"]["jun30_or_dec31_n"],
                                       q4["realised_through_2023_24"]["all"]["N"])),
              ("through 2023/24 non-plain", frac(q4["realised_through_2023_24"]["not_plain"]["jun30_or_dec31_n"],
                                                 q4["realised_through_2023_24"]["not_plain"]["N"]))],
        "5": ex + [("both episodes", f0(q5a["both"]["real_transfer_episodes"])),
                   ("either episodes", f0(q5a["either"]["real_transfer_episodes"])),
                   ("realised both/either", f"{f0(q5a['both_realised_only_loans'])} and "
                                            f"{f0(q5a['either_realised_only_loans'])}"),
                   ("calls", f"{o['calls']} extraction calls across {o['families']}"),
                   ("calls/case", f"{o['calls_per_family']:.2f}"),
                   ("median", f0(o["total_tokens_median"])), ("mean", f0(o["total_tokens_mean"])),
                   ("p90", f0(o["total_tokens_p90"])),
                   ("researched", f"{o['researched_cases_reaching_extraction']} of the {o['researched_cases']}"),
                   ("LOW tpc", f0(sc["LOW"]["tokens_per_call"])), ("HIGH cpc", f"{sc['HIGH']['calls_per_case']:.2f}"),
                   ("BASE 50% example", tok(loan_scope.extraction_tokens(q5a["both"]["loan_episodes"], .5, "BASE")))],
        "6": [("rows", f0(q1["raw_rows"])), ("players", f0(q5b["transfer_players"])),
              ("old players", f0(q5b["players_last_season_2012_2021"])),
              ("2012 squad", f"{f0(q5b['squad_rows'][0]['squad_players'])} players"),
              ("2012 with", f"{f0(q5b['squad_rows'][0]['with_transfer_history'])} "
                            f"({q5b['squad_rows'][0]['pct_with_transfer_history']:.1f}%)"),
              ("episode players", f0(q5b["episode_players"])), ("raw clubs", f0(q5b["raw_clubs"])),
              ("episode clubs", f0(q5b["episode_clubs"])),
              ("clubs mapped", frac(q5b["episode_clubs_with_country"], q5b["episode_clubs"])),
              ("loans mapped", frac(q5b["loans_both_countries_known"], q2["loans"])),
              ("episodes mapped", frac(q5b["episodes_both_countries_known"], q1["real_transfer_episodes"])),
              ("on a squad", f"{f0(q5b['transfer_players_on_2023_2025_squad'])} of the {f0(q5b['transfer_players'])}")]
             + [(f"capture {c['source_season_file']}", f0(c["players"])) for c in q5b["capture"]],
    }


def expected_in_provenance_doc(N: dict) -> list[tuple[str, str]]:
    q1, q2, q5b = N["q1"], N["q2"], N["q5b"]
    out = [("rows", f0(q1["raw_rows"])), ("players", f0(q5b["transfer_players"])),
           ("episodes", f0(q1["real_transfer_episodes"])), ("loans", f0(q2["loans"])),
           ("episode players", f0(q5b["episode_players"])), ("raw clubs", f0(q5b["raw_clubs"])),
           ("episode clubs", f0(q5b["episode_clubs"])), ("clubs table", f0(q5b["clubs_table"])),
           ("clubs mapped", frac(q5b["episode_clubs_with_country"], q5b["episode_clubs"])),
           ("loans mapped", f"{f0(q5b['loans_both_countries_known'])} / {f0(q2['loans'])}"),
           ("old players", f0(q5b["players_last_season_2012_2021"])),
           ("only via national team", f0(q5b["transfer_players_only_via_2025_national_team"])),
           ("on a 2023-2025 squad", f0(q5b["transfer_players_on_2023_2025_squad"])),
           ("pre-2012 players", f0(q5b["pre_2012_players"]))]
    for c in q5b["capture"]:
        out += [(f"capture {c['source_season_file']} players", f0(c["players"])),
                (f"capture {c['source_season_file']} rows", f0(c["rows"]))]
    for r in q5b["squad_rows"]:
        out.append((f"squad {r['season']}", f"| {r['season']} | {f0(r['squad_players'])} |"))
    return out


def crosscheck(N: dict) -> pd.DataFrame:
    rows = []
    if ANSWERS_MD.exists():
        secs = _sections(ANSWERS_MD.read_text())
        for q, items in expected_in_answer_sheet(N).items():
            for label, s in items:
                rows.append({"document": ANSWERS_MD.name, "section": f"Q{q}", "check": label, "expected": s,
                             "present": s in secs.get(q, "")})
    if PROVENANCE_MD.exists():
        text = PROVENANCE_MD.read_text()
        for label, s in expected_in_provenance_doc(N):
            rows.append({"document": PROVENANCE_MD.name, "section": "all", "check": label, "expected": s,
                         "present": s in text})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def main() -> dict:
    res = compute()
    N = res["N"]
    res["cube"].to_csv(OUT / "loan_scope_cube.csv", index=False)
    res["grid"].to_csv(OUT / "token_workload_by_scope_size.csv", index=False)
    FINAL_MD.write_text(render(N))
    (OUT / "daniel_dataset_scope_final_numbers.json").write_text(json.dumps(_jsonable(N), indent=2, default=str))
    cc = crosscheck(N)
    cc.to_csv(OUT / "daniel_dataset_scope_crosscheck.csv", index=False)
    res["crosscheck"] = cc
    return res


if __name__ == "__main__":
    r = main()
    cc = r["crosscheck"]
    print(f"wrote {FINAL_MD.name}; crosscheck: {int(cc.present.sum())}/{len(cc)} expected values present")
    for (doc, sec), g in cc.groupby(["document", "section"]):
        miss = g[~g.present]
        print(f"  {doc} {sec}: {'all present' if miss.empty else 'MISSING ' + '; '.join(miss.check + ' ' + miss.expected)}")
