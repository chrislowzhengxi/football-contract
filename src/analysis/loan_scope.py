"""How big is a candidate Stage 2 sample?

    python -m src.analysis.loan_scope --countries Italy,England,Spain \\
        --seasons 2021/22,2022/23,2023/24 --country-mode both

or from Python:

    from src.analysis.loan_scope import scope
    scope(countries=["Italy"], seasons=["2023/24"], country_mode="both")

Counts are over real transfer episodes (see `loan_episodes`), so a loan and
its return are one episode. Seasons refer to the season of the outbound
move.

Country filters use only clubs whose country the upstream snapshot records;
unmapped clubs are never guessed:

  * `both`   - both clubs are mapped AND both are in the selected countries;
  * `either` - at least one club is mapped and in the selected countries.

Neither is a complete census. `both` misses loans whose other club is
unmapped; `either` also admits cross-border loans, and it still misses a
loan whose two clubs are both unmapped. So neither is an exact lower or
upper bound on the true count.

Two optional filters, both off by default so existing counts are unchanged:

  * `--realised-only` drops episodes Transfermarkt flags as scheduled
    (future) moves, and loans whose recorded ending is a scheduled return.
    Loans with no ending at all stay in and are counted as unresolved.
  * `--through-season 2024/25` keeps outbound moves up to that season.

The workload estimate multiplies observed per-call EXTRACTION token usage
from the Stage 2 cache by the number of loans in scope and by an explicit
share of loans assumed to reach an extraction call. That share is NOT
observed for ordinary loans - every family researched so far was chosen
because evidence was likely - so it is a parameter, shown in the output.
Search / evidence-discovery cost is not measured and not included.
"""
from __future__ import annotations

import argparse
import glob
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .loan_episodes import (ROOT, build_universe, club_geography, load_canonical,
                            normalise_season_arg)

EXTRACTION_CACHE = ROOT / "data" / "outputs" / "contract_research" / "stage2_cache"
# Every case researched so far, whether or not it reached an extraction call.
RESEARCHED_CASES_LOG = ROOT / "data" / "outputs" / "rebuild" / "stage2" / "daniel_20_selection_log.csv"
CURRENT_PROMPT_CACHE = "extraction_v4i"
EARLIER_PROMPT_CACHE = "extraction_v4"     # before the identity gate; re-runs inflate calls/family

NONSTANDARD = ("fee_bearing_return", "early_termination",
               "purchase_option_or_permanent_conversion", "immediate_follow_on_transfer",
               "third_party_sale_related", "other_nonstandard")


@lru_cache(maxsize=1)
def universe():
    df = load_canonical()
    return build_universe(df, club_geography())


def _in_countries(frame: pd.DataFrame, countries, mode: str) -> pd.Series:
    if not countries:
        return pd.Series(True, index=frame.index)
    cs = {c.strip().casefold() for c in countries}
    f = frame.from_country.fillna("").str.casefold().isin(cs)
    t = frame.to_country.fillna("").str.casefold().isin(cs)
    if mode == "both":
        return f & t
    if mode == "either":
        return f | t
    raise ValueError("country_mode must be 'both' or 'either'")


def select(countries=None, seasons=None, country_mode: str = "both",
           confederation: str | None = None, realised_only: bool = False,
           through_season: str | None = None) -> pd.DataFrame:
    """The real transfer episodes in scope."""
    ep = universe().episodes
    mask = _in_countries(ep, countries, country_mode)
    if confederation:
        if country_mode == "both":
            mask &= (ep.from_confederation == confederation) & (ep.to_confederation == confederation)
        else:
            mask &= (ep.from_confederation == confederation) | (ep.to_confederation == confederation)
    if seasons:
        wanted = {normalise_season_arg(s) for s in seasons}
        mask &= ep.season.isin(wanted)
    if through_season:
        mask &= ep.season <= normalise_season_arg(through_season)
    if realised_only:
        mask &= ~ep.transfermarkt_future_transfer.astype(bool) & ~ep.terminal_scheduled_future
    return ep[mask]


def scope(countries=None, seasons=None, country_mode: str = "both",
          confederation: str | None = None, share_reaching_llm=(0.1, 0.5, 0.9),
          realised_only: bool = False, through_season: str | None = None) -> dict:
    ep = select(countries, seasons, country_mode, confederation, realised_only, through_season)
    loans = ep[ep.episode_type == "loan"]
    e = loans.economic_ending
    ended = loans.terminal_event_id.notna()
    out = {
        "filters": {"countries": countries, "seasons": seasons, "country_mode": country_mode,
                    "confederation": confederation, "realised_only": realised_only,
                    "through_season": through_season},
        "real_transfer_episodes": int(len(ep)),
        "loan_episodes": int(len(loans)),
        "loans_with_recorded_ending": int(ended.sum()),
        "ordinary_loan_endings": int((e == "ordinary_end_of_loan").sum()),
        "nonstandard_loan_endings": int(e.isin(NONSTANDARD).sum()),
        "fee_bearing_returns": int(loans.ending_fee_on_return_eur.notna().sum()),
        "unresolved_or_open_loans": int((e == "unresolved").sum()),
        # Of loans with an ending: the ending row's label is not a plain "End of loan".
        "raw_label_not_plain_end_of_loan": int((ended & (loans.raw_label_class != "End of loan")).sum()),
        "by_economic_ending": {k: int(v) for k, v in e.value_counts().items()},
    }
    out["workload"] = workload(out["loan_episodes"], out["nonstandard_loan_endings"],
                               share_reaching_llm)
    return out


# ---------------------------------------------------------------------------
# Token workload
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def token_profile(cache_name: str = CURRENT_PROMPT_CACHE) -> dict:
    """Observed per-call usage from cached Stage 2 extractions.

    Parley's `prompt_tokens` field is unreliable (it reports 3 on a call
    whose total is 13,092), so input is taken as total - completion.
    """
    rows = []
    for f in glob.glob(str(EXTRACTION_CACHE / cache_name / "*.json")):
        d = json.loads(Path(f).read_text())
        md = d.get("provider_metadata") or {}
        if not md.get("total_tokens"):
            continue
        try:
            cost = float(md.get("parley_cost_header"))
        except (TypeError, ValueError):
            cost = float("nan")
        rows.append({"family": d.get("anchor_event_id"), "total": md["total_tokens"],
                     "completion": md.get("completion_tokens") or 0, "cost_usd": cost})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"calls": 0}
    t["input"] = t.total - t.completion
    q = t.total.quantile
    researched = (set(pd.read_csv(RESEARCHED_CASES_LOG).event_id)
                  if RESEARCHED_CASES_LOG.exists() else set())
    return {
        "cache": cache_name, "calls": int(len(t)), "families": int(t.family.nunique()),
        "researched_cases": len(researched),
        "researched_cases_reaching_extraction": len(researched & set(t.family)),
        "calls_per_family": round(len(t) / t.family.nunique(), 3),
        "total_tokens_p25": float(q(.25)), "total_tokens_median": float(q(.5)),
        "total_tokens_mean": float(t.total.mean()), "total_tokens_p90": float(q(.9)),
        "input_tokens_mean": float(t.input.mean()), "completion_tokens_mean": float(t.completion.mean()),
        "input_tokens_median": float(t.input.median()), "completion_tokens_median": float(t.completion.median()),
        "cost_usd_mean_per_call": float(t.cost_usd.mean()),
        "cost_usd_p90_per_call": float(t.cost_usd.quantile(.9)),
    }


def scenarios() -> dict[str, tuple[float, float]]:
    """Scenario name -> (tokens per call, calls per case). Assumptions built
    from observed percentiles; not forecasts."""
    cur, old = token_profile(CURRENT_PROMPT_CACHE), token_profile(EARLIER_PROMPT_CACHE)
    if not cur.get("calls"):
        return {}
    return {
        "LOW": (cur["total_tokens_p25"], 1.0),
        "BASE": (cur["total_tokens_mean"], cur["calls_per_family"]),
        "HIGH": (cur["total_tokens_p90"], old.get("calls_per_family", cur["calls_per_family"])),
    }


def extraction_tokens(n_cases: int, share_reaching_extraction: float, scenario: str = "BASE") -> int:
    """Extraction tokens for researching `n_cases` loans under one scenario."""
    tok, cpf = scenarios()[scenario]
    return round(n_cases * share_reaching_extraction * cpf * tok)


def workload(n_loans: int, n_nonstandard: int, shares=(0.1, 0.5, 0.9)) -> dict:
    """LOW / BASE / HIGH extraction tokens for researching `n_loans` loans.

        tokens = loans x share_reaching_llm x calls_per_family x tokens_per_call

    LOW  = p25 tokens per call x 1.0 call per family
    BASE = mean tokens per call x observed calls per family (current prompt)
    HIGH = p90 tokens per call x calls per family seen with the earlier prompt,
           which includes re-extraction during development
    Evidence acquisition (search / page retrieval) is not included: the 50
    families researched so far ran entirely from cache.
    """
    cur, old = token_profile(CURRENT_PROMPT_CACHE), token_profile(EARLIER_PROMPT_CACHE)
    if not cur.get("calls"):
        return {"available": False,
                "formula": "tokens = loans x share_reaching_llm x calls_per_family x tokens_per_call"}
    scen = scenarios()
    grid = {}
    for name in scen:
        for sh in shares:
            for label, n in (("all_loans", n_loans), ("nonstandard_only", n_nonstandard)):
                grid[f"{name}|share={sh}|{label}"] = extraction_tokens(n, sh, name)
    return {
        "available": True,
        "formula": "tokens = loans x share_reaching_llm x calls_per_family x tokens_per_call",
        "observed": {"current_prompt": cur, "earlier_prompt": old},
        "scenarios": {k: {"tokens_per_call": v[0], "calls_per_family": v[1]} for k, v in scen.items()},
        "share_reaching_llm_note": (
            f"{cur['researched_cases_reaching_extraction']} of the {cur['researched_cases']} cases "
            "researched so far reached an extraction call, but they were chosen because evidence "
            "was likely; for ordinary loans the share is unobserved and probably much lower."),
        "grid": grid,
    }


def _fmt(n: float) -> str:
    return f"{n / 1e6:,.1f}M" if n >= 1e6 else f"{n:,.0f}"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--countries", help="comma-separated, e.g. Italy,England,Spain")
    ap.add_argument("--seasons", help="comma-separated, e.g. 2021/22,2022/23")
    ap.add_argument("--country-mode", choices=("both", "either"), default="both")
    ap.add_argument("--confederation", help="e.g. UEFA")
    ap.add_argument("--realised-only", action="store_true",
                    help="drop scheduled (future-flagged) moves and loans whose ending is scheduled")
    ap.add_argument("--through-season", help="keep outbound moves up to this season, e.g. 2024/25")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args(argv)
    res = scope(a.countries.split(",") if a.countries else None,
                a.seasons.split(",") if a.seasons else None,
                a.country_mode, a.confederation,
                realised_only=a.realised_only, through_season=a.through_season)
    if a.json:
        print(json.dumps(res, indent=2, default=str))
        return res
    print(f"filters: {res['filters']}")
    for k in ("real_transfer_episodes", "loan_episodes", "loans_with_recorded_ending",
              "ordinary_loan_endings", "raw_label_not_plain_end_of_loan", "nonstandard_loan_endings",
              "fee_bearing_returns", "unresolved_or_open_loans"):
        print(f"  {k:32s} {res[k]:>9,}")
    w = res["workload"]
    if w.get("available"):
        print("\nextraction tokens (loans x share x calls/family x tokens/call):")
        for name in ("LOW", "BASE", "HIGH"):
            cells = [f"share {sh}: {_fmt(w['grid'][f'{name}|share={sh}|all_loans'])}"
                     for sh in (0.1, 0.5, 0.9)]
            print(f"  {name:5s} " + "   ".join(cells))
        print(f"  note: {w['share_reaching_llm_note']}")
        print("  note: extraction tokens only; search / evidence-discovery cost is not measured.")
    return res


if __name__ == "__main__":
    main()
