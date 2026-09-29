"""How big is a candidate Stage 2 sample?

    python -m src.analysis.loan_scope --countries Italy,England,Spain \\
        --seasons 2021/22,2022/23,2023/24 --country-mode both

or from Python:

    from src.analysis.loan_scope import scope
    scope(countries=["Italy"], seasons=["2023/24"], country_mode="both")

Counts are over real transfer episodes (see `loan_episodes`), so a loan and
its return are one episode. Seasons refer to the season of the outbound
move. Clubs whose country the upstream snapshot does not record are never
guessed: with `--country-mode both` an episode with an unmapped club is out
of scope, with `either` it is in scope only if the other club qualifies.

The workload estimate multiplies observed per-call token usage from the
Stage 2 extraction cache by the number of loans in scope and by an explicit
share of loans assumed to reach an extraction call. That share is NOT
observed for ordinary loans - every family researched so far was chosen
because evidence was likely - so it is a parameter, shown in the output.
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
           confederation: str | None = None) -> pd.DataFrame:
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
    return ep[mask]


def scope(countries=None, seasons=None, country_mode: str = "both",
          confederation: str | None = None, share_reaching_llm=(0.1, 0.5, 0.9)) -> dict:
    ep = select(countries, seasons, country_mode, confederation)
    loans = ep[ep.episode_type == "loan"]
    e = loans.economic_ending
    out = {
        "filters": {"countries": countries, "seasons": seasons, "country_mode": country_mode,
                    "confederation": confederation},
        "real_transfer_episodes": int(len(ep)),
        "loan_episodes": int(len(loans)),
        "ordinary_loan_endings": int((e == "ordinary_end_of_loan").sum()),
        "nonstandard_loan_endings": int(e.isin(NONSTANDARD).sum()),
        "fee_bearing_returns": int(loans.ending_fee_on_return_eur.notna().sum()),
        "unresolved_or_open_loans": int((e == "unresolved").sum()),
        "raw_label_not_plain_end_of_loan": int((loans.raw_label_class != "End of loan").sum()),
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
    return {
        "cache": cache_name, "calls": int(len(t)), "families": int(t.family.nunique()),
        "calls_per_family": round(len(t) / t.family.nunique(), 3),
        "total_tokens_p25": float(q(.25)), "total_tokens_median": float(q(.5)),
        "total_tokens_mean": float(t.total.mean()), "total_tokens_p90": float(q(.9)),
        "input_tokens_mean": float(t.input.mean()), "completion_tokens_mean": float(t.completion.mean()),
        "input_tokens_median": float(t.input.median()), "completion_tokens_median": float(t.completion.median()),
        "cost_usd_mean_per_call": float(t.cost_usd.mean()),
        "cost_usd_p90_per_call": float(t.cost_usd.quantile(.9)),
    }


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
    scen = {
        "LOW": (cur["total_tokens_p25"], 1.0),
        "BASE": (cur["total_tokens_mean"], cur["calls_per_family"]),
        "HIGH": (cur["total_tokens_p90"], old.get("calls_per_family", cur["calls_per_family"])),
    }
    grid = {}
    for name, (tok, cpf) in scen.items():
        for sh in shares:
            for label, n in (("all_loans", n_loans), ("nonstandard_only", n_nonstandard)):
                grid[f"{name}|share={sh}|{label}"] = round(n * sh * cpf * tok)
    return {
        "available": True,
        "formula": "tokens = loans x share_reaching_llm x calls_per_family x tokens_per_call",
        "observed": {"current_prompt": cur, "earlier_prompt": old},
        "scenarios": {k: {"tokens_per_call": v[0], "calls_per_family": v[1]} for k, v in scen.items()},
        "share_reaching_llm_note": (
            "0.90 of the 50 families researched so far reached an extraction call, but "
            "they were chosen because evidence was likely; for ordinary loans the share "
            "is unobserved and probably much lower."),
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
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args(argv)
    res = scope(a.countries.split(",") if a.countries else None,
                a.seasons.split(",") if a.seasons else None,
                a.country_mode, a.confederation)
    if a.json:
        print(json.dumps(res, indent=2, default=str))
        return res
    print(f"filters: {res['filters']}")
    for k in ("real_transfer_episodes", "loan_episodes", "ordinary_loan_endings",
              "nonstandard_loan_endings", "fee_bearing_returns", "unresolved_or_open_loans"):
        print(f"  {k:28s} {res[k]:>9,}")
    w = res["workload"]
    if w.get("available"):
        print("\nextraction tokens (loans x share x calls/family x tokens/call):")
        for name in ("LOW", "BASE", "HIGH"):
            cells = [f"share {sh}: {_fmt(w['grid'][f'{name}|share={sh}|all_loans'])}"
                     for sh in (0.1, 0.5, 0.9)]
            print(f"  {name:5s} " + "   ".join(cells))
        print(f"  note: {w['share_reaching_llm_note']}")
    return res


if __name__ == "__main__":
    main()
