"""Stage 2 pilot runner: search -> retrieve -> gate -> extract -> fuse.

    python -m src.stage2.run_pilot --dry-run            # no network, no cost
    python -m src.stage2.run_pilot --category fee_bearing_loan_return   # 8-event smoke test
    python -m src.stage2.run_pilot                      # full 26-event pilot

Credentials are read from the environment, or from `.pytest_cache/.env` - the
same gitignored location the earlier pilot used. Only TAVILY_API_KEY and
PARLEY_API_KEY are ever read from that file, and no value is ever logged.

Stage 1 facts and Stage 2 observations are kept in separate columns throughout
and combined only in the final fused output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..stage1.backbone import REBUILD_DIR
from .queries import build_queries
from .sources import can_establish_alone, classify, domain_of, tier_of

PILOT_CSV = REBUILD_DIR / "stage2_pilot_selection.csv"
OUT_DIR = REBUILD_DIR / "stage2"
CACHE_DIR = Path("data/outputs/contract_research/stage2_cache")
ENV_FILE = Path(".pytest_cache/.env")
ALLOWED_ENV_KEYS = {"TAVILY_API_KEY", "PARLEY_API_KEY", "PARLEY_MODEL"}

# The 11 Stage 2 target fields. Stage 1C settles everything else.
TARGET_FIELDS = [
    "purchase_option", "purchase_obligation", "obligation_trigger",
    "add_ons", "sell_on", "buy_back", "parent_contract_expiry",
    "years_remaining_at_transfer", "release_or_purchase_clause_exercised",
    "fee_on_return_interpretation", "deal_summary",
]

# A page only reaches the LLM if it contains contract language. This is the
# main cost control: events whose evidence says nothing cost zero extraction.
GATE_TERMS = [
    "option to buy", "purchase option", "obligation to buy", "obligation",
    "add-on", "add on", "bonus", "sell-on", "sell on", "buy-back", "buy back",
    "release clause", "clause", "per cent", "percent", "%", "contract until",
    "signed until", "loan fee", "exercised", "triggered", "appearances",
    "riscatto", "obbligo", "diritto", "bonus", "clausola", "percentuale",
    "opción de compra", "obligación", "cláusula", "opção de compra",
    "option d'achat", "obligation d'achat", "clause", "Kaufoption",
    "Kaufpflicht", "Ausstiegsklausel", "satın alma opsiyonu",
    "zorunlu satın alma", "bonservis", "opsiyon",
]


def load_credentials() -> dict[str, str]:
    """Environment first, then the gitignored .pytest_cache/.env. Values are
    never returned to the caller for logging - only presence is reported."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in ALLOWED_ENV_KEYS and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")
    return {k: os.environ[k] for k in ALLOWED_ENV_KEYS if os.environ.get(k)}


def missing_credentials() -> list[str]:
    have = load_credentials()
    return [k for k in ("TAVILY_API_KEY", "PARLEY_API_KEY") if k not in have]


@dataclass
class Metrics:
    events_attempted: int = 0
    events_completed: int = 0
    search_queries_issued: int = 0
    search_cache_hits: int = 0
    search_results_returned: int = 0
    pages_retrieved: int = 0
    page_cache_hits: int = 0
    pages_relevant: int = 0
    pages_passing_gate: int = 0
    llm_extraction_calls: int = 0
    llm_skipped_insufficient_evidence: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model_used: str = ""
    errors: list[str] = field(default_factory=list)


def _cache_path(kind: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    path = CACHE_DIR / kind
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{digest}.json"


def cached_search(query: str, provider, metrics: Metrics, dry_run: bool) -> list[dict]:
    path = _cache_path("search", query)
    if path.exists():
        metrics.search_cache_hits += 1
        return json.loads(path.read_text()).get("results", [])
    if dry_run:
        return []
    payload = provider._search(query)
    metrics.search_queries_issued += 1
    results = payload.get("results", []) if isinstance(payload, dict) else []
    path.write_text(json.dumps({"query": query, "results": results}, ensure_ascii=False))
    metrics.search_results_returned += len(results)
    time.sleep(0.4)          # courtesy pacing
    return results


def cached_page(url: str, fetcher, metrics: Metrics, dry_run: bool) -> str | None:
    path = _cache_path("page", url)
    if path.exists():
        metrics.page_cache_hits += 1
        return json.loads(path.read_text()).get("text")
    if dry_run:
        return None
    try:
        result = fetcher.fetch(url)
        text = getattr(result, "text", None) or getattr(result, "visible_text", None)
    except Exception as exc:                       # noqa: BLE001
        metrics.errors.append(f"retrieval failed {domain_of(url)}: {type(exc).__name__}")
        return None
    metrics.pages_retrieved += 1
    path.write_text(json.dumps({"url": url, "text": text}, ensure_ascii=False))
    time.sleep(0.4)
    return text


def event_matches(text: str, event: dict) -> bool:
    """Guard against extracting from a page about a different transfer."""
    if not text:
        return False
    low = text.lower()
    surname = str(event.get("player_name", "")).split()[-1].lower()
    clubs = [str(event.get("from_club_name", "")).lower().split()[0],
             str(event.get("to_club_name", "")).lower().split()[0]]
    return surname in low and any(c and c in low for c in clubs)


def passes_gate(text: str) -> tuple[bool, list[str]]:
    low = (text or "").lower()
    hits = sorted({t for t in GATE_TERMS if t.lower() in low})
    return (len(hits) >= 2, hits)


def research_event(event: dict, provider, fetcher, extractor, metrics: Metrics,
                   dry_run: bool) -> dict:
    """One event end to end. Returns a record; never raises for data reasons."""
    event_id = event["event_id"]
    queries = build_queries(event)
    evidence, planned = [], []
    for query in queries:
        planned.append(query.as_dict())
        for result in cached_search(query.query, provider, metrics, dry_run)[:5]:
            url = result.get("url", "")
            source_class = classify(url, event.get("from_club_name"), event.get("to_club_name"))
            evidence.append({
                "event_id": event_id, "url": url, "domain": domain_of(url),
                "title": result.get("title"), "snippet": result.get("content"),
                "source_class": source_class, "tier": tier_of(source_class),
                "from_query": query.query, "target_field": query.target_field,
            })

    # Retrieve only Tier 1 and Tier 2, best tier first, capped per event.
    seen, retrieved = set(), []
    for record in sorted(evidence, key=lambda r: (r["tier"], r["domain"])):
        if record["tier"] >= 3 or record["url"] in seen or len(retrieved) >= 6:
            continue
        seen.add(record["url"])
        text = cached_page(record["url"], fetcher, metrics, dry_run)
        if not text:
            continue
        record["retrieved"] = True
        record["matches_event"] = event_matches(text, event)
        gated, hits = passes_gate(text)
        record["passes_gate"] = gated
        record["gate_terms"] = hits[:12]
        record["text_chars"] = len(text)
        record["text"] = text[:20000]
        if record["matches_event"]:
            metrics.pages_relevant += 1
        if gated and record["matches_event"]:
            metrics.pages_passing_gate += 1
        retrieved.append(record)

    usable = [r for r in retrieved if r.get("passes_gate") and r.get("matches_event")]
    record = {
        "event_id": event_id,
        "player_id": int(event["player_id"]),
        "candidate_category": event.get("candidate_category"),
        "queries_planned": planned,
        "evidence_found": len(evidence),
        "pages_retrieved": len(retrieved),
        "pages_usable": len(usable),
        "tier1_usable": sum(1 for r in usable if can_establish_alone(r["source_class"])),
        "source_classes_seen": sorted({r["source_class"] for r in evidence}),
        "extraction_status": None,
        "result": None,
    }
    if not usable:
        metrics.llm_skipped_insufficient_evidence += 1
        record["extraction_status"] = "insufficient_evidence"
        return record
    if dry_run:
        record["extraction_status"] = "would_extract"
        return record
    try:
        result = extractor(event, usable)
        metrics.llm_extraction_calls += 1
        record["extraction_status"] = "extracted"
        record["result"] = result
    except Exception as exc:                        # noqa: BLE001
        metrics.errors.append(f"{event_id} extraction: {type(exc).__name__}: {exc}")
        record["extraction_status"] = "extraction_failed"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2 pilot")
    parser.add_argument("--pilot", default=str(PILOT_CSV))
    parser.add_argument("--category", help="restrict to one candidate_category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="plan queries and read caches only; no network, no cost")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_csv(args.pilot)
    if args.category:
        pilot = pilot[pilot.candidate_category == args.category]
    if args.limit:
        pilot = pilot.head(args.limit)

    missing = missing_credentials()
    if missing and not args.dry_run:
        print("STOP - missing credentials: " + ", ".join(missing))
        print(f"Set them in the environment, or put them in {ENV_FILE} as KEY=value.")
        print("Re-run: python -m src.stage2.run_pilot --category fee_bearing_loan_return")
        raise SystemExit(2)

    provider = fetcher = extractor = None
    if not args.dry_run:
        from ..official_page_retrieval import OfficialPageFetcher
        from ..source_discovery import TavilySearchProvider
        provider = TavilySearchProvider.from_environment()
        fetcher = OfficialPageFetcher()
        extractor = _build_extractor()

    metrics = Metrics(model_used=os.environ.get("PARLEY_MODEL", "bedrock/claude-haiku-4-5"))
    records = []
    for _, row in pilot.iterrows():
        metrics.events_attempted += 1
        record = research_event(row.to_dict(), provider, fetcher, extractor, metrics, args.dry_run)
        if record["extraction_status"] in ("extracted", "would_extract"):
            metrics.events_completed += 1
        records.append(record)
        print(f"  {record['event_id'][:14]}  {str(row.player_name)[:22]:24s} "
              f"ev={record['evidence_found']:>3} pages={record['pages_retrieved']} "
              f"usable={record['pages_usable']} -> {record['extraction_status']}")

    suffix = ("_dryrun" if args.dry_run else "") + (f"_{args.category}" if args.category else "")
    (output_dir / f"stage2_research_records{suffix}.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str))
    (output_dir / f"stage2_metrics{suffix}.json").write_text(
        json.dumps(asdict(metrics), indent=2, default=str))
    print("\n" + json.dumps(asdict(metrics), indent=2, default=str))


def _build_extractor():
    """Parley extraction over pooled, gated evidence. One call per event."""
    from ..contract_schemas import render_model_schema_instructions
    from ..research_contract import ParleyProvider
    prompt_path = Path("prompts/contract_research.md")
    base = prompt_path.read_text() if prompt_path.exists() else ""
    provider = ParleyProvider(
        os.environ["PARLEY_API_KEY"],
        os.environ.get("PARLEY_MODEL", "bedrock/claude-haiku-4-5"),
        base + "\n\n" + render_model_schema_instructions(),
    )

    def extract(event: dict, usable: list[dict]):
        from ..source_discovery import SourceCandidate
        sources = [
            SourceCandidate(url=r["url"], title=r.get("title") or "", snippet=r.get("text", "")[:12000],
                            source_type=r["source_class"], published_date=None)
            for r in usable
        ]
        return provider.research(event, sources).to_dict()
    return extract


if __name__ == "__main__":
    main()
