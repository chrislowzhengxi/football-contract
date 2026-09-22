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
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..stage1.backbone import REBUILD_DIR
from .pdf_text import fetch_text as pdf_fetch_text
from .queries import build_queries
from .sources import can_establish_alone, classify, domain_of, tier_of

PILOT_CSV = REBUILD_DIR / "stage2_pilot_selection.csv"
OUT_DIR = REBUILD_DIR / "stage2"
CACHE_DIR = Path("data/outputs/contract_research/stage2_cache")
ENV_FILE = Path(".pytest_cache/.env")
PDF_URL_HINT = re.compile(r"\.pdf($|\?)|/api/file/download/|/download/|filings?/", re.I)
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
# A page reaches the LLM only if it carries contract language. Splitting the
# vocabulary by strength matters: a flat count both admitted match reports
# (which say "appearances" and "%") and rejected genuine announcements whose
# only signal was a single explicit term like "obligation to buy".
STRONG_GATE_TERMS = [
    "option to buy", "purchase option", "obligation to buy", "obligation to purchase",
    "mandatory purchase", "sell-on", "sell on clause", "buy-back", "buy back clause",
    "release clause", "exercised the option", "triggered the obligation",
    "permanent deal", "permanent transfer for", "made permanent",
    "diritto di riscatto", "obbligo di riscatto", "controriscatto",
    "clausola rescissoria", "percentuale sulla futura rivendita",
    "opción de compra", "obligación de compra", "opción de recompra",
    "cláusula de rescisión", "porcentaje de una futura venta",
    "opção de compra", "opção de recompra", "percentagem de mais-valia",
    "option d'achat", "obligation d'achat", "clause de rachat",
    "pourcentage à la revente", "kaufoption", "kaufpflicht",
    "weiterverkaufsbeteiligung", "rückkaufoption",
    "satın alma opsiyonu", "zorunlu satın alma", "geri alma opsiyonu",
    "bonservis bedeli", "opsiyon hakkı",
    # Vocabulary observed in KAP (Borsa Istanbul) filings, which itemise every
    # transfer of a season: these are the phrases the disclosures actually use.
    "sonraki satıştan pay", "sonraki satış payı", "geçici transfer bedeli",
    "kesin transferini gerçekleştirme opsiyonu", "şarta bağlı zorunlu",
    "satın alma önceliği", "transfer bedeli ödenecektir",
]
WEAK_GATE_TERMS = [
    "add-on", "add on", "bonus", "bonuses", "clause", "contract until",
    "signed until", "loan fee", "exercised", "triggered", "appearances",
    "per cent", "percent", "undisclosed fee", "transfer fee", "riscatto",
    "obbligo", "clausola", "bónus", "variables", "boni",
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
    pdf_pages_extracted: int = 0
    pages_failed_retrieval: int = 0
    page_cache_hits: int = 0
    pages_relevant: int = 0
    pages_passing_gate: int = 0
    llm_extraction_calls: int = 0
    extraction_cache_hits: int = 0
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


def cached_page(url: str, fetcher, metrics: Metrics, dry_run: bool) -> tuple[str | None, str]:
    """Return (text, retrieval_status).

    The status is kept so a retrieval failure is never confused with a page
    that was fetched fine but simply says nothing about the contract.
    """
    path = _cache_path("page", url)
    if path.exists():
        payload = json.loads(path.read_text())
        text = payload.get("text")
        if text:
            metrics.page_cache_hits += 1
            return text, payload.get("status", "ok")
        # a cached empty body is a cached failure, not a cache miss
        metrics.pages_failed_retrieval += 1
        return None, payload.get("status", "empty")
    if dry_run:
        return None, "dry_run"
    # Regulated filings are usually PDFs. The HTML fetcher returns parsed
    # binary for those, which is why the pilot could not read any KAP
    # disclosure. Try the PDF path first for anything that looks like a file.
    if PDF_URL_HINT.search(url):
        text, status = pdf_fetch_text(url, CACHE_DIR / "pdf_bin")
        if text:
            metrics.pages_retrieved += 1
            metrics.pdf_pages_extracted += 1
            path.write_text(json.dumps({"url": url, "text": text, "status": status},
                                       ensure_ascii=False))
            return text, status
    try:
        result = fetcher.fetch(url)
        text = getattr(result, "extracted_text", "") or ""
        status = getattr(result, "retrieval_status", "unknown")
        http = getattr(result, "http_status", None)
    except Exception as exc:                       # noqa: BLE001
        metrics.pages_failed_retrieval += 1
        metrics.errors.append(f"retrieval raised {domain_of(url)}: {type(exc).__name__}")
        return None, "exception"
    if text:
        metrics.pages_retrieved += 1
    else:
        metrics.pages_failed_retrieval += 1
        status = f"{status}:{http}" if http else status
    path.write_text(json.dumps({"url": url, "text": text, "status": status},
                               ensure_ascii=False))
    time.sleep(0.4)
    return (text or None), status


def excerpt_around_player(text: str, event: dict, window: int = 3500) -> str:
    """For a long filing, keep only the passages that name the player.

    A club's annual report itemises every transfer of the season; sending all
    132,000 characters would be wasteful and would bury the relevant clause.
    """
    if len(text) <= window * 2:
        return text
    surname = str(event.get("player_name", "")).split()[-1].lower()
    low, spans = text.lower(), []
    start = 0
    while len(spans) < 4:
        i = low.find(surname, start)
        if i < 0:
            break
        spans.append((max(0, i - window // 2), min(len(text), i + window)))
        start = i + len(surname)
    if not spans:
        return text[:window * 2]
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return "\n...\n".join(text[a:b] for a, b in merged)


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
    """One explicit contract term is enough; otherwise three weak signals."""
    low = (text or "").lower()
    if low.lstrip().startswith("%pdf") or "%PDF-" in (text or "")[:400]:
        return False, ["__binary_pdf__"]          # unreadable without a PDF parser
    strong = sorted({t for t in STRONG_GATE_TERMS if t in low})
    weak = sorted({t for t in WEAK_GATE_TERMS if t in low})
    return (bool(strong) or len(weak) >= 3), strong + weak


def gather_evidence(event: dict, provider, fetcher, metrics: Metrics,
                    dry_run: bool) -> tuple[dict, list[dict]]:
    """Discovery and retrieval only. Returns (record stub, usable evidence).

    Split out from research_event so the extraction-quality audit can rebuild
    the exact evidence set from cache without issuing a single new request.
    """
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
        if record["tier"] >= 3 or record["url"] in seen or len(retrieved) >= 8:
            continue
        seen.add(record["url"])
        text, status = cached_page(record["url"], fetcher, metrics, dry_run)
        record["retrieval_status"] = status
        if not text:
            record["retrieved"] = False
            retrieved.append(record)          # kept, so failures are countable
            continue
        record["retrieved"] = True
        record["matches_event"] = event_matches(text, event)
        gated, hits = passes_gate(text)
        record["passes_gate"] = gated
        record["gate_terms"] = hits[:12]
        record["gate_strength"] = ("strong" if any(t in STRONG_GATE_TERMS for t in hits)
                                   else "weak" if gated else "none")
        record["text_chars"] = len(text)
        record["text"] = excerpt_around_player(text, event)[:20000]
        if record["matches_event"]:
            metrics.pages_relevant += 1
        if gated and record["matches_event"]:
            metrics.pages_passing_gate += 1
        retrieved.append(record)

    got_text = [r for r in retrieved if r.get("retrieved")]
    usable = [r for r in got_text if r.get("passes_gate") and r.get("matches_event")]
    record = {
        "event_id": event_id,
        "player_id": int(event["player_id"]),
        "candidate_category": event.get("candidate_category"),
        "queries_planned": planned,
        "evidence_found": len(evidence),
        "pages_attempted": len(retrieved),
        "pages_retrieved": len(got_text),
        "pages_matching_event": sum(1 for r in got_text if r.get("matches_event")),
        "pages_usable": len(usable),
        "tier1_usable": sum(1 for r in usable if can_establish_alone(r["source_class"])),
        "source_classes_seen": sorted({r["source_class"] for r in evidence}),
        "extraction_status": None,
        "result": None,
    }
    return record, usable


def research_event(event: dict, provider, fetcher, extractor, metrics: Metrics,
                   dry_run: bool) -> dict:
    """One event end to end. Returns a record; never raises for data reasons."""
    event_id = event["event_id"]
    record, usable = gather_evidence(event, provider, fetcher, metrics, dry_run)
    if not usable:
        metrics.llm_skipped_insufficient_evidence += 1
        record["extraction_status"] = "insufficient_evidence"
        return record
    if dry_run:
        record["extraction_status"] = "would_extract"
        return record

    # Extraction is the only step that costs real money, so its result is
    # cached against the evidence it was derived from. A re-run after a crash
    # re-reads rather than re-pays; changing the evidence changes the key.
    fingerprint = hashlib.sha256(
        (event_id + "|" + "|".join(sorted(r["url"] for r in usable))).encode()).hexdigest()
    cache_file = _cache_path("extraction", fingerprint)
    if cache_file.exists():
        metrics.extraction_cache_hits += 1
        record["extraction_status"] = "extracted"
        record["result"] = json.loads(cache_file.read_text())
        return record
    try:
        result = extractor(event, usable)
        metrics.llm_extraction_calls += 1
        cache_file.write_text(json.dumps(result, ensure_ascii=False, default=str))
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

    suffix = ("_dryrun" if args.dry_run else "") + (f"_{args.category}" if args.category else "")
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
              f"usable={record['pages_usable']} -> {record['extraction_status']}", flush=True)
        # Checkpoint after every event; a later crash cannot lose earlier work.
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
            SourceCandidate(
                source_url=r["url"],
                source_title=r.get("title") or "",
                publisher=r.get("domain"),
                publication_date=None,
                source_type=r["source_class"],
                evidence_text=(r.get("text") or "")[:12000],
                language="en",
                evidence_id=f"ev{i:02d}",
                source_tier=r.get("tier"),
                retrieved_text=(r.get("text") or "")[:12000],
                retrieval_status=r.get("retrieval_status"),
            )
            for i, r in enumerate(usable, start=1)
        ]
        return provider.research(event, sources).to_dict()
    return extract


if __name__ == "__main__":
    main()
