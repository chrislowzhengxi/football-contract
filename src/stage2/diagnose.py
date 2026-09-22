"""Bounded diagnostic: is the Stage 2 ceiling the pipeline or the public record?

    python -m src.stage2.diagnose

Runs a deliberately expensive, high-recall pass over 12 events and compares it
against exactly what the bounded pipeline saw, so each unresolved case can be
attributed to a layer instead of being written off as scarcity.

The old baseline is not taken from the old report; it is replayed from cache by
calling the same gather_evidence the pilot used, so the comparison is exact.
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
from .audit import audit_payload, classify_resolution
from .deep_search import (
    CLAUSE_TERMS,
    build_deep_queries,
    chunk_document,
    pdf_pages_of_interest,
    tavily_deep_search,
)
from .pdf_text import extract_pdf_text, fetch_binary, looks_like_pdf
from .reextract import call_parley, target_event_block
from .run_pilot import CACHE_DIR, Metrics, _cache_path, gather_evidence, load_credentials
from .sources import classify, domain_of, tier_of

SAMPLE_CSV = REBUILD_DIR / "stage2_diagnostic_sample.csv"
OUT_DIR = REBUILD_DIR / "stage2"
PROMPT_PATH = Path("prompts/contract_research_v3.md")
PROMPT_VERSION = "v3-deep"

STAGE2_TARGET = ["purchase_option", "purchase_obligation", "obligation_trigger", "add_ons",
                 "sell_on", "buy_back", "parent_contract_expiry", "release_or_purchase_clause"]

# Classes that may never establish a term alone. Everything else is judged on
# its merits; an unknown domain is discoverable and readable, not auto-trusted.
LEADS_ONLY = {"aggregator_low_quality", "transfermarkt_backbone"}


@dataclass
class DeepMetrics:
    events: int = 0
    queries_issued: int = 0
    query_cache_hits: int = 0
    results_returned: int = 0
    unique_urls: int = 0
    pages_with_text: int = 0
    pages_from_raw_content: int = 0
    pages_fetched_separately: int = 0
    pdfs_parsed: int = 0
    chunks_built: int = 0
    extraction_calls: int = 0
    extraction_cache_hits: int = 0
    completion_tokens: int = 0
    errors: list = field(default_factory=list)


def cached_deep_search(query: str, api_key: str, m: DeepMetrics) -> dict:
    path = _cache_path("deep_search", PROMPT_VERSION + "|" + query)
    if path.exists():
        m.query_cache_hits += 1
        return json.loads(path.read_text())
    # The first 12-event run lost one event entirely to transient HTTP errors,
    # so a failed query is retried with backoff rather than silently dropped.
    payload, last = None, None
    for attempt in range(3):
        try:
            payload = tavily_deep_search(query, api_key)
            break
        except Exception as exc:                            # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    if payload is None:
        raise last
    m.queries_issued += 1
    path.write_text(json.dumps(payload, ensure_ascii=False))
    time.sleep(0.3)
    return payload


def page_text(url: str, raw_content: str | None, m: DeepMetrics) -> tuple[str | None, str]:
    """Prefer content the search provider already returned; fall back to our
    own fetch, and parse PDFs properly. Full text, no excerpting here."""
    if raw_content and len(raw_content.strip()) > 200:
        m.pages_from_raw_content += 1
        return raw_content, "tavily_raw_content"
    path = _cache_path("deep_page", url)
    if path.exists():
        d = json.loads(path.read_text())
        return d.get("text") or None, d.get("status", "cache")
    blob = fetch_binary(url, timeout=30)
    if blob is None:
        path.write_text(json.dumps({"url": url, "text": "", "status": "fetch_failed"}))
        return None, "fetch_failed"
    if looks_like_pdf(blob):
        text = extract_pdf_text(blob, max_pages=80)
        status = "pdf_ok" if text else "pdf_unreadable"
        if text:
            m.pdfs_parsed += 1
    else:
        from ..official_page_retrieval import extract_visible_text
        try:
            text, status = extract_visible_text(blob.decode("utf-8", "ignore")), "html_ok"
        except Exception:                                   # noqa: BLE001
            text, status = "", "html_parse_failed"
    m.pages_fetched_separately += 1
    path.write_text(json.dumps({"url": url, "text": text, "status": status}, ensure_ascii=False))
    time.sleep(0.2)
    return (text or None), status


def clause_terms_in(text: str) -> list[str]:
    low = (text or "").lower()
    return sorted({t for t in CLAUSE_TERMS if t.lower() in low})


def outside_excerpt_analysis(full_text: str, old_excerpt: str, event: dict) -> dict:
    """Part 4: did the old fixed player-window hide clause language?"""
    full_terms = set(clause_terms_in(full_text))
    seen_terms = set(clause_terms_in(old_excerpt or ""))
    missed = sorted(full_terms - seen_terms)
    return {
        "full_chars": len(full_text or ""),
        "old_excerpt_chars": len(old_excerpt or ""),
        "clause_terms_in_full": sorted(full_terms),
        "clause_terms_missed_by_old_excerpt": missed,
        "old_excerpt_hid_clause_language": bool(missed),
    }


def build_evidence(event: dict, api_key: str, m: DeepMetrics,
                   max_docs: int = 18) -> tuple[list[dict], dict]:
    """Deep discovery + retrieval + multi-chunking for one event."""
    queries = build_deep_queries(event)
    hits: dict[str, dict] = {}
    for q in queries:
        try:
            payload = cached_deep_search(q.query, api_key, m)
        except Exception as exc:                            # noqa: BLE001
            m.errors.append(f"search failed [{q.family}]: {type(exc).__name__}")
            continue
        for rank, item in enumerate(payload.get("results", []), start=1):
            url = item.get("url")
            if not url:
                continue
            m.results_returned += 1
            prev = hits.get(url)
            if prev is None:
                hits[url] = {
                    "url": url, "title": item.get("title"),
                    "snippet": item.get("content") or "",
                    "raw_content": item.get("raw_content"),
                    "score": item.get("score") or 0.0,
                    "published_date": item.get("published_date"),
                    "found_by": [q.family], "best_rank": rank,
                }
            else:
                prev["found_by"].append(q.family)
                prev["best_rank"] = min(prev["best_rank"], rank)
                prev["score"] = max(prev["score"], item.get("score") or 0.0)
    m.unique_urls += len(hits)

    # Rank by how many different query families surfaced it, then by provider
    # score. Deliberately NOT by domain: discovery must not be whitelisted.
    ranked = sorted(hits.values(),
                    key=lambda h: (-len(set(h["found_by"])), -h["score"], h["best_rank"]))
    evidence = []
    for h in ranked[:max_docs]:
        text, status = page_text(h["url"], h.get("raw_content"), m)
        if not text:
            continue
        m.pages_with_text += 1
        cls = classify(h["url"], event.get("from_club"), event.get("to_club"))
        chunks = chunk_document(text, event)
        m.chunks_built += len(chunks)
        evidence.append({
            "url": h["url"], "domain": domain_of(h["url"]), "title": h["title"],
            "source_class": cls, "tier": tier_of(cls),
            "leads_only": cls in LEADS_ONLY,
            "found_by_families": sorted(set(h["found_by"])),
            "n_families": len(set(h["found_by"])), "best_rank": h["best_rank"],
            "published_date": h.get("published_date"),
            "retrieval_status": status, "full_text": text,
            "full_chars": len(text), "clause_terms": clause_terms_in(text),
            "chunks": chunks,
        })
    return evidence, {"queries": [q.__dict__ for q in queries], "unique_urls": len(hits)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2 deep-search diagnostic")
    ap.add_argument("--sample", default=str(SAMPLE_CSV))
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    load_credentials()
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key or not os.environ.get("PARLEY_API_KEY"):
        raise SystemExit("STOP - TAVILY_API_KEY and PARLEY_API_KEY are required")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(args.sample)
    if args.limit:
        sample = sample.head(args.limit)
    prompt = PROMPT_PATH.read_text()
    model = os.environ.get("PARLEY_MODEL", "bedrock/claude-haiku-4-5")
    m = DeepMetrics()
    records, rows = [], []

    for _, row in sample.iterrows():
        event = row.to_dict()
        m.events += 1
        eid = event["event_id"]

        # --- baseline: replay exactly what the bounded pipeline saw ---
        base_event = dict(event)
        base_event["from_club_name"] = event["from_club"]
        base_event["to_club_name"] = event["to_club"]
        base_event["player_name"] = event["player"]
        _, old_usable = gather_evidence(base_event, None, None, Metrics(), dry_run=True)
        old_urls = {r["url"] for r in old_usable}
        old_excerpts = {r["url"]: r.get("text", "") for r in old_usable}

        evidence, qinfo = build_evidence(event, api_key, m)

        # --- Part 4: what did the old excerpt hide? ---
        retrieval_audit = []
        for ev in evidence:
            if ev["url"] in old_urls:
                a = outside_excerpt_analysis(ev["full_text"], old_excerpts[ev["url"]], event)
                a["url"] = ev["url"]
                retrieval_audit.append(a)

        # --- extraction over chunked evidence ---
        admissible = [e for e in evidence if e["clause_terms"] or e["tier"] <= 2]
        admissible = admissible[:12]
        srcs = []
        for i, e in enumerate(admissible, start=1):
            body = "\n\n---\n".join(
                f"[chunk {c['kind']} @{c['offset']}]\n{c['text']}" for c in e["chunks"][:8])
            srcs.append({
                "evidence_id": f"ev{i:02d}", "source_url": e["url"],
                "source_title": e["title"] or "", "publisher": e["domain"],
                "source_class": e["source_class"], "source_tier": e["tier"],
                "leads_only": e["leads_only"], "publication_date": e.get("published_date"),
                "retrieval_date": time.strftime("%Y-%m-%d"),
                "evidence_text": body[:16000],
            })
        raw, status = None, "insufficient_evidence"
        if srcs:
            fp = hashlib.sha256((PROMPT_VERSION + "|" + eid + "|" +
                                 "|".join(sorted(s["source_url"] for s in srcs))).encode()).hexdigest()
            cf = _cache_path("extraction_v3", fp)
            if cf.exists():
                raw, status = json.loads(cf.read_text()), "extracted"
                m.extraction_cache_hits += 1
            else:
                try:
                    raw = call_parley(prompt, {"target_event": target_event_block(base_event),
                                               "sources": srcs}, model, timeout=240)
                    m.extraction_calls += 1
                    m.completion_tokens += (raw.get("provider_metadata") or {}).get("completion_tokens") or 0
                    cf.write_text(json.dumps(raw, ensure_ascii=False, default=str))
                    status = "extracted"
                except Exception as exc:                    # noqa: BLE001
                    m.errors.append(f"{eid} extraction: {type(exc).__name__}: {exc}")
                    status = "extraction_failed"

        audit = audit_payload(raw, base_event) if raw else {
            "violations": [], "downgraded": [], "direction_flags": [], "fields_supported": [],
            "mechanism_fields_supported": [], "scope_declared": False,
            "target_evidence_count": 0, "related_event_evidence": []}
        new_s2 = [f for f in audit["fields_supported"] if f in STAGE2_TARGET]
        new_res = ("resolved_strong" if new_s2 and audit["mechanism_fields_supported"]
                   and not audit["direction_flags"]
                   else "resolved_partial" if new_s2
                   else "unresolved_stage1_facts_only"
                   if audit["fields_supported"] and status == "extracted"
                   else classify_resolution(audit, status, len(srcs)))

        best = ""
        if new_s2 and raw:
            ids = {i for f in new_s2 for i in (raw.get(f, {}) or {}).get("evidence_ids", [])}
            best = ";".join(s["source_url"] for s in srcs if s["evidence_id"] in ids)[:300]

        records.append({"event_id": eid, "player": event["player"], "new_resolution": new_res,
                        "new_stage2_fields": new_s2, "audit": audit, "result": raw,
                        "evidence": [{k: v for k, v in e.items() if k != "full_text"} for e in evidence],
                        "retrieval_audit": retrieval_audit,
                        "old_urls": sorted(old_urls), "queries": qinfo["queries"]})
        rows.append({
            "event_id": eid, "player": event["player"], "from_club": event["from_club"],
            "to_club": event["to_club"], "transfer_date": event["transfer_date"],
            "stratum": event["stratum"], "old_resolution": event["old_resolution"],
            "old_stage2_fields": event.get("old_stage2_fields") or "",
            "old_pages_usable": event["old_pages_usable"],
            "new_resolution": new_res, "new_stage2_fields": ";".join(new_s2),
            "new_urls_discovered": qinfo["unique_urls"],
            "new_docs_with_text": len(evidence),
            "docs_new_vs_old": len([e for e in evidence if e["url"] not in old_urls]),
            "docs_with_clause_terms": len([e for e in evidence if e["clause_terms"]]),
            "old_excerpt_hid_clauses": sum(1 for a in retrieval_audit
                                           if a["old_excerpt_hid_clause_language"]),
            "best_source": best,
            "review_required": bool((raw or {}).get("review_required")),
        })
        pd.DataFrame(rows).to_csv(out / "stage2_diagnostic_results.csv", index=False)
        (out / "stage2_diagnostic_records.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False, default=str))
        (out / "stage2_diagnostic_metrics.json").write_text(
            json.dumps(asdict(m), indent=2, default=str))
        print(f"  {event['player'][:20]:22s} old={event['old_resolution'][:28]:30s} "
              f"new={new_res:18s} fields={new_s2} urls={qinfo['unique_urls']} "
              f"docs={len(evidence)}", flush=True)

    print("\n" + json.dumps(asdict(m), indent=2, default=str))


if __name__ == "__main__":
    main()
