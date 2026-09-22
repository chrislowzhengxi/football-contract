"""Extraction-quality pass over the 43 fee-bearing loan returns.

    python -m src.stage2.reextract

Reuses the cached search results, retrieved pages and PDFs from the earlier
run. It issues no new searches and fetches no new pages; the only outbound
calls are extraction, and those are cached too.

Why this exists: the first 43-event run reported 23 events as `extracted`, but
review showed `extracted` did not mean correctly resolved. A direction was
reversed, a "definitive acquisition" became a purchase obligation, and a later
buyback populated an earlier loan-return. This pass re-extracts under a
stricter prompt and then enforces every rule again in code, because a prompt
cannot be relied on to police itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from ..stage1.backbone import REBUILD_DIR
from .audit import audit_payload, classify_resolution, normalise_status
from .run_pilot import (
    CACHE_DIR,
    Metrics,
    _cache_path,
    gather_evidence,
    load_credentials,
    missing_credentials,
)

POPULATION_CSV = REBUILD_DIR / "stage2_fee_return_population.csv"
OUT_DIR = REBUILD_DIR / "stage2"
PROMPT_PATH = Path("prompts/contract_research_v2.md")
PARLEY_URL = "https://parley.api.mit.edu/v1/chat/completions"
PROMPT_VERSION = "v2-eventlock"


def target_event_block(event: dict) -> dict:
    """The event as an unambiguous instruction, not a row dump.

    Stage 1 fee columns are deliberately excluded: handing the model the
    fee_on_return amount invites it to explain that number rather than report
    what the evidence says.
    """
    return {
        "event_id": event["event_id"],
        "player_id": int(event["player_id"]),
        "player_name": event["player_name"],
        "from_club": event["from_club_name"],
        "to_club": event["to_club_name"],
        "transfer_date": str(event["transfer_date"]),
        "season": event.get("transfer_season"),
        "direction": (
            f"On {event['transfer_date']}, {event['player_name']} LEFT "
            f"{event['from_club_name']} and JOINED {event['to_club_name']}. "
            f"This movement, and only this movement, is the target event. "
            f"Any other transfer of this player is a different transaction."
        ),
        "stage1_transfer_type": event.get("transfer_type_normalized"),
    }


def call_parley(prompt: str, payload: dict, model: str, timeout: float = 180) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
        "response_format": {"type": "json_object"},
    }
    request = Request(
        PARLEY_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['PARLEY_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        cost = response.headers.get("x-parley-v1-cost")
        data = json.loads(response.read())
    content = data["choices"][0]["message"]["content"]
    if content.strip().startswith("```"):
        content = content.strip().strip("`")
        content = content[content.find("{"):content.rfind("}") + 1]
    parsed = json.loads(content)
    usage = data.get("usage", {})
    parsed["provider_metadata"] = {
        "provider": "parley", "model": data.get("model", model),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "parley_cost_header": cost,
    }
    return parsed


def extract_one(event: dict, usable: list[dict], prompt: str, model: str,
                metrics: Metrics) -> tuple[dict | None, str]:
    """(raw payload, status). One technical retry, no second web search."""
    fingerprint = hashlib.sha256(
        (PROMPT_VERSION + "|" + event["event_id"] + "|"
         + "|".join(sorted(r["url"] for r in usable))).encode()).hexdigest()
    cache_file = _cache_path("extraction_v2", fingerprint)
    if cache_file.exists():
        metrics.extraction_cache_hits += 1
        return json.loads(cache_file.read_text()), "extracted"

    sources = [
        {
            "evidence_id": f"ev{i:02d}",
            "source_url": r["url"],
            "source_title": r.get("title") or "",
            "publisher": r.get("domain"),
            "source_class": r["source_class"],
            "source_tier": r.get("tier"),
            "retrieval_date": time.strftime("%Y-%m-%d"),
            "evidence_text": (r.get("text") or "")[:12000],
        }
        for i, r in enumerate(usable, start=1)
    ]
    payload = {"target_event": target_event_block(event), "sources": sources}

    last_error = ""
    for attempt in range(2):
        try:
            raw = call_parley(prompt, payload, model)
            metrics.llm_extraction_calls += 1
            cache_file.write_text(json.dumps(raw, ensure_ascii=False, default=str))
            return raw, "extracted"
        except (HTTPError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            metrics.llm_extraction_calls += 1
            if attempt == 0:
                time.sleep(2.0)
    metrics.errors.append(f"{event['event_id']} extraction: {last_error}")
    return None, "extraction_failed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 extraction-quality pass")
    parser.add_argument("--population", default=str(POPULATION_CSV))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_credentials()
    missing = missing_credentials()
    if missing:
        raise SystemExit(f"STOP - missing credentials: {', '.join(missing)}")

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.population)
    if args.limit:
        events = events.head(args.limit)
    prompt = PROMPT_PATH.read_text()
    model = os.environ.get("PARLEY_MODEL", "bedrock/claude-haiku-4-5")

    prior = {}
    prior_path = output_dir / "stage2_research_records_fee_bearing_loan_return.json"
    if prior_path.exists():
        prior = {r["event_id"]: r for r in json.loads(prior_path.read_text())}

    metrics = Metrics(model_used=model)
    rows, records = [], []
    for _, row in events.iterrows():
        event = row.to_dict()
        metrics.events_attempted += 1
        # dry_run=True: cache only. No search, no page fetch, no spend.
        stub, usable = gather_evidence(event, None, None, metrics, dry_run=True)

        if not usable:
            status, raw, audit = "insufficient_evidence", None, None
        else:
            raw, status = extract_one(event, usable, prompt, model, metrics)
            audit = audit_payload(raw, event) if raw else None

        if audit is None:
            audit = {"violations": [], "downgraded": [], "direction_flags": [],
                     "fields_supported": [], "mechanism_fields_supported": [],
                     "scope_declared": False, "target_evidence_count": 0,
                     "related_event_evidence": []}
        resolution = classify_resolution(audit, status, len(usable))
        if resolution.startswith("resolved"):
            metrics.events_completed += 1

        prior_status = prior.get(event["event_id"], {}).get("extraction_status", "n/a")
        review_reasons = list((raw or {}).get("review_reasons") or [])
        review_reasons += audit["downgraded"] + audit["direction_flags"]

        rows.append({
            "event_id": event["event_id"],
            "player_id": int(event["player_id"]),
            "player": event["player_name"],
            "from_club": event["from_club_name"],
            "to_club": event["to_club_name"],
            "transfer_date": event["transfer_date"],
            "stage1_fee_on_return_eur": event.get("fee_on_return_eur"),
            "prior_extraction_status": prior_status,
            "revised_resolution": resolution,
            "fields_supported": ";".join(audit["fields_supported"]),
            "mechanism_fields_supported": ";".join(audit["mechanism_fields_supported"]),
            "n_fields_supported": len(audit["fields_supported"]),
            "evidence_sources": ";".join(r["url"] for r in usable)[:600],
            "source_classes": ";".join(sorted({r["source_class"] for r in usable})),
            "pages_usable": len(usable),
            "event_match_quality": (
                "scope_declared" if audit["scope_declared"] else "scope_not_declared"),
            "target_evidence_count": audit["target_evidence_count"],
            "direction_check": ("fail" if audit["direction_flags"]
                                else "pass" if audit["fields_supported"] else "n/a"),
            "status_repairs": len([v for v in audit["violations"] if "status" in v]),
            "downgrades": len(audit["downgraded"]),
            "review_required": bool((raw or {}).get("review_required")) or bool(
                audit["downgraded"] or audit["direction_flags"]),
            "review_reasons": " | ".join(str(r) for r in review_reasons)[:900],
            "related_event_evidence": len(audit["related_event_evidence"]),
        })
        records.append({"event_id": event["event_id"], "resolution": resolution,
                        "audit": audit, "result": raw})

        pd.DataFrame(rows).to_csv(output_dir / "stage2_extraction_audit.csv", index=False)
        (output_dir / "stage2_reextraction_records.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False, default=str))
        (output_dir / "stage2_reextraction_metrics.json").write_text(
            json.dumps(asdict(metrics), indent=2, default=str))
        print(f"  {event['event_id'][:14]} {str(event['player_name'])[:20]:22s} "
              f"{prior_status:22s} -> {resolution:34s} "
              f"fields={len(audit['fields_supported'])} dg={len(audit['downgraded'])}",
              flush=True)

    print("\n" + json.dumps(asdict(metrics), indent=2, default=str))


if __name__ == "__main__":
    main()
