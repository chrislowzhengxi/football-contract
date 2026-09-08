from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .build_research_dataset import coverage
from .config import DEFAULT_OUTPUT_DIR
from .fullpage_parley_3 import (
    _class_counts,
    _coverage,
    _field_changes,
    _provider,
    _read_json,
    _regenerate_datasets,
    _run_parley,
    _update_batch_counts,
    _write_json,
)
from .official_page_retrieval import OfficialPageFetcher
from .research_contract import load_event
from .source_discovery import (
    SourceCandidate,
    TavilySearchProvider,
    assess_source_sufficiency,
    build_query_plan,
    discover_transfer,
    filter_admissible_sources,
    load_source_candidates,
    source_tier,
)
from .retry_4_discovery import _load_allowed_env


BASE_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
BATCH_PATH = BASE_DIR / "batch_20_results.json"
BASELINE_PATH = BASE_DIR / "fresh_8_baseline.json"
COMPARISON_PATH = BASE_DIR / "fresh_8_discovery_comparison.csv"
UPDATE_LOG_PATH = BASE_DIR / "fresh_8_update_log.md"
STATUS_REPORT_PATH = BASE_DIR / "pilot_20_status_report.md"
FRESH_DISCOVERED_DIR = BASE_DIR / "fresh_8_sources" / "discovered"
FRESH_ADMISSIBLE_DIR = BASE_DIR / "fresh_8_sources" / "admissible"
ARCHIVE_DIR = BASE_DIR / "fresh_8_archive"
SUMMARY_PATH = BASE_DIR / "fresh_8_summary.json"

FIELDS = (
    "transfer_type", "transfer_fee", "loan_fee", "add_ons", "purchase_option",
    "purchase_obligation", "obligation_trigger", "sell_on", "buy_back",
    "parent_contract_expiry",
)


def _analysis_rows() -> list[dict[str, Any]]:
    with (DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _current_insufficient_ids() -> list[str]:
    return [row["event_id"] for row in _analysis_rows() if row["classification"] == "insufficient_evidence"]


def _load_sources(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("sources", payload if isinstance(payload, list) else [])


def _old_discovered_artifact(event_id: str) -> Path:
    retry = BASE_DIR / "retry_4_sources" / "discovered" / f"{event_id}.json"
    if retry.exists():
        return retry
    return DEFAULT_OUTPUT_DIR / "discovered_sources" / f"{event_id}.json"


def build_baseline(event_ids: list[str]) -> list[dict[str, Any]]:
    batch = _read_json(BATCH_PATH)
    batch_rows = {event["event_id"]: event for event in batch["events"]}
    rows = []
    for event_id in event_ids:
        event = load_event(DEFAULT_OUTPUT_DIR / "structured_transfers.csv", event_id)
        sources = _load_sources(_old_discovered_artifact(event_id))
        tier_counts = Counter(source.get("source_tier") for source in sources)
        admissible = _load_sources(DEFAULT_OUTPUT_DIR / "discovered_sources" / "admissible" / f"{event_id}.json")
        rows.append({
            "event_id": event_id,
            "player": event.get("player_name"),
            "departing_club": event.get("from_club_name"),
            "receiving_club": event.get("to_club_name"),
            "date": event.get("transfer_date"),
            "current_classification": batch_rows[event_id].get("classification"),
            "old_discovered_source_count": len(sources),
            "old_tier1_count": tier_counts.get(1, 0),
            "old_tier2_count": tier_counts.get(2, 0),
            "old_admissible_count": len(admissible),
            "old_sufficiency": bool(batch_rows[event_id].get("sufficient")),
            "unresolved_reasons": batch_rows[event_id].get("sufficiency_reasons", []),
        })
    _write_json(BASELINE_PATH, rows)
    return rows


def _write_source_artifacts(event: dict[str, Any], candidates: list[SourceCandidate], admissible: list[SourceCandidate], sufficient: bool, reasons: list[str]) -> None:
    common = {
        "event_id": event["event_id"],
        "event": {
            "player_name": event.get("player_name"),
            "from_club_name": event.get("from_club_name"),
            "to_club_name": event.get("to_club_name"),
            "transfer_date": event.get("transfer_date"),
            "transfer_fee": event.get("transfer_fee"),
        },
        "sufficient": sufficient,
        "sufficiency_reasons": reasons,
    }
    _write_json(FRESH_DISCOVERED_DIR / f"{event['event_id']}.json", {**common, "sources": [candidate.to_dict() for candidate in candidates]})
    _write_json(FRESH_ADMISSIBLE_DIR / f"{event['event_id']}.json", {**common, "sources": [candidate.to_dict() for candidate in admissible]})


def _evidence_sources_for_result(sources: list[SourceCandidate]) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        row = source.to_dict()
        row["retrieval_date"] = (source.retrieval_timestamp or "")[:10] or "2026-09-08"
        rows.append(row)
    return rows


def _archive_existing(path: Path) -> None:
    if path.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, ARCHIVE_DIR / path.name)


def _run_parley_for_event(event: dict[str, Any], admissible: list[SourceCandidate]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    provider = _provider()
    result, info = _run_parley(provider, event, admissible)
    if result:
        result["sources"] = _evidence_sources_for_result(admissible)
    return result, info


def _comparison_row(baseline: dict[str, Any], candidates: list[SourceCandidate], admissible: list[SourceCandidate], sufficient: bool, pages_retrieved: int, retrieval_successes: int, parley_called: bool, classification_after: str) -> dict[str, Any]:
    tiers = Counter(candidate.source_tier for candidate in candidates)
    exact = sum(1 for candidate in candidates if candidate.event_match_status == "exact")
    likely = sum(1 for candidate in candidates if candidate.event_match_status == "likely")
    return {
        "event_id": baseline["event_id"],
        "player": baseline["player"],
        "old_discovered_sources": baseline["old_discovered_source_count"],
        "new_discovered_sources": len(candidates),
        "old_tier1": baseline["old_tier1_count"],
        "new_tier1": tiers.get(1, 0),
        "old_tier2": baseline["old_tier2_count"],
        "new_tier2": tiers.get(2, 0),
        "old_admissible": baseline["old_admissible_count"],
        "new_admissible": len(admissible),
        "old_sufficient": baseline["old_sufficiency"],
        "new_sufficient": sufficient,
        "new_exact_matches": exact,
        "new_likely_matches": likely,
        "pages_retrieved": pages_retrieved,
        "retrieval_successes": retrieval_successes,
        "parley_called": parley_called,
        "classification_before": baseline["current_classification"],
        "classification_after": classification_after,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _failure_category(row: dict[str, Any]) -> str:
    if row["classification_after"] != "insufficient_evidence":
        return "rescued"
    if int(row["new_tier1"]) + int(row["new_tier2"]) == 0:
        return "discovery failure"
    if int(row["new_admissible"]) == 0 or int(row["new_exact_matches"]) + int(row["new_likely_matches"]) == 0:
        return "event-identity failure"
    if row["new_sufficient"] is False:
        return "disclosure/missing-data failure"
    return "extraction/validation failure"


def _update_log(rows: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any], attempts: int, cost: float) -> None:
    lines = [
        "# Fresh 8 Update Log",
        "",
        f"- Parley attempts: {attempts}",
        f"- Parley cost: {cost:.6f}",
        f"- Classification counts before: {json.dumps(before['class_counts'], sort_keys=True)}",
        f"- Classification counts after: {json.dumps(after['class_counts'], sort_keys=True)}",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row['player']} ({row['event_id']})",
            "",
            f"- Classification before: `{row['classification_before']}`",
            f"- Classification after: `{row['classification_after']}`",
            f"- Parley called: `{row['parley_called']}`",
            f"- Parley attempts: {row['parley_attempts']}",
            f"- Parley cost: {row['parley_cost']:.6f}",
            f"- Newly established fields: {json.dumps(row['fields_newly_established'], ensure_ascii=False)}",
            f"- Fields still unknown: {json.dumps(row['fields_still_unknown'], ensure_ascii=False)}",
            "",
        ])
    UPDATE_LOG_PATH.write_text("\n".join(lines))


def _status_report(comparison: list[dict[str, Any]], updates: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any], attempts: int, cost: float, tavily_queries: int, raw_results: int) -> None:
    failure_counts = Counter(_failure_category(row) for row in comparison)
    tier1 = sum(int(row["new_tier1"]) for row in comparison)
    tier2 = sum(int(row["new_tier2"]) for row in comparison)
    pages = sum(int(row["pages_retrieved"]) for row in comparison)
    retrieved = sum(int(row["retrieval_successes"]) for row in comparison)
    lines = [
        "# Pilot 20 Status Report",
        "",
        "The standard architecture is now Tavily discovery, targeted full-page retrieval for Tier 1/2 URLs, event matching on retrieved text, admissibility filtering, sufficiency gating, and Parley extraction only after sufficiency.",
        "",
        "## Current Status",
        "",
        f"- Clean: {after['class_counts'].get('clean', 0)}",
        f"- Usable with review: {after['class_counts'].get('usable_with_review', 0)}",
        f"- Insufficient evidence: {after['class_counts'].get('insufficient_evidence', 0)}",
        f"- Invalid: {after['class_counts'].get('invalid', 0)}",
        f"- Fresh-8 Tavily queries: {tavily_queries}",
        f"- Fresh-8 raw Tavily results: {raw_results}",
        f"- Fresh-8 Tier 1 sources found: {tier1}",
        f"- Fresh-8 Tier 2 sources found: {tier2}",
        f"- Fresh-8 pages fetched/cached: {pages}",
        f"- Fresh-8 retrieval successes: {retrieved}",
        f"- Fresh-8 Parley attempts: {attempts}",
        f"- Fresh-8 Parley cost: {cost:.6f}",
        "",
        "## Field Coverage",
        "",
    ]
    for field in FIELDS:
        lines.append(f"- {field}: {before['coverage'].get(field, 0)} -> {after['coverage'].get(field, 0)}")
    lines.extend([
        "",
        "## Failure Categories",
        "",
    ])
    for category, count in failure_counts.items():
        lines.append(f"- {category}: {count}")
    lines.extend([
        "",
        "Sell-on and buy-back remain systematically sparse in public evidence. Transfer type and broad loan/permanent structures are much more available than secondary clauses.",
        "",
        "Recommendation: B. Scale only selected transfer types / clubs where evidence quality is high.",
    ])
    STATUS_REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    _load_allowed_env()
    tavily = TavilySearchProvider.from_environment()
    fetcher = OfficialPageFetcher()
    before = {"class_counts": _class_counts(), "coverage": _coverage()}
    event_ids = _current_insufficient_ids()
    baseline = build_baseline(event_ids)
    batch = _read_json(BATCH_PATH)
    batch_events = {event["event_id"]: event for event in batch["events"]}
    comparison_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    total_attempts = 0
    total_cost = 0.0
    total_queries = 0
    total_raw_results = 0

    for base in baseline:
        event = load_event(DEFAULT_OUTPUT_DIR / "structured_transfers.csv", base["event_id"])
        before_searches = tavily.search_count
        before_raw = tavily.raw_result_count
        result = discover_transfer(event, tavily, fullpage_fetcher=fetcher)
        candidates = result.candidates
        admissible = filter_admissible_sources(candidates)
        sufficient, reasons = assess_source_sufficiency(admissible)
        _write_source_artifacts(event, candidates, admissible, sufficient, reasons)
        query_count = tavily.search_count - before_searches
        raw_count = tavily.raw_result_count - before_raw
        total_queries += query_count
        total_raw_results += raw_count
        pages = sum(1 for candidate in candidates if source_tier(candidate) <= 2)
        retrieved = sum(1 for candidate in candidates if source_tier(candidate) <= 2 and candidate.retrieval_status == "success")
        parley_called = False
        classification_after = base["current_classification"]
        established: list[str] = []
        unknown: list[str] = []
        parley_attempts = 0
        parley_cost = 0.0

        if sufficient:
            parley_result, info = _run_parley_for_event(event, admissible)
            parley_attempts = int(info["attempts"])
            parley_cost = float(info["total_cost"])
            total_attempts += parley_attempts
            total_cost += parley_cost
            parley_called = bool(parley_result)
            established, unknown = _field_changes(parley_result)
            if parley_result:
                output_path = BASE_DIR / f"{base['event_id']}.json"
                _archive_existing(output_path)
                output_path.write_text(json.dumps(parley_result, indent=2, ensure_ascii=False) + "\n")
                classification_after = "usable_with_review" if parley_result.get("review_required") else "clean"
                meta = parley_result.get("provider_metadata", {})
                batch_events[base["event_id"]].update({
                    "classification": classification_after,
                    "parley_called": True,
                    "parley_attempts": parley_attempts,
                    "parley_successful_extraction": True,
                    "parley_cost": meta.get("total_request_cost") or parley_cost,
                    "parley_total_attempt_cost": parley_cost,
                    "output_path": str(output_path),
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "completion_tokens": meta.get("completion_tokens"),
                    "total_tokens": meta.get("total_tokens"),
                    "review_required": parley_result.get("review_required"),
                    "review_reasons": parley_result.get("review_reasons") or [],
                })

        batch_events[base["event_id"]].update({
            "tavily_search_count": query_count,
            "raw_result_count": raw_count,
            "source_count": len(candidates),
            "admissible_source_count": len(admissible),
            "sufficient": sufficient,
            "sufficiency_reasons": reasons,
        })
        comparison_rows.append(_comparison_row(base, candidates, admissible, sufficient, pages, retrieved, parley_called, classification_after))
        update_rows.append({
            "event_id": base["event_id"],
            "player": base["player"],
            "classification_before": base["current_classification"],
            "classification_after": classification_after,
            "parley_called": parley_called,
            "parley_attempts": parley_attempts,
            "parley_cost": parley_cost,
            "fields_newly_established": established,
            "fields_still_unknown": unknown,
        })

    _write_csv(COMPARISON_PATH, comparison_rows)
    batch["fresh_8_tavily_queries"] = total_queries
    batch["fresh_8_raw_results"] = total_raw_results
    batch["fresh_8_parley_attempts"] = total_attempts
    batch["fresh_8_successful_parley_extractions"] = sum(1 for row in update_rows if row["parley_called"])
    batch["fresh_8_parley_cost"] = round(total_cost, 6)
    _update_batch_counts(batch)
    _write_json(BATCH_PATH, batch)
    _regenerate_datasets(batch)
    after = {"class_counts": _class_counts(), "coverage": _coverage()}
    _update_log(update_rows, before, after, total_attempts, total_cost)
    _status_report(comparison_rows, update_rows, before, after, total_attempts, total_cost, total_queries, total_raw_results)
    _write_json(SUMMARY_PATH, {
        "before": before,
        "after": after,
        "comparison": comparison_rows,
        "updates": update_rows,
        "tavily_queries": total_queries,
        "raw_results": total_raw_results,
        "parley_attempts": total_attempts,
        "parley_cost": round(total_cost, 6),
    })


if __name__ == "__main__":
    main()
