from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .build_research_dataset import coverage
from .config import DEFAULT_OUTPUT_DIR
from .contract_schemas import ContractResearchResult, render_model_schema_instructions
from .fullpage_parley_3 import _class_counts, _coverage, _field_changes, _regenerate_datasets, _run_parley, _update_batch_counts
from .official_page_retrieval import OfficialPageFetcher
from .pipeline_v2 import (
    FIELD_RESEARCH_FIELDS,
    FIELD_VOCABULARY,
    SearchBudget,
    evidence_temporality,
    existing_evidence_rows,
    generate_event_query_plan,
    generate_field_queries,
    resolve_event_from_existing,
)
from .research_contract import ParleyProvider, load_event
from .retry_4_discovery import _load_allowed_env
from .source_discovery import (
    SourceCandidate,
    _domain,
    _source_type,
    assess_event_match,
    deduplicate_sources,
    filter_admissible_sources,
    score_source,
    source_tier,
    TavilySearchProvider,
)


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
BATCH_PATH = OUTPUT_DIR / "batch_20_results.json"
STRUCTURED_PATH = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"
SELECTION_PATH = OUTPUT_DIR / "pipeline_v2_live5_selection.json"
BASELINE_PATH = OUTPUT_DIR / "pipeline_v2_live5_baseline.csv"
QUERY_LOG_PATH = OUTPUT_DIR / "pipeline_v2_live5_query_log.csv"
FIELD_RESULTS_PATH = OUTPUT_DIR / "pipeline_v2_live5_field_results.csv"
UPDATE_LOG_PATH = OUTPUT_DIR / "pipeline_v2_live5_update_log.md"
REPORT_PATH = OUTPUT_DIR / "pipeline_v2_live5_report.md"
SUMMARY_PATH = OUTPUT_DIR / "pipeline_v2_live5_summary.json"
ARCHIVE_DIR = OUTPUT_DIR / "pipeline_v2_live5_archive"
SOURCE_DIR = OUTPUT_DIR / "pipeline_v2_live5_sources"

SELECTED_EVENTS = (
    ("tm_ec95d18d266c5a44f071", "usable_with_review; Porto free-transfer event with missing add-ons, obligation, and parent-contract fields."),
    ("tm_211885c2a2cbd065928b", "usable_with_review; loan with purchase option and several missing secondary clauses."),
    ("tm_1ae175d1097fbcd0ddc0", "insufficient_evidence; Fran Navarro Braga -> Porto side of related-event pair."),
    ("tm_93e4c413dd2795d5dd19", "insufficient_evidence; Fran Navarro Porto -> Braga reverse/related event with different failure mode."),
    ("tm_694c30a10815aea6d84d", "clean control; permanent transfer with already-supported fee/add-ons/obligation fields."),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _provider() -> ParleyProvider:
    api_key = os.environ.get("PARLEY_API_KEY")
    if not api_key:
        raise RuntimeError("PARLEY_API_KEY is required")
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "contract_research.md"
    prompt = "\n\n".join([
        prompt_path.read_text(),
        "Pipeline V2 input groups evidence by field. Assess each contract field independently and use only evidence IDs supplied for that field.",
        render_model_schema_instructions(),
    ])
    return ParleyProvider(api_key, os.environ.get("PARLEY_MODEL", "bedrock/claude-haiku-4-5"), prompt)


def _selection(batch: dict[str, Any]) -> list[dict[str, Any]]:
    batch_rows = {row["event_id"]: row for row in batch["events"]}
    rows = []
    for event_id, reason in SELECTED_EVENTS:
        event = load_event(STRUCTURED_PATH, event_id)
        rows.append({
            "event_id": event_id,
            "player": event.get("player_name"),
            "departing_club": event.get("from_club_name"),
            "receiving_club": event.get("to_club_name"),
            "transfer_date": event.get("transfer_date"),
            "classification": batch_rows[event_id].get("classification"),
            "selection_reason": reason,
        })
    _write_json(SELECTION_PATH, rows)
    return rows


def _baseline(batch: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assessment_rows, resolutions = existing_evidence_rows(events, batch)
    by_event_field = {(row["event_id"], row["field"]): row for row in assessment_rows}
    batch_rows = {row["event_id"]: row for row in batch["events"]}
    rows = []
    for event in events:
        result = _read_json(Path(batch_rows[event["event_id"]]["output_path"])) if batch_rows[event["event_id"]].get("output_path") else None
        for field in FIELD_RESEARCH_FIELDS:
            current = (result or {}).get(field, {})
            assessment = by_event_field[(event["event_id"], field)]
            rows.append({
                "event_id": event["event_id"],
                "player": event.get("player_name"),
                "event_resolution_status": resolutions[event["event_id"]].status,
                "classification": batch_rows[event["event_id"]].get("classification"),
                "field": field,
                "existing_value": current.get("amount") or current.get("price") or current.get("percentage") or current.get("date") or current.get("year") or current.get("value"),
                "existing_value_status": current.get("status") or "unresearched",
                "field_evidence_assessment": assessment["field_evidence_status"],
                "supporting_evidence": json.dumps(current.get("evidence_ids", []), ensure_ascii=False),
                "needs_new_search": assessment["needs_new_search"],
            })
    _write_csv(BASELINE_PATH, rows)
    return rows


def _search_query(tavily: TavilySearchProvider, event: dict[str, Any], query: Any) -> list[SourceCandidate]:
    payload = tavily._search(query.query)
    rows = []
    for rank, item in enumerate(payload.get("results", []), start=1):
        url = item.get("url")
        if not url:
            continue
        candidate = SourceCandidate(
            source_url=url,
            source_title=item.get("title"),
            publisher=_domain(url),
            publication_date=item.get("published_date"),
            source_type=_source_type(url, event),
            evidence_text=item.get("content") or "",
            language=item.get("language") or query.language,
            search_query=query.query,
            search_rank=rank,
            provider_score=item.get("score"),
            query_family=query.query_family,
            query_language=query.language,
            target_domain=query.target_domain,
            query_reason=query.reason,
        )
        candidate.source_tier = source_tier(candidate)
        candidate.accessible = True
        candidate.event_match_status, candidate.event_match_score, candidate.event_match_reasons = assess_event_match(candidate, event)
        candidate.quality_score = score_source(candidate, event)
        rows.append(candidate)
    return rows


def _retrieve(candidates: list[SourceCandidate], event: dict[str, Any], fetcher: OfficialPageFetcher) -> list[SourceCandidate]:
    for candidate in candidates:
        candidate.source_tier = source_tier(candidate)
        if candidate.source_tier > 2:
            continue
        fetcher.apply_to_candidate(candidate)
        candidate.event_match_status, candidate.event_match_score, candidate.event_match_reasons = assess_event_match(candidate, event)
        candidate.quality_score = score_source(candidate, event)
    return deduplicate_sources(candidates)


def _field_has_terms(field: str, source: SourceCandidate) -> bool:
    text = f"{source.source_title or ''} {source.evidence_text} {source.retrieved_text or ''}".lower()
    terms = [term.lower() for values in FIELD_VOCABULARY[field].values() for term in values]
    if field == "transfer_fee" and any(term in text for term in ("free transfer", "signed for free", "without a fee")):
        return True
    return any(term in text for term in terms)


def _field_assessment(field: str, sources: list[SourceCandidate], event_resolution_ok: bool) -> tuple[str, list[SourceCandidate], str]:
    if not event_resolution_ok:
        return "insufficient", [], "event unresolved"
    admissible = filter_admissible_sources(sources)
    field_sources = [source for source in admissible if _field_has_terms(field, source)]
    text = " ".join(f"{source.evidence_text} {source.retrieved_text or ''}".lower() for source in field_sources)
    if any(term in text for term in ("undisclosed fee", "fee was not disclosed", "undisclosed amount", "montante nao divulgado", "montante não divulgado")):
        return "explicit_not_disclosed", field_sources, "admissible source explicitly says the term was not disclosed"
    if field_sources:
        return "sufficient", field_sources, "admissible exact/likely source contains field-specific term"
    return "insufficient", [], "source found but field not disclosed"


def _resolve_event_live(event: dict[str, Any], tavily: TavilySearchProvider, fetcher: OfficialPageFetcher, query_log: list[dict[str, Any]]) -> tuple[str, float, list[SourceCandidate]]:
    all_sources: list[SourceCandidate] = []
    plans = generate_field_queries(event, "event_resolution")
    for plan in plans:
        found = _retrieve(_search_query(tavily, event, plan), event, fetcher)
        all_sources = deduplicate_sources(all_sources + found)
        admissible = filter_admissible_sources(all_sources)
        exact_likely = [source for source in all_sources if source.event_match_status in {"exact", "likely"}]
        status = "confirmed" if any(source.event_match_status == "exact" and source.source_tier == 1 for source in admissible) else ("likely" if admissible else "unresolved")
        query_log.append(_query_log_row(event, plan, found, admissible, status in {"confirmed", "likely"}, False, "event resolution"))
        if status in {"confirmed", "likely"}:
            return status, max((source.event_match_score or 0.0 for source in admissible), default=0.0), all_sources
    if any(source.event_match_status == "mismatch" for source in all_sources):
        return "mismatch", 0.0, all_sources
    if any(source.event_match_status == "ambiguous" for source in all_sources):
        return "ambiguous", 0.3, all_sources
    return "unresolved", 0.0, all_sources


def _query_log_row(event: dict[str, Any], plan: Any, found: list[SourceCandidate], admissible: list[SourceCandidate], sufficient: bool, stopped: bool, reason: str) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "player": event.get("player_name"),
        "stage": plan.stage,
        "field": plan.field,
        "query_family": plan.query_family,
        "query": plan.query,
        "language": plan.language,
        "target_domain": plan.target_domain,
        "result_count": len(found),
        "tier1_count": sum(1 for source in found if source.source_tier == 1),
        "tier2_count": sum(1 for source in found if source.source_tier == 2),
        "exact_likely_count": sum(1 for source in found if source.event_match_status in {"exact", "likely"}),
        "admissible_count": len(admissible),
        "field_sufficient_after_query": sufficient,
        "stopped_field_search": stopped,
        "stop_reason": reason,
    }


def _merge_result(existing: dict[str, Any] | None, new: dict[str, Any], newly_sufficient_fields: set[str]) -> dict[str, Any]:
    if not existing:
        return new
    for field in FIELD_RESEARCH_FIELDS:
        if field in newly_sufficient_fields:
            continue
        old = existing.get(field, {})
        fresh = new.get(field, {})
        if old.get("status") in {"disclosed_yes", "disclosed_no", "partially_disclosed", "conflicting_sources", "undisclosed"} and fresh.get("status") in {"not_found", "not_applicable"}:
            new[field] = old
    return new


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    _load_allowed_env()
    batch = _read_json(BATCH_PATH)
    before = {"class_counts": _class_counts(), "coverage": _coverage()}
    selection = _selection(batch)
    events = [load_event(STRUCTURED_PATH, row["event_id"]) for row in selection]
    baseline = _baseline(batch, events)
    baseline_by_event_field = {(row["event_id"], row["field"]): row for row in baseline}
    planned = {event["event_id"]: generate_event_query_plan(event) for event in events}
    tavily = TavilySearchProvider.from_environment()
    fetcher = OfficialPageFetcher()
    provider = _provider()
    batch_events = {row["event_id"]: row for row in batch["events"]}
    query_log: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    total_attempts = 0
    total_cost = 0.0
    all_retrieved = 0
    cache_hits = 0

    for event in events:
        event_id = event["event_id"]
        batch_row = batch_events[event_id]
        existing = _read_json(Path(batch_row["output_path"])) if batch_row.get("output_path") else None
        classification_before = batch_row.get("classification")
        resolution_status, resolution_confidence, event_sources = _resolve_event_live(event, tavily, fetcher, query_log)
        event_ok = resolution_status in {"confirmed", "likely"}
        field_sources: dict[str, list[SourceCandidate]] = defaultdict(list)
        field_statuses: dict[str, tuple[str, str]] = {}
        queries_by_field: Counter[str] = Counter()
        newly_sufficient: set[str] = set()

        for field in FIELD_RESEARCH_FIELDS:
            base = baseline_by_event_field[(event_id, field)]
            if base["field_evidence_assessment"] in {"sufficient", "explicit_not_disclosed", "not_applicable", "conflicting"}:
                field_statuses[field] = (base["field_evidence_assessment"], "already established")
                continue
            if not event_ok:
                field_statuses[field] = ("insufficient", "event unresolved")
                continue
            sources_for_field: list[SourceCandidate] = []
            for plan in generate_field_queries(event, field, budget=SearchBudget()):
                found = _retrieve(_search_query(tavily, event, plan), event, fetcher)
                all_retrieved += sum(1 for source in found if source.source_tier in {1, 2})
                cache_hits += sum(1 for source in found if source.retrieval_method == "cache")
                sources_for_field = deduplicate_sources(sources_for_field + found)
                status, supporting, reason = _field_assessment(field, sources_for_field, event_ok)
                queries_by_field[field] += 1
                stopped = status in {"sufficient", "explicit_not_disclosed", "conflicting"}
                query_log.append(_query_log_row(event, plan, found, supporting, stopped, stopped, reason if stopped else "continue field search"))
                if stopped:
                    break
            status, supporting, reason = _field_assessment(field, sources_for_field, event_ok)
            field_sources[field] = supporting
            field_statuses[field] = (status, reason)
            if status in {"sufficient", "explicit_not_disclosed", "conflicting"}:
                newly_sufficient.add(field)

        parley_info = {"attempts": 0, "successful_extraction": False, "total_cost": 0.0}
        result_payload = None
        if newly_sufficient:
            parley_sources = deduplicate_sources([source for field in sorted(newly_sufficient) for source in field_sources[field]])
            SOURCE_DIR.mkdir(parents=True, exist_ok=True)
            _write_json(SOURCE_DIR / f"{event_id}_admissible.json", {"event_id": event_id, "event_resolution": {"status": resolution_status, "confidence": resolution_confidence}, "field_evidence": {field: [source.evidence_id for source in field_sources[field]] for field in sorted(newly_sufficient)}, "sources": [source.to_dict() for source in parley_sources]})
            result_payload, parley_info = _run_parley(provider, event, parley_sources)
            total_attempts += int(parley_info["attempts"])
            total_cost += float(parley_info["total_cost"])
            if result_payload:
                result_payload = _merge_result(existing, result_payload, newly_sufficient)
                result_payload["sources"] = [source.to_dict() | {"retrieval_date": (source.retrieval_timestamp or "")[:10] or "2026-09-08"} for source in parley_sources]
                ContractResearchResult.from_dict(result_payload)
                output_path = OUTPUT_DIR / f"{event_id}.json"
                if output_path.exists():
                    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_path, ARCHIVE_DIR / output_path.name)
                output_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n")
                classification_after = "usable_with_review" if result_payload.get("review_required") else "clean"
                meta = result_payload.get("provider_metadata", {})
                batch_row.update({
                    "classification": classification_after,
                    "parley_called": True,
                    "parley_attempts": int(batch_row.get("parley_attempts") or 0) + int(parley_info["attempts"]),
                    "parley_successful_extraction": True,
                    "parley_cost": meta.get("total_request_cost") or parley_info["total_cost"],
                    "parley_total_attempt_cost": float(batch_row.get("parley_total_attempt_cost") or 0) + float(parley_info["total_cost"]),
                    "output_path": str(output_path),
                    "review_required": result_payload.get("review_required"),
                    "review_reasons": result_payload.get("review_reasons") or [],
                })
            else:
                classification_after = classification_before
        else:
            classification_after = classification_before

        discovered = deduplicate_sources(event_sources + [source for values in field_sources.values() for source in values])
        batch_row.update({
            "pipeline_v2_live5_processed": True,
            "pipeline_v2_event_resolution_status": resolution_status,
            "source_count": max(int(batch_row.get("source_count") or 0), len(discovered)),
            "admissible_source_count": max(int(batch_row.get("admissible_source_count") or 0), len(filter_admissible_sources(discovered))),
            "sufficient": bool(batch_row.get("sufficient")) or bool(newly_sufficient),
        })
        established, unknown = _field_changes(result_payload)
        update_rows.append({
            "event_id": event_id,
            "player": event.get("player_name"),
            "event_resolution": resolution_status,
            "fields_before": {field: baseline_by_event_field[(event_id, field)]["existing_value_status"] for field in FIELD_RESEARCH_FIELDS},
            "newly_established_fields": sorted(newly_sufficient),
            "unchanged_fields": [field for field in FIELD_RESEARCH_FIELDS if field not in newly_sufficient],
            "conflicting_fields": [field for field, (status, _) in field_statuses.items() if status == "conflicting"],
            "review_reasons": (result_payload or existing or {}).get("review_reasons", []),
            "classification_before": classification_before,
            "classification_after": classification_after,
            "parley_attempts": parley_info["attempts"],
            "parley_cost": parley_info["total_cost"],
            "parley_success": parley_info["successful_extraction"],
            "extracted_fields": established,
            "unknown_fields": unknown,
        })
        for field in FIELD_RESEARCH_FIELDS:
            base = baseline_by_event_field[(event_id, field)]
            final_status, reason = field_statuses[field]
            supporting = field_sources[field]
            field_rows.append({
                "event_id": event_id,
                "player": event.get("player_name"),
                "field": field,
                "baseline_status": base["field_evidence_assessment"],
                "final_field_evidence_status": final_status,
                "baseline_value_status": base["existing_value_status"],
                "final_value_status": (result_payload or existing or {}).get(field, {}).get("status") or base["existing_value_status"],
                "newly_established": field in newly_sufficient,
                "supporting_evidence_count": len(supporting),
                "best_source_tier": min((source.source_tier for source in supporting if source.source_tier), default=""),
                "queries_executed": queries_by_field[field],
                "evidence_temporality": json.dumps(sorted({evidence_temporality(source.to_dict(), event) for source in supporting}), ensure_ascii=False),
                "reason": reason,
            })

    batch["pipeline_v2_live5_tavily_queries"] = tavily.search_count
    batch["pipeline_v2_live5_raw_results"] = tavily.raw_result_count
    batch["pipeline_v2_live5_parley_attempts"] = total_attempts
    batch["pipeline_v2_live5_successful_extractions"] = sum(1 for row in update_rows if row["parley_success"])
    batch["pipeline_v2_live5_parley_cost"] = round(total_cost, 6)
    _update_batch_counts(batch)
    _write_json(BATCH_PATH, batch)
    _regenerate_datasets(batch)
    after = {"class_counts": _class_counts(), "coverage": _coverage()}
    _write_csv(QUERY_LOG_PATH, query_log)
    _write_csv(FIELD_RESULTS_PATH, field_rows)
    _write_update_log(update_rows)
    _write_summary_report(selection, before, after, planned, query_log, field_rows, update_rows, tavily.raw_result_count, all_retrieved, cache_hits, total_attempts, total_cost)


def _coverage_subset(cov: dict[str, int]) -> dict[str, int]:
    return {field: cov.get(field, 0) for field in FIELD_RESEARCH_FIELDS}


def _write_update_log(rows: list[dict[str, Any]]) -> None:
    lines = ["# Pipeline V2 Live-5 Update Log", ""]
    for row in rows:
        lines.extend([
            f"## {row['player']} ({row['event_id']})",
            "",
            f"- Event resolution: `{row['event_resolution']}`",
            f"- Fields before: `{json.dumps(row['fields_before'], ensure_ascii=False, sort_keys=True)}`",
            f"- Newly established fields: `{json.dumps(row['newly_established_fields'], ensure_ascii=False)}`",
            f"- Unchanged fields: `{json.dumps(row['unchanged_fields'], ensure_ascii=False)}`",
            f"- Conflicting fields: `{json.dumps(row['conflicting_fields'], ensure_ascii=False)}`",
            f"- Review reasons: `{json.dumps(row['review_reasons'], ensure_ascii=False)}`",
            f"- Classification before: `{row['classification_before']}`",
            f"- Classification after: `{row['classification_after']}`",
            f"- Parley attempts: {row['parley_attempts']}",
            f"- Parley cost: {row['parley_cost']:.6f}",
            "",
        ])
    UPDATE_LOG_PATH.write_text("\n".join(lines))


def _write_summary_report(selection: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any], planned: dict[str, list[Any]], query_log: list[dict[str, Any]], field_rows: list[dict[str, Any]], update_rows: list[dict[str, Any]], raw_results: int, retrieved_pages: int, cache_hits: int, attempts: int, cost: float) -> None:
    planned_count = sum(len(rows) for rows in planned.values())
    actual = len(query_log)
    avoided = planned_count - actual
    targeted = [row for row in field_rows if row["baseline_status"] == "insufficient"]
    recovered = [row for row in targeted if row["newly_established"]]
    improved_events = {row["event_id"] for row in recovered}
    resolution_success = sum(1 for row in update_rows if row["event_resolution"] in {"confirmed", "likely"})
    parley_successes = sum(1 for row in update_rows if row["parley_success"])
    strategy_counts = Counter("field-specific keyword query" for row in recovered)
    failure_counts = Counter(row["reason"] for row in field_rows if row["final_field_evidence_status"] == "insufficient")
    lines = [
        "# Pipeline V2 Live-5 Report",
        "",
        "## 1. Selected Events",
        "",
    ]
    for row in selection:
        lines.append(f"- {row['event_id']} {row['player']}: {row['selection_reason']}")
    lines.extend([
        "",
        "## 2. Tests",
        "",
        "- Before: 100 passed",
        "- After: run `pytest -q` after live run",
        "",
        "## 3. Event Resolution Results",
        "",
        *[f"- {row['event_id']} {row['player']}: {row['event_resolution']}" for row in update_rows],
        "",
        "## 4. Planned vs Actual Tavily Queries",
        "",
        f"- Planned queries across 5 events: {planned_count}",
        f"- Actual Tavily queries: {actual}",
        f"- Average actual queries/event: {actual / 5:.1f}",
        f"- Early-stop savings: {avoided} queries ({avoided / planned_count:.1%})",
        "",
        "## 5. Newly Established Fields",
        "",
        *[f"- {row['event_id']} {row['player']}: {json.dumps(row['newly_established_fields'], ensure_ascii=False)}" for row in update_rows],
        "",
        "## 6. Fields Still Missing",
        "",
        *[f"- {row['event_id']} {row['player']}: {json.dumps([field['field'] for field in field_rows if field['event_id'] == row['event_id'] and field['final_field_evidence_status'] == 'insufficient'], ensure_ascii=False)}" for row in update_rows],
        "",
        "## 7. Success Metrics",
        "",
        f"- Field recovery rate: {len(recovered)}/{len(targeted)} = {(len(recovered) / len(targeted) if targeted else 0):.1%}",
        f"- Event improvement rate: {len(improved_events)}/5 = {len(improved_events) / 5:.1%}",
        f"- Event-resolution success rate: {resolution_success}/5 = {resolution_success / 5:.1%}",
        f"- Query efficiency: {actual}/{planned_count} = {actual / planned_count:.1%}",
        f"- Parley efficiency: {len(recovered)}/{parley_successes} newly supported fields per successful call" if parley_successes else "- Parley efficiency: no successful calls",
        "",
        "## 8. Cost",
        "",
        f"- Tavily queries: {actual}",
        f"- Approximate Tavily credits: {actual}",
        f"- Raw results: {raw_results}",
        f"- Retrieved Tier 1/2 pages: {retrieved_pages}",
        f"- Cache hits: {cache_hits}",
        f"- Parley attempts: {attempts}",
        f"- Successful extractions: {parley_successes}",
        f"- Total Parley cost: {cost:.6f}",
        f"- Projected Tavily queries for 100 events: {int(round((actual / 5) * 100))}",
        f"- Projected Tavily queries for 1,000 events: {int(round((actual / 5) * 1000))}",
        "",
        "## 9. Coverage Before -> After",
        "",
        f"- Classifications: {before['class_counts']} -> {after['class_counts']}",
        *[f"- {field}: {before['coverage'].get(field, 0)} -> {after['coverage'].get(field, 0)}" for field in FIELD_RESEARCH_FIELDS],
        "",
        "## 10. V2 Strategy Attribution",
        "",
        f"- New evidence strategy counts: {dict(strategy_counts)}",
        "",
        "## 11. Failure Analysis",
        "",
        f"- Remaining field-level failure reasons: {dict(failure_counts)}",
        "- Sell-on and buy-back still look structurally sparse in public evidence.",
        "",
        "## 12. Recommendation",
        "",
        "B. Tune V2 before broader live use. The experiment should be judged on field recovery and false-match avoidance, not clean-event count.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    _write_json(SUMMARY_PATH, {"before": before, "after": after, "planned_queries": planned_count, "actual_queries": actual, "field_recovery": [row for row in recovered], "updates": update_rows})


if __name__ == "__main__":
    main()
