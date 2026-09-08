from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .build_research_dataset import main as build_dataset_main
from .config import DEFAULT_OUTPUT_DIR
from .contract_schemas import render_model_schema_instructions
from .research_contract import ParleyProvider, ProviderFailure, load_event
from .source_discovery import (
    SourceCandidate,
    TavilySearchProvider,
    _check_accessibility,
    _domain,
    _source_type,
    assess_event_match,
    assess_source_sufficiency,
    build_query_plan,
    deduplicate_sources,
    filter_admissible_sources,
    source_tier,
    score_source,
)


SELECTED_EVENTS = {
    "tm_ec95d18d266c5a44f071": "Porto inbound free/permanent-looking case; old discovery found only Tier 3/weak evidence.",
    "tm_890a6e2a4b41a1f8dd46": "Portuguese-club Porto inbound case; good test for Portuguese query terms and official Porto domain.",
    "tm_1ae175d1097fbcd0ddc0": "Portuguese club-to-club Braga-to-Porto direction; tests paired official-domain queries.",
    "tm_211885c2a2cbd065928b": "Porto outbound to France; diversifies receiving club and may surface loan/permanent ambiguity.",
}


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
STRUCTURED_PATH = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"
BATCH_PATH = OUTPUT_DIR / "batch_20_results.json"
DISCOVERED_DIR = OUTPUT_DIR / "retry_4_sources" / "discovered"
ADMISSIBLE_DIR = OUTPUT_DIR / "retry_4_sources" / "admissible"
PREVIOUS_RESULTS_DIR = OUTPUT_DIR / "retry_4_previous_results"
ALLOWED_ENV_KEYS = {"TAVILY_API_KEY", "PARLEY_API_KEY"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _load_allowed_env(path: Path = Path(".pytest_cache/.env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in ALLOWED_ENV_KEYS and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _sources_from_artifact(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return payload
    return payload.get("sources", [])


def _analysis_rows() -> dict[str, dict[str, Any]]:
    with (DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis.csv").open(newline="") as handle:
        return {row["event_id"]: row for row in csv.DictReader(handle)}


def _batch_event_map(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {event["event_id"]: event for event in batch["events"]}


def _baseline_record(event_id: str, event: dict[str, Any], batch_row: dict[str, Any], analysis_row: dict[str, Any]) -> dict[str, Any]:
    discovered = _sources_from_artifact(DEFAULT_OUTPUT_DIR / "discovered_sources" / f"{event_id}.json")
    admissible = _sources_from_artifact(DEFAULT_OUTPUT_DIR / "discovered_sources" / "admissible" / f"{event_id}.json")
    return {
        "event_id": event_id,
        "player": event.get("player_name"),
        "transfer_date": event.get("transfer_date"),
        "departing_club": event.get("from_club_name"),
        "receiving_club": event.get("to_club_name"),
        "current_classification": batch_row.get("classification") or analysis_row.get("classification"),
        "existing_discovered_source_count": len(discovered),
        "existing_admissible_source_count": len(admissible),
        "current_sufficiency_result": bool(batch_row.get("sufficient")),
        "current_unresolved_issues": batch_row.get("sufficiency_reasons") or analysis_row.get("sufficiency_reasons"),
        "selection_reason": SELECTED_EVENTS[event_id],
    }


def _candidate_from_tavily_item(item: dict[str, Any], event: dict[str, Any], plan: Any, rank: int, provider: TavilySearchProvider) -> SourceCandidate | None:
    url = item.get("url")
    if not url:
        return None
    candidate = SourceCandidate(
        source_url=url,
        source_title=item.get("title"),
        publisher=_domain(url),
        publication_date=item.get("published_date"),
        source_type=_source_type(url, event),
        evidence_text=item.get("content") or "",
        language=item.get("language") or plan.language,
        search_query=plan.query,
        search_rank=rank,
        provider_score=item.get("score"),
        query_family=plan.family,
        query_language=plan.language,
        target_domain=plan.target_domain,
        query_reason=plan.reason,
    )
    candidate.accessible = _check_accessibility(url, provider.timeout)
    candidate.event_match_status, candidate.event_match_score, candidate.event_match_reasons = assess_event_match(candidate, event)
    candidate.source_tier = source_tier(candidate)
    candidate.quality_score = score_source(candidate, event)
    return candidate


def _discover_event(event: dict[str, Any], provider: TavilySearchProvider) -> tuple[list[SourceCandidate], list[SourceCandidate], list[dict[str, Any]], int]:
    all_candidates: list[SourceCandidate] = []
    query_log: list[dict[str, Any]] = []
    raw_results = 0
    for plan in build_query_plan(event):
        payload = provider._search(plan.query)
        results = payload.get("results", [])
        raw_results += len(results)
        query_candidates = [
            candidate
            for rank, item in enumerate(results, start=1)
            if (candidate := _candidate_from_tavily_item(item, event, plan, rank, provider)) is not None
        ]
        useful = [candidate for candidate in query_candidates if candidate.event_match_status in {"exact", "likely"}]
        admissible = filter_admissible_sources(query_candidates)
        query_log.append({
            "event_id": event["event_id"],
            "player": event.get("player_name"),
            "query": plan.query,
            "query_family": plan.family,
            "language": plan.language,
            "target_domain": plan.target_domain,
            "reason": plan.reason,
            "result_count": len(results),
            "useful_result_count": len(useful),
            "admissible_result_count": len(admissible),
        })
        all_candidates.extend(query_candidates)
    discovered = deduplicate_sources(all_candidates)
    admissible = filter_admissible_sources(discovered)
    return discovered, admissible, query_log, raw_results


def _write_source_artifacts(event: dict[str, Any], discovered: list[SourceCandidate], admissible: list[SourceCandidate], sufficient: bool, reasons: list[str]) -> None:
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
    _write_json(DISCOVERED_DIR / f"{event['event_id']}.json", {**common, "sources": [candidate.to_dict() for candidate in discovered]})
    _write_json(ADMISSIBLE_DIR / f"{event['event_id']}.json", {**common, "sources": [candidate.to_dict() for candidate in admissible]})


def _compare(event_id: str, baseline: dict[str, Any], discovered: list[SourceCandidate], admissible: list[SourceCandidate], sufficient: bool) -> dict[str, Any]:
    tiers = Counter(candidate.source_tier for candidate in discovered)
    admissible_tiers = Counter(candidate.source_tier for candidate in admissible)
    exact = sum(1 for candidate in discovered if candidate.event_match_status == "exact")
    likely = sum(1 for candidate in discovered if candidate.event_match_status == "likely")
    event_identity_improved = exact + likely > 0
    admissible_evidence_improved = len(admissible) > int(baseline["existing_admissible_source_count"])
    evidence_sufficiency_converted = sufficient and not bool(baseline["current_sufficiency_result"])
    return {
        "event_id": event_id,
        "player_name": baseline["player"],
        "old_discovered_sources": baseline["existing_discovered_source_count"],
        "new_discovered_sources": len(discovered),
        "old_admissible_sources": baseline["existing_admissible_source_count"],
        "new_admissible_sources": len(admissible),
        "old_sufficient": baseline["current_sufficiency_result"],
        "new_sufficient": sufficient,
        "new_exact_match_sources": exact,
        "new_likely_match_sources": likely,
        "new_tier1_sources": tiers.get(1, 0),
        "new_tier2_sources": tiers.get(2, 0),
        "new_admissible_tier1_sources": admissible_tiers.get(1, 0),
        "new_admissible_tier2_sources": admissible_tiers.get(2, 0),
        "event_identity_improved": event_identity_improved,
        "admissible_evidence_improved": admissible_evidence_improved,
        "evidence_sufficiency_converted": evidence_sufficiency_converted,
        "improved": event_identity_improved or admissible_evidence_improved or evidence_sufficiency_converted,
    }


def _write_comparison_csv(rows: list[dict[str, Any]]) -> None:
    path = OUTPUT_DIR / "retry_4_discovery_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _provider() -> ParleyProvider:
    api_key = os.environ.get("PARLEY_API_KEY")
    if not api_key:
        raise RuntimeError("PARLEY_API_KEY is required for sufficient retry cases")
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "contract_research.md"
    return ParleyProvider(api_key, os.environ.get("PARLEY_MODEL", "bedrock/claude-haiku-4-5"), prompt_path.read_text() + "\n\n" + render_model_schema_instructions())


def _cost_from_metadata(result: dict[str, Any] | None) -> float:
    if not result:
        return 0.0
    value = result.get("provider_metadata", {}).get("total_request_cost")
    return float(value or 0)


def _cost_from_diagnostics(diagnostics: dict[str, Any] | None) -> float:
    if not diagnostics:
        return 0.0
    value = diagnostics.get("total_request_cost")
    if value is not None:
        return float(value or 0)
    header = diagnostics.get("parley_cost_header")
    if isinstance(header, str):
        try:
            return float(header.removeprefix("$"))
        except ValueError:
            return 0.0
    return 0.0


def retry_parley_metrics(update_rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = sum(int(row.get("parley_info", {}).get("attempts") or 0) for row in update_rows)
    successes = sum(1 for row in update_rows if row.get("parley_info", {}).get("successful_extraction"))
    total_cost = sum(float(row.get("parley_info", {}).get("total_attempt_cost") or 0) for row in update_rows)
    return {
        "attempts": attempts,
        "successful_extractions": successes,
        "total_cost": round(total_cost, 6),
    }


def retry_improvement_metrics(rows: list[dict[str, Any]], attempted: int | None = None) -> dict[str, Any]:
    denominator = attempted if attempted is not None else len(rows)
    if denominator == 0:
        return {
            "attempted": 0,
            "event_identity_improved": 0,
            "admissible_evidence_improved": 0,
            "evidence_sufficiency_converted": 0,
            "contract_field_improved": 0,
        }
    event_identity = sum(1 for row in rows if int(row.get("new_exact_match_sources") or 0) + int(row.get("new_likely_match_sources") or 0) > 0)
    admissible = sum(1 for row in rows if int(row.get("new_admissible_sources") or 0) > int(row.get("old_admissible_sources") or 0))
    sufficient = sum(1 for row in rows if _truthy(row.get("new_sufficient")) and not _truthy(row.get("old_sufficient")))
    fields = sum(1 for row in rows if row.get("fields_newly_established"))
    return {
        "attempted": denominator,
        "event_identity_improved": event_identity,
        "admissible_evidence_improved": admissible,
        "evidence_sufficiency_converted": sufficient,
        "contract_field_improved": fields,
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _run_parley(event: dict[str, Any], admissible: list[SourceCandidate]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    provider = _provider()
    diagnostics: dict[str, Any] = {"called": True, "error": None, "attempts": 0, "attempt_costs": []}
    for attempt in range(1, 3):
        try:
            result = provider.research(event, admissible)
            result_payload = result.to_dict()
            attempt_costs = diagnostics["attempt_costs"] + [_cost_from_metadata(result_payload)]
            output_path = OUTPUT_DIR / f"{event['event_id']}.json"
            if output_path.exists():
                PREVIOUS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                (PREVIOUS_RESULTS_DIR / f"{event['event_id']}.json").write_text(output_path.read_text())
            output_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n")
            return result_payload, {
                **diagnostics,
                "attempts": attempt,
                "attempt_costs": attempt_costs,
                "total_attempt_cost": round(sum(attempt_costs), 6),
                "successful_extraction": True,
                "output_path": str(output_path),
            }
        except ProviderFailure as error:
            attempt_costs = diagnostics["attempt_costs"] + [_cost_from_diagnostics(error.diagnostics)]
            diagnostics = {
                **diagnostics,
                "attempts": attempt,
                "attempt_costs": attempt_costs,
                "total_attempt_cost": round(sum(attempt_costs), 6),
                "successful_extraction": False,
                "error": str(error),
                "diagnostics": error.diagnostics,
            }
            if attempt == 2:
                return None, diagnostics
    return None, diagnostics


def _update_batch_counts(batch: dict[str, Any]) -> None:
    counts = Counter(event["classification"] for event in batch["events"])
    batch["clean"] = counts.get("clean", 0)
    batch["usable_with_review"] = counts.get("usable_with_review", 0)
    batch["insufficient_evidence"] = counts.get("insufficient_evidence", 0)
    batch["invalid"] = counts.get("invalid", 0)
    batch["sufficient_source_events"] = sum(1 for event in batch["events"] if event.get("sufficient"))
    batch["parley_calls"] = sum(1 for event in batch["events"] if event.get("parley_called"))
    batch["total_parley_cost"] = round(sum(float(event.get("parley_cost") or 0) for event in batch["events"]), 6)
    batch["average_parley_cost_per_call"] = round(batch["total_parley_cost"] / batch["parley_calls"], 6) if batch["parley_calls"] else 0
    batch["total_tavily_searches"] = sum(int(event.get("tavily_search_count") or 0) for event in batch["events"])


def _field_summary(result: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not result:
        return [], []
    established = []
    unknown = []
    for field_name, field in result.items():
        if not isinstance(field, dict) or "status" not in field:
            continue
        status = field.get("status")
        value = field.get("amount") or field.get("price") or field.get("percentage") or field.get("date") or field.get("year") or field.get("value") or field.get("description")
        if status in {"disclosed_yes", "disclosed_no", "partially_disclosed", "conflicting_sources", "undisclosed"} and value is not None:
            established.append(f"{field_name}: {value} ({status})")
        elif status in {"not_found", "undisclosed", "not_applicable"}:
            unknown.append(f"{field_name}: {status}")
    return established, unknown


def main() -> None:
    _load_allowed_env()
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        raise SystemExit("TAVILY_API_KEY is required for retry_4 live discovery")

    batch = _read_json(BATCH_PATH)
    batch_events = _batch_event_map(batch)
    analysis = _analysis_rows()
    baselines = []
    events = {}
    for event_id in SELECTED_EVENTS:
        event = load_event(STRUCTURED_PATH, event_id)
        events[event_id] = event
        baselines.append(_baseline_record(event_id, event, batch_events[event_id], analysis[event_id]))
    _write_json(OUTPUT_DIR / "retry_4_baseline.json", baselines)

    provider = TavilySearchProvider(tavily_key)
    search_log: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for baseline in baselines:
        event_id = baseline["event_id"]
        event = events[event_id]
        before_classification = batch_events[event_id].get("classification")
        discovered, admissible, query_log, raw_results = _discover_event(event, provider)
        sufficient, sufficiency_reasons = assess_source_sufficiency(admissible)
        _write_source_artifacts(event, discovered, admissible, sufficient, sufficiency_reasons)
        search_log.extend(query_log)
        comparison = _compare(event_id, baseline, discovered, admissible, sufficient)
        comparisons.append(comparison)

        parley_result = None
        parley_info: dict[str, Any] = {"called": False}
        classification_after = before_classification
        if sufficient:
            exact_likely_admissible = [candidate for candidate in admissible if candidate.event_match_status in {"exact", "likely"}]
            parley_result, parley_info = _run_parley(event, exact_likely_admissible)
            if parley_result:
                classification_after = "usable_with_review" if parley_result.get("review_required") else "clean"
            else:
                classification_after = "invalid"

        established, unknown = _field_summary(parley_result)
        batch_row = batch_events[event_id]
        batch_row.update({
            "tavily_search_count": len(build_query_plan(event)),
            "raw_result_count": raw_results,
            "source_count": len(discovered),
            "admissible_source_count": len(admissible),
            "sufficient": sufficient,
            "sufficiency_reasons": sufficiency_reasons,
            "classification": classification_after,
            "parley_called": bool(parley_info.get("called") and parley_result),
            "parley_attempts": int(parley_info.get("attempts") or 0),
            "parley_successful_extraction": bool(parley_info.get("successful_extraction")),
            "parley_cost": _cost_from_metadata(parley_result),
            "parley_total_attempt_cost": float(parley_info.get("total_attempt_cost") or 0),
            "output_path": parley_info.get("output_path") if parley_result else batch_row.get("output_path"),
            "review_required": (parley_result or {}).get("review_required"),
            "review_reasons": (parley_result or {}).get("review_reasons") or batch_row.get("review_reasons", []),
        })
        if parley_result and parley_result.get("provider_metadata"):
            meta = parley_result["provider_metadata"]
            batch_row.update({
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "total_tokens": meta.get("total_tokens"),
            })

        update_rows.append({
            "event_id": event_id,
            "player": event.get("player_name"),
            "sufficiency_changed": baseline["current_sufficiency_result"] != sufficient,
            "parley_called": batch_row["parley_called"],
            "fields_newly_established": established,
            "fields_still_unknown": unknown,
            "classification_before": before_classification,
            "classification_after": classification_after,
            "parley_info": parley_info,
        })
        report_rows.append({
            **comparison,
            "raw_results": raw_results,
            "parley_called": batch_row["parley_called"],
            "parley_cost": batch_row.get("parley_cost") or 0,
            "classification_before": before_classification,
            "classification_after": classification_after,
            "fields_newly_established": established,
        })

    batch["retry_4_tavily_queries"] = len(search_log)
    batch["retry_4_raw_results"] = sum(row["result_count"] for row in search_log)
    parley_metrics = retry_parley_metrics(update_rows)
    batch["retry_4_parley_attempts"] = parley_metrics["attempts"]
    batch["retry_4_successful_parley_extractions"] = parley_metrics["successful_extractions"]
    batch["retry_4_parley_calls"] = parley_metrics["successful_extractions"]
    batch["retry_4_parley_cost"] = parley_metrics["total_cost"]
    _update_batch_counts(batch)
    _write_json(BATCH_PATH, batch)
    _write_json(OUTPUT_DIR / "retry_4_search_log.json", search_log)
    _write_comparison_csv(comparisons)
    _write_update_log(update_rows)
    _write_report(baselines, search_log, report_rows)
    build_dataset_main()


def _write_update_log(rows: list[dict[str, Any]]) -> None:
    lines = ["# Retry 4 Update Log", ""]
    for row in rows:
        lines.extend([
            f"## {row['player']} ({row['event_id']})",
            "",
            f"- Sufficiency changed: `{row['sufficiency_changed']}`",
            f"- Parley called: `{row['parley_called']}`",
            f"- Classification before: `{row['classification_before']}`",
            f"- Classification after: `{row['classification_after']}`",
            f"- Fields newly established: {_json_list(row['fields_newly_established'])}",
            f"- Fields still unknown: {_json_list(row['fields_still_unknown'])}",
            "",
        ])
    (OUTPUT_DIR / "retry_4_update_log.md").write_text("\n".join(lines))


def _write_report(baselines: list[dict[str, Any]], search_log: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    metrics = retry_improvement_metrics(rows)
    sufficient = metrics["evidence_sufficiency_converted"]
    identity_improved = metrics["event_identity_improved"]
    admissible_improved = metrics["admissible_evidence_improved"]
    field_improved = metrics["contract_field_improved"]
    parley_calls = sum(1 for row in rows if row["parley_called"])
    total_cost = sum(float(row["parley_cost"] or 0) for row in rows)
    recommendation = "A. Retry all remaining insufficient cases now" if sufficient >= 2 and admissible_improved >= 3 else (
        "C. Use a mixed strategy: retry only cases matching patterns that succeeded" if admissible_improved >= 2 else
        "B. Improve source discovery further before retrying all"
    )
    lines = [
        "# Retry 4 Discovery Report",
        "",
        "## Selected Cases",
        "",
    ]
    for baseline in baselines:
        lines.append(f"- `{baseline['event_id']}` {baseline['player']}: {baseline['selection_reason']}")
    lines.extend([
        "",
        "## Aggregate Results",
        "",
        f"- Total Tavily queries: {len(search_log)}",
        f"- Raw results: {sum(row['result_count'] for row in search_log)}",
        f"- New Tier 1 sources: {sum(row['new_tier1_sources'] for row in rows)}",
        f"- New Tier 2 sources: {sum(row['new_tier2_sources'] for row in rows)}",
        f"- Exact/likely event matches: {sum(row['new_exact_match_sources'] + row['new_likely_match_sources'] for row in rows)}",
        f"- Cases that became sufficient: {sufficient}",
        f"- Cases sent to Parley: {parley_calls}",
        f"- Parley calls: {parley_calls}",
        f"- Total Parley cost: {total_cost:.6f}",
        f"- Event-identity improvement rate: {identity_improved}/4 = {identity_improved / 4:.2%}",
        f"- Admissible-evidence improvement rate: {admissible_improved}/4 = {admissible_improved / 4:.2%}",
        f"- Evidence-sufficiency conversion rate: {sufficient}/4 = {sufficient / 4:.2%}",
        f"- Contract-field improvement rate: {field_improved}/4 = {field_improved / 4:.2%}",
        f"- Recommendation: {recommendation}",
        "",
        "## Case Results",
        "",
    ])
    for row in rows:
        lines.extend([
            f"### {row['player_name']} ({row['event_id']})",
            "",
            f"- Old vs new admissible sources: {row['old_admissible_sources']} -> {row['new_admissible_sources']}",
            f"- Sufficiency: {row['old_sufficient']} -> {row['new_sufficient']}",
            f"- Classifications: `{row['classification_before']}` -> `{row['classification_after']}`",
            f"- New exact/likely matches: {row['new_exact_match_sources'] + row['new_likely_match_sources']}",
            f"- Newly established fields: {_json_list(row['fields_newly_established'])}",
            "",
        ])
    (OUTPUT_DIR / "retry_4_report.md").write_text("\n".join(lines))


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False) if values else "[]"


if __name__ == "__main__":
    main()
