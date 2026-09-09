from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .official_page_retrieval import OfficialPageFetcher
from .pipeline_v2 import (
    EventResolution,
    FIELD_RESEARCH_FIELDS,
    SearchBudget,
    generate_event_query_plan,
    link_evidence_to_event,
)
from .pipeline_v21_diagnostics import field_signal, load_saved_sources
from .pipeline_v2_live5 import _retrieve, _search_query
from .research_contract import load_event
from .retry_4_discovery import _load_allowed_env
from .source_discovery import SourceCandidate, deduplicate_sources, filter_admissible_sources, TavilySearchProvider


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
STRUCTURED_PATH = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"
BATCH_PATH = OUTPUT_DIR / "batch_20_results.json"
SELECTION_PATH = OUTPUT_DIR / "pipeline_v21_live3_selection.json"
BASELINE_PATH = OUTPUT_DIR / "pipeline_v21_live3_baseline.csv"
QUERY_LOG_PATH = OUTPUT_DIR / "pipeline_v21_live3_query_log.csv"
FIELD_COMPARISON_PATH = OUTPUT_DIR / "pipeline_v21_live3_field_comparison.csv"
REPORT_PATH = OUTPUT_DIR / "pipeline_v21_live3_report.md"
SOURCE_DIR = OUTPUT_DIR / "pipeline_v21_live3_sources"
REGISTRY_CANDIDATES_PATH = OUTPUT_DIR / "pipeline_v21_registry_candidates.md"

EVENT_SELECTION = (
    ("tm_ec95d18d266c5a44f071", "Luuk de Jong", "relatively clean permanent/free-control style case with missing secondary fields"),
    ("tm_211885c2a2cbd065928b", "Danny Namaso", "loan/options case with missing secondary clause fields"),
    ("tm_1ae175d1097fbcd0ddc0", "Fran Navarro", "difficult related-event direction case; tests reverse contamination safeguards"),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_registry_candidates() -> None:
    audit = pd.read_csv(OUTPUT_DIR / "pipeline_v21_source_registry_audit.csv")
    candidates = audit[audit["recommended_action"] == "review_for_possible_tier2"]
    lines = ["# Pipeline V2.1 Registry Candidates", ""]
    for _, row in candidates.iterrows():
        domain = row["domain"]
        examples = []
        for event_id in json.loads(row["events_seen"]):
            for source in load_saved_sources(event_id):
                if source.source_url and domain in source.source_url:
                    examples.append(source)
        lines.extend([
            f"## {domain}",
            "",
            f"- Country: {row.get('country') if not pd.isna(row.get('country')) else 'unknown'}",
            f"- Current tier: {row['current_tier']}",
            f"- Frequency in saved Tavily results: {row['result_frequency']}",
            f"- Events: `{row['events_seen']}`",
            f"- Exact/likely matches: {row['exact_likely_frequency']}",
            f"- Contract-field signals: {row['field_signal_frequency']}",
            "- Example saved artifacts:",
        ])
        for source in examples[:3]:
            snippet = " ".join((source.evidence_text or source.retrieved_text or "")[:260].split())
            lines.append(f"  - {source.source_title or '(untitled)'} | {source.source_url} | {snippet}")
        lines.extend([
            "- Why it might qualify as Tier 2: saved results include event and field signals, so it may contain football reporting that V1/V2 did not classify.",
            "- Why it might not: current audit has not established editorial standards, independence, or reporting reliability; frequency is not enough.",
            "- Recommendation: needs_human_review",
            "",
        ])
    REGISTRY_CANDIDATES_PATH.write_text("\n".join(lines) + "\n")


def write_selection(batch: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {row["event_id"]: row for row in batch["events"]}
    rows = []
    for event_id, label, reason in EVENT_SELECTION:
        event = load_event(STRUCTURED_PATH, event_id)
        rows.append({
            "event_id": event_id,
            "player": event.get("player_name") or label,
            "departing_club": event.get("from_club_name"),
            "receiving_club": event.get("to_club_name"),
            "transfer_date": event.get("transfer_date"),
            "classification": by_id[event_id].get("classification"),
            "selection_reason": reason,
        })
    _write_json(SELECTION_PATH, rows)
    return rows


def existing_field_statuses(batch_row: dict[str, Any]) -> dict[str, str]:
    if not batch_row.get("output_path"):
        return {field: "unresearched" for field in FIELD_RESEARCH_FIELDS}
    result = _read_json(Path(batch_row["output_path"]))
    return {field: result.get(field, {}).get("status") or "unresearched" for field in FIELD_RESEARCH_FIELDS}


def fields_needing_search(statuses: dict[str, str]) -> list[str]:
    return [field for field, status in statuses.items() if status in {"not_found", "unresearched"}]


def write_baseline(selection: list[dict[str, Any]], batch: dict[str, Any]) -> list[dict[str, Any]]:
    live5_queries = pd.read_csv(OUTPUT_DIR / "pipeline_v2_live5_query_log.csv")
    live5_fields = pd.read_csv(OUTPUT_DIR / "pipeline_v2_live5_field_results.csv")
    live5_summary = _read_json(OUTPUT_DIR / "pipeline_v2_live5_summary.json")
    live5_resolution = {row["event_id"]: row["event_resolution"] for row in live5_summary["updates"]}
    replay = pd.read_csv(OUTPUT_DIR / "pipeline_v21_offline_replay.csv")
    by_id = {row["event_id"]: row for row in batch["events"]}
    rows = []
    for selected in selection:
        event_id = selected["event_id"]
        statuses = existing_field_statuses(by_id[event_id])
        q = live5_queries[live5_queries["event_id"] == event_id]
        f = live5_fields[live5_fields["event_id"] == event_id]
        r = replay[replay["event_id"] == event_id]
        rows.append({
            "event_id": event_id,
            "player": selected["player"],
            "classification": selected["classification"],
            "v2_event_resolution_status": live5_resolution.get(event_id, "unknown"),
            "v21_offline_event_link_status": json.dumps(sorted(set(str(value) for value in r["event_link_status"].dropna() if str(value))), ensure_ascii=False),
            "existing_field_statuses": json.dumps(statuses, ensure_ascii=False, sort_keys=True),
            "fields_needing_search": json.dumps(fields_needing_search(statuses), ensure_ascii=False),
            "old_v2_executed_query_count": len(q),
            "old_v2_tier12_result_count": int(q["tier1_count"].sum() + q["tier2_count"].sum()),
            "old_v2_admissible_count": int(q["admissible_count"].sum()),
            "old_v2_field_sufficient_count": int(f["newly_established"].sum()),
        })
    _write_csv(BASELINE_PATH, rows)
    return rows


def _query_round(plan: Any, executed_index: int) -> str:
    if plan.field == "event_resolution":
        return "event_resolution"
    return {1: "round_1", 2: "round_2"}.get(executed_index, "round_3")


def _resolve_event(event: dict[str, Any], tavily: TavilySearchProvider, fetcher: OfficialPageFetcher) -> tuple[EventResolution, list[SourceCandidate], list[dict[str, Any]]]:
    sources: list[SourceCandidate] = []
    logs = []
    status = "unresolved"
    confidence = 0.0
    for plan in [item for item in generate_event_query_plan(event, set()) if item.field == "event_resolution"]:
        found = _retrieve(_search_query(tavily, event, plan), event, fetcher)
        sources = deduplicate_sources(sources + found)
        admissible = filter_admissible_sources(sources)
        if any(source.event_match_status == "exact" for source in admissible):
            status = "confirmed"
        elif admissible:
            status = "likely"
        elif any(source.event_match_status == "mismatch" for source in sources):
            status = "mismatch"
        elif any(source.event_match_status == "ambiguous" for source in sources):
            status = "ambiguous"
        confidence = max((source.event_match_score or 0 for source in admissible), default=0.0)
        logs.append(query_log_row(event, plan, "event_resolution", found, [], EventResolution(str(event["event_id"]), status, confidence), "event resolution"))
        if status in {"confirmed", "likely"}:
            break
    return EventResolution(str(event["event_id"]), status, confidence), sources, logs


def assess_linked_field(field: str, sources: list[SourceCandidate], event: dict[str, Any], resolution: EventResolution) -> tuple[str, list[SourceCandidate], str, Counter[str]]:
    link_counts: Counter[str] = Counter()
    supporting = []
    for source in sources:
        if source.source_tier not in {1, 2} or not field_signal(field, source):
            continue
        link = link_evidence_to_event(source, resolution, event)
        link_counts[link.status] += 1
        if link.status in {"direct_match", "anchored_match"}:
            supporting.append(source)
    if resolution.status not in {"confirmed", "likely"}:
        return "insufficient", [], "event unresolved", link_counts
    if supporting:
        return "sufficient", supporting, "direct or safely anchored Tier 1/2 field signal", link_counts
    if link_counts["mismatch"] or link_counts["related_event_only"]:
        return "insufficient", [], "false or related-event match prevented", link_counts
    return "insufficient", [], "no safely linked field evidence", link_counts


def query_log_row(event: dict[str, Any], plan: Any, round_name: str, found: list[SourceCandidate], supporting: list[SourceCandidate], resolution: EventResolution, stop_reason: str) -> dict[str, Any]:
    links = [link_evidence_to_event(source, resolution, event) for source in found if source.source_tier in {1, 2}]
    signals = [source for source in found if source.source_tier in {1, 2} and plan.field != "event_resolution" and field_signal(plan.field, source)]
    return {
        "event_id": event["event_id"],
        "player": event.get("player_name"),
        "field": plan.field,
        "round": round_name,
        "query_family": plan.query_family,
        "query": plan.query,
        "language": plan.language,
        "target_domain": plan.target_domain,
        "raw_results": len(found),
        "tier1_results": sum(1 for source in found if source.source_tier == 1),
        "tier2_results": sum(1 for source in found if source.source_tier == 2),
        "pages_retrieved": sum(1 for source in found if source.source_tier in {1, 2} and source.retrieval_status == "success"),
        "direct_matches": sum(1 for link in links if link.status == "direct_match"),
        "anchored_matches": sum(1 for link in links if link.status == "anchored_match"),
        "related_event_only": sum(1 for link in links if link.status == "related_event_only"),
        "ambiguous": sum(1 for link in links if link.status == "ambiguous"),
        "mismatch": sum(1 for link in links if link.status == "mismatch"),
        "field_signal_found": bool(signals),
        "field_sufficient_after_query": bool(supporting),
        "stop_reason": stop_reason,
    }


def run_live3() -> None:
    _load_allowed_env()
    batch = _read_json(BATCH_PATH)
    selection = write_selection(batch)
    baseline = write_baseline(selection, batch)
    baseline_by_event = {row["event_id"]: row for row in baseline}
    tavily = TavilySearchProvider.from_environment()
    fetcher = OfficialPageFetcher()
    query_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    for selected in selection:
        event = load_event(STRUCTURED_PATH, selected["event_id"])
        statuses = json.loads(baseline_by_event[event["event_id"]]["existing_field_statuses"])
        established = {field for field, status in statuses.items() if status not in {"not_found", "unresearched"}}
        planned = generate_event_query_plan(event, established, budget=SearchBudget(event_total_max_queries=18))
        resolution, event_sources, resolution_logs = _resolve_event(event, tavily, fetcher)
        query_rows.extend(resolution_logs)
        sources_by_field: dict[str, list[SourceCandidate]] = defaultdict(list)
        field_status: dict[str, tuple[str, list[SourceCandidate], str, Counter[str], int]] = {}
        if resolution.status in {"confirmed", "likely"}:
            for field in FIELD_RESEARCH_FIELDS:
                if field in established:
                    field_status[field] = (statuses[field], [], "already established", Counter(), 0)
                    continue
                executed = 0
                field_plans = [plan for plan in planned if plan.field == field]
                for plan in field_plans:
                    executed += 1
                    found = _retrieve(_search_query(tavily, event, plan), event, fetcher)
                    sources_by_field[field] = deduplicate_sources(sources_by_field[field] + found)
                    status, supporting, reason, link_counts = assess_linked_field(field, sources_by_field[field], event, resolution)
                    stopped = status == "sufficient"
                    query_rows.append(query_log_row(event, plan, _query_round(plan, executed), found, supporting, resolution, "field sufficient" if stopped else reason))
                    if stopped:
                        break
                field_status[field] = assess_linked_field(field, sources_by_field[field], event, resolution) + (executed,)
        else:
            for field in FIELD_RESEARCH_FIELDS:
                field_status[field] = ("insufficient", [], "event unresolved", Counter(), 0)

        all_sources = deduplicate_sources(event_sources + [source for sources in sources_by_field.values() for source in sources])
        _write_json(SOURCE_DIR / f"{event['event_id']}.json", {
            "event_id": event["event_id"],
            "event_resolution": resolution.to_dict(),
            "sources": [source.to_dict() for source in all_sources],
        })
        old_live5_fields = pd.read_csv(OUTPUT_DIR / "pipeline_v2_live5_field_results.csv")
        offline = pd.read_csv(OUTPUT_DIR / "pipeline_v21_offline_replay.csv")
        for field in FIELD_RESEARCH_FIELDS:
            status, supporting, reason, link_counts, executed = field_status[field]
            v2_status = old_live5_fields[(old_live5_fields.event_id == event["event_id"]) & (old_live5_fields.field == field)]["final_field_evidence_status"].iloc[0]
            pre_live = offline[(offline.event_id == event["event_id"]) & (offline.field == field)]["v21_status"].iloc[0]
            link_statuses = [link_evidence_to_event(source, resolution, event).status for source in supporting]
            field_rows.append({
                "event_id": event["event_id"],
                "player": event.get("player_name"),
                "field": field,
                "v2_status": v2_status,
                "v21_pre_live_status": pre_live,
                "v21_post_live_status": "sufficient" if status == "sufficient" else status,
                "newly_sufficient": v2_status == "insufficient" and status == "sufficient",
                "supporting_sources": json.dumps([source.source_url for source in supporting], ensure_ascii=False),
                "best_source_tier": min((source.source_tier for source in supporting if source.source_tier), default=""),
                "event_link_status": json.dumps(sorted(set(link_statuses)), ensure_ascii=False),
                "queries_executed": executed,
                "reason": reason,
            })
    _write_csv(QUERY_LOG_PATH, query_rows)
    _write_csv(FIELD_COMPARISON_PATH, field_rows)
    write_report(selection, baseline, query_rows, field_rows)


def write_report(selection: list[dict[str, Any]], baseline: list[dict[str, Any]], query_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> None:
    q = pd.DataFrame(query_rows)
    f = pd.DataFrame(field_rows)
    planned = sum(len(generate_event_query_plan(load_event(STRUCTURED_PATH, row["event_id"]), set(), budget=SearchBudget(event_total_max_queries=18))) for row in selection)
    actual = len(q)
    newly = f[f["newly_sufficient"] == True]
    false_prevented = int(q["mismatch"].sum() + q["related_event_only"].sum())
    lines = [
        "# Pipeline V2.1 Live3 Report",
        "",
        "## Registry Candidates",
        "",
        "See `pipeline_v21_registry_candidates.md`. Both candidates remain `needs_human_review`; neither was auto-promoted.",
        "",
        "## Selected Events",
        "",
    ]
    for row in selection:
        lines.append(f"- {row['event_id']} {row['player']}: {row['selection_reason']}")
    lines.extend([
        "",
        "## Tests",
        "",
        "- Before: 109 passed",
        "- After: pending",
        "",
        "## Tavily Usage",
        "",
        f"- Total Tavily queries: {actual}",
        f"- Actual query counts/event: {q.groupby('event_id').size().to_dict()}",
        f"- Planned queries avoided by early stopping/gating: {planned - actual} ({((planned - actual) / planned if planned else 0):.1%})",
        f"- Raw results: {int(q['raw_results'].sum())}",
        f"- Tier 1 results: {int(q['tier1_results'].sum())}",
        f"- Tier 2 results: {int(q['tier2_results'].sum())}",
        f"- Pages retrieved: {int(q['pages_retrieved'].sum())}",
        "",
        "## Direct vs Anchored Matches",
        "",
        f"- Direct matches: {int(q['direct_matches'].sum())}",
        f"- Anchored matches: {int(q['anchored_matches'].sum())}",
        f"- Related-event-only matches: {int(q['related_event_only'].sum())}",
        f"- Ambiguous/mismatch results: {int(q['ambiguous'].sum() + q['mismatch'].sum())}",
        "",
        "## Field Evidence",
        "",
        f"- Newly sufficient fields: {len(newly)}",
        f"- Parley-ready fields: {newly[['event_id', 'field']].to_dict('records') if not newly.empty else []}",
        f"- False matches prevented: {false_prevented}",
        "",
        "## V2 vs V2.1",
        "",
    ])
    for row in baseline:
        event_id = row["event_id"]
        lines.append(f"- {event_id}: V2 queries {row['old_v2_executed_query_count']} -> V2.1 queries {int((q.event_id == event_id).sum())}; field-sufficient {row['old_v2_field_sufficient_count']} -> {int((f.event_id == event_id).sum() and f[(f.event_id == event_id) & (f.newly_sufficient == True)].shape[0])}")
    lines.extend([
        "",
        "## Decision",
        "",
        "Measured V2.1 recall improvement depends on newly sufficient fields above. Remaining missingness is classified in `pipeline_v21_live3_field_comparison.csv` by field.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_registry_candidates()
    run_live3()
