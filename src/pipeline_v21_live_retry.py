from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_DIR
from .official_page_retrieval import OfficialPageFetcher
from .pipeline_v2 import FIELD_RESEARCH_FIELDS, SearchBudget, generate_event_query_plan
from .pipeline_v2_live5 import _field_assessment, _query_log_row, _retrieve, _search_query
from .research_contract import load_event
from .retry_4_discovery import _load_allowed_env
from .source_discovery import SourceCandidate, deduplicate_sources, filter_admissible_sources, TavilySearchProvider


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
STRUCTURED_PATH = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"
OLD_LOG_PATH = OUTPUT_DIR / "pipeline_v2_live5_query_log.csv"
RETRY_PATH = OUTPUT_DIR / "pipeline_v21_live_retry.csv"
SOURCE_DIR = OUTPUT_DIR / "pipeline_v21_live_retry_sources"
EVENT_IDS = (
    "tm_ec95d18d266c5a44f071",
    "tm_211885c2a2cbd065928b",
    "tm_1ae175d1097fbcd0ddc0",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _old_query_counts() -> dict[tuple[str, str], int]:
    with OLD_LOG_PATH.open(newline="") as handle:
        return Counter((row["event_id"], row["field"]) for row in csv.DictReader(handle))


def _resolve(event: dict[str, Any], tavily: TavilySearchProvider, fetcher: OfficialPageFetcher) -> tuple[str, list[SourceCandidate], list[dict[str, Any]]]:
    all_sources: list[SourceCandidate] = []
    logs: list[dict[str, Any]] = []
    for plan in [item for item in generate_event_query_plan(event) if item.field == "event_resolution"]:
        found = _retrieve(_search_query(tavily, event, plan), event, fetcher)
        all_sources = deduplicate_sources(all_sources + found)
        admissible = filter_admissible_sources(all_sources)
        if any(source.event_match_status == "exact" for source in admissible):
            status = "confirmed"
        elif admissible:
            status = "likely"
        elif any(source.event_match_status == "mismatch" for source in all_sources):
            status = "mismatch"
        else:
            status = "unresolved"
        logs.append(_query_log_row(event, plan, found, admissible, status in {"confirmed", "likely"}, False, "event resolution"))
        if status in {"confirmed", "likely"}:
            return status, all_sources, logs
    return status if "status" in locals() else "unresolved", all_sources, logs


def main() -> None:
    _load_allowed_env()
    tavily = TavilySearchProvider.from_environment()
    fetcher = OfficialPageFetcher()
    old_counts = _old_query_counts()
    rows: list[dict[str, Any]] = []

    for event_id in EVENT_IDS:
        event = load_event(STRUCTURED_PATH, event_id)
        plan = generate_event_query_plan(event, set(), budget=SearchBudget(event_total_max_queries=18))
        planned_by_field = Counter(item.field for item in plan)
        event_resolution, event_sources, resolution_logs = _resolve(event, tavily, fetcher)
        sources_by_field: dict[str, list[SourceCandidate]] = defaultdict(list)
        status_by_field: dict[str, tuple[str, str]] = {}
        executed_by_field: Counter[str] = Counter()

        if event_resolution in {"confirmed", "likely"}:
            for field in FIELD_RESEARCH_FIELDS:
                field_plans = [item for item in plan if item.field == field]
                for field_plan in field_plans:
                    found = _retrieve(_search_query(tavily, event, field_plan), event, fetcher)
                    sources_by_field[field] = deduplicate_sources(sources_by_field[field] + found)
                    status, supporting, reason = _field_assessment(field, sources_by_field[field], True)
                    executed_by_field[field] += 1
                    if status in {"sufficient", "explicit_not_disclosed", "conflicting"}:
                        break
                status_by_field[field] = _field_assessment(field, sources_by_field[field], True)[:2] + (_field_assessment(field, sources_by_field[field], True)[2],)
        else:
            for field in FIELD_RESEARCH_FIELDS:
                status_by_field[field] = ("insufficient", [], "event unresolved")

        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        all_sources = deduplicate_sources(event_sources + [source for sources in sources_by_field.values() for source in sources])
        (SOURCE_DIR / f"{event_id}.json").write_text(json.dumps({
            "event_id": event_id,
            "event_resolution": event_resolution,
            "sources": [source.to_dict() for source in all_sources],
        }, indent=2, ensure_ascii=False) + "\n")

        for field in FIELD_RESEARCH_FIELDS:
            status, supporting, reason = status_by_field[field]
            rows.append({
                "event_id": event_id,
                "player": event.get("player_name"),
                "field": field,
                "old_v2_query_count": old_counts[(event_id, field)],
                "new_v21_planned_query_count": planned_by_field[field],
                "new_v21_executed_query_count": executed_by_field[field],
                "tier1_results": sum(1 for source in sources_by_field[field] if source.source_tier == 1),
                "tier2_results": sum(1 for source in sources_by_field[field] if source.source_tier == 2),
                "event_resolution": event_resolution,
                "field_sufficient_after_query": status in {"sufficient", "explicit_not_disclosed", "conflicting"},
                "field_evidence_status": status,
                "supporting_evidence_count": len(supporting),
                "reason": reason,
            })
        for log in resolution_logs:
            rows.append({
                "event_id": event_id,
                "player": event.get("player_name"),
                "field": "event_resolution",
                "old_v2_query_count": old_counts[(event_id, "event_resolution")],
                "new_v21_planned_query_count": planned_by_field["event_resolution"],
                "new_v21_executed_query_count": 1,
                "tier1_results": log["tier1_count"],
                "tier2_results": log["tier2_count"],
                "event_resolution": event_resolution,
                "field_sufficient_after_query": log["field_sufficient_after_query"],
                "field_evidence_status": event_resolution,
                "supporting_evidence_count": log["admissible_count"],
                "reason": log["stop_reason"],
            })
    _write_csv(RETRY_PATH, rows)


if __name__ == "__main__":
    main()
