from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .build_research_dataset import (
    build_analysis_table,
    build_audit_table,
    build_research_dataset,
    build_review_queue,
    coverage,
    write_analysis_data_dictionary,
    write_data_dictionary,
    write_qa_summary,
)
from .config import DEFAULT_OUTPUT_DIR
from .contract_schemas import ContractResearchResult, render_model_schema_instructions
from .research_contract import ParleyProvider, ProviderFailure, load_event
from .retry_4_discovery import _cost_from_diagnostics, _cost_from_metadata, _load_allowed_env
from .source_discovery import SourceCandidate, assess_source_sufficiency, load_source_candidates


EVENT_IDS = (
    "tm_ec95d18d266c5a44f071",
    "tm_890a6e2a4b41a1f8dd46",
    "tm_211885c2a2cbd065928b",
)
OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
BATCH_PATH = OUTPUT_DIR / "batch_20_results.json"
STRUCTURED_PATH = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"
ADMISSIBLE_DIR = OUTPUT_DIR / "fullpage_rescue_sources" / "admissible"
ARCHIVE_DIR = OUTPUT_DIR / "fullpage_parley_3_archive"
LOG_PATH = OUTPUT_DIR / "fullpage_parley_3_update_log.md"
SUMMARY_PATH = OUTPUT_DIR / "fullpage_parley_3_summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _batch_event_map(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {event["event_id"]: event for event in batch["events"]}


def _class_counts() -> dict[str, int]:
    df = pd.read_csv(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis.csv")
    return {key: int(value) for key, value in df["classification"].value_counts().to_dict().items()}


def _coverage() -> dict[str, int]:
    dataset = pd.read_csv(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20.csv")
    return coverage(dataset)


def _provider() -> ParleyProvider:
    _load_allowed_env()
    api_key = os.environ.get("PARLEY_API_KEY")
    if not api_key:
        raise RuntimeError("PARLEY_API_KEY is required")
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "contract_research.md"
    model = os.environ.get("PARLEY_MODEL", "bedrock/claude-haiku-4-5")
    return ParleyProvider(api_key, model, prompt_path.read_text() + "\n\n" + render_model_schema_instructions())


def _validate_evidence(event_id: str, sources: list[SourceCandidate]) -> tuple[bool, list[str]]:
    if not sources:
        raise ValueError(f"{event_id} has no full-page admissible sources")
    bad = [
        source.source_url
        for source in sources
        if source.source_tier not in {1, 2} or source.event_match_status not in {"exact", "likely"}
    ]
    if bad:
        raise ValueError(f"{event_id} evidence contains non-admissible sources: {bad}")
    sufficient, reasons = assess_source_sufficiency(sources)
    if not sufficient:
        raise ValueError(f"{event_id} evidence is not sufficient: {reasons}")
    return sufficient, reasons


def _run_parley(provider: ParleyProvider, event: dict[str, Any], sources: list[SourceCandidate]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    attempt_costs: list[float] = []
    last_error: dict[str, Any] | None = None
    for attempt in range(1, 3):
        try:
            result = provider.research(event, sources)
            payload = result.to_dict()
            attempt_costs.append(_cost_from_metadata(payload))
            return payload, {
                "attempts": attempt,
                "successful_extraction": True,
                "attempt_costs": attempt_costs,
                "total_cost": round(sum(attempt_costs), 6),
                "error": None,
            }
        except ProviderFailure as error:
            cost = _cost_from_diagnostics(error.diagnostics)
            attempt_costs.append(cost)
            last_error = {"message": str(error), "diagnostics": error.diagnostics}
    return None, {
        "attempts": 2,
        "successful_extraction": False,
        "attempt_costs": attempt_costs,
        "total_cost": round(sum(attempt_costs), 6),
        "error": last_error,
    }


def _field_changes(result: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not result:
        return [], []
    established = []
    unknown = []
    for name, value in result.items():
        if not isinstance(value, dict) or "status" not in value:
            continue
        status = value.get("status")
        field_value = value.get("amount") or value.get("price") or value.get("percentage") or value.get("date") or value.get("year") or value.get("value") or value.get("description")
        if status in {"disclosed_yes", "disclosed_no", "partially_disclosed", "conflicting_sources", "undisclosed"} and field_value is not None:
            established.append(f"{name}: {field_value} ({status})")
        elif status in {"not_found", "not_applicable", "undisclosed"}:
            unknown.append(f"{name}: {status}")
    return established, unknown


def _archive_existing(path: Path) -> None:
    if path.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, ARCHIVE_DIR / path.name)


def _update_batch_counts(batch: dict[str, Any]) -> None:
    counts = Counter(event.get("classification") for event in batch["events"])
    batch["clean"] = counts.get("clean", 0)
    batch["usable_with_review"] = counts.get("usable_with_review", 0)
    batch["insufficient_evidence"] = counts.get("insufficient_evidence", 0)
    batch["invalid"] = counts.get("invalid", 0)
    batch["sufficient_source_events"] = sum(1 for event in batch["events"] if event.get("sufficient"))
    batch["parley_calls"] = sum(1 for event in batch["events"] if event.get("parley_called"))
    batch["total_parley_cost"] = round(sum(float(event.get("parley_cost") or 0) for event in batch["events"]), 6)
    batch["average_parley_cost_per_call"] = round(batch["total_parley_cost"] / batch["parley_calls"], 6) if batch["parley_calls"] else 0


def _regenerate_datasets(batch: dict[str, Any]) -> None:
    raw_path = DEFAULT_OUTPUT_DIR / "research_dataset_batch_20.csv"
    analysis_path = DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis.csv"
    audit_path = DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_audit.csv"
    review_path = DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_review_queue.csv"
    dataset = build_research_dataset(STRUCTURED_PATH, BATCH_PATH, raw_path)
    analysis = build_analysis_table(dataset)
    audit = build_audit_table(dataset)
    review = build_review_queue(analysis, audit)
    analysis.to_csv(analysis_path, index=False)
    audit.to_csv(audit_path, index=False)
    review.to_csv(review_path, index=False)
    write_data_dictionary(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_data_dictionary.md", dataset)
    write_analysis_data_dictionary(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis_data_dictionary.md", analysis)
    write_qa_summary(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_qa.md", analysis, batch, DEFAULT_OUTPUT_DIR / "discovered_sources")


def main() -> None:
    before = {"class_counts": _class_counts(), "coverage": _coverage()}
    batch = _read_json(BATCH_PATH)
    batch_events = _batch_event_map(batch)
    provider = _provider()
    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    total_attempts = 0

    for event_id in EVENT_IDS:
        event = load_event(STRUCTURED_PATH, event_id)
        sources = load_source_candidates(ADMISSIBLE_DIR / f"{event_id}.json")
        sufficient, reasons = _validate_evidence(event_id, sources)
        batch_row = batch_events[event_id]
        before_classification = batch_row.get("classification")
        output_path = OUTPUT_DIR / f"{event_id}.json"
        result_payload, parley_info = _run_parley(provider, event, sources)
        total_cost += float(parley_info["total_cost"])
        total_attempts += int(parley_info["attempts"])
        established, unknown = _field_changes(result_payload)

        if result_payload:
            ContractResearchResult.from_dict(result_payload)
            _archive_existing(output_path)
            output_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n")
            classification_after = "usable_with_review" if result_payload.get("review_required") else "clean"
            meta = result_payload.get("provider_metadata", {})
            batch_row.update({
                "classification": classification_after,
                "parley_called": True,
                "parley_attempts": parley_info["attempts"],
                "parley_successful_extraction": True,
                "parley_cost": meta.get("total_request_cost") or parley_info["total_cost"],
                "parley_total_attempt_cost": parley_info["total_cost"],
                "output_path": str(output_path),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "total_tokens": meta.get("total_tokens"),
                "review_required": result_payload.get("review_required"),
                "review_reasons": result_payload.get("review_reasons") or [],
            })
        else:
            classification_after = "invalid"
            failure_path = OUTPUT_DIR / "fullpage_parley_3_failures" / f"{event_id}.json"
            _write_json(failure_path, parley_info)
            batch_row.update({
                "classification": "invalid",
                "parley_called": False,
                "parley_attempts": parley_info["attempts"],
                "parley_successful_extraction": False,
                "parley_total_attempt_cost": parley_info["total_cost"],
            })

        batch_row.update({
            "sufficient": sufficient,
            "sufficiency_reasons": reasons,
            "admissible_source_count": len(sources),
        })
        rows.append({
            "event_id": event_id,
            "player": event.get("player_name"),
            "classification_before": before_classification,
            "evidence_sufficient_before": sufficient,
            "sources_used": [source.evidence_id for source in sources],
            "source_urls": [source.source_url for source in sources],
            "fields_newly_established": established,
            "fields_still_unknown": unknown,
            "review_reasons": (result_payload or {}).get("review_reasons") or [],
            "classification_after": classification_after,
            "parley_attempts": parley_info["attempts"],
            "parley_cost": parley_info["total_cost"],
            "parley_success": parley_info["successful_extraction"],
        })

    batch["fullpage_parley_3_attempts"] = total_attempts
    batch["fullpage_parley_3_successful_extractions"] = sum(1 for row in rows if row["parley_success"])
    batch["fullpage_parley_3_cost"] = round(total_cost, 6)
    _update_batch_counts(batch)
    _write_json(BATCH_PATH, batch)
    _regenerate_datasets(batch)
    after = {"class_counts": _class_counts(), "coverage": _coverage()}
    _write_json(SUMMARY_PATH, {"before": before, "after": after, "events": rows, "total_attempts": total_attempts, "total_cost": round(total_cost, 6)})
    _write_log(rows, before, after, total_attempts, total_cost)


def _write_log(rows: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any], total_attempts: int, total_cost: float) -> None:
    lines = [
        "# Full-Page Parley 3 Update Log",
        "",
        f"- Total Parley attempts: {total_attempts}",
        f"- Total Parley cost: {total_cost:.6f}",
        f"- Classification counts before: {json.dumps(before['class_counts'], sort_keys=True)}",
        f"- Classification counts after: {json.dumps(after['class_counts'], sort_keys=True)}",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row['player']} ({row['event_id']})",
            "",
            f"- Classification before: `{row['classification_before']}`",
            f"- Evidence sufficiency before Parley: `{row['evidence_sufficient_before']}`",
            f"- Sources used: {json.dumps(row['sources_used'], ensure_ascii=False)}",
            f"- Source URLs: {json.dumps(row['source_urls'], ensure_ascii=False)}",
            f"- Fields newly established: {json.dumps(row['fields_newly_established'], ensure_ascii=False)}",
            f"- Fields still unknown: {json.dumps(row['fields_still_unknown'], ensure_ascii=False)}",
            f"- Review reasons: {json.dumps(row['review_reasons'], ensure_ascii=False)}",
            f"- Classification after: `{row['classification_after']}`",
            f"- Parley attempts: {row['parley_attempts']}",
            f"- Parley cost: {row['parley_cost']:.6f}",
            "",
        ])
    LOG_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
