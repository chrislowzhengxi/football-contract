from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .build_research_dataset import coverage
from .config import DEFAULT_OUTPUT_DIR
from .evaluate_fullpage_rescue import evaluate_events, mechanism_signals
from .fullpage_parley_3 import (
    _class_counts,
    _coverage,
    _field_changes,
    _provider,
    _read_json,
    _regenerate_datasets,
    _run_parley,
    _update_batch_counts,
    _validate_evidence,
    _write_json,
)
from .research_contract import load_event
from .source_discovery import load_source_candidates


BASE_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
BATCH_PATH = BASE_DIR / "batch_20_results.json"
BASELINE_PATH = BASE_DIR / "remaining_8_baseline.json"
COMPARISON_PATH = BASE_DIR / "remaining_8_fullpage_comparison.csv"
STATUS_REPORT_PATH = BASE_DIR / "pilot_20_status_report.md"
PARLEY_LOG_PATH = BASE_DIR / "remaining_8_parley_update_log.md"


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


def _discovered_artifact(event_id: str) -> Path:
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
        sources = _load_sources(_discovered_artifact(event_id))
        tier_counts = Counter(source.get("source_tier") for source in sources)
        admissible = _load_sources(DEFAULT_OUTPUT_DIR / "discovered_sources" / "admissible" / f"{event_id}.json")
        rows.append({
            "event_id": event_id,
            "player": event.get("player_name"),
            "from_club": event.get("from_club_name"),
            "to_club": event.get("to_club_name"),
            "date": event.get("transfer_date"),
            "old_discovered_source_count": len(sources),
            "existing_tier1_count": tier_counts.get(1, 0),
            "existing_tier2_count": tier_counts.get(2, 0),
            "existing_admissible_count": len(admissible),
            "unresolved_reasons": batch_rows[event_id].get("sufficiency_reasons", []),
        })
    _write_json(BASELINE_PATH, rows)
    return rows


def _comparison_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _group(row: dict[str, Any]) -> str:
    tier12 = int(row["old_tier12_source_count"])
    new_admissible = int(row["new_admissible_count"])
    new_matches = int(row["new_exact_likely_tier12_matches"])
    if new_admissible > 0 and row["new_sufficient"].lower() == "true":
        return "rescued_by_fullpage"
    if tier12 == 0:
        return "still_no_credible_Tier1_2_sources"
    if new_matches == 0:
        return "official_sources_exist_but_event_identity_still_unresolved"
    return "official_sources_exist_but_not_sufficient"


def _run_parley_for_rescued(rows: list[dict[str, Any]], before: dict[str, Any]) -> tuple[list[dict[str, Any]], int, float]:
    rescued = [row for row in rows if _group(row) == "rescued_by_fullpage"]
    if not rescued:
        _write_parley_log([], before, before, 0, 0.0)
        return [], 0, 0.0
    batch = _read_json(BATCH_PATH)
    batch_events = {event["event_id"]: event for event in batch["events"]}
    provider = _provider()
    updates = []
    total_attempts = 0
    total_cost = 0.0
    for row in rescued:
        event_id = row["event_id"]
        sources = load_source_candidates(BASE_DIR / "fullpage_rescue_sources" / "admissible" / f"{event_id}.json")
        sufficient, reasons = _validate_evidence(event_id, sources)
        event = load_event(DEFAULT_OUTPUT_DIR / "structured_transfers.csv", event_id)
        result, info = _run_parley(provider, event, sources)
        total_attempts += int(info["attempts"])
        total_cost += float(info["total_cost"])
        established, unknown = _field_changes(result)
        batch_row = batch_events[event_id]
        before_classification = batch_row.get("classification")
        classification_after = before_classification
        if result:
            output_path = BASE_DIR / f"{event_id}.json"
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            classification_after = "usable_with_review" if result.get("review_required") else "clean"
            meta = result.get("provider_metadata", {})
            batch_row.update({
                "classification": classification_after,
                "parley_called": True,
                "parley_attempts": info["attempts"],
                "parley_successful_extraction": True,
                "parley_cost": meta.get("total_request_cost") or info["total_cost"],
                "parley_total_attempt_cost": info["total_cost"],
                "output_path": str(output_path),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "total_tokens": meta.get("total_tokens"),
                "review_required": result.get("review_required"),
                "review_reasons": result.get("review_reasons") or [],
            })
        batch_row.update({
            "sufficient": sufficient,
            "sufficiency_reasons": reasons,
            "admissible_source_count": len(sources),
        })
        updates.append({
            "event_id": event_id,
            "player": row["player_name"],
            "classification_before": before_classification,
            "classification_after": classification_after,
            "fields_newly_established": established,
            "fields_still_unknown": unknown,
            "parley_attempts": info["attempts"],
            "parley_cost": info["total_cost"],
            "parley_success": bool(result),
        })
    batch["remaining_8_parley_attempts"] = total_attempts
    batch["remaining_8_successful_parley_extractions"] = sum(1 for update in updates if update["parley_success"])
    batch["remaining_8_parley_cost"] = round(total_cost, 6)
    _update_batch_counts(batch)
    _write_json(BATCH_PATH, batch)
    _regenerate_datasets(batch)
    _write_parley_log(updates, before, {"class_counts": _class_counts(), "coverage": _coverage()}, total_attempts, total_cost)
    return updates, total_attempts, total_cost


def _write_parley_log(updates: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any], attempts: int, cost: float) -> None:
    lines = [
        "# Remaining 8 Parley Update Log",
        "",
        f"- Parley attempts: {attempts}",
        f"- Parley cost: {cost:.6f}",
        f"- Classification counts before: {json.dumps(before['class_counts'], sort_keys=True)}",
        f"- Classification counts after: {json.dumps(after['class_counts'], sort_keys=True)}",
        "",
    ]
    for update in updates:
        lines.extend([
            f"## {update['player']} ({update['event_id']})",
            "",
            f"- Classification before: `{update['classification_before']}`",
            f"- Classification after: `{update['classification_after']}`",
            f"- Fields newly established: {json.dumps(update['fields_newly_established'], ensure_ascii=False)}",
            f"- Fields still unknown: {json.dumps(update['fields_still_unknown'], ensure_ascii=False)}",
            f"- Parley attempts: {update['parley_attempts']}",
            f"- Parley cost: {update['parley_cost']:.6f}",
            "",
        ])
    PARLEY_LOG_PATH.write_text("\n".join(lines))


def _status_report(baseline: list[dict[str, Any]], comparison: list[dict[str, Any]], updates: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any], attempts: int, cost: float) -> None:
    groups = Counter(_group(row) for row in comparison)
    had_tier12 = sum(int(row["old_tier12_source_count"]) > 0 for row in comparison)
    term_rich = sum(int(row["contract_term_rich_pages"]) for row in comparison)
    mechanisms = Counter()
    for row in comparison:
        mechanisms.update(json.loads(row["mechanism_signals"]))
    lines = [
        "# Pilot 20 Status Report",
        "",
        "The current pilot now uses the validated architecture: Tavily discovery, targeted full-page retrieval for saved Tier 1/2 URLs, event matching on retrieved text, admissibility filtering, sufficiency gating, and Parley extraction only after sufficiency.",
        "",
        "## Current Status",
        "",
        f"- Clean: {after['class_counts'].get('clean', 0)}",
        f"- Usable with review: {after['class_counts'].get('usable_with_review', 0)}",
        f"- Insufficient evidence: {after['class_counts'].get('insufficient_evidence', 0)}",
        f"- Invalid: {after['class_counts'].get('invalid', 0)}",
        f"- Remaining-8 cases with saved Tier 1/2 URLs: {had_tier12}",
        f"- Rescued by full-page retrieval in this pass: {groups.get('rescued_by_fullpage', 0)}",
        f"- Parley attempts in this pass: {attempts}",
        f"- Parley cost in this pass: {cost:.6f}",
        "",
        "## Field Coverage",
        "",
    ]
    for field in FIELDS:
        lines.append(f"- {field}: {before['coverage'].get(field, 0)} -> {after['coverage'].get(field, 0)}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The strongest technical bottleneck was Tavily snippet thinness for already-discovered official pages. Full-page retrieval fixes that when official URLs are present. The remaining technical bottleneck is discovery coverage: several insufficient cases have no saved Tier 1/2 URLs to retrieve.",
        "",
        "Public-data missingness remains separate from technical failure. Even when official pages support event identity and transfer type, they often omit sell-on, buy-back, add-ons, exact option prices, and fee mechanics. Transfer type and broad loan/permanent structure are realistic public fields; sell-on and buy-back appear systematically sparse.",
        "",
        f"Contract-term-rich pages in remaining-8 retrieval: {term_rich}. Mechanism signals: {json.dumps(dict(mechanisms), ensure_ascii=False, sort_keys=True)}",
        "",
        "## Remaining Insufficient Cases",
        "",
    ])
    baseline_by_id = {row["event_id"]: row for row in baseline}
    for row in comparison:
        base = baseline_by_id[row["event_id"]]
        lines.append(f"- `{row['event_id']}` {row['player_name']}: {_group(row)}; {base['from_club']} -> {base['to_club']}")
    STATUS_REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    before = {"class_counts": _class_counts(), "coverage": _coverage()}
    event_ids = _current_insufficient_ids()
    baseline = build_baseline(event_ids)
    evaluate_events(event_ids, COMPARISON_PATH)
    comparison = _comparison_rows(COMPARISON_PATH)
    updates, attempts, cost = _run_parley_for_rescued(comparison, before)
    if attempts == 0:
        batch = _read_json(BASE_DIR / "batch_20_results.json")
        batch["remaining_8_parley_attempts"] = 0
        batch["remaining_8_successful_parley_extractions"] = 0
        batch["remaining_8_parley_cost"] = 0.0
        _write_json(BASE_DIR / "batch_20_results.json", batch)
    after = {"class_counts": _class_counts(), "coverage": _coverage()}
    _status_report(baseline, comparison, updates, before, after, attempts, cost)


if __name__ == "__main__":
    main()
