from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .source_fusion_providers import ExistingResearchProvider, StructuredSourceProvider, TransfermarktBackboneProvider
from .source_fusion_schemas import CONTRACT_FIELDS, DETERMINISTIC_FIELDS, FieldObservation, FieldResolution, FusionResult


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "source_fusion"
STRUCTURED_PATH = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"
BATCH_PATH = DEFAULT_OUTPUT_DIR / "contract_research" / "batch_20_results.json"
PLAYING_TIME_PATH = DEFAULT_OUTPUT_DIR / "playing_time_outcomes.csv"
AFFIRMATIVE_STATUSES = {"disclosed_yes", "disclosed_no", "partially_disclosed", "undisclosed", "conflicting_sources"}
DETERMINISTIC_CANONICAL_FIELDS = set(DETERMINISTIC_FIELDS) | {"transfer_type", "tm_transfer_fee", "market_value_at_signing"}
EXACT_FINANCIAL_FIELDS = {"transfer_fee", "loan_fee", "add_ons", "sell_on", "buy_back", "release_or_purchase_clause"}


def _value_key(observation: FieldObservation) -> tuple[Any, str | None, str | None]:
    return observation.normalized_value, observation.currency, observation.precision


def _has_safe_affirmative(observation: FieldObservation) -> bool:
    if observation.status not in AFFIRMATIVE_STATUSES:
        return False
    if observation.status == "disclosed_no":
        return observation.value is False
    if observation.status == "undisclosed":
        return True
    return observation.normalized_value is not None or observation.value is not None


def _is_researched_contract_field(observation: FieldObservation) -> bool:
    return observation.source_type != "deterministic_backbone" and observation.field in CONTRACT_FIELDS


def _priority(observation: FieldObservation) -> tuple[int, float]:
    source_rank = {
        "structured_external_candidate": 0,
        "official_structured_disclosure": 0,
        "web_research": 1,
        "deterministic_backbone": 2,
    }.get(observation.source_type, 2)
    tier_rank = observation.source_tier if observation.source_tier is not None else 9
    return source_rank + tier_rank, -observation.confidence


def resolve_field(event_id: str, field: str, observations: list[FieldObservation]) -> FieldResolution:
    relevant = [observation for observation in observations if observation.field == field]
    provenance = [observation.to_dict() for observation in relevant]
    if not relevant:
        return FieldResolution(event_id, field, [], resolution_status="not_found")

    if field in DETERMINISTIC_CANONICAL_FIELDS:
        affirmative = [observation for observation in relevant if observation.source_type == "deterministic_backbone" and _has_safe_affirmative(observation)]
        if affirmative:
            preferred = sorted(affirmative, key=_priority)[0]
            return FieldResolution(event_id, field, relevant, preferred.normalized_value, "resolved_single_source", preferred, provenance=provenance)

    affirmative = [
        observation for observation in relevant
        if _has_safe_affirmative(observation)
        and _is_researched_contract_field(observation)
        and not (observation.source_tier == 3 and field in EXACT_FINANCIAL_FIELDS)
    ]
    if not affirmative:
        if any(observation.status == "not_applicable" for observation in relevant):
            return FieldResolution(event_id, field, relevant, resolution_status="not_applicable", provenance=provenance)
        if any(observation.status == "undisclosed" for observation in relevant):
            preferred = next(observation for observation in relevant if observation.status == "undisclosed")
            return FieldResolution(event_id, field, relevant, None, "explicit_not_disclosed", preferred, provenance=provenance)
        if any(observation.status == "not_found" for observation in relevant):
            return FieldResolution(event_id, field, relevant, resolution_status="not_found", provenance=provenance)
        return FieldResolution(event_id, field, relevant, resolution_status="insufficient_evidence", provenance=provenance)

    groups: dict[tuple[Any, str | None, str | None], list[FieldObservation]] = defaultdict(list)
    for observation in affirmative:
        groups[_value_key(observation)].append(observation)
    non_empty_keys = [key for key in groups if key[0] is not None or any(observation.status == "undisclosed" for observation in groups[key])]
    if len(non_empty_keys) > 1:
        preferred = sorted(affirmative, key=_priority)[0]
        return FieldResolution(
            event_id,
            field,
            relevant,
            None,
            "unresolved_conflict",
            preferred,
            conflicting_observations=affirmative,
            review_required=True,
            review_reasons=[f"conflicting observations for {field}; values, currencies, or precision differ"],
            provenance=provenance,
        )

    preferred = sorted(affirmative, key=_priority)[0]
    if len({observation.source_name for observation in affirmative}) >= 2:
        status = "resolved_corroborated"
    elif preferred.status == "undisclosed":
        status = "explicit_not_disclosed"
    else:
        status = "resolved_single_source"
    review_reasons = sorted({reason for observation in affirmative for reason in observation.review_reasons})
    return FieldResolution(
        event_id,
        field,
        relevant,
        preferred.normalized_value,
        status,
        preferred,
        review_required=bool(review_reasons or any(observation.review_required for observation in affirmative)),
        review_reasons=review_reasons,
        provenance=provenance,
    )


def fuse_event(event: dict[str, Any], providers: list[StructuredSourceProvider]) -> FusionResult:
    observations = []
    for provider in providers:
        observations.extend(provider.get_event_observations(event))
    fields = list(DETERMINISTIC_FIELDS) + ["transfer_type", "tm_transfer_fee", "market_value_at_signing"] + list(CONTRACT_FIELDS)
    resolutions = {field: resolve_field(str(event["event_id"]), field, observations) for field in dict.fromkeys(fields)}
    review_reasons = sorted({reason for resolution in resolutions.values() for reason in resolution.review_reasons})
    return FusionResult(str(event["event_id"]), resolutions, observations, bool(review_reasons), review_reasons)


def _load_pilot_events() -> list[dict[str, Any]]:
    structured = pd.read_csv(STRUCTURED_PATH, dtype={"event_id": str})
    batch = json.loads(BATCH_PATH.read_text())
    ids = [row["event_id"] for row in batch["events"]]
    rows = []
    for event_id in ids:
        row = structured[structured["event_id"] == event_id].iloc[0].to_dict()
        rows.append({key: (None if pd.isna(value) else value) for key, value in row.items()})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def run_pilot_20(output_dir: Path = OUTPUT_DIR) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    providers: list[StructuredSourceProvider] = [TransfermarktBackboneProvider(), ExistingResearchProvider()]
    events = _load_pilot_events()
    observation_rows: list[dict[str, Any]] = []
    fused_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for event in events:
        result = fuse_event(event, providers)
        player = event.get("player_name")
        for observation in result.observations:
            observation_rows.append({
                "event_id": observation.event_id,
                "player": player,
                "field": observation.field,
                "value": observation.value,
                "status": observation.status,
                "source_name": observation.source_name,
                "source_type": observation.source_type,
                "source_tier": observation.source_tier,
                "confidence": observation.confidence,
                "precision": observation.precision,
                "currency": observation.currency,
                "review_required": observation.review_required,
                "source_reference": observation.source_reference or observation.source_url,
            })
        fused_row = {"event_id": result.event_id, "player": player}
        for field, resolution in result.resolutions.items():
            fused_row[field] = resolution.resolved_value
            fused_row[f"{field}_resolution_status"] = resolution.resolution_status
        fused_rows.append(fused_row)
        for field, resolution in result.resolutions.items():
            audit_rows.append({
                "event_id": result.event_id,
                "player": player,
                "field": field,
                "observation_count": len(resolution.observations),
                "sources": json.dumps(sorted({observation.source_name for observation in resolution.observations}), ensure_ascii=False),
                "resolved_value": resolution.resolved_value,
                "resolution_status": resolution.resolution_status,
                "conflict_count": len(resolution.conflicting_observations),
                "review_required": resolution.review_required,
                "review_reasons": json.dumps(resolution.review_reasons, ensure_ascii=False),
            })
    _write_csv(output_dir / "pilot_20_observations.csv", observation_rows)
    _write_csv(output_dir / "pilot_20_fused.csv", fused_rows)
    _write_csv(output_dir / "pilot_20_fusion_audit.csv", audit_rows)
    return observation_rows, fused_rows, audit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline source fusion over existing pilot artifacts")
    parser.add_argument("--pilot-20", action="store_true", help="Run source fusion over the 20-event pilot")
    parser.add_argument("--event-id", help="Run source fusion for one structured transfer event")
    args = parser.parse_args()
    if args.pilot_20 or not args.event_id:
        observations, fused, audit = run_pilot_20()
        print(f"Wrote {len(observations)} observations, {len(fused)} fused rows, and {len(audit)} audit rows")
        return
    structured = pd.read_csv(STRUCTURED_PATH, dtype={"event_id": str})
    matches = structured[structured["event_id"] == args.event_id]
    if matches.empty:
        raise SystemExit(f"event_id not found: {args.event_id}")
    event = {key: (None if pd.isna(value) else value) for key, value in matches.iloc[0].to_dict().items()}
    result = fuse_event(event, [TransfermarktBackboneProvider(), ExistingResearchProvider()])
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
