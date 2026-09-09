from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .contract_schemas import FIELD_NAMES, ContractResearchResult
from .source_fusion_schemas import FieldObservation, SourceProviderMetadata, utc_now


class StructuredSourceProvider(Protocol):
    name: str
    source_type: str

    def get_event_observations(self, event: dict[str, Any]) -> list[FieldObservation]:
        ...


def _missing(value: Any) -> bool:
    try:
        return pd.isna(value)
    except (TypeError, ValueError):
        return value is None


def _field_value(field: dict[str, Any]) -> Any:
    for key in ("amount", "price", "percentage", "date", "value", "year"):
        value = field.get(key)
        if value is not None:
            return value
    return None


class TransfermarktBackboneProvider:
    name = "Transfermarkt"
    source_type = "deterministic_backbone"

    def __init__(self, structured_transfers_path: Path = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"):
        self.structured_transfers_path = structured_transfers_path

    @property
    def metadata(self) -> SourceProviderMetadata:
        return SourceProviderMetadata(self.name, self.source_type, source_tier=None, source_reference=str(self.structured_transfers_path))

    def get_event_observations(self, event: dict[str, Any]) -> list[FieldObservation]:
        event_id = str(event["event_id"])
        observed_at = utc_now()
        rows = [
            ("player_name", event.get("player_name"), "string", None),
            ("player_age_at_transfer", event.get("age_at_transfer"), "years", None),
            ("departing_club", event.get("from_club_name"), "club_name", None),
            ("receiving_club", event.get("to_club_name"), "club_name", None),
            ("transfer_date", event.get("transfer_date"), "date", "date"),
            ("transfer_type", event.get("transfer_type"), "transfer_type", None),
            ("tm_transfer_fee", event.get("transfer_fee"), "fee", "exact" if not _missing(event.get("transfer_fee")) else None),
            ("market_value_at_signing", event.get("market_value_nearest_transfer"), "market_value", "exact" if not _missing(event.get("market_value_nearest_transfer")) else None),
        ]
        observations = []
        for field, value, notes, precision in rows:
            observations.append(FieldObservation(
                event_id=event_id,
                field=field,
                value=None if _missing(value) else value,
                normalized_value=None if _missing(value) else value,
                status="not_found" if _missing(value) else "disclosed_yes",
                source_name=self.name,
                source_type=self.source_type,
                source_reference=str(self.structured_transfers_path),
                observed_at=observed_at,
                evidence_date=str(event.get("transfer_date") or "") or None,
                confidence=1.0 if not _missing(value) else 0.0,
                notes=f"Transfermarkt deterministic {notes}; not independent contract research.",
                raw_value=None if _missing(value) else value,
                currency="EUR" if field in {"tm_transfer_fee", "market_value_at_signing"} and not _missing(value) else None,
                precision=precision,
            ))
        return observations


class ExistingResearchProvider:
    name = "existing_web_research"
    source_type = "web_research"

    def __init__(self, batch_results_path: Path = DEFAULT_OUTPUT_DIR / "contract_research" / "batch_20_results.json"):
        self.batch_results_path = batch_results_path
        self._batch = json.loads(batch_results_path.read_text())
        self._paths = {row["event_id"]: row.get("output_path") for row in self._batch["events"]}

    def get_event_observations(self, event: dict[str, Any]) -> list[FieldObservation]:
        path = self._paths.get(str(event["event_id"]))
        if not path or not Path(path).exists():
            return []
        result = ContractResearchResult.from_dict(json.loads(Path(path).read_text())).to_dict()
        source_by_id = {source["evidence_id"]: source for source in result.get("sources", [])}
        observations = []
        for field_name in FIELD_NAMES:
            field = result.get(field_name, {})
            status = field.get("status") or "not_found"
            evidence_ids = [str(item) for item in field.get("evidence_ids", [])]
            references = [
                source_by_id[evidence_id].get("source_url")
                for evidence_id in evidence_ids
                if evidence_id in source_by_id and source_by_id[evidence_id].get("source_url")
            ]
            tier_values = [
                source_by_id[evidence_id].get("source_tier")
                for evidence_id in evidence_ids
                if evidence_id in source_by_id and source_by_id[evidence_id].get("source_tier")
            ]
            observations.append(FieldObservation(
                event_id=str(event["event_id"]),
                field=field_name,
                value=_field_value(field),
                normalized_value=_field_value(field),
                status=status,
                source_name=self.name,
                source_type=self.source_type,
                source_tier=min(tier_values) if tier_values else None,
                source_url=references[0] if len(references) == 1 else None,
                source_reference=json.dumps(references, ensure_ascii=False) if references else str(self.batch_results_path),
                observed_at=result.get("research_timestamp") or utc_now(),
                evidence_date=None,
                confidence=float(field.get("confidence") or 0.0),
                notes=field.get("description"),
                raw_value=field.get("reported_values") or _field_value(field),
                currency=field.get("currency"),
                precision=field.get("precision"),
                evidence_ids=evidence_ids,
                review_required=bool(result.get("review_required")),
                review_reasons=list(result.get("review_reasons") or []),
            ))
        return observations


class OfflineJSONSourceProvider:
    def __init__(self, path: Path, name: str = "external_contract_source"):
        self.path = path
        self.name = name
        self.source_type = "structured_external_candidate"
        self.payload = json.loads(path.read_text())

    def get_event_observations(self, event: dict[str, Any]) -> list[FieldObservation]:
        event_rows = [row for row in self.payload.get("events", []) if str(row.get("event_id")) == str(event["event_id"])]
        observations = []
        for row in event_rows:
            for item in row.get("observations", []):
                observations.append(FieldObservation(
                    event_id=str(event["event_id"]),
                    field=str(item["field"]),
                    value=item.get("value"),
                    normalized_value=item.get("normalized_value", item.get("value")),
                    status=item.get("status", "disclosed_yes"),
                    source_name=item.get("source_name", self.name),
                    source_type=item.get("source_type", self.source_type),
                    source_tier=item.get("source_tier"),
                    source_url=item.get("source_url"),
                    source_reference=item.get("source_reference", str(self.path)),
                    observed_at=item.get("observed_at", utc_now()),
                    evidence_date=item.get("evidence_date"),
                    contemporaneous_or_retrospective=item.get("contemporaneous_or_retrospective", "contemporaneous"),
                    confidence=float(item.get("confidence", 0.8)),
                    notes=item.get("notes"),
                    raw_value=item.get("raw_value", item.get("value")),
                    currency=item.get("currency"),
                    precision=item.get("precision"),
                    evidence_ids=[str(value) for value in item.get("evidence_ids", [])],
                    review_required=bool(item.get("review_required", False)),
                    review_reasons=[str(value) for value in item.get("review_reasons", [])],
                ))
        return observations
