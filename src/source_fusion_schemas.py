from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


OBSERVATION_STATUSES = {
    "disclosed_yes",
    "disclosed_no",
    "partially_disclosed",
    "undisclosed",
    "not_found",
    "conflicting_sources",
    "not_applicable",
}
RESOLUTION_STATUSES = {
    "resolved_single_source",
    "resolved_corroborated",
    "unresolved_conflict",
    "insufficient_evidence",
    "explicit_not_disclosed",
    "not_found",
    "not_applicable",
}
CONTRACT_FIELDS = (
    "transfer_type",
    "transfer_fee",
    "loan_fee",
    "add_ons",
    "sell_on",
    "buy_back",
    "years_contract_left",
    "parent_contract_expiry",
    "purchase_option",
    "purchase_obligation",
    "obligation_trigger",
    "release_or_purchase_clause",
)
DETERMINISTIC_FIELDS = (
    "player_name",
    "player_age_at_transfer",
    "receiving_club",
    "departing_club",
    "transfer_date",
    "tm_transfer_fee",
    "market_value_at_signing",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SourceProviderMetadata:
    name: str
    source_type: str
    source_tier: int | None = None
    provider_version: str | None = None
    retrieved_at: str | None = None
    source_reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceObservation:
    event_id: str
    field: str
    value: Any = None
    normalized_value: Any = None
    status: str = "not_found"
    source_name: str = ""
    source_type: str = ""
    source_tier: int | None = None
    source_url: str | None = None
    source_reference: str | None = None
    observed_at: str = field(default_factory=utc_now)
    evidence_date: str | None = None
    contemporaneous_or_retrospective: str = "contemporaneous"
    confidence: float = 0.0
    notes: str | None = None
    raw_value: Any = None
    currency: str | None = None
    precision: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in OBSERVATION_STATUSES:
            raise ValueError(f"invalid observation status: {self.status}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("observation confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldObservation(SourceObservation):
    pass


@dataclass
class FieldResolution:
    event_id: str
    field: str
    observations: list[FieldObservation]
    resolved_value: Any = None
    resolution_status: str = "not_found"
    preferred_observation: FieldObservation | None = None
    conflicting_observations: list[FieldObservation] = field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.resolution_status not in RESOLUTION_STATUSES:
            raise ValueError(f"invalid resolution status: {self.resolution_status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "field": self.field,
            "observations": [observation.to_dict() for observation in self.observations],
            "resolved_value": self.resolved_value,
            "resolution_status": self.resolution_status,
            "preferred_observation": self.preferred_observation.to_dict() if self.preferred_observation else None,
            "conflicting_observations": [observation.to_dict() for observation in self.conflicting_observations],
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "provenance": self.provenance,
        }


@dataclass
class FusionResult:
    event_id: str
    resolutions: dict[str, FieldResolution]
    observations: list[FieldObservation]
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "resolutions": {field: resolution.to_dict() for field, resolution in self.resolutions.items()},
            "observations": [observation.to_dict() for observation in self.observations],
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
        }
