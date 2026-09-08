from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


STATUS_VALUES = {
    "disclosed_yes",
    "disclosed_no",
    "partially_disclosed",
    "undisclosed",
    "not_found",
    "conflicting_sources",
    "not_applicable",
}


@dataclass
class Evidence:
    evidence_id: str
    source_url: str
    source_title: str | None
    publisher: str | None
    source_type: str
    publication_date: str | None
    retrieval_date: str
    evidence_text: str
    language: str
    source_tier: int | None = None


@dataclass
class ProviderMetadata:
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    parley_cost_header: str | None = None
    total_request_cost: float | None = None


@dataclass
class ContractField:
    value: Any = None
    status: str = "not_found"
    confidence: float = 0.0
    amount: float | None = None
    currency: str | None = None
    description: str | None = None
    price: float | None = None
    percentage: float | None = None
    basis: str | None = None
    metric: str | None = None
    threshold: float | None = None
    unit: str | None = None
    additional_condition: str | None = None
    date: str | None = None
    exercised: bool | None = None
    evidence_ids: list[str] = field(default_factory=list)
    reported_values: list[dict[str, Any]] = field(default_factory=list)


FIELD_NAMES = (
    "transfer_type",
    "transfer_fee",
    "loan_fee",
    "add_ons",
    "purchase_option",
    "purchase_obligation",
    "obligation_trigger",
    "sell_on",
    "buy_back",
    "parent_contract_expiry",
    "release_or_purchase_clause",
)


def render_model_schema_instructions() -> str:
    allowed = ", ".join(sorted(STATUS_VALUES))
    field_lines = []
    for name in FIELD_NAMES:
        field_lines.append(
            f'  "{name}": {{"value": nullable, "status": one of [{allowed}], '
            '"confidence": number from 0 to 1, "evidence_ids": array of source evidence_id strings, '
            'plus only relevant nullable fields such as amount, currency, price, percentage, '
            'description, metric, threshold, unit, additional_condition, date, exercised, '
            'or reported_values: array of original per-source amount/currency objects}}'
        )
    return "\n".join([
        "The response schema is authoritative and every contract field is required:",
        '{',
        '  "research_timestamp": ISO-8601 string,',
        '  "sources": array of evidence objects,',
        '  "deal_summary": string,',
        '  "review_required": boolean,',
        '  "review_reasons": array of strings,',
        *field_lines,
        '}',
        f"Every status MUST be exactly one of: [{allowed}].",
        "Never invent status values such as confirmed, yes, no, known, or unknown.",
        "disclosed_yes means the supplied evidence explicitly establishes the feature exists or is true.",
        "disclosed_no means the supplied evidence explicitly states the feature does not exist or is false.",
        "not_found means the supplied evidence does not establish the term; absence of discussion is not disclosed_no.",
        "partially_disclosed means only part of a contractual term is known.",
        "Use not_applicable only when the field genuinely does not apply to this transaction.",
        "Use null for unknown values, and link every non-not_found contractual claim through evidence_ids.",
        "When sources report amounts, preserve each source's original amount and currency in reported_values; do not convert currencies or invent a consensus number.",
        "Only Tier 1 or Tier 2 sources are admissible contractual evidence; Tier 3 sources are context only and must not independently establish exact contractual fields.",
        "Deterministic Transfermarkt fee and market-value fields in the event are background metadata only; never copy them into researched contract fields without admissible source evidence.",
        "Return exactly one JSON object with no prose and no markdown fences.",
    ])


@dataclass
class ContractResearchResult:
    event_id: str
    research_timestamp: str
    sources: list[Evidence]
    deal_summary: str
    review_required: bool
    review_reasons: list[str]
    transfer_type: ContractField
    transfer_fee: ContractField
    loan_fee: ContractField
    add_ons: ContractField
    purchase_option: ContractField
    purchase_obligation: ContractField
    obligation_trigger: ContractField
    sell_on: ContractField
    buy_back: ContractField
    parent_contract_expiry: ContractField
    release_or_purchase_clause: ContractField
    provider_metadata: ProviderMetadata = field(default_factory=ProviderMetadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContractResearchResult":
        missing = {"event_id", "deal_summary", "sources"} - payload.keys()
        if missing:
            raise ValueError(f"research result missing fields: {sorted(missing)}")
        fields = {name: _field_from_dict(payload.get(name, {})) for name in FIELD_NAMES}
        sources = [_evidence_from_dict(source) for source in payload.get("sources", [])]
        result = cls(
            event_id=str(payload["event_id"]),
            research_timestamp=payload.get("research_timestamp") or utc_now(),
            sources=sources,
            deal_summary=str(payload["deal_summary"]),
            review_required=bool(payload.get("review_required", False)),
            review_reasons=[str(reason) for reason in payload.get("review_reasons", [])],
            provider_metadata=ProviderMetadata(**payload.get("provider_metadata", {})),
            **fields,
        )
        validate_research_result(result)
        return result


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _field_from_dict(value: Any) -> ContractField:
    if not isinstance(value, dict):
        raise ValueError("contract fields must be objects")
    allowed = set(ContractField.__dataclass_fields__)
    return ContractField(**{key: item for key, item in value.items() if key in allowed})


def _evidence_from_dict(value: Any) -> Evidence:
    if not isinstance(value, dict):
        raise ValueError("sources must be objects")
    required = {"evidence_id", "source_url", "source_type", "retrieval_date", "evidence_text", "language"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"source missing fields: {sorted(missing)}")
    return Evidence(
        evidence_id=str(value["evidence_id"]),
        source_url=str(value["source_url"]),
        source_title=value.get("source_title"),
        publisher=value.get("publisher"),
        source_type=str(value["source_type"]),
        publication_date=value.get("publication_date"),
        retrieval_date=str(value["retrieval_date"]),
        evidence_text=str(value["evidence_text"]),
        language=str(value["language"]),
        source_tier=value.get("source_tier"),
    )


def validate_research_result(result: ContractResearchResult) -> None:
    source_ids = {source.evidence_id for source in result.sources}
    reasons = set(result.review_reasons)
    _add_specific_review_reasons(result, reasons)
    for name in FIELD_NAMES:
        finding = getattr(result, name)
        if finding.status not in STATUS_VALUES:
            raise ValueError(f"invalid status for {name}: {finding.status}")
        if not 0 <= finding.confidence <= 1:
            raise ValueError(f"confidence for {name} must be between 0 and 1")
        if not set(finding.evidence_ids) <= source_ids:
            raise ValueError(f"{name} references unknown evidence")
        if finding.status in {"disclosed_yes", "disclosed_no", "partially_disclosed", "undisclosed", "conflicting_sources"} and not finding.evidence_ids:
            raise ValueError(f"{name} requires evidence_ids for status {finding.status}")
        if finding.status in {"not_found", "partially_disclosed"} and finding.value is False:
            raise ValueError(f"{name}: {finding.status} cannot be represented as false")
        if finding.status == "conflicting_sources":
            reasons.add(f"conflicting sources for {name}")
        if name == "purchase_obligation" and finding.status in {"disclosed_yes", "partially_disclosed"}:
            reasons.add("purchase obligation requires human review")
        if name == "obligation_trigger" and finding.status in {"disclosed_yes", "partially_disclosed"}:
            reasons.add("obligation trigger requires human review")
        if finding.confidence < 0.5 and finding.status not in {"not_found", "not_applicable"}:
            reasons.add(f"low-confidence extraction for {name}")
        if name == "parent_contract_expiry" and (finding.value is not None or finding.date is not None):
            linked_text = " ".join(
                source.evidence_text.lower()
                for source in result.sources
                if source.evidence_id in finding.evidence_ids
            )
            explicit_date_terms = ("contract expires", "contract expiry", "contract until", "contract through", "years remaining", "year contract", "contract duration")
            date_is_explicit = any(term in linked_text for term in explicit_date_terms)
            if not date_is_explicit:
                reasons.discard("parent contract expiry is not explicitly sourced")
                reasons.add("unsupported_contract_expiry")
        if finding.evidence_ids and any(source.source_type == "aggregator" for source in result.sources if source.evidence_id in finding.evidence_ids):
            if any(value is not None for value in (finding.amount, finding.price, finding.percentage, finding.threshold)):
                reasons.add(f"exact figure for {name} relies on an aggregator")
    if reasons:
        result.review_required = True
    if result.review_required and not reasons:
        raise ValueError("review_required requires at least one review reason")
    result.review_reasons[:] = sorted(reasons)


def _add_specific_review_reasons(result: ContractResearchResult, reasons: set[str]) -> None:
    all_text = " ".join(source.evidence_text.lower() for source in result.sources)
    fee_text = " ".join(
        source.evidence_text.lower()
        for source in result.sources
        if source.evidence_id in set(result.transfer_fee.evidence_ids + result.add_ons.evidence_ids)
    )
    currencies = set()
    for marker, currency in (("€", "EUR"), ("£", "GBP"), ("$", "USD"), ("eur", "EUR"), ("gbp", "GBP"), ("usd", "USD")):
        if marker in fee_text:
            currencies.add(currency)
    if len(currencies) > 1:
        reasons.add("currency_reporting_difference")
    package_terms = ("add-on", "add on", "bonus", "could increase", "up to", "total package", "total fee")
    if any(term in fee_text for term in package_terms) and (
        result.transfer_fee.status in {"partially_disclosed", "conflicting_sources"}
        or result.add_ons.status in {"partially_disclosed", "conflicting_sources"}
    ):
        reasons.add("base_fee_vs_total_package_ambiguity")
    if result.transfer_fee.status == "conflicting_sources":
        reasons.add("conflicting_base_fee_reports")
    if result.transfer_fee.amount is not None and len(result.transfer_fee.evidence_ids) == 1:
        linked_sources = [source for source in result.sources if source.evidence_id in result.transfer_fee.evidence_ids]
        if linked_sources and linked_sources[0].source_type not in {"official", "regulatory", "governing_body"}:
            reasons.add("weak_source_for_exact_financial_term")
    option = result.purchase_option
    if option.status in {"disclosed_yes", "partially_disclosed"} and option.price is None and option.amount is None:
        reasons.add("option_terms_incomplete")
    later_transfer_language = all_text + " " + " ".join(reasons).lower()
    if any(term in later_transfer_language for term in ("later permanent", "permanent transfer", "subsequently signed", "joined permanently", "transitions from loan")) and "loan" in later_transfer_language:
        reasons.add("later_transfer_used_to_interpret_original_deal")
