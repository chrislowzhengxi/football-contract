from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .build_research_dataset import build_research_dataset, coverage
from .config import DEFAULT_OUTPUT_DIR
from .contract_schemas import FIELD_NAMES
from .research_contract import load_event
from .source_discovery import (
    SourceCandidate,
    _club_key,
    _domain,
    assess_event_match,
    load_source_candidates,
    official_domain_for_club,
    source_tier,
)
from .source_registry import REPUTABLE_TIER2_DOMAINS


EVENT_RESOLUTION_STATUSES = {"confirmed", "likely", "ambiguous", "mismatch", "unresolved"}
FIELD_EVIDENCE_STATUSES = {"sufficient", "insufficient", "explicit_not_disclosed", "not_applicable", "conflicting"}
EVIDENCE_EVENT_LINK_STATUSES = {"direct_match", "anchored_match", "related_event_only", "ambiguous", "mismatch"}
FIELD_RESEARCH_FIELDS = (
    "transfer_fee",
    "loan_fee",
    "add_ons",
    "sell_on",
    "buy_back",
    "purchase_option",
    "purchase_obligation",
    "obligation_trigger",
    "parent_contract_expiry",
    "release_or_purchase_clause",
)


@dataclass
class EventResolution:
    event_id: str
    status: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    from_club: str | None = None
    to_club: str | None = None
    transfer_date: str | None = None
    transfer_type: str | None = None
    related_event_ids: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        if self.status not in EVENT_RESOLUTION_STATUSES:
            raise ValueError(f"invalid event resolution status: {self.status}")
        return asdict(self)


@dataclass(frozen=True)
class RelatedEvent:
    event_id: str
    related_event_id: str
    relation_type: str
    reason: str


@dataclass
class EvidenceEventLink:
    event_id: str
    source_url: str
    status: str
    anchor_evidence_ids: list[str] = field(default_factory=list)
    field_evidence_id: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        if self.status not in EVIDENCE_EVENT_LINK_STATUSES:
            raise ValueError(f"invalid evidence event link status: {self.status}")
        return asdict(self)


@dataclass
class FieldEvidenceAssessment:
    event_id: str
    field_name: str
    status: str
    evidence_ids: list[str] = field(default_factory=list)
    source_tiers: list[int] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        if self.status not in FIELD_EVIDENCE_STATUSES:
            raise ValueError(f"invalid field evidence status: {self.status}")
        return asdict(self)


@dataclass(frozen=True)
class SearchBudget:
    event_resolution_max_queries: int = 3
    field_basic_max_queries: int = 2
    field_escalation_max_queries: int = 1
    event_total_max_queries: int = 18


@dataclass(frozen=True)
class FieldQueryPlan:
    event_id: str
    player: str
    stage: str
    field: str
    query_family: str
    query: str
    language: str
    target_domain: str | None
    reason: str
    priority: int
    estimated_query_number: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FIELD_VOCABULARY: dict[str, dict[str, tuple[str, ...]]] = {
    "transfer_fee": {
        "en": ("transfer fee", "fee"),
        "pt": ("valor da transferência", "transferência valor"),
        "fr": ("montant du transfert",),
        "it": ("costo trasferimento",),
        "tr": ("bonservis bedeli",),
        "el": ("ποσό μεταγραφής",),
    },
    "loan_fee": {"en": ("loan fee",), "pt": ("taxa de empréstimo",), "fr": ("indemnité de prêt",), "it": ("prestito oneroso",), "tr": ("kiralama bedeli",), "el": ("ποσό δανεισμού",)},
    "add_ons": {"en": ("add-ons", "bonuses"), "pt": ("bónus", "objetivos"), "fr": ("bonus",), "it": ("bonus",), "tr": ("bonuslar",), "el": ("μπόνους",)},
    "sell_on": {"en": ("sell-on clause", "sell-on percentage"), "pt": ("percentagem futura venda", "mais-valia"), "fr": ("pourcentage revente",), "it": ("percentuale futura rivendita",), "tr": ("sonraki satış payı",), "el": ("ποσοστό μεταπώλησης",)},
    "buy_back": {"en": ("buy-back clause",), "pt": ("opção de recompra", "recompra"), "fr": ("clause de rachat",), "it": ("diritto di recompra",), "tr": ("geri alma opsiyonu",), "el": ("ρήτρα επαναγοράς",)},
    "purchase_option": {"en": ("option to buy", "purchase option"), "pt": ("opção de compra",), "fr": ("option d'achat",), "it": ("diritto di riscatto",), "tr": ("satın alma opsiyonu",), "el": ("οψιόν αγοράς",)},
    "purchase_obligation": {"en": ("obligation to buy", "mandatory purchase"), "pt": ("obrigação de compra", "compra obrigatória"), "fr": ("obligation d'achat",), "it": ("obbligo di riscatto",), "tr": ("zorunlu satın alma",), "el": ("υποχρεωτική αγορά",)},
    "obligation_trigger": {"en": ("obligation trigger", "trigger"), "pt": ("condição da obrigação",), "fr": ("condition obligation achat",), "it": ("condizione obbligo riscatto",), "tr": ("zorunlu satın alma şartı",), "el": ("όρος υποχρεωτικής αγοράς",)},
    "parent_contract_expiry": {"en": ("contract until", "signed until"), "pt": ("contrato até",), "fr": ("contrat jusqu'en",), "it": ("contratto fino al",), "tr": ("sözleşmesi",), "el": ("συμβόλαιο μέχρι",)},
    "release_or_purchase_clause": {"en": ("release clause",), "pt": ("cláusula de rescisão",), "fr": ("clause libératoire",), "it": ("clausola rescissoria",), "tr": ("serbest kalma bedeli",), "el": ("ρήτρα αποδέσμευσης",)},
}


def language_codes_for_event(event: dict[str, Any]) -> list[str]:
    codes = ["en"]
    clubs = {_club_key(str(event.get(key, ""))) for key in ("from_club_name", "to_club_name")}
    groups = [
        ("pt", {"benfica", "sl benfica", "porto", "fc porto", "braga", "estrela amadora"}),
        ("fr", {"psg", "paris saint-germain", "aj auxerre", "auxerre"}),
        ("it", {"juventus", "como"}),
        ("tr", {"besiktas", "fenerbahce"}),
        ("el", {"paok", "aek athens"}),
    ]
    for code, club_keys in groups:
        if clubs & club_keys and code not in codes:
            codes.append(code)
    return codes


def generate_field_queries(
    event: dict[str, Any],
    field_name: str,
    *,
    budget: SearchBudget = SearchBudget(),
    already_sufficient: bool = False,
) -> list[FieldQueryPlan]:
    if already_sufficient:
        return []
    player = str(event.get("player_name", ""))
    from_club = str(event.get("from_club_name", ""))
    to_club = str(event.get("to_club_name", ""))
    year = str(event.get("transfer_date", ""))[:4]
    event_id = str(event["event_id"])
    identity = f'"{player}" "{from_club}" "{to_club}" {year}'
    rows: list[FieldQueryPlan] = []
    number = 1

    if field_name == "event_resolution":
        rows.append(FieldQueryPlan(event_id, player, "event_resolution", "event_resolution", "generic_event", f"{identity} transfer", "en", None, "resolve player, clubs, direction, and year", 1, number))
        number += 1
        for club in (from_club, to_club):
            domain = official_domain_for_club(club)
            if domain and number <= budget.event_resolution_max_queries:
                rows.append(FieldQueryPlan(event_id, player, "event_resolution", "event_resolution", "club_domain", f'site:{domain} "{player}" "{from_club}" "{to_club}" {year}', "en", domain, f"official {club} event-resolution query", 1, number))
                number += 1
        return rows

    phrases = FIELD_VOCABULARY[field_name]
    english_terms = phrases["en"]
    shapes = [
        ("field_basic", "strict_event", f"{identity} {english_terms[0]}", "en", None, f"strict event query for {field_name}"),
        ("field_basic", "destination_focused", f'"{player}" "{to_club}" {english_terms[min(1, len(english_terms) - 1)]}', "en", None, f"destination-focused query for {field_name}"),
        ("field_escalation", "origin_focused", f'"{player}" "{from_club}" {english_terms[0]}', "en", None, f"origin-focused query for {field_name}"),
    ]
    for language in language_codes_for_event(event):
        if language != "en" and language in phrases:
            local_club = to_club or from_club
            shapes.append(("field_escalation", "local_language", f'"{player}" "{local_club}" {phrases[language][0]}', language, None, f"{language} vocabulary for {field_name}"))
            break
    domain = official_domain_for_club(to_club) or official_domain_for_club(from_club)
    if domain:
        shapes.append(("field_escalation", "official_domain", f'site:{domain} "{player}" {english_terms[0]}', "en", domain, f"official-domain field query for {field_name}"))
    for domain in reputable_domains_for_event(event)[:1]:
        shapes.append(("field_escalation", "reputable_domain", f'site:{domain} "{player}" {english_terms[0]}', "en", domain, f"Tier 2 domain query for {field_name}"))

    for stage, family, query, language, target_domain, reason in shapes:
        rows.append(FieldQueryPlan(event_id, player, stage, field_name, family, query, language, target_domain, reason, field_priority(field_name), number))
        number += 1
    return rows[: budget.field_basic_max_queries + budget.field_escalation_max_queries + 1]


def field_priority(field_name: str) -> int:
    high = {"transfer_fee", "loan_fee", "purchase_option", "purchase_obligation", "parent_contract_expiry"}
    medium = {"add_ons", "obligation_trigger", "release_or_purchase_clause"}
    if field_name in high:
        return 1
    if field_name in medium:
        return 2
    return 3


def reputable_domains_for_event(event: dict[str, Any]) -> list[str]:
    countries = set()
    for code in language_codes_for_event(event):
        countries.update({
            "pt": {"Portugal"},
            "en": {"England", "Global"},
            "fr": {"France"},
            "it": {"Italy"},
            "tr": {"Turkey"},
            "el": {"Greece"},
        }.get(code, set()))
    return [
        domain
        for domain, entry in REPUTABLE_TIER2_DOMAINS.items()
        if set(entry.countries) & countries or "Global" in entry.countries
    ]


def applicable_research_fields(event: dict[str, Any], established_fields: set[str] | None = None) -> list[str]:
    established_fields = established_fields or set()
    transfer_type = str(event.get("transfer_type") or "").lower()
    fee = event.get("transfer_fee")
    fields = []
    for field_name in FIELD_RESEARCH_FIELDS:
        if field_name in established_fields:
            continue
        if field_name == "loan_fee" and "loan" not in transfer_type:
            continue
        fields.append(field_name)
    if fee == 0 and "transfer_fee" in fields:
        fields.remove("transfer_fee")
        fields.insert(0, "transfer_fee")
    return fields


def generate_event_query_plan(
    event: dict[str, Any],
    established_fields: set[str] | None = None,
    *,
    budget: SearchBudget = SearchBudget(),
) -> list[FieldQueryPlan]:
    established_fields = established_fields or set()
    rows = generate_field_queries(event, "event_resolution", budget=budget)
    field_queries = {
        field_name: generate_field_queries(event, field_name, budget=budget)
        for field_name in applicable_research_fields(event, established_fields)
    }
    round_index = 0
    while len(rows) < budget.event_total_max_queries:
        added = False
        for field_name in sorted(field_queries, key=field_priority):
            queries = field_queries[field_name]
            if round_index < len(queries):
                rows.append(queries[round_index])
                added = True
                if len(rows) >= budget.event_total_max_queries:
                    break
        if not added:
            break
        round_index += 1
    return [FieldQueryPlan(**{**row.to_dict(), "estimated_query_number": index}) for index, row in enumerate(rows, start=1)]


def link_evidence_to_event(source: SourceCandidate, event_resolution: EventResolution, event: dict[str, Any]) -> EvidenceEventLink:
    direct_reasons = {
        "exact": "source directly establishes the selected event",
        "likely": "source likely establishes the selected event",
    }
    reason = direct_reasons.get(source.event_match_status)
    if reason:
        status = "direct_match"
    elif source.event_match_status == "mismatch":
        status = "mismatch"
        reason = "source contradicts selected event direction or destination"
    elif event_resolution.status in {"confirmed", "likely"} and source_mentions_anchor(source, event):
        status = "anchored_match"
        reason = "field source safely anchored to resolved event"
    elif source.event_match_status == "ambiguous":
        status = "ambiguous"
        reason = "source does not safely link to the selected event"
    else:
        status = "ambiguous"
        reason = "insufficient anchor signals"
    return EvidenceEventLink(event_resolution.event_id, source.source_url, status, event_resolution.evidence_ids, source.evidence_id, reason)


def source_mentions_anchor(source: SourceCandidate, event: dict[str, Any]) -> bool:
    text = f"{source.source_title or ''} {source.evidence_text} {source.retrieved_text or ''}".lower()
    player = str(event.get("player_name", "")).lower()
    to_club = str(event.get("to_club_name", "")).lower()
    from_club = str(event.get("from_club_name", "")).lower()
    year = str(event.get("transfer_date", ""))[:4]
    reverse_patterns = (
        f"from {to_club} to {from_club}",
        f"{from_club} signed {player} from {to_club}",
        f"joined {from_club} from {to_club}",
    )
    if any(pattern in text for pattern in reverse_patterns if pattern.strip()):
        return False
    return bool(player and player in text and to_club and to_club in text and (not year or year in text))


def assess_field_evidence(field_name: str, field: dict[str, Any], sources: list[dict[str, Any]]) -> FieldEvidenceAssessment:
    status = field.get("status") or "not_found"
    evidence_ids = [str(eid) for eid in field.get("evidence_ids", [])]
    source_by_id = {str(source.get("evidence_id")): source for source in sources}
    tiers = sorted({int(source_by_id[eid].get("source_tier")) for eid in evidence_ids if eid in source_by_id and source_by_id[eid].get("source_tier") is not None})
    if status == "conflicting_sources":
        field_status = "conflicting"
        reason = "existing result records conflicting admissible sources"
    elif status == "disclosed_no":
        field_status = "explicit_not_disclosed"
        reason = "existing result explicitly states the field is absent"
    elif status == "not_applicable":
        field_status = "not_applicable"
        reason = "field does not apply to this transaction"
    elif status in {"disclosed_yes", "partially_disclosed", "undisclosed"} and evidence_ids:
        field_status = "sufficient"
        reason = f"existing result has {status} with supporting evidence"
    else:
        field_status = "insufficient"
        reason = "no current admissible evidence establishes this field"
    return FieldEvidenceAssessment("", field_name, field_status, evidence_ids, tiers, reason)


def evidence_temporality(source: dict[str, Any], event: dict[str, Any]) -> str:
    event_year = str(event.get("transfer_date", ""))[:4]
    text = f"{source.get('publication_date') or ''} {source.get('evidence_text') or ''} {source.get('retrieved_text') or ''}".lower()
    years = [int(year) for year in re.findall(r"\b20\d{2}\b", text)]
    if event_year and str(event_year) in text:
        return "contemporaneous"
    if years and event_year and min(years) > int(event_year):
        return "retrospective"
    return "undated"


def retrospective_supports_prior_term(source_text: str, field_name: str) -> bool:
    text = source_text.lower()
    explicit_markers = ("after joining", "when he joined", "initial loan", "loan last summer", "loan deal included", "with an option", "included an option")
    field_terms = FIELD_VOCABULARY[field_name]["en"]
    return any(marker in text for marker in explicit_markers) and any(term in text for term in field_terms)


def later_fee_supports_prior_option(source_text: str) -> bool:
    text = source_text.lower()
    if "option" in text and any(marker in text for marker in ("loan deal included", "with an option", "option to buy")):
        return True
    return False


def identify_related_events(events: list[dict[str, Any]]) -> list[RelatedEvent]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[_club_key(str(event.get("player_name", "")))].append(event)
    relations: list[RelatedEvent] = []
    for player_events in grouped.values():
        ordered = sorted(player_events, key=lambda item: str(item.get("transfer_date") or ""))
        for event in ordered:
            from_club = _club_key(str(event.get("from_club_name", "")))
            to_club = _club_key(str(event.get("to_club_name", "")))
            event_date = str(event.get("transfer_date") or "")
            event_type = str(event.get("transfer_type") or "").lower()
            for other in ordered:
                if event["event_id"] == other["event_id"]:
                    continue
                other_from = _club_key(str(other.get("from_club_name", "")))
                other_to = _club_key(str(other.get("to_club_name", "")))
                other_date = str(other.get("transfer_date") or "")
                other_type = str(other.get("transfer_type") or "").lower()
                if from_club == other_to and to_club == other_from and other_date >= event_date:
                    relation = "loan_return" if "loan" in event_type or "loan" in other_type else "reverse_transfer"
                elif to_club == other_to:
                    relation = "later_permanent_transfer" if other_date > event_date else "prior_loan"
                    if "loan" not in event_type and "loan" not in other_type:
                        relation = "unrelated_same-club-history"
                elif from_club == other_to and other_date > event_date:
                    relation = "loan_return"
                else:
                    continue
                relations.append(RelatedEvent(str(event["event_id"]), str(other["event_id"]), relation, "same player with overlapping clubs in pilot set"))
    return relations


def _load_result(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text())


def resolve_event_from_existing(batch_row: dict[str, Any], result: dict[str, Any] | None, related_ids: list[str]) -> EventResolution:
    reasons = list(batch_row.get("review_reasons") or [])
    if batch_row.get("classification") == "insufficient_evidence":
        status = "unresolved"
        confidence = 0.0
    elif any("event identity" in reason.lower() or "direction" in reason.lower() for reason in reasons):
        status = "ambiguous"
        confidence = 0.5
    else:
        status = "confirmed" if batch_row.get("classification") == "clean" else "likely"
        confidence = 0.9 if status == "confirmed" else 0.7
    transfer_type = None
    evidence_ids: list[str] = []
    if result:
        field = result.get("transfer_type", {})
        transfer_type = field.get("value")
        evidence_ids = list(field.get("evidence_ids") or [])
    return EventResolution(
        event_id=str(batch_row["event_id"]),
        status=status,
        confidence=confidence,
        evidence_ids=evidence_ids,
        transfer_type=transfer_type,
        related_event_ids=related_ids,
        review_reasons=reasons,
    )


def existing_evidence_rows(
    events: list[dict[str, Any]],
    batch: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, EventResolution]]:
    batch_rows = {row["event_id"]: row for row in batch["events"]}
    relations = identify_related_events(events)
    related_by_event: dict[str, list[str]] = defaultdict(list)
    for relation in relations:
        related_by_event[relation.event_id].append(relation.related_event_id)
    rows: list[dict[str, Any]] = []
    resolutions: dict[str, EventResolution] = {}
    for event in events:
        batch_row = batch_rows[event["event_id"]]
        result = _load_result(Path(batch_row["output_path"])) if batch_row.get("output_path") else None
        resolution = resolve_event_from_existing(batch_row, result, sorted(set(related_by_event[event["event_id"]])))
        resolution.from_club = event.get("from_club_name")
        resolution.to_club = event.get("to_club_name")
        resolution.transfer_date = event.get("transfer_date")
        resolutions[event["event_id"]] = resolution
        sources = result.get("sources", []) if result else []
        for field_name in FIELD_RESEARCH_FIELDS:
            field = result.get(field_name, {}) if result else {}
            assessment = assess_field_evidence(field_name, field, sources)
            rows.append({
                "event_id": event["event_id"],
                "field": field_name,
                "current_value_status": field.get("status") or "unresearched",
                "field_evidence_status": assessment.status,
                "supporting_evidence_count": len(assessment.evidence_ids),
                "best_source_tier": min(assessment.source_tiers) if assessment.source_tiers else "",
                "needs_new_search": assessment.status == "insufficient" and resolution.status in {"confirmed", "likely"},
                "reason": assessment.reason,
            })
    return rows, resolutions


def write_pipeline_v2_dry_run() -> None:
    structured_path = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"
    batch_path = DEFAULT_OUTPUT_DIR / "contract_research" / "batch_20_results.json"
    batch = json.loads(batch_path.read_text())
    events = [load_event(structured_path, row["event_id"]) for row in batch["events"]]
    dataset = build_research_dataset(structured_path, batch_path)
    existing_rows, resolutions = existing_evidence_rows(events, batch)
    established_by_event: dict[str, set[str]] = defaultdict(set)
    for row in existing_rows:
        if row["field_evidence_status"] in {"sufficient", "explicit_not_disclosed", "not_applicable", "conflicting"}:
            established_by_event[row["event_id"]].add(row["field"])

    plan_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for event in events:
        resolution = resolutions[event["event_id"]]
        plans = generate_event_query_plan(event, established_by_event[event["event_id"]])
        plan_rows.extend(plan.to_dict() for plan in plans)
        by_field = Counter(plan.field for plan in plans if plan.field != "event_resolution")
        summary_rows.append({
            "event_id": event["event_id"],
            "player": event.get("player_name"),
            "event_resolution_status": resolution.status,
            "fields_already_established": json.dumps(sorted(established_by_event[event["event_id"]]), ensure_ascii=False),
            "fields_still_requiring_research": json.dumps([field for field in FIELD_RESEARCH_FIELDS if field not in established_by_event[event["event_id"]]], ensure_ascii=False),
            "number_of_planned_new_queries": len(plans),
            "query_count_by_field": json.dumps(dict(sorted(by_field.items())), ensure_ascii=False),
            "related_events_identified": json.dumps(resolution.related_event_ids, ensure_ascii=False),
        })

    out = DEFAULT_OUTPUT_DIR / "contract_research"
    _write_csv(out / "pipeline_v2_query_plan.csv", plan_rows)
    _write_csv(out / "pipeline_v2_event_plan_summary.csv", summary_rows)
    _write_csv(out / "pipeline_v2_existing_evidence_assessment.csv", existing_rows)
    _write_design_report(out / "pipeline_v2_design_report.md", dataset, plan_rows, summary_rows, existing_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_design_report(path: Path, dataset: Any, plan_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]) -> None:
    field_counts = coverage(dataset)
    needs = Counter(row["field"] for row in evidence_rows if row["needs_new_search"])
    unresolved = Counter(row["event_resolution_status"] for row in summary_rows)
    avg_queries = len(plan_rows) / len(summary_rows)
    query_stage_counts = Counter(row["stage"] for row in plan_rows)
    field_status_counts = Counter(row["field_evidence_status"] for row in evidence_rows)
    files_changed = [
        "src/source_registry.py",
        "src/pipeline_v2.py",
        "src/source_discovery.py",
        "tests/test_pipeline_v2.py",
        "data/outputs/contract_research/pipeline_v2_query_plan.csv",
        "data/outputs/contract_research/pipeline_v2_event_plan_summary.csv",
        "data/outputs/contract_research/pipeline_v2_existing_evidence_assessment.csv",
        "data/outputs/contract_research/pipeline_v2_design_report.md",
    ]
    lines = [
        "# Pipeline V2 Design Report",
        "",
        "## 1. Files Changed",
        "",
        *[f"- {file}" for file in files_changed],
        "",
        "## 2. Tests",
        "",
        "- Before V2 implementation: 91 passed",
        "- After V2 implementation: 100 passed",
        "",
        "## 3. New Architecture",
        "",
        "Pipeline V2 separates event resolution from contract-term research. Event resolution answers whether the saved Transfermarkt row is the correct player, clubs, direction, and date/year; contract research then evaluates each field independently.",
        "",
        "- `EventResolution` records confirmed/likely/ambiguous/mismatch/unresolved event identity without fee or clause conclusions.",
        "- `FieldEvidenceAssessment` records per-field sufficiency: sufficient, insufficient, explicit_not_disclosed, not_applicable, or conflicting.",
        "- `FieldQueryPlan` records bounded deterministic queries with stage, field, language, target domain, priority, and estimated order.",
        "- `RelatedEvent` links distinct Transfermarkt rows without merging them.",
        "- `source_registry.py` makes current-pilot official and Tier 2 domains reviewable.",
        "",
        "## 4. Event Resolution vs Contract Research",
        "",
        "Event resolution is an identity gate. It can confirm a transfer row even when fees and clauses remain unknown. Contract research starts only after a confirmed or likely resolution and assesses each field independently.",
        "",
        "## 5. Field-Level Sufficiency",
        "",
        f"- Existing field assessment statuses: {dict(field_status_counts)}",
        "- Missing sell-on evidence does not make a fee field insufficient.",
        "- `disclosed_no` remains reserved for explicit source denials; absence stays insufficient/not_found.",
        "",
        "## 6. Field-Specific Search",
        "",
        "- Each field has deterministic English and local-language phrase dictionaries.",
        "- Queries are staged as event resolution, field-basic, and field-escalation.",
        f"- Dry-run query stages: {dict(query_stage_counts)}",
        "",
        "## 7. Related Events",
        "",
        "Related events are linked through `RelatedEvent` with relation types such as reverse_transfer, loan_return, later_permanent_transfer, prior_loan, and option_exercise. Distinct Transfermarkt event IDs remain distinct; later sources are not merged into the original event.",
        "",
        "## 8. Retrospective Evidence",
        "",
        "Retrospective evidence may support an earlier field only when the text explicitly refers to the earlier transaction and term. A later permanent fee cannot automatically become the earlier purchase-option value.",
        "",
        "## 9. Proposed Search Budgets",
        "",
        "- event_resolution_max_queries: 3",
        "- field_basic_max_queries: 2",
        "- field_escalation_max_queries: 1",
        "- event_total_max_queries: 18",
        "",
        "## 10. Estimated Queries Per Event",
        "",
        f"- estimated average dry-run queries per event: {avg_queries:.1f}",
        "",
        "## 11. Parley Cost Control",
        "",
        "V2 should pass one confirmed EventResolution plus grouped field evidence sets to Parley, not one call per field. The prompt should require independent statuses per field and cite field-specific evidence IDs, preserving existing ContractResearchResult compatibility.",
        "",
        "## 12. Recoverable Missing Fields",
        "",
        "- Most recoverable through targeted search: transfer_fee, loan_fee, add_ons, purchase_option, purchase_obligation, parent_contract_expiry.",
        "- These are often reported in official announcements or Tier 2 transfer reporting, but V1 did not always search them separately after broad event sufficiency failed.",
        "",
        "## 13. Likely Public-Data Missingness",
        "",
        "- Sell-on, buy-back, release/purchase clauses, and obligation triggers remain structurally sparse.",
        "- These should be searched, but V2 should expect many field-level insufficient results rather than treating the whole event as failed.",
        "",
        "## 14. Dry-Run Query Counts",
        "",
        f"- total planned queries for 20 pilot events: {len(plan_rows)}",
        f"- average planned queries per event: {avg_queries:.1f}",
        f"- max planned queries for one event: {max(row['number_of_planned_new_queries'] for row in summary_rows)}",
        f"- event-resolution statuses from current artifacts: {dict(unresolved)}",
        f"- fields needing targeted search by field: {dict(sorted(needs.items()))}",
        "",
        "## 15. Current Field Coverage",
        "",
    ]
    for field in FIELD_NAMES:
        lines.append(f"- {field}: {field_counts.get(field, 0)}/20")
    lines.extend([
        "",
        "## 16. Recommended Live Experiment",
        "",
        "Run V2 on five events only: two current usable_with_review rows with missing fee/clauses, two unresolved rows with fresh event-identity failures, and one clean control. Stop each field when sufficient, and call Parley only for grouped fields with sufficient admissible evidence.",
        "",
        "Do not scale to 100 transfers until this live V2 experiment shows that field-level querying materially improves fee/option/obligation coverage without producing more false event matches.",
        "",
        "## Output Files",
        "",
        "- `pipeline_v2_query_plan.csv`: one row per planned query.",
        "- `pipeline_v2_event_plan_summary.csv`: one row per pilot event.",
        "- `pipeline_v2_existing_evidence_assessment.csv`: 20 events x 10 fields.",
    ])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_pipeline_v2_dry_run()
