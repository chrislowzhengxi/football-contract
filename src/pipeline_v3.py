from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .official_page_retrieval import OfficialPageFetcher
from .pipeline_v2 import (
    EvidenceEventLink,
    EventResolution,
    FIELD_RESEARCH_FIELDS,
    FIELD_VOCABULARY,
    FieldQueryPlan,
    SearchBudget,
    field_priority,
    language_codes_for_event,
    link_evidence_to_event,
)
from .pipeline_v21_diagnostics import field_signal, load_saved_sources
from .pipeline_v21_live3 import assess_linked_field
from .contract_schemas import render_model_schema_instructions
from .research_contract import ParleyProvider, ProviderFailure, load_event
from .source_discovery import (
    SourceCandidate,
    _domain,
    _source_type,
    deduplicate_sources,
    filter_admissible_sources,
    official_domain_for_club,
    score_source,
    source_tier,
    TavilySearchProvider,
    assess_event_match,
)
from .source_registry import OFFICIAL_CLUB_DOMAINS, REPUTABLE_TIER2_DOMAINS


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
BATCH_PATH = OUTPUT_DIR / "batch_20_results.json"
STRUCTURED_PATH = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"

CORE_FIELDS = ("transfer_fee", "loan_fee", "purchase_option", "purchase_obligation", "parent_contract_expiry")
SECONDARY_FIELDS = ("add_ons", "obligation_trigger", "release_or_purchase_clause")
OPPORTUNISTIC_FIELDS = ("sell_on", "buy_back")
SOURCE_CLASSES = (
    "official_buying_club",
    "official_selling_club",
    "official_other_club",
    "regulatory_disclosure",
    "stock_exchange_disclosure",
    "financial_filing",
    "major_national_media",
    "reputable_local_sports_media",
    "reputable_transfer_specialist",
    "retrospective_reporting",
    "other_tier2",
)
FIELD_SOURCE_TARGETS = {
    "transfer_fee": ("regulatory_disclosure", "financial_filing", "official_buying_club", "official_selling_club", "major_national_media", "reputable_local_sports_media", "reputable_transfer_specialist"),
    "loan_fee": ("official_buying_club", "official_selling_club", "regulatory_disclosure", "major_national_media", "reputable_local_sports_media", "reputable_transfer_specialist"),
    "purchase_option": ("official_buying_club", "official_selling_club", "major_national_media", "reputable_local_sports_media", "reputable_transfer_specialist"),
    "purchase_obligation": ("official_buying_club", "official_selling_club", "regulatory_disclosure", "financial_filing", "major_national_media", "reputable_local_sports_media"),
    "parent_contract_expiry": ("official_buying_club", "official_selling_club", "major_national_media"),
    "add_ons": ("regulatory_disclosure", "financial_filing", "major_national_media", "reputable_local_sports_media", "reputable_transfer_specialist"),
    "obligation_trigger": ("official_buying_club", "official_selling_club", "major_national_media", "reputable_local_sports_media"),
    "release_or_purchase_clause": ("regulatory_disclosure", "financial_filing", "major_national_media", "reputable_local_sports_media"),
    "sell_on": ("regulatory_disclosure", "financial_filing", "reputable_local_sports_media"),
    "buy_back": ("regulatory_disclosure", "financial_filing", "reputable_local_sports_media"),
}
SPECIALIST_CANDIDATES = {
    "footballtransfers.com": {"specialist_candidate": True, "reviewed": False, "tier": 3},
    "rotowire.com": {"specialist_candidate": True, "reviewed": False, "tier": 3},
}
ALLOWED_LIVE_ENV_KEYS = {"TAVILY_API_KEY"}
ALLOWED_PARLEY_ENV_KEYS = {"PARLEY_API_KEY"}
FINANCIAL_DOMAINS_BY_CLUB = {
    "benfica": ("slbenfica.pt",),
    "sl benfica": ("slbenfica.pt",),
    "porto": ("fcporto.pt",),
    "fc porto": ("fcporto.pt",),
    "juventus": ("juventus.com",),
}


@dataclass(frozen=True)
class V3Budget:
    event_resolution_max_queries: int = 3
    core_field_basic_queries: int = 1
    core_field_escalation_queries: int = 2
    secondary_field_queries: int = 2
    opportunistic_field_queries: int = 1
    event_total_max_queries: int = 20


@dataclass(frozen=True)
class V3Query:
    event_id: str
    player: str
    field: str
    field_class: str
    round: str
    query_family: str
    query: str
    language: str
    target_domain: str | None
    target_source_class: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def field_class(field: str) -> str:
    if field in CORE_FIELDS:
        return "core"
    if field in SECONDARY_FIELDS:
        return "secondary"
    return "opportunistic"


def classify_source(source: SourceCandidate, event: dict[str, Any]) -> str:
    domain = _domain(source.source_url)
    if domain == official_domain_for_club(str(event.get("to_club_name", ""))):
        return "official_buying_club"
    if domain == official_domain_for_club(str(event.get("from_club_name", ""))):
        return "official_selling_club"
    if domain in {entry.domain for entry in OFFICIAL_CLUB_DOMAINS.values()}:
        return "official_other_club"
    if any(token in domain for token in ("stockexchange", "borsa", "cmvm")):
        return "stock_exchange_disclosure"
    if any(token in domain for token in ("investor", "annualreport", "financial", "sec.gov")):
        return "financial_filing"
    if source.source_type in {"regulatory", "governing_body"}:
        return "regulatory_disclosure"
    entry = REPUTABLE_TIER2_DOMAINS.get(domain)
    if entry and entry.source_type == "major_news":
        return "major_national_media"
    if entry and entry.source_type == "football_reporting":
        return "reputable_local_sports_media"
    if domain in SPECIALIST_CANDIDATES:
        return "reputable_transfer_specialist"
    if source.source_tier == 2:
        return "other_tier2"
    return "retrospective_reporting" if _looks_retrospective(source, event) else "other_tier2"


def _looks_retrospective(source: SourceCandidate, event: dict[str, Any]) -> bool:
    event_year = str(event.get("transfer_date", ""))[:4]
    text = f"{source.publication_date or ''} {source.source_title or ''} {source.evidence_text or ''} {source.retrieved_text or ''}"
    return bool(event_year and any(year > event_year for year in ("2026", "2027", "2028", "2029", "2030") if year in text))


def configured_domains_for_source_class(event: dict[str, Any], source_class: str) -> list[str]:
    if source_class == "official_buying_club":
        return [domain for domain in [official_domain_for_club(str(event.get("to_club_name", "")))] if domain]
    if source_class == "official_selling_club":
        return [domain for domain in [official_domain_for_club(str(event.get("from_club_name", "")))] if domain]
    if source_class in {"regulatory_disclosure", "stock_exchange_disclosure", "financial_filing"}:
        domains = []
        for club in (str(event.get("from_club_name", "")).lower(), str(event.get("to_club_name", "")).lower()):
            domains.extend(FINANCIAL_DOMAINS_BY_CLUB.get(club, ()))
        return sorted(set(domains))
    if source_class == "major_national_media":
        return [domain for domain, entry in REPUTABLE_TIER2_DOMAINS.items() if entry.source_type == "major_news"][:2]
    if source_class == "reputable_local_sports_media":
        languages = set(language_codes_for_event(event))
        country_by_language = {"pt": "Portugal", "fr": "France", "it": "Italy", "el": "Greece", "en": "England"}
        countries = {country_by_language[language] for language in languages if language in country_by_language}
        return [domain for domain, entry in REPUTABLE_TIER2_DOMAINS.items() if entry.source_type == "football_reporting" and set(entry.countries) & countries][:2]
    if source_class == "reputable_transfer_specialist":
        return []
    return []


def v3_queries_for_field(event: dict[str, Any], field: str, budget: V3Budget = V3Budget()) -> list[V3Query]:
    player = str(event.get("player_name", ""))
    from_club = str(event.get("from_club_name", ""))
    to_club = str(event.get("to_club_name", ""))
    year = str(event.get("transfer_date", ""))[:4]
    term = FIELD_VOCABULARY[field]["en"][0]
    klass = field_class(field)
    max_queries = {"core": budget.core_field_basic_queries + budget.core_field_escalation_queries, "secondary": budget.secondary_field_queries, "opportunistic": budget.opportunistic_field_queries}[klass]
    rows = [
        V3Query(str(event["event_id"]), player, field, klass, "round_1", "broad_event_field", f'"{player}" "{from_club}" "{to_club}" {year} {term}', "en", None, "broad", field_priority(field)),
    ]
    round_name = "round_2"
    for source_class in FIELD_SOURCE_TARGETS[field]:
        for domain in configured_domains_for_source_class(event, source_class)[:1]:
            rows.append(V3Query(str(event["event_id"]), player, field, klass, round_name, "source_class_domain", f'site:{domain} "{player}" {term}', "en", domain, source_class, field_priority(field)))
            round_name = "round_3"
            if len(rows) >= max_queries:
                return rows
    for language in language_codes_for_event(event):
        if language != "en" and language in FIELD_VOCABULARY[field] and len(rows) < max_queries:
            rows.append(V3Query(str(event["event_id"]), player, field, klass, "round_3", "local_language", f'"{player}" "{to_club}" {FIELD_VOCABULARY[field][language][0]}', language, None, "local_language", field_priority(field)))
            break
    return rows[:max_queries]


def v3_event_plan(event: dict[str, Any], established: set[str] | None = None, budget: V3Budget = V3Budget()) -> list[V3Query]:
    established = established or set()
    player = str(event.get("player_name", ""))
    from_club = str(event.get("from_club_name", ""))
    to_club = str(event.get("to_club_name", ""))
    year = str(event.get("transfer_date", ""))[:4]
    rows = [V3Query(str(event["event_id"]), player, "event_resolution", "event", "event_resolution", "broad_event", f'"{player}" "{from_club}" "{to_club}" {year} transfer', "en", None, "broad", 0)]
    for club, source_class in ((to_club, "official_buying_club"), (from_club, "official_selling_club")):
        domain = official_domain_for_club(club)
        if domain and len(rows) < budget.event_resolution_max_queries:
            rows.append(V3Query(str(event["event_id"]), player, "event_resolution", "event", "event_resolution", "official_event_domain", f'site:{domain} "{player}" "{from_club}" "{to_club}" {year}', "en", domain, source_class, 0))
    for group in (CORE_FIELDS, SECONDARY_FIELDS, OPPORTUNISTIC_FIELDS):
        grouped_queries = {
            field: v3_queries_for_field(event, field, budget)
            for field in group
            if field not in established
        }
        round_index = 0
        while grouped_queries and len(rows) < budget.event_total_max_queries:
            added = False
            for field in group:
                if field not in grouped_queries:
                    continue
                queries = grouped_queries[field]
                if round_index < len(queries):
                    rows.append(queries[round_index])
                    added = True
                    if len(rows) >= budget.event_total_max_queries:
                        return rows[:budget.event_total_max_queries]
            if not added:
                break
            round_index += 1
        if len(rows) >= budget.event_total_max_queries:
            return rows[:budget.event_total_max_queries]
    return rows[:budget.event_total_max_queries]


def _result_status(event_id: str, field: str) -> str:
    batch = json.loads(BATCH_PATH.read_text())
    row = next(item for item in batch["events"] if item["event_id"] == event_id)
    if not row.get("output_path"):
        return "unresearched"
    result = json.loads(Path(row["output_path"]).read_text())
    return result.get(field, {}).get("status") or "unresearched"


def current_established_fields(event_id: str) -> set[str]:
    statuses = {field: _result_status(event_id, field) for field in FIELD_RESEARCH_FIELDS}
    return {field for field, status in statuses.items() if status not in {"not_found", "unresearched"}}


def offline_source_replay() -> list[dict[str, Any]]:
    batch = json.loads(BATCH_PATH.read_text())
    rows = []
    for item in batch["events"]:
        event = load_event(STRUCTURED_PATH, item["event_id"])
        resolution = EventResolution(str(event["event_id"]), "likely" if item.get("classification") != "insufficient_evidence" else "unresolved", 0.7)
        sources = [source for source in load_saved_sources(str(event["event_id"])) if source.source_tier in {1, 2}]
        for field in FIELD_RESEARCH_FIELDS:
            signal_sources = [source for source in sources if field_signal(field, source)]
            links = [link_evidence_to_event(source, resolution, event) for source in signal_sources]
            classes = sorted({classify_source(source, event) for source in signal_sources})
            prioritized = any(source_class in FIELD_SOURCE_TARGETS[field] for source_class in classes)
            rows.append({
                "event_id": event["event_id"],
                "field": field,
                "current_evidence_status": _result_status(str(event["event_id"]), field),
                "source_classes_found": json.dumps(classes, ensure_ascii=False),
                "best_quality_tier": min((source.source_tier for source in signal_sources), default=""),
                "direct_matches": sum(1 for link in links if link.status == "direct_match"),
                "anchored_matches": sum(1 for link in links if link.status == "anchored_match"),
                "field_signals": len(signal_sources),
                "v3_would_prioritize": prioritized,
                "live_search_still_needed": _result_status(str(event["event_id"]), field) in {"not_found", "unresearched"} and not signal_sources,
                "reason": "saved prioritized source signal exists" if prioritized else "no prioritized saved source signal",
            })
    path = OUTPUT_DIR / "pipeline_v3_offline_source_replay.csv"
    _write_csv(path, rows)
    return rows


def select_live5(replay: list[dict[str, Any]]) -> list[dict[str, Any]]:
    choices = (
        ("tm_64a02c80719320e6347a", ("transfer_fee", "parent_contract_expiry"), "permanent fee case with missing core fields"),
        ("tm_211885c2a2cbd065928b", ("parent_contract_expiry", "add_ons"), "loan/option case with missing expiry and secondary fields"),
        ("tm_163d23e38fec40c7a994", ("add_ons", "parent_contract_expiry"), "obligation case with remaining missing fields"),
        ("tm_890a6e2a4b41a1f8dd46", ("transfer_fee",), "lower-profile permanent control with missing fee"),
        ("tm_694c30a10815aea6d84d", ("release_or_purchase_clause", "sell_on"), "high-profile permanent control with sparse-clause targets"),
    )
    rows = []
    for event_id, fields, reason in choices:
        event = load_event(STRUCTURED_PATH, event_id)
        rows.append({
            "event_id": event_id,
            "player": event.get("player_name"),
            "departing_club": event.get("from_club_name"),
            "receiving_club": event.get("to_club_name"),
            "transfer_date": event.get("transfer_date"),
            "selection_reason": reason,
            "target_fields": list(fields),
            "expected_source_classes": sorted({source_class for field in fields for source_class in FIELD_SOURCE_TARGETS[field][:4]}),
        })
    _write_json(OUTPUT_DIR / "pipeline_v3_live5_selection.json", rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_tavily_env_only(path: Path = Path(".pytest_cache/.env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in ALLOWED_LIVE_ENV_KEYS and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _load_parley_env_only(path: Path = Path(".pytest_cache/.env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in ALLOWED_PARLEY_ENV_KEYS and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def write_baseline() -> None:
    analysis = pd.read_csv(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis.csv")
    raw = pd.read_csv(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20.csv")
    coverage = {
        field: int(raw[f"researched_{field}_status"].isin({"disclosed_yes", "partially_disclosed", "conflicting_sources", "undisclosed"}).sum())
        for field in FIELD_RESEARCH_FIELDS
    }
    lines = [
        "# Pipeline V3 Baseline",
        "",
        f"- Tests before V3: run result recorded separately.",
        f"- Classifications: {analysis['classification'].value_counts().to_dict()}",
        f"- Field coverage/status-bearing counts: {coverage}",
        f"- Official club aliases in registry: {len(OFFICIAL_CLUB_DOMAINS)}",
        f"- Tier 2 domains in registry: {len(REPUTABLE_TIER2_DOMAINS)}",
        "- V3 core fields: " + ", ".join(CORE_FIELDS),
        "- V3 secondary fields: " + ", ".join(SECONDARY_FIELDS),
        "- V3 opportunistic fields: " + ", ".join(OPPORTUNISTIC_FIELDS),
        "- V3 event total query cap: 20.",
    ]
    (OUTPUT_DIR / "pipeline_v3_baseline.md").write_text("\n".join(lines) + "\n")


def write_specialist_review() -> None:
    lines = ["# Pipeline V3 Specialist Registry Review", ""]
    for domain, meta in SPECIALIST_CANDIDATES.items():
        lines.extend([
            f"## {domain}",
            "",
            f"- Current tier: {meta['tier']}",
            f"- specialist_candidate: {meta['specialist_candidate']}",
            f"- reviewed: {meta['reviewed']}",
            "- Saved examples: see `pipeline_v21_registry_candidates.md`.",
            "- Fields it tends to report in saved artifacts: transfer/player profile signals, possible option/contract snippets.",
            "- Recommendation: needs_human_review; do not allow independent exact financial claims until reviewed/promoted.",
            "- Rationale: candidate status is based on saved-result behavior only, not verified editorial reliability.",
            "",
        ])
    (OUTPUT_DIR / "pipeline_v3_specialist_registry_review.md").write_text("\n".join(lines))


def _search_query(tavily: TavilySearchProvider, event: dict[str, Any], query: V3Query) -> list[SourceCandidate]:
    payload = tavily._search(query.query)
    rows = []
    for rank, item in enumerate(payload.get("results", []), start=1):
        url = item.get("url")
        if not url:
            continue
        source = SourceCandidate(
            source_url=url,
            source_title=item.get("title"),
            publisher=_domain(url),
            publication_date=item.get("published_date"),
            source_type=_source_type(url, event),
            evidence_text=item.get("content") or "",
            language=item.get("language") or query.language,
            search_query=query.query,
            search_rank=rank,
            provider_score=item.get("score"),
            query_family=query.query_family,
            query_language=query.language,
            target_domain=query.target_domain,
            query_reason=query.target_source_class,
        )
        source.source_tier = source_tier(source)
        source.event_match_status, source.event_match_score, source.event_match_reasons = assess_event_match(source, event)
        source.quality_score = score_source(source, event)
        rows.append(source)
    return rows


def _retrieve(sources: list[SourceCandidate], event: dict[str, Any], fetcher: OfficialPageFetcher) -> list[SourceCandidate]:
    for source in sources:
        if source.source_tier not in {1, 2}:
            continue
        fetcher.apply_to_candidate(source)
        source.event_match_status, source.event_match_score, source.event_match_reasons = assess_event_match(source, event)
        source.quality_score = score_source(source, event)
    return sources


def live5() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _load_tavily_env_only()
    selection = _read_json(OUTPUT_DIR / "pipeline_v3_live5_selection.json")
    tavily = TavilySearchProvider.from_environment()
    fetcher = OfficialPageFetcher()
    query_rows = []
    field_rows = []
    yield_rows = []
    for selected in selection:
        event = load_event(STRUCTURED_PATH, selected["event_id"])
        established = current_established_fields(selected["event_id"])
        resolution = EventResolution(str(event["event_id"]), "unresolved", 0.0)
        all_sources: list[SourceCandidate] = []
        field_support: dict[str, list[SourceCandidate]] = defaultdict(list)
        field_status = {field: "insufficient" for field in FIELD_RESEARCH_FIELDS}
        for query in v3_event_plan(event, established):
            if query.field != "event_resolution" and resolution.status not in {"confirmed", "likely"}:
                continue
            if query.field != "event_resolution" and field_status.get(query.field) == "sufficient":
                continue
            found = _retrieve(_search_query(tavily, event, query), event, fetcher)
            all_sources = deduplicate_sources(all_sources + found)
            if query.field == "event_resolution":
                admissible = filter_admissible_sources(all_sources)
                if any(source.event_match_status == "exact" for source in admissible):
                    resolution = EventResolution(str(event["event_id"]), "confirmed", max(source.event_match_score or 0 for source in admissible))
                elif admissible:
                    resolution = EventResolution(str(event["event_id"]), "likely", max(source.event_match_score or 0 for source in admissible))
            else:
                links = [link_evidence_to_event(source, resolution, event) for source in found if source.source_tier in {1, 2}]
                supporting = [
                    source for source in found
                    if source.source_tier in {1, 2}
                    and field_signal(query.field, source)
                    and classify_source(source, event) in FIELD_SOURCE_TARGETS[query.field]
                    and link_evidence_to_event(source, resolution, event).status in {"direct_match", "anchored_match"}
                ]
                field_support[query.field].extend(supporting)
                if supporting:
                    field_status[query.field] = "sufficient"
            query_rows.append({
                "event_id": event["event_id"],
                "player": event.get("player_name"),
                "field": query.field,
                "field_class": query.field_class,
                "round": query.round,
                "query_family": query.query_family,
                "query": query.query,
                "language": query.language,
                "target_domain": query.target_domain,
                "target_source_class": query.target_source_class,
                "raw_results": len(found),
                "tier1_results": sum(1 for source in found if source.source_tier == 1),
                "tier2_results": sum(1 for source in found if source.source_tier == 2),
                "pages_retrieved": sum(1 for source in found if source.source_tier in {1, 2} and source.retrieval_status == "success"),
                "direct_matches": sum(1 for source in found if source.source_tier in {1, 2} and link_evidence_to_event(source, resolution, event).status == "direct_match"),
                "anchored_matches": sum(1 for source in found if source.source_tier in {1, 2} and link_evidence_to_event(source, resolution, event).status == "anchored_match"),
                "field_signal_found": any(field_signal(query.field, source) for source in found) if query.field != "event_resolution" else False,
                "field_sufficient_after_query": field_status.get(query.field) == "sufficient",
                "stop_reason": "field sufficient" if field_status.get(query.field) == "sufficient" else "continue or budget exhausted",
            })
        for field in FIELD_RESEARCH_FIELDS:
            baseline = _result_status(selected["event_id"], field)
            support = deduplicate_sources(field_support[field])
            field_rows.append({
                "event_id": event["event_id"],
                "player": event.get("player_name"),
                "field": field,
                "field_class": field_class(field),
                "baseline_status": baseline,
                "final_evidence_status": field_status[field] if baseline in {"not_found", "unresearched"} else baseline,
                "newly_sufficient": baseline in {"not_found", "unresearched"} and bool(support),
                "supporting_source_count": len(support),
                "best_source_tier": min((source.source_tier for source in support), default=""),
                "source_classes_used": json.dumps(sorted({classify_source(source, event) for source in support}), ensure_ascii=False),
                "queries_executed": sum(1 for row in query_rows if row["event_id"] == event["event_id"] and row["field"] == field),
                "reason": "source-class targeted evidence found" if support else "not disclosed by targeted source classes",
                "would_be_parley_ready": baseline in {"not_found", "unresearched"} and bool(support),
            })
        _write_json(OUTPUT_DIR / "pipeline_v3_sources" / f"{event['event_id']}.json", {"event_id": event["event_id"], "event_resolution": resolution.to_dict(), "sources": [source.to_dict() for source in all_sources]})
    _write_csv(OUTPUT_DIR / "pipeline_v3_live5_query_log.csv", query_rows)
    _write_csv(OUTPUT_DIR / "pipeline_v3_live5_field_results.csv", field_rows)
    ydf = pd.DataFrame(query_rows)
    fdf = pd.DataFrame(field_rows)
    for source_class in SOURCE_CLASSES:
        class_queries = ydf[ydf["target_source_class"] == source_class]
        supported = fdf[fdf["source_classes_used"].str.contains(source_class, regex=False)]
        yield_rows.append({
            "source_class": source_class,
            "queries_run": len(class_queries),
            "events_hit": int(class_queries["event_id"].nunique()) if not class_queries.empty else 0,
            "tier1_results": int(class_queries["tier1_results"].sum()) if not class_queries.empty else 0,
            "tier2_results": int(class_queries["tier2_results"].sum()) if not class_queries.empty else 0,
            "field_signals": int(class_queries["field_signal_found"].sum()) if not class_queries.empty else 0,
            "direct_anchored_matches": int(class_queries["direct_matches"].sum() + class_queries["anchored_matches"].sum()) if not class_queries.empty else 0,
            "newly_sufficient_fields": int(supported["newly_sufficient"].sum()) if not supported.empty else 0,
            "fields_supported": json.dumps(sorted(supported["field"].unique()), ensure_ascii=False),
        })
    _write_csv(OUTPUT_DIR / "pipeline_v3_source_class_yield.csv", yield_rows)
    return query_rows, field_rows, yield_rows


def _source_from_dict(payload: dict[str, Any]) -> SourceCandidate:
    allowed = SourceCandidate.__dataclass_fields__
    return SourceCandidate(**{key: payload.get(key) for key in allowed})


def _parse_cost_header(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(value.removeprefix("$"))
    except ValueError:
        return 0.0


def _has_amount_signal(source: SourceCandidate) -> bool:
    text = f"{source.evidence_text or ''} {source.retrieved_text or ''}".lower()
    return any(token in text for token in ("€", "eur", "euro", "euros", "million", "milhão", "milhões"))


def _parley_provider() -> ParleyProvider:
    _load_parley_env_only()
    api_key = os.environ.get("PARLEY_API_KEY")
    if not api_key:
        raise RuntimeError("PARLEY_API_KEY is required for V3 candidate extraction")
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "contract_research.md"
    prompt = prompt_path.read_text() + "\n\n" + render_model_schema_instructions()
    return ParleyProvider(api_key, os.environ.get("PARLEY_MODEL", "bedrock/claude-haiku-4-5"), prompt)


def _supporting_sources(event: dict[str, Any], target_fields: set[str]) -> list[SourceCandidate]:
    payload = _read_json(OUTPUT_DIR / "pipeline_v3_sources" / f"{event['event_id']}.json")
    resolution = EventResolution(**payload["event_resolution"])
    sources = [_source_from_dict(source) for source in payload["sources"]]
    supporting: list[SourceCandidate] = []
    for source in sources:
        if source.source_tier not in {1, 2}:
            continue
        link = link_evidence_to_event(source, resolution, event)
        if link.status not in {"direct_match", "anchored_match"}:
            continue
        if any(
            field_signal(field, source)
            and classify_source(source, event) in FIELD_SOURCE_TARGETS[field]
            and (field != "transfer_fee" or _has_amount_signal(source))
            for field in target_fields
        ):
            supporting.append(source)
    for index, source in enumerate(sorted(supporting, key=lambda item: item.quality_score, reverse=True), start=1):
        source.evidence_id = f"s{index}"
    return supporting


def parley_candidate_extractions() -> dict[str, Any]:
    field_results = pd.read_csv(OUTPUT_DIR / "pipeline_v3_live5_field_results.csv")
    newly = field_results[field_results["newly_sufficient"] == True]  # noqa: E712 - pandas boolean mask
    candidates_dir = OUTPUT_DIR / "pipeline_v3_candidates"
    attempts = 0
    successful = 0
    total_cost = 0.0
    rows = []
    if newly.empty:
        payload = {"attempts": 0, "successful_extractions": 0, "total_cost": 0.0, "events": []}
        _write_json(candidates_dir / "pipeline_v3_candidate_log.json", payload)
        return payload

    provider = _parley_provider()
    for event_id, group in newly.groupby("event_id"):
        event = load_event(STRUCTURED_PATH, str(event_id))
        target_fields = set(group["field"])
        sources = _supporting_sources(event, target_fields)
        if not sources:
            rows.append({"event_id": event_id, "target_fields": sorted(target_fields), "attempts": 0, "success": False, "cost": 0.0, "reason": "no supporting sources after reload"})
            continue
        event_attempts = 0
        event_cost = 0.0
        success = False
        reason = ""
        for attempt_number in (1, 2):
            attempts += 1
            event_attempts += 1
            try:
                result = provider.research(event, sources)
                cost = result.provider_metadata.total_request_cost or 0.0
                total_cost += cost
                event_cost += cost
                _write_json(candidates_dir / f"{event_id}.json", result.to_dict())
                successful += 1
                success = True
                reason = "candidate extracted"
                break
            except ProviderFailure as error:
                cost = _parse_cost_header(error.diagnostics.get("parley_cost_header"))
                total_cost += cost
                event_cost += cost
                reason = str(error)
                _write_json(candidates_dir / "failures" / f"{event_id}_attempt_{attempt_number}.json", error.diagnostics)
                if error.diagnostics.get("failure_stage") != "schema_validation":
                    break
        rows.append({"event_id": event_id, "target_fields": sorted(target_fields), "attempts": event_attempts, "success": success, "cost": event_cost, "reason": reason})
    payload = {"attempts": attempts, "successful_extractions": successful, "total_cost": total_cost, "events": rows}
    _write_json(candidates_dir / "pipeline_v3_candidate_log.json", payload)
    return payload


def _public_disclosure_classification(row: dict[str, Any]) -> str:
    if row.get("newly_sufficient"):
        return "resolved_by_v3"
    baseline = row.get("baseline_status")
    if baseline not in {"not_found", "unresearched"}:
        return "already_established_or_not_applicable"
    if int(row.get("queries_executed") or 0) == 0:
        return "query_budget_exhausted"
    if int(row.get("supporting_source_count") or 0) == 0:
        return "credible_source_found_but_field_not_disclosed"
    return "other"


def write_public_disclosure_classification(field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in field_rows:
        item = dict(row)
        item["public_disclosure_classification"] = _public_disclosure_classification(row)
        rows.append(item)
    if rows:
        _write_csv(OUTPUT_DIR / "pipeline_v3_public_disclosure_classification.csv", rows)
    return rows


def write_report(
    live_ran: bool,
    query_rows: list[dict[str, Any]] | None = None,
    field_rows: list[dict[str, Any]] | None = None,
    yield_rows: list[dict[str, Any]] | None = None,
    parley_attempts: int = 0,
    parley_successful: int = 0,
    parley_cost: float = 0.0,
) -> None:
    replay = pd.read_csv(OUTPUT_DIR / "pipeline_v3_offline_source_replay.csv")
    selection = _read_json(OUTPUT_DIR / "pipeline_v3_live5_selection.json")
    field_rows = field_rows or []
    query_rows = query_rows or []
    yield_rows = yield_rows or []
    classified_rows = write_public_disclosure_classification(field_rows) if field_rows else []
    newly = [row for row in field_rows if row.get("newly_sufficient")]
    missing_core = [row for row in field_rows if row.get("field") in CORE_FIELDS and row.get("baseline_status") in {"not_found", "unresearched"}]
    newly_core = [row for row in newly if row.get("field_class") == "core"]
    event_improvements = {row["event_id"] for row in newly}
    core_recovery_rate = len(newly_core) / len(missing_core) if missing_core else 0.0
    event_improvement_rate = len(event_improvements) / len(selection) if selection else 0.0
    failures = Counter()
    failure_classes = Counter(row["public_disclosure_classification"] for row in classified_rows if row["public_disclosure_classification"] not in {"resolved_by_v3", "already_established_or_not_applicable"})
    for row in field_rows:
        if row.get("newly_sufficient"):
            continue
        if row["baseline_status"] in {"not_found", "unresearched"}:
            failures[row["field"]] += 1
    lines = [
        "# Pipeline V3 Report",
        "",
        "## Why V3",
        "",
        "V3 was introduced because V2.1 improved query allocation and event-link guardrails but did not recover new fields. The limiting factor appeared to be source availability/public disclosure, so V3 targets source classes by field.",
        "",
        "## Field Classes",
        "",
        f"- Core: {', '.join(CORE_FIELDS)}",
        f"- Secondary: {', '.join(SECONDARY_FIELDS)}",
        f"- Opportunistic: {', '.join(OPPORTUNISTIC_FIELDS)}",
        "",
        "## Source Classes",
        "",
        "- " + ", ".join(SOURCE_CLASSES),
        "",
        "## Targeting Rules",
        "",
        "Core fields are prioritized before secondary fields; sell_on and buy_back run only opportunistically when budget remains. Transfer-specialist candidates remain Tier 3 unless reviewed.",
        "",
        "## Registry Coverage",
        "",
        f"- Official aliases: {len(OFFICIAL_CLUB_DOMAINS)}",
        f"- Tier 2 domains: {len(REPUTABLE_TIER2_DOMAINS)}",
        "- Weak coverage remains for regulatory/financial domains and reviewed transfer-specialist sources.",
        "",
        "## Offline Replay",
        "",
        f"- Rows: {len(replay)}",
        f"- Saved prioritized source signals: {int(replay['v3_would_prioritize'].sum())}",
        f"- Live search still needed rows: {int(replay['live_search_still_needed'].sum())}",
        "",
        "## Selected Live5 Cases",
        "",
    ]
    for row in selection:
        lines.append(f"- {row['event_id']} {row['player']}: targets {row['target_fields']}; {row['selection_reason']}")
    lines.extend([
        "",
        "## Live Results",
        "",
        f"- Live Tavily run performed: {live_ran}",
        f"- Tavily queries: {len(query_rows)}",
        f"- Raw results: {sum(int(row['raw_results']) for row in query_rows) if query_rows else 0}",
        f"- Tier 1 results: {sum(int(row['tier1_results']) for row in query_rows) if query_rows else 0}",
        f"- Tier 2 results: {sum(int(row['tier2_results']) for row in query_rows) if query_rows else 0}",
        f"- Pages retrieved: {sum(int(row['pages_retrieved']) for row in query_rows) if query_rows else 0}",
        "",
        "## Source-Class Yield",
        "",
        json.dumps(yield_rows, ensure_ascii=False),
        "",
        "## Field Recovery",
        "",
        f"- Newly sufficient core fields: {[row for row in newly if row['field_class'] == 'core']}",
        f"- Core recovery rate: {len(newly_core)} / {len(missing_core)} = {core_recovery_rate:.2%}",
        f"- Newly sufficient secondary fields: {[row for row in newly if row['field_class'] == 'secondary']}",
        f"- Opportunistic results: {[row for row in newly if row['field_class'] == 'opportunistic']}",
        f"- Event improvement rate: {len(event_improvements)} / {len(selection)} = {event_improvement_rate:.2%}",
        f"- Parley attempts/successful/cost: {parley_attempts} / {parley_successful} / {parley_cost:.6f}",
        "",
        "## Remaining Failure Taxonomy",
        "",
        f"- Still missing by field: {dict(failures)}",
        f"- Public-disclosure classifications: {dict(failure_classes)}",
        "- likely_public_nondisclosure remains likely for sell_on and buy_back.",
        "",
        "## Scale-Readiness Decision",
        "",
        "C. V3 still produces very low core-field recovery: public observability is the dominant limitation; narrow the empirical dataset before scaling." if live_ran and core_recovery_rate < 0.3 else "B. V3 improves selected fields/source classes: keep a narrower field-aware V3 and then test 50-100 events." if newly else "D. Source registry/discovery is still insufficient: perform one more source-strategy expansion before scaling.",
    ])
    (OUTPUT_DIR / "pipeline_v3_report.md").write_text("\n".join(lines) + "\n")


def prepare() -> None:
    write_baseline()
    write_specialist_review()
    replay = offline_source_replay()
    select_live5(replay)
    write_report(False)


if __name__ == "__main__":
    if "--parley-candidates-only" in sys.argv:
        candidate_log = parley_candidate_extractions()
        query_rows = pd.read_csv(OUTPUT_DIR / "pipeline_v3_live5_query_log.csv").to_dict("records")
        field_rows = pd.read_csv(OUTPUT_DIR / "pipeline_v3_live5_field_results.csv").to_dict("records")
        yield_rows = pd.read_csv(OUTPUT_DIR / "pipeline_v3_source_class_yield.csv").to_dict("records")
        write_report(
            True,
            query_rows,
            field_rows,
            yield_rows,
            parley_attempts=int(candidate_log["attempts"]),
            parley_successful=int(candidate_log["successful_extractions"]),
            parley_cost=float(candidate_log["total_cost"]),
        )
    elif "--live5" in sys.argv:
        queries, fields, yields = live5()
        candidate_log = {"attempts": 0, "successful_extractions": 0, "total_cost": 0.0}
        if "--parley-candidates" in sys.argv:
            candidate_log = parley_candidate_extractions()
        write_report(
            True,
            queries,
            fields,
            yields,
            parley_attempts=int(candidate_log["attempts"]),
            parley_successful=int(candidate_log["successful_extractions"]),
            parley_cost=float(candidate_log["total_cost"]),
        )
    else:
        prepare()
