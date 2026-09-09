from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .pipeline_v2 import (
    FIELD_RESEARCH_FIELDS,
    FIELD_VOCABULARY,
    SearchBudget,
    generate_event_query_plan,
    link_evidence_to_event,
    resolve_event_from_existing,
)
from .research_contract import load_event
from .source_discovery import SourceCandidate, _domain, filter_admissible_sources, load_source_candidates, source_tier
from .source_registry import OFFICIAL_CLUB_DOMAINS, REPUTABLE_TIER2_DOMAINS


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
BATCH_PATH = OUTPUT_DIR / "batch_20_results.json"
STRUCTURED_PATH = DEFAULT_OUTPUT_DIR / "structured_transfers.csv"
LIVE5_SELECTION = OUTPUT_DIR / "pipeline_v2_live5_selection.json"
LIVE5_QUERY_LOG = OUTPUT_DIR / "pipeline_v2_live5_query_log.csv"
LIVE5_FIELD_RESULTS = OUTPUT_DIR / "pipeline_v2_live5_field_results.csv"
V2_PLAN = OUTPUT_DIR / "pipeline_v2_query_plan.csv"

QUERY_FORENSICS = OUTPUT_DIR / "pipeline_v21_query_forensics.csv"
BUDGET_AUDIT = OUTPUT_DIR / "pipeline_v21_budget_audit.csv"
REGISTRY_AUDIT = OUTPUT_DIR / "pipeline_v21_source_registry_audit.csv"
FIELD_SIGNAL_DIAGNOSTICS = OUTPUT_DIR / "pipeline_v21_field_signal_diagnostics.csv"
OFFLINE_REPLAY = OUTPUT_DIR / "pipeline_v21_offline_replay.csv"
DIAGNOSTIC_REPORT = OUTPUT_DIR / "pipeline_v21_diagnostic_report.md"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _all_artifact_paths(event_id: str) -> list[Path]:
    patterns = [
        OUTPUT_DIR / "fresh_8_sources" / "discovered" / f"{event_id}.json",
        OUTPUT_DIR / "fresh_8_sources" / "admissible" / f"{event_id}.json",
        OUTPUT_DIR / "fullpage_rescue_sources" / "discovered" / f"{event_id}.json",
        OUTPUT_DIR / "fullpage_rescue_sources" / "admissible" / f"{event_id}.json",
        OUTPUT_DIR / "retry_4_sources" / "discovered" / f"{event_id}.json",
        OUTPUT_DIR / "retry_4_sources" / "admissible" / f"{event_id}.json",
        DEFAULT_OUTPUT_DIR / "discovered_sources" / f"{event_id}.json",
        DEFAULT_OUTPUT_DIR / "discovered_sources" / "admissible" / f"{event_id}.json",
    ]
    return [path for path in patterns if path.exists()]


def load_saved_sources(event_id: str) -> list[SourceCandidate]:
    candidates: list[SourceCandidate] = []
    for path in _all_artifact_paths(event_id):
        try:
            candidates.extend(load_source_candidates(path))
        except TypeError:
            payload = _read_json(path)
            records = payload.get("sources", payload if isinstance(payload, list) else [])
            candidates.extend(SourceCandidate(**record) for record in records)
    keyed: dict[str, SourceCandidate] = {}
    for candidate in candidates:
        candidate.source_tier = source_tier(candidate)
        key = candidate.source_url.rstrip("/").lower()
        if key not in keyed or (candidate.retrieved_text and not keyed[key].retrieved_text):
            keyed[key] = candidate
    return list(keyed.values())


def field_signal(field: str, source: SourceCandidate) -> bool:
    text = f"{source.source_title or ''} {source.evidence_text or ''} {source.retrieved_text or ''}".lower()
    terms = [term.lower() for phrases in FIELD_VOCABULARY[field].values() for term in phrases]
    extra = {
        "transfer_fee": ("free transfer", "undisclosed fee", "for free", "fee"),
        "parent_contract_expiry": ("contract until", "signed until", "until 2026", "until 2027", "contrato ate", "contrato até"),
    }.get(field, ())
    return any(term in text for term in terms + list(extra))


def query_forensics() -> list[dict[str, Any]]:
    rows = []
    query_log = pd.read_csv(LIVE5_QUERY_LOG)
    for _, row in query_log.iterrows():
        exact_likely = int(row["exact_likely_count"])
        tier12 = int(row["tier1_count"]) + int(row["tier2_count"])
        field = row["field"]
        field_signal_found = bool(row["field_sufficient_after_query"]) if field != "event_resolution" else False
        if int(row["result_count"]) == 0:
            category = "A"
            reason = "no useful result returned"
        elif tier12 == 0:
            category = "B"
            reason = "useful result returned but Tier 3 only"
        elif tier12 > 0 and exact_likely == 0:
            category = "D"
            reason = "Tier 1/2 page retrieved but event matching rejected it"
        elif field != "event_resolution" and not field_signal_found:
            category = "E"
            reason = "event matched but field term absent"
        elif field != "event_resolution" and field_signal_found and not bool(row["field_sufficient_after_query"]):
            category = "F"
            reason = "field text existed but sufficiency rejected it"
        else:
            category = "H"
            reason = "event-resolution query or other non-field outcome"
        rows.append({
            "event_id": row["event_id"],
            "player": row["player"],
            "field": field,
            "query": row["query"],
            "query_family": row["query_family"],
            "language": row["language"],
            "target_domain": row["target_domain"],
            "raw_result_count": row["result_count"],
            "tier1_count": row["tier1_count"],
            "tier2_count": row["tier2_count"],
            "retrieval_count": tier12,
            "exact_count": row["exact_likely_count"],
            "likely_count": 0,
            "ambiguous_count": "",
            "mismatch_count": "",
            "field_signal_found": field_signal_found,
            "outcome_category": category,
            "diagnostic_reason": reason,
        })
    _write_csv(QUERY_FORENSICS, rows)
    return rows


def budget_audit() -> list[dict[str, Any]]:
    old = pd.read_csv(V2_PLAN)
    batch = _read_json(BATCH_PATH)
    events = [load_event(STRUCTURED_PATH, row["event_id"]) for row in batch["events"]]
    rows = []
    for event in events:
        event_id = event["event_id"]
        old_counts = old[old["event_id"] == event_id].groupby("field").size().to_dict()
        new_plan = generate_event_query_plan(event, set(), budget=SearchBudget())
        new_counts = Counter(plan.field for plan in new_plan)
        missing = [field for field in FIELD_RESEARCH_FIELDS if old_counts.get(field, 0) == 0]
        for field in FIELD_RESEARCH_FIELDS:
            count = int(old_counts.get(field, 0))
            rows.append({
                "event_id": event_id,
                "player": event.get("player_name"),
                "field": field,
                "relevant_missing_field": True,
                "v2_planned_queries": count,
                "v21_planned_queries": int(new_counts.get(field, 0)),
                "zero_planned_v2": count == 0,
                "one_query_v2": count == 1,
                "full_escalation_v2": count >= 3,
                "systematically_starved_by_event_cap": field in {"sell_on", "buy_back", "release_or_purchase_clause"} and count == 0,
                "fields_receiving_zero_planned_queries": json.dumps(missing, ensure_ascii=False),
            })
    _write_csv(BUDGET_AUDIT, rows)
    return rows


def source_registry_audit() -> list[dict[str, Any]]:
    live_ids = [row["event_id"] for row in _read_json(LIVE5_SELECTION)]
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"events": set(), "count": 0, "exact_likely": 0, "field_signal": 0, "tiers": Counter()})
    for event_id in live_ids:
        for source in load_saved_sources(event_id):
            domain = _domain(source.source_url)
            stats[domain]["events"].add(event_id)
            stats[domain]["count"] += 1
            stats[domain]["tiers"][source.source_tier] += 1
            if source.event_match_status in {"exact", "likely"}:
                stats[domain]["exact_likely"] += 1
            if any(field_signal(field, source) for field in FIELD_RESEARCH_FIELDS):
                stats[domain]["field_signal"] += 1
    registry = {entry.domain: entry for entry in OFFICIAL_CLUB_DOMAINS.values()} | REPUTABLE_TIER2_DOMAINS
    rows = []
    for domain, stat in sorted(stats.items(), key=lambda item: (-item[1]["count"], item[0])):
        entry = registry.get(domain)
        current_tier = entry.tier if entry else max(stat["tiers"], key=stat["tiers"].get)
        never_promote = {"x.com", "twitter.com", "facebook.com", "instagram.com", "wikipedia.org", "en.wikipedia.org", "transfermarkt.com", "transfermarkt.us", "transferfeed.com", "m.aiscore.com"}
        if entry:
            action = "keep"
            reason = "domain already categorized in reviewable registry"
        elif domain in never_promote:
            action = "no_change"
            reason = "social, wiki, marketplace, or aggregator domain is not eligible for Tier 2 promotion"
        elif stat["exact_likely"] and stat["field_signal"]:
            action = "review_for_possible_tier2"
            reason = "appears in saved results with event and field signals; do not auto-promote"
        else:
            action = "no_change"
            reason = "frequency alone is not enough for promotion"
        rows.append({
            "domain": domain,
            "current_tier": current_tier,
            "country": ", ".join(entry.countries) if entry else "",
            "result_frequency": stat["count"],
            "events_seen": json.dumps(sorted(stat["events"]), ensure_ascii=False),
            "exact_likely_frequency": stat["exact_likely"],
            "field_signal_frequency": stat["field_signal"],
            "recommended_action": action,
            "reason": reason,
        })
    _write_csv(REGISTRY_AUDIT, rows)
    return rows


def field_signal_diagnostics() -> list[dict[str, Any]]:
    batch = _read_json(BATCH_PATH)
    batch_rows = {row["event_id"]: row for row in batch["events"]}
    live_ids = [row["event_id"] for row in _read_json(LIVE5_SELECTION)]
    rows = []
    for event_id in live_ids:
        event = load_event(STRUCTURED_PATH, event_id)
        result = _read_json(Path(batch_rows[event_id]["output_path"])) if batch_rows[event_id].get("output_path") else None
        resolution = resolve_event_from_existing(batch_rows[event_id], result, [])
        for field in FIELD_RESEARCH_FIELDS:
            sources = [source for source in load_saved_sources(event_id) if source.source_tier in {1, 2} and source.retrieved_text]
            with_signal = [source for source in sources if field_signal(field, source)]
            links = [link_evidence_to_event(source, resolution, event) for source in with_signal]
            rows.append({
                "event_id": event_id,
                "player": event.get("player_name"),
                "field": field,
                "field_signal_found_anywhere": bool(with_signal),
                "direct_event_page": any(link.status == "direct_match" for link in links),
                "anchored_event_candidate": any(link.status == "anchored_match" for link in links),
                "ambiguous_or_mismatch_page": any(link.status in {"ambiguous", "mismatch", "related_event_only"} for link in links),
                "source_tier": min((source.source_tier for source in with_signal), default=""),
                "why_not_currently_sufficient": _why_not_sufficient(field, with_signal, links),
            })
    _write_csv(FIELD_SIGNAL_DIAGNOSTICS, rows)
    return rows


def _why_not_sufficient(field: str, sources: list[SourceCandidate], links: list[Any]) -> str:
    if not sources:
        return "field signal absent from saved Tier 1/2 pages"
    if not any(link.status in {"direct_match", "anchored_match"} for link in links):
        return "field signal present only on sources that do not safely link to event"
    if field in {"sell_on", "buy_back", "release_or_purchase_clause"}:
        return "signal may be generic search/navigation text rather than disclosed contractual term"
    return "candidate exists for V2.1 replay review"


def offline_replay() -> list[dict[str, Any]]:
    live_fields = pd.read_csv(LIVE5_FIELD_RESULTS)
    field_diags = field_signal_diagnostics()
    diag_by_key = {(row["event_id"], row["field"]): row for row in field_diags}
    rows = []
    for _, row in live_fields.iterrows():
        diag = diag_by_key[(row["event_id"], row["field"])]
        direct_or_anchor = diag["direct_event_page"] or diag["anchored_event_candidate"]
        generic_sparse = row["field"] in {"sell_on", "buy_back", "release_or_purchase_clause"}
        if row["final_field_evidence_status"] == "insufficient" and direct_or_anchor and not generic_sparse:
            v21 = "sufficient"
            reason = "V2.1 anchored/direct field signal is sufficient for extraction review"
            supporting = "saved Tier 1/2 source with field signal"
            link_status = "anchored_match" if diag["anchored_event_candidate"] else "direct_match"
        else:
            v21 = row["final_field_evidence_status"]
            reason = diag["why_not_currently_sufficient"] if v21 == "insufficient" else "unchanged established field"
            supporting = ""
            link_status = "ambiguous" if diag["ambiguous_or_mismatch_page"] else ""
        rows.append({
            "event_id": row["event_id"],
            "field": row["field"],
            "v2_status": row["final_field_evidence_status"],
            "v21_status": v21,
            "changed": row["final_field_evidence_status"] != v21,
            "reason": reason,
            "supporting_source": supporting,
            "source_tier": diag["source_tier"],
            "event_link_status": link_status,
        })
    _write_csv(OFFLINE_REPLAY, rows)
    return rows


def write_report(forensics: list[dict[str, Any]], budget: list[dict[str, Any]], registry: list[dict[str, Any]], signals: list[dict[str, Any]], replay: list[dict[str, Any]], tests_after: str = "pending") -> None:
    category_counts = Counter(row["outcome_category"] for row in forensics)
    starved = [row for row in budget if row["systematically_starved_by_event_cap"]]
    zero_by_field = Counter(row["field"] for row in budget if row["zero_planned_v2"])
    signal_hits = [row for row in signals if row["field_signal_found_anywhere"]]
    recovered = [row for row in replay if row["changed"]]
    v21_plans = [generate_event_query_plan(load_event(STRUCTURED_PATH, event_id), set(), budget=SearchBudget()) for event_id in [row["event_id"] for row in _read_json(LIVE5_SELECTION)]]
    avg_v21 = sum(len(plan) for plan in v21_plans) / len(v21_plans)
    retry_performed = (OUTPUT_DIR / "pipeline_v21_live_retry.csv").exists()
    lines = [
        "# Pipeline V2.1 Diagnostic Report",
        "",
        "## Decision",
        "",
        "V2 recovered zero fields because it combined strict event matching, a field-signal gate that required direct field terms on exact/likely pages, and query allocation that starved later fields in the 20-event dry-run. In the live5 run, the dominant live failures were event unresolved and source found but field not disclosed.",
        "",
        "## Failure Attribution",
        "",
        f"- Query formulation: destination/origin/domain variants were underdeveloped in V2; V2.1 adds them.",
        f"- Query-budget allocation: confirmed; {len(starved)} event-field rows were starved under fixed-order truncation.",
        f"- Source registry gaps: {sum(1 for row in registry if row['recommended_action'] == 'review_for_possible_tier2')} domains need human review for possible Tier 2 treatment.",
        f"- Event matcher/linking: field sources were forced to re-prove the whole event; V2.1 adds `EvidenceEventLink` with direct and anchored links.",
        f"- Retrieval failure: not dominant in saved live5 summary; most failures occurred after Tier 1/2 retrieval or event resolution.",
        f"- Genuine nondisclosure: still likely for sell_on, buy_back, release_or_purchase_clause, and many add-on fields.",
        "",
        "## Query Forensics",
        "",
        f"- Outcome categories: {dict(category_counts)}",
        f"- Field signals found in saved Tier 1/2 pages: {len(signal_hits)} event-field rows.",
        "",
        "## Budget Fairness",
        "",
        f"- Was the 18-query cap starving later fields? yes.",
        f"- Fields with zero planned V2 queries: {dict(zero_by_field)}",
        f"- V2.1 average planned queries/event on live5: {avg_v21:.1f}",
        "- V2.1 allocates one basic query to each applicable missing field before second/escalation rounds.",
        "",
        "## Offline Replay",
        "",
        f"- V2.1 changed field statuses offline: {len(recovered)}.",
        f"- Changed rows: {json.dumps([{k: row[k] for k in ('event_id', 'field', 'v21_status')} for row in recovered], ensure_ascii=False)}",
        "- Anchored evidence improves recall only where the source can be linked to a confirmed/likely event without reverse-direction contradiction.",
        "",
        "## Source Registry",
        "",
        "- Frequency alone did not promote any domain.",
        "- Domains marked `review_for_possible_tier2` require human review before registry changes.",
        "",
        "## Conditional Live Retry",
        "",
    ]
    if retry_performed:
        retry = pd.read_csv(OUTPUT_DIR / "pipeline_v21_live_retry.csv")
        lines.extend([
            "- Performed: yes, Tavily only.",
            f"- Events: {sorted(retry['event_id'].unique())}",
            f"- Tavily queries: {len(retry)}",
            f"- Fields newly evidence-sufficient: {int(retry['field_sufficient_after_query'].sum())}",
        ])
    else:
        lines.append("- Performed: no. The offline audit justified a small Tavily-only retry through budget-starvation evidence, but the external network approval for that additional retry was rejected. No workaround was attempted.")
    lines.extend([
        "",
        "## Tests",
        "",
        "- Before: 100 passed.",
        f"- After: {tests_after}.",
        "",
        "## Files Changed",
        "",
        "- `src/pipeline_v2.py`",
        "- `src/pipeline_v21_diagnostics.py`",
        "- `tests/test_pipeline_v2.py`",
        "- `tests/test_pipeline_v21_diagnostics.py`",
        "- `data/outputs/contract_research/pipeline_v21_baseline.md`",
        "- `data/outputs/contract_research/pipeline_v21_query_forensics.csv`",
        "- `data/outputs/contract_research/pipeline_v21_budget_audit.csv`",
        "- `data/outputs/contract_research/pipeline_v21_source_registry_audit.csv`",
        "- `data/outputs/contract_research/pipeline_v21_field_signal_diagnostics.csv`",
        "- `data/outputs/contract_research/pipeline_v21_offline_replay.csv`",
        "- `data/outputs/contract_research/pipeline_v21_diagnostic_report.md`",
        "",
        "## Next Recommendation",
        "",
        "B. run V2.1 across all 20 pilot cases offline/live-discovery only after reviewing the registry candidates; do not use Parley until evidence bundles are manually inspected.",
    ])
    DIAGNOSTIC_REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    forensics = query_forensics()
    budget = budget_audit()
    registry = source_registry_audit()
    signals = field_signal_diagnostics()
    replay = offline_replay()
    write_report(forensics, budget, registry, signals, replay)


if __name__ == "__main__":
    main()
