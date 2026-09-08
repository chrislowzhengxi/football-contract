from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_DIR
from .retry_4_discovery import retry_improvement_metrics


BASE_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
DISCOVERED_DIR = BASE_DIR / "retry_4_sources" / "discovered"
ANALYSIS_CSV = BASE_DIR / "retry_4_official_source_failure_analysis.csv"
REPORT_MD = BASE_DIR / "retry_4_official_source_failure_report.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _domain(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0].removeprefix("www.").lower()


def classify_rejection(source: dict[str, Any]) -> tuple[str, list[str], str]:
    reasons = list(source.get("event_match_reasons") or [])
    title = source.get("source_title") or ""
    text = source.get("evidence_text") or ""
    haystack = f"{title} {text}".lower()
    status = source.get("event_match_status")
    secondary: list[str] = []
    notes = ""

    if not source.get("accessible"):
        return "inaccessible", secondary, "Saved candidate failed the accessibility check."
    if status == "mismatch":
        if "reverse_direction" in reasons:
            return "wrong_transfer_event", ["reverse_direction"], "Snippet supports the opposite transfer direction."
        if "loan_return_without_target_direction" in reasons:
            return "wrong_transfer_event", ["loan_return"], "Snippet discusses a loan return rather than the original transfer."
        return "wrong_transfer_event", reasons, "Event matcher classified the candidate as a mismatch."
    if "loan_followed_by_permanent_transfer" in reasons:
        return "loan_vs_permanent_mismatch", reasons, "Snippet mixes a loan with a later permanent-transfer event."
    if "player_match" not in reasons:
        return "player_not_matched", reasons, "Saved snippet/title does not contain the player name."
    if "departing_club_match" not in reasons:
        secondary.append("departing_club_not_matched")
    if "receiving_club_match" not in reasons:
        secondary.append("receiving_club_not_matched")
    if source.get("publication_date") and str(source.get("publication_date"))[:4] not in {"2025", ""}:
        secondary.append("wrong_year_or_date")
    correct_official_announcement = (
        source.get("source_tier") in {1, 2}
        and source.get("accessible")
        and source.get("target_domain")
        and _domain(source.get("source_url") or "") == source.get("target_domain")
        and "player_match" in reasons
        and any(term in haystack for term in (
            "transfer", "signs", "signed", "joins", "joined", "leaves for",
            "cedido ao", "cedido a", "emprestado ao", "emprestado a",
            "e dragao", "é dragão", "transferencia", "transferência",
            "vertrekt naar", "se marcha al", "rejoint",
        ))
    )
    if correct_official_announcement and status == "ambiguous":
        return "admissibility_logic_issue", secondary, "Correct-looking official-domain announcement failed the saved event matcher."
    if "direction_match" not in reasons:
        if any(term in haystack for term in ("transfer", "transferred", "signs", "signed", "joins", "joined", "loan", "empréstimo", "transferência")):
            return "direction_not_established", secondary, "Snippet has transfer language but does not establish the saved event direction."
        return "snippet_insufficient_for_event_identity", secondary, "Snippet is too thin to establish event identity."
    if status in {"ambiguous", None}:
        return "snippet_insufficient_for_event_identity", secondary, "Candidate has partial identity but not enough saved context for admissibility."
    return "admissibility_logic_issue", secondary, "Tier 1/2 source appears exact/likely and accessible but was not admissible."


def build_failure_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(DISCOVERED_DIR.glob("tm_*.json")):
        payload = _load_json(path)
        event = payload.get("event", {})
        for source in payload.get("sources", []):
            if source.get("source_tier") not in {1, 2}:
                continue
            primary, secondary, notes = classify_rejection(source)
            rows.append({
                "event_id": payload.get("event_id"),
                "player": event.get("player_name"),
                "source_url": source.get("source_url"),
                "source_domain": _domain(source.get("source_url") or ""),
                "title": source.get("source_title"),
                "query_family": source.get("query_family"),
                "query": source.get("search_query"),
                "source_tier": source.get("source_tier"),
                "accessible": source.get("accessible"),
                "event_match_status": source.get("event_match_status"),
                "event_match_score": source.get("event_match_score"),
                "event_match_reasons": json.dumps(source.get("event_match_reasons") or [], ensure_ascii=False),
                "quality_score": source.get("quality_score"),
                "primary_rejection_reason": primary,
                "secondary_rejection_reasons": json.dumps(secondary, ensure_ascii=False),
                "would_be_admissible_if_event_identity_resolved": bool(source.get("accessible") and source.get("source_tier") in {1, 2}),
                "notes": notes,
            })
    return rows


def write_analysis_csv(rows: list[dict[str, Any]]) -> None:
    ANALYSIS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ANALYSIS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _comparison_rows() -> list[dict[str, Any]]:
    with (BASE_DIR / "retry_4_discovery_comparison.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _search_log_rows() -> list[dict[str, Any]]:
    return _load_json(BASE_DIR / "retry_4_search_log.json")


def write_report(rows: list[dict[str, Any]]) -> None:
    by_reason = Counter(row["primary_rejection_reason"] for row in rows)
    by_event: dict[str, Counter[str]] = defaultdict(Counter)
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_event[f"{row['player']} ({row['event_id']})"][row["primary_rejection_reason"]] += 1
        by_family[str(row["query_family"])][row["primary_rejection_reason"]] += 1

    comparison = _comparison_rows()
    metrics = retry_improvement_metrics(comparison)
    total_tier1 = sum(1 for row in rows if int(row["source_tier"]) == 1)
    total_tier2 = sum(1 for row in rows if int(row["source_tier"]) == 2)
    correct_but_failed = sum(1 for row in rows if row["primary_rejection_reason"] in {"direction_not_established", "snippet_insufficient_for_event_identity"})
    inaccessible = by_reason.get("inaccessible", 0)
    unrelated = by_reason.get("search_result_unrelated", 0) + by_reason.get("player_not_matched", 0)
    bug_like = by_reason.get("admissibility_logic_issue", 0)
    component_counts = {
        "SEARCH_RETRIEVAL_FAILURE": unrelated,
        "EVENT_MATCHING_FAILURE": bug_like,
        "ACCESS_CONTENT_FAILURE": by_reason.get("snippet_insufficient_for_event_identity", 0) + by_reason.get("direction_not_established", 0) + inaccessible,
        "TRUE_DISCLOSURE_FAILURE": 0,
    }

    lines = [
        "# Retry 4 Official Source Failure Report",
        "",
        "## Totals",
        "",
        f"- Total Tier 1 results: {total_tier1}",
        f"- Total Tier 2 results: {total_tier2}",
        f"- Genuinely unrelated/player-missing: {unrelated}",
        f"- Correct-looking official pages that failed matching/content identity: {correct_but_failed}",
        f"- Snippet/direction limitations: {by_reason.get('snippet_insufficient_for_event_identity', 0) + by_reason.get('direction_not_established', 0)}",
        f"- Inaccessible: {inaccessible}",
        f"- Apparent matching/admissibility bugs: {bug_like}",
        "",
        "## Rejection Reasons",
        "",
    ]
    for reason, count in by_reason.most_common():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## By Event", ""])
    for event, counts in sorted(by_event.items()):
        parts = ", ".join(f"{reason}: {count}" for reason, count in counts.most_common())
        lines.append(f"- {event}: {parts}")
    lines.extend(["", "## By Query Family", ""])
    for family, counts in sorted(by_family.items()):
        parts = ", ".join(f"{reason}: {count}" for reason, count in counts.most_common())
        lines.append(f"- {family}: {parts}")
    lines.extend([
        "",
        "## Diagnosis",
        "",
        "Primary diagnosis: E. A mixture, dominated by access/content failure in saved snippets rather than a broad admissibility bug.",
        "",
    ])
    for component, count in component_counts.items():
        lines.append(f"- {component}: {count}")
    lines.extend([
        "",
        "The saved official-domain results often came from the right club sites, but the saved title/snippet usually did not include enough player + both clubs + direction information to pass the strict event gate. The exact/likely matches found in this retry were Tier 3, so they were correctly excluded from admissible evidence.",
        "",
        "## Corrected Retry Metrics",
        "",
        f"- Event-identity improvement rate: {metrics['event_identity_improved']}/{metrics['attempted']} = {metrics['event_identity_improved'] / metrics['attempted']:.2%}",
        f"- Admissible-evidence improvement rate: {metrics['admissible_evidence_improved']}/{metrics['attempted']} = {metrics['admissible_evidence_improved'] / metrics['attempted']:.2%}",
        f"- Evidence-sufficiency conversion rate: {metrics['evidence_sufficiency_converted']}/{metrics['attempted']} = {metrics['evidence_sufficiency_converted'] / metrics['attempted']:.2%}",
        f"- Contract-field improvement rate: {metrics['contract_field_improved']}/{metrics['attempted']} = {metrics['contract_field_improved'] / metrics['attempted']:.2%}",
        "",
        "## Query Performance",
        "",
    ])
    log = _search_log_rows()
    for family in sorted({row["query_family"] for row in log}):
        family_rows = [row for row in log if row["query_family"] == family]
        lines.append(
            f"- {family}: {len(family_rows)} queries, "
            f"{sum(row['result_count'] for row in family_rows)} raw results, "
            f"{sum(row['useful_result_count'] for row in family_rows)} exact/likely results, "
            f"{sum(row['admissible_result_count'] for row in family_rows)} admissible results"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = build_failure_rows()
    if not rows:
        raise SystemExit("No Tier 1/2 retry sources found")
    write_analysis_csv(rows)
    write_report(rows)


if __name__ == "__main__":
    main()
