from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_DIR
from .official_page_retrieval import OfficialPageFetcher
from .research_contract import load_event
from .source_discovery import (
    SourceCandidate,
    assess_event_match,
    assess_source_sufficiency,
    filter_admissible_sources,
    score_source,
    source_tier,
)


BASE_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
RETRY4_DISCOVERED = BASE_DIR / "retry_4_sources" / "discovered"
FULLPAGE_DIR = BASE_DIR / "fullpage_rescue_sources"
RETRY4_IDS = (
    "tm_ec95d18d266c5a44f071",
    "tm_890a6e2a4b41a1f8dd46",
    "tm_1ae175d1097fbcd0ddc0",
    "tm_211885c2a2cbd065928b",
)
MECHANISM_TERMS = {
    "transfer_fee": ("transfer fee", "fee", "valor", "montante", "verba", "custo", "€", "eur"),
    "loan_fee": ("loan fee", "empréstimo", "emprestimo", "cedido", "loan"),
    "option_to_buy": ("option to buy", "opção de compra", "opcao de compra"),
    "obligation_to_buy": ("obligation to buy", "mandatory purchase", "obrigação de compra", "compra obrigatória"),
    "add_ons": ("add-ons", "addons", "bónus", "bonus", "variáveis", "variaveis"),
    "sell_on": ("sell-on", "mais-valia", "percentagem de futura transferência"),
    "buy_back": ("buy-back", "recompra"),
    "release_clause": ("release clause", "cláusula de rescisão", "clausula de rescisao"),
    "contract_duration": ("contract until", "até 20", "ate 20", "contrato até", "valid until", "season"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _candidate(record: dict[str, Any]) -> SourceCandidate:
    return SourceCandidate(**record)


def _tier12_candidates(path: Path) -> list[SourceCandidate]:
    payload = _load_json(path)
    return [_candidate(record) for record in payload.get("sources", []) if record.get("source_tier") in {1, 2}]


def _old_admissible_count(event_id: str) -> int:
    path = BASE_DIR / "retry_4_sources" / "admissible" / f"{event_id}.json"
    return len(_load_json(path).get("sources", [])) if path.exists() else 0


def mechanism_signals(text: str) -> list[str]:
    lowered = text.lower()
    return [name for name, terms in MECHANISM_TERMS.items() if any(term in lowered for term in terms)]


def evaluate_event(event_id: str, discovered_dir: Path, fetcher: OfficialPageFetcher) -> dict[str, Any]:
    event = load_event(DEFAULT_OUTPUT_DIR / "structured_transfers.csv", event_id)
    candidates = _tier12_candidates(discovered_dir / f"{event_id}.json")
    old_exact_likely = sum(1 for candidate in candidates if candidate.event_match_status in {"exact", "likely"})
    old_admissible = _old_admissible_count(event_id)
    old_sufficient = bool(_load_json(discovered_dir / f"{event_id}.json").get("sufficient"))
    retrieval_counts = Counter()
    failures = Counter()
    rescued_sources: list[SourceCandidate] = []
    term_rich = 0
    mechanisms = Counter()

    for candidate in candidates:
        result = fetcher.apply_to_candidate(candidate)
        retrieval_counts[result.retrieval_status] += 1
        if result.retrieval_status != "success":
            failures[result.retrieval_error or result.retrieval_status] += 1
        candidate.event_match_status, candidate.event_match_score, candidate.event_match_reasons = assess_event_match(candidate, event)
        candidate.source_tier = source_tier(candidate)
        candidate.quality_score = score_source(candidate, event)
        rescued_sources.append(candidate)
        found = mechanism_signals(candidate.retrieved_text or "")
        if len(found) >= 2:
            term_rich += 1
        mechanisms.update(found)

    admissible = filter_admissible_sources(rescued_sources)
    sufficient, reasons = assess_source_sufficiency(admissible)
    new_exact_likely = sum(1 for candidate in rescued_sources if candidate.event_match_status in {"exact", "likely"})
    _write_rescued_sources(event_id, event, rescued_sources, admissible, sufficient, reasons)
    return {
        "event_id": event_id,
        "player_name": event.get("player_name"),
        "old_tier12_source_count": len(candidates),
        "successfully_retrieved_page_count": retrieval_counts["success"],
        "old_exact_likely_tier12_matches": old_exact_likely,
        "new_exact_likely_tier12_matches": new_exact_likely,
        "old_admissible_count": old_admissible,
        "new_admissible_count": len(admissible),
        "old_sufficient": old_sufficient,
        "new_sufficient": sufficient,
        "retrieval_failures": json.dumps(dict(failures), ensure_ascii=False, sort_keys=True),
        "contract_term_rich_pages": term_rich,
        "mechanism_signals": json.dumps(dict(mechanisms), ensure_ascii=False, sort_keys=True),
    }


def _write_rescued_sources(event_id: str, event: dict[str, Any], sources: list[SourceCandidate], admissible: list[SourceCandidate], sufficient: bool, reasons: list[str]) -> None:
    common = {
        "event_id": event_id,
        "event": {
            "player_name": event.get("player_name"),
            "from_club_name": event.get("from_club_name"),
            "to_club_name": event.get("to_club_name"),
            "transfer_date": event.get("transfer_date"),
        },
        "sufficient": sufficient,
        "sufficiency_reasons": reasons,
    }
    for name, payload_sources in {"discovered": sources, "admissible": admissible}.items():
        path = FULLPAGE_DIR / name / f"{event_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**common, "sources": [source.to_dict() for source in payload_sources]}, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _insufficient_event_ids() -> list[str]:
    with (DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis.csv").open(newline="") as handle:
        return [row["event_id"] for row in csv.DictReader(handle) if row["classification"] == "insufficient_evidence"]


def _discovered_path_for_event(event_id: str) -> Path | None:
    retry_path = RETRY4_DISCOVERED / f"{event_id}.json"
    if retry_path.exists():
        return retry_path
    original_path = DEFAULT_OUTPUT_DIR / "discovered_sources" / f"{event_id}.json"
    return original_path if original_path.exists() else None


def evaluate_events(event_ids: list[str], output_csv: Path) -> list[dict[str, Any]]:
    fetcher = OfficialPageFetcher()
    rows = []
    for event_id in event_ids:
        discovered_path = _discovered_path_for_event(event_id)
        if discovered_path is None:
            continue
        rows.append(evaluate_event(event_id, discovered_path.parent, fetcher))
    if rows:
        _write_csv(output_csv, rows)
    return rows


def write_report(path: Path, rows: list[dict[str, Any]], title: str, conditional_ran: bool | None = None) -> None:
    attempted_urls = sum(int(row["old_tier12_source_count"]) for row in rows)
    retrieved = sum(int(row["successfully_retrieved_page_count"]) for row in rows)
    identity_rescued = sum(int(row["new_exact_likely_tier12_matches"]) > int(row["old_exact_likely_tier12_matches"]) for row in rows)
    admissible_rescued = sum(int(row["new_admissible_count"]) > int(row["old_admissible_count"]) for row in rows)
    sufficient = sum(_truthy(row["new_sufficient"]) and not _truthy(row["old_sufficient"]) for row in rows)
    term_rich = sum(int(row["contract_term_rich_pages"]) for row in rows)
    mechanisms = Counter()
    failures = Counter()
    for row in rows:
        mechanisms.update(json.loads(row["mechanism_signals"]))
        failures.update(json.loads(row["retrieval_failures"]))
    lines = [
        f"# {title}",
        "",
        f"- Known Tier 1/2 URLs attempted: {attempted_urls}",
        f"- Successful retrievals: {retrieved}",
        f"- Retrieval success rate: {retrieved}/{attempted_urls} = {(retrieved / attempted_urls if attempted_urls else 0):.2%}",
        f"- Event-match rescue rate: {identity_rescued}/{len(rows)} = {(identity_rescued / len(rows) if rows else 0):.2%}",
        f"- Admissible-evidence rescue rate: {admissible_rescued}/{len(rows)} = {(admissible_rescued / len(rows) if rows else 0):.2%}",
        f"- Evidence-sufficiency conversion rate: {sufficient}/{len(rows)} = {(sufficient / len(rows) if rows else 0):.2%}",
        f"- Contract-term-rich retrieved pages: {term_rich}",
        f"- Retrieval failures by reason: {json.dumps(dict(failures), ensure_ascii=False, sort_keys=True)}",
        f"- Mechanism signals: {json.dumps(dict(mechanisms), ensure_ascii=False, sort_keys=True)}",
        "- Recommended architecture: A. Tavily discovery → direct full-page retrieval for promising Tier 1/2 URLs → event matching → admissibility → sufficiency → Parley",
        "- Bottleneck diagnosed: Tavily snippets, not official-page availability, were the main retry-4 bottleneck.",
    ]
    if conditional_ran is not None:
        lines.append(f"- Conditional broader evaluation ran: {conditional_ran}")
    lines.extend(["", "## Events", ""])
    for row in rows:
        lines.append(
            f"- `{row['event_id']}` {row['player_name']}: exact/likely {row['old_exact_likely_tier12_matches']} -> "
            f"{row['new_exact_likely_tier12_matches']}; admissible {row['old_admissible_count']} -> "
            f"{row['new_admissible_count']}; sufficient {row['old_sufficient']} -> {row['new_sufficient']}; "
            f"term-rich pages {row['contract_term_rich_pages']}"
        )
    path.write_text("\n".join(lines) + "\n")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def main() -> None:
    retry_rows = evaluate_events(list(RETRY4_IDS), BASE_DIR / "retry_4_fullpage_comparison.csv")
    identity_rescued = sum(int(row["new_exact_likely_tier12_matches"]) > int(row["old_exact_likely_tier12_matches"]) for row in retry_rows)
    admissible_rescued = sum(int(row["new_admissible_count"]) > int(row["old_admissible_count"]) for row in retry_rows)
    should_extend = admissible_rescued >= 2 or identity_rescued >= 2
    write_report(BASE_DIR / "retry_4_fullpage_report.md", retry_rows, "Retry 4 Full-Page Rescue Report", conditional_ran=should_extend)
    if should_extend:
        remaining = [event_id for event_id in _insufficient_event_ids() if event_id not in RETRY4_IDS]
        all_rows = evaluate_events(remaining, BASE_DIR / "fullpage_rescue_20_comparison.csv")
        write_report(BASE_DIR / "fullpage_rescue_20_report.md", all_rows, "Full-Page Rescue 20 Report")


if __name__ == "__main__":
    main()
