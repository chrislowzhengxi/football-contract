from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_DIR
from .pipeline_v2 import FIELD_RESEARCH_FIELDS
from .pipeline_v21_diagnostics import field_signal, load_saved_sources
from .research_contract import load_event
from .source_discovery import _domain, official_domain_for_club
from .source_registry import REPUTABLE_TIER2_DOMAINS


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "contract_research"
AUDIT_PATH = OUTPUT_DIR / "source_strategy_audit.csv"
REPORT_PATH = OUTPUT_DIR / "source_strategy_report.md"
REGISTRY_CANDIDATES = ("footballtransfers.com", "rotowire.com")

SOURCE_CLASSES = (
    "official_buying_club",
    "official_selling_club",
    "regulatory_financial_disclosure",
    "major_national_sports_media",
    "reputable_local_sports_media",
    "reputable_transfer_specialist_journalism",
    "later_retrospective_reporting",
)

SELECTED_EVENTS = (
    "tm_694c30a10815aea6d84d",  # Dodi Lukebakio, high-profile permanent, fee/add-ons.
    "tm_211885c2a2cbd065928b",  # Danny Namaso, loan with option.
    "tm_163d23e38fec40c7a994",  # Florentino, obligation case.
    "tm_890a6e2a4b41a1f8dd46",  # Joao Costa, lower-profile permanent.
    "tm_ec95d18d266c5a44f071",  # Luuk de Jong, high-profile free/permanent control.
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_class(source: Any, event: dict[str, Any]) -> str:
    domain = _domain(source.source_url)
    if domain == official_domain_for_club(str(event.get("to_club_name", ""))):
        return "official_buying_club"
    if domain == official_domain_for_club(str(event.get("from_club_name", ""))):
        return "official_selling_club"
    if source.source_type in {"regulatory", "governing_body"} or any(token in domain for token in ("cmvm", "stockexchange", "sec.gov")):
        return "regulatory_financial_disclosure"
    entry = REPUTABLE_TIER2_DOMAINS.get(domain)
    if entry and entry.source_type == "major_news":
        return "major_national_sports_media"
    if entry and entry.source_type == "football_reporting":
        return "reputable_local_sports_media"
    if domain in {"footballtransfers.com", "transferfeed.com", "rotowire.com"}:
        return "reputable_transfer_specialist_journalism"
    text = f"{source.source_title or ''} {source.evidence_text or ''} {source.retrieved_text or ''}".lower()
    event_year = str(event.get("transfer_date", ""))[:4]
    years = [year for year in ("2026", "2027", "2028", "2029", "2030") if year in text]
    if event_year and years and all(year > event_year for year in years):
        return "later_retrospective_reporting"
    return "reputable_transfer_specialist_journalism" if source.source_tier == 2 else "major_national_sports_media"


def _result_payload(event_id: str) -> dict[str, Any]:
    batch = _read_json(OUTPUT_DIR / "batch_20_results.json")
    batch_row = next(row for row in batch["events"] if row["event_id"] == event_id)
    if not batch_row.get("output_path"):
        return {}
    return _read_json(Path(batch_row["output_path"]))


def _field_disclosed(result: dict[str, Any], field: str, source_class: str, event: dict[str, Any]) -> tuple[bool, bool, bool, str]:
    current = result.get(field, {})
    status = current.get("status")
    if status not in {"disclosed_yes", "disclosed_no", "partially_disclosed", "undisclosed", "conflicting_sources"}:
        return False, False, False, ""
    supporting_ids = set(current.get("evidence_ids") or [])
    supporting_sources = [
        source
        for source in result.get("sources", [])
        if source.get("evidence_id") in supporting_ids
    ]
    class_sources = [source for source in supporting_sources if _source_class(_dict_source(source), event) == source_class]
    if not class_sources:
        return False, False, False, ""
    has_exact = any(current.get(key) is not None for key in ("amount", "price", "percentage", "date", "year"))
    partial = status in {"partially_disclosed", "undisclosed", "conflicting_sources"} and not has_exact
    return True, has_exact, partial, class_sources[0].get("source_url", "")


class _dict_source:
    def __init__(self, payload: dict[str, Any]):
        self.source_url = payload.get("source_url", "")
        self.source_type = payload.get("source_type", "")
        self.source_tier = payload.get("source_tier")
        self.source_title = payload.get("source_title")
        self.evidence_text = payload.get("evidence_text")
        self.retrieved_text = payload.get("retrieved_text")


def build_audit() -> list[dict[str, Any]]:
    rows = []
    for event_id in SELECTED_EVENTS:
        event = load_event(DEFAULT_OUTPUT_DIR / "structured_transfers.csv", event_id)
        result = _result_payload(event_id)
        sources = [source for source in load_saved_sources(event_id) if source.source_tier in {1, 2}]
        by_class: dict[str, list[Any]] = defaultdict(list)
        for source in sources:
            by_class[_source_class(source, event)].append(source)
        for field in FIELD_RESEARCH_FIELDS:
            for source_class in SOURCE_CLASSES:
                class_sources = by_class[source_class]
                signal_sources = [source for source in class_sources if field_signal(field, source)]
                disclosed, exact, partial, supporting_example = _field_disclosed(result, field, source_class, event)
                best = min((source.source_tier for source in signal_sources), default="")
                example = supporting_example or (signal_sources[0].source_url if signal_sources else "")
                rows.append({
                    "event_id": event_id,
                    "player": event.get("player_name"),
                    "field": field,
                    "source_class": source_class,
                    "credible_source_found": bool(class_sources),
                    "field_disclosed": disclosed,
                    "exact_value_disclosed": exact,
                    "partial_information": partial,
                    "source_count": len(signal_sources),
                    "best_source_tier": best,
                    "example_source": example,
                    "notes": _notes(field, source_class, class_sources, signal_sources, disclosed),
                })
    _write_csv(AUDIT_PATH, rows)
    return rows


def _notes(field: str, source_class: str, class_sources: list[Any], signal_sources: list[Any], disclosed: bool) -> str:
    if disclosed:
        return "existing result contains supported disclosure and this source class has matching field signal"
    if signal_sources:
        return "field signal present, but saved evidence does not establish a usable disclosed value"
    if class_sources:
        return "credible source class present, but no field-specific signal found"
    return "no saved source from this source class"


def write_report(rows: list[dict[str, Any]]) -> None:
    by_field_class = Counter((row["field"], row["source_class"]) for row in rows if row["field_disclosed"])
    exact_by_field = Counter(row["field"] for row in rows if row["exact_value_disclosed"])
    signal_by_field = Counter(row["field"] for row in rows if row["source_count"])
    class_disclosures = Counter(row["source_class"] for row in rows if row["field_disclosed"])
    lines = [
        "# Source Strategy / Public Observability Audit",
        "",
        "## Selected Events",
        "",
        "- Dodi Lukebakio: high-profile permanent transfer with fee/add-ons/obligation evidence.",
        "- Danny Namaso: loan/options case.",
        "- Florentino: option/obligation case.",
        "- Joao Costa: lower-profile permanent transfer.",
        "- Luuk de Jong: high-profile free/permanent control-style case.",
        "",
        "## Registry Candidates",
        "",
    ]
    for domain in REGISTRY_CANDIDATES:
        lines.append(f"- `{domain}`: retain current tier pending human review; do not auto-promote from frequency or snippets alone.")
    lines.extend([
        "",
        "## Disclosure Patterns",
        "",
        f"- Source classes with actual supported disclosures in the selected sample: {dict(class_disclosures)}",
        f"- Exact-value disclosures by field: {dict(exact_by_field)}",
        f"- Field signals by field, whether usable or not: {dict(signal_by_field)}",
        "",
        "## Questions",
        "",
        "- Fees: disclosed mainly by official club pages where clubs announce a free/permanent fee, and by reputable local/major reporting when a numeric fee is public.",
        "- Options/obligations: official buying-club pages often disclose existence, but not complete trigger/amount detail.",
        "- Add-ons: occasionally disclosed in high-profile Tier 2 reporting; rarely in official club copy.",
        "- Sell-on/buy-back: no robust disclosure pattern in the saved sample.",
        "- Contract duration: official club pages are the strongest source class, but precision is often year/season rather than exact date.",
        "- Official club pages are mainly event-confirmation and broad-structure sources, not reliable financial-detail sources.",
        "- Structurally sparse fields: sell_on, buy_back, release_or_purchase_clause, obligation_trigger.",
        "- Fields that might improve with broader discovery: transfer_fee, add_ons, purchase_option, purchase_obligation, parent_contract_expiry.",
        "- Future V3 should deliberately target regulatory disclosures for listed clubs, local sports media by country, and reputable transfer-specialist journalism after manual domain review.",
        "",
        "## Recommendation",
        "",
        "C. Broaden sources for selected fields and treat sparse clauses as opportunistic. Keep sell-on, buy-back, release clauses, and obligation triggers in the schema, but do not treat them as core coverage targets unless higher-signal source classes are found.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = build_audit()
    write_report(rows)


if __name__ == "__main__":
    main()
