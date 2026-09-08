from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .contract_schemas import FIELD_NAMES, STATUS_VALUES, ContractResearchResult


CLASSIFICATIONS = {"clean", "usable_with_review", "insufficient_evidence", "invalid"}
STATUS_BREAKDOWN = (
    "disclosed_yes",
    "disclosed_no",
    "partially_disclosed",
    "conflicting_sources",
    "undisclosed",
    "not_applicable",
    "not_found",
    "unresearched",
)
BINARY_FIELDS = {
    "purchase_option",
    "purchase_obligation",
    "sell_on",
    "buy_back",
    "release_or_purchase_clause",
}
NUMERIC_FIELDS = {"transfer_fee", "loan_fee", "add_ons"}
ANALYSIS_COLUMNS = [
    "event_id",
    "player_name",
    "player_age_at_transfer",
    "transfer_date",
    "season",
    "departing_club",
    "receiving_club",
    "deterministic_transfer_type",
    "tm_transfer_fee",
    "tm_market_value_at_signing",
    "researched_transfer_type",
    "researched_transfer_type_status",
    "researched_transfer_fee",
    "researched_transfer_fee_status",
    "researched_loan_fee",
    "researched_loan_fee_status",
    "researched_add_ons",
    "researched_add_ons_status",
    "researched_purchase_option",
    "researched_purchase_option_status",
    "researched_purchase_obligation",
    "researched_purchase_obligation_status",
    "researched_obligation_trigger",
    "researched_obligation_trigger_status",
    "researched_sell_on",
    "researched_sell_on_status",
    "researched_buy_back",
    "researched_buy_back_status",
    "researched_parent_contract_expiry",
    "researched_parent_contract_expiry_status",
    "classification",
    "review_required",
    "research_performed",
    "evidence_sufficient",
    "discovered_source_count",
    "admissible_source_count",
]
REVIEW_QUEUE_COLUMNS = [
    "event_id",
    "player_name",
    "departing_club",
    "receiving_club",
    "transfer_date",
    "researched_transfer_fee",
    "researched_transfer_fee_status",
    "researched_add_ons",
    "researched_purchase_option",
    "researched_purchase_obligation",
    "researched_sell_on",
    "researched_parent_contract_expiry",
    "review_reasons",
    "admissible_source_count",
    "source_urls",
    "deal_summary",
]


def _json_text(value: Any) -> str | None:
    if value in (None, [], {}):
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _field_value(field: dict[str, Any]) -> Any:
    for key in ("amount", "price", "percentage", "date", "value"):
        value = field.get(key)
        if value is not None:
            return value
    return None


def _flatten_contract_field(prefix: str, field: dict[str, Any]) -> dict[str, Any]:
    return {
        prefix: _field_value(field),
        f"{prefix}_status": field.get("status"),
        f"{prefix}_amount": field.get("amount"),
        f"{prefix}_currency": field.get("currency"),
        f"{prefix}_price": field.get("price"),
        f"{prefix}_percentage": field.get("percentage"),
        f"{prefix}_date": field.get("date"),
        f"{prefix}_description": field.get("description"),
        f"{prefix}_evidence_ids": _json_text(field.get("evidence_ids")),
        f"{prefix}_reported_values": _json_text(field.get("reported_values")),
    }


def _load_research_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return ContractResearchResult.from_dict(payload).to_dict()


def _source_urls(result: dict[str, Any]) -> str | None:
    urls = [source["source_url"] for source in result.get("sources", []) if source.get("source_url")]
    return _json_text(urls)


def _source_ids(result: dict[str, Any]) -> str | None:
    ids = [source["evidence_id"] for source in result.get("sources", []) if source.get("evidence_id")]
    return _json_text(ids)


def build_research_dataset(
    structured_transfers_path: Path,
    batch_results_path: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    structured = pd.read_csv(structured_transfers_path, dtype={"event_id": str})
    batch = json.loads(batch_results_path.read_text())
    batch_events = pd.DataFrame(batch["events"])
    if batch_events["event_id"].duplicated().any():
        raise ValueError("batch results contain duplicate event_id values")

    deterministic_columns = [
        "event_id",
        "player_name",
        "age_at_transfer",
        "transfer_date",
        "transfer_season",
        "from_club_name",
        "to_club_name",
        "transfer_type",
        "transfer_fee",
        "market_value_nearest_transfer",
    ]
    missing = set(deterministic_columns) - set(structured.columns)
    if missing:
        raise ValueError(f"structured transfers missing columns: {sorted(missing)}")
    base = batch_events[["event_id"]].merge(structured[deterministic_columns], on="event_id", how="left")
    if base["player_name"].isna().any():
        missing_ids = base.loc[base["player_name"].isna(), "event_id"].tolist()
        raise ValueError(f"batch event IDs missing from structured transfers: {missing_ids}")

    rows: list[dict[str, Any]] = []
    for _, base_row in base.iterrows():
        event_id = str(base_row["event_id"])
        batch_row = batch_events.loc[batch_events["event_id"] == event_id].iloc[0].to_dict()
        output_path_value = batch_row.get("output_path")
        result_path = Path(output_path_value) if isinstance(output_path_value, str) and output_path_value else None
        research_performed = bool(batch_row.get("parley_called"))
        result = _load_research_result(result_path) if result_path else None
        if research_performed and result is None:
            raise ValueError(f"batch says research was performed but result is missing: {event_id}")

        row: dict[str, Any] = {
            "event_id": event_id,
            "player_name": base_row["player_name"],
            "player_age_at_transfer": base_row["age_at_transfer"],
            "transfer_date": base_row["transfer_date"],
            "season": base_row["transfer_season"],
            "departing_club": base_row["from_club_name"],
            "receiving_club": base_row["to_club_name"],
            "deterministic_transfer_type": base_row["transfer_type"],
            "tm_transfer_fee": base_row["transfer_fee"],
            "tm_market_value_at_signing": base_row["market_value_nearest_transfer"],
            "classification": batch_row.get("classification"),
            "review_required": result.get("review_required") if result else None,
            "review_reasons": _json_text(result.get("review_reasons") if result else batch_row.get("review_reasons")),
            "discovered_source_count": batch_row.get("source_count"),
            "admissible_source_count": batch_row.get("admissible_source_count"),
            "research_performed": research_performed,
            "evidence_sufficient": bool(batch_row.get("sufficient")),
            "sufficiency_reasons": _json_text(batch_row.get("sufficiency_reasons")),
            "provider": result.get("provider_metadata", {}).get("provider") if result else None,
            "model": result.get("provider_metadata", {}).get("model") if result else None,
            "parley_cost": batch_row.get("parley_cost"),
            "source_ids": _source_ids(result) if result else None,
            "source_urls": _source_urls(result) if result else None,
            "deal_summary": result.get("deal_summary") if result else None,
        }
        for field_name in FIELD_NAMES:
            field = result.get(field_name, {}) if result else {}
            row.update(_flatten_contract_field(f"researched_{field_name}", field))
        rows.append(row)

    dataset = pd.DataFrame(rows)
    validate_dataset(dataset, batch)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
    return dataset


def build_analysis_table(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    missing = set(ANALYSIS_COLUMNS) - set(raw_dataset.columns)
    if missing:
        raise ValueError(f"raw dataset missing analysis columns: {sorted(missing)}")
    analysis = raw_dataset[ANALYSIS_COLUMNS].copy()
    validate_analysis_table(analysis)
    return analysis


def build_audit_table(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    return raw_dataset.copy()


def build_review_queue(analysis: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    queue = analysis[analysis["classification"] == "usable_with_review"].merge(
        audit[["event_id", "review_reasons", "source_urls", "deal_summary"]],
        on="event_id",
        how="left",
    )
    return queue[REVIEW_QUEUE_COLUMNS].copy()


def validate_dataset(dataset: pd.DataFrame, batch: dict[str, Any]) -> None:
    if len(dataset) != int(batch["attempted_events"]):
        raise ValueError("dataset row count does not match attempted batch events")
    if dataset["event_id"].duplicated().any():
        raise ValueError("dataset event_id values are not unique")
    invalid_classes = set(dataset["classification"].dropna()) - CLASSIFICATIONS
    if invalid_classes:
        raise ValueError(f"invalid classifications: {sorted(invalid_classes)}")
    status_columns = [column for column in dataset.columns if column.startswith("researched_") and column.endswith("_status")]
    for column in status_columns:
        invalid = set(dataset[column].dropna()) - STATUS_VALUES
        if invalid:
            raise ValueError(f"invalid status values in {column}: {sorted(invalid)}")
    insufficient = dataset["classification"] == "insufficient_evidence"
    if dataset.loc[insufficient, "research_performed"].any():
        raise ValueError("insufficient-evidence rows must not be marked research_performed")
    researched = dataset["research_performed"]
    if not (dataset.loc[researched, "researched_transfer_fee_status"].notna()).all():
        raise ValueError("researched rows must retain researched transfer fee status")
    suspicious = dataset[
        dataset["tm_transfer_fee"].notna()
        & dataset["researched_transfer_fee"].notna()
        & (dataset["tm_transfer_fee"] == dataset["researched_transfer_fee"])
        & (dataset["researched_transfer_fee_status"] == "not_found")
    ]
    if not suspicious.empty:
        raise ValueError("researched transfer fee appears backfilled from Transfermarkt")


def validate_analysis_table(analysis: pd.DataFrame) -> None:
    if len(analysis) != 20:
        raise ValueError("analysis table must contain exactly 20 rows for batch 20")
    if analysis["event_id"].duplicated().any():
        raise ValueError("analysis table event_id values are not unique")
    if len(analysis.columns) > 40:
        raise ValueError("analysis table is too wide")
    forbidden = ("source_url", "source_urls", "evidence_ids", "reported_values", "provider", "model", "token")
    bad = [column for column in analysis.columns if any(marker in column for marker in forbidden)]
    if bad:
        raise ValueError(f"analysis table contains audit/provenance columns: {bad}")


def coverage(dataset: pd.DataFrame) -> dict[str, int]:
    return {field_name: counts["usable_value_count"] for field_name, counts in status_breakdown(dataset).items()}


def status_breakdown(dataset: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts = {}
    for field_name in FIELD_NAMES:
        status_column = f"researched_{field_name}_status"
        if status_column not in dataset:
            continue
        value_column = f"researched_{field_name}"
        field_counts = {status: 0 for status in STATUS_BREAKDOWN}
        statuses = dataset[status_column]
        for status in STATUS_BREAKDOWN:
            if status == "unresearched":
                field_counts[status] = int(statuses.isna().sum())
            else:
                field_counts[status] = int((statuses == status).sum())
        field_counts["usable_value_count"] = usable_value_count(dataset, field_name, value_column, status_column)
        counts[field_name] = field_counts
    return counts


def usable_value_count(dataset: pd.DataFrame, field_name: str, value_column: str, status_column: str) -> int:
    statuses = dataset[status_column]
    values = dataset[value_column] if value_column in dataset else pd.Series([None] * len(dataset), index=dataset.index)
    if field_name in NUMERIC_FIELDS:
        return int((statuses.isin({"disclosed_yes", "partially_disclosed", "conflicting_sources"}) & values.notna()).sum())
    if field_name in BINARY_FIELDS:
        return int((statuses.isin({"disclosed_yes", "disclosed_no", "partially_disclosed", "conflicting_sources"}) & values.notna()).sum())
    return int((statuses.isin({"disclosed_yes", "partially_disclosed", "conflicting_sources", "undisclosed"}) & values.notna()).sum())


def tier_metrics(event_ids: pd.Series, discovered_dir: Path) -> dict[str, int]:
    metrics = {
        "events_with_discovered_tier1": 0,
        "events_with_discovered_tier2": 0,
        "events_with_admissible_tier1": 0,
        "events_with_admissible_tier2": 0,
    }
    for event_id in event_ids:
        discovered = _load_sources(discovered_dir / f"{event_id}.json")
        admissible = _load_sources(discovered_dir / "admissible" / f"{event_id}.json")
        if any(source.get("source_tier") == 1 for source in discovered):
            metrics["events_with_discovered_tier1"] += 1
        if any(source.get("source_tier") == 2 for source in discovered):
            metrics["events_with_discovered_tier2"] += 1
        if any(source.get("source_tier") == 1 for source in admissible):
            metrics["events_with_admissible_tier1"] += 1
        if any(source.get("source_tier") == 2 for source in admissible):
            metrics["events_with_admissible_tier2"] += 1
    return metrics


def _load_sources(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("sources", payload if isinstance(payload, list) else [])


def write_data_dictionary(path: Path, dataset: pd.DataFrame) -> None:
    rows = []
    deterministic = {
        "event_id", "player_name", "player_age_at_transfer", "transfer_date", "season",
        "departing_club", "receiving_club", "deterministic_transfer_type",
        "tm_transfer_fee", "tm_market_value_at_signing",
    }
    metadata = {
        "classification", "review_required", "review_reasons", "discovered_source_count",
        "admissible_source_count", "research_performed", "evidence_sufficient",
        "sufficiency_reasons", "provider", "model", "parley_cost", "source_ids",
        "source_urls", "deal_summary",
    }
    meanings = {
        "tm_transfer_fee": "Transfermarkt reported fee from the deterministic event backbone; never used to fill researched fee fields.",
        "tm_market_value_at_signing": "Nearest Transfermarkt market value around the signing date.",
        "research_performed": "True when a Parley extraction exists for the event.",
        "evidence_sufficient": "True when the source sufficiency gate passed.",
        "source_urls": "Compact JSON list of URLs used in the ContractResearchResult; no article bodies included.",
    }
    for column in dataset.columns:
        if column in deterministic:
            source = "Transfermarkt/deterministic"
            missing = "Blank means absent in structured Transfermarkt output."
        elif column in metadata:
            source = "derived metadata"
            missing = "Blank means no research artifact exists or the metadata was unavailable."
        elif column.startswith("researched_"):
            source = "researched"
            missing = "Blank value columns mean unknown/not applicable; companion status columns preserve the reason when research exists."
        else:
            source = "derived metadata"
            missing = "Blank means unavailable."
        rows.append({
            "column": column,
            "meaning": meanings.get(column, _default_meaning(column)),
            "source": source,
            "expected_type": str(dataset[column].dtype),
            "missing_value_interpretation": missing,
        })
    lines = ["# Research Dataset Batch 20 Data Dictionary", "", "| column | meaning | source | expected type | missing-value interpretation |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]).replace("|", "/") for key in ("column", "meaning", "source", "expected_type", "missing_value_interpretation")) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_analysis_data_dictionary(path: Path, analysis: pd.DataFrame) -> None:
    meanings = {
        "event_id": "Stable transfer event identifier.",
        "player_name": "Player name from the deterministic Transfermarkt backbone.",
        "player_age_at_transfer": "Age on the transfer date.",
        "transfer_date": "Transfer event date.",
        "season": "Transfer season.",
        "departing_club": "Club the player departed.",
        "receiving_club": "Club receiving the player.",
        "deterministic_transfer_type": "Transfer type placeholder from deterministic data when available.",
        "tm_transfer_fee": "Transfermarkt reported fee; kept separate from researched fee and never used for backfilling.",
        "tm_market_value_at_signing": "Nearest Transfermarkt market value at signing.",
        "classification": "Batch-level research classification.",
        "review_required": "Whether validated extraction requires human review.",
        "research_performed": "True when a Parley extraction artifact exists.",
        "evidence_sufficient": "True when admissible evidence passed the sufficiency gate.",
        "discovered_source_count": "Number of deduplicated discovered sources.",
        "admissible_source_count": "Number of admissible Tier 1/2 sources retained for extraction.",
    }
    lines = ["# Research Dataset Batch 20 Analysis Data Dictionary", "", "| column | meaning | source | expected type | missing-value interpretation |", "| --- | --- | --- | --- | --- |"]
    for column in analysis.columns:
        if column.startswith("researched_"):
            source = "researched"
            missing = "Blank value means no supported analytic value; status column distinguishes not_found, not_applicable, undisclosed, and unresearched where available."
            meaning = _default_meaning(column)
        elif column.startswith("tm_") or column in {"event_id", "player_name", "player_age_at_transfer", "transfer_date", "season", "departing_club", "receiving_club", "deterministic_transfer_type"}:
            source = "Transfermarkt/deterministic"
            missing = "Blank means unavailable in deterministic source."
            meaning = meanings.get(column, column.replace("_", " "))
        else:
            source = "derived metadata"
            missing = "Blank means no extraction artifact exists or metadata is unavailable."
            meaning = meanings.get(column, column.replace("_", " "))
        lines.append(f"| {column} | {meaning} | {source} | {analysis[column].dtype} | {missing} |")
    path.write_text("\n".join(lines) + "\n")


def _default_meaning(column: str) -> str:
    if column.startswith("researched_") and column.endswith("_status"):
        return f"Evidence status for {column.removeprefix('researched_').removesuffix('_status')}."
    if column.startswith("researched_") and column.endswith("_evidence_ids"):
        return f"JSON list of evidence IDs supporting {column.removeprefix('researched_').removesuffix('_evidence_ids')}."
    if column.startswith("researched_") and column.endswith("_reported_values"):
        return f"JSON list of per-source reported values for {column.removeprefix('researched_').removesuffix('_reported_values')}."
    if column.startswith("researched_"):
        return f"Structured researched subfield for {column.removeprefix('researched_')}."
    return column.replace("_", " ")


def write_qa_summary(path: Path, dataset: pd.DataFrame, batch: dict[str, Any], discovered_dir: Path) -> None:
    class_counts = dataset["classification"].value_counts().to_dict()
    breakdown = status_breakdown(dataset)
    tiers = tier_metrics(dataset["event_id"], discovered_dir)
    lines = [
        "# Research Dataset Batch 20 QA",
        "",
        f"- Row count: {len(dataset)}",
        f"- Unique events: {dataset['event_id'].nunique()}",
        f"- Columns in compact analysis table: {len(dataset.columns)}",
        f"- Number researched: {int(dataset['research_performed'].sum())}",
        f"- Number insufficient evidence: {class_counts.get('insufficient_evidence', 0)}",
        f"- Clean: {class_counts.get('clean', 0)}",
        f"- Usable with review: {class_counts.get('usable_with_review', 0)}",
        f"- Invalid: {class_counts.get('invalid', 0)}",
        f"- Events with at least one discovered Tier 1 source: {tiers['events_with_discovered_tier1']}",
        f"- Events with at least one discovered Tier 2 source: {tiers['events_with_discovered_tier2']}",
        f"- Events with at least one admissible Tier 1 source: {tiers['events_with_admissible_tier1']}",
        f"- Events with at least one admissible Tier 2 source: {tiers['events_with_admissible_tier2']}",
        f"- Events with no sufficient admissible evidence: {int((~dataset['evidence_sufficient']).sum())}",
        "",
        "## Field Status Breakdown",
        "",
        "| field | usable values | disclosed_yes | disclosed_no | partially_disclosed | conflicting_sources | undisclosed | not_applicable | not_found | unresearched |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field_name, counts in breakdown.items():
        lines.append(
            f"| {field_name} | {counts['usable_value_count']} | {counts['disclosed_yes']} | {counts['disclosed_no']} | "
            f"{counts['partially_disclosed']} | {counts['conflicting_sources']} | {counts['undisclosed']} | "
            f"{counts['not_applicable']} | {counts['not_found']} | {counts['unresearched']} |"
        )
    lines.extend([
        "",
        "Usable values are intentionally conservative: numeric fields require a supported numeric value, not_applicable and not_found are not counted, and explicit disclosed_no counts only for binary clause fields.",
        "",
        "## Playing-Time Outcomes",
        "",
        "playing_time_outcomes.csv remains a separate long-form table. It can contain multiple rows per event and multiple clubs within a window, so this first one-row-per-event consolidation does not collapse those outcomes into a single ambiguous measure.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-row-per-event contract research dataset from existing artifacts")
    parser.add_argument("--structured-transfers", default=str(DEFAULT_OUTPUT_DIR / "structured_transfers.csv"))
    parser.add_argument("--batch-results", default=str(DEFAULT_OUTPUT_DIR / "contract_research" / "batch_20_results.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20.csv"))
    parser.add_argument("--analysis-output", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis.csv"))
    parser.add_argument("--audit-output", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_audit.csv"))
    parser.add_argument("--review-queue-output", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_review_queue.csv"))
    parser.add_argument("--data-dictionary", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_data_dictionary.md"))
    parser.add_argument("--analysis-data-dictionary", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_analysis_data_dictionary.md"))
    parser.add_argument("--qa-summary", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_qa.md"))
    parser.add_argument("--discovered-dir", default=str(DEFAULT_OUTPUT_DIR / "discovered_sources"))
    args = parser.parse_args()

    batch = json.loads(Path(args.batch_results).read_text())
    dataset = build_research_dataset(Path(args.structured_transfers), Path(args.batch_results), Path(args.output))
    analysis = build_analysis_table(dataset)
    audit = build_audit_table(dataset)
    review_queue = build_review_queue(analysis, audit)
    analysis.to_csv(args.analysis_output, index=False)
    audit.to_csv(args.audit_output, index=False)
    review_queue.to_csv(args.review_queue_output, index=False)
    write_data_dictionary(Path(args.data_dictionary), dataset)
    write_analysis_data_dictionary(Path(args.analysis_data_dictionary), analysis)
    write_qa_summary(Path(args.qa_summary), analysis, batch, Path(args.discovered_dir))
    print(f"Wrote {len(dataset)} rows to {args.output}")
    print(f"Wrote {len(analysis)} rows to {args.analysis_output}")
    print(f"Wrote {len(audit)} rows to {args.audit_output}")
    print(f"Wrote {len(review_queue)} rows to {args.review_queue_output}")


if __name__ == "__main__":
    main()
