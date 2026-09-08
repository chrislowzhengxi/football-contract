from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .contract_schemas import FIELD_NAMES, STATUS_VALUES, ContractResearchResult


CLASSIFICATIONS = {"clean", "usable_with_review", "insufficient_evidence", "invalid"}


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


def coverage(dataset: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for field_name in FIELD_NAMES:
        status_column = f"researched_{field_name}_status"
        if status_column not in dataset:
            continue
        counts[field_name] = int(dataset[status_column].isin({"disclosed_yes", "disclosed_no", "partially_disclosed", "undisclosed", "conflicting_sources", "not_applicable"}).sum())
    return counts


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
    cov = coverage(dataset)
    tier1 = tier2 = 0
    for event_id in dataset["event_id"]:
        source_path = discovered_dir / f"{event_id}.json"
        if not source_path.exists():
            continue
        payload = json.loads(source_path.read_text())
        sources = payload.get("sources", [])
        if any(source.get("source_tier") == 1 for source in sources):
            tier1 += 1
        if any(source.get("source_tier") == 2 for source in sources):
            tier2 += 1
    lines = [
        "# Research Dataset Batch 20 QA",
        "",
        f"- Row count: {len(dataset)}",
        f"- Unique events: {dataset['event_id'].nunique()}",
        f"- Number researched: {int(dataset['research_performed'].sum())}",
        f"- Number insufficient evidence: {class_counts.get('insufficient_evidence', 0)}",
        f"- Clean: {class_counts.get('clean', 0)}",
        f"- Usable with review: {class_counts.get('usable_with_review', 0)}",
        f"- Invalid: {class_counts.get('invalid', 0)}",
        f"- Events with at least one Tier 1 source: {tier1}",
        f"- Events with at least one admissible Tier 2 source: {tier2}",
        f"- Events with no sufficient admissible evidence: {int((~dataset['evidence_sufficient']).sum())}",
        "",
        "## Field Coverage",
        "",
    ]
    for field_name, count in cov.items():
        lines.append(f"- {field_name}: {count}/{len(dataset)} established")
    lines.extend([
        "",
        "Coverage treats disclosed_yes, disclosed_no, partially_disclosed, undisclosed, conflicting_sources, and not_applicable as established statuses. not_found and unresearched blanks are not counted as established.",
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
    parser.add_argument("--data-dictionary", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_data_dictionary.md"))
    parser.add_argument("--qa-summary", default=str(DEFAULT_OUTPUT_DIR / "research_dataset_batch_20_qa.md"))
    parser.add_argument("--discovered-dir", default=str(DEFAULT_OUTPUT_DIR / "discovered_sources"))
    args = parser.parse_args()

    batch = json.loads(Path(args.batch_results).read_text())
    dataset = build_research_dataset(Path(args.structured_transfers), Path(args.batch_results), Path(args.output))
    write_data_dictionary(Path(args.data_dictionary), dataset)
    write_qa_summary(Path(args.qa_summary), dataset, batch, Path(args.discovered_dir))
    print(f"Wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
