import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.build_research_dataset import (
    build_analysis_table,
    build_audit_table,
    build_research_dataset,
    build_review_queue,
    status_breakdown,
    tier_metrics,
)
from src.contract_schemas import STATUS_VALUES


ROOT = Path(__file__).resolve().parents[1]
STRUCTURED = ROOT / "data" / "outputs" / "structured_transfers.csv"
BATCH = ROOT / "data" / "outputs" / "contract_research" / "batch_20_results.json"


def dataset():
    return build_research_dataset(STRUCTURED, BATCH)


def analysis():
    return build_analysis_table(dataset())


def audit():
    return build_audit_table(dataset())


def test_batch_dataset_has_exactly_20_unique_events():
    df = dataset()
    assert len(df) == 20
    assert df["event_id"].is_unique


def test_all_batch_events_are_retained_including_insufficient_evidence():
    df = dataset()
    batch = json.loads(BATCH.read_text())
    assert set(df["event_id"]) == {event["event_id"] for event in batch["events"]}
    assert (df["classification"] == "insufficient_evidence").sum() == 11


def test_researched_fee_is_not_backfilled_from_transfermarkt():
    df = dataset().set_index("event_id")
    examples = {
        "tm_64a02c80719320e6347a": 7500000.0,
        "tm_5a39adbdd28687ae4f98": 5000000.0,
        "tm_ab856d20243ac4c7732c": 3750000.0,
    }
    for event_id, tm_fee in examples.items():
        row = df.loc[event_id]
        assert row["tm_transfer_fee"] == tm_fee
        assert pd.isna(row["researched_transfer_fee"])
        assert row["researched_transfer_fee_status"] == "not_found"


def test_researched_status_values_remain_valid():
    df = dataset()
    status_columns = [column for column in df.columns if column.startswith("researched_") and column.endswith("_status")]
    for column in status_columns:
        assert set(df[column].dropna()) <= STATUS_VALUES


def test_classification_counts_match_batch_summary():
    df = dataset()
    batch = json.loads(BATCH.read_text())
    counts = Counter(df["classification"])
    assert counts["clean"] == batch["clean"] == 2
    assert counts["usable_with_review"] == batch["usable_with_review"] == 7
    assert counts["insufficient_evidence"] == batch["insufficient_evidence"] == 11
    assert counts["invalid"] == batch["invalid"] == 0


def test_research_performed_matches_parley_extraction():
    df = dataset().set_index("event_id")
    batch = json.loads(BATCH.read_text())
    for event in batch["events"]:
        assert bool(df.loc[event["event_id"], "research_performed"]) is bool(event["parley_called"])


def test_no_api_keys_or_secrets_appear_in_dataset(tmp_path):
    output = tmp_path / "research_dataset.csv"
    build_research_dataset(STRUCTURED, BATCH, output)
    text = output.read_text()
    assert not re.search(r"(TAVILY_API_KEY|PARLEY_API_KEY|sk-parley|tvly-)", text, re.IGNORECASE)


def test_build_does_not_modify_deterministic_sources(tmp_path):
    before = {
        STRUCTURED: STRUCTURED.read_bytes(),
        ROOT / "data" / "outputs" / "playing_time_outcomes.csv": (ROOT / "data" / "outputs" / "playing_time_outcomes.csv").read_bytes(),
    }
    build_research_dataset(STRUCTURED, BATCH, tmp_path / "research_dataset.csv")
    for path, contents in before.items():
        assert path.read_bytes() == contents


def test_analysis_table_has_20_unique_rows_and_is_compact():
    df = analysis()
    assert len(df) == 20
    assert df["event_id"].is_unique
    assert len(df.columns) <= 40


def test_analysis_table_excludes_audit_provenance_columns():
    columns = set(analysis().columns)
    forbidden = {"source_urls", "source_ids", "provider", "model", "deal_summary", "parley_cost"}
    assert not (columns & forbidden)
    assert not any("evidence_ids" in column or "reported_values" in column or "description" in column for column in columns)


def test_audit_table_preserves_provenance():
    df = audit()
    assert "source_urls" in df
    assert "provider" in df
    assert "model" in df
    assert "researched_transfer_fee_reported_values" in df
    assert df["source_urls"].notna().sum() == 9


def test_review_queue_contains_only_usable_with_review_events():
    queue = build_review_queue(analysis(), audit())
    assert len(queue) == 7
    assert set(queue["event_id"]) == set(analysis().loc[analysis()["classification"] == "usable_with_review", "event_id"])
    assert "source_urls" in queue
    assert queue["source_urls"].notna().all()


def test_status_breakdown_counts_are_internally_consistent():
    df = analysis()
    breakdown = status_breakdown(df)
    for counts in breakdown.values():
        total = sum(counts[status] for status in (
            "disclosed_yes",
            "disclosed_no",
            "partially_disclosed",
            "conflicting_sources",
            "undisclosed",
            "not_applicable",
            "not_found",
            "unresearched",
        ))
        assert total == len(df)


def test_not_applicable_is_not_counted_as_usable_value():
    counts = status_breakdown(analysis())
    assert counts["loan_fee"]["not_applicable"] > 0
    assert counts["loan_fee"]["usable_value_count"] == 0


def test_explicit_disclosed_no_for_binary_fields_counts_as_usable():
    counts = status_breakdown(analysis())
    assert counts["purchase_option"]["disclosed_no"] == 1
    assert counts["purchase_option"]["usable_value_count"] >= 1


def test_numeric_usable_values_require_actual_values():
    counts = status_breakdown(analysis())
    assert counts["transfer_fee"]["disclosed_yes"] == 3
    assert counts["transfer_fee"]["usable_value_count"] == 3


def test_admissible_tier2_metrics_use_admissible_artifacts():
    metrics = tier_metrics(analysis()["event_id"], ROOT / "data" / "outputs" / "discovered_sources")
    assert metrics["events_with_discovered_tier2"] >= metrics["events_with_admissible_tier2"]
    assert metrics["events_with_admissible_tier2"] == 7
