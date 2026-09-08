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
    assert (df["classification"] == "insufficient_evidence").sum() == 7


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
    assert counts["clean"] == batch["clean"] == 6
    assert counts["usable_with_review"] == batch["usable_with_review"] == 7
    assert counts["insufficient_evidence"] == batch["insufficient_evidence"] == 7
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
    assert df["source_urls"].notna().sum() == 13


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


def test_corrected_clean_cases_have_no_unresolved_review_reasons():
    df = audit().set_index("event_id")
    for event_id in ["tm_1c61e919afc42ed4ee08", "tm_5a39adbdd28687ae4f98", "tm_694c30a10815aea6d84d", "tm_890a6e2a4b41a1f8dd46"]:
        assert df.loc[event_id, "classification"] == "clean"
        assert not bool(df.loc[event_id, "review_required"])
        assert pd.isna(df.loc[event_id, "review_reasons"])


def test_thiago_silva_remains_review_required_for_expiry_ambiguity():
    df = audit().set_index("event_id")
    row = df.loc["tm_78e3733d67be84096fe7"]
    assert row["classification"] == "usable_with_review"
    assert bool(row["review_required"])
    assert row["researched_parent_contract_expiry"] == "through 2025-26 season; option to extend through summer 2027"
    assert pd.isna(row["researched_parent_contract_expiry_date"])
    assert row["researched_parent_contract_expiry_precision"] == "year_or_season"


def test_joao_costa_expiry_uses_year_precision_only():
    df = audit().set_index("event_id")
    row = df.loc["tm_890a6e2a4b41a1f8dd46"]
    assert row["classification"] == "clean"
    assert row["researched_parent_contract_expiry"] == "2030"
    assert pd.isna(row["researched_parent_contract_expiry_date"])
    assert row["researched_parent_contract_expiry_year"] == 2030
    assert row["researched_parent_contract_expiry_precision"] == "year"


def test_unresolved_cases_remain_unchanged():
    df = audit().set_index("event_id")
    assert df.loc["tm_b8d2830a76ece80d8d96", "classification"] == "usable_with_review"
    assert df.loc["tm_b8d2830a76ece80d8d96", "researched_purchase_option_status"] == "partially_disclosed"
    assert df.loc["tm_46a910a95150314a1717", "classification"] == "usable_with_review"
    assert df.loc["tm_46a910a95150314a1717", "researched_transfer_type"] == "loan"
    assert df.loc["tm_163d23e38fec40c7a994", "classification"] == "usable_with_review"
    assert df.loc["tm_163d23e38fec40c7a994", "researched_transfer_fee_status"] == "undisclosed"


def test_not_applicable_is_not_counted_as_usable_value():
    counts = status_breakdown(analysis())
    assert counts["loan_fee"]["not_applicable"] > 0
    assert counts["loan_fee"]["usable_value_count"] == 1


def test_explicit_disclosed_no_for_binary_fields_counts_as_usable():
    df = analysis()
    df.loc[df.index[0], "researched_purchase_option"] = False
    df.loc[df.index[0], "researched_purchase_option_status"] = "disclosed_no"
    counts = status_breakdown(df)
    assert counts["purchase_option"]["disclosed_no"] >= 1
    assert counts["purchase_option"]["usable_value_count"] >= 1


def test_numeric_usable_values_require_actual_values():
    counts = status_breakdown(analysis())
    assert counts["transfer_fee"]["disclosed_yes"] == 4
    assert counts["transfer_fee"]["usable_value_count"] == 4


def test_derived_tables_regenerate_successfully(tmp_path):
    raw = build_research_dataset(STRUCTURED, BATCH, tmp_path / "raw.csv")
    analysis_df = build_analysis_table(raw)
    audit_df = build_audit_table(raw)
    queue = build_review_queue(analysis_df, audit_df)
    assert (tmp_path / "raw.csv").exists()
    assert len(analysis_df) == 20
    assert len(queue) == 7
    assert set(raw["event_id"]) == set(analysis_df["event_id"])


def test_admissible_tier2_metrics_use_admissible_artifacts():
    metrics = tier_metrics(analysis()["event_id"], ROOT / "data" / "outputs" / "discovered_sources")
    assert metrics["events_with_discovered_tier2"] >= metrics["events_with_admissible_tier2"]
    assert metrics["events_with_admissible_tier2"] == 7
