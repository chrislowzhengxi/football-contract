import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.build_research_dataset import build_research_dataset, coverage
from src.contract_schemas import STATUS_VALUES


ROOT = Path(__file__).resolve().parents[1]
STRUCTURED = ROOT / "data" / "outputs" / "structured_transfers.csv"
BATCH = ROOT / "data" / "outputs" / "contract_research" / "batch_20_results.json"


def dataset():
    return build_research_dataset(STRUCTURED, BATCH)


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


def test_field_coverage_counts_are_computed_from_statuses():
    cov = coverage(dataset())
    assert cov["transfer_type"] == 7
    assert cov["transfer_fee"] == 5
    assert cov["loan_fee"] == 5
    assert cov["purchase_obligation"] == 6
