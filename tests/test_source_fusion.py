import json
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_OUTPUT_DIR
from src.source_fusion import fuse_event, resolve_field, run_pilot_20
from src.source_fusion_providers import ExistingResearchProvider, OfflineJSONSourceProvider, TransfermarktBackboneProvider
from src.source_fusion_schemas import FieldObservation


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "source_fusion" / "mock_contract_source.json"


def event(**overrides):
    values = {
        "event_id": "tm_test_match",
        "player_name": "Player",
        "age_at_transfer": 25,
        "from_club_name": "Club A",
        "to_club_name": "Club B",
        "transfer_date": "2025-07-01",
        "transfer_type": "permanent",
        "transfer_fee": 7500000,
        "market_value_nearest_transfer": 10000000,
    }
    values.update(overrides)
    return values


def obs(**overrides):
    values = {
        "event_id": "tm_test",
        "field": "transfer_fee",
        "value": 100,
        "normalized_value": 100,
        "status": "disclosed_yes",
        "source_name": "source_a",
        "source_type": "structured_external_candidate",
        "source_tier": 1,
        "confidence": 0.9,
        "currency": "EUR",
    }
    values.update(overrides)
    return FieldObservation(**values)


def test_transfermarkt_provider_preserves_backbone_ownership():
    observations = TransfermarktBackboneProvider().get_event_observations(event())
    by_field = {observation.field: observation for observation in observations}

    assert by_field["tm_transfer_fee"].value == 7500000
    assert by_field["tm_transfer_fee"].source_type == "deterministic_backbone"
    assert "not independent contract research" in by_field["tm_transfer_fee"].notes


def test_existing_research_provider_keeps_researched_fee_separate_from_tm_fee():
    structured = pd.read_csv(DEFAULT_OUTPUT_DIR / "structured_transfers.csv", dtype={"event_id": str})
    row = structured[structured["event_id"] == "tm_64a02c80719320e6347a"].iloc[0].to_dict()
    observations = ExistingResearchProvider().get_event_observations(row)
    fee = next(observation for observation in observations if observation.field == "transfer_fee")

    assert row["transfer_fee"] == 7500000
    assert fee.status == "not_found"
    assert fee.value is None


def test_offline_json_provider_reads_synthetic_records():
    provider = OfflineJSONSourceProvider(FIXTURE)
    observations = provider.get_event_observations(event())

    assert observations[0].source_name == "external_contract_source"
    assert observations[0].value == 7500000
    assert observations[0].currency == "EUR"


def test_matching_independent_values_are_corroborated():
    resolution = resolve_field("tm_test", "transfer_fee", [
        obs(source_name="source_a"),
        obs(source_name="source_b"),
    ])

    assert resolution.resolution_status == "resolved_corroborated"
    assert resolution.resolved_value == 100


def test_conflicting_fee_values_remain_unresolved():
    resolution = resolve_field("tm_test", "transfer_fee", [
        obs(value=100, normalized_value=100, currency="EUR"),
        obs(value=110, normalized_value=110, currency="EUR", source_name="source_b"),
    ])

    assert resolution.resolution_status == "unresolved_conflict"
    assert resolution.review_required
    assert resolution.conflicting_observations


def test_currency_difference_is_not_silently_reconciled():
    resolution = resolve_field("tm_test", "transfer_fee", [
        obs(value=100, normalized_value=100, currency="EUR"),
        obs(value=100, normalized_value=100, currency="GBP", source_name="source_b"),
    ])

    assert resolution.resolution_status == "unresolved_conflict"


def test_year_vs_exact_date_precision_is_preserved_as_conflict():
    resolution = resolve_field("tm_test", "parent_contract_expiry", [
        obs(field="parent_contract_expiry", value="2029", normalized_value="2029", precision="year"),
        obs(field="parent_contract_expiry", value="2029-06-30", normalized_value="2029-06-30", precision="date", source_name="source_b"),
    ])

    assert resolution.resolution_status == "unresolved_conflict"


def test_not_found_does_not_override_supported_observation():
    resolution = resolve_field("tm_test", "purchase_option", [
        obs(field="purchase_option", value=True, normalized_value=True, currency=None),
        obs(field="purchase_option", value=None, normalized_value=None, status="not_found", source_name="source_b", currency=None),
    ])

    assert resolution.resolution_status == "resolved_single_source"
    assert resolution.resolved_value is True


def test_not_found_never_becomes_disclosed_no():
    resolution = resolve_field("tm_test", "sell_on", [
        obs(field="sell_on", value=None, normalized_value=None, status="not_found", currency=None),
    ])

    assert resolution.resolution_status == "not_found"
    assert resolution.resolved_value is None


def test_tier3_cannot_establish_exact_financial_term():
    resolution = resolve_field("tm_test", "transfer_fee", [
        obs(source_tier=3, source_type="web_research"),
    ])

    assert resolution.resolution_status in {"not_found", "insufficient_evidence"}
    assert resolution.resolved_value is None


def test_fusion_keeps_transfermarkt_fee_distinguishable_from_researched_fee():
    provider = OfflineJSONSourceProvider(FIXTURE)
    result = fuse_event(event(), [TransfermarktBackboneProvider(), provider])

    assert result.resolutions["tm_transfer_fee"].resolved_value == 7500000
    assert result.resolutions["transfer_fee"].resolved_value == 7500000
    assert result.resolutions["tm_transfer_fee"].preferred_observation.source_name == "Transfermarkt"
    assert result.resolutions["transfer_fee"].preferred_observation.source_name == "external_contract_source"


def test_pilot_20_outputs_are_written_offline(tmp_path):
    observations, fused, audit = run_pilot_20(tmp_path)

    assert len(fused) == 20
    assert observations
    assert audit
    assert (tmp_path / "pilot_20_observations.csv").exists()
    assert (tmp_path / "pilot_20_fused.csv").exists()
    assert (tmp_path / "pilot_20_fusion_audit.csv").exists()


def test_observation_provenance_is_serializable():
    observation = obs(evidence_ids=["s1"], source_reference="mock:source")

    payload = observation.to_dict()
    assert payload["evidence_ids"] == ["s1"]
    assert json.dumps(payload)
