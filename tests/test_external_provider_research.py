import csv
import json
from pathlib import Path

import pytest

from src.external_provider_research import (
    REQUESTED_FIELDS,
    ProviderAccessModel,
    ProviderCandidate,
    ProviderCapability,
    provider_research_results,
    score_provider,
    write_outputs,
)
from src.source_fusion_providers import OfflineJSONSourceProvider


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "source_fusion"


def _provider(name):
    return next(result for result in provider_research_results() if result.candidate.provider_name == name)


def _capabilities(name):
    return {capability.field: capability for capability in _provider(name).capabilities}


def test_provider_schemas_reject_invalid_controlled_values():
    with pytest.raises(ValueError):
        ProviderCapability("Provider", "transfer_fee", support_status="maybe")

    with pytest.raises(ValueError):
        ProviderCapability("Provider", "not_a_contract_field")

    with pytest.raises(ValueError):
        ProviderAccessModel("Provider", api_available="yes")


def test_unknown_capability_does_not_count_as_supported():
    candidate = ProviderCandidate(
        provider_name="Unknown Contract API",
        provider_url="https://example.com",
        documentation_url="https://example.com/docs",
        provider_category="fixture",
        fields_claimed=[],
        api_available="true",
        player_level="true",
        transfer_level="true",
    )
    unknown_caps = [
        ProviderCapability(
            provider="Unknown Contract API",
            field=field,
            support_status="unknown",
            data_format="unknown",
        )
        for field in REQUESTED_FIELDS
    ]
    supported_caps = [
        ProviderCapability(
            provider="Unknown Contract API",
            field=field,
            support_status="supported" if field == "transfer_fee" else "unknown",
            data_format="structured" if field == "transfer_fee" else "unknown",
        )
        for field in REQUESTED_FIELDS
    ]

    assert not [capability for capability in unknown_caps if capability.support_status == "supported"]
    assert score_provider(candidate, supported_caps) > score_provider(candidate, unknown_caps)


def test_capology_contract_scope_does_not_promote_transfer_fee():
    capabilities = _capabilities("Capology API")

    assert capabilities["transfer_fee"].support_status == "not_supported"
    assert capabilities["add_ons"].support_status == "not_supported"
    assert capabilities["parent_contract_expiry"].support_status == "supported"
    assert capabilities["salary"].support_status == "supported"


def test_transfer_api_candidates_only_support_documented_transfer_fields():
    sportmonks = _capabilities("Sportmonks Football API")
    api_football = _capabilities("API-Football")

    assert sportmonks["transfer_type"].support_status == "supported"
    assert sportmonks["transfer_fee"].support_status == "supported"
    assert sportmonks["purchase_option"].support_status == "unknown"
    assert api_football["transfer_type"].support_status == "supported"
    assert api_football["loan_fee"].support_status == "unknown"


def test_provider_outputs_are_deterministic_and_rectangular(tmp_path):
    results = write_outputs(tmp_path)

    candidates = list(csv.DictReader((tmp_path / "provider_candidates.csv").open()))
    capabilities = list(csv.DictReader((tmp_path / "provider_field_capabilities.csv").open()))
    rankings = list(csv.DictReader((tmp_path / "provider_rankings.csv").open()))

    assert len(candidates) == len(results)
    assert len(capabilities) == len(results) * len(REQUESTED_FIELDS)
    assert rankings[0]["provider"] == "Sportmonks Football API"
    assert "transfer_fee" in json.loads(rankings[0]["supported_fields"])
    assert (tmp_path / "provider_incremental_value.csv").exists()
    assert (tmp_path / "provider_integration_plan.md").exists()


def test_top_provider_fixtures_can_be_ingested_by_source_fusion_interface():
    fixture_cases = [
        ("sportmonks_example.json", "tm_provider_sportmonks", {"transfer_type", "transfer_fee"}),
        ("capology_example.json", "tm_provider_capology", {"parent_contract_expiry", "salary"}),
        ("api_football_example.json", "tm_provider_api_football", {"transfer_type", "transfer_fee"}),
    ]

    for fixture_name, event_id, fields in fixture_cases:
        provider = OfflineJSONSourceProvider(FIXTURE_DIR / fixture_name)
        observations = provider.get_event_observations({"event_id": event_id})

        assert {observation.field for observation in observations} == fields
        assert all(observation.source_type == "structured_external_candidate" for observation in observations)
        assert all(observation.source_reference for observation in observations)
