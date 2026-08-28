import pytest
import pandas as pd

from src.contract_schemas import ContractResearchResult
from src.research_contract import FixtureResearchProvider, run


def result_payload(**fields):
    payload = {"event_id": "tm_test", "deal_summary": "Research incomplete", "sources": []}
    payload.update(fields)
    return payload


def source():
    return [{
        "evidence_id": "s1",
        "source_url": "https://example.com/source",
        "source_title": "Source",
        "publisher": "Publisher",
        "source_type": "major_news",
        "publication_date": None,
        "retrieval_date": "2026-08-28",
        "evidence_text": "Evidence",
        "language": "en",
    }]


def test_not_found_does_not_become_false():
    with pytest.raises(ValueError):
        ContractResearchResult.from_dict(result_payload(sell_on={"value": False, "status": "not_found"}))


def test_partial_disclosure_preserves_unknown_percentage():
    result = ContractResearchResult.from_dict(result_payload(sell_on={
        "evidence_ids": ["s1"],
        "value": True,
        "percentage": None,
        "status": "partially_disclosed",
        "confidence": 0.8,
    }, sources=source()))
    assert result.sell_on.value is True
    assert result.sell_on.percentage is None


def test_conflicting_sources_trigger_review():
    result = ContractResearchResult.from_dict(result_payload(
        sources=source(),
        transfer_fee={"status": "conflicting_sources", "confidence": 0.7, "evidence_ids": ["s1"]}
    ))
    assert result.review_required
    assert "conflicting sources for transfer_fee" in result.review_reasons


def test_obligation_trigger_triggers_review():
    result = ContractResearchResult.from_dict(result_payload(
        sources=source(),
        purchase_obligation={"value": True, "status": "disclosed_yes", "confidence": 0.9, "evidence_ids": ["s1"]},
        obligation_trigger={"value": True, "status": "disclosed_yes", "confidence": 0.9, "evidence_ids": ["s1"]},
    ))
    assert result.review_required
    assert "purchase obligation requires human review" in result.review_reasons
    assert "obligation trigger requires human review" in result.review_reasons


def test_undisclosed_fee_remains_null():
    result = ContractResearchResult.from_dict(result_payload(
        sources=source(),
        transfer_fee={"amount": None, "status": "undisclosed", "confidence": 0.9, "evidence_ids": ["s1"]}
    ))
    assert result.transfer_fee.amount is None


@pytest.mark.parametrize("field", [
    {"status": "unknown"},
    {"status": "not_found", "confidence": 1.1},
])
def test_malformed_status_or_confidence_is_rejected(field):
    with pytest.raises(ValueError):
        ContractResearchResult.from_dict(result_payload(transfer_fee=field))


def test_fixture_provider_runs_one_event(tmp_path):
    events_path = tmp_path / "structured_transfers.csv"
    fixture_path = tmp_path / "provider.json"
    output_dir = tmp_path / "contract_research"
    pd.DataFrame([{"event_id": "tm_test", "player_name": "Player", "from_club_name": "A", "to_club_name": "B", "transfer_date": "2024-07-01"}]).to_csv(events_path, index=False)
    fixture_path.write_text('{"deal_summary":"No reliable terms found","sources":[]}')

    output_path = run("tm_test", events_path, output_dir, FixtureResearchProvider(fixture_path))

    assert output_path.exists()
    assert output_path.parent == output_dir
    assert output_path.read_text().find('"event_id": "tm_test"') >= 0
