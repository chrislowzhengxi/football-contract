import pytest
import pandas as pd
import json

from src.contract_schemas import ContractResearchResult, render_model_schema_instructions
from src.research_contract import FixtureResearchProvider, ParleyProvider, ProviderFailure, _completion_json, run
from src.source_discovery import SourceCandidate


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
    {"status": "confirmed"},
    {"status": "yes"},
    {"status": "not_found", "confidence": 1.1},
])
def test_malformed_status_or_confidence_is_rejected(field):
    with pytest.raises(ValueError):
        ContractResearchResult.from_dict(result_payload(transfer_fee=field))


@pytest.mark.parametrize("status", ["disclosed_yes", "disclosed_no"])
def test_explicit_disclosure_statuses_are_accepted(status):
    result = ContractResearchResult.from_dict(result_payload(
        sources=source(),
        transfer_type={"value": status == "disclosed_yes", "status": status, "confidence": 0.9, "evidence_ids": ["s1"]},
    ))
    assert result.transfer_type.status == status


def test_missing_evidence_remains_not_found():
    result = ContractResearchResult.from_dict(result_payload())
    assert result.sell_on.status == "not_found"
    assert result.sell_on.value is None


def test_inferred_parent_expiry_triggers_review():
    result = ContractResearchResult.from_dict(result_payload(
        sources=source(),
        parent_contract_expiry={
            "value": "2025-06-30",
            "date": "2025-06-30",
            "status": "partially_disclosed",
            "confidence": 0.8,
            "evidence_ids": ["s1"],
        },
    ))
    assert result.review_required
    assert "unsupported_contract_expiry" in result.review_reasons


def test_explicit_parent_expiry_does_not_trigger_inference_review():
    evidence = source()
    evidence[0]["evidence_text"] = "The player's contract expires in June 2025."
    result = ContractResearchResult.from_dict(result_payload(
        sources=evidence,
        parent_contract_expiry={
            "value": "2025-06-30",
            "date": "2025-06-30",
            "status": "disclosed_yes",
            "confidence": 0.9,
            "evidence_ids": ["s1"],
        },
    ))
    assert "parent contract expiry is not explicitly sourced" not in result.review_reasons


def test_base_fee_and_total_package_are_distinguished():
    evidence = source()
    evidence[0]["evidence_text"] = "The base fee was €28m and could increase to €30m with add-ons."
    result = ContractResearchResult.from_dict(result_payload(
        sources=evidence,
        transfer_fee={"amount": 28000000, "currency": "EUR", "status": "partially_disclosed", "confidence": 0.8, "evidence_ids": ["s1"]},
        add_ons={"amount": 2000000, "currency": "EUR", "status": "partially_disclosed", "confidence": 0.8, "evidence_ids": ["s1"]},
    ))
    assert "base_fee_vs_total_package_ambiguity" in result.review_reasons
    assert "conflicting_base_fee_reports" not in result.review_reasons


def test_different_currency_reports_get_difference_reason_without_conversion():
    evidence = source()
    evidence[0]["evidence_text"] = "The fee was €28m with add-ons to €30m."
    evidence.append({**evidence[0], "evidence_id": "s2", "evidence_text": "The fee was £23.8m with add-ons to £25.5m."})
    result = ContractResearchResult.from_dict(result_payload(
        sources=evidence,
        transfer_fee={"status": "partially_disclosed", "confidence": 0.8, "evidence_ids": ["s1", "s2"], "reported_values": [{"amount": 28000000, "currency": "EUR", "evidence_id": "s1"}, {"amount": 23.8, "currency": "GBP", "evidence_id": "s2"}]},
        add_ons={"status": "partially_disclosed", "confidence": 0.8, "evidence_ids": ["s1", "s2"]},
    ))
    assert "currency_reporting_difference" in result.review_reasons
    assert result.transfer_fee.reported_values[0]["currency"] == "EUR"
    assert result.transfer_fee.reported_values[1]["currency"] == "GBP"


def test_genuinely_conflicting_base_fees_get_specific_reason():
    result = ContractResearchResult.from_dict(result_payload(
        sources=source(),
        transfer_fee={"status": "conflicting_sources", "confidence": 0.5, "evidence_ids": ["s1"]},
    ))
    assert "conflicting_base_fee_reports" in result.review_reasons


def test_later_permanent_fee_is_not_attached_to_loan():
    evidence = source()
    evidence[0]["evidence_text"] = "The player joined on loan; a later permanent transfer was agreed separately."
    result = ContractResearchResult.from_dict(result_payload(
        sources=evidence,
        transfer_type={"value": "loan", "status": "disclosed_yes", "confidence": 0.9, "evidence_ids": ["s1"]},
        loan_fee={"amount": 7000000, "status": "disclosed_yes", "confidence": 0.9, "evidence_ids": ["s1"]},
    ))
    assert "later_transfer_used_to_interpret_original_deal" in result.review_reasons
    assert result.loan_fee.amount == 7000000


def test_incomplete_option_terms_get_specific_reason():
    result = ContractResearchResult.from_dict(result_payload(
        sources=source(),
        purchase_option={"value": True, "status": "disclosed_yes", "confidence": 0.9, "evidence_ids": ["s1"]},
    ))
    assert "option_terms_incomplete" in result.review_reasons


def test_single_secondary_exact_fee_requires_review():
    result = ContractResearchResult.from_dict(result_payload(
        sources=source(),
        transfer_fee={"amount": 50000000, "currency": "EUR", "status": "disclosed_yes", "confidence": 0.9, "evidence_ids": ["s1"]},
    ))
    assert "weak_source_for_exact_financial_term" in result.review_reasons


def test_deterministic_fee_is_not_researched_fee_without_evidence():
    event = {"transfer_fee": 12000000, "player_name": "Player"}
    result = ContractResearchResult.from_dict(result_payload())
    assert event["transfer_fee"] == 12000000
    assert result.transfer_fee.amount is None
    assert result.transfer_fee.status == "not_found"
    instructions = render_model_schema_instructions()
    assert "never copy them into researched contract fields" in instructions


def test_fenced_json_response_is_parsed():
    assert _completion_json({"choices": [{"message": {"content": "```json\n{\"deal_summary\": \"ok\"}\n```"}}]}) == {"deal_summary": "ok"}


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


def test_parley_provider_sends_sources_and_captures_metadata(monkeypatch):
    calls = {}

    class Response:
        headers = {"x-parley-v1-cost": "0.0123"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "model": "bedrock/claude-haiku-4-5",
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                "choices": [{"message": {"content": json.dumps({
                    "deal_summary": "Terms not found",
                    "sources": [],
                })}}],
            }).encode()

    def fake_urlopen(request, timeout):
        calls["url"] = request.full_url
        calls["headers"] = dict(request.headers)
        calls["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("src.research_contract.urlopen", fake_urlopen)
    provider = ParleyProvider("sk-parley-v1-test-only", "bedrock/claude-haiku-4-5", "prompt")
    result = provider.research(
        {"event_id": "tm_test", "player_name": "Player"},
        [SourceCandidate("https://example.com", "Title", "Publisher", None, "major_news", "Evidence", "en")],
    )

    assert calls["url"] == "https://parley.api.mit.edu/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer sk-parley-v1-test-only"
    assert calls["body"]["response_format"] == {"type": "json_object"}
    assert calls["body"]["messages"][1]["content"].find("Evidence") >= 0
    assert result.provider_metadata.provider == "parley"
    assert result.provider_metadata.model == "bedrock/claude-haiku-4-5"
    assert result.provider_metadata.prompt_tokens == 12
    assert result.provider_metadata.completion_tokens == 8
    assert result.provider_metadata.parley_cost_header == "0.0123"
    assert result.provider_metadata.total_request_cost == 0.0123


def test_failed_schema_validation_preserves_safe_metadata(monkeypatch):
    class Response:
        status = 200
        headers = {"x-parley-v1-cost": "0.0042"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "model": "claude-haiku-4-5",
                "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
                "choices": [{"message": {"content": json.dumps({
                    "deal_summary": "Invalid status test",
                    "sources": [],
                    "transfer_type": {"status": "confirmed", "confidence": 0.9},
                })}}],
            }).encode()

    monkeypatch.setattr("src.research_contract.urlopen", lambda request, timeout: Response())
    provider = ParleyProvider("sk-parley-v1-never-store", "claude-haiku-4-5", "prompt")
    with pytest.raises(ProviderFailure) as failure:
        provider.research({"event_id": "tm_test"}, [])
    diagnostics = failure.value.diagnostics
    assert diagnostics["failure_stage"] == "schema_validation"
    assert diagnostics["prompt_tokens"] == 20
    assert diagnostics["completion_tokens"] == 7
    assert diagnostics["total_tokens"] == 27
    assert diagnostics["parley_cost_header"] == "0.0042"
    assert "sk-parley" not in json.dumps(diagnostics)
