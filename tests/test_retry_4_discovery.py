from src.retry_4_discovery import retry_improvement_metrics, retry_parley_metrics


def test_retry_parley_metrics_distinguish_attempts_successes_and_costs():
    rows = [
        {
            "parley_info": {
                "attempts": 2,
                "successful_extraction": True,
                "total_attempt_cost": 0.031,
            }
        },
        {
            "parley_info": {
                "attempts": 1,
                "successful_extraction": False,
                "total_attempt_cost": 0.004,
            }
        },
        {"parley_info": {"called": False}},
    ]

    assert retry_parley_metrics(rows) == {
        "attempts": 3,
        "successful_extractions": 1,
        "total_cost": 0.035,
    }


def test_retry_improvement_metrics_do_not_treat_tier3_identity_as_admissible_lift():
    rows = [
        {
            "old_admissible_sources": 0,
            "new_admissible_sources": 0,
            "old_sufficient": False,
            "new_sufficient": False,
            "new_exact_match_sources": 1,
            "new_likely_match_sources": 0,
            "fields_newly_established": [],
        },
        {
            "old_admissible_sources": 0,
            "new_admissible_sources": 0,
            "old_sufficient": False,
            "new_sufficient": False,
            "new_exact_match_sources": 0,
            "new_likely_match_sources": 0,
            "fields_newly_established": [],
        },
        {
            "old_admissible_sources": 0,
            "new_admissible_sources": 1,
            "old_sufficient": False,
            "new_sufficient": True,
            "new_exact_match_sources": 1,
            "new_likely_match_sources": 0,
            "fields_newly_established": ["transfer_type: loan"],
        },
    ]

    assert retry_improvement_metrics(rows) == {
        "attempted": 3,
        "event_identity_improved": 2,
        "admissible_evidence_improved": 1,
        "evidence_sufficiency_converted": 1,
        "contract_field_improved": 1,
    }
