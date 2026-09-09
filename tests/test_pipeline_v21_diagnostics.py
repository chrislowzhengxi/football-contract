import urllib.request

from src.pipeline_v21_diagnostics import field_signal, offline_replay, source_registry_audit
from src.source_discovery import SourceCandidate, source_tier


def test_source_tiers_unchanged_by_frequency_alone():
    rows = source_registry_audit()
    uncategorized = [row for row in rows if row["recommended_action"] == "no_change"]

    assert uncategorized
    assert all("promotion" in row["reason"] or "not eligible" in row["reason"] for row in uncategorized)


def test_tier3_never_becomes_contractual_signal_source():
    source = SourceCandidate(
        source_url="https://transfermarkt.com/player",
        source_title="Player transfer fee",
        publisher="transfermarkt.com",
        publication_date=None,
        source_type="aggregator",
        evidence_text="Player transfer fee EUR 10m",
        language="en",
    )

    assert source_tier(source) == 3
    assert field_signal("transfer_fee", source)


def test_offline_replay_performs_no_network_calls(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("offline replay must not call network APIs")

    monkeypatch.setattr(urllib.request, "urlopen", fail)

    rows = offline_replay()

    assert rows
    assert {"event_id", "field", "v2_status", "v21_status", "changed"} <= set(rows[0])
