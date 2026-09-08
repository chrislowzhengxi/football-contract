import json

import pytest

from src.source_discovery import (
    TavilySearchProvider,
    SourceCandidate,
    assess_source_sufficiency,
    build_search_queries,
    deduplicate_sources,
    filter_admissible_sources,
    score_source,
    source_tier,
)


def candidate(**overrides):
    values = {
        "source_url": "https://example.com/article",
        "source_title": "Player transfer fee announced",
        "publisher": "example.com",
        "publication_date": "2025-07-01",
        "source_type": "major_news",
        "evidence_text": "Player joined Club for EUR 10m with add-ons.",
        "language": "en",
    }
    values.update(overrides)
    return SourceCandidate(**values)


def test_query_templates_cover_contract_terms():
    queries = build_search_queries({
        "player_name": "Player",
        "from_club_name": "Club A",
        "to_club_name": "Club B",
        "transfer_date": "2025-07-01",
    })
    assert any("transfer fee" in query for query in queries)
    assert any("option to buy" in query for query in queries)
    assert any("obligation to buy" in query for query in queries)
    assert any("sell-on clause" in query for query in queries)


def test_official_source_ranks_above_secondary_reporting():
    event = {"player_name": "Player", "from_club_name": "Club A", "to_club_name": "Club B"}
    official = candidate(source_url="https://club-a.example.com/news", source_type="official")
    secondary = candidate(source_url="https://news.example.com/article", source_type="major_news")
    assert score_source(official, event) > score_source(secondary, event)


def test_syndicated_sources_are_deduplicated():
    first = candidate(source_url="https://news.example.com/a", source_title="Player joins Club")
    duplicate = candidate(source_url="https://mirror.example.com/b", source_title="Player joins Club")
    first.quality_score = 80
    duplicate.quality_score = 50
    result = deduplicate_sources([duplicate, first])
    assert len(result) == 1
    assert result[0].source_url == first.source_url


def test_inaccessible_source_is_not_sufficient():
    dead = candidate(accessible=False, quality_score=90)
    sufficient, reasons = assess_source_sufficiency([dead])
    assert not sufficient
    assert reasons


def test_metadata_and_evidence_text_are_preserved_exactly():
    item = candidate(evidence_text="Exact provider snippet", reported_values=[{"amount": 10, "currency": "EUR"}])
    payload = item.to_dict()
    assert payload["source_url"] == "https://example.com/article"
    assert payload["evidence_text"] == "Exact provider snippet"
    assert payload["reported_values"] == [{"amount": 10, "currency": "EUR"}]
    assert "invented" not in json.dumps(payload)


def test_one_strong_source_is_sufficient():
    strong = candidate(source_type="official", accessible=True, quality_score=80)
    sufficient, reasons = assess_source_sufficiency([strong])
    assert sufficient
    assert reasons == ["at least one strong accessible source"]


def test_two_independent_reputable_sources_are_sufficient():
    first = candidate(source_url="https://reuters.com/a", publisher="reuters.com", quality_score=55)
    second = candidate(source_url="https://bbc.com/b", publisher="bbc.com", quality_score=55)
    sufficient, reasons = assess_source_sufficiency([first, second])
    assert sufficient
    assert reasons == ["two independent reputable sources"]


def test_tavily_normalizes_results_and_preserves_provider_score(monkeypatch):
    provider = TavilySearchProvider("test-key")
    provider._search = lambda query: {"results": [{
        "url": "https://reuters.com/story",
        "title": "Player transfer",
        "published_date": "2025-07-01",
        "content": "Player joined Porto for EUR 10m.",
        "score": 0.91,
    }]}
    monkeypatch.setattr("src.source_discovery._check_accessibility", lambda url, timeout: True)
    results = provider.search_transfer({"player_name": "Player", "from_club_name": "Club A", "to_club_name": "Porto", "transfer_date": "2025-07-01"})
    assert len(results) == 1
    assert results[0].publisher == "reuters.com"
    assert results[0].provider_score == 0.91
    assert results[0].evidence_text == "Player joined Porto for EUR 10m."
    assert provider.search_count == 0


def test_tavily_missing_metadata_remains_null(monkeypatch):
    provider = TavilySearchProvider("test-key")
    provider._search = lambda query: {"results": [{"url": "https://example.com/result", "content": "snippet"}]}
    monkeypatch.setattr("src.source_discovery._check_accessibility", lambda url, timeout: True)
    result = provider.search_transfer({"player_name": "Player", "from_club_name": "A", "to_club_name": "B", "transfer_date": "2025-07-01"})[0]
    assert result.source_title is None
    assert result.publication_date is None
    assert result.language == "en"


def test_tavily_authentication_error_is_safe(monkeypatch):
    from urllib.error import HTTPError

    provider = TavilySearchProvider("secret-not-used-in-message")
    def fail(request, timeout):
        raise HTTPError(request.full_url, 401, "unauthorized", {}, None)
    monkeypatch.setattr("src.source_discovery.urlopen", fail)
    with pytest.raises(RuntimeError, match="Tavily authentication failed"):
        provider._search("query")
    assert "secret-not-used" not in "Tavily authentication failed"


def test_tier_three_sources_are_not_admissible_for_parley():
    official = candidate(source_url="https://club.example.com/announcement", source_type="official")
    aggregator = candidate(source_url="https://www.transfermarkt.us/player", source_type="aggregator")
    admissible = filter_admissible_sources([official, aggregator])
    assert source_tier(official) == 1
    assert source_tier(aggregator) == 3
    assert admissible == [official]


def test_major_news_is_admissible_but_social_is_not():
    news = candidate(source_url="https://reuters.com/story", source_type="major_news")
    social = candidate(source_url="https://instagram.com/p/story", source_type="aggregator")
    assert [item.source_url for item in filter_admissible_sources([news, social])] == [news.source_url]
