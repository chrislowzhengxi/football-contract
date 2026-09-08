import json

import pytest

from src.source_discovery import (
    TavilySearchProvider,
    SourceCandidate,
    assess_event_match,
    assess_source_sufficiency,
    build_query_plan,
    build_search_queries,
    deduplicate_sources,
    discover_transfer,
    filter_admissible_sources,
    official_domain_for_club,
    retrieve_fullpage_for_promising_sources,
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


def test_query_plan_has_bounded_multistage_queries():
    plans = build_query_plan({
        "player_name": "Florentino",
        "from_club_name": "SL Benfica",
        "to_club_name": "Burnley",
        "transfer_date": "2025-07-01",
    })
    assert len(plans) <= 6
    assert [plan.family for plan in plans][:2] == ["generic_event", "club_domain"]
    assert any(plan.family == "mechanism" for plan in plans)
    assert any(plan.family == "local_language" and plan.language == "pt" for plan in plans)
    assert any("obrigação de compra" in plan.query for plan in plans)


def test_mechanism_query_uses_unresolved_fields():
    plans = build_query_plan(
        {
            "player_name": "Player",
            "from_club_name": "Club A",
            "to_club_name": "Club B",
            "transfer_date": "2025-07-01",
        },
        unresolved_fields=["loan_fee", "purchase_obligation"],
    )
    mechanism = [plan.query for plan in plans if plan.family == "mechanism"][0]
    assert "loan fee" in mechanism
    assert "obligation to buy" in mechanism


def test_remaining_case_local_language_queries_are_bounded():
    examples = [
        ("João Mário", "Besiktas", "Benfica", "pt"),
        ("Kerem Aktürkoğlu", "Benfica", "Fenerbahçe", "pt"),
        ("Renato Sanches", "Benfica", "PSG", "pt"),
        ("Marko Grujić", "Porto", "AEK Athens", "pt"),
    ]
    for player, from_club, to_club, language in examples:
        plans = build_query_plan({
            "player_name": player,
            "from_club_name": from_club,
            "to_club_name": to_club,
            "transfer_date": "2025-06-30",
        })
        assert len(plans) <= 6
        assert any(plan.family == "local_language" and plan.language == language for plan in plans)


def test_official_domains_cover_remaining_case_clubs():
    assert official_domain_for_club("Besiktas") == "bjk.com.tr"
    assert official_domain_for_club("Fenerbahçe") == "fenerbahce.org"
    assert official_domain_for_club("AEK Athens") == "aekfc.gr"
    assert official_domain_for_club("Al-Ain") == "alainclub.ae"


def test_exact_transfer_direction_is_admissible():
    event = {
        "player_name": "Soualiho Meïté",
        "from_club_name": "Benfica",
        "to_club_name": "PAOK",
        "transfer_date": "2025-07-01",
    }
    item = candidate(
        source_url="https://paokfc.gr/news",
        source_type="official",
        source_title="PAOK signs Soualiho Meite",
        evidence_text="Soualiho Meite has joined PAOK from Benfica on a permanent transfer in 2025.",
    )
    item.event_match_status, item.event_match_score, item.event_match_reasons = assess_event_match(item, event)
    assert item.event_match_status == "exact"
    assert filter_admissible_sources([item]) == [item]


def test_reverse_transfer_direction_is_rejected():
    event = {
        "player_name": "Andrea Belotti",
        "from_club_name": "Benfica",
        "to_club_name": "Como",
        "transfer_date": "2025-07-01",
    }
    item = candidate(
        source_url="https://comofootball.com/news",
        source_type="official",
        source_title="Andrea Belotti joins Benfica",
        evidence_text="Andrea Belotti joined Benfica on loan from Como in 2024.",
    )
    item.event_match_status, _, reasons = assess_event_match(item, event)
    assert item.event_match_status == "mismatch"
    assert "reverse_direction" in reasons
    assert filter_admissible_sources([item]) == []


def test_receiving_club_official_announcement_can_be_likely_without_departing_club():
    event = {
        "player_name": "Danny Namaso",
        "from_club_name": "Porto",
        "to_club_name": "AJ Auxerre",
        "transfer_date": "2025-08-17",
    }
    item = candidate(
        source_url="https://www.aja.fr/danny-namaso-rejoint-laja",
        source_type="official",
        source_title="Actualités Danny Namaso rejoint l'AJA",
        evidence_text="Danny Namaso rejoint l'AJA pour la saison 2025.",
    )
    item.event_match_status, _, reasons = assess_event_match(item, event)
    assert item.event_match_status == "likely"
    assert "receiving_club_official_announcement" in reasons


def test_receiving_club_official_rule_still_rejects_reverse_direction():
    event = {
        "player_name": "Andrea Belotti",
        "from_club_name": "Benfica",
        "to_club_name": "Como",
        "transfer_date": "2025-06-30",
    }
    item = candidate(
        source_url="https://comofootball.com/en/andrea-belotti-loan-from-como-1907-to-benfica",
        source_type="official",
        source_title="Andrea Belotti loan from Como 1907 to Benfica",
        evidence_text="Andrea Belotti joined Benfica on loan from Como in 2024.",
    )
    status, _, reasons = assess_event_match(item, event)
    assert status == "mismatch"
    assert "reverse_direction" in reasons


def test_receiving_club_official_page_for_different_destination_is_rejected():
    event = {
        "player_name": "Renato Sanches",
        "from_club_name": "Benfica",
        "to_club_name": "PSG",
        "transfer_date": "2025-06-30",
    }
    item = candidate(
        source_url="https://www.psg.fr/en/content/renato-sanches-loaned-to-panathinaikos-fc",
        source_type="official",
        source_title="Renato Sanches loaned to Panathinaikos FC",
        evidence_text=(
            "Portuguese midfielder Renato Sanches joins Greek club Panathinaikos FC on loan. "
            "Last season, he returned on loan to his formative club Benfica. "
            "He moved to Paris Saint-Germain during the summer transfer window in 2022."
        ),
    )
    status, _, reasons = assess_event_match(item, event)
    assert status == "mismatch"
    assert "different_destination_announcement" in reasons


def test_later_permanent_transfer_does_not_establish_original_loan_terms():
    event = {
        "player_name": "Player",
        "from_club_name": "Benfica",
        "to_club_name": "Burnley",
        "transfer_date": "2024-08-01",
    }
    item = candidate(
        source_type="major_news",
        evidence_text="Player made a permanent transfer to Burnley from Benfica after a season-long loan.",
    )
    item.event_match_status, _, reasons = assess_event_match(item, event)
    assert item.event_match_status == "ambiguous"
    assert "loan_followed_by_permanent_transfer" in reasons
    assert filter_admissible_sources([item]) == []


def test_loan_return_article_does_not_establish_original_loan():
    event = {
        "player_name": "Player",
        "from_club_name": "Como",
        "to_club_name": "Benfica",
        "transfer_date": "2024-08-01",
    }
    item = candidate(
        source_type="major_news",
        evidence_text="Player returned to Como after his Benfica loan ended.",
    )
    status, _, reasons = assess_event_match(item, event)
    assert status == "mismatch"
    assert "loan_return_without_target_direction" in reasons


def test_official_source_ranks_above_secondary_reporting():
    event = {"player_name": "Player", "from_club_name": "Club A", "to_club_name": "Club B"}
    official = candidate(source_url="https://club-a.example.com/news", source_type="official")
    secondary = candidate(source_url="https://news.example.com/article", source_type="major_news")
    assert score_source(official, event) > score_source(secondary, event)


def test_official_club_domain_ranks_above_aggregator():
    event = {"player_name": "Player", "from_club_name": "Benfica", "to_club_name": "PAOK", "transfer_date": "2025-07-01"}
    official = candidate(
        source_url="https://slbenfica.pt/news/player",
        source_type="official",
        target_domain=official_domain_for_club("Benfica"),
        event_match_status="exact",
    )
    aggregator = candidate(
        source_url="https://example.com/player",
        source_type="aggregator",
        event_match_status="exact",
    )
    assert score_source(official, event) > score_source(aggregator, event)


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
    assert results[0].query_family
    assert results[0].event_match_status
    assert provider.search_count == 0


def test_tavily_missing_metadata_remains_null(monkeypatch):
    provider = TavilySearchProvider("test-key")
    provider._search = lambda query: {"results": [{"url": "https://example.com/result", "content": "snippet"}]}
    monkeypatch.setattr("src.source_discovery._check_accessibility", lambda url, timeout: True)
    result = provider.search_transfer({"player_name": "Player", "from_club_name": "A", "to_club_name": "B", "transfer_date": "2025-07-01"})[0]
    assert result.source_title is None
    assert result.publication_date is None
    assert result.language == "en"


def test_normal_discovery_performs_fullpage_before_sufficiency():
    class Provider:
        def search_transfer(self, event):
            return [candidate(
                source_url="https://www.fcporto.pt/pt/noticias/20250803-pt-luuk-de-jong-e-dragao",
                source_type="official",
                source_tier=1,
                evidence_text="Luuk de Jong é Dragão.",
                event_match_status="ambiguous",
                quality_score=80,
            )]

    class Fetcher:
        calls = 0

        def apply_to_candidate(self, item):
            self.calls += 1
            item.retrieved_text = "Luuk de Jong completed a transfer from PSV to FC Porto in 2025."
            item.retrieval_status = "success"

    fetcher = Fetcher()
    result = discover_transfer(
        {"player_name": "Luuk de Jong", "from_club_name": "PSV", "to_club_name": "Porto", "transfer_date": "2025-08-03"},
        Provider(),
        fullpage_fetcher=fetcher,
    )
    assert fetcher.calls == 1
    assert result.sufficient
    assert result.candidates[0].event_match_status == "exact"
    assert result.candidates[0].evidence_text == "Luuk de Jong é Dragão."


def test_fullpage_retrieval_skips_tier_three_by_default():
    class Fetcher:
        calls = 0

        def apply_to_candidate(self, item):
            self.calls += 1

    fetcher = Fetcher()
    items = [
        candidate(source_url="https://www.transfermarkt.com/player", source_type="aggregator", source_tier=3),
        candidate(source_url="https://www.fcporto.pt/news", source_type="official", source_tier=1),
    ]
    retrieve_fullpage_for_promising_sources(items, {"player_name": "Player", "from_club_name": "A", "to_club_name": "B"}, fetcher)
    assert fetcher.calls == 1


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


def test_tier_three_cannot_satisfy_exact_financial_terms():
    tm = candidate(
        source_url="https://www.transfermarkt.com/player",
        source_type="aggregator",
        quality_score=95,
        event_match_status="exact",
    )
    admissible = filter_admissible_sources([tm])
    sufficient, reasons = assess_source_sufficiency(admissible)
    assert admissible == []
    assert not sufficient
    assert reasons


def test_major_news_is_admissible_but_social_is_not():
    news = candidate(source_url="https://reuters.com/story", source_type="major_news")
    social = candidate(source_url="https://instagram.com/p/story", source_type="aggregator")
    assert [item.source_url for item in filter_admissible_sources([news, social])] == [news.source_url]
