from urllib.error import URLError

import pytest

from src.official_page_retrieval import OfficialPageFetcher, extract_visible_text
from src.source_discovery import SourceCandidate, assess_event_match, source_tier


class FakeHeaders:
    def get_content_charset(self):
        return "utf-8"


class FakeResponse:
    headers = FakeHeaders()
    status = 200

    def __init__(self, body: str, final_url: str = "https://example.com/final"):
        self.body = body.encode()
        self.final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, limit):
        return self.body[:limit]

    def geturl(self):
        return self.final_url


def candidate(**overrides):
    values = {
        "source_url": "https://www.fcporto.pt/pt/noticias/20250803-pt-luuk-de-jong-e-dragao",
        "source_title": "FC Porto - Notícias - Luuk de Jong é Dragão",
        "publisher": "fcporto.pt",
        "publication_date": "2025-08-03",
        "source_type": "official",
        "evidence_text": "Luuk de Jong é Dragão.",
        "language": "pt",
        "source_tier": 1,
        "event_match_status": "ambiguous",
    }
    values.update(overrides)
    return SourceCandidate(**values)


def test_extract_visible_text_removes_script_and_preserves_article_text():
    text = extract_visible_text("<html><script>bad()</script><article><h1>Title</h1><p>Luuk de Jong joins Porto.</p></article></html>")
    assert "bad" not in text
    assert "Luuk de Jong joins Porto" in text


def test_snippet_ambiguous_but_retrieved_body_exact_receiving_club():
    event = {"player_name": "Luuk de Jong", "from_club_name": "PSV", "to_club_name": "Porto", "transfer_date": "2025-08-03"}
    item = candidate(evidence_text="Luuk de Jong é Dragão.")
    assert assess_event_match(item, event)[0] == "ambiguous"
    item.retrieved_text = "On 3 August 2025 Luuk de Jong completed a transfer from PSV to FC Porto."
    assert assess_event_match(item, event)[0] == "exact"


def test_departing_club_full_body_resolves_direction():
    event = {"player_name": "Luuk de Jong", "from_club_name": "PSV", "to_club_name": "Porto", "transfer_date": "2025-08-03"}
    item = candidate(
        source_url="https://www.psv.nl/en/media/article/transfer-luuk-de-jong-leaves-to-fc-porto",
        publisher="psv.nl",
        evidence_text="Transfer | Luuk de Jong leaves for FC Porto",
        retrieved_text="PSV confirms Luuk de Jong leaves from PSV to FC Porto in 2025.",
    )
    assert assess_event_match(item, event)[0] == "exact"


def test_reverse_transfer_remains_mismatch_with_full_body():
    event = {"player_name": "Andrea Belotti", "from_club_name": "Benfica", "to_club_name": "Como", "transfer_date": "2025-06-30"}
    item = candidate(
        source_url="https://comofootball.com/en/andrea-belotti-loan-from-como-1907-to-benfica",
        publisher="comofootball.com",
        source_title="Andrea Belotti loan from Como 1907 to Benfica",
        evidence_text="Andrea Belotti joined Benfica on loan from Como.",
        retrieved_text="Como announces Andrea Belotti has joined Benfica from Como on loan.",
    )
    assert assess_event_match(item, event)[0] == "mismatch"


def test_later_permanent_transfer_does_not_establish_earlier_loan_with_full_body():
    event = {"player_name": "Player", "from_club_name": "Benfica", "to_club_name": "Burnley", "transfer_date": "2024-08-01"}
    item = candidate(retrieved_text="Player made a permanent transfer to Burnley from Benfica after a season-long loan.")
    status, _, reasons = assess_event_match(item, event)
    assert status == "ambiguous"
    assert "loan_followed_by_permanent_transfer" in reasons


def test_wrong_year_article_remains_ambiguous():
    event = {"player_name": "Player", "from_club_name": "PSV", "to_club_name": "Porto", "transfer_date": "2025-08-03"}
    item = candidate(retrieved_text="Player joined Porto from PSV in 2021.")
    assert assess_event_match(item, event)[0] == "ambiguous"


def test_club_aliases_work_with_retrieved_body():
    event = {"player_name": "Danny Namaso", "from_club_name": "Porto", "to_club_name": "AJ Auxerre", "transfer_date": "2025-08-17"}
    item = candidate(
        source_url="https://www.aja.fr/danny-namaso-rejoint-laja",
        publisher="aja.fr",
        source_title="Danny Namaso rejoint l'AJA",
        evidence_text="Danny Namaso rejoint l'AJA pour la saison 2025.",
    )
    assert assess_event_match(item, event)[0] == "likely"


def test_cache_prevents_redundant_network_retrieval(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        return FakeResponse("<article>Luuk de Jong joins Porto from PSV.</article>", "https://example.com/final")

    monkeypatch.setattr("src.official_page_retrieval.urlopen", fake_urlopen)
    fetcher = OfficialPageFetcher(cache_dir=tmp_path, retry_delay=0)
    first = fetcher.fetch("https://example.com/page")
    second = fetcher.fetch("https://example.com/page")
    assert first.retrieval_status == "success"
    assert second.from_cache
    assert calls["count"] == 1


def test_failed_retrieval_is_cached_safely(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout):
        raise URLError("nope")

    monkeypatch.setattr("src.official_page_retrieval.urlopen", fake_urlopen)
    fetcher = OfficialPageFetcher(cache_dir=tmp_path, retry_delay=0)
    result = fetcher.fetch("https://example.com/missing")
    cached = fetcher.fetch("https://example.com/missing")
    assert result.retrieval_status == "failed"
    assert cached.from_cache


def test_retrieval_does_not_change_tier_three_source():
    item = candidate(source_url="https://www.transfermarkt.com/player", source_type="aggregator", source_tier=3)
    item.retrieved_text = "Official-looking transfer text."
    assert source_tier(item) == 3


def test_retrieval_preserves_original_snippet_and_does_not_fabricate_terms(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout):
        return FakeResponse("<article>Player joined Porto from PSV.</article>")

    monkeypatch.setattr("src.official_page_retrieval.urlopen", fake_urlopen)
    item = candidate(evidence_text="Original Tavily snippet")
    fetcher = OfficialPageFetcher(cache_dir=tmp_path)
    fetcher.apply_to_candidate(item)
    assert item.evidence_text == "Original Tavily snippet"
    assert "fee" not in item.retrieved_text.lower()
