from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass
class SourceCandidate:
    source_url: str
    source_title: str | None
    publisher: str | None
    publication_date: str | None
    source_type: str
    evidence_text: str
    language: str
    evidence_id: str | None = None
    reported_values: list[dict[str, Any]] | None = None
    quality_score: float = 0.0
    accessible: bool = True
    search_query: str | None = None
    search_rank: int | None = None
    provider_score: float | None = None
    source_tier: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchProvider(Protocol):
    def search_transfer(self, event: dict[str, Any]) -> list[SourceCandidate]:
        ...


@dataclass
class SourceDiscoveryResult:
    candidates: list[SourceCandidate]
    sufficient: bool
    reasons: list[str]


REPUTABLE_DOMAINS = {
    "reuters.com", "bbc.com", "espn.com", "theathletic.com", "nytimes.com",
    "apnews.com", "theguardian.com", "skysports.com", "goal.com",
}
SPECIALIST_DOMAINS = {
    "football-italia.net", "maisfutebol.iol.pt", "ojogo.pt", "record.pt",
    "a-bola.pt", "lequipe.fr", "marca.com", "as.com",
}


def build_search_queries(event: dict[str, Any]) -> list[str]:
    player = event.get("player_name", "")
    from_club = event.get("from_club_name", "")
    to_club = event.get("to_club_name", "")
    date = event.get("transfer_date", "")
    identity = f'"{player}" "{from_club}" "{to_club}"'
    return [
        f"{identity} {date}",
        f"{identity} transfer fee",
        f"{identity} loan option to buy",
        f"{identity} obligation to buy",
        f"{identity} official announcement",
        f"{identity} add-ons bonuses",
        f"{identity} sell-on clause",
    ]


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _source_type(url: str, event: dict[str, Any]) -> str:
    domain = _domain(url)
    if domain in REPUTABLE_DOMAINS:
        return "major_news"
    if domain in SPECIALIST_DOMAINS:
        return "football_reporting"
    club_tokens = re.findall(r"[a-z0-9]+", " ".join(str(event.get(key, "")) for key in ("from_club_name", "to_club_name")).lower())
    if any(token and token in domain for token in club_tokens if len(token) > 3):
        return "official"
    if any(token in domain for token in ("sec.gov", "cmvm", "stockexchange", "regulator")):
        return "regulatory"
    return "aggregator"


def source_tier(candidate: SourceCandidate) -> int:
    domain = _domain(candidate.source_url)
    if domain in {"transfermarkt.com", "transfermarkt.us", "transfermarkt.co.uk", "wikipedia.org", "instagram.com", "facebook.com", "reddit.com"}:
        return 3
    if candidate.source_type in {"official", "regulatory", "governing_body"}:
        return 1
    if candidate.source_type in {"major_news", "football_reporting"}:
        return 2
    return 3


def filter_admissible_sources(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    """Keep full discovery provenance separate from contractual evidence input."""
    admissible = []
    for candidate in candidates:
        candidate.source_tier = source_tier(candidate)
        if candidate.source_tier <= 2 and candidate.accessible:
            admissible.append(candidate)
    return admissible


def score_source(candidate: SourceCandidate, event: dict[str, Any]) -> float:
    score = {
        "official": 60,
        "regulatory": 65,
        "major_news": 50,
        "football_reporting": 35,
        "aggregator": 15,
    }.get(candidate.source_type, 10)
    haystack = f"{candidate.source_title or ''} {candidate.evidence_text}".lower()
    identity = " ".join(str(event.get(key, "")) for key in ("player_name", "from_club_name", "to_club_name")).lower()
    if event.get("player_name", "").lower() in haystack:
        score += 12
    if any(term in haystack for term in ("€", "£", "$", "fee", "loan", "option", "obligation", "bonus", "add-on")):
        score += 10
    if "said" in haystack or "statement" in haystack or "announced" in haystack:
        score += 5
    if candidate.publication_date and str(event.get("transfer_date", ""))[:4] in candidate.publication_date:
        score += 3
    if identity and all(token in haystack for token in identity.split() if len(token) > 3):
        score += 5
    if not candidate.accessible:
        score -= 50
    return max(0.0, min(100.0, score))


def deduplicate_sources(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    retained: list[SourceCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.quality_score, reverse=True):
        normalized_url = candidate.source_url.split("#", 1)[0].rstrip("/").lower()
        duplicate = False
        for existing in retained:
            existing_url = existing.source_url.split("#", 1)[0].rstrip("/").lower()
            title_similarity = SequenceMatcher(None, (candidate.source_title or "").lower(), (existing.source_title or "").lower()).ratio()
            if normalized_url == existing_url or (title_similarity >= 0.92 and candidate.publisher == existing.publisher):
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    for index, candidate in enumerate(retained, start=1):
        candidate.evidence_id = candidate.evidence_id or f"s{index}"
    return retained


def assess_source_sufficiency(candidates: list[SourceCandidate]) -> tuple[bool, list[str]]:
    usable = [candidate for candidate in candidates if candidate.accessible]
    strong = [candidate for candidate in usable if candidate.quality_score >= 70]
    independent_domains = {_domain(candidate.source_url) for candidate in usable if candidate.quality_score >= 50}
    if strong:
        return True, ["at least one strong accessible source"]
    if len(independent_domains) >= 2:
        return True, ["two independent reputable sources"]
    return False, ["fewer than one strong or two independent reputable accessible sources"]


class BraveSearchProvider:
    """Search the Brave Web Search API; it never extracts contract claims."""

    def __init__(self, api_key: str, timeout: float = 20, max_results_per_query: int = 5):
        self.api_key = api_key
        self.timeout = timeout
        self.max_results_per_query = max_results_per_query

    @classmethod
    def from_environment(cls) -> "BraveSearchProvider":
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is required for Brave source discovery")
        return cls(api_key)

    def search_transfer(self, event: dict[str, Any]) -> list[SourceCandidate]:
        candidates: list[SourceCandidate] = []
        for query in build_search_queries(event):
            request = Request(
                "https://api.search.brave.com/res/v1/web/search?" + urlencode({"q": query, "count": self.max_results_per_query}),
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key, "User-Agent": "football-contract-source-discovery/1.0"},
            )
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
            for rank, item in enumerate(payload.get("web", {}).get("results", []), start=1):
                url = item.get("url")
                if not url:
                    continue
                candidate = SourceCandidate(
                    source_url=url,
                    source_title=item.get("title"),
                    publisher=_domain(url),
                    publication_date=item.get("age") or item.get("page_age"),
                    source_type=_source_type(url, event),
                    evidence_text=item.get("description") or "",
                    language=item.get("language") or "en",
                    search_query=query,
                    search_rank=rank,
                )
                candidate.accessible = _check_accessibility(url, self.timeout)
                candidate.quality_score = score_source(candidate, event)
                candidates.append(candidate)
        return deduplicate_sources(candidates)


class TavilySearchProvider:
    """Search Tavily's API and normalize results without extracting claims."""

    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 20, max_results_per_query: int = 5):
        self.api_key = api_key
        self.timeout = timeout
        self.max_results_per_query = max_results_per_query
        self.search_count = 0
        self.raw_result_count = 0

    @classmethod
    def from_environment(cls) -> "TavilySearchProvider":
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is required for Tavily source discovery")
        return cls(api_key)

    def search_transfer(self, event: dict[str, Any]) -> list[SourceCandidate]:
        candidates: list[SourceCandidate] = []
        for query in build_tavily_queries(event):
            payload = self._search(query)
            for rank, item in enumerate(payload.get("results", []), start=1):
                url = item.get("url")
                if not url:
                    continue
                candidate = SourceCandidate(
                    source_url=url,
                    source_title=item.get("title"),
                    publisher=_domain(url),
                    publication_date=item.get("published_date"),
                    source_type=_source_type(url, event),
                    evidence_text=item.get("content") or "",
                    language=item.get("language") or "en",
                    search_query=query,
                    search_rank=rank,
                    provider_score=item.get("score"),
                )
                candidate.accessible = _check_accessibility(url, self.timeout)
                candidate.quality_score = score_source(candidate, event)
                candidates.append(candidate)
        return deduplicate_sources(candidates)

    def _search(self, query: str) -> dict[str, Any]:
        body = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": self.max_results_per_query,
            "include_answer": False,
            "include_raw_content": False,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers={"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "football-contract-source-discovery/1.0"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                self.search_count += 1
                payload = json.loads(response.read())
        except HTTPError as error:
            if error.code in {401, 403}:
                raise RuntimeError("Tavily authentication failed") from error
            raise RuntimeError(f"Tavily search failed with HTTP {error.code}") from error
        self.raw_result_count += len(payload.get("results", []))
        return payload


def build_tavily_queries(event: dict[str, Any]) -> list[str]:
    player = event.get("player_name", "")
    from_club = event.get("from_club_name", "")
    to_club = event.get("to_club_name", "")
    date = event.get("transfer_date", "")
    identity = f'"{player}" "{from_club}" "{to_club}"'
    return [
        f"{identity} transfer {date}",
        f"{identity} transfer fee add-ons",
        f"{identity} loan option obligation to buy",
        f"{identity} official announcement",
    ]


def _check_accessibility(url: str, timeout: float) -> bool:
    try:
        request = Request(url, headers={"User-Agent": "football-contract-source-discovery/1.0"}, method="HEAD")
        with urlopen(request, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 400
    except Exception:
        return False


def discover_transfer(event: dict[str, Any], provider: SearchProvider) -> SourceDiscoveryResult:
    candidates = provider.search_transfer(event)
    sufficient, reasons = assess_source_sufficiency(candidates)
    return SourceDiscoveryResult(candidates, sufficient, reasons)


def load_source_candidates(path: Path) -> list[SourceCandidate]:
    payload = json.loads(path.read_text())
    records = payload.get("sources", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("source file must contain a JSON list or a {\"sources\": [...]} object")
    candidates = [SourceCandidate(**record) for record in records]
    for index, candidate in enumerate(candidates, start=1):
        if not candidate.evidence_id:
            candidate.evidence_id = f"s{index}"
    return candidates
