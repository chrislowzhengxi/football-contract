"""Provider-independent discovery types.

Nothing downstream of discovery may know which provider produced a URL. The
Stage 2 audit found that search-engine text and directly-retrieved page text
were being conflated, so a span could not be traced to how it was obtained.
`SearchResult` therefore keeps `snippet` and `raw_content` separate, and
`Evidence` keeps both apart from `retrieved_page_text`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, unquote, quote

# Tracking parameters that change the URL string but not the document.
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref_", "igshid")


def canonical_url(url: str) -> str:
    """Stable identity for a page, so the same document found by two providers
    deduplicates to one retrieval."""
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    scheme = "https" if parts.scheme in ("http", "https", "") else parts.scheme
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not any(k.lower().startswith(p) for p in _TRACKING_PREFIXES)])
    # Percent-encoding is not identity: /wiki/Alexander_S%C3%B8rloth and
    # /wiki/Alexander_Sørloth are one document, and treating them as two
    # inflated the cache index's unique-URL count.
    path = unquote(parts.path or "/")
    path = quote(path, safe="/:@!$&'()*+,;=~-._")
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, host, path, query, ""))


def domain_of(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


@dataclass(frozen=True)
class SearchQuery:
    """A request for discovery. Carries no provider-specific fields."""
    query: str
    language: str = "en"
    query_family: str = "general"
    event_id: str | None = None
    event_family_id: str | None = None
    preferred_domains: tuple = ()
    excluded_domains: tuple = ()
    priority: int = 1
    rationale: str = ""

    def key(self) -> str:
        base = f"{self.query}|{self.language}|{sorted(self.preferred_domains)}"
        return hashlib.sha256(base.encode()).hexdigest()[:24]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["preferred_domains"] = list(self.preferred_domains)
        d["excluded_domains"] = list(self.excluded_domains)
        return d


@dataclass
class SearchResult:
    """One discovered URL, normalised across providers."""
    url: str
    title: str | None = None
    snippet: str | None = None            # provider's own summary text
    provider: str = "unknown"
    rank: int = 0
    raw_content: str | None = None        # provider-rendered page text, if any
    published_date: str | None = None
    provider_score: float | None = None   # 0..1 where the provider supplies one
    query: str | None = None
    query_family: str | None = None
    language: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.url = canonical_url(self.url)

    @property
    def domain(self) -> str:
        return domain_of(self.url)

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class SearchProvider(Protocol):
    """Every provider is interchangeable from the caller's point of view."""
    provider_name: str
    supports_raw_content: bool
    supports_language: bool
    supports_domain_filtering: bool

    def search(self, query: SearchQuery, max_results: int = 10) -> list[SearchResult]:
        ...

    def available(self) -> tuple[bool, str]:
        """(usable now, human-readable reason). Never raises; a provider that
        cannot run is skipped rather than aborting the pipeline."""
        ...


@dataclass
class Evidence:
    """A retrieved document, with discovery and retrieval provenance kept apart.

    `raw_search_content` is what a search provider handed us. `retrieved_page_text`
    is what we fetched and parsed ourselves. Conflating them was a real defect in
    the diagnostic pipeline: it made spans untraceable to their acquisition path.
    """
    source_url: str
    discovery_provider: str = "unknown"
    discovery_query: str | None = None
    search_rank: int = 0
    raw_search_content: str | None = None
    retrieved_page_text: str | None = None
    retrieval_method: str = "none"        # cache | http | pdf | provider_raw | none
    retrieval_status: str = "not_attempted"
    title: str | None = None
    published_date: str | None = None
    identity_strength: str = "ungated"
    source_class: str = "unknown"
    source_tier: int = 3
    content_type: str = "html"
    event_id: str | None = None
    event_family_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return domain_of(self.source_url)

    @property
    def best_text(self) -> str:
        """Prefer text we retrieved and parsed; fall back to provider rendering."""
        return self.retrieved_page_text or self.raw_search_content or ""

    @property
    def text_origin(self) -> str:
        if self.retrieved_page_text:
            return "retrieved_page_text"
        if self.raw_search_content:
            return "raw_search_content"
        return "none"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["domain"] = self.domain
        d["text_origin"] = self.text_origin
        d["n_chars"] = len(self.best_text)
        return d


def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """Collapse by canonical URL, keeping the richest copy and remembering every
    query family that surfaced it - cross-family agreement is a ranking signal."""
    by_url: dict[str, SearchResult] = {}
    for r in results:
        if not r.url:
            continue
        cur = by_url.get(r.url)
        if cur is None:
            r.metadata.setdefault("found_by", [])
            if r.query_family and r.query_family not in r.metadata["found_by"]:
                r.metadata["found_by"].append(r.query_family)
            r.metadata.setdefault("providers", [r.provider])
            by_url[r.url] = r
            continue
        if r.query_family and r.query_family not in cur.metadata.setdefault("found_by", []):
            cur.metadata["found_by"].append(r.query_family)
        if r.provider not in cur.metadata.setdefault("providers", []):
            cur.metadata["providers"].append(r.provider)
        if not cur.raw_content and r.raw_content:
            cur.raw_content = r.raw_content
        if not cur.snippet and r.snippet:
            cur.snippet = r.snippet
        if not cur.title and r.title:
            cur.title = r.title
        if (r.provider_score or 0) > (cur.provider_score or 0):
            cur.provider_score = r.provider_score
        cur.rank = min(cur.rank or 10**6, r.rank or 10**6)
    return list(by_url.values())


def rank_results(results: list[SearchResult], tier_of=None) -> list[SearchResult]:
    """Provider-agnostic ordering: cross-family agreement, then source tier,
    then the provider's own score, then rank. `provider_score` is optional, so a
    provider that supplies none is not penalised relative to one that does."""
    def key(r: SearchResult):
        fams = len(r.metadata.get("found_by", []) or [])
        tier = tier_of(r.url) if tier_of else 3
        return (-fams, tier, -(r.provider_score or 0.0), r.rank or 10**6)
    return sorted(results, key=key)
