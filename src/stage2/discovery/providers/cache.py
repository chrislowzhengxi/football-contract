"""Discovery from evidence we already hold. Always tried first: it is free,
instant, and the project holds ~1,800 readable documents that were previously
unqueryable by player or event."""
from __future__ import annotations

from ..cache_index import CachedEvidenceIndex
from ..types import SearchQuery, SearchResult


class CacheProvider:
    provider_name = "cache"
    supports_raw_content = True
    supports_language = False
    supports_domain_filtering = True
    accepts = ("index",)

    def __init__(self, index: CachedEvidenceIndex | None = None):
        self._index = index

    @property
    def index(self) -> CachedEvidenceIndex:
        if self._index is None:
            self._index = CachedEvidenceIndex().build()
        return self._index

    def available(self) -> tuple[bool, str]:
        try:
            return (True, f"{len(self.index.docs)} cached documents")
        except Exception as exc:                               # noqa: BLE001
            return (False, f"cache unreadable: {exc}")

    def search(self, query: SearchQuery, max_results: int = 10) -> list[SearchResult]:
        """Token lookup over the index. The `query` string is treated as free
        text; club and player tokens in it drive the match."""
        docs = self.index.lookup(query.query, min_score=2)
        if query.preferred_domains:
            pref = set(query.preferred_domains)
            docs.sort(key=lambda d: (d.domain not in pref,))
        if query.excluded_domains:
            ex = set(query.excluded_domains)
            docs = [d for d in docs if d.domain not in ex]
        out = self.index.as_results(docs[:max_results], query.query)
        for r in out:
            r.query_family = query.query_family
            r.language = query.language
        return out

    def search_family(self, family_rows, max_results: int = 25) -> list[SearchResult]:
        """Preferred entry point: match on every club in the family, not one string."""
        docs = self.index.for_family(family_rows)
        return self.index.as_results(docs[:max_results])
