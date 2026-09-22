"""Tavily adapter.

Kept working, but nothing downstream may depend on it. Constructing this class
without a key is fine; `available()` reports the problem and the pipeline skips
it. That is what lets the suite pass with TAVILY_API_KEY unset.
"""
from __future__ import annotations

import json
import os
import time
from urllib.request import Request, urlopen

from ..types import SearchQuery, SearchResult

ENDPOINT = "https://api.tavily.com/search"


class TavilyProvider:
    provider_name = "tavily"
    supports_raw_content = True
    supports_language = False
    supports_domain_filtering = True
    accepts = ("api_key", "depth", "timeout", "allow_network")

    def __init__(self, api_key: str | None = None, depth: str = "basic",
                 timeout: float = 45, allow_network: bool = True):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.depth, self.timeout = depth, timeout
        self.allow_network = allow_network
        self.queries_issued = 0

    def available(self) -> tuple[bool, str]:
        if not self.allow_network:
            return (False, "network disabled for this run")
        if not self.api_key:
            return (False, "TAVILY_API_KEY not set")
        return (True, f"tavily ({self.depth})")

    def search(self, query: SearchQuery, max_results: int = 10) -> list[SearchResult]:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"TavilyProvider unavailable: {reason}")
        body = {"api_key": self.api_key, "query": query.query, "search_depth": self.depth,
                "topic": "general", "max_results": max_results,
                "include_answer": False, "include_raw_content": True}
        if query.preferred_domains:
            body["include_domains"] = list(query.preferred_domains)
        if query.excluded_domains:
            body["exclude_domains"] = list(query.excluded_domains)
        req = Request(ENDPOINT, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        self.queries_issued += 1
        time.sleep(0.3)
        return self.normalize(payload, query)

    @staticmethod
    def normalize(payload: dict, query: SearchQuery | None = None) -> list[SearchResult]:
        """Read a Tavily payload - live or from the existing cache - as generic
        results. This is what keeps 510 cached Tavily files usable."""
        out = []
        for rank, item in enumerate(payload.get("results") or [], start=1):
            if not item.get("url"):
                continue
            out.append(SearchResult(
                url=item["url"], title=item.get("title"), snippet=item.get("content"),
                provider="tavily", rank=rank, raw_content=item.get("raw_content"),
                published_date=item.get("published_date"),
                provider_score=item.get("score"),
                query=(query.query if query else payload.get("query")),
                query_family=(query.query_family if query else None),
                language=(query.language if query else None),
                metadata={"tavily_id": item.get("id")}))
        return out
