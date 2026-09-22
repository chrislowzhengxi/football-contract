"""Generic web search, decoupled from any specific vendor.

The Python pipeline cannot call the host's browser/search tooling in-process,
so this provider reads from a handoff store: a directory of JSON files, one per
query, that ANY out-of-process capability can populate - an agent's web-search
tool, a browser automation step, a different search API, or a manual export.

That keeps the contract honest. The provider does not pretend to search; it
consumes results that something legitimate produced, and reports plainly when a
query has not been answered yet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..types import SearchQuery, SearchResult, canonical_url

HANDOFF_DIR = Path("data/outputs/contract_research/stage2_cache/generic_search")


class GenericWebSearchProvider:
    provider_name = "generic"
    supports_raw_content = False
    supports_language = True
    supports_domain_filtering = True
    accepts = ("handoff_dir",)

    def __init__(self, handoff_dir: Path = HANDOFF_DIR):
        self.dir = Path(handoff_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.misses: list[str] = []

    def available(self) -> tuple[bool, str]:
        n = len(list(self.dir.glob("*.json")))
        return (True, f"handoff store with {n} answered queries")

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:24]

    def path_for(self, query: str) -> Path:
        return self.dir / f"{self._key(query)}.json"

    def record(self, query: str, results: list[dict], source: str = "agent_web_search") -> Path:
        """Write results obtained elsewhere into the handoff store.

        `results` items need only `url`; `title` and `snippet` are used if given.
        """
        p = self.path_for(query)
        p.write_text(json.dumps(
            {"query": query, "source": source,
             "results": [{"url": canonical_url(r.get("url", "")),
                          "title": r.get("title"), "snippet": r.get("snippet")}
                         for r in results if r.get("url")]},
            indent=2, ensure_ascii=False))
        return p

    def search(self, query: SearchQuery, max_results: int = 10) -> list[SearchResult]:
        p = self.path_for(query.query)
        if not p.exists():
            self.misses.append(query.query)
            return []
        payload = json.loads(p.read_text())
        ex = set(query.excluded_domains)
        out = []
        for rank, item in enumerate(payload.get("results") or [], start=1):
            r = SearchResult(url=item["url"], title=item.get("title"),
                             snippet=item.get("snippet"), provider="generic", rank=rank,
                             query=query.query, query_family=query.query_family,
                             language=query.language,
                             metadata={"handoff_source": payload.get("source")})
            if r.domain in ex:
                continue
            out.append(r)
        return out[:max_results]

    def pending(self) -> list[str]:
        """Queries asked for but not yet answered - the work list for whatever
        external capability is wired in."""
        return list(dict.fromkeys(self.misses))
