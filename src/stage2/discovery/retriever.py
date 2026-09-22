"""Retrieval, independent of how a URL was discovered.

A URL from the cache, a club sitemap, a search API or a citation link is
treated identically here. The one thing this layer refuses to do is conflate a
search engine's rendering of a page with a page we fetched and parsed
ourselves: both are preserved, separately, on the Evidence object.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..pdf_text import extract_pdf_text, looks_like_pdf
from ..sources import classify, tier_of
from .types import Evidence, SearchResult, canonical_url, domain_of

CACHE_ROOT = Path("data/outputs/contract_research/stage2_cache")
PAGE_DIRS = ("page", "deep_page", "generic_page")
WRITE_DIR = CACHE_ROOT / "generic_page"
USER_AGENT = ("FootballContractResearch/1.0 (academic transfer-terms research; "
              "contact via repository owner)")


class PageRetriever:
    """Cache-first, then optional polite HTTP. Never raises on a bad page."""

    def __init__(self, allow_network: bool = False, pacing: float = 1.5,
                 index=None):
        self.allow_network = allow_network
        self.pacing = pacing
        self.index = index
        self._disk: dict[str, tuple[str, str]] | None = None
        self.stats = {"cache_hits": 0, "network_fetches": 0, "pdf_parsed": 0,
                      "provider_raw_used": 0, "failures": 0, "skipped_no_network": 0}
        WRITE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- cache ----------
    def _disk_cache(self) -> dict:
        if self._disk is None:
            self._disk = {}
            for sub in PAGE_DIRS:
                d = CACHE_ROOT / sub
                if not d.is_dir():
                    continue
                for f in d.glob("*.json"):
                    try:
                        p = json.loads(f.read_text())
                    except Exception:                          # noqa: BLE001
                        continue
                    u = canonical_url(p.get("url") or "")
                    t = p.get("text") or ""
                    if u and (u not in self._disk or len(t) > len(self._disk[u][0])):
                        self._disk[u] = (t, p.get("status") or "cache")
        return self._disk

    # ---------- main ----------
    def retrieve(self, result: SearchResult, event_id: str | None = None,
                 event_family_id: str | None = None,
                 from_club: str | None = None, to_club: str | None = None) -> Evidence:
        url = canonical_url(result.url)
        cls = classify(url, from_club, to_club)
        ev = Evidence(
            source_url=url, discovery_provider=result.provider,
            discovery_query=result.query, search_rank=result.rank,
            raw_search_content=result.raw_content,
            title=result.title, published_date=result.published_date,
            source_class=cls, source_tier=tier_of(cls),
            event_id=event_id, event_family_id=event_family_id,
            metadata={"snippet": result.snippet,
                      "found_by": result.metadata.get("found_by", []),
                      "providers": result.metadata.get("providers", [result.provider])})

        hit = self._disk_cache().get(url)
        if hit and hit[0]:
            ev.retrieved_page_text, ev.retrieval_status = hit[0], hit[1]
            ev.retrieval_method = "cache"
            self.stats["cache_hits"] += 1
        elif self.allow_network:
            text, status, method = self._fetch(url)
            ev.retrieved_page_text, ev.retrieval_status, ev.retrieval_method = text, status, method
            if not text:
                self.stats["failures"] += 1
        else:
            ev.retrieval_status = "not_fetched_offline"
            self.stats["skipped_no_network"] += 1

        # Provider-rendered text is a fallback, and the Evidence records which
        # one the span came from via `text_origin`.
        if not ev.retrieved_page_text and ev.raw_search_content:
            self.stats["provider_raw_used"] += 1
            if ev.retrieval_method == "none":
                ev.retrieval_method = "provider_raw"
        ev.content_type = "pdf" if ev.retrieval_status.startswith("pdf") else "html"
        return ev

    def _fetch(self, url: str) -> tuple[str | None, str, str]:
        out = WRITE_DIR / (re.sub(r"\W+", "_", url)[:180] + ".json")
        if out.exists():
            try:
                p = json.loads(out.read_text())
                return (p.get("text") or None), p.get("status", "cache"), "cache"
            except Exception:                                  # noqa: BLE001
                pass
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=25) as resp:
                blob = resp.read()
            self.stats["network_fetches"] += 1
            time.sleep(self.pacing)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            return None, f"fetch_failed:{type(exc).__name__}", "http"
        if looks_like_pdf(blob):
            text = extract_pdf_text(blob, max_pages=80)
            status, method = ("pdf_ok" if text else "pdf_unreadable"), "pdf"
            if text:
                self.stats["pdf_parsed"] += 1
        else:
            try:
                from ...official_page_retrieval import extract_visible_text
                text, status, method = extract_visible_text(
                    blob.decode("utf-8", "ignore")), "html_ok", "http"
            except Exception:                                  # noqa: BLE001
                text, status, method = "", "html_parse_failed", "http"
        out.write_text(json.dumps({"url": url, "text": text, "status": status},
                                  ensure_ascii=False))
        return (text or None), status, method

    def retrieve_all(self, results, **kw) -> list[Evidence]:
        seen, out = set(), []
        for r in results:
            u = canonical_url(r.url)
            if u in seen:
                continue
            seen.add(u)
            out.append(self.retrieve(r, **kw))
        return out
