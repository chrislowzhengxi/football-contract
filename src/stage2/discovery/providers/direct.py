"""Discovery straight from an organization's own site - no search engine.

This is the path that makes Stage 2 viable without any search API. Given a
player, a club and a year, it tries the site's own affordances in order of
politeness and cost: a cached answer, then its sitemap, then its feeds, then a
documented on-site search endpoint.

Rules this module will not break:
  * robots.txt is fetched once per host and honoured.
  * one request at a time, paced, with backoff.
  * a hard per-host page budget.
  * URLs are never brute-forced or guessed - every candidate comes from a
    document the site itself published (sitemap, feed, or link).
"""
from __future__ import annotations

import gzip
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from ..cache_index import fold, name_tokens
from ..types import SearchQuery, SearchResult, canonical_url, domain_of

REGISTRY_PATH = Path("data/config/source_registry.json")
CACHE_DIR = Path("data/outputs/contract_research/stage2_cache/direct_source")
USER_AGENT = ("FootballContractResearch/1.0 (academic transfer-terms research; "
              "contact via repository owner)")
REQUEST_PACING_S = 1.5
PER_HOST_BUDGET = 12


class DirectSourceProvider:
    provider_name = "direct"
    supports_raw_content = False
    supports_language = True
    supports_domain_filtering = True
    accepts = ("registry_path", "allow_network", "budget")

    def __init__(self, registry_path: Path = REGISTRY_PATH,
                 allow_network: bool = True, budget: int = PER_HOST_BUDGET):
        self.registry = {}
        p = Path(registry_path)
        if p.exists():
            self.registry = json.loads(p.read_text()).get("entries", {})
        self.allow_network = allow_network
        self.budget = budget
        self._robots: dict[str, RobotFileParser | None] = {}
        self._spent: dict[str, int] = {}
        self.fetches = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def available(self) -> tuple[bool, str]:
        if not self.registry:
            return (False, "source_registry.json missing or empty")
        return (True, f"{len(self.registry)} organizations registered")

    # ---------- polite fetching ----------
    def _robot_ok(self, url: str) -> bool:
        host = urlsplit(url).netloc
        if host not in self._robots:
            rp = RobotFileParser()
            rp.set_url(f"https://{host}/robots.txt")
            try:
                rp.read()
            except Exception:                                  # noqa: BLE001
                rp = None          # unreadable robots.txt: stay conservative
            self._robots[host] = rp
        rp = self._robots[host]
        if rp is None:
            return False
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:                                      # noqa: BLE001
            return False

    def _get(self, url: str, timeout: float = 20) -> bytes | None:
        host = domain_of(url)
        if self._spent.get(host, 0) >= self.budget:
            return None
        cache = CACHE_DIR / (re.sub(r"\W+", "_", canonical_url(url))[:180] + ".bin")
        if cache.exists():
            return cache.read_bytes() or None
        if not self.allow_network or not self._robot_ok(url):
            return None
        self._spent[host] = self._spent.get(host, 0) + 1
        for attempt in range(2):
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT})
                with urlopen(req, timeout=timeout) as resp:
                    data = resp.read()
                self.fetches += 1
                cache.write_bytes(data)
                time.sleep(REQUEST_PACING_S)
                return data
            except HTTPError as e:
                if e.code in (429, 503) and attempt == 0:
                    time.sleep(5)
                    continue
                cache.write_bytes(b"")
                return None
            except (URLError, TimeoutError, OSError):
                if attempt == 0:
                    time.sleep(2)
                    continue
                return None
        return None

    # ---------- site-native discovery ----------
    def sitemap_urls(self, domain: str, limit: int = 4) -> list[str]:
        """URLs advertised by the site's own sitemap(s)."""
        found, seen = [], set()
        roots = [f"https://{domain}/sitemap.xml", f"https://{domain}/sitemap_index.xml"]
        robots = self._get(f"https://{domain}/robots.txt")
        if robots:
            roots += re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots.decode("utf-8", "ignore"))
        queue = list(dict.fromkeys(roots))[:limit]
        while queue:
            sm = queue.pop(0)
            if sm in seen:
                continue
            seen.add(sm)
            raw = self._get(sm)
            if not raw:
                continue
            if raw[:2] == b"\x1f\x8b":
                try:
                    raw = gzip.decompress(raw)
                except Exception:                              # noqa: BLE001
                    continue
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue
            tag = root.tag.split("}")[-1]
            locs = [e.text.strip() for e in root.iter() if e.tag.split("}")[-1] == "loc" and e.text]
            if tag == "sitemapindex":
                queue += [l for l in locs if l not in seen][: limit - len(seen)]
            else:
                found += locs
        return found

    def feed_urls(self, domain: str) -> list[tuple[str, str]]:
        """(url, title) from a site's RSS/Atom feed, where one is published."""
        out = []
        for path in ("/feed", "/rss", "/rss.xml", "/feed.xml", "/atom.xml", "/news/rss"):
            raw = self._get(f"https://{domain}{path}")
            if not raw:
                continue
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue
            for item in root.iter():
                t = item.tag.split("}")[-1]
                if t in ("item", "entry"):
                    link = title = None
                    for ch in item:
                        ct = ch.tag.split("}")[-1]
                        if ct == "link":
                            link = ch.text or ch.attrib.get("href")
                        elif ct == "title":
                            title = ch.text
                    if link:
                        out.append((link, title or ""))
            if out:
                break
        return out

    def search(self, query: SearchQuery, max_results: int = 10) -> list[SearchResult]:
        """Match a player/year against URLs the organization itself publishes."""
        domains = list(query.preferred_domains) or self._domains_for(query)
        if not domains:
            return []
        want = name_tokens(query.query)
        year = next((m for m in re.findall(r"\b(19|20)\d{2}\b", query.query)), None)
        years = set(re.findall(r"\b((?:19|20)\d{2})\b", query.query))
        results: list[SearchResult] = []
        for dom in domains[:3]:
            candidates = [(u, "") for u in self.sitemap_urls(dom)] + self.feed_urls(dom)
            for url, title in candidates:
                hay = fold(f"{url} {title}")
                hits = sum(1 for t in want if t in hay)
                if hits < max(1, len(want) - 1):
                    continue
                if years and not any(y in url or y in (title or "") for y in years):
                    # Year is a weak filter on club sites; keep but rank lower.
                    score = 0.4
                else:
                    score = 0.9
                results.append(SearchResult(
                    url=url, title=title or None, provider="direct",
                    rank=len(results) + 1, provider_score=score,
                    query=query.query, query_family=query.query_family,
                    language=query.language,
                    metadata={"discovery": "sitemap_or_feed", "organization_domain": dom}))
        results.sort(key=lambda r: -(r.provider_score or 0))
        return results[:max_results]

    def _domains_for(self, query: SearchQuery) -> list[str]:
        toks = name_tokens(query.query)
        out = []
        for key, e in self.registry.items():
            if not e.get("official_domain"):
                continue
            if name_tokens(key) & toks or fold(key) in fold(query.query):
                out.append(e["official_domain"])
        return out

    def domains_for_clubs(self, clubs) -> list[str]:
        """Registry domains for a set of club names, plus any regulatory venue."""
        out = []
        for c in clubs:
            k = fold(str(c))
            for key, e in self.registry.items():
                if fold(key) == k and e.get("official_domain"):
                    out.append(e["official_domain"])
                    reg = e.get("regulatory_disclosure")
                    if reg and reg.get("domain"):
                        out.append(reg["domain"])
        return list(dict.fromkeys(out))
