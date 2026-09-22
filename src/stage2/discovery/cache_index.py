"""Cache-first discovery.

Before paying any search provider we must know what we already hold. The
project has accumulated 171 deep searches (~48MB with provider-rendered page
text), 339 basic searches, 486 retrieved pages and 20 PDFs. None of it was
queryable by player or event before this index existed.

The index is built purely from files on disk. It makes no network calls.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .types import SearchResult, canonical_url, domain_of

CACHE_ROOT = Path("data/outputs/contract_research/stage2_cache")
SEARCH_DIRS = {"search": "tavily_basic", "deep_search": "tavily_deep"}
# generic_page holds documents retrieved through the host's own web
# capability. Omitting it meant generic-search evidence was written to
# disk but never became discoverable.
PAGE_DIRS = ("page", "deep_page", "generic_page")


def fold(s: str) -> str:
    """Accent- and case-insensitive text key. 'Sørloth' and 'Sorloth' must match,
    because sources spell players inconsistently and a miss here silently drops
    real evidence."""
    if not s:
        return ""
    s = s.replace("ø", "o").replace("Ø", "O").replace("ð", "d").replace("þ", "th")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


# Name particles that are part of a surname but useless as a match key.
# A token in more than this share of the corpus is treated as an ordinary word
# rather than a distinctive name.
COMMON_TOKEN_DF = 0.02

PARTICLES = {"de", "del", "della", "di", "da", "dos", "das", "van", "von", "der",
             "den", "la", "le", "el", "al", "bin", "ibn", "mc", "mac", "st"}


def name_tokens(name: str) -> set[str]:
    """Tokens worth matching on: drop initials and very short particles."""
    return {t for t in fold(name).split() if len(t) > 2}


def surname_of(name: str) -> str | None:
    """The LAST substantive token, not the longest.

    Picking the longest token made "Milan Heca" match on "Milan" and pull in
    every AC Milan document in the cache - 509 of them - and did the same for
    every "Álvaro ...". Surname position, not length, is the reliable signal.
    """
    toks = [t for t in fold(name).split() if len(t) > 2 and t not in PARTICLES]
    return toks[-1] if toks else None


@dataclass
class CachedDoc:
    url: str
    title: str | None = None
    snippet: str | None = None
    raw_content: str | None = None
    page_text: str | None = None
    page_status: str | None = None
    queries: list = field(default_factory=list)
    providers: set = field(default_factory=set)

    @property
    def domain(self) -> str:
        return domain_of(self.url)

    @property
    def best_text(self) -> str:
        return self.page_text or self.raw_content or self.snippet or ""

    @property
    def readable(self) -> bool:
        return len(self.best_text) >= 200


class CachedEvidenceIndex:
    """Everything already acquired, queryable by player / club / event family."""

    def __init__(self, root: Path = CACHE_ROOT):
        self.root = Path(root)
        self.docs: dict[str, CachedDoc] = {}
        self.by_token: dict[str, set] = defaultdict(set)
        self.by_domain: dict[str, set] = defaultdict(set)
        self.query_to_urls: dict[str, list] = defaultdict(list)
        self.stats: dict = {}

    # ---------- build ----------
    def build(self) -> "CachedEvidenceIndex":
        pages = self._load_pages()
        n_search_files = 0
        for sub, provider in SEARCH_DIRS.items():
            d = self.root / sub
            if not d.is_dir():
                continue
            for f in d.glob("*.json"):
                try:
                    payload = json.loads(f.read_text())
                except Exception:                              # noqa: BLE001
                    continue
                n_search_files += 1
                query = payload.get("query") or ""
                for item in payload.get("results") or []:
                    url = canonical_url(item.get("url") or "")
                    if not url:
                        continue
                    doc = self.docs.get(url)
                    if doc is None:
                        doc = CachedDoc(url=url)
                        self.docs[url] = doc
                    doc.title = doc.title or item.get("title")
                    doc.snippet = doc.snippet or item.get("content")
                    if item.get("raw_content") and not doc.raw_content:
                        doc.raw_content = item["raw_content"]
                    doc.providers.add(provider)
                    if query and query not in doc.queries:
                        doc.queries.append(query)
                    self.query_to_urls[query].append(url)

        # Attach directly-retrieved page text, and keep pages we fetched even if
        # no search result references them.
        for url, (text, status) in pages.items():
            doc = self.docs.get(url)
            if doc is None:
                doc = CachedDoc(url=url)
                self.docs[url] = doc
                doc.providers.add("page_cache")
            doc.page_text, doc.page_status = text, status

        for url, doc in self.docs.items():
            self.by_domain[doc.domain].add(url)
            blob = fold(" ".join(filter(None, [doc.title, doc.snippet, url.replace("-", " ")])))
            # Index the head of the body too - enough to catch names, cheap enough
            # to hold in memory across ~1.3k documents.
            blob += " " + fold((doc.best_text or "")[:4000])
            for tok in set(blob.split()):
                if len(tok) > 2:
                    self.by_token[tok].add(url)

        readable = [d for d in self.docs.values() if d.readable]
        self.stats = {
            "search_files_read": n_search_files,
            "unique_urls": len(self.docs),
            "readable_documents": len(readable),
            "with_provider_raw_content": sum(1 for d in self.docs.values() if d.raw_content),
            "with_retrieved_page_text": sum(1 for d in self.docs.values() if d.page_text),
            "distinct_domains": len(self.by_domain),
            "distinct_queries": len(self.query_to_urls),
            "duplicate_rate": round(
                1 - len(self.docs) / max(1, sum(len(v) for v in self.query_to_urls.values())), 4),
        }
        return self

    def _load_pages(self) -> dict:
        out = {}
        for sub in PAGE_DIRS:
            d = self.root / sub
            if not d.is_dir():
                continue
            for f in d.glob("*.json"):
                try:
                    payload = json.loads(f.read_text())
                except Exception:                              # noqa: BLE001
                    continue
                url = canonical_url(payload.get("url") or "")
                text = payload.get("text") or ""
                if not url:
                    continue
                prev = out.get(url)
                if prev is None or len(text) > len(prev[0]):
                    out[url] = (text, payload.get("status"))
        return out

    # ---------- query ----------
    def lookup(self, player_name: str, clubs: list[str] | None = None,
               aliases: list[str] | None = None, min_score: int = 2,
               require_surname: bool = True) -> list[CachedDoc]:
        """Candidate cached documents for a player and their clubs.

        `require_surname` defaults True and is load-bearing. Without it, a
        forename that is also a club - "Milan Joksimovic" - matches every AC
        Milan document in the cache. That is the same class of name-collision
        bug that put six different Vitinhos into one Stage 1 sample, so the
        surname must be present, not merely some token of the name.
        """
        want = name_tokens(player_name)
        for a in aliases or []:
            want |= name_tokens(a)
        if not want:
            return []
        club_toks = set()
        for c in clubs or []:
            club_toks |= name_tokens(c)

        scores: dict[str, int] = defaultdict(int)
        surname = surname_of(player_name)
        if surname:
            want.add(surname)

        # Some surnames are ordinary words - Old, Long, May, Said, Post, Just,
        # Can, Main. Matching on those alone pulls in hundreds of unrelated
        # documents. A surname occurring in more than COMMON_TOKEN_DF of the
        # corpus is not distinctive enough to stand on its own, so it must be
        # corroborated by a club token.
        n_docs = max(1, len(self.docs))
        surname_df = len(self.by_token.get(surname, ())) / n_docs if surname else 0.0
        needs_club = surname_df > COMMON_TOKEN_DF
        surname_urls = set(self.by_token.get(surname, ())) if surname else set()
        for tok in want:
            for url in self.by_token.get(tok, ()):
                scores[url] += 3 if tok == surname else 1
        for tok in club_toks:
            for url in self.by_token.get(tok, ()):
                if url in scores:
                    scores[url] += 1
        club_urls: set = set()
        for tok in club_toks:
            club_urls |= set(self.by_token.get(tok, ()))
        out = [(s, self.docs[u]) for u, s in scores.items()
               if s >= min_score
               and (not require_surname or u in surname_urls)
               and (not needs_club or u in club_urls)]
        out.sort(key=lambda t: (-t[0], not t[1].readable, -len(t[1].best_text)))
        return [d for _, d in out]

    def mentions_full_name(self, doc: "CachedDoc", player_name: str) -> bool:
        """Does the document actually name THIS player, forename and surname
        together?

        Token scoring alone confuses namesakes: "Fábio Ronaldo" matched Cristiano
        Ronaldo's coverage, and "Aaron Long" matched the word "long". Requiring
        the forename within a short window of the surname removes both.
        """
        toks = [t for t in fold(player_name).split() if len(t) > 2 and t not in PARTICLES]
        if len(toks) < 2:
            return False
        fore, sur = toks[0], toks[-1]
        hay = fold(" ".join(filter(None, [doc.title, doc.snippet, doc.best_text[:200000]])))
        start = 0
        while True:
            i = hay.find(sur, start)
            if i < 0:
                return False
            window = hay[max(0, i - 60): i + 60]
            if fore in window:
                return True
            start = i + len(sur)

    def lookup_strict(self, player_name: str, clubs: list[str] | None = None,
                      min_score: int = 2) -> list["CachedDoc"]:
        """Cache lookup that additionally verifies the full name appears."""
        return [d for d in self.lookup(player_name, clubs, min_score=min_score)
                if self.mentions_full_name(d, player_name)]

    def for_family(self, family_rows, min_score: int = 2) -> list[CachedDoc]:
        """Cached documents relevant to any member of an event family."""
        if not len(family_rows):
            return []
        first = family_rows.iloc[0]
        clubs = sorted({str(c) for c in
                        list(family_rows.from_club_name) + list(family_rows.to_club_name)
                        if isinstance(c, str)})
        # Strict: the document must actually name this player, not merely share
        # a token with them. Namesakes and word-surnames are otherwise pulled in.
        return self.lookup_strict(str(first.player_name), clubs, min_score=min_score)

    def as_results(self, docs: list[CachedDoc], query: str = "") -> list[SearchResult]:
        return [SearchResult(url=d.url, title=d.title, snippet=d.snippet,
                             provider="cache", rank=i + 1, raw_content=d.raw_content,
                             query=query or (d.queries[0] if d.queries else None),
                             query_family="cache_lookup",
                             metadata={"page_status": d.page_status,
                                       "cached_providers": sorted(d.providers)})
                for i, d in enumerate(docs)]
