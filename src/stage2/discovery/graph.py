"""Source-lead expansion.

A tier-3 page may not establish a contractual term, but it often names the
source that can: "according to KAP", "the club announced", "as reported by La
Gazzetta". Throwing those pages away discards a map to better evidence. This
module extracts outbound links and named-publication references and records
them as edges, so weak sources can lead us upstream without ever being allowed
to establish a term themselves.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass

from ..sources import classify, tier_of
from .types import canonical_url, domain_of

# "according to X", "per X", "X reports", "as reported by X", plus the common
# non-English forms seen in the cached corpus.
# The attribution phrase is matched case-insensitively, but the captured
# publication name must still start with a capital, so "according to the club"
# does not become a citation of a publication called "The".
ATTRIBUTION_PATTERNS = [
    r"(?i:according to) ((?!The\b|A\b)[A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3})",
    r"(?i:as reported by) ([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3})",
    r"(?i:per|via) ([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3}) report",
    r"([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3}) reports? that",
    r"(?i:secondo)(?: il | la | lo | )([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3})",
    r"(?i:seg[uú]n)(?: el | la | )([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3})",
    r"(?i:segundo)(?: o | a | )([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3})",
    r"(?i:d'apr[eè]s) ([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3})",
    r"(?i:laut) ([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3})",
    r"([A-Z][\w&.']{1,20}(?: [A-Z][\w&.']{1,20}){0,3})'(?:in|nin) haberine g[oö]re",
]
# Named venues worth resolving upstream to.
AUTHORITATIVE_MENTIONS = {
    "kap": "kap.org.tr", "borsa istanbul": "kap.org.tr",
    "fifa": "fifa.com", "cas": "tas-cas.org",
    "annual report": None, "financial statement": None, "investor relations": None,
    "official website": None, "club statement": None, "official announcement": None,
    "the club announced": None, "club confirmed": None, "statement read": None,
}
HREF = re.compile(r'href=["\'](https?://[^"\'>\s]+)', re.I)


@dataclass
class Edge:
    src_url: str
    src_domain: str
    src_tier: int
    relation: str          # links_to | cites_publication | mentions_venue
    dst: str               # URL or a named publication/venue
    dst_domain: str | None
    dst_tier: int | None
    context: str
    event_family_id: str | None = None


def extract_edges(url: str, text: str, html: str | None = None,
                  event_family_id: str | None = None, limit: int = 40) -> list[Edge]:
    src_cls = classify(url, None, None)
    src_tier = tier_of(src_cls)
    src_dom = domain_of(url)
    edges: list[Edge] = []

    if html:
        for href in list(dict.fromkeys(HREF.findall(html)))[:limit]:
            h = canonical_url(href)
            d = domain_of(h)
            if not d or d == src_dom:
                continue
            cls = classify(h, None, None)
            edges.append(Edge(url, src_dom, src_tier, "links_to", h, d, tier_of(cls),
                              "", event_family_id))

    body = text or ""
    for pat in ATTRIBUTION_PATTERNS:
        for m in re.finditer(pat, body):
            name = m.group(1).strip().rstrip(".,;:")
            if len(name) < 3:
                continue
            ctx = body[max(0, m.start() - 90): m.end() + 90].replace("\n", " ")
            edges.append(Edge(url, src_dom, src_tier, "cites_publication", name,
                              None, None, ctx, event_family_id))
    low = body.lower()
    for mention, dom in AUTHORITATIVE_MENTIONS.items():
        i = low.find(mention)
        if i >= 0:
            ctx = body[max(0, i - 90): i + 120].replace("\n", " ")
            edges.append(Edge(url, src_dom, src_tier, "mentions_venue", mention,
                              dom, tier_of(classify(f"https://{dom}", None, None)) if dom else None,
                              ctx, event_family_id))
    return edges


def summarise(edges: list[Edge]) -> dict:
    """How often does a weak source point at a strong one? That ratio is the
    justification for keeping tier-3 pages in the pipeline at all."""
    lead = [e for e in edges if e.src_tier == 3 and (e.dst_tier or 3) <= 2]
    return {
        "edges": len(edges),
        "by_relation": dict(Counter(e.relation for e in edges)),
        "tier3_sources": len({e.src_url for e in edges if e.src_tier == 3}),
        "tier3_to_tier12_edges": len(lead),
        "tier3_sources_with_upstream_lead": len({e.src_url for e in lead}),
        "top_cited_publications": dict(Counter(
            e.dst for e in edges if e.relation == "cites_publication").most_common(25)),
        "top_link_targets": dict(Counter(
            e.dst_domain for e in edges if e.relation == "links_to" and e.dst_domain
        ).most_common(25)),
    }


def to_jsonl(edges: list[Edge]) -> list[str]:
    return [json.dumps(asdict(e), ensure_ascii=False) for e in edges]
