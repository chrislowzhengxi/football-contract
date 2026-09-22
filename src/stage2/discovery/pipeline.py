"""Provider-independent Stage 2 research pipeline.

Runs event-family construction, cache-first discovery, optional direct-source
discovery, retrieval and evidence selection. It calls no search API unless a
provider that needs one is explicitly selected, so a `cache,direct` run needs
no credentials of any kind.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..deep_search import chunk_document
from ..event_family import build_families, families_to_rows
from ..sources import classify, tier_of
from . import identity
from .cache_index import CachedEvidenceIndex
from .planner import plan_for_family, vocabulary
from .providers import build_providers
from .retriever import PageRetriever
from .types import Evidence, SearchQuery, deduplicate, rank_results

# Search-need classes (Part 14)
NO_SEARCH = "NO_EXTERNAL_SEARCH_NEEDED"
DIRECT_ONLY = "DIRECT_SOURCE_ONLY"
GEN_T1 = "GENERAL_SEARCH_TIER1"
GEN_T2 = "GENERAL_SEARCH_TIER2"
DONT_STANDALONE = "DO_NOT_SEARCH_STANDALONE"
REVIEW_FIRST = "REVIEW_FIRST"


def clause_terms():
    v = vocabulary().get("mechanisms", {})
    return sorted({t for m in v.values() for ts in m.values() for t in ts}, key=len, reverse=True)


_TERMS = None


def terms_in(text: str) -> list[str]:
    global _TERMS
    if _TERMS is None:
        _TERMS = clause_terms()
    low = (text or "").lower()
    return sorted({t for t in _TERMS if t.lower() in low})


@dataclass
class FamilyResult:
    event_family_id: str
    anchor_event_id: str
    player_name: str
    family_classification: str
    evidence: list = field(default_factory=list)
    queries: list = field(default_factory=list)
    search_need: str = REVIEW_FIRST
    notes: str = ""
    rejected_identity: list = field(default_factory=list)

    @property
    def readable(self):
        return [e for e in self.evidence if len(e.best_text) >= 200]

    @property
    def with_clause(self):
        return [e for e in self.readable if terms_in(e.best_text)]

    @property
    def tier1(self):
        return [e for e in self.readable if e.source_tier == 1]


class Stage2Pipeline:
    def __init__(self, providers=("cache", "direct", "generic"), allow_network: bool = False,
                 index: CachedEvidenceIndex | None = None, identity_gate: bool = True):
        self.index = index or CachedEvidenceIndex().build()
        # On by default. Without it the packet for a family can contain
        # documents about other players that merely mention ours in passing,
        # and a contract term then gets read off the wrong transaction.
        self.identity_gate = identity_gate
        self.providers, self.skipped = build_providers(
            providers, index=self.index, allow_network=allow_network)
        self.retriever = PageRetriever(allow_network=allow_network, index=self.index)
        self.canonical = None

    def provider_names(self):
        return [p.provider_name for p in self.providers]

    def run_family(self, family_rows, escalate: bool = False,
                   max_docs: int = 20) -> FamilyResult:
        anchor = family_rows[family_rows.is_family_anchor].iloc[0]
        clubs = sorted({str(x) for x in list(family_rows.from_club_name) +
                        list(family_rows.to_club_name)})
        res = FamilyResult(
            event_family_id=str(anchor.event_family_id),
            anchor_event_id=str(anchor.event_id),
            player_name=str(anchor.player_name),
            family_classification=str(anchor.family_interpretation_status))

        listed = []
        direct = next((p for p in self.providers if p.provider_name == "direct"), None)
        if direct is not None:
            listed = direct.domains_for_clubs(clubs)
        queries = plan_for_family(family_rows, escalate=escalate,
                                  listed_domains=tuple(listed))
        res.queries = [q.to_dict() for q in queries]

        results = []
        for p in self.providers:
            if p.provider_name == "cache":
                results += p.search_family(family_rows, max_results=max_docs * 2)
            else:
                for q in queries:
                    qq = q
                    if p.provider_name == "direct" and listed:
                        qq = SearchQuery(**{**q.to_dict(),
                                            "preferred_domains": tuple(listed),
                                            "excluded_domains": tuple(q.excluded_domains)})
                    try:
                        results += p.search(qq, max_results=10)
                    except Exception as exc:                   # noqa: BLE001
                        res.notes += f"[{p.provider_name} failed: {type(exc).__name__}]"

        merged = rank_results(deduplicate(results),
                              tier_of=lambda u: tier_of(classify(u, None, None)))
        retrieved = self.retriever.retrieve_all(
            merged[:max_docs * 2 if self.identity_gate else max_docs],
            event_id=res.anchor_event_id, event_family_id=res.event_family_id,
            from_club=str(anchor.from_club_name), to_club=str(anchor.to_club_name))

        if self.identity_gate:
            years = sorted({int(str(d)[:4]) for d in family_rows.transfer_date
                            if str(d)[:4].isdigit()})
            kept = []
            for e in retrieved:
                verdict = identity.assess(
                    e.title, e.source_url, e.best_text, str(anchor.player_name),
                    clubs, years=tuple(years))
                e.identity_strength = verdict.strength
                if verdict.admit:
                    kept.append(e)
                else:
                    res.rejected_identity.append(
                        {"url": e.source_url, "title": e.title,
                         "reason": verdict.reason_text, "signals": verdict.signals})
            retrieved = kept
        res.evidence = retrieved[:max_docs]
        res.search_need = classify_search_need(res, family_rows)
        return res


def classify_search_need(res: FamilyResult, family_rows) -> str:
    """Decide what, if anything, we should pay a search provider for.

    Ordered so the cheapest correct answer wins: a family that should not be
    researched standalone is never queued for search at all, however much
    evidence happens to exist for it.
    """
    anchor = family_rows[family_rows.is_family_anchor].iloc[0]
    fam_class = str(anchor.family_interpretation_status)
    review = bool(anchor.family_review_required)

    if fam_class in ("third_party_sale_related", "immediate_follow_on_transfer"):
        return DONT_STANDALONE
    if fam_class == "unresolved_family_semantics":
        return REVIEW_FIRST
    n_clause, n_t1 = len(res.with_clause), len(res.tier1)
    if n_t1 >= 1 and n_clause >= 1:
        return NO_SEARCH
    if n_clause >= 2:
        return NO_SEARCH
    if n_clause == 1:
        return DIRECT_ONLY
    if fam_class == "early_termination" and review:
        return REVIEW_FIRST
    if len(res.readable) >= 8:
        return GEN_T2          # plenty read, nothing found: needs depth or nothing
    return GEN_T1
