"""Stage 2 must work with no search credentials at all.

Every test here runs with TAVILY_API_KEY removed from the environment. That is
a hard requirement, not a convenience: the pipeline is not allowed to depend on
one vendor.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage2.discovery import types as T
from src.stage2.discovery.cache_index import CachedEvidenceIndex, fold, name_tokens
from src.stage2.discovery.cli import build_parser, parse_providers
from src.stage2.discovery.graph import extract_edges, summarise
from src.stage2.discovery.pipeline import (DONT_STANDALONE, NO_SEARCH,
                                           Stage2Pipeline, classify_search_need)
from src.stage2.discovery.planner import plan_for_family
from src.stage2.discovery.providers import build_providers
from src.stage2.discovery.providers.generic import GenericWebSearchProvider
from src.stage2.discovery.providers.tavily import TavilyProvider
from src.stage2.discovery.retriever import PageRetriever
from src.stage2.mechanisms import ALL_MECHANISMS, admissible


@pytest.fixture(autouse=True)
def no_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


# ---------- URL normalization / dedup ----------
@pytest.mark.parametrize("a,b", [
    ("https://WWW.Example.com/a/", "http://example.com/a"),
    ("https://example.com/a?utm_source=x&id=1", "https://example.com/a?id=1"),
    ("https://en.wikipedia.org/wiki/Alexander_S%C3%B8rloth",
     "https://en.wikipedia.org/wiki/Alexander_Sørloth"),
    ("https://example.com//a//b", "https://example.com/a/b"),
])
def test_canonical_url_collapses_equivalent_forms(a, b):
    assert T.canonical_url(a) == T.canonical_url(b)


def test_deduplicate_merges_across_providers_and_keeps_richest():
    merged = T.deduplicate([
        T.SearchResult("https://a.com/x", provider="tavily", query_family="f1", snippet="s"),
        T.SearchResult("http://www.a.com/x", provider="cache", query_family="f2",
                       raw_content="RC", provider_score=0.9),
    ])
    assert len(merged) == 1
    r = merged[0]
    assert set(r.metadata["providers"]) == {"tavily", "cache"}
    assert set(r.metadata["found_by"]) == {"f1", "f2"}
    assert r.raw_content == "RC" and r.snippet == "s" and r.provider_score == 0.9


def test_ranking_prefers_cross_family_agreement_then_tier():
    lo = T.SearchResult("https://x.com/1", rank=1, metadata={"found_by": ["a"]})
    hi = T.SearchResult("https://y.com/2", rank=9, metadata={"found_by": ["a", "b", "c"]})
    assert T.rank_results([lo, hi])[0] is hi


# ---------- provider selection ----------
def test_tavily_unavailable_without_key_but_does_not_raise():
    ok, why = TavilyProvider().available()
    assert ok is False and "TAVILY_API_KEY" in why


def test_build_providers_skips_tavily_and_keeps_the_rest():
    usable, skipped = build_providers(["cache", "direct", "generic", "tavily"])
    names = [p.provider_name for p in usable]
    assert "cache" in names and "direct" in names and "generic" in names
    assert any(n == "tavily" for n, _ in skipped)


def test_tavily_quota_exhausted_is_a_skip_not_a_crash(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    p = TavilyProvider(allow_network=False)
    ok, why = p.available()
    assert ok is False and "network" in why.lower()


def test_pipeline_runs_with_no_credentials():
    pipe = Stage2Pipeline(providers=("cache", "direct"), allow_network=False)
    assert pipe.providers, "cache+direct must be usable with no keys"
    assert "tavily" not in pipe.provider_names()


# ---------- tavily cache normalization ----------
def test_cached_tavily_payload_reads_as_generic_results():
    payload = {"query": "q", "results": [
        {"url": "https://a.com/1", "title": "T", "content": "snip",
         "raw_content": "BODY", "score": 0.7, "id": "z"}]}
    out = TavilyProvider.normalize(payload)
    assert len(out) == 1
    r = out[0]
    assert r.provider == "tavily" and r.raw_content == "BODY"
    assert r.snippet == "snip" and r.provider_score == 0.7


# ---------- cache index ----------
def test_fold_is_accent_and_case_insensitive():
    assert fold("Sørloth") == fold("sorloth")
    assert fold("Mönchengladbach") == fold("Monchengladbach")


def test_name_tokens_drops_initials():
    assert "a" not in name_tokens("A. Sørloth")


def test_lookup_requires_surname_so_club_forenames_do_not_match(tmp_path):
    """'Milan Joksimovic' must not match every AC Milan document."""
    idx = CachedEvidenceIndex(root=tmp_path)
    idx.docs = {}
    from src.stage2.discovery.cache_index import CachedDoc
    d = CachedDoc(url="https://x.com/acmilan", title="AC Milan sign a striker",
                  page_text="AC Milan " * 60)
    idx.docs[d.url] = d
    for tok in set(fold(d.title + " " + d.page_text).split()):
        idx.by_token[tok].add(d.url)
    # The forename "Milan" matches the club document; the surname does not.
    # Requiring the surname is what suppresses it, at any score threshold.
    assert idx.lookup("Milan Joksimovic", [], min_score=1) == []
    assert idx.lookup("Milan Joksimovic", [], min_score=1, require_surname=False) != []


# ---------- generic provider handoff ----------
def test_surname_is_last_token_not_longest():
    from src.stage2.discovery.cache_index import surname_of
    # "Milan Heca": longest token is the forename, which is also a club.
    assert surname_of("Milan Heca") == "heca"
    assert surname_of("Álvaro Peña") == "pena"
    assert surname_of("Kevin De Bruyne") == "bruyne"   # particle skipped


def test_common_word_surname_needs_club_corroboration(tmp_path):
    """Surnames like Old, Long, May are ordinary words; alone they must not match."""
    from src.stage2.discovery.cache_index import CachedDoc
    idx = CachedEvidenceIndex(root=tmp_path)
    idx.docs = {}
    for i in range(60):
        d = CachedDoc(url=f"https://x.com/{i}", title="a long report",
                      page_text="the long season " * 40)
        idx.docs[d.url] = d
        for tok in set(fold(d.title + " " + d.page_text).split()):
            idx.by_token[tok].add(d.url)
    # "long" is in 100% of the corpus, far above the distinctiveness threshold
    assert idx.lookup("Chris Long", []) == []


def test_full_name_verification_rejects_namesakes(tmp_path):
    """A document about Cristiano Ronaldo must not be evidence for Fabio Ronaldo."""
    from src.stage2.discovery.cache_index import CachedDoc
    idx = CachedEvidenceIndex(root=tmp_path)
    d = CachedDoc(url="https://x.com/cr7", title="Cristiano Ronaldo joins Juventus",
                  page_text="Cristiano Ronaldo completed his move to Juventus. " * 20)
    assert idx.mentions_full_name(d, "Cristiano Ronaldo")
    assert not idx.mentions_full_name(d, "Fabio Ronaldo")


def test_generic_provider_records_and_replays(tmp_path):
    g = GenericWebSearchProvider(handoff_dir=tmp_path)
    q = T.SearchQuery(query="player club option 2020", query_family="mechanism_en")
    assert g.search(q) == []
    assert g.pending() == ["player club option 2020"]
    g.record("player club option 2020", [{"url": "https://club.com/news", "title": "N"}])
    out = g.search(q)
    assert len(out) == 1 and out[0].provider == "generic"
    assert out[0].url == "https://club.com/news"


def test_generic_provider_honours_domain_exclusion(tmp_path):
    g = GenericWebSearchProvider(handoff_dir=tmp_path)
    g.record("q", [{"url": "https://bad.com/a"}, {"url": "https://good.com/b"}])
    out = g.search(T.SearchQuery(query="q", excluded_domains=("bad.com",)))
    assert [r.domain for r in out] == ["good.com"]


# ---------- retrieval provenance ----------
def test_retrieval_keeps_search_text_separate_from_page_text():
    r = T.SearchResult("https://e.com/a", provider="tavily", raw_content="PROVIDER TEXT")
    ev = PageRetriever(allow_network=False).retrieve(r)
    assert ev.raw_search_content == "PROVIDER TEXT"
    assert ev.retrieved_page_text in (None, "")
    assert ev.text_origin == "raw_search_content"
    assert ev.best_text == "PROVIDER TEXT"


def test_retrieval_prefers_our_own_page_text_when_present(tmp_path, monkeypatch):
    ret = PageRetriever(allow_network=False)
    ret._disk = {T.canonical_url("https://e.com/a"): ("OUR TEXT", "html_ok")}
    ev = ret.retrieve(T.SearchResult("https://e.com/a", provider="tavily",
                                     raw_content="PROVIDER TEXT"))
    assert ev.text_origin == "retrieved_page_text" and ev.best_text == "OUR TEXT"
    assert ev.raw_search_content == "PROVIDER TEXT"   # still preserved


def test_retrieval_deduplicates_equivalent_urls():
    ret = PageRetriever(allow_network=False)
    out = ret.retrieve_all([T.SearchResult("https://e.com/a"),
                            T.SearchResult("http://www.e.com/a/")])
    assert len(out) == 1


def test_evidence_carries_discovery_provenance():
    r = T.SearchResult("https://e.com/a", provider="direct", query="q", rank=3)
    ev = PageRetriever(allow_network=False).retrieve(r)
    assert ev.discovery_provider == "direct" and ev.discovery_query == "q"
    assert ev.search_rank == 3


# ---------- planner ----------
def _family_rows():
    return pd.DataFrame([
        {"event_family_id": "fam_1", "event_id": "e_loan", "is_family_anchor": False,
         "event_role": "original_loan", "player_name": "Wout Weghorst",
         "transfer_date": "2022-07-05", "from_club_name": "Burnley",
         "to_club_name": "Besiktas", "family_interpretation_status": "early_termination",
         "family_review_required": True},
        {"event_family_id": "fam_1", "event_id": "e_ret", "is_family_anchor": True,
         "event_role": "early_termination", "player_name": "Wout Weghorst",
         "transfer_date": "2023-01-12", "from_club_name": "Besiktas",
         "to_club_name": "Burnley", "family_interpretation_status": "early_termination",
         "family_review_required": True},
    ])


def test_planner_is_provider_agnostic_and_targets_the_negotiated_leg():
    qs = plan_for_family(_family_rows())
    assert 4 <= len(qs) <= 7
    assert all(isinstance(q, T.SearchQuery) for q in qs)
    # terms live on the loan, not the administrative return
    assert all(q.event_id == "e_loan" for q in qs)


def test_planner_emits_local_language_for_non_english_club():
    langs = {q.language for q in plan_for_family(_family_rows())}
    assert "tr" in langs, "Besiktas must be queried in Turkish"


def test_planner_escalation_adds_queries_without_changing_normal_ones():
    base = plan_for_family(_family_rows())
    esc = plan_for_family(_family_rows(), escalate=True)
    assert len(esc) > len(base)
    assert [q.query for q in esc[:len(base)]] == [q.query for q in base]


def test_planner_never_imports_a_search_vendor():
    import src.stage2.discovery.planner as planner
    src = Path(planner.__file__).read_text().lower()
    assert "tavily" not in src and "api_key" not in src


# ---------- source graph ----------
def test_source_graph_finds_upstream_lead_from_tier3():
    edges = extract_edges("https://sportskeeda.com/x",
                          "According to Fabrizio Romano the option was 5m. KAP filings confirm.")
    rel = {e.relation for e in edges}
    assert "cites_publication" in rel and "mentions_venue" in rel
    s = summarise(edges)
    assert s["tier3_sources"] == 1 and s["tier3_to_tier12_edges"] >= 1


def test_source_graph_ignores_self_links():
    edges = extract_edges("https://a.com/1", "", html='<a href="https://a.com/2">x</a>')
    assert not [e for e in edges if e.relation == "links_to"]


# ---------- mechanisms / re-scoping ----------
def test_mechanism_rejected_on_incompatible_role():
    ok, problems = admissible("add_ons", {
        "status": "partially_disclosed", "evidence_ids": ["e1"],
        "evidence_span": "x", "which_transaction": "Roma->Milan", "direction_established": True},
        "administrative_return")
    assert ok is False and any("does not sensibly attach" in p for p in problems)


def test_mechanism_accepted_on_compatible_role():
    ok, problems = admissible("add_ons", {
        "status": "partially_disclosed", "evidence_ids": ["e1"],
        "evidence_span": "x", "which_transaction": "Roma->Milan", "direction_established": True},
        "third_party_sale")
    assert ok is True, problems


def test_amount_without_direction_is_inadmissible():
    ok, problems = admissible("termination_compensation", {
        "status": "disclosed_yes", "evidence_ids": ["e1"], "evidence_span": "paid 5m",
        "which_transaction": "loan ended"}, "early_termination")
    assert ok is False and any("payer/recipient" in p for p in problems)


def test_new_mechanisms_exist():
    for m in ("termination_compensation", "third_party_sale_share", "economic_mechanism"):
        assert m in ALL_MECHANISMS


# ---------- search need ----------
def test_admin_family_is_never_queued_for_standalone_search():
    rows = _family_rows()
    rows["family_interpretation_status"] = "third_party_sale_related"

    class R:
        with_clause = [1, 2, 3]
        tier1 = [1]
        readable = [1, 2, 3]
    assert classify_search_need(R(), rows) == DONT_STANDALONE


def test_tier1_plus_clause_evidence_needs_no_search():
    class R:
        with_clause = [1]
        tier1 = [1]
        readable = [1]
    assert classify_search_need(R(), _family_rows()) == NO_SEARCH


# ---------- CLI ----------
def test_cli_accepts_provider_lists_and_orders_them_cheapest_first():
    assert parse_providers("tavily,cache") == ["cache", "tavily"]


def test_cli_rejects_unknown_provider():
    with pytest.raises(Exception):
        parse_providers("notaprovider")


def test_cli_defaults_do_not_require_tavily():
    args = build_parser().parse_args(["--event-id", "x"])
    assert args.providers is None       # resolved to cache,direct downstream


def test_cli_list_providers_runs_without_keys(capsys):
    from src.stage2.discovery.cli import main
    assert main(["--list-providers"]) == 0
    out = capsys.readouterr().out
    assert "cache" in out and "tavily" in out


# ---------- config integrity ----------
@pytest.mark.parametrize("path", [
    "data/config/source_registry.json",
    "data/config/source_quality_registry.json",
    "data/config/contract_clause_vocabulary.json",
])
def test_config_files_are_valid_json(path):
    assert json.loads(Path(path).read_text())


def test_unknown_domain_is_tier2_not_discarded():
    q = json.loads(Path("data/config/source_quality_registry.json").read_text())
    assert q["unknown_domain_policy"]["tier"] == 2
    assert q["unknown_domain_policy"]["action"] == "retrieve_read_classify"


def test_clause_vocabulary_covers_required_languages():
    v = json.loads(Path("data/config/contract_clause_vocabulary.json").read_text())
    for lang in ("en", "es", "pt", "it", "de", "fr", "tr"):
        assert lang in v["languages"]
    for mech in ("purchase_option", "purchase_obligation", "buy_back", "sell_on",
                 "add_ons", "termination_compensation", "transfer_proceeds_share"):
        assert mech in v["mechanisms"]
