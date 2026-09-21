from __future__ import annotations

import json
import os
import re
import csv
import argparse
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import DEFAULT_OUTPUT_DIR
from .source_registry import OFFICIAL_CLUB_DOMAINS, REPUTABLE_TIER2_DOMAINS


@dataclass
class SourceCandidate:
    source_url: str
    source_title: str | None
    publisher: str | None
    publication_date: str | None
    source_type: str
    evidence_text: str
    language: str
    evidence_id: str | None = None
    reported_values: list[dict[str, Any]] | None = None
    quality_score: float = 0.0
    accessible: bool = True
    search_query: str | None = None
    search_rank: int | None = None
    provider_score: float | None = None
    source_tier: int | None = None
    event_match_status: str | None = None
    event_match_score: float | None = None
    event_match_reasons: list[str] | None = None
    query_family: str | None = None
    query_language: str | None = None
    target_domain: str | None = None
    query_reason: str | None = None
    retrieved_text: str | None = None
    retrieval_status: str | None = None
    retrieval_http_status: int | None = None
    retrieval_final_url: str | None = None
    retrieval_timestamp: str | None = None
    retrieval_method: str | None = None
    retrieval_error: str | None = None
    content_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchProvider(Protocol):
    def search_transfer(self, event: dict[str, Any]) -> list[SourceCandidate]:
        ...


@dataclass
class SourceDiscoveryResult:
    candidates: list[SourceCandidate]
    sufficient: bool
    reasons: list[str]


@dataclass
class QueryPlan:
    query: str
    family: str
    language: str = "en"
    target_domain: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REPUTABLE_DOMAINS = {
    "reuters.com", "bbc.com", "espn.com", "theathletic.com", "nytimes.com",
    "apnews.com", "theguardian.com", "skysports.com", "goal.com",
} | {domain for domain, entry in REPUTABLE_TIER2_DOMAINS.items() if entry.source_type == "major_news"}
SPECIALIST_DOMAINS = {
    "football-italia.net", "maisfutebol.iol.pt", "ojogo.pt", "record.pt",
    "a-bola.pt", "lequipe.fr", "marca.com", "as.com",
} | {domain for domain, entry in REPUTABLE_TIER2_DOMAINS.items() if entry.source_type == "football_reporting"}
CLUB_DOMAINS = {club: entry.domain for club, entry in OFFICIAL_CLUB_DOMAINS.items()}
CLUB_ALIASES = {
    "porto": ("porto", "fc porto"),
    "fc porto": ("porto", "fc porto"),
    "braga": ("braga", "sc braga"),
    "sc braga": ("braga", "sc braga"),
    "aj auxerre": ("aj auxerre", "auxerre", "aja"),
    "auxerre": ("aj auxerre", "auxerre", "aja"),
    "estrela amadora": ("estrela amadora", "estrela da amadora"),
}
PORTUGUESE_CLUBS = {"benfica", "sl benfica", "porto", "fc porto", "braga"}
TURKISH_CLUBS = {"besiktas", "beşiktaş", "fenerbahce", "fenerbahçe"}
FRENCH_CLUBS = {"psg", "paris saint-germain"}
GREEK_CLUBS = {"aek athens"}
PORTUGUESE_MECHANISM_TERMS = (
    "transferência",
    "empréstimo",
    "opção de compra",
    "obrigação de compra",
    "compra obrigatória",
    "valor",
    "cláusula",
    "mais-valia",
    "percentagem",
)
LOCAL_LANGUAGE_TERMS = {
    "pt": PORTUGUESE_MECHANISM_TERMS[:7],
    "tr": ("transfer", "kiralık", "satın alma opsiyonu", "zorunlu satın alma", "bonservis", "sözleşme"),
    "fr": ("transfert", "prêt", "option d'achat", "obligation d'achat", "montant", "contrat"),
    "el": ("μεταγραφή", "δανεισμός", "οψιόν αγοράς", "υποχρεωτική αγορά", "συμβόλαιο"),
}


def build_search_queries(event: dict[str, Any]) -> list[str]:
    return [plan.query for plan in build_query_plan(event, max_queries=7)]


def build_query_plan(event: dict[str, Any], max_queries: int = 6, unresolved_fields: list[str] | None = None) -> list[QueryPlan]:
    player = event.get("player_name", "")
    from_club = event.get("from_club_name", "")
    to_club = event.get("to_club_name", "")
    year = str(event.get("transfer_date", ""))[:4]
    identity = f'"{player}" "{from_club}" "{to_club}" {year}'
    plans = [
        QueryPlan(f"{identity} transfer", "generic_event", reason="confirm exact transfer event, clubs, direction, and year"),
    ]
    for club in (from_club, to_club):
        domain = official_domain_for_club(str(club))
        if domain:
            plans.append(QueryPlan(
                f'site:{domain} "{player}" "{from_club}" "{to_club}" {year}',
                "club_domain",
                target_domain=domain,
                reason=f"search official {club} domain",
            ))
    terms = mechanism_terms_for_event(event, unresolved_fields)
    if terms:
        plans.append(QueryPlan(f"{identity} {' '.join(terms[:4])}", "mechanism", reason="search for specific contract mechanisms"))
    language = local_language_for_event(event)
    if language:
        local_terms = " ".join(LOCAL_LANGUAGE_TERMS[language][:4])
        plans.append(QueryPlan(
            f'"{player}" "{from_club}" "{to_club}" {year} {local_terms}',
            "local_language",
            language=language,
            reason=f"{language} mechanism vocabulary for local clubs",
        ))
    return plans[:max_queries]


def official_domain_for_club(club_name: str) -> str | None:
    return CLUB_DOMAINS.get(_club_key(club_name))


def local_language_for_event(event: dict[str, Any]) -> str | None:
    club_keys = {_club_key(str(event.get(key, ""))) for key in ("from_club_name", "to_club_name")}
    if club_keys & PORTUGUESE_CLUBS:
        return "pt"
    if club_keys & TURKISH_CLUBS:
        return "tr"
    if club_keys & FRENCH_CLUBS:
        return "fr"
    if club_keys & GREEK_CLUBS:
        return "el"
    return None


def _club_key(club_name: str) -> str:
    text = str(club_name).lower()
    replacements = {"á": "a", "à": "a", "ã": "a", "â": "a", "é": "e", "ê": "e", "í": "i", "ó": "o", "ô": "o", "ú": "u", "ç": "c", "ü": "u"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _club_aliases(club_name: str) -> tuple[str, ...]:
    key = _club_key(club_name)
    return CLUB_ALIASES.get(key, (key,) if key else ())


def mechanism_terms_for_event(event: dict[str, Any], unresolved_fields: list[str] | None = None) -> list[str]:
    fields = set(unresolved_fields or ("transfer_fee", "loan_fee", "purchase_option", "purchase_obligation", "add_ons"))
    mapping = [
        ("transfer_fee", "transfer fee"),
        ("loan_fee", "loan fee"),
        ("purchase_option", "option to buy"),
        ("purchase_obligation", "obligation to buy"),
        ("obligation_trigger", "trigger"),
        ("sell_on", "sell-on"),
        ("buy_back", "buy-back"),
        ("release_or_purchase_clause", "release clause"),
        ("add_ons", "add-ons"),
    ]
    return [term for field, term in mapping if field in fields][:4]


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _source_type(url: str, event: dict[str, Any]) -> str:
    domain = _domain(url)
    if domain in set(CLUB_DOMAINS.values()):
        return "official"
    if domain in REPUTABLE_DOMAINS:
        return "major_news"
    if domain in SPECIALIST_DOMAINS:
        return "football_reporting"
    club_tokens = re.findall(r"[a-z0-9]+", " ".join(str(event.get(key, "")) for key in ("from_club_name", "to_club_name")).lower())
    if any(token and token in domain for token in club_tokens if len(token) > 3):
        return "official"
    if any(token in domain for token in ("sec.gov", "cmvm", "stockexchange", "regulator")):
        return "regulatory"
    return "aggregator"


def source_tier(candidate: SourceCandidate) -> int:
    domain = _domain(candidate.source_url)
    if domain in {"transfermarkt.com", "transfermarkt.us", "transfermarkt.co.uk", "wikipedia.org", "instagram.com", "facebook.com", "reddit.com"}:
        return 3
    if candidate.source_type in {"official", "regulatory", "governing_body"}:
        return 1
    if candidate.source_type in {"major_news", "football_reporting"}:
        return 2
    return 3


def filter_admissible_sources(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    """Keep full discovery provenance separate from contractual evidence input."""
    admissible = []
    for candidate in candidates:
        candidate.source_tier = source_tier(candidate)
        if candidate.event_match_status in {"ambiguous", "mismatch"}:
            continue
        if candidate.source_tier <= 2 and candidate.accessible:
            admissible.append(candidate)
    return admissible


def assess_event_match(candidate: SourceCandidate, event: dict[str, Any]) -> tuple[str, float, list[str]]:
    haystack = _text_key(f"{candidate.source_title or ''} {candidate.evidence_text} {candidate.retrieved_text or ''}")
    player = _text_key(str(event.get("player_name", "")))
    from_club = _text_key(str(event.get("from_club_name", "")))
    to_club = _text_key(str(event.get("to_club_name", "")))
    from_aliases = _club_aliases(from_club)
    to_aliases = _club_aliases(to_club)
    year = str(event.get("transfer_date", ""))[:4]
    reasons: list[str] = []
    score = 0.0
    if player and player in haystack:
        score += 0.25
        reasons.append("player_match")
    if from_aliases and any(alias in haystack for alias in from_aliases):
        score += 0.2
        reasons.append("departing_club_match")
    if to_aliases and any(alias in haystack for alias in to_aliases):
        score += 0.2
        reasons.append("receiving_club_match")
    if year and year in haystack:
        score += 0.1
        reasons.append("year_match")
    mentioned_years = set(re.findall(r"\b20\d{2}\b", haystack))

    forward = _has_direction_any(haystack, from_aliases, to_aliases)
    reverse = _has_direction_any(haystack, to_aliases, from_aliases)
    loan_return = any(term in haystack for term in ("end of loan", "return from loan", "loan return", "returned to"))
    later_permanent = any(term in haystack for term in ("made permanent", "permanent switch", "permanent transfer")) and "loan" in haystack
    if later_permanent:
        reasons.append("loan_followed_by_permanent_transfer")
        return "ambiguous", min(score, 0.65), reasons
    if forward:
        score += 0.25
        reasons.append("direction_match")
    if reverse and not forward:
        reasons.append("reverse_direction")
        return "mismatch", min(score, 0.35), reasons
    if loan_return and not forward:
        reasons.append("loan_return_without_target_direction")
        return "mismatch", min(score, 0.35), reasons
    if forward and year and mentioned_years and year not in mentioned_years:
        reasons.append("wrong_year_or_date")
        return "ambiguous", min(score, 0.65), reasons
    hosted_receiving = official_domain_for_club(str(event.get("to_club_name", ""))) == _domain(candidate.source_url)
    transfer_language = any(term in haystack for term in (
        "transfer", "transferred", "signs", "signed", "joins", "joined", "leaves for",
        "e dragao", "é dragão", "cedido ao", "cedido a", "emprestado ao", "emprestado a",
        "transferencia", "transferência", "vertrekt naar", "se marcha al", "rejoint",
    ))
    if not forward and hosted_receiving and _mentions_transfer_to_other_club(haystack, to_aliases):
        reasons.append("different_destination_announcement")
        return "mismatch", min(score, 0.35), reasons
    if not forward and hosted_receiving and transfer_language and player and player in haystack and to_aliases and any(alias in haystack for alias in to_aliases):
        score += 0.18
        reasons.append("receiving_club_official_announcement")
        if year and year in haystack:
            return "likely", min(score, 1.0), reasons
    if score >= 0.75 and forward:
        return "exact", min(score, 1.0), reasons
    if score >= 0.5 and forward:
        return "likely", min(score, 1.0), reasons
    return "ambiguous", min(score, 1.0), reasons


def _has_direction(text: str, from_club: str, to_club: str) -> bool:
    if not from_club or not to_club:
        return False
    patterns = (
        rf"from\s+{re.escape(from_club)}.{{0,80}}\bto\s+{re.escape(to_club)}",
        rf"{re.escape(from_club)}.{{0,80}}\bto\s+{re.escape(to_club)}",
        rf"{re.escape(to_club)}.{{0,80}}\bfrom\s+{re.escape(from_club)}",
        rf"{re.escape(to_club)}.{{0,80}}\bsign(?:ed|s|ing)?\b.{{0,80}}\bfrom\s+{re.escape(from_club)}",
        rf"{re.escape(to_club)}.{{0,80}}\bjoin(?:ed|s|ing)?\b.{{0,80}}\bfrom\s+{re.escape(from_club)}",
        rf"{re.escape(from_club)}.{{0,80}}\btransfer(?:red)?\b.{{0,80}}\bto\s+{re.escape(to_club)}",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _has_direction_any(text: str, from_aliases: tuple[str, ...], to_aliases: tuple[str, ...]) -> bool:
    return any(_has_direction(text, from_alias, to_alias) for from_alias in from_aliases for to_alias in to_aliases)


def _mentions_transfer_to_other_club(text: str, to_aliases: tuple[str, ...]) -> bool:
    patterns = (
        r"\bjoin(?:s|ed|ing)?\s+(?:greek club\s+|club\s+)?([a-z0-9 .'-]{3,80})",
        r"\bloan(?:ed)?\s+to\s+([a-z0-9 .'-]{3,80})",
        r"\bpre(?:t|te|ted|tee)\s+(?:au|a|to)\s+([a-z0-9 .'-]{3,80})",
        r"\brejoint\s+([a-z0-9 .'-]{3,80})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            destination = match.group(1)
            if any(alias in destination for alias in to_aliases):
                continue
            if any(marker in destination for marker in ("panathinaikos", "roma", "benfica", "porto", "aek", "fenerbahce", "besiktas", "auxerre", "braga")):
                return True
    return False


def _text_key(text: str) -> str:
    replacements = {
        "á": "a", "à": "a", "ã": "a", "â": "a", "ä": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "õ": "o", "ô": "o", "ö": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ç": "c",
    }
    normalized = str(text).lower()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return re.sub(r"\s+", " ", normalized).strip()


def score_source(candidate: SourceCandidate, event: dict[str, Any]) -> float:
    score = {
        "official": 60,
        "regulatory": 65,
        "major_news": 50,
        "football_reporting": 35,
        "aggregator": 15,
    }.get(candidate.source_type, 10)
    haystack = f"{candidate.source_title or ''} {candidate.evidence_text} {candidate.retrieved_text or ''}".lower()
    identity = " ".join(str(event.get(key, "")) for key in ("player_name", "from_club_name", "to_club_name")).lower()
    if event.get("player_name", "").lower() in haystack:
        score += 12
    if any(term in haystack for term in ("€", "£", "$", "fee", "loan", "option", "obligation", "bonus", "add-on")):
        score += 10
    if "said" in haystack or "statement" in haystack or "announced" in haystack:
        score += 5
    if candidate.publication_date and str(event.get("transfer_date", ""))[:4] in candidate.publication_date:
        score += 3
    if identity and all(token in haystack for token in identity.split() if len(token) > 3):
        score += 5
    if candidate.target_domain and _domain(candidate.source_url) == candidate.target_domain:
        score += 12
    if candidate.event_match_status is None:
        candidate.event_match_status, candidate.event_match_score, candidate.event_match_reasons = assess_event_match(candidate, event)
    if candidate.event_match_status == "exact":
        score += 18
    elif candidate.event_match_status == "likely":
        score += 8
    elif candidate.event_match_status == "ambiguous":
        score -= 12
    elif candidate.event_match_status == "mismatch":
        score -= 60
    if not candidate.accessible:
        score -= 50
    return max(0.0, min(100.0, score))


def deduplicate_sources(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    retained: list[SourceCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.quality_score, reverse=True):
        normalized_url = candidate.source_url.split("#", 1)[0].rstrip("/").lower()
        duplicate = False
        for existing in retained:
            existing_url = existing.source_url.split("#", 1)[0].rstrip("/").lower()
            title_similarity = SequenceMatcher(None, (candidate.source_title or "").lower(), (existing.source_title or "").lower()).ratio()
            if normalized_url == existing_url or (title_similarity >= 0.92 and candidate.publisher == existing.publisher):
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    for index, candidate in enumerate(retained, start=1):
        candidate.evidence_id = candidate.evidence_id or f"s{index}"
    return retained


def assess_source_sufficiency(candidates: list[SourceCandidate]) -> tuple[bool, list[str]]:
    usable = [candidate for candidate in candidates if candidate.accessible]
    event_scored = any(candidate.event_match_status is not None for candidate in usable)
    if event_scored and not any(candidate.event_match_status in {"exact", "likely"} for candidate in usable):
        return False, ["no admissible source establishes exact or likely event identity"]
    strong = [candidate for candidate in usable if candidate.quality_score >= 70]
    independent_domains = {_domain(candidate.source_url) for candidate in usable if candidate.quality_score >= 50}
    if strong:
        return True, ["at least one strong accessible source"]
    if len(independent_domains) >= 2:
        return True, ["two independent reputable sources"]
    return False, ["fewer than one strong or two independent reputable accessible sources"]


def retrieve_fullpage_for_promising_sources(
    candidates: list[SourceCandidate],
    event: dict[str, Any],
    fetcher: Any | None = None,
) -> list[SourceCandidate]:
    """Retrieve known Tier 1/2 result pages before final event matching."""
    if fetcher is None:
        from .official_page_retrieval import OfficialPageFetcher
        fetcher = OfficialPageFetcher()
    for candidate in candidates:
        candidate.source_tier = source_tier(candidate)
        if candidate.source_tier > 2:
            continue
        fetcher.apply_to_candidate(candidate)
        candidate.event_match_status, candidate.event_match_score, candidate.event_match_reasons = assess_event_match(candidate, event)
        candidate.quality_score = score_source(candidate, event)
    return deduplicate_sources(candidates)


class TavilySearchProvider:
    """Search Tavily's API and normalize results without extracting claims."""

    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 20, max_results_per_query: int = 5):
        self.api_key = api_key
        self.timeout = timeout
        self.max_results_per_query = max_results_per_query
        self.search_count = 0
        self.raw_result_count = 0

    @classmethod
    def from_environment(cls) -> "TavilySearchProvider":
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is required for Tavily source discovery")
        return cls(api_key)

    def search_transfer(self, event: dict[str, Any]) -> list[SourceCandidate]:
        candidates: list[SourceCandidate] = []
        for plan in build_query_plan(event):
            payload = self._search(plan.query)
            for rank, item in enumerate(payload.get("results", []), start=1):
                url = item.get("url")
                if not url:
                    continue
                candidate = SourceCandidate(
                    source_url=url,
                    source_title=item.get("title"),
                    publisher=_domain(url),
                    publication_date=item.get("published_date"),
                    source_type=_source_type(url, event),
                    evidence_text=item.get("content") or "",
                    language=item.get("language") or "en",
                    search_query=plan.query,
                    search_rank=rank,
                    provider_score=item.get("score"),
                    query_family=plan.family,
                    query_language=plan.language,
                    target_domain=plan.target_domain,
                    query_reason=plan.reason,
                )
                candidate.accessible = _check_accessibility(url, self.timeout)
                candidate.event_match_status, candidate.event_match_score, candidate.event_match_reasons = assess_event_match(candidate, event)
                candidate.quality_score = score_source(candidate, event)
                candidates.append(candidate)
        return deduplicate_sources(candidates)

    def _search(self, query: str) -> dict[str, Any]:
        body = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": self.max_results_per_query,
            "include_answer": False,
            "include_raw_content": False,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers={"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "football-contract-source-discovery/1.0"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                self.search_count += 1
                payload = json.loads(response.read())
        except HTTPError as error:
            if error.code in {401, 403}:
                raise RuntimeError("Tavily authentication failed") from error
            raise RuntimeError(f"Tavily search failed with HTTP {error.code}") from error
        self.raw_result_count += len(payload.get("results", []))
        return payload


def build_tavily_queries(event: dict[str, Any]) -> list[str]:
    return [plan.query for plan in build_query_plan(event)]


def _check_accessibility(url: str, timeout: float) -> bool:
    try:
        request = Request(url, headers={"User-Agent": "football-contract-source-discovery/1.0"}, method="HEAD")
        with urlopen(request, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 400
    except Exception:
        return False


def discover_transfer(
    event: dict[str, Any],
    provider: SearchProvider,
    *,
    use_fullpage: bool = True,
    fullpage_fetcher: Any | None = None,
) -> SourceDiscoveryResult:
    candidates = provider.search_transfer(event)
    if use_fullpage:
        candidates = retrieve_fullpage_for_promising_sources(candidates, event, fullpage_fetcher)
    admissible = filter_admissible_sources(candidates)
    sufficient, reasons = assess_source_sufficiency(admissible)
    return SourceDiscoveryResult(candidates, sufficient, reasons)


def load_source_candidates(path: Path) -> list[SourceCandidate]:
    payload = json.loads(path.read_text())
    records = payload.get("sources", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("source file must contain a JSON list or a {\"sources\": [...]} object")
    candidates = [SourceCandidate(**record) for record in records]
    for index, candidate in enumerate(candidates, start=1):
        if not candidate.evidence_id:
            candidate.evidence_id = f"s{index}"
    return candidates


def _load_event_from_csv(path: Path, event_id: str) -> dict[str, Any]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_id") == event_id:
                return dict(row)
    raise ValueError(f"event_id not found in {path}: {event_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer source-discovery utilities")
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--events", type=Path, default=DEFAULT_OUTPUT_DIR / "structured_transfers.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-queries", type=int, default=6)
    parser.add_argument("--unresolved-field", action="append", default=None)
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit("Only --dry-run is supported by this CLI entry point.")

    event = _load_event_from_csv(args.events, args.event_id)
    plans = build_query_plan(event, max_queries=args.max_queries, unresolved_fields=args.unresolved_field)
    print(json.dumps([plan.to_dict() for plan in plans], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
