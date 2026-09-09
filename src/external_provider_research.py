from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_DIR


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "source_fusion"
TRISTATE_VALUES = {"true", "false", "unknown"}
SUPPORT_STATUSES = {"supported", "not_supported", "unknown"}
DATA_FORMATS = {"structured", "semi_structured", "text_only", "unknown"}
REQUESTED_FIELDS = (
    "transfer_type",
    "transfer_fee",
    "loan_fee",
    "purchase_option",
    "purchase_obligation",
    "parent_contract_expiry",
    "years_contract_left",
    "add_ons",
    "obligation_trigger",
    "release_or_purchase_clause",
    "sell_on",
    "buy_back",
    "salary",
)
FIELD_WEIGHTS = {
    "transfer_fee": 16,
    "loan_fee": 10,
    "purchase_option": 9,
    "purchase_obligation": 9,
    "parent_contract_expiry": 8,
    "years_contract_left": 5,
    "transfer_type": 5,
    "add_ons": 4,
    "obligation_trigger": 3,
    "release_or_purchase_clause": 3,
    "salary": 3,
    "sell_on": 2,
    "buy_back": 2,
}
ACCESS_OVERRIDES: dict[str, dict[str, str]] = {
    "FootyStats API": {
        "free_tier": "unknown",
    },
    "SportsDataIO Soccer API": {
        "authentication_required": "true",
    },
    "StatsBomb": {
        "free_tier": "unknown",
        "paid_only": "unknown",
        "pricing_public": "unknown",
    },
    "Stats Perform / Opta": {
        "free_tier": "unknown",
        "paid_only": "unknown",
    },
    "Wyscout API": {
        "free_tier": "unknown",
        "paid_only": "unknown",
        "pricing_public": "unknown",
        "authentication_required": "true",
    },
    "FIFA Transfer Reports / TMS aggregate data": {
        "authentication_required": "false",
    },
    "TransferRoom API": {
        "free_tier": "false",
        "paid_only": "true",
        "authentication_required": "true",
    },
}


def _tri(value: str) -> str:
    if value not in TRISTATE_VALUES:
        raise ValueError(f"invalid tri-state value: {value}")
    return value


@dataclass(frozen=True)
class ProviderCapability:
    provider: str
    field: str
    support_status: str = "unknown"
    data_format: str = "unknown"
    historical_available: str = "unknown"
    player_level: str = "unknown"
    transfer_level: str = "unknown"
    contract_level: str = "unknown"
    evidence_url: str | None = None
    evidence_type: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.field not in REQUESTED_FIELDS:
            raise ValueError(f"unknown provider capability field: {self.field}")
        if self.support_status not in SUPPORT_STATUSES:
            raise ValueError(f"invalid support status: {self.support_status}")
        if self.data_format not in DATA_FORMATS:
            raise ValueError(f"invalid data format: {self.data_format}")
        _tri(self.historical_available)
        _tri(self.player_level)
        _tri(self.transfer_level)
        _tri(self.contract_level)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderAccessModel:
    provider: str
    api_available: str = "unknown"
    bulk_export_available: str = "unknown"
    free_tier: str = "unknown"
    paid_only: str = "unknown"
    pricing_public: str = "unknown"
    academic_access: str = "unknown"
    authentication_required: str = "unknown"
    research_use_known: str = "unknown"
    redistribution_terms_known: str = "unknown"
    notes: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.api_available,
            self.bulk_export_available,
            self.free_tier,
            self.paid_only,
            self.pricing_public,
            self.academic_access,
            self.authentication_required,
            self.research_use_known,
            self.redistribution_terms_known,
        ):
            _tri(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderCandidate:
    provider_name: str
    provider_url: str
    documentation_url: str
    provider_category: str
    fields_claimed: list[str] = field(default_factory=list)
    historical_data: str = "unknown"
    historical_start_if_known: str | None = None
    competition_coverage: str = "unknown"
    country_coverage: str = "unknown"
    player_level: str = "unknown"
    transfer_level: str = "unknown"
    contract_level: str = "unknown"
    api_available: str = "unknown"
    bulk_export_available: str = "unknown"
    csv_available: str = "unknown"
    database_access: str = "unknown"
    academic_access: str = "unknown"
    free_tier: str = "unknown"
    paid_only: str = "unknown"
    pricing_public: str = "unknown"
    authentication_required: str = "unknown"
    stable_player_id: str = "unknown"
    stable_transfer_id: str = "unknown"
    provenance_available: str = "unknown"
    redistribution_terms_known: str = "unknown"
    research_use_known: str = "unknown"
    last_checked: str = field(default_factory=lambda: date.today().isoformat())
    evidence_urls: list[str] = field(default_factory=list)
    notes: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.historical_data,
            self.player_level,
            self.transfer_level,
            self.contract_level,
            self.api_available,
            self.bulk_export_available,
            self.csv_available,
            self.database_access,
            self.academic_access,
            self.free_tier,
            self.paid_only,
            self.pricing_public,
            self.authentication_required,
            self.stable_player_id,
            self.stable_transfer_id,
            self.provenance_available,
            self.redistribution_terms_known,
            self.research_use_known,
        ):
            _tri(value)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fields_claimed"] = json.dumps(self.fields_claimed, ensure_ascii=False)
        payload["evidence_urls"] = json.dumps(self.evidence_urls, ensure_ascii=False)
        return payload


@dataclass(frozen=True)
class ProviderResearchResult:
    candidate: ProviderCandidate
    access: ProviderAccessModel
    capabilities: list[ProviderCapability]
    score: float = 0.0
    incremental_value_score: float = 0.0
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "access": self.access.to_dict(),
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "score": self.score,
            "incremental_value_score": self.incremental_value_score,
            "notes": self.notes,
        }


def _cap(provider: str, field_name: str, support: str, fmt: str, evidence: str, notes: str, historical: str = "unknown", player: str = "unknown", transfer: str = "unknown", contract: str = "unknown") -> ProviderCapability:
    return ProviderCapability(provider, field_name, support, fmt, historical, player, transfer, contract, evidence, "official_documentation", notes)


def _unknown_caps(provider: str, evidence: str) -> list[ProviderCapability]:
    return [_cap(provider, field_name, "unknown", "unknown", evidence, "No public documentation located for this field.") for field_name in REQUESTED_FIELDS]


def capability_map(provider: str, evidence: str, overrides: dict[str, tuple[str, str, str, str, str, str, str]] | None = None) -> list[ProviderCapability]:
    overrides = overrides or {}
    rows = []
    for field_name in REQUESTED_FIELDS:
        support, fmt, notes, historical, player, transfer, contract = overrides.get(field_name, ("unknown", "unknown", "No public documentation located for this field.", "unknown", "unknown", "unknown", "unknown"))
        rows.append(_cap(provider, field_name, support, fmt, evidence, notes, historical, player, transfer, contract))
    return rows


def provider_research_results() -> list[ProviderResearchResult]:
    results: list[ProviderResearchResult] = []
    data = [
        (
            ProviderCandidate(
                "Sportmonks Football API",
                "https://www.sportmonks.com/football-api/",
                "https://docs.sportmonks.com/football/endpoints-and-entities/endpoints/transfers/get-all-transfers",
                "football_data_api",
                ["transfers", "transfer amount", "transfer type ids", "player/team ids"],
                "true",
                None,
                "available within subscription",
                "global football coverage, subscription dependent",
                "true",
                "true",
                "false",
                "true",
                "unknown",
                "false",
                "true",
                "unknown",
                "true",
                "unknown",
                "true",
                "true",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=[
                    "https://docs.sportmonks.com/football/endpoints-and-entities/endpoints/transfers/get-all-transfers",
                    "https://www.sportmonks.com/blogs/exploring-club-transfers-with-sportmonks-football-api-and-dart/",
                ],
                notes="Strongest structured transfer candidate located; docs show transfer id, player/team ids, date, type_id, completed, and amount.",
            ),
            capability_map("Sportmonks Football API", "https://docs.sportmonks.com/football/endpoints-and-entities/endpoints/transfers/get-all-transfers", {
                "transfer_type": ("supported", "structured", "Transfer rows expose type_id and completed/career fields.", "true", "true", "true", "false"),
                "transfer_fee": ("supported", "structured", "Transfer rows expose numeric amount; currency/fee basis needs validation.", "true", "true", "true", "false"),
                "loan_fee": ("unknown", "unknown", "Loan rows may have amount but docs do not establish loan-fee semantics.", "unknown", "true", "true", "false"),
            }),
        ),
        (
            ProviderCandidate(
                "API-Football",
                "https://www.api-football.com/",
                "https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide",
                "football_data_api",
                ["transfers", "transfer type string", "team/player ids"],
                "true",
                None,
                "more than 1200 leagues and cups claimed in official guide",
                "worldwide, provider coverage dependent",
                "true",
                "true",
                "false",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=["https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide"],
                notes="Official guide says /transfers returns player or team transfer history with date, in/out clubs, and type strings such as fee, Free, Loan, or N/A.",
            ),
            capability_map("API-Football", "https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide", {
                "transfer_type": ("supported", "semi_structured", "Type field encodes Free, Loan, N/A, or a fee string.", "true", "true", "true", "false"),
                "transfer_fee": ("supported", "semi_structured", "Fee appears as formatted string when known; not enough docs for currency basis or exactness.", "true", "true", "true", "false"),
                "loan_fee": ("unknown", "unknown", "Loan type is present but separate loan-fee amount is not documented.", "unknown", "true", "true", "false"),
            }),
        ),
        (
            ProviderCandidate(
                "football-data.org",
                "https://www.football-data.org/",
                "https://docs.football-data.org/general/v4/team.html",
                "football_data_api",
                ["team squad", "player contract start/until", "market value"],
                "unknown",
                None,
                "documented competitions/resources, plan dependent",
                "selected football competitions",
                "true",
                "false",
                "true",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "true",
                "unknown",
                "true",
                "true",
                "false",
                "unknown",
                "true",
                "unknown",
                evidence_urls=[
                    "https://docs.football-data.org/general/v4/team.html",
                    "https://docs.football-data.org/general/v4/policies.html",
                ],
                notes="Useful for current contract-until fields and IDs, but transfer fee/options are not documented.",
            ),
            capability_map("football-data.org", "https://docs.football-data.org/general/v4/team.html", {
                "parent_contract_expiry": ("supported", "structured", "Team resource exposes player contract start/until.", "unknown", "true", "false", "true"),
                "years_contract_left": ("supported", "structured", "Can be derived from documented contract until value if event date alignment is valid.", "unknown", "true", "false", "true"),
            }),
        ),
        (
            ProviderCandidate(
                "Capology API",
                "https://www.capology.com/",
                "https://www.capology.com/documentation/api/v2/get-started/",
                "contract_salary_database",
                ["contracts", "salaries", "renewals", "payrolls", "recaps", "club finances"],
                "true",
                None,
                "website/API coverage; Basic covers selected leagues, Plus all salary leagues",
                "largest leagues and additional salary coverage",
                "true",
                "unknown",
                "true",
                "true",
                "unknown",
                "false",
                "true",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "true",
                "true",
                "unknown",
                "true",
                "true",
                evidence_urls=[
                    "https://www.capology.com/documentation/api/v2/get-started/",
                    "https://capology.com/support/",
                    "https://www.deweydata.io/doi/10-82551-kt4k-2279",
                ],
                notes="High value for salaries/contract info; official support states no transfer fees and no detailed bonus categories.",
            ),
            capability_map("Capology API", "https://www.capology.com/documentation/api/v2/get-started/", {
                "parent_contract_expiry": ("supported", "structured", "Contracts endpoints are documented as player contract information.", "true", "true", "unknown", "true"),
                "years_contract_left": ("supported", "structured", "Derivable from contract endpoint if dates/season are available.", "true", "true", "unknown", "true"),
                "salary": ("supported", "structured", "Salary endpoints and Dewey salary dataset are documented.", "true", "true", "false", "true"),
                "transfer_fee": ("not_supported", "unknown", "Official support says transfer fees are not available.", "false", "unknown", "unknown", "false"),
                "add_ons": ("not_supported", "unknown", "Official support says detailed signing/performance/incentive bonuses are not available.", "false", "unknown", "unknown", "false"),
            }),
        ),
        (
            ProviderCandidate(
                "Dewey Data - Capology dataset",
                "https://www.deweydata.io/",
                "https://www.deweydata.io/doi/10-82551-kt4k-2279",
                "academic_data_marketplace",
                ["Capology professional football salary data"],
                "true",
                None,
                "Capology salary dataset through Dewey",
                "biggest leagues and second divisions listed by dataset page",
                "true",
                "false",
                "true",
                "unknown",
                "true",
                "true",
                "true",
                "true",
                "unknown",
                "true",
                "true",
                "unknown",
                "false",
                "unknown",
                "true",
                "true",
                evidence_urls=[
                    "https://www.deweydata.io/doi/10-82551-kt4k-2279",
                    "https://docs.deweydata.io/docs/faq",
                    "https://www.deweydata.io/pricing",
                ],
                notes="Academic access route for Capology salary data; provider itself is a marketplace, not a football data producer.",
            ),
            capability_map("Dewey Data - Capology dataset", "https://www.deweydata.io/doi/10-82551-kt4k-2279", {
                "salary": ("supported", "structured", "Dataset page identifies Professional Football Salary Data.", "true", "true", "false", "true"),
                "parent_contract_expiry": ("unknown", "unknown", "Dataset title/description do not prove expiry fields in exported schema.", "unknown", "true", "false", "unknown"),
            }),
        ),
        (
            ProviderCandidate(
                "TheSportsDB",
                "https://www.thesportsdb.com/",
                "https://www.thesportsdb.com/documentation",
                "football_data_api",
                ["player contracts", "former teams", "player/team ids"],
                "unknown",
                None,
                "wide sports/leagues, exact football contract coverage undocumented",
                "global sports coverage",
                "true",
                "unknown",
                "true",
                "true",
                "unknown",
                "unknown",
                "true",
                "unknown",
                "true",
                "false",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=["https://www.thesportsdb.com/documentation"],
                notes="Docs expose player contract lookup endpoints, but public field schema for contract details is not specific enough.",
            ),
            capability_map("TheSportsDB", "https://www.thesportsdb.com/documentation", {
                "parent_contract_expiry": ("unknown", "unknown", "Contract lookup exists, but exact fields are not documented in the public page.", "unknown", "true", "unknown", "true"),
                "salary": ("unknown", "unknown", "Contract lookup exists, but salary fields are not established.", "unknown", "true", "unknown", "true"),
            }),
        ),
        (
            ProviderCandidate(
                "Sportradar Soccer API",
                "https://developer.sportradar.com/soccer",
                "https://developer.sportradar.com/soccer/reference/soccer-season-transfers",
                "football_data_api",
                ["season transfers", "role_type", "from/to competitor ids", "player ids"],
                "true",
                None,
                "competition coverage matrix, package dependent",
                "global football coverage, package dependent",
                "true",
                "true",
                "false",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "unknown",
                "true",
                "unknown",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=[
                    "https://developer.sportradar.com/soccer/reference/soccer-season-transfers",
                    "https://developer.sportradar.com/soccer/reference/soccer-extended-faq",
                ],
                notes="Good transfer/event identity provider, but public docs do not expose fee/options/contract terms.",
            ),
            capability_map("Sportradar Soccer API", "https://developer.sportradar.com/soccer/reference/soccer-season-transfers", {
                "transfer_type": ("supported", "structured", "role_type includes player/on_loan/unemployed/other.", "true", "true", "true", "false"),
            }),
        ),
        (
            ProviderCandidate(
                "Wyscout API",
                "https://apidocs.wyscout.com/",
                "https://apidocs.wyscout.com/index.html",
                "football_data_api",
                ["player transfers", "player ids", "team details"],
                "unknown",
                None,
                "customer subscription dependent",
                "global football scouting/data coverage",
                "true",
                "true",
                "false",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "true",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=["https://apidocs.wyscout.com/index.html"],
                notes="Players transfers endpoint exists; exact financial/contract terms were not visible in public docs.",
            ),
            capability_map("Wyscout API", "https://apidocs.wyscout.com/index.html", {
                "transfer_type": ("unknown", "unknown", "Player transfer endpoint exists but public snippet does not establish type/fee fields.", "unknown", "true", "true", "false"),
            }),
        ),
        (
            ProviderCandidate(
                "TransferRoom API",
                "https://www.transferroom.com/",
                "https://knowledge.transferroom.com/how-to-use-the-transferroom-api",
                "transfer_specific_service",
                ["club API", "player/head-coach data intelligence JSON"],
                "unknown",
                None,
                "club-user access",
                "unknown",
                "true",
                "unknown",
                "unknown",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=["https://knowledge.transferroom.com/how-to-use-the-transferroom-api"],
                notes="Potentially valuable transfer-market source, but public API docs do not disclose field-level contract capabilities.",
            ),
            _unknown_caps("TransferRoom API", "https://knowledge.transferroom.com/how-to-use-the-transferroom-api"),
        ),
        (
            ProviderCandidate(
                "SportsDataIO Soccer API",
                "https://sportsdata.io/",
                "https://sportsdata.io/developers/api-documentation/global",
                "football_data_api",
                ["teams", "players", "rosters", "active memberships"],
                "true",
                None,
                "soccer competitions documented in help center",
                "selected global competitions",
                "true",
                "unknown",
                "false",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "true",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=[
                    "https://sportsdata.io/developers/api-documentation/global",
                    "https://sportsdata.io/help/soccer-guide",
                ],
                notes="Likely useful for roster/membership and loans, but not documented for transfer fee or contract clauses.",
            ),
            capability_map("SportsDataIO Soccer API", "https://sportsdata.io/help/soccer-guide", {
                "transfer_type": ("unknown", "unknown", "Help notes active membership can show loan club; transfer-record endpoint not established.", "unknown", "true", "unknown", "false"),
            }),
        ),
        (
            ProviderCandidate(
                "Stats Perform / Opta",
                "https://www.statsperform.com/",
                "https://www.statsperform.com/faqs/stats-perform-faqs-apis-data-delivery/",
                "football_data_api",
                ["sports APIs", "historical statistics", "player/team ids"],
                "true",
                None,
                "enterprise coverage",
                "global sports coverage",
                "true",
                "unknown",
                "unknown",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=["https://www.statsperform.com/faqs/stats-perform-faqs-apis-data-delivery/"],
                notes="Public pages confirm APIs and historical statistics, but not contract/transfer-fee products.",
            ),
            _unknown_caps("Stats Perform / Opta", "https://www.statsperform.com/faqs/stats-perform-faqs-apis-data-delivery/"),
        ),
        (
            ProviderCandidate(
                "CIES Football Observatory",
                "https://football-observatory.com/",
                "https://prod-website.football-observatory.com/-Value-",
                "academic_market_value_source",
                ["estimated transfer values", "player valuation model"],
                "true",
                "2013",
                "70 leagues for value tool",
                "global football sample, league dependent",
                "true",
                "false",
                "false",
                "unknown",
                "unknown",
                "false",
                "true",
                "true",
                "unknown",
                "unknown",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=["https://prod-website.football-observatory.com/-Value-"],
                notes="Estimated player values, not actual transfer contract terms; mostly redundant with market value objectives.",
            ),
            capability_map("CIES Football Observatory", "https://prod-website.football-observatory.com/-Value-", {}),
        ),
        (
            ProviderCandidate(
                "FIFA Transfer Reports / TMS aggregate data",
                "https://inside.fifa.com/transfer-system/transfer-reports",
                "https://inside.fifa.com/transfer-system/transfer-reports/methodology",
                "financial_regulatory_aggregate",
                ["aggregate transfer reports", "transfer type definitions", "fee methodology"],
                "true",
                None,
                "international transfers only",
                "global associations",
                "false",
                "false",
                "false",
                "false",
                "false",
                "false",
                "false",
                "false",
                "false",
                "false",
                "unknown",
                "true",
                "false",
                "true",
                "true",
                "true",
                evidence_urls=["https://inside.fifa.com/transfer-system/transfer-reports/methodology"],
                notes="Excellent aggregate methodology source, but not a public player-level data provider.",
            ),
            capability_map("FIFA Transfer Reports / TMS aggregate data", "https://inside.fifa.com/transfer-system/transfer-reports/methodology", {
                "transfer_type": ("not_supported", "unknown", "Public reports are aggregate and not player-level.", "true", "false", "false", "false"),
                "transfer_fee": ("not_supported", "unknown", "Fee data is aggregate/rounded in reports, not public transfer-level records.", "true", "false", "false", "false"),
            }),
        ),
        (
            ProviderCandidate(
                "FootyStats API",
                "https://footystats.org/api/",
                "https://footystats.org/api/documentations",
                "football_stats_api",
                ["matches", "league stats", "team/player stats"],
                "true",
                None,
                "selected subscribed leagues",
                "league package dependent",
                "true",
                "false",
                "false",
                "true",
                "false",
                "false",
                "false",
                "unknown",
                "true",
                "unknown",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=["https://footystats.org/api/documentations"],
                notes="Focused on match/team/player statistics, not transfer contracts.",
            ),
            capability_map("FootyStats API", "https://footystats.org/api/documentations", {}),
        ),
        (
            ProviderCandidate(
                "StatsBomb",
                "https://statsbomb.com/",
                "https://live-data-api-guide.statsbomb.com/api-reference/",
                "football_stats_api",
                ["match events", "lineups", "squads", "player statistics"],
                "true",
                None,
                "licensed competitions",
                "competition-license dependent",
                "true",
                "false",
                "false",
                "true",
                "unknown",
                "false",
                "false",
                "unknown",
                "true",
                "true",
                "true",
                "unknown",
                "false",
                "unknown",
                "unknown",
                "unknown",
                evidence_urls=["https://live-data-api-guide.statsbomb.com/api-reference/"],
                notes="Excellent event/match data source but no documented contract/transfer-term fields.",
            ),
            capability_map("StatsBomb", "https://live-data-api-guide.statsbomb.com/api-reference/", {}),
        ),
    ]
    for candidate, capabilities in data:
        candidate = replace(candidate, **ACCESS_OVERRIDES.get(candidate.provider_name, {}))
        access = ProviderAccessModel(
            provider=candidate.provider_name,
            api_available=candidate.api_available,
            bulk_export_available=candidate.bulk_export_available,
            free_tier=candidate.free_tier,
            paid_only=candidate.paid_only,
            pricing_public=candidate.pricing_public,
            academic_access=candidate.academic_access,
            authentication_required=candidate.authentication_required,
            research_use_known=candidate.research_use_known,
            redistribution_terms_known=candidate.redistribution_terms_known,
            notes=candidate.notes,
        )
        score = score_provider(candidate, capabilities)
        incremental = incremental_value_score(capabilities)
        results.append(ProviderResearchResult(candidate, access, capabilities, score, incremental, candidate.notes))
    return sorted(results, key=lambda item: (item.score, item.incremental_value_score), reverse=True)


def score_provider(candidate: ProviderCandidate, capabilities: list[ProviderCapability]) -> float:
    total_weight = sum(FIELD_WEIGHTS.values())
    value = 0.0
    for capability in capabilities:
        if capability.support_status != "supported":
            continue
        multiplier = {"structured": 1.0, "semi_structured": 0.75, "text_only": 0.35, "unknown": 0.0}[capability.data_format]
        value += FIELD_WEIGHTS[capability.field] * multiplier
    field_score = 50 * value / total_weight
    automation = 0.0
    automation += 8 if candidate.api_available == "true" else 0
    automation += 4 if candidate.bulk_export_available == "true" or candidate.csv_available == "true" else 0
    automation += 4 if candidate.stable_player_id == "true" else 0
    automation += 4 if candidate.stable_transfer_id == "true" else 0
    history = 15 if candidate.historical_data == "true" else 5 if candidate.historical_data == "unknown" else 0
    access = 0.0
    access += 3 if candidate.free_tier == "true" else 1 if candidate.free_tier == "unknown" else 0
    access += 3 if candidate.academic_access == "true" else 1 if candidate.academic_access == "unknown" else 0
    access += 2 if candidate.pricing_public == "true" else 0
    access += 2 if candidate.paid_only == "false" else 0
    provenance = 5 if candidate.provenance_available == "true" else 2 if candidate.provenance_available == "unknown" else 0
    penalty = 0.0
    if not any(capability.support_status == "supported" for capability in capabilities):
        penalty += 8
    if candidate.transfer_level != "true" and not any(capability.field in {"parent_contract_expiry", "salary"} and capability.support_status == "supported" for capability in capabilities):
        penalty += 4
    return round(max(0.0, field_score + automation + history + access + provenance - penalty), 3)


def incremental_value_score(capabilities: list[ProviderCapability]) -> float:
    value = 0.0
    for capability in capabilities:
        if capability.support_status != "supported":
            continue
        if capability.field in {"transfer_type"}:
            value += 1
        elif capability.field in {"transfer_fee", "loan_fee", "purchase_option", "purchase_obligation", "parent_contract_expiry", "years_contract_left"}:
            value += 3
        elif capability.field == "salary":
            value += 2
        else:
            value += 2.5
    return round(value, 3)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path = OUTPUT_DIR) -> list[ProviderResearchResult]:
    results = provider_research_results()
    candidates = [result.candidate.to_dict() for result in results]
    capabilities = [capability.to_dict() for result in results for capability in result.capabilities]
    access = [result.access.to_dict() for result in results]
    rankings = [
        {
            "rank": index,
            "provider": result.candidate.provider_name,
            "provider_category": result.candidate.provider_category,
            "score": result.score,
            "incremental_value_score": result.incremental_value_score,
            "supported_fields": json.dumps([cap.field for cap in result.capabilities if cap.support_status == "supported"], ensure_ascii=False),
            "unknown_fields": json.dumps([cap.field for cap in result.capabilities if cap.support_status == "unknown"], ensure_ascii=False),
            "evidence_urls": json.dumps(result.candidate.evidence_urls, ensure_ascii=False),
            "notes": result.notes,
        }
        for index, result in enumerate(results, start=1)
    ]
    incremental = []
    for result in results:
        supported = [cap.field for cap in result.capabilities if cap.support_status == "supported"]
        duplicates = [field_name for field_name in supported if field_name in {"transfer_type"}]
        new_fields = [field_name for field_name in supported if field_name not in {"transfer_type"}]
        could_replace = [field_name for field_name in supported if field_name in {"transfer_fee", "loan_fee", "parent_contract_expiry", "years_contract_left"}]
        incremental.append({
            "provider": result.candidate.provider_name,
            "new_fields": json.dumps(new_fields, ensure_ascii=False),
            "partially_new_fields": json.dumps([cap.field for cap in result.capabilities if cap.support_status == "unknown"], ensure_ascii=False),
            "duplicate_fields": json.dumps(duplicates, ensure_ascii=False),
            "could_replace_web_research": json.dumps(could_replace, ensure_ascii=False),
            "highest_value_field": max(new_fields, key=lambda field_name: FIELD_WEIGHTS[field_name], default=""),
            "incremental_value_score": result.incremental_value_score,
            "notes": result.notes,
        })
    automation_rows = [
        {
            "provider": result.candidate.provider_name,
            "api_pagination": "documented" if result.candidate.provider_name in {"Sportradar Soccer API"} else "unknown",
            "rate_limits": "documented" if result.candidate.provider_name in {"football-data.org", "TheSportsDB"} else "unknown",
            "batch_endpoints": "true" if result.candidate.bulk_export_available == "true" else "unknown",
            "stable_ids": "true" if result.candidate.stable_player_id == "true" or result.candidate.stable_transfer_id == "true" else "unknown",
            "historical_query_ability": result.candidate.historical_data,
            "transfer_lookup_ability": result.candidate.transfer_level,
            "player_lookup_ability": result.candidate.player_level,
            "club_lookup_ability": "true" if result.candidate.api_available == "true" else "unknown",
            "export_size_constraints": "unknown",
            "notes": result.candidate.notes,
        }
        for result in results[:8]
    ]
    _write_csv(output_dir / "provider_candidates.csv", candidates)
    _write_csv(output_dir / "provider_field_capabilities.csv", capabilities)
    _write_csv(output_dir / "provider_access_matrix.csv", access)
    _write_csv(output_dir / "provider_rankings.csv", rankings)
    _write_csv(output_dir / "provider_incremental_value.csv", incremental)
    _write_csv(output_dir / "provider_automation_feasibility.csv", automation_rows)
    write_markdown_outputs(output_dir, results)
    return results


def write_markdown_outputs(output_dir: Path, results: list[ProviderResearchResult]) -> None:
    top = results[:5]
    integration_lines = [
        "# Provider Integration Plan",
        "",
        "This is a design document only. No real provider adapter has been implemented.",
        "",
    ]
    for result in top:
        supported = [cap.field for cap in result.capabilities if cap.support_status == "supported"]
        unknown = [cap.field for cap in result.capabilities if cap.support_status == "unknown"]
        integration_lines.extend([
            f"## {result.candidate.provider_name}",
            "",
            f"- Fields it would contribute: {', '.join(supported) if supported else 'none established'}",
            f"- Existing fields it might replace: {', '.join(field_name for field_name in supported if field_name in {'transfer_fee', 'parent_contract_expiry', 'years_contract_left'}) or 'none'}",
            f"- Fields still requiring web research: {', '.join(field_name for field_name in REQUESTED_FIELDS if field_name not in supported and field_name != 'salary')}",
            "- Matching approach: map provider player/team/transfer IDs to the Transfermarkt event using player, from club, to club, date, and transfer direction; then persist provider IDs as provenance.",
            f"- Expected integration complexity: {'medium' if result.candidate.api_available == 'true' else 'high'}",
            f"- Access blocker: {access_blocker(result.candidate)}",
            f"- Build adapter now: {'yes, benchmark first' if result.candidate.provider_name in {'Sportmonks Football API', 'API-Football', 'football-data.org', 'Capology API'} else 'no, research access/fields first'}",
            f"- Unknowns: {', '.join(unknown[:6])}{'...' if len(unknown) > 6 else ''}",
            "",
        ])
    (output_dir / "provider_integration_plan.md").write_text("\n".join(integration_lines))

    architecture = [
        "# Thousand-Transfer Source-Fusion Architecture",
        "",
        "This plan assumes provider access is evaluated before purchase or integration. It uses call-count formulas rather than invented coverage or cost estimates.",
        "",
        "## Shared Stages",
        "",
        "1. Load the Transfermarkt backbone once and keep deterministic identifiers, dates, clubs, market values, and Transfermarkt fee columns separate from independently sourced contract fields.",
        "2. Resolve provider IDs with cached player/team/event lookup tables.",
        "3. Pull structured provider observations for documented fields only.",
        "4. Fuse observations with existing web-research results and preserve provider-level provenance.",
        "5. Identify unresolved high-value fields: transfer_fee, loan_fee, purchase_option, purchase_obligation, parent_contract_expiry, and years_contract_left.",
        "6. Run targeted web fallback only for unresolved high-value fields.",
        "7. Call extraction only where admissible evidence is sufficient.",
        "8. Run field-level QA for missingness, source disagreement, exactness, currency, and transfer-direction errors.",
        "",
        "## 1,000 Events",
        "",
        "- Run Transfermarkt backbone once.",
        "- Query structured providers for all event/player IDs where access exists.",
        "- Fuse deterministic and structured observations.",
        "- Run web research only on unresolved high-value fields.",
        "- Call extraction only when admissible evidence is sufficient.",
        "- Call-count framework: 1,000 x provider_calls_per_event + unresolved_rate x bounded_web_queries + sufficient_rate x extraction_calls.",
        "",
        "## 5,000 Events",
        "",
        "- Batch provider lookups by player/team/date where APIs allow pagination or bulk export.",
        "- Cache provider IDs and source observations before web fallback.",
        "- Process LOW/MEDIUM/HIGH provider-coverage scenarios without assuming exact rates.",
        "- Call-count framework: 5,000 x provider_calls_per_event + unresolved_rate x bounded_web_queries + sufficient_rate x extraction_calls.",
        "",
        "## 10,000 Events",
        "",
        "- Prefer bulk export or paginated provider syncs over one transfer at a time.",
        "- Partition by season/league/club and resume from persisted observation checkpoints.",
        "- QA with field-level missingness, provider disagreement, source-tier distribution, and event-link conflicts.",
        "- Call-count framework: 10,000 x provider_calls_per_event + unresolved_rate x bounded_web_queries + sufficient_rate x extraction_calls.",
        "",
        "## Coverage Scenarios",
        "",
        "- LOW provider coverage: external source resolves only expiry/salary or basic transfer type; targeted web fallback remains central.",
        "- MEDIUM provider coverage: one transfer API resolves many fees/types; web fallback focuses options, obligations, expiry, and sparse clauses.",
        "- HIGH provider coverage: structured providers resolve most high-priority fields; web fallback becomes an audit and exception path.",
        "",
        "## Provider Mix Suggested By This Audit",
        "",
        "- Structured transfer API: Sportmonks or API-Football for transfer_fee and transfer_type, subject to a real-access benchmark of amount/type semantics.",
        "- Contract/salary API: Capology or a Capology dataset route for parent_contract_expiry, years_contract_left, and salary, subject to coverage and licensing.",
        "- Event identity provider: Sportradar can improve transfer identity checks but does not materially improve contract-term coverage from public docs alone.",
        "- Fallback web research: remains necessary for loan fees, purchase options, obligations, triggers, add-ons, sell-on, buy-back, and release clauses.",
    ]
    (output_dir / "thousand_transfer_architecture.md").write_text("\n".join(architecture) + "\n")

    fallback = [
        "# Automated Fallback Architecture",
        "",
        "If no structured provider materially improves contract fields, do not redesign general web search.",
        "",
        "- Transfermarkt supplies the event backbone.",
        "- Known official/media/financial domains supply targeted field discovery.",
        "- Web research is reserved for fees, options, obligations, and expiry.",
        "- Add-ons, sell-on, buy-back, exact triggers, and release clauses remain opportunistic.",
        "- Use bounded query budgets, field-level early stopping, asynchronous queues, cached pages, and explicit missingness preservation.",
        "",
        "## Scalable Workflow",
        "",
        "- Precompute field-specific query plans from club, league, season, transfer type, and language hints.",
        "- Stop searching a field once admissible exact evidence is found and corroboration rules are satisfied.",
        "- Skip extraction for fields with no admissible evidence and write explicit not_found statuses.",
        "- Batch page retrieval and extraction queues independently so failed retrieval does not block deterministic backbone generation.",
        "- Preserve sparse-clause missingness rather than forcing every transfer through human review.",
        "",
        "## Field Treatment",
        "",
        "- Core web fallback fields: transfer_fee, loan_fee, purchase_option, purchase_obligation, parent_contract_expiry, years_contract_left.",
        "- Opportunistic fields: add_ons, obligation_trigger, release_or_purchase_clause, sell_on, buy_back.",
        "- Do not let Transfermarkt fees populate researched_contract_fee without independent admissible evidence.",
    ]
    (output_dir / "automated_fallback_architecture.md").write_text("\n".join(fallback) + "\n")

    decision = [
        "# Provider Decision Tree",
        "",
        "A. A structured provider covers several high-value contract fields -> integrate provider and benchmark on the existing 20.",
        "",
        "B. A provider covers one major field extremely well -> integrate for that field only.",
        "",
        "C. Multiple providers each cover different fields -> use multi-provider fusion and preserve disagreements.",
        "",
        "D. No provider materially improves contract data -> scale Transfermarkt plus targeted web fallback with explicit missingness.",
        "",
        "E. Best source is commercial but promising -> document access requirements and do not purchase automatically.",
    ]
    (output_dir / "provider_decision_tree.md").write_text("\n".join(decision) + "\n")

    summary = [
        "# External Provider Research Summary",
        "",
        "- Search mode: public web documentation only; no Tavily, no Parley, no provider accounts, no purchases.",
        "- Providers investigated: " + ", ".join(result.candidate.provider_name for result in results),
        "- Top ranked providers: " + ", ".join(result.candidate.provider_name for result in results[:5]),
        "- Main finding: source fusion looks materially better than web-search-only only if a structured transfer endpoint is licensed or benchmarked; otherwise, most APIs duplicate identity/match/stat data.",
        "- Scalable structured coverage appears strongest for transfer_type, transfer_fee, parent_contract_expiry, years_contract_left, and salary.",
        "- No public documentation found a scalable structured source for loan_fee, purchase_option, purchase_obligation, obligation_trigger, add_ons, sell_on, buy_back, or release_or_purchase_clause.",
        "- Recommended next engineering action: benchmark Sportmonks Football API through the offline adapter contract first, then decide whether access is worth requesting.",
        "",
        "## Top Provider Field Fit",
        "",
    ]
    for result in results[:5]:
        supported = [cap.field for cap in result.capabilities if cap.support_status == "supported"]
        summary.append(f"- {result.candidate.provider_name}: {', '.join(supported) if supported else 'no supported requested fields'}")
    summary.extend([
        "",
        "## Evidence Sources",
        "",
    ])
    for result in results:
        summary.append(f"- {result.candidate.provider_name}: " + ", ".join(result.candidate.evidence_urls))
    (output_dir / "provider_research_summary.md").write_text("\n".join(summary) + "\n")


def access_blocker(candidate: ProviderCandidate) -> str:
    if candidate.paid_only == "true":
        return "paid or sales-gated access"
    if candidate.authentication_required == "true":
        return "authentication/API key required"
    if candidate.api_available != "true":
        return "no documented API path"
    return "none established from public docs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write offline external provider research matrices")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    results = write_outputs(Path(args.output_dir))
    print(f"Wrote provider research for {len(results)} providers")


if __name__ == "__main__":
    main()
