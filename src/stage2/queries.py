"""Field-aware query construction from Stage 1C facts.

Queries are built from the frozen backbone - player, both clubs, date, season,
and the Stage 1 transfer type - so an event is always searched as itself and
never as "some transfer involving this name". Identity is player_id upstream;
the name only ever enters the query string.

Query volume is bounded per event. Which mechanism terms are used depends on
the candidate category, because the question differs: for a fee-bearing loan
return we want to know what the money WAS, for a loan with a fee we want to
know whether an option came with it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sources import club_key, disclosure_venue, official_domain

# Contract vocabulary, per field, per language. Extends the earlier pilot's
# vocabulary with the terms needed for return-leg interpretation.
VOCABULARY: dict[str, dict[str, tuple[str, ...]]] = {
    "purchase_option": {
        "en": ("option to buy", "purchase option"), "it": ("diritto di riscatto",),
        "es": ("opción de compra",), "pt": ("opção de compra",), "fr": ("option d'achat",),
        "de": ("Kaufoption",), "tr": ("satın alma opsiyonu",), "nl": ("optie tot koop",),
        "el": ("οψιόν αγοράς",),
    },
    "purchase_obligation": {
        "en": ("obligation to buy", "mandatory purchase"), "it": ("obbligo di riscatto",),
        "es": ("obligación de compra", "compra obligatoria"), "pt": ("obrigação de compra",),
        "fr": ("obligation d'achat",), "de": ("Kaufpflicht",),
        "tr": ("zorunlu satın alma",), "nl": ("verplichte koopoptie",),
        "el": ("υποχρεωτική αγορά",),
    },
    "obligation_trigger": {
        "en": ("obligation trigger", "appearances clause", "conditions met"),
        "it": ("condizione obbligo riscatto",), "es": ("condición obligación compra",),
        "pt": ("condição obrigação compra",), "fr": ("condition obligation achat",),
        "de": ("Bedingung Kaufpflicht",), "tr": ("zorunlu satın alma şartı",),
    },
    "add_ons": {
        "en": ("add-ons", "bonuses", "performance related"), "it": ("bonus",),
        "es": ("variables", "bonus"), "pt": ("bónus",), "fr": ("bonus",),
        "de": ("Boni",), "tr": ("bonuslar",), "nl": ("bonussen",),
    },
    "sell_on": {
        "en": ("sell-on clause", "sell-on percentage"), "it": ("percentuale futura rivendita",),
        "es": ("porcentaje de futura venta",), "pt": ("percentagem futura venda", "mais-valia"),
        "fr": ("pourcentage à la revente",), "de": ("Weiterverkaufsbeteiligung",),
        "tr": ("sonraki satış payı",), "nl": ("doorverkooppercentage",),
    },
    "buy_back": {
        "en": ("buy-back clause",), "it": ("diritto di recompra", "contro-riscatto"),
        "es": ("opción de recompra",), "pt": ("opção de recompra",),
        "fr": ("clause de rachat",), "de": ("Rückkaufoption",), "tr": ("geri alma opsiyonu",),
    },
    "parent_contract_expiry": {
        "en": ("contract until", "signed until"), "it": ("contratto fino al",),
        "es": ("contrato hasta",), "pt": ("contrato até",), "fr": ("contrat jusqu'en",),
        "de": ("Vertrag bis",), "tr": ("sözleşme",), "nl": ("contract tot",),
    },
    "release_or_purchase_clause": {
        "en": ("release clause", "exercised the option"), "it": ("clausola rescissoria", "riscattato"),
        "es": ("cláusula de rescisión", "ejerció la opción"), "pt": ("cláusula de rescisão",),
        "fr": ("clause libératoire",), "de": ("Ausstiegsklausel",),
        "tr": ("serbest kalma bedeli",),
    },
}

# Which fields each candidate category should lead with.
CATEGORY_FIELDS: dict[str, tuple[str, ...]] = {
    "fee_bearing_loan_return": ("release_or_purchase_clause", "purchase_obligation",
                                "purchase_option", "obligation_trigger"),
    "loan_with_fee": ("purchase_option", "purchase_obligation", "obligation_trigger", "add_ons"),
    "high_value_undisclosed": ("add_ons", "sell_on", "purchase_option"),
    "control_ordinary_permanent": ("add_ons", "sell_on", "buy_back"),
}

CLUB_LANGUAGE: dict[str, str] = {
    "besiktas": "tr", "galatasaray": "tr", "trabzonspor": "tr",
    "ac milan": "it", "inter": "it", "cagliari": "it", "fiorentina": "it",
    "atalanta": "it", "juventus": "it", "ssc napoli": "it", "torino": "it",
    "fc nantes": "fr", "stade rennais": "fr", "psg": "fr", "r. strasbourg": "fr",
    "girona": "es", "atlético": "es", "real betis": "es", "sevilla fc": "es",
    "málaga cf": "es", "braga": "pt", "benfica": "pt", "ec bahia": "pt",
    "werder bremen": "de", "bayern munich": "de", "mönchengladbach": "de",
    "leipzig": "de", "fc luzern": "de", "ajax": "nl", "olympiacos": "el",
    "independiente": "es", "rosario central": "es",
}


@dataclass(frozen=True)
class Query:
    event_id: str
    query: str
    intent: str          # what this query is trying to reach
    target_field: str | None
    language: str
    expected_source_class: str

    def as_dict(self) -> dict:
        return {"event_id": self.event_id, "query": self.query, "intent": self.intent,
                "target_field": self.target_field, "language": self.language,
                "expected_source_class": self.expected_source_class}


def language_for(event: dict) -> tuple[str, str]:
    """(language code, the club that speaks it). Pairing the local-language
    term with an English-speaking club produces nonsense queries."""
    for club in (event.get("to_club_name"), event.get("from_club_name")):
        code = CLUB_LANGUAGE.get(club_key(club))
        if code:
            return code, str(club)
    return "en", str(event.get("to_club_name") or event.get("from_club_name") or "")


def build_queries(event: dict, max_queries: int = 4) -> list[Query]:
    """A small, ordered set of queries. Cheapest-highest-precision first."""
    event_id = event["event_id"]
    player = str(event.get("player_name", "")).strip()
    from_club = str(event.get("from_club_name", "")).strip()
    to_club = str(event.get("to_club_name", "")).strip()
    year = str(event.get("transfer_date", ""))[:4]
    category = event.get("candidate_category", "")
    fields = CATEGORY_FIELDS.get(category, ("purchase_option", "add_ons", "sell_on"))
    lead, second = fields[0], fields[1] if len(fields) > 1 else fields[0]
    language, language_club = language_for(event)
    out: list[Query] = []

    # 1. Buying club's own announcement - highest precision available.
    buying = official_domain(to_club)
    if buying:
        out.append(Query(event_id, f'site:{buying} "{player}"', "official announcement, buying club",
                         None, "en", "official_club_buying"))
    # 2. Regulated filing, where a club is listed. Untested class; the pilot
    #    contains three Borsa Istanbul clubs whose filings go through KAP.
    for club in (to_club, from_club):
        venue = disclosure_venue(club)
        if venue and len(out) < max_queries:
            out.append(Query(event_id, f'site:{venue[0]} "{player}" {club}',
                             f"regulated disclosure ({venue[1]})", None, "en",
                             "regulatory_filing" if venue[0] == "kap.org.tr" else "financial_disclosure"))
            break
    # 3. Field-aware general query in English on the leading mechanism.
    terms = VOCABULARY[lead]["en"][0]
    if len(out) < max_queries:
        out.append(Query(event_id, f'"{player}" "{from_club}" "{to_club}" {terms} {year}',
                         f"field query: {lead}", lead, "en", "national_media_major"))
    # 4. Local-language query on the second mechanism, or English if no local.
    if len(out) < max_queries:
        phrases = VOCABULARY[second]
        code = language if language in phrases else "en"
        club_for_term = language_club if code != "en" else (to_club or from_club)
        out.append(Query(event_id, f'"{player}" {club_for_term} {phrases[code][0]}',
                         f"field query: {second} ({code})", second, code,
                         "local_club_media" if code != "en" else "national_media_major"))
    # 5. Selling club, if there is room left.
    selling = official_domain(from_club)
    if selling and len(out) < max_queries:
        out.append(Query(event_id, f'site:{selling} "{player}"', "official announcement, selling club",
                         None, "en", "official_club_selling"))
    return out[:max_queries]
