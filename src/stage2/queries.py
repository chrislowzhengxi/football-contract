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


def build_queries(event: dict, max_queries: int = 6) -> list[Query]:
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

    # Priority order corrected after the 26-event pilot. Measured yields were:
    # national media 100%, uncatalogued domains 23%, official club pages 7%.
    # The single richest document found was a listed club's annual report,
    # which itemises fee, options, instalments and sell-on for every transfer
    # of the season. Official club pages are kept, but last and only one.

    # 1-2. Regulated filings first, for clubs that are publicly listed.
    for club in (to_club, from_club):
        venue = disclosure_venue(club)
        if not venue:
            continue
        cls = "regulatory_filing" if venue[0] == "kap.org.tr" else "financial_disclosure"
        out.append(Query(event_id, f'site:{venue[0]} "{player}" {club}',
                         f"regulated disclosure ({venue[1]})", None, "en", cls))
        out.append(Query(event_id,
                         f'site:{venue[0]} {club} faaliyet raporu futbolcu transfer {year}'
                         if venue[0] == "kap.org.tr"
                         else f'site:{venue[0]} {club} annual report transfer {year}',
                         "club annual report (itemised transfer disclosures)", None,
                         "tr" if venue[0] == "kap.org.tr" else "en", cls))
        break

    # 3-4. Field-aware media queries on the two leading mechanisms.
    for mechanism in (lead, second):
        if len(out) >= max_queries:
            break
        out.append(Query(event_id,
                         f'"{player}" "{from_club}" "{to_club}" '
                         f'{VOCABULARY[mechanism]["en"][0]} {year}',
                         f"field query: {mechanism}", mechanism, "en", "national_media_major"))

    # 5. Local-language query, paired with the club that speaks the language.
    if len(out) < max_queries:
        phrases = VOCABULARY[lead]
        code = language if language in phrases else "en"
        club_for_term = language_club if code != "en" else (to_club or from_club)
        out.append(Query(event_id, f'"{player}" {club_for_term} {phrases[code][0]} {year}',
                         f"field query: {lead} ({code})", lead, code,
                         "local_club_media" if code != "en" else "national_media_major"))

    # 6. Open deal-structure query - the phrasing media actually use.
    if len(out) < max_queries:
        out.append(Query(event_id,
                         f'"{player}" {from_club} {to_club} transfer deal structure fee clause',
                         "deal structure, open web", None, "en", "national_media_major"))

    # 7. Official club page, as supporting evidence only. Deliberately one
    # query, not two: 74 official pages in the pilot yielded 6 usable ones.
    buying = official_domain(to_club) or official_domain(from_club)
    if buying and len(out) < max_queries:
        out.append(Query(event_id, f'site:{buying} "{player}"',
                         "official club announcement (supporting evidence)", None, "en",
                         "official_club_buying"))
    return out[:max_queries]
