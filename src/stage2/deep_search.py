"""Deep, high-recall discovery for the Stage 2 diagnostic experiment.

The bounded pilot asked "can we find this cheaply". This module asks the
different question "does this information exist publicly at all", so it trades
cost for recall: many more query families, many more results per query, no
domain whitelist at the discovery step, and raw page content pulled straight
from the search provider so a separate fetch cannot silently drop a page.

Discoverability and admissibility are deliberately separated. An unknown domain
is allowed to be *found* here; whether it may *establish* a contractual term is
decided later, by source class.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen

TAVILY_ENDPOINT = "https://api.tavily.com/search"

# Language of the football press that would cover each club. Broader than the
# bounded pilot's map because discovery, not cost, is the constraint here.
CLUB_LANG = {
    "trabzonspor": "tr", "besiktas": "tr", "galatasaray": "tr", "fenerbahce": "tr",
    "genoa": "it", "atalanta": "it", "spezia calcio": "it", "inter": "it",
    "lecce": "it", "ac milan": "it", "parma": "it", "sampdoria": "it", "roma": "it",
    "cagliari": "it", "fiorentina": "it", "spal": "it", "juventus": "it",
    "juve next gen": "it", "pisa": "it", "monza": "it", "napoli": "it",
    "hellas verona": "it", "brescia": "it", "pescara": "it", "torino": "it",
    "cesena": "it", "us palermo": "it", "pavia": "it", "cremonese": "it",
    "reggiana": "it", "frosinone": "it", "sassuolo": "it", "bologna": "it",
    "fc nantes": "fr", "stade rennais": "fr", "clermont foot": "fr", "psg": "fr",
    "r. strasbourg": "fr", "braga": "pt", "benfica": "pt", "porto": "pt",
    "málaga cf": "es", "malaga cf": "es", "girona": "es", "atlético": "es",
    "sevilla fc": "es", "real betis": "es", "peñarol": "es", "necaxa": "es",
    "werder bremen": "de", "frankfurt": "de", "freiburg": "de", "leverkusen": "de",
    "arm. bielefeld": "de", "sturm graz": "de", "leipzig": "de", "bayern munich": "de",
    "mönchengladbach": "de", "fc luzern": "de", "atromitos": "el", "paok": "el",
    "olympiacos": "el", "ajax": "nl", "genk": "nl", "rsc anderlecht": "nl",
    "club brugge": "nl", "cfr cluj": "ro", "sepsi osk": "ro",
}

# One entry per query family. Each is a distinct way of asking, not a rephrase.
MECHANISM_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "purchase_option": {
        "en": ("option to buy", "purchase option"), "it": ("diritto di riscatto",),
        "es": ("opción de compra",), "pt": ("opção de compra",),
        "fr": ("option d'achat",), "de": ("Kaufoption",),
        "tr": ("satın alma opsiyonu",), "nl": ("optie tot koop",),
        "el": ("οψιόν αγοράς",), "ro": ("opțiune de cumpărare",),
    },
    "purchase_obligation": {
        "en": ("obligation to buy", "mandatory purchase"), "it": ("obbligo di riscatto",),
        "es": ("obligación de compra",), "pt": ("obrigação de compra",),
        "fr": ("obligation d'achat",), "de": ("Kaufpflicht",),
        "tr": ("zorunlu satın alma",), "nl": ("verplichte koopoptie",),
        "el": ("υποχρεωτική αγορά",), "ro": ("obligație de cumpărare",),
    },
    "buy_back": {
        "en": ("buy-back clause",), "it": ("controriscatto", "diritto di recompra"),
        "es": ("opción de recompra",), "pt": ("opção de recompra",),
        "fr": ("clause de rachat",), "de": ("Rückkaufoption",),
        "tr": ("geri alma opsiyonu",), "nl": ("terugkoopoptie",),
        "el": ("ρήτρα επαναγοράς",), "ro": ("clauză de răscumpărare",),
    },
    "sell_on": {
        "en": ("sell-on clause", "percentage of future sale"),
        "it": ("percentuale sulla futura rivendita",),
        "es": ("porcentaje de una futura venta",), "pt": ("percentagem de mais-valia",),
        "fr": ("pourcentage à la revente",), "de": ("Weiterverkaufsbeteiligung",),
        "tr": ("sonraki satış payı",), "nl": ("doorverkooppercentage",),
        "el": ("ποσοστό μεταπώλησης",), "ro": ("procent din transfer",),
    },
    "add_ons": {
        "en": ("add-ons bonuses", "contingent payments"), "it": ("bonus",),
        "es": ("variables bonus",), "pt": ("bónus",), "fr": ("bonus",),
        "de": ("Boni",), "tr": ("bonuslar",), "nl": ("bonussen",),
        "el": ("μπόνους",), "ro": ("bonusuri",),
    },
}


@dataclass(frozen=True)
class DeepQuery:
    family: str
    query: str
    language: str
    rationale: str


def _lang(event: dict) -> tuple[str, str, str, str]:
    """(lang_to, club_to, lang_from, club_from)."""
    def key(n): return re.sub(r"\s+", " ", str(n or "")).strip().casefold()
    to_c, from_c = str(event.get("to_club") or event.get("to_club_name") or ""), \
                   str(event.get("from_club") or event.get("from_club_name") or "")
    return CLUB_LANG.get(key(to_c), "en"), to_c, CLUB_LANG.get(key(from_c), "en"), from_c


def build_deep_queries(event: dict) -> list[DeepQuery]:
    """8-13 materially different query families for one event."""
    player = str(event.get("player") or event.get("player_name") or "").strip()
    lang_to, to_club, lang_from, from_club = _lang(event)
    date = str(event.get("transfer_date") or "")
    year = date[:4]
    prev_year = str(int(year) - 1) if year.isdigit() else year
    surname = player.split()[-1] if player else ""
    langs = list(dict.fromkeys([lang_to, lang_from, "en"]))
    out: list[DeepQuery] = []
    add = lambda f, q, l, r: out.append(DeepQuery(f, q, l, r))

    # 1-2. Both official clubs, named rather than site-restricted, so archived
    #      and syndicated copies of the announcement are reachable too.
    add("official_buying", f'"{player}" {to_club} official statement signing {year}', "en",
        "buying club's own announcement, including archived copies")
    add("official_selling", f'"{player}" {from_club} official announcement transfer {year}', "en",
        "selling club's announcement, which often names sell-on and add-ons")

    # 3-7. One family per contractual mechanism. Prefer a NON-English language
    #      when either club has one: English phrasing is already covered by the
    #      other families, whereas the local press is where "diritto di
    #      riscatto" or "satın alma opsiyonu" is actually written. Without this
    #      preference an English-speaking counterpart club silently suppressed
    #      every local-language query.
    local = [(l, c) for l, c in ((lang_to, to_club), (lang_from, from_club)) if l != "en"]
    for mech, terms in MECHANISM_TERMS.items():
        lang, club = (local[0] if local else ("en", to_club or from_club))
        if lang not in terms:
            lang, club = "en", to_club or from_club
        add(f"mechanism_{mech}", f'"{player}" {club} {terms[lang][0]} {year}', lang,
            f"explicit {mech} wording in {lang}")
    # English phrasing for the two mechanisms most likely to attach to a loan.
    if local:
        for mech in ("purchase_option", "purchase_obligation"):
            add(f"mechanism_{mech}_en",
                f'"{player}" {from_club} {to_club} {MECHANISM_TERMS[mech]["en"][0]} {year}',
                "en", f"English phrasing for {mech}, since local-language took the slot above")
    # A second local language, when the two clubs speak different ones.
    if len(local) > 1:
        lang2, club2 = local[1]
        if lang2 in MECHANISM_TERMS["purchase_option"]:
            add("mechanism_purchase_option_lang2",
                f'"{player}" {club2} {MECHANISM_TERMS["purchase_option"][lang2][0]} {year}',
                lang2, "the other club's language")

    # 8. Contract duration.
    add("contract_duration", f'"{player}" {to_club} contract until signed until {year}', "en",
        "contract length and expiry at the time of the move")

    # 9. Money that is not a headline fee: loan fees, termination, compensation.
    add("payment_terms",
        f'"{player}" {from_club} {to_club} loan fee termination payment compensation {year}', "en",
        "loan fee, termination settlement or compensation rather than a transfer fee")

    # 10. Regulated and financial disclosure, for any club that files.
    add("regulatory_financial",
        f'{to_club} OR {from_club} annual report financial statement player transfers {year}', "en",
        "annual report or regulated filing itemising transfer consideration")

    # 11. Legal and tribunal record. Disputes force terms into the public record.
    add("legal_tribunal",
        f'"{player}" {from_club} {to_club} FIFA CAS tribunal dispute ruling transfer', "en",
        "FIFA/CAS decisions, which quote contractual clauses verbatim")

    # 12-13. Retrospective reporting. A later article that explains the original
    #        agreement is valid evidence, and the bounded pilot never asked for it.
    add("retrospective", f'"{player}" {from_club} {to_club} deal explained how the transfer worked', "en",
        "retrospective explainer describing the original agreement")
    lang_r = next((l for l in langs if l in MECHANISM_TERMS["purchase_option"]), "en")
    club_r = to_club if lang_r == lang_to else from_club
    add("retrospective_local",
        f'"{surname}" {club_r} {MECHANISM_TERMS["purchase_option"][lang_r][0]} accordo {prev_year} {year}',
        lang_r, "local-language retrospective covering the original agreement")
    return out


def tavily_deep_search(query: str, api_key: str, max_results: int = 20,
                       timeout: float = 45) -> dict:
    """Advanced-depth Tavily search that also returns page text.

    `include_raw_content` matters: the bounded pilot lost 28 pages to fetch
    failures, and content returned inline here cannot fail that way.
    """
    body = {
        "api_key": api_key, "query": query, "search_depth": "advanced",
        "topic": "general", "max_results": max_results,
        "include_answer": False, "include_raw_content": True,
    }
    request = Request(TAVILY_ENDPOINT, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


# --- chunking -------------------------------------------------------------

CLAUSE_TERMS = [t for terms in MECHANISM_TERMS.values() for group in terms.values() for t in group]
CLAUSE_TERMS += ["release clause", "clausola rescissoria", "cláusula de rescisión",
                 "riscatto", "obbligo", "opzione", "bonservis", "opsiyon",
                 "contract until", "signed until", "contratto fino", "sell-on",
                 "buy-back", "recompra", "termination", "compensation", "settlement"]


def chunk_document(text: str, event: dict, width: int = 2200,
                   max_chunks: int = 12) -> list[dict]:
    """Multiple evidence chunks, not one window around the player's name.

    A fixed player-centred window was the bounded pilot's blind spot: in a long
    filing the clause often sits in a different section from the name, in a
    table, or on a later page. So we cut three ways - around the player, around
    each clause term, and around the two clubs - then merge overlaps.
    """
    if not text:
        return []
    low = text.lower()
    player = str(event.get("player") or event.get("player_name") or "")
    surname = player.split()[-1].lower() if player else ""
    anchors: list[tuple[int, str]] = []

    def find_all(needle: str, kind: str, limit: int) -> None:
        start, n = 0, 0
        while n < limit:
            i = low.find(needle, start)
            if i < 0:
                break
            anchors.append((i, kind))
            start, n = i + len(needle), n + 1

    if surname:
        find_all(surname, "player", 6)
    for term in CLAUSE_TERMS:
        find_all(term.lower(), f"clause:{term}", 2)
    for club_key in ("from_club", "to_club", "from_club_name", "to_club_name"):
        club = str(event.get(club_key) or "").lower().split()
        if club:
            find_all(club[0], "club", 2)

    if not anchors:
        return [{"kind": "head", "text": text[: width * 2], "offset": 0}]
    anchors.sort()
    spans: list[list] = []
    for pos, kind in anchors:
        a, b = max(0, pos - width // 3), min(len(text), pos + width)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
            spans[-1][2].add(kind.split(":")[0])
        else:
            spans.append([a, b, {kind.split(":")[0]}])
    spans = spans[:max_chunks]
    return [{"kind": "+".join(sorted(k)), "text": text[a:b], "offset": a} for a, b, k in spans]


def pdf_pages_of_interest(pages: list[str], event: dict) -> list[int]:
    """Indices of PDF pages naming the player or carrying clause vocabulary,
    plus the page either side, because tables often continue across a break."""
    player = str(event.get("player") or event.get("player_name") or "")
    surname = player.split()[-1].lower() if player else ""
    hits: set[int] = set()
    for i, page in enumerate(pages):
        low = (page or "").lower()
        if (surname and surname in low) or any(t.lower() in low for t in CLAUSE_TERMS):
            hits.update({i - 1, i, i + 1})
    return sorted(i for i in hits if 0 <= i < len(pages))
