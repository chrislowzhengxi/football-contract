"""Source classes, domain registry and admissibility for Stage 2.

The earlier pilot leaned on general search plus national media. This module
adds the classes that were never tested - financial and regulatory disclosure,
local club media, transfer specialists, structured providers - and tags every
retrieved page with a `source_class` so the report can rank classes by contract
fields established per page rather than by assumption.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

# Tier 1 can establish a term alone. Tier 2 needs corroboration. Tier 3 is
# discovery context only and can never establish a contractual term.
SOURCE_CLASS_TIER = {
    "official_club_buying": 1,
    "official_club_selling": 1,
    "official_club_other": 1,
    "financial_disclosure": 1,
    "regulatory_filing": 1,
    "league_or_federation": 1,
    "national_media_major": 2,
    "local_club_media": 2,
    "sports_business_media": 2,
    "transfer_specialist": 2,
    "structured_data_provider": 2,
    "archived_retrospective": 2,
    "aggregator_low_quality": 3,
    "transfermarkt_backbone": 3,   # our own backbone; never independent evidence
    "unknown": 3,
}

# Official club sites for every club in the 26-event pilot. Keyed on the
# lowercase Transfermarkt club name; aliases point at the same domain.
PILOT_CLUB_DOMAINS: dict[str, str] = {
    "besiktas": "bjk.com.tr", "burnley": "burnleyfootballclub.com",
    "galatasaray": "galatasaray.org", "ac milan": "acmilan.com",
    "fc nantes": "fcnantes.com", "stade rennais": "staderennais.com",
    "fc luzern": "fcl.ch", "inter": "inter.it",
    "cagliari": "cagliaricalcio.com", "fiorentina": "acffiorentina.com",
    "trabzonspor": "trabzonspor.org.tr", "crystal palace": "cpfc.co.uk",
    "braga": "scbraga.pt", "málaga cf": "malagacf.com", "malaga cf": "malagacf.com",
    "atalanta": "atalanta.it", "juventus": "juventus.com",
    "independiente": "clubaindependiente.com", "rosario central": "rosariocentral.com",
    "girona": "gironafc.cat", "wolves": "wolves.co.uk",
    "chelsea": "chelseafc.com", "werder bremen": "werder.de",
    "ssc napoli": "sscnapoli.it", "spartak moscow": "spartak.com",
    "bayern munich": "fcbayern.com", "torino": "torinofc.it",
    "atlético": "atleticodemadrid.com", "atletico": "atleticodemadrid.com",
    "fulham": "fulhamfc.com", "real betis": "realbetisbalompie.es",
    "tottenham": "tottenhamhotspur.com", "shabab al-ahli": "shababalahliclub.ae",
    "al-nasr": "alnasrsc.ae", "nott'm forest": "nottinghamforest.co.uk",
    "olympiacos": "olympiacos.org", "r. strasbourg": "rcstrasbourgalsace.fr",
    "psg": "psg.fr", "qatar sc": "qatarsc.com", "sevilla fc": "sevillafc.es",
    "benfica": "slbenfica.pt", "ec bahia": "esporteclubebahia.com.br",
    "neom sc": "neomsc.com", "mönchengladbach": "borussia.de",
    "monchengladbach": "borussia.de", "leipzig": "rbleipzig.com",
    "ajax": "ajax.nl", "sunderland": "safc.com",
}

# Clubs that are publicly listed and therefore file disclosures itemising
# transfer consideration, contingent payments and sell-on rights. This is the
# class the earlier pilot never tested, and the pilot contains three Turkish
# clubs whose filings all go through KAP.
LISTED_CLUB_DISCLOSURE: dict[str, tuple[str, str]] = {
    "galatasaray": ("kap.org.tr", "Borsa Istanbul - KAP material-event disclosure"),
    "besiktas": ("kap.org.tr", "Borsa Istanbul - KAP material-event disclosure"),
    "trabzonspor": ("kap.org.tr", "Borsa Istanbul - KAP material-event disclosure"),
    "fenerbahce": ("kap.org.tr", "Borsa Istanbul - KAP material-event disclosure"),
    "juventus": ("juventus.com", "Borsa Italiana - investor relations / price-sensitive releases"),
    "ajax": ("ajax.nl", "Euronext Amsterdam - regulated information"),
    "benfica": ("slbenfica.pt", "Euronext Lisbon - SL Benfica SAD"),
    "borussia dortmund": ("bvb.de", "Frankfurt - BVB investor relations"),
}

# Domains whose class we know without fetching.
DOMAIN_CLASS: dict[str, str] = {
    "kap.org.tr": "regulatory_filing",
    "bvb.de": "financial_disclosure",
    "transfermarkt.com": "transfermarkt_backbone",
    "transfermarkt.co.uk": "transfermarkt_backbone",
    "transfermarkt.de": "transfermarkt_backbone",
    # national media
    "bbc.co.uk": "national_media_major", "theguardian.com": "national_media_major",
    "telegraph.co.uk": "national_media_major", "thetimes.co.uk": "national_media_major",
    "lequipe.fr": "national_media_major", "gazzetta.it": "national_media_major",
    "corrieredellosport.it": "national_media_major", "tuttosport.com": "national_media_major",
    "marca.com": "national_media_major", "as.com": "national_media_major",
    "mundodeportivo.com": "national_media_major", "sport.es": "national_media_major",
    "kicker.de": "national_media_major", "bild.de": "national_media_major",
    "record.pt": "national_media_major", "abola.pt": "national_media_major",
    "ad.nl": "national_media_major", "telegraaf.nl": "national_media_major",
    "hurriyet.com.tr": "national_media_major", "sabah.com.tr": "national_media_major",
    "sozcu.com.tr": "national_media_major", "milliyet.com.tr": "national_media_major",
    "sport24.gr": "national_media_major", "clarin.com": "national_media_major",
    "lanacion.com.ar": "national_media_major", "ole.com.ar": "national_media_major",
    "globo.com": "national_media_major", "ge.globo.com": "national_media_major",
    # sports business
    "swissramble.substack.com": "sports_business_media",
    "footballbenchmark.com": "sports_business_media",
    "sportico.com": "sports_business_media", "sportbusiness.com": "sports_business_media",
    "calcioefinanza.it": "sports_business_media", "2playbook.com": "sports_business_media",
    # transfer specialists
    "fabrizioromano.com": "transfer_specialist", "theathletic.com": "transfer_specialist",
    "di-marzio.com": "transfer_specialist", "skysports.com": "transfer_specialist",
    "gianlucadimarzio.com": "transfer_specialist",
    # structured providers
    "api-football.com": "structured_data_provider", "sportmonks.com": "structured_data_provider",
    "statsbomb.com": "structured_data_provider", "opta.com": "structured_data_provider",
    "soccerment.com": "structured_data_provider",
    # league / federation
    "premierleague.com": "league_or_federation", "legaseriea.it": "league_or_federation",
    "bundesliga.com": "league_or_federation", "ligue1.com": "league_or_federation",
    "laliga.com": "league_or_federation", "tff.org": "league_or_federation",
    "fifa.com": "league_or_federation", "uefa.com": "league_or_federation",
    # archives
    "web.archive.org": "archived_retrospective",
}

LOW_QUALITY_MARKERS = (
    "sportskeeda", "givemesport", "footballtransfers", "caughtoffside",
    "90min.com", "tribalfootball", "football-italia", "onefootball",
    "msn.com", "yahoo.com", "pinterest", "reddit.com", "wikipedia.org",
)


def domain_of(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def club_key(name: str | None) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip().casefold()


def official_domain(club_name: str | None) -> str | None:
    key = club_key(club_name)
    if key in PILOT_CLUB_DOMAINS:
        return PILOT_CLUB_DOMAINS[key]
    # fall back to the registry ported from the earlier pilot
    try:
        from ..source_registry import OFFICIAL_CLUB_DOMAINS
        entry = OFFICIAL_CLUB_DOMAINS.get(key)
        return entry.domain if entry else None
    except Exception:
        return None


def disclosure_venue(club_name: str | None) -> tuple[str, str] | None:
    """(domain, description) where this club's regulated filings appear."""
    return LISTED_CLUB_DISCLOSURE.get(club_key(club_name))


def classify(url: str, from_club: str | None, to_club: str | None) -> str:
    """Assign a source_class to a URL, using the event's two clubs for context."""
    domain = domain_of(url)
    if not domain:
        return "unknown"
    buying, selling = official_domain(to_club), official_domain(from_club)
    if buying and domain.endswith(buying):
        return "official_club_buying"
    if selling and domain.endswith(selling):
        return "official_club_selling"
    for known, source_class in DOMAIN_CLASS.items():
        if domain == known or domain.endswith("." + known):
            return source_class
    if any(marker in domain for marker in LOW_QUALITY_MARKERS):
        return "aggregator_low_quality"
    # An unrecognised club-looking domain in the right country is most often
    # local coverage; treated as Tier 2 and never allowed to stand alone.
    if re.search(r"\b(fc|cf|sc|ac|club)\b", domain) or domain.endswith((".gr", ".tr", ".pt", ".ar", ".br")):
        return "local_club_media"
    return "unknown"


def tier_of(source_class: str) -> int:
    return SOURCE_CLASS_TIER.get(source_class, 3)


def can_establish_alone(source_class: str) -> bool:
    return tier_of(source_class) == 1
