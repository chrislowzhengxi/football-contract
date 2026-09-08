from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRegistryEntry:
    domain: str
    source_type: str
    tier: int
    countries: tuple[str, ...]
    notes: str


OFFICIAL_CLUB_DOMAINS: dict[str, SourceRegistryEntry] = {
    "benfica": SourceRegistryEntry("slbenfica.pt", "official", 1, ("Portugal",), "Official club website."),
    "sl benfica": SourceRegistryEntry("slbenfica.pt", "official", 1, ("Portugal",), "Official club website."),
    "porto": SourceRegistryEntry("fcporto.pt", "official", 1, ("Portugal",), "Official club website."),
    "fc porto": SourceRegistryEntry("fcporto.pt", "official", 1, ("Portugal",), "Official club website."),
    "braga": SourceRegistryEntry("scbraga.pt", "official", 1, ("Portugal",), "Official club website."),
    "psv": SourceRegistryEntry("psv.nl", "official", 1, ("Netherlands",), "Official club website."),
    "paok": SourceRegistryEntry("paokfc.gr", "official", 1, ("Greece",), "Official club website."),
    "aek athens": SourceRegistryEntry("aekfc.gr", "official", 1, ("Greece",), "Official club website."),
    "como": SourceRegistryEntry("comofootball.com", "official", 1, ("Italy",), "Official club website."),
    "burnley": SourceRegistryEntry("burnleyfootballclub.com", "official", 1, ("England",), "Official club website."),
    "southampton": SourceRegistryEntry("southamptonfc.com", "official", 1, ("England",), "Official club website."),
    "juventus": SourceRegistryEntry("juventus.com", "official", 1, ("Italy",), "Official club website."),
    "psg": SourceRegistryEntry("psg.fr", "official", 1, ("France",), "Official club website."),
    "paris saint-germain": SourceRegistryEntry("psg.fr", "official", 1, ("France",), "Official club website."),
    "rosario central": SourceRegistryEntry("rosariocentral.com", "official", 1, ("Argentina",), "Official club website."),
    "fenerbahce": SourceRegistryEntry("fenerbahce.org", "official", 1, ("Turkey",), "Official club website."),
    "fenerbahçe": SourceRegistryEntry("fenerbahce.org", "official", 1, ("Turkey",), "Official club website."),
    "besiktas": SourceRegistryEntry("bjk.com.tr", "official", 1, ("Turkey",), "Official club website."),
    "beşiktaş": SourceRegistryEntry("bjk.com.tr", "official", 1, ("Turkey",), "Official club website."),
    "aj auxerre": SourceRegistryEntry("aja.fr", "official", 1, ("France",), "Official club website."),
    "auxerre": SourceRegistryEntry("aja.fr", "official", 1, ("France",), "Official club website."),
    "basel": SourceRegistryEntry("fcb.ch", "official", 1, ("Switzerland",), "Official club website."),
    "estrela amadora": SourceRegistryEntry("estrelamadora.pt", "official", 1, ("Portugal",), "Official club website."),
    "al-ain": SourceRegistryEntry("alainclub.ae", "official", 1, ("United Arab Emirates",), "Official club website."),
    "al ain": SourceRegistryEntry("alainclub.ae", "official", 1, ("United Arab Emirates",), "Official club website."),
    "al-ain fc": SourceRegistryEntry("alainclub.ae", "official", 1, ("United Arab Emirates",), "Official club website."),
}


REPUTABLE_TIER2_DOMAINS: dict[str, SourceRegistryEntry] = {
    "reuters.com": SourceRegistryEntry("reuters.com", "major_news", 2, ("Global",), "Wire reporting; useful for official-fee confirmation when terms are explicit."),
    "bbc.com": SourceRegistryEntry("bbc.com", "major_news", 2, ("England", "Global"), "Public-service sports reporting."),
    "theguardian.com": SourceRegistryEntry("theguardian.com", "major_news", 2, ("England", "Global"), "Established newspaper sports reporting."),
    "skysports.com": SourceRegistryEntry("skysports.com", "major_news", 2, ("England",), "Established broadcast sports reporting."),
    "espn.com": SourceRegistryEntry("espn.com", "major_news", 2, ("Global",), "Established sports reporting."),
    "theathletic.com": SourceRegistryEntry("theathletic.com", "major_news", 2, ("Global",), "Subscription sports reporting; snippets may be limited."),
    "apnews.com": SourceRegistryEntry("apnews.com", "major_news", 2, ("Global",), "Wire reporting."),
    "ojogo.pt": SourceRegistryEntry("ojogo.pt", "football_reporting", 2, ("Portugal",), "Portuguese football reporting; claims need explicit source text."),
    "record.pt": SourceRegistryEntry("record.pt", "football_reporting", 2, ("Portugal",), "Portuguese football reporting; claims need explicit source text."),
    "a-bola.pt": SourceRegistryEntry("a-bola.pt", "football_reporting", 2, ("Portugal",), "Portuguese football reporting; claims need explicit source text."),
    "maisfutebol.iol.pt": SourceRegistryEntry("maisfutebol.iol.pt", "football_reporting", 2, ("Portugal",), "Portuguese football reporting; claims need explicit source text."),
    "lequipe.fr": SourceRegistryEntry("lequipe.fr", "football_reporting", 2, ("France",), "French football reporting."),
    "football-italia.net": SourceRegistryEntry("football-italia.net", "football_reporting", 2, ("Italy",), "Italian football specialist in English."),
    "marca.com": SourceRegistryEntry("marca.com", "football_reporting", 2, ("Spain",), "Spanish sports reporting."),
    "as.com": SourceRegistryEntry("as.com", "football_reporting", 2, ("Spain",), "Spanish sports reporting."),
}


def registry_domains() -> dict[str, SourceRegistryEntry]:
    return {**{entry.domain: entry for entry in OFFICIAL_CLUB_DOMAINS.values()}, **REPUTABLE_TIER2_DOMAINS}
