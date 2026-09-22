"""Strict evidence-identity gate.

A document earns a place in an extraction packet only if it is ABOUT the
family we are researching. Mere mention is not enough, and that distinction is
load-bearing: the first demonstration dataset gave Santiago Gimenez a Riccardo
Sottil announcement and an Alvaro Morata loan report, because both articles
name Gimenez in a passing aside ("Morata was replaced at Milan by Santiago
Gimenez, who signed from Feyenoord"). Every identity check upstream passed -
the full name really was present - and a contract term was then read off a
sentence about a different transaction entirely.

Three questions decide admission:

  1. Is this player the SUBJECT of the document, or a bystander in it?
  2. Do the family's clubs appear where the player is named?
  3. Is the document's timeframe compatible with the family's?

The subject test leans on a gazetteer built from Stage 1C. When a title names
a known footballer who is not ours, and ours appears nowhere in the title, URL
or opening paragraph, the document is about someone else. That is a fact about
the document, not a heuristic about relevance.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .cache_index import PARTICLES, fold

CANONICAL_CSV = Path("data/outputs/rebuild/stage1c_canonical_transfers.csv")

# How much of the body counts as the lead. Long enough to cover a headline,
# standfirst and opening paragraph; short enough that a sidebar does not qualify.
LEAD_CHARS = 700
# Window, in characters, inside which a club name corroborates a player mention.
CLUB_WINDOW = 320
# A document naming the player this many times or more is plausibly about them
# even when the title is generic ("Official announcements, July 2022").
SUBJECT_MENTION_FLOOR = 4
# Seasons of slack around the family. Retrospective coverage is explicitly
# allowed, so this only excludes documents written long BEFORE the family.
YEARS_BEFORE = 1


def _tok(s: str) -> list[str]:
    return [t for t in fold(s).split() if t]


def name_parts(name: str) -> tuple[str, str] | None:
    """(forename, surname) as folded tokens, or None if unusable."""
    toks = [t for t in _tok(name) if len(t) > 2 and t not in PARTICLES]
    if len(toks) < 2:
        return None
    return toks[0], toks[-1]


@lru_cache(maxsize=1)
def player_gazetteer(path: str = str(CANONICAL_CSV)) -> dict[str, set]:
    """surname -> {forenames}, over every player Stage 1C knows about.

    Used only to recognise that a title is about somebody else. Absence from
    the gazetteer never rejects a document.
    """
    import pandas as pd
    try:
        names = pd.read_csv(path, usecols=["player_name"], low_memory=False)
    except (FileNotFoundError, ValueError):
        return {}
    g: dict[str, set] = defaultdict(set)
    for n in names.player_name.dropna().unique():
        p = name_parts(str(n))
        if p:
            g[p[1]].add(p[0])
    return dict(g)


def named_players_in(text: str, gaz: dict[str, set] | None = None) -> set[tuple[str, str]]:
    """Known footballers named in a short piece of text, e.g. a headline."""
    gaz = player_gazetteer() if gaz is None else gaz
    if not gaz:
        return set()
    toks = _tok(text)
    found = set()
    for i, t in enumerate(toks):
        fores = gaz.get(t)
        if not fores:
            continue
        for j in range(max(0, i - 3), i):
            if toks[j] in fores:
                found.add((toks[j], t))
                break
    return found


def _years_in(text: str) -> set[int]:
    return {int(y) for y in re.findall(r"\b(19[89]\d|20[0-4]\d)\b", text or "")}


@dataclass
class IdentityVerdict:
    admit: bool
    strength: str                       # strong | medium | rejected
    reasons: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)


def assess(title: str | None, url: str, text: str, player_name: str,
           clubs, aliases=(), years=(), gaz: dict[str, set] | None = None
           ) -> IdentityVerdict:
    """Should this document enter the extraction packet for this family?"""
    parts = name_parts(player_name)
    if parts is None:
        return IdentityVerdict(False, "rejected", ["player name has no usable forename+surname"])
    fore, sur = parts

    title_f = fold(title or "")
    url_f = fold((url or "").replace("-", " ").replace("/", " ").replace("_", " "))
    body_f = fold(text or "")
    lead_f = body_f[:LEAD_CHARS]

    name_variants = [(fore, sur)]
    for a in aliases or []:
        p = name_parts(a)
        if p:
            name_variants.append(p)

    def has_full_name(hay: str) -> bool:
        for f, s in name_variants:
            start = 0
            while True:
                i = hay.find(s, start)
                if i < 0:
                    break
                if f in hay[max(0, i - 60): i + 60]:
                    return True
                start = i + len(s)
        return False

    in_title = has_full_name(title_f)
    # A URL slug rarely carries a forename adjacent to the surname, so a slug
    # match needs the surname plus either the forename or a family club.
    club_tokens = {t for c in clubs or [] for t in _tok(str(c)) if len(t) > 3}
    in_url = sur in url_f and (fore in url_f or bool(club_tokens & set(url_f.split())))
    in_lead = has_full_name(lead_f)
    in_body = has_full_name(body_f)

    mentions = body_f.count(sur)
    signals = {"in_title": in_title, "in_url": in_url, "in_lead": in_lead,
               "in_body": in_body, "surname_mentions": mentions}

    if not in_body:
        return IdentityVerdict(False, "rejected",
                               ["document never names this player in full"], signals)

    # --- competing subject ---------------------------------------------------
    others = {p for p in named_players_in(title or "", gaz)
              if p[1] != sur and p not in name_variants}
    signals["other_players_in_title"] = sorted(f"{a} {b}" for a, b in others)
    # A lead mention does NOT rescue a document whose headline is about someone
    # else. In a short article the aside that "Morata was replaced by Santiago
    # Gimenez, who signed from Feyenoord" falls inside the opening paragraph,
    # and it is still an aside. Only the title or the URL slug establishes that
    # a document is about our player.
    if others and not (in_title or in_url):
        other = sorted(others)[0]
        return IdentityVerdict(
            False, "rejected",
            [f"document is about {other[0]} {other[1]}, not {player_name}; "
             "our player appears only in passing"], signals)

    # --- club corroboration --------------------------------------------------
    club_near = False
    if club_tokens:
        hay = " ".join([title_f, url_f, body_f])
        for f, s in name_variants:
            start = 0
            while not club_near:
                i = hay.find(s, start)
                if i < 0:
                    break
                window = hay[max(0, i - CLUB_WINDOW): i + CLUB_WINDOW]
                if club_tokens & set(window.split()):
                    club_near = True
                start = i + len(s)
    else:
        club_near = True
    signals["family_club_near_name"] = club_near

    # --- timeframe -----------------------------------------------------------
    doc_years = _years_in(" ".join([title_f, url_f, body_f[:6000]]))
    year_ok = True
    if years and doc_years:
        year_ok = any(y >= min(years) - YEARS_BEFORE for y in doc_years)
    signals["years_in_document"] = sorted(doc_years)[:12]
    signals["year_compatible"] = year_ok

    reasons = []
    if not club_near:
        reasons.append("no family club appears near any mention of the player")
    if not year_ok:
        reasons.append(
            f"document predates the family (mentions only {sorted(doc_years)[:4]}, "
            f"family begins {min(years)})")

    if reasons:
        return IdentityVerdict(False, "rejected", reasons, signals)

    if in_title or in_url:
        return IdentityVerdict(True, "strong", ["player named in title or URL"], signals)
    if in_lead:
        return IdentityVerdict(True, "strong", ["player named in the opening paragraph"], signals)
    if mentions >= SUBJECT_MENTION_FLOOR and not others:
        return IdentityVerdict(True, "medium",
                               [f"player named {mentions}x with a family club nearby"], signals)
    return IdentityVerdict(
        False, "rejected",
        [f"player mentioned only in passing ({mentions}x, not in title, URL or lead)"],
        signals)


# --- club recognition, for the family-scope gate -----------------------------

# Tokens that appear in hundreds of club names and identify nothing on their own.
CLUB_STOPWORDS = {
    "fc", "cf", "sc", "ac", "as", "ss", "us", "sv", "tsv", "vfb", "vfl", "fsv",
    "club", "clube", "cd", "ca", "afc", "cfc", "sk", "fk", "nk", "hk", "ik",
    "united", "city", "town", "county", "rovers", "athletic", "atletico",
    "sporting", "real", "deportivo", "olympique", "racing", "u17", "u18", "u19",
    "u20", "u21", "u23", "ii", "iii", "youth", "reserves", "academy", "team",
    "the", "and", "de", "del", "der", "van", "von", "calcio", "futbol", "football",
    "without", "retired", "unknown", "free", "agent", "career", "break",
}


@lru_cache(maxsize=1)
def club_gazetteer(path: str = str(CANONICAL_CSV)) -> dict[str, set]:
    """distinctive token -> {club names containing it}.

    Used to notice that an evidence span is about a transaction involving a
    club outside the family - the signal that separates "this quote describes
    our deal" from "this quote describes the sale three years later".
    """
    import pandas as pd
    try:
        df = pd.read_csv(path, usecols=["from_club_name", "to_club_name"],
                         low_memory=False)
    except (FileNotFoundError, ValueError):
        return {}
    names = pd.unique(pd.concat([df.from_club_name, df.to_club_name]).dropna())
    g: dict[str, set] = defaultdict(set)
    for n in names:
        clean = str(n)
        for t in _tok(clean):
            if len(t) > 3 and t not in CLUB_STOPWORDS:
                g[t].add(clean)
    return dict(g)


def club_tokens_of(clubs) -> set[str]:
    """Distinctive tokens for a set of club names."""
    return {t for c in clubs or [] for t in _tok(str(c))
            if len(t) > 3 and t not in CLUB_STOPWORDS}


def clubs_mentioned(text: str, gaz: dict[str, set] | None = None,
                    exclude: set | None = None) -> set[str]:
    """Distinctive club tokens present in a piece of text.

    Only capitalised words count. Stage 1C contains 18,651 club names, among
    them Right to Dream, Para Hills, 9 de Julio, Asia Euro Utd and Wolayta
    Dicha, so a plain token match treats "right", "para", "julio", "euro" and
    "dicha" as clubs and then concludes that a perfectly good Spanish or
    English quotation is about some other team. A club is a proper noun; the
    running words that collide with one are not.
    """
    gaz = club_gazetteer() if gaz is None else gaz
    if not gaz:
        return set()
    exclude = exclude or set()
    out = set()
    for raw in re.findall(r"[^\W\d_][\w'’-]*", text or "", flags=re.UNICODE):
        if not raw[:1].isupper():
            continue
        t = fold(raw)
        if len(t) > 3 and t in gaz and t not in exclude and t not in CLUB_STOPWORDS:
            out.add(t)
    return out


def name_tokens_of(name: str) -> set[str]:
    """Folded tokens of a player's name, for excluding them from club matching."""
    return {t for t in _tok(name) if len(t) > 3}


# Shortest prefix at which two spellings of one club are treated as the same.
CLUB_PREFIX = 4


def club_tokens_overlap(a: set, b: set, prefix: int = CLUB_PREFIX) -> set:
    """Club tokens in `a` that name a club also named in `b`.

    Clubs are written many ways and the variants often share no whole token:
    Stade Rennais is "Rennes" in running French, Internazionale is "Inter",
    Manchester City is "Man City". Exact token matching therefore concluded
    that "Rennes a fait marcher sa clause de rachat" described a transaction
    outside a family whose clubs were Stade Rennais and FC Nantes, and threw
    away a correctly reported buy-back. Matching on a shared prefix keeps the
    check useful without that failure; it only ever makes the gate more
    permissive, and the document-level identity gate has already established
    that the page is about the right player.
    """
    out = set()
    for t in a:
        for u in b:
            if t == u or (len(t) >= prefix and len(u) >= prefix
                          and (t.startswith(u[:prefix]) or u.startswith(t[:prefix]))):
                out.add(t)
                break
    return out
