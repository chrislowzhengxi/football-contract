"""Provider-independent query planning.

Emits plain query strings with metadata. Knows nothing about any search vendor,
and must never import one. Plans are built from an EVENT FAMILY, not a single
Transfermarkt row, so a query can target the leg that actually carries the
terms - the outbound loan or the onward sale - rather than an administrative
return leg that has none.
"""
from __future__ import annotations

import json
from pathlib import Path

from .types import SearchQuery

VOCAB_PATH = Path("data/config/contract_clause_vocabulary.json")
_CACHE: dict | None = None

CLUB_LANG_PATH = None   # languages come from deep_search.CLUB_LANG via _club_lang


def vocabulary() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(VOCAB_PATH.read_text()) if VOCAB_PATH.exists() else {"mechanisms": {}}
    return _CACHE


def _term(mechanism: str, lang: str) -> str | None:
    m = vocabulary().get("mechanisms", {}).get(mechanism, {})
    terms = m.get(lang) or []
    return terms[0] if terms else None


def _club_lang(club: str) -> str:
    from ..deep_search import CLUB_LANG        # data table only, no transport
    import re
    return CLUB_LANG.get(re.sub(r"\s+", " ", str(club or "")).strip().casefold(), "en")


def plan_for_family(family_rows, escalate: bool = False,
                    listed_domains: tuple = ()) -> list[SearchQuery]:
    """4-6 normal queries for a family, plus escalation queries on request.

    The research leg is chosen first: terms live on the negotiated leg. For a
    family whose return is administrative, we ask about the original loan and
    the onward sale instead of the return.
    """
    rows = family_rows.sort_values("transfer_date")
    anchor = rows[rows.is_family_anchor].iloc[0] if "is_family_anchor" in rows and \
        rows.is_family_anchor.any() else rows.iloc[-1]
    fam_id = anchor.get("event_family_id")
    player = str(anchor.player_name)

    # Prefer a leg that can carry terms.
    PRIORITY = ("original_loan", "permanent_transfer", "third_party_sale", "loan_return")
    target = None
    for role in PRIORITY:
        m = rows[rows.event_role == role] if "event_role" in rows else rows.iloc[0:0]
        if len(m):
            target = m.iloc[0]
            break
    if target is None:
        target = anchor

    from_club, to_club = str(target.from_club_name), str(target.to_club_name)
    year = str(target.transfer_date)[:4]
    lang_from, lang_to = _club_lang(from_club), _club_lang(to_club)
    local = next((l for l in (lang_to, lang_from) if l != "en"), None)
    local_club = to_club if local == lang_to else from_club

    Q: list[SearchQuery] = []

    def add(fam, q, lang, prio, why, prefer=()):
        Q.append(SearchQuery(query=q, language=lang, query_family=fam,
                             event_id=str(target.event_id), event_family_id=fam_id,
                             preferred_domains=tuple(prefer), priority=prio, rationale=why))

    add("official_buying", f'"{player}" {to_club} official announcement signing {year}', "en", 1,
        "buying club's own statement, including archived copies")
    add("official_selling", f'"{player}" {from_club} official statement transfer {year}', "en", 1,
        "selling club's statement, where sell-on and add-ons often appear")

    if local:
        t = _term("purchase_option", local) or _term("purchase_obligation", local)
        if t:
            add("mechanism_local", f'"{player}" {local_club} {t} {year}', local, 1,
                f"explicit mechanism wording in {local}")
    add("mechanism_en", f'"{player}" {from_club} {to_club} '
        f'{_term("purchase_option","en") or "option to buy"} '
        f'{_term("purchase_obligation","en") or "obligation to buy"} {year}', "en", 1,
        "English mechanism phrasing")
    add("contract_duration", f'"{player}" {to_club} contract until signed until {year}', "en", 2,
        "contract length and expiry at the time of the move")
    if listed_domains:
        add("regulatory", f'{to_club} OR {from_club} annual report player transfers {year}',
            "en", 2, "regulated filing itemising transfer consideration",
            prefer=tuple(listed_domains))

    if escalate:
        for mech in ("buy_back", "sell_on", "add_ons", "termination_compensation",
                     "transfer_proceeds_share"):
            for lang in filter(None, (local, "en")):
                t = _term(mech, lang)
                if t:
                    add(f"esc_{mech}_{lang}",
                        f'"{player}" {local_club if lang == local else to_club} {t} {year}',
                        lang, 3, f"escalation: {mech} in {lang}")
        add("esc_legal", f'"{player}" {from_club} {to_club} FIFA CAS tribunal ruling transfer',
            "en", 3, "tribunal decisions quote clauses verbatim")
        add("esc_retrospective",
            f'"{player}" {from_club} {to_club} deal explained how the transfer worked',
            "en", 3, "retrospective explainer describing the original agreement")
    return Q


def plan_to_jsonl(plans: dict[str, list[SearchQuery]]) -> list[str]:
    out = []
    for fam_id, queries in plans.items():
        for q in queries:
            d = q.to_dict()
            d["event_family_id"] = fam_id
            out.append(json.dumps(d, ensure_ascii=False))
    return out
