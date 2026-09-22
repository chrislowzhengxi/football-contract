"""Deterministic post-extraction audit for Stage 2.

The 43-event run showed that a model marked `extracted` is not the same as an
event correctly resolved. Three failure modes appeared, and none of them are
caught by schema validation:

  * direction reversal - the payer and recipient of a fee swapped;
  * mechanism inference - "definitive acquisition" read as a purchase
    obligation, which the summary itself admitted was not stated;
  * event bleed - a later buyback populating an earlier loan-return event.

Prompt wording alone cannot be trusted to prevent these, so every rule here is
enforced in code after the model replies. A rule that fires downgrades the
field rather than discarding the extraction, and records why.
"""
from __future__ import annotations

import re
from typing import Any

STATUS_VALUES = {
    "disclosed_yes", "disclosed_no", "partially_disclosed", "undisclosed",
    "not_found", "conflicting_sources", "not_applicable",
}

# Values models actually emitted instead of the vocabulary, mapped to the
# nearest legal status. Anything unrecognised becomes not_found, never a claim.
STATUS_REPAIR = {
    "confirmed": "disclosed_yes", "yes": "disclosed_yes", "true": "disclosed_yes",
    "disclosed": "disclosed_yes", "established": "disclosed_yes",
    "no": "disclosed_no", "false": "disclosed_no", "absent": "disclosed_no",
    "none": "not_found", "unknown": "not_found", "not_disclosed": "undisclosed",
    "not_reported": "not_found", "unavailable": "not_found", "null": "not_found",
    "partial": "partially_disclosed", "partially": "partially_disclosed",
    "conflicting": "conflicting_sources", "conflict": "conflicting_sources",
    "n/a": "not_applicable", "na": "not_applicable", "inapplicable": "not_applicable",
}

# Fields that assert a contractual MECHANISM. These may never be inferred; the
# evidence must name the mechanism, so a quote is required and is checked.
MECHANISM_FIELDS = {
    "purchase_option": (
        "option to buy", "purchase option", "option to purchase", "buy option",
        "diritto di riscatto", "opzione", "opción de compra", "opcion de compra",
        "opção de compra", "option d'achat", "kaufoption",
        "satın alma opsiyonu", "opsiyon", "koopoptie", "optie tot koop",
    ),
    "purchase_obligation": (
        "obligation to buy", "obligation to purchase", "mandatory purchase",
        "obligatory", "obbligo di riscatto", "obbligo", "obligación de compra",
        "compra obligatoria", "obrigação de compra", "obligation d'achat",
        "kaufpflicht", "zorunlu satın alma", "verplichte",
    ),
    "buy_back": (
        "buy-back", "buyback", "buy back", "recompra", "riacquisto",
        "contro-riscatto", "controriscatto", "clause de rachat", "rückkauf",
        "ruckkauf", "geri alma", "terugkoop",
    ),
    "sell_on": (
        "sell-on", "sell on", "future sale", "futura rivendita", "mais-valia",
        "percentagem", "porcentaje de una futura", "pourcentage à la revente",
        "weiterverkaufsbeteiligung", "sonraki satış", "doorverkoop",
        "percentage of any future", "share of any future",
    ),
    "release_or_purchase_clause": (
        "release clause", "clausola rescissoria", "cláusula de rescisión",
        "clausula de rescision", "cláusula de rescisão", "clause libératoire",
        "ausstiegsklausel", "serbest kalma", "buyout clause", "minimum fee release",
    ),
    "obligation_trigger": (
        "trigger", "triggered", "condition", "conditional", "appearances",
        "promotion", "relegation", "survival", "qualification", "condizione",
        "şart", "sarta bagli", "şarta bağlı", "bedingung", "condición",
    ),
}

CLAIM_STATUSES = {"disclosed_yes", "partially_disclosed", "conflicting_sources"}


def normalise_status(raw: Any) -> tuple[str, bool]:
    """(status, was_repaired). Unrecognised values collapse to not_found."""
    if not isinstance(raw, str):
        return "not_found", raw is not None
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if value in STATUS_VALUES:
        return value, False
    return STATUS_REPAIR.get(value, "not_found"), True


def _norm_club(name: Any) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower())
    text = re.sub(r"\b(fc|cf|sc|ac|afc|ssc|club|calcio|cp|as|ec|sv|vfl|bv|rc)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _club_matches(claim: Any, club: Any) -> bool:
    a, b = _norm_club(claim), _norm_club(club)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    at, bt = set(a.split()), set(b.split())
    return bool(at & bt)


def audit_payload(payload: dict, event: dict) -> dict:
    """Apply every rule to one raw model payload. Returns an audit record and
    mutates `payload` so downstream consumers see only surviving claims."""
    from_club, to_club = event.get("from_club_name"), event.get("to_club_name")
    sources = payload.get("sources") or []
    target_ids = {
        s.get("evidence_id") for s in sources
        if isinstance(s, dict) and s.get("describes_target_event") is True
    }
    known_ids = {s.get("evidence_id") for s in sources if isinstance(s, dict)}
    # A payload that never marks scope is not trusted to have considered it.
    scope_declared = any(
        isinstance(s, dict) and "describes_target_event" in s for s in sources
    )

    violations: list[str] = []
    downgraded: list[str] = []
    supported: list[str] = []
    direction_flags: list[str] = []

    for name in list(MECHANISM_FIELDS) + [
        "transfer_type", "transfer_fee", "loan_fee", "add_ons", "parent_contract_expiry"
    ]:
        field = payload.get(name)
        if not isinstance(field, dict):
            continue

        status, repaired = normalise_status(field.get("status"))
        if repaired:
            violations.append(f"{name}: invalid status {field.get('status')!r} repaired to {status}")
        field["status"] = status
        if status not in CLAIM_STATUSES:
            continue

        ids = [i for i in (field.get("evidence_ids") or []) if i]
        unknown = [i for i in ids if i not in known_ids]
        if unknown:
            violations.append(f"{name}: cites unknown evidence {unknown}")
            ids = [i for i in ids if i in known_ids]
            field["evidence_ids"] = ids

        # Rule 1 - event lock. A claim resting only on evidence about another
        # transaction cannot describe this one.
        if not ids:
            field["status"] = "not_found"
            downgraded.append(f"{name}: claim without evidence_ids")
            continue
        if scope_declared and target_ids and not (set(ids) & target_ids):
            field["status"] = "not_found"
            field["audit_note"] = "evidence describes a different transaction"
            downgraded.append(f"{name}: event mismatch, evidence is about another transaction")
            continue
        if field.get("describes_target_event") is False:
            field["status"] = "not_found"
            downgraded.append(f"{name}: model marked this claim as not about the target event")
            continue

        # Rule 2 - no inference of mechanism. The wording must name it.
        if name in MECHANISM_FIELDS:
            quote = " ".join(str(field.get(k) or "") for k in
                             ("quote", "description", "value", "additional_condition")).lower()
            if not any(term in quote for term in MECHANISM_FIELDS[name]):
                field["status"] = "not_found"
                field["audit_note"] = "mechanism not explicitly stated in the cited wording"
                downgraded.append(f"{name}: mechanism inferred, not quoted")
                continue

        # Rule 3 - direction of money.
        amount = field.get("amount") or field.get("price")
        if amount:
            payer, recipient = field.get("payer"), field.get("recipient")
            if not payer or not recipient:
                field["direction_established"] = False
                direction_flags.append(f"{name}: amount without payer/recipient")
            else:
                known_pair = (
                    (_club_matches(payer, to_club) and _club_matches(recipient, from_club))
                    or (_club_matches(payer, from_club) and _club_matches(recipient, to_club))
                )
                if not known_pair:
                    field["direction_established"] = False
                    direction_flags.append(
                        f"{name}: payer/recipient ({payer} -> {recipient}) are not this event's clubs")
                else:
                    field["direction_established"] = True

        supported.append(name)

    mechanism_supported = [f for f in supported if f in MECHANISM_FIELDS]
    return {
        "violations": violations,
        "downgraded": downgraded,
        "direction_flags": direction_flags,
        "fields_supported": supported,
        "mechanism_fields_supported": mechanism_supported,
        "scope_declared": scope_declared,
        "target_evidence_count": len(target_ids),
        "related_event_evidence": payload.get("related_event_evidence") or [],
    }


def classify_resolution(audit: dict, status: str, pages_usable: int) -> str:
    """One of the five outcome classes the audit reports against."""
    if status == "extraction_failed":
        return "extraction_failed"
    if status != "extracted":
        return "unresolved_insufficient_evidence"
    if audit["fields_supported"]:
        # Strong means a genuine contractual mechanism survived every rule.
        if audit["mechanism_fields_supported"] and not audit["direction_flags"]:
            return "resolved_strong"
        return "resolved_partial"
    if audit["downgraded"] and any("event mismatch" in d for d in audit["downgraded"]):
        return "unresolved_event_mismatch"
    if pages_usable:
        return "unresolved_insufficient_evidence"
    return "unresolved_insufficient_evidence"
