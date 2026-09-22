"""Event-family extraction (prompt v4) with deterministic validation.

A Parley-populated field is a claim, not a fact. Everything the model returns is
re-checked here against the supplied evidence: the span must actually occur in
the source we sent, the mechanism term must actually appear in the span, the
target event must belong to the family, and the role must be able to carry the
term. Anything that fails is downgraded, with a reason.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .mechanisms import (ALL_MECHANISMS, ECONOMIC_MECHANISM_VALUES, STATUS_VALUES,
                         admissible)
from .money import find_all as find_money, normalise_amount, normalise_percentage
from .discovery import identity
from .discovery.planner import vocabulary

STAGE2_MECHANISM_FIELDS = [
    "purchase_option", "purchase_obligation", "obligation_trigger", "sell_on",
    "buy_back", "add_ons", "release_or_purchase_clause",
    "termination_compensation", "third_party_sale_share",
]
STAGE2_ALL_FIELDS = STAGE2_MECHANISM_FIELDS + ["parent_contract_expiry"]
STAGE1_OWNED = {"transfer_type", "transfer_fee", "loan_fee"}

# An `economic_mechanism` is a claim about what the money IS, so it must rest
# on a mechanism field that survived validation. The first dataset asserted
# `termination_compensation` for Morata and `third_party_sale_share` for
# Sorloth while neither field was supported.
MECHANISM_REQUIRES = {
    "buy_back_exercise": {"buy_back"},
    "option_exercise": {"purchase_option"},
    "obligation_settlement": {"purchase_obligation", "obligation_trigger"},
    "termination_compensation": {"termination_compensation"},
    "third_party_sale_share": {"third_party_sale_share"},
    "sell_on_settlement": {"sell_on"},
}
# Mechanisms that merely name a Stage 1 fact. Permitted, but they are not a
# Stage 2 research finding and never justify a quality grade.
STAGE1_MECHANISMS = {"transfer_fee", "loan_fee", "unknown"}

STATUS_REPAIR = {
    "confirmed": "disclosed_yes", "yes": "disclosed_yes", "true": "disclosed_yes",
    "no": "disclosed_no", "false": "disclosed_no", "unknown": "not_found",
    "none": "not_found", "null": "not_found", "n/a": "not_applicable",
    "partial": "partially_disclosed", "conflicting": "conflicting_sources",
}


def normalise_status(v):
    if v in STATUS_VALUES:
        return v, False
    s = STATUS_REPAIR.get(str(v).strip().lower())
    return (s, True) if s else ("not_found", True)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _numbers_in(text: str) -> set:
    """Monetary magnitudes mentioned in a piece of text, normalised to units."""
    out = set()
    for raw, suffix in re.findall(r"(\d[\d.,]*)\s*(m|mln|million|milioni|millones|millions|k|mila)?",
                                  (text or "").lower()):
        t = raw.rstrip(".,").replace(",", ".") if raw.count(",") == 1 and raw.count(".") == 0 \
            else raw.replace(",", "")
        try:
            val = float(t)
        except ValueError:
            continue
        if suffix in ("m", "mln", "million", "milioni", "millones", "millions"):
            val *= 1_000_000
        elif suffix in ("k", "mila"):
            val *= 1_000
        out.add(round(val, 2))
    return out


def _amount_consistent(amount, span: str) -> bool:
    """Does the recorded amount actually appear in the quoted span?

    Sottil's buy_back was recorded as 12,200,000 while its own span read
    "contro-riscatto per 13,5 milioni". A figure that contradicts its own
    quotation is a false positive however good the source is.
    """
    nums = _numbers_in(span)
    if not nums:
        return True                      # nothing to contradict
    try:
        a = float(amount)
    except (TypeError, ValueError):
        return True
    cands = {a, a / 1_000_000, a / 1_000, a * 1_000_000}
    return any(abs(c - n) <= max(0.05, abs(n) * 0.02) for c in cands for n in nums)


def _span_supported(span: str, eids: list, src_text: dict,
                    overlap: float = 0.75) -> bool:
    """Is this span traceable to text we actually supplied?

    Exact substring first. Failing that, require most of the span's distinctive
    tokens to appear in one source - which survives a chunk boundary cutting a
    quote in half, without letting an invented span through.
    """
    probe = span[:110]
    pool = [src_text.get(e, "") for e in eids] or list(src_text.values())
    if any(probe and probe in t for t in pool):
        return True
    joined = " ".join(src_text.values())
    if probe and probe in joined:
        return True
    toks = [t for t in re.findall(r"[\w']+", span) if len(t) > 3]
    if len(toks) < 4:
        return False
    for t_src in pool + [joined]:
        hit = sum(1 for t in toks if t in t_src)
        if hit / len(toks) >= overlap:
            return True
    return False


def build_family_block(family_rows, anchor_event_id: str) -> dict:
    rows = family_rows.sort_values("transfer_date")
    comps = []
    for _, r in rows.iterrows():
        comps.append({
            "event_id": str(r.event_id), "event_role": str(r.event_role),
            "is_anchor": str(r.event_id) == str(anchor_event_id),
            "transfer_date": str(r.transfer_date),
            "from_club": str(r.from_club_name), "to_club": str(r.to_club_name),
            "stage1_transfer_type": str(r.transfer_type_normalized),
            "direction": (f"the player LEFT {r.from_club_name} and JOINED "
                          f"{r.to_club_name} on {r.transfer_date}"),
        })
    a = rows[rows.event_id == anchor_event_id].iloc[0]
    return {
        "event_family_id": str(a.event_family_id),
        "anchor_event_id": str(anchor_event_id),
        "player_name": str(a.player_name),
        "player_id": str(a.family_player_id),
        "family_interpretation_status": str(a.family_interpretation_status),
        "family_note": str(a.family_notes or ""),
        "component_events": comps,
        "instruction": ("Attribute each term to the component event whose transaction "
                        "the evidence describes, via scoped_to_event_id. The anchor is "
                        "not necessarily where the terms live."),
    }


def _mech_terms(field_name: str) -> list[str]:
    v = vocabulary().get("mechanisms", {})
    key = {"release_or_purchase_clause": "release_clause",
           "third_party_sale_share": "transfer_proceeds_share"}.get(field_name, field_name)
    return [t for ts in v.get(key, {}).values() for t in ts]


@dataclass
class Validation:
    supported: list = field(default_factory=list)
    downgraded: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    status_repairs: list = field(default_factory=list)
    wrong_event: list = field(default_factory=list)
    span_not_found: list = field(default_factory=list)
    direction_missing: list = field(default_factory=list)
    stage1_restated: list = field(default_factory=list)
    tier3_only: list = field(default_factory=list)
    amount_conflicts: list = field(default_factory=list)
    composite_spans: list = field(default_factory=list)
    scoped: dict = field(default_factory=dict)
    # --- added by the data-integrity hardening pass ---
    money_rescaled: list = field(default_factory=list)
    money_unsupported: list = field(default_factory=list)
    out_of_family: list = field(default_factory=list)
    related_event_evidence: list = field(default_factory=list)
    conflicting: dict = field(default_factory=dict)
    normalised: dict = field(default_factory=dict)
    mechanism_rejected: str = ""
    mechanism_value: str = ""


def _is_composite(span: str) -> bool:
    """A span stitched together from several places is not a quotation.

    Colombo's buy-back span read "contropzione; buyback clause; Milan activated
    the buyback clause; Milan eserciterà il controriscatto; €3m or €3.5m" -
    five fragments from four sources, presented as one quote, concealing a
    disagreement about the fee.
    """
    raw = span or ""
    return " / " in raw or "..." in raw or raw.count("; ") >= 2


# Wording that marks two figures as ALTERNATIVES rather than as two facts.
DISJUNCTION_CUES = (
    " or ", " vs ", " versus ", "oppure", " ovvero ", "secondo altre fonti",
    "other sources", "reports put", "reported as", "variously", "either",
    "while others", "altri riportano", "según otras", "conflicting",
)
# Wording that marks them as parts of one total, which is not a disagreement.
ADDITIVE_CUES = (
    "additionally", "in addition", "plus", "of which", "comprising", "inoltre",
    "oltre a", "además", "en plus", "bonus", "waived", "including",
)


def multiple_large_figures(span: str) -> list[float]:
    """Distinct large same-currency figures quoted in one span."""
    found = find_money(span or "")
    if len(found) < 2:
        return []
    by_cur: dict = {}
    for m in found:
        by_cur.setdefault(m.currency or "?", set()).add(round(m.value, 2))
    for vals in by_cur.values():
        big = sorted(x for x in vals if x >= 100_000)
        if len(big) >= 2 and (max(big) - min(big)) > max(1.0, min(big) * 0.02):
            return big
    return []


def _conflicting_figures(span: str, has_amount: bool = True) -> list[float]:
    """Figures that genuinely disagree about the same quantity.

    Two large numbers in one quotation are usually not a disagreement. Morata's
    span reads "a termination fee of 5,000,000 EUR. Additionally, the footballer
    has waived his receivables amounting to 651,562 EUR" - two components of one
    settlement. Ricardo Horta's quotes a tiered sell-on schedule whose 2.5m and
    5m are thresholds, not rival valuations. Only an explicit disjunction makes
    two figures alternatives, and a field carrying no amount has no quantity to
    disagree about.
    """
    if not has_amount:
        return []
    big = multiple_large_figures(span)
    if not big:
        return []
    low = (span or "").lower()
    if any(c in low for c in ADDITIVE_CUES) and not any(c in low for c in DISJUNCTION_CUES):
        return []
    if not any(c in low for c in DISJUNCTION_CUES):
        return []
    return big


def _scope_check(span: str, comp: dict, family_comps: list, club_gaz,
                 exclude: set | None = None) -> tuple[str, str]:
    """Does this span describe the component event it is scoped to?

    Returns (verdict, detail) where verdict is one of `ok`, `rescope:<event_id>`
    or `out_of_family`. The rule that matters: a span naming only clubs from
    outside the family describes a different transaction, whatever the model
    scoped it to. That is how a later Milan sale populated a term on Rafael
    Leao's 2018 Sporting -> Lille move.
    """
    span_clubs = identity.clubs_mentioned(span, club_gaz, exclude=exclude)
    if not span_clubs:
        return "ok", "span names no club; falls back to source-level identity"
    scope_tokens = identity.club_tokens_of([comp.get("from_club"), comp.get("to_club")])
    if identity.club_tokens_overlap(span_clubs, scope_tokens):
        return "ok", ""
    family_tokens = identity.club_tokens_of(
        [c.get("from_club") for c in family_comps] + [c.get("to_club") for c in family_comps])
    if not identity.club_tokens_overlap(span_clubs, family_tokens):
        return "out_of_family", (
            f"span describes a transaction involving {sorted(span_clubs)[:3]}, none of "
            "which belongs to this family")
    # Re-scope only when the span points at exactly ONE other leg. A loan and
    # its return name the same two clubs, so "the span mentions Palace and
    # Trabzonspor" cannot choose between them; guessing moved Sorloth's
    # third-party share onto the loan leg, where it could not attach at all.
    matches = [c for c in family_comps
               if identity.club_tokens_overlap(
                   span_clubs, identity.club_tokens_of([c.get("from_club"), c.get("to_club")]))]
    if len(matches) == 1 and matches[0]["event_id"] != comp["event_id"]:
        c = matches[0]
        return f"rescope:{c['event_id']}", (
            f"span describes the {c.get('from_club')} -> {c.get('to_club')} leg")
    return "ok", "span names a family club other than the scoped leg's; scope left as given"


def validate(payload: dict, family_rows, sources: list[dict],
             family_block: dict | None = None) -> Validation:
    """Deterministic re-check of everything the model claimed."""
    v = Validation()
    if not isinstance(payload, dict):
        v.reasons.append("payload is not an object")
        return v
    roles = {str(r.event_id): str(r.event_role) for _, r in family_rows.iterrows()}
    comps = {str(r.event_id): {"event_id": str(r.event_id),
                               "from_club": str(r.from_club_name),
                               "to_club": str(r.to_club_name),
                               "transfer_date": str(r.transfer_date),
                               "event_role": str(r.event_role)}
             for _, r in family_rows.iterrows()}
    comp_list = list(comps.values())
    by_id = {s["evidence_id"]: s for s in sources}
    src_text = {s["evidence_id"]: _norm(s.get("evidence_text", "")) for s in sources}
    try:
        club_gaz = identity.club_gazetteer()
    except Exception:                                          # noqa: BLE001
        club_gaz = {}
    # A player's own surname can be a club - UA Horta, for Ricardo Horta - so
    # it must not count as evidence that the span is about another team.
    player_name = str(family_rows.iloc[0].player_name) if len(family_rows) else ""
    player_tokens = identity.name_tokens_of(player_name)

    for fname in STAGE2_ALL_FIELDS + sorted(STAGE1_OWNED):
        d = payload.get(fname)
        if not isinstance(d, dict) or not d:
            continue
        status, repaired = normalise_status(d.get("status"))
        if repaired:
            v.status_repairs.append(f"{fname}:{d.get('status')}->{status}")
            d["status"] = status
        if status in ("not_found", "not_applicable"):
            continue

        eids = [e for e in (d.get("evidence_ids") or []) if e in by_id]
        if not eids:
            v.downgraded.append(fname)
            v.reasons.append(f"{fname}: no valid evidence_ids")
            continue

        # Tier 3 is leads-only and may never establish a term on its own. The
        # first v4 run let a Transfermarkt news page establish a purchase
        # obligation and an aggregator establish a contract expiry; both are
        # exactly the false positives the tiering exists to prevent.
        tiers = [int(by_id[e].get("source_tier") or 3) for e in eids]
        if min(tiers) >= 3:
            v.downgraded.append(fname)
            v.tier3_only.append(fname)
            v.reasons.append(
                f"{fname}: only tier-3 support ({[by_id[e].get('publisher') for e in eids]}); "
                "leads-only sources cannot establish a term")
            continue

        raw_span = d.get("evidence_span") or d.get("quote") or ""
        span = _norm(raw_span)
        if not span:
            v.downgraded.append(fname)
            v.reasons.append(f"{fname}: no evidence_span")
            continue
        # The span must be traceable to something we sent. Exact substring is
        # the strong test, but chunk boundaries and whitespace normalisation
        # break it on spans that are genuinely present, so a high token-overlap
        # match is accepted as a fallback. Both are far stricter than trusting
        # the model, and neither admits a paraphrase of absent text.
        if not _span_supported(span, eids, src_text):
            v.span_not_found.append(fname)
            v.downgraded.append(fname)
            v.reasons.append(f"{fname}: evidence_span not traceable to supplied sources")
            continue

        scoped = str(d.get("scoped_to_event_id") or "")
        if scoped not in roles:
            v.wrong_event.append(fname)
            v.downgraded.append(fname)
            v.reasons.append(
                f"{fname}: scoped_to_event_id '{scoped}' is not a member of this family")
            continue

        # --- hard family-scope gate -------------------------------------
        verdict, detail = _scope_check(raw_span, comps[scoped], comp_list, club_gaz,
                                       exclude=player_tokens)
        if verdict == "out_of_family":
            v.out_of_family.append(fname)
            v.downgraded.append(fname)
            v.related_event_evidence.append(
                {"field": fname, "evidence_span": raw_span[:400],
                 "evidence_ids": eids, "reason": detail})
            v.reasons.append(f"{fname}: {detail}")
            continue
        if verdict.startswith("rescope:"):
            scoped = verdict.split(":", 1)[1]
            d["scoped_to_event_id"] = scoped
            v.reasons.append(f"{fname}: re-scoped to {scoped} - {detail}")
        role = roles[scoped]

        if fname in STAGE1_OWNED:
            v.stage1_restated.append(fname)
            continue

        ok, problems = admissible(fname, {**d, "status": status}, role)
        if not ok:
            v.downgraded.append(fname)
            v.reasons.append(f"{fname}: " + "; ".join(problems))
            if any("payer/recipient" in p for p in problems):
                v.direction_missing.append(fname)
            continue

        terms = _mech_terms(fname)
        if fname in STAGE2_MECHANISM_FIELDS and terms:
            if not any(t.lower() in span for t in terms):
                v.downgraded.append(fname)
                v.reasons.append(
                    f"{fname}: span does not contain any recognised mechanism term")
                continue

        # --- money: the span is the authority on scale and on value ------
        amt_key = "amount" if d.get("amount") is not None else "price"
        amt = d.get(amt_key)
        if amt is not None:
            n = normalise_amount(amt, d.get("currency"), raw_span)
            if n.status == "contradicts_span":
                # The span quotes a figure and it is not this one. Something is
                # wrong with the reading, so the whole claim goes.
                v.downgraded.append(fname)
                v.money_unsupported.append(fname)
                v.amount_conflicts.append(fname)
                v.reasons.append(f"{fname}: {n.note}")
                continue
            if n.status == "no_figure_in_span":
                # The quotation establishes the mechanism but not the number,
                # which came from somewhere else in the source. Keep the
                # finding, drop the figure: Sottil's option is real even though
                # "decide di riscattarlo" does not say what it cost.
                d.pop("amount", None)
                d.pop("price", None)
                d["currency"] = None
                d["status"] = status = "partially_disclosed"
                v.money_unsupported.append(fname)
                v.reasons.append(
                    f"{fname}: amount {amt} dropped - its evidence span carries no "
                    "figure; the mechanism is recorded without a value")
                amt = None
            if amt is not None and n.scale_applied != 1.0:
                v.money_rescaled.append(f"{fname}:{amt}->{n.amount:,.0f}")
                v.reasons.append(f"{fname}: {n.note}")
            if amt is not None:
                d[amt_key] = n.amount
                d["currency"] = n.currency or d.get("currency")
                d["amount_raw_text"] = n.raw_text
                v.normalised[fname] = {"amount": n.amount, "currency": d.get("currency"),
                                       "raw_text": n.raw_text, "scale": n.scale_applied}
        elif d.get("currency") and not d.get("percentage"):
            # A currency with nothing denominated in it is noise.
            d["currency"] = None

        pct = d.get("percentage")
        if isinstance(pct, (int, float)):
            got, why = normalise_percentage(pct, raw_span)
            if got is None:
                v.downgraded.append(fname)
                v.money_unsupported.append(fname)
                v.reasons.append(f"{fname}: {why}")
                continue

        # --- credible disagreement is a finding, not a tie to break ------
        has_amount = d.get("amount") is not None or d.get("price") is not None
        conflict = _conflicting_figures(raw_span, has_amount)
        if conflict:
            d["status"] = status = "conflicting_sources"
            d["reported_values"] = conflict
            v.conflicting[fname] = conflict
            v.reasons.append(
                f"{fname}: sources disagree ({', '.join(f'{c:,.0f}' for c in conflict)}); "
                "recorded as conflicting_sources with all values preserved")
        elif has_amount:
            others = multiple_large_figures(raw_span)
            if others:
                d["other_figures_in_span"] = others
                v.reasons.append(
                    f"{fname}: span also quotes {', '.join(f'{c:,.0f}' for c in others)} - "
                    "check which figure the recorded value refers to")

        if _is_composite(raw_span):
            v.composite_spans.append(fname)

        # disclosed_yes claims the TERMS are known. Knowing only that a clause
        # exists and was or was not exercised is partial disclosure: Pio
        # Esposito's option and counter-option were both announced without a
        # price, which is a different fact from a disclosed price.
        carries = any(d.get(k) is not None for k in
                      ("amount", "price", "percentage", "date", "value"))
        if status == "disclosed_yes" and not carries:
            d["status"] = status = "partially_disclosed"
            v.status_repairs.append(
                f"{fname}:disclosed_yes->partially_disclosed (clause stated, terms not)")

        # An obligation_trigger exists to say WHAT converts the option into a
        # duty. Dzeko's recorded "if certain performance conditions were met"
        # with metric "not_specified" - a statement that we do not know the
        # trigger, published as though we did. A trigger needs a named metric
        # or a figure; Sorloth's "started 50 per cent of their games" has both.
        if fname == "obligation_trigger":
            metric = str(d.get("metric") or "").strip().lower()
            threshold = str(d.get("threshold") or "")
            vague = metric in ("", "not_specified", "unspecified", "unknown", "none")
            if vague and not re.search(r"\d", threshold + " " + raw_span):
                v.downgraded.append(fname)
                v.reasons.append(
                    f"{fname}: names no metric and no threshold - records that the "
                    "trigger is unknown rather than what it was")
                continue

        v.supported.append(fname)
        v.scoped[fname] = {"event_id": scoped, "role": role,
                           "statement_type": d.get("statement_type"),
                           "status": d.get("status")}

    # --- economic_mechanism must rest on a field that survived -----------
    em = payload.get("economic_mechanism")
    if isinstance(em, dict):
        em_status, _ = normalise_status(em.get("status"))
        value = str(em.get("value") or "")
        v.mechanism_value = value if em_status not in ("not_found", "not_applicable") else ""
        need = MECHANISM_REQUIRES.get(value)
        if v.mechanism_value and need and not (need & set(v.supported)):
            v.mechanism_rejected = value
            v.reasons.append(
                f"economic_mechanism '{value}' asserted while {sorted(need)} is not "
                "supported by validated evidence")
            em["status"] = "not_found"
            em["value"] = None
            v.mechanism_value = ""
        elif v.mechanism_value and value not in ECONOMIC_MECHANISM_VALUES:
            v.mechanism_rejected = value
            v.reasons.append(
                f"economic_mechanism '{value}' is not in the controlled vocabulary")
            em["status"] = "not_found"
            em["value"] = None
            v.mechanism_value = ""
    return v


def summarise(payload: dict, val: Validation) -> dict:
    mech = [f for f in val.supported if f in STAGE2_MECHANISM_FIELDS]
    if mech and not val.direction_missing:
        res = "resolved_strong"
    elif val.supported:
        res = "resolved_partial"
    else:
        res = "unresolved_insufficient_evidence"
    return {
        "resolution": res,
        "supported_fields": val.supported,
        "mechanism_fields": mech,
        "downgraded_fields": sorted(set(val.downgraded)),
        "wrong_event_attachments": val.wrong_event,
        "span_not_found": val.span_not_found,
        "stage1_restated": val.stage1_restated,
        "tier3_only_rejected": val.tier3_only,
        "amount_conflicts": val.amount_conflicts,
        "composite_spans": val.composite_spans,
        "status_repairs": val.status_repairs,
        "money_rescaled": val.money_rescaled,
        "money_unsupported": val.money_unsupported,
        "out_of_family": val.out_of_family,
        "related_event_evidence": val.related_event_evidence,
        "conflicting_fields": val.conflicting,
        "normalised": val.normalised,
        "economic_mechanism": val.mechanism_value,
        "economic_mechanism_rejected": val.mechanism_rejected,
        "reasons": val.reasons,
        "review_required": (bool(payload.get("review_required")) or bool(val.downgraded)
                            or bool(val.composite_spans) or bool(val.conflicting)),
        "n_supported": len(val.supported),
        "scoped": val.scoped,
    }
