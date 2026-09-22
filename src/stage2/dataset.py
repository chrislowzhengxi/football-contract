"""Assemble validated Stage 2 records into the published tables.

One row per player/event family. Only fields that survived validation are
written: a term rejected by the money, scope, tier or role gates leaves its
column empty rather than appearing with a caveat, because a populated cell
with a footnote is read as a finding.

Conflicts are preserved rather than resolved. Where sources disagree the
status is `conflicting_sources` and every reported value is kept, so the
disagreement is visible in the data instead of being settled silently.
"""
from __future__ import annotations

import json
import re

from .money import find_all as find_money

# Wording in a model's own review note that reports a cross-source
# disagreement. A row carrying one of these cannot also claim high confidence.
CONFLICT_WORDS = ("conflicting", "conflict", "disagree", "discrepan", "contradict")

TERM_FIELDS = [
    "transfer_fee", "loan_fee", "add_ons", "sell_on", "buy_back",
    "purchase_option", "purchase_obligation", "obligation_trigger",
    "release_or_purchase_clause", "termination_compensation",
    "third_party_sale_share", "parent_contract_expiry",
]
# Stage 1C already settles these; repeating one is not a research finding.
STAGE1_FIELDS = {"transfer_fee", "loan_fee"}
RESEARCH_FIELDS = [f for f in TERM_FIELDS if f not in STAGE1_FIELDS]
# A negotiated mechanism, as distinct from a contract length. Both are Stage 2
# findings, but a row whose only finding is an expiry date is much weaker
# evidence that clause recovery works, so the two are counted separately.
MECHANISM_FIELDS = [f for f in RESEARCH_FIELDS if f != "parent_contract_expiry"]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else f


def format_percentage(pct) -> str:
    """A percentage, or a tiered sell-on schedule, in words a reader can scan.

    Horta's sell-on is three bands keyed to the future sale price. As raw JSON
    it is unreadable in a spreadsheet, which defeats the point of the column.
    """
    if isinstance(pct, (int, float)):
        return f"{_num(pct)}%"
    if isinstance(pct, dict):
        pct = [pct]
    if not isinstance(pct, list):
        return str(pct)[:160]
    parts = []
    for band in pct:
        if not isinstance(band, dict):
            parts.append(str(band))
            continue
        v = band.get("percentage_value", band.get("percentage"))
        lo, hi = band.get("lower_threshold_eur"), band.get("upper_threshold_eur")
        def m(x):
            return f"EUR {x/1e6:g}m" if isinstance(x, (int, float)) and x else None
        if lo and hi:
            parts.append(f"{v}% between {m(lo)} and {m(hi)}")
        elif hi:
            parts.append(f"{v}% up to {m(hi)}")
        elif lo:
            parts.append(f"{v}% above {m(lo)}")
        else:
            parts.append(f"{v}%")
    return "; ".join(parts)[:220]


def format_term(d: dict) -> str:
    """A term as one readable cell, with the quoted wording kept beside it."""
    if not isinstance(d, dict):
        return ""
    bits = []
    amt = d.get("amount") if d.get("amount") is not None else d.get("price")
    n = _num(amt)
    if n is not None:
        bits.append(f"{n:,}" if isinstance(n, int) else f"{n:,.2f}")
        if d.get("currency"):
            bits.append(str(d["currency"]))
    pct = d.get("percentage")
    if pct not in (None, "", []):
        bits.append(format_percentage(pct))
    if d.get("date"):
        bits.append(str(d["date"]))
    if d.get("value") not in (None, "", []):
        bits.append(str(d["value"])[:80])
    if d.get("exercised") is not None:
        bits.append(f"exercised={d['exercised']}")
    if d.get("status") == "conflicting_sources" and d.get("reported_values"):
        # reported_values arrive either as numbers or as the wording a source
        # used, e.g. "EUR 2.5m (ev01)", so read a figure out of the text too.
        vals = []
        for v in d["reported_values"]:
            n = _num(v)
            if n is None:
                found = find_money(str(v))
                n = found[0].value if found else None
            if n is not None and f"{n:,.0f}" not in vals:
                vals.append(f"{n:,.0f}")
        if vals:
            return f"CONFLICTING: {' | '.join(vals)} {d.get('currency') or ''}".strip()
    if not bits:
        return str(d.get("status") or "")
    out = " ".join(bits)
    raw = d.get("amount_raw_text")
    return f'{out} (quoted "{raw}")' if raw else out


def family_summary(block: dict) -> str:
    return " -> ".join(
        f"{c['transfer_date']} {c['from_club']}->{c['to_club']} [{c['event_role']}]"
        for c in block.get("component_events", []))


def review_reports_a_conflict(rec: dict) -> bool:
    text = " ".join(str(r) for r in
                    ((rec.get("payload") or {}).get("review_reasons") or [])).lower()
    return any(w in text for w in CONFLICT_WORDS)


def disputed_values_in_review(rec: dict) -> list[float]:
    """Figures the model's own review note reports other sources giving."""
    text = " ".join(str(r) for r in
                    ((rec.get("payload") or {}).get("review_reasons") or []))
    if not any(w in text.lower() for w in CONFLICT_WORDS):
        return []
    return sorted({round(m.value, 2) for m in find_money(text) if m.value >= 100_000})


def publishes_a_figure(row: dict) -> bool:
    """Does this row put a monetary amount in front of a reader?

    Comma-grouped, because `format_term` writes amounts that way and a date
    like 2023-06-19 must not count as a published figure.
    """
    return any(re.search(r"\d{1,3}(?:,\d{3})+", str(row.get(f) or ""))
               for f in RESEARCH_FIELDS)


def grade(rec: dict) -> tuple[str, str]:
    """A / B / C, with the reason. Contaminated or mis-built cases cannot be A.

    A: at least one validated non-Stage-1 term, correctly scoped, admissible
       source, no unresolved material contradiction affecting that term.
    B: useful contract evidence exists but material review remains.
    C: no reliable non-Stage-1 term established.
    """
    v = rec.get("validation") or {}
    sup = [f for f in v.get("supported_fields", []) if f in RESEARCH_FIELDS]
    if not sup:
        if not rec.get("sources"):
            n = rec.get("n_identity_rejected", 0)
            return "C", (f"no admissible evidence: all {n} candidate documents were "
                         "about other players" if n else "no cached evidence at all")
        return "C", (f"{len(rec.get('sources', []))} admissible documents, but no "
                     "contract term survived validation")
    if v.get("wrong_event_attachments"):
        return "B", "a term was attached to an event outside the family"
    if v.get("out_of_family"):
        return "B", "evidence describing a transaction outside the family was removed"
    tiers = {int(s.get("source_tier") or 3) for s in rec.get("sources", [])}
    if not ({1, 2} & tiers):
        return "C", "no tier-1 or tier-2 source in the packet"
    conflicting = set(v.get("conflicting_fields") or {})
    if conflicting & set(sup):
        return "B", (f"sources disagree on {sorted(conflicting & set(sup))}; "
                     "values preserved, not resolved")
    if set(v.get("composite_spans") or []) & set(sup):
        return "B", "a supporting quotation is stitched from several fragments"
    if review_reports_a_conflict(rec):
        vals = disputed_values_in_review(rec)
        return "B", ("the review note reports a cross-source disagreement"
                     + (f" ({', '.join(f'{x:,.0f}' for x in vals)})" if vals else ""))
    return "A", f"validated: {', '.join(sup)}"


def build_row(rec: dict, meta) -> dict:
    """One dataset row. `meta` is the Stage 1C row for the anchor event."""
    v = rec.get("validation") or {}
    payload = rec.get("payload") or {}
    block = rec.get("block") or {}
    sup = set(v.get("supported_fields") or [])
    roles = {c["event_id"]: c for c in block.get("component_events", [])}
    anchor = roles.get(rec["event_id"], {})
    used = {e for f in sup for e in ((payload.get(f) or {}).get("evidence_ids") or [])}
    srcs = [s for s in rec.get("sources", []) if s["evidence_id"] in used]
    g, why = grade(rec)
    research = sorted(sup & set(RESEARCH_FIELDS))

    row = {
        "player": str(meta.player_name),
        "age_at_transfer": meta.age_at_transfer,
        "transfer_date": str(meta.transfer_date),
        "departing_club": str(meta.from_club_name),
        "receiving_club": str(meta.to_club_name),
        "transfer_type": str(meta.transfer_type_normalized),
        "event_family_id": rec.get("event_family_id"),
        "event_id": rec["event_id"],
        "anchor_event_role": anchor.get("event_role", ""),
        "family_classification": rec.get("family_classification", ""),
        "family_notes": rec.get("family_notes", ""),
        "family_event_ids": ";".join(roles),
        "event_family_summary": family_summary(block),
    }
    for f in TERM_FIELDS:
        d = payload.get(f) if f in sup else None
        row[f] = format_term(d) if d else ""
        row[f + "_status"] = (d or {}).get("status", "") if d else ""
        row[f + "_scope_event_id"] = (d or {}).get("scoped_to_event_id", "") if d else ""
        row[f + "_scope_role"] = (
            roles.get((d or {}).get("scoped_to_event_id", ""), {}).get("event_role", "")
            if d else "")
        row[f + "_evidence_span"] = ((d or {}).get("evidence_span") or "")[:500] if d else ""
        rv = (d or {}).get("reported_values") if d else None
        row[f + "_reported_values"] = "|".join(str(_num(x)) for x in rv) if rv else ""

    row.update({
        "market_value_at_signing_eur": meta.market_value_eur,
        "subsequent_season_minutes": "",          # not present in Stage 1C
        "stage1_permanent_fee_eur": meta.permanent_transfer_fee_eur,
        "stage1_loan_fee_eur": meta.loan_fee_eur,
        "stage1_fee_on_return_eur": meta.fee_on_return_eur,
        # "unknown" is the model declining to name a mechanism; publishing it
        # as a value would read as a finding.
        "economic_mechanism": ("" if v.get("economic_mechanism") == "unknown"
                               else v.get("economic_mechanism", "")),
        "deal_summary": str(payload.get("family_summary")
                            or payload.get("deal_summary") or "")[:700],
        "disputed_values_in_review": "|".join(
            f"{x:,.0f}" for x in disputed_values_in_review(rec)),
        "n_researched_fields": len(research),
        "n_mechanism_fields": len([f for f in research if f in MECHANISM_FIELDS]),
        "researched_fields": ";".join(research),
        "field_statuses": ";".join(f"{f}={(payload.get(f) or {}).get('status', '')}"
                                   for f in research),
        "reported_values": ";".join(
            f"{f}:{'|'.join(str(_num(x)) for x in (payload.get(f) or {}).get('reported_values') or [])}"
            for f in research if (payload.get(f) or {}).get("reported_values")),
        "sources": ";".join(s["source_url"] for s in srcs)[:1200],
        "source_tiers": ";".join(str(t) for t in sorted({s["source_tier"] for s in srcs})),
        "source_classes": ";".join(sorted({s["source_class"] for s in srcs})),
        "direct_or_retrospective": ";".join(sorted(
            {(payload.get(f) or {}).get("statement_type") for f in research
             if (payload.get(f) or {}).get("statement_type")})),
        "n_sources_in_packet": len(rec.get("sources", [])),
        "n_sources_used": len(srcs),
        "n_documents_rejected_by_identity_gate": rec.get("n_identity_rejected", 0),
        "confidence": {"A": "high", "B": "medium", "C": "none"}[g],
        "quality_grade": g,
        "grade_reason": why,
        "review_required": bool(v.get("review_required")),
        "review_reasons": " | ".join(
            [str(r) for r in (payload.get("review_reasons") or [])[:3]]
            + [str(r) for r in (v.get("reasons") or [])[:4]])[:900],
        "resolution": v.get("resolution", ""),
        "validator_flags": ";".join(filter(None, [
            "money_rescaled" if v.get("money_rescaled") else "",
            "money_unsupported" if v.get("money_unsupported") else "",
            "out_of_family" if v.get("out_of_family") else "",
            "conflicting" if v.get("conflicting_fields") else "",
            "composite_span" if v.get("composite_spans") else "",
            "tier3_rejected" if v.get("tier3_only_rejected") else "",
            "wrong_event" if v.get("wrong_event_attachments") else "",
            "mechanism_rejected" if v.get("economic_mechanism_rejected") else ""])),
        "related_event_evidence": json.dumps(
            v.get("related_event_evidence") or [], ensure_ascii=False)[:900],
    })
    return row


def build_field_rows(rec: dict, meta) -> list[dict]:
    """One row per validated term, with its full provenance."""
    v = rec.get("validation") or {}
    payload = rec.get("payload") or {}
    by_id = {s["evidence_id"]: s for s in rec.get("sources", [])}
    out = []
    for f in v.get("supported_fields", []):
        d = payload.get(f) or {}
        eids = [e for e in (d.get("evidence_ids") or []) if e in by_id]
        amt = d.get("amount") if d.get("amount") is not None else d.get("price")
        out.append({
            "player": str(meta.player_name), "event_id": rec["event_id"],
            "event_family_id": rec.get("event_family_id"), "field": f,
            "status": d.get("status", ""),
            "amount_normalised": _num(amt),
            "currency": d.get("currency") or "",
            "amount_quoted_text": d.get("amount_raw_text") or "",
            "percentage": json.dumps(d.get("percentage"), ensure_ascii=False)[:300]
                          if d.get("percentage") not in (None, "", []) else "",
            "date": d.get("date") or "",
            "exercised": d.get("exercised"),
            "reported_values": "|".join(str(_num(x)) for x in (d.get("reported_values") or [])),
            "scoped_to_event_id": d.get("scoped_to_event_id", ""),
            "which_transaction": str(d.get("which_transaction") or "")[:300],
            "statement_type": d.get("statement_type", ""),
            "evidence_span": str(d.get("evidence_span") or "")[:600],
            "evidence_ids": ";".join(eids),
            "source_urls": ";".join(by_id[e]["source_url"] for e in eids)[:800],
            "source_tiers": ";".join(str(by_id[e]["source_tier"]) for e in eids),
        })
    return out


def build_source_rows(rec: dict, meta, used_only: bool = True) -> list[dict]:
    v = rec.get("validation") or {}
    payload = rec.get("payload") or {}
    used = {e for f in v.get("supported_fields", [])
            for e in ((payload.get(f) or {}).get("evidence_ids") or [])}
    out = []
    for s in rec.get("sources", []):
        if used_only and s["evidence_id"] not in used:
            continue
        out.append({
            "player": str(meta.player_name), "event_id": rec["event_id"],
            "event_family_id": rec.get("event_family_id"),
            "evidence_id": s["evidence_id"], "source_url": s["source_url"],
            "source_title": s.get("source_title", ""), "publisher": s.get("publisher", ""),
            "source_class": s.get("source_class", ""), "source_tier": s.get("source_tier"),
            "identity_strength": s.get("identity_strength", ""),
            "discovery_provider": s.get("discovery_provider", ""),
            "text_origin": s.get("text_origin", ""),
            "used_for_a_validated_field": s["evidence_id"] in used,
        })
    return out
