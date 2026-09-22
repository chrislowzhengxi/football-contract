"""Adversarial checks over an assembled Stage 2 dataset.

Everything here looks for a contradiction between two things the dataset says
about itself, rather than for a missing value. Each check is named for a
defect that actually reached the first demonstration artifact, so a regression
shows up as a failing check rather than as a plausible-looking row.

`run_all` returns one row per violation. An empty result is the release gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .discovery import identity
from .money import find_all as find_money

# A contract figure below this, quoted from text that says "million", is a
# scaling error rather than a genuinely tiny fee.
IMPLAUSIBLY_SMALL = 1_000
MILLION_WORDS = ("million", "milioni", "millones", "milhões", "millions", "mln",
                 "mio", "milione", "millon", "millón", "m ", "m)", "m.", "m,")

RETURN_WORDS = ("loan_return", "administrative_return", "early_termination")
PERMANENT_TYPES = ("permanent_transfer", "free_transfer", "undisclosed_transfer",
                   "no_fee_shown")


@dataclass
class Violation:
    check: str
    player: str
    event_id: str
    detail: str

    def as_row(self) -> dict:
        return {"check": self.check, "player": self.player,
                "event_id": self.event_id, "detail": self.detail}


def _text(v) -> str:
    return "" if v is None else str(v)


def _has(v) -> bool:
    s = _text(v).strip()
    return bool(s) and s.lower() not in ("nan", "none", "")


def check_transfer_type_matches_family_roles(row) -> list[Violation]:
    """A permanent transfer must not be described as a return."""
    out = []
    ttype = _text(row.get("transfer_type"))
    summary = _text(row.get("event_family_summary"))
    fam = _text(row.get("family_classification"))
    if ttype in PERMANENT_TYPES:
        anchor_role = _text(row.get("anchor_event_role"))
        if anchor_role in RETURN_WORDS:
            out.append(Violation(
                "permanent_transfer_with_return_role", _text(row.get("player")),
                _text(row.get("event_id")),
                f"transfer_type={ttype} but anchor event_role={anchor_role}"))
        if "negotiated_return" in fam or "administrative_return" in fam:
            out.append(Violation(
                "permanent_transfer_classified_as_return", _text(row.get("player")),
                _text(row.get("event_id")),
                f"transfer_type={ttype} but family_classification={fam}"))
        if re.search(r"\breturn\b", _text(row.get("family_notes")), re.I):
            out.append(Violation(
                "permanent_transfer_described_as_return", _text(row.get("player")),
                _text(row.get("event_id")), "family notes call this leg a return"))
    return out


def check_amount_scaling(row, fields) -> list[Violation]:
    """A figure under 1,000 quoted from text that says "million"."""
    out = []
    for f in fields:
        val = row.get(f)
        if not _has(val):
            continue
        nums = [float(x) for x in re.findall(r"amount=([\d.]+)", _text(val))]
        span = _text(row.get(f + "_evidence_span")).lower()
        for n in nums:
            if n < IMPLAUSIBLY_SMALL and any(w in span for w in MILLION_WORDS):
                out.append(Violation(
                    "amount_scaling_error", _text(row.get("player")),
                    _text(row.get("event_id")),
                    f"{f} records {n:g} while its span says million: {span[:90]}"))
    return out


def check_mechanism_supported(row, fields) -> list[Violation]:
    """`economic_mechanism` must rest on a field that survived validation."""
    from .extract_v4 import MECHANISM_REQUIRES
    mech = _text(row.get("economic_mechanism"))
    need = MECHANISM_REQUIRES.get(mech)
    if not mech or not need:
        return []
    present = {f for f in fields if _has(row.get(f))}
    if not (need & present):
        return [Violation("mechanism_without_field", _text(row.get("player")),
                          _text(row.get("event_id")),
                          f"economic_mechanism={mech} but none of {sorted(need)} is populated")]
    return []


def check_scope_in_family(row, fields) -> list[Violation]:
    """Every term must be scoped to a member of its own family."""
    members = {e for e in _text(row.get("family_event_ids")).split(";") if e}
    out = []
    for f in fields:
        scoped = _text(row.get(f + "_scope_event_id"))
        if _has(row.get(f)) and scoped and members and scoped not in members:
            out.append(Violation("scope_outside_family", _text(row.get("player")),
                                 _text(row.get("event_id")),
                                 f"{f} scoped to {scoped}, not in {sorted(members)}"))
    return out


def check_grade_has_a_finding(row, fields) -> list[Violation]:
    """An A grade requires at least one validated non-Stage-1 field."""
    grade = _text(row.get("quality_grade")).upper()
    n = int(row.get("n_researched_fields") or 0)
    if grade == "A" and n == 0:
        return [Violation("grade_a_without_finding", _text(row.get("player")),
                          _text(row.get("event_id")), "quality_grade=A with no researched field")]
    if grade in ("A", "B") and not any(_has(row.get(f)) for f in fields):
        return [Violation("grade_without_populated_field", _text(row.get("player")),
                          _text(row.get("event_id")),
                          f"quality_grade={grade} but no term column is populated")]
    return []


def check_sources_belong_to_the_player(row, source_rows, gaz=None) -> list[Violation]:
    """No source in the packet may be about a different footballer."""
    player = _text(row.get("player"))
    out = []
    for s in source_rows:
        title = _text(s.get("source_title"))
        if not title:
            continue
        ours = identity.name_parts(player)
        others = {p for p in identity.named_players_in(title, gaz)
                  if not ours or p[1] != ours[1]}
        if others and (not ours or not identity.name_parts(player) or
                       ours not in identity.named_players_in(title, gaz)):
            out.append(Violation(
                "source_about_another_player", player, _text(row.get("event_id")),
                f"{s.get('source_url', '')[:70]} is titled about "
                f"{sorted(others)[0][0]} {sorted(others)[0][1]}"))
    return out


def check_conflicts_are_declared(row, fields) -> list[Violation]:
    """Figures quoted as alternatives must be recorded as conflicting_sources.

    Mirrors the validator rule exactly: only an explicit disjunction makes two
    figures rival readings of the same quantity, and a field carrying no amount
    has no quantity to disagree about.
    """
    from .extract_v4 import _conflicting_figures
    out = []
    for f in fields:
        span = _text(row.get(f + "_evidence_span"))
        if not _has(row.get(f)) or not span:
            continue
        has_amount = bool(re.search(r"\d{1,3}(?:,\d{3})+", _text(row.get(f))))
        if not _conflicting_figures(span, has_amount):
            continue
        status = _text(row.get(f + "_status"))
        if status != "conflicting_sources" and not _has(row.get(f + "_reported_values")):
            out.append(Violation("undeclared_conflict", _text(row.get("player")),
                                 _text(row.get("event_id")),
                                 f"{f} span quotes rival figures but status={status}"))
    return out


def check_review_reasons_match_output(row) -> list[Violation]:
    """A review note that names a conflict, while the row declares none."""
    reasons = _text(row.get("review_reasons")).lower()
    if not reasons:
        return []
    says_conflict = any(w in reasons for w in
                        ("conflicting", "conflict", "disagree", "discrepan"))
    declares = ("conflicting_sources" in _text(row.get("field_statuses")).lower()
                or _has(row.get("reported_values")))
    # A review note about figures we did not publish is not a contradiction:
    # Colombo's note reports a 3m/3.5m disagreement and the validator dropped
    # both amounts because neither span carried one. What IS a contradiction is
    # claiming high confidence while the note reports a live disagreement.
    publishes = any(re.search(r"\d{1,3}(?:,\d{3})+", _text(row.get(f)))
                    for f in ("purchase_option", "buy_back", "sell_on", "add_ons",
                              "termination_compensation", "third_party_sale_share",
                              "purchase_obligation"))
    claims_high = _text(row.get("quality_grade")).upper() == "A"
    if says_conflict and not declares and publishes and claims_high:
        return [Violation("review_reason_contradicts_output", _text(row.get("player")),
                          _text(row.get("event_id")),
                          "review reasons report a conflict the structured fields do not")]
    return []


def check_summary_subject(row, gaz=None) -> list[Violation]:
    """A deal summary whose subject is somebody else."""
    player = _text(row.get("player"))
    text = " ".join([_text(row.get("deal_summary")), _text(row.get("evidence_summary"))])
    if not text.strip():
        return []
    ours = identity.name_parts(player)
    if not ours:
        return []
    others = {p for p in identity.named_players_in(text, gaz) if p[1] != ours[1]}
    ours_present = any(p[1] == ours[1] for p in identity.named_players_in(text, gaz))
    if others and not ours_present:
        return [Violation("summary_about_another_player", player, _text(row.get("event_id")),
                          f"summary names {sorted(others)[0][1]} but never {player}")]
    return []


def run_all(rows, sources_by_event: dict, term_fields) -> list[dict]:
    """Every check over every row. An empty list is the release gate."""
    try:
        gaz = identity.player_gazetteer()
    except Exception:                                          # noqa: BLE001
        gaz = {}
    out: list[Violation] = []
    for row in rows:
        eid = _text(row.get("event_id"))
        out += check_transfer_type_matches_family_roles(row)
        out += check_amount_scaling(row, term_fields)
        out += check_mechanism_supported(row, term_fields)
        out += check_scope_in_family(row, term_fields)
        out += check_grade_has_a_finding(row, term_fields)
        out += check_conflicts_are_declared(row, term_fields)
        out += check_review_reasons_match_output(row)
        out += check_summary_subject(row, gaz)
        out += check_sources_belong_to_the_player(row, sources_by_event.get(eid, []), gaz)
    return [v.as_row() for v in out]
