"""Stage 2 event-family layer.

A Transfermarkt row is a *registration movement*, not necessarily a negotiated
transaction. The Stage 2 diagnostic (2026-09-21) showed that most fee-bearing
loan-return rows are the administrative unwinding of an agreement struck on a
different leg, or with a third club. Researching such a row on its own asks a
question the row cannot answer.

This module groups related rows into an `event_family` so terms can be attached
to the leg that actually carries them. It is deterministic and offline: it reads
Stage 1C, applies sequence logic, and writes a separate layer. Stage 1C rows are
never modified.

Grouping establishes only that rows are RELATED. It never asserts an economic
mechanism - that requires evidence, and lives in `mechanisms.py`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

# A loan and its return leg. Deliberately generous: Pio Esposito's Inter->Spezia
# loan ran 680 days, and multi-season loans are common.
MAX_LOAN_DAYS = 950

# Two follow-on windows. Inside IMMEDIATE the link is strong enough to treat the
# rows as one economic episode; between IMMEDIATE and WINDOW the rows are
# probably related but the family is flagged for review.
FOLLOW_ON_IMMEDIATE_DAYS = 21
FOLLOW_ON_WINDOW_DAYS = 60

# Distance from 30 June within which a return looks like ordinary season-end
# expiry rather than an early exit.
SEASON_END_TOLERANCE_DAYS = 10

LOAN_TYPES = {"loan"}
RETURN_TYPES = {"loan_return"}
PERMANENT_TYPES = {"permanent_transfer", "free_transfer", "undisclosed_transfer", "no_fee_shown"}

EVENT_ROLES = (
    "original_loan", "loan_return", "early_termination", "permanent_transfer",
    "third_party_sale", "administrative_return", "internal_registration", "unknown",
)

# Family classes span the whole Stage 1C population, not just loan returns.
# The first four describe families whose anchor is NOT a return leg; the module
# originally had only the return-anchored classes, so a standalone permanent
# transfer fell through to "likely_negotiated_return" and was described in the
# notes as a "return".
FAMILY_CLASSES = (
    "standalone_permanent_transfer", "standalone_loan", "loan_family",
    "likely_negotiated_return", "early_termination", "immediate_follow_on_transfer",
    "third_party_sale_related", "administrative_return", "unresolved_family_semantics",
)

# Classes whose anchor is a return leg. Only these may carry the return-leg
# role refinements and the fee_on_return caveat.
RETURN_ANCHORED_CLASSES = frozenset({
    "likely_negotiated_return", "early_termination", "immediate_follow_on_transfer",
    "third_party_sale_related", "administrative_return",
})


def _family_id(player_id, seed_event_id: str) -> str:
    h = hashlib.sha256(f"{player_id}|{seed_event_id}".encode()).hexdigest()[:20]
    return f"fam_{h}"


def _days_from_season_end(ts: pd.Timestamp) -> int:
    """Absolute days to the nearest 30 June."""
    best = None
    for year in (ts.year - 1, ts.year, ts.year + 1):
        d = abs((ts - pd.Timestamp(year=year, month=6, day=30)).days)
        best = d if best is None else min(best, d)
    return best


@dataclass
class Family:
    family_id: str
    player_id: object
    events: list = field(default_factory=list)      # ordered event_ids
    roles: dict = field(default_factory=dict)       # event_id -> role
    start_date: object = None
    end_date: object = None
    classification: str = "unresolved_family_semantics"
    review_required: bool = False
    review_reasons: list = field(default_factory=list)
    notes: str = ""


def _anchor_role_for(transfer_type: str) -> str:
    """The anchor's role comes from its own Stage 1 type.

    This module was written for loan-return anchors and defaulted every anchor
    to "loan_return", which mislabelled permanent transfers and loans once the
    layer was pointed at the wider research population.
    """
    return {
        "loan_return": "loan_return", "loan": "original_loan",
        "permanent_transfer": "permanent_transfer",
        "free_transfer": "permanent_transfer",
        "undisclosed_transfer": "permanent_transfer",
        "no_fee_shown": "permanent_transfer",
        "youth_or_internal": "internal_registration",
    }.get(str(transfer_type), "unknown")


def _follow_on_role(f, pivot_from_club_id) -> str:
    """Role of a movement that happens after the pivot leg."""
    if f.transfer_type_normalized == "youth_or_internal" or f.get("is_internal_move"):
        return "internal_registration"
    if f.transfer_type_normalized in PERMANENT_TYPES:
        # A sale back to the club that held the player on loan is the
        # option/counter-option story; a sale to anyone else makes the pivot a
        # conduit for a third party's money.
        return ("permanent_transfer" if f.to_club_id == pivot_from_club_id
                else "third_party_sale")
    if f.transfer_type_normalized in LOAN_TYPES:
        return "original_loan"          # seeds the next family
    return "unknown"


def build_families(canonical: pd.DataFrame, target_event_ids: list[str]) -> dict[str, Family]:
    """One family per target event, whatever kind of event it is.

    The family is the set of rows belonging to the same economic episode. Which
    rows those are depends on what the anchor is:

      * a return leg  -> the originating loan, the return, and movements close
                         after the return;
      * a loan        -> the loan, its own return leg, and movements close after
                         that return;
      * a permanent   -> the transfer, any loan of the player AT the buying club
                         that it converts, and movements close after it.

    Anchoring on the return leg only was correct for the population Stage 2
    first selected and wrong for everything else: a loan anchor was given no
    return leg, so it looked standalone, and a permanent anchor was handed the
    return-leg classifier.
    """
    df = canonical.copy()
    df["_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    by_player = {pid: g.sort_values(["_date", "event_id"]) for pid, g in df.groupby("player_id")}
    families: dict[str, Family] = {}

    for eid in target_event_ids:
        row = df[df.event_id == eid]
        if row.empty:
            continue
        anchor = row.iloc[0]
        chain = by_player[anchor.player_id]
        adate = anchor._date
        atype = str(anchor.transfer_type_normalized)
        anchor_role = _anchor_role_for(atype)

        loan = None           # originating loan, when there is one
        ret = None            # the return leg, when there is one
        pivot = anchor        # the leg that follow-ons are measured from
        ordered = []

        if atype in RETURN_TYPES:
            # --- originating loan: latest prior loan with reversed clubs ---
            prior = chain[(chain._date <= adate) & (chain.event_id != eid)
                          & (chain.transfer_type_normalized.isin(LOAN_TYPES))
                          & (chain.to_club_id == anchor.from_club_id)
                          & (chain.from_club_id == anchor.to_club_id)]
            prior = prior[(adate - prior._date).dt.days <= MAX_LOAN_DAYS]
            loan = prior.iloc[-1] if len(prior) else None
            ret = anchor
            if loan is not None:
                ordered.append((loan._date, loan.event_id, "original_loan"))

        elif atype in LOAN_TYPES:
            # --- the loan's own return leg, if it has happened yet ---
            loan = anchor
            after = chain[(chain._date > adate)
                          & (chain.transfer_type_normalized.isin(RETURN_TYPES))
                          & (chain.from_club_id == anchor.to_club_id)
                          & (chain.to_club_id == anchor.from_club_id)]
            after = after[(after._date - adate).dt.days <= MAX_LOAN_DAYS]
            if len(after):
                ret = after.iloc[0]
                ordered.append((ret._date, ret.event_id, "loan_return"))
                pivot = ret

        else:
            # --- a permanent move may convert a loan the player is already on ---
            prior = chain[(chain._date < adate) & (chain.event_id != eid)
                          & (chain.transfer_type_normalized.isin(LOAN_TYPES))
                          & (chain.to_club_id == anchor.to_club_id)]
            prior = prior[(adate - prior._date).dt.days <= MAX_LOAN_DAYS]
            if len(prior):
                loan = prior.iloc[-1]
                ordered.append((loan._date, loan.event_id, "original_loan"))

        fam = Family(_family_id(anchor.player_id, eid), anchor.player_id)
        ordered.append((adate, eid, anchor_role))

        # --- follow-on movements after the pivot leg ---
        pdate = pivot._date
        seen = {e for _, e, _ in ordered}
        after = chain[(chain._date > pdate)
                      & ((chain._date - pdate).dt.days <= FOLLOW_ON_WINDOW_DAYS)
                      & (~chain.event_id.isin(seen))]
        follow = []
        for _, f in after.iterrows():
            role = _follow_on_role(f, pivot.from_club_id)
            follow.append({"event_id": f.event_id, "gap": (f._date - pdate).days,
                           "role": role, "type": f.transfer_type_normalized,
                           "to_club_id": f.to_club_id, "to_club": f.to_club_name,
                           "date": f._date})
            ordered.append((f._date, f.event_id, role))

        ordered.sort(key=lambda t: (t[0], t[1]))
        fam.events = [e for _, e, _ in ordered]
        fam.roles = {e: r for _, e, r in ordered}
        fam.start_date = min(d for d, _, _ in ordered)
        fam.end_date = max(d for d, _, _ in ordered)

        _classify(fam, anchor, loan, ret, follow, pdate)
        families[eid] = fam
    return families


def _classify(fam: Family, anchor, loan, ret, follow: list, pdate) -> None:
    """Label the family, and refine the return leg's role where there is one.

    Dispatches on what the anchor actually is. The return-leg branch is the
    original logic, unchanged in behaviour; the other branches exist because
    applying it to a permanent transfer produced the contradiction of a
    `permanent_transfer` row classified `likely_negotiated_return` and
    described in the notes as a "return".
    """
    is_return = str(anchor.transfer_type_normalized) in RETURN_TYPES
    is_loan = str(anchor.transfer_type_normalized) in LOAN_TYPES

    season_end = _days_from_season_end(pdate) <= SEASON_END_TOLERANCE_DAYS
    immediate = [f for f in follow if f["gap"] <= FOLLOW_ON_IMMEDIATE_DAYS]
    windowed = [f for f in follow if f["gap"] <= FOLLOW_ON_WINDOW_DAYS]
    third_party = [f for f in windowed if f["role"] == "third_party_sale"]
    back_to_loan_club = [f for f in windowed if f["role"] == "permanent_transfer"]

    if is_return:
        _classify_return_anchor(fam, anchor, loan, follow, pdate, season_end,
                                immediate, windowed, third_party, back_to_loan_club)
    elif is_loan:
        _classify_loan_anchor(fam, anchor, ret, windowed, third_party)
    else:
        _classify_permanent_anchor(fam, anchor, loan, immediate, windowed, third_party)

    # The fee_on_return caveat belongs to return legs. A permanent transfer has
    # no fee_on_return to misread.
    if (fam.classification in ("third_party_sale_related", "early_termination",
                               "immediate_follow_on_transfer")
            and fam.classification in RETURN_ANCHORED_CLASSES and is_return):
        fam.review_required = True
        fam.review_reasons.append(
            "fee_on_return_eur should not be read as a bilateral transfer price here")


def _classify_return_anchor(fam, ret, loan, follow, rdate, season_end, immediate,
                            windowed, third_party, back_to_loan_club) -> None:
    """Original return-leg logic. Priority order matters: a third-party sale
    explains the return more specifically than 'there was a follow-on'."""
    if loan is None:
        fam.review_required = True
        fam.review_reasons.append("no originating loan found within %d days" % MAX_LOAN_DAYS)

    if third_party:
        f = third_party[0]
        fam.classification = "third_party_sale_related"
        fam.roles[ret.event_id] = "administrative_return"
        fam.notes = (f"Parent sold the player to a third club ({f['to_club']}) "
                     f"{f['gap']}d after the return; the return leg is the conduit.")
        if f["gap"] > FOLLOW_ON_IMMEDIATE_DAYS:
            fam.review_required = True
            fam.review_reasons.append(
                f"third-party sale is {f['gap']}d after the return, beyond the "
                f"{FOLLOW_ON_IMMEDIATE_DAYS}d immediate window; link is weaker")
    elif back_to_loan_club:
        f = back_to_loan_club[0]
        fam.classification = "likely_negotiated_return"
        fam.roles[ret.event_id] = "loan_return"
        fam.notes = (f"Player returned and was then transferred back to the loan club "
                     f"({f['to_club']}) {f['gap']}d later - consistent with an option "
                     f"or counter-option being settled. Mechanism NOT asserted.")
    elif immediate and not season_end:
        fam.classification = "early_termination"
        fam.roles[ret.event_id] = "early_termination"
        fam.notes = (f"Return is {_days_from_season_end(rdate)}d from any season end and is "
                     f"followed within {immediate[0]['gap']}d by another movement.")
    elif immediate:
        fam.classification = "immediate_follow_on_transfer"
        fam.roles[ret.event_id] = "administrative_return"
        fam.notes = (f"Season-end return immediately followed by a further move "
                     f"({immediate[0]['type']} to {immediate[0]['to_club']}, "
                     f"{immediate[0]['gap']}d).")
    elif not follow:
        fam.classification = "likely_negotiated_return"
        fam.roles[ret.event_id] = "loan_return"
        fam.notes = ("Standalone fee-bearing return with no follow-on movement; the "
                     "fee most plausibly settles something agreed on this leg.")
    elif any(f["role"] == "original_loan" for f in windowed):
        # Serial loan cycling: the parent takes the player back at season end and
        # re-loans him inside the same window. The return is a staging step, and
        # the money on it settles the loan that just ended rather than a new deal.
        f = next(f for f in windowed if f["role"] == "original_loan")
        fam.classification = "immediate_follow_on_transfer"
        fam.roles[ret.event_id] = "administrative_return"
        fam.notes = (f"Serial loan cycling: re-loaned to {f['to_club']} {f['gap']}d after "
                     f"the return, within the same transfer window.")
        fam.review_required = True
        fam.review_reasons.append(
            f"follow-on loan is {f['gap']}d after the return, beyond the "
            f"{FOLLOW_ON_IMMEDIATE_DAYS}d immediate window")
    else:
        fam.classification = "unresolved_family_semantics"
        fam.review_required = True
        fam.review_reasons.append("follow-on exists but does not match a known pattern")


def _classify_loan_anchor(fam, anchor, ret, windowed, third_party) -> None:
    """A loan. Either its return leg is known, or the loan is still open."""
    if third_party:
        f = third_party[0]
        fam.classification = "third_party_sale_related"
        fam.notes = (f"Loan to {anchor.to_club_name}; after the return the parent sold "
                     f"the player to a third club ({f['to_club']}) {f['gap']}d later.")
        return
    if ret is None:
        fam.classification = "standalone_loan"
        fam.notes = (f"Loan from {anchor.from_club_name} to {anchor.to_club_name} with no "
                     f"return leg recorded within {MAX_LOAN_DAYS}d. Any option, obligation "
                     f"or buy-back agreed on this loan belongs to THIS leg.")
        fam.review_required = True
        fam.review_reasons.append(
            f"no return leg found within {MAX_LOAN_DAYS}d; the loan may still be "
            "running, or the return may be missing from Stage 1")
        return
    bought = [f for f in windowed if f["role"] == "permanent_transfer"]
    if bought:
        f = bought[0]
        fam.classification = "likely_negotiated_return"
        fam.notes = (f"Loan to {anchor.to_club_name}, returned, then transferred back to "
                     f"{f['to_club']} {f['gap']}d later - consistent with an option or "
                     f"counter-option being settled. Mechanism NOT asserted.")
        return
    fam.classification = "loan_family"
    fam.notes = (f"Loan from {anchor.from_club_name} to {anchor.to_club_name} and its "
                 f"return on {ret.transfer_date}. Terms agreed for the loan attach to "
                 f"the loan leg, not the return.")


def _classify_permanent_anchor(fam, anchor, loan, immediate, windowed, third_party) -> None:
    """A permanent, free or undisclosed transfer. Never a 'return'."""
    if third_party:
        f = third_party[0]
        fam.classification = "third_party_sale_related"
        fam.notes = (f"{anchor.from_club_name} -> {anchor.to_club_name} followed "
                     f"{f['gap']}d later by an onward sale to {f['to_club']}.")
        fam.review_required = True
        fam.review_reasons.append(
            "an onward sale follows this transfer closely; terms may belong to either leg")
        return
    if immediate:
        f = immediate[0]
        fam.classification = "immediate_follow_on_transfer"
        fam.notes = (f"{anchor.from_club_name} -> {anchor.to_club_name} immediately "
                     f"followed by a further move ({f['type']} to {f['to_club']}, "
                     f"{f['gap']}d).")
        fam.review_required = True
        fam.review_reasons.append(
            "a further movement follows within "
            f"{FOLLOW_ON_IMMEDIATE_DAYS}d; the two legs may be one negotiation")
        return
    if loan is not None:
        fam.classification = "loan_family"
        fam.notes = (f"Permanent move to {anchor.to_club_name} following a loan at the "
                     f"same club that began {loan.transfer_date}; consistent with a "
                     f"purchase option or obligation being settled. Mechanism NOT asserted.")
        return
    if windowed:
        fam.classification = "unresolved_family_semantics"
        fam.review_required = True
        fam.review_reasons.append("follow-on exists but does not match a known pattern")
        return
    fam.classification = "standalone_permanent_transfer"
    fam.notes = (f"Standalone {str(anchor.transfer_type_normalized).replace('_', ' ')} "
                 f"from {anchor.from_club_name} to {anchor.to_club_name} with no related "
                 f"movement nearby. Terms attach to this leg.")


def families_to_rows(families: dict[str, Family], canonical: pd.DataFrame) -> pd.DataFrame:
    """Flatten to one row per (family, member event) for the Stage 2 layer."""
    meta = canonical.set_index("event_id")
    out = []
    for target_eid, fam in families.items():
        for eid in fam.events:
            if eid not in meta.index:
                continue
            m = meta.loc[eid]
            out.append({
                "event_family_id": fam.family_id,
                "event_id": eid,
                "is_family_anchor": eid == target_eid,
                "event_role": fam.roles.get(eid, "unknown"),
                "family_player_id": fam.player_id,
                "player_name": m.player_name,
                "transfer_date": m.transfer_date,
                "from_club_name": m.from_club_name,
                "to_club_name": m.to_club_name,
                "transfer_type_normalized": m.transfer_type_normalized,
                "permanent_transfer_fee_eur": m.permanent_transfer_fee_eur,
                "loan_fee_eur": m.loan_fee_eur,
                "fee_on_return_eur": m.fee_on_return_eur,
                "family_start_date": fam.start_date.date().isoformat(),
                "family_end_date": fam.end_date.date().isoformat(),
                "family_events": ";".join(fam.events),
                "family_interpretation_status": fam.classification,
                "family_review_required": fam.review_required,
                "family_review_reasons": " | ".join(fam.review_reasons),
                "family_notes": fam.notes,
            })
    return pd.DataFrame(out)
