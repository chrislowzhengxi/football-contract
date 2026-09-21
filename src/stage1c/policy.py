"""Normalization and research-target rules for the canonical backbone.

Pure functions only, so every rule in this file is directly testable and can be
read without running anything.
"""
from __future__ import annotations

import re

import pandas as pd

PLACEHOLDER_CLUBS = {"Without Club", "Retired", "Career break", "Ban", "Unknown"}

# Matches Transfermarkt's naming for academy and reserve sides. Validated
# against all 18,651 club names in the snapshot: 7,760 match, and the senior
# clubs involved in transfers above EUR 20m are never matched except where the
# selling side genuinely is an academy squad (e.g. "Man City U18" selling
# Jadon Sancho). Anchoring "B"/"II"/"III"/"C" to the END of the name keeps
# senior clubs such as "B. Banja Luka" out.
YOUTH_OR_RESERVE = re.compile(
    r"(?:\bU\s?1[0-9]\b|\bU\s?2[0-3]\b|Yth\.?|Youth|Jgd\.?|Sub-\d|Academy"
    r"|Castilla|Res\.|\bII$|\bIII$|\bB$|\bC$)"
)

# ---------------------------------------------------------------------------
# Transfer type
# ---------------------------------------------------------------------------

# Transfermarkt's own fee label -> our normalized type. This mapping is the
# whole reason Stage 1C exists: the label is an explicit statement by the
# source, not an inference of ours.
LABEL_TO_TYPE = {
    "paid_transfer":        "permanent_transfer",
    "free_transfer":        "free_transfer",
    "loan_transfer":        "loan",
    "loan_fee":             "loan",
    "end_of_loan":          "loan_return",
    "end_of_loan_with_fee": "loan_return",
    "undisclosed":          "undisclosed_transfer",
    "no_fee_shown":         "no_fee_shown",
    "draft":                "other",
    "unknown":              "unknown",
}

# Labels that say nothing about the nature of the movement, and may therefore
# be refined by the internal-move rule. Every other label stands.
UNINFORMATIVE_LABELS = {"no_fee_shown", "undisclosed", "unknown"}

# Stage 1 heuristic class -> normalized type, used ONLY where no raw label
# exists. Kept deliberately coarse: the heuristic never earns a confident type.
HEURISTIC_TO_TYPE = {
    "permanent_with_fee":        "permanent_transfer",
    "loan_out":                  "loan",
    "loan_return":               "loan_return",
    "zero_fee_move":             "no_fee_shown",
    "no_fee_recorded":           "no_fee_shown",
    "arrival_from_no_club":      "free_transfer",
    "exit_to_no_club":           "other",
    "retirement":                "other",
    "same_club_record_artifact": "other",
}


def normalize_club_name(name) -> str:
    """Strip an academy/reserve suffix so sides of one organisation collide."""
    if pd.isna(name):
        return ""
    stripped = YOUTH_OR_RESERVE.sub("", str(name))
    return re.sub(r"\s+", " ", stripped).strip(" .-").casefold()


def is_youth_or_reserve(name) -> bool:
    return bool(YOUTH_OR_RESERVE.search(str(name))) if not pd.isna(name) else False


def classify_type(raw_label, heuristic_class, internal_move: bool) -> tuple[str, str]:
    """Return (transfer_type_normalized, transfer_type_source).

    Precedence, strictly:
      1. an explicit Transfermarkt label;
      2. the internal-move rule, but ONLY where the label says nothing about
         the movement (a labelled loan between a club and its own B team stays
         a loan);
      3. the Stage 1 timeline heuristic, only where no raw row was matched.
    """
    if raw_label and not pd.isna(raw_label) and raw_label in LABEL_TO_TYPE:
        normalized = LABEL_TO_TYPE[raw_label]
        if internal_move and raw_label in UNINFORMATIVE_LABELS:
            return "youth_or_internal", "club_name_pattern"
        return normalized, "transfermarkt_raw_label"
    if internal_move:
        return "youth_or_internal", "club_name_pattern"
    return HEURISTIC_TO_TYPE.get(heuristic_class, "unknown"), "stage1_heuristic"


# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------

# What the source told us about the fee. Deliberately separate from the amount.
#   disclosed_amount  a number was published
#   disclosed_free    Transfermarkt states "free transfer": an asserted zero
#   undisclosed       Transfermarkt shows "?": a deal happened, terms withheld
#   no_fee_shown      Transfermarkt shows "-": no fee cell at all
#   not_applicable    end of a loan; there is no fee concept
#   unknown           no raw row matched
FEE_DISCLOSURE = {
    "paid_transfer":        "disclosed_amount",
    "loan_fee":             "disclosed_amount",
    "end_of_loan_with_fee": "disclosed_amount",
    "free_transfer":        "disclosed_free",
    "undisclosed":          "undisclosed",
    "no_fee_shown":         "no_fee_shown",
    # A bare "loan transfer" shows no fee. We cannot tell a genuinely free loan
    # from a loan whose fee was not published, so we do NOT assert zero.
    "loan_transfer":        "no_fee_shown",
    "end_of_loan":          "not_applicable",
    "draft":                "not_applicable",
    "unknown":              "unknown",
}


def split_fees(raw_label, fee_numeric_eur) -> dict:
    """Route one parsed amount into the one field where it belongs.

    A loan fee must never land in `permanent_transfer_fee`, and an amount paid
    on a return leg is neither, so it gets its own field rather than being
    forced into one of the other two.
    """
    amount = None if pd.isna(fee_numeric_eur) else float(fee_numeric_eur)
    fees = {
        "permanent_transfer_fee_eur": None,
        "loan_fee_eur": None,
        "fee_on_return_eur": None,
        "fee_disclosure_status": FEE_DISCLOSURE.get(raw_label, "unknown"),
    }
    if raw_label == "paid_transfer":
        fees["permanent_transfer_fee_eur"] = amount
    elif raw_label == "free_transfer":
        # Transfermarkt asserting "free transfer" is a positive statement that
        # no fee was paid, so 0 here is a disclosed value, not an imputed one.
        fees["permanent_transfer_fee_eur"] = 0.0
    elif raw_label == "loan_fee":
        fees["loan_fee_eur"] = amount
    elif raw_label == "end_of_loan_with_fee":
        fees["fee_on_return_eur"] = amount
    return fees


# ---------------------------------------------------------------------------
# Research population
# ---------------------------------------------------------------------------

# Types that can carry contract terms worth researching.
RESEARCHABLE_TYPES = {"permanent_transfer", "free_transfer", "loan", "undisclosed_transfer"}


def research_decision(row) -> tuple[bool, str]:
    """Should this event go to Stage 2? Returns (is_target, exclusion_reason).

    The policy in one sentence: an explicit monetary amount always qualifies an
    event; otherwise we require a senior-club move of a researchable type.
    """
    if row["involves_placeholder_club"]:
        return False, "placeholder_club_not_a_club_to_club_transfer"
    if row["is_internal_move"]:
        return False, "internal_registration_change_within_one_organisation"

    kind = row["transfer_type_normalized"]

    # A fee paid on a return leg is the trace of an exercised option or
    # obligation. Stage 1B found 43 of them; they are kept, not discarded.
    if kind == "loan_return":
        if row["return_leg_has_fee"]:
            return True, ""
        return False, "loan_return_bookkeeping_row_no_deal_terms"

    if kind == "youth_or_internal":
        return False, "youth_or_internal_movement"
    if kind == "other":
        return False, "not_a_standard_club_to_club_transfer"
    if kind == "unknown":
        return False, "transfer_type_could_not_be_established"
    if kind == "no_fee_shown":
        # "-" is the absence of a fee cell, not evidence that a deal with terms
        # took place. Excluded by default; the flags stay in the table so this
        # can be revisited without rebuilding.
        return False, "no_fee_shown_no_evidence_of_a_negotiated_deal"

    if kind in RESEARCHABLE_TYPES:
        if row["has_monetary_amount"]:
            return True, ""
        if row["is_senior_move"]:
            return True, ""
        return False, "youth_or_reserve_side_and_no_disclosed_amount"

    return False, "unhandled_transfer_type"
