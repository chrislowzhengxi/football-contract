"""Derived movement classification for raw Transfermarkt transfer rows.

IMPORTANT: the raw ``transfers`` table has ten columns and none of them says
whether a movement was a loan, a free transfer, or a permanent deal. Everything
in this module except the pseudo-club rules is a HEURISTIC over the sequence of
a player's rows. ``movement_class_basis`` records which is which.
"""
from __future__ import annotations

import pandas as pd

# Transfermarkt uses these as placeholder "clubs". They are explicit raw markers,
# not real clubs, so rules that read them are raw facts rather than heuristics.
PSEUDO_CLUB_NAMES = {"Without Club", "Retired", "Career break", "Ban", "Unknown"}

# Longest gap between the outbound leg and the return leg that we are willing to
# read as one loan. 18-month loans exist; 98.4% of all reverse-direction round
# trips in the snapshot close within 750 days, 91.6% within 400.
MAX_LOAN_DAYS = 550

# A move back to the club that just hosted a loan, this soon after the return
# leg, is the "loan turned permanent" pattern (Mbappé 2018, Morata 2020).
MAX_DAYS_AFTER_LOAN_RETURN = 31

MOVEMENT_CLASSES = (
    "same_club_record_artifact",
    "retirement",
    "exit_to_no_club",
    "arrival_from_no_club",
    "loan_out",
    "loan_return",
    "permanent_with_fee",
    "zero_fee_move",
    "no_fee_recorded",
)

# How much the class name is actually worth:
#   explicit_raw_marker - Transfermarkt itself put a marker in the row.
#   raw_fee_value       - the class is nothing more than a restatement of the raw
#                         fee value. It does NOT establish permanent/free/loan.
#   derived_heuristic   - we inferred it from the shape of the player's timeline.
CLASS_BASIS = {
    "same_club_record_artifact": "explicit_raw_marker",
    "retirement": "explicit_raw_marker",
    "exit_to_no_club": "explicit_raw_marker",
    "arrival_from_no_club": "explicit_raw_marker",
    "permanent_with_fee": "raw_fee_value",
    "zero_fee_move": "raw_fee_value",
    "no_fee_recorded": "raw_fee_value",
    "loan_out": "derived_heuristic",
    "loan_return": "derived_heuristic",
}


def _fee_is_zero_or_missing(fee) -> bool:
    return pd.isna(fee) or float(fee) == 0.0


def order_player_timeline(transfers: pd.DataFrame) -> pd.DataFrame:
    """Return the rows in a deterministic, chain-aware order per player.

    Sorting by date alone is ambiguous: 118 player/date pairs in the snapshot
    hold more than one row (a loan return and the next loan are often stamped
    with the same day). Within such a group we greedily pick the row whose
    ``from_club_id`` continues from where the player was last seen, and flag the
    group when that is impossible so the ambiguity stays visible.
    """
    frame = transfers.sort_values(
        ["player_id", "transfer_date", "from_club_id", "to_club_id"], kind="mergesort"
    ).reset_index(drop=True)
    order: list[int] = []
    ambiguous: set[int] = set()
    for _, player_rows in frame.groupby("player_id", sort=False):
        current_club = None
        for _, day_rows in player_rows.groupby("transfer_date", sort=True):
            remaining = list(day_rows.index)
            while remaining:
                follows = [i for i in remaining if frame.at[i, "from_club_id"] == current_club]
                if follows:
                    chosen = follows[0]
                elif len(remaining) == 1:
                    chosen = remaining[0]
                else:
                    chosen = remaining[0]
                    ambiguous.update(remaining)
                remaining.remove(chosen)
                order.append(chosen)
                current_club = frame.at[chosen, "to_club_id"]
    ordered = frame.loc[order].reset_index(drop=True)
    ordered["derived_same_date_order_ambiguous"] = [index in ambiguous for index in order]
    return ordered


def classify_movements(ordered: pd.DataFrame) -> pd.DataFrame:
    """Add the derived movement columns to a chain-ordered transfer frame."""
    rows = ordered.to_dict("records")
    count = len(rows)
    player = [row["player_id"] for row in rows]
    fee = [row["transfer_fee"] for row in rows]
    date = [pd.Timestamp(row["transfer_date"]) for row in rows]
    from_id = [row["from_club_id"] for row in rows]
    to_id = [row["to_club_id"] for row in rows]
    from_name = [row["from_club_name"] for row in rows]
    to_name = [row["to_club_name"] for row in rows]

    klass: list[str] = [""] * count
    rule: list[str] = [""] * count
    loan_partner: list[int | None] = [None] * count
    loan_days: list[float | None] = [None] * count

    def same_player_next(i: int) -> int | None:
        j = i + 1
        return j if j < count and player[j] == player[i] else None

    # Pass 1 - explicit raw markers and the fee-only fallbacks.
    for i in range(count):
        if from_id[i] == to_id[i]:
            klass[i], rule[i] = "same_club_record_artifact", "from_club_id == to_club_id"
        elif to_name[i] == "Retired":
            klass[i], rule[i] = "retirement", "to_club_name == 'Retired'"
        elif to_name[i] in PSEUDO_CLUB_NAMES:
            klass[i], rule[i] = "exit_to_no_club", f"to_club_name == '{to_name[i]}'"
        elif from_name[i] in PSEUDO_CLUB_NAMES:
            klass[i], rule[i] = "arrival_from_no_club", f"from_club_name == '{from_name[i]}'"
        elif not pd.isna(fee[i]) and float(fee[i]) > 0:
            klass[i], rule[i] = "permanent_with_fee", "transfer_fee > 0"
        elif not pd.isna(fee[i]):
            klass[i], rule[i] = "zero_fee_move", "transfer_fee == 0"
        else:
            klass[i], rule[i] = "no_fee_recorded", "transfer_fee IS NULL"

    # Pass 2 - loan pairs. Only rows that pass 1 left as zero/missing-fee moves
    # can be loan legs, so real money and pseudo-club rows are never overwritten.
    loanable = {"zero_fee_move", "no_fee_recorded"}
    for i in range(count):
        j = same_player_next(i)
        if j is None or klass[i] not in loanable or klass[j] not in loanable:
            continue
        if not (from_id[j] == to_id[i] and to_id[j] == from_id[i]):
            continue
        if not (_fee_is_zero_or_missing(fee[i]) and _fee_is_zero_or_missing(fee[j])):
            continue
        gap = (date[j] - date[i]).days
        if gap < 0 or gap > MAX_LOAN_DAYS:
            continue
        klass[i], rule[i] = "loan_out", (
            f"next row for the player reverses this move within {MAX_LOAN_DAYS} days "
            "and both legs carry no fee"
        )
        klass[j], rule[j] = "loan_return", "reverses the immediately preceding loan_out leg"
        loan_partner[i], loan_partner[j] = j, i
        loan_days[i] = loan_days[j] = gap

    # Pass 3 - the "permanent move after a loan" pattern.
    follows_loan_return = [False] * count
    days_after_return: list[float | None] = [None] * count
    for i in range(1, count):
        if player[i] != player[i - 1] or klass[i - 1] != "loan_return":
            continue
        # The previous row brought the player home from club X; this row sends
        # him back to club X.
        if to_id[i] != from_id[i - 1]:
            continue
        gap = (date[i] - date[i - 1]).days
        if 0 <= gap <= MAX_DAYS_AFTER_LOAN_RETURN:
            follows_loan_return[i] = True
            days_after_return[i] = gap

    result = ordered.copy()
    result["derived_movement_class"] = klass
    result["derived_movement_rule"] = rule
    result["derived_movement_basis"] = [CLASS_BASIS[value] for value in klass]
    result["derived_loan_partner_row"] = loan_partner
    result["derived_loan_duration_days"] = loan_days
    result["derived_follows_loan_return"] = follows_loan_return
    result["derived_days_after_loan_return"] = days_after_return
    return result


def fee_semantics(fee) -> str:
    """What the raw ``transfer_fee`` value can and cannot mean."""
    if pd.isna(fee):
        return "no_fee_recorded"
    if float(fee) == 0.0:
        return "zero_recorded"
    return "fee_reported"
