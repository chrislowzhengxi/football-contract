"""Build the Stage 2 candidate queue and select a bounded pilot.

    python -m src.stage2.candidates

Writes, and spends nothing:
    stage2_candidate_queue.csv    every Stage 1C research target, categorised
                                  and prioritised
    stage2_pilot_selection.csv    the 26-event pilot, with the rule that chose
                                  each row

Identity is `player_id` throughout. `player_name` is carried for reading only:
267 names in the snapshot belong to more than one player.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..stage1.backbone import REBUILD_DIR

CANONICAL_CSV = REBUILD_DIR / "stage1c_canonical_transfers.csv"

# The raw snapshot was captured on this date; anything later was not observable
# when the backbone was built.
SNAPSHOT_DATE = pd.Timestamp("2026-06-12")
# Pilot events must already have happened, so reporting has had time to appear.
TODAY = pd.Timestamp("2026-09-21")
# "Recent" = the last three completed seasons, where contractual reporting is
# richest and club media is still online.
RECENT_FROM = pd.Timestamp("2023-07-01")

QUEUE_COLUMNS = [
    "priority", "candidate_category", "selection_reason",
    "event_id", "player_id", "player_name", "date_of_birth", "age_at_transfer",
    "transfer_date", "transfer_season", "from_club_name", "to_club_name",
    "transfer_type_normalized", "transfer_type_raw", "fee_disclosure_status",
    "permanent_transfer_fee_eur", "loan_fee_eur", "fee_on_return_eur",
    "market_value_eur", "fee_display_raw",
    "transfermarkt_transfer_id", "event_chain_id",
    "is_senior_move", "transfermarkt_future_transfer", "has_happened",
]

# Why each category is worth spending a search on. Ordered by priority.
CATEGORY_RATIONALE = {
    "fee_bearing_loan_return": (
        "Money attached to an 'End of loan' row. Stage 1C deliberately refuses to "
        "say what this amount is. An exercised purchase option or obligation is the "
        "most likely explanation, but only external evidence can establish it. "
        "43 events exist in total - the entire population is small enough to research."),
    "loan_with_fee": (
        "A loan with a published loan fee. A club that pays to borrow a player "
        "usually negotiates an option or obligation at the same time, so these are "
        "the densest expected source of option/obligation terms."),
    "high_value_undisclosed": (
        "Transfermarkt shows '?', meaning a deal happened and the fee was withheld. "
        "The highest-value examples available - note these are only EUR 4-25m, "
        "because '?' concentrates in small deals rather than big secretive ones."),
    "control_ordinary_permanent": (
        "An ordinary recent permanent transfer with a published mid-range fee. The "
        "control group: if add-ons and sell-ons are not recoverable HERE, they are "
        "not recoverable at scale anywhere."),
    "high_value_recent_permanent": (
        "Large recent permanent transfer. Richest reporting, but unrepresentative - "
        "queued below the pilot as scale-up candidates."),
    "other_research_target": (
        "Remaining Stage 1C research targets. Not prioritised for the pilot."),
}


def build_queue(canonical: pd.DataFrame) -> pd.DataFrame:
    """Categorise and prioritise every Stage 1C research target."""
    frame = canonical[canonical.is_research_target].copy()
    dates = pd.to_datetime(frame.transfer_date)
    frame["has_happened"] = dates <= TODAY
    recent = dates >= RECENT_FROM

    permanent = frame.permanent_transfer_fee_eur.fillna(0)
    market = frame.market_value_eur.fillna(0)

    # Priority 1 first, and each event lands in exactly one category.
    conditions = [
        (1, "fee_bearing_loan_return", frame.return_leg_has_fee),
        (2, "loan_with_fee", (frame.transfer_type_normalized == "loan")
                             & frame.loan_fee_eur.notna()),
        # EUR 4m, not a rounder number, because undisclosed deals are mostly
        # SMALL: the median recent "?" transfer has a market value of EUR 350k
        # and only 6 exceed EUR 5m. Transfermarkt usually obtains the fee for
        # big moves, so "?" concentrates in obscure deals. A higher bar would
        # leave the category empty.
        (3, "high_value_undisclosed", (frame.transfer_type_normalized == "undisclosed_transfer")
                                      & recent & (market >= 4_000_000)),
        (4, "control_ordinary_permanent", (frame.transfer_type_normalized == "permanent_transfer")
                                          & recent & permanent.between(5_000_000, 20_000_000)),
        (5, "high_value_recent_permanent", (frame.transfer_type_normalized == "permanent_transfer")
                                           & recent & (permanent > 20_000_000)),
    ]
    frame["priority"] = 6
    frame["candidate_category"] = "other_research_target"
    assigned = pd.Series(False, index=frame.index)
    for priority, name, mask in conditions:
        take = mask.fillna(False) & ~assigned
        frame.loc[take, "priority"] = priority
        frame.loc[take, "candidate_category"] = name
        assigned |= take
    frame["selection_reason"] = frame.candidate_category.map(CATEGORY_RATIONALE)

    # Largest relevant amount first inside each band, then most recent, then a
    # stable tiebreak on event_id so the ordering is fully reproducible.
    frame["_amount"] = frame[["fee_on_return_eur", "loan_fee_eur",
                              "permanent_transfer_fee_eur", "market_value_eur"]].max(axis=1)
    frame = frame.sort_values(
        ["priority", "_amount", "transfer_date", "event_id"],
        ascending=[True, False, False, True],
    )
    return frame[QUEUE_COLUMNS].reset_index(drop=True)


def _pick(pool: pd.DataFrame, by: str, n: int, rule: str) -> pd.DataFrame:
    chosen = pool.sort_values([by, "transfer_date", "event_id"],
                              ascending=[False, False, True]).head(n).copy()
    chosen["pilot_rule"] = rule
    return chosen


def select_pilot(queue: pd.DataFrame) -> pd.DataFrame:
    """A 26-event pilot. Every rule is deterministic and stated on the row.

    The pilot is built to answer "which contract fields are observable, for
    which kinds of transfer", not to maximise fields filled. That is why it
    spans the value range and carries a control group.
    """
    # Only events that have actually happened, so reporting exists to find.
    live = queue[queue.has_happened]
    picks: list[pd.DataFrame] = []

    # P1 - the whole population is only 43, so take the 8 largest.
    picks.append(_pick(live[live.candidate_category == "fee_bearing_loan_return"],
                       "fee_on_return_eur", 8,
                       "8 largest fee_on_return_eur among the 43 that exist"))

    # P2 - 6 largest, PLUS 4 spread down the value range. The spread is the
    # point: if only headline loans are reported, that is the finding.
    loans = live[live.candidate_category == "loan_with_fee"]
    top_loans = _pick(loans, "loan_fee_eur", 6, "6 largest loan_fee_eur")
    picks.append(top_loans)
    rest = loans[~loans.event_id.isin(top_loans.event_id)].sort_values(
        ["loan_fee_eur", "transfer_date", "event_id"], ascending=[False, False, True])
    if len(rest):
        # Fixed quantile positions - reproducible, and deliberately not the top.
        positions = sorted({int(round(q * (len(rest) - 1))) for q in (0.10, 0.30, 0.55, 0.80)})
        spread = rest.iloc[positions].copy()
        spread["pilot_rule"] = (
            "loan_fee_eur at fixed quantiles 10/30/55/80% of the 2,050 - tests "
            "whether smaller loan fees are reported at all")
        picks.append(spread)

    # P3 - 4 highest-market-value undisclosed moves.
    picks.append(_pick(live[live.candidate_category == "high_value_undisclosed"],
                       "market_value_eur", 4,
                       "4 highest market_value_eur among recent undisclosed senior moves"))

    # P4 - the control group.
    picks.append(_pick(live[live.candidate_category == "control_ordinary_permanent"],
                       "permanent_transfer_fee_eur", 4,
                       "4 largest ordinary permanent transfers in the EUR 5-20m band (control)"))

    pilot = pd.concat(picks, ignore_index=True)
    pilot = pilot.sort_values(["priority", "event_id"]).reset_index(drop=True)
    pilot.insert(0, "pilot_rank", range(1, len(pilot) + 1))
    return pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage 2 candidate queue and pilot")
    parser.add_argument("--canonical", default=str(CANONICAL_CSV))
    parser.add_argument("--output-dir", default=str(REBUILD_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    canonical = pd.read_csv(args.canonical, low_memory=False)
    queue = build_queue(canonical)
    pilot = select_pilot(queue)
    queue.to_csv(output_dir / "stage2_candidate_queue.csv", index=False)
    pilot.to_csv(output_dir / "stage2_pilot_selection.csv", index=False)

    summary = {
        "canonical_events": len(canonical),
        "research_targets_queued": len(queue),
        "queue_by_category": queue.candidate_category.value_counts().to_dict(),
        "pilot_events": len(pilot),
        "pilot_by_category": pilot.candidate_category.value_counts().to_dict(),
        "distinct_players_in_pilot": int(pilot.player_id.nunique()),
        "api_calls_made": 0,
    }
    (output_dir / "stage2_pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
