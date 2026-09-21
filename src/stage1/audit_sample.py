"""Pick ~20 Stage 1 events that stress-test the backbone, and explain each one.

Selection reads the RAW columns (fee, dates, club ids, the player's raw row
sequence). It never trusts a normalized type field to decide what a case is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_DATABASE
from ..load_data import connect_database
from .backbone import REBUILD_DIR

PSEUDO = {"Without Club", "Retired", "Career break", "Ban", "Unknown"}
YOUTH_PATTERN = r"(?:U1[0-9]|U2[0-3]|Yth|Youth|Jgd|Sub-1|Academy|Castilla|Res\.| II$| B$)"


def _raw_reference(row: pd.Series) -> str:
    return (
        "transfers WHERE player_id={} AND transfer_date=DATE '{}' "
        "AND from_club_id={} AND to_club_id={}"
    ).format(
        int(row.raw_player_id), pd.Timestamp(row.raw_transfer_date).date(),
        int(row.raw_from_club_id), int(row.raw_to_club_id),
    )


def select_cases(events: pd.DataFrame, covered_club_ids: set[int]) -> pd.DataFrame:
    """Return one event per stress-test category, chosen deterministically."""
    frame = events.copy()
    frame["raw_transfer_date"] = pd.to_datetime(frame["raw_transfer_date"])
    frame["_both_covered"] = frame.raw_from_club_id.isin(covered_club_ids) & frame.raw_to_club_id.isin(covered_club_ids)
    frame["_senior"] = ~(
        frame.raw_from_club_name.str.contains(YOUTH_PATTERN, na=False, regex=True)
        | frame.raw_to_club_name.str.contains(YOUTH_PATTERN, na=False, regex=True)
        | frame.raw_from_club_name.isin(PSEUDO) | frame.raw_to_club_name.isin(PSEUDO)
    )
    # Prefer cases a human can check quickly: well-covered clubs, decent market value.
    frame["_checkable"] = frame._both_covered & frame._senior & (frame.raw_market_value_in_eur.fillna(0) >= 1_000_000)
    multi = frame.groupby("raw_player_id").size()
    repeat_players = set(multi[multi >= 6].index)

    picks: list[tuple[str, pd.DataFrame, str]] = []

    def add(category: str, subset: pd.DataFrame, reason: str, sort_by=None, ascending=False, n=1):
        if subset.empty:
            return
        ordered = subset.sort_values(sort_by, ascending=ascending) if sort_by else subset.sort_values(
            ["raw_transfer_date", "raw_player_id"], ascending=False)
        picks.append((category, ordered.head(n), reason))

    add("ordinary_permanent_transfer",
        frame[frame._checkable & (frame.derived_movement_class == "permanent_with_fee")
              & ~frame.derived_follows_loan_return & frame.raw_transfer_fee.between(20e6, 60e6)
              & (frame.raw_transfer_date.dt.year == 2024)],
        "raw transfer_fee is a positive number, no surrounding loan legs: the plain buy case")

    add("ordinary_loan",
        frame[frame._checkable & (frame.derived_movement_class == "loan_out")
              & frame.derived_loan_duration_days.between(300, 370)
              & (frame.raw_market_value_in_eur >= 10e6)
              & (frame.raw_transfer_date.dt.year == 2024)],
        "raw rows show A->B then B->A about one season later, both with fee 0: the plain loan case")

    add("loan_return",
        frame[frame._checkable & (frame.derived_movement_class == "loan_return")
              & (frame.raw_transfer_date.dt.year == 2025)
              & frame.derived_date_on_season_boundary],
        "raw row reverses the previous move on a season boundary date: the loan-return bookkeeping row")

    add("permanent_after_loan",
        frame[frame._checkable & frame.derived_follows_loan_return & (frame.raw_transfer_fee > 5e6)],
        "raw rows show loan out, return, then a fee-bearing move back to the same club within days")

    add("free_transfer_out_of_contract",
        frame[frame._senior & frame._both_covered & (frame.derived_movement_class == "zero_fee_move")
              & (frame.raw_market_value_in_eur.fillna(0) >= 5e6)
              & (frame.raw_transfer_date.dt.month == 7) & (frame.raw_transfer_date.dt.day == 1)
              & (frame.raw_transfer_date.dt.year >= 2024)],
        "raw fee is exactly 0 on 1 July between two senior clubs with no return leg: reads as a free transfer, but 0 also covers loans and undisclosed fees")

    add("signed_as_free_agent",
        frame[(frame.derived_movement_class == "arrival_from_no_club")
              & frame.raw_to_club_id.isin(covered_club_ids)
              & (frame.raw_market_value_in_eur.fillna(0) >= 3e6)],
        "raw from_club_name is the placeholder 'Without Club': an explicit raw marker, not a guess")

    add("exit_to_no_club",
        frame[(frame.derived_movement_class == "exit_to_no_club")
              & frame.raw_from_club_id.isin(covered_club_ids)
              & (frame.raw_market_value_in_eur.fillna(0) >= 3e6)],
        "raw to_club_name is the placeholder 'Without Club': contract ended or player released")

    add("highest_fee",
        frame[frame.raw_transfer_fee > 0], "largest raw transfer_fee in the snapshot",
        sort_by="raw_transfer_fee")

    add("recent_high_fee",
        frame[(frame.raw_transfer_date.dt.year >= 2026) & (frame.raw_transfer_fee > 20e6)],
        "most recent large fee: tests whether the snapshot edge is handled",
        sort_by="raw_transfer_date")

    add("older_transfer",
        frame[frame._checkable & (frame.raw_transfer_fee > 10e6)
              & (frame.raw_transfer_date.dt.year <= 2008)],
        "pre-2009 event: tests date handling and market-value availability far from the snapshot",
        sort_by="raw_transfer_date", ascending=True)

    add("same_player_multiple_moves",
        frame[frame.raw_player_id.isin(repeat_players) & frame._checkable
              & (frame.raw_transfer_date.dt.year >= 2024)],
        "player has 6+ raw rows: tests event_id uniqueness and timeline ordering", n=2)

    add("scheduled_future_loan_return",
        frame[(frame.derived_movement_class == "loan_return")
              & (frame.raw_transfer_date > pd.Timestamp("2026-08-28"))
              & (frame.raw_market_value_in_eur.fillna(0) >= 5e6)],
        "raw transfer_date is in the future relative to the snapshot: a pre-registered loan end, not a completed event",
        sort_by="raw_market_value_in_eur")

    add("long_loan_over_one_season",
        frame[frame._checkable & (frame.derived_movement_class == "loan_out")
              & frame.derived_loan_duration_days.between(500, 550)],
        "raw legs are ~18 months apart: sits at the edge of the 550-day loan window")

    add("early_loan_recall",
        frame[frame._checkable & (frame.derived_movement_class == "loan_return")
              & (frame.derived_loan_duration_days < 200)
              & ~frame.derived_date_on_season_boundary
              & (frame.raw_transfer_date.dt.year >= 2024)],
        "return leg lands mid-season on a non-boundary date: probably a cut-short loan")

    nxt = frame.groupby("raw_player_id").shift(-1)
    reversed_next = (nxt.raw_from_club_id == frame.raw_to_club_id) & (nxt.raw_to_club_id == frame.raw_from_club_id)
    add("round_trip_that_is_not_a_loan",
        frame[frame._checkable & (frame.derived_movement_class == "permanent_with_fee")
              & reversed_next & (frame.raw_transfer_fee > 10e6)],
        "fee-bearing move that the player's NEXT raw row reverses: a genuine two-way permanent pair, "
        "the case the loan heuristic must not capture",
        sort_by="raw_transfer_fee")

    add("undisclosed_or_no_fee_between_senior_clubs",
        frame[frame._senior & frame._both_covered & frame.raw_transfer_fee.isna()
              & (frame.raw_market_value_in_eur.fillna(0) >= 3e6)
              & (frame.raw_transfer_date.dt.year >= 2022)],
        "raw transfer_fee is NULL between two senior clubs: Transfermarkt showed no fee at all")

    add("youth_or_internal_promotion",
        frame[frame.raw_transfer_fee.isna() & ~frame._senior
              & frame.raw_to_club_id.isin(covered_club_ids)
              & (frame.raw_transfer_date.dt.year >= 2024)
              & frame.raw_from_club_name.str.contains("U19|U21|B$| II$", na=False, regex=True)],
        "youth or reserve side to first team: a registration change, not a transfer")

    add("same_day_ordering_ambiguous",
        frame[frame.derived_same_date_order_ambiguous],
        "two raw rows share a player and a date and cannot be chained: ordering, and therefore loan pairing, is uncertain here",
        sort_by="raw_transfer_date", n=2)

    add("cross_league_transfer",
        frame[frame._checkable & (frame.derived_movement_class == "permanent_with_fee")
              & (frame.raw_transfer_fee > 30e6) & (frame.raw_transfer_date.dt.year == 2025)],
        "large fee between clubs in different domestic competitions")

    add("zero_fee_between_big_clubs",
        frame[frame._senior & frame._both_covered & (frame.raw_transfer_fee == 0)
              & (frame.raw_market_value_in_eur.fillna(0) >= 20e6)
              & (frame.derived_movement_class == "zero_fee_move")],
        "raw fee 0 with a very high market value: exactly where 'fee 0' is least believable",
        sort_by="raw_market_value_in_eur")

    rows = []
    seen: set[str] = set()
    for category, subset, reason in picks:
        for _, row in subset.iterrows():
            if row.event_id in seen:
                continue
            seen.add(row.event_id)
            rows.append({"case_category": category, "reason_selected": reason, **row.to_dict()})
    return pd.DataFrame(rows)


def build_case_table(cases: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": cases.event_id,
        "player": cases.raw_player_name,
        "from_club": cases.raw_from_club_name,
        "to_club": cases.raw_to_club_name,
        "transfer_date": pd.to_datetime(cases.raw_transfer_date).dt.date,
        "fee": cases.raw_transfer_fee,
        "market_value": cases.raw_market_value_in_eur,
        # The column the legacy parser produced. It is blank for every row on
        # purpose: enrich_structured.py assigns None to it. See the audit.
        "current_transfer_type": "",
        "stage1_derived_movement_class": cases.derived_movement_class,
        "stage1_movement_basis": cases.derived_movement_basis,
        "case_category": cases.case_category,
        "reason_selected": cases.reason_selected,
        "raw_record_reference": [_raw_reference(row) for _, row in cases.iterrows()],
        "validation_status": "pending_human_verification",
        "validation_notes": "",
    })


def player_timeline(connection, player_id: int) -> pd.DataFrame:
    return connection.execute(
        "SELECT transfer_date, transfer_season, from_club_name, to_club_name, "
        "transfer_fee, market_value_in_eur FROM transfers WHERE player_id = ? "
        "ORDER BY transfer_date", [int(player_id)],
    ).fetchdf()


def _money(value) -> str:
    if pd.isna(value):
        return "NULL"
    return f"EUR {float(value):,.0f}"


def write_markdown(cases: pd.DataFrame, connection, path: Path) -> None:
    lines = ["# Stage 1 - 20-event audit sample\n",
             "Each case shows the normalized event, the raw Transfermarkt values behind it, and "
             "how every normalized value was produced. You should be able to judge the parser "
             "from this file without reading any Python.\n",
             "`fee` values are euros. `NULL` means Transfermarkt recorded no fee at all; `0` "
             "means Transfermarkt recorded something non-numeric (free transfer, loan transfer, "
             "end of loan, or a loan fee that this dataset drops).\n"]
    for number, (_, case) in enumerate(cases.iterrows(), start=1):
        date = pd.Timestamp(case.raw_transfer_date).date()
        lines.append(f"\n---\n\n## {number}. {case.raw_player_name}: "
                     f"{case.raw_from_club_name} -> {case.raw_to_club_name} ({date})\n")
        lines.append(f"**Case category:** `{case.case_category}`  ")
        lines.append(f"**Why selected:** {case.reason_selected}  ")
        lines.append(f"**event_id:** `{case.event_id}`\n")

        lines.append("### 1. Normalized event\n")
        lines.append("| field | value |")
        lines.append("| --- | --- |")
        age = "NULL" if pd.isna(case.derived_age_at_transfer) else int(case.derived_age_at_transfer)
        for label, value in [
            ("player", case.raw_player_name), ("age at transfer", age),
            ("from club", case.raw_from_club_name), ("to club", case.raw_to_club_name),
            ("date", date), ("season", case.raw_transfer_season),
            ("fee", _money(case.raw_transfer_fee)),
            ("market value", _money(case.raw_market_value_in_eur)),
            ("derived_movement_class", f"`{case.derived_movement_class}`"),
            ("derived_movement_basis", f"`{case.derived_movement_basis}`"),
            ("legacy transfer_type", "(blank - never populated)"),
        ]:
            lines.append(f"| {label} | {value} |")
        lines.append("")

        lines.append("### 2. Raw Transfermarkt record\n")
        lines.append(f"```sql\nSELECT * FROM {_raw_reference(case)}\n```\n")
        lines.append("| raw column | raw value |")
        lines.append("| --- | --- |")
        for column in ["raw_player_id", "raw_transfer_date", "raw_transfer_season",
                       "raw_from_club_id", "raw_to_club_id", "raw_from_club_name",
                       "raw_to_club_name", "raw_transfer_fee", "raw_market_value_in_eur",
                       "raw_player_name"]:
            value = case[column]
            shown = "NULL" if pd.isna(value) else value
            lines.append(f"| `transfers.{column[4:]}` | {shown} |")
        lines.append("")
        lines.append(f"`players.date_of_birth` = {case.joined_date_of_birth}, "
                     f"`players.name` = {case.joined_player_name_players}, "
                     f"`players.position` = {case.joined_position}\n")

        timeline = player_timeline(connection, case.raw_player_id)
        lines.append("Surrounding raw rows for this player (this is the only evidence the loan "
                     "rules have):\n")
        lines.append("| date | season | from | to | fee | market value |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, step in timeline.iterrows():
            marker = " **<- this event**" if (
                pd.Timestamp(step.transfer_date).date() == date
                and step.from_club_name == case.raw_from_club_name
                and step.to_club_name == case.raw_to_club_name) else ""
            lines.append(
                f"| {pd.Timestamp(step.transfer_date).date()} | {step.transfer_season} | "
                f"{step.from_club_name} | {step.to_club_name} | {_money(step.transfer_fee)} | "
                f"{_money(step.market_value_in_eur)}{marker} |")
        lines.append("")

        lines.append("### 3. Transformation\n")
        lines.append(f"**Player:** raw `transfers.player_name` = \"{case.raw_player_name}\"; "
                     f"`players.name` = \"{case.joined_player_name_players}\" joined on "
                     f"`player_id = {case.raw_player_id}`. Kept verbatim; no name matching, no fuzzy join.\n")
        lines.append(f"**From club:** raw `transfers.from_club_name` = \"{case.raw_from_club_name}\" "
                     f"(id {case.raw_from_club_id}) -> normalized without modification. Direction is "
                     "taken from the raw column names, not inferred.\n")
        lines.append(f"**To club:** raw `transfers.to_club_name` = \"{case.raw_to_club_name}\" "
                     f"(id {case.raw_to_club_id}) -> normalized without modification.\n")
        lines.append(f"**Date:** raw `transfers.transfer_date` = {date} -> copied. No precision is "
                     "added. Transfermarkt files many events on bookkeeping days "
                     f"(1 July, 30 June, 31 December); this one "
                     f"{'IS' if case.derived_date_on_season_boundary else 'is NOT'} on such a day.\n")
        lines.append(f"**Age:** `{date}` minus `players.date_of_birth` = "
                     f"{case.joined_date_of_birth}, with the birthday-not-yet-reached adjustment "
                     f"-> **{age}**. Derived, but from two exact raw dates.\n")
        fee_note = {
            "fee_reported": "a real number, so money was reported for this move",
            "zero_recorded": "exactly 0. Transfermarkt collapses 'free transfer', 'loan transfer', "
                             "'End of loan' and 'loan fee: EUR x' into this value. It does NOT mean no money moved",
            "no_fee_recorded": "NULL. Transfermarkt showed no fee cell at all (youth move, free "
                               "agency, retirement, or simply unknown)",
        }[case.derived_fee_semantics]
        lines.append(f"**Fee:** raw `transfers.transfer_fee` = {_money(case.raw_transfer_fee)} -> "
                     f"copied verbatim, no parsing or unit conversion. Semantics: {fee_note}. "
                     "There is no loan-fee column anywhere in this dataset.\n")
        mv_line = (f"**Market value:** raw `transfers.market_value_in_eur` = "
                   f"{_money(case.raw_market_value_in_eur)}. Cross-checked against "
                   f"`player_valuations` as of the transfer date: "
                   f"{_money(case.derived_mv_asof_eur)} dated {case.derived_mv_asof_date} "
                   f"({case.derived_mv_asof_lag_days} days earlier) - "
                   f"{'matches' if case.derived_mv_raw_matches_asof else 'DOES NOT match'}. ")
        mv_line += (f"The legacy parser instead used the *nearest* valuation in either direction: "
                    f"{_money(case.derived_mv_nearest_eur)} dated {case.derived_mv_nearest_date}.")
        lines.append(mv_line + "\n")
        rule = case.derived_movement_rule
        lines.append(f"**Movement class:** `{case.derived_movement_class}`, basis "
                     f"`{case.derived_movement_basis}`. Rule that fired: {rule}.")
        if case.derived_movement_basis == "derived_heuristic":
            lines.append(f"  Paired leg: `{case.derived_loan_partner_event_id}`, "
                         f"{case.derived_loan_duration_days} days apart. "
                         "Limitation: nothing in the raw data says 'loan'. This is inferred purely "
                         "from the reverse-direction pair and the absence of a fee on both legs.\n")
        elif case.derived_movement_basis == "raw_fee_value":
            lines.append("  Limitation: this class only restates the raw fee. It does not establish "
                         "permanent vs free vs loan. Stage 5 must settle that from sources.\n")
        else:
            lines.append("  This comes from an explicit Transfermarkt placeholder value, so it is a "
                         "raw fact rather than an inference.\n")
        if case.derived_follows_loan_return:
            lines.append(f"**Loan-to-permanent pattern:** the previous raw row returned the player "
                         f"from this same club {case.derived_days_after_loan_return:.0f} days "
                         "earlier, so this looks like a purchase option or obligation being "
                         "exercised. Stage 5 must confirm which.\n")
        if case.derived_same_date_order_ambiguous:
            lines.append("**Ordering warning:** another raw row shares this player and date and the "
                         "two cannot be chained by club. The loan pairing here is unreliable.\n")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: build the 20-event audit sample")
    parser.add_argument("--db", default=str(DEFAULT_DATABASE))
    parser.add_argument("--input", default=str(REBUILD_DIR / "stage1_normalized_transfers.csv"))
    parser.add_argument("--output-dir", default=str(REBUILD_DIR))
    args = parser.parse_args()
    connection = connect_database(Path(args.db), read_only=True)
    covered = {int(row[0]) for row in connection.execute(
        "SELECT CAST(club_id AS INTEGER) FROM clubs").fetchall()}
    events = pd.read_csv(args.input, low_memory=False)
    cases = select_cases(events, covered)
    output_dir = Path(args.output_dir)
    build_case_table(cases).to_csv(output_dir / "stage1_validation_20.csv", index=False)
    write_markdown(cases, connection, output_dir / "stage1_validation_20.md")
    print(f"selected {len(cases)} cases across {cases.case_category.nunique()} categories")


if __name__ == "__main__":
    main()
