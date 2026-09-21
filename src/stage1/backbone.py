"""Stage 1 build: raw Transfermarkt records -> normalized transfer events.

Run:
    python -m src.stage1.backbone                       # whole snapshot
    python -m src.stage1.backbone --club Porto --club Benfica --season 24/25

Outputs go to data/outputs/rebuild/ and never touch data/raw or the historical
outputs under data/processed/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_DATABASE, ROOT
from ..enrich_structured import age_at_transfer, stable_event_id
from ..load_data import connect_database, inspect_schema, require_columns
from .movement import classify_movements, fee_semantics, order_player_timeline

REBUILD_DIR = ROOT / "data" / "outputs" / "rebuild"

RAW_TRANSFER_COLUMNS = [
    "player_id", "transfer_date", "transfer_season", "from_club_id", "to_club_id",
    "from_club_name", "to_club_name", "transfer_fee", "market_value_in_eur", "player_name",
]

# Dates Transfermarkt files on the last day of a competition season. A return
# leg landing here is the usual bookkeeping date, not a real event date.
SEASON_BOUNDARY_DAYS = {(6, 30), (5, 31), (12, 31), (7, 1), (1, 1)}


def load_raw_transfers(connection) -> pd.DataFrame:
    schema = inspect_schema(connection)
    require_columns(schema, "transfers", RAW_TRANSFER_COLUMNS)
    frame = connection.execute(
        "SELECT " + ", ".join(RAW_TRANSFER_COLUMNS) + " FROM transfers"
    ).fetchdf()
    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"])
    frame["transfer_fee"] = frame["transfer_fee"].astype("float64")
    frame["market_value_in_eur"] = frame["market_value_in_eur"].astype("float64")
    return frame


def market_value_lookups(connection, events: pd.DataFrame) -> pd.DataFrame:
    """As-of and nearest market values from ``player_valuations``.

    Two joins, kept side by side on purpose:

    * ``asof``   - last valuation dated on or before the transfer. No look-ahead.
    * ``nearest``- closest valuation in either direction. This is what the legacy
      parser used; it is reproduced only so the difference stays measurable.
    """
    valuations = connection.execute(
        "SELECT player_id, date, market_value_in_eur FROM player_valuations"
    ).fetchdf()
    valuations["date"] = pd.to_datetime(valuations["date"])
    valuations["market_value_in_eur"] = valuations["market_value_in_eur"].astype("float64")
    right = valuations.sort_values(["date", "player_id"], kind="mergesort")

    keys = events[["player_id", "transfer_date"]].drop_duplicates()
    left = keys.sort_values(["transfer_date", "player_id"], kind="mergesort")
    out = keys.copy()
    for direction, prefix in (("backward", "asof"), ("nearest", "nearest")):
        matched = pd.merge_asof(
            left, right, left_on="transfer_date", right_on="date",
            by="player_id", direction=direction,
        )
        matched = matched.rename(columns={
            "market_value_in_eur": f"derived_mv_{prefix}_eur",
            "date": f"derived_mv_{prefix}_date",
        })[["player_id", "transfer_date", f"derived_mv_{prefix}_eur", f"derived_mv_{prefix}_date"]]
        out = out.merge(matched, on=["player_id", "transfer_date"], how="left")
    out["derived_mv_asof_lag_days"] = (
        out["transfer_date"] - out["derived_mv_asof_date"]
    ).dt.days
    out["derived_mv_nearest_abs_lag_days"] = (
        out["derived_mv_nearest_date"] - out["transfer_date"]
    ).dt.days.abs()
    return out


def add_event_chains(events: pd.DataFrame) -> pd.DataFrame:
    """Link a move that immediately continues the previous one (B->A then A->C).

    Same rule as the legacy parser: same player, this row leaves the club the
    previous row arrived at, within 14 days. Both events are kept.
    """
    chain_ids: list[str] = []
    previous: dict | None = None
    for row in events.itertuples():
        joins = (
            previous is not None
            and row.raw_player_id == previous["player"]
            and row.raw_from_club_id == previous["to_club"]
            and (pd.Timestamp(row.raw_transfer_date) - previous["date"]).days <= 14
        )
        chain_id = previous["chain"] if joins else "chain_" + row.event_id[3:]
        chain_ids.append(chain_id)
        previous = {
            "player": row.raw_player_id,
            "to_club": row.raw_to_club_id,
            "date": pd.Timestamp(row.raw_transfer_date),
            "chain": chain_id,
        }
    events = events.copy()
    events["derived_event_chain_id"] = chain_ids
    return events


def build_backbone(connection) -> pd.DataFrame:
    raw = load_raw_transfers(connection)
    ordered = order_player_timeline(raw)
    classified = classify_movements(ordered)

    players = connection.execute(
        "SELECT player_id, name, date_of_birth, position, sub_position, "
        "country_of_citizenship FROM players"
    ).fetchdf()
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"])
    merged = classified.merge(
        players.rename(columns={
            "name": "joined_player_name_players",
            "date_of_birth": "joined_date_of_birth",
            "position": "joined_position",
            "sub_position": "joined_sub_position",
            "country_of_citizenship": "joined_country_of_citizenship",
        }),
        on="player_id", how="left",
    )
    merged["joined_player_row_found"] = merged["joined_player_name_players"].notna()

    events = pd.DataFrame({
        "raw_player_id": merged["player_id"],
        "raw_player_name": merged["player_name"],
        "raw_transfer_date": merged["transfer_date"],
        "raw_transfer_season": merged["transfer_season"],
        "raw_from_club_id": merged["from_club_id"],
        "raw_to_club_id": merged["to_club_id"],
        "raw_from_club_name": merged["from_club_name"],
        "raw_to_club_name": merged["to_club_name"],
        "raw_transfer_fee": merged["transfer_fee"],
        "raw_market_value_in_eur": merged["market_value_in_eur"],
    })
    for column in [
        "joined_player_name_players", "joined_player_row_found", "joined_date_of_birth",
        "joined_position", "joined_sub_position", "joined_country_of_citizenship",
    ]:
        events[column] = merged[column]

    # event_id keeps the legacy recipe verbatim (sha256 over
    # "player_id|str(Timestamp)|from_club_id|to_club_id") so IDs stay joinable
    # with everything produced by the previous pipeline.
    events["event_id"] = [
        stable_event_id(*values)
        for values in events[[
            "raw_player_id", "raw_transfer_date", "raw_from_club_id", "raw_to_club_id",
        ]].itertuples(index=False, name=None)
    ]

    events["derived_age_at_transfer"] = [
        age_at_transfer(dob, moved)
        for dob, moved in zip(events["joined_date_of_birth"], events["raw_transfer_date"])
    ]
    events["derived_player_move_index"] = merged.groupby("player_id").cumcount()
    events["derived_fee_semantics"] = merged["transfer_fee"].map(fee_semantics)
    events["derived_movement_class"] = merged["derived_movement_class"]
    events["derived_movement_basis"] = merged["derived_movement_basis"]
    events["derived_movement_rule"] = merged["derived_movement_rule"]
    events["derived_loan_duration_days"] = merged["derived_loan_duration_days"]
    events["derived_follows_loan_return"] = merged["derived_follows_loan_return"]
    events["derived_days_after_loan_return"] = merged["derived_days_after_loan_return"]
    events["derived_same_date_order_ambiguous"] = merged["derived_same_date_order_ambiguous"]

    partner = merged["derived_loan_partner_row"]
    events["derived_loan_partner_event_id"] = [
        events["event_id"].iat[int(index)] if pd.notna(index) else None for index in partner
    ]
    events["derived_date_on_season_boundary"] = [
        (date.month, date.day) in SEASON_BOUNDARY_DAYS for date in events["raw_transfer_date"]
    ]

    values = market_value_lookups(connection, merged)
    events = events.merge(
        values.rename(columns={"player_id": "raw_player_id", "transfer_date": "raw_transfer_date"}),
        on=["raw_player_id", "raw_transfer_date"], how="left", validate="many_to_one",
    )
    events["derived_mv_raw_matches_asof"] = (
        events["raw_market_value_in_eur"] == events["derived_mv_asof_eur"]
    )
    return add_event_chains(events)


def filter_events(
    events: pd.DataFrame, clubs: list[str] | None, seasons: list[str] | None,
) -> pd.DataFrame:
    if clubs:
        wanted = {club.strip().casefold() for club in clubs}
        keep = events["raw_from_club_name"].str.casefold().isin(wanted) | events[
            "raw_to_club_name"
        ].str.casefold().isin(wanted)
        events = events[keep]
    if seasons:
        normalized = {
            f"{season[2:4]}/{season[-2:]}" if len(season) == 7 and season[4] == "/" else season
            for season in seasons
        }
        events = events[events["raw_transfer_season"].isin(normalized)]
    return events.reset_index(drop=True)


def snapshot_metadata(connection, database: Path) -> dict:
    stat = database.stat()
    commit = None
    tables = inspect_schema(connection)
    if "version" in tables:
        commit = connection.execute("SELECT * FROM version").fetchdf().iloc[0].to_dict()
    return {
        "database_path": str(database),
        "database_bytes": stat.st_size,
        "database_mtime_local": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")
        .tz_convert("America/New_York").isoformat(),
        "upstream_project": "dcaribou/transfermarkt-datasets",
        "upstream_snapshot_url": "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb",
        "version_table": commit,
        "tables": {name: connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                   for name in tables},
        "max_transfer_date": str(connection.execute("SELECT max(transfer_date) FROM transfers").fetchone()[0]),
        "max_player_valuation_date": str(connection.execute("SELECT max(date) FROM player_valuations").fetchone()[0]),
        "max_game_date": str(connection.execute("SELECT max(date) FROM games").fetchone()[0]),
        "max_appearance_date": str(connection.execute("SELECT max(date) FROM appearances").fetchone()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: build the Transfermarkt backbone")
    parser.add_argument("--db", default=str(DEFAULT_DATABASE))
    parser.add_argument("--output-dir", default=str(REBUILD_DIR))
    parser.add_argument("--club", action="append")
    parser.add_argument("--season", action="append")
    parser.add_argument("--name", default="stage1_normalized_transfers")
    args = parser.parse_args()

    database = Path(args.db)
    connection = connect_database(database, read_only=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = build_backbone(connection)
    metadata = snapshot_metadata(connection, database)
    metadata["rows_in_full_backbone"] = int(len(events))

    selected = filter_events(events, args.club, args.season)
    metadata["rows_written"] = int(len(selected))
    metadata["selection"] = {"clubs": args.club, "seasons": args.season}

    selected.to_csv(output_dir / f"{args.name}.csv", index=False)
    metadata_name = ("stage1_snapshot_metadata.json" if not (args.club or args.season)
                     else f"{args.name}_metadata.json")
    (output_dir / metadata_name).write_text(json.dumps(metadata, indent=2))
    print(f"wrote {len(selected)} events to {output_dir / (args.name + '.csv')}")


if __name__ == "__main__":
    main()
