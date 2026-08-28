from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from .compute_minutes import compute_minutes
from .config import DEFAULT_DATABASE, DEFAULT_OUTPUT_DIR, SOURCE_URL
from .enrich_structured import enrich_events
from .load_data import connect_database, download_database, inspect_schema, require_columns
from .schemas import raise_for_errors, validate_events, validate_outcomes


def select_transfers(connection, clubs: list[str] | None = None, seasons: list[str] | None = None, competition: str | None = None, limit: int | None = None) -> pd.DataFrame:
    schema = inspect_schema(connection)
    require_columns(schema, "transfers", ["player_id", "transfer_date", "from_club_id", "to_club_id"])
    conditions, params = ["1=1"], []
    if clubs:
        variants = sorted({variant for club in clubs for variant in (club, re.sub(r"^(FC|AC|SC|RC)\\s+", "", club, flags=re.IGNORECASE))})
        conditions.append("(lower(from_club_name) IN (SELECT lower(UNNEST(?))) OR lower(to_club_name) IN (SELECT lower(UNNEST(?))))")
        params.extend([variants, variants])
    if seasons:
        normalized = [f"{season[2:4]}/{season[-2:]}" if len(season) == 7 and season[4] == "/" else season for season in seasons]
        conditions.append("transfer_season IN (SELECT UNNEST(?))")
        params.append(normalized)
    query = "SELECT * FROM transfers WHERE " + " AND ".join(conditions) + " ORDER BY transfer_date, player_id"
    if limit:
        query += f" LIMIT {int(limit)}"
    transfers = connection.execute(query, params).fetchdf()
    if competition and "competition_id" in schema["transfers"]:
        transfers = transfers[transfers.competition_id == competition]
    return transfers


def run(args: argparse.Namespace) -> None:
    database = Path(args.db)
    if not database.exists():
        if args.no_download:
            raise FileNotFoundError(database)
        download_database(database, args.source_url)
    connection = connect_database(database)
    schema = inspect_schema(connection)
    print(json.dumps(schema, indent=2))
    transfers = select_transfers(connection, args.club, args.season, args.competition, args.limit)
    events = enrich_events(connection, transfers)
    outcomes, appearance_ids = compute_minutes(connection, events)
    event_report, outcome_report = validate_events(events), validate_outcomes(outcomes, appearance_ids)
    raise_for_errors(event_report, outcome_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "structured_transfers.csv", index=False)
    outcomes.to_csv(output_dir / "playing_time_outcomes.csv", index=False)
    output_db = output_dir / "football_contracts.duckdb"
    out = duckdb.connect(str(output_db))
    out.register("events_df", events)
    out.register("outcomes_df", outcomes)
    out.execute("CREATE OR REPLACE TABLE structured_transfers AS SELECT * FROM events_df")
    out.execute("CREATE OR REPLACE TABLE playing_time_outcomes AS SELECT * FROM outcomes_df")
    out.close()
    report = {"errors": event_report.errors + outcome_report.errors, "warnings": event_report.warnings + outcome_report.warnings}
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"Selected {len(events)} transfer events; wrote outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and enrich Transfermarkt transfer events")
    parser.add_argument("--db", default=str(DEFAULT_DATABASE))
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--club", action="append", help="Club name; repeat for multiple clubs")
    parser.add_argument("--season", action="append", help="Transfer season such as 2024/25")
    parser.add_argument("--competition", help="Competition ID when present in transfers")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-download", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()