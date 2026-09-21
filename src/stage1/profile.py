"""Profile a Stage 1 backbone file and write a human-readable markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backbone import REBUILD_DIR

PSEUDO = {"Without Club", "Retired", "Career break", "Ban", "Unknown"}


def _pct(part: int, whole: int) -> str:
    return f"{part:,} ({part / whole * 100:.1f}%)" if whole else "0"


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_none_\n"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def profile(events: pd.DataFrame, snapshot: dict, source: str) -> str:
    n = len(events)
    dates = pd.to_datetime(events["raw_transfer_date"])
    snapshot_date = pd.Timestamp(snapshot["database_mtime_local"][:10])
    lines: list[str] = []
    add = lines.append

    add("# Stage 1 - structured transfer profile\n")
    add(f"Source file: `{source}`  ")
    add(f"Raw snapshot: `{snapshot['database_path']}` (downloaded {snapshot['database_mtime_local']})  ")
    add(f"Upstream: {snapshot['upstream_project']}, version table `{snapshot['version_table']}`\n")

    add("## 1. Size and coverage\n")
    add(f"- rows (transfer events): **{n:,}**")
    add(f"- unique players: **{events['raw_player_id'].nunique():,}**")
    add(f"- unique club ids appearing on either side: **{pd.concat([events.raw_from_club_id, events.raw_to_club_id]).nunique():,}**")
    add(f"- unique club names appearing on either side: **{pd.concat([events.raw_from_club_name, events.raw_to_club_name]).nunique():,}**")
    season_order = events.groupby("raw_transfer_season")["raw_transfer_date"].min().sort_values()
    add(f"- transfer_season values: **{events['raw_transfer_season'].nunique()}** "
        f"(earliest `{season_order.index[0]}`, latest `{season_order.index[-1]}`; the labels are "
        "two-digit strings, so they do not sort chronologically as text)")
    add(f"- transfer_date range: **{dates.min().date()} .. {dates.max().date()}**\n")
    add("Rows per season (top 15):\n")
    add(_table(events["raw_transfer_season"].value_counts().head(15).rename_axis("season")
              .reset_index(name="rows")))

    add("## 2. Missingness\n")
    add(f"- `raw_transfer_fee` NULL: {_pct(int(events.raw_transfer_fee.isna().sum()), n)}")
    add(f"- `raw_transfer_fee` exactly 0: {_pct(int((events.raw_transfer_fee == 0).sum()), n)}")
    add(f"- `raw_transfer_fee` > 0: {_pct(int((events.raw_transfer_fee > 0).sum()), n)}")
    add(f"- `raw_market_value_in_eur` NULL: {_pct(int(events.raw_market_value_in_eur.isna().sum()), n)}")
    add(f"- `derived_mv_asof_eur` NULL (no earlier valuation exists): "
        f"{_pct(int(events.derived_mv_asof_eur.isna().sum()), n)}")
    add(f"- player row not found in `players`: {_pct(int((~events.joined_player_row_found).sum()), n)}")
    add(f"- `joined_date_of_birth` NULL (so age is NULL): "
        f"{_pct(int(events.joined_date_of_birth.isna().sum()), n)}")
    add("- **`transfer_type` missingness: not applicable.** The raw `transfers` table has no "
        "transfer-type column at all, so Stage 1 emits `derived_movement_class` instead. "
        "Its missingness is 0% by construction; see the basis split below.\n")

    add("## 3. Derived movement classification\n")
    counts = events["derived_movement_class"].value_counts().rename_axis("derived_movement_class").reset_index(name="rows")
    counts["share"] = (counts["rows"] / n * 100).round(1).astype(str) + "%"
    add(_table(counts))
    add("Basis (is the class a raw fact or our heuristic?):\n")
    add(_table(events["derived_movement_basis"].value_counts().rename_axis("basis").reset_index(name="rows")))

    add("## 4. Duplicate and near-duplicate events\n")
    dup_ids = int(events["event_id"].duplicated().sum())
    add(f"- duplicate `event_id`: **{dup_ids}**")
    key = ["raw_player_id", "raw_transfer_date", "raw_from_club_id", "raw_to_club_id"]
    add(f"- duplicate (player, date, from, to) tuples: **{int(events.duplicated(key).sum())}**")
    same_day = events.duplicated(["raw_player_id", "raw_transfer_date"], keep=False)
    add(f"- rows sharing a (player, date) pair with another row: **{int(same_day.sum())}** "
        f"in {int(events[same_day].groupby(['raw_player_id','raw_transfer_date']).ngroups)} groups")
    add(f"- rows where same-day ordering could not be resolved by club chaining: "
        f"**{int(events.derived_same_date_order_ambiguous.sum())}**")
    add(f"- rows where `from_club_id == to_club_id`: **{int((events.raw_from_club_id == events.raw_to_club_id).sum())}**\n")
    if same_day.any():
        add("Examples of same-day pairs:\n")
        add(_table(events[same_day].sort_values(["raw_player_id", "raw_transfer_date"])
                   .head(8)[["raw_player_name", "raw_transfer_date", "raw_from_club_name",
                             "raw_to_club_name", "raw_transfer_fee", "derived_movement_class"]]))

    add("## 5. Reverse-direction and repeated movement\n")
    reverse = int((events.derived_movement_class == "loan_return").sum())
    add(f"- rows that reverse the immediately preceding move and were paired as a loan: **{reverse:,}**")
    add(f"- rows flagged `derived_follows_loan_return` (a move back to the loan club within 31 days "
        f"of the return leg - the classic 'loan turned permanent'): "
        f"**{int(events.derived_follows_loan_return.sum()):,}**")
    repeats = events.groupby("raw_player_id").size()
    add(f"- players with more than one recorded movement: **{int((repeats > 1).sum()):,}** "
        f"of {repeats.size:,}")
    add(f"- maximum movements for a single player: **{int(repeats.max())}**\n")

    add("## 6. Date sanity\n")
    add(f"- transfer_date after the snapshot download date ({snapshot_date.date()}): "
        f"**{int((dates > snapshot_date).sum()):,}** - these are scheduled future rows, "
        "overwhelmingly pre-registered loan end dates")
    add(f"- transfer_date before 1990: **{int((dates < pd.Timestamp('1990-01-01')).sum())}**")
    add(f"- transfer_date NULL: **{int(dates.isna().sum())}**")
    dob = pd.to_datetime(events["joined_date_of_birth"])
    add(f"- transfer_date before date_of_birth: **{int((dates < dob).sum())}**")
    add(f"- transfer_date falling on a season boundary day: "
        f"{_pct(int(events.derived_date_on_season_boundary.sum()), n)}\n")
    add("Most common calendar days (bookkeeping dates dominate):\n")
    add(_table(dates.dt.strftime("%m-%d").value_counts().head(8).rename_axis("mm-dd").reset_index(name="rows")))

    add("## 7. Age anomalies\n")
    age = events["derived_age_at_transfer"]
    add(f"- age NULL: {_pct(int(age.isna().sum()), n)}")
    add(f"- age < 14: **{int((age < 14).sum()):,}**")
    add(f"- age < 10: **{int((age < 10).sum()):,}**")
    add(f"- age > 42: **{int((age > 42).sum()):,}**")
    add(f"- age range: **{age.min()} .. {age.max()}**\n")

    add("## 8. Fee anomalies\n")
    fee = events["raw_transfer_fee"]
    add(f"- negative fee: **{int((fee < 0).sum())}**")
    add(f"- fee > EUR 150m: **{int((fee > 150_000_000).sum())}**")
    add(f"- fee > 0 on a row classified `loan_out` or `loan_return`: "
        f"**{int(((fee > 0) & events.derived_movement_class.isin(['loan_out','loan_return'])).sum())}** "
        "(should be 0 by construction)")
    add(f"- fee > 0 where the receiving club is a pseudo club: "
        f"**{int(((fee > 0) & events.raw_to_club_name.isin(PSEUDO)).sum())}**\n")
    add("Largest fees:\n")
    add(_table(events.nlargest(5, "raw_transfer_fee")[
        ["raw_player_name", "raw_transfer_date", "raw_from_club_name", "raw_to_club_name",
         "raw_transfer_fee"]]))

    add("## 9. Market value join\n")
    both = events.dropna(subset=["raw_market_value_in_eur", "derived_mv_asof_eur"])
    add(f"- `raw_market_value_in_eur` equals the as-of valuation: "
        f"**{(both.raw_market_value_in_eur == both.derived_mv_asof_eur).mean() * 100:.2f}%** "
        f"of {len(both):,} comparable rows. The raw column already IS the as-of value.")
    nearest_both = events.dropna(subset=["raw_market_value_in_eur", "derived_mv_nearest_eur"])
    add(f"- `raw_market_value_in_eur` equals the *nearest* valuation (what the legacy parser used): "
        f"**{(nearest_both.raw_market_value_in_eur == nearest_both.derived_mv_nearest_eur).mean() * 100:.2f}%**")
    forward = pd.to_datetime(events.derived_mv_nearest_date) > dates
    add(f"- rows whose nearest valuation is dated AFTER the transfer (look-ahead): "
        f"{_pct(int(forward.sum()), n)}")
    add(f"- rows with no market value in `transfers` but a nearest valuation the legacy parser "
        f"would have filled in: **{int((events.raw_market_value_in_eur.isna() & events.derived_mv_nearest_eur.notna()).sum()):,}**")
    add(f"- median as-of lag: **{events.derived_mv_asof_lag_days.median():.0f} days**; "
        f"median nearest absolute lag: **{events.derived_mv_nearest_abs_lag_days.median():.0f} days**\n")

    add("## 10. Pseudo clubs\n")
    add("Transfermarkt records non-club states as placeholder clubs. They are raw facts, not "
        "derivations, and they are not transfers between clubs:\n")
    rows = [{"pseudo_club": p,
             "as departing club": int((events.raw_from_club_name == p).sum()),
             "as receiving club": int((events.raw_to_club_name == p).sum())} for p in sorted(PSEUDO)]
    add(_table(pd.DataFrame(rows)))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a Stage 1 backbone CSV")
    parser.add_argument("--input", default=str(REBUILD_DIR / "stage1_normalized_transfers.csv"))
    parser.add_argument("--metadata", default=str(REBUILD_DIR / "stage1_snapshot_metadata.json"))
    parser.add_argument("--output", default=str(REBUILD_DIR / "stage1_structured_profile.md"))
    args = parser.parse_args()
    events = pd.read_csv(args.input, low_memory=False)
    snapshot = json.loads(Path(args.metadata).read_text())
    text = profile(events, snapshot, Path(args.input).name)
    Path(args.output).write_text(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
