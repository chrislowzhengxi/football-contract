"""Stage 1B feasibility run: parse the raw Transfermarkt rows for the audit
players and compare them, field by field, against the Stage 1 DuckDB backbone.

    python -m src.stage1b.build

Writes to data/outputs/rebuild/:
    stage1b_transfermarkt_page_rows.csv   every parsed row for the audit players
    stage1b_comparison.csv                backbone vs raw label, per event
    stage1b_feasibility_summary.md        what the raw layer actually adds
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_DATABASE
from ..load_data import connect_database
from ..stage1.backbone import REBUILD_DIR
from .parse import parse_player
from .raw_api import collect_players

AUDIT_CSV = REBUILD_DIR / "stage1_validation_20.csv"
BACKBONE_CSV = REBUILD_DIR / "stage1_normalized_transfers.csv"

LOAN_LABELS = {"loan_transfer", "end_of_loan", "loan_fee", "end_of_loan_with_fee"}


def audit_player_ids(path: Path = AUDIT_CSV) -> set[int]:
    frame = pd.read_csv(path)
    return {
        int(re.search(r"player_id=(\d+)", reference).group(1))
        for reference in frame["raw_record_reference"]
    }


def audit_event_ids(path: Path = AUDIT_CSV) -> set[str]:
    return set(pd.read_csv(path)["event_id"])


def build_page_rows(player_ids: set[int], names: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    responses, origin = collect_players(player_ids)
    rows: list[dict] = []
    for player_id, response in responses.items():
        for row in parse_player(player_id, names.get(player_id, ""), response):
            row["source_season_file"] = origin[player_id]
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["player_name", "transfer_date"]).reset_index(drop=True)
    coverage = {
        "players_requested": len(player_ids),
        "players_resolved": len(responses),
        "players_missing": sorted(player_ids - set(responses)),
        "rows_parsed": len(frame),
        "season_files_used": {season: sum(1 for value in origin.values() if value == season)
                              for season in sorted(set(origin.values()))},
    }
    return frame, coverage


def compare(backbone: pd.DataFrame, page_rows: pd.DataFrame, audit_ids: set[str]) -> pd.DataFrame:
    """Left join the backbone events for these players onto the raw rows.

    Join key is (player_id, date, from_club_id, to_club_id) - the same tuple the
    upstream dbt model deduplicates on, so a miss is a real discrepancy rather
    than a key mismatch.
    """
    left = backbone.copy()
    left["_date"] = pd.to_datetime(left["raw_transfer_date"]).dt.strftime("%Y-%m-%d")
    right = page_rows.copy()
    right["_date"] = pd.to_datetime(right["transfer_date"]).dt.strftime("%Y-%m-%d")
    keys = ["raw_player_id", "_date", "raw_from_club_id", "raw_to_club_id"]
    right_keyed = right.rename(columns={
        "player_id": "raw_player_id", "from_club_id": "raw_from_club_id",
        "to_club_id": "raw_to_club_id",
    })
    merged = left.merge(
        right_keyed, on=keys, how="left", suffixes=("", "_web"), indicator=True,
    )

    def matched(condition, present):
        return [None if not ok else bool(value) for ok, value in zip(present, condition)]

    present = merged["_merge"] == "both"
    web_mv = merged["market_value_eur"]
    web_fee = merged["fee_numeric_eur"]
    db_fee = merged["raw_transfer_fee"]
    label = merged["transfer_type_raw"]

    out = pd.DataFrame({
        "player": merged["raw_player_name"],
        "event_date": merged["_date"],
        "from_club": merged["raw_from_club_name"],
        "to_club": merged["raw_to_club_name"],
        "duckdb_fee": db_fee,
        "duckdb_market_value": merged["raw_market_value_in_eur"],
        "derived_movement_class": merged["derived_movement_class"],
        "website_raw_label": label,
        "website_fee_display_raw": merged["fee_display_raw"],
        "website_fee": web_fee,
        "website_fee_role": merged["fee_amount_role"],
        "website_market_value": web_mv,
        "field_match_from_club": matched(
            merged["from_club"].fillna("") == merged["raw_from_club_name"].fillna(""), present),
        "field_match_to_club": matched(
            merged["to_club"].fillna("") == merged["raw_to_club_name"].fillna(""), present),
        "field_match_date": matched(
            merged["transfer_date"].fillna("") == merged["_date"].fillna(""), present),
        "field_match_market_value": matched(
            merged["raw_market_value_in_eur"].fillna(-1) == web_mv.fillna(-1), present),
        "field_match_fee": matched(db_fee.fillna(-1) == web_fee.fillna(-1), present),
        "loan_fee_recovered": [
            bool(lab == "loan_fee" and pd.notna(fee) and fee > 0 and (pd.isna(d) or d == 0))
            for lab, fee, d in zip(label, web_fee, db_fee)
        ],
        "event_id": merged["event_id"],
        "is_audit_event": merged["event_id"].isin(audit_ids),
        "matched_in_raw_api": present,
        "website_future_transfer": merged["future_transfer"],
        "notes": "",
    })

    notes = []
    for row in out.itertuples():
        parts = []
        if not row.matched_in_raw_api:
            parts.append("no matching row in the raw API snapshot")
        else:
            if row.website_raw_label in LOAN_LABELS:
                parts.append(f"raw label '{row.website_raw_label}' is absent from the DuckDB")
            if row.loan_fee_recovered:
                parts.append(f"LOAN FEE RECOVERED: EUR {row.website_fee:,.0f} where DuckDB has "
                             f"{'NULL' if pd.isna(row.duckdb_fee) else int(row.duckdb_fee)}")
            if row.website_raw_label == "undisclosed":
                parts.append("fee was '?' (undisclosed) - DuckDB stores NULL, "
                             "indistinguishable from '-' (no fee shown)")
            if row.website_raw_label == "end_of_loan_with_fee":
                parts.append(f"fee paid ON RETURN: EUR {row.website_fee:,.0f} - likely an "
                             "exercised option/obligation")
            if row.field_match_fee is False and row.website_raw_label == "paid_transfer":
                parts.append("permanent fee disagrees between DuckDB and raw API")
            if row.field_match_market_value is False:
                parts.append("market value disagrees")
        notes.append("; ".join(parts))
    out["notes"] = notes
    return out


def summarise(comparison: pd.DataFrame, page_rows: pd.DataFrame, coverage: dict) -> str:
    matched = comparison[comparison.matched_in_raw_api]
    audit = comparison[comparison.is_audit_event]
    n = len(matched)

    def share(mask, total=None):
        total = total if total is not None else n
        count = int(mask.sum())
        return f"{count} / {total} ({count / total * 100:.0f}%)" if total else "0"

    lines = ["# Stage 1B - what the raw Transfermarkt layer adds over the DuckDB\n"]
    lines.append("Source: the upstream project's own raw acquisition files, pulled from the same "
                 "public R2 bucket as the DuckDB. **No requests were made to transfermarkt.com.**\n")
    lines.append(f"- audit players requested: {coverage['players_requested']}")
    lines.append(f"- players resolved in the raw snapshot: {coverage['players_resolved']}")
    lines.append(f"- players not found: {coverage['players_missing'] or 'none'}")
    lines.append(f"- raw transfer rows parsed for those players: {coverage['rows_parsed']}")
    lines.append(f"- season files used: {coverage['season_files_used']}")
    lines.append(f"- DuckDB backbone events for those players: {len(comparison)}")
    lines.append(f"- of those, matched to a raw row: {share(comparison.matched_in_raw_api, len(comparison))}")
    lines.append(f"- of those, the 22 Stage 1 audit events: {len(audit)}, "
                 f"matched {int(audit.matched_in_raw_api.sum())}\n")

    lines.append("## Field-by-field: does the raw layer agree with the DuckDB?\n")
    lines.append("| field | agrees | reading |")
    lines.append("| --- | --- | --- |")
    for field, label, reading in [
        ("field_match_from_club", "departing club", "DuckDB is faithful"),
        ("field_match_to_club", "receiving club", "DuckDB is faithful"),
        ("field_match_date", "transfer date", "DuckDB is faithful"),
        ("field_match_market_value", "market value", "DuckDB is faithful"),
        ("field_match_fee", "fee (numeric)", "disagreements are the upstream `else 0` collapse"),
    ]:
        agree = int((matched[field] == True).sum())  # noqa: E712
        lines.append(f"| {label} | {agree} / {n} ({agree / n * 100:.0f}%) | {reading} |")
    lines.append("")

    lines.append("## What the raw layer ADDS that the DuckDB does not contain\n")
    lines.append("| added information | events | note |")
    lines.append("| --- | --- | --- |")
    label_counts = matched.website_raw_label.value_counts()
    lines.append(f"| semantic transfer-type label (any) | {share(matched.website_raw_label.notna())} "
                 "| the DuckDB has no label column at all |")
    lines.append(f"| `loan transfer` label | {share(matched.website_raw_label == 'loan_transfer')} "
                 "| DuckDB stores fee 0 |")
    lines.append(f"| `End of loan` label | {share(matched.website_raw_label == 'end_of_loan')} "
                 "| DuckDB stores fee 0 |")
    lines.append(f"| `free transfer` label | {share(matched.website_raw_label == 'free_transfer')} "
                 "| DuckDB stores fee 0 - correct value, but indistinguishable from a loan |")
    lines.append(f"| **loan fee with an amount** | {share(matched.website_raw_label == 'loan_fee')} "
                 "| **DuckDB stores 0. Pure loss.** |")
    lines.append(f"| fee paid on return (`End of loan €x`) | "
                 f"{share(matched.website_raw_label == 'end_of_loan_with_fee')} "
                 "| DuckDB stores 0 |")
    lines.append(f"| `?` = explicitly undisclosed | {share(matched.website_raw_label == 'undisclosed')} "
                 "| DuckDB stores NULL, same as `-` |")
    lines.append(f"| ordinary permanent fee | {share(matched.website_raw_label == 'paid_transfer')} "
                 "| DuckDB already has this, and it agrees |")
    lines.append(f"| Transfermarkt `transfer_id` | {share(page_rows.transfermarkt_transfer_id.notna(), len(page_rows))} "
                 "| stable per-transfer key, dropped upstream |")
    lines.append(f"| `futureTransfer` / `upcoming` flag | {share(page_rows.future_transfer, len(page_rows))} "
                 "| marks scheduled rows explicitly; Stage 1 has to infer this from the date |")
    lines.append("")

    lines.append("## Does the raw label VALIDATE the Stage 1 loan heuristic?\n")
    lines.append("This is the first ground truth we have had for the loan rules. Rows are "
                 "Stage 1's derived class, columns are Transfermarkt's own label.\n")
    crosstab = pd.crosstab(matched.derived_movement_class, matched.website_raw_label)
    lines.append("| derived_movement_class | " + " | ".join(f"`{c}`" for c in crosstab.columns) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in crosstab.columns) + " |")
    for index, row in crosstab.iterrows():
        lines.append(f"| `{index}` | " + " | ".join(str(v) if v else "" for v in row) + " |")
    lines.append("")

    loan_truth = matched.website_raw_label.isin(LOAN_LABELS)
    loan_guess = matched.derived_movement_class.isin(["loan_out", "loan_return"])
    tp = int((loan_truth & loan_guess).sum())
    fp = int((~loan_truth & loan_guess).sum())
    fn = int((loan_truth & ~loan_guess).sum())
    lines.append(f"- Transfermarkt says loan-related: **{int(loan_truth.sum())}** events")
    lines.append(f"- Stage 1 heuristic said loan: **{int(loan_guess.sum())}** events")
    lines.append(f"- correctly identified: **{tp}**")
    lines.append(f"- **false positives** (heuristic said loan, label says otherwise): **{fp}**")
    lines.append(f"- **false negatives** (label says loan, heuristic missed it): **{fn}**")
    if int(loan_truth.sum()):
        lines.append(f"- recall {tp / int(loan_truth.sum()) * 100:.0f}%, "
                     f"precision {tp / max(int(loan_guess.sum()), 1) * 100:.0f}%")
    lines.append("")
    misses = matched[loan_truth & ~loan_guess]
    if not misses.empty:
        lines.append("Missed loans:\n")
        lines.append("| player | date | from | to | raw label | Stage 1 called it |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in misses.itertuples():
            lines.append(f"| {row.player} | {row.event_date} | {row.from_club} | {row.to_club} "
                         f"| `{row.website_raw_label}` | `{row.derived_movement_class}` |")
        lines.append("")
    wrong = matched[~loan_truth & loan_guess]
    if not wrong.empty:
        lines.append("False positives:\n")
        lines.append("| player | date | from | to | raw label | Stage 1 called it |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in wrong.itertuples():
            lines.append(f"| {row.player} | {row.event_date} | {row.from_club} | {row.to_club} "
                         f"| `{row.website_raw_label}` | `{row.derived_movement_class}` |")
        lines.append("")

    lines.append("## Loan fees recovered on the audit sample\n")
    recovered = matched[matched.loan_fee_recovered]
    if recovered.empty:
        lines.append("_none in this sample_\n")
    else:
        lines.append("| player | date | from | to | DuckDB fee | raw label | recovered loan fee |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in recovered.itertuples():
            db = "NULL" if pd.isna(row.duckdb_fee) else f"{int(row.duckdb_fee)}"
            lines.append(f"| {row.player} | {row.event_date} | {row.from_club} | {row.to_club} "
                         f"| {db} | `{row.website_raw_label}` | EUR {row.website_fee:,.0f} |")
        lines.append("")

    lines.append("## Raw label distribution across all parsed rows for these players\n")
    lines.append("| transfer_type_raw | rows |")
    lines.append("| --- | --- |")
    for value, count in page_rows.transfer_type_raw.value_counts().items():
        lines.append(f"| `{value}` | {count} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1B feasibility run")
    parser.add_argument("--output-dir", default=str(REBUILD_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    player_ids = audit_player_ids()
    audit_ids = audit_event_ids()
    connection = connect_database(DEFAULT_DATABASE, read_only=True)
    names = {
        int(row[0]): row[1] for row in connection.execute(
            "SELECT player_id, name FROM players WHERE player_id IN ("
            + ",".join(str(i) for i in sorted(player_ids)) + ")").fetchall()
    }

    page_rows, coverage = build_page_rows(player_ids, names)
    page_rows.to_csv(output_dir / "stage1b_transfermarkt_page_rows.csv", index=False)

    backbone = pd.read_csv(BACKBONE_CSV, low_memory=False)
    backbone = backbone[backbone.raw_player_id.isin(player_ids)].reset_index(drop=True)
    comparison = compare(backbone, page_rows, audit_ids)
    comparison.to_csv(output_dir / "stage1b_comparison.csv", index=False)

    (output_dir / "stage1b_feasibility_summary.md").write_text(
        summarise(comparison, page_rows, coverage))
    (output_dir / "stage1b_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(f"parsed {len(page_rows)} raw rows for {coverage['players_resolved']} players; "
          f"compared {len(comparison)} backbone events")


if __name__ == "__main__":
    main()
