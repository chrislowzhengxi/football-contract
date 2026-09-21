"""Stage 1C: build the canonical, frozen Transfermarkt backbone.

    python -m src.stage1c.build

Writes to data/outputs/rebuild/:
    stage1c_canonical_transfers.csv   the authoritative Stage 1 output
    stage1c_validation_sample.csv     the 20 hand-audited players, readable
    stage1c_research_population.md    counts, policy, and every edge case
    stage1c_regression_checks.md      the named checks, pass/fail
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ..stage1.backbone import REBUILD_DIR
from .enrichment import load_enrichment
from .policy import (
    PLACEHOLDER_CLUBS,
    is_youth_or_reserve,
    normalize_club_name,
    research_decision,
    split_fees,
)
from .policy import classify_type

BACKBONE_CSV = REBUILD_DIR / "stage1_normalized_transfers.csv"
CANONICAL_CSV = REBUILD_DIR / "stage1c_canonical_transfers.csv"
SAMPLE_CSV = REBUILD_DIR / "stage1c_validation_sample.csv"

AUDIT_CSV = REBUILD_DIR / "stage1_validation_20.csv"


def validation_player_ids(path: Path = AUDIT_CSV) -> list[int]:
    """The player_ids audited in Stage 1, read from the audit sample itself.

    Keyed on player_id, never on name. Transfermarkt has 267 names shared by
    more than one player: six different footballers are called "Vitinho", and
    there are two Idrissa Gueyes and two Ladislav Krejcis. Selecting the sample
    by name silently pulled all of them in and interleaved their careers.
    """
    frame = pd.read_csv(path)
    return sorted({
        int(re.search(r"player_id=(\d+)", reference).group(1))
        for reference in frame["raw_record_reference"]
    })

CANONICAL_COLUMNS = [
    # identity
    "event_id", "transfermarkt_transfer_id", "event_chain_id",
    # player
    "player_id", "player_name", "date_of_birth", "age_at_transfer",
    "position", "country_of_citizenship",
    # movement
    "transfer_date", "transfer_season",
    "from_club_id", "from_club_name", "to_club_id", "to_club_name",
    # type
    "transfer_type_raw", "transfer_type_normalized", "transfer_type_source",
    "stage1_heuristic_class", "heuristic_agrees_with_label",
    # fees
    "fee_display_raw", "fee_numeric_eur", "fee_role", "fee_disclosure_status",
    "permanent_transfer_fee_eur", "loan_fee_eur", "fee_on_return_eur",
    "duckdb_transfer_fee", "fee_matches_duckdb",
    # market value
    "market_value_eur", "market_value_display",
    # flags
    "is_loan_related", "return_leg_has_fee", "transfermarkt_future_transfer",
    "involves_placeholder_club", "is_youth_or_reserve_side", "is_internal_move",
    "is_senior_move", "has_monetary_amount", "enrichment_matched",
    # research
    "is_research_target", "research_exclusion_reason",
]

LOAN_LABELS = {"loan_transfer", "loan_fee", "end_of_loan", "end_of_loan_with_fee"}


def build_canonical(backbone: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    left = backbone.copy()
    left["join_date"] = pd.to_datetime(left["raw_transfer_date"]).dt.strftime("%Y-%m-%d")
    merged = left.merge(
        enrichment, on=["raw_player_id", "join_date", "raw_from_club_id", "raw_to_club_id"],
        how="left", validate="one_to_one", indicator=True,
    )

    out = pd.DataFrame(index=merged.index)
    out["event_id"] = merged["event_id"]
    out["transfermarkt_transfer_id"] = merged["transfermarkt_transfer_id"]
    out["event_chain_id"] = merged["derived_event_chain_id"]
    out["player_id"] = merged["raw_player_id"]
    out["player_name"] = merged["raw_player_name"]
    out["date_of_birth"] = merged["joined_date_of_birth"]
    out["age_at_transfer"] = merged["derived_age_at_transfer"]
    out["position"] = merged["joined_position"]
    out["country_of_citizenship"] = merged["joined_country_of_citizenship"]
    out["transfer_date"] = merged["join_date"]
    out["transfer_season"] = merged["raw_transfer_season"]
    out["from_club_id"] = merged["raw_from_club_id"]
    out["from_club_name"] = merged["raw_from_club_name"]
    out["to_club_id"] = merged["raw_to_club_id"]
    out["to_club_name"] = merged["raw_to_club_name"]

    out["enrichment_matched"] = merged["_merge"] == "both"
    out["transfer_type_raw"] = merged["transfermarkt_raw_label"]
    out["fee_display_raw"] = merged["transfermarkt_raw_fee_text"]
    out["fee_numeric_eur"] = merged["transfermarkt_fee_numeric_eur"]
    out["market_value_display"] = merged["transfermarkt_market_value_display"]
    out["transfermarkt_future_transfer"] = merged["transfermarkt_future_transfer"].fillna(False).astype(bool)
    out["stage1_heuristic_class"] = merged["derived_movement_class"]

    # --- club-shape flags (needed before typing, because an internal move can
    # refine an uninformative label) ---
    out["involves_placeholder_club"] = (
        merged["raw_from_club_name"].isin(PLACEHOLDER_CLUBS)
        | merged["raw_to_club_name"].isin(PLACEHOLDER_CLUBS)
    )
    out["is_youth_or_reserve_side"] = (
        merged["raw_from_club_name"].map(is_youth_or_reserve)
        | merged["raw_to_club_name"].map(is_youth_or_reserve)
    )
    from_base = merged["raw_from_club_name"].map(normalize_club_name)
    to_base = merged["raw_to_club_name"].map(normalize_club_name)
    out["is_internal_move"] = (
        (from_base == to_base) & (from_base != "")
        & (merged["raw_from_club_id"] != merged["raw_to_club_id"])
        & ~out["involves_placeholder_club"]
    )
    out["is_senior_move"] = ~(
        out["involves_placeholder_club"] | out["is_youth_or_reserve_side"] | out["is_internal_move"]
    )

    # --- type ---
    typed = [
        classify_type(label, heuristic, internal)
        for label, heuristic, internal in zip(
            out["transfer_type_raw"], out["stage1_heuristic_class"], out["is_internal_move"])
    ]
    out["transfer_type_normalized"] = [t for t, _ in typed]
    out["transfer_type_source"] = [s for _, s in typed]

    # --- fees ---
    fees = pd.DataFrame([
        split_fees(label, amount)
        for label, amount in zip(out["transfer_type_raw"], out["fee_numeric_eur"])
    ], index=out.index)
    for column in ["permanent_transfer_fee_eur", "loan_fee_eur", "fee_on_return_eur",
                   "fee_disclosure_status"]:
        out[column] = fees[column]
    out["fee_role"] = [
        "permanent_fee" if label == "paid_transfer"
        else "loan_fee" if label == "loan_fee"
        else "fee_on_return" if label == "end_of_loan_with_fee"
        else "none_free" if label == "free_transfer"
        else None
        for label in out["transfer_type_raw"]
    ]
    out["duckdb_transfer_fee"] = merged["raw_transfer_fee"]
    out["market_value_eur"] = merged["raw_market_value_in_eur"]

    # The DuckDB number is only comparable where Transfermarkt published a
    # permanent fee or asserted "free"; everywhere else it is the `else 0`
    # artefact and a mismatch is expected, not a defect.
    comparable = out["transfer_type_raw"].isin(["paid_transfer", "free_transfer"])
    out["fee_matches_duckdb"] = [
        bool(abs((p if p is not None and not pd.isna(p) else -1)
                 - (d if not pd.isna(d) else -1)) < 0.5) if ok else None
        for ok, p, d in zip(comparable, out["permanent_transfer_fee_eur"], out["duckdb_transfer_fee"])
    ]

    out["is_loan_related"] = out["transfer_type_raw"].isin(LOAN_LABELS)
    out["return_leg_has_fee"] = (
        (out["transfer_type_normalized"] == "loan_return")
        & out["fee_on_return_eur"].notna() & (out["fee_on_return_eur"].fillna(0) > 0)
    )
    out["has_monetary_amount"] = (
        out[["permanent_transfer_fee_eur", "loan_fee_eur", "fee_on_return_eur"]]
        .fillna(0).gt(0).any(axis=1)
    )
    # Placeholder-club rows are excluded from the comparison: the heuristic
    # classifies them by the placeholder marker and the label describes the fee
    # cell, so the two are not measuring the same thing and counting them as
    # agreement would flatter the heuristic.
    not_comparable = {"arrival_from_no_club", "exit_to_no_club", "retirement",
                      "same_club_record_artifact"}
    out["heuristic_agrees_with_label"] = [
        None if (not matched or heuristic in not_comparable) else bool(
            (heuristic == "loan_out" and label in ("loan_transfer", "loan_fee"))
            or (heuristic == "loan_return" and label in ("end_of_loan", "end_of_loan_with_fee"))
            or (heuristic == "permanent_with_fee" and label == "paid_transfer")
            or (heuristic == "zero_fee_move" and label == "free_transfer")
            or (heuristic == "no_fee_recorded" and label in ("no_fee_shown", "undisclosed"))
        )
        for matched, heuristic, label in zip(
            out["enrichment_matched"], out["stage1_heuristic_class"], out["transfer_type_raw"])
    ]

    # --- research population ---
    decisions = [research_decision(row) for _, row in out.iterrows()]
    out["is_research_target"] = [t for t, _ in decisions]
    out["research_exclusion_reason"] = [r for _, r in decisions]

    return out[CANONICAL_COLUMNS]


def population_report(canonical: pd.DataFrame, meta: dict) -> str:
    n = len(canonical)
    lines = ["# Stage 1C - canonical backbone and research population\n"]
    lines.append(f"Canonical events: **{n:,}**  ")
    lines.append(f"Raw Transfermarkt enrichment matched: "
                 f"**{int(canonical.enrichment_matched.sum()):,} "
                 f"({canonical.enrichment_matched.mean() * 100:.2f}%)**  ")
    lines.append(f"Requests to transfermarkt.com: **{meta.get('transfermarkt_com_requests', 0)}**\n")

    lines.append("## Transfer type (normalized)\n")
    lines.append("| transfer_type_normalized | events | share | source is a raw label |")
    lines.append("| --- | --- | --- | --- |")
    for kind, count in canonical.transfer_type_normalized.value_counts().items():
        subset = canonical[canonical.transfer_type_normalized == kind]
        from_label = (subset.transfer_type_source == "transfermarkt_raw_label").sum()
        lines.append(f"| `{kind}` | {count:,} | {count / n * 100:.1f}% "
                     f"| {from_label:,} ({from_label / count * 100:.0f}%) |")
    lines.append("")

    lines.append("## Where each type came from\n")
    lines.append("| transfer_type_source | events | share |")
    lines.append("| --- | --- | --- |")
    for source, count in canonical.transfer_type_source.value_counts().items():
        lines.append(f"| `{source}` | {count:,} | {count / n * 100:.1f}% |")
    lines.append("")

    lines.append("## The categories you asked for, counted\n")
    loans = canonical.transfer_type_normalized == "loan"
    returns = canonical.transfer_type_normalized == "loan_return"
    senior = canonical.is_senior_move
    rows = [
        ("permanent transfer (fee published)", int((canonical.transfer_type_raw == "paid_transfer").sum())),
        ("free transfer", int((canonical.transfer_type_normalized == "free_transfer").sum())),
        ("loan (total)", int(loans.sum())),
        ("loan WITH a loan fee", int((loans & canonical.loan_fee_eur.notna()).sum())),
        ("loan with no fee published", int((loans & canonical.loan_fee_eur.isna()).sum())),
        ("loan return (total)", int(returns.sum())),
        ("loan return WITH a fee", int(canonical.return_leg_has_fee.sum())),
        ("undisclosed fee ('?'), senior clubs", int(((canonical.transfer_type_normalized == "undisclosed_transfer") & senior).sum())),
        ("undisclosed fee ('?'), non-senior", int(((canonical.transfer_type_normalized == "undisclosed_transfer") & ~senior).sum())),
        ("no fee shown ('-'), senior clubs", int(((canonical.transfer_type_normalized == "no_fee_shown") & senior).sum())),
        ("no fee shown ('-'), non-senior", int(((canonical.transfer_type_normalized == "no_fee_shown") & ~senior).sum())),
        ("youth / internal registration change", int((canonical.transfer_type_normalized == "youth_or_internal").sum())),
        ("other (draft, retirement, placeholder)", int((canonical.transfer_type_normalized == "other").sum())),
        ("unknown", int((canonical.transfer_type_normalized == "unknown").sum())),
    ]
    lines.append("| category | events |")
    lines.append("| --- | --- |")
    for label, count in rows:
        lines.append(f"| {label} | {count:,} |")
    lines.append(f"| **total** | **{n:,}** |")
    lines.append("")

    lines.append("## Money recovered that the DuckDB does not hold\n")
    lines.append("| field | events with a value | total EUR |")
    lines.append("| --- | --- | --- |")
    for column, label in [
        ("permanent_transfer_fee_eur", "permanent_transfer_fee_eur (> 0)"),
        ("loan_fee_eur", "**loan_fee_eur**"),
        ("fee_on_return_eur", "**fee_on_return_eur**"),
    ]:
        positive = canonical[canonical[column].fillna(0) > 0]
        lines.append(f"| {label} | {len(positive):,} | {positive[column].sum():,.0f} |")
    lines.append("")

    lines.append("## Fee disclosure status\n")
    lines.append("| fee_disclosure_status | events | meaning |")
    lines.append("| --- | --- | --- |")
    meanings = {
        "disclosed_amount": "a number was published",
        "disclosed_free": "Transfermarkt states 'free transfer' - an asserted zero",
        "undisclosed": "Transfermarkt shows '?' - a deal happened, terms withheld",
        "no_fee_shown": "Transfermarkt shows '-' - no fee cell at all",
        "not_applicable": "end of a loan, or a draft; no fee concept",
        "unknown": "no raw row matched",
    }
    for status, count in canonical.fee_disclosure_status.value_counts().items():
        lines.append(f"| `{status}` | {count:,} | {meanings.get(status, '')} |")
    lines.append("")

    lines.append("## Research population\n")
    targets = int(canonical.is_research_target.sum())
    lines.append(f"- `is_research_target = true`: **{targets:,}** ({targets / n * 100:.1f}%)")
    lines.append(f"- `is_research_target = false`: **{n - targets:,}** ({(n - targets) / n * 100:.1f}%)\n")
    lines.append("### Targets by type\n")
    lines.append("| transfer_type_normalized | targets |")
    lines.append("| --- | --- |")
    for kind, count in canonical[canonical.is_research_target].transfer_type_normalized.value_counts().items():
        lines.append(f"| `{kind}` | {count:,} |")
    lines.append("")
    lines.append("### Why events were excluded\n")
    lines.append("| research_exclusion_reason | events |")
    lines.append("| --- | --- |")
    excluded = canonical[~canonical.is_research_target]
    for reason, count in excluded.research_exclusion_reason.value_counts().items():
        lines.append(f"| `{reason}` | {count:,} |")
    lines.append("")

    lines.append("## Does the Stage 1 heuristic still agree with the raw label?\n")
    checked = canonical[canonical.enrichment_matched & canonical.heuristic_agrees_with_label.notna()]
    agree = int(checked.heuristic_agrees_with_label.sum())
    lines.append(f"- comparable events: {len(checked):,}")
    lines.append(f"- heuristic agrees with the raw label: **{agree:,} "
                 f"({agree / len(checked) * 100:.1f}%)**")
    lines.append(f"- heuristic disagrees: **{len(checked) - agree:,}** - in every one of these the "
                 "raw label wins, and the heuristic value is retained in "
                 "`stage1_heuristic_class` for inspection\n")
    disagree = checked[~checked.heuristic_agrees_with_label.astype(bool)]
    if not disagree.empty:
        lines.append("| stage1_heuristic_class | transfer_type_raw | events |")
        lines.append("| --- | --- | --- |")
        for (heuristic, label), count in disagree.groupby(
                ["stage1_heuristic_class", "transfer_type_raw"]).size().sort_values(
                ascending=False).items():
            lines.append(f"| `{heuristic}` | `{label}` | {count:,} |")
        lines.append("")

    lines.append("## Remaining ambiguity, stated plainly\n")
    lines.append("1. **`no_fee_shown` ('-') between two senior clubs** - "
                 f"{int(((canonical.transfer_type_normalized == 'no_fee_shown') & senior).sum()):,} events. "
                 "Transfermarkt shows no fee cell. Could be a free move, an unrecorded deal, or an "
                 "administrative row. Excluded by default; flip `research_exclusion_reason == "
                 "'no_fee_shown_no_evidence_of_a_negotiated_deal'` to include them.")
    lines.append("2. **`loan_transfer` with no fee** - "
                 f"{int((loans & canonical.loan_fee_eur.isna()).sum()):,} events. We cannot tell a "
                 "genuinely free loan from a loan whose fee was not published, so "
                 "`fee_disclosure_status` is `no_fee_shown`, not a zero.")
    lines.append("3. **`fee_on_return_eur`** - "
                 f"{int(canonical.return_leg_has_fee.sum()):,} events where money is attached to the "
                 "END of a loan. Most plausibly an exercised option or obligation, but Transfermarkt "
                 "does not say. Kept, flagged, and routed to research.")
    lines.append("4. **`free_transfer` asserts 0** - this is the one place Stage 1C writes a zero. "
                 "Justification: 'free transfer' is a positive statement by the source, unlike '-' "
                 "or '?'. Recorded as `fee_disclosure_status = disclosed_free` so it stays "
                 "distinguishable from a parsed amount.")
    lines.append("5. **Youth/reserve naming is a club-name pattern**, not a Transfermarkt field. A "
                 "cross-club sale from an academy squad (Sancho, Man City U18 -> Dortmund, EUR 20.6m) "
                 "is still a research target because a fee was published; only same-organisation "
                 "moves are excluded as internal.\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1C: build the canonical backbone")
    parser.add_argument("--backbone", default=str(BACKBONE_CSV))
    parser.add_argument("--output-dir", default=str(REBUILD_DIR))
    parser.add_argument("--rebuild-enrichment", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    backbone = pd.read_csv(args.backbone, low_memory=False)
    enrichment, meta = load_enrichment(rebuild=args.rebuild_enrichment)
    canonical = build_canonical(backbone, enrichment)
    canonical.to_csv(output_dir / "stage1c_canonical_transfers.csv", index=False)

    # A focused column set, ordered so a row can be read straight across
    # against a Transfermarkt transfer-history page. The full 43-column record
    # for these events is in the canonical file.
    sample_columns = [
        "player_id", "player_name", "date_of_birth",
        "age_at_transfer", "transfer_season", "transfer_date",
        "from_club_name", "to_club_name",
        "fee_display_raw", "transfer_type_raw", "transfer_type_normalized",
        "permanent_transfer_fee_eur", "loan_fee_eur", "fee_on_return_eur",
        "fee_disclosure_status", "market_value_display", "market_value_eur",
        "duckdb_transfer_fee", "stage1_heuristic_class", "heuristic_agrees_with_label",
        "is_senior_move", "is_internal_move", "transfermarkt_future_transfer",
        "is_research_target", "research_exclusion_reason",
        "transfermarkt_transfer_id", "event_id",
    ]
    sample = canonical[canonical.player_id.isin(validation_player_ids())].sort_values(
        ["player_id", "transfer_date"])[sample_columns]
    sample.to_csv(output_dir / "stage1c_validation_sample.csv", index=False)

    (output_dir / "stage1c_research_population.md").write_text(
        population_report(canonical, meta))
    print(f"canonical events: {len(canonical):,}; "
          f"enrichment matched {canonical.enrichment_matched.mean() * 100:.2f}%; "
          f"research targets {int(canonical.is_research_target.sum()):,}; "
          f"validation sample rows {len(sample)}")


if __name__ == "__main__":
    main()
