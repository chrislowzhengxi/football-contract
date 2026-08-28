from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CONTRACT_FIELDS = [
    "transfer_type", "add_ons", "sell_on_terms", "buy_back_terms",
    "purchase_option", "purchase_obligation", "obligation_trigger",
    "release_clause_or_purchase_option", "historical_contract_expiry",
    "source_evidence", "contract_summary",
]
STATUS_VALUES = {"disclosed_yes", "disclosed_no", "partially_disclosed", "undisclosed", "not_found", "conflicting_sources", "not_applicable"}


@dataclass
class ValidationReport:
    errors: list[str]
    warnings: list[str]


def validate_events(events: pd.DataFrame) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    if events["event_id"].duplicated().any():
        errors.append("duplicate transfer/event IDs")
    if events["age_at_transfer"].dropna().map(lambda age: age < 0 or age > 60).any():
        errors.append("impossible ages")
    for column in events.columns:
        if column.endswith("_status"):
            values = set(events[column].dropna().unique())
            invalid = values - STATUS_VALUES
            if invalid:
                errors.append(f"invalid status values in {column}: {sorted(invalid)}")
    return ValidationReport(errors, warnings)


def validate_outcomes(outcomes: pd.DataFrame, appearance_ids: pd.Series | None = None) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    if outcomes.empty:
        return ValidationReport(errors, ["no outcome rows produced"])
    if (outcomes["minutes_played"] < 0).any():
        errors.append("negative minutes")
    if (outcomes["window_start"] > outcomes["window_end"]).any():
        errors.append("transfer dates after outcome windows")
    if appearance_ids is not None and appearance_ids.duplicated().any():
        errors.append("duplicate appearance counting")
    if outcomes["missing_appearance_data"].all():
        warnings.append("suspiciously missing appearance data for all selected outcomes")
    return ValidationReport(errors, warnings)


def raise_for_errors(*reports: ValidationReport) -> None:
    errors = [error for report in reports for error in report.errors]
    if errors:
        raise ValueError("; ".join(errors))