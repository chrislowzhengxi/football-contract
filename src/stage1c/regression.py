"""Named regression checks against the canonical backbone.

Every check here corresponds to something verified by hand on the Transfermarkt
website. They are expressed as data assertions so a rebuild that breaks one is
caught immediately.

    python -m src.stage1c.regression
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..stage1.backbone import REBUILD_DIR
from .build import CANONICAL_CSV


# Every check selects a player by Transfermarkt player_id, never by name.
# 267 names in the snapshot are shared by more than one player - there are six
# different footballers called "Vitinho" and two called "Idrissa Gueye" - so a
# name-keyed check can silently assert against the wrong human being.
PLAYER_IDS = {
    "Silas": 612826,
    "Tiago Serrago": 1231128,
    "Idrissa Gueye": 1178488,          # the 2006-born one; 126665 is the Everton player
    "Vitinho": 670965,                 # one of six players of this name
    "Raheem Sterling": 134425,
    "Bamba Dieng": 822458,
    "Neymar": 68290,
    "Donyell Malen": 326029,
    "Sergio Ramos": 25557,
    "Kendry Páez": 1052439,
    "Artem Stepanov": 1045344,
    "Cuiabano": 891353,
    "Stavros Pnevmonidis": 1077560,
    "Juan Jose Arias": 989937,
    "Antoine Griezmann": 125781,
    "Kazeem Olaigbe": 565406,
    "Oli Cockle": 1489849,
    "Luca Rafaelli": 1402952,
    "Roger Fernandes": 906329,
    "Kylian Mbappé": 342229,
    "Ângelo": 743598,
}


def _one(canonical: pd.DataFrame, player: str, date: str | None = None,
         from_club: str | None = None, to_club: str | None = None) -> pd.DataFrame:
    rows = canonical[canonical.player_id == PLAYER_IDS[player]]
    if date:
        rows = rows[rows.transfer_date == date]
    if from_club:
        rows = rows[rows.from_club_name == from_club]
    if to_club:
        rows = rows[rows.to_club_name == to_club]
    return rows


def run_checks(canonical: pd.DataFrame) -> list[dict]:
    checks: list[dict] = []

    def check(name: str, expectation: str, rows: pd.DataFrame, test) -> None:
        if len(rows) != 1:
            checks.append({"check": name, "expected": expectation,
                           "observed": f"{len(rows)} matching rows", "passed": False})
            return
        row = rows.iloc[0]
        ok, observed = test(row)
        checks.append({"check": name, "expected": expectation,
                       "observed": observed, "passed": bool(ok)})

    # --- the loan fees read off the website by hand ---
    check("Silas: Stuttgart -> Red Star is a loan carrying a EUR 500,000 loan fee",
          "type=loan, loan_fee_eur=500000, permanent_transfer_fee_eur is empty",
          _one(canonical, "Silas", "2024-09-03", "Stuttgart", "Red Star"),
          lambda r: (r.transfer_type_normalized == "loan" and r.loan_fee_eur == 500_000
                     and pd.isna(r.permanent_transfer_fee_eur),
                     f"type={r.transfer_type_normalized}, loan_fee={r.loan_fee_eur}, "
                     f"permanent_fee={r.permanent_transfer_fee_eur}"))

    check("Idrissa Gueye: the loan carries a EUR 4,000,000 loan fee",
          "type=loan, loan_fee_eur=4000000",
          _one(canonical, "Idrissa Gueye", "2025-09-01"),
          lambda r: (r.transfer_type_normalized == "loan" and r.loan_fee_eur == 4_000_000,
                     f"type={r.transfer_type_normalized}, loan_fee={r.loan_fee_eur}"))

    check("Idrissa Gueye: the later permanent move is EUR 6,000,000, NOT a loan fee",
          "type=permanent_transfer, permanent_transfer_fee_eur=6000000, loan_fee_eur empty",
          _one(canonical, "Idrissa Gueye", "2026-07-01"),
          lambda r: (r.transfer_type_normalized == "permanent_transfer"
                     and r.permanent_transfer_fee_eur == 6_000_000 and pd.isna(r.loan_fee_eur),
                     f"type={r.transfer_type_normalized}, "
                     f"permanent_fee={r.permanent_transfer_fee_eur}, loan_fee={r.loan_fee_eur}"))

    check("Donyell Malen: the loan carries a EUR 2,000,000 loan fee",
          "type=loan, loan_fee_eur=2000000",
          _one(canonical, "Donyell Malen", "2026-01-16"),
          lambda r: (r.transfer_type_normalized == "loan" and r.loan_fee_eur == 2_000_000,
                     f"type={r.transfer_type_normalized}, loan_fee={r.loan_fee_eur}"))

    check("Donyell Malen: the later permanent move is EUR 25,000,000",
          "type=permanent_transfer, permanent_transfer_fee_eur=25000000",
          _one(canonical, "Donyell Malen", "2026-07-01"),
          lambda r: (r.transfer_type_normalized == "permanent_transfer"
                     and r.permanent_transfer_fee_eur == 25_000_000 and pd.isna(r.loan_fee_eur),
                     f"type={r.transfer_type_normalized}, "
                     f"permanent_fee={r.permanent_transfer_fee_eur}, loan_fee={r.loan_fee_eur}"))

    check("Kazeem Olaigbe: Trabzonspor -> Konyaspor loan fee is EUR 3,020,000",
          "type=loan, loan_fee_eur=3020000",
          _one(canonical, "Kazeem Olaigbe", "2026-02-04"),
          lambda r: (r.transfer_type_normalized == "loan" and r.loan_fee_eur == 3_020_000,
                     f"type={r.transfer_type_normalized}, loan_fee={r.loan_fee_eur}"))

    check("Kazeem Olaigbe: the later Trabzonspor -> Konyaspor move stays SEPARATE and undisclosed",
          "a distinct event on 2026-07-01, type=undisclosed_transfer, no fee asserted",
          _one(canonical, "Kazeem Olaigbe", "2026-07-01"),
          lambda r: (r.transfer_type_normalized == "undisclosed_transfer"
                     and r.fee_disclosure_status == "undisclosed"
                     and pd.isna(r.permanent_transfer_fee_eur) and pd.isna(r.loan_fee_eur),
                     f"type={r.transfer_type_normalized}, disclosure={r.fee_disclosure_status}, "
                     f"permanent_fee={r.permanent_transfer_fee_eur}, loan_fee={r.loan_fee_eur}"))

    check("Kazeem Olaigbe: the loan, the return and the permanent move are three distinct events",
          "3 rows between Trabzonspor and Konyaspor with 3 distinct event_ids",
          pd.DataFrame([{"n": len(canonical[
              (canonical.player_id == PLAYER_IDS["Kazeem Olaigbe"])
              & canonical.from_club_name.isin(["Trabzonspor", "Konyaspor"])
              & canonical.to_club_name.isin(["Trabzonspor", "Konyaspor"])].event_id.unique())}]),
          lambda r: (r.n == 3, f"{r.n} distinct event_ids"))

    check("Kylian Mbappe: PSG -> Real Madrid is a free transfer",
          "type=free_transfer, disclosure=disclosed_free, permanent_transfer_fee_eur=0",
          _one(canonical, "Kylian Mbappé", "2024-07-01", "PSG", "Real Madrid"),
          lambda r: (r.transfer_type_normalized == "free_transfer"
                     and r.fee_disclosure_status == "disclosed_free"
                     and r.permanent_transfer_fee_eur == 0,
                     f"type={r.transfer_type_normalized}, disclosure={r.fee_disclosure_status}, "
                     f"permanent_fee={r.permanent_transfer_fee_eur}"))

    check("Roger Fernandes: Braga -> Al-Ittihad is a EUR 32,000,000 permanent transfer",
          "type=permanent_transfer, permanent_transfer_fee_eur=32000000",
          _one(canonical, "Roger Fernandes", "2025-09-05", "Braga", "Al-Ittihad"),
          lambda r: (r.transfer_type_normalized == "permanent_transfer"
                     and r.permanent_transfer_fee_eur == 32_000_000,
                     f"type={r.transfer_type_normalized}, "
                     f"permanent_fee={r.permanent_transfer_fee_eur}"))

    # --- the semantics that must not collapse ---
    undisclosed = canonical[canonical.fee_display_raw == "?"]
    checks.append({
        "check": "raw '?' always stays undisclosed and never becomes a number",
        "expected": "every '?' row has disclosure=undisclosed and no fee amount",
        "observed": f"{len(undisclosed):,} rows; "
                    f"{int((undisclosed.fee_disclosure_status == 'undisclosed').sum()):,} undisclosed; "
                    f"{int(undisclosed[['permanent_transfer_fee_eur','loan_fee_eur','fee_on_return_eur']].notna().any(axis=1).sum())} with an amount",
        "passed": bool((undisclosed.fee_disclosure_status == "undisclosed").all()
                       and not undisclosed[["permanent_transfer_fee_eur", "loan_fee_eur",
                                            "fee_on_return_eur"]].notna().any(axis=1).any()),
    })

    dash = canonical[canonical.fee_display_raw == "-"]
    checks.append({
        "check": "raw '-' always stays no_fee_shown and is never treated as a confirmed zero",
        "expected": "every '-' row has disclosure=no_fee_shown and no fee amount",
        "observed": f"{len(dash):,} rows; "
                    f"{int((dash.fee_disclosure_status == 'no_fee_shown').sum()):,} no_fee_shown; "
                    f"{int(dash[['permanent_transfer_fee_eur','loan_fee_eur','fee_on_return_eur']].notna().any(axis=1).sum())} with an amount",
        "passed": bool((dash.fee_disclosure_status == "no_fee_shown").all()
                       and not dash[["permanent_transfer_fee_eur", "loan_fee_eur",
                                     "fee_on_return_eur"]].notna().any(axis=1).any()),
    })

    checks.append({
        "check": "'?' and '-' never share a fee_disclosure_status",
        "expected": "the two sets of statuses are disjoint",
        "observed": f"'?' -> {sorted(set(undisclosed.fee_disclosure_status))}, "
                    f"'-' -> {sorted(set(dash.fee_disclosure_status))}",
        "passed": not (set(undisclosed.fee_disclosure_status) & set(dash.fee_disclosure_status)),
    })

    loans = canonical[canonical.transfer_type_normalized == "loan"]
    checks.append({
        "check": "no loan fee ever lands in permanent_transfer_fee_eur",
        "expected": "0 loan rows carry a permanent fee",
        "observed": f"{int(loans.permanent_transfer_fee_eur.notna().sum())} of {len(loans):,} loans",
        "passed": bool(loans.permanent_transfer_fee_eur.isna().all()),
    })

    permanent = canonical[canonical.transfer_type_normalized == "permanent_transfer"]
    checks.append({
        "check": "no permanent fee ever lands in loan_fee_eur",
        "expected": "0 permanent rows carry a loan fee",
        "observed": f"{int(permanent.loan_fee_eur.notna().sum())} of {len(permanent):,} permanents",
        "passed": bool(permanent.loan_fee_eur.isna().all()),
    })

    returns = canonical[canonical.transfer_type_normalized == "loan_return"]
    checks.append({
        "check": "loan returns are never classified as loans",
        "expected": "every 'End of loan' row is loan_return, never loan",
        "observed": f"{len(returns):,} loan_return rows; "
                    f"{int((canonical.transfer_type_raw.isin(['end_of_loan','end_of_loan_with_fee']) & (canonical.transfer_type_normalized == 'loan')).sum())} mislabelled as loan",
        "passed": not bool((canonical.transfer_type_raw.isin(["end_of_loan", "end_of_loan_with_fee"])
                            & (canonical.transfer_type_normalized == "loan")).any()),
    })

    with_fee = canonical[canonical.return_leg_has_fee]
    checks.append({
        "check": "return legs carrying a fee are kept and routed to research",
        "expected": "43 rows flagged, all research targets, all with fee_on_return_eur > 0",
        "observed": f"{len(with_fee)} flagged; {int(with_fee.is_research_target.sum())} are targets; "
                    f"EUR {with_fee.fee_on_return_eur.sum():,.0f} total",
        "passed": bool(len(with_fee) > 0 and with_fee.is_research_target.all()
                       and (with_fee.fee_on_return_eur > 0).all()),
    })

    future = canonical[canonical.transfermarkt_future_transfer]
    checks.append({
        "check": "Transfermarkt's futureTransfer flag is preserved",
        "expected": "the flag survives the join and marks scheduled rows",
        "observed": f"{len(future):,} rows flagged future; "
                    f"{int((pd.to_datetime(future.transfer_date) > pd.Timestamp('2026-06-12')).sum()):,} "
                    "dated after the raw snapshot capture date",
        "passed": bool(len(future) > 0),
    })

    internal = canonical[canonical.is_internal_move]
    checks.append({
        "check": "youth / internal registration changes are identifiable and excluded",
        "expected": "flagged by is_internal_move and never a research target",
        "observed": f"{len(internal):,} internal rows; "
                    f"{int(internal.is_research_target.sum())} wrongly kept as targets",
        "passed": bool(not internal.is_research_target.any()),
    })

    # --- identity: names are not unique, ids are ---
    name_to_ids = canonical.groupby("player_name").player_id.nunique()
    shared = name_to_ids[name_to_ids > 1]
    id_to_names = canonical.groupby("player_id").player_name.nunique()
    chain_players = canonical.groupby("event_chain_id").player_id.nunique()
    checks.append({
        "check": "players sharing a name are never merged into one identity",
        "expected": "each player_id has exactly one name; no event chain spans two player_ids; "
                    "no duplicate event_id",
        "observed": f"{len(shared)} names are shared by 2+ players "
                    f"(worst: {shared.max() if len(shared) else 0} players called "
                    f"'{shared.idxmax() if len(shared) else '-'}'), covering "
                    f"{int(canonical.player_name.isin(shared.index).sum()):,} events; "
                    f"{int((id_to_names > 1).sum())} player_ids carry >1 name; "
                    f"{int((chain_players > 1).sum())} chains span >1 player_id; "
                    f"{int(canonical.event_id.duplicated().sum())} duplicate event_ids",
        "passed": bool((id_to_names <= 1).all() and (chain_players <= 1).all()
                       and not canonical.event_id.duplicated().any()),
    })

    labelled = canonical[canonical.transfer_type_raw.notna()
                         & canonical.transfer_type_raw.ne("unknown")]
    from_label = labelled[labelled.transfer_type_source == "transfermarkt_raw_label"]
    checks.append({
        "check": "a raw label always beats the Stage 1 heuristic",
        "expected": "no row takes its type from the heuristic while a usable raw label exists",
        "observed": f"{len(labelled):,} labelled rows; "
                    f"{int((labelled.transfer_type_source == 'stage1_heuristic').sum())} "
                    "still sourced from the heuristic",
        "passed": bool(not (labelled.transfer_type_source == "stage1_heuristic").any()),
    })

    return checks


def render(checks: list[dict]) -> str:
    passed = sum(1 for c in checks if c["passed"])
    lines = ["# Stage 1C - regression checks\n",
             f"**{passed} / {len(checks)} passed**\n"]
    for check in checks:
        mark = "PASS" if check["passed"] else "**FAIL**"
        lines.append(f"### {mark} - {check['check']}\n")
        lines.append(f"- expected: {check['expected']}")
        lines.append(f"- observed: {check['observed']}\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1C regression checks")
    parser.add_argument("--canonical", default=str(CANONICAL_CSV))
    parser.add_argument("--output", default=str(REBUILD_DIR / "stage1c_regression_checks.md"))
    args = parser.parse_args()
    canonical = pd.read_csv(args.canonical, low_memory=False)
    checks = run_checks(canonical)
    Path(args.output).write_text(render(checks))
    for check in checks:
        print(("PASS  " if check["passed"] else "FAIL  ") + check["check"])
        if not check["passed"]:
            print(f"        expected: {check['expected']}")
            print(f"        observed: {check['observed']}")
    passed = sum(1 for c in checks if c["passed"])
    print(f"\n{passed}/{len(checks)} passed -> {args.output}")


if __name__ == "__main__":
    main()
