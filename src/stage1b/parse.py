"""Parse one Transfermarkt transfer row, preserving the label AND the amount.

The whole point of Stage 1B is the thing the upstream dbt model does not do:

    upstream:  "Loan fee:<br /><i>€500k</i>"  ->  transfer_fee = 0
    here:      transfer_type_raw = "loan_fee"
               fee_display_raw   = 'Loan fee:<br /><i class="normaler-text">€500k</i>'
               fee_numeric_eur   = 500000
               fee_amount_role   = "loan_fee"

`fee_display_raw` is always the byte-exact string Transfermarkt returned, kept
before any normalization so a wrong parse can always be traced back.
"""
from __future__ import annotations

import re

# Every label observed across all 135,065 transfer rows in the 2025 raw file.
# Counts are from that pass and are recorded so a new label shows up as
# "unknown" rather than being silently folded into an existing bucket.
LABEL_RULES = [
    ("no_fee_shown",         lambda s: s == "-"),                              # 33,554
    ("free_transfer",        lambda s: s.lower() == "free transfer"),          # 27,330
    ("end_of_loan",          lambda s: s.lower() == "end of loan"),            # 22,841
    ("loan_transfer",        lambda s: s.lower() == "loan transfer"),          # 21,185
    ("undisclosed",          lambda s: s == "?"),                              # 13,634
    ("loan_fee",             lambda s: s.lower().startswith("loan fee")),      #  1,700
    ("end_of_loan_with_fee", lambda s: s.lower().startswith("end of loan")),   #     18
    ("draft",                lambda s: s.lower() == "draft"),                  #    220
    ("paid_transfer",        lambda s: bool(re.fullmatch(r"€[\d.,]+\s*[mk]?", s, re.I))),
]

# Which kind of money the parsed amount represents. This is the distinction the
# research needs and the one the DuckDB destroys.
AMOUNT_ROLE = {
    "paid_transfer": "permanent_fee",
    "loan_fee": "loan_fee",
    "end_of_loan_with_fee": "fee_on_return",
    "free_transfer": "none_free",
}

_TAGS = re.compile(r"<[^>]+>")
_AMOUNT = re.compile(r"€\s*([\d.,]+)\s*([mk])?", re.I)


def classify_label(fee_display_raw) -> str:
    """Map Transfermarkt's displayed fee text to a stable label. Never guesses."""
    if fee_display_raw is None:
        return "unknown"
    text = str(fee_display_raw).strip()
    if not text:
        return "unknown"
    for label, matches in LABEL_RULES:
        if matches(text):
            return label
    return "unknown"


def parse_amount_eur(fee_display_raw) -> float | None:
    """Extract a euro amount, or None when the text carries no amount.

    Returns None - never 0 - when no amount is present, so that "no amount
    stated" stays distinguishable from "the amount was zero".
    """
    if fee_display_raw is None:
        return None
    text = _TAGS.sub(" ", str(fee_display_raw))
    match = _AMOUNT.search(text)
    if not match:
        return None
    number = match.group(1).replace(",", "")
    if number.count(".") > 1:          # thousands separators, e.g. 1.234.567
        number = number.replace(".", "")
    try:
        value = float(number)
    except ValueError:
        return None
    suffix = (match.group(2) or "").lower()
    if suffix == "m":
        return value * 1_000_000
    if suffix == "k":
        return value * 1_000
    return value


def club_id_from_href(href: str | None) -> int | None:
    """Club id out of '/al-orooba/transfers/verein/51771/saison_id/2025'."""
    if not href:
        return None
    match = re.search(r"/verein/(\d+)", href)
    return int(match.group(1)) if match else None


def transfer_id_from_url(url: str | None) -> int | None:
    """Transfermarkt's own per-transfer id, which the upstream model discards."""
    if not url:
        return None
    match = re.search(r"/transfer_id/(\d+)", url)
    return int(match.group(1)) if match else None


def parse_transfer_row(player_id: int, player_name: str, transfer: dict) -> dict:
    """One raw transfer object -> one flat row. Raw text is always preserved."""
    fee_raw = transfer.get("fee")
    mv_raw = transfer.get("marketValue")
    label = classify_label(fee_raw)
    amount = parse_amount_eur(fee_raw)
    # "free transfer" states an amount of zero; every other label either carries
    # its own amount or states none at all.
    fee_numeric = 0.0 if label == "free_transfer" else amount
    return {
        "player_id": player_id,
        "player_name": player_name,
        "season": transfer.get("season"),
        "transfer_date": transfer.get("dateUnformatted"),
        "transfer_date_display": transfer.get("date"),
        "from_club": (transfer.get("from") or {}).get("clubName"),
        "to_club": (transfer.get("to") or {}).get("clubName"),
        "from_club_id": club_id_from_href((transfer.get("from") or {}).get("href")),
        "to_club_id": club_id_from_href((transfer.get("to") or {}).get("href")),
        "market_value_display": mv_raw,
        "market_value_eur": parse_amount_eur(mv_raw),
        "fee_display_raw": fee_raw,
        "transfer_type_raw": label,
        "fee_numeric_eur": fee_numeric,
        "fee_amount_role": AMOUNT_ROLE.get(label),
        "is_loan_related": label in {"loan_transfer", "end_of_loan", "loan_fee",
                                     "end_of_loan_with_fee"},
        "transfermarkt_transfer_id": transfer_id_from_url(transfer.get("url")),
        "future_transfer": bool(transfer.get("futureTransfer")),
        "upcoming": bool(transfer.get("upcoming")),
        "transfer_url_path": transfer.get("url"),
    }


def parse_player(player_id: int, player_name: str, response: dict) -> list[dict]:
    return [
        parse_transfer_row(player_id, player_name, transfer)
        for transfer in (response.get("transfers") or [])
    ]
