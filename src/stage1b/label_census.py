"""Census of Transfermarkt's fee labels across the whole backbone.

Streams the three published upstream raw season files and answers, for every
one of our 175,165 backbone events, what Transfermarkt's displayed fee text
actually said - and therefore exactly what the upstream dbt model threw away.

Makes no requests to transfermarkt.com.

    python -m src.stage1b.label_census
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

from ..stage1.backbone import REBUILD_DIR
from .parse import classify_label, club_id_from_href, parse_amount_eur
from .raw_api import SEASON_PREFERENCE, USER_AGENT, _blob_url, snapshot_manifest, transfers_file_md5

# What the upstream dbt model (base_transfers.sql) does with each label.
UPSTREAM_BEHAVIOUR = {
    "no_fee_shown":         ("NULL", "faithful - Transfermarkt showed no fee"),
    "undisclosed":          ("NULL", "LOSS - '?' means explicitly undisclosed, stored the same as '-'"),
    "free_transfer":        ("0", "faithful value, but the 'free' label itself is lost"),
    "loan_transfer":        ("0", "LOSS - the loan label is destroyed"),
    "end_of_loan":          ("0", "LOSS - the end-of-loan label is destroyed"),
    "loan_fee":             ("0", "LOSS - a real loan fee is destroyed"),
    "end_of_loan_with_fee": ("0", "LOSS - a fee paid on return is destroyed"),
    "draft":                ("0", "LOSS - MLS draft label destroyed"),
    "paid_transfer":        ("parsed euros", "faithful"),
    "unknown":              ("0", "unclassified label"),
}


def census(backbone: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = set(zip(
        backbone.raw_player_id,
        pd.to_datetime(backbone.raw_transfer_date).dt.strftime("%Y-%m-%d"),
        backbone.raw_from_club_id, backbone.raw_to_club_id,
    ))
    manifest = snapshot_manifest()
    seen: set[int] = set()
    raw: dict[tuple, tuple[str, float | None]] = {}
    per_season: dict[str, int] = {}
    for season in SEASON_PREFERENCE:
        md5 = transfers_file_md5(manifest, season)
        if md5 is None:
            continue
        added = 0
        request = urllib.request.Request(_blob_url(md5), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response:
            for line in io.TextIOWrapper(response, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                player_id, payload = record.get("player_id"), record.get("response")
                if not payload or player_id in seen:
                    continue
                seen.add(player_id)
                added += 1
                for transfer in payload.get("transfers") or []:
                    key = (player_id, transfer.get("dateUnformatted"),
                           club_id_from_href((transfer.get("from") or {}).get("href")),
                           club_id_from_href((transfer.get("to") or {}).get("href")))
                    raw[key] = (classify_label(transfer.get("fee")),
                                parse_amount_eur(transfer.get("fee")))
        per_season[season] = added

    matched = keys & set(raw)
    counts = collections.Counter(raw[key][0] for key in matched)
    amounts = collections.defaultdict(float)
    for key in matched:
        label, amount = raw[key]
        if amount:
            amounts[label] += amount

    total = sum(counts.values())
    rows = [{
        "transfer_type_raw": label,
        "backbone_events": count,
        "share_pct": round(count / total * 100, 2),
        "total_amount_eur": round(amounts.get(label, 0.0)) or None,
        "upstream_duckdb_stores": UPSTREAM_BEHAVIOUR.get(label, ("?", "?"))[0],
        "assessment": UPSTREAM_BEHAVIOUR.get(label, ("?", "?"))[1],
    } for label, count in counts.most_common()]

    coverage = {
        "backbone_events": len(keys),
        "backbone_players": int(backbone.raw_player_id.nunique()),
        "raw_players_read": len(seen),
        "new_players_per_season_file": per_season,
        "backbone_events_matched_to_raw": len(matched),
        "backbone_event_coverage_pct": round(len(matched) / len(keys) * 100, 2),
        "transfermarkt_com_requests": 0,
    }
    return pd.DataFrame(rows), coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-backbone Transfermarkt label census")
    parser.add_argument("--backbone", default=str(REBUILD_DIR / "stage1_normalized_transfers.csv"))
    parser.add_argument("--output-dir", default=str(REBUILD_DIR))
    args = parser.parse_args()
    backbone = pd.read_csv(args.backbone, low_memory=False)
    table, coverage = census(backbone)
    output_dir = Path(args.output_dir)
    table.to_csv(output_dir / "stage1b_fee_label_census.csv", index=False)
    (output_dir / "stage1b_label_census_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
