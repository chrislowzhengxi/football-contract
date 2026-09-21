"""Build the full raw-Transfermarkt enrichment table for every backbone event.

Streams the three published upstream raw season files (Stage 1B established
these are public on the same R2 bucket as the DuckDB, and cover 100% of the
backbone) and emits one enrichment row per (player, date, from_club, to_club).

Makes no requests to transfermarkt.com.
"""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

from ..stage1.backbone import REBUILD_DIR
from ..stage1b.parse import (
    classify_label,
    club_id_from_href,
    parse_amount_eur,
    transfer_id_from_url,
)
from ..stage1b.raw_api import SEASON_PREFERENCE, USER_AGENT, _blob_url, snapshot_manifest, transfers_file_md5

ENRICHMENT_CSV = REBUILD_DIR / "stage1c_transfermarkt_enrichment.csv"

JOIN_KEYS = ["raw_player_id", "join_date", "raw_from_club_id", "raw_to_club_id"]


def build_enrichment() -> tuple[pd.DataFrame, dict]:
    """One row per raw transfer, keyed for joining onto the Stage 1 backbone."""
    manifest = snapshot_manifest()
    seen_players: set[int] = set()
    rows: list[dict] = []
    per_season: dict[str, int] = {}
    duplicate_keys = 0
    keys_seen: set[tuple] = set()

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
                # Upstream takes the most recent season file that has the
                # player; we mirror that so our rows match the DuckDB's.
                if not payload or player_id in seen_players:
                    continue
                seen_players.add(player_id)
                added += 1
                for transfer in payload.get("transfers") or []:
                    fee_raw = transfer.get("fee")
                    label = classify_label(fee_raw)
                    key = (
                        player_id,
                        transfer.get("dateUnformatted"),
                        club_id_from_href((transfer.get("from") or {}).get("href")),
                        club_id_from_href((transfer.get("to") or {}).get("href")),
                    )
                    if key in keys_seen:
                        duplicate_keys += 1
                        continue
                    keys_seen.add(key)
                    rows.append({
                        "raw_player_id": key[0],
                        "join_date": key[1],
                        "raw_from_club_id": key[2],
                        "raw_to_club_id": key[3],
                        "transfermarkt_transfer_id": transfer_id_from_url(transfer.get("url")),
                        "transfermarkt_raw_fee_text": fee_raw,
                        "transfermarkt_raw_label": label,
                        "transfermarkt_fee_numeric_eur": (
                            0.0 if label == "free_transfer" else parse_amount_eur(fee_raw)),
                        "transfermarkt_market_value_display": transfer.get("marketValue"),
                        "transfermarkt_future_transfer": bool(transfer.get("futureTransfer")),
                        "transfermarkt_upcoming": bool(transfer.get("upcoming")),
                        "transfermarkt_source_season_file": season,
                    })
        per_season[season] = added

    frame = pd.DataFrame(rows)
    meta = {
        "raw_players_read": len(seen_players),
        "new_players_per_season_file": per_season,
        "raw_transfer_rows": len(frame),
        "duplicate_join_keys_dropped": duplicate_keys,
        "transfermarkt_com_requests": 0,
        "source": "dcaribou/transfermarkt-datasets raw acquisition files, DVC md5 "
                  "3e2a95f72dfccad1d3f2f3677e661102.dir (the hash pinned by the commit "
                  "our DuckDB snapshot declares)",
    }
    return frame, meta


def load_enrichment(path: Path = ENRICHMENT_CSV, rebuild: bool = False) -> tuple[pd.DataFrame, dict]:
    """Cached accessor. The cache is a plain CSV so it stays inspectable."""
    meta_path = path.with_name(path.stem + "_meta.json")
    if path.exists() and not rebuild:
        frame = pd.read_csv(path, low_memory=False)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["loaded_from_cache"] = True
        return frame, meta
    frame, meta = build_enrichment()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    meta["loaded_from_cache"] = False
    return frame, meta
