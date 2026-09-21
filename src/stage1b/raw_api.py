"""Fetch the upstream project's RAW Transfermarkt acquisition files.

Provenance chain, all verifiable:

  dcaribou/transfermarkt-datasets @ 154367df   (the commit our DuckDB declares)
    -> data/raw/transfermarkt-api.dvc          md5 3e2a95f7...dir
      -> https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/dvc/files/md5/3e/2a95...dir
        -> {season}/transfers.json             one JSON object per player per line

That is the same public bucket `src/config.py` already downloads the DuckDB
from. We are re-reading the upstream project's own inputs, not scraping
transfermarkt.com.

Each line is {"player_id": int, "response": {...} | null}, and
response.transfers is the list of transfer rows as Transfermarkt's own
`/ceapi/transferHistory/list/{player_id}` endpoint returned them.
"""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

from ..config import ROOT

DVC_HTTP_REMOTE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/dvc/files/md5"
SNAPSHOT_DIR_MD5 = "3e2a95f72dfccad1d3f2f3677e661102"
CACHE_DIR = ROOT / "data" / "raw" / "transfermarkt-api"
USER_AGENT = "football-contract-research/0.1 (MIT academic research)"

# Upstream takes, per player, the most recent season file that has data for
# them. We mirror that order.
SEASON_PREFERENCE = ["2025", "2024", "2023"]


def _blob_url(md5: str) -> str:
    return f"{DVC_HTTP_REMOTE}/{md5[:2]}/{md5[2:]}"


def _open(url: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    )


def snapshot_manifest() -> list[dict]:
    """The file listing for the raw acquisition directory our DuckDB was built from."""
    with _open(_blob_url(SNAPSHOT_DIR_MD5 + ".dir")) as response:
        return json.load(response)


def transfers_file_md5(manifest: list[dict], season: str) -> str | None:
    for entry in manifest:
        if entry["relpath"] == f"{season}/transfers.json":
            return entry["md5"]
    return None


def stream_players(
    season: str, manifest: list[dict], wanted: set[int] | None = None,
    cache_dir: Path | None = None,
) -> dict[int, dict]:
    """Return {player_id: raw API response} for `wanted` players in one season file.

    Streams the remote file rather than downloading it whole, unless a local
    cache copy exists. `wanted=None` returns every player, which is large.
    """
    md5 = transfers_file_md5(manifest, season)
    if md5 is None:
        return {}
    cache_dir = cache_dir or CACHE_DIR
    cached = cache_dir / season / "transfers.json"

    found: dict[int, dict] = {}
    source = cached.open(encoding="utf-8", errors="replace") if cached.exists() else None
    handle = None
    if source is None:
        handle = _open(_blob_url(md5))
        source = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
    try:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            player_id = record.get("player_id")
            if wanted is not None and player_id not in wanted:
                continue
            if not record.get("response"):
                continue
            found[player_id] = record["response"]
            if wanted is not None and len(found) == len(wanted):
                break
    finally:
        source.close()
        if handle is not None:
            handle.close()
    return found


def collect_players(player_ids: set[int], cache_dir: Path | None = None) -> tuple[dict[int, dict], dict[int, str]]:
    """Resolve each player from the most recent season file that has them.

    Returns ({player_id: response}, {player_id: season the data came from}).
    """
    manifest = snapshot_manifest()
    responses: dict[int, dict] = {}
    origin: dict[int, str] = {}
    outstanding = set(player_ids)
    for season in SEASON_PREFERENCE:
        if not outstanding:
            break
        found = stream_players(season, manifest, outstanding, cache_dir)
        for player_id, response in found.items():
            responses[player_id] = response
            origin[player_id] = season
        outstanding -= set(found)
    return responses, origin
