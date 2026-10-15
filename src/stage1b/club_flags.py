"""Club -> Transfermarkt country flag, from the upstream raw transfer histories Stage 1B already reads.

    python -m src.stage1b.club_flags

Each `from`/`to` side of a raw transfer row carries the club's Transfermarkt id (in `href`) and a
`countryFlag` image whose file name is Transfermarkt's country id, the same id as the snapshot's
`countries.country_id`. This module keeps those fields, per club, for every Stage 1C player, across every
cached season file. It reads the local cache only (`raw_api.cache_season_files` fills it once) and leaves the
Stage 1B parse that Stage 1C was built from untouched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..config import ROOT
from .parse import club_id_from_href
from .raw_api import CACHE_DIR, SEASON_PREFERENCE

FLAG = re.compile(r"/flagge/[^/]+/(\d+)\.png")
OUT = ROOT / "data" / "outputs" / "rebuild"
RAW_FLAGS_CSV = OUT / "club_country_raw_flags.csv"
STATUSES = ("single_flag", "single_flag_not_in_snapshot_countries", "conflicting_flags", "no_flag", "special_entry")


def side_rows(response: dict, player_id: int, season: str) -> list[dict]:
    """One row per from/to side of every transfer in one raw response."""
    rows = []
    for t in response.get("transfers") or []:
        f, to = t.get("from") or {}, t.get("to") or {}
        key = t.get("url") or f"{player_id}|{t.get('dateUnformatted')}|{f.get('href')}|{to.get('href')}"
        for side, c in (("from", f), ("to", to)):
            m = FLAG.search(c.get("countryFlag") or "")
            rows.append({"player_id": player_id, "season_file": season, "transfer_key": key, "side": side,
                         "club_id": club_id_from_href(c.get("href")), "club_name": c.get("clubName"),
                         "flag_id": int(m.group(1)) if m else None, "special": bool(c.get("isSpecial")),
                         "date": t.get("dateUnformatted")})
    return rows


def read_sides(player_ids: set[int], cache_dir: Path | None = None,
               seasons: list[str] | None = None) -> pd.DataFrame:
    """Every side of every raw transfer row for `player_ids`, from every cached season file.

    `preferred` marks rows from the response Stage 1B used for that player (the most recent season file that
    has data for them), which is the response Stage 1C was built from.
    """
    cache_dir = cache_dir or CACHE_DIR
    seasons = seasons or SEASON_PREFERENCE
    rows, have = [], {}
    for season in seasons:
        path = cache_dir / season / "transfers.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} is not cached; run raw_api.cache_season_files() once")
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("player_id")
                if pid not in player_ids or not rec.get("response"):
                    continue
                have.setdefault(pid, season)
                rows.extend(side_rows(rec["response"], pid, season))
    S = pd.DataFrame(rows, columns=["player_id", "season_file", "transfer_key", "side", "club_id", "club_name",
                                    "flag_id", "special", "date"])
    S["preferred"] = S.season_file.eq(S.player_id.map(have))
    S["club_id"] = S.club_id.astype("Int64")
    S["flag_id"] = S.flag_id.astype("Int64")
    return S


def aggregate(S: pd.DataFrame, countries: dict[int, str]) -> pd.DataFrame:
    """One row per club id: its flag(s), support and status. The same transfer side read from two season files
    counts once."""
    S = S[S.club_id.notna()].copy()
    d = S.drop_duplicates(["transfer_key", "side", "season_file"])
    uniq = d.drop_duplicates(["transfer_key", "side"])
    g = d.groupby("club_id")
    A = pd.DataFrame({
        "club_names": g.club_name.agg(lambda s: " | ".join(sorted(set(s.dropna())))),
        "flag_ids": g.flag_id.agg(lambda s: "|".join(str(int(x)) for x in sorted(set(s.dropna())))),
        "n_distinct_flags": g.flag_id.agg(lambda s: s.dropna().nunique()),
        "rows_without_flag": g.flag_id.agg(lambda s: int(s.isna().sum())),
        "special_entry": g.special.any(),
        "n_players": g.player_id.nunique(),
        "season_files": g.season_file.agg(lambda s: "|".join(sorted(set(s)))),
        "in_stage1b_preferred_response": g.preferred.any(),
        "first_date": g.date.min(), "last_date": g.date.max(),
        "example_transfer": g.transfer_key.min()})
    A["n_supporting_rows"] = uniq.groupby("club_id").size().reindex(A.index).fillna(0).astype(int)
    A = A.reset_index()
    one = A.n_distinct_flags.eq(1)
    A["country_flag_id"] = A.flag_ids.where(one, "").map(lambda x: int(x) if x else None).astype("Int64")
    A["country_name"] = A.country_flag_id.map(lambda x: countries.get(int(x), "") if pd.notna(x) else "")
    A["mapping_status"] = [
        "special_entry" if sp else "no_flag" if n == 0 else "conflicting_flags" if n > 1
        else "single_flag" if name else "single_flag_not_in_snapshot_countries"
        for sp, n, name in zip(A.special_entry, A.n_distinct_flags, A.country_name)]
    cols = ["club_id", "club_names", "country_flag_id", "country_name", "n_supporting_rows", "n_distinct_flags",
            "flag_ids", "rows_without_flag", "special_entry", "mapping_status", "n_players", "season_files",
            "in_stage1b_preferred_response", "first_date", "last_date", "example_transfer"]
    return A[cols].sort_values("club_id").reset_index(drop=True)


def snapshot_countries(db_path=None) -> dict[int, str]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    try:
        d = con.sql("select cast(country_id as integer) id, country_name from countries").df()
    finally:
        con.close()
    return dict(zip(d.id.astype(int), d.country_name))


def build(player_ids: set[int] | None = None, cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(side rows, per-club flag table) for the Stage 1C player universe, from the local cache only."""
    if player_ids is None:
        from ..analysis.loan_episodes import load_canonical
        player_ids = set(load_canonical().player_id.astype(int))
    S = read_sides(set(player_ids), cache_dir)
    return S, aggregate(S, snapshot_countries())


def main() -> pd.DataFrame:
    from .raw_api import cache_season_files

    cache_season_files()                       # no-op when the verified cache is present
    S, A = build()
    A.to_csv(RAW_FLAGS_CSV, index=False)
    print(f"wrote {RAW_FLAGS_CSV.name}: {len(A):,} clubs from {S.player_id.nunique():,} players")
    print(A.mapping_status.value_counts().to_string())
    return A


if __name__ == "__main__":
    main()
