"""What does the Transfermarkt snapshot actually cover?

    python -m src.analysis.provenance_audit              # local evidence only
    python -m src.analysis.provenance_audit --upstream   # + upstream raw files

Local mode reads the DuckDB snapshot, the frozen Stage 1C table and the
Stage 1C enrichment (read-only) and writes a per-season coverage table.

`--upstream` also reads the upstream project's own raw acquisition files -
all 14 per-season squad lists (2012-2025) and the three transfer-history files
(2023-2025) pinned by the commit our DuckDB declares - from the project's
public DVC bucket (the same bucket `src/config.py` downloads the DuckDB from),
and writes one squad-coverage row per season. It never contacts
transfermarkt.com. It streams about 400 MB and keeps only per-player summaries.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import urllib.request

import duckdb
import pandas as pd

from ..config import DEFAULT_DATABASE
from .loan_episodes import ROOT, build_universe, club_geography, load_canonical

OUT = ROOT / "data" / "outputs" / "rebuild"
ENRICHMENT_CSV = OUT / "stage1c_transfermarkt_enrichment.csv"
DVC_REMOTE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/dvc/files/md5"
# data/raw/*.dvc at upstream commit 154367df
API_DIR_MD5 = "3e2a95f72dfccad1d3f2f3677e661102"
SCRAPER_DIR_MD5 = "c9ec80fd8b18310f7bded68fe92b67d3"
TRANSFER_SEASONS = ("2023", "2024", "2025")   # the only transfers.json files upstream holds
SQUAD_SEASONS = tuple(str(y) for y in range(2012, 2026))   # every club-squad season upstream holds
USER_AGENT = "football-contract-research/0.1 (MIT academic research)"


# ---------------------------------------------------------------------------
# Local evidence
# ---------------------------------------------------------------------------

def upstream_tables(con) -> dict:
    q = lambda s: con.sql(s).fetchall()
    return {
        "competitions": q("select count(*), count(distinct country_name) from competitions")[0],
        "competition_types": con.sql("select type, count(*) n from competitions group by 1 order by 1").df(),
        "game_competitions_missing_from_table": [r[0] for r in q(
            "select distinct competition_id from games where competition_id not in "
            "(select competition_id from competitions) order by 1")],
        "games": q("select count(*), min(date), max(date), min(season), max(season) from games")[0],
        "games_by_type": con.sql("""select c.type, min(g.season) first_season, max(g.season) last_season,
                                           min(g.date) first_date, max(g.date) last_date, count(*) games
                                    from games g join competitions c using (competition_id)
                                    group by 1 order by 1""").df(),
        "leagues_by_season": con.sql("""select season, count(distinct competition_id) leagues,
                                               string_agg(distinct competition_id, ',' order by competition_id) ids
                                        from games where competition_type = 'domestic_league'
                                        group by 1 order by 1""").df(),
        "clubs": q("select count(*) from clubs")[0][0],
        "clubs_in_games": q("select count(distinct c) from (select unnest([home_club_id, away_club_id]) c from games)")[0][0],
        "players": q("select count(*) from players")[0][0],
        "players_with_appearances": q("select count(distinct player_id) from appearances")[0][0],
        "appearances": q("select min(date), max(date) from appearances")[0],
        "valuations": q("select min(date), max(date) from player_valuations")[0],
        "transfers": q("select count(*), count(distinct player_id), min(transfer_date), max(transfer_date) from transfers")[0],
        "players_by_last_season": con.sql("""
            with tp as (select distinct player_id from transfers)
            select p.last_season, count(*) players, count(tp.player_id) with_transfers
            from players p left join tp using (player_id) group by 1 order by 1""").df(),
        "earliest_transfers": con.sql("""
            select t.player_id, p.name, p.date_of_birth, t.transfer_date, t.from_club_name, t.to_club_name
            from transfers t join players p using (player_id)
            order by t.transfer_date, t.player_id limit 5""").df(),
    }


def player_source_file(enrichment_csv=ENRICHMENT_CSV) -> pd.Series:
    """player_id -> the upstream season file that supplied his history."""
    e = pd.read_csv(enrichment_csv, usecols=["raw_player_id", "transfermarkt_source_season_file"])
    e = e.drop_duplicates()
    assert e.raw_player_id.is_unique, "a player's history came from more than one file"
    return e.set_index("raw_player_id").transfermarkt_source_season_file.astype(str)


def source_file_capture(rows: pd.DataFrame, enrichment_csv=ENRICHMENT_CSV) -> pd.DataFrame:
    """Per upstream season file, over Stage 1C rows: rows, players, last
    realised and first scheduled date. The file was captured between those two
    dates, because Transfermarkt flags a move as future only while it has not
    happened."""
    src = rows.player_id.map(player_source_file(enrichment_csv))
    assert src.notna().all(), "a Stage 1C player has no source file"
    fut = rows.transfermarkt_future_transfer.astype(bool)
    out = []
    for s, g in rows.groupby(src):
        f = fut.loc[g.index]
        out.append({"source_season_file": str(s), "rows": len(g), "players": g.player_id.nunique(),
                    "last_realised_date": g.loc[~f, "_date"].max().date(),
                    "first_scheduled_date": g.loc[f, "_date"].min().date(), "scheduled_rows": int(f.sum())})
    return pd.DataFrame(out)


def season_table(u, geo, source_of_player: dict | None = None) -> pd.DataFrame:
    rows, ep = u.rows, u.episodes
    mapped = set(geo.club_id)
    age = (rows._date - pd.to_datetime(rows.date_of_birth, errors="coerce")).dt.days / 365.25
    out = []
    for s in sorted(set(rows.season)):
        rr, ee = rows[rows.season == s], ep[ep.season == s]
        clubs = set(ee.from_club_id) | set(ee.to_club_id)
        rec = {"season": s, "raw_rows": len(rr), "raw_players": rr.player_id.nunique(),
               "episodes": len(ee), "episode_players": ee.player_id.nunique(),
               "episode_clubs": len(clubs), "episode_clubs_with_country": len(clubs & mapped),
               "pct_episode_clubs_with_country": round(100 * len(clubs & mapped) / len(clubs), 1) if clubs else None,
               "loans": int((ee.episode_type == "loan").sum()),
               "pct_rows_youth_or_reserve_side": round(100 * rr.is_youth_or_reserve_side.mean(), 1),
               "median_age_at_move": round(age.loc[rr.index].median(), 1),
               "scheduled_future_rows": int(rr.transfermarkt_future_transfer.sum())}
        if source_of_player:
            src = rr.player_id.map(source_of_player)
            for f in TRANSFER_SEASONS:
                rec[f"rows_from_{f}_file"] = int((src == f).sum())
        out.append(rec)
    return pd.DataFrame(out)


def club_country_counts(u, geo, con) -> dict:
    rows, ep, L = u.rows, u.episodes, u.loans
    clubs_tbl = {int(r[0]) for r in con.sql("select club_id from clubs").fetchall()}   # stored as text
    raw_clubs = set(rows.from_club_id) | set(rows.to_club_id)
    ep_clubs = set(ep.from_club_id) | set(ep.to_club_id)
    games_clubs = {int(r[0]) for r in con.sql(
        "select distinct c from (select unnest([home_club_id, away_club_id]) c from games) where c is not null").fetchall()}
    f, b = L.from_country.notna(), L.to_country.notna()
    return {
        "raw_transfer_clubs": len(raw_clubs), "episode_clubs": len(ep_clubs),
        "clubs_table": len(clubs_tbl), "clubs_table_in_transfer_histories": len(clubs_tbl & raw_clubs),
        "clubs_in_covered_games": len(games_clubs), "episode_clubs_in_covered_games": len(ep_clubs & games_clubs),
        "clubs_with_country": len(geo), "clubs_with_country_by_source": geo.geo_source.value_counts().to_dict(),
        "episode_clubs_with_country": len(ep_clubs & set(geo.club_id)),
        "episodes_touching_clubs_table_club": int((ep.from_club_id.isin(clubs_tbl) | ep.to_club_id.isin(clubs_tbl)).sum()),
        "loans": len(L), "loans_lender_country_known": int(f.sum()), "loans_borrower_country_known": int(b.sum()),
        "loans_both_known": int((f & b).sum()), "loans_neither_known": int((~f & ~b).sum()),
    }


def pre_2012_players(u, con) -> dict:
    rows = u.rows
    pre = set(rows.loc[rows._date < "2012-07-01", "player_id"])
    pl = con.sql("select player_id, last_season from players").df().set_index("player_id")
    app = con.sql("select player_id, max(date) last_app from appearances group by 1").df().set_index("player_id")
    in_app = [p for p in pre if p in app.index]
    r = rows[rows._date < "2012-07-01"]
    age = (r._date - pd.to_datetime(r.date_of_birth, errors="coerce")).dt.days / 365.25
    return {"players": len(pre), "in_players_table": int(pl.index.isin(list(pre)).sum()),
            "last_season_2023_or_later": int((pl.loc[list(pre), "last_season"] >= "2023").sum()),
            "with_covered_appearance": len(in_app),
            "with_covered_appearance_from_2023_07": int((pd.to_datetime(app.loc[in_app, "last_app"]) >= "2023-07-01").sum()),
            "rows": len(r), "median_age_at_move": round(age.median(), 1),
            "pct_rows_youth_or_reserve_side": round(100 * r.is_youth_or_reserve_side.mean(), 1)}


# ---------------------------------------------------------------------------
# Upstream raw files (public DVC bucket; never transfermarkt.com)
# ---------------------------------------------------------------------------

def _get(md5: str):
    return urllib.request.urlopen(urllib.request.Request(f"{DVC_REMOTE}/{md5[:2]}/{md5[2:]}",
                                                         headers={"User-Agent": USER_AGENT}))


def manifest(dir_md5: str) -> dict[str, str]:
    with _get(dir_md5 + ".dir") as r:
        return {e["relpath"]: e["md5"] for e in json.load(r)}


def squad_entries(md5: str) -> pd.DataFrame:
    """One row per (player, squad) in a season's squad file: player_id,
    squad_type ('club' or 'national_team') and the squad's Transfermarkt id."""
    with _get(md5) as r:
        text = gzip.decompress(r.read()).decode("utf-8", "replace")
    rows = []
    for line in text.splitlines():
        if line.startswith("{"):
            p = json.loads(line)
            parent = p.get("parent") or {}
            href = (parent.get("href") or "").rstrip("/")
            rows.append((int(p["href"].rstrip("/").split("/")[-1]), parent.get("type") or "club",
                         int(href.split("/")[-1]) if href.split("/")[-1].isdigit() else None))
    return pd.DataFrame(rows, columns=["player_id", "squad_type", "squad_id"]).drop_duplicates()


def squad_kind(entries: pd.DataFrame) -> dict[int, str]:
    """player_id -> 'club' if on any club squad, else 'national_team'."""
    club = set(entries.loc[entries.squad_type == "club", "player_id"])
    return {p: "club" if p in club else "national_team" for p in entries.player_id}


def transfer_file(md5: str) -> dict[int, str]:
    """player_id -> 'data' | 'empty' | 'null' for one season's transfers.json."""
    out = {}
    with _get(md5) as r:
        for line in io.TextIOWrapper(r, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = d.get("response")
            dated = [t for t in ((resp or {}).get("transfers") or [])
                     if t.get("dateUnformatted") not in (None, "", "0000-00-00")]
            out[d.get("player_id")] = "null" if not resp else ("data" if dated else "empty")
    return out


def squad_coverage_row(season: str, squad: dict[int, str], tf: dict[str, dict[int, str]],
                       db_players: set[int]) -> dict:
    """One squad season: who was on it, who was fetched in that season's
    transfer acquisition, and who has a history only because a different
    season's acquisition fetched him."""
    ids = set(squad)
    this = tf.get(season)
    fetched = {p for p, v in (this or {}).items() if v != "null"}
    later = {p for s, f in tf.items() if s > season for p, v in f.items() if v == "data"}
    hist = ids & db_players
    via_other = hist - fetched
    nt_only = {p for p, v in squad.items() if v == "national_team"}
    return {
        "squad_season": season, "season": f"{season}/{str(int(season) + 1)[2:]}",
        "squad_players": len(ids),
        "club_squad_players": len(ids) - len(nt_only),
        "national_team_only_players": len(nt_only),
        "transfers_json_exists": this is not None,
        "transfers_json_players_total": len(this) if this is not None else 0,
        "squad_players_in_this_seasons_transfers_json": len(ids & set(this or {})),
        "fetched_non_null_response": len(ids & fetched),
        "fetched_null_response": len(ids & {p for p, v in (this or {}).items() if v == "null"}),
        "fetched_but_no_dated_transfers": len(ids & {p for p, v in (this or {}).items() if v == "empty"}),
        "with_transfer_history": len(hist),
        "without_transfer_history": len(ids) - len(hist),
        "pct_with_transfer_history": round(100 * len(hist) / len(ids), 1),
        "history_fetched_in_this_season": len(hist & fetched),
        "history_only_via_other_season_fetch": len(via_other),
        "of_which_via_later_season_fetch": len(via_other & later),
        "of_which_via_earlier_season_fetch_only": len(via_other - later),
        "club_squad_with_history": len((ids - nt_only) & db_players),
        "national_team_only_with_history": len(nt_only & db_players),
    }


def upstream_audit(db_transfer_players: set[int]) -> tuple[pd.DataFrame, dict[int, str], dict]:
    api, scraper = manifest(API_DIR_MD5), manifest(SCRAPER_DIR_MD5)
    files = sorted(k for k in api if k.endswith("transfers.json"))
    assert files == [f"{s}/transfers.json" for s in TRANSFER_SEASONS], files
    squad_files = sorted(k for k in scraper if k.endswith("players.json.gz"))
    assert squad_files == [f"{s}/players.json.gz" for s in SQUAD_SEASONS], squad_files
    tf = {s: transfer_file(api[f"{s}/transfers.json"]) for s in TRANSFER_SEASONS}
    # base_transfers.sql: each player's history comes from the latest season
    # file with a non-null response.
    source = {}
    for s in TRANSFER_SEASONS:
        source.update({p: s for p, v in tf[s].items() if v != "null"})
    entries = {s: squad_entries(scraper[f"{s}/players.json.gz"]) for s in SQUAD_SEASONS}
    squads = {s: squad_kind(e) for s, e in entries.items()}
    rows = [squad_coverage_row(s, squads[s], tf, db_transfer_players) for s in SQUAD_SEASONS]
    leagues = league_coverage(entries, tf, db_transfer_players)
    club2325 = set().union(*(set(entries[s].loc[entries[s].squad_type == "club", "player_id"])
                             for s in TRANSFER_SEASONS))
    nt2025 = set(entries["2025"].loc[entries["2025"].squad_type == "national_team", "player_id"])
    cohort = set().union(*(set(squads[s]) for s in TRANSFER_SEASONS))
    outside = sorted(db_transfer_players - cohort)
    last_squad = {p: max((s for s in SQUAD_SEASONS if p in squads[s]), default=None) for p in outside}
    with_data = {p for f in tf.values() for p, v in f.items() if v == "data"}
    facts = {"transfer_files": files, "squad_files": squad_files,
             "players_per_transfer_file": {s: len(v) for s, v in tf.items()},
             "responses_per_transfer_file": {s: pd.Series(list(v.values())).value_counts().to_dict()
                                             for s, v in tf.items()},
             "db_transfer_players": len(db_transfer_players),
             "db_players_all_in_raw_files": db_transfer_players <= set(source),
             "db_players_by_source_file": pd.Series([source[p] for p in db_transfer_players
                                                     if p in source]).value_counts().sort_index().to_dict(),
             "raw_players_with_dated_history_not_in_db": len(with_data - db_transfer_players),
             "squad_players_2023_2025_union": len(cohort),
             "squad_players_2023_2025_union_with_history": len(cohort & db_transfer_players),
             "db_players_on_a_2023_2025_squad_file": len(db_transfer_players & cohort),
             "db_players_on_a_2023_2025_club_squad": len(db_transfer_players & club2325),
             "db_players_only_via_2025_national_team_squad": len((db_transfer_players - club2325) & nt2025),
             "db_players_not_on_a_2023_2025_squad_file": [
                 {"player_id": p, "last_squad_season_file": last_squad[p], "history_from_file": source.get(p)}
                 for p in outside]}
    return pd.DataFrame(rows), source, facts, leagues


def league_coverage(entries: dict[str, pd.DataFrame], tf: dict[str, dict[int, str]],
                    db_players: set[int]) -> pd.DataFrame:
    """Squad players by the domestic league their club played that season
    (from the snapshot's own games), with transfer-file and history coverage.
    National-team squads form one row per season."""
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    club_league = con.sql("""
        select distinct season, x.club_id, competition_id
        from games, lateral (select unnest([home_club_id, away_club_id]) club_id) x
        where competition_type = 'domestic_league'""").df()
    con.close()
    rows = []
    for s, e in entries.items():
        cl = club_league[club_league.season.astype(str) == s].drop_duplicates("club_id")
        e = e.merge(cl[["club_id", "competition_id"]], how="left", left_on="squad_id", right_on="club_id")
        e["league"] = e.competition_id.where(e.squad_type == "club", "national_team").fillna("(club not in a covered league's games)")
        this = tf.get(s, {})
        for lg, g in e.groupby("league"):
            ids = set(g.player_id)
            rows.append({"squad_season": s, "league": lg, "squad_players": len(ids),
                         "in_this_seasons_transfers_json": len(ids & set(this)),
                         "with_transfer_history": len(ids & db_players),
                         "pct_in_this_seasons_transfers_json": round(100 * len(ids & set(this)) / len(ids), 1),
                         "pct_with_transfer_history": round(100 * len(ids & db_players) / len(ids), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", action="store_true", help="also read upstream raw files (network, ~380 MB)")
    a = ap.parse_args(argv)
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    geo = club_geography()
    u = build_universe(load_canonical(), geo)
    res = {"tables": upstream_tables(con), "capture": source_file_capture(u.rows),
           "clubs": club_country_counts(u, geo, con), "pre_2012": pre_2012_players(u, con)}
    source = None
    if a.upstream:
        squads, source, facts, leagues = upstream_audit(set(u.rows.player_id))
        squads.to_csv(OUT / "dataset_provenance_squad_coverage.csv", index=False)
        leagues.to_csv(OUT / "dataset_provenance_league_coverage.csv", index=False)
        (OUT / "dataset_provenance_upstream_facts.json").write_text(json.dumps(facts, indent=2, default=str))
        res["squads"], res["upstream_facts"] = squads, facts
    res["seasons"] = season_table(u, geo, source)
    res["seasons"].to_csv(OUT / "dataset_provenance_season_coverage.csv", index=False)
    for k, v in res.items():
        print(f"\n== {k}")
        if isinstance(v, pd.DataFrame):
            print(v.to_string(index=False))
        else:
            for kk, vv in (v.items() if isinstance(v, dict) else [(k, v)]):
                print(f"  {kk}: {vv.to_string(index=False) if isinstance(vv, pd.DataFrame) else vv}")
    return res


if __name__ == "__main__":
    main()
