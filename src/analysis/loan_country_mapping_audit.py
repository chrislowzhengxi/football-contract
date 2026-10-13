"""Country-mapping coverage audit: why loan clubs are unmapped, and what could recover them without guessing.

    python -m src.analysis.loan_country_mapping_audit [--refresh-external]

Diagnostic only. Nothing here changes Stage 1C, `club_geography`, the calendar
rule or its classifications; every recovery is a *proposal* written to its own
table. Club names are used to describe the missing clubs (youth, reserve,
renamed) and, at most, to propose a parent-club link that also needs a recorded
move between the two clubs; a country is never read off a name.

The one external test reads (a) the upstream project's own raw transfer files,
from the same public bucket the DuckDB snapshot comes from, and (b) Wikidata,
for the audit sample's clubs only. Both results are cached in
`loan_country_mapping_external_probe.json`; a rerun makes no web request unless
`--refresh-external` is passed.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from . import loan_scope
from .daniel_scope_final import f0, md_table, pc
from .loan_calendar_rule import N_MIN, _json, apply, learn, population, segment_cross_border
from .loan_country_dates import context
from .loan_episodes import ROOT, club_geography
from .stage1_scope_report import display_country as dc
from ..stage1c.policy import PLACEHOLDER_CLUBS, YOUTH_OR_RESERVE, normalize_club_name

OUT = ROOT / "data" / "outputs" / "rebuild"
AUDIT_MD = OUT / "loan_country_mapping_audit.md"
CLUBS_CSV = OUT / "loan_country_mapping_unmapped_clubs.csv"
SAMPLE_CSV = OUT / "loan_country_mapping_sample_audit.csv"
CANDIDATES_CSV = OUT / "loan_country_mapping_candidate_recoveries.csv"
COVERAGE_CSV = OUT / "loan_country_mapping_coverage_estimate.csv"
PROBE_JSON = OUT / "loan_country_mapping_external_probe.json"
NUMBERS_JSON = OUT / "loan_country_mapping_numbers.json"
WRITES = (AUDIT_MD, CLUBS_CSV, SAMPLE_CSV, CANDIDATES_CSV, COVERAGE_CSV, PROBE_JSON, NUMBERS_JSON)

UNMAPPED = "one or both clubs unmapped"

# Diagnostic split of Stage 1C's own academy/reserve pattern into its two halves. Used to DESCRIBE clubs only.
YOUTH = re.compile(r"(?:\bU\s?1[0-9]\b|\bU\s?2[0-3]\b|Yth\.?|Youth|Jgd\.?|Sub-\d|Academy)")
RESERVE = re.compile(r"(?:Castilla|Res\.|\bII$|\bIII$|\bB$|\bC$)")
CATEGORIES = ("placeholder", "youth / development side", "reserve / B side", "senior club, no youth or reserve marker")

PLAYERS_PER_CLUB = 3          # raw records fetched per audit club
FLAG = re.compile(r"/flagge/[^/]+/(\d+)\.png")
WIKIDATA_API = "https://www.wikidata.org/w/api.php"   # Action API; the SPARQL service was rate-limited

# Audit sample: (stratum, club filter, ranking column, clubs). Ranked by the endings the club blocks, then club id.
SAMPLE_PLAN = (
    ("senior, missing as borrower", lambda C: C.primary_category.eq(CATEGORIES[3]), "as_borrower", 5),
    ("senior, missing as lender", lambda C: C.primary_category.eq(CATEGORIES[3]), "as_lender", 5),
    ("senior, in endings with both clubs missing", lambda C: C.primary_category.eq(CATEGORIES[3]), "in_both_missing", 3),
    ("reserve / B side", lambda C: C.primary_category.eq(CATEGORIES[2]), "episodes_affected", 5),
    ("youth / development side", lambda C: C.primary_category.eq(CATEGORIES[1]), "episodes_affected", 4),
    ("several names under one id", lambda C: C.n_names.gt(1), "episodes_affected", 2),
    ("in the snapshot clubs table, league code without a country", lambda C: C.in_snapshot_clubs_table,
     "episodes_affected", 2),
    ("senior, exact normalised name shared with a mapped club",
     lambda C: C.primary_category.eq(CATEGORIES[3]) & C.same_name_mapped_ids.ne(""), "episodes_affected", 2),
)


# ---------------------------------------------------------------------------
# Club names and the unmapped population
# ---------------------------------------------------------------------------

def club_names(rows: pd.DataFrame) -> pd.DataFrame:
    """Every (club_id, name) pair Stage 1C records, with its row count and first/last transfer date."""
    parts = [rows[[f"{s}_club_id", f"{s}_club_name", "transfer_date"]].set_axis(["club_id", "name", "date"], axis=1)
             for s in ("from", "to")]
    s = pd.concat(parts).dropna(subset=["club_id", "name"])
    s["club_id"] = s.club_id.astype("int64")
    s["date"] = pd.to_datetime(s.date, errors="coerce")
    return s.groupby(["club_id", "name"]).agg(rows=("date", "size"), first=("date", "min"),
                                              last=("date", "max")).reset_index()


def category(name: str) -> str:
    if name in PLACEHOLDER_CLUBS:
        return CATEGORIES[0]
    if YOUTH.search(name):
        return CATEGORIES[1]
    if RESERVE.search(name):
        return CATEGORIES[2]
    return CATEGORIES[3]


def slots(R: pd.DataFrame) -> pd.DataFrame:
    """One row per (unmapped ending, unmapped side)."""
    U = R[R.context.eq(UNMAPPED)]
    both = U.lender_country.isna() & U.borrower_country.isna()
    out = []
    for side in ("lender", "borrower"):
        m = U[f"{side}_country"].isna()
        out.append(pd.DataFrame({"loan_event_id": U.loan_event_id[m], "club_id": U[f"{side}_club_id"][m].astype("int64"),
                                 "club": U[f"{side}_club"][m], "side": side, "both_missing": both[m]}))
    return pd.concat(out, ignore_index=True)


def missing_sides(R: pd.DataFrame) -> dict:
    U = R[R.context.eq(UNMAPPED)]
    return {"lender only missing": int((U.lender_country.isna() & U.borrower_country.notna()).sum()),
            "borrower only missing": int((U.lender_country.notna() & U.borrower_country.isna()).sum()),
            "both missing": int((U.lender_country.isna() & U.borrower_country.isna()).sum())}


# ---------------------------------------------------------------------------
# Internal evidence: the upstream snapshot tables, searched by club id
# ---------------------------------------------------------------------------

def upstream_evidence(ids: set[int], db_path=None) -> pd.DataFrame:
    """Where each club id appears in the snapshot, and whether that appearance carries a country."""
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    con.register("ids", pd.DataFrame({"club_id": sorted(ids)}))
    try:
        E = pd.DataFrame({"club_id": sorted(ids)})
        q = {
            "clubs": """select cast(c.club_id as bigint) club_id, c.domestic_competition_id snapshot_competition,
                               k.country_name snapshot_competition_country
                        from clubs c left join competitions k on c.domestic_competition_id = k.competition_id
                        where cast(c.club_id as bigint) in (select club_id from ids)""",
            "games": """with gc as (select home_club_id club_id, competition_id, game_id from games
                                    union all select away_club_id, competition_id, game_id from games)
                        select cast(gc.club_id as bigint) club_id,
                               string_agg(distinct gc.competition_id, ',' order by gc.competition_id) game_competitions,
                               count(distinct gc.game_id) n_games,
                               string_agg(distinct k.country_name, ',') game_competition_countries
                        from gc left join competitions k using (competition_id)
                        where cast(gc.club_id as bigint) in (select club_id from ids) group by 1""",
            "appearances": """select cast(player_club_id as bigint) club_id,
                                     string_agg(distinct competition_id, ',' order by competition_id) appearance_competitions
                              from appearances where cast(player_club_id as bigint) in (select club_id from ids)
                              group by 1""",
            "players": """select cast(current_club_id as bigint) club_id, count(*) players_current_club_rows,
                                 string_agg(distinct current_club_domestic_competition_id, ',') players_competition
                          from players where cast(current_club_id as bigint) in (select club_id from ids) group by 1""",
            "valuations": """select cast(current_club_id as bigint) club_id, count(*) valuation_rows,
                                    string_agg(distinct player_club_domestic_competition_id, ',') valuations_competition
                             from player_valuations where cast(current_club_id as bigint) in (select club_id from ids)
                             group by 1""",
            "transfers": """with t as (select from_club_id club_id from transfers union all select to_club_id from transfers)
                            select cast(club_id as bigint) club_id, count(*) snapshot_transfer_rows from t
                            where cast(club_id as bigint) in (select club_id from ids) group by 1""",
        }
        for sql in q.values():
            E = E.merge(con.sql(sql).df(), on="club_id", how="left")
        comps = set(con.sql("select competition_id from competitions where country_name is not null").df().competition_id)
    finally:
        con.close()
    E["in_snapshot_clubs_table"] = E.snapshot_competition.notna()

    def resolvable(codes):
        return "" if pd.isna(codes) else ",".join(c for c in str(codes).split(",") if c in comps)

    for col in ("players_competition", "valuations_competition"):
        E[f"{col}_with_country"] = E[col].map(resolvable)
    E["internal_country_evidence"] = (E.snapshot_competition_country.notna() | E.game_competition_countries.notna()
                                      | E.players_competition_with_country.ne("")
                                      | E.valuations_competition_with_country.ne(""))
    return E


def ambiguous_clubs(db_path=None) -> set[int]:
    """Clubs `club_geography` drops because its two sources place them in two countries."""
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    try:
        d = con.sql("""
            with t1 as (select cast(c.club_id as bigint) club_id
                        from clubs c join competitions k on c.domestic_competition_id = k.competition_id
                        where k.country_name is not null),
                 t2 as (select distinct cast(x.club_id as bigint) club_id, k.country_name country
                        from games g join competitions k using (competition_id),
                             lateral (select unnest([g.home_club_id, g.away_club_id]) club_id) x
                        where k.country_name is not null)
            select club_id from t2 where club_id not in (select club_id from t1)
            group by 1 having count(distinct country) > 1""").df()
    finally:
        con.close()
    return set(d.club_id.astype("int64"))


def competition_status(db_path=None) -> dict[str, str]:
    """Every competition code in `games` -> what the snapshot says about its country."""
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    try:
        d = con.sql("""select g.competition_id code, max(k.competition_id) is not null listed, max(k.country_name) country
                       from (select distinct competition_id from games) g
                       left join competitions k using (competition_id) group by 1""").df()
    finally:
        con.close()
    return {r.code: (r.country if isinstance(r.country, str) else
                     "international, no country" if r.listed else "no row in competitions")
            for r in d.itertuples()}


def orphan_competitions(geo: pd.DataFrame, db_path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Competitions played in `games` but missing from `competitions`, which `club_geography` cannot see.

    A code gets a country only if every one of its participants that is already mapped is in that one country
    (competition-country membership). Mixed participants (a Club World Cup) or none mapped give nothing.
    Returns (one row per code, one row per unmapped participant with the country its codes imply).
    """
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    try:
        M = con.sql("""with gc as (select home_club_id club_id, competition_id from games
                                   union all select away_club_id, competition_id from games)
                       select competition_id code, cast(club_id as bigint) club_id, count(*) games from gc
                       where competition_id not in (select competition_id from competitions) group by 1, 2""").df()
    finally:
        con.close()
    g = geo.set_index("club_id").country
    M["country"] = M.club_id.map(g)
    K = M.groupby("code").agg(participants=("club_id", "nunique"), mapped=("country", "count"),
                              unmapped=("country", lambda s: int(s.isna().sum())),
                              countries=("country", lambda s: "|".join(sorted(s.dropna().unique())))).reset_index()
    K["membership_country"] = K.countries.where(K.countries.ne("") & ~K.countries.str.contains(r"\|"), "")
    U = M[M.country.isna()].merge(K[["code", "membership_country", "mapped"]], on="code")
    per = U.groupby("club_id").agg(orphan_competitions=("code", lambda s: ",".join(sorted(s))),
                                   implied=("membership_country", lambda s: sorted(set(s) - {""}))).reset_index()
    per["membership_country"] = per.implied.map(lambda v: v[0] if len(v) == 1 else "")
    per["membership_conflict"] = per.implied.map(len).gt(1)
    return K, per.drop(columns="implied")


def geography_sources(geo: pd.DataFrame) -> dict:
    return {k: int(v) for k, v in geo.geo_source.value_counts().items()}


def snapshot_leagues(db_path=None) -> dict:
    """Domestic leagues with a country in the snapshot, and the clubs-table league codes that have none."""
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    try:
        lg = con.sql("select count(*) from competitions where type = 'domestic_league' and country_name is not null")
        orphan = con.sql("""select domestic_competition_id code, count(*) n from clubs
                            where domestic_competition_id not in (select competition_id from competitions)
                            group by 1 order by 2 desc""").df()
        return {"leagues": int(lg.fetchone()[0]), "orphan_codes": dict(zip(orphan.code, orphan.n.astype(int)))}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Name links (exact normalised names only) and their validation
# ---------------------------------------------------------------------------

def name_links(names: pd.DataFrame, geo: pd.DataFrame, rows: pd.DataFrame, targets: set[int]) -> pd.DataFrame:
    """For each target club: mapped clubs sharing an exact normalised name, their countries, and direct moves.

    `normalize_club_name` is Stage 1C's own rule (strip an academy/reserve suffix, casefold). No fuzzy match.
    """
    g = geo.set_index("club_id").country
    n = names.assign(org=names.name.map(normalize_club_name))
    n = n[n.org.ne("")]
    mapped = n[n.club_id.isin(g.index)][["org", "club_id"]].drop_duplicates()
    by_org = mapped.groupby("org").club_id.apply(lambda s: sorted(set(s)))
    pair = pd.concat([rows[["from_club_id", "to_club_id"]].set_axis(["a", "b"], axis=1),
                      rows[["to_club_id", "from_club_id"]].set_axis(["a", "b"], axis=1)]).dropna().astype("int64")
    moves = pair.value_counts()
    out = []
    for cid, grp in n[n.club_id.isin(targets)].groupby("club_id"):
        ids = sorted({i for o in grp.org.unique() if o in by_org.index for i in by_org[o] if i != cid})
        countries = sorted({g[i] for i in ids})
        direct = {i: int(moves.get((cid, i), 0)) for i in ids}
        out.append({"club_id": cid, "has_marker": bool(grp.name.map(lambda s: bool(YOUTH_OR_RESERVE.search(s))).any()),
                    "same_name_mapped_ids": "|".join(map(str, ids)),
                    "same_name_mapped_countries": "|".join(countries),
                    "n_same_name_countries": len(countries),
                    "direct_moves_with_same_name_club": sum(direct.values()),
                    "linked_club_ids": "|".join(str(i) for i in ids if direct[i] > 0)})
    L = pd.DataFrame(out, columns=["club_id", "has_marker", "same_name_mapped_ids", "same_name_mapped_countries",
                                   "n_same_name_countries", "direct_moves_with_same_name_club", "linked_club_ids"])
    L["name_rule"] = np.select(
        [L.n_same_name_countries.eq(0), L.n_same_name_countries.gt(1),
         L.has_marker & L.direct_moves_with_same_name_club.gt(0), L.has_marker],
        ["no mapped club shares the name", "name shared across countries (ambiguous)",
         "parent link: marker + same name + recorded move", "parent link without a recorded move"],
        default="senior exact-name identity")
    L["name_country"] = L.same_name_mapped_countries.where(L.n_same_name_countries.eq(1), "")
    return L


PROPOSED_RULE = "parent link: marker + same name + recorded move"
MEMBERSHIP_RULE = "competition membership: code missing from competitions, mapped participants in one country"
REVIEW_RULES = ("parent link without a recorded move", "senior exact-name identity")


def validate_name_rules(names: pd.DataFrame, geo: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out on already mapped clubs: would the name rule have returned their known country?"""
    g = geo.set_index("club_id").country
    L = name_links(names, geo, rows, set(g.index))
    L = L[L.name_country.ne("")].assign(true_country=lambda d: d.club_id.map(g))
    return L.assign(agrees=L.name_country.eq(L.true_country))


# ---------------------------------------------------------------------------
# Audit sample
# ---------------------------------------------------------------------------

def select_sample(C: pd.DataFrame) -> pd.DataFrame:
    chosen, out = set(), []
    for stratum, f, col, k in SAMPLE_PLAN:
        pool = C[f(C) & ~C.club_id.isin(chosen) & C[col].gt(0)].sort_values([col, "club_id"], ascending=[False, True])
        for _, r in pool.head(k).iterrows():
            chosen.add(r.club_id)
            out.append({"club_id": r.club_id, "stratum": stratum})
    return pd.DataFrame(out)


def internal_verdict(r, status: dict[str, str]) -> tuple[str, str, str, str]:
    """(evidence_source, recoverable yes/no, proposed country, reason) from repo data only."""
    ev = []
    if r.in_snapshot_clubs_table:
        ev.append(f"snapshot clubs table: league {r.snapshot_competition}"
                  + ("" if pd.notna(r.snapshot_competition_country) else " (no such competition row, so no country)"))
    if pd.notna(r.game_competitions):
        ev.append("games: " + ", ".join(f"{c} ({status.get(c, '?')})" for c in r.game_competitions.split(",")))
    if pd.notna(r.appearance_competitions):
        ev.append(f"appearances: {r.appearance_competitions}")
    if pd.notna(r.players_competition) and r.players_competition:
        ev.append(f"players.current_club league {r.players_competition}")
    if pd.notna(r.valuations_competition) and r.valuations_competition:
        ev.append(f"valuations league {r.valuations_competition}")
    if r.same_name_mapped_ids:
        ev.append(f"Stage 1C name shared with mapped club(s) {r.same_name_mapped_ids} "
                  f"[{r.same_name_mapped_countries}], {r.direct_moves_with_same_name_club} direct move(s)")
    ev.append(f"Stage 1C: {r.stage1c_rows} rows, {r.n_names} name(s)")
    if r.internal_country_evidence:
        return "; ".join(ev), "yes", "", "snapshot metadata carries a country"
    if r.membership_country:
        return ("; ".join(ev), "yes", r.membership_country,
                f"competition membership: plays {r.orphan_competitions}, which has no row in competitions, and every "
                f"mapped club in it is in {r.membership_country}")
    if r.name_rule == PROPOSED_RULE:
        return ("; ".join(ev), "yes", r.name_country,
                "parent-club link: Stage 1C's own academy/reserve normalisation gives one mapped club, one country, "
                "and the two ids exchange players directly")
    why = {"no mapped club shares the name": "no snapshot table gives a country and no mapped club shares the name",
           "name shared across countries (ambiguous)": "same normalised name exists in more than one country",
           "parent link without a recorded move": "name matches one mapped club but the two ids never exchange a player",
           "senior exact-name identity": "senior homonym only; validation shows exact senior names cross countries"}
    reason = why[r.name_rule]
    if r.in_snapshot_clubs_table and pd.isna(r.snapshot_competition_country):
        reason = f"league code {r.snapshot_competition} has no row in the snapshot's competitions table; " + reason
    return "; ".join(ev), "no", "", reason


# ---------------------------------------------------------------------------
# External probe (audit sample only, cached)
# ---------------------------------------------------------------------------

def probe_players(sample: pd.DataFrame, S: pd.DataFrame) -> dict[int, int]:
    """Up to PLAYERS_PER_CLUB players per audit club (lowest player ids among its unmapped endings)."""
    pid = S.merge(sample[["club_id"]], on="club_id")
    out = {}
    for cid, g in pid.groupby("club_id"):
        for p in sorted(g.player_id.unique())[:PLAYERS_PER_CLUB]:
            out.setdefault(int(p), int(cid))
    return out


def fetch_raw_flags(sample_ids: list[int], players: dict[int, int]) -> dict:
    """Stream the upstream raw transfer files for the audit players only; keep each club's country flag id."""
    from ..stage1b.raw_api import DVC_HTTP_REMOTE, SNAPSHOT_DIR_MD5, collect_players, snapshot_manifest, \
        transfers_file_md5
    from ..stage1b.parse import club_id_from_href

    responses, origin = collect_players(set(players))
    flags: dict[int, dict] = {}
    for p, resp in responses.items():
        for t in resp.get("transfers") or []:
            for side in ("from", "to"):
                c = t.get(side) or {}
                cid = club_id_from_href(c.get("href") or "")
                if cid is None:
                    continue
                m = FLAG.search(c.get("countryFlag") or "")
                rec = flags.setdefault(int(cid), {"names": [], "flag_ids": [], "rows": 0, "special": False})
                rec["rows"] += 1
                rec["special"] |= bool(c.get("isSpecial"))
                if c.get("clubName") and c["clubName"] not in rec["names"]:
                    rec["names"].append(c["clubName"])
                fid = int(m.group(1)) if m else None
                if fid not in rec["flag_ids"]:
                    rec["flag_ids"].append(fid)
    manifest = snapshot_manifest()
    return {"fetched_on": dt.date.today().isoformat(),
            "raw_source": {"bucket": DVC_HTTP_REMOTE, "snapshot_dir_md5": SNAPSHOT_DIR_MD5,
                           "season_files": {s: transfers_file_md5(manifest, s) for s in sorted(set(origin.values()))}},
            "sample_club_ids": [int(i) for i in sample_ids],
            "players": {str(p): {"for_club": c, "season_file": origin.get(p)} for p, c in players.items()},
            "players_found": len(responses),
            "club_flags": {str(k): v for k, v in sorted(flags.items())}}


def _wikidata_get(params: dict) -> dict:
    from ..stage1b.raw_api import USER_AGENT

    url = f"{WIKIDATA_API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
        return json.load(r)


def fetch_wikidata(sample_ids: list[int]) -> dict:
    """Three Wikidata Action API requests for the audit clubs: find items whose P7223 (Transfermarkt team id) is
    one of ours, read their P7223 and P17 (country) claims, then the country labels. Exact id match only."""
    import urllib.error

    search = "haswbstatement:" + "|".join(f"P7223={i}" for i in sample_ids)
    try:
        hits = _wikidata_get({"action": "query", "list": "search", "srsearch": search, "srlimit": "50",
                              "srnamespace": "0"})["query"]["search"]
        qids = [h["title"] for h in hits]
        ents = _wikidata_get({"action": "wbgetentities", "ids": "|".join(qids), "props": "claims"})["entities"] \
            if qids else {}

        def values(e, prop):
            return [c["mainsnak"].get("datavalue", {}).get("value") for c in e.get("claims", {}).get(prop, [])]

        countries = sorted({v["id"] for e in ents.values() for v in values(e, "P17") if v})
        labels = _wikidata_get({"action": "wbgetentities", "ids": "|".join(countries), "props": "labels",
                                "languages": "en"})["entities"] if countries else {}
        wiki: dict[str, list] = {}
        wanted = {str(i) for i in sample_ids}
        for q, e in ents.items():
            cs = [labels.get(v["id"], {}).get("labels", {}).get("en", {}).get("value", v["id"])
                  for v in values(e, "P17") if v] or [None]
            for tm in {str(v) for v in values(e, "P7223") if v} & wanted:
                for c in cs:
                    wiki.setdefault(tm, []).append({"item": q, "country": c})
        return {"wikidata_method": "Action API: list=search haswbstatement P7223, wbgetentities P7223/P17",
                "wikidata_search": search, "wikidata": wiki, "wikidata_error": None}
    except (urllib.error.URLError, KeyError, TimeoutError) as e:
        return {"wikidata_search": search, "wikidata": {}, "wikidata_error": f"{type(e).__name__}: {e}"}


def external_probe(sample_ids: list[int], players: dict[int, int], refresh: bool = False) -> dict:
    """Cached in PROBE_JSON. The raw-flag part is saved before Wikidata is asked, so a Wikidata failure
    never forces the raw files to be streamed again."""
    P = json.loads(PROBE_JSON.read_text()) if PROBE_JSON.exists() and not refresh else None
    if P is not None and sorted(P["sample_club_ids"]) != sorted(int(i) for i in sample_ids):
        raise RuntimeError("cached probe was made for a different audit sample; rerun with --refresh-external")
    if P is None:
        P = fetch_raw_flags(sample_ids, players)
        PROBE_JSON.write_text(json.dumps(P, indent=1))
    if refresh or "wikidata" not in P or P.get("wikidata_error"):
        P.update(fetch_wikidata(sample_ids))
        PROBE_JSON.write_text(json.dumps(P, indent=1))
    return P


def flag_countries(db_path=None) -> dict[int, str]:
    """Transfermarkt country id -> country name, from the snapshot's own `countries` table."""
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    try:
        d = con.sql("select cast(country_id as integer) id, country_name from countries").df()
    finally:
        con.close()
    return dict(zip(d.id, d.country_name))


def flag_table(P: dict, fc: dict[int, str]) -> pd.DataFrame:
    rows = []
    for cid, v in P["club_flags"].items():
        ids = [i for i in v["flag_ids"] if i is not None]
        one = len(set(ids)) == 1
        rows.append({"club_id": int(cid), "flag_names": " | ".join(v["names"]), "flag_ids": "|".join(map(str, ids)),
                     "flag_country": (fc.get(ids[0], f"Transfermarkt country {ids[0]}") if one else ""),
                     "flag_id_in_snapshot": one and ids[0] in fc, "flag_rows": v["rows"], "special": v["special"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Projection with a proposed mapping (calendars fixed, never re-learned)
# ---------------------------------------------------------------------------

def project(R: pd.DataFrame, cal: dict, extra: pd.Series | None = None) -> pd.DataFrame:
    """Fill ONLY missing club countries from `extra`, re-derive context and apply the existing calendars."""
    P = R[["loan_event_id", "lender_club_id", "borrower_club_id", "lender_country", "borrower_country",
           "lender_confederation", "borrower_confederation", "doy", "covid"]].copy()
    if extra is not None and len(extra):
        P["lender_country"] = P.lender_country.fillna(P.lender_club_id.map(extra))
        P["borrower_country"] = P.borrower_country.fillna(P.borrower_club_id.map(extra))
    P["context"] = context(P)
    P["calendar_key"] = np.select([P.context.eq("domestic"), P.context.eq("cross-border")],
                                  ["domestic: " + P.lender_country.fillna(""), segment_cross_border(P)], default="")
    return P.join(apply(P, cal))


def scenario(name: str, R: pd.DataFrame, cal: dict, extra: pd.Series | None, base: pd.DataFrame, note: str) -> dict:
    P = project(R, cal, extra)
    cls = P.calendar_class.ne("geography_unclassified")
    newly = cls & base.calendar_class.eq("geography_unclassified")
    filled = (P.lender_country.notna() & R.lender_country.isna()) | (P.borrower_country.notna() & R.borrower_country.isna())
    was_um = R.context.eq(UNMAPPED)
    return {"scenario": name, "clubs_added": 0 if extra is None else int(len(extra)),
            "endings_with_a_side_filled": int(filled.sum()),
            "endings_now_fully_mapped": int((was_um & P.context.ne(UNMAPPED)).sum()),
            "still_unmapped": int(P.context.eq(UNMAPPED).sum()),
            "newly_classifiable": int(newly.sum()), "classifiable": int(cls.sum()), "realised": len(P),
            "classifiable_share": float(cls.mean()),
            "below_history_threshold": int(P.unclassified_reason.eq("country below the history threshold").sum()),
            "note": note}


def bounds_after_full_recovery(R: pd.DataFrame, cal: dict) -> dict:
    """If every unmapped club got a country: which endings are classifiable whatever that country is?

    One side known with country K: the ending becomes domestic-K or cross-border. If K has a calendar, both
    outcomes are classifiable. Everything else depends on the recovered countries.
    """
    U = R[R.context.eq(UNMAPPED)]
    known = U.lender_country.fillna(U.borrower_country)
    one_side = known.notna()
    has_cal = known.map(lambda k: f"domestic: {k}" in cal if pd.notna(k) else False)
    sure = int((one_side & has_cal).sum())
    classified_now = int(R.context.eq("cross-border").sum()
                         + R[R.context.eq("domestic")].calendar_key.isin(cal.keys()).sum())
    return {"guaranteed": sure, "depends_on_country": int(len(U) - sure),
            "one_side_known_no_calendar": int((one_side & ~has_cal).sum()), "both_missing": int((~one_side).sum()),
            "classified_now": classified_now, "lower": classified_now + sure, "upper": classified_now + len(U)}


# ---------------------------------------------------------------------------
# Selection bias: mapped vs unmapped endings
# ---------------------------------------------------------------------------

BIAS_GROUPS = ("both clubs mapped", "lender only missing", "borrower only missing", "both missing")
PERIODS = ((0, 2010, "loan season starts before 2010"), (2010, 2015, "2010–2014"), (2015, 2020, "2015–2019"),
           (2020, 9999, "2020 or later"))


def bias(R: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    L = loans.set_index("loan_event_id")
    B = R.copy()
    B["group"] = np.select([B.context.ne(UNMAPPED), B.lender_country.isna() & B.borrower_country.notna(),
                            B.lender_country.notna() & B.borrower_country.isna()], BIAS_GROUPS[:3],
                           default=BIAS_GROUPS[3])
    B["start_year"] = B.loan_season.str[:4].astype(int)
    B["fee_bearing"] = B.loan_event_id.map(L.outbound_loan_fee_eur).fillna(0).gt(0)
    B["age"] = B.loan_event_id.map(L.age_at_transfer)
    B["mv"] = B.loan_event_id.map(L.market_value_eur)
    B["lender_marker"] = B.lender_club.fillna("").map(lambda s: bool(YOUTH_OR_RESERVE.search(s)))
    B["borrower_marker"] = B.borrower_club.fillna("").map(lambda s: bool(YOUTH_OR_RESERVE.search(s)))
    out = []
    for gname in BIAS_GROUPS + ("any club missing",):
        g = B[B.group.eq(gname)] if gname != "any club missing" else B[B.group.ne(BIAS_GROUPS[0])]
        rec = {"group": gname, "endings": len(g), "median loan start year": float(g.start_year.median())}
        for lo, hi, lab in PERIODS:
            rec[lab] = float(g.start_year.between(lo, hi - 1).mean())
        rec.update({"fee-bearing loan": float(g.fee_bearing.mean()),
                    "median duration (days)": float(g.loan_duration_days.median()),
                    "median age at loan": float(g.age.median()),
                    "market value recorded": float(g.mv.notna().mean()),
                    "median market value (EUR)": float(g.mv.median()),
                    "lender name has youth/reserve marker": float(g.lender_marker.mean()),
                    "borrower name has youth/reserve marker": float(g.borrower_marker.mean()),
                    "ends 30 June": float(g.ending_mmdd.eq("06-30").mean()),
                    "ends 31 May": float(g.ending_mmdd.eq("05-31").mean()),
                    "ends 31 December": float(g.ending_mmdd.eq("12-31").mean())})
        nk = g.next_move_kind.fillna("none recorded")
        for k in ("permanent_to_borrower", "loan", "sale_to_third_club", "free_move_to_third_club", "none recorded"):
            rec[f"next move: {k}"] = float(nk.eq(k).mean())
        out.append(rec)
    return pd.DataFrame(out)


def known_side(R: pd.DataFrame, cal: dict) -> dict:
    U = R[R.context.eq(UNMAPPED)]
    res = {}
    for lab, m, col in (("borrower missing, lender known", U.lender_country.notna(), "lender_country"),
                        ("lender missing, borrower known", U.borrower_country.notna(), "borrower_country")):
        k = U.loc[m, col]
        vc = k.value_counts()
        res[lab] = {"n": int(m.sum()), "top": {c: int(v) for c, v in vc.head(8).items()},
                    "with_calendar": int(k.map(lambda c: f"domestic: {c}" in cal).sum()),
                    "countries": int(vc.size)}
    return res


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(u=None, refresh: bool = False) -> tuple[dict, dict]:
    u = u or loan_scope.universe()
    R = population(u)
    geo = club_geography()
    cal, _ = learn(R[~R.covid])
    base = project(R, cal)
    N: dict = {"realised": len(R), "context": {k: int(v) for k, v in R.context.value_counts().items()}}
    N["sides"] = missing_sides(R)
    N["below_threshold"] = int(base.unclassified_reason.eq("country below the history threshold").sum())
    N["unclassified"] = int(base.calendar_class.eq("geography_unclassified").sum())
    N["classifiable_now"] = len(R) - N["unclassified"]
    N["calendars"] = sorted(cal)
    N["geo_clubs"] = len(geo)
    N["geo_sources"] = geography_sources(geo)
    N["geo_countries"] = int(geo.country.nunique())
    N["snapshot"] = snapshot_leagues()
    N["code_status"] = competition_status()
    below = R[base.unclassified_reason.eq("country below the history threshold")]
    N["below_countries"] = {k: int(v) for k, v in below.lender_country.value_counts().items()}
    assert sum(N["sides"].values()) == N["context"][UNMAPPED]
    assert N["unclassified"] == N["context"][UNMAPPED] + N["below_threshold"]

    # ---- the unmapped clubs
    S = slots(R).merge(R[["loan_event_id", "player_id"]], on="loan_event_id")
    assert not S.duplicated(["loan_event_id", "club_id"]).any(), "a club is on both sides of one loan"
    names = club_names(u.rows)
    N["stage1c_clubs"] = int(names.club_id.nunique())
    ids = set(S.club_id)
    C = S.groupby("club_id").agg(
        club_name=("club", lambda s: s.value_counts().index[0]),
        episodes_affected=("loan_event_id", "nunique"),
        as_lender=("side", lambda s: int(s.eq("lender").sum())),
        as_borrower=("side", lambda s: int(s.eq("borrower").sum())),
        in_both_missing=("both_missing", "sum"),
        sole_unmapped_side=("both_missing", lambda s: int((~s).sum()))).reset_index()
    nm = names[names.club_id.isin(ids)]
    C = C.merge(nm.groupby("club_id").agg(n_names=("name", "nunique"),
                                          all_names=("name", lambda s: " | ".join(sorted(s))),
                                          stage1c_rows=("rows", "sum"), first_seen=("first", "min"),
                                          last_seen=("last", "max")).reset_index(), on="club_id", how="left")
    C["primary_category"] = C.club_name.map(category)
    E = upstream_evidence(ids)
    amb = ambiguous_clubs()
    C = C.merge(E, on="club_id", how="left")
    C["dropped_as_two_country"] = C.club_id.isin(amb)
    K, Mper = orphan_competitions(geo)
    C = C.merge(Mper, on="club_id", how="left")
    C["orphan_competitions"] = C.orphan_competitions.fillna("")
    C["membership_country"] = C.membership_country.fillna("")
    C["membership_conflict"] = C.membership_conflict.astype("boolean").fillna(False).astype(bool)
    N["orphan_codes"] = K.set_index("code").to_dict(orient="index")
    L = name_links(names, geo, u.rows, ids)
    C = C.merge(L, on="club_id", how="left")
    for col in ("same_name_mapped_ids", "same_name_mapped_countries", "linked_club_ids", "name_country"):
        C[col] = C[col].fillna("")
    C["name_rule"] = C.name_rule.fillna("no mapped club shares the name")
    C = C.sort_values(["episodes_affected", "club_id"], ascending=[False, True]).reset_index(drop=True)
    assert len(C) == len(ids) and C.episodes_affected.sum() == len(S)
    assert (C.as_lender + C.as_borrower).eq(C.episodes_affected).all()

    N["clubs"] = {"unique_ids": len(C), "unique_names": int(S.club.nunique()), "slots": len(S),
                  "endings_per_club": {lab: int(C.episodes_affected.between(lo, hi).sum()) for lab, lo, hi in
                                       (("1", 1, 1), ("2–4", 2, 4), ("5–9", 5, 9), ("10–19", 10, 19),
                                        ("20–49", 20, 49), ("50+", 50, 10 ** 9))},
                  "median_endings": float(C.episodes_affected.median()),
                  "top50_share_of_slots": float(C.episodes_affected.head(50).sum() / len(S)),
                  "top500_share_of_slots": float(C.episodes_affected.head(500).sum() / len(S))}
    cat = C.groupby("primary_category").agg(clubs=("club_id", "size"), slots=("episodes_affected", "sum"))
    N["categories"] = {k: {"clubs": int(cat.clubs.get(k, 0)), "slots": int(cat.slots.get(k, 0))} for k in CATEGORIES}
    sc = S.merge(C[["club_id", "primary_category"]], on="club_id")
    N["category_endings"] = {k: int(sc.loc[sc.primary_category.eq(k), "loan_event_id"].nunique()) for k in CATEGORIES}
    N["flags"] = {
        "several names under one id": int(C.n_names.gt(1).sum()),
        "shares exact normalised name with a mapped club": int(C.same_name_mapped_ids.ne("").sum()),
        "in snapshot clubs table": int(C.in_snapshot_clubs_table.sum()),
        "dropped as two-country": int(C.dropped_as_two_country.sum()),
        "in snapshot games": int(C.game_competitions.notna().sum()),
        "in snapshot appearances": int(C.appearance_competitions.notna().sum()),
        "as players' current club": int(C.players_current_club_rows.notna().sum()),
        "as valuations' current club": int(C.valuation_rows.notna().sum()),
        "in snapshot transfers table": int(C.snapshot_transfer_rows.notna().sum()),
        "any snapshot evidence with a country": int(C.internal_country_evidence.sum())}
    N["tier_counts"] = {"clubs table": int(C.snapshot_competition_country.notna().sum()),
                        "games": int(C.game_competition_countries.notna().sum()),
                        "other tables": int((C.players_competition_with_country.ne("")
                                             | C.valuations_competition_with_country.ne("")).sum())}

    def codes(col):
        return sorted(set(C[col].dropna().str.split(",").explode()) - {""})

    N["players_codes"], N["valuations_codes"] = codes("players_competition"), codes("valuations_competition")
    N["appearance_codes"] = codes("appearance_competitions")
    N["snapshot_club_codes"] = {k: int(v) for k, v in C.loc[C.in_snapshot_clubs_table, "snapshot_competition"]
                                .value_counts().items()}
    N["snapshot_club_endings"] = int(C.loc[C.in_snapshot_clubs_table, "episodes_affected"].sum())
    N["game_competitions"] = {k: int(v) for k, v in C.game_competitions.dropna().str.split(",").explode()
                              .value_counts().items()}
    N["last_seen_year"] = {"unmapped": float(C.last_seen.dt.year.median())}
    mapped_last = names[names.club_id.isin(geo.club_id)].groupby("club_id")["last"].max()
    N["last_seen_year"]["mapped"] = float(mapped_last.dt.year.median())
    N["last_seen_before_2015"] = int(C.last_seen.dt.year.lt(2015).sum())
    N["name_rules"] = {k: {"clubs": int(v.club_id.size), "slots": int(v.episodes_affected.sum())}
                       for k, v in C.groupby("name_rule")}
    N["parent_links_by_category"] = {k: int(v) for k, v in
                                     C[C.name_rule.eq(PROPOSED_RULE)].primary_category.value_counts().items()}

    V = validate_name_rules(names, geo, u.rows)
    N["validation_loo"] = {k: {"n": int(len(g)), "agree": int(g.agrees.sum()),
                               "disagree_examples": [f"{a} ({b} vs {c})" for a, b, c in
                                                     g.loc[~g.agrees, ["club_id", "name_country", "true_country"]]
                                                     .head(6).itertuples(index=False)]}
                           for k, g in V.groupby("name_rule")}
    N["mapped_youth_reserve_clubs"] = int(names[names.club_id.isin(geo.club_id)].groupby("club_id").name
                                          .apply(lambda s: s.map(lambda x: bool(YOUTH_OR_RESERVE.search(x))).any()).sum())

    # ---- audit sample (+ the cached external probe)
    smp = select_sample(C)
    N["sample_size"] = len(smp)
    assert 20 <= len(smp) <= 30
    players = probe_players(smp, S)
    P = external_probe(list(smp.club_id), players, refresh)
    fc = flag_countries()
    F = flag_table(P, fc)
    N["probe"] = {"players_requested": len(players), "players_found": P["players_found"],
                  "season_files": P["raw_source"]["season_files"], "fetched_on": P["fetched_on"],
                  "clubs_seen": len(F), "wikidata_rows": sum(len(v) for v in P["wikidata"].values()),
                  "wikidata_error": P.get("wikidata_error")}
    A = smp.merge(C, on="club_id", how="left")
    A = A.merge(F[["club_id", "flag_country", "flag_ids", "flag_rows", "flag_id_in_snapshot"]], on="club_id", how="left")
    A["wikidata_country"] = A.club_id.map(lambda c: " | ".join(sorted({w["country"] or "no P17" for w in
                                                                         P["wikidata"].get(str(c), [])})))
    A["wikidata_items"] = A.club_id.map(lambda c: " | ".join(w["item"] for w in P["wikidata"].get(str(c), [])))
    status = competition_status()
    v = A.apply(lambda r: internal_verdict(r, status), axis=1, result_type="expand")
    A[["evidence_source", "recoverable_from_repo_yes_no", "proposed_country_if_deterministically_supported",
       "reason"]] = v
    A["current_country"] = ""
    A["external_agrees_with_proposal"] = np.where(
        A.proposed_country_if_deterministically_supported.eq(""), "",
        np.where(A.flag_country.eq(A.proposed_country_if_deterministically_supported), "yes", "no"))
    A["missing_side_mix"] = A.apply(lambda r: f"lender {r.as_lender} / borrower {r.as_borrower} "
                                              f"/ both-missing {int(r.in_both_missing)}", axis=1)
    N["sample"] = {"recoverable_repo": int(A.recoverable_from_repo_yes_no.eq("yes").sum()),
                   "by_stratum": {k: int(v) for k, v in A.stratum.value_counts().items()},
                   "flag_found": int(A.flag_country.fillna("").ne("").sum()),
                   "flag_in_snapshot": int(A.flag_id_in_snapshot.fillna(False).astype(bool).sum()),
                   "flag_countries": {k: int(v) for k, v in A.flag_country.fillna("").replace("", "none")
                                      .value_counts().items()},
                   "wikidata_found": int(A.wikidata_items.ne("").sum()),
                   "wikidata_with_country": int(A.wikidata_country.ne("").sum()
                                                - A.wikidata_country.eq("no P17").sum()),
                   "proposals": int(A.proposed_country_if_deterministically_supported.ne("").sum()),
                   "proposal_agree": int(A.external_agrees_with_proposal.eq("yes").sum()),
                   "episodes": int(A.episodes_affected.sum())}
    wd_uk = A.wikidata_country.str.contains("United Kingdom")
    N["sample"]["wikidata_uk"] = int(wd_uk.sum())
    wd_cmp = A[A.wikidata_country.ne("") & A.wikidata_country.ne("no P17") & A.flag_country.fillna("").ne("")]
    N["sample"]["wikidata_vs_flag"] = {"n": len(wd_cmp), "agree": int((wd_cmp.wikidata_country == wd_cmp.flag_country)
                                                                      .sum()),
                                       "uk_vs_home_nation": int(wd_cmp.wikidata_country.eq("United Kingdom").sum())}

    # ---- external flag validated on every club seen in the fetched records
    g = geo.set_index("club_id").country
    Fm = F[F.club_id.isin(g.index) & F.flag_country.ne("")].assign(ours=lambda d: d.club_id.map(g))
    Fm["agree"] = Fm.flag_country.eq(Fm.ours)
    N["flag_validation_mapped"] = {"n": len(Fm), "agree": int(Fm.agree.sum()),
                                   "disagreements": [f"{r.flag_names} ({r.club_id}): flag {r.flag_country}, "
                                                     f"snapshot {r.ours}" for r in Fm[~Fm.agree].itertuples()]}
    Lall = name_links(names, geo, u.rows, set(F.club_id) - set(g.index))
    Fu = F.merge(Lall, on="club_id")
    Fu = Fu[Fu.name_country.ne("") & Fu.flag_country.ne("")]
    N["flag_validation_name_rules"] = {
        k: {"n": int(len(d)), "agree": int(d.name_country.eq(d.flag_country).sum()),
            "disagreements": [f"{r.flag_names} ({r.club_id}): name rule {r.name_country}, flag {r.flag_country}"
                              for r in d[d.name_country.ne(d.flag_country)].itertuples()][:8]}
        for k, d in Fu.groupby("name_rule")}
    N["flag_unmapped_seen"] = int(F.club_id.isin(ids).sum())
    N["flag_null"] = int(F.flag_ids.eq("").sum())
    N["flag_multi"] = int(F.flag_ids.str.contains(r"\|").sum())
    N["flag_null_special"] = int((F.flag_ids.eq("") & F.special).sum())
    N["flag_not_in_snapshot"] = int((F.flag_country.ne("") & ~F.flag_id_in_snapshot).sum())

    # ---- proposed recovered mapping (separate from production) and coverage
    memb = C[C.membership_country.ne("") & ~C.internal_country_evidence]
    parent = C[C.name_rule.eq(PROPOSED_RULE) & ~C.internal_country_evidence & ~C.club_id.isin(memb.club_id)]
    both_tiers = C[C.membership_country.ne("") & C.name_rule.eq(PROPOSED_RULE)]
    N["membership_vs_parent"] = {"both": len(both_tiers),
                                 "disagree": int(both_tiers.membership_country.ne(both_tiers.name_country).sum())}
    prop = pd.concat([memb.assign(proposed=memb.membership_country, tier=MEMBERSHIP_RULE),
                      parent.assign(proposed=parent.name_country, tier=PROPOSED_RULE)])
    assert prop.club_id.is_unique
    review_parent = C[C.name_rule.eq(REVIEW_RULES[0]) & ~C.club_id.isin(prop.club_id)]
    review_senior = C[C.name_rule.eq(REVIEW_RULES[1]) & ~C.club_id.isin(prop.club_id)]
    flag_sample = A[A.flag_country.fillna("").ne("") & A.flag_id_in_snapshot.fillna(False).astype(bool)]
    ext_map = pd.Series(flag_sample.flag_country.values, index=flag_sample.club_id.values)
    m_memb = pd.Series(memb.membership_country.values, index=memb.club_id.values)
    m_par = pd.Series(parent.name_country.values, index=parent.club_id.values)
    m_prop = pd.Series(prop.proposed.values, index=prop.club_id.values)
    m_rev = pd.Series(review_parent.name_country.values, index=review_parent.club_id.values)
    m_sen = pd.Series(review_senior.name_country.values, index=review_senior.club_id.values)
    code_clubs = C[C.snapshot_competition.isin(N["snapshot"]["orphan_codes"])]
    seen_codes = code_clubs.merge(F[["club_id", "flag_country"]], on="club_id")
    N["orphan_code_flags"] = {code: {"clubs": int(len(g)), "flag_seen": int(len(seen_codes[seen_codes.snapshot_competition
                                                                                          .eq(code)])),
                                     "flag_countries": {k: int(v) for k, v in seen_codes.loc[
                                         seen_codes.snapshot_competition.eq(code), "flag_country"].value_counts().items()},
                                     "slots": int(g.episodes_affected.sum())}
                              for code, g in code_clubs.groupby("snapshot_competition")}
    code_map = {code: next(iter(v["flag_countries"])) for code, v in N["orphan_code_flags"].items()
                if len(v["flag_countries"]) == 1}
    m_code = pd.Series(code_clubs.snapshot_competition.map(code_map).values, index=code_clubs.club_id.values).dropna()
    S_INT = "internal: both proposed tiers"
    sc_rows = [
        scenario("current mapping", R, cal, None, base, "production club_geography; reproduces the calendar rule"),
        scenario("internal: competition membership (proposed)", R, cal, m_memb, base,
                 "competition missing from `competitions`; every mapped participant in one country"),
        scenario("internal: parent links with a recorded move (proposed)", R, cal, m_par, base,
                 "Stage 1C academy/reserve normalisation -> one mapped club in one country, and a direct move"),
        scenario(S_INT, R, cal, m_prop, base, "the two proposed internal tiers together"),
        scenario("internal: + parent links without a recorded move (review only)", R, cal,
                 pd.concat([m_prop, m_rev]), base, "not recommended without external confirmation"),
        scenario("internal: + senior exact-name identity (review only)", R, cal,
                 pd.concat([m_prop, m_rev, m_sen]), base, "senior homonyms cross countries in validation"),
        scenario("external probe: raw countryFlag, audit clubs only", R, cal, ext_map, base,
                 f"{len(ext_map)} audit clubs; a test, not a full mapping"),
        scenario("external-confirmed league code: " + ", ".join(f"{k} -> {v}" for k, v in code_map.items()), R, cal,
                 m_code, base, "snapshot clubs-table league code read as the country every probed club's flag shows"),
        scenario("internal proposed + external probe on audit clubs", R, cal,
                 pd.concat([m_prop, ext_map[~ext_map.index.isin(m_prop.index)]]), base, "combination"),
    ]
    assert sc_rows[0]["classifiable"] == N["classifiable_now"] and sc_rows[0]["newly_classifiable"] == 0
    Bd = bounds_after_full_recovery(R, cal)
    assert Bd["classified_now"] == N["classifiable_now"]
    for lab, val, note in (("external, every unmapped club recovered: guaranteed lower bound", Bd["lower"],
                            "one side known in a country with a calendar; classifiable whatever the other country"),
                           ("external, every unmapped club recovered: upper bound", Bd["upper"],
                            "every recovered ending lands in a calendar")):
        sc_rows.append({"scenario": lab, "clubs_added": len(C), "endings_with_a_side_filled": N["context"][UNMAPPED],
                        "endings_now_fully_mapped": N["context"][UNMAPPED], "still_unmapped": 0,
                        "newly_classifiable": val - N["classifiable_now"], "classifiable": val, "realised": len(R),
                        "classifiable_share": val / len(R), "below_history_threshold": np.nan, "note": note})
    COV = pd.DataFrame(sc_rows)
    N["coverage"] = {r["scenario"]: {k: r[k] for k in ("clubs_added", "endings_with_a_side_filled",
                                                       "endings_now_fully_mapped", "newly_classifiable",
                                                       "classifiable", "classifiable_share", "still_unmapped")}
                     for r in sc_rows}
    N["code_scenario"] = sc_rows[7]["scenario"]
    N["bounds"] = Bd
    Pp = project(R, cal, m_prop)
    newdom = Pp[Pp.context.eq("domestic") & base.context.ne("domestic") & ~Pp.covid]
    N["prop_new_domestic_by_country"] = {k: int(v) for k, v in newdom.lender_country.value_counts().head(10).items()}
    N["prop_new_xb"] = int((Pp.context.eq("cross-border") & base.context.ne("cross-border")).sum())
    train_after = Pp[Pp.context.eq("domestic") & ~Pp.covid].lender_country.value_counts()
    N["countries_reaching_n_min"] = sorted(k for k, v in train_after.items() if v >= N_MIN
                                           and f"domestic: {k}" not in cal)
    Pc = project(R, cal, m_code)
    nocal = Pc[Pc.context.eq("domestic") & ~Pc.covid & Pc.calendar_class.eq("geography_unclassified")]
    N["code_domestic_training"] = {k: int(v) for k, v in nocal.lender_country.value_counts().head(5).items()}
    N["code_reaching_n_min"] = sorted(k for k, v in nocal.lender_country.value_counts().items() if v >= N_MIN)
    ci = N["coverage"][S_INT]
    N["internal_recoverable"] = {"clubs": len(prop), "membership_clubs": len(memb), "parent_clubs": len(parent),
                                 "slots": int(prop.episodes_affected.sum()),
                                 "endings_touched": ci["endings_with_a_side_filled"],
                                 "endings_fully_mapped": ci["endings_now_fully_mapped"],
                                 "newly_classifiable": ci["newly_classifiable"],
                                 "classifiable_share": ci["classifiable_share"]}

    # ---- candidate table
    cands = []
    for r in prop.itertuples():
        ev = (f"plays {r.orphan_competitions} (no competitions row); mapped participants: "
              + "; ".join(f"{c}: {N['orphan_codes'][c]['mapped']} mapped, {N['orphan_codes'][c]['countries'] or 'none'}"
                          for c in r.orphan_competitions.split(","))
              if r.tier == MEMBERSHIP_RULE else
              f"same normalised name as mapped club(s) {r.same_name_mapped_ids}; "
              f"{r.direct_moves_with_same_name_club} direct Stage 1C move(s)")
        cands.append({"club_id": r.club_id, "club_name": r.club_name, "primary_category": r.primary_category,
                      "episodes_affected": r.episodes_affected, "tier": r.tier, "proposed_country": r.proposed,
                      "status": "proposed", "evidence": ev, "linked_club_ids": r.linked_club_ids})
    for lab, df in ((REVIEW_RULES[0], review_parent), (REVIEW_RULES[1], review_senior)):
        for r in df.itertuples():
            cands.append({"club_id": r.club_id, "club_name": r.club_name, "primary_category": r.primary_category,
                          "episodes_affected": r.episodes_affected, "tier": lab,
                          "proposed_country": r.name_country, "status": "review only",
                          "evidence": f"same normalised name as mapped club(s) {r.same_name_mapped_ids}; "
                                      f"{r.direct_moves_with_same_name_club} direct Stage 1C move(s)",
                          "linked_club_ids": r.linked_club_ids})
    for r in code_clubs[code_clubs.club_id.isin(m_code.index)].itertuples():
        cands.append({"club_id": r.club_id, "club_name": r.club_name, "primary_category": r.primary_category,
                      "episodes_affected": r.episodes_affected,
                      "tier": "external-confirmed league code (snapshot clubs table)",
                      "proposed_country": m_code[r.club_id], "status": "option, needs sign-off",
                      "evidence": f"clubs.domestic_competition_id = {r.snapshot_competition}; every probed club with "
                                  f"that code has flag {m_code[r.club_id]}", "linked_club_ids": ""})
    for r in flag_sample.itertuples():
        cands.append({"club_id": r.club_id, "club_name": r.club_name, "primary_category": r.primary_category,
                      "episodes_affected": r.episodes_affected,
                      "tier": "external probe: upstream raw countryFlag (audit sample only)",
                      "proposed_country": r.flag_country, "status": "external test only",
                      "evidence": f"Transfermarkt country id {r.flag_ids} on {int(r.flag_rows)} raw transfer row(s)",
                      "linked_club_ids": ""})
    CAND = pd.DataFrame(cands)
    CAND["external_flag_country"] = CAND.club_id.map(F.set_index("club_id").flag_country).fillna("")
    CAND["agrees_with_external_flag"] = np.where(CAND.external_flag_country.eq(""), "",
                                                 np.where(CAND.external_flag_country.eq(CAND.proposed_country),
                                                          "yes", "no"))
    N["candidates"] = {f"{a} [{b}]": int(v) for (a, b), v in CAND.groupby(["tier", "status"]).size().items()}
    N["candidate_flag_check"] = {t: {"checked": int(g.external_flag_country.ne("").sum()),
                                     "agree": int(g.agrees_with_external_flag.eq("yes").sum())}
                                 for t, g in CAND[~CAND.tier.str.startswith("external probe")].groupby("tier")}

    # ---- selection bias
    BI = bias(R, u.loans)
    N["bias"] = BI.set_index("group").to_dict(orient="index")
    N["known_side"] = known_side(R, cal)

    tb = {"C": C, "A": A, "CAND": CAND, "COV": COV, "BI": BI, "F": F, "R": R}
    return N, tb


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


def yn(n: int) -> str:
    return "yes" if n else "no"


def compare(a: float, b: float, more: str, less: str, same: str = "about the same") -> str:
    return more if a > b else less if a < b else same


def render(N: dict, tb: dict) -> str:
    C, A, COV = tb["C"], tb["A"], tb["COV"]
    R = N["realised"]
    um = N["context"][UNMAPPED]
    sd = N["sides"]
    cats, ce = N["categories"], N["category_endings"]
    ir = N["internal_recoverable"]
    cv = N["coverage"]
    bd = N["bounds"]
    sm = N["sample"]
    fv = N["flag_validation_mapped"]
    fr = N["flag_validation_name_rules"]
    loo = N["validation_loo"]
    prop_share = ir["classifiable_share"]
    oc = N["orphan_codes"]
    memb_codes = {k: v for k, v in oc.items() if v["membership_country"] and v["unmapped"]}
    cf = N["orphan_code_flags"]
    code_cov = cv[N["code_scenario"]]
    sen = fr.get(REVIEW_RULES[1], {"n": 0, "agree": 0})
    par = fr.get(PROPOSED_RULE, {"n": 0, "agree": 0})
    senior = CATEGORIES[3]
    ls_ = loo.get(REVIEW_RULES[1], {"n": 0, "agree": 0})
    W: list[str] = []
    W += ["# Loan country mapping: why clubs are unmapped and what can recover them", "",
          "*Generated by `python -m src.analysis.loan_country_mapping_audit`. Diagnostic only: Stage 1C, "
          "`club_geography`, the calendar rule and its classifications are unchanged. Every recovered country below "
          "is a proposal in a separate table. Club names are used to describe clubs, never to assign a country on "
          "their own.*", ""]

    # ---- short answers
    W += ["## Short answers", ""]
    W += [f"1. **Why are {pc(N['unclassified'], R)} unclassified?** Of {f0(R)} realised loan endings, "
          f"{f0(um)} ({pc(um, R)}) have at least one club with no country, and {f0(N['below_threshold'])} "
          f"({pc(N['below_threshold'], R)}) are domestic loans in a country with fewer than {N_MIN} training endings. "
          f"The country of a club comes only from the snapshot: its `clubs` table ({f0(N['geo_sources'].get('clubs_table_domestic_league', 0))} "
          f"clubs in the {N['snapshot']['leagues']} first-tier leagues the snapshot covers) "
          f"and clubs that played a game in one of those countries' domestic leagues, cups or super cups "
          f"({f0(N['geo_sources'].get('domestic_competition_participation', 0))} more). Any other club has no "
          f"country.",
          f"2. **What kinds of clubs are missing?** {f0(N['clubs']['unique_ids'])} club ids. "
          f"{f0(cats[senior]['clubs'])} are senior clubs with no youth or reserve marker "
          f"({pc(cats[senior]['slots'], N['clubs']['slots'])} of the missing club slots): lower divisions of "
          f"covered countries and clubs in countries the snapshot does not cover, which repo data cannot tell apart. "
          f"{f0(cats[CATEGORIES[2]]['clubs'])} are reserve or B sides and {f0(cats[CATEGORIES[1]]['clubs'])} youth or "
          f"development sides. {f0(cats[CATEGORIES[0]]['clubs'])} are placeholders. The list is long-tailed: "
          f"half the clubs leave {f0(N['clubs']['median_endings'])} or fewer endings unmapped.",
          f"3. **Can repo data fix it?** Partly. No snapshot table gives any of these clubs a country directly "
          f"({N['flags']['any snapshot evidence with a country']} clubs). The {N['flags']['in snapshot clubs table']} "
          f"that are in the `clubs` table carry league code {', '.join(N['snapshot_club_codes'])}, which has no row "
          f"in `competitions`. Two deterministic internal routes exist: competition membership "
          f"({f0(ir['membership_clubs'])} clubs that play "
          + ", ".join(f"{k}, a competition with no `competitions` row whose {v['mapped']} mapped participants are all "
                      f"in {v['membership_country']}" for k, v in memb_codes.items())
          + f") and the parent-club link ({f0(ir['parent_clubs'])} reserve/youth sides whose Stage 1C normalised "
          f"name matches mapped clubs in exactly one country and which exchange players directly with that club). "
          f"Together, {f0(ir['clubs'])} clubs fill a side of {f0(ir['endings_touched'])} endings, complete "
          f"{f0(ir['endings_fully_mapped'])} and make {f0(ir['newly_classifiable'])} newly classifiable.",
          f"4. **Cleanest external source:** the upstream project's own raw transfer files (same public bucket as the "
          f"DuckDB). Each side of each transfer row carries Transfermarkt's country flag, keyed by the same club id: "
          f"{f0(N['probe']['clubs_seen'] - N['flag_null'] - N['flag_multi'])} of the {f0(N['probe']['clubs_seen'])} "
          f"clubs in the fetched records have exactly one, {N['flag_multi']} have several, and "
          f"{N['flag_null_special']} of the {N['flag_null']} without one are Transfermarkt placeholders such as "
          f"Without Club. "
          f"On the {N['sample_size']} audit clubs it returned a country for {sm['flag_found']}; on "
          f"{f0(fv['n'])} already-mapped clubs in the same records it agreed with our mapping {f0(fv['agree'])} times. "
          + (f"Wikidata could not be queried ({N['probe']['wikidata_error']})." if N["probe"]["wikidata_error"]
             else f"Wikidata has an item for {sm['wikidata_found']} of the {N['sample_size']} audit clubs, "
                  f"{sm['wikidata_with_country']} with a country."),
          f"5. **Recoverable calendar coverage:** today {f0(N['classifiable_now'])} of {f0(R)} endings "
          f"({pc(N['classifiable_now'], R)}) are classifiable. The two internal routes raise that to "
          f"{pct(prop_share)}. Reading the snapshot's "
          + ", ".join(f"{k} code as {next(iter(v['flag_countries']), '?')}" for k, v in cf.items())
          + f" (confirmed by the raw flag) adds {f0(code_cov['newly_classifiable'])} more on its own. Recovering every "
          f"club from the raw flags would give between {pc(bd['lower'], R)} and {pc(bd['upper'], R)} before any "
          f"calendar is re-learned.",
          f"6. **Safe to improve without guessing?** The raw flag is a recorded attribute keyed by club id, "
          f"not an inference ({pc(fv['agree'], fv['n'])} agreement on mapped clubs; the differences are explained "
          f"in section 4). The parent link agreed with the flag {par['agree']} of {par['n']} times in the probe, so "
          f"it is usable as a cross-check, but it is still name-based. Exact senior names are not safe: "
          f"{sen['agree']} of {sen['n']} agreed with the flag and {ls_['agree']} of {ls_['n']} in the leave-one-out "
          f"check on mapped clubs. Fuzzy matching was not used.", ""]

    # ---- Part 1
    W += ["## 1. The unmapped population", ""]
    W += md_table(["Realised endings", "N", "Share"],
                  [["Both clubs mapped, same country (domestic)", f0(N["context"].get("domestic", 0)),
                    pc(N["context"].get("domestic", 0), R)],
                   ["Both clubs mapped, different countries (cross-border)", f0(N["context"].get("cross-border", 0)),
                    pc(N["context"].get("cross-border", 0), R)],
                   ["Lender missing, borrower known", f0(sd["lender only missing"]), pc(sd["lender only missing"], R)],
                   ["Borrower missing, lender known", f0(sd["borrower only missing"]),
                    pc(sd["borrower only missing"], R)],
                   ["Both clubs missing", f0(sd["both missing"]), pc(sd["both missing"], R)],
                   ["**All with a club missing**", f"**{f0(um)}**", f"**{pc(um, R)}**"],
                   [f"Domestic, country known but under {N_MIN} training endings", f0(N["below_threshold"]),
                    pc(N["below_threshold"], R)],
                   ["**Unclassified in the calendar rule**", f"**{f0(N['unclassified'])}**",
                    f"**{pc(N['unclassified'], R)}**"]])
    bc = N["below_countries"]
    W += ["", f"The below-threshold endings come from {len(bc)} countries, led by "
          + ", ".join(f"{dc(k)} ({v})" for k, v in list(bc.items())[:5]) + ".", ""]
    W += [f"**Why clubs have no country.** Stage 1C names {f0(N['stage1c_clubs'])} club ids. "
          "`club_geography` has two sources, both in the snapshot: "
          f"(1) the `clubs` table, {f0(N['geo_sources'].get('clubs_table_domestic_league', 0))} clubs, each with its "
          f"domestic league; (2) participation in a covered country's domestic league, cup or super cup in `games`, "
          f"{f0(N['geo_sources'].get('domestic_competition_participation', 0))} further clubs. Together "
          f"{f0(N['geo_clubs'])} clubs in {N['geo_countries']} countries. The rest are lower-division clubs, "
          "academies and reserve sides, and clubs in countries the snapshot does not cover, which players pass "
          "through on the way to or from the covered leagues.", ""]
    fl = N["flags"]
    W += md_table(["Where the unmapped club ids appear in the snapshot", "Clubs", "Carries a country?"],
                  [["`clubs` table", fl["in snapshot clubs table"],
                    "no: " + ", ".join(f"{k} ({v})" for k, v in N["snapshot_club_codes"].items())
                    + " has no row in `competitions`"],
                   ["`games` (home or away)", fl["in snapshot games"],
                    yn(N["tier_counts"]["games"]) + ": " + ", ".join(f"{c} ({N['code_status'].get(c, '?')})"
                                                                     for c in N["game_competitions"])],
                   ["`appearances`", fl["in snapshot appearances"],
                    "no: the same competitions (" + ", ".join(N["appearance_codes"]) + ")"],
                   ["`players.current_club_id`", fl["as players' current club"],
                    yn(N["tier_counts"]["other tables"]) + ": league code empty or "
                    + (", ".join(N["players_codes"]) or "none")],
                   ["`player_valuations.current_club_id`", f0(fl["as valuations' current club"]),
                    yn(N["tier_counts"]["other tables"]) + ": league code empty or "
                    + (", ".join(N["valuations_codes"]) or "none")],
                   ["`transfers` (snapshot)", f0(fl["in snapshot transfers table"]), "no: names only"],
                   ["dropped by `club_geography` for two countries", fl["dropped as two-country"], "—"],
                   ["**any of these with a usable country**", f"**{fl['any snapshot evidence with a country']}**", ""]],
                  right={1})
    W += [""]
    W += ["**What the missing clubs are** (name-based description only; no country is assigned from it).", ""]
    W += md_table(["Category", "Clubs", "Missing-club slots", "Endings involving one"],
                  [[k, f0(cats[k]["clubs"]), f0(cats[k]["slots"]), f0(ce[k])] for k in CATEGORIES]
                  + [["**Total**", f"**{f0(N['clubs']['unique_ids'])}**", f"**{f0(N['clubs']['slots'])}**",
                      f"**{f0(um)}**"]])
    W += ["", f"Slots count a club once per ending it leaves unmapped ({f0(N['clubs']['slots'])} across "
          f"{f0(um)} endings, because {f0(sd['both missing'])} endings miss both clubs). Endings involving a "
          f"category overlap.", ""]
    W += [f"Overlays: {f0(fl['several names under one id'])} club ids carry several names in Stage 1C (renamed or "
          f"respelled); {f0(fl['shares exact normalised name with a mapped club'])} share an exact normalised name with "
          f"a mapped club under another id. The median unmapped club was last seen in Stage 1C in "
          f"{N['last_seen_year']['unmapped']:.0f}, against {N['last_seen_year']['mapped']:.0f} for mapped clubs; "
          f"{f0(N['last_seen_before_2015'])} were last seen before 2015, so historical clubs are a minor part. "
          f"There are {N['clubs']['unique_names']:,} distinct names on the {f0(N['clubs']['unique_ids'])} ids.", ""]
    epc = N["clubs"]["endings_per_club"]
    W += md_table(["Endings a club leaves unmapped"] + list(epc), [["Clubs"] + [f0(v) for v in epc.values()]])
    W += ["", f"The top 50 clubs account for {pct(N['clubs']['top50_share_of_slots'])} of the missing-club slots and "
          f"the top 500 for {pct(N['clubs']['top500_share_of_slots'])}.", "", "**Top 50 unmapped clubs by endings "
          "affected**", ""]
    W += md_table(["#", "Club", "Id", "Category", "Endings", "As lender", "As borrower", "Name rule"],
                  [[i + 1, r.club_name, r.club_id, r.primary_category.split(",")[0], r.episodes_affected,
                    r.as_lender, r.as_borrower, r.name_rule] for i, r in enumerate(C.head(50).itertuples())],
                  right={0, 2, 4, 5, 6})
    W += [""]

    # ---- Part 2
    W += ["## 2. Manual audit sample", "",
          f"{N['sample_size']} clubs, chosen reproducibly: within each stratum the clubs leaving the most endings "
          f"unmapped, ties broken by club id, no club twice. Strata: "
          + "; ".join(f"{k} ({v})" for k, v in sm["by_stratum"].items()) + ". "
          f"Together they leave {f0(sm['episodes'])} endings unmapped.", ""]
    W += md_table(["Club", "Id", "Stratum", "Endings", "Repo recovers?", "Proposed (repo)", "Raw flag", "Wikidata",
                   "Reason"],
                  [[r.club_name, r.club_id, r.stratum, r.episodes_affected, r.recoverable_from_repo_yes_no,
                    r.proposed_country_if_deterministically_supported or "—", r.flag_country or "—",
                    r.wikidata_country or "—", r.reason] for r in A.itertuples()], right={1, 3})
    via_parent = int(A.recoverable_from_repo_yes_no.eq("yes").sum() - A.internal_country_evidence.sum())
    W += ["", f"From repo data alone {sm['recoverable_repo']} of {N['sample_size']} are recoverable, {via_parent} "
          f"of them through the parent link. The raw flag gives {sm['flag_found']} of {N['sample_size']} a country; where both exist "
          f"they agree {sm['proposal_agree']} of {sm['proposals']} times. The full evidence per club is in "
          f"`{SAMPLE_CSV.name}`.", ""]

    # ---- Part 3
    W += ["## 3. Internal sources, ranked", ""]
    W += md_table(["Rank", "Source", "Match key", "Unmapped clubs it gives a country", "Status"],
                  [[1, "`clubs.domestic_competition_id` → `competitions.country_name`", "club id",
                    f"{N['tier_counts']['clubs table']} ({N['flags']['in snapshot clubs table']} listed, league code "
                    f"{', '.join(N['snapshot_club_codes'])} absent from `competitions`)", "in production"],
                   [2, "domestic league/cup/super-cup games → competition country", "club id",
                    f"{N['tier_counts']['games']}", "in production"],
                   ["2b", "competition membership: games in a competition with no `competitions` row, all of whose "
                          "mapped participants are in one country", "club id + competition code",
                    f"{f0(ir['membership_clubs'])}", "proposed (separate table)"],
                   [3, "parent-club link: Stage 1C academy/reserve normalisation → one mapped club, one country, "
                       "plus a direct Stage 1C move between the ids", "club id pair + exact normalised name",
                    f"{f0(ir['parent_clubs'])}", "proposed (separate table)"],
                   [4, "same club id in `players`, `player_valuations`, `appearances`, `club_games`", "club id",
                    f"{N['tier_counts']['other tables']} (league codes: "
                    f"{', '.join(sorted(set(N['players_codes']) | set(N['valuations_codes']))) or 'none'})",
                    "nothing to add" if not N["tier_counts"]["other tables"] else "usable"],
                   [5, "exact normalised senior name with one country", "name",
                    f"{f0(N['name_rules'].get(REVIEW_RULES[1], {}).get('clubs', 0))} candidates", "review only"],
                   ["–", "fuzzy name matching", "name", "not run", "excluded"]], right={0})
    W += ["", "**Competitions in `games` with no `competitions` row** (invisible to `club_geography`).", ""]
    W += md_table(["Code", "Participants", "Already mapped", "Countries of mapped participants", "Unmapped",
                   "Membership country"],
                  [[k, v["participants"], v["mapped"], ", ".join(dc(c) for c in v["countries"].split("|") if c) or "—",
                    v["unmapped"],
                    v["membership_country"] or "none"] for k, v in sorted(oc.items(), key=lambda kv: -kv[1]["unmapped"])],
                  right={1, 2, 4})
    mv = N["membership_vs_parent"]
    W += ["", f"Clubs that qualify under both 2b and 3: {mv['both']} ({mv['disagree']} disagree); 2b takes precedence."]
    lp = loo.get(PROPOSED_RULE, {"n": 0, "agree": 0})
    ls = loo.get(REVIEW_RULES[1], {"n": 0, "agree": 0})
    W += ["", "**Name rules on the full unmapped list.**", ""]
    W += md_table(["Exact normalised-name outcome", "Clubs", "Missing-club slots"],
                  [[k, f0(v["clubs"]), f0(v["slots"])] for k, v in
                   sorted(N["name_rules"].items(), key=lambda kv: -kv[1]["slots"])])
    W += ["", f"**Validation.** Leave-one-out on already-mapped clubs: the senior exact-name rule returns the right "
          f"country for {ls['agree']} of {ls['n']} (" + ", ".join(ls.get("disagree_examples", [])) + " are wrong). "
          f"Only {lp['n']} mapped club passes the parent rule, so internal validation of the parent link is thin "
          f"({N['mapped_youth_reserve_clubs']} mapped clubs have a youth/reserve name at all). The external probe "
          f"below is the real check.", ""]

    # ---- Part 4
    W += ["## 4. External sources (feasibility, audit sample only)", ""]
    W += md_table(["Source", "Match key", "Deterministic?", "Licence", "Coverage (audit test)", "Requests", "Verdict"],
                  [["Upstream raw transfer files (`countryFlag` on each `from`/`to`)", "Transfermarkt club id "
                    "(same as ours)", "yes: recorded attribute, Transfermarkt country id = `countries.country_id`",
                    "same lineage and terms as the DuckDB we already use", f"{sm['flag_found']}/{N['sample_size']}",
                    f"{len(N['probe']['season_files'])} public file(s) streamed ("
                    + ", ".join(N["probe"]["season_files"]) + "), no scraping",
                    "**best**"],
                   ["Wikidata (P7223 Transfermarkt team id → P17 country)", "Transfermarkt club id", "yes, where linked",
                    "CC0", (f"not available: {N['probe']['wikidata_error']}" if N["probe"]["wikidata_error"] else
                            f"{sm['wikidata_found']}/{N['sample_size']} items, {sm['wikidata_with_country']} with a "
                            "country"), "3 Action API requests",
                    f"secondary: misses {N['sample_size'] - sm['wikidata_found']} of {N['sample_size']} audit clubs; "
                    f"{sm['wikidata_uk']} come back as 'United Kingdom', not a home nation"],
                   ["Open name-keyed databases (openfootball, football-data.co.uk)", "club name", "no", "open",
                    "not tested", "—", "rejected: name matching"],
                   ["transfermarkt.com pages", "club id", "yes", "site terms", "not tested", "per club",
                    "rejected: scraping"]], right=set())
    W += ["", f"**Probe.** {N['probe']['players_requested']} players ({PLAYERS_PER_CLUB} or fewer per audit club) "
          f"were read from the raw files the snapshot was built from; {N['probe']['players_found']} were found. "
          f"Their full transfer histories name {f0(N['probe']['clubs_seen'])} clubs. Fetched {N['probe']['fetched_on']}, "
          f"cached in `{PROBE_JSON.name}`.", ""]
    W += [f"- **Flag vs our mapping**, on {f0(fv['n'])} already-mapped clubs in those records: "
          f"{f0(fv['agree'])} agree ({pc(fv['agree'], fv['n'])})."
          + (" Disagreements: " + "; ".join(fv["disagreements"][:8]) + ". These are clubs that play in another "
             "country's league; the flag is the club's association, the snapshot uses its league." if fv["disagreements"]
             else "")]
    for k, d in fr.items():
        W += [f"- **Flag vs the {k} rule**, on every club in those records that the snapshot does not map (loan "
              f"clubs or not): {d['agree']} of {d['n']} agree."
              + (" Disagreements: " + "; ".join(d["disagreements"]) + "." if d["disagreements"] else "")]
    W += [f"- Clubs in the records with no usable flag: {N['flag_null']} ({N['flag_null_special']} are Transfermarkt "
          f"'special' entries such as Without Club); flags outside the snapshot's `countries` table: "
          f"{N['flag_not_in_snapshot']}.",
          f"- Wikidata vs flag where both exist: {sm['wikidata_vs_flag']['agree']} of {sm['wikidata_vs_flag']['n']} "
          f"agree; {sm['wikidata_vs_flag']['uk_vs_home_nation']} differ only because Wikidata says 'United Kingdom'.",
          ""]

    # ---- Part 5
    W += ["## 5. How much can be recovered", "",
          "Calendars are the existing ones, learned once and not re-learned; only missing club countries are "
          "filled. A recovered domestic ending is classifiable only if its country already has a calendar.", ""]
    W += md_table(["Scenario", "Clubs added", "Endings with a side filled", "Endings fully mapped", "Newly "
                   "classifiable", "Classifiable", "Share"],
                  [[r.scenario, f0(r.clubs_added), f0(r.endings_with_a_side_filled), f0(r.endings_now_fully_mapped),
                    f0(r.newly_classifiable), f0(r.classifiable), pct(r.classifiable_share)] for r in COV.itertuples()])
    W += ["", f"The lower bound counts endings with one club known in a country that has a calendar "
          f"({f0(bd['guaranteed'])}): whatever country the other club turns out to have, the ending is either domestic "
          f"in that calendar or cross-border. The remaining {f0(bd['depends_on_country'])} depend on the recovered "
          f"countries ({f0(bd['both_missing'])} with both clubs missing, {f0(bd['one_side_known_no_calendar'])} with "
          f"the known club in a country without a calendar).", "",
          f"The league-code row adds little on its own because a recovered domestic ending needs a calendar: after it, "
          f"the largest groups of domestic endings still without one are "
          + ", ".join(f"{dc(k)} ({v} non-COVID)" for k, v in N["code_domestic_training"].items())
          + ". " + (", ".join(dc(c) for c in N["code_reaching_n_min"]) + f" would pass the {N_MIN}-ending threshold "
                    "and get its own calendar if the rule were re-learned; that is not done here."
                    if N["code_reaching_n_min"] else f"None reaches the {N_MIN}-ending threshold."), ""]
    W += [f"Under the proposed internal links, {f0(N['prop_new_xb'])} endings become cross-border and the new domestic "
          "endings fall in " + ", ".join(f"{dc(k)} ({v})" for k, v in N["prop_new_domestic_by_country"].items()) + "."
          + (" Re-learning would bring " + ", ".join(dc(c) for c in N["countries_reaching_n_min"])
             + f" over the {N_MIN}-ending threshold." if N["countries_reaching_n_min"] else
             f" No new country would cross the {N_MIN}-ending threshold."), ""]

    # ---- Part 6
    BI = tb["BI"].set_index("group")
    W += ["## 6. Are the unmapped endings different? (selection check)", ""]
    cols = list(BI.index)
    rows = []
    for var in BI.columns:
        if var == "endings":
            rows.append([var] + [f0(BI.loc[c, var]) for c in cols])
        elif var.startswith("median"):
            fmt = ((lambda x: f"€{x / 1e3:,.0f}k") if "EUR" in var else (lambda x: f"{x:.0f}") if "year" in var
                   else (lambda x: f"{x:,.0f}"))
            rows.append([var] + [fmt(BI.loc[c, var]) for c in cols])
        else:
            rows.append([var] + [pct(BI.loc[c, var]) for c in cols])
    W += md_table(["Realised endings"] + cols, rows)
    b0, b1 = N["bias"][BIAS_GROUPS[0]], N["bias"]["any club missing"]
    mv1, mv0 = b1["median market value (EUR)"], b0["median market value (EUR)"]
    W += ["", "Compared with endings where both clubs are mapped, endings with a club missing start in "
          + ("the same" if b1["median loan start year"] == b0["median loan start year"] else
             compare(b1["median loan start year"], b0["median loan start year"], "a later", "an earlier"))
          + f" median loan year ({b1['median loan start year']:.0f} vs {b0['median loan start year']:.0f}; "
          f"{pct(b1['2020 or later'])} vs {pct(b0['2020 or later'])} start in 2020 or later), involve "
          + compare(b1["median age at loan"], b0["median age at loan"], "older", "younger", "same-age")
          + f" players (median age {b1['median age at loan']:.0f} vs {b0['median age at loan']:.0f}) with a "
          + compare(mv1, mv0, "higher", "lower", "similar")
          + f" median market value (€{mv1 / 1e3:,.0f}k vs €{mv0 / 1e3:,.0f}k), carry a loan fee "
          + compare(b1["fee-bearing loan"], b0["fee-bearing loan"], "more", "less", "as")
          + f" often ({pct(b1['fee-bearing loan'])} vs {pct(b0['fee-bearing loan'])}), and have a borrower with a "
          f"youth or reserve name in {pct(b1['borrower name has youth/reserve marker'])} of cases against "
          f"{pct(b0['borrower name has youth/reserve marker'])}. Next moves differ too: a permanent move to the "
          f"borrower follows {pct(b1['next move: permanent_to_borrower'])} against "
          f"{pct(b0['next move: permanent_to_borrower'])}. The classified set is therefore a selected subset, and "
          "any off-calendar share computed on it describes the mapped kind of loan, not all loans.", ""]
    ks = N["known_side"]
    for lab, d in ks.items():
        W += [f"- **{lab}** ({f0(d['n'])} endings, {d['countries']} known countries): "
              + ", ".join(f"{dc(k)} {f0(v)}" for k, v in d["top"].items())
              + f". {f0(d['with_calendar'])} ({pc(d['with_calendar'], d['n'])}) have the known club in a country "
                "with a calendar."]
    W += [""]

    # ---- recommendation
    W += ["## What is safe, and the next step", "",
          f"- **Safe now (proposal):** the {f0(ir['membership_clubs'])} competition-membership clubs and the "
          f"{f0(ir['parent_clubs'])} parent links, kept in `{CANDIDATES_CSV.name}` with their evidence. Both are "
          "deterministic and auditable. The parent link is name-based, so confirm it against the raw flag before use. "
          "Where the probe saw a flag for one of these candidates: "
          + "; ".join(f"{lab} {v['agree']} of {v['checked']} agree" for lab, v in
                      (("competition-membership", N["candidate_flag_check"].get(MEMBERSHIP_RULE, {"checked": 0, "agree": 0})),
                       ("parent-link", N["candidate_flag_check"].get(PROPOSED_RULE, {"checked": 0, "agree": 0}))))
          + ".",
          "- **Not safe:** exact senior-name identity and any fuzzy matching.",
          "- **Best next step:** extend the Stage 1B parser, which already reads these raw files, to keep each "
          "side's `countryFlag` id, and build a club id → Transfermarkt country table from it as a separate "
          "file. Map the id through the snapshot's own `countries` table. Keep the league-based snapshot country "
          "where both exist, and report the clubs whose flag differs (clubs playing in another country's league). "
          "Re-run the calendar rule only after Professor Freund agrees to the new mapping.", ""]
    return "\n".join(W)


def numbers_check(md: str, N: dict) -> list[str]:
    """Key generated numbers that must appear verbatim in the markdown."""
    want = [f0(N["realised"]), f0(N["context"][UNMAPPED]), f0(N["clubs"]["unique_ids"]), f0(N["below_threshold"]),
            f0(N["internal_recoverable"]["clubs"]), f0(N["classifiable_now"])]
    return [w for w in want if w not in md]


def main(u=None, refresh: bool = False) -> dict:
    sha_before = _stage1c_sha()
    N, tb = compute(u, refresh)
    C, A, CAND, COV = tb["C"], tb["A"], tb["CAND"], tb["COV"]
    keep = ["club_id", "club_name", "all_names", "n_names", "primary_category", "episodes_affected", "as_lender",
            "as_borrower", "in_both_missing", "sole_unmapped_side", "stage1c_rows", "first_seen", "last_seen",
            "in_snapshot_clubs_table", "snapshot_competition", "snapshot_competition_country", "game_competitions",
            "n_games", "appearance_competitions", "players_competition", "valuations_competition",
            "snapshot_transfer_rows", "dropped_as_two_country", "internal_country_evidence", "same_name_mapped_ids",
            "same_name_mapped_countries", "direct_moves_with_same_name_club", "linked_club_ids", "name_rule",
            "name_country"]
    out = C[keep].copy()
    for c in ("first_seen", "last_seen"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(CLUBS_CSV, index=False)
    A[["club_id", "club_name", "stratum", "primary_category", "episodes_affected", "missing_side_mix",
       "current_country", "evidence_source", "recoverable_from_repo_yes_no",
       "proposed_country_if_deterministically_supported", "reason", "flag_country", "flag_ids", "wikidata_country",
       "wikidata_items", "external_agrees_with_proposal"]].to_csv(SAMPLE_CSV, index=False)
    CAND.to_csv(CANDIDATES_CSV, index=False)
    COV.to_csv(COVERAGE_CSV, index=False)
    md = render(N, tb)
    miss = numbers_check(md, N)
    assert not miss, f"numbers missing from the markdown: {miss}"
    AUDIT_MD.write_text(md)
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    assert _stage1c_sha() == sha_before
    print_summary(N)
    return N


def _stage1c_sha() -> str:
    import hashlib
    return hashlib.sha256((OUT / "stage1c_canonical_transfers.csv").read_bytes()).hexdigest()


def print_summary(N: dict) -> None:
    R, um, ir, bd = N["realised"], N["context"][UNMAPPED], N["internal_recoverable"], N["bounds"]
    cats = N["categories"]
    sm = N["sample"]
    print("UNMAPPED:")
    print(f"  realised endings missing geography: {um:,} of {R:,} ({pc(um, R)}); "
          f"+{N['below_threshold']:,} domestic in countries under {N_MIN} endings")
    print(f"  unique clubs involved: {N['clubs']['unique_ids']:,}")
    print(f"  top causes: snapshot geography reaches only clubs in its {N['snapshot']['leagues']} top flights and "
          f"clubs that met them in domestic competitions -> {cats[CATEGORIES[3]]['clubs']:,} senior clubs "
          f"(lower divisions or uncovered countries; {pc(cats[CATEGORIES[3]]['slots'], N['clubs']['slots'])} of "
          f"missing-club slots), {cats[CATEGORIES[2]]['clubs']:,} reserve/B and {cats[CATEGORIES[1]]['clubs']:,} "
          f"youth sides; league code " + ", ".join(f"{k} ({v['clubs']} clubs, {v['slots']:,} missing-club slots)"
                                                   for k, v in N["orphan_code_flags"].items())
          + " has no row in `competitions`")
    print("RECOVERABLE INTERNALLY:")
    print(f"  clubs: {ir['clubs']:,} ({ir['membership_clubs']} competition membership + {ir['parent_clubs']} "
          f"parent links with a recorded move; proposal only)")
    print(f"  loan endings: {ir['endings_touched']:,} get a side filled, {ir['endings_fully_mapped']:,} fully mapped, "
          f"{ir['newly_classifiable']:,} newly classifiable")
    print(f"  projected classification coverage: {pc(N['classifiable_now'], R)} -> {pct(ir['classifiable_share'])} "
          f"of {R:,} realised endings")
    print("EXTERNAL OPTION:")
    print(f"  best candidate source: upstream raw transfer files' countryFlag (same club id, same lineage); "
          f"audit clubs with a flag {sm['flag_found']}/{N['sample_size']}; agreement with mapped clubs "
          f"{N['flag_validation_mapped']['agree']}/{N['flag_validation_mapped']['n']}")
    print(f"  expected benefit: all {um:,} endings mapped; classifiable {pc(bd['lower'], R)}–{pc(bd['upper'], R)} "
          f"of realised endings (now {pc(N['classifiable_now'], R)}) before re-learning calendars")
    print("RECOMMENDATION:")
    print("  Add the raw countryFlag id to the Stage 1B parse as a separate club-country table, validate it against "
          "the snapshot mapping, and only then re-run the calendar rule; keep parent links as a cross-check.")


if __name__ == "__main__":
    main(refresh="--refresh-external" in sys.argv)
