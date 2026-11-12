"""Season-specific club -> league / tier lookup for Transfermarkt club ids.

    python -m src.analysis.league_tiers            # download (cached), parse, match, write outputs
    python -m src.analysis.league_tiers --offline  # never touch the network (fails if a file is missing)

Sources
  * Transfermarkt snapshot (data/raw/transfermarkt-datasets.duckdb, read-only): `games` rows of the 31
    `domestic_league` competitions give tier 1 for (club_id, season) directly.
  * openfootball (CC0 1.0, public domain): football.json files (preferred) and, only where no football.json file
    exists for a league-season, the upstream Football.TXT files of the same project (england, deutschland, espana,
    italy, europe repositories), parsed here.
  * football-data.co.uk: NOT downloaded. Its robots.txt disallows the user agents Anthropic-AI, Claude-Web and
    ClaudeBot for the whole site; this lookup is built by an AI agent, so fetching the data CSVs under a generic
    User-Agent would sidestep that rule. Its disclaimer page carries no reuse licence. See README.

Season convention: integer `season` = Transfermarkt season id = start year (2024 means 2024/25), identical to
`games.season` in the snapshot. Calendar-year leagues (file label "2025") follow Transfermarkt's convention that
calendar year Y is season id Y-1 (verified in the snapshot: NO1/SE1/BRA1/JAP1/MLS1 season 2024 games are dated 2025).

Matching is deterministic (no similarity scores): normalised exact name match within the league's country (family),
a hand-written alias table, and aliases learned from tier-1 league-seasons present in both openfootball and the
snapshot (unique restricted match or forced elimination). Ambiguous names stay unmatched. Reserve / youth sides are
never matched to senior clubs.

New files only: writes data/processed/league_tiers/*, data/raw/external_leagues/* (cache + manifest).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "external_leagues"
OF_DIR = RAW / "openfootball"
FD_DIR = RAW / "football-data.co.uk"
MANIFEST = RAW / "manifest.json"
OUT = ROOT / "data" / "processed" / "league_tiers"
DUCKDB_PATH = ROOT / "data" / "raw" / "transfermarkt-datasets.duckdb"
MASTER_PATH = ROOT / "data" / "outputs" / "rebuild" / "analysis_episode_master.csv"
GEO_PATH = ROOT / "data" / "outputs" / "rebuild" / "club_geography_v2.csv"
ORG_PATH = ROOT / "data" / "outputs" / "rebuild" / "club_organisation_map.csv"

TABLE_CSV = OUT / "club_season_league.csv"
AUDIT_CSV = OUT / "name_match_audit.csv"
COVERAGE_CSV = OUT / "league_season_coverage.csv"
CONFLICTS_CSV = OUT / "club_season_conflicts.csv"
ALIASES_CSV = OUT / "learned_aliases.csv"
MASTER_COV_CSV = OUT / "master_coverage_by_country.csv"
NUMBERS_JSON = OUT / "league_tiers_numbers.json"
README_MD = OUT / "README.md"

USER_AGENT = "football-contract-research/1.0"
MIN_INTERVAL_S = 1.1
SEASON_MIN, SEASON_MAX = 2010, 2026  # external files downloaded (start year); analysis focus 2012-2025

SRC_SNAPSHOT = "transfermarkt_snapshot_games"
SRC_OF = "openfootball"

OPENFOOTBALL_TERMS = ("CC0 1.0 Universal public-domain dedication (LICENSE.md of openfootball/football.json and of "
                      "the upstream openfootball/england, deutschland, espana, italy, europe repositories). "
                      "Reuse for any purpose permitted; no attribution required (given anyway).")
GITHUB_API_TERMS = "GitHub REST API (unauthenticated, 60 requests/hour); repository tree listing only."
FOOTBALL_DATA_TERMS = ("NOT USED for data. robots.txt (fetched 2026-11-07) allows '*' but disallows Anthropic-AI, "
                       "Claude-Web, ClaudeBot (and GPTBot, PerplexityBot, CCBot, ...) for '/'. The disclaimer page is a "
                       "liability/gambling disclaimer and states no licence for reuse. Because this lookup is built by "
                       "an AI agent, downloading the CSVs with a generic User-Agent would circumvent the robots rule, "
                       "so no data CSV was requested. Only robots.txt, disclaimer.php and notes.txt were fetched.")

# --------------------------------------------------------------------------------------------------------------------
# League definitions: code -> (country, tier, league name, country family of member clubs, snapshot competition id)
# Tier = level in that country's league pyramid in the given season. Names are the competition names in that season.
# --------------------------------------------------------------------------------------------------------------------
UK_ENG = ("England", "Wales")


def _l(country, tier, name, family=None, snap=None, calendar=False):
    return {"country": country, "tier": tier, "name": name, "family": tuple(family or (country,)), "snap": snap,
            "calendar": calendar}


LEAGUES: dict[str, dict] = {
    # England (English pyramid includes Welsh clubs Cardiff, Swansea, Newport, Wrexham, ...)
    "en.1": _l("England", 1, "Premier League", UK_ENG, "GB1"),
    "en.2": _l("England", 2, "EFL Championship", UK_ENG),
    "en.3": _l("England", 3, "EFL League One", UK_ENG),
    "en.4": _l("England", 4, "EFL League Two", UK_ENG),
    "en.5": _l("England", 5, "National League", UK_ENG),
    "sco.1": _l("Scotland", 1, "Scottish Premiership", None, "SC1"),
    "de.1": _l("Germany", 1, "Bundesliga", None, "L1"),
    "de.2": _l("Germany", 2, "2. Bundesliga"),
    "de.3": _l("Germany", 3, "3. Liga"),
    "de.4.bayern": _l("Germany", 4, "Regionalliga Bayern"),
    "de.4.nord": _l("Germany", 4, "Regionalliga Nord"),
    "de.4.nordost": _l("Germany", 4, "Regionalliga Nordost"),
    "de.4.suedwest": _l("Germany", 4, "Regionalliga Südwest"),
    "de.4.west": _l("Germany", 4, "Regionalliga West"),
    "es.1": _l("Spain", 1, "LaLiga (Primera División)", ("Spain", "Andorra"), "ES1"),
    "es.2": _l("Spain", 2, "Segunda División", ("Spain", "Andorra")),
    "it.1": _l("Italy", 1, "Serie A", None, "IT1"),
    "it.2": _l("Italy", 2, "Serie B", None),
    "it.3.a": _l("Italy", 3, "Serie C Girone A", ("Italy", "San Marino")),
    "it.3.b": _l("Italy", 3, "Serie C Girone B", ("Italy", "San Marino")),
    "it.3.c": _l("Italy", 3, "Serie C Girone C", ("Italy", "San Marino")),
    "fr.1": _l("France", 1, "Ligue 1", ("France", "Monaco"), "FR1"),
    "fr.2": _l("France", 2, "Ligue 2", ("France", "Monaco")),
    "nl.1": _l("Netherlands", 1, "Eredivisie", None, "NL1"),
    "nl.2": _l("Netherlands", 2, "Eerste Divisie"),
    "pt.1": _l("Portugal", 1, "Primeira Liga", None, "PO1"),
    "pt.2": _l("Portugal", 2, "Liga Portugal 2 (Segunda Liga)"),
    "be.1": _l("Belgium", 1, "Belgian Pro League (First Division A)", None, "BE1"),
    "gr.1": _l("Greece", 1, "Super League Greece", None, "GR1"),
    "tr.1": _l("Türkiye", 1, "Süper Lig", None, "TR1"),
    "tr.2": _l("Türkiye", 2, "TFF 1. Lig"),
    "at.1": _l("Austria", 1, "Austrian Bundesliga", None, "A1"),
    "at.2": _l("Austria", 2, "2. Liga (Erste Liga until 2017/18)"),
    "ch.1": _l("Switzerland", 1, "Swiss Super League", ("Switzerland", "Liechtenstein"), "C1"),
    "ch.2": _l("Switzerland", 2, "Challenge League", ("Switzerland", "Liechtenstein")),
    "ru.1": _l("Russia", 1, "Russian Premier League", None, "RU1"),
    "ru.2": _l("Russia", 2, "Russian First League (FNL)"),
    "cz.1": _l("Czech Republic", 1, "Czech First League", None, "TS1"),
    "cz.2": _l("Czech Republic", 2, "Czech National Football League"),
    "hu.1": _l("Hungary", 1, "Nemzeti Bajnokság I"),
    "dk.1": _l("Denmark", 1, "Danish Superliga", None, "DK1"),
    "dk.2": _l("Denmark", 2, "Danish 1st Division"),
    "pl.1": _l("Poland", 1, "Ekstraklasa", None, "PL1"),
    "pl.2": _l("Poland", 2, "I liga"),
    "no.1": _l("Norway", 1, "Eliteserien", None, "NO1", True),
    "no.2": _l("Norway", 2, "1. divisjon (OBOS-ligaen)", None, None, True),
    "se.1": _l("Sweden", 1, "Allsvenskan", None, "SE1", True),
    "se.2": _l("Sweden", 2, "Superettan", None, None, True),
    "se.3s": _l("Sweden", 3, "Ettan Södra", None, None, True),
    "fi.1": _l("Finland", 1, "Veikkausliiga", None, None, True),
    "is.1": _l("Iceland", 1, "Besta deild", None, None, True),
    "ie.1": _l("Ireland", 1, "League of Ireland Premier Division", None, None, True),
    "ee.1": _l("Estonia", 1, "Meistriliiga", None, None, True),
    "lv.1": _l("Latvia", 1, "Virslīga", None, None, True),
    "lt.1": _l("Lithuania", 1, "A Lyga", None, None, True),
    "ge.1": _l("Georgia", 1, "Erovnuli Liga", None, None, True),
    "by.1": _l("Belarus", 1, "Belarusian Premier League", None, None, True),
    "fo.1": _l("Faroe Islands", 1, "Faroe Islands Premier League", None, None, True),
    "al.1": _l("Albania", 1, "Kategoria Superiore"),
    "ad.1": _l("Andorra", 1, "Primera Divisió"),
    "am.1": _l("Armenia", 1, "Armenian Premier League"),
    "az.1": _l("Azerbaijan", 1, "Azerbaijan Premier League"),
    "ba.1": _l("Bosnia-Herzegovina", 1, "Premijer Liga BiH"),
    "bg.1": _l("Bulgaria", 1, "First League (Bulgaria)"),
    "hr.1": _l("Croatia", 1, "SuperSport HNL", None, "KR1"),
    "cy.1": _l("Cyprus", 1, "Cypriot First Division"),
    "gi.1": _l("Gibraltar", 1, "Gibraltar Football League"),
    "kos.1": _l("Kosovo", 1, "Football Superleague of Kosovo"),
    "lu.1": _l("Luxembourg", 1, "BGL Ligue"),
    "mt.1": _l("Malta", 1, "Maltese Premier League"),
    "md.1": _l("Moldova", 1, "Super Liga (Moldova)"),
    "me.1": _l("Montenegro", 1, "Montenegrin First League"),
    "mk.1": _l("North Macedonia", 1, "Macedonian First League"),
    "nir.1": _l("Northern Ireland", 1, "NIFL Premiership"),
    "rs.1": _l("Serbia", 1, "Serbian SuperLiga", None, "SER1"),
    "sk.1": _l("Slovakia", 1, "Slovak First League"),
    "si.1": _l("Slovenia", 1, "Slovenian PrvaLiga"),
    "sm.1": _l("San Marino", 1, "Campionato Sammarinese"),
    "ua.1": _l("Ukraine", 1, "Ukrainian Premier League", None, "UKR1"),
    "wal.1": _l("Wales", 1, "Cymru Premier"),
    "ro.1": _l("Romania", 1, "Romanian SuperLiga", None, "RO1"),
    # outside Europe (football.json only)
    "ar.1": _l("Argentina", 1, "Liga Profesional (Argentina)", None, "ARG1"),  # snapshot ARG1 = apertura only
    "au.1": _l("Australia", 1, "A-League Men", ("Australia", "New Zealand"), "AUS1"),
    "br.1": _l("Brazil", 1, "Campeonato Brasileiro Série A", None, "BRA1", True),
    "br.2": _l("Brazil", 2, "Campeonato Brasileiro Série B", None, None, True),
    "cn.1": _l("China", 1, "Chinese Super League", None, None, True),
    "co.1": _l("Colombia", 1, "Primera A (Colombia)", None, None, True),
    "jp.1": _l("Japan", 1, "J1 League", None, "JAP1", True),
    "mls": _l("United States", 1, "Major League Soccer", ("United States", "Canada"), "MLS1", True),
    "mx.1": _l("Mexico", 1, "Liga MX", None, "MEX1"),  # snapshot MEX1 = clausura only
    "eg.1": _l("Egypt", 1, "Egyptian Premier League"),
    "ma.1": _l("Morocco", 1, "Botola Pro"),
    "dz.1": _l("Algeria", 1, "Ligue Professionnelle 1 (Algeria)"),
}
# Season-specific league names (start-year ranges, inclusive).
LEAGUE_NAME_BY_SEASON = {
    "sco.1": [(2000, 2012, "Scottish Premier League"), (2013, 2100, "Scottish Premiership")],
    "en.5": [(2000, 2014, "Conference Premier"), (2015, 2100, "National League")],
    "at.2": [(2000, 2017, "Erste Liga"), (2018, 2100, "2. Liga")],
    "be.1": [(2000, 2015, "Belgian Pro League"), (2016, 2021, "First Division A"), (2022, 2100, "Belgian Pro League")],
}
# Snapshot competition id -> openfootball code (tier-1 only; used for snapshot rows' names and alias learning)
SNAP_TO_CODE = {v["snap"]: k for k, v in LEAGUES.items() if v["snap"]}

# football.json codes (file stem) are the canonical codes above, except these spellings.
JSON_CODE_FIX = {"mls": "mls"}

# Upstream Football.TXT file -> code (only used where no football.json file exists for the league-season).
TXT_REPOS = ("england", "deutschland", "espana", "italy", "europe")
TXT_PATTERNS = {
    "england": [(r"^(\d{4}-\d{2})/1-premierleague\.txt$", "en.1"), (r"^(\d{4}-\d{2})/2-championship\.txt$", "en.2"),
                (r"^(\d{4}-\d{2})/3-league1\.txt$", "en.3"), (r"^(\d{4}-\d{2})/4-league2\.txt$", "en.4"),
                (r"^(\d{4}-\d{2})/5-nationalleague\.txt$", "en.5")],
    "deutschland": [(r"^(\d{4}-\d{2})/1-bundesliga\.txt$", "de.1"), (r"^(\d{4}-\d{2})/2-bundesliga2\.txt$", "de.2"),
                    (r"^(\d{4}-\d{2})/3-liga3\.txt$", "de.3"),
                    (r"^(\d{4}-\d{2})/4-regionalliga-(bayern|nord|nordost|suedwest|west)\.txt$", "de.4.{1}")],
    "espana": [(r"^(\d{4}-\d{2})/1-liga\.txt$", "es.1"), (r"^(\d{4}-\d{2})/2-liga2\.txt$", "es.2")],
    "italy": [(r"^(\d{4}-\d{2})/1-seriea\.txt$", "it.1"), (r"^(\d{4}-\d{2})/2-serieb\.txt$", "it.2"),
              (r"^(\d{4}-\d{2})/3-seriec_([abc])\.txt$", "it.3.{1}")],
    "europe": [(r"^[a-z-]+/(\d{4}(?:-\d{2})?)_([a-z]+)(\d)(s?)\.txt$", "{1}.{2}{3}")],
}
# Football.TXT files also downloaded where football.json exists, only to validate the TXT parser (never used as rows).
TXT_PARSER_VALIDATION = [("england", "2012-13/3-league1.txt"), ("deutschland", "2014-15/3-liga3.txt"),
                         ("espana", "2013-14/2-liga2.txt"), ("europe", "netherlands/2020-21_nl1.txt"),
                         ("italy", "2015-16/2-serieb.txt"), ("europe", "france/2016-17_fr2.txt")]


def league_name(code: str, season: int) -> str:
    for lo, hi, name in LEAGUE_NAME_BY_SEASON.get(code, []):
        if lo <= season <= hi:
            return name
    return LEAGUES[code]["name"]


def season_from_label(label: str) -> int:
    """'2015-16' -> 2015; calendar-year '2025' -> 2024 (Transfermarkt season id convention)."""
    if re.fullmatch(r"\d{4}-\d{2}", label):
        return int(label[:4])
    if re.fullmatch(r"\d{4}", label):
        return int(label) - 1
    raise ValueError(label)


# --------------------------------------------------------------------------------------------------------------------
# Download (idempotent, cached, rate-limited, manifest)
# --------------------------------------------------------------------------------------------------------------------
class ServerRefused(RuntimeError):
    pass


_last_request: dict[str, float] = {}
_blocked_hosts: set[str] = set()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"generated_by": "src/analysis/league_tiers.py", "user_agent": USER_AGENT, "files": {}}


def save_manifest(man: dict) -> None:
    man["files"] = dict(sorted(man["files"].items()))
    MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")


def register(man: dict, url: str, local: Path, terms: str, retrieved_at: str | None = None, status: int = 200):
    rel = str(local.relative_to(ROOT))
    old = man["files"].get(rel, {})
    man["files"][rel] = {"url": url, "local_path": rel, "sha256": _sha256(local), "bytes": local.stat().st_size,
                         "retrieved_at": old.get("retrieved_at") or retrieved_at or
                         dt.datetime.fromtimestamp(local.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds"),
                         "http_status": old.get("http_status", status), "terms": terms}


def fetch(url: str, local: Path, man: dict, terms: str, offline: bool = False) -> Path | None:
    """Download url to local unless it already exists. Returns None on 404. Raises ServerRefused on 403/429."""
    if local.exists():
        register(man, url, local, terms)
        return local
    if offline:
        raise FileNotFoundError(f"offline and not cached: {local}")
    host = urllib.parse.urlparse(url).netloc
    if host in _blocked_hosts:
        raise ServerRefused(host)
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request.get(host, -1e9))
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    _last_request[host] = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            _blocked_hosts.add(host)
            man.setdefault("refusals", []).append({"url": url, "status": e.code, "at": _now()})
            raise ServerRefused(f"{host} returned {e.code}") from e
        if e.code == 404:
            man.setdefault("not_found", []).append({"url": url, "at": _now()})
            return None
        raise
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(local)
    register(man, url, local, terms, retrieved_at=_now())
    return local


def _tree_paths(repo: str) -> list[str]:
    t = json.loads((OF_DIR / "trees" / f"{repo}.json").read_text())
    assert not t.get("truncated"), repo
    return [x["path"] for x in t["tree"] if x["type"] == "blob"]


def plan_downloads() -> pd.DataFrame:
    """List of external league files to use, from the cached GitHub trees (one tree request per repository)."""
    rows = []
    for path in _tree_paths("football.json"):
        m = re.fullmatch(r"(\d{4}(?:-\d{2})?)/([a-z0-9.]+)\.json", path)
        if not m:
            continue
        label, code = m.groups()
        if code not in LEAGUES:
            continue
        season = season_from_label(label)
        if not SEASON_MIN <= season <= SEASON_MAX:
            continue
        rows.append({"format": "json", "repo": "football.json", "path": path, "code": code, "season_label": label,
                     "season": season, "role": "data",
                     "url": f"https://raw.githubusercontent.com/openfootball/football.json/master/{path}",
                     "local": OF_DIR / "json" / path})
    have = {(r["code"], r["season"]) for r in rows}
    for repo in TXT_REPOS:
        for path in _tree_paths(repo):
            for pat, code_t in TXT_PATTERNS[repo]:
                m = re.fullmatch(pat, path)
                if not m:
                    continue
                g = m.groups()
                label = g[0]
                code = code_t.format(*g)
                if code not in LEAGUES:
                    continue
                season = season_from_label(label)
                if not SEASON_MIN <= season <= SEASON_MAX:
                    continue
                if (repo, path) in TXT_PARSER_VALIDATION:
                    role = "parser_validation"
                elif (code, season) in have:
                    continue
                else:
                    role = "data"
                rows.append({"format": "txt", "repo": repo, "path": path, "code": code, "season_label": label,
                             "season": season, "role": role,
                             "url": f"https://raw.githubusercontent.com/openfootball/{repo}/master/{path}",
                             "local": OF_DIR / "txt" / repo / path})
    df = pd.DataFrame(rows)
    dup = df[df.role == "data"].duplicated(["code", "season"], keep=False)
    assert not dup.any(), df[df.role == "data"][dup]
    return df.sort_values(["code", "season", "format"]).reset_index(drop=True)


def download(offline: bool = False) -> pd.DataFrame:
    man = load_manifest()
    man["user_agent"] = USER_AGENT
    man["sources"] = {"openfootball": OPENFOOTBALL_TERMS, "football-data.co.uk": FOOTBALL_DATA_TERMS,
                      "api.github.com": GITHUB_API_TERMS}
    # metadata captured once (trees, licences, terms pages)
    for repo in ("football.json",) + TXT_REPOS:
        fetch(f"https://api.github.com/repos/openfootball/{repo}/git/trees/master?recursive=1",
              OF_DIR / "trees" / f"{repo}.json", man, GITHUB_API_TERMS, offline)
        fetch(f"https://raw.githubusercontent.com/openfootball/{repo}/master/LICENSE.md",
              OF_DIR / f"{repo}_LICENSE.md", man, OPENFOOTBALL_TERMS, offline)
    fetch("https://raw.githubusercontent.com/openfootball/football.json/master/README.md",
          OF_DIR / "football.json_README.md", man, OPENFOOTBALL_TERMS, offline)
    for name, url in (("robots.txt", "https://www.football-data.co.uk/robots.txt"),
                      ("disclaimer.php.html", "https://www.football-data.co.uk/disclaimer.php"),
                      ("notes.txt", "https://www.football-data.co.uk/notes.txt")):
        fetch(url, FD_DIR / "terms" / name, man, FOOTBALL_DATA_TERMS, offline)
    for repo in ("football.json",) + TXT_REPOS:
        lic = (OF_DIR / f"{repo}_LICENSE.md").read_text()
        assert "CC0 1.0 Universal" in lic, f"{repo} licence is not CC0"
    plan = plan_downloads()
    ok = []
    try:
        for r in plan.itertuples():
            p = fetch(r.url, r.local, man, OPENFOOTBALL_TERMS, offline)
            ok.append(p is not None)
    finally:
        save_manifest(man)
    plan["downloaded"] = ok + [False] * (len(plan) - len(ok))
    return plan


# --------------------------------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------------------------------
_SCORE = r"(?:\d+\s*-\s*\d+(?:\s*\(\d+\s*-\s*\d+\))?(?:\s*(?:a\.?e\.?t\.?|n\.?\s?V\.?|i\.?\s?E\.?|pen\.?))?" \
         r"(?:\s*\d+\s*-\s*\d+\s*(?:pen\.?|i\.?\s?E\.?))?|-)"


def parse_txt_matches(text: str) -> list[tuple[str, str]]:
    """Return (team1, team2) for every fixture line in a Football.TXT league file."""
    out = []
    for raw in text.splitlines():
        s = raw.split("#", 1)[0] if raw.lstrip().startswith("#") else raw
        s = s.strip()
        if not s or s[0] in "=▪[" or s.lower().startswith(("matchday", "round", "group", "date", "teams", "matches")):
            continue
        s = re.sub(r"\[[^\]]*\]", " ", s)            # notes like [cancelled]
        s = re.sub(r"\s+@\s+.*$", "", s)               # venue
        s = re.sub(r"^\d{1,2}[:.h]\d{2}\s+", "", s)    # kick-off time
        s = s.strip()
        m = re.fullmatch(r"(?P<t1>.+?)\s+v\s+(?P<t2>.+?)(?:\s+" + _SCORE + r")?(?:\s+(?:awd\.?|abd\.?|n/p))?", s)
        if m is None:
            m = re.fullmatch(r"(?P<t1>.+?)\s{2,}" + _SCORE + r"\s{2,}(?P<t2>.+?)", s) or \
                re.fullmatch(r"(?P<t1>.+?)\s+" + _SCORE + r"\s+(?P<t2>\D.+?)", s)
        if m is None:
            continue
        t1, t2 = m.group("t1").strip(), m.group("t2").strip()
        if re.search(r"\d+\s*-\s*\d+", t1 + t2) or len(t1) < 2 or len(t2) < 2:
            continue
        # A date line ("Fri Aug 28 2020") never contains ' v ' or a score, so it cannot reach here.
        out.append((t1, t2))
    return out


def parse_json_fixtures(obj: dict) -> list[tuple[str, str, str]]:
    """(date, team1, team2) of every dated fixture in a football.json file."""
    ms = list(obj.get("matches", []))
    for rd in obj.get("rounds", []) or []:
        ms.extend(rd.get("matches", []))
    out = []
    for m in ms:
        t1, t2, d = m.get("team1"), m.get("team2"), m.get("date")
        t1 = t1.get("name") if isinstance(t1, dict) else t1
        t2 = t2.get("name") if isinstance(t2, dict) else t2
        if t1 and t2 and d:
            out.append((d, t1.strip(), t2.strip()))
    return out


def parse_json_matches(obj: dict) -> list[tuple[str, str]]:
    ms = list(obj.get("matches", []))
    for rd in obj.get("rounds", []) or []:
        ms.extend(rd.get("matches", []))
    out = []
    for m in ms:
        t1, t2 = m.get("team1"), m.get("team2")
        if isinstance(t1, dict):
            t1 = t1.get("name")
        if isinstance(t2, dict):
            t2 = t2.get("name")
        if t1 and t2:
            out.append((t1.strip(), t2.strip()))
    return out


def teams_from_matches(matches: list[tuple[str, str]]) -> pd.DataFrame:
    """Teams with their fixture counts. A team with fewer than half the median count of fixtures is flagged
    (play-off / relegation-decider participants from another division, or teams removed early)."""
    cnt: dict[str, int] = defaultdict(int)
    for a, b in matches:
        cnt[a] += 1
        cnt[b] += 1
    df = pd.DataFrame({"external_team_name": list(cnt), "n_fixtures": list(cnt.values())})
    if df.empty:
        return df.assign(member=pd.Series(dtype=bool))
    med = df.n_fixtures.median()
    df["member"] = df.n_fixtures >= 0.5 * med
    return df


def parse_files(plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (teams, files, fixtures). teams: one row per (code, season, external_team_name) from data files;
    fixtures: dated fixtures of football.json data files (used for fixture pairing with snapshot games)."""
    teams, files, fixtures = [], [], []
    for r in plan.itertuples():
        p = Path(r.local)
        if not p.exists():
            files.append({"code": r.code, "season": r.season, "file": str(p.relative_to(ROOT)), "role": r.role,
                          "status": "missing"})
            continue
        text = p.read_text(encoding="utf-8")
        header_teams = None
        if r.format == "json":
            obj = json.loads(text)
            matches = parse_json_matches(obj)
            if r.role == "data":
                fixtures += [(r.code, r.season, d, a, b) for d, a, b in parse_json_fixtures(obj)]
        else:
            matches = parse_txt_matches(text)
            m = re.search(r"^#\s*Teams\s+(\d+)", text, flags=re.M)
            header_teams = int(m.group(1)) if m else None
        t = teams_from_matches(matches)
        n_member = int(t.member.sum()) if len(t) else 0
        status = "ok"
        if not len(t):
            status = "no_fixtures_parsed"
        elif header_teams is not None and header_teams != n_member:
            status = f"team_count_mismatch_header_{header_teams}"
        files.append({"code": r.code, "season": r.season, "season_label": r.season_label, "format": r.format,
                      "file": str(p.relative_to(ROOT)), "url": r.url, "role": r.role, "n_fixtures": len(matches),
                      "n_teams_parsed": len(t), "n_teams_member": n_member, "header_teams": header_teams,
                      "status": status})
        if r.role == "data" and len(t):
            t = t.assign(code=r.code, season=r.season, season_label=r.season_label, format=r.format,
                         file=str(p.relative_to(ROOT)), file_status=status)
            teams.append(t)
    # A club cannot play in two divisions of one country in the same season: if two data files of different codes
    # for one country-season share >= 3 team names, the files are not trustworthy (e.g. a second-tier file that is a
    # copy of the first tier) and both are dropped.
    files_df = pd.DataFrame(files)
    if teams:
        tt = pd.concat(teams, ignore_index=True)
        tt = tt[tt.member]
        tt["country"] = tt.code.map(lambda c: LEAGUES[c]["country"])
        bad = {}
        for (ctry, season), g in tt.groupby(["country", "season"]):
            sets = {c: set(x.external_team_name) for c, x in g.groupby("code")}
            codes = sorted(sets)
            for i, a in enumerate(codes):
                for b in codes[i + 1:]:
                    ov = sets[a] & sets[b]
                    if len(ov) >= 3:
                        bad[(a, season)] = f"cross_division_overlap_with_{b}_{len(ov)}_teams"
                        bad[(b, season)] = f"cross_division_overlap_with_{a}_{len(ov)}_teams"
        for k, v in bad.items():
            m = (files_df.code == k[0]) & (files_df.season == k[1]) & (files_df.role == "data")
            files_df.loc[m, "status"] = v
        for t in teams:
            k = (t.code.iloc[0], t.season.iloc[0])
            if k in bad:
                t["file_status"] = bad[k]
    files = files_df
    fx = pd.DataFrame(fixtures, columns=["code", "season", "date", "team1", "team2"])
    return pd.concat(teams, ignore_index=True), files, fx


# --------------------------------------------------------------------------------------------------------------------
# Name normalisation and reserve/youth detection
# --------------------------------------------------------------------------------------------------------------------
_CHAR_MAP = str.maketrans({"ß": "ss", "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "ł": "l", "Ł": "l", "đ": "d",
                           "Đ": "d", "ı": "i", "œ": "oe", "Œ": "oe", "þ": "th", "Þ": "th", "ð": "d", "’": "'",
                           "`": "'", "´": "'"})
# Trailing tokens that mark a reserve (second) team of a club, and the youth markers. Never stripped silently: they
# become a class suffix of the key, so "Bayern München II" can only equal another "... II"-class name.
RESERVE_TRAILING = {"ii": "res", "2": "res", "iii": "res3", "b": "res", "j": "res", "castilla": "res",
                    "atletic": "res", "atletico": "res", "reserves": "res", "reserve": "res", "res": "res",
                    "futuro": "res", "promesas": "res", "yth": "youth", "youth": "youth", "academy": "youth",
                    "primavera": "youth", "jgd": "youth"}
# Not markers: a trailing "C." abbreviates Central/City (Barracas C.), "Juniors" is part of senior names
# (Boca Juniors, Argentinos Juniors).
GENERIC_TOKENS = {"fc", "afc", "cf", "sc", "sv", "ac", "ssc", "ss", "us", "as", "asd", "cd", "ud", "sd", "rcd", "fk",
                  "sk", "bk", "if", "ff", "tsv", "vfl", "vfb", "fsv", "tsg", "spvgg", "ssv", "sg", "club", "calcio",
                  "football", "futbol", "futebol", "de", "the", "and", "cfc", "kv", "krc", "rsc", "rc", "royal", "nk",
                  "hnk", "pfc", "ofk", "sfc", "bsc", "vv", "sbv", "ec", "se", "cr", "ca", "fcs", "aff", "ik"}
_TOKEN_SUB = {"utd": "united", "saint": "st", "ste": "st", "munich": "munchen"}
# Reserve sides whose name carries no generic marker.
RESERVE_NAMES = {("bilbao", "athletic"): "res"}
AGE_SUFFIX = {"15", "16", "17", "18", "19", "20", "21", "23"}  # Athletic Club reserve side; base tokens kept


def _ascii(s: str) -> str:
    s = s.translate(_CHAR_MAP)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def tokens(name: str) -> list[str]:
    s = _ascii(name).lower().replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [_TOKEN_SUB.get(t, t) for t in s.split()]


def reserve_class(name: str) -> tuple[str | None, list[str]]:
    """(class, base tokens). class None = senior team; 'res' = reserve/B/II/Jong team; 'u19' etc = age group."""
    toks = tokens(name)
    cls = None
    base = list(toks)
    # age groups anywhere: U19, U-21, Sub-23, Under 23
    for i, t in enumerate(toks):
        m = re.fullmatch(r"[uoys](\d{2})", t)  # U19 / Dutch O19 / Portuguese Y19, S17
        if m:
            cls = f"u{m.group(1)}"
            base = toks[:i] + toks[i + 1:]
            break
        if t in ("sub", "under") and i + 1 < len(toks) and re.fullmatch(r"\d{2}", toks[i + 1]):
            cls = f"u{toks[i + 1]}"
            base = toks[:i] + toks[i + 2:]
            break
        if t == "u" and i + 1 < len(toks) and re.fullmatch(r"\d{2}", toks[i + 1]):
            cls = f"u{toks[i + 1]}"
            base = toks[:i] + toks[i + 2:]
            break
    if toks[:2] == ["willem", "ii"]:  # Willem II (Tilburg) is a senior club named after King Willem II
        c2, b2 = reserve_class(" ".join(toks[2:])) if len(toks) > 2 else (None, [])
        return c2, toks[:2] + b2
    if cls is None and len(toks) >= 2 and toks[-1] in AGE_SUFFIX and not toks[-2].isdigit():
        cls, base = f"u{toks[-1]}", toks[:-1]  # TM youth sides such as "MVV 19", "Cova Piedade 17"
    if cls is None and tuple(toks) in RESERVE_NAMES:
        cls = RESERVE_NAMES[tuple(toks)]
    elif cls is None and len(toks) >= 2:
        if toks[0] == "jong":
            cls, base = "res", toks[1:]
        elif toks[-2:] == ["next", "gen"]:
            cls, base = "res", toks[:-2]
        elif toks[-1] in RESERVE_TRAILING and not (toks[-1] == "atletico" and toks[0] in ("club", "atletico")):
            cls, base = RESERVE_TRAILING[toks[-1]], toks[:-1]
    return cls, base


def is_reserve_name(name: str) -> bool:
    return reserve_class(name)[0] is not None


def norm_keys(name: str) -> tuple[str, str]:
    """(strict, loose) exact-match keys. strict: accents/punctuation/case folded; loose: additionally drops legal-form
    tokens (FC, AFC, SV, VfB, ...) and 1-2 digit numbers (Schalke 04, Hannover 96). Reserve class kept as suffix."""
    cls, base = reserve_class(name)
    suffix = f" #{cls}" if cls else ""
    strict = " ".join(base)
    loose_toks = [t for t in base if t not in GENERIC_TOKENS and not re.fullmatch(r"\d{1,2}", t)]
    loose = " ".join(loose_toks) or strict
    return strict + suffix, loose + suffix


def normalise_name(name: str) -> str:
    return norm_keys(name)[0]


# --------------------------------------------------------------------------------------------------------------------
# Manual alias table. Each entry: (league country, external name exactly as in openfootball, TM club id, a TM name of
# that id that must exist in the universe -> guards against typos in the id, note). Added only where the identity is
# unambiguous (same club, renamed or abbreviated). Checked in build(): the id must exist, carry the stated TM name,
# be in the league's country family and have the same reserve status as the external name.
# --------------------------------------------------------------------------------------------------------------------
DUP = "other candidate is a Transfermarkt youth/academy id with the same name (its episodes are players aged <=16)"
MANUAL_ALIASES: list[tuple[str, str, int, str, str]] = [
    # Austria
    ("Austria", "Austria Lustenau", 541, "A. Lustenau", "SC Austria Lustenau; TM abbreviation"),
    ("Austria", "Austria Salzburg", 11853, "A. Salzburg", "SV Austria Salzburg (not Red Bull Salzburg); TM abbreviation"),
    ("Austria", "FC Trenkwalder Admira", 503, "Admira Wacker", "sponsor name of FC Admira Wacker Mödling 2010-12"),
    ("Austria", "FC Wacker Innsbruck", 2444, "FC Wacker", "FC Wacker Innsbruck; TM short name 'FC Wacker'"),
    ("Austria", "FC Wacker Innsbruck II", 18344, "FC Wacker II", "reserve side of FC Wacker Innsbruck"),
    ("Austria", "KSV Superfund", 2394, "SV Kapfenberg", "sponsor name of Kapfenberger SV 2010/11"),
    ("Austria", "Kapfenberger SV 1919", 2394, "SV Kapfenberg", "Kapfenberger SV"),
    ("Austria", "LASK Linz", 413, "LASK", "LASK (Linzer ASK)"),
    ("Austria", "SC Magna Wiener Neustadt", 17444, "Wiener Neustadt", "sponsor name of SC Wiener Neustadt"),
    ("Austria", "1. Wiener Neustädter SC", 17444, "Wiener Neustadt", "SC Wiener Neustadt (openfootball's later name)"),
    ("Austria", "SV Stripfing/Weiden", 20296, "SV Stripfing", "SV Stripfing (Weiden an der March)"),
    ("Austria", "Schwarz-Weiß Bregenz", 936, "SW Bregenz", "SC Schwarz-Weiß Bregenz; TM abbreviation"),
    # England
    ("England", "Halifax Town", 1127, "FC Halifax Town", "FC Halifax Town (founded 2008); other candidate 62966 is "
     "Halifax Town AFC, dissolved 2008 (episodes 2005-08 only)"),
    # France
    ("France", "Chamois Niortais", 1161, "Chamois Niort", "Chamois Niortais FC"),
    ("France", "FC Chambly", 35012, "FC Chambly Oise", "FC Chambly Oise"),
    ("France", "Grenoble Foot 38", 1290, "Grenoble", "Grenoble Foot 38"),
    ("France", "Havre AC", 738, "Le Havre AC", "Le Havre AC"),
    ("France", "US Créteil-Lusitanos", 2067, "US Créteil-L", "US Créteil-Lusitanos; TM abbreviation"),
    # Germany
    ("Germany", "1. FC Magdeburg", 187, "1.FC Magdeburg", DUP + " (17587)"),
    ("Germany", "Rot-Weiss Essen", 56, "Rot-Weiss Essen", DUP + " (11877)"),
    ("Germany", "Hessen Kassel", 373, "KSV Hessen Kassel", "KSV Hessen Kassel"),
    # Italy
    ("Italy", "Ascoli Picchio FC", 408, "Ascoli", "Ascoli Picchio FC 1898 / Ascoli Calcio 1898"),
    ("Italy", "L.R. Vicenza", 2655, "LR Vicenza", "LR Vicenza"),
    ("Italy", "L.R. Vicenza Virtus", 2655, "LR Vicenza", "LR Vicenza Virtus (club name 2018-21)"),
    ("Italy", "Mantova 1911 SSD", 2581, "Mantova 1911", "Mantova 1911"),
    ("Italy", "Reggina 1914", 1386, "Reggina", "Reggina 1914 (club name 2016-23)"),
    # Netherlands
    ("Netherlands", "MVV", 384, "MVV Maastricht", "MVV Maastricht"),
    ("Netherlands", "Roda JC Kerkrade", 192, "Roda JC Kerkrade", DUP + " (32802)"),
    # Portugal
    ("Portugal", "FC Felgueiras 1932", 1701, "Felgueiras", "FC Felgueiras 1932"),
    ("Portugal", "Sporting Covilhã", 10140, "SC Covilhã", "Sporting Clube da Covilhã"),
    ("Portugal", "União Leiria", 2639, "Leiria", "União de Leiria"),
    ("Portugal", "União Torreense", 2432, "Torreense", "SCU Torreense"),
    # Spain
    ("Spain", "Athletic Club B", 6688, "Bilbao Athletic", "Bilbao Athletic = Athletic Club's reserve side"),
    ("Spain", "CF Fuenlabrada", 16486, "CF Fuenlabrada", DUP + " (16620)"),
    ("Spain", "Cultural Leonesa", 4542, "Cultural Leonesa", "other candidate 41165 has one episode, a 17-year-old "
     "moving to Real Madrid U18 (academy id)"),
    # Switzerland
    ("Switzerland", "FC Thun Berner Oberland", 938, "FC Thun", "FC Thun (name from 2023)"),
    ("Switzerland", "FC Baden", 747, "FC Baden 1897", "FC Baden 1897"),
    # Türkiye
    ("Türkiye", "Altay Izmir", 2375, "Altay SK", "Altay SK (İzmir)"),
    ("Türkiye", "Menemen Belediyespor", 20664, "Menemen Belediye", "Menemen Belediyespor"),
]


# --------------------------------------------------------------------------------------------------------------------
# Transfermarkt club universe (master + snapshot)
# --------------------------------------------------------------------------------------------------------------------
def load_master() -> pd.DataFrame:
    cols = ["event_id", "from_club_id", "from_club_name", "to_club_id", "to_club_name", "from_country_v2",
            "to_country_v2", "is_youth_or_reserve_side", "transfer_date", "transfer_season", "season", "kind"]
    return pd.read_csv(MASTER_PATH, usecols=cols, low_memory=False)


def connect():
    import duckdb
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


def index_universe(names: pd.DataFrame, countries: dict, parent: dict | None = None,
                   display: dict | None = None) -> dict:
    """Exact-match index over TM name variants. names: club_id, name (one row per variant)."""
    names = names.copy()
    keys = names["name"].map(norm_keys)
    names["strict"] = keys.str[0]
    names["loose"] = keys.str[1]
    names["cls"] = names["name"].map(lambda n: reserve_class(n)[0])
    reserve = names.groupby("club_id").cls.agg(lambda s: s.notna().any()).to_dict()
    classes = names.dropna(subset=["cls"]).groupby("club_id").cls.agg(set).to_dict()
    index: dict[tuple, set] = defaultdict(set)
    keys_by_id: dict[int, set] = defaultdict(set)
    for r in names.itertuples(index=False):
        keys_by_id[r.club_id].update({("strict", r.strict), ("loose", r.loose)})
        for k in countries.get(r.club_id, ()):
            index[(k, "strict", r.strict)].add(r.club_id)
            index[(k, "loose", r.loose)].add(r.club_id)
    if display is None:
        display = names.drop_duplicates("club_id").set_index("club_id")["name"].to_dict()
    return {"names": names, "countries": countries, "reserve": reserve, "classes": classes, "index": index,
            "keys_by_id": keys_by_id, "parent": parent or {}, "display": display}


def tm_universe(con, master: pd.DataFrame) -> dict:
    names = []
    for side in ("from", "to"):
        d = master[[f"{side}_club_id", f"{side}_club_name"]].dropna()
        names.append(pd.DataFrame({"club_id": d.iloc[:, 0].astype(int), "name": d.iloc[:, 1], "src": "master"}))
    t = con.sql("select from_club_id a, from_club_name b from transfers union select to_club_id, to_club_name "
                "from transfers").df().dropna()
    names.append(pd.DataFrame({"club_id": t.a.astype(int), "name": t.b, "src": "snapshot_transfers"}))
    c = con.sql("select club_id, name, domestic_competition_id from clubs").df()
    names.append(pd.DataFrame({"club_id": c.club_id.astype(int), "name": c.name, "src": "snapshot_clubs"}))
    g = con.sql("""select g.home_club_id a, g.home_club_name b, c.country_name k, c.type ty from games g
                   join competitions c using(competition_id) where g.home_club_id is not null
                   union select g.away_club_id, g.away_club_name, c.country_name, c.type from games g
                   join competitions c using(competition_id) where g.away_club_id is not null""").df()
    gn = g.dropna(subset=["b"])
    names.append(pd.DataFrame({"club_id": gn.a.astype(int), "name": gn.b, "src": "snapshot_games"}))
    names = pd.concat(names).drop_duplicates(["club_id", "name"]).reset_index(drop=True)

    countries: dict[int, set] = defaultdict(set)
    geo = pd.read_csv(GEO_PATH, usecols=["club_id", "calendar_country", "league_country"])
    for col in ("calendar_country", "league_country"):
        for cid, k in geo[["club_id", col]].dropna().itertuples(index=False):
            countries[int(cid)].add(k)
    for side in ("from", "to"):
        for cid, k in master[[f"{side}_club_id", f"{side}_country_v2"]].dropna().itertuples(index=False):
            countries[int(cid)].add(k)
    comp = con.sql("select competition_id, country_name from competitions").df().set_index("competition_id")
    for cid, dc in c[["club_id", "domestic_competition_id"]].dropna().itertuples(index=False):
        if dc in comp.index and pd.notna(comp.loc[dc, "country_name"]):
            countries[int(cid)].add(comp.loc[dc, "country_name"])
    for cid, k in g[g.ty.isin(["domestic_league", "domestic_cup"])][["a", "k"]].dropna().itertuples(index=False):
        countries[int(cid)].add(k)

    disp = names[names.src == "snapshot_clubs"].drop_duplicates("club_id").set_index("club_id")["name"].to_dict()
    for side in ("from", "to"):
        d = master[[f"{side}_club_id", f"{side}_club_name"]].dropna()
        mf = d.groupby(f"{side}_club_id")[f"{side}_club_name"].agg(lambda s: s.value_counts().index[0])
        for cid, n in mf.items():
            disp.setdefault(int(cid), n)
    for cid, n in names.drop_duplicates("club_id")[["club_id", "name"]].itertuples(index=False):
        disp.setdefault(int(cid), n)
    org = pd.read_csv(ORG_PATH, usecols=["child_club_id", "parent_club_id", "confidence"])
    org = org[org.confidence.isin(["high", "medium"]) & org.parent_club_id.notna()]
    parent = {int(a): int(b) for a, b in zip(org.child_club_id, org.parent_club_id)}
    tm = index_universe(names, countries, parent, disp)
    master_ids = set(pd.concat([master.from_club_id, master.to_club_id]).dropna().astype(int))
    snap_ids = set(c.club_id.astype(int)) | set(g.a.dropna().astype(int))
    tm.update({"master_ids": master_ids, "snapshot_ids": snap_ids})
    return tm


def lookup(name: str, family: tuple, tm: dict, level: str) -> set:
    k = norm_keys(name)[0 if level == "strict" else 1]
    out = set()
    for c in family:
        out |= tm["index"].get((c, level, k), set())
    return out


# --------------------------------------------------------------------------------------------------------------------
# Snapshot tier-1 memberships
# --------------------------------------------------------------------------------------------------------------------
SNAP_LEAGUE_NAME_OVERRIDE = {"ARG1": "Liga Profesional (Argentina), Torneo Apertura",
                             "MEX1": "Liga MX, Clausura", "RSK1": "K League 1", "SA1": "Saudi Pro League"}


def snapshot_league_games(con) -> pd.DataFrame:
    d = con.sql("""select competition_id, cast(season as int) season, cast(date as varchar) date,
                   home_club_id::int home, away_club_id::int away from games
                   where competition_type='domestic_league' and home_club_id is not null
                   and away_club_id is not null""").df()
    return d


def snapshot_tier1(con) -> pd.DataFrame:
    d = con.sql("""
        with x as (select competition_id, season, home_club_id club_id from games where competition_type='domestic_league'
                   union all select competition_id, season, away_club_id from games where competition_type='domestic_league')
        select x.competition_id, cast(x.season as int) season, x.club_id, count(*) n_games, c.country_name, c.name
        from x join competitions c using(competition_id) where x.club_id is not null group by all""").df()
    d["club_id"] = d.club_id.astype(int)
    med = d.groupby(["competition_id", "season"]).n_games.transform("median")
    d["member"] = d.n_games >= 0.5 * med
    return d


# --------------------------------------------------------------------------------------------------------------------
# Alias learning from tier-1 league-seasons present in both openfootball and the snapshot
# --------------------------------------------------------------------------------------------------------------------
def _restricted_candidates(name: str, ids: set, tm: dict) -> set:
    strict, loose = norm_keys(name)
    hit = {i for i in ids if ("strict", strict) in tm["keys_by_id"].get(i, ())}
    if not hit:
        hit = {i for i in ids if ("loose", loose) in tm["keys_by_id"].get(i, ())}
    return hit


FIXTURE_MIN_EVIDENCE = 3


def _fixture_candidates(e: str, fx: pd.DataFrame, games: pd.DataFrame, name_to_id: dict) -> list[int]:
    """For external name e: for each of its dated fixtures whose opponent is already paired, the snapshot game of the
    same league-season on the same date with that opponent on the same home/away side names e's club id."""
    out = []
    by = games.groupby(["date", "home"]).away.apply(list).to_dict()
    by_a = games.groupby(["date", "away"]).home.apply(list).to_dict()
    for d, t1, t2 in fx[["date", "team1", "team2"]].itertuples(index=False):
        if t1 == e and t2 in name_to_id:
            out += by_a.get((d, name_to_id[t2]), [])
        elif t2 == e and t1 in name_to_id:
            out += by.get((d, name_to_id[t1]), [])
    return out


def learn_aliases(ext: pd.DataFrame, snap: pd.DataFrame, tm: dict, fixtures: pd.DataFrame | None = None,
                  snap_games: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair external names with snapshot club ids in linked tier-1 league-seasons.

    Per league-season: an external name is paired when exactly one still-unpaired snapshot club of that league-season
    has the same normalised key (restricted exact match), or by elimination when exactly one external name and one
    snapshot club remain unpaired and both sets have the same size, or by fixture pairing (at least
    FIXTURE_MIN_EVIDENCE of its dated fixtures against already-paired opponents coincide with snapshot games of that
    league-season on the same date, same opponent, same home/away side, all naming one club). Pairs propagate across the seasons of the same
    league until a fixpoint. An alias is kept only if, in every linked season where the name appears, its club id is
    a member of the snapshot league-season, and no two names of one season share an id.
    Returns (aliases, linked league-season table)."""
    ext = ext[ext.member]
    snapm = snap[snap.member]
    linked = []
    for (code, season), e in ext.groupby(["code", "season"]):
        comp = LEAGUES[code]["snap"]
        if not comp:
            continue
        t = snapm[(snapm.competition_id == comp) & (snapm.season == season)]
        if len(t):
            linked.append((code, season, sorted(e.external_team_name), set(t.club_id)))
    known: dict[tuple, int] = {}
    how: dict[tuple, tuple] = {}
    changed = True
    while changed:
        changed = False
        for code, season, E, T in linked:
            c = LEAGUES[code]["country"]
            assigned = {e: known[(c, e)] for e in E if (c, e) in known}
            remT = T - set(assigned.values())
            for e in E:
                if e in assigned:
                    continue
                cand = _restricted_candidates(e, remT, tm)
                if len(cand) == 1:
                    i = cand.pop()
                    known[(c, e)] = i
                    how[(c, e)] = ("restricted_exact", code, season)
                    assigned[e] = i
                    remT.discard(i)
                    changed = True
            remE = [e for e in E if e not in assigned]
            if len(remE) == 1 and len(remT) == 1 and len(E) == len(T):
                known[(c, remE[0])] = next(iter(remT))
                how[(c, remE[0])] = ("elimination", code, season)
                changed = True
                continue
            if remE and fixtures is not None and snap_games is not None:
                fx = fixtures[(fixtures.code == code) & (fixtures.season == season)]
                gm = snap_games[(snap_games.competition_id == LEAGUES[code]["snap"]) & (snap_games.season == season)]
                if fx.empty or gm.empty:
                    continue
                for e in remE:
                    ev = _fixture_candidates(e, fx, gm, assigned)
                    if len(ev) >= FIXTURE_MIN_EVIDENCE and len(set(ev)) == 1 and ev[0] in remT:
                        known[(c, e)] = ev[0]
                        how[(c, e)] = ("fixture_pairing", code, season)
                        assigned[e] = ev[0]
                        remT.discard(ev[0])
                        changed = True
    # consistency checks
    bad: dict[tuple, str] = {}
    for code, season, E, T in linked:
        c = LEAGUES[code]["country"]
        ids = defaultdict(list)
        for e in E:
            if (c, e) in known:
                ids[known[(c, e)]].append(e)
                if known[(c, e)] not in T:
                    bad[(c, e)] = f"id {known[(c, e)]} not in snapshot {code} {season}"
        for i, es in ids.items():
            if len(es) > 1:
                for e in es:
                    bad[(c, e)] = f"id {i} shared by {es} in {code} {season}"
    rows = []
    for (c, e), i in known.items():
        is_res_e = is_reserve_name(e)
        if is_res_e != tm["reserve"].get(i, False):
            bad.setdefault((c, e), "reserve/senior mismatch")
        seasons = sorted({s for code, s, E, T in linked if LEAGUES[code]["country"] == c and e in E})
        rows.append({"country": c, "external_team_name": e, "club_id": i, "club_name_tm": tm["display"].get(i),
                     "learned_by": how[(c, e)][0], "learned_in": f"{how[(c, e)][1]} {how[(c, e)][2]}",
                     "linked_seasons": " ".join(map(str, seasons)), "status": "ok" if (c, e) not in bad else "rejected",
                     "reject_reason": bad.get((c, e), "")})
    al = pd.DataFrame(rows).sort_values(["country", "external_team_name"]).reset_index(drop=True)
    link_tab = []
    for code, season, E, T in linked:
        c = LEAGUES[code]["country"]
        got = {known[(c, e)] for e in E if (c, e) in known and (c, e) not in bad}
        link_tab.append({"code": code, "season": season, "n_external": len(E), "n_snapshot": len(T),
                         "n_paired": len(got), "fully_paired": len(got) == len(E) == len(T)})
    return al, pd.DataFrame(link_tab)


# --------------------------------------------------------------------------------------------------------------------
# Matching every external team name
# --------------------------------------------------------------------------------------------------------------------
def check_manual_aliases(tm: dict) -> dict[tuple, int]:
    out = {}
    for country, ext_name, cid, tm_name, note in MANUAL_ALIASES:
        assert cid in tm["display"], f"manual alias {ext_name}: club id {cid} not in universe"
        tm_names = set(tm["names"][tm["names"].club_id == cid].name)
        assert tm_name in tm_names, f"manual alias {ext_name}: {cid} has no TM name {tm_name!r} ({tm_names})"
        fam = next(v["family"] for v in LEAGUES.values() if v["country"] == country)
        assert tm["countries"].get(cid, set()) & set(fam), f"manual alias {ext_name}: {cid} not in {fam}"
        assert is_reserve_name(ext_name) == tm["reserve"].get(cid, False), f"manual alias {ext_name}: reserve mismatch"
        assert (country, ext_name) not in out, f"duplicate manual alias {ext_name}"
        out[(country, ext_name)] = cid
    return out


def match_name(name: str, code: str, tm: dict, manual: dict, learned: dict) -> dict:
    lg = LEAGUES[code]
    c, fam = lg["country"], lg["family"]
    raw_name = name
    ext_res = is_reserve_name(name)
    res: dict = {"club_id": None, "match_method": None, "unmatched_reason": None, "candidates": ""}
    # Strict key first; the loose key is used only when the strict key matches nothing. Either level must point to
    # exactly one club id, otherwise the name is ambiguous.
    exact, level_used = set(), None
    for level in ("strict", "loose"):
        cand = {i for i in lookup(name, fam, tm, level) if tm["reserve"].get(i, False) == ext_res}
        if cand:
            exact, level_used = cand, level
            break
    if (c, raw_name) in manual:
        i = manual[(c, raw_name)]
        if len(exact) == 1 and i not in exact:
            raise AssertionError(f"manual alias {name} -> {i} contradicts unique exact match {exact}")
        return {**res, "club_id": i, "match_method": "alias_manual"}
    if (c, raw_name) in learned:
        i = learned[(c, raw_name)]
        if len(exact) == 1 and i not in exact:
            return {**res, "unmatched_reason": f"conflict: learned alias {i} vs exact {sorted(exact)}",
                    "candidates": " ".join(map(str, sorted(exact | {i})))}
        return {**res, "club_id": i, "match_method": "alias_learned"}
    if len(exact) == 1:
        return {**res, "club_id": next(iter(exact)), "match_method": "exact_normalised", "key_level": level_used}
    if len(exact) > 1:
        return {**res, "unmatched_reason": f"ambiguous: {len(exact)} TM clubs with this normalised name in {'/'.join(fam)}",
                "candidates": " ".join(map(str, sorted(exact)))}
    # derived reserve alias: "<learned/known senior name> II" -> the TM reserve club of that senior club
    cls, base = reserve_class(name)
    if cls is not None:
        base_name = " ".join(base)
        senior = None
        for (cc, en), i in list(learned.items()) + list(manual.items()):
            if cc == c and not is_reserve_name(en) and (norm_keys(en)[0] == norm_keys(base_name)[0] or
                                                         norm_keys(en)[1] == norm_keys(base_name)[1]):
                senior = i if senior in (None, i) else -1
        if senior == -1:  # two different learned/manual seniors share the base name: give up
            senior = None
        elif senior is None:
            for lvl in ("strict", "loose"):
                s = {i for i in lookup(base_name, fam, tm, lvl) if not tm["reserve"].get(i, False)}
                if s:
                    senior = next(iter(s)) if len(s) == 1 else None
                    break
        if senior is not None:
            sk = {k for lvl, k in tm["keys_by_id"][senior] if "#" not in k}
            cands = set()
            for k in sk:
                for lvl in ("strict", "loose"):
                    for cc in fam:
                        cands |= tm["index"].get((cc, lvl, f"{k} #{cls}"), set())
            # reserve sides attached to that senior club by the audited organisation map (high/medium confidence)
            org = {r for r, par in tm["parent"].items() if par == senior}
            same = {i for i in cands | org if i != senior and cls in tm["classes"].get(i, set())}
            if not same and cls == "res":
                # Transfermarkt labels some second teams by age group (Jong Ajax = "Ajax U21", Atalanta U23 in
                # Serie C): accept a U21/U23 side of the same senior club, only if it is the unique one.
                same = {i for i in org if i != senior and tm["classes"].get(i, set()) & {"u21", "u23"}}
            cands = same
            if len(cands) == 1:
                return {**res, "club_id": next(iter(cands)), "match_method": "alias_learned",
                        "key_level": f"reserve_of_{senior}"}
            if len(cands) > 1:
                return {**res, "unmatched_reason": f"ambiguous reserve side of {senior}",
                        "candidates": " ".join(map(str, sorted(cands)))}
    # diagnose
    opp = set()
    for level in ("strict", "loose"):
        opp |= {i for i in lookup(name, fam, tm, level) if tm["reserve"].get(i, False) != ext_res}
        k = norm_keys(" ".join(base) if cls else name)[0 if level == "strict" else 1]
        for cc in fam:
            for kk in ([k] if cls else [f"{k} #res", f"{k} #res3"]):
                opp |= {i for i in tm["index"].get((cc, level, kk), set()) if tm["reserve"].get(i, False) != ext_res}
    if opp:
        return {**res, "unmatched_reason": "reserve/senior mismatch only", "candidates": " ".join(map(str, sorted(opp)))}
    return {**res, "unmatched_reason": f"no TM club with this normalised name in {'/'.join(fam)}"}


# --------------------------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------------------------
TABLE_COLS = ["club_id", "club_name_tm", "season", "country", "league_name", "tier", "source", "source_league_code",
              "external_team_name", "match_method", "n_sources_agreeing", "sources", "external_season_label",
              "source_file"]


def match_external(ext: pd.DataFrame, tm: dict, manual: dict, learned: dict) -> pd.DataFrame:
    rows = []
    cache: dict[tuple, dict] = {}
    for r in ext.itertuples(index=False):
        base = {"source": SRC_OF, "source_league_code": r.code, "season": r.season, "season_label": r.season_label,
                "external_team_name": r.external_team_name, "n_fixtures": r.n_fixtures, "file": r.file,
                "country": LEAGUES[r.code]["country"], "tier": LEAGUES[r.code]["tier"]}
        if r.file_status != "ok":
            rows.append({**base, "club_id": None, "match_method": None, "candidates": "",
                         "unmatched_reason": f"file failed validation ({r.file_status})"})
            continue
        if not r.member:
            rows.append({**base, "club_id": None, "match_method": None, "candidates": "",
                         "unmatched_reason": "not a member of this division (fewer than half the median fixtures in "
                                             "file: play-off/decider participant)"})
            continue
        key = (r.external_team_name, r.code)
        if key not in cache:
            cache[key] = match_name(r.external_team_name, r.code, tm, manual, learned)
        rows.append({**base, **cache[key]})
    a = pd.DataFrame(rows)
    # one TM club cannot be two teams of one league-season
    m = a.club_id.notna()
    dup = a[m].duplicated(["source_league_code", "season", "club_id"], keep=False)
    di = a[m][dup].index
    a.loc[di, "unmatched_reason"] = "two external names of this league-season matched the same TM club " + \
        a.loc[di, "club_id"].astype("Int64").astype(str)
    a.loc[di, "candidates"] = a.loc[di, "club_id"].astype("Int64").astype(str)
    a.loc[di, ["club_id", "match_method"]] = None
    a["club_id"] = a.club_id.astype("Int64")
    a["club_name_tm"] = a.club_id.map(lambda i: tm["display"].get(int(i)) if pd.notna(i) else None)
    return a


def combine(audit: pd.DataFrame, snap: pd.DataFrame, tm: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    of = audit[audit.club_id.notna()].copy()
    of = pd.DataFrame({"club_id": of.club_id.astype(int), "season": of.season, "country": of.country,
                       "league_name": [league_name(c, s) for c, s in zip(of.source_league_code, of.season)],
                       "tier": of.tier, "source": SRC_OF, "source_league_code": of.source_league_code,
                       "league_key": of.source_league_code, "external_team_name": of.external_team_name,
                       "match_method": of.match_method, "external_season_label": of.season_label,
                       "source_file": of.file})
    sm = snap[snap.member]
    sn = pd.DataFrame({"club_id": sm.club_id, "season": sm.season, "country": sm.country_name,
                       "league_name": [SNAP_LEAGUE_NAME_OVERRIDE.get(c) or
                                       (league_name(SNAP_TO_CODE[c], s) if c in SNAP_TO_CODE else n)
                                       for c, s, n in zip(sm.competition_id, sm.season, sm.name)],
                       "tier": 1, "source": SRC_SNAPSHOT, "source_league_code": sm.competition_id,
                       "league_key": [SNAP_TO_CODE.get(c, c) for c in sm.competition_id],
                       "external_team_name": None, "match_method": "snapshot_games",
                       "external_season_label": None, "source_file": "data/raw/transfermarkt-datasets.duckdb:games"})
    allr = pd.concat([sn, of], ignore_index=True)
    g = allr.groupby(["club_id", "season"])
    n_tier = g.tier.transform("nunique")
    n_league = g.league_key.transform("nunique")
    bad = (n_tier > 1) | (n_league > 1)
    conflicts = allr[bad].sort_values(["club_id", "season", "source"]).copy()
    conflicts["conflict_type"] = (n_tier[bad] > 1).map({True: "different tiers", False: "same tier, different league"})
    ok = allr[~bad].copy()
    ok["src_rank"] = ok.source.map({SRC_SNAPSHOT: 0, SRC_OF: 1})
    ok = ok.sort_values(["club_id", "season", "src_rank"])
    agg = ok.groupby(["club_id", "season"]).agg(
        n_sources_agreeing=("source", "nunique"), sources=("source", lambda s: ";".join(sorted(set(s)))),
        ext_name=("external_team_name", lambda s: next((x for x in s if pd.notna(x)), None)),
        ext_method=("match_method", lambda s: next((x for x in s if x != "snapshot_games"), None)),
        ext_label=("external_season_label", lambda s: next((x for x in s if pd.notna(x)), None)),
        ext_code=("source_league_code", lambda s: ";".join(dict.fromkeys(s))),
        n_rows=("source", "size"))
    first = ok.drop_duplicates(["club_id", "season"]).set_index(["club_id", "season"])
    t = first.join(agg)
    t["external_team_name"] = t.ext_name
    t["source_league_code"] = t.ext_code
    t["external_season_label"] = t.ext_label
    t = t.reset_index()
    t["club_name_tm"] = t.club_id.map(tm["display"])
    t["tier"] = t.tier.astype(int)
    t["season"] = t.season.astype(int)
    return t[TABLE_COLS].sort_values(["club_id", "season"]).reset_index(drop=True), conflicts.drop(columns="league_key")


def league_season_coverage(audit: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    a = audit.copy()
    a["is_member_row"] = ~a.unmatched_reason.fillna("").str.startswith("not a member")
    g = a.groupby(["source", "source_league_code", "season"])
    cov = pd.DataFrame({
        "external_season_label": g.season_label.first(), "country": g.country.first(), "tier": g.tier.first(),
        "teams_in_file": g.is_member_row.sum(),
        "teams_matched": g.club_id.apply(lambda s: s.notna().sum()),
        "excluded_non_members": g.is_member_row.apply(lambda s: (~s).sum()),
        "file": g.file.first()}).reset_index()
    cov["league_name"] = [league_name(c, s) for c, s in zip(cov.source_league_code, cov.season)]
    cov["teams_unmatched"] = cov.teams_in_file - cov.teams_matched
    cov["unmatched_names"] = [
        "; ".join(sorted(a[(a.source_league_code == c) & (a.season == s) & a.club_id.isna() & a.is_member_row]
                         .external_team_name))
        for c, s in zip(cov.source_league_code, cov.season)]
    fs = files[files.role == "data"].set_index(["code", "season"])[["format", "status"]]
    cov = cov.join(fs, on=["source_league_code", "season"])
    return cov[["source", "source_league_code", "season", "external_season_label", "country", "league_name", "tier",
                "format", "status", "teams_in_file", "teams_matched", "teams_unmatched", "excluded_non_members",
                "unmatched_names", "file"]]


def build(offline: bool = False, do_download: bool = True) -> dict:
    plan = download(offline) if do_download else plan_downloads()
    ext, files, fixtures = parse_files(plan)
    con = connect()
    master = load_master()
    tm = tm_universe(con, master)
    snap = snapshot_tier1(con)
    snap_games = snapshot_league_games(con)
    aliases, linked = learn_aliases(ext[ext.file_status == "ok"], snap, tm, fixtures, snap_games)
    learned = {(r.country, r.external_team_name): int(r.club_id) for r in aliases[aliases.status == "ok"].itertuples()}
    manual = check_manual_aliases(tm)
    audit = match_external(ext, tm, manual, learned)
    table, conflicts = combine(audit, snap, tm)
    cov = league_season_coverage(audit, files)
    return {"plan": plan, "files": files, "ext": ext, "tm": tm, "snap": snap, "aliases": aliases, "linked": linked,
            "audit": audit, "table": table, "conflicts": conflicts, "coverage": cov, "master": master}


# --------------------------------------------------------------------------------------------------------------------
# Coverage on the master and outputs
# --------------------------------------------------------------------------------------------------------------------
FOCUS = (2012, 2025)


def master_coverage(master: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["season_int"] = m.season.astype(str).str[:4].astype(int)
    tier = table.set_index(["club_id", "season"]).tier
    rows = []
    for side in ("from", "to"):
        ids = m[f"{side}_club_id"]
        idx = pd.MultiIndex.from_arrays([ids.fillna(-1).astype(int), m.season_int])
        m[f"{side}_tier"] = tier.reindex(idx).to_numpy()
        m[f"{side}_class"] = m[f"{side}_club_name"].fillna("").map(
            lambda n: "senior" if reserve_class(n)[0] is None else
            ("reserve" if reserve_class(n)[0] in ("res", "res3") else "youth"))
    for scope, mask in (("all_seasons", pd.Series(True, index=m.index)),
                        ("2012-2025", m.season_int.between(*FOCUS))):
        mm = m[mask]
        for side in ("from", "to"):
            ctry = mm[f"{side}_country_v2"].fillna("(unknown)")
            has = mm[f"{side}_tier"].notna()
            for c, grp in has.groupby(ctry):
                rows.append({"scope": scope, "side": side, "country": c, "episodes": len(grp),
                             "with_tier": int(grp.sum()), "share": grp.mean()})
            rows.append({"scope": scope, "side": side, "country": "ALL", "episodes": len(has),
                         "with_tier": int(has.sum()), "share": has.mean()})
            for cl, grp in has.groupby(mm[f"{side}_class"]):
                rows.append({"scope": scope, "side": side, "country": f"ALL ({cl} sides)", "episodes": len(grp),
                             "with_tier": int(grp.sum()), "share": grp.mean()})
    out = pd.DataFrame(rows)
    out["share"] = out.share.round(4)
    return out


# Hand-picked spot checks: (club_id, season, expected tier, expected league code, why we believe it).
# The expected values are public football history; build() asserts the table agrees.
SPOT_CHECKS: list[tuple[int, int, int, str, str]] = [
    (350, 2018, 2, "en.2", "Sheffield United were Championship runners-up in 2018/19 (promoted)"),
    (350, 2019, 1, "GB1", "...and played the 2019/20 Premier League: same club, different tier one season later"),
    (41, 2018, 2, "de.2", "Hamburger SV's first relegation (May 2018) put them in the 2. Bundesliga 2018/19"),
    (1112, 2022, 5, "en.5", "Wrexham (a Welsh club in the English pyramid) won the National League in 2022/23"),
    (1112, 2024, 3, "en.3", "Wrexham's second straight promotion: League One 2024/25"),
    (17, 2021, 3, "de.3", "Borussia Dortmund II (reserve side) were promoted to the 3. Liga for 2021/22"),
    (8817, 2022, 2, "nl.2", "Jong Ajax (TM 'Ajax U21') play in the Eerste Divisie"),
    (2464, 2012, 2, "es.2", "FC Barcelona B played the Segunda División 2012/13"),
    (124, 2016, 1, "SC1", "Rangers returned to the Scottish Premiership in 2016/17 (2012/13-2015/16 in lower "
                          "divisions, which have no verified source, so those seasons are absent)"),
    (371, 2015, 1, "SC1", "Celtic, Scottish Premiership 2015/16"),
    (2, 2019, 3, "de.3", "1. FC Kaiserslautern played the 3. Liga from 2018/19 to 2021/22"),
    (142, 2013, 2, "es.2", "Real Zaragoza were relegated from LaLiga in 2013 and played the Segunda in 2013/14"),
    (289, 2018, 3, "en.3", "Sunderland's second consecutive relegation: League One 2018/19"),
    (826, 2019, 2, "fr.2", "RC Lens won promotion from Ligue 2 in 2019/20..."),
    (826, 2020, 1, "FR1", "...and played Ligue 1 in 2020/21"),
    (130, 2017, 2, "it.2", "Parma, refounded in 2015, reached Serie B for 2017/18"),
]


def run_spot_checks(table: pd.DataFrame) -> pd.DataFrame:
    t = table.set_index(["club_id", "season"])
    rows = []
    for cid, season, tier, code, why in SPOT_CHECKS:
        got = t.loc[(cid, season)] if (cid, season) in t.index else None
        rows.append({"club_id": cid, "club_name_tm": None if got is None else got.club_name_tm, "season": season,
                     "expected_tier": tier, "expected_code": code,
                     "table_tier": None if got is None else int(got.tier),
                     "table_league": None if got is None else got.league_name,
                     "table_code": None if got is None else got.source_league_code,
                     "table_external_name": None if got is None else got.external_team_name,
                     "table_method": None if got is None else got.match_method,
                     "ok": got is not None and int(got.tier) == tier and code in str(got.source_league_code),
                     "why": why})
    return pd.DataFrame(rows)


def parser_validation(plan: pd.DataFrame | None = None) -> pd.DataFrame:
    """Football.TXT parser check: for league-seasons where both the generated football.json and the upstream
    Football.TXT file were downloaded, the member team sets must be identical."""
    plan = plan_downloads() if plan is None else plan
    rows = []
    for r in plan[plan.role == "parser_validation"].itertuples():
        j = plan[(plan.code == r.code) & (plan.season == r.season) & (plan.format == "json")]
        if j.empty or not Path(r.local).exists() or not Path(j.iloc[0].local).exists():
            continue
        tt = teams_from_matches(parse_txt_matches(Path(r.local).read_text(encoding="utf-8")))
        tj = teams_from_matches(parse_json_matches(json.loads(Path(j.iloc[0].local).read_text(encoding="utf-8"))))
        st, sj = set(tt[tt.member].external_team_name), set(tj[tj.member].external_team_name)
        rows.append({"code": r.code, "season": r.season, "txt_file": str(Path(r.local).relative_to(ROOT)),
                     "n_txt": len(st), "n_json": len(sj), "identical": st == sj,
                     "only_txt": "; ".join(sorted(st - sj)), "only_json": "; ".join(sorted(sj - st))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------------
# README (all numbers interpolated from the build)
# --------------------------------------------------------------------------------------------------------------------
PRIORITY_DEPTH = {"England": 5, "Scotland": 4, "Germany": 4, "Spain": 3, "Italy": 3, "France": 3, "Netherlands": 2,
                  "Portugal": 2, "Belgium": 2, "Türkiye": 2, "Greece": 2, "Austria": 2, "Switzerland": 2}


def _ranges(seasons) -> str:
    s = sorted(set(int(x) for x in seasons))
    out, start, prev = [], None, None
    for x in s:
        if start is None:
            start = prev = x
        elif x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    if start is not None:
        out.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in out)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def numbers(r: dict, mcov: pd.DataFrame, spots: pd.DataFrame, pv: pd.DataFrame) -> dict:
    t, a, cov = r["table"], r["audit"], r["coverage"]
    member = ~a.unmatched_reason.fillna("").str.startswith(("not a member", "file failed"))
    um = a[member & a.club_id.isna()]
    reason = um.unmatched_reason.str.extract(r"^(ambiguous reserve|ambiguous|reserve/senior|no TM club|conflict|two "
                                             r"external)")[0].fillna("other")
    g = lambda scope, side, c: mcov[(mcov.scope == scope) & (mcov.side == side) & (mcov.country == c)].iloc[0]
    return {
        "rows": len(t),
        "rows_by_source_tier": {f"{s}|tier{k}": int(v) for (s, k), v in t.groupby(["source", "tier"]).size().items()},
        "rows_by_method": t.match_method.value_counts().to_dict(),
        "rows_two_sources": int((t.n_sources_agreeing >= 2).sum()),
        "files_used": int((r["files"].role == "data").sum()),
        "files_json": int(((r["files"].role == "data") & (r["files"].format == "json")).sum()),
        "files_txt": int(((r["files"].role == "data") & (r["files"].format == "txt")).sum()),
        "files_not_ok": int(((r["files"].role == "data") & (r["files"].status != "ok")).sum()),
        "files_not_ok_detail": {f"{c} {sn}": st for c, sn, st in r["files"][(r["files"].role == "data") &
                                (r["files"].status != "ok")][["code", "season", "status"]].itertuples(index=False)},
        "league_seasons": len(cov),
        "external_member_names": int(member.sum()),
        "external_matched": int((member & a.club_id.notna()).sum()),
        "external_unmatched": int(len(um)),
        "unmatched_by_reason": reason.value_counts().to_dict(),
        "non_member_rows": int(a.unmatched_reason.fillna("").str.startswith("not a member").sum()),
        "distinct_unmatched_names": int(um.drop_duplicates(["source_league_code", "external_team_name"])
                                        .pipe(lambda d: d.groupby("country").external_team_name.nunique().sum())),
        "conflict_club_seasons": int(r["conflicts"][["club_id", "season"]].drop_duplicates().shape[0]),
        "conflict_types": r["conflicts"].drop_duplicates(["club_id", "season"]).conflict_type.value_counts().to_dict(),
        "aliases_learned_ok": int((r["aliases"].status == "ok").sum()),
        "aliases_rejected": int((r["aliases"].status != "ok").sum()),
        "aliases_by_rule": r["aliases"][r["aliases"].status == "ok"].learned_by.value_counts().to_dict(),
        "linked_league_seasons": len(r["linked"]),
        "linked_fully_paired": int(r["linked"].fully_paired.sum()),
        "manual_aliases": len(MANUAL_ALIASES),
        "spot_checks": len(spots), "spot_checks_ok": int(spots.ok.sum()),
        "parser_validation_files": len(pv), "parser_validation_identical": int(pv.identical.sum()) if len(pv) else 0,
        "master_rows": len(r["master"]),
        "coverage": {f"{scope}|{side}": {"episodes": int(g(scope, side, "ALL").episodes),
                                         "with_tier": int(g(scope, side, "ALL").with_tier),
                                         "share": float(g(scope, side, "ALL").share)}
                     for scope in ("all_seasons", "2012-2025") for side in ("from", "to")},
    }


def write_readme(r: dict, n: dict, mcov: pd.DataFrame, spots: pd.DataFrame, pv: pd.DataFrame) -> str:
    t, cov, files = r["table"], r["coverage"], r["files"]
    man = load_manifest()
    nfiles = len(man["files"])
    nbytes = sum(v["bytes"] for v in man["files"].values())
    refusals = man.get("refusals", [])
    # tier definitions per league code
    defs = []
    for code, g in cov.groupby("source_league_code"):
        lg = LEAGUES[code]
        names = sorted({league_name(code, s) for s in g.season})
        defs.append({"code": code, "country": lg["country"], "tier": lg["tier"], "league": " / ".join(names),
                     "clubs from": "/".join(lg["family"]),
                     "seasons (start year)": _ranges(g.season),
                     "calendar-year": "yes" if lg["calendar"] else ""})
    snap_seasons = r["snap"].groupby("competition_id").season.agg(_ranges)
    for comp, ss in snap_seasons.items():
        code = SNAP_TO_CODE.get(comp)
        defs.append({"code": comp, "country": r["snap"][r["snap"].competition_id == comp].country_name.iloc[0],
                     "tier": 1, "league": SNAP_LEAGUE_NAME_OVERRIDE.get(comp) or
                     (LEAGUES[code]["name"] if code else comp), "clubs from": "snapshot games",
                     "seasons (start year)": ss, "calendar-year": ""})
    defs = pd.DataFrame(defs).sort_values(["country", "tier", "code"])
    # gaps for priority countries, 2012-2025
    have = set()
    for code, s in zip(cov.source_league_code, cov.season):
        have.add((LEAGUES[code]["country"], LEAGUES[code]["tier"], int(s)))
    for comp, s in r["snap"][["competition_id", "season"]].drop_duplicates().itertuples(index=False):
        have.add((r["snap"][r["snap"].competition_id == comp].country_name.iloc[0], 1, int(s)))
    gaps = []
    for c, depth in PRIORITY_DEPTH.items():
        for tier in range(1, depth + 1):
            miss = [s for s in range(FOCUS[0], FOCUS[1] + 1) if (c, tier, s) not in have]
            if miss:
                gaps.append({"country": c, "tier": tier, "seasons with no verified source": _ranges(miss)})
    gaps = pd.DataFrame(gaps)
    # coverage tables
    def covtab(scope):
        d = mcov[mcov.scope == scope]
        f = d[d.side == "from"].set_index("country")
        to = d[d.side == "to"].set_index("country")
        x = f[["episodes", "with_tier", "share"]].join(to[["episodes", "with_tier", "share"]], how="outer",
                                                       lsuffix="_from", rsuffix="_to").fillna(0)
        x = x.sort_values("episodes_from", ascending=False)
        top = [i for i in x.index if not i.startswith("ALL")][:25]
        x = x.loc[[i for i in x.index if i.startswith("ALL")] + top]
        return pd.DataFrame({"club country": x.index, "episodes (from)": x.episodes_from.astype(int),
                             "verified tier (from)": x.share_from.map(_pct),
                             "episodes (to)": x.episodes_to.astype(int),
                             "verified tier (to)": x.share_to.map(_pct)})
    cbr = n["unmatched_by_reason"]
    unl = r["audit"]
    member = ~unl.unmatched_reason.fillna("").str.startswith(("not a member", "file failed"))
    worst = cov[cov.tier <= 5].sort_values("teams_unmatched", ascending=False)
    worst = worst[worst.teams_unmatched > 0].head(15)[["source_league_code", "season", "teams_in_file",
                                                         "teams_matched", "unmatched_names"]]
    sp = spots[["club_name_tm", "club_id", "season", "table_league", "table_tier", "table_code", "table_method",
                "ok", "why"]]
    st = n["rows_by_source_tier"]
    tier_tab = t.groupby(["tier", "source"]).size().unstack(fill_value=0)
    tier_tab = tier_tab.reset_index().rename(columns={"tier": "tier"})
    cov_all = n["coverage"]
    txt = f"""# Club → league / tier lookup (season-specific)

Generated by `python -m src.analysis.league_tiers` (module `src/analysis/league_tiers.py`); every number below is
computed by that run. Re-running with `--offline` uses only the cached downloads.

## Files

| file | content |
|---|---|
| `club_season_league.csv` | one row per verified (club_id, season): {n['rows']:,} rows |
| `name_match_audit.csv` | every (source, league code, season, external team name) with the matched TM club id or the reason it is unmatched ({len(r['audit']):,} rows) |
| `league_season_coverage.csv` | per external league-season: teams in file, matched, unmatched ({len(cov):,} league-seasons) |
| `club_season_conflicts.csv` | club-seasons where sources disagree (excluded from the table): {n['conflict_club_seasons']} club-seasons |
| `learned_aliases.csv` | external name → TM club id pairs learned from tier-1 league-seasons present in both openfootball and the snapshot |
| `master_coverage_by_country.csv` | coverage of the master's from/to clubs by club country |
| `league_tiers_numbers.json` | the numbers quoted here |

Columns of `club_season_league.csv`: `club_id`, `club_name_tm`, `season` (int start year; 2024 = 2024/25),
`country` (country of the league), `league_name`, `tier` (1 = top division of that country's pyramid), `source`
(the preferred source: `{SRC_SNAPSHOT}` when the snapshot has the club in a first-tier league that season, else
`{SRC_OF}`), `source_league_code` (all agreeing source codes, `;`-separated, e.g. `GB1;en.1`), `external_team_name`
(openfootball spelling, if any), `match_method` (`snapshot_games` when the snapshot row exists, else how the
openfootball name was matched: `exact_normalised` / `alias_manual` / `alias_learned`), `n_sources_agreeing`
(1 or 2 distinct sources), `sources`, `external_season_label` (label of the openfootball file, e.g. `2015-16` or
`2025`), `source_file`.

Rows by tier and source (`source` = preferred source of the row):

{_md(tier_tab)}

Rows confirmed by both sources: {n['rows_two_sources']:,}. Caveat: in tier-1 league-seasons covered by both the
snapshot and openfootball, many openfootball names were matched with aliases learned *from that snapshot league*, so
agreement there is a consistency check of the alias, not independent evidence.

## Sources and terms

* **Transfermarkt snapshot** (`data/raw/transfermarkt-datasets.duckdb`, read-only, already in the repo): `games` of
  the 31 `domestic_league` competitions (all `first_tier`). A club is a member of a league-season if it played at least
  half the median number of games of that league-season. Tier 1 by construction. Transfermarkt itself was not
  contacted (its terms forbid scraping).
* **openfootball** — {OPENFOOTBALL_TERMS} Used: `football.json` files ({n['files_json']} league-season files) and,
  only where no football.json file exists for a league-season, the upstream Football.TXT files of the same project
  ({n['files_txt']} files, parsed here). The TXT parser was validated on {n['parser_validation_files']} league-seasons
  that exist in both formats: identical team sets in {n['parser_validation_identical']} of
  {n['parser_validation_files']}. A file is dropped when its parsed team count disagrees with its own `# Teams` header,
  or when it shares 3 or more teams with another division's file of the same country-season (a club cannot play in
  two divisions at once). Dropped: {n['files_not_ok']} ({', '.join(f'{k}: {v}' for k, v in n['files_not_ok_detail'].items()) or 'none'}).
* **football-data.co.uk** — {FOOTBALL_DATA_TERMS}
* Downloads: User-Agent `{USER_AGENT}` (no e-mail), at most one request per {MIN_INTERVAL_S} s per host, cached under
  `data/raw/external_leagues/<source>/…`, never re-downloaded. `data/raw/external_leagues/manifest.json` lists
  {nfiles} files ({nbytes / 1e6:.1f} MB) with URL, local path, sha256, bytes, retrieval time and terms note.
  HTTP 403/429 refusals: {len(refusals)}. GitHub tree listings were requested once per repository via the REST API.

## Season convention

`season` = Transfermarkt season id = start year of a European season (2024 = 2024/25), identical to
`games.season` in the snapshot and to the first four digits of the master's `season` ('2024/25'). For
calendar-year leagues (openfootball label `2025`; Nordic, Baltic, Irish, Brazilian, Chinese, Japanese, MLS, …) the
label year Y is stored as season Y−1, which is Transfermarkt's own convention (checked in the snapshot: NO1, SE1,
BRA1, JAP1, MLS1 season 2024 games are all dated in calendar 2025; and openfootball `se.1 2025` pairs with snapshot
`SE1 2024`). Argentina's `ar.1` files labelled `2025` likewise map to 2024 (snapshot ARG1 2024 = Apertura 2025).

Note for the join: for a move in the summer window the club's league in the *season of the move* already reflects
the promotion/relegation that just happened; the previous season's league is the row with `season − 1`.

## Matching rules (deterministic, no similarity scores)

1. Universe of TM clubs: every club id in the master (both sides) or in the snapshot (`clubs`, `games`), with every
   name variant seen in the master, snapshot `transfers`, `clubs` and `games`. A club's countries are its
   `calendar_country`/`league_country` (club_geography_v2), the master's `from/to_country_v2`, and the country of
   snapshot domestic league/cup games it played.
2. Normalisation: accents folded (ß→ss, ø→o, …), lower-case, `&`→and, punctuation removed, `utd`→united,
   `saint`→st, `munich`→munchen. *Strict key* = that. *Loose key* = strict minus legal-form tokens
   (`{", ".join(sorted(GENERIC_TOKENS))}`) and 1–2 digit numbers (Schalke 04, Hannover 96); 4-digit years are kept.
3. Reserve/youth: a trailing `II`, `III`, `B`, `2`, `J` (openfootball's Dutch Jong), `Castilla`, `Atlètic`,
   `Atlético` (as last word), `Next Gen`, `Futuro`, `Reserves`, a leading `Jong`, the name `Bilbao Athletic`, an age
   group token (`U19`, `U-21`, `Sub-23`, Dutch `O19`, Portuguese `Y19`/`S17`) or a trailing age number 15–23 (TM's
   "MVV 19", "Cova Piedade 17") gives the name a class (`res`, `u19`, …) that is part of the key, so a reserve name
   can only equal a reserve name of the same class. Exception: `Willem II` is a senior club. A TM club is a
   reserve/youth club if any of its names carries a class; an external name and a TM club must have the same status
   for any match (exact, alias, learned).
4. Exact match: within the league's country family (England+Wales for the English pyramid, Switzerland+Liechtenstein,
   France+Monaco, Spain+Andorra, Italy+San Marino, US+Canada for MLS, Australia+New Zealand), the strict key must match
   exactly one club id; only if the strict key matches nothing is the loose key used, and it must also match exactly
   one id (`exact_normalised`). More than one id at the level used → `ambiguous`, unmatched, candidate ids listed.
   (Stripping the Brazilian state suffixes openfootball appends, e.g. "Flamengo RJ", was tried and rejected: it
   turned "Botafogo SP" into Botafogo RJ, which the conflict check caught.)
5. Learned aliases (`alias_learned`): for each tier-1 league-season present in both openfootball and the snapshot
   ({n['linked_league_seasons']} league-seasons; {n['linked_fully_paired']} fully paired), an external name is paired
   with a snapshot club of *that* league-season if it is the only remaining one with the same key (restricted exact
   match), by elimination when exactly one name and one club remain and both sets have equal size, or by fixture
   pairing (football.json files only: at least {FIXTURE_MIN_EVIDENCE} of the name's dated fixtures against
   already-paired opponents coincide with snapshot games of that league-season on the same date, same opponent, same
   home/away side, and all of them name the same club id); pairs propagate
   across seasons to a fixpoint. An alias is kept only if, in every linked season where the name appears, its club is
   in the snapshot league and no two names share a club. Kept: {n['aliases_learned_ok']}
   ({', '.join(f'{k} {v}' for k, v in n['aliases_by_rule'].items())}); rejected: {n['aliases_rejected']}.
   Aliases are keyed by (league country, exact external name) and reused for the same club in lower divisions.
   A learned alias that contradicts a unique exact match is not used (unmatched, reason `conflict`).
6. Derived reserve aliases (`alias_learned`): an external reserve name "X II/B/J/Jong X" whose senior part X is
   matched (learned/manual/unique exact) to senior club A is matched to the unique TM club of the same reserve class
   that either has a name "<a name of A> <marker>" or is attached to A in the audited
   `club_organisation_map.csv` (confidence high/medium). If no side of the same class exists, a unique U21/U23 side of
   A in the organisation map is accepted (Transfermarkt files Jong Ajax as "Ajax U21" and Atalanta's Serie C team as
   "Atalanta U23"). Otherwise unmatched.
7. Manual aliases (`alias_manual`): {n['manual_aliases']} hand-written entries (`MANUAL_ALIASES` in the module), each
   checked at build time: the id exists, carries the stated TM name, is in the league's country family, has the same
   reserve status, and does not contradict a unique exact match.
8. Within one external league-season two names may not match the same club (both become unmatched).
9. Teams appearing in a league file with fewer than half the median number of fixtures (play-off/decider participants
   from another division, placeholders such as `N.N.`) are not members of that division ({n['non_member_rows']} rows).
10. Combining sources per (club_id, season): if sources give different tiers, or the same tier but different leagues
   (e.g. two Regionalliga groups), the club-season is written to `club_season_conflicts.csv` and left out.

## Tier definitions (per league code and season)

Tier = level of the division in that country's league pyramid in that season (1 = top). League names follow the
season (Scottish Premier League until 2012/13, Scottish Premiership from 2013/14; Conference Premier → National
League (tier 5) in 2015/16; Austrian Erste Liga → 2. Liga (tier 2) in 2018/19). Seasons below are those for which a
verified file exists.

{_md(defs)}

## Known gaps (priority countries, seasons 2012/13–2025/26)

No verified source exists for these country-tier-seasons. football-data.co.uk, whose file list includes Scotland
tiers 2–4 (SC1–SC3) and England tier 5 (EC), was not used (see terms); openfootball has no Scottish lower divisions,
no Belgian/Greek second tier and no Spanish/French third tier:

{_md(gaps)}

Other gaps: tiers below 2 for most other countries; most non-European leagues before 2024; Spanish Segunda B /
Primera RFEF (tier 3), Italian Serie C except 2024/25, French National (tier 3); youth/academy sides never play in
these leagues, so youth-side episodes cannot get a tier.

## Name matching results

External member names (league-season × team): {n['external_member_names']:,}; matched {n['external_matched']:,};
unmatched {n['external_unmatched']:,} ({', '.join(f'{k}: {v}' for k, v in cbr.items())}); distinct unmatched
(country, name) pairs: {n['distinct_unmatched_names']}. Conflicting club-seasons: {n['conflict_club_seasons']}{(' (' + ', '.join(f'{k}: {v}' for k, v in n['conflict_types'].items()) + ')') if n['conflict_types'] else ''}.

League-seasons (tier ≤ 5) with the most unmatched names:

{_md(worst)}

## Coverage on the master (`analysis_episode_master.csv`, {n['master_rows']:,} episodes)

Share of episodes whose club has a verified (club_id, season) row, joining on the master's `season` start year.
All seasons: from side {_pct(cov_all['all_seasons|from']['share'])}, to side {_pct(cov_all['all_seasons|to']['share'])}.
Seasons 2012/13–2025/26 ({cov_all['2012-2025|from']['episodes']:,} episodes): from side
{_pct(cov_all['2012-2025|from']['share'])}, to side {_pct(cov_all['2012-2025|to']['share'])}.

Seasons 2012/13–2025/26, by club country (25 largest by from-side episodes; full table in
`master_coverage_by_country.csv`). "ALL (youth sides)" etc. split by whether the club name is a senior, reserve or
youth side:

{_md(covtab('2012-2025'))}

## Spot checks

Asserted by the build and by `tests/test_league_tiers.py` ({n['spot_checks_ok']}/{n['spot_checks']} agree).

{_md(sp)}
"""
    README_MD.write_text(txt)
    return txt


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    r = build(offline=args.offline)
    OUT.mkdir(parents=True, exist_ok=True)
    r["table"].to_csv(TABLE_CSV, index=False)
    audit_cols = ["source", "source_league_code", "season", "season_label", "country", "tier", "external_team_name",
                  "n_fixtures", "club_id", "club_name_tm", "match_method", "key_level", "unmatched_reason",
                  "candidates", "file"]
    a = r["audit"].reindex(columns=audit_cols)
    a.sort_values(["source_league_code", "season", "external_team_name"]).to_csv(AUDIT_CSV, index=False)
    r["coverage"].sort_values(["source_league_code", "season"]).to_csv(COVERAGE_CSV, index=False)
    r["conflicts"].to_csv(CONFLICTS_CSV, index=False)
    r["aliases"].to_csv(ALIASES_CSV, index=False)
    mcov = master_coverage(r["master"], r["table"])
    mcov.to_csv(MASTER_COV_CSV, index=False)
    spots = run_spot_checks(r["table"])
    pv = parser_validation(r["plan"])
    n = numbers(r, mcov, spots, pv)
    NUMBERS_JSON.write_text(json.dumps(n, indent=1, ensure_ascii=False, default=str) + "\n")
    write_readme(r, n, mcov, spots, pv)
    assert spots.ok.all(), spots[~spots.ok].to_string()
    assert not r["table"].duplicated(["club_id", "season"]).any()
    print(json.dumps({k: n[k] for k in ("rows", "rows_by_source_tier", "external_unmatched", "unmatched_by_reason",
                                        "conflict_club_seasons", "coverage")}, indent=1, default=str))
    return r


if __name__ == "__main__":
    main()
