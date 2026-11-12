"""Week 10: extend the season-specific club -> league / tier lookup with English Wikipedia league-season articles.

    python -m src.analysis.league_tiers_wiki --download   # discover + fetch (cached, 1 request/s), then build
    python -m src.analysis.league_tiers_wiki              # build from the cache only (never touches the network)

Why: the Week 9 lookup (src/analysis/league_tiers.py: Transfermarkt snapshot games + openfootball) has no list for
several second divisions of the 14 playing-time countries (Belgium, Greece, Russia, Scotland, Ukraine, most Danish
and Turkish seasons, Dutch and Portuguese seasons before 2020/21) and almost no first-division lists for other
countries before 2024/25. Wikipedia has one article per league-season with the final standings.

Source and terms
  * en.wikipedia.org article pages (/wiki/<title>) only. robots.txt: "Friendly, low-speed bots are welcome viewing
    article pages"; /w/ and /api/ are disallowed for generic agents, so neither is used. One request per second,
    User-Agent `football-contract-research/1.0 (non-commercial academic research)` (no personal e-mail). Text is
    CC BY-SA 4.0; only league membership facts are extracted, and every article URL is recorded for attribution.
  * Cache: data/raw/external_leagues/wikipedia/<sha1 of title>.html.gz, manifest wikipedia_manifest.json (url,
    canonical title, sha256, bytes, retrieval time, HTTP status, terms note).

Method
  1. Discovery: for each league series, fetch seed articles and collect, from every navbox on every fetched page,
     links whose title is "<season> <league name>" for one of the series' name patterns (each with a season window
     and a tier); red links (page does not exist) are recorded as missing, never fetched.
  2. Parsing: standings tables (header with Pos, Team/Club and Pld/P). The largest table is the division; further
     tables of at least 60% of its size and sharing no team with it are groups of the same division (e.g. Super
     League Greece 2 North/South). Smaller or overlapping tables (championship/relegation rounds, promotion
     play-offs with lower-division teams) are ignored.
  3. Matching: src.analysis.league_tiers.match_name (unchanged, deterministic) on the link text and on the linked
     article title; the two must not point to different clubs. Learned aliases: league_tiers.learn_aliases on
     Wikipedia first-division seasons that overlap the snapshot, plus the Week 9 openfootball aliases.
  4. Combination with the Week 9 table (data/processed/league_tiers/club_season_league.csv, read-only): an existing
     club-season always wins; a Wikipedia row that disagrees with it is written to the overlap audit and dropped. A
     club found in two divisions of one country-season by Wikipedia is dropped (conflict).
  5. Completeness: a (country, season, tier) list is complete when some source lists every team of the division and
     every listed team is matched to a Transfermarkt club id. Completeness of tiers 1 AND 2 in a country-season is
     what later allows "not in tier 1 or 2" to be stated for a club of that country (week10_followup).

Writes only data/processed/league_tiers_v2/* and data/raw/external_leagues/wikipedia/*.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import league_tiers as lt

ROOT = lt.ROOT
WIKI_DIR = ROOT / "data" / "raw" / "external_leagues" / "wikipedia"
WIKI_MANIFEST = WIKI_DIR / "wikipedia_manifest.json"
OUT = ROOT / "data" / "processed" / "league_tiers_v2"
OLD_TABLE = lt.TABLE_CSV
OLD_COVERAGE = lt.COVERAGE_CSV
OLD_ALIASES = lt.ALIASES_CSV

USER_AGENT = "football-contract-research/1.0 (non-commercial academic research)"
MIN_INTERVAL_S = 1.0
WIKI_TERMS = ("English Wikipedia article page (/wiki/), CC BY-SA 4.0; only league-membership facts extracted. "
              "robots.txt welcomes friendly low-speed bots on article pages; /w/ and /api/ (disallowed) not used. "
              "One request per second, identifying User-Agent without personal contact data.")
SEASONS = (2005, 2025)          # Transfermarkt season ids (start year) fetched
FOURTEEN = ("England", "Scotland", "Germany", "Spain", "Italy", "France", "Netherlands", "Portugal", "Belgium",
            "Greece", "Türkiye", "Russia", "Ukraine", "Denmark")

# season-label kinds: split "2015–16" -> 2015; calendar "2016" -> 2015 (Transfermarkt convention, as in Week 9);
# apertura "Apertura 2016" -> 2016; clausura "Clausura 2017" -> 2016.
SPLIT = r"(?P<y1>\d{4})–(?P<y2>\d{2}|\d{4})"
CAL = r"(?P<cy>\d{4})"


def P(name_regex: str, tier: int, lo: int = SEASONS[0], hi: int = SEASONS[1], fmt: str = "prefix"):
    return {"re": name_regex, "tier": tier, "lo": lo, "hi": hi, "fmt": fmt}


def S(code, country, tier, seeds, pats, family=None, snap=None, lo=SEASONS[0]):
    return {"code": code, "country": country, "tier": tier, "seeds": seeds, "pats": pats,
            "family": tuple(family or (country,)), "snap": snap, "lo": lo}


UK = ("England", "Wales")
SERIES = [
    # ---- the 14 playing-time countries, tiers 1 and 2 ----
    S("en.1", "England", 1, ["2015–16 Premier League"], [P(r"(?:FA )?Premier League", 1)], UK, "GB1"),
    S("en.2", "England", 2, ["2015–16 Football League Championship"], [P(r"(?:Football League|EFL) Championship", 2)], UK),
    S("sco.1", "Scotland", 1, ["2015–16 Scottish Premiership", "2010–11 Scottish Premier League"],
      [P(r"Scottish Premier(?:ship| League)", 1)], None, "SC1"),
    S("sco.2", "Scotland", 2, ["2015–16 Scottish Championship", "2010–11 Scottish First Division"],
      [P(r"Scottish Championship", 2, 2013), P(r"Scottish First Division", 2, hi=2012)]),
    S("de.1", "Germany", 1, ["2015–16 Bundesliga"], [P(r"Bundesliga", 1)], None, "L1"),
    S("de.2", "Germany", 2, ["2015–16 2. Bundesliga"], [P(r"2\. Bundesliga", 2)]),
    S("es.1", "Spain", 1, ["2015–16 La Liga"], [P(r"La Liga", 1)], ("Spain", "Andorra"), "ES1"),
    S("es.2", "Spain", 2, ["2015–16 Segunda División"], [P(r"Segunda División", 2)], ("Spain", "Andorra")),
    S("it.1", "Italy", 1, ["2015–16 Serie A"], [P(r"Serie A", 1)], None, "IT1"),
    S("it.2", "Italy", 2, ["2015–16 Serie B"], [P(r"Serie B", 2)]),
    S("fr.1", "France", 1, ["2015–16 Ligue 1"], [P(r"Ligue 1", 1)], ("France", "Monaco"), "FR1"),
    S("fr.2", "France", 2, ["2015–16 Ligue 2"], [P(r"Ligue 2", 2)], ("France", "Monaco")),
    S("nl.1", "Netherlands", 1, ["2015–16 Eredivisie"], [P(r"Eredivisie", 1)], None, "NL1"),
    S("nl.2", "Netherlands", 2, ["2015–16 Eerste Divisie"], [P(r"Eerste Divisie", 2)]),
    S("pt.1", "Portugal", 1, ["2015–16 Primeira Liga"], [P(r"Primeira Liga", 1)], None, "PO1"),
    S("pt.2", "Portugal", 2, ["2015–16 Segunda Liga", "2018–19 LigaPro", "2021–22 Liga Portugal 2",
                               "2010–11 Liga de Honra"],
      [P(r"(?:Segunda Liga|LigaPro|Liga Portugal 2|Liga de Honra|Liga Vitalis|Liga Orangina)", 2)]),
    S("be.1", "Belgium", 1, ["2015–16 Belgian Pro League", "2017–18 Belgian First Division A",
                              "2007–08 Belgian First Division"],
      [P(r"Belgian (?:Pro League|First Division A)", 1), P(r"Belgian First Division", 1, hi=2015)], None, "BE1"),
    S("be.2", "Belgium", 2, ["2015–16 Belgian Second Division", "2017–18 Belgian First Division B",
                              "2022–23 Challenger Pro League"],
      [P(r"Belgian Second Division", 2, hi=2015), P(r"Belgian First Division B", 2, 2016, 2021),
       P(r"Challenger Pro League", 2, 2022)]),
    S("gr.1", "Greece", 1, ["2015–16 Super League Greece", "2005–06 Alpha Ethniki"],
      [P(r"Super League Greece(?: 1)?", 1), P(r"Alpha Ethniki", 1, hi=2005)], None, "GR1"),
    S("gr.2", "Greece", 2, ["2015–16 Football League (Greece)", "2008–09 Beta Ethniki", "2020–21 Super League Greece 2"],
      [P(r"Beta Ethniki", 2, hi=2009), P(r"Football League \(Greece\)", 2, 2010, 2018),
       P(r"Super League (?:Greece )?2", 2, 2019)]),
    S("tr.1", "Türkiye", 1, ["2015–16 Süper Lig"], [P(r"Süper Lig", 1)], None, "TR1"),
    S("tr.2", "Türkiye", 2, ["2015–16 TFF First League"], [P(r"TFF (?:First League|1\. Lig)", 2)]),
    S("ru.1", "Russia", 1, ["2015–16 Russian Premier League"], [P(r"Russian Premier League", 1, 2011)], None, "RU1",
      lo=2011),
    S("ru.2", "Russia", 2, ["2015–16 Russian National Football League", "2022–23 Russian First League"],
      [P(r"Russian (?:National Football League|Football National League|First League)", 2, 2011)], lo=2011),
    S("ua.1", "Ukraine", 1, ["2015–16 Ukrainian Premier League", "2007–08 Vyshcha Liha"],
      [P(r"Ukrainian Premier League", 1), P(r"Vyshcha Liha", 1, hi=2007)], None, "UKR1"),
    S("ua.2", "Ukraine", 2, ["2015–16 Ukrainian First League"], [P(r"Ukrainian First League", 2)]),
    S("dk.1", "Denmark", 1, ["2015–16 Danish Superliga"], [P(r"Danish Superliga", 1)], None, "DK1"),
    S("dk.2", "Denmark", 2, ["2015–16 Danish 1st Division"], [P(r"Danish 1st Division", 2)]),
    # ---- other countries: first divisions (and a few second divisions) for origin-tier recovery ----
    S("at.1", "Austria", 1, ["2015–16 Austrian Football Bundesliga"], [P(r"Austrian Football Bundesliga", 1)],
      None, "A1"),
    S("at.2", "Austria", 2, ["2015–16 Austrian Football First League", "2019–20 2. Liga (Austria)"],
      [P(r"Austrian Football First League", 2, hi=2017), P(r"2\. Liga \(Austria\)", 2, 2018)]),
    S("ch.1", "Switzerland", 1, ["2015–16 Swiss Super League"], [P(r"Swiss Super League", 1)],
      ("Switzerland", "Liechtenstein"), "C1"),
    S("ch.2", "Switzerland", 2, ["2015–16 Swiss Challenge League"], [P(r"Swiss Challenge League", 2)],
      ("Switzerland", "Liechtenstein")),
    S("cz.1", "Czech Republic", 1, ["2015–16 Czech First League"], [P(r"Czech First League", 1)], None, "TS1"),
    S("cz.2", "Czech Republic", 2, ["2015–16 Czech National Football League"],
      [P(r"Czech National Football League", 2)]),
    S("pl.1", "Poland", 1, ["2015–16 Ekstraklasa"], [P(r"Ekstraklasa", 1)], None, "PL1"),
    S("pl.2", "Poland", 2, ["2015–16 I liga"], [P(r"I liga", 2)]),
    S("hr.1", "Croatia", 1, ["2015–16 Croatian First Football League", "2022–23 Croatian Football League"],
      [P(r"Croatian (?:First )?Football League", 1)], None, "KR1"),
    S("rs.1", "Serbia", 1, ["2015–16 Serbian SuperLiga"], [P(r"Serbian Super[Ll]iga", 1)], None, "SER1"),
    S("ro.1", "Romania", 1, ["2015–16 Liga I"], [P(r"Liga I", 1)], None, "RO1"),
    S("sk.1", "Slovakia", 1, ["2015–16 Slovak First Football League", "2022–23 Slovak First Football League"],
      [P(r"Slovak (?:First Football League|Super Liga)", 1)]),
    S("si.1", "Slovenia", 1, ["2015–16 Slovenian PrvaLiga"], [P(r"Slovenian PrvaLiga", 1)]),
    S("hu.1", "Hungary", 1, ["2015–16 Nemzeti Bajnokság I"], [P(r"Nemzeti Bajnokság I", 1)]),
    S("bg.1", "Bulgaria", 1, ["2015–16 A Football Group", "2018–19 First Professional Football League (Bulgaria)"],
      [P(r"A Football Group", 1, hi=2015), P(r"First Professional Football League \(Bulgaria\)", 1, 2016),
       P(r"Bulgarian First League", 1, 2016)]),
    S("ba.1", "Bosnia-Herzegovina", 1, ["2015–16 Premier League of Bosnia and Herzegovina"],
      [P(r"Premier League of Bosnia and Herzegovina", 1)]),
    S("cy.1", "Cyprus", 1, ["2015–16 Cypriot First Division"], [P(r"Cypriot First Division", 1)]),
    S("il.1", "Israel", 1, ["2015–16 Israeli Premier League"], [P(r"Israeli Premier League", 1)]),
    S("se.1", "Sweden", 1, ["2016 Allsvenskan"], [P(r"Allsvenskan", 1, fmt="calendar")], None, "SE1"),
    S("se.2", "Sweden", 2, ["2016 Superettan"], [P(r"Superettan", 2, fmt="calendar")]),
    S("no.1", "Norway", 1, ["2016 Eliteserien", "2015 Tippeligaen"],
      [P(r"(?:Eliteserien|Tippeligaen)", 1, fmt="calendar")], None, "NO1"),
    S("no.2", "Norway", 2, ["2016 Norwegian First Division"],
      [P(r"(?:Norwegian First Division|1\. divisjon|OBOS-ligaen)", 2, fmt="calendar")]),
    S("br.1", "Brazil", 1, ["2016 Campeonato Brasileiro Série A"],
      [P(r"Campeonato Brasileiro Série A", 1, fmt="calendar")], None, "BRA1"),
    S("br.2", "Brazil", 2, ["2016 Campeonato Brasileiro Série B"],
      [P(r"Campeonato Brasileiro Série B", 2, fmt="calendar")]),
    S("ar.1", "Argentina", 1, ["2016–17 Argentine Primera División", "2021 Argentine Primera División"],
      [P(r"Argentine Primera División", 1, fmt="any")], None, "ARG1"),
    S("uy.1", "Uruguay", 1, ["2016 Uruguayan Primera División season"],
      [P(r"Uruguayan Primera División season", 1, fmt="any")]),
    S("cl.1", "Chile", 1, ["2017 Chilean Primera División season"],
      [P(r"Chilean Primera División season", 1, fmt="any")]),
    S("co.1", "Colombia", 1, ["2016 Categoría Primera A season"],
      [P(r"Categoría Primera A season", 1, fmt="calendar")]),
    S("ec.1", "Ecuador", 1, ["2016 Ecuadorian Serie A"], [P(r"Ecuadorian Serie A", 1, fmt="calendar")]),
    S("py.1", "Paraguay", 1, ["2016 Paraguayan Primera División season"],
      [P(r"Paraguayan Primera División season", 1, fmt="calendar")]),
    S("mx.1", "Mexico", 1, ["Apertura 2016 Liga MX"],
      [P(r"Liga MX", 1, fmt="apertura"), P(r"Liga MX", 1, fmt="clausura")], None, "MEX1"),
    S("mls", "United States", 1, ["2016 Major League Soccer season"],
      [P(r"Major League Soccer season", 1, fmt="calendar")], ("United States", "Canada"), "MLS1"),
    S("jp.1", "Japan", 1, ["2016 J1 League", "2014 J.League Division 1"],
      [P(r"(?:J1 League|J\. ?League Division 1)", 1, fmt="calendar")], None, "JAP1"),
    S("jp.2", "Japan", 2, ["2016 J2 League"], [P(r"(?:J2 League|J\. ?League Division 2)", 2, fmt="calendar")]),
    S("kr.1", "Korea, South", 1, ["2016 K League Classic", "2019 K League 1"],
      [P(r"(?:K League 1|K League Classic|K League)", 1, fmt="calendar")], None, "RSK1"),
    S("cn.1", "China", 1, ["2016 Chinese Super League"], [P(r"Chinese Super League", 1, fmt="calendar")]),
    S("au.1", "Australia", 1, ["2015–16 A-League"], [P(r"A-League(?: Men)?", 1)], ("Australia", "New Zealand"),
      "AUS1"),
    S("sa.1", "Saudi Arabia", 1, ["2015–16 Saudi Professional League"],
      [P(r"Saudi (?:Professional|Pro) League", 1)], None, "SA1"),
    S("qa.1", "Qatar", 1, ["2015–16 Qatar Stars League"], [P(r"Qatar Stars League", 1)]),
    S("ae.1", "United Arab Emirates", 1, ["2015–16 UAE Pro-League"],
      [P(r"(?:UAE Pro[- ]League|UAE Arabian Gulf League)", 1)]),
    S("kz.1", "Kazakhstan", 1, ["2016 Kazakhstan Premier League"], [P(r"Kazakhstan Premier League", 1, fmt="calendar")]),
    S("by.1", "Belarus", 1, ["2016 Belarusian Premier League"], [P(r"Belarusian Premier League", 1, fmt="calendar")]),
]
SERIES_BY_CODE = {s["code"]: s for s in SERIES}
assert len(SERIES_BY_CODE) == len(SERIES)

# Register series that the Week 9 module does not know, so league_tiers.match_name / learn_aliases accept the codes.
for s in SERIES:
    if s["code"] not in lt.LEAGUES:
        lt.LEAGUES[s["code"]] = lt._l(s["country"], s["tier"], s["code"], s["family"], s["snap"])
    else:
        L = lt.LEAGUES[s["code"]]
        assert L["country"] == s["country"] and L["tier"] == s["tier"], s["code"]
        if s["snap"] and not L["snap"]:
            L["snap"] = s["snap"]


# --------------------------------------------------------------------------------------------------------------------
# Season labels
# --------------------------------------------------------------------------------------------------------------------
def title_season(title: str, pat: dict) -> tuple[str, int] | None:
    """(label, Transfermarkt season id) if `title` is a season article of this name pattern, else None."""
    name = pat["re"]
    fmt = pat["fmt"]
    tries = []
    if fmt in ("prefix", "any"):
        tries.append(("split", rf"^{SPLIT} {name}$"))
    if fmt in ("calendar", "any"):
        tries.append(("calendar", rf"^{CAL} {name}$"))
    if fmt == "apertura":
        tries.append(("apertura", rf"^Apertura {CAL} {name}$"))
    if fmt == "clausura":
        tries.append(("clausura", rf"^Clausura {CAL} {name}$"))
    for kind, rx in tries:
        m = re.fullmatch(rx, title)
        if not m:
            continue
        if kind == "split":
            y1, y2 = int(m["y1"]), m["y2"]
            if int(y2[-2:]) != (y1 + 1) % 100:
                return None
            return f"{y1}-{str(y1 + 1)[-2:]}", y1
        y = int(m["cy"])
        if kind == "calendar":
            return str(y), y - 1
        if kind == "apertura":
            return f"Apertura {y}", y
        return f"Clausura {y}", y - 1
    return None


def classify_title(title: str, series: dict) -> dict | None:
    for p in series["pats"]:
        r = title_season(title, p)
        if r and p["lo"] <= r[1] <= p["hi"] and r[1] >= series["lo"] and SEASONS[0] <= r[1] <= SEASONS[1]:
            return {"label": r[0], "season": r[1], "tier": p["tier"]}
    return None


# --------------------------------------------------------------------------------------------------------------------
# Download (article pages only, cached, rate-limited)
# --------------------------------------------------------------------------------------------------------------------
_last = [0.0]


def _key(title: str) -> str:
    return hashlib.sha1(title.encode()).hexdigest()


def load_manifest() -> dict:
    if WIKI_MANIFEST.exists():
        return json.loads(WIKI_MANIFEST.read_text())
    return {"generated_by": "src/analysis/league_tiers_wiki.py", "user_agent": USER_AGENT, "terms": WIKI_TERMS,
            "pages": {}}


def save_manifest(man: dict) -> None:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False, sort_keys=True))


def fetch(title: str, man: dict, offline: bool) -> str | None:
    """Article HTML for `title` (cached). None when the page does not exist or offline and not cached."""
    rec = man["pages"].get(title)
    path = WIKI_DIR / f"{_key(title)}.html.gz"
    if rec and rec["http_status"] == 200 and path.exists():
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    if rec and rec["http_status"] == 404:
        return None
    if offline:
        return None
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="()_,.'!-")
    wait = MIN_INTERVAL_S - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    status, body = None, None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                status, body = r.status, r.read()
            break
        except urllib.error.HTTPError as e:
            status = e.code
            if e.code in (429, 503):
                raise RuntimeError(f"Wikipedia refused ({e.code}); stopping, not retrying around a rate limit")
            break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    _last[0] = time.time()
    entry = {"url": url, "http_status": status, "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "terms": "see manifest terms"}
    if status == 200 and body is not None:
        WIKI_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(body))
        html = body.decode("utf-8")
        m = re.search(r'<link rel="canonical" href="https://en\.wikipedia\.org/wiki/([^"]+)"', html)
        entry.update({"local_path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(body).hexdigest(),
                      "bytes": len(body), "canonical_title": urllib.parse.unquote(m.group(1)).replace("_", " ") if m else None})
        man["pages"][title] = entry
        return html
    man["pages"][title] = entry
    return None


def navbox_links(html: str) -> list[tuple[str, bool]]:
    """(link title, is_red_link) for every link inside navboxes."""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(html, "lxml")
    out = []
    for nb in s.select("div.navbox, table.navbox"):
        for a in nb.select("a[title]"):
            t = a.get("title") or ""
            red = "new" in (a.get("class") or [])
            t = re.sub(r" \(page does not exist\)$", "", t)
            out.append((t, red))
    return out


def discover_and_fetch(offline: bool) -> pd.DataFrame:
    """Breadth-first discovery per series. Returns one row per (series, title) with status."""
    man = load_manifest()
    rows = []
    try:
        for s in SERIES:
            queue = list(s["seeds"])
            seen, red = set(), set()
            while queue:
                t = queue.pop(0)
                if t in seen:
                    continue
                seen.add(t)
                html = fetch(t, man, offline)
                cls = classify_title(t, s)
                if html is None:
                    rows.append({"code": s["code"], "title": t, "status": "not_found" if not offline else "not_cached",
                                 **(cls or {})})
                    continue
                rows.append({"code": s["code"], "title": t, "status": "fetched", **(cls or {})})
                for lt_, is_red in navbox_links(html):
                    c = classify_title(lt_, s)
                    if not c:
                        continue
                    if is_red:
                        red.add(lt_)
                    elif lt_ not in seen:
                        queue.append(lt_)
            for t in sorted(red - seen):
                rows.append({"code": s["code"], "title": t, "status": "red_link_page_does_not_exist",
                             **classify_title(t, s)})
            save_manifest(man)
    finally:
        save_manifest(man)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------------
# Parsing standings tables
# --------------------------------------------------------------------------------------------------------------------
TEAM_H = ("team", "club", "teams", "clubs")
PLD_H = ("pld", "p", "gp", "played", "mp", "m", "pj", "games")
POS_H = ("pos", "position", "pos.", "#", "rank")
ANNOT = re.compile(r"\s*\((?:[A-Z]{1,2}|[a-z])(?:,\s*[A-Z]{1,2})*\)\s*$")


def _clean(t: str) -> str:
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = ANNOT.sub("", t.strip())
    return re.sub(r"\s+", " ", t).strip()


def standings_tables(html: str) -> list[list[dict]]:
    from bs4 import BeautifulSoup
    s = BeautifulSoup(html, "lxml")
    tabs = []
    for tb in s.select("table.wikitable"):
        trs = tb.find_all("tr")
        if len(trs) < 4:
            continue
        hdr = [c.get_text(" ", strip=True).lower().rstrip(".").strip() for c in trs[0].find_all(["th", "td"])]
        hdr = [re.sub(r"\[.*?\]", "", h).strip() for h in hdr]
        ti = next((i for i, h in enumerate(hdr) if h in TEAM_H or h.split(" ")[0] in ("team", "club")), None)
        if ti is None or not any(h in PLD_H for h in hdr) or not any(h in POS_H for h in hdr):
            continue
        teams = []
        for tr in trs[1:]:
            cells = tr.find_all(["th", "td"], recursive=False)
            if len(cells) < len(hdr) - 2 or len(cells) <= ti:
                continue
            cell = cells[ti]
            a = next((x for x in cell.find_all("a") if x.get("title") and not x.find_parent(class_="flagicon")
                      and not (x.get("href") or "").startswith("/wiki/File:") and x.get_text(strip=True)), None)
            text = _clean(cell.get_text(" ", strip=True))
            if not text or text.lower() in ("team", "club"):
                continue
            title = re.sub(r" \(page does not exist\)$", "", a.get("title")) if a else None
            teams.append({"text": _clean(a.get_text(" ", strip=True)) if a else text, "cell_text": text,
                          "title": title, "red": bool(a and "new" in (a.get("class") or []))})
        if len(teams) >= 4:
            tabs.append(teams)
    return tabs


def division_teams(html: str) -> tuple[list[dict], str]:
    """Teams of the division (largest standings table plus disjoint groups of similar size)."""
    tabs = standings_tables(html)
    if not tabs:
        return [], "no_standings_table"
    key = lambda t: (t["title"] or t["text"]).lower()
    tabs = sorted(tabs, key=len, reverse=True)
    main = tabs[0]
    chosen = [main]
    used = {key(t) for t in main}
    for tb in tabs[1:]:
        ks = {key(t) for t in tb}
        if len(tb) >= 0.6 * len(main) and not (ks & used):
            chosen.append(tb)
            used |= ks
    teams, seen = [], set()
    for tb in chosen:
        for t in tb:
            if key(t) not in seen:
                seen.add(key(t))
                teams.append(t)
    return teams, ("ok" if len(chosen) == 1 else f"ok_{len(chosen)}_groups")


# --------------------------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------------------------
def parse_all(disc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    man = load_manifest()
    pages = disc[(disc.status == "fetched") & disc.season.notna()].drop_duplicates(["code", "title"])
    teams, files = [], []
    for r in pages.itertuples():
        html = fetch(r.title, man, offline=True)
        tt, st = division_teams(html)
        files.append({"code": r.code, "season": int(r.season), "label": r.label, "tier": int(r.tier),
                      "title": r.title, "url": man["pages"][r.title]["url"], "status": st, "n_teams": len(tt)})
        for t in tt:
            teams.append({"code": r.code, "season": int(r.season), "season_label": r.label, "tier": int(r.tier),
                          "article": r.title, **t})
    return pd.DataFrame(teams), pd.DataFrame(files)


def _strip_disamb(t: str | None) -> str | None:
    if t is None:
        return None
    return re.sub(r"\s*\((?:football club|soccer club|footballer|football|club|[A-Z][a-z]+ club)\)$", "", t)


def _undot(t: str | None) -> str | None:
    """Collapse dotted abbreviations (K.V.C. -> KVC, F.C. -> FC, R. -> R) so they normalise like Transfermarkt's."""
    if t is None:
        return None
    t = re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda m: m.group(0).replace(".", ""), t)
    return re.sub(r"\b([A-Za-z])\.(?=\s|$)", r"\1", t)


# Transfermarkt writes Chinese clubs with a city code ("BJ Guoan", "SH Shenhua"); applied to cn.1 names only.
CN_CITY = {"Beijing": "BJ", "Shanghai": "SH", "Guangzhou": "GZ", "Tianjin": "TJ", "Changchun": "CC", "Dalian": "DL",
           "Wuhan": "WH", "Shenzhen": "SZ", "Jiangsu": "JS", "Hebei": "HB", "Shandong": "SD", "Henan": "HN",
           "Chongqing": "CQ", "Hangzhou": "HZ", "Qingdao": "QD", "Chengdu": "CD", "Shijiazhuang": "SJZ",
           "Guizhou": "GZ", "Liaoning": "LN", "Yanbian": "YB", "Meizhou": "MZ", "Cangzhou": "CZ", "Nantong": "NT",
           "Zhejiang": "ZJ", "Xiamen": "XM", "Shaanxi": "SX", "Harbin": "HRB", "Kunming": "KM", "Qingdao Hainiu": "QD"}


def variants(r) -> list[str]:
    """Name variants of one listed team: link text, cell text, article title without disambiguation, undotted;
    for Chinese clubs also with the city code Transfermarkt uses."""
    title = r.title if isinstance(r.title, str) else None
    v = [r.text, r.cell_text, _strip_disamb(title)]
    v += [_undot(x) for x in v if x]
    if getattr(r, "code", None) == "cn.1":
        v += [re.sub(rf"^{k}\b", c, x) for x in list(v) if x for k, c in CN_CITY.items() if x.startswith(k + " ")]
    return [x for x in dict.fromkeys(v) if x]


YOUTH_SUFFIX = re.compile(r"(?:\b(?:Yth|Y|U\s?-?\d{2}|Jgd|Jugend|Juniors?|Youth|Academy|Sub-?\d{2})\.?)$", re.I)
SUBSET_MIN_COVER = 0.67     # smaller token set must be at least 2/3 of the larger (measured precision: README)
# Disabled after the 12 Nov 2026 manual audit: on a stratified sample of club-seasons that only this step matched,
# 4 of 23 were wrong (a successor club matched to seasons before it existed: Koge BK -> HB Koge, Nafta Lendava ->
# NK Nafta 1903, and earlier Malatyaspor -> Yeni Malatyaspor), although it is right for 99.9% of names that other
# rules also match. Kept for the precision check only.
USE_SUBSET_FALLBACK = False


def _word(t: str) -> bool:
    """Tokens used for containment: distinctive words (3+ letters) and single-letter initials ("Y." of Yeni);
    two-letter codes (state suffixes such as MT/PR, legal forms such as CS) are ignored, as before."""
    return not t.isdigit() and (len(t) >= MIN_TOKEN or len(t) == 1)


class SubsetIndex:
    """Country-wide token-subset fallback: candidate TM clubs of the league's country family, same reserve status,
    not a youth side by name, whose distinctive loose tokens contain (or are contained in) the name's tokens with
    coverage >= SUBSET_MIN_COVER. Accepted only when exactly one club qualifies over all name variants."""

    def __init__(self, tm: dict):
        self.tm = tm
        self.by_country: dict[str, list] = {}
        youth = set(tm["names"].club_id[tm["names"].name.map(lambda n: bool(YOUTH_SUFFIX.search(str(n).strip())))])
        for cid, loose in zip(tm["names"].club_id, tm["names"].loose):
            if int(cid) in youth:
                continue
            # containment and coverage count single-letter initials, so "Y." in "Y. Malatyaspor" is a word that
            # "Malatyaspor" lacks; a name needs at least one distinctive (3+ letter) token to be used
            full = frozenset(t for t in loose.split(" #")[0].split() if _word(t))
            if any(len(t) >= MIN_TOKEN for t in full):
                for k in tm["countries"].get(int(cid), ()):
                    self.by_country.setdefault(k, []).append((int(cid), full))

    def candidates(self, names: list[str], code: str) -> set:
        fam = SERIES_BY_CODE[code]["family"]
        res = any(lt.is_reserve_name(x) for x in names)
        out = set()
        for x in names:
            vt = frozenset(t for t in lt.norm_keys(x)[1].split(" #")[0].split() if _word(t))
            if not any(len(t) >= MIN_TOKEN for t in vt):
                continue
            for k in fam:
                for cid, ts in self.by_country.get(k, []):
                    if self.tm["reserve"].get(cid, False) != res or not (ts <= vt or vt <= ts):
                        continue
                    small, big = (ts, vt) if len(ts) <= len(vt) else (vt, ts)
                    if len(small) / len(big) >= SUBSET_MIN_COVER:
                        out.add(cid)
        return out


def match_wiki(teams: pd.DataFrame, tm: dict, manual: dict, learned: dict, sub: "SubsetIndex | None" = None) -> pd.DataFrame:
    cache: dict = {}
    out = []
    for r in teams.itertuples(index=False):
        res = {}
        ids, methods, reasons = set(), [], []
        for nm in variants(r):
            if not nm:
                continue
            k = (nm, r.code)
            if k not in cache:
                cache[k] = lt.match_name(nm, r.code, tm, manual, learned)
            m = cache[k]
            if m["club_id"] is not None:
                ids.add(int(m["club_id"]))
                methods.append(f"{m['match_method']}:{'text' if nm == r.text else 'variant'}")
            elif m.get("unmatched_reason"):
                reasons.append(m["unmatched_reason"])
        ambiguous_exact = any(x.startswith(("ambiguous", "conflict")) for x in reasons)
        if not ids and sub is not None and not ambiguous_exact:
            c = sub.candidates(variants(r), r.code)
            if len(c) == 1:
                ids, methods = set(c), ["token_subset_country"]
            elif len(c) > 1:
                reasons = [f"token-subset fallback ambiguous: {len(c)} clubs"] + reasons
        if len(ids) == 1:
            res = {"club_id": ids.pop(), "match_method": methods[0], "unmatched_reason": None}
        elif len(ids) > 1:
            res = {"club_id": None, "match_method": None,
                   "unmatched_reason": f"link text and article title match different clubs {sorted(ids)}"}
        else:
            res = {"club_id": None, "match_method": None, "unmatched_reason": reasons[0] if reasons else "no name"}
        out.append({**r._asdict(), **res})
    a = pd.DataFrame(out)
    m = a.club_id.notna()
    dup = a[m].duplicated(["code", "season", "club_id"], keep=False)
    # the same club listed twice in one league-season (two article titles for one season id, e.g. Argentina's
    # transitional tournaments) is one membership; two different names of one article mapping to one club is not
    di = a[m][dup].groupby(["code", "season", "club_id"]).article.transform("nunique").eq(1)
    bad = di[di].index
    a.loc[bad, "unmatched_reason"] = "two team names of this article matched the same TM club"
    a.loc[bad, ["club_id", "match_method"]] = None
    a["club_id"] = a.club_id.astype("Int64")
    a["club_name_tm"] = a.club_id.map(lambda i: tm["display"].get(int(i)) if pd.notna(i) else None)
    a["country"] = a.code.map(lambda c: SERIES_BY_CODE[c]["country"])
    return a


def _tokset(key: str) -> frozenset:
    return frozenset(key.split(" #")[0].split())


def club_end_years(tm: dict) -> dict[int, int]:
    """Transfermarkt marks dissolved clubs with '(- YYYY)' in the name: club id -> last year."""
    end: dict[int, int] = {}
    for cid, nm in zip(tm["names"].club_id, tm["names"].name):
        m = re.search(r"\(\s*-\s*(\d{4})\s*\)", str(nm))
        if m:
            end[int(cid)] = max(end.get(int(cid), 0), int(m.group(1)))
    return end


def reject_after_end(audit: pd.DataFrame, tm: dict) -> pd.DataFrame:
    """A listed team matched to a club in a season after that club's end year is a refounded successor club with a
    new id (Lierse SK (- 2018) vs Lierse Kempenzonen): the match is rejected and the team left unmatched (added
    12 Nov 2026 after the manual audit)."""
    end = club_end_years(tm)
    a = audit.copy()
    e = a.club_id.map(lambda i: end.get(int(i)) if pd.notna(i) else None)
    bad = e.notna() & (a.season > e)
    a.loc[bad, "unmatched_reason"] = "matched club ended in " + e[bad].astype("Int64").astype(str) + \
        " (a refounded club has a different id)"
    a.loc[bad, ["club_id", "match_method", "club_name_tm"]] = None
    a["club_id"] = a.club_id.astype("Int64")
    return a


def learned_aliases(teams: pd.DataFrame, tm: dict, snap: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Week 9 openfootball aliases plus aliases learned on Wikipedia first-division seasons that overlap the
    snapshot. Within one linked league-season (Wikipedia teams E, snapshot member clubs T), iterate to a fixpoint:
    (a) restricted exact: exactly one remaining club of T shares a normalised key with any name variant;
    (b) token subset: exactly one remaining club of T has a name whose loose tokens contain all loose tokens of a
        variant (or vice versa), same reserve status;
    (c) elimination: one team and one club left and |E| = |T|.
    A variant is kept as an alias only if it maps to the same club in every linked season where it appears and no two
    teams of one season share a club. A key mapping to two clubs (across sources) is dropped."""
    snapm = snap[snap.member]
    tok_by_id = {i: {_tokset(k) for lvl, k in ks if lvl == "loose"} for i, ks in tm["keys_by_id"].items()}
    votes: dict[tuple, set] = {}
    how: dict[tuple, str] = {}
    log = []
    for (code, season), e in teams.groupby(["code", "season"]):
        comp = SERIES_BY_CODE[code]["snap"]
        if not comp:
            continue
        T = set(snapm[(snapm.competition_id == comp) & (snapm.season == season)].club_id.astype(int))
        if not T:
            continue
        c = SERIES_BY_CODE[code]["country"]
        ents = e.drop_duplicates(["text", "title"])
        V = {k: variants(r) for k, r in zip(range(len(ents)), ents.itertuples(index=False))}
        res = {k: any(lt.is_reserve_name(x) for x in v) for k, v in V.items()}
        got: dict[int, tuple] = {}
        changed = True
        while changed:
            changed = False
            remT = T - {i for i, _ in got.values()}
            for k, v in V.items():
                if k in got:
                    continue
                cand = set()
                for x in v:
                    cand |= {i for i in lt._restricted_candidates(x, remT, tm)}
                cand = {i for i in cand if tm["reserve"].get(i, False) == res[k]}
                step = "restricted_exact"
                if len(cand) != 1:
                    cand = set()
                    for x in v:
                        ts = _tokset(lt.norm_keys(x)[1])
                        if not ts:
                            continue
                        for i in remT:
                            if tm["reserve"].get(i, False) != res[k]:
                                continue
                            if any(ts <= t2 or t2 <= ts for t2 in tok_by_id.get(i, ()) if t2):
                                cand.add(i)
                    step = "token_subset"
                if len(cand) == 1:
                    got[k] = (cand.pop(), step)
                    remT = T - {i for i, _ in got.values()}
                    changed = True
            left = [k for k in V if k not in got]
            remT = T - {i for i, _ in got.values()}
            if len(left) == 1 and len(remT) == 1 and len(V) == len(T):
                got[left[0]] = (next(iter(remT)), "elimination")
                changed = True
        ids = [i for i, _ in got.values()]
        dupl = {i for i in ids if ids.count(i) > 1}
        for k, (i, step) in got.items():
            if i in dupl:
                continue
            for x in V[k]:
                votes.setdefault((c, x), set()).add(i)
                how.setdefault((c, x), f"{step} {code} {season}")
        log.append({"code": code, "season": season, "n_wiki": len(V), "n_snapshot": len(T),
                    "n_paired": len(got) - sum(1 for i, _ in got.values() if i in dupl)})
    rows = []
    for (c, x), ids in votes.items():
        rows.append({"country": c, "external_team_name": x, "club_id": min(ids) if len(ids) == 1 else None,
                     "club_name_tm": tm["display"].get(min(ids)) if len(ids) == 1 else None,
                     "learned_by": how[(c, x)], "status": "ok" if len(ids) == 1 else "rejected",
                     "reject_reason": "" if len(ids) == 1 else f"maps to several clubs {sorted(ids)}"})
    al = pd.DataFrame(rows)
    learned: dict = {}
    clash = set()
    old = pd.read_csv(OLD_ALIASES)
    old = old[old.status.eq("ok")]
    pairs = [(r.country, r.external_team_name, int(r.club_id)) for r in old.itertuples()]
    pairs += [(r.country, r.external_team_name, int(r.club_id)) for r in al[al.status.eq("ok")].itertuples()]
    for c, n, i in pairs:
        if (c, n) in learned and learned[(c, n)] != i:
            clash.add((c, n))
        learned.setdefault((c, n), i)
    for k in clash:
        learned.pop(k, None)
    al["dropped_cross_source_clash"] = [(r.country, r.external_team_name) in clash for r in al.itertuples()]
    return learned, al.assign(_log=None), pd.DataFrame(log)


def combine(audit: pd.DataFrame, old: pd.DataFrame, tm: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    w = audit[audit.club_id.notna()].copy()
    w["club_id"] = w.club_id.astype(int)
    w["league_name"] = w.article.str.replace(r"^(?:\d{4}(?:–\d{2,4})? |Apertura \d{4} |Clausura \d{4} )", "", regex=True)
    w = w.drop_duplicates(["club_id", "season", "code"])
    # Wikipedia-internal conflicts: one club in two divisions (codes) of the same season
    g = w.groupby(["club_id", "season"]).code.transform("nunique")
    wconf = w[g > 1].assign(conflict_type="Wikipedia lists the club in two divisions in one season")
    w = w[g == 1]
    # overlap with the Week 9 table
    oldk = old.set_index(["club_id", "season"])
    key = pd.MultiIndex.from_arrays([w.club_id, w.season])
    inold = key.isin(oldk.index)
    ov = w[inold].copy()
    o = oldk.reindex(pd.MultiIndex.from_arrays([ov.club_id, ov.season]))
    ov["old_tier"] = o.tier.to_numpy()
    ov["old_league"] = o.league_name.to_numpy()
    ov["old_source"] = o.source.to_numpy()
    ov["old_code"] = o.source_league_code.to_numpy()
    ov["agrees"] = ov.tier.to_numpy() == ov.old_tier.to_numpy()
    new = w[~inold]
    nt = pd.DataFrame({"club_id": new.club_id, "club_name_tm": new.club_name_tm, "season": new.season,
                       "country": new.country, "league_name": new.league_name, "tier": new.tier,
                       "source": "wikipedia", "source_league_code": new.code, "external_team_name": new.text,
                       "match_method": new.match_method, "n_sources_agreeing": 1, "sources": "wikipedia",
                       "external_season_label": new.season_label,
                       "source_file": "https://en.wikipedia.org/wiki/" + new.article.str.replace(" ", "_")})
    table = pd.concat([old.assign(source_week="week9"), nt.assign(source_week="week10_wikipedia")],
                      ignore_index=True).sort_values(["club_id", "season"]).reset_index(drop=True)
    assert not table.duplicated(["club_id", "season"]).any()
    return table, ov, wconf


MIN_TOKEN = 3


def country_tokens(tm: dict) -> dict[str, set]:
    """Distinctive name tokens (loose key, at least MIN_TOKEN letters) of every Transfermarkt club in our universe,
    by country."""
    out: dict[str, set] = {}
    names = tm["names"]
    for cid, loose in zip(names.club_id, names.loose):
        toks = {t for t in loose.split(" #")[0].split() if len(t) >= MIN_TOKEN and not t.isdigit()}
        for k in tm["countries"].get(int(cid), ()):
            out.setdefault(k, set()).update(toks)
    return out


def could_be_known(audit: pd.DataFrame, tm: dict) -> pd.Series:
    """For each unmatched listed team: does any of its name variants share a distinctive token with a club of that
    country family in our universe? If not, the team cannot be the club of any episode."""
    ct = country_tokens(tm)
    out = []
    for r in audit.itertuples(index=False):
        if pd.notna(r.club_id):
            out.append(False)
            continue
        fam = SERIES_BY_CODE[r.code]["family"]
        pool = set().union(*(ct.get(k, set()) for k in fam))
        toks = set()
        for x in variants(r):
            toks |= {t for t in lt.norm_keys(x)[1].split(" #")[0].split() if len(t) >= MIN_TOKEN and not t.isdigit()}
        amb = isinstance(r.unmatched_reason, str) and r.unmatched_reason.startswith(("ambiguous", "conflict",
                                                                                     "link text", "two team",
                                                                                     "reserve/senior"))
        out.append(bool(amb or (toks & pool)))
    return pd.Series(out, index=audit.index)


def completeness(audit: pd.DataFrame, files: pd.DataFrame, old_cov: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """One row per (country, season, tier, source list). `complete_strict`: every listed team matched.
    `complete`: every listed team matched, or (unmatched) sharing no distinctive name token with any club of that
    country in our universe, so it cannot be the club of any episode. At least 8 teams either way."""
    rows = []
    a = audit.drop_duplicates(["code", "season", "text", "title"])
    a = a.groupby(["country", "code", "season", "tier"]).agg(n=("text", "size"),
                                                             matched=("club_id", lambda s: s.notna().sum()),
                                                             risky=("could_be_known_club", "sum"))
    for (c, code, s, t), r in a.iterrows():
        rows.append({"country": c, "season": s, "tier": t, "source": "wikipedia", "code": code, "teams": r.n,
                     "matched": r.matched, "unmatched_possibly_known": int(r.risky),
                     "complete_strict": bool(r.n >= 8 and r.matched == r.n),
                     "complete": bool(r.n >= 8 and r.risky == 0)})
    oc = old_cov[old_cov.status.eq("ok")]
    for r in oc.itertuples():
        ok = bool(r.teams_in_file >= 8 and r.teams_unmatched == 0)
        rows.append({"country": r.country, "season": r.season, "tier": r.tier, "source": "openfootball",
                     "code": r.source_league_code, "teams": r.teams_in_file, "matched": r.teams_matched,
                     "unmatched_possibly_known": int(r.teams_unmatched), "complete_strict": ok, "complete": ok})
    sm = snap[snap.member].groupby(["country_name", "competition_id", "season"]).club_id.nunique()
    for (c, comp, s), n in sm.items():
        rows.append({"country": c, "season": s, "tier": 1, "source": "transfermarkt_snapshot_games", "code": comp,
                     "teams": n, "matched": n, "unmatched_possibly_known": 0, "complete_strict": True,
                     "complete": True})
    C = pd.DataFrame(rows)
    return C


def build(offline: bool) -> dict:
    disc = discover_and_fetch(offline)
    teams, files = parse_all(disc)
    con = lt.connect()
    master = lt.load_master()
    tm = lt.tm_universe(con, master)
    snap = lt.snapshot_tier1(con)
    learned, al, pairlog = learned_aliases(teams, tm, snap)
    manual = lt.check_manual_aliases(tm)
    sub = SubsetIndex(tm)
    audit = match_wiki(teams, tm, manual, learned, sub if USE_SUBSET_FALLBACK else None)
    audit = reject_after_end(audit, tm)
    audit["could_be_known_club"] = could_be_known(audit, tm)
    # precision of the fallback, measured on every distinct team already matched by the other rules
    known = audit[audit.club_id.notna() & ~audit.match_method.eq("token_subset_country")].drop_duplicates(
        ["code", "text", "title"])
    ev = [(sub.candidates(variants(r), r.code), int(r.club_id)) for r in known.itertuples(index=False)]
    uniq = [(c, t) for c, t in ev if len(c) == 1]
    fallback_eval = {"evaluated": len(ev), "unique_candidate": len(uniq),
                     "unique_correct": sum(c == {t} for c, t in uniq),
                     "unique_wrong": sum(c != {t} for c, t in uniq)}
    old = pd.read_csv(OLD_TABLE)
    table, overlap, wconf = combine(audit, old, tm)
    comp = completeness(audit, files, pd.read_csv(OLD_COVERAGE), snap)
    return {"disc": disc, "files": files, "audit": audit, "aliases": al.drop(columns="_log"), "pairlog": pairlog,
            "fallback_eval": fallback_eval, "table": table, "overlap": overlap,
            "wiki_conflicts": wconf, "completeness": comp, "old": old}


def write(r: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    r["disc"].to_csv(OUT / "wiki_discovery.csv", index=False)
    r["files"].to_csv(OUT / "wiki_league_seasons.csv", index=False)
    r["audit"].to_csv(OUT / "wiki_name_match_audit.csv", index=False)
    r["aliases"].to_csv(OUT / "wiki_learned_aliases.csv", index=False)
    r["pairlog"].to_csv(OUT / "wiki_snapshot_pairing_by_league_season.csv", index=False)
    r["table"].to_csv(OUT / "club_season_league_v2.csv", index=False)
    r["overlap"].to_csv(OUT / "wiki_vs_week9_overlap.csv", index=False)
    r["wiki_conflicts"].to_csv(OUT / "wiki_internal_conflicts.csv", index=False)
    r["completeness"].to_csv(OUT / "league_list_completeness.csv", index=False)
    a, ov, f = r["audit"], r["overlap"], r["files"]
    n = {
        "series": len(SERIES), "pages_fetched": int((r["disc"].status == "fetched").sum()),
        "season_pages_parsed": int(len(f)), "season_pages_without_table": int((f.n_teams == 0).sum()),
        "red_links_missing_articles": int((r["disc"].status == "red_link_page_does_not_exist").sum()),
        "team_rows": int(len(a)), "team_rows_matched": int(a.club_id.notna().sum()),
        "overlap_rows": int(len(ov)), "overlap_agree_tier": int(ov.agrees.sum()),
        "overlap_agree_share": float(ov.agrees.mean()) if len(ov) else None,
        "wiki_internal_conflict_rows": int(len(r["wiki_conflicts"])),
        "table_rows_week9": int((r["table"].source_week == "week9").sum()),
        "table_rows_added": int((r["table"].source_week == "week10_wikipedia").sum()),
        "matched_by_token_subset_fallback": int(a.match_method.eq("token_subset_country").sum()),
        "rejected_after_club_end_year": int(a.unmatched_reason.fillna("").str.startswith("matched club ended").sum()),
        "fallback_precision_check": r["fallback_eval"],
        "token_subset_fallback_used": USE_SUBSET_FALLBACK,
    }
    (OUT / "league_tiers_v2_numbers.json").write_text(json.dumps(n, indent=1))
    return n


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch missing pages (else cache only)")
    args = ap.parse_args(argv)
    r = build(offline=not args.download)
    n = write(r)
    print(json.dumps(n, indent=1))
    return r


if __name__ == "__main__":
    main()
