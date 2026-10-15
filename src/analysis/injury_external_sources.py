"""Existing third-party football injury datasets: provenance, licence, schema fit, and the gate decision.

    python -m src.analysis.injury_external_sources

Reads only the metadata and documentation captured once in `data/raw/injury_source_metadata/` (GitHub, Kaggle and
DataCite metadata, README files, one PubMed abstract). No injury data file was downloaded: no candidate with
Transfermarkt player ids has a clear reuse basis (see the report), so the join, validation and coverage tests were
not run. One check uses only the five sample rows printed in the primary candidate's README, against our own
snapshot. Not legal advice. New files only.
"""
from __future__ import annotations

import hashlib
import json
import re

import pandas as pd

from .daniel_scope_final import f0, md_table
from .loan_episodes import ROOT

OUT = ROOT / "data" / "outputs" / "rebuild"
META = ROOT / "data" / "raw" / "injury_source_metadata"
AUDIT_MD = OUT / "injury_external_sources_audit.md"
COMPARISON_CSV = OUT / "injury_external_sources_comparison.csv"
NUMBERS_JSON = OUT / "injury_external_numbers.json"
WRITES = (AUDIT_MD, COMPARISON_CSV, NUMBERS_JSON)
NOT_WRITTEN = ("injury_external_join_audit.csv", "injury_external_validation.csv", "injury_external_coverage.csv")


def load() -> dict:
    cap = json.loads((META / "capture.json").read_text())
    for name, h in cap["files"].items():
        assert hashlib.sha256((META / name).read_bytes()).hexdigest() == h, f"{name} changed since capture"
    j = lambda n: json.loads((META / n).read_text())
    return {"capture": cap, "salimt_repo": j("salimt_repo.json"), "salimt_tree": j("salimt_tree.json"),
            "salimt_readme": (META / "salimt_README.md").read_text(),
            "salimt_readme_data": (META / "salimt_README_data.md").read_text(),
            "salimt_kaggle": j("kaggle_xfkzujqjvx97n_football-datasets.json"),
            "figshare": {v: j(f"datacite_10.6084_m9.figshare.25648788{v}.json") for v in ("", ".v1", ".v2")},
            "sanan": j("kaggle_sananmuzaffarov_european-football-injuries-2020-2025.json"),
            "amrit": j("kaggle_amritbiswas007_player-injuries-and-team-performance-dataset.json"),
            "kberkek": j("kaggle_kberkek00_transfermarkt-datas.json"),
            "dcaribou": j("dcaribou_repo.json"), "hoenig": (META / "hoenig_2022_abstract.txt").read_text()}


def num(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1).replace(",", "")) if m else None


def sentence(text: str, start: str) -> str:
    t = re.sub(r"\s+", " ", text)
    m = re.search(re.escape(start) + r".*?\.(?= [A-Z]|$)", t)
    return m.group(0) if m else ""


def readme_preview(text: str) -> pd.DataFrame:
    """The injury sample rows printed in the primary candidate's README_data.md."""
    block = text.split("## player_injury_histories", 1)[1].split("##", 1)[0]
    rows = [r.strip().strip("|").split("|") for r in block.splitlines() if r.strip().startswith("|")]
    head = [c.strip() for c in rows[0]]
    body = [[c.strip() for c in r] for r in rows[2:]]
    return pd.DataFrame(body, columns=head)


def preview_check(P: pd.DataFrame) -> dict:
    """Do the README's sample spells line up with our own snapshot (same player id, no appearances inside)?"""
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        pid = int(P.player_id.iloc[0])
        who = con.sql(f"select name, date_of_birth from players where player_id = {pid}").fetchone()
        rows = []
        for r in P.itertuples():
            a = pd.to_datetime(r.from_date, format="%d.%m.%Y")
            b = pd.to_datetime(r.end_date, format="%d.%m.%Y")
            apps = con.sql(f"""select count(*) from appearances where player_id = {pid}
                               and date between '{a.date()}' and '{b.date()}'""").fetchone()[0]
            club = con.sql(f"""select player_club_id from appearances where player_id = {pid}
                               and date between '{(a - pd.Timedelta(days=200)).date()}' and '{a.date()}'
                               group by 1 order by count(*) desc limit 1""").fetchone()
            lg = con.sql(f"""select count(*) from games where competition_type = 'domestic_league'
                             and {club[0] if club else -1} in (home_club_id, away_club_id)
                             and date between '{a.date()}' and '{b.date()}'""").fetchone()[0]
            rows.append({"injury": r.injury_reason, "from": str(a.date()), "until": str(b.date()),
                         "games_missed_source": r.games_missed, "club_league_games_in_spell": int(lg),
                         "player_appearances_in_spell": int(apps)})
    finally:
        con.close()
    return {"player_id": pid, "our_name": who[0] if who else None, "rows": rows}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute() -> dict:
    m = load()
    sr, sk, fv = m["salimt_repo"], m["salimt_kaggle"], m["figshare"]
    tree = [t["path"] for t in m["salimt_tree"]["tree"]]
    injury_files = {t["path"]: t.get("size") for t in m["salimt_tree"]["tree"] if "injur" in t["path"] and t["type"] == "blob"}
    fdesc_html = fv[".v2"]["descriptions"][0]["description"]
    fdesc = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fdesc_html))
    P = readme_preview(m["salimt_readme_data"])
    N: dict = {"captured_on": m["capture"]["captured_on"]}
    N["primary"] = {
        "name": sr["full_name"], "creator": sr["owner"]["login"], "github_license": sr["license"],
        "license_file_in_repo": any(p.lower().startswith("license") for p in tree),
        "readme_license_badge": "github/license/salimt/football-datasets" in m["salimt_readme"],
        "kaggle_license": sk["licenseName"], "kaggle_last_updated": sk["lastUpdated"], "kaggle_owner": sk["ownerName"],
        "github_pushed_at": sr["pushed_at"], "created_at": sr["created_at"],
        "readme_last_updated_badge": re.search(r"Last%20Updated-([A-Za-z]+%20\d{4})", m["salimt_readme"]).group(1).replace("%20", " "),
        "stated_source": "Transfermarkt" if "from Transfermarkt" in (sr["description"] or "") else None,
        "injury_rows": num(r"\*\*([\d,]+) Player Injury Histories", m["salimt_readme"]),
        "players_in_datalake": num(r"Total Players\*\*: ([\d,]+)", m["salimt_readme"]),
        "injury_files": injury_files, "injury_columns": list(P.columns),
        "collection_code_in_repo": any(p.endswith(".py") for p in tree)}
    last, _, first = fv[".v2"]["creators"][0]["name"].partition(",")
    N["figshare"] = {"doi": "10.6084/m9.figshare.25648788", "creator": f"{first.strip().title()} {last.strip().title()}",
                     "license": fv[".v2"]["rightsList"][0]["rights"],
                     "versions": {v or "concept": {"title": d["titles"][0]["title"], "updated": d["dates"][-2]["date"]
                                  if len(d["dates"]) > 1 else None, "size": (d.get("sizes") or [None])[0]}
                                  for v, d in fv.items()},
                     "injuries": num(r"\+([\d,]+) injuries", fdesc), "players": num(r"accross \+([\d,]+) players", fdesc),
                     "collection": sentence(fdesc, "To collect injury data"),
                     "fields": re.findall(r"<b>([^<]+?)\s*</b>", fdesc_html)}
    N["others"] = {k: {"title": m[k]["title"], "owner": m[k]["ownerName"], "license": m[k]["licenseName"],
                       "last_updated": m[k]["lastUpdated"], "bytes": m[k]["totalBytes"]} for k in ("sanan", "amrit", "kberkek")}
    N["others"]["sanan"]["records"] = num(r"\*\*([\d,]+) records\*\*", m["sanan"]["description"])
    N["others"]["amrit"]["records"] = num(r"over ([\d,]+) injury records", m["amrit"]["description"])
    N["others"]["kberkek"]["mentions_injuries"] = "injur" in (m["kberkek"]["description"] or "").lower()
    N["dcaribou"] = {"license": m["dcaribou"]["license"]["spdx_id"], "description": m["dcaribou"]["description"]}
    hz = m["hoenig"]
    N["hoenig"] = {"citation": "Hoenig T, Edouard P, Krause M, Malhan D, Relógio A, Junge A, Hollander K. J Sci Med "
                               "Sport. 2022;25(4):300-305. doi:10.1016/j.jsams.2021.11.038",
                   "results": sentence(hz, "RESULTS:")[len("RESULTS: "):] + " " + sentence(hz, "In comparison to other databases"),
                   "conclusion": sentence(hz, "CONCLUSIONS:")[len("CONCLUSIONS: "):]}
    N["preview_check"] = preview_check(P)
    N["downloaded_injury_files"] = 0
    return N


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def comparison(N: dict) -> pd.DataFrame:
    p, f, o = N["primary"], N["figshare"], N["others"]
    Y, NO, DOC = "yes", "no", "per documentation"
    rows = [
        {"source": "salimt/football-datasets (GitHub)", "creator": p["creator"],
         "location": "https://github.com/salimt/football-datasets",
         "stated_license": "none (no LICENSE file; GitHub detects none; README shows a licence badge that links to a "
                           "missing file)", "license_status": "no license found",
         "data_files_licensed": NO, "attribution_required": "not stated", "original_source": "Transfermarkt (stated)",
         "collection_method": "not documented; no collection code in the repo", "last_updated": p["github_pushed_at"][:10],
         "redistribution_statement": "none", "research_restrictions": "none stated", "provenance": "unclear: copy of a "
         "site whose terms prohibit automated copying",
         "transfermarkt_player_id": Y + f" ({DOC}; README sample id {N['preview_check']['player_id']} = "
                                   f"{N['preview_check']['our_name']} in our data)",
         "player_name": "no (in a separate profiles table)", "season": Y, "injury_type": Y, "start_date": Y,
         "end_date": Y, "days_missed": "yes (text, e.g. '11 days')", "games_missed": "yes ('-' when blank)",
         "club": NO, "rows": p["injury_rows"], "players": f"{p['players_in_datalake']:,} in the datalake; injured players not stated",
         "seasons": "not documented", "leagues": "not documented ('all major leagues')",
         "join_without_fuzzy_matching": Y, "downloaded": NO, "viable": "technically yes; reuse basis unclear"},
        {"source": "salimt football-datasets (Kaggle mirror)", "creator": p["kaggle_owner"],
         "location": "https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets",
         "stated_license": p["kaggle_license"], "license_status": "explicit on Kaggle; conflicts with the GitHub copy",
         "data_files_licensed": "yes on Kaggle", "attribution_required": "no (CC0)", "original_source": "Transfermarkt (stated)",
         "collection_method": "not documented", "last_updated": p["kaggle_last_updated"][:10],
         "redistribution_statement": "CC0 dedication by the uploader", "research_restrictions": "none stated",
         "provenance": "unclear: the uploader's dedication cannot settle rights in Transfermarkt's content (a question for review)",
         "transfermarkt_player_id": Y + f" ({DOC})", "player_name": "no", "season": Y, "injury_type": Y, "start_date": Y,
         "end_date": Y, "days_missed": Y, "games_missed": Y, "club": NO, "rows": p["injury_rows"], "players": "as GitHub",
         "seasons": "not documented", "leagues": "not documented", "join_without_fuzzy_matching": Y,
         "downloaded": "no (needs a Kaggle account; not attempted)", "viable": "technically yes; needs approval"},
        {"source": "Vermaut, Injuries - Transfermarkt.com (Figshare)", "creator": f["creator"],
         "location": f"https://doi.org/{f['doi']}", "stated_license": f["license"], "license_status": "explicit (CC BY 4.0)",
         "data_files_licensed": "yes", "attribution_required": "yes (CC BY)", "original_source": "Transfermarkt (stated)",
         "collection_method": f["collection"], "last_updated": f["versions"]["concept"]["updated"],
         "redistribution_statement": "CC BY 4.0", "research_restrictions": "none stated",
         "provenance": "stated web scraping of Transfermarkt; record now blank (title 'blank', size "
                       f"{f['versions']['concept']['size']}; API returns 404)",
         "transfermarkt_player_id": Y + f" ({DOC})", "player_name": Y, "season": Y, "injury_type": Y, "start_date": Y,
         "end_date": Y, "days_missed": Y, "games_missed": Y, "club": NO, "rows": f["injuries"], "players": f["players"],
         "seasons": "not documented", "leagues": "not documented", "join_without_fuzzy_matching": Y,
         "downloaded": "no (record appears withdrawn)", "viable": "no: not currently available"},
        {"source": o["sanan"]["title"] + " (Kaggle)", "creator": o["sanan"]["owner"],
         "location": "https://www.kaggle.com/datasets/sananmuzaffarov/european-football-injuries-2020-2025",
         "stated_license": o["sanan"]["license"], "license_status": "explicit (share-alike)", "data_files_licensed": "yes",
         "attribution_required": "yes (BY-SA)", "original_source": "not stated (fields mirror Transfermarkt's)",
         "collection_method": "thesis project; not documented", "last_updated": o["sanan"]["last_updated"][:10],
         "redistribution_statement": "CC BY-SA 4.0", "research_restrictions": "share-alike on derivatives",
         "provenance": "unclear", "transfermarkt_player_id": "no id described", "player_name": Y, "season": Y,
         "injury_type": Y, "start_date": Y, "end_date": Y, "days_missed": Y, "games_missed": Y, "club": Y,
         "rows": o["sanan"]["records"], "players": "not stated", "seasons": "2020/21–2024/25", "leagues": "Big Five only",
         "join_without_fuzzy_matching": "no (names)", "downloaded": NO, "viable": "no: name matching, 5 seasons, 5 leagues"},
        {"source": o["amrit"]["title"] + " (Kaggle)", "creator": o["amrit"]["owner"],
         "location": "https://www.kaggle.com/datasets/amritbiswas007/player-injuries-and-team-performance-dataset",
         "stated_license": o["amrit"]["license"], "license_status": "explicit", "data_files_licensed": "yes",
         "attribution_required": "no", "original_source": "Transfermarkt, Football Critic, Sky Sports (stated)",
         "collection_method": "not documented", "last_updated": o["amrit"]["last_updated"][:10],
         "redistribution_statement": "CC0", "research_restrictions": "none stated", "provenance": "mixed sources",
         "transfermarkt_player_id": NO, "player_name": Y, "season": Y, "injury_type": Y, "start_date": Y, "end_date": Y,
         "days_missed": "not stated", "games_missed": "not stated", "club": Y, "rows": o["amrit"]["records"],
         "players": "not stated", "seasons": "2019–2023", "leagues": "Premier League, 8 clubs",
         "join_without_fuzzy_matching": NO, "downloaded": NO, "viable": "no: tiny, one league, names"},
        {"source": "SoccerMon (Zenodo, 10.5281/zenodo.10033832)", "creator": "Simula / SoccerMon authors",
         "location": "https://www.nature.com/articles/s41597-024-03386-x", "stated_license": "CC BY 4.0 (per search result)",
         "license_status": "explicit (not verified here)", "data_files_licensed": "yes", "attribution_required": "yes",
         "original_source": "team monitoring system", "collection_method": "athlete self-reports and GPS",
         "last_updated": "", "redistribution_statement": "CC BY 4.0", "research_restrictions": "none stated",
         "provenance": "clear (academic)", "transfermarkt_player_id": NO, "player_name": "anonymised", "season": Y,
         "injury_type": "partly", "start_date": "partly", "end_date": "partly", "days_missed": NO, "games_missed": NO,
         "club": "two Norwegian women's teams", "rows": "", "players": "", "seasons": "2020–2021", "leagues": "Norway (women)",
         "join_without_fuzzy_matching": NO, "downloaded": NO, "viable": "no: different population, tiny injury subset"},
        {"source": "dcaribou/transfermarkt-datasets (our upstream)", "creator": "dcaribou",
         "location": "https://github.com/dcaribou/transfermarkt-datasets", "stated_license": N["dcaribou"]["license"],
         "license_status": "explicit", "data_files_licensed": "yes", "attribution_required": "no",
         "original_source": "Transfermarkt", "collection_method": "documented extraction pipeline",
         "last_updated": "", "redistribution_statement": "CC0", "research_restrictions": "none stated",
         "provenance": "the project already relies on it", "transfermarkt_player_id": Y, "player_name": Y, "season": "",
         "injury_type": NO, "start_date": NO, "end_date": NO, "days_missed": NO, "games_missed": NO, "club": Y,
         "rows": 0, "players": "", "seasons": "", "leagues": "", "join_without_fuzzy_matching": Y, "downloaded": "already in repo",
         "viable": "no injury assets"},
        {"source": "live scrapers and unofficial APIs (worldfootballR, Apify actors, community Transfermarkt APIs)",
         "creator": "various", "location": "", "stated_license": "varies", "license_status": "n/a",
         "data_files_licensed": "n/a", "attribution_required": "", "original_source": "Transfermarkt, live",
         "collection_method": "scrape on request", "last_updated": "", "redistribution_statement": "",
         "research_restrictions": "", "provenance": "our own automated access by another name",
         "transfermarkt_player_id": Y, "player_name": Y, "season": Y, "injury_type": Y, "start_date": Y, "end_date": Y,
         "days_missed": Y, "games_missed": Y, "club": "", "rows": "", "players": "", "seasons": "", "leagues": "",
         "join_without_fuzzy_matching": Y, "downloaded": NO, "viable": "excluded"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(N: dict, T: pd.DataFrame) -> str:
    p, f, pc, h = N["primary"], N["figshare"], N["preview_check"], N["hoenig"]
    inj = next(iter(p["injury_files"]))
    W = ["# External football injury datasets: audit", "",
         f"*Generated by `python -m src.analysis.injury_external_sources` from metadata and documentation captured on "
         f"{N['captured_on']} in `data/raw/injury_source_metadata/`. No injury data file was downloaded; no Transfermarkt "
         "page was fetched. Licence facts are reported as published; no legal conclusion is drawn.*", ""]
    W += ["## Decision: C, technically useful but licensing and provenance need approval", "",
          f"- Every candidate with Transfermarkt player ids and injury dates is itself a copy of Transfermarkt's injury "
          f"pages. The primary candidate (`{p['name']}`) has **no licence on GitHub** and a **{p['kaggle_license']}** "
          "dedication on its Kaggle mirror by the same uploader. The Figshare dataset carries "
          f"**{f['license']}** but says it was collected by web scraping, and its record now appears to be blank.",
          "- Transfermarkt's own terms (captured in the previous phase) prohibit copying its content by automated means "
          "and reserve text and data mining rights. Whether a third party's licence or dedication makes the data "
          "reusable for this project is the question for review. This audit does not answer it.",
          "- Because no candidate has a clear reuse basis, nothing was downloaded, and the join, validation and coverage "
          f"tests were not run ({', '.join(NOT_WRITTEN)} are not written).",
          "- Even with approval, a published comparison finds lower injury incidence and a higher share of severe "
          "injuries in Transfermarkt data than in scientific surveillance, with differences by country and year "
          "(section 4). That points to a restricted robustness use (severe injuries), not the main analysis.", ""]

    W += ["## 1. Primary candidate: salimt/football-datasets", ""]
    W += md_table(["", "Finding"],
                  [["creator / location", f"GitHub `{p['name']}` (owner {p['creator']}); Kaggle mirror by {p['kaggle_owner']}"],
                   ["licence (GitHub)", "none: no LICENSE file in the repository, GitHub detects none"
                    + ("; the README shows a licence badge pointing to the missing file" if p["readme_license_badge"] else "")],
                   ["licence (Kaggle mirror)", f"{p['kaggle_license']} (Kaggle metadata)"],
                   ["original source", f"{p['stated_source']} (the repository description says the data are from it)"],
                   ["collection method", "not documented; " + ("collection code is in the repo" if p["collection_code_in_repo"]
                                                               else "no collection code in the repo")],
                   ["last updated", f"GitHub push {p['github_pushed_at'][:10]}; Kaggle {p['kaggle_last_updated'][:10]}; "
                                    f"README badge '{p['readme_last_updated_badge']}'"],
                   ["redistribution / research terms", "nothing stated on GitHub; CC0 dedication on Kaggle"],
                   ["injury table", f"{f0(p['injury_rows'])} rows (README); files: "
                    + "; ".join(f"`{k.split('/')[-1]}` ({v / 1e6:.1f} MB)" for k, v in p["injury_files"].items())],
                   ["columns", ", ".join(f"`{c}`" for c in p["injury_columns"])],
                   ["players", f"{f0(p['players_in_datalake'])} in the datalake; how many have injury rows is not stated"],
                   ["seasons / leagues", "not documented"]], right=set())
    W += ["", "**Schema fit.** It has everything the fit-adjusted outcome needs (Transfermarkt player id, season, "
          "injury type, start and end date, days and games missed), so it would join on player id without any name "
          "matching. It has no club column and stores days as text ('11 days') and blank games as '-'. Rows include "
          "non-medical reasons, such as 'Rest', which would need classifying.", "",
          f"**ID check on the README's own sample rows** (documentation only, not a download): id {pc['player_id']} is "
          f"{pc['our_name']} in our snapshot too. In every sample spell he has no appearance:", ""]
    W += md_table(["Injury", "From", "Until", "Games missed (source)", "Club league games in spell (ours)",
                   "His appearances in spell (ours)"],
                  [[r["injury"], r["from"], r["until"], r["games_missed_source"], r["club_league_games_in_spell"],
                    r["player_appearances_in_spell"]] for r in pc["rows"]])
    W += ["", "Games missed in the source count all competitions, so they can exceed our league count. Five rows show "
          "that the ids line up, not that coverage is complete.", ""]

    W += ["## 2. Other candidates", ""]
    cols = ["source", "stated_license", "license_status", "original_source", "transfermarkt_player_id", "start_date",
            "end_date", "rows", "seasons", "leagues", "viable"]
    W += md_table(["Source", "Licence", "Licence status", "Original source", "TM player id", "Start", "End", "Rows",
                   "Seasons", "Leagues", "Viable?"],
                  [[f0(r[c]) if isinstance(r[c], (int, float)) and not isinstance(r[c], bool) and r[c] == r[c]
                    else str(r[c]) for c in cols] for _, r in T.iterrows()], right=set())
    W += ["", f"The Figshare record (DOI {f['doi']}) by {f['creator']}: version 2 lists {f0(f['injuries'])} injuries "
          f"across {f0(f['players'])} players with the Transfermarkt player id and name. Its description says: "
          f"\"{f['collection']}\" The newest metadata (updated {f['versions']['concept']['updated']}) show the title "
          f"'{f['versions']['concept']['title']}' and size {f['versions']['concept']['size']}, and Figshare's API returns "
          "404, so the files no longer appear to be available.", "",
          "This search found no academic release of professional men's injury spells with Transfermarkt ids, and no "
          "public release of the club injury-surveillance data used in sports-medicine research. Paid feeds are "
          "excluded by this project's rules.", ""]

    W += ["## 3. Join, validation and coverage", "",
          "Not run. They need the data files, and no candidate with player ids has a clear reuse basis. If a reuse basis "
          "is approved, the tests are already specified by this phase and can reuse the previous phase's code: exact "
          "join on player id for the 10,188 players; in-match injury substitutions as the validation signal (same day, "
          "±1, ±3, ±7 days); coverage by league, season, move type and age; and the rules for medical unavailability "
          "written down in `injury_pilot_summary.md`.", ""]

    W += ["## 4. What the literature says about Transfermarkt injury data", "",
          f"{h['citation']}.", "", f"> {h['results']}", "", f"> {h['conclusion']}", "",
          "For the fit-adjusted denominator this means two things. Lower incidence with more severe injuries suggests "
          "minor injuries are missing, so the adjustment would mostly remove long absences. And recording differs by country and year, so a loan/permanent "
          "comparison could pick up recording differences between the clubs and leagues loanees and permanent signings "
          "move to. A coverage audit against our in-match injury substitutions would be the first test.", ""]

    W += ["## 5. The question to settle first", "",
          f"The project already uses Transfermarkt-derived data: our snapshot comes from `dcaribou/transfermarkt-"
          f"datasets`, published under {N['dcaribou']['license']}, which has no injury table. The decision for Professor "
          "Freund (and, if needed, MIT counsel) is whether a third-party copy such as the salimt Kaggle mirror "
          f"({p['kaggle_license']}) meets the same standard for a robustness check. If it does: download that one "
          "file, run the join and validation, and use it only for a severe-injury robustness analysis. If it does not: "
          "ask Transfermarkt for permission or a licence, or leave injury adjustment out.", ""]
    return "\n".join(W)


def main() -> dict:
    from .club_geography_v2 import CANONICAL_CSV, STAGE1C_SHA256, sha256

    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    old = [OUT / n for n in ("injury_pilot_summary.md", "injury_pilot_numbers.json", "playing_time_injury_feasibility.md",
                             "playing_time_injury_numbers.json", "playing_time_sample_redesign_numbers.json")]
    before = {p.name: sha256(p) for p in old if p.exists()}
    assert not {p.name for p in WRITES} & set(before)
    N = compute()
    T = comparison(N)
    T.to_csv(COMPARISON_CSV, index=False)
    AUDIT_MD.write_text(render(N, T))
    N["old_output_sha256"] = before
    NUMBERS_JSON.write_text(json.dumps(N, indent=1, default=str))
    assert {p.name: sha256(p) for p in old if p.exists()} == before
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print_summary(N)
    return N


def print_summary(N: dict) -> None:
    p, pc = N["primary"], N["preview_check"]
    print("PRIMARY CANDIDATE")
    print(f"- source: {p['name']} (GitHub; Kaggle mirror by {p['kaggle_owner']})")
    print(f"- rows: {p['injury_rows']:,} injury rows (README)")
    print(f"- players: {p['players_in_datalake']:,} in the datalake; players with injury rows not stated")
    print("- seasons: not documented (README sample rows are 2015/16)")
    print(f"- license: none on GitHub (no LICENSE file); {p['kaggle_license']} declared on the Kaggle mirror")
    print(f"- original source: {p['stated_source']} (collection method not documented)")
    print("JOIN")
    print("- our 10,188 players matched: not tested (no file downloaded: no clear reuse basis)")
    print(f"- exact-id match rate: n/a; README sample id {pc['player_id']} = {pc['our_name']} in our data, "
          f"{sum(r['player_appearances_in_spell'] == 0 for r in pc['rows'])}/{len(pc['rows'])} sample spells without an appearance")
    print("VALIDATION")
    print("- injury substitution events checked: 0 (not run)")
    print("- matched ±3d: n/a\n- matched ±7d: n/a")
    print("COVERAGE")
    print("- usable loan episodes: unknown\n- usable permanent episodes: unknown")
    print("- strong/weak seasons: unknown; Hoenig et al. 2022 find lower injury incidence and more severe injuries "
          "in Transfermarkt data than in scientific surveillance, with differences by country and year")
    print("PROVENANCE")
    print("- needs review: Transfermarkt-derived; no licence on GitHub, CC0 claimed on Kaggle; Figshare CC BY 4.0 copy "
          "stated as scraped and now appears withdrawn")
    print("RECOMMENDATION")
    print("- SEEK APPROVAL (category C). If approved, use the salimt Kaggle copy only for a severe-injury robustness "
          "check after a coverage and validation audit; not for the main analysis.")


if __name__ == "__main__":
    main()
