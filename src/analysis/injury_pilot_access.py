"""Transfermarkt injury-history pilot: access and terms check, and the stop decision.

    python -m src.analysis.injury_pilot_access

Reads only local files: the policy snapshot captured once in `data/raw/transfermarkt_policy_cache/` (robots.txt and
the published terms of use, with URL, status and sha256) and the existing injury-feasibility numbers. It makes no
web request and contains no code that fetches player pages. The published terms prohibit automated access, so the
pilot collection was not run; this module documents why, and records the source-independent rules that any
injury-adjusted outcome would need. New files only.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.robotparser

from .daniel_scope_final import f0, md_table
from .loan_episodes import ROOT

OUT = ROOT / "data" / "outputs" / "rebuild"
POLICY_DIR = ROOT / "data" / "raw" / "transfermarkt_policy_cache"
INJURY_CACHE = ROOT / "data" / "raw" / "transfermarkt_injury_cache"      # must stay absent or empty in this phase
SUMMARY_MD = OUT / "injury_pilot_summary.md"
NUMBERS_JSON = OUT / "injury_pilot_numbers.json"
WRITES = (SUMMARY_MD, NUMBERS_JSON)
FEASIBILITY_JSON = OUT / "playing_time_injury_numbers.json"
INJURY_PATH = "/player-slug/verletzungen/spieler/1"        # a representative injury-history path for robots.txt


# ---------------------------------------------------------------------------
# Policy snapshot
# ---------------------------------------------------------------------------

def load_snapshot() -> dict:
    meta = json.loads((POLICY_DIR / "policy_snapshot.json").read_text())
    for name, f in meta["files"].items():
        got = hashlib.sha256((POLICY_DIR / name).read_bytes()).hexdigest()
        assert got == f["sha256"], f"{name} does not match its recorded hash"
    return meta


def robots_verdict(meta: dict) -> dict:
    out = {}
    for name, f in meta["files"].items():
        if not name.startswith("robots_"):
            continue
        rp = urllib.robotparser.RobotFileParser()
        rp.parse((POLICY_DIR / name).read_text().splitlines())
        out[f["url"]] = {"injury_path_allowed_for_our_agent": rp.can_fetch(meta["user_agent"], INJURY_PATH),
                         "allowed_for_*": rp.can_fetch("*", INJURY_PATH), "disallowed_for_wget": not rp.can_fetch("wget", "/"),
                         "text": (POLICY_DIR / name).read_text().strip()}
    return out


def clean_text(page: str) -> str:
    page = re.sub(r"(?s)<(script|style)[^>]*>.*?</\1>", " ", page)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page))).strip()


def terms_clauses(meta: dict) -> dict:
    """Clause 11.1 of the terms (automated access) in each captured language, verbatim."""
    pats = {"terms_www.transfermarkt.com.html": r"11\.1\. The User is not permitted.*?expressly reserved\.",
            "terms_www.transfermarkt.de.html": r"11\.1\. Dem Nutzer ist es nicht gestattet.*?ausdrücklich vorbehalten\."}
    out = {}
    for name, pat in pats.items():
        m = re.search(pat, clean_text((POLICY_DIR / name).read_text()))
        out[meta["files"][name]["url"]] = m.group(0) if m else None
    return out


# ---------------------------------------------------------------------------
# Source-independent rules for medical unavailability (Part 9)
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("physical injury", "counts", "primary concept"),
    ("illness", "counts", "primary concept"),
    ("fitness / rehabilitation (e.g. 'lack of match fitness', 'rehab')", "does not count in the primary rule",
     "a separate sensitivity; the player may be available"),
    ("unknown medical issue", "counts only if the source labels it medical", "flagged; sensitivity without them"),
    ("suspension", "does not count", "only in a separate 'available for selection' definition"),
    ("international duty", "does not count", "not medical; league games clash rarely"),
    ("non-selection / absent from the squad", "never counts", "absence is not evidence of injury"),
]
EDGE_RULES = [
    ("injury begins on a match day", "If the player appears in that game, the game stays in the fit denominator (he "
     "was fit at kick-off). If he does not appear, the game is unavailable only if the spell starts that day or earlier."),
    ("injury ends on a match day", "Treat the end date as the last day out: the game is unavailable unless the player "
     "appears in it. An appearance always overrides a spell."),
    ("open injury (no end date)", "Do not infer a return date. Episodes whose window overlaps an open spell are flagged "
     "incomplete and left out of the fit-adjusted sample; a bound treats them as out until the next appearance."),
    ("overlapping injury rows", "Take the union of the intervals; count each game once."),
    ("return date before the next appearance", "The spell ends at its recorded end date. Later games count as fit, "
     "even if he does not play (that is non-selection)."),
    ("zero or blank games missed", "Use the dates, not the games-missed field; report where the two disagree as a "
     "quality check."),
    ("appearance inside a spell", "The appearance wins: the game counts as fit, and the conflict is logged."),
    ("missing start date", "The row cannot be placed in time: excluded and counted; the player's history is flagged."),
    ("player with no injury rows", "Not 'healthy' until historical coverage is shown to be complete for that league "
     "and season; otherwise the episode is missing, not zero."),
]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def compute() -> dict:
    meta = load_snapshot()
    F = json.loads(FEASIBILITY_JSON.read_text())
    cache_files = [p for p in INJURY_CACHE.rglob("*") if p.is_file()] if INJURY_CACHE.exists() else []
    N = {"decision": "STOP: published terms prohibit automated access; no injury page was requested",
         "snapshot": meta, "robots": robots_verdict(meta), "terms_11_1": terms_clauses(meta),
         "injury_pages_requested": 0, "injury_cache_files": len(cache_files),
         "players_attempted": 0, "pages_retrieved": 0, "rows_parsed": 0,
         "revised": F["revised"], "distinct_players": F["distinct_players"],
         "validation_pool": {k: {"episodes_with_injury_substitution": F["waterfall"][k]["with_any_injury_substitution_in_window"],
                                 "players_with_injury_substitution": F["waterfall"][k]["players_with_injury_substitution"]}
                             for k in ("loan", "permanent")}}
    assert N["injury_cache_files"] == 0, "injury pages are cached although collection was stopped"
    assert all(N["terms_11_1"].values()), "clause 11.1 not found in a captured terms page"
    return N


def render(N: dict) -> str:
    m = N["snapshot"]
    rb = N["robots"]
    t = N["terms_11_1"]
    en = next(v for k, v in t.items() if ".com/" in k)
    de = next(v for k, v in t.items() if ".de/" in k)
    vp = N["validation_pool"]
    W = ["# Transfermarkt injury-history pilot: stopped at the access check", "",
         "*Generated by `python -m src.analysis.injury_pilot_access` from the policy snapshot in "
         "`data/raw/transfermarkt_policy_cache/` (captured " + m["captured_on"] + "). No injury page was requested; "
         "no injury data exist in this repository. This is not legal advice.*", ""]
    W += ["## Decision", "",
          "**The pilot collection was not run.** Transfermarkt's published terms of use prohibit retrieving its content "
          "with bots, spiders, screen scraping or other automated processes. The phase instructions say to stop in "
          "that case, so no player page was fetched, not even the 3–5 pages meant for checking the page structure "
          "(a script fetching them would itself be automated access). Parts 1 and 3–8 are therefore not done; Part 9 "
          "(the rules) does not depend on the source and is written out below.", ""]
    W += ["## 1. What was checked", ""]
    W += md_table(["File", "URL", "HTTP", "sha256 (first 16)"],
                  [[name, f["url"], f["http_status"], f["sha256"][:16]] for name, f in m["files"].items()], right=set())
    W += ["", f"Each URL was requested once, with the user agent `{m['user_agent']}`.", "",
          "**robots.txt** (identical on .com and .co.uk):", "", "```", next(iter(rb.values()))["text"], "```", "",
          "For an injury-history path, robots.txt "
          + ("allows" if all(v["allowed_for_*"] for v in rb.values()) else "does not allow")
          + " any agent other than `wget`. robots.txt is a crawling convention; it does not override the terms of use.", "",
          "**Terms of use, clause 11.1** (verbatim):", "", f"> {en}", "", f"> {de}", "",
          "**What the published material says, without legal interpretation:** automated retrieval or copying of the "
          "site's content is not permitted; using the content to train AI is not permitted; and Transfermarkt reserves "
          "its rights against text and data mining under §44b of the German Copyright Act. Whether any statutory "
          "exception for scientific research applies to this project is a legal question for MIT's counsel. This "
          "report does not answer it.", ""]
    W += ["## 2. What the pilot would have answered, and what is still open", ""]
    W += md_table(["Question", "Status"],
                  [["URL pattern, server HTML vs JavaScript, available fields (Part 1)", "not inspected"],
                   ["cached, rate-limited collector (Part 3)", "not written"],
                   ["stratified pilot of about 150 players (Part 4)", "not drawn"],
                   ["parsed injury rows, page statuses A–E (Part 5)", "none"],
                   ["historical coverage by season, league, value, move type, age (Part 6)", "unknown"],
                   ["agreement with our in-match injury substitutions (Part 7)", "unknown"],
                   ["match-level demonstration (Part 8)", "not run: no confirmed spells exist"]], right=set())
    W += ["", "The central methodological risk is unchanged: without the pilot we do not know whether an empty "
          "Transfermarkt injury history means a healthy player or an unrecorded one, nor whether older seasons and "
          "smaller leagues are as complete as recent ones. The validation set for that check is ready in our own "
          f"data: {f0(vp['loan']['players_with_injury_substitution'])} loan players and "
          f"{f0(vp['permanent']['players_with_injury_substitution'])} permanent players in the revised sample have at "
          "least one in-match substitution tagged \"Injury\" during their window "
          f"({f0(vp['loan']['episodes_with_injury_substitution'])} and "
          f"{f0(vp['permanent']['episodes_with_injury_substitution'])} episodes).", ""]
    W += ["## 3. Rules for medical unavailability (any source)", "",
          "Primary concept: **a confirmed injury or illness spell from the source that covers the date of a receiving-"
          "club league game, unless the player appears in that game.** Suspensions and non-selection never count.", ""]
    W += md_table(["Category", "Primary rule", "Note"], [list(r) for r in CATEGORIES], right=set())
    W += ["", "Deterministic edge cases:", ""]
    W += md_table(["Case", "Rule"], [list(r) for r in EDGE_RULES], right=set())
    W += ["", "Outcome definitions once spells exist: current share = minutes / (90 × all league games of B in the "
          "window); fit-adjusted share = minutes / (90 × those games not covered by a confirmed spell under the rules "
          "above). An episode with an open spell, or in a league-season whose coverage is not validated, gets no "
          "fit-adjusted value.", ""]
    W += ["## 4. Routes that respect the terms", "",
          "1. **Ask Transfermarkt** for written permission, or a data licence, to retrieve injury histories for "
          f"about {f0(N['distinct_players'])} players for academic research. With permission, run the pilot as "
          "specified (one player at a time, cached, backing off on errors), starting with the coverage audit, not with "
          "scale.",
          "2. **Ask MIT counsel** whether a research exception applies despite clause 11.1, before any automated "
          "collection.",
          "3. **Licensed data:** a provider that sells injury spells would be compliant, but paid sources are excluded "
          "by this project's rules and would need a player-id mapping. Community datasets scraped from Transfermarkt "
          "carry the same terms problem.",
          "4. **Until then:** keep the current playing-time outcome as primary, and do not present any injury-adjusted "
          "number.", ""]
    return "\n".join(W)


def main() -> dict:
    from .club_geography_v2 import CANONICAL_CSV, STAGE1C_SHA256, sha256

    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    old = [OUT / n for n in ("playing_time_join_episodes.csv", "playing_time_sample_redesign_numbers.json",
                             "playing_time_age_numbers.json", "playing_time_injury_numbers.json",
                             "playing_time_regression_numbers.json")]
    before = {p.name: sha256(p) for p in old if p.exists()}
    N = compute()
    SUMMARY_MD.write_text(render(N))
    N["old_output_sha256"] = before
    NUMBERS_JSON.write_text(json.dumps(N, indent=1, default=str))
    assert {p.name: sha256(p) for p in old if p.exists()} == before
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print_summary(N)
    return N


def print_summary(N: dict) -> None:
    rb = next(iter(N["robots"].values()))
    print("ACCESS")
    print("- direct HTTP works?: not tested on injury pages (stopped before any was requested); robots.txt and the terms "
          "pages were served over plain HTTP (200)")
    print("- browser required?: not tested")
    print(f"- robots / terms issue?: robots.txt allows injury paths for generic agents ({rb['allowed_for_*']}); the terms "
          "of use, clause 11.1, prohibit access or copying by bots, spiders, screen scraping or other automated "
          "processes, and reserve rights against text and data mining (§44b UrhG)")
    print("- blocking / rate-limit behavior?: none seen on the 4 policy requests; injury pages not requested")
    print("PILOT")
    for k in ("players_attempted", "pages_retrieved"):
        print(f"- {k.replace('_', ' ')}: {N[k]}")
    print("- explicit empty histories: 0\n- pages with injury rows: 0\n- parse failures: 0 (no parsing)")
    print("HISTORICAL COVERAGE")
    print("- early seasons: unknown\n- recent seasons: unknown\n- obvious league differences?: unknown")
    print("VALIDATION AGAINST KNOWN INJURY SUBSTITUTIONS")
    vp = N["validation_pool"]
    print(f"- events checked: 0 (validation pool ready: {vp['loan']['players_with_injury_substitution']:,} loan and "
          f"{vp['permanent']['players_with_injury_substitution']:,} permanent players with an in-window injury substitution)")
    print("- matched same day: n/a\n- matched +/-3d: n/a\n- matched +/-7d: n/a")
    print("INJURY SPELL QUALITY")
    print("- rows parsed: 0\n- missing start date: n/a\n- missing end date: n/a\n- open injuries: n/a\n- duplicates/conflicts: n/a")
    print("MATCH-LEVEL DEMO")
    print("- episodes demonstrated: 0\n- examples where fit-adjusted denominator differs: none (no confirmed spells)")
    print("RECOMMENDATION")
    print("- none of READY TO SCALE / SCALE WITH RESTRICTIONS / NOT RELIABLE ENOUGH can be assigned: the pilot stopped "
          "at the access check. Seek written permission from Transfermarkt (or counsel's view) before any collection; "
          "until then keep the current outcome and present no injury-adjusted figure.")


if __name__ == "__main__":
    main()
