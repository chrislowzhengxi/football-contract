"""Geography v2: a calendar country for every Stage 1C club, with exact provenance.

    python -m src.analysis.club_geography_v2 [--refresh-wikidata]

Precedence, fixed in advance:
  1. the league/competition country from `club_geography` (v1) is kept whenever it exists;
  2. otherwise the raw Transfermarkt `countryFlag` fills it, when every raw row for the club carries the same
     flag (`src.stage1b.club_flags`, read from the verified local cache of Stage 1B's own source files);
  3. clubs still without a country try, in order: A competition membership, B a validated parent-club link,
     C Wikidata (Transfermarkt team id P7223 -> country P17). D (another structured source) was not needed.
A disagreement between sources is recorded, never resolved by overwriting. No country comes from a club name:
the parent link needs Stage 1C's own academy/reserve normalisation, one mapped parent country, and a recorded
move between the two ids, and it is used only where the raw flag is missing or self-contradictory.

Writes new files only; v1 geography, Stage 1C and the v1 calendar outputs are not touched.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from . import loan_country_mapping_audit as audit
from .daniel_scope_final import f0, md_table, pc
from .loan_calendar_rule import population
from .loan_country_dates import context
from .loan_episodes import CANONICAL_CSV, ROOT, club_geography
from .stage1_scope_report import display_country as dc
from ..stage1b import club_flags
from ..stage1c.policy import PLACEHOLDER_CLUBS, YOUTH_OR_RESERVE

OUT = ROOT / "data" / "outputs" / "rebuild"
RAW_FLAGS_CSV = club_flags.RAW_FLAGS_CSV
GEO_V2_CSV = OUT / "club_geography_v2.csv"
RESIDUAL_CSV = OUT / "club_geography_v2_residual_audit.csv"
SUMMARY_MD = OUT / "club_geography_v2_summary.md"
NUMBERS_JSON = OUT / "club_geography_v2_numbers.json"
WRITES = (RAW_FLAGS_CSV, GEO_V2_CSV, RESIDUAL_CSV, SUMMARY_MD, NUMBERS_JSON)
WIKIDATA_CACHE = ROOT / "data" / "raw" / "wikidata" / "transfermarkt_team_p7223.json"
STAGE1C_SHA256 = "92f383d72a3ed59925c922f0d1b5f98528bc66ad3a9f49bb990867ed1d1d50dd"
UNMAPPED = "one or both clubs unmapped"

CONF = {"europa": "UEFA", "amerika": "CONMEBOL/CONCACAF", "asien": "AFC", "afrika": "CAF"}
SOURCES = {1: "v1 league/competition geography", 2: "raw Transfermarkt countryFlag",
           3: "fallback A: competition membership", 4: "fallback B: parent-club link",
           5: "fallback C: Wikidata P7223 -> P17"}
FLAG_USABLE = ("single_flag", "single_flag_not_in_snapshot_countries")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unnamed(fid) -> str:
    """Calendar key for a Transfermarkt country id the snapshot's `countries` table does not name."""
    return f"Transfermarkt country {int(fid)}"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def snapshot_country_table(db_path=None) -> pd.DataFrame:
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    try:
        d = con.sql("select cast(country_id as integer) id, country_name, confederation from countries").df()
    finally:
        con.close()
    d["confederation"] = d.confederation.map(CONF).fillna(d.confederation)
    return d


def wikidata_countries(ids: list[int], refresh: bool = False) -> dict[str, list[dict]]:
    """Wikidata items carrying each Transfermarkt team id, with their P17 country. Cached per id in
    WIKIDATA_CACHE; only ids never asked before trigger a request (25 ids per request)."""
    cache = json.loads(WIKIDATA_CACHE.read_text()) if WIKIDATA_CACHE.exists() and not refresh else \
        {"method": "Wikidata Action API: list=search haswbstatement:P7223, wbgetentities P7223/P17", "queried": {}}
    todo = [i for i in sorted(set(int(x) for x in ids)) if str(i) not in cache["queried"]]
    for k in range(0, len(todo), 25):
        chunk = todo[k:k + 25]
        res = audit.fetch_wikidata(chunk)
        if res["wikidata_error"]:
            raise RuntimeError(f"Wikidata request failed: {res['wikidata_error']}")
        for i in chunk:
            cache["queried"][str(i)] = res["wikidata"].get(str(i), [])
    if todo or not WIKIDATA_CACHE.exists():
        WIKIDATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        WIKIDATA_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return {str(i): cache["queried"].get(str(i), []) for i in ids}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(rows: pd.DataFrame, geo1: pd.DataFrame, flags: pd.DataFrame, ctab: pd.DataFrame,
          refresh_wikidata: bool = False) -> tuple[pd.DataFrame, dict]:
    """One row per club (Stage 1C clubs plus every v1-mapped club) with calendar_country and provenance."""
    names = audit.club_names(rows)
    stage1c_ids = set(names.club_id)
    ids = sorted(stage1c_ids | set(geo1.club_id.astype("int64")))
    top_name = names.sort_values(["club_id", "rows", "name"], ascending=[True, False, True]) \
        .drop_duplicates("club_id").set_index("club_id").name
    F = flags.set_index("club_id")
    fname = dict(zip(ctab.id, ctab.country_name))
    fconf = dict(zip(ctab.id, ctab.confederation))
    name_conf = {n: c for n, c in zip(ctab.country_name, ctab.confederation)}
    v1 = geo1.set_index("club_id")

    G = pd.DataFrame({"club_id": ids})
    G["club_name"] = G.club_id.map(top_name).fillna(G.club_id.map(F.club_names)).fillna("")
    G["in_stage1c"] = G.club_id.isin(stage1c_ids)
    G["league_country"] = G.club_id.map(v1.country).fillna("")
    G["league_source"] = G.club_id.map(v1.geo_source).fillna("")
    G["raw_flag_status"] = G.club_id.map(F.mapping_status).fillna("not_in_raw_files")
    G["raw_flag_ids"] = G.club_id.map(F.flag_ids).fillna("")
    G["raw_flag_rows"] = G.club_id.map(F.n_supporting_rows).fillna(0).astype(int)
    G["raw_first_date"] = G.club_id.map(F.first_date).fillna("")
    G["raw_last_date"] = G.club_id.map(F.last_date).fillna("")
    G["raw_example_transfer"] = G.club_id.map(F.example_transfer).fillna("")

    def label(fid):
        return fname.get(int(fid), unnamed(fid))

    G["raw_flag_countries_observed"] = G.raw_flag_ids.map(
        lambda s: "|".join(label(int(x)) for x in s.split("|")) if s else "")
    usable = G.raw_flag_status.isin(FLAG_USABLE)
    G["raw_flag_country"] = np.where(usable, G.raw_flag_countries_observed, "")
    G["special_entry"] = (G.club_id.map(F.special_entry).astype("boolean").fillna(False).astype(bool)
                          | G.club_name.isin(PLACEHOLDER_CLUBS))

    # ---- secondary evidence, for every club the league geography does not map
    unleague = set(G.loc[G.league_country.eq(""), "club_id"])
    K, Mper = audit.orphan_competitions(geo1)
    Mper = Mper.set_index("club_id")
    G["membership_competitions"] = G.club_id.map(Mper.orphan_competitions).fillna("")
    G["membership_country"] = G.club_id.map(Mper.membership_country).fillna("")
    L = audit.name_links(names, geo1, rows, unleague).set_index("club_id")
    G["parent_rule"] = G.club_id.map(L.name_rule).fillna("")
    G["parent_country"] = np.where(G.parent_rule.eq(audit.PROPOSED_RULE),
                                   G.club_id.map(L.name_country).fillna(""), "")
    G["parent_club_ids"] = G.club_id.map(L.linked_club_ids).fillna("")
    G["parent_direct_moves"] = G.club_id.map(L.direct_moves_with_same_name_club).fillna(0).astype(int)

    # ---- precedence 1 and 2
    G["calendar_country"], G["source_priority"] = "", 0          # 0 = no source
    lg = G.league_country.ne("") & ~G.special_entry
    G.loc[lg, "calendar_country"], G.loc[lg, "source_priority"] = G.league_country[lg], 1
    fl = G.calendar_country.eq("") & usable & ~G.special_entry
    G.loc[fl, "calendar_country"], G.loc[fl, "source_priority"] = G.raw_flag_country[fl], 2

    # ---- fallbacks, only for real clubs still without a country
    observed = G.raw_flag_countries_observed.str.split("|")

    def allowed(country, i):
        """A fallback may not contradict the raw rows: with conflicting flags it must pick one of them."""
        return bool(country) and (G.raw_flag_status[i] != "conflicting_flags" or country in observed[i])

    residual = G.index[G.calendar_country.eq("") & ~G.special_entry]
    wiki = wikidata_countries(list(G.club_id[residual]), refresh_wikidata) if len(residual) else {}
    snapshot_names = set(ctab.country_name)
    G["wikidata_items"], G["wikidata_country"] = "", ""
    for i in residual:
        w = wiki.get(str(G.club_id[i]), [])
        G.loc[i, "wikidata_items"] = "|".join(sorted({x["item"] for x in w}))
        cs = sorted({x["country"] for x in w if x["country"]})
        G.loc[i, "wikidata_country"] = "|".join(cs)
    for i in residual:
        for prio, country in ((3, G.membership_country[i]), (4, G.parent_country[i]),
                              (5, G.wikidata_country[i] if "|" not in G.wikidata_country[i]
                               and G.wikidata_country[i] in snapshot_names else "")):
            if allowed(country, i):
                G.loc[i, "calendar_country"], G.loc[i, "source_priority"] = country, prio
                break

    G["source"] = G.source_priority.map(lambda p: SOURCES.get(p, ""))
    G.loc[G.special_entry, "source"] = "special entry (not a club)"
    G["status"] = np.select(
        [G.special_entry, G.source_priority.eq(1), G.source_priority.eq(2) & G.raw_flag_status.eq("single_flag"),
         G.source_priority.eq(2), G.source_priority.isin([3, 4, 5]) & G.raw_flag_status.eq("conflicting_flags"),
         G.source_priority.isin([3, 4, 5])],
        ["special_entry", "league_country", "raw_flag", "raw_flag_country_id_only", "fallback_among_conflicting_flags",
         "fallback"], default="unresolved")
    G["confidence"] = G.status.map({"league_country": "high", "raw_flag": "high", "raw_flag_country_id_only": "high",
                                    "fallback_among_conflicting_flags": "medium", "fallback": "medium",
                                    "special_entry": "", "unresolved": ""})

    # ---- conflicts (recorded, never resolved by overwriting)
    flag_known = G.raw_flag_countries_observed.ne("")
    lvf = G.league_country.ne("") & flag_known & (G.raw_flag_country.ne(G.league_country)
                                                  | G.raw_flag_status.eq("conflicting_flags"))
    sec = G.source_priority.eq(2) & (
        (G.membership_country.ne("") & G.membership_country.ne(G.calendar_country))
        | (G.parent_country.ne("") & G.parent_country.ne(G.calendar_country)))
    G["conflict"] = (lvf | sec | G.raw_flag_status.eq("conflicting_flags")).fillna(False).astype(bool)
    G["conflict_type"] = np.select(
        [lvf, G.raw_flag_status.eq("conflicting_flags"), sec],
        ["league country differs from raw flag(s); league kept",
         "raw rows carry more than one flag", "raw flag differs from membership/parent evidence; flag kept"],
        default="")
    G["confederation"] = np.where(
        G.source_priority.eq(1), G.club_id.map(v1.confederation).fillna(""),
        G.calendar_country.map(name_conf).fillna(""))
    G.loc[G.source_priority.eq(2), "confederation"] = [
        fconf.get(int(x.split("|")[0]), "") if x else "" for x in G.loc[G.source_priority.eq(2), "raw_flag_ids"]]
    G["provenance"] = [provenance(r, K) for r in G.itertuples()]
    assert G.club_id.is_unique
    return G, {"orphan_codes": K.set_index("code").to_dict(orient="index"), "wikidata_queried": len(wiki)}


def provenance(r, K) -> str:
    p = r.source_priority
    if r.special_entry:
        return f"Transfermarkt special entry; raw status {r.raw_flag_status}"
    if p == 0:
        return ""
    if p == 1:
        return f"club_geography v1 ({r.league_source})"
    if p == 2:
        return (f"raw countryFlag id {r.raw_flag_ids} on {r.raw_flag_rows} raw row(s), {r.raw_first_date} to "
                f"{r.raw_last_date}; e.g. {r.raw_example_transfer}")
    if p == 3:
        k = K.set_index("code")
        return "; ".join(f"plays {c} (no competitions row): {int(k.mapped[c])} mapped participants, all "
                         f"{k.membership_country[c]}" for c in r.membership_competitions.split(",")
                         if k.membership_country.get(c))
    if p == 4:
        return (f"parent club id(s) {r.parent_club_ids}: same Stage 1C normalised organisation, one mapped country, "
                f"{r.parent_direct_moves} direct move(s)")
    return f"Wikidata {r.wikidata_items}: P7223 = {r.club_id}, P17 = {r.wikidata_country}"


def residual_audit(G: pd.DataFrame, endings: pd.Series) -> pd.DataFrame:
    """Every real club that had no country after the raw-flag pass, with what was tried and why it ended there."""
    R = G[(G.source_priority.eq(0) | G.source_priority.ge(3)) & ~G.special_entry].copy()
    R["endings_affected"] = R.club_id.map(endings).fillna(0).astype(int)
    R["attempted_sources"] = "v1 league geography; raw countryFlag; A membership; B parent link; C Wikidata"
    R["evidence"] = [
        f"raw: {r.raw_flag_status} [{r.raw_flag_countries_observed or 'none'}] on {r.raw_flag_rows} row(s); "
        f"A: {r.membership_country or 'none'}; B: {r.parent_rule or 'not run'} -> {r.parent_country or 'none'}; "
        f"C: {r.wikidata_country or 'no item'} {r.wikidata_items}".strip() for r in R.itertuples()]

    def why(r):
        if r.status != "unresolved":
            return ""
        if r.raw_flag_status == "conflicting_flags":
            return "raw rows carry several flags and no fallback picks one of them"
        if r.raw_flag_status == "no_flag":
            return "raw rows carry no flag and no fallback applies"
        if r.raw_flag_status == "not_in_raw_files":
            return "club id absent from the raw files and no fallback applies"
        return "no source gives a country"

    R["final_status"] = R.status
    R["reason_unresolved"] = [why(r) for r in R.itertuples()]
    cols = ["club_id", "club_name", "endings_affected", "attempted_sources", "evidence", "final_status",
            "calendar_country", "source", "reason_unresolved"]
    return R[cols].sort_values(["final_status", "endings_affected", "club_id"],
                               ascending=[False, False, True]).reset_index(drop=True)


def endings_coverage(R: pd.DataFrame, country: pd.Series) -> dict:
    """Realised endings by how many of the two clubs have a country under `country` (club_id -> country)."""
    lc = R.lender_club_id.astype("int64").map(country).replace("", np.nan)
    bc = R.borrower_club_id.astype("int64").map(country).replace("", np.nan)
    both, none = lc.notna() & bc.notna(), lc.isna() & bc.isna()
    return {"realised": len(R), "both_mapped": int(both.sum()), "one_missing": int((~both & ~none).sum()),
            "both_missing": int(none.sum()), "coverage": float(both.mean())}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def plural(n: int, word: str) -> str:
    return f"{f0(n)} {word}{'' if n == 1 else 's'}"


def flag_periods(sides: pd.DataFrame, ids, names: dict[int, str]) -> dict[int, str]:
    """For clubs with several flags: each flag's first and last transfer date in the raw rows."""
    out = {}
    s = sides[sides.club_id.isin(list(ids)) & sides.flag_id.notna()]
    for cid, g in s.groupby("club_id"):
        per = g.groupby("flag_id").date.agg(["min", "max", "size"]).sort_values("min")
        out[int(cid)] = "; ".join(f"{names.get(int(f), unnamed(f))} {r['min']} to {r['max']} ({r['size']} rows)"
                                  for f, r in per.iterrows())
    return out


def compute(u=None, flags: pd.DataFrame | None = None, refresh_wikidata: bool = False,
            sides: pd.DataFrame | None = None) -> tuple[dict, dict]:
    from . import loan_scope

    u = u or loan_scope.universe()
    geo1 = club_geography()
    ctab = snapshot_country_table()
    if flags is None:
        sides, flags = club_flags.build(set(u.rows.player_id.astype(int)))
    G, meta = build(u.rows, geo1, flags, ctab, refresh_wikidata)
    R = population(u)
    N: dict = {"realised": len(R), "geo1_clubs": len(geo1), "orphan_codes": meta["orphan_codes"],
               "wikidata_queried": meta["wikidata_queried"]}
    assert G.loc[G.source_priority.eq(1), "calendar_country"].eq(G.league_country[G.source_priority.eq(1)]).all()

    # ---- Part 1: raw flags
    N["raw"] = {"clubs": len(flags), "status": {k: int(v) for k, v in flags.mapping_status.value_counts().items()},
                "unnamed_ids": int(flags.loc[flags.mapping_status.eq("single_flag_not_in_snapshot_countries"),
                                             "country_flag_id"].nunique()),
                "players": int(u.rows.player_id.nunique())}
    cf = flags[flags.mapping_status.eq("conflicting_flags")]
    periods = flag_periods(sides, cf.club_id, dict(zip(ctab.id, ctab.country_name))) if sides is not None else {}
    N["conflicting_flag_clubs"] = [f"{r.club_names} ({r.club_id}): " + periods.get(int(r.club_id), r.flag_ids)
                                   for r in cf.itertuples()]

    # ---- the 4,541 v1-unmapped loan clubs
    S = audit.slots(R)
    ends = S.groupby("club_id").loan_event_id.nunique()
    allend = pd.concat([R.lender_club_id, R.borrower_club_id]).astype("int64").value_counts()
    S["club_id"] = S.club_id.astype("int64")
    U = G[G.club_id.isin(ends.index)].copy()
    assert len(U) == ends.size
    U["endings"] = U.club_id.map(ends)
    N["unmapped_before"] = {"clubs": len(U), "endings": int(R.context.eq(UNMAPPED).sum())}
    st = U.status.value_counts()
    N["unmapped_after"] = {
        "raw_flag_named": int(st.get("raw_flag", 0)), "raw_flag_id_only": int(st.get("raw_flag_country_id_only", 0)),
        "fallback": {SOURCES[p]: int(v) for p, v in U[U.source_priority.ge(3)].source_priority.value_counts().items()},
        "special": int(st.get("special_entry", 0)), "unresolved": int(st.get("unresolved", 0)),
        "raw_status": {k: int(v) for k, v in U.raw_flag_status.value_counts().items()},
        "conflicting_flags": int(U.raw_flag_status.eq("conflicting_flags").sum()),
        "flag_not_in_snapshot": int(U.raw_flag_status.eq("single_flag_not_in_snapshot_countries").sum()),
        "not_in_raw": int(U.raw_flag_status.eq("not_in_raw_files").sum()),
        "no_flag": int(U.raw_flag_status.eq("no_flag").sum())}
    N["unmapped_after"]["recovered_raw"] = N["unmapped_after"]["raw_flag_named"] + N["unmapped_after"]["raw_flag_id_only"]
    N["unmapped_after"]["recovered_fallback"] = sum(N["unmapped_after"]["fallback"].values())
    N["unmapped_after"]["unresolved_endings"] = int(U.loc[U.status.eq("unresolved"), "endings"].sum())
    N["unmapped_after"]["id_only_countries"] = {k: int(v) for k, v in U.loc[U.status.eq("raw_flag_country_id_only"),
                                                                             "calendar_country"].value_counts().head(8).items()}
    N["unmapped_after"]["id_only_endings"] = int(U.loc[U.status.eq("raw_flag_country_id_only"), "endings"].sum())
    N["unmapped_after"]["top_countries"] = {k: int(v) for k, v in U.loc[U.source_priority.eq(2), "calendar_country"]
                                            .value_counts().head(12).items()}

    # ---- conflicts and cross-checks
    lg = G[G.league_country.ne("") & G.raw_flag_countries_observed.ne("")]
    N["league_vs_flag"] = {"compared": len(lg), "single_flag_agree": int(lg.raw_flag_country.eq(lg.league_country).sum()),
                           "single_flag_disagree": int((lg.raw_flag_country.ne("") & lg.raw_flag_country.ne(lg.league_country)).sum()),
                           "conflicting_flags": int(lg.raw_flag_status.eq("conflicting_flags").sum()),
                           "examples": [f"{r.club_name} ({r.club_id}): league {r.league_country}, flag(s) "
                                        f"{r.raw_flag_countries_observed}" for r in
                                        lg[lg.conflict].sort_values("raw_flag_rows", ascending=False).head(25).itertuples()]}
    fl = G[G.source_priority.eq(2)]
    for col, lab in (("membership_country", "membership"), ("parent_country", "parent")):
        x = fl[fl[col].ne("")]
        N[f"flag_vs_{lab}"] = {"compared": len(x), "agree": int(x[col].eq(x.calendar_country).sum()),
                               "disagree_examples": [f"{r.club_name} ({r.club_id}): flag {r.calendar_country}, "
                                                     f"{lab} {getattr(r, col)}" for r in
                                                     x[x[col].ne(x.calendar_country)].itertuples()][:12]}
    N["conflicts"] = {k: int(v) for k, v in G[G.conflict].conflict_type.value_counts().items()}
    Gc = G[G.conflict].assign(endings=lambda d: d.club_id.map(allend).fillna(0).astype(int))
    N["conflict_rows"] = [f"{r.club_name} ({r.club_id}): calendar country {r.calendar_country or 'none'} "
                          f"from {r.source or 'no source'}; flags {r.raw_flag_countries_observed or 'none'}"
                          + (f"; league {r.league_country}" if r.league_country else "")
                          + (f"; parent {r.parent_country}" if r.parent_country else "")
                          + f"; {plural(r.endings, 'realised ending')}" for r in Gc.itertuples()]
    N["conflict_endings"] = int(Gc.endings.sum())
    fb = G[G.source_priority.ge(3)].assign(endings=lambda d: d.club_id.map(allend).fillna(0).astype(int))
    N["fallback_rows"] = [f"{r.club_name} ({r.club_id}): {r.calendar_country} via {r.source} "
                          f"(flags {r.raw_flag_countries_observed or 'none'}); {plural(r.endings, 'realised ending')}"
                          for r in fb.itertuples()]

    # ---- Part 5 coverage
    stage = G[G.in_stage1c]
    real = stage[~stage.special_entry]
    N["clubs"] = {"stage1c": len(stage), "special": int(stage.special_entry.sum()), "real": len(real),
                  "with_country": int(real.calendar_country.ne("").sum()),
                  "real_missing": int(real.calendar_country.eq("").sum()),
                  "coverage": float(real.calendar_country.ne("").mean()),
                  "by_status": {k: int(v) for k, v in real.status.value_counts().items()},
                  "v1_with_country": int(real.league_country.ne("").sum())}
    v1c = pd.Series(geo1.country.values, index=geo1.club_id.astype("int64"))
    v2c = G.set_index("club_id").calendar_country
    N["endings_v1"] = endings_coverage(R, v1c)
    N["endings_v2"] = endings_coverage(R, v2c)
    N["endings_v1"]["context_check"] = int(R.context.ne(UNMAPPED).sum())
    assert N["endings_v1"]["both_mapped"] == N["endings_v1"]["context_check"]
    li, bi = R.lender_club_id.astype("int64"), R.borrower_club_id.astype("int64")
    lcv, bcv = li.map(v2c).replace("", np.nan), bi.map(v2c).replace("", np.nan)
    gstat = G.set_index("club_id").status
    N["endings_v2"]["missing_side_status"] = {
        k: int(v) for k, v in pd.concat([li[lcv.isna()], bi[bcv.isna()]]).map(gstat).fillna("not in geography table")
        .value_counts().items()}
    RA = residual_audit(G, allend)
    N["residual"] = {"rows": len(RA), "resolved": int(RA.final_status.ne("unresolved").sum()),
                     "unresolved": int(RA.final_status.eq("unresolved").sum()),
                     "unresolved_in_loans": int((RA.final_status.eq("unresolved") & RA.endings_affected.gt(0)).sum()),
                     "unresolved_list": [f"{r.club_name} ({r.club_id}, {plural(r.endings_affected, 'ending')}): "
                                         f"{r.reason_unresolved}"
                                         for r in RA[RA.final_status.eq("unresolved")].head(20).itertuples()]}
    N["sources_used"] = {SOURCES[int(p)]: int(v) for p, v in
                         stage.source_priority[stage.source_priority.gt(0)].value_counts().sort_index().items()}
    return N, {"G": G, "U": U, "RA": RA, "R": R, "flags": flags}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(N: dict) -> str:
    ua, ub, cl = N["unmapped_after"], N["unmapped_before"], N["clubs"]
    e1, e2 = N["endings_v1"], N["endings_v2"]
    lvf, raw = N["league_vs_flag"], N["raw"]
    rs = N["residual"]
    W = ["# Club geography v2: a calendar country for every club", "",
         "*Generated by `python -m src.analysis.club_geography_v2`. New files only: Stage 1C, v1 `club_geography` "
         "and the v1 calendar outputs are unchanged. No country is read off a club name.*", ""]
    W += ["## Headline", "",
          f"- **Clubs.** Of {f0(cl['stage1c'])} Stage 1C club ids, {f0(cl['special'])} are Transfermarkt special "
          f"entries (Without Club, Retired, ...), not clubs. Of the {f0(cl['real'])} real clubs, "
          f"{f0(cl['with_country'])} now have a calendar country: **{pc(cl['with_country'], cl['real'], 2)}** "
          f"(v1: {f0(cl['v1_with_country'])}, {pc(cl['v1_with_country'], cl['real'], 1)}). "
          f"{f0(cl['real_missing'])} real club{'s remain' if cl['real_missing'] != 1 else ' remains'} without one.",
          f"- **Loan endings.** Of {f0(e2['realised'])} realised loan endings, both clubs now have a country in "
          f"{f0(e2['both_mapped'])}: **{pc(e2['both_mapped'], e2['realised'], 2)}** (v1: {f0(e1['both_mapped'])}, "
          f"{pc(e1['both_mapped'], e1['realised'])}). {f0(e2['one_missing'])} miss one club and "
          f"{f0(e2['both_missing'])} miss both.",
          f"- **The {f0(ub['clubs'])} v1-unmapped loan clubs.** {f0(ua['recovered_raw'])} recovered from the raw "
          f"flag, {f0(ua['recovered_fallback'])} by a fallback, {f0(ua['unresolved'])} unresolved, "
          f"{f0(ua['special'])} special entries.", ""]
    if cl["coverage"] >= 0.98:
        W += [f"Coverage of genuine clubs is {pc(cl['with_country'], cl['real'], 2)}: effectively complete.", ""]

    W += ["## 1. Raw Transfermarkt club → country flags", "",
          f"Read from the three cached season files of the upstream raw transfer histories (md5-verified against the "
          f"snapshot manifest), for all {f0(raw['players'])} Stage 1C players. Each from/to side carries the club id "
          f"and a `countryFlag` whose file name is Transfermarkt's country id, the id used by the snapshot's "
          f"`countries` table. One row per club in `{RAW_FLAGS_CSV.name}`."
          + (" Cached files: " + "; ".join(f"{k} md5 {v['md5']} ({'verified' if v['verified'] else 'NOT verified'})"
                                          for k, v in N["raw_cache"].items()) + "." if "raw_cache" in N else ""), ""]
    W += md_table(["Raw flag status", "Clubs"], [[k, f0(v)] for k, v in raw["status"].items()])
    W += ["", f"`single_flag_not_in_snapshot_countries`: a single, consistent country id that the snapshot's 124-country "
          f"table does not name ({raw['unnamed_ids']} distinct ids). These clubs get the id as their calendar key "
          "(\"Transfermarkt country N\"): two clubs with the same id are domestic, different ids are cross-border. "
          "Only the label is missing.", "",
          "**What the flag means.** The flag is recorded per transfer row, not once per club. For the clubs below "
          "the rows disagree. In some the change is clean in time (one flag ends before the other starts); in "
          "others the two flags interleave, so the raw rows alone cannot pick one. Each flag's date range:", ""]
    W += [f"- {x}" for x in N["conflicting_flag_clubs"]]
    W += ["", "These clubs are not assigned from the flag. A fallback may assign one only if it picks one of the "
          "observed flags:", ""]
    W += [f"- {x}" for x in N["fallback_rows"]] + [""]

    W += ["## 2. Precedence: the country for the calendar", ""]
    W += md_table(["Priority", "Source", "When used", "Clubs (Stage 1C)"],
                  [[1, SOURCES[1], "always, when it exists (the competition calendar the club plays under)",
                    f0(N["sources_used"].get(SOURCES[1], 0))],
                   [2, SOURCES[2], "missing league country; every raw row carries the same flag",
                    f0(N["sources_used"].get(SOURCES[2], 0))],
                   [3, SOURCES[3], "still missing; competition with no `competitions` row, all mapped participants "
                                   "in one country", f0(N["sources_used"].get(SOURCES[3], 0))],
                   [4, SOURCES[4], "still missing; youth/reserve marker, one mapped parent country, a recorded move",
                    f0(N["sources_used"].get(SOURCES[4], 0))],
                   [5, SOURCES[5], "still missing; exact Transfermarkt team id, one country named in the snapshot",
                    f0(N["sources_used"].get(SOURCES[5], 0))]], right={0, 3})
    W += ["", f"**League vs flag.** {f0(lvf['compared'])} league-mapped clubs also have raw flags: "
          f"{f0(lvf['single_flag_agree'])} agree, {f0(lvf['single_flag_disagree'])} carry a single different flag and "
          f"{f0(lvf['conflicting_flags'])} carry several. In every case the league country is kept and the conflict "
          "is recorded:", ""]
    W += [f"- {x}" for x in lvf["examples"]]
    for lab in ("membership", "parent"):
        d = N[f"flag_vs_{lab}"]
        W += ["", f"**Flag vs {lab} evidence** (flag-filled clubs that also have {lab} evidence): "
              f"{f0(d['agree'])} of {f0(d['compared'])} agree."
              + (" Disagreements, flag kept and recorded: " + "; ".join(d["disagree_examples"]) + "."
                 if d["disagree_examples"] else "")]
    W += ["", "All conflict types: " + "; ".join(f"{k}: {v}" for k, v in N["conflicts"].items())
          + f". Every recorded conflict, with the realised loan endings it touches ({f0(N['conflict_endings'])} in "
          "total):", ""]
    W += [f"- {x}" for x in N["conflict_rows"]]
    W += ["", "Where a youth side keeps its own flag (priority 2) while its parent is mapped by league, the two "
          "differ (Wales or Andorra vs England or Spain). If Professor Freund prefers the parent's league country "
          "for such sides, these are the rows to switch; the choice changes only the endings listed.", ""]

    W += ["## 3. The v1-unmapped loan clubs", ""]
    W += md_table(["Outcome", "Clubs"],
                  [["v1-unmapped club ids in realised loan endings", f0(ub["clubs"])],
                   ["recovered from the raw flag (named country)", f0(ua["raw_flag_named"])],
                   ["recovered from the raw flag (country id not named in the snapshot)", f0(ua["raw_flag_id_only"])],
                   *[[f"recovered by {k}", f0(v)] for k, v in ua["fallback"].items()],
                   ["special entries", f0(ua["special"])],
                   ["**unresolved**", f"**{f0(ua['unresolved'])}**"]])
    W += ["", "Raw status of the same clubs: " + "; ".join(f"{k} {v}" for k, v in ua["raw_status"].items()) + ".",
          "", "Largest recovered countries (raw flag): "
          + ", ".join(f"{dc(k)} {v}" for k, v in ua["top_countries"].items()) + ".",
          "", f"Clubs recovered with a country id the snapshot does not name take part in {f0(ua['id_only_endings'])} "
          f"of the endings they were missing from; the ids are led "
          f"by " + ", ".join(f"{k} ({v} clubs)" for k, v in ua["id_only_countries"].items()) + ".", ""]

    W += ["## 4. Residual hierarchy", "",
          f"{f0(rs['rows'])} real clubs had no country after the raw-flag pass. {f0(rs['resolved'])} were resolved by "
          f"a fallback and {f0(rs['unresolved'])} remain ({rs['unresolved_in_loans']} of them in realised loan "
          f"endings). Full detail in `{RESIDUAL_CSV.name}`. Wikidata was asked about {N['wikidata_queried']} club ids "
          "(cached).", ""]
    if rs["unresolved_list"]:
        W += [f"- {x}" for x in rs["unresolved_list"]] + [""]

    W += ["## 5. Coverage", "", "**A. Clubs** (Stage 1C club ids)", ""]
    W += md_table(["", "N"], [["Stage 1C unique club ids", f0(cl["stage1c"])],
                              ["special / non-club entries", f0(cl["special"])],
                              ["real clubs (denominator)", f0(cl["real"])],
                              ["real clubs with a calendar country (numerator)", f0(cl["with_country"])],
                              ["real clubs still missing", f0(cl["real_missing"])],
                              ["**coverage**", f"**{pc(cl['with_country'], cl['real'], 2)}**"]])
    W += ["", "**B. Realised loan endings**", ""]
    W += md_table(["", "v1", "v2"],
                  [["realised endings (denominator)", f0(e1["realised"]), f0(e2["realised"])],
                   ["both clubs mapped (numerator)", f0(e1["both_mapped"]), f0(e2["both_mapped"])],
                   ["one club missing", f0(e1["one_missing"]), f0(e2["one_missing"])],
                   ["both clubs missing", f0(e1["both_missing"]), f0(e2["both_missing"])],
                   ["**geography coverage**", f"**{pc(e1['both_mapped'], e1['realised'], 2)}**",
                    f"**{pc(e2['both_mapped'], e2['realised'], 2)}**"]])
    W += ["", "Status of the clubs behind the remaining gaps: "
          + ("; ".join(f"{k} {v}" for k, v in e2["missing_side_status"].items()) or "none") + ".", ""]
    return "\n".join(W)


def main(u=None, refresh_wikidata: bool = False) -> dict:
    from ..stage1b.raw_api import cache_season_files

    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    cache = cache_season_files()                       # verified cache: no download when present
    from . import loan_scope
    u = u or loan_scope.universe()
    sides, flags = club_flags.build(set(u.rows.player_id.astype(int)))
    N, tb = compute(u, flags, refresh_wikidata, sides)
    N["raw_cache"] = {s: {"md5": v["md5"], "verified": v["verified"]} for s, v in cache.items()}
    flags.to_csv(RAW_FLAGS_CSV, index=False)
    G = tb["G"]
    cols = ["club_id", "club_name", "calendar_country", "league_country", "raw_flag_country", "source",
            "source_priority", "conflict", "conflict_type", "special_entry", "status", "confidence", "confederation",
            "provenance", "league_source", "raw_flag_status", "raw_flag_ids", "raw_flag_countries_observed",
            "raw_flag_rows", "raw_first_date", "raw_last_date", "membership_competitions", "membership_country",
            "parent_rule", "parent_country", "parent_club_ids", "parent_direct_moves", "wikidata_items",
            "wikidata_country", "in_stage1c"]
    G[cols].to_csv(GEO_V2_CSV, index=False)
    tb["RA"].to_csv(RESIDUAL_CSV, index=False)
    md = render(N)
    for v in (N["clubs"]["stage1c"], N["clubs"]["with_country"], N["endings_v2"]["both_mapped"],
              N["unmapped_before"]["clubs"]):
        assert f0(v) in md, v
    N["outputs_sha256"] = {p.name: sha256(p) for p in (RAW_FLAGS_CSV, GEO_V2_CSV, RESIDUAL_CSV)}
    SUMMARY_MD.write_text(md)
    NUMBERS_JSON.write_text(json.dumps(N, indent=1, default=str))
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print(f"wrote {GEO_V2_CSV.name}, {RAW_FLAGS_CSV.name}, {RESIDUAL_CSV.name}, {SUMMARY_MD.name}")
    return N


def geography_frame(G: pd.DataFrame | None = None) -> pd.DataFrame:
    """geography_v2 in the shape `build_universe` expects (club_id, country, confederation, geo_source)."""
    G = pd.read_csv(GEO_V2_CSV, keep_default_na=False) if G is None else G
    g = G[G.calendar_country.astype(str).ne("")]
    return pd.DataFrame({"club_id": g.club_id.astype("int64"), "country": g.calendar_country,
                         "confederation": g.confederation.replace("", np.nan), "geo_source": g.source}) \
        .reset_index(drop=True)


if __name__ == "__main__":
    main(refresh_wikidata="--refresh-wikidata" in sys.argv)
