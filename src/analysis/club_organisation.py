"""Which youth/reserve side belongs to which senior club: a conservative, auditable organisation map (v2).

    python -m src.analysis.club_organisation

Stage 1C marks a move as internal only when the two club names are identical once an academy/reserve suffix is
stripped, so promotions such as "B. Dortmund II -> Dortmund" or "Schalke 04 U19 -> FC Schalke 04" enter the master as
permanent transfers. Transfermarkt publishes no parent-club field, so this module links a youth/reserve side to a
senior club only where two independent kinds of evidence agree:

  1. names: every Transfermarkt name ever shown for the two club ids (raw transfer histories, Stage 1C, the snapshot's
     `clubs` table), compared token by token after removing the youth/reserve marker and legal-form words. Tokens may
     match exactly, as an abbreviation ("Olymp." -> Olympique, "Utd" -> United), as an initial ("B." -> Borussia) or as
     an acronym ("RM" -> Real Madrid). No similarity scores are used.
  2. player flows: how many distinct players moved between the two clubs in either direction anywhere in Stage 1C,
     and whether another name-compatible club is about as connected.

Country flags must agree. Senior clubs whose names happen to end in "II" or "B" (Willem II) are never treated as a
youth side: any club with a top-flight league game in the snapshot is senior by definition. Stage 1C is read only.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from .daniel_scope_final import f0, md_table
from .loan_episodes import CANONICAL_CSV, ROOT
from ..stage1c.policy import PLACEHOLDER_CLUBS, YOUTH_OR_RESERVE, normalize_club_name

OUT = ROOT / "data" / "outputs" / "rebuild"
RAW_FLAGS_CSV = OUT / "club_country_raw_flags.csv"
MAP_CSV = OUT / "club_organisation_map.csv"
NUMBERS_JSON = OUT / "club_organisation_numbers.json"
WRITES = (MAP_CSV, NUMBERS_JSON)
STAGE1C_SHA256 = "92f383d72a3ed59925c922f0d1b5f98528bc66ad3a9f49bb990867ed1d1d50dd"

# Reserve-side naming that the Stage 1C pattern does not cover: numbered second teams ("Spartak 2", "Zenit 2 S-Pb",
# "Chornomorets-2"), "Jong X", "AKA X", "X Jugend", "X JV", "X Academy"/"X Acad.", "X Espoirs", "Reserve(s)"/"Res"/"Re.",
# age-group labels "Y19"/"M-17", and
# Transfermarkt's 15-character truncation of "Youth" ("CA Osasuna Yout", "RCD Mallorca Yo"). Used only here; Stage 1C's
# own flags are left as they are. Senior clubs are protected by the top-flight guard, not by this pattern.
V2_MARKERS = re.compile(r"(?:[\s-][23](?=\s|$)|\bJV$|Academy|\bAcade?\.$|Espoirs?$|\bYo(?:u|ut)?$|Reserves?$|\bRes$"
                        r"|\bRe\.$|Jugend$|^Jong\s|^AKA\s|\b[YM]-?1\d$)")
LEGAL = {"fc", "sc", "cf", "ac", "afc", "cd", "ud", "sd", "rc", "rcd", "ca", "cs", "fk", "nk", "sk", "hnk", "pfc", "sv",
         "tsv", "vfl", "vfb", "vfr", "fsv", "bsc", "ssc", "as", "us", "kv", "kaa", "krc", "rsc", "ogc", "cp", "if", "bk",
         "ff", "ik", "sco", "tsg", "1", "de", "la", "le", "the", "cfc", "ssd", "asd", "spvgg", "ks", "gks", "ofk"}
GENERIC = {"city", "united", "utd", "real", "sporting", "club", "atletico", "athletic", "olympique", "racing", "union",
           "stade", "sport", "dynamo", "dinamo", "deportivo", "calcio", "football", "futebol", "clube", "royal", "inter",
           "olympic", "rapid", "young", "boys", "town", "rovers", "county", "wanderers", "athletico", "fussball"}
# Hand-entered token equivalences, each checked against the club pages' own names in the data: Transfermarkt writes
# the youth sides with a city or nickname that never appears in the senior club's names.
TOKEN_ALIASES = {
    "rennes": "rennais",       # Rennes B / U19 -> Stade Rennais
    "hearts": "heart",         # Hearts FC B / U20 -> Heart of Midl.
    "bruges": "brugge",        # Club Bruges U19 -> Club Brugge
    "nizza": "nice",           # Nizza U19 -> Nice (Italian exonym)
    "nurnberg": "nuremberg",   # Nürnberg U19 -> 1.FC Nuremberg (German/English)
    "saloniki": "salonika",    # PAOK Saloniki B -> PAOK (its own variant "PAOK Salonika")
    "spurs": "tottenham",      # Spurs U18 -> Tottenham
}
MASTER_COLUMNS = ("from_club_parent_id_v2", "same_organisation_move_v2", "academy_promotion_candidate",
                  "same_org_confidence", "same_org_reason")
MIN_PLAYERS_HIGH = 2           # the move itself plus at least one other player
DOMINANCE = 2.0                # parent must have at least twice the players of any other name-compatible club
CONF_ORDER = {"high": 3, "medium": 2, "low": 1}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def side_marker(name: str) -> str | None:
    """Which youth/reserve marker a club name carries ("stage1c" pattern, "v2" extension), after trimming spaces."""
    n = str(name).strip()
    if YOUTH_OR_RESERVE.search(n):
        return "stage1c"
    if V2_MARKERS.search(n):
        return "v2"
    return None


def fold(name: str) -> list[str]:
    s = str(name).replace("ß", "ss").replace("ø", "o").replace("æ", "ae").replace("ł", "l").replace("đ", "d")
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).casefold()
    return [TOKEN_ALIASES.get(t, t) for t in re.split(r"[^a-z0-9]+", s) if t]


def base_tokens(name: str) -> list[str]:
    """The organisation part of a youth/reserve side's name: markers and legal-form words removed."""
    stripped = V2_MARKERS.sub(" ", YOUTH_OR_RESERVE.sub(" ", str(name).strip()))
    return [t for t in fold(stripped) if t not in LEGAL]


def senior_tokens(name: str) -> list[str]:
    return [t for t in fold(name) if t not in LEGAL]


def _subsequence(a: str, b: str) -> bool:
    it = iter(b)
    return all(ch in it for ch in a)


def tok_match(a: str, b: str) -> str | None:
    """How token a can stand for token b."""
    if a == b:
        return "exact"
    if len(a) >= 3 and b.startswith(a):
        return "prefix"
    if len(a) == 1 and b.startswith(a):
        return "initial"
    if len(a) >= 2 and a[0] == b[0] and len(a) < len(b) and _subsequence(a, b):
        return "abbreviation"
    return None


def _acronym(a: str, toks: list[str]) -> bool:
    return 2 <= len(a) <= 4 and any("".join(t[0] for t in toks[i:i + len(a)]) == a for i in range(len(toks) - len(a) + 1))


def _covers(A: list[str], B: list[str]) -> list[tuple[str, str]] | None:
    out = []
    for a in A:
        kinds = [k for b in B if (k := tok_match(a, b))] + (["acronym"] if _acronym(a, B) else [])
        if not kinds:
            return None
        out.append((a, min(kinds, key=["exact", "prefix", "acronym", "abbreviation", "initial"].index)))
    return out


def _distinctive(a: str, kind: str) -> bool:
    if a in GENERIC:
        return False
    return (kind == "exact" and len(a) >= 2) or (kind in ("prefix", "abbreviation") and len(a) >= 4) or kind == "acronym"


def name_evidence(child: list[str], parent: list[str]) -> tuple[int, str] | None:
    """(tier, text): 1 same name, 2 all tokens accounted for with a distinctive one, 3 only generic words shared."""
    if not child or not parent:
        return None
    if set(child) == set(parent):
        return 1, "same name once the youth/reserve marker and legal form are removed"
    best = None
    for A, B, who in ((child, parent, "youth side's"), (parent, child, "senior club's")):
        k = _covers(A, B)
        if k is None:
            continue
        if who == "senior club's":
            # the youth side may carry extra words only if they are initials, generic words or their abbreviations
            # ("B. Dortmund II" -> Dortmund), never another club's name ("CSKA Moscow II" -> FC Moscow)
            extra = [t for t in B if not any(tok_match(a, t) for a in A)]
            if any(len(t) > 1 and t not in GENERIC and not any(tok_match(t, g) for g in GENERIC) for t in extra):
                continue
        dist = [f"{a} ({kind})" for a, kind in k if _distinctive(a, kind)]
        if dist:
            return 2, f"every {who} name word is matched in the other name; distinctive: {', '.join(dist)}"
        best = best or (3, f"every {who} name word is matched, but only generic words: "
                           f"{', '.join(f'{a} ({kind})' for a, kind in k)}")
    return best


def club_names(S: pd.DataFrame, flags: pd.DataFrame, clubs: pd.DataFrame) -> pd.Series:
    """club_id -> every name Transfermarkt has shown for it (raw histories, Stage 1C, the snapshot's clubs table)."""
    raw = flags[["club_id", "club_names"]].dropna()
    raw = raw.assign(name=raw.club_names.str.split(r" \| ")).explode("name")[["club_id", "name"]]
    parts = [raw, clubs[["club_id", "name"]],
             S[["from_club_id", "from_club_name"]].set_axis(["club_id", "name"], axis=1),
             S[["to_club_id", "to_club_name"]].set_axis(["club_id", "name"], axis=1)]
    nm = pd.concat(parts).dropna()
    nm["club_id"] = nm.club_id.astype(int)
    nm["name"] = nm.name.astype(str).str.strip()
    nm = nm[nm.name.ne("")].drop_duplicates()
    return nm.groupby("club_id").name.agg(lambda s: sorted(set(s)))


def senior_league_clubs(con) -> set[int]:
    """Every club with a top-flight domestic-league game in the snapshot: senior by definition."""
    q = ("select home_club_id c from games where competition_type = 'domestic_league' union "
         "select away_club_id from games where competition_type = 'domestic_league'")
    return set(int(x) for x in con.execute(q).fetchdf().c.dropna())


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def flows(S: pd.DataFrame, side: pd.Series, placeholder: set[int]) -> pd.DataFrame:
    """Distinct players moving youth side -> senior club (up) and senior club -> youth side (down), per pair."""
    E = S.dropna(subset=["from_club_id", "to_club_id"]).astype({"from_club_id": int, "to_club_id": int})
    E = E[~E.from_club_id.isin(placeholder) & ~E.to_club_id.isin(placeholder)]
    fy, ty = E.from_club_id.map(side).fillna(False), E.to_club_id.map(side).fillna(False)
    up = E[fy & ~ty].groupby(["from_club_id", "to_club_id"]).player_id.nunique()
    down = E[~fy & ty].groupby(["to_club_id", "from_club_id"]).player_id.nunique()
    up.index.names = down.index.names = ["child", "cand"]
    F = pd.concat([up.rename("up"), down.rename("down")], axis=1).fillna(0).astype(int).reset_index()
    F["players"] = F.up + F.down
    F = F.sort_values(["child", "players", "up", "cand"], ascending=[True, False, False, True])
    F["overall_rank"] = F.groupby("child").cumcount() + 1
    return F


def build_map(S: pd.DataFrame, names: pd.Series, flag: pd.Series, senior: set[int]) -> tuple[pd.DataFrame, dict]:
    placeholder = {c for c, v in names.items() if any(n in PLACEHOLDER_CLUBS for n in v)}
    marker = names.map(lambda v: next((m for m in map(side_marker, v) if m), None))
    side = marker.notna() & ~names.index.isin(senior) & ~names.index.isin(placeholder)
    guard = marker.notna() & names.index.isin(senior)
    F = flows(S, side, placeholder)
    # the side's main senior destination in its own country (academies also loan players abroad: Burnley U23 -> Dundee)
    domestic = F[F.child.map(flag).eq(F.cand.map(flag)) | F.child.map(flag).isna() | F.cand.map(flag).isna()]
    top_all = domestic.sort_values(["child", "players", "up", "cand"], ascending=[True, False, False, True]) \
        .drop_duplicates("child").set_index("child")
    rows = []
    for child, g in F.groupby("child", sort=True):
        # "G.Antep" abbreviates one word (Gaziantep): also try each name with a dot that has no space after it joined
        bases = [(n, b) for n in names[child] if side_marker(n)
                 for b in {tuple(base_tokens(n)), tuple(base_tokens(re.sub(r"\.(?=\w)", "", n)))}]
        bases = [(n, list(b)) for n, b in bases]
        cands = []
        for r in g.itertuples(index=False):
            ev = None
            for cn, b in bases:
                for pn in names.get(r.cand, []):
                    e = name_evidence(b, senior_tokens(pn))
                    if e and (ev is None or e[0] < ev[0]):
                        ev = (e[0], e[1], cn, pn)
                if ev and ev[0] == 1:
                    break
            if ev:
                fc, fp = flag.get(child), flag.get(r.cand)
                country = "unknown" if pd.isna(fc) or pd.isna(fp) else ("same" if fc == fp else "different")
                cands.append({"cand": r.cand, "up": r.up, "down": r.down, "players": r.players,
                              "overall_rank": r.overall_rank, "tier": ev[0], "name_evidence": ev[1],
                              "child_name_used": ev[2], "parent_name_used": ev[3], "country": country})
        C = pd.DataFrame(cands)
        top = top_all.loc[child] if child in top_all.index else g.iloc[0]
        rec = {"child_club_id": child, "child_names": " | ".join(names[child]), "child_marker": marker[child],
               "child_country_flag": flag.get(child), "senior_clubs_connected": len(g),
               "players_connected": int(g.players.sum()), "top_destination_id": int(top.cand),
               "top_destination_name": names[int(top.cand)][0], "top_destination_players": int(top.players)}
        C = C[C.country.ne("different")] if len(C) else C
        if not len(C):
            rec.update(confidence="unresolved", reason="no connected senior club has a compatible name in the same country")
            rows.append(rec)
            continue
        C = C.sort_values(["tier", "players", "up", "cand"], ascending=[True, False, False, True])
        p = C.iloc[0]
        others = C.iloc[1:]
        nxt = int(others.players.max()) if len(others) else 0
        dominant = p.players >= DOMINANCE * nxt
        share_of_top = p.players / rec["top_destination_players"]
        if p.country != "same":
            conf, why = "medium", "country flag of one side unknown"
        elif p.tier == 3:
            conf, why = (("medium", "only generic name words shared, but the clearly dominant connected club")
                         if p.players >= MIN_PLAYERS_HIGH and dominant and p.overall_rank == 1 else
                         ("low", "only generic name words shared"))
        elif not dominant:
            conf, why = ("medium" if p.tier == 1 else "low",
                         f"another name-compatible club is about as connected ({nxt} vs {p.players} players)")
        elif p.players < MIN_PLAYERS_HIGH:
            conf, why = "medium", "names agree, but this move is the only one between the two clubs"
        elif share_of_top < 0.5:
            conf, why = "medium", (f"names agree, but the side's main senior destination is another club "
                                   f"({rec['top_destination_name']}, {rec['top_destination_players']} players)")
        else:
            conf, why = "high", (f"names agree, same country, {p.players} players connect them and no other "
                                 f"name-compatible club comes close")
        rec.update({"parent_club_id": int(p.cand), "parent_names": " | ".join(names[int(p.cand)]),
                    "parent_country_flag": flag.get(p.cand), "name_tier": int(p.tier), "name_evidence": p.name_evidence,
                    "child_name_used": p.child_name_used, "parent_name_used": p.parent_name_used,
                    "players_up": int(p.up), "players_down": int(p.down), "players_between": int(p.players),
                    "parent_overall_rank": int(p.overall_rank), "parent_share_of_top_destination": float(share_of_top),
                    "other_compatible_clubs": len(others), "next_compatible_players": nxt,
                    "other_compatible_ids": "|".join(str(int(x)) for x in others.cand),
                    "confidence": conf, "reason": why})
        rows.append(rec)
    MAP = pd.DataFrame(rows)
    lone = sorted(set(names.index[side]) - set(MAP.child_club_id))
    MAP = pd.concat([MAP, pd.DataFrame({
        "child_club_id": lone, "child_names": [" | ".join(names[c]) for c in lone],
        "child_marker": [marker[c] for c in lone], "child_country_flag": [flag.get(c) for c in lone],
        "senior_clubs_connected": 0, "players_connected": 0, "confidence": "unresolved",
        "reason": "no player moved between this side and any senior club in Stage 1C"})], ignore_index=True)
    gd = pd.DataFrame({"child_club_id": sorted(names.index[guard]),
                       "child_names": [" | ".join(names[c]) for c in sorted(names.index[guard])],
                       "child_marker": [marker[c] for c in sorted(names.index[guard])],
                       "confidence": "rejected_senior_club",
                       "reason": "plays top-flight league games in the snapshot, so a senior club; the 'II'/'B' is part "
                                 "of its name"})
    MAP = pd.concat([MAP, gd], ignore_index=True)
    for c in ("parent_club_id", "top_destination_id", "child_country_flag", "parent_country_flag"):
        MAP[c] = MAP[c].astype("Int64")
    N = {"club_ids": len(names), "youth_or_reserve_sides": int(side.sum()),
         "by_marker": {k: int(v) for k, v in marker[side].value_counts().items()},
         "senior_guard": int(guard.sum()), "senior_guard_names": sorted(names[c][0] for c in names.index[guard]),
         "sides_with_flows": int(F.child.nunique()), "pairs": len(F),
         "by_confidence": {k: int(v) for k, v in MAP.confidence.value_counts().items()},
         "players_between_by_confidence": {k: {kk: float(vv) for kk, vv in g.players_between.describe().items()}
                                          for k, g in MAP.dropna(subset=["players_between"]).groupby("confidence")}}
    return MAP, N


# Hand verdicts on the stratified review sample (review_sample): event_id -> (verdict, note). Verdicts:
# "same organisation", "not same organisation", "unclear".
_S, _N, _U = "same organisation", "not same organisation", "unclear"
MANUAL_REVIEW: dict[str, tuple[str, str]] = {
    # high
    "tm_01959025e7705ebae4d8": (_S, "Club Brugge's U19"),
    "tm_0296ebbca4e37be9d22c": (_S, "Jong KAA Gent is Gent's reserve side"),
    "tm_021748251dd957807f6a": (_S, "Silkeborg academy"),
    "tm_09092584a86f588efa65": (_S, "Nordsjælland academy (Right to Dream pathway)"),
    "tm_009611985626d76a7bee": (_S, "Gabri Veiga: Celta B to the first team"),
    "tm_00a1226dbba0829ee722": (_S, "Atlético Madrid B to the first team"),
    "tm_01ee4abab7c0dae0e97f": (_S, "Nsoki: PSG reserve to first team"),
    "tm_02d7c80ddb2f2c149473": (_S, "Guessand: Nice B to first team"),
    "tm_0601984742e3dfd46139": (_S, "Valery: Southampton U23 to first team"),
    "tm_09130da7b70d483c8f6a": (_S, "Ozoh: Palace academy"),
    "tm_0048dc929aa1724f394b": (_S, "PAO = Panathinaikos"),
    "tm_00ea7062fe498b9ed045": (_S, "Asteras Tripolis academy"),
    "tm_0c1978f676beff6790ef": (_S, "Guarino: Empoli Primavera"),
    "tm_0e1f3764eaa9229eaa8d": (_S, "Milan Primavera"),
    "tm_0265564a2eb7e457a1ba": (_S, "Stanišić: Bayern II to first team"),
    "tm_0514f677a5d5e46636bb": (_S, "Pavlović: Bayern II to first team"),
    "tm_006b8bf534dbb5dd2ca1": (_S, "Fortuna Sittard U21"),
    "tm_019be098dc050f2333c5": (_S, "FC Volendam U21"),
    "tm_0079c0354f6dacf05403": (_S, "Vitória Guimarães B"),
    "tm_03874d123da4fd56c87f": (_S, "Vitória Guimarães B"),
    "tm_0013e23a5bf588fe4faa": (_S, "Rubin-2 is Rubin Kazan's second team"),
    "tm_00a53a7956ed2212c019": (_S, "Rostov-2 is FK Rostov's second team"),
    "tm_0284872b71962dbe39fa": (_S, "Ryan Gauld: Dundee United youth"),
    "tm_0492e405863bcf6ecdd2": (_S, "Dundee United youth"),
    "tm_0c4893f5710f0be96bd5": (_S, "Gençlerbirliği U21"),
    "tm_0d0dfca66814f70b89d6": (_S, "Fenerbahçe U21"),
    "tm_00841bab560082a36645": (_S, "Shakhtar Donetsk U19"),
    "tm_033498c7548fcb9782be": (_S, "Zorya Luhansk U19"),
    # medium / low
    "tm_5219f19625882000b4fb": (_S, "Celta U19 (low only because Celta's B side is about as connected)"),
    "tm_21e21e23272f24aff869": (_S, "Sint-Truiden U18"),
    "tm_cd6a19a9d100f821f8c7": (_S, "AC Horsens youth"),
    "tm_19692d52fd47e7e44ba9": (_S, "Yéremy Pino: Villarreal C to first team"),
    "tm_79406699e3fe7efeddb9": (_S, "Rennes U17 to Stade Rennais"),
    "tm_f93a9aba47e35e75acda": (_S, "Panserraikos U17"),
    "tm_6ff56bcb18c10857ea0e": (_S, "Parma youth"),
    "tm_f3971ef2d4bfaadfa412": (_S, "Laurin Ulrich: VfB Stuttgart U17"),
    "tm_02cce31d7532d7fd1f35": (_S, "Telstar U19"),
    "tm_12d3f4642a3bb7af1e7f": (_S, "Matchoi Djaló: Paços de Ferreira U17"),
    "tm_074bce1e9bfec7c491c3": (_S, "Dynamo-2 Moscow is Dynamo's second team"),
    "tm_794a82c379b703f1c3ee": (_S, "Falkirk reserves"),
    "tm_67b661c4f6d650bc9451": (_S, "Bodrum FK U19 (parent's other name BB Bodrumspor)"),
    "tm_124ac6ac2ba4316a3bc5": (_S, "Olimpik Donetsk II"),
    # rejected: earlier rough screen only
    "tm_290b880e6ec32e17ce4c": (_N, "Dynamo Kyiv II to Arsenal Kyiv: two organisations sharing a city"),
    "tm_45339f2e0a4a597c3d81": (_N, "Dynamo Kyiv II to Arsenal Kyiv"),
    "tm_66b1c25a8c8c4ae67f44": (_N, "Dynamo Kyiv II to Arsenal Kyiv"),
    "tm_b180753cf87a24b1f082": (_N, "Dynamo Kyiv II to Arsenal Kyiv"),
    # rejected: senior club named like a reserve side
    "tm_2f7262a7c89080c2d4c9": (_N, "Willem II is a senior Eredivisie club"),
    "tm_c0eced7aade81eabb64b": (_N, "Willem II is a senior club"),
    "tm_ce75c608a8e4a04049a5": (_N, "Willem II is a senior club"),
    "tm_5fc455cb1876e2da9c1c": (_N, "Willem II is a senior club"),
    "tm_764265ffe72baaa69c7c": (_N, "Willem II is a senior club"),
    "tm_c87333d7f860da162d56": (_N, "Willem II is a senior club"),
    "tm_009c30b1285544d11531": (_N, "MVV to Willem II: two senior clubs"),
    "tm_01cd5f90e59cb0195f32": (_N, "Hamburg's reserve side to Willem II: different organisations"),
    "tm_0257d68e6576cfd1fb76": (_N, "Feyenoord U21 loaned to Willem II"),
    "tm_0b4b16618bff095631d1": (_N, "Hamilton Academical is a senior club"),
    "tm_2e9ce353d20efbbad40d": (_N, "Hamilton Academical is a senior club"),
    "tm_2f5c914dfb953f517b8f": (_N, "Lewis Ferguson: Hamilton Academical sold to Aberdeen"),
    "tm_1d4a506012617e0ed64e": (_N, "Willem II is a senior club"),
    # different organisation or unresolved
    "tm_0271535ab8e832971238": (_N, "Leicester and OH Leuven share an owner but are separate clubs"),
    "tm_0d72c05a50ebea721827": (_S, "MISSED: SVZW = SV Zulte Waregem; acronym not recoverable from its names"),
    "tm_0477ead8716db55d49bd": (_N, "Barça Atlètic to Viborg"),
    "tm_088ff5392d40dc464c01": (_N, "Barcelona B to Girona for a fee"),
    "tm_00ced0a8e52abb7d303b": (_N, "Lausanne II to Le Havre"),
    "tm_059eec9af4731fa98142": (_N, "Gallagher loaned from Chelsea to Crystal Palace"),
    "tm_0002b94d787768362c31": (_N, "Panathinaikos B to OFI"),
    "tm_081bde5ef860972b4ce7": (_N, "Ajax U21 loaned to Pescara"),
    "tm_0246ecb359b2593167fb": (_N, "Bayern U19 to Eintracht Frankfurt"),
    "tm_0093c9f8d8fe0a1a0967": (_N, "Danjuma: PSV U19 to NEC"),
    "tm_1570afae296dcde55c84": (_U, "Twente and Heracles run a joint academy; the side maps to FC Twente"),
    "tm_0003ff4429057afbe17e": (_N, "Benfica B to Alverca"),
    "tm_0101fb8a448e519e77e3": (_N, "Volgar (senior; one old name ended '-2') to Anzhi"),
    "tm_07548b726c0e69e90e3a": (_S, "MISSED: Lokomotiv-Kazanka is Lokomotiv Moscow's farm side; names share only 'Loko'"),
    "tm_05f0c15fe36ea11e72d7": (_N, "Leicester U21 to Motherwell"),
    "tm_13ecfedd0f5e57b079b2": (_N, "Rapid Wien II to Kasımpaşa"),
    "tm_00e20d9f000f75986ed7": (_N, "Dynamo Kyiv U19 loaned to Zorya"),
}


def load_map() -> pd.DataFrame:
    return pd.read_csv(MAP_CSV, dtype={"parent_club_id": "Int64", "child_club_id": "int64"})


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

ROUGH_GENERIC = {"city", "united", "real", "sporting", "club", "atletico", "athletic", "olympique", "racing", "union",
                 "stade", "sport", "dynamo", "dinamo", "deportivo", "calcio", "football", "futebol", "clube"}


def rough_stem_tokens(name: str) -> set:
    """The earlier playing-time QA screen: name words of four letters or more shared by the two clubs."""
    return {w for w in normalize_club_name(name).replace(".", " ").split() if len(w) >= 4 and w not in ROUGH_GENERIC}


def rough_screen(F: pd.DataFrame) -> pd.Series:
    a_youth = F.from_club_name.map(lambda x: bool(YOUTH_OR_RESERVE.search(str(x))))
    shared = pd.Series([bool(rough_stem_tokens(a) & rough_stem_tokens(b)) for a, b in zip(F.from_club_name, F.to_club_name)],
                       index=F.index)
    return F.is_youth_or_reserve_side.astype(bool) & a_youth & shared


def episode_flags(F: pd.DataFrame, MAP: pd.DataFrame) -> pd.DataFrame:
    """Per move: does it stay inside one organisation (each youth/reserve side mapped to its proposed senior parent)?"""
    K = MAP.set_index("child_club_id")
    resolved = K[K.confidence.isin(CONF_ORDER)]
    par, conf = resolved.parent_club_id, resolved.confidence
    fid = F.from_club_id.astype("Int64")
    tid = F.to_club_id.astype("Int64")
    f_org = fid.map(par).fillna(fid)
    t_org = tid.map(par).fillna(tid)
    rank = lambda s: s.map(CONF_ORDER)
    fc, tc = rank(fid.map(conf)), rank(tid.map(conf))
    same = fid.notna() & tid.notna() & fid.ne(tid) & f_org.eq(t_org) & (fc.notna() | tc.notna())
    level = pd.concat([fc, tc], axis=1).min(axis=1)
    inv = {v: k for k, v in CONF_ORDER.items()}
    confidence = level.where(same).map(inv)
    up = same & fc.notna() & tid.eq(fid.map(par))
    from_side = fid.isin(K.index[K.confidence.ne("rejected_senior_club")])
    to_side = tid.isin(K.index[K.confidence.ne("rejected_senior_club")])
    guard = fid.isin(K.index[K.confidence.eq("rejected_senior_club")]) | tid.isin(K.index[K.confidence.eq("rejected_senior_club")])
    rough = rough_screen(F)

    def why(i):
        if same[i]:
            k = fid[i] if up[i] else (tid[i] if pd.notna(tc[i]) else fid[i])
            r = K.loc[k]
            kind = "promotion from its own youth/reserve side" if up[i] else "move between sides of one organisation"
            return f"{kind}: {r.child_names.split(' | ')[0]} -> parent {r.parent_names.split(' | ')[0]}; {r.reason}"
        if from_side[i] and pd.notna(fid.map(par)[i]):
            p = K.loc[fid[i]]
            return (f"different organisation: {p.child_names.split(' | ')[0]} belongs to "
                    f"{p.parent_names.split(' | ')[0]} ({p.confidence})")
        if guard[i]:
            return "senior club whose name ends in a reserve-like suffix (e.g. Willem II); not a youth side"
        if from_side[i] or to_side[i]:
            return "youth/reserve side with no resolved parent club"
        return ""
    reason = pd.Series([why(i) for i in F.index], index=F.index)
    # 'rejected_name_overlap_only': the earlier rough name screen called this internal, but club B is a senior club other
    # than club A's resolved parent (Dynamo Kyiv II -> Arsenal Kyiv)
    status = np.select(
        [same & confidence.eq("high"), same & confidence.eq("medium"), same & confidence.eq("low"), guard,
         rough & fid.map(par).notna() & ~to_side, from_side | to_side],
        ["high", "medium", "low", "rejected_senior_club", "rejected_name_overlap_only",
         "different_organisation_or_unresolved"], default="no_youth_or_reserve_side")
    return pd.DataFrame({
        "from_club_parent_id_v2": fid.map(par).astype("Int64"),
        "same_organisation_move_v2": (same & confidence.eq("high")).astype(bool),
        "academy_promotion_candidate": (same & up).astype(bool),
        "same_org_confidence": status,
        "same_org_reason": reason.replace("", np.nan),
        "_rough_screen": rough.astype(bool),
        "_from_youth_side_v2": from_side.astype(bool),
    }, index=F.index)


def youth_side_ids(MAP: pd.DataFrame) -> set[int]:
    return set(MAP.loc[MAP.confidence.ne("rejected_senior_club"), "child_club_id"].astype(int))


CANDIDATE_COLUMNS = ["event_id", "player_id", "player_name", "transfer_date", "season", "kind", "transfer_type_raw",
                     "fee_display_raw", "from_club_id", "from_club_name", "to_club_id", "to_club_name", "receiving_league",
                     "same_org_confidence", "same_organisation_move_v2", "academy_promotion_candidate",
                     "from_club_parent_id_v2", "parent_names", "name_evidence", "players_between",
                     "next_compatible_players", "parent_share_of_top_destination", "same_org_reason",
                     "earlier_rough_screen", "is_youth_or_reserve_side", "in_cf_revised_sample",
                     "in_cf_main_model_sample", "in_cf_injury_model_sample", "pt_reliable", "minutes_share", "true_zero"]


def candidate_table(M: pd.DataFrame, MAP: pd.DataFrame) -> pd.DataFrame:
    """Every master episode that touches a youth/reserve side (v2 or Stage 1C pattern), with the evidence behind it."""
    K = MAP.set_index("child_club_id")
    rough = rough_screen(M)
    keep = M.same_org_confidence.ne("no_youth_or_reserve_side") | M.is_youth_or_reserve_side.astype(bool) | rough
    C = M[keep].copy()
    C["earlier_rough_screen"] = rough[keep]
    fid = C.from_club_id.astype("Int64")
    for c in ("parent_names", "name_evidence", "players_between", "next_compatible_players",
              "parent_share_of_top_destination"):
        C[c] = fid.map(K[c])
    C["true_zero"] = C.true_zero.where(C.pt_reliable)          # only interpretable on reliable episodes
    order = {"high": 0, "medium": 1, "low": 2, "rejected_name_overlap_only": 3, "rejected_senior_club": 4,
             "different_organisation_or_unresolved": 5, "no_youth_or_reserve_side": 6}
    C = C.assign(_o=C.same_org_confidence.map(order)).sort_values(["_o", "from_club_name", "transfer_date", "event_id"])
    return C[CANDIDATE_COLUMNS].reset_index(drop=True)


# Stratified review sample: within the calendar-first revised sample, per status and receiving league, the first rows
# by event_id (a hash, so effectively random and stable).
REVIEW_PER_CELL = {"high": 2, "medium": 1, "low": 5, "rejected_name_overlap_only": 5, "rejected_senior_club": 3,
                   "different_organisation_or_unresolved": 2}


def review_sample(C: pd.DataFrame) -> pd.DataFrame:
    R = C[C.in_cf_revised_sample & C.same_org_confidence.isin(REVIEW_PER_CELL)]
    parts = []
    for (st, lg), g in R.groupby(["same_org_confidence", "receiving_league"]):
        g = g.sort_values("event_id")
        if st == "different_organisation_or_unresolved":     # one '-' labelled and one other, the likeliest misses
            g = pd.concat([g[g.transfer_type_raw.eq("no_fee_shown")].head(1), g[g.transfer_type_raw.ne("no_fee_shown")].head(1)])
        parts.append(g.head(REVIEW_PER_CELL[st]))
    S = pd.concat(parts).sort_values(["same_org_confidence", "receiving_league", "event_id"]).reset_index(drop=True)
    v = S.event_id.map(MANUAL_REVIEW)
    S["manual_verdict"] = v.map(lambda x: x[0] if isinstance(x, tuple) else None)
    S["manual_note"] = v.map(lambda x: x[1] if isinstance(x, tuple) else None)
    return S


def review_summary(S: pd.DataFrame) -> dict:
    assert S.manual_verdict.notna().all(), f"{int(S.manual_verdict.isna().sum())} sampled rows have no verdict"
    flagged = S.same_org_confidence.isin(CONF_ORDER)
    out = {"rows": len(S), "leagues": int(S.receiving_league.nunique()),
           "by_status": {k: {kk: int(vv) for kk, vv in g.manual_verdict.value_counts().items()}
                         for k, g in S.groupby("same_org_confidence")},
           "false_positives": S.loc[flagged & S.manual_verdict.eq("not same organisation"),
                                    ["from_club_name", "to_club_name", "same_org_confidence", "manual_note"]]
           .values.tolist(),
           "false_negatives": S.loc[~flagged & S.manual_verdict.eq("same organisation"),
                                    ["from_club_name", "to_club_name", "same_org_confidence", "manual_note"]]
           .values.tolist()}
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _c(m: dict) -> str:
    return f"{m['coef']:+.4f}".replace("-0.0000", "0.0000").replace("+0.0000", "0.0000")


def _ci(m: dict) -> str:
    return f"[{m['lo']:+.4f}, {m['hi']:+.4f}]"


def render(A: dict, MN: dict, S: pd.DataFrame) -> str:
    V, R = A["variants"], A["review"]
    a = V["A"]
    W = ["# Academy and reserve-team promotions: organisation map and sensitivity", "",
         "Generated by `python -m src.analysis.club_organisation` (the map) and `python -m src.analysis.data_lineage` "
         "(flags, samples, regressions). Stage 1C is read only (SHA-256 "
         f"`{MN['stage1c_sha256'][:16]}…`) and its `is_internal_move` is not changed. **The main calendar-first sample is "
         "not replaced**: everything below is a sensitivity.", "",
         "## 1. Why", "",
         "Stage 1C treats a move as internal only when the two club names are identical once an academy/reserve suffix "
         "is stripped. Transfermarkt abbreviates names differently on its youth and senior pages (\"B. Dortmund II\" vs "
         "\"Dortmund\", \"Schalke 04 U19\" vs \"FC Schalke 04\", \"RM Castilla\" vs \"Real Madrid\"), so many "
         "promotions from a club's own academy enter the master as permanent arrivals. Transfermarkt publishes no "
         "parent-club field.", "",
         "## 2. How a youth/reserve side is linked to a senior club", "",
         f"Club ids: {f0(MN['club_ids'])}. Youth/reserve sides: {f0(MN['youth_or_reserve_sides'])} (any name the id "
         f"has carried matches the Stage 1C pattern, or one of the extra markers listed in `club_organisation.py`, e.g. "
         f"\"Spartak 2\", \"Jong KVM\", \"Rennes Jugend\"). Any club with a top-flight league game in the snapshot "
         f"({f0(MN['senior_league_clubs'])} clubs) is senior by definition; this guard removes "
         f"{', '.join(MN['senior_guard_names'])}.", "",
         "A side is linked to the senior club it is connected to by player moves (either direction, anywhere in Stage "
         "1C) only if:", "",
         "1. **names agree**: after removing the youth/reserve marker and legal-form words, every word of one name "
         "stands for a word of the other (exactly, as an abbreviation such as Olymp. → Olympique, as an initial such as "
         "B. → Borussia, or as an acronym such as RM → Real Madrid), with at least one distinctive (non-generic) word. "
         f"No similarity score is used. {len(MN['token_aliases'])} hand-entered word equivalences cover names "
         "Transfermarkt writes differently on the two pages: "
         + ", ".join(f"{k} → {v}" for k, v in MN["token_aliases"].items()) + ";",
         "2. **same country** (Transfermarkt country flag of both clubs);",
         "3. it is the **most connected** name-compatible club (at least twice the players of any other).", "",
         "Confidence of a link:", "",
         "- **high**: names agree, same country, at least 2 players connect the two clubs (the move itself plus one more), "
         "no other name-compatible club comes close, and it carries at least half the players of the side's main senior "
         "destination in that country;",
         "- **medium**: names agree and same country, but only this one move connects them, or the side's main domestic "
         "destination is another club, or only generic words are shared with a clearly dominant club;",
         "- **low**: a weaker name match with a competing compatible club;",
         "- **unresolved**: no connected senior club has a compatible name in the same country (left unlinked).", "",
         "Links by confidence (sides): " + "; ".join(f"{k} {f0(v)}" for k, v in MN["by_confidence"].items()) + ".", "",
         "A move is a **same-organisation move** when both clubs belong to one organisation once each linked side is "
         "replaced by its parent; its confidence is that of the weakest link used. `same_organisation_move_v2` = high "
         "only; `academy_promotion_candidate` = club A is a youth/reserve side and club B its parent, any confidence.", "",
         "## 3. What the flags find in the analysis samples", ""]
    st = A["status_by_sample"]
    order = ["high", "medium", "low", "rejected_name_overlap_only", "rejected_senior_club",
             "different_organisation_or_unresolved", "no_youth_or_reserve_side"]
    W += md_table(["same_org_confidence", "Revised: loans", "Revised: permanent", "Main: permanent",
                   "Injury: permanent"],
                  [[f"`{k}`", f0(st["revised"].get(k, {}).get("loan", 0)), f0(st["revised"].get(k, {}).get("permanent", 0)),
                    f0(st["main"].get(k, {}).get("permanent", 0)), f0(st["injury_model"].get(k, {}).get("permanent", 0))]
                   for k in order])
    rv = A["rough_vs_v2"]
    lab = A["candidate_labels"]
    W += ["", f"No loan in any analysis sample is flagged ({f0(A['loans_flagged'])} flagged loans in the whole master, "
          "all outside the samples). Of the earlier rough name screen's "
          f"{f0(rv['main']['rough_screen'])} main-regression hits, {f0(rv['main']['rough_and_v2'])} are confirmed and "
          f"{f0(rv['main']['rough_only'])} rejected; the map adds {f0(rv['main']['v2_only'])} the screen missed. Revised "
          f"sample: {f0(rv['revised']['rough_screen'])} screen hits, {f0(rv['revised']['rough_and_v2'])} confirmed, "
          f"{f0(rv['revised']['rough_only'])} rejected, {f0(rv['revised']['v2_only'])} added.", "",
          "Consistency check on the fee label: Transfermarkt usually shows \"-\" (no fee shown) for a promotion inside "
          f"a club. Of all {f0(sum(lab.values()))} flagged candidates in the master, {f0(lab.get('no_fee_shown', 0))} carry "
          "that label (" + ", ".join(f"{k} {f0(v)}" for k, v in lab.items() if k != "no_fee_shown") + "). The label is "
          "consistent with the promotion classification but is not by itself evidence of a promotion: "
          f"{f0(A['no_fee_shown_permanent_unflagged'])} other permanent arrivals in the master also carry \"-\" and are "
          "not flagged.", ""]
    G = A["permanent_groups"]
    W += ["Calendar-first revised sample, permanent arrivals:", ""]
    W += md_table(["Group", "Episodes", "Mean minutes share", "True zeros", "Label '-'", "Mean age",
                   "Mean log relative value (main)"],
                  [[k, f0(g["n"]), f"{g['mean_share']:.3f}", f"{100 * g['true_zero']:.1f}%",
                    f"{100 * g['no_fee_shown']:.1f}%", f"{g['mean_age']:.1f}", f"{g['mean_log_relative_value_main']:.2f}"]
                   for k, g in G.items()])
    W += ["", "## 4. Sensitivity (calendar-first sequence; only the flagged rows are removed before the playing-time steps)",
          ""]
    rows = []
    for k, v in V.items():
        t, e = v["totals"], v["estimates"]
        rows.append([k, v["label"], f"{f0(t['revised']['total'])} ({f0(t['revised']['loan'])} / "
                     f"{f0(t['revised']['permanent'])})", f0(t["main"]["total"]), f0(t["injury_model"]["total"]),
                     f"{100 * e['raw_gap_revised']:+.1f} pp", f"{100 * e['raw_gap_main']:+.1f} pp"])
    W += md_table(["", "Sample", "Revised N (loans / permanent)", "Main N", "Injury N", "Raw gap, revised",
                   "Raw gap, main"], rows, right={2, 3, 4, 5, 6})
    W += [""]
    rows = []
    for k, v in V.items():
        e = v["estimates"]
        m, f = e["m3"], e["injury_m3_fit"]
        rows.append([k, _c(m), f"{m['se']:.4f}", _ci(m), f"{m['p']:.3f}", _c(f), f"{f['se']:.4f}", _ci(f),
                     f"{f['p']:.3f}"])
    W += md_table(["", "M3 loan coef", "SE", "95% CI", "p", "Injury fit-adjusted coef", "SE", "95% CI", "p"], rows)
    W += ["", "Observations removed relative to A:", ""]
    W += md_table(["", "Revised: loans", "Revised: permanent", "Main: loans", "Main: permanent", "Injury: loans",
                   "Injury: permanent"],
                  [[k, *(f0(v["removed"][s][kk]) for s in ("revised", "main", "injury_model") for kk in ("loan", "permanent"))]
                   for k, v in V.items() if k != "A"])
    b = V["B"]["estimates"]
    W += ["", "Loan coefficient on the main regression sample as controls are added (all with position and "
          "league-season fixed effects; M1 adds the share of the season remaining at arrival, M2 age and age², M3 log "
          "relative value and log squad value; standard errors clustered by receiving club):", ""]
    W += md_table(["", "M1", "M2", "M3 (preferred)"],
                  [[k, *(f"{_c(v['ladder'][s])} ({v['ladder'][s]['se']:.4f})" for s in ("M1", "M2", "M3"))]
                   for k, v in V.items()])
    G1, G0 = G["high-confidence same organisation"], G["other permanent arrivals"]
    W += ["", f"Removing the {f0(V['B']['removed']['main']['total'])} high-confidence promotions from the main "
          f"regression moves the raw loan-minus-permanent gap from {100 * a['estimates']['raw_gap_main']:+.1f} to "
          f"{100 * b['raw_gap_main']:+.1f} pp and the M1 coefficient from {_c(a['ladder']['M1'])} to "
          f"{_c(V['B']['ladder']['M1'])}, but the preferred M3 coefficient only from {_c(a['estimates']['m3'])} to "
          f"{_c(b['m3'])}. The promoted players' mean log relative value is {G1['mean_log_relative_value_main']:.2f} "
          f"against {G0['mean_log_relative_value_main']:.2f} for other permanent arrivals, so the value controls in M3 "
          "already absorb their shortfall. D removes every arrival from any youth/reserve side, including genuine sales "
          "by other clubs' academies, and is an extreme bound only.", "",
          "## 5. Manual inspection", "",
          f"Stratified sample: {R['rows']} rows of the calendar-first revised sample, by status and receiving league "
          f"({R['leagues']} leagues), the first rows by `event_id` in each cell. Verdicts are written in "
          "`club_organisation.MANUAL_REVIEW`; every sampled row must have one or the pipeline stops. Rows and verdicts: "
          "`academy_promotion_manual_review.csv`.", ""]
    W += md_table(["Status", "Same organisation", "Not same organisation", "Unclear"],
                  [[f"`{k}`", f0(v.get("same organisation", 0)), f0(v.get("not same organisation", 0)),
                    f0(v.get("unclear", 0))] for k, v in sorted(R["by_status"].items(), key=lambda x: order.index(x[0]))])
    fp, fn = R["false_positives"], R["false_negatives"]
    n_flag = sum(sum(R["by_status"].get(k, {}).values()) for k in CONF_ORDER)
    W += ["", (f"No false positives were found among the {n_flag} flagged observations in the manual review sample."
               if not fp else f"False positives among the {n_flag} flagged observations in the manual review sample: "
               f"{len(fp)}: " + "; ".join(f"{a} → {b} ({c}): {d}" for a, b, c, d in fp) + "."),
          f"False negatives among unflagged rows: {len(fn)}: " + "; ".join(f"{a} → {b}: {d}" for a, b, c, d in fn) + ".",
          "", "Examples of high-confidence inclusions:", ""]
    hi = S[S.same_org_confidence.eq("high")].head(8)
    W += [f"- {r.player_name} ({str(r.transfer_date)[:10]}): {r.from_club_name} → {r.to_club_name}; {r.manual_note}"
          for r in hi.itertuples()]
    W += ["", "Rejected look-alikes:", ""]
    rj = S[S.same_org_confidence.isin(["rejected_name_overlap_only", "rejected_senior_club"])].drop_duplicates(
        ["from_club_name", "to_club_name"]).drop_duplicates("from_club_name").head(5)
    W += [f"- {r.from_club_name} → {r.to_club_name} (`{r.same_org_confidence}`): {r.manual_note}" for r in rj.itertuples()]
    W += ["", "## 6. Known limits", "",
          "- Sides whose names share nothing usable with the parent stay unresolved (SV Zulte Waregem's \"SVZW U21\" and "
          "\"Jong Essevee\", Lokomotiv Moscow's farm side \"Loko-Kazanka\"). The Twente/Heracles joint academy is "
          "mapped to FC Twente, so its moves to Heracles are not flagged.",
          "- Clubs under common ownership (Leicester and OH Leuven; City Football Group) are separate organisations here.",
          "- A senior club whose earlier name carried a reserve marker can be treated as a side (Volgar, once "
          "\"Volgar-Gazprom-2\"); this matters only if a parent is then found, and Volgar has none.",
          "- The repeat-spell rule (`prior_spell_at_b`) is not recomputed when a promotion is removed; a later spell at "
          "the same club stays marked as a repeat.", "",
          "## 7. Files", "",
          "- `club_organisation_map.csv`: one row per youth/reserve side (and the senior-club guard), with the proposed "
          "parent, the names used, players connecting them, competing clubs, confidence and reason.",
          f"- `youth_reserve_review_universe.csv`: the broader review universe, {f0(A['candidates']['rows'])} master "
          "episodes that touch a youth/reserve side (or carry Stage 1C's youth flag), with status, evidence and sample "
          f"flags. Only {f0(A['candidates']['academy_promotion_candidate'])} of them have "
          "`academy_promotion_candidate = True`.",
          "- `academy_promotion_manual_review.csv`: the stratified review sample with verdicts.",
          "- Master columns: `from_club_parent_id_v2`, `same_organisation_move_v2`, `academy_promotion_candidate`, "
          "`same_org_confidence`, `same_org_reason`."]
    return "\n".join(W) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute(con=None) -> tuple[pd.DataFrame, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    S = pd.read_csv(CANONICAL_CSV, low_memory=False,
                    usecols=["player_id", "from_club_id", "to_club_id", "from_club_name", "to_club_name"])
    flags = pd.read_csv(RAW_FLAGS_CSV)
    own = con is None
    con = con or duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        clubs = con.execute("select club_id, name from clubs").fetchdf()
        senior = senior_league_clubs(con)
    finally:
        if own:
            con.close()
    names = club_names(S, flags, clubs)
    flag = flags.set_index("club_id").country_flag_id
    MAP, N = build_map(S, names, flag, senior)
    N["stage1c_sha256"] = STAGE1C_SHA256
    N["senior_league_clubs"] = len(senior)
    N["token_aliases"] = TOKEN_ALIASES
    return MAP, N


def main() -> dict:
    MAP, N = compute()
    MAP.to_csv(MAP_CSV, index=False)
    NUMBERS_JSON.write_text(json.dumps(N, indent=2, default=str))
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print(f"wrote {MAP_CSV.name}: {len(MAP)} youth/reserve sides; {N['by_confidence']}")
    return N


if __name__ == "__main__":
    main()
