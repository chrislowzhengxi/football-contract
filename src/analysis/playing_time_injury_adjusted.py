"""Injury-adjusted playing time from the external Kaggle injury histories (salimt, CC0, version 2, 2025-09-15).

    python -m src.analysis.playing_time_injury_adjusted

Raw files (manually downloaded by the user from Kaggle, verified here against Kaggle's file listing and fixed
SHA-256 hashes) are never modified. The revised playing-time sample, its windows and the current outcome come from
the earlier modules unchanged. The fit-adjusted outcome keeps the numerator and removes from the denominator only
receiving-club league games covered by a confirmed medical injury or illness spell in which the player did not
appear. Nothing is inferred: no return date for open spells, no health from empty histories, no injury from squad
absence. Observational; not causal. New files only.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from . import loan_scope
from . import playing_time_age as pta
from . import playing_time_regression as reg
from .daniel_scope_final import f0, md_table, pc
from .loan_episodes import ROOT
from .playing_time_descriptives import COLOR, GRID, INK, LABEL, MUTED, _style, describe, diff
from .playing_time_injury_feasibility import episode_games, revised_sample

OUT = ROOT / "data" / "outputs" / "rebuild"
RAW = ROOT / "data" / "raw" / "external_injuries" / "salimt_kaggle"
RAW_META = ROOT / "data" / "raw" / "external_injuries" / "salimt_kaggle_metadata"
PROCESSED = ROOT / "data" / "processed" / "external_injuries" / "salimt_kaggle_injuries_normalized.csv"
INJ_FILE, PROF_FILE = RAW / "player_injuries" / "player_injuries.csv", RAW / "player_profiles" / "player_profiles.csv"
MANIFEST = OUT / "injury_external_download_manifest.json"
JOIN_CSV = OUT / "injury_external_join_audit.csv"
VALID_CSV = OUT / "injury_external_validation.csv"
COVER_CSV = OUT / "injury_external_coverage.csv"
EPISODES_CSV = OUT / "playing_time_injury_adjusted_episodes.csv"
MODELS_CSV = OUT / "playing_time_injury_adjusted_models.csv"
SUMMARY_MD = OUT / "playing_time_injury_adjusted_summary.md"
NUMBERS_JSON = OUT / "playing_time_injury_adjusted_numbers.json"
CLASS_CSV = OUT / "injury_reason_classification.csv"
FIGS = {"dist": "figures/playing_time_injury_adjustment_distribution.png",
        "coef": "figures/playing_time_injury_adjustment_coefficients.png"}
WRITES = (MANIFEST, JOIN_CSV, VALID_CSV, COVER_CSV, EPISODES_CSV, MODELS_CSV, SUMMARY_MD, NUMBERS_JSON, CLASS_CSV) \
    + tuple(OUT / f for f in FIGS.values())

SOURCE = {"dataset": "xfkzujqjvx97n/football-datasets", "owner": "salimt", "version": 2,
          "version_notes": "Update 2025-09-15", "version_date": "2025-09-15", "license": "CC0: Public Domain",
          "url": "https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets/versions/2",
          "how_obtained": "manually downloaded from Kaggle by the user (Kaggle API downloads need an account token, "
                          "which this machine does not have); placed in data/raw/external_injuries/salimt_kaggle/",
          "original_source_stated": "Transfermarkt"}
EXPECTED = {"player_injuries/player_injuries.csv": ("fab7edd70cd4be43c682172a3d7ac8c63dc8e119cd2c90ec2acd8a8f801bf192", 8051385),
            "player_profiles/player_profiles.csv": ("7ef9afbacf97577d099f21a0ef0ea0d46381bd4806a119f095b9562762a20096", 26380409)}
SNAPSHOT = pd.Timestamp("2025-09-15")      # Kaggle version 2 date: later injuries are not in the file
TRUNCATION_CAP = 15                         # no player has more rows; the source shows only the latest 15
CONSEQ = ("missed the next two league games", "missed only the next league game", "no later league game in the window",
          "played the next league game")
EARLY_LAST, EARLY_LABEL, LATE_LABEL = 2016, "2012/13–2016/17", "2017/18–2024/25"
RAW_COLS = ["player_id", "season_name", "injury_reason", "from_date", "end_date", "days_missed", "games_missed"]
KINDS = ("loan", "permanent")

# ---------------------------------------------------------------------------
# Classification of injury_reason (every label listed in injury_reason_classification.csv)
# ---------------------------------------------------------------------------

NON_MEDICAL = {"rest": "non-medical: rest"}
FLAG_ONLY = {"fitness": "fitness / rehabilitation (not counted)",
             "quarantine": "unclear: quarantine / isolation (not counted)"}
UNSPECIFIED = {"unknown injury"}                                         # an injury, type unknown
MINOR_UNCLEAR = {"muscle fatigue", "muscle stiffness", "calf stiffness", "sore muscles"}
ILLNESS_TERMS = ("ill", "flu", "influenza", "cold", "fever", "virus", "infection", "tonsillitis", "stomach",
                 "mononucleation", "pneumonia", "angina", "food poisoning", "bronchitis", "cancer", "malaria",
                 "meningitis", "shingles", "chickenpox", "skin disease", "tuberculosis", "kidney problems",
                 "heart problems", "circulation problems", "appendicitis", "allergic reaction", "depression",
                 "nerval disease", "pancreas", "blood poisoning", "blood clots", "vein occlusion", "tumor", "stroke",
                 "cerebral hemorrhage", "coma", "varicose vein", "liver transplant", "splenic infarction",
                 "cyst in the lungs", "toothache", "abscess", "balance disorder", "corona", "dental", "kidney stone")
INJURY_TERMS = ("injur", "problem", "tear", "torn", "fractur", "broken", "sprain", "strain", "bruise", "contusion",
                "surgery", "knock", "rupture", "ripture", "ligament", "tendon", "muscle", "muscular", "pain", "damage",
                "dislocation", "inflammation", "irritation", "edema", "crack", "wound", "cut", "laceration",
                "concussion", "arthroscopy", "appendectomy", "hernia", "lumbago", "pubalgia", "contracture",
                "overstretch", "stretch", "dead leg", "cartilage", "meniscus", "tendonitis", "tendinopathy", "bursitis",
                "sciatica", "whiplash", "spine", "vertebra", "syndrome", "impingement", "splinter", "spur", "cyst",
                "blockage", "fissure", "obliquity", "swelling", "bleeding", "trauma", "eardrum", "accident", "insect bite",
                "corn", "combustion", "screws", "pinched nerve", "contortion", "pneumothorax", "collapsed lung",
                "compression", "gehörgangsfraktur", "bone", "capsul", "hearing", "pelvic", "lung")


def classify_reason(label: str) -> dict:
    """category, counts in PRIMARY, counts in STRICT. The order of the checks is the rule."""
    s = label.strip().lower()
    if s in NON_MEDICAL:
        return {"category": NON_MEDICAL[s], "primary": False, "strict": False}
    if s in FLAG_ONLY:
        return {"category": FLAG_ONLY[s], "primary": False, "strict": False}
    if s in UNSPECIFIED:
        return {"category": "injury, type unspecified (primary only)", "primary": True, "strict": False}
    if s in MINOR_UNCLEAR:
        return {"category": "minor / unclear muscular complaint (primary only)", "primary": True, "strict": False}
    words = set(re.findall(r"[a-zäöü]+", s))
    if any((t in words) if " " not in t and len(t) <= 4 else (t in s) for t in ILLNESS_TERMS):
        return {"category": "illness", "primary": True, "strict": True}
    if any(t in s for t in INJURY_TERMS):
        return {"category": "physical injury", "primary": True, "strict": True}
    return {"category": "UNCLASSIFIED", "primary": False, "strict": False}


# ---------------------------------------------------------------------------
# Raw files and normalisation
# ---------------------------------------------------------------------------

def verify_raw() -> dict:
    out = {}
    listing = json.loads((RAW_META / "kaggle_file_listing.json").read_text())
    listed = {f["name"]: f for f in listing["datasetFiles"]}
    for rel, (sha, size) in EXPECTED.items():
        b = (RAW / rel).read_bytes()
        got = hashlib.sha256(b).hexdigest()
        out[rel] = {"bytes": len(b), "sha256": got, "kaggle_listed_bytes": listed[rel]["totalBytes"],
                    "kaggle_file_created": listed[rel]["creationDate"],
                    "git_blob_sha1": hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()}
        assert got == sha and len(b) == size == listed[rel]["totalBytes"], f"{rel} does not match its verified copy"
    return out


def normalize(R: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Long-form injury table. Raw strings kept; only exact duplicate rows dropped; problems flagged, never fixed."""
    R = R.copy()
    R["source_row"] = np.arange(len(R))
    rows_per_player = R.groupby("player_id").size()
    exact = R.duplicated(subset=RAW_COLS, keep="first")
    D = R[~exact].copy()
    out = pd.DataFrame({"source_row": D.source_row, "player_id": D.player_id.astype("int64"),
                        "season": D.season_name, "injury_reason_raw": D.injury_reason,
                        "injury_reason_clean": D.injury_reason.str.strip().str.lower(),
                        "from_date_raw": D.from_date, "end_date_raw": D.end_date,
                        "days_missed_raw": D.days_missed, "games_missed_raw": D.games_missed})
    out["from_date"] = pd.to_datetime(out.from_date_raw.replace("", None), format="%Y-%m-%d", errors="coerce")
    out["end_date"] = pd.to_datetime(out.end_date_raw.replace("", None), format="%Y-%m-%d", errors="coerce")
    out["date_parse_error"] = (out.from_date.isna() & out.from_date_raw.ne("")) | (out.end_date.isna() & out.end_date_raw.ne(""))
    out["days_missed"] = pd.to_numeric(out.days_missed_raw.replace("", None), errors="coerce").astype("Int64")
    out["games_missed"] = pd.to_numeric(out.games_missed_raw.replace("", None), errors="coerce").astype("Int64")
    out["missing_start"] = out.from_date.isna()
    out["open_injury"] = out.end_date.isna() & ~out.missing_start
    out["end_before_start"] = out.end_date < out.from_date
    out["starts_after_snapshot"] = out.from_date > SNAPSHOT
    grp = out.groupby(["player_id", "from_date"], dropna=False)
    out["conflicting_duplicate"] = grp.source_row.transform("size").gt(1) & out.from_date.notna()
    out["player_rows_in_source"] = out.player_id.map(rows_per_player.rename(index=lambda x: int(x)))
    out["player_history_truncated"] = out.player_rows_in_source >= TRUNCATION_CAP
    out["player_history_floor"] = out.player_id.map(out.groupby("player_id").from_date.min())
    cl = out.injury_reason_raw.map(classify_reason)
    out["category"] = cl.map(lambda d: d["category"])
    out["counts_primary"] = cl.map(lambda d: d["primary"])
    out["counts_strict"] = cl.map(lambda d: d["strict"])
    info = {"raw_rows": len(R), "exact_duplicates_dropped": int(exact.sum()), "rows": len(out),
            "players": int(out.player_id.nunique()), "missing_start": int(out.missing_start.sum()),
            "open": int(out.open_injury.sum()), "end_before_start": int(out.end_before_start.sum()),
            "date_parse_errors": int(out.date_parse_error.sum()), "conflicting_duplicate_rows": int(out.conflicting_duplicate.sum()),
            "starts_after_snapshot": int(out.starts_after_snapshot.sum()),
            "players_truncated": int(out.loc[out.player_history_truncated, "player_id"].nunique()),
            "max_rows_per_player": int(rows_per_player.max())}
    return out.reset_index(drop=True), info


def spells(N: pd.DataFrame, rule: str = "primary") -> pd.DataFrame:
    """Placeable medical spells for a rule ('primary' or 'strict'): start and end both known."""
    col = {"primary": "counts_primary", "strict": "counts_strict"}[rule]
    S = N[N[col] & N.from_date.notna() & N.end_date.notna() & ~N.end_before_start]
    return S[["player_id", "source_row", "from_date", "end_date", "injury_reason_raw", "category"]].rename(
        columns={"source_row": "spell_id"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Game-level unavailability and episode outcomes
# ---------------------------------------------------------------------------

def game_flags(W: pd.DataFrame, Sp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For every (episode, window game): does a spell cover the date, did he appear, is he medically unavailable?
    Inclusive dates (a spell starting or ending on match day covers it); an appearance always overrides; several
    covering spells count the game once (union)."""
    G = W[["event_id", "player_id", "game_id", "date", "played"]].copy()
    M = G.merge(Sp[["player_id", "spell_id", "from_date", "end_date"]], on="player_id")
    M = M[(M.from_date <= M.date) & (M.date <= M.end_date)]
    cov = M.groupby(["event_id", "game_id"]).spell_id.nunique().rename("covering_spells")
    G = G.join(cov, on=["event_id", "game_id"])
    G["covering_spells"] = G.covering_spells.fillna(0).astype(int)
    G["spell_covers"] = G.covering_spells > 0
    G["appearance_conflict"] = G.spell_covers & G.played
    G["medical_unavailable"] = G.spell_covers & ~G.played
    return G, M[~M.played]


def union_days(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> int:
    days, cur = 0, None
    for a, b in sorted(intervals):
        if cur is None or a > cur[1] + pd.Timedelta(days=1):
            if cur is not None:
                days += (cur[1] - cur[0]).days + 1
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur is not None:
        days += (cur[1] - cur[0]).days + 1
    return days


def episode_outcomes(S: pd.DataFrame, G: pd.DataFrame, Sp: pd.DataFrame) -> pd.DataFrame:
    a = G.groupby("event_id").agg(available_games_current=("game_id", "size"),
                                  confirmed_injury_games=("medical_unavailable", "sum"),
                                  appearance_conflicts=("appearance_conflict", "sum"))
    E = S[["event_id", "player_id", "kind", "minutes", "minutes_share", "window_start", "window_end"]].set_index("event_id").join(a)
    E["available_games_fit"] = E.available_games_current - E.confirmed_injury_games
    E["current_share"] = E.minutes / (90 * E.available_games_current)
    assert np.allclose(E.current_share, E.minutes_share), "current share does not reproduce"
    E["fit_adjusted_share"] = (E.minutes / (90 * E.available_games_fit.where(E.available_games_fit > 0))).astype(float)
    sp = Sp.merge(E[["player_id", "window_start", "window_end"]].reset_index(), on="player_id")
    sp = sp[(sp.from_date <= sp.window_end) & (sp.end_date >= sp.window_start)]
    sp["a"], sp["b"] = sp[["from_date", "window_start"]].max(axis=1), sp[["end_date", "window_end"]].min(axis=1)
    days = sp.groupby("event_id").apply(lambda g: union_days(list(zip(g.a, g.b))), include_groups=False)
    E["injury_days_overlap"] = days.reindex(E.index).fillna(0).astype(int)
    E["injury_spell_count"] = sp.groupby("event_id").spell_id.nunique().reindex(E.index).fillna(0).astype(int)
    return E.drop(columns=["minutes_share"]).reset_index()


def usability(E: pd.DataFrame, N: pd.DataFrame, status: pd.Series) -> pd.Series:
    """First failing reason, or 'usable'. Order: datalake, snapshot, truncation, open spell, missing start, games."""
    open_sp = N[N.counts_primary & N.open_injury].groupby("player_id").from_date.min()
    miss = set(N.loc[N.counts_primary & N.missing_start, "player_id"])
    floor = N[N.player_history_truncated].groupby("player_id").player_history_floor.first()
    st = E.player_id.map(status)
    trunc = E.player_id.map(floor)
    op = E.player_id.map(open_sp)
    return pd.Series(np.select(
        [st.eq("C"), st.eq("D"), E.window_end >= SNAPSHOT, trunc.notna() & (E.window_start < trunc),
         op.notna() & (op <= E.window_end), E.player_id.isin(miss), E.available_games_fit < 1],
        ["player not in the external datalake", "ambiguous player id", "window reaches the 2025-09-15 snapshot",
         "injury history truncated before the window", "open injury (no end date) could cover the window",
         "player has a medical row without a start date", "no fit games left"], default="usable"), index=E.index)


def sensitivity_flags(G: pd.DataFrame, M_unplayed: pd.DataFrame, k: int) -> pd.Series:
    """Games removed when only spells that remove at least k window games (in that episode) are kept."""
    per = M_unplayed.groupby(["event_id", "spell_id"]).game_id.nunique()
    keep = per[per >= k].reset_index()[["event_id", "spell_id"]]
    hit = M_unplayed.merge(keep, on=["event_id", "spell_id"])[["event_id", "game_id"]].drop_duplicates()
    return G.set_index(["event_id", "game_id"]).index.isin(pd.MultiIndex.from_frame(hit))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

M3 = ["loan", "remain", "age_c", "age_c2", "log_relative_value", "log_squad_value"]
AGE = ["loan", "older", "loan_x_older", "remain", "log_relative_value", "log_squad_value"]


def coef_row(m: dict, name: str, model: str, outcome: str, sample: str) -> dict:
    r = m["res"]
    b, se = float(r.params[name]), float(r.bse[name])
    return {"model": model, "outcome": outcome, "sample": sample, "term": name, "coef": b, "se": se,
            "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": float(r.pvalues[name]), "N": m["N"], "clusters": m["clusters"]}


def pair_models(C: pd.DataFrame, x: list[str], terms: list[str], label: str, fit_col: str = "fit_adjusted_share") -> list[dict]:
    rows = []
    for outcome, y in (("current", "current_share"), ("fit-adjusted", fit_col)):
        m = reg.fit(C, x, fe="league_season", cats=["position_clean"], y=y)
        for t in terms:
            rows.append(coef_row(m, t, label, outcome, f"N={m['N']}"))
    return rows


def age_contrasts(C: pd.DataFrame, y: str) -> dict:
    m = reg.fit(C, AGE, fe="league_season", cats=["position_clean"], y=y)
    r = m["res"]
    return {"young": pta.lincom(r, ["loan"]), "older": pta.lincom(r, ["loan", "loan_x_older"]),
            "interaction": pta.lincom(r, ["loan_x_older"]), "N": m["N"]}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(u=None) -> tuple[dict, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    Nn: dict = {"source": SOURCE, "files": verify_raw(), "snapshot": str(SNAPSHOT.date())}
    raw = pd.read_csv(INJ_FILE, dtype=str, keep_default_na=False)
    INJ, Nn["normalize"] = normalize(raw)
    prof = pd.read_csv(PROF_FILE, dtype=str, keep_default_na=False, usecols=["player_id", "player_name", "date_of_birth"])
    prof_ids = prof.player_id.astype("int64")
    Nn["profiles"] = {"rows": len(prof), "unique_ids": int(prof_ids.nunique()), "duplicate_ids": int(prof_ids.duplicated().sum())}
    assert set(INJ.player_id) <= set(prof_ids), "injury players missing from the profiles table"
    labels = INJ.groupby("injury_reason_raw").agg(rows=("source_row", "size"), category=("category", "first"),
                                                  counts_primary=("counts_primary", "first"),
                                                  counts_strict=("counts_strict", "first")).reset_index() \
        .sort_values("rows", ascending=False)
    assert not labels.category.eq("UNCLASSIFIED").any(), "an injury label has no classification"

    u = u or loan_scope.universe()
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        S, covered = revised_sample(u, con)
        W = episode_games(S, con)
        pv = reg.valuations(con)
        names = con.sql("select player_id, name, date_of_birth from players").df()
    finally:
        con.close()
    Nn["revised"] = {k: int((S.kind == k).sum()) for k in KINDS}
    Nn["revised_players"] = int(S.player_id.nunique())

    # ---- Part 3: exact player-id join
    dup = set(prof_ids[prof_ids.duplicated()])
    with_rows = set(INJ.player_id)
    in_lake = set(prof_ids)
    status = pd.Series({p: ("D" if p in dup else "A" if p in with_rows else "B" if p in in_lake else "C")
                        for p in S.player_id.astype("int64").unique()})
    S = S.assign(player_id=S.player_id.astype("int64"))
    S["status"] = S.player_id.map(status)
    Nn["join_players"] = {k: int(v) for k, v in status.value_counts().items()}
    ourd = names.assign(player_id=names.player_id.astype("int64")).set_index("player_id")
    pr = prof.assign(player_id=prof_ids).set_index("player_id")
    both = status[status.isin(["A", "B"])].index
    dob_ours = pd.to_datetime(ourd.date_of_birth.reindex(both), errors="coerce")
    dob_ext = pd.to_datetime(pr.date_of_birth.reindex(both), errors="coerce")
    comp = dob_ours.notna() & dob_ext.notna()
    Nn["id_check"] = {"compared": int(comp.sum()), "dob_equal": int((dob_ours[comp] == dob_ext[comp]).sum()),
                      "examples_mismatch": [f"{p}: ours {ourd.name.get(p)} {dob_ours[p].date()}, external "
                                            f"{pr.player_name.get(p)} {dob_ext[p].date()}"
                                            for p in comp[comp].index if dob_ours[p] != dob_ext[p]][:8]}

    # ---- covariates, groups
    B = reg.build(S, pv)
    B = B.assign(player_id=B.player_id.astype("int64"))
    cov = B.set_index("event_id")[["age", "remain", "log_relative_value", "log_squad_value", "position_clean",
                                   "league_season", "relative_value", "player_value", "b_club_id"]]
    S = S.join(cov, on="event_id", rsuffix="_cov")
    S["age_group"] = np.where(np.floor(S.age) <= pta.CUTOFF, f"young (≤{pta.CUTOFF})", f"older (≥{pta.CUTOFF + 1})")
    S["rv_quintile"] = pd.qcut(S.log_relative_value, 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str).replace("nan", "no value")

    # ---- Part 7-8: game flags, outcomes, usability
    W = W.assign(player_id=W.player_id.astype("int64"))
    SpP = spells(INJ, "primary")
    G, Mu = game_flags(W, SpP)
    E = episode_outcomes(S, G, SpP)
    E = E.merge(S[["event_id", "status", "receiving_league", "arrival_season", "age", "age_group", "rv_quintile", "remain",
                   "log_relative_value", "log_squad_value", "position_clean", "league_season", "relative_value",
                   "player_value"]], on="event_id")
    E["b_club_id"] = E.event_id.map(S.set_index("event_id").b_club_id)
    E["data_quality_flag"] = usability(E, INJ, status).values
    E["usable"] = E.data_quality_flag.eq("usable")
    assert (E.available_games_fit <= E.available_games_current).all()
    ok = E.fit_adjusted_share.notna()
    assert (E.loc[ok, "fit_adjusted_share"] >= E.loc[ok, "current_share"] - 1e-12).all()
    assert E.loc[E.confirmed_injury_games.eq(0) & ok].pipe(lambda d: np.allclose(d.fit_adjusted_share, d.current_share))
    # strict and severity variants on the same games
    SpS = spells(INJ, "strict")
    Gs, _ = game_flags(W, SpS)
    G["unavail_strict"] = Gs.medical_unavailable.values
    for k in (3, 5):
        G[f"unavail_sev{k}"] = sensitivity_flags(G, Mu, k)
    for name, col in (("strict", "unavail_strict"), ("severe3", "unavail_sev3"), ("long5", "unavail_sev5")):
        games = G.groupby("event_id")[col].sum()
        E[f"injury_games_{name}"] = E.event_id.map(games).fillna(0).astype(int)
        fg = E.available_games_current - E[f"injury_games_{name}"]
        E[f"fit_share_{name}"] = (E.minutes / (90 * fg.where(fg > 0))).astype(float)
    Nn["appearance_conflicts"] = {"games": int(G.appearance_conflict.sum()),
                                  "episodes": int((E.appearance_conflicts > 0).sum())}

    # ---- usability tables
    Nn["usability"] = {k: {r: int(v) for r, v in E[E.kind == k].data_quality_flag.value_counts().items()} for k in KINDS}
    U = E[E.usable].copy()
    Nn["usable"] = {k: int((U.kind == k).sum()) for k in KINDS}

    # ---- Part 3 join audit table
    rows = []
    for dim, col in (("all", None), ("league", "receiving_league"), ("season", "arrival_season"), ("kind", "kind"),
                     ("age group", "age_group"), ("relative-value quintile", "rv_quintile")):
        for key, g in (E.groupby(["kind", col]) if col and col != "kind" else E.groupby("kind")):
            key = key if isinstance(key, tuple) else (key,)
            rec = {"dimension": dim, "kind": key[0], "value": key[1] if len(key) > 1 else "all", "episodes": len(g),
                   "players": int(g.player_id.nunique())}
            for s_ in "ABCD":
                rec[f"episodes_{s_}"] = int(g.status.eq(s_).sum())
                rec[f"players_{s_}"] = int(g.loc[g.status.eq(s_), "player_id"].nunique())
            rec["exact_id_match_rate_players"] = (rec["players_A"] + rec["players_B"]) / rec["players"]
            rec["usable_episodes"] = int(g.usable.sum())
            rows.append(rec)
    JOIN = pd.DataFrame(rows)
    Nn["join_by_kind"] = {k: {"players": int(E.loc[E.kind == k, "player_id"].nunique()),
                              "players_in_lake": int(E.loc[(E.kind == k) & E.status.isin(["A", "B"]), "player_id"].nunique()),
                              "players_with_rows": int(E.loc[(E.kind == k) & E.status.eq("A"), "player_id"].nunique())}
                          for k in KINDS}

    # ---- Part 4 coverage by league-season
    inj_rows = INJ.groupby("player_id").size()
    rows = []
    for dim, keys in (("league_season", ["receiving_league", "arrival_season"]), ("league", ["receiving_league"]),
                      ("season", ["arrival_season"]), ("kind", ["kind"]), ("age group", ["age_group"])):
        for key, g in E.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            pl = g.player_id.unique()
            inl = [p for p in pl if status.get(p) in ("A", "B")]
            rows.append({"dimension": dim, "key": " / ".join(map(str, key)), "episodes": len(g), "players": len(pl),
                         "players_in_lake": len(inl), "players_with_injury_rows": int(sum(status.get(p) == "A" for p in pl)),
                         "players_with_injury_in_window": int(g.loc[g.injury_spell_count > 0, "player_id"].nunique()),
                         "injury_rows": int(inj_rows.reindex(pl).fillna(0).sum()),
                         "median_injuries_per_player_in_lake": float(inj_rows.reindex(inl).fillna(0).median()) if inl else np.nan,
                         "injury_days_in_windows": int(g.injury_days_overlap.sum()),
                         "usable_episodes": int(g.usable.sum()), "usable_share": float(g.usable.mean()),
                         "truncated_excluded": int(g.data_quality_flag.eq("injury history truncated before the window").sum()),
                         "snapshot_excluded": int(g.data_quality_flag.eq("window reaches the 2025-09-15 snapshot").sum())})
    COV = pd.DataFrame(rows)
    sea = COV[COV.dimension == "season"].set_index("key")
    Nn["coverage_by_season"] = {k: {"episodes": int(r.episodes), "usable_share": float(r.usable_share),
                                    "players_with_injury_rows_share": float(r.players_with_injury_rows / r.players),
                                    "players_with_injury_in_window_share": float(r.players_with_injury_in_window / r.players),
                                    "truncated_excluded": int(r.truncated_excluded)} for k, r in sea.iterrows()}
    lea = COV[COV.dimension == "league"].set_index("key")
    Nn["coverage_by_league"] = {k: {"episodes": int(r.episodes), "usable_share": float(r.usable_share),
                                    "players_with_injury_rows_share": float(r.players_with_injury_rows / r.players),
                                    "players_with_injury_in_window_share": float(r.players_with_injury_in_window / r.players)}
                                for k, r in lea.iterrows()}

    # ---- Part 6 validation against in-match injury substitutions
    ev = W[(W.injury_off > 0) & W.player_id.map(status).isin(["A", "B"]) & (W.date < SNAPSHOT)]
    ev = ev[["event_id", "player_id", "game_id", "date"]].drop_duplicates(["player_id", "game_id"]).merge(
        S[["event_id", "receiving_league", "arrival_season", "kind"]], on="event_id")
    allrows = INJ[INJ.from_date.notna()][["player_id", "source_row", "from_date", "end_date", "injury_reason_raw", "counts_primary"]]
    V = ev.merge(allrows, on="player_id", how="left")
    V["lag"] = (V.from_date - V.date).dt.days
    V["abs_lag"] = V.lag.abs()
    V["covers"] = (V.from_date <= V.date + pd.Timedelta(days=7)) & (V.end_date.fillna(V.from_date) >= V.date)
    best = V.sort_values(["abs_lag", "source_row"]).groupby(["player_id", "game_id"]).head(1).reset_index(drop=True)
    best["n_rows_within_7"] = V[V.abs_lag <= 7].groupby(["player_id", "game_id"]).size().reindex(
        pd.MultiIndex.from_frame(best[["player_id", "game_id"]])).fillna(0).astype(int).values
    best["covered_by_interval"] = V.groupby(["player_id", "game_id"]).covers.any().reindex(
        pd.MultiIndex.from_frame(best[["player_id", "game_id"]])).fillna(False).values
    best["player_status"] = best.player_id.map(status)
    for k in (0, 1, 3, 7):
        best[f"match_{k}d"] = best.abs_lag <= k
    VAL = best.rename(columns={"date": "event_date", "from_date": "nearest_injury_start",
                               "end_date": "nearest_injury_end", "injury_reason_raw": "nearest_injury_reason"})
    VAL["ambiguous"] = VAL.n_rows_within_7 > 1
    Gs_ = G.sort_values(["event_id", "date"]).copy()
    Gs_["next_played"] = Gs_.groupby("event_id").played.shift(-1)
    Gs_["next2_played"] = Gs_.groupby("event_id").played.shift(-2)
    nxt = Gs_.drop_duplicates(["player_id", "game_id"]).set_index(["player_id", "game_id"])[["next_played", "next2_played"]]
    VAL = VAL.join(nxt, on=["player_id", "game_id"])
    VAL["after_event"] = np.select(
        [VAL.next_played.isna(), VAL.next_played.eq(False) & VAL.next2_played.eq(False), VAL.next_played.eq(False)],
        [CONSEQ[2], CONSEQ[0], CONSEQ[1]], default=CONSEQ[3])
    VAL["period"] = np.where(VAL.arrival_season.astype(int) <= EARLY_LAST, EARLY_LABEL, LATE_LABEL)
    Nn["validation"] = {"events": len(VAL), "players": int(VAL.player_id.nunique()),
                        **{f"match_{k}d": int(VAL[f"match_{k}d"].sum()) for k in (0, 1, 3, 7)},
                        "covered_by_interval": int(VAL.covered_by_interval.sum()),
                        "events_of_players_without_rows": int(VAL.player_status.eq("B").sum()),
                        "lag_distribution": {int(k): int(v) for k, v in VAL.loc[VAL.abs_lag <= 7, "lag"].value_counts().sort_index().items()},
                        "by_consequence": {c: {"events": int((VAL.after_event == c).sum()),
                                               **{f"match_{k}d": float(VAL.loc[VAL.after_event == c, f"match_{k}d"].mean())
                                                  for k in (0, 1, 3, 7)}} for c in CONSEQ},
                        "missed_two_by_period": {p_: {"events": int(len(g)), "match_7d": float(g.match_7d.mean())}
                                                 for p_, g in VAL[VAL.after_event == CONSEQ[0]].groupby("period")},
                        "missed_two_by_season": {int(s): {"events": int(len(g)), "match_7d": float(g.match_7d.mean())}
                                                 for s, g in VAL[VAL.after_event == CONSEQ[0]].groupby("arrival_season")},
                        "by_season": {int(s): {"events": len(g), "match_3d": float(g.match_3d.mean()), "match_7d": float(g.match_7d.mean())}
                                      for s, g in VAL.groupby("arrival_season")},
                        "by_league": {lg: {"events": len(g), "match_3d": float(g.match_3d.mean()), "match_7d": float(g.match_7d.mean())}
                                      for lg, g in VAL.groupby("receiving_league")}}
    pick = lambda d, n: [f"player {r.player_id}, event {r.event_date.date()}: nearest injury {r.nearest_injury_reason} "
                         f"from {r.nearest_injury_start.date() if pd.notna(r.nearest_injury_start) else '—'} "
                         f"(lag {'' if pd.isna(r.lag) else int(r.lag)} days)" for r in d.head(n).itertuples()]
    Nn["validation_examples"] = {"strong": pick(VAL[VAL.match_1d & ~VAL.ambiguous].sort_values(["arrival_season", "player_id"]), 4),
                                 "misses": pick(VAL[~VAL.match_7d & VAL.player_status.eq("A")].sort_values(["arrival_season", "player_id"]), 4),
                                 "ambiguous": pick(VAL[VAL.ambiguous].sort_values(["arrival_season", "player_id"]), 3)}

    # ---- Part 9 descriptives on the usable sample
    U = E[E.usable].copy()
    U["delta"] = U.fit_adjusted_share - U.current_share
    burden = {}
    for k in KINDS:
        g = U[U.kind == k]
        burden[k] = {"N": len(g), "current": describe(g.current_share), "fit": describe(g.fit_adjusted_share),
                     "mean_injury_games": float(g.confirmed_injury_games.mean()),
                     **{f"share_ge_{t}": float((g.confirmed_injury_games >= t).mean()) for t in (1, 3, 5, 10)},
                     "share_zero": float((g.confirmed_injury_games == 0).mean()),
                     "delta": describe(g.delta)}
    Nn["burden"] = burden
    Nn["raw_gap"] = {"current": diff(U.loc[U.kind == "loan", "current_share"], U.loc[U.kind == "permanent", "current_share"]),
                     "fit": diff(U.loc[U.kind == "loan", "fit_adjusted_share"], U.loc[U.kind == "permanent", "fit_adjusted_share"])}
    for name in ("strict", "severe3", "long5"):
        Nn["raw_gap"][name] = diff(U.loc[U.kind == "loan", f"fit_share_{name}"], U.loc[U.kind == "permanent", f"fit_share_{name}"])

    # ---- Part 10-13 models on the usable, covariate-complete sample
    need = ["current_share", "fit_adjusted_share", "remain", "age", "position_clean", "log_relative_value", "log_squad_value"]
    C = U.dropna(subset=need).copy()
    C["loan"] = (C.kind == "loan").astype(int)
    C["age_c"] = C.age - C.age.mean()
    C["age_c2"] = C.age_c ** 2
    C["older"] = (np.floor(C.age) > pta.CUTOFF).astype(float)
    C["loan_x_older"] = C.loan * C.older
    Nn["model_sample"] = {k: int((C.kind == k).sum()) for k in KINDS}
    mrows = []
    mrows += pair_models(C, M3, ["loan"], "preferred (M3)")
    cur, fit_ = mrows[0], mrows[1]
    assert cur["N"] == fit_["N"]
    Nn["preferred"] = {"current": cur, "fit": fit_, "change": fit_["coef"] - cur["coef"]}
    Nn["age"] = {"current": age_contrasts(C, "current_share"), "fit": age_contrasts(C, "fit_adjusted_share")}
    for side in ("current", "fit"):
        for t in ("young", "older", "interaction"):
            mrows.append({"model": "age (young ≤23 vs older ≥24)", "outcome": "current" if side == "current" else "fit-adjusted",
                          "sample": f"N={Nn['age'][side]['N']}", "term": t, **{k: Nn["age"][side][t][k] for k in ("coef", "se", "lo", "hi", "p")},
                          "N": Nn["age"][side]["N"]})
    eng = C[C.receiving_league == "GB1"]
    er = pair_models(eng, M3, ["loan"], "England (GB1) preferred")
    mrows += er
    Nn["england"] = {"current": er[0], "fit": er[1], "N": er[0]["N"], "loans": int(eng.loan.sum())}
    Nn["sensitivity"] = {}
    for name, col in (("A primary", "fit_adjusted_share"), ("B strict", "fit_share_strict"),
                      ("C severe (spells removing ≥3 games)", "fit_share_severe3"), ("D long (spells removing ≥5 games)", "fit_share_long5")):
        r = pair_models(C, M3, ["loan"], f"sensitivity {name}", fit_col=col)
        mrows += r[1:]
        Nn["sensitivity"][name] = {"coef": r[1]["coef"], "lo": r[1]["lo"], "hi": r[1]["hi"], "N": r[1]["N"],
                                   "mean_injury_games": {k: float(C.loc[C.kind == k, {"fit_adjusted_share": "confirmed_injury_games",
                                                                                     "fit_share_strict": "injury_games_strict",
                                                                                     "fit_share_severe3": "injury_games_severe3",
                                                                                     "fit_share_long5": "injury_games_long5"}[col]].mean())
                                                         for k in KINDS}}
    late = C[C.arrival_season.astype(int) > EARLY_LAST]
    rl = pair_models(late, M3, ["loan"], f"post-hoc check: {LATE_LABEL} only")
    mrows += rl
    Nn["late_only"] = {"current": rl[0], "fit": rl[1]}
    A_only = C[C.status == "A"]
    ra = pair_models(A_only, M3, ["loan"], "coverage check: players with injury rows only")
    mrows += ra
    Nn["a_only"] = {"current": ra[0], "fit": ra[1]}
    MOD = pd.DataFrame(mrows)
    Nn["model_event_ids_identical"] = True

    # ---- Part 14 coverage bias
    def prof_(g):
        return {"N": len(g), "loan_share": float(g.kind.eq("loan").mean()), "age_median": float(g.age.median()),
                "player_value_median": float(g.player_value.median()), "relative_value_median": float(g.relative_value.median()),
                "current_share_mean": float(g.current_share.mean()), "season_median": float(g.arrival_season.astype(float).median()),
                "gb1_share": float(g.receiving_league.eq("GB1").mean())}
    Nn["bias"] = {"usable": prof_(E[E.usable]), "unusable": prof_(E[~E.usable]), "all": prof_(E)}
    Nn["unusable_by_league"] = {lg: float(1 - g.usable.mean()) for lg, g in E.groupby("receiving_league")}
    return Nn, {"INJ": INJ, "labels": labels, "E": E, "G": G, "JOIN": JOIN, "COV": COV, "VAL": VAL, "MOD": MOD, "C": C}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(N: dict, E: pd.DataFrame) -> None:
    plt = _style()
    (OUT / "figures").mkdir(exist_ok=True)
    U = E[E.usable]
    d = U.fit_adjusted_share - U.current_share
    fig, ax = plt.subplots(figsize=(10, 4.4))
    edges = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    labs = ["0", "0–0.01", "0.01–0.02", "0.02–0.05", "0.05–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.5", "> 0.5"]
    x = np.arange(len(labs))
    for off, k in ((-0.18, "loan"), (0.18, "permanent")):
        s = d[U.kind == k]
        h = [(s == 0).mean() * 100] + [((s > lo) & (s <= hi)).mean() * 100 for lo, hi in zip(edges[:-1], edges[1:])]
        ax.bar(x + off, h, width=0.34, color=COLOR[k], label=f"{LABEL[k]} (N = {len(s):,})")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_xlabel("Fit-adjusted share minus current share (same episode)")
    ax.set_ylabel("Share of episodes (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("How much the injury adjustment moves each episode (injury-usable sample)", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["dist"], dpi=160)
    plt.close(fig)

    rows = [("Preferred model, loan", N["preferred"]["current"], N["preferred"]["fit"])]
    for t, lab in (("young", "Young (≤23): loan − permanent"), ("older", "Older (≥24): loan − permanent"),
                   ("interaction", "Loan × older")):
        rows.append((lab, N["age"]["current"][t], N["age"]["fit"][t]))
    rows.append(("England (GB1), loan", N["england"]["current"], N["england"]["fit"]))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, c, f) in zip(y, rows):
        ax.errorbar(100 * c["coef"], yi + 0.14, xerr=[[100 * (c["coef"] - c["lo"])], [100 * (c["hi"] - c["coef"])]],
                    fmt="o", ms=7, mfc="white", color=MUTED, elinewidth=1.5, capsize=0,
                    label="current outcome" if yi == y[0] else None)
        ax.errorbar(100 * f["coef"], yi - 0.14, xerr=[[100 * (f["coef"] - f["lo"])], [100 * (f["hi"] - f["coef"])]],
                    fmt="o", ms=8, color=INK, elinewidth=2, capsize=0, label="fit-adjusted outcome" if yi == y[0] else None)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Estimate (pp of league minutes), 95% interval")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Same observations, two outcomes (N = {N['preferred']['current']['N']:,}; England N = {N['england']['N']:,})",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["coef"], dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def pp(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.1f} pp"


def ci(d) -> str:
    return f"[{pp(d['lo'])}, {pp(d['hi'])}]"


def p3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


DECISION = "Injury adjustment does not materially alter the main results."


def recommendation(N: dict) -> tuple[str, str]:
    """The decision is a recorded judgement. The checks below are the facts it rests on: if a rerun changes any of
    them, this raises and the decision must be reviewed rather than silently kept."""
    pr, ag, en = N["preferred"], N["age"], N["england"]
    checks = {"preferred loan coefficient moves by less than 1 pp": abs(pr["change"]) < 0.01,
              "age contrasts move by less than 1 pp": all(abs(ag["fit"][t]["coef"] - ag["current"][t]["coef"]) < 0.01
                                                         for t in ("young", "older", "interaction")),
              "England moves by less than 1 pp": abs(en["fit"]["coef"] - en["current"]["coef"]) < 0.01,
              "every severity definition moves the loan coefficient by less than 1 pp":
                  all(abs(x["coef"] - pr["current"]["coef"]) < 0.01 for x in N["sensitivity"].values())}
    if not all(checks.values()):
        raise AssertionError(f"recorded decision no longer supported: {checks}")
    v, us, rv = N["validation"], N["usable"], N["revised"]
    mt = v["by_consequence"][CONSEQ[0]]
    share = (us["loan"] + us["permanent"]) / (rv["loan"] + rv["permanent"])
    return DECISION, (f"on the same {pr['current']['N']:,} observations the adjustment moves the preferred loan coefficient "
                      f"by {pp(pr['change'])}, the young/older contrasts and England by under 1 pp, and no severity "
                      f"definition changes that. Use it as a robustness check, not as the headline outcome: it covers "
                      f"{pct(share)} of revised episodes, and the source records an injury for {pct(mt['match_7d'])} of "
                      f"in-match injury substitutions followed by two missed league games "
                      f"({pct(v['missed_two_by_period'][EARLY_LABEL]['match_7d'])} in {EARLY_LABEL})")


def render(N: dict, labels: pd.DataFrame) -> str:
    nz, jp, us, rv = N["normalize"], N["join_players"], N["usable"], N["revised"]
    v, bu, rg = N["validation"], N["burden"], N["raw_gap"]
    pr, ag, en = N["preferred"], N["age"], N["england"]
    rec, why = recommendation(N)
    nplayers = N["revised_players"]
    W = ["# Injury-adjusted playing time (external Kaggle injury histories)", "",
         "*Generated by `python -m src.analysis.playing_time_injury_adjusted`. Source: Kaggle "
         f"`{SOURCE['dataset']}` version {SOURCE['version']} ({SOURCE['version_date']}), {SOURCE['license']}, manually "
         "downloaded by the user and verified against Kaggle's file listing; original stated source Transfermarkt. No "
         "Transfermarkt page was fetched and the GitHub copy was not used. Observational; not causal.*", ""]
    W += [f"## Recommendation: {rec}", "", f"Reason: {why}.", ""]

    W += ["## 1–2. Source and normalisation", ""]
    W += md_table(["File", "Bytes", "Kaggle-listed bytes", "SHA-256"],
                  [[k, f0(f["bytes"]), f0(f["kaggle_listed_bytes"]), f"`{f['sha256']}`"] for k, f in N["files"].items()], right={1, 2})
    W += ["", f"{f0(nz['raw_rows'])} raw rows, {f0(nz['exact_duplicates_dropped'])} exact duplicates dropped, "
          f"{f0(nz['rows'])} rows for {f0(nz['players'])} players. Flagged, not fixed: {nz['missing_start']} without a "
          f"start date, {f0(nz['open'])} open (no end date), {nz['end_before_start']} ending before they start, "
          f"{f0(nz['conflicting_duplicate_rows'])} rows sharing a player and start date, {nz['starts_after_snapshot']} "
          "starting after the snapshot.", "",
          f"**History truncation.** No player has more than {nz['max_rows_per_player']} rows, and "
          f"{f0(nz['players_truncated'])} players have exactly {TRUNCATION_CAP}, listed newest first: the source kept only "
          "the latest page of each injury history. For these players, injuries before their oldest listed one are "
          "unobserved, so episodes that start earlier get no fit-adjusted value.", ""]
    cats = labels.groupby("category").agg(labels_n=("injury_reason_raw", "size"), rows=("rows", "sum"),
                                          primary=("counts_primary", "first"), strict=("counts_strict", "first"))
    W += ["**Classification** (every label is listed in `injury_reason_classification.csv`):", ""]
    W += md_table(["Category", "Distinct labels", "Rows", "PRIMARY", "STRICT"],
                  [[c, f0(r.labels_n), f0(r.rows), "counts" if r.primary else "–", "counts" if r.strict else "–"]
                   for c, r in cats.sort_values("rows", ascending=False).iterrows()], right={1, 2})
    W += ["", "Judgement calls, stated: 'unknown injury' is an injury of unknown type and counts in PRIMARY only; muscle "
          "fatigue / stiffness and sore muscles count in PRIMARY only; 'Fitness' and 'Quarantine' never count; 'Rest' is "
          "non-medical. The source has no suspension or international-duty labels.", ""]

    W += ["## 3. Exact player-id join", ""]
    W += md_table(["Revised-sample players", "N"],
                  [["distinct players", f0(nplayers)],
                   ["A. in the external datalake with injury rows", f0(jp.get("A", 0))],
                   ["B. in the datalake, zero injury rows", f0(jp.get("B", 0))],
                   ["C. absent from the datalake", f0(jp.get("C", 0))],
                   ["D. ambiguous id", f0(jp.get("D", 0))],
                   ["**exact-id match rate (A + B)**", f"**{pc(jp.get('A', 0) + jp.get('B', 0), nplayers)}**"]])
    ic = N["id_check"]
    W += ["", f"Id sanity check: of {f0(ic['compared'])} matched players with a birth date on both sides, "
          f"{f0(ic['dob_equal'])} ({pc(ic['dob_equal'], ic['compared'])}) have the same date of birth."
          + (" Mismatches, e.g.: " + "; ".join(ic["examples_mismatch"][:4]) + "." if ic["examples_mismatch"] else ""), "",
          "By move type: " + "; ".join(f"{LABEL[k]}: {f0(N['join_by_kind'][k]['players_in_lake'])} of "
                                       f"{f0(N['join_by_kind'][k]['players'])} players in the datalake, "
                                       f"{f0(N['join_by_kind'][k]['players_with_rows'])} with injury rows" for k in KINDS)
          + f". By league, season, age group and relative-value quintile in `{JOIN_CSV.name}`.", ""]

    W += ["## 4. Coverage and usability", ""]
    reasons = sorted((set(N["usability"]["loan"]) | set(N["usability"]["permanent"])) - {"usable"})
    W += md_table(["Episode status", "Loans", "Permanent"],
                  [[r, f0(N["usability"]["loan"].get(r, 0)), f0(N["usability"]["permanent"].get(r, 0))] for r in reasons]
                  + [["**usable**", f"**{f0(us['loan'])}** of {f0(rv['loan'])}", f"**{f0(us['permanent'])}** of {f0(rv['permanent'])}"]])
    cs = N["coverage_by_season"]
    W += ["", "By arrival season:", ""]
    W += md_table(["Season", "Episodes", "Usable", "Players with any injury row", "Players with an injury in the window",
                   "Excluded: truncated history"],
                  [[f"{int(float(s))}/{str(int(float(s)) + 1)[2:]}", f0(r["episodes"]), pct(r["usable_share"]),
                    pct(r["players_with_injury_rows_share"]), pct(r["players_with_injury_in_window_share"]), f0(r["truncated_excluded"])]
                   for s, r in cs.items()])
    cl = N["coverage_by_league"]
    W += ["", "By league:", ""]
    W += md_table(["League", "Episodes", "Usable", "Players with any injury row", "Players with an injury in the window"],
                  [[lg, f0(r["episodes"]), pct(r["usable_share"]), pct(r["players_with_injury_rows_share"]),
                    pct(r["players_with_injury_in_window_share"])] for lg, r in sorted(cl.items(), key=lambda kv: -kv[1]["episodes"])])
    W += ["", f"League-season detail in `{COVER_CSV.name}`. 2025/26 windows reach the snapshot and are never usable.", ""]

    W += ["## 6. Validation against in-match injury substitutions", "",
          f"{f0(v['events'])} in-window injury substitutions (before the snapshot) of {f0(v['players'])} players in the "
          "datalake. For each, the nearest external injury start (any label); one external spell may match several "
          "events.", ""]
    W += md_table(["Window", "Events matched", "Share"],
                  [[lab, f0(v[f"match_{k}d"]), pc(v[f"match_{k}d"], v["events"])] for k, lab in
                   ((0, "same day"), (1, "±1 day"), (3, "±3 days"), (7, "±7 days"))]
                  + [["covered by a spell interval (start ≤ event + 7 days, end ≥ event)", f0(v["covered_by_interval"]),
                      pc(v["covered_by_interval"], v["events"])]])
    W += ["", "**By what happened next.** An injury substitution need not cause an absence, and the source records "
          "absences. Splitting the events by whether the player then appeared in B's next league games (used only to "
          "group the events, never to create an injury):", ""]
    W += md_table(["After the event", "Events", "Same day", "±1 day", "±3 days", "±7 days"],
                  [[c, f0(r["events"]), pct(r["match_0d"]), pct(r["match_1d"]), pct(r["match_3d"]), pct(r["match_7d"])]
                   for c, r in v["by_consequence"].items()])
    mp = v["missed_two_by_period"]
    W += ["", f"For the clearest absences (two missed league games) the source has an injury starting within 7 days in "
          + "; ".join(f"{pct(r['match_7d'])} of {f0(r['events'])} events in {p_}" for p_, r in mp.items())
          + ". Roughly three in ten such absences are missing from the source, more in early seasons.", ""]
    W += ["", f"{f0(v['events_of_players_without_rows'])} events belong to players with no injury rows at all (B): "
          "each is a known injury the source does not have. Signed lag (external start minus event) within ±7 days: "
          + ", ".join(f"{k:+d}: {f0(c)}" for k, c in v["lag_distribution"].items()) + ".", ""]
    W += md_table(["Season", "Events", "Matched ±3d", "Matched ±7d"],
                  [[f"{s}/{str(s + 1)[2:]}", f0(r["events"]), pct(r["match_3d"]), pct(r["match_7d"])] for s, r in
                   sorted(v["by_season"].items())])
    W += [""]
    W += md_table(["League", "Events", "Matched ±3d", "Matched ±7d"],
                  [[lg, f0(r["events"]), pct(r["match_3d"]), pct(r["match_7d"])] for lg, r in sorted(v["by_league"].items())])
    ex = N["validation_examples"]
    W += ["", "Strong matches: " + "; ".join(ex["strong"]) + ".", "", "Clear misses: " + "; ".join(ex["misses"]) + ".", "",
          "Ambiguous (several external rows within ±7 days): " + "; ".join(ex["ambiguous"]) + ".", ""]

    W += ["## 7–9. The adjustment on the injury-usable sample", ""]
    W += md_table(["", "Loans", "Permanent"],
                  [["N usable", f0(bu["loan"]["N"]), f0(bu["permanent"]["N"])],
                   ["mean current share", p3(bu["loan"]["current"]["mean"]), p3(bu["permanent"]["current"]["mean"])],
                   ["mean fit-adjusted share", p3(bu["loan"]["fit"]["mean"]), p3(bu["permanent"]["fit"]["mean"])],
                   ["median current share", p3(bu["loan"]["current"]["median"]), p3(bu["permanent"]["current"]["median"])],
                   ["median fit-adjusted share", p3(bu["loan"]["fit"]["median"]), p3(bu["permanent"]["fit"]["median"])],
                   ["mean injury games removed", f"{bu['loan']['mean_injury_games']:.2f}", f"{bu['permanent']['mean_injury_games']:.2f}"],
                   ["zero injury games removed", pct(bu["loan"]["share_zero"]), pct(bu["permanent"]["share_zero"])],
                   *[[f"≥{t} injury games removed", pct(bu["loan"][f"share_ge_{t}"]), pct(bu["permanent"][f"share_ge_{t}"])]
                     for t in (1, 3, 5, 10)],
                   ["mean change in share", p3(bu["loan"]["delta"]["mean"]), p3(bu["permanent"]["delta"]["mean"])]])
    W += ["", f"**Raw loan − permanent, same usable sample:** current {pp(rg['current']['mean_diff'])} "
          f"[{pp(rg['current']['mean_diff_lo'])}, {pp(rg['current']['mean_diff_hi'])}]; fit-adjusted "
          f"{pp(rg['fit']['mean_diff'])} [{pp(rg['fit']['mean_diff_lo'])}, {pp(rg['fit']['mean_diff_hi'])}]. "
          f"Appearances inside a spell (the appearance wins): {f0(N['appearance_conflicts']['games'])} games in "
          f"{f0(N['appearance_conflicts']['episodes'])} episodes.", "", f"![Distribution]({FIGS['dist']})", ""]

    W += ["## 10–12. Same-sample regressions", "",
          f"Preferred model (loan, arrival timing, age, age², log relative value, log squad value, position, league × "
          f"season fixed effects, errors clustered by receiving club), run twice on the same {f0(pr['current']['N'])} "
          f"episodes ({f0(N['model_sample']['loan'])} loans, {f0(N['model_sample']['permanent'])} permanent).", ""]
    rows = [["preferred: loan", f"{pp(pr['current']['coef'])} {ci(pr['current'])}", f"{pp(pr['fit']['coef'])} {ci(pr['fit'])}",
             pp(pr["change"])]]
    for t, lab in (("young", "young (≤23): loan − permanent"), ("older", "older (≥24): loan − permanent"),
                   ("interaction", "loan × older")):
        rows.append([lab, f"{pp(ag['current'][t]['coef'])} {ci(ag['current'][t])}", f"{pp(ag['fit'][t]['coef'])} {ci(ag['fit'][t])}",
                     pp(ag["fit"][t]["coef"] - ag["current"][t]["coef"])])
    rows.append([f"England (GB1, N = {f0(en['N'])}, {f0(en['loans'])} loans): loan", f"{pp(en['current']['coef'])} {ci(en['current'])}",
                 f"{pp(en['fit']['coef'])} {ci(en['fit'])}", pp(en["fit"]["coef"] - en["current"]["coef"])])
    W += md_table(["", "Current outcome", "Fit-adjusted outcome", "Change"], rows)
    W += ["", f"![Coefficients]({FIGS['coef']})", ""]

    W += ["## 13. Data-quality sensitivities (same observations, loan coefficient)", ""]
    W += md_table(["Definition", "Fit-adjusted loan coefficient", "Mean injury games removed (loan / permanent)"],
                  [[k, f"{pp(s['coef'])} [{pp(s['lo'])}, {pp(s['hi'])}]",
                    f"{s['mean_injury_games']['loan']:.2f} / {s['mean_injury_games']['permanent']:.2f}"] for k, s in N["sensitivity"].items()]
                  + [["current outcome (reference)", f"{pp(pr['current']['coef'])} {ci(pr['current'])}", "0 / 0"]])
    lo_ = N["late_only"]
    W += ["", f"Post-hoc data-quality check, added after the validation showed weaker early capture and reported "
          f"whatever it showed: {LATE_LABEL} only (N = {f0(lo_['current']['N'])}): current {pp(lo_['current']['coef'])} "
          f"{ci(lo_['current'])}, fit-adjusted {pp(lo_['fit']['coef'])} {ci(lo_['fit'])}."]
    ao = N["a_only"]
    W += ["", f"Coverage check, players with injury rows only (N = {f0(ao['current']['N'])}): current {pp(ao['current']['coef'])}, "
          f"fit-adjusted {pp(ao['fit']['coef'])}.", ""]

    b = N["bias"]
    W += ["## 14. Is the usable sample selected?", ""]
    W += md_table(["", "Usable", "Not usable", "All revised"],
                  [["episodes", f0(b["usable"]["N"]), f0(b["unusable"]["N"]), f0(b["all"]["N"])],
                   ["loan share", pct(b["usable"]["loan_share"]), pct(b["unusable"]["loan_share"]), pct(b["all"]["loan_share"])],
                   ["median age", f"{b['usable']['age_median']:.1f}", f"{b['unusable']['age_median']:.1f}", f"{b['all']['age_median']:.1f}"],
                   ["median player value", f"€{b['usable']['player_value_median'] / 1e3:,.0f}k", f"€{b['unusable']['player_value_median'] / 1e3:,.0f}k",
                    f"€{b['all']['player_value_median'] / 1e3:,.0f}k"],
                   ["median relative value", f"{b['usable']['relative_value_median']:.3f}", f"{b['unusable']['relative_value_median']:.3f}",
                    f"{b['all']['relative_value_median']:.3f}"],
                   ["mean current share", p3(b["usable"]["current_share_mean"]), p3(b["unusable"]["current_share_mean"]),
                    p3(b["all"]["current_share_mean"])],
                   ["median arrival season", f"{b['usable']['season_median']:.0f}", f"{b['unusable']['season_median']:.0f}",
                    f"{b['all']['season_median']:.0f}"],
                   ["England share", pct(b["usable"]["gb1_share"]), pct(b["unusable"]["gb1_share"]), pct(b["all"]["gb1_share"])]])
    ul, upm = N["usability"]["loan"], N["usability"]["permanent"]
    g_ = lambda r: ul.get(r, 0) + upm.get(r, 0)
    W += ["", f"Most unusable episodes are 2025/26 windows that reach the snapshot ({f0(g_('window reaches the 2025-09-15 snapshot'))}), "
          f"players absent from the datalake ({f0(g_('player not in the external datalake'))}) and truncated histories "
          f"({f0(g_('injury history truncated before the window'))}). On age, player value, relative value and current share "
          "the usable sample is close to the full revised sample, so apart from losing 2025/26 and more episodes in "
          + ", ".join(sorted(N["unusable_by_league"], key=N["unusable_by_league"].get, reverse=True)[:4])
          + " it is not strongly selected."]
    W += ["", "Share of episodes not usable, by league: "
          + ", ".join(f"{lg} {pct(x)}" for lg, x in sorted(N["unusable_by_league"].items(), key=lambda kv: -kv[1])) + ".", ""]
    return "\n".join(W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _json(x):
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


def old_outputs() -> list:
    names = ["playing_time_sample_redesign.md", "playing_time_sample_redesign_numbers.json", "playing_time_age_analysis.md",
             "playing_time_age_numbers.json", "playing_time_regression_numbers.json", "playing_time_regression_models.csv",
             "playing_time_injury_feasibility.md", "playing_time_injury_numbers.json", "injury_pilot_summary.md",
             "injury_pilot_numbers.json", "injury_external_sources_audit.md", "injury_external_numbers.json",
             "injury_external_sources_comparison.csv", "playing_time_join_episodes.csv"]
    return [OUT / n for n in names]


def main(u=None) -> dict:
    from .club_geography_v2 import CANONICAL_CSV, STAGE1C_SHA256, sha256

    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    before = {p.name: sha256(p) for p in old_outputs() if p.exists()}
    assert not {p.name for p in WRITES} & set(before)
    raw_before = {k: sha256(RAW / k) for k in EXPECTED}
    N, tb = compute(u)
    MANIFEST.write_text(json.dumps(_json({**SOURCE, "files": N["files"], "verified_on": pd.Timestamp.now().isoformat(timespec="seconds"),
                                          "kaggle_listing": str(RAW_META / "kaggle_file_listing.json"),
                                          "note": "2025/26 injury coverage is incomplete: the snapshot is dated 2025-09-15"}), indent=1))
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    tb["INJ"].to_csv(PROCESSED, index=False)
    tb["labels"].to_csv(CLASS_CSV, index=False)
    tb["JOIN"].to_csv(JOIN_CSV, index=False)
    tb["VAL"].to_csv(VALID_CSV, index=False)
    tb["COV"].to_csv(COVER_CSV, index=False)
    cols = ["event_id", "player_id", "kind", "current_share", "fit_adjusted_share", "available_games_current",
            "confirmed_injury_games", "available_games_fit", "injury_days_overlap", "injury_spell_count", "data_quality_flag",
            "usable", "status", "appearance_conflicts", "receiving_league", "arrival_season", "age_group", "rv_quintile",
            "injury_games_strict", "fit_share_strict", "injury_games_severe3", "fit_share_severe3", "injury_games_long5",
            "fit_share_long5", "window_start", "window_end"]
    out = tb["E"][cols].copy()
    for c in ("window_start", "window_end"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(EPISODES_CSV, index=False)
    tb["MOD"].to_csv(MODELS_CSV, index=False)
    figures(N, tb["E"])
    md = render(N, tb["labels"])
    for v in (N["usable"]["loan"], N["usable"]["permanent"], N["revised_players"], N["validation"]["events"]):
        assert f0(v) in md, v
    SUMMARY_MD.write_text(md)
    N["model_event_ids"] = sorted(tb["C"].event_id)
    N["old_output_sha256"] = before
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    assert {p.name: sha256(p) for p in old_outputs() if p.exists()} == before, "an old output changed"
    assert {k: sha256(RAW / k) for k in EXPECTED} == raw_before, "a raw file changed"
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print_summary(N)
    return N


def print_summary(N: dict) -> None:
    f = N["files"]["player_injuries/player_injuries.csv"]
    jp, us, v, bu, rg, pr, ag, en = (N["join_players"], N["usable"], N["validation"], N["burden"], N["raw_gap"],
                                    N["preferred"], N["age"], N["england"])
    cs = N["coverage_by_season"]
    weak_s = [f"{int(float(s))}/{str(int(float(s)) + 1)[2:]}" for s, r in cs.items() if r["usable_share"] < 0.5]
    rec, why = recommendation(N)
    print("SOURCE")
    print(f"- Kaggle dataset: {SOURCE['dataset']} v{SOURCE['version']} ({SOURCE['version_date']}), manually downloaded")
    print(f"- licence: {SOURCE['license']}")
    print(f"- injury rows: {N['normalize']['raw_rows']:,} ({N['normalize']['rows']:,} after dropping exact duplicates)")
    print(f"- unique players: {N['normalize']['players']:,} with injuries; {N['profiles']['unique_ids']:,} profiles")
    print(f"- SHA256: {f['sha256']} (injuries); {N['files']['player_profiles/player_profiles.csv']['sha256']} (profiles)")
    print("JOIN")
    print(f"- revised distinct players: {N['revised_players']:,}")
    print(f"- found in external datalake: {jp.get('A', 0) + jp.get('B', 0):,}")
    print(f"- exact-id match rate: {pc(jp.get('A', 0) + jp.get('B', 0), N['revised_players'])}")
    print(f"- players with injury rows: {jp.get('A', 0):,}")
    print("COVERAGE")
    print(f"- usable loan episodes: {us['loan']:,} of {N['revised']['loan']:,}")
    print(f"- usable permanent episodes: {us['permanent']:,} of {N['revised']['permanent']:,}")
    mp = v["missed_two_by_period"]
    print(f"- earliest reliable season: {LATE_LABEL.split('–')[0]} (injury recorded for {pct(mp[LATE_LABEL]['match_7d'])} of "
          f"two-game absences after an injury substitution, against {pct(mp[EARLY_LABEL]['match_7d'])} in {EARLY_LABEL})")
    print("- latest season: 2024/25; 2025/26 windows reach the 2025-09-15 snapshot and are never usable")
    lw = sorted(N["coverage_by_league"].items(), key=lambda kv: kv[1]["players_with_injury_rows_share"])[:3]
    print(f"- weak league/seasons: {', '.join(weak_s)} (under half usable); fewest players with any injury row: "
          + ", ".join(f"{k} {pct(r['players_with_injury_rows_share'])}" for k, r in lw))
    print("VALIDATION")
    print(f"- injury substitution events checked: {v['events']:,}")
    print(f"- matched same day: {v['match_0d']:,} ({pc(v['match_0d'], v['events'])})")
    print(f"- matched +/-3d: {v['match_3d']:,} ({pc(v['match_3d'], v['events'])})")
    print(f"- matched +/-7d: {v['match_7d']:,} ({pc(v['match_7d'], v['events'])}); for events followed by two missed "
          f"league games: {pct(v['by_consequence'][CONSEQ[0]]['match_7d'])} of {v['by_consequence'][CONSEQ[0]]['events']:,}")
    print("INJURY BURDEN")
    print(f"- loans mean confirmed injury games removed: {bu['loan']['mean_injury_games']:.2f}")
    print(f"- permanent mean: {bu['permanent']['mean_injury_games']:.2f}")
    print(f"- loans with >=1 injury game: {pct(bu['loan']['share_ge_1'])}")
    print(f"- permanent with >=1: {pct(bu['permanent']['share_ge_1'])}")
    print("RAW SAME-SAMPLE")
    print(f"- current loan-minus-permanent gap: {pp(rg['current']['mean_diff'])}")
    print(f"- fit-adjusted loan-minus-permanent gap: {pp(rg['fit']['mean_diff'])}")
    print("PREFERRED REGRESSION SAME-SAMPLE")
    print(f"- current outcome loan coefficient: {pp(pr['current']['coef'])} {ci(pr['current'])} (N {pr['current']['N']:,})")
    print(f"- fit-adjusted outcome loan coefficient: {pp(pr['fit']['coef'])} {ci(pr['fit'])}")
    print(f"- change: {pp(pr['change'])}")
    print("AGE")
    for t, lab in (("young", "young"), ("older", "older"), ("interaction", "interaction")):
        print(f"- {lab} current / fit-adjusted: {pp(ag['current'][t]['coef'])} / {pp(ag['fit'][t]['coef'])}")
    print("ENGLAND")
    print(f"- current: {pp(en['current']['coef'])} {ci(en['current'])} (N {en['N']:,})")
    print(f"- fit-adjusted: {pp(en['fit']['coef'])} {ci(en['fit'])}")
    print("RECOMMENDATION")
    print(f"- {rec}")
    print(f"- {why}")


if __name__ == "__main__":
    main()
