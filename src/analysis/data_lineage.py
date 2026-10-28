"""Data lineage, data dictionary and sample construction for the whole project. Documentation only.

    python -m src.analysis.data_lineage

Rebuilds every sample in memory with the project's own functions (no definition is changed or re-implemented),
checks each one against its saved output, and writes new files only:

    stage1c_column_dictionary.csv     every Stage 1C column
    data_source_inventory.csv / .md   every raw and intermediate source
    analysis_episode_master.csv       one row per transfer episode, the variables we use
    analysis_variable_dictionary.csv  one row per master-table variable
    data_lineage_overview.md          the lineage, waterfalls, worked examples and Q&A
    data_lineage_numbers.json         every number quoted in the documents

Every other file under data/ is hashed before and after; the run fails if any of them changed.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import club_geography_v2 as g2
from . import appearance_audit as aud
from . import club_organisation as co
from . import injury_method_example as ime
from . import loan_calendar_rule as lc
from . import loan_calendar_rule_v2 as lcv2
from . import loan_scope
from . import loan_timing_exploration as lte
from . import playing_time_age as pta
from . import playing_time_injury_adjusted as ia
from . import playing_time_join as ptj
from . import playing_time_regression as reg
from .daniel_scope_final import f0, md_table, pc
from .column_guide_specs import (CORE_SHORTLIST, DOMAINS, MASTER_GUIDE, ORIGINS, S1C_ROLES, STAGE1C_GUIDE, USEFULNESS,
                                 VALID_WHEN)
from .data_lineage_specs import LOAN_FIELDS, SOURCES, STAGE1C, VARS
from .loan_episodes import CANONICAL_CSV, ROOT, build_universe, load_canonical
from ..stage1c.build import CANONICAL_COLUMNS
from ..stage2 import event_family as ef

OUT = ROOT / "data" / "outputs" / "rebuild"
STAGE1C_DICT_CSV = OUT / "stage1c_column_dictionary.csv"
INVENTORY_CSV = OUT / "data_source_inventory.csv"
INVENTORY_MD = OUT / "data_source_inventory.md"
MASTER_CSV = OUT / "analysis_episode_master.csv"
VARDICT_CSV = OUT / "analysis_variable_dictionary.csv"
LINEAGE_MD = OUT / "data_lineage_overview.md"
NUMBERS_JSON = OUT / "data_lineage_numbers.json"
CALFIRST_MD = OUT / "calendar_first_samples.md"
CALFIRST_CSV = OUT / "calendar_first_samples.csv"
CORE_MD = OUT / "master_core_columns.md"
ACADEMY_MD = OUT / "academy_promotion_sensitivity.md"
ACADEMY_CSV = OUT / "youth_reserve_review_universe.csv"
ACADEMY_REVIEW_CSV = OUT / "academy_promotion_manual_review.csv"
WRITES = (STAGE1C_DICT_CSV, INVENTORY_CSV, INVENTORY_MD, MASTER_CSV, VARDICT_CSV, LINEAGE_MD, NUMBERS_JSON, CALFIRST_MD,
          CALFIRST_CSV, CORE_MD, ACADEMY_MD, ACADEMY_CSV, ACADEMY_REVIEW_CSV)

KINDS = ("loan", "permanent")
ROLE_ORDER = ("internal_registration", "placeholder_non_club_movement", "non_transfer_other",
              "loan_return_collapsed", "loan_return_unmatched")
ROLE_TEXT = {
    "internal_registration": ("Stage 1C `is_internal_move` is True (same organisation on both sides, e.g. Ajax U21 → "
                              "Ajax)", "relabelled; not an episode"),
    "placeholder_non_club_movement": ("Stage 1C `involves_placeholder_club` is True (Without Club / Retired / Career "
                                      "break / Ban / Unknown)", "relabelled; not an episode"),
    "non_transfer_other": ("`transfer_type_normalized == 'other'` (Transfermarkt 'draft' label)",
                           "relabelled; not an episode"),
    "loan_return_collapsed": ("an End-of-loan row that `pair_loans` matched to an open outbound loan",
                              "collapsed into that loan's episode (it becomes the loan's `terminal_event_id`)"),
    "loan_return_unmatched": ("an End-of-loan row with no open loan from that club to close",
                              "set aside; not an episode"),
}
CODE_TOKENS = ("calendar_class", "off_calendar", "calendar_normal", "v2_state", "economic_ending",
               "early_termination", "geography_frame", "GEO_V2_CSV", "loan_calendar_rule")
DOWNSTREAM = ("playing_time_join", "playing_time_sample_redesign", "playing_time_regression", "playing_time_age",
              "playing_time_injury_feasibility", "playing_time_injury_adjusted")
NEED_OLD = ["share", "loan", "remain", "age", "position_clean", "log_relative_value", "log_squad_value"]
NEED_INJ = ["current_share", "fit_adjusted_share", "remain", "age", "position_clean", "log_relative_value",
            "log_squad_value"]
ANTON = 193004


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def watched() -> dict[str, str]:
    """SHA-256 of every file under data/ that this module does not write."""
    out = {}
    for p in sorted((ROOT / "data").rglob("*")):
        if p.is_file() and p not in WRITES and p.name != ".DS_Store":
            out[str(p.relative_to(ROOT))] = g2.sha256(p)
    return out


def cell(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).replace("|", "\\|").replace("\n", " ")


def ints(d) -> dict:
    return {str(k): int(v) for k, v in d.items()}


def dstr(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")


def nplayers(df: pd.DataFrame) -> int:
    return int(df.player_id.nunique())


def module_text(name: str) -> str:
    return (ROOT / "src" / "analysis" / f"{name}.py").read_text()


def references(column: str) -> list[str]:
    """Modules whose code names the column as a string or attribute (a textual scan, not a call graph)."""
    pat = re.compile(r"""["'.]""" + re.escape(column) + r"\b")
    mods = sorted(p for p in (ROOT / "src" / "analysis").glob("*.py") if not p.stem.startswith("data_lineage"))
    mods.append(ROOT / "src" / "stage2" / "event_family.py")
    return [p.stem for p in mods if pat.search(p.read_text())]


# ---------------------------------------------------------------------------
# Part 1: Stage 1C
# ---------------------------------------------------------------------------

def stage1c_facts(S: pd.DataFrame) -> dict:
    d = pd.to_datetime(S.transfer_date)
    per = S.groupby("player_id").size()
    seasons = S.transfer_season.map(lambda s: int(s[:2]))
    four = seasons.map(lambda y: 1900 + y if y >= 90 else 2000 + y)
    return {"rows": len(S), "columns": S.shape[1], "players": nplayers(S), "date_min": str(d.min().date()),
            "date_max": str(d.max().date()), "season_min": f"{four.min()}/{str(four.min() + 1)[2:]}",
            "season_max": f"{four.max()}/{str(four.max() + 1)[2:]}", "rows_before_2000": int((d.dt.year < 2000).sum()),
            "rows_from_2010": int((d.dt.year >= 2010).sum()), "future_flagged": int(S.transfermarkt_future_transfer.sum()),
            "rows_per_player_median": float(per.median()), "rows_per_player_max": int(per.max()),
            "raw_label": ints(S.transfer_type_raw.value_counts()), "normalized": ints(S.transfer_type_normalized.value_counts()),
            "label_to_type": {f"{a} -> {b}": int(n) for (a, b), n in
                              S.groupby(["transfer_type_raw", "transfer_type_normalized"]).size().items()},
            "type_source": ints(S.transfer_type_source.value_counts()),
            "enrichment_matched": int(S.enrichment_matched.sum()), "columns_list": list(S.columns)}


def stage1c_dictionary(S: pd.DataFrame, Sr: pd.DataFrame) -> pd.DataFrame:
    assert list(S.columns) == CANONICAL_COLUMNS and set(STAGE1C) == set(CANONICAL_COLUMNS)
    rows = []
    for c in CANONICAL_COLUMNS:
        m, raw = STAGE1C[c], Sr[c]
        nb = raw[raw.ne("")]
        vals = S[c].value_counts(dropna=False)
        used = m.used
        status = ("no" if used.startswith("Not used") else
                  "audit / display only" if used.startswith(("Display", "Audit", "QA", "Provenance", "Tie-breaker",
                                                              "Kept for comparison", "Descriptive")) else
                  "Stage 1C build only" if used.startswith("Stage 1C build only") else
                  "Stage 2 only" if used.startswith("Stage 2") else "yes")
        rows.append({"column": c, "position": CANONICAL_COLUMNS.index(c) + 1, "dtype": str(S[c].dtype),
                     "example_value": nb.iloc[0] if len(nb) else "", "raw_or_derived": m.raw_or_derived,
                     "original_source": m.origin, "plain_english_meaning": m.meaning,
                     "n_rows": len(S), "n_blank": int(raw.eq("").sum()), "n_zero": int(raw.isin(["0", "0.0"]).sum()),
                     "n_question_mark": int(raw.eq("?").sum()), "n_dash": int(raw.eq("-").sum()),
                     "n_distinct": int(S[c].nunique()),
                     "values": "; ".join(f"{'(blank)' if pd.isna(k) else k} ({int(n):,})" for k, n in vals.items())
                     if len(vals) <= 12 else "",
                     "missing_value_meaning": m.missing, "zero_meaningful": m.zero, "special_tokens": m.tokens,
                     "used_in_analysis": status, "downstream_use": used,
                     "referenced_by_modules": ", ".join(references(c)),
                     **guide_fields(c, S[c])})
    return pd.DataFrame(rows)


def guide_fields(c: str, s: pd.Series) -> dict:
    """Reader-facing role, origin, construction and explained categories; the categories must match the data exactly."""
    g = STAGE1C_GUIDE[c]
    cats = ""
    if g.values is not None:
        vc = s.astype("string").fillna("(blank)").replace("", "(blank)").value_counts()
        if pd.api.types.is_bool_dtype(s) or s.dropna().isin([True, False]).all() and s.dtype == object:
            vc.index = vc.index.map(lambda x: {"true": "True", "false": "False"}.get(str(x).lower(), x))
        assert set(vc.index) == set(g.values), (c, set(vc.index) ^ set(g.values))
        cats = "; ".join(f"{k} ({int(vc[k]):,}): {g.values[k]}" for k in vc.index)
    return {"role": S1C_ROLES[g.role], "comes_from": g.source, "how_created": g.how, "categories_explained": cats}


# ---------------------------------------------------------------------------
# Part 3: rows -> episodes
# ---------------------------------------------------------------------------

def episode_waterfall(u) -> dict:
    r, ep = u.rows, u.episodes
    steps, cur = [], len(r)
    for role in ROLE_ORDER:
        g = r[r.row_role == role]
        steps.append({"role": role, "before": cur, "removed": len(g), "after": cur - len(g),
                      "players_with_such_rows": nplayers(g), "by_type": ints(g.transfer_type_normalized.value_counts())})
        cur -= len(g)
    assert cur == len(ep) == int((r.row_role == "episode").sum())
    ret = u.returns
    closed = ret.dropna(subset=["loan_idx"])
    real_loan = set(r.index[(r.row_role == "episode") & (r.transfer_type_normalized == "loan")])
    to_real = closed[closed.loan_idx.isin(real_loan)]
    L = u.loans
    out = {"rows": len(r), "steps": steps, "episodes": len(ep), "episode_players": nplayers(ep),
           "rows_players": nplayers(r),
           "role_by_type": {f"{a} | {b}": int(n) for (a, b), n in r.groupby(["row_role", "transfer_type_normalized"]).size().items()},
           "episode_type": ints(ep.episode_type.value_counts()),
           "episode_type_players": ints(ep.groupby("episode_type").player_id.nunique()),
           "permanent_pooled": int(ep.episode_type.isin(["permanent_paid", "free_transfer", "undisclosed_fee",
                                                         "no_fee_shown"]).sum()),
           "permanent_players": nplayers(ep[ep.episode_type.isin(["permanent_paid", "free_transfer", "undisclosed_fee",
                                                                  "no_fee_shown"])]),
           "outbound_loan_rows": int((r.transfer_type_normalized == "loan").sum()), "spells": len(u.spells),
           "spells_internal": int(u.spells.loan_is_internal.sum()),
           "spells_placeholder": int(u.spells.loan_involves_placeholder.sum()),
           "loans": len(L), "loan_players": nplayers(L), "return_rows": int((r.transfer_type_normalized == "loan_return").sum()),
           "returns_matched": len(closed), "returns_matched_roles": ints(r.loc[closed.index, "row_role"].value_counts()),
           "returns_closing_real_loans": len(to_real),
           "returns_closing_real_loans_roles": ints(r.loc[to_real.index, "row_role"].value_counts()),
           "returns_unmatched": int(ret.loan_idx.isna().sum()),
           "returns_unmatched_roles": ints(r.loc[ret[ret.loan_idx.isna()].index, "row_role"].value_counts()),
           "return_match_reason": ints(ret.match_reason.value_counts()),
           "loan_match_reason": ints(L.match_reason.value_counts())}
    assert out["returns_closing_real_loans"] == int(L.has_matched_return.sum())
    return out


def loan_example(u) -> dict:
    """A loan with its outbound and return rows, by a fixed rule: a plain loan from the 2022/23 season that ended in a
    recorded End-of-loan row on 30 June 2023 with no follow-on, lowest loan event id."""
    L = u.loans
    c = L[L.loan_season.eq("2022/23") & L.has_matched_return & L.economic_ending.eq("ordinary_end_of_loan")
          & L.terminal_date.eq(pd.Timestamp("2023-06-30"))].sort_values("loan_event_id")
    sp = c.iloc[0]
    r = u.rows.set_index("event_id")
    cols = ["transfer_date", "transfer_season", "from_club_name", "to_club_name", "fee_display_raw",
            "transfer_type_raw", "transfer_type_normalized", "row_role"]
    rows = [{"event_id": e, **{k: str(r.loc[e, k]) for k in cols}} for e in (sp.loan_event_id, sp.terminal_event_id)]
    return {"player": sp.player_name, "player_id": int(sp.player_id), "rows": rows,
            "episode": {k: str(int(sp[k]) if k == "duration_days" else sp[k])
                        for k in ("loan_event_id", "terminal_event_id", "match_reason", "duration_days",
                                  "raw_label_class", "economic_ending")}}


# ---------------------------------------------------------------------------
# Part 4: loan endings
# ---------------------------------------------------------------------------

def loan_facts(u, T: pd.DataFrame, nm: pd.DataFrame) -> dict:
    L = u.loans
    assert lte.check_follow_on(L, nm) == 0, "windowed next move differs from the production follow-on"
    assert (lte.reclassify(T).values == T.economic_ending.values).all(), "reclassify does not reproduce production"
    ret = T.ending_is_return.to_numpy(bool)
    gap = T.next_move_gap_days
    has = T.ending_is_return & gap.le(ef.FOLLOW_ON_WINDOW_DAYS) & ~T.next_move_involves_placeholder
    typ = T.next_move_type.fillna("")
    perm = typ.isin(lte.PERMANENT_TYPES)
    to_b = T.next_move_to_club_id.eq(T.borrower_club_id)
    sale = typ.isin(lte.SALE_TYPES)
    f1 = has & ~(perm & to_b)
    f2 = f1 & ~(perm & sale)
    f3 = f2 & gap.le(ef.FOLLOW_ON_IMMEDIATE_DAYS)
    f4 = f3 & T.return_boundary_days.gt(ef.SEASON_END_TOLERANCE_DAYS)
    et = T.economic_ending.eq("early_termination")
    assert (f4 == et).all()
    jun = T.ending_date.map(lambda d: ef._days_from_season_end(d) if pd.notna(d) else np.nan)
    et_jun = lte.reclassify(T.assign(return_boundary_days=jun.where(T.ending_is_return)))
    dec_only = f3 & ~T.return_boundary_days.gt(ef.SEASON_END_TOLERANCE_DAYS) & jun.gt(ef.SEASON_END_TOLERANCE_DAYS)
    assert int(et_jun.eq("early_termination").sum()) == int(et.sum() + dec_only.sum())
    return {
        "loans": len(L), "ending_observed": int(T.ending_observed.sum()), "open": int((~T.ending_observed).sum()),
        "ended_by_return": int(ret.sum()), "ended_by_other_movement": int((T.ending_observed & ~T.ending_is_return).sum()),
        "terminal_type": ints(L.terminal_type.fillna("(none: open)").value_counts()),
        "scheduled_endings": int((T.ending_observed & T.ending_scheduled).sum()),
        "scheduled_returns": int((T.ending_scheduled & T.ending_is_return).sum()),
        "realised": int(T.ending_realised.sum()), "outbound_scheduled": int(T.loan_start_scheduled.sum()),
        "economic_ending": ints(L.economic_ending.value_counts()),
        "raw_label_class": ints(L.raw_label_class.value_counts()),
        "follow_on_present": int(L.follow_on_event_id.notna().sum()),
        "next_move_present": int(T.next_move_event_id.notna().sum()),
        "funnel": {"loans ended by a return row": int(ret.sum()),
                   "with a follow-on (next deal within 60 days, not a release)": int(has.sum()),
                   "follow-on is not a permanent move to the borrower": int(f1.sum()),
                   "follow-on is not a paid or undisclosed-fee move to a third club": int(f2.sum()),
                   "follow-on within 21 days of the return": int(f3.sum()),
                   "return more than 10 days from 30 June and 31 December": int(f4.sum())},
        "early_termination": int(et.sum()),
        "early_termination_if_30_june_only": int(et_jun.eq("early_termination").sum()),
        "only_far_from_30_june_not_31_dec": int(dec_only.sum()),
        "early_termination_terminal_months": ints(T.loc[et, "ending_date"].dt.month.value_counts().sort_index()),
        "early_termination_realised": int((et & T.ending_realised).sum()),
    }


def et_example(u, T: pd.DataFrame) -> dict:
    """Rule: early_termination loans whose loan season is 2023/24 and whose ending is realised; lowest loan event id."""
    c = T[T.economic_ending.eq("early_termination") & T.ending_realised & T.loan_season.eq("2023/24")]
    t = c.sort_values("loan_event_id").iloc[0]
    L = u.loans.set_index("loan_event_id").loc[t.loan_event_id]
    r = u.rows.set_index("event_id")
    ids = [t.loan_event_id, t.ending_event_id, L.follow_on_event_id]
    rows = [{"role": role, "event_id": e, "transfer_date": str(r.loc[e, "transfer_date"]),
             "from": r.loc[e, "from_club_name"], "to": r.loc[e, "to_club_name"], "fee_display_raw": r.loc[e, "fee_display_raw"],
             "transfer_type_normalized": r.loc[e, "transfer_type_normalized"]}
            for role, e in zip(("outbound loan", "End-of-loan row (terminal)", "follow-on"), ids)]
    jun = ef._days_from_season_end(t.ending_date)
    return {"player": t.player_name, "player_id": int(t.player_id), "rows": rows,
            "duration_days": int(t.loan_duration_days), "return_boundary_days": int(t.return_boundary_days),
            "days_from_30_june": int(jun), "follow_on_gap_days": int(L.follow_on_gap_days),
            "follow_on_type": L.follow_on_type, "economic_ending": L.economic_ending, "ending_detail": L.ending_detail}


# ---------------------------------------------------------------------------
# Part 5: calendar
# ---------------------------------------------------------------------------

def calendar_facts(R1: pd.DataFrame, R2: pd.DataFrame, N2: dict, T: pd.DataFrame, G2: pd.DataFrame) -> dict:
    out = {"params": lcv2.PARAMS, "covid_window": [str(lc.COVID_START.date()), str(lc.COVID_END.date())]}
    for k, R in (("v1", R1), ("v2", R2)):
        s = lcv2.state(R)
        mapped = R.lender_country.notna() & R.borrower_country.notna()
        out[k] = {"realised": len(R), "mapped": int(mapped.sum()), "unmapped": int((~mapped).sum()),
                  "domestic": int(R.context.eq("domestic").sum()), "cross_border": int(R.context.eq("cross-border").sum()),
                  "classifiable": int(s.isin(lcv2.STATES[:2]).sum()), "normal": int(s.eq(lcv2.STATES[0]).sum()),
                  "off": int(s.eq(lcv2.STATES[1]).sum()), "country_unknown": int(s.eq(lcv2.STATES[2]).sum()),
                  "below_threshold": int(s.eq(lcv2.STATES[3]).sum()),
                  "classes": ints(R.calendar_class.value_counts()),
                  "covid": int(R.covid.sum()), "non_covid": int((~R.covid).sum()),
                  "domestic_calendars": int(R.loc[s.isin(lcv2.STATES[:2]).values
                                                  & R.calendar_key.str.startswith("domestic"), "calendar_key"].nunique())}
    cal = N2["calendars"]
    out["calendars_v2"] = {k: {"N": int(v["N"]), "centers": [lc.mmdd_of(c) for c in v["centers"]],
                               "coverage": float(v["coverage"]), "stopped": v["stopped"]} for k, v in cal.items()}
    train = R2[~R2.covid & R2.calendar_key.isin(cal.keys())]
    assert len(train) == sum(int(v["N"]) for v in cal.values())
    out["training_endings"] = len(train)
    bt = R2[lcv2.state(R2).eq(lcv2.STATES[3])]
    out["below_threshold_countries"] = int(bt.lender_country.nunique())
    out["below_threshold_top"] = ints(bt.lender_country.value_counts().head(12))
    out["below_threshold_all_domestic"] = bool(bt.context.eq("domestic").all())
    dn = R2[R2.context.eq("domestic") & ~R2.covid].lender_country.value_counts()
    out["largest_country_without_calendar"] = {"country": dn[dn < lcv2.PARAMS["N_MIN"]].index[0],
                                              "n": int(dn[dn < lcv2.PARAMS["N_MIN"]].iloc[0])}
    unk = R2[lcv2.state(R2).eq(lcv2.STATES[2]).values]
    sides = pd.DataFrame({"name": pd.concat([unk.lender_club, unk.borrower_club]).values,
                          "id": pd.concat([unk.lender_club_id, unk.borrower_club_id]).astype(int).values,
                          "country": pd.concat([unk.lender_country, unk.borrower_country]).values})
    sides = sides[sides.country.isna()]
    out["country_unknown_clubs"] = [f"{' / '.join(sorted(set(g.name)))} ({i}, {len(g)} ending{'s' if len(g) > 1 else ''})"
                                    for i, g in sides.groupby("id")]
    saved1 = pd.read_csv(lc.CLASSIFIED_CSV, usecols=["loan_event_id", "calendar_class"]).set_index("loan_event_id")
    saved2 = pd.read_csv(lcv2.CLASSIFIED_CSV, usecols=["loan_event_id", "calendar_class"]).set_index("loan_event_id")
    out["reproduces_saved_v1"] = bool((saved1.calendar_class.reindex(R1.loan_event_id).values == R1.calendar_class.values).all())
    out["reproduces_saved_v2"] = bool((saved2.calendar_class.reindex(R2.loan_event_id).values == R2.calendar_class.values).all())
    assert out["reproduces_saved_v1"] and out["reproduces_saved_v2"]
    # early_termination label by calendar state (the table in the calendar summary)
    C = R2.assign(state=lcv2.state(R2).values)
    out["et_by_state"] = {s: {"N": int((C.state == s).sum()),
                              "early_termination": int(((C.state == s) & C.economic_ending.eq("early_termination")).sum())}
                          for s in lcv2.STATES[:2]}
    nrm_et = C[(C.state == lcv2.STATES[0]) & C.economic_ending.eq("early_termination")]
    out["et_normal_matched_dates"] = ints(nrm_et.matched_standard_mmdd.value_counts().head(6))
    # geography sources
    st = G2[G2.in_stage1c.astype(str).eq("True")]
    out["geo_v2_status_stage1c_clubs"] = ints(st.status.value_counts())
    out["geo_v2_source_stage1c_clubs"] = ints(st.source.replace("", "(none)").value_counts())
    return out


def code_check() -> dict:
    hits = {}
    for m in DOWNSTREAM + ("injury_method_example",):
        txt = module_text(m)
        hits[m] = [t for t in CODE_TOKENS if t in txt]
    return hits


# ---------------------------------------------------------------------------
# Parts 6-7: playing time
# ---------------------------------------------------------------------------

def pt_waterfall(E: pd.DataFrame, J: dict) -> dict:
    out = {}
    for k in KINDS:
        g = E[E.kind == k]
        steps = [{"step": "candidate episodes", "n": len(g), "players": nplayers(g)}]
        cur = g
        for reason in ptj.REASONS:
            m = cur.exclusion_reason.eq(reason)
            steps.append({"step": f"excluded: {ptj.REASONS[reason]}", "code": reason, "removed": int(m.sum()),
                          "n": int((~m).sum()), "players": nplayers(cur[~m])})
            cur = cur[~m]
        assert len(cur) == int(g.reliable.sum())
        for s in ("realised", "covered_league_season", "matched", "reliable"):
            assert int(g[s].sum()) == J["waterfall"][k][s], (k, s)
        R = g[g.reliable]
        rev_steps = []
        c = R
        for lab, f in (("loan did not end with a return to the lender ('other movement' or 'no ending recorded')",
                        lambda x: x.kind.eq("loan") & x.loan_ending.ne("return to lender")),
                       ("repeat spell: an earlier loan or permanent move of the player into the same club B",
                        lambda x: x.prior_spell_at_b)):
            m = f(c)
            rev_steps.append({"step": f"excluded: {lab}", "removed": int(m.sum()), "n": int((~m).sum()),
                              "players": nplayers(c[~m])})
            c = c[~m]
        old_steps = []
        o = R
        for lab, f in (("fewer than two-thirds of B's league games left at arrival", lambda x: ~x.arrived_full_season),
                       ("repeat spell at B", lambda x: x.prior_spell_at_b),
                       ("loan did not end with a return to the lender",
                        lambda x: x.kind.eq("loan") & x.loan_ending.ne("return to lender"))):
            m = f(o)
            old_steps.append({"step": f"excluded: {lab}", "removed": int(m.sum()), "n": int((~m).sum())})
            o = o[~m]
        assert len(o) == int(g.primary_sample.sum())
        out[k] = {"steps": steps, "reliable": len(R), "reliable_players": nplayers(R), "revised_steps": rev_steps,
                  "revised": len(c), "revised_players": nplayers(c), "old_steps": old_steps, "old": len(o),
                  "reliable_true_zero": int(R.true_zero.sum()), "revised_true_zero": int(c.true_zero.sum()),
                  "revised_mean_share": float(c.minutes_share.mean()),
                  "revised_departed_in_window": int(c.departed_in_window.sum()),
                  "revised_window_games_median": float(c.available_games.median()),
                  "two_thirds_jan_feb": int((~R.arrived_full_season & R.move_date.dt.month.isin([1, 2])).sum())}
    return out


def game_table(con, b: int, season: int, ws, we, pid: int) -> pd.DataFrame:
    g = con.sql(f"""select g.game_id, g.date, g.home_club_id, g.home_club_name, g.away_club_name,
                           (select count(*) from appearances a where cast(a.game_id as varchar) = g.game_id
                                                              and a.player_club_id = {b}) b_rows,
                           (select max(a.minutes_played) from appearances a where cast(a.game_id as varchar) = g.game_id
                                                              and a.player_id = {pid} and a.player_club_id = {b}) player_minutes
                    from games g where g.competition_type = 'domestic_league' and cast(g.season as int) = {season}
                    and {b} in (g.home_club_id, g.away_club_id)
                    and g.date between '{pd.Timestamp(ws).date()}' and '{pd.Timestamp(we).date()}' order by g.date""").df()
    g["opponent"] = [a if h == b else hn for h, hn, a in zip(g.home_club_id, g.home_club_name, g.away_club_name)]
    g["complete"] = g.b_rows >= ptj.COMPLETE_ROWS
    return g


def pt_example(E: pd.DataFrame, con, mask: pd.Series, rule: str) -> dict:
    e = E[mask].sort_values(["arrival_season", "event_id"], ascending=[False, True]).iloc[0]
    g = game_table(con, int(e.b_club_id), int(e.arrival_season), e.window_start, e.window_end, int(e.player_id))
    assert len(g) == e.available_games and int(g.complete.sum()) == e.complete_games
    return {"rule": rule, "player": e.player_name, "player_id": int(e.player_id), "event_id": e.event_id,
            "kind": e.kind, "a_club": e.a_club, "b_club": e.b_club, "b_club_id": int(e.b_club_id),
            "move_date": str(e.move_date.date()), "league": e.receiving_league, "arrival_season": int(e.arrival_season),
            "window_start": str(e.window_start.date()), "window_end": str(e.window_end.date()),
            "departure_date": None if pd.isna(e.departure_date) else str(e.departure_date.date()),
            "season_games": int(e.season_games), "available_games": int(e.available_games),
            "complete_games": int(e.complete_games), "player_rows": int(g.player_minutes.notna().sum()),
            "minutes": None if pd.isna(e.minutes) else int(e.minutes),
            "minutes_share": None if pd.isna(e.minutes_share) else float(e.minutes_share),
            "exclusion_reason": e.exclusion_reason or "none (reliable)",
            "games": [{"date": str(r.date.date()), "opponent": r.opponent, "b_rows": int(r.b_rows), "complete": bool(r.complete),
                       "player_minutes": None if pd.isna(r.player_minutes) else int(r.player_minutes)} for r in g.itertuples()]}


# ---------------------------------------------------------------------------
# Parts 8-9: market value, regression samples
# ---------------------------------------------------------------------------

def missing_breakdown(F: pd.DataFrame, need: list[str]) -> dict:
    checks = [("age missing (no date of birth)", F.age.isna()), ("position missing", F.position_clean.isna()),
              ("no player valuation strictly before the move", F.player_value.isna()),
              ("player value not positive", F.player_value.le(0)),
              (f"squad value missing (fewer than {reg.MIN_SQUAD_PLAYERS} valued players)", F.squad_value.isna()),
              ("squad value not positive", F.squad_value.le(0))]
    seq, rem = [], pd.Series(True, index=F.index)
    for lab, m in checks:
        n = int((m & rem).sum())
        seq.append({"reason": lab, "any": int(m.sum()), "sequential": n,
                    **{f"{k}_sequential": int((m & rem & F.kind.eq(k)).sum()) for k in KINDS}})
        rem &= ~m
    C = F.dropna(subset=need)
    assert int(rem.sum()) == len(C)
    return {"N": len(F), "complete": len(C), "dropped": len(F) - len(C), "steps": seq,
            "no_pv_only_post_move": int((F.player_value.isna() & F.next_vdate.notna()).sum()),
            "no_pv_none_at_all": int((F.player_value.isna() & F.next_vdate.isna()).sum())}


def m3(C: pd.DataFrame, y: str = "share") -> dict:
    return reg.coef(reg.fit(C, reg.M3, fe="league_season", cats=["position_clean"], y=y))


def raw_gap(F: pd.DataFrame, y: str) -> float:
    return float(F.loc[F.kind == "loan", y].mean() - F.loc[F.kind == "permanent", y].mean())


def regression_samples(E: pd.DataFrame, B: pd.DataFrame, Cinj: pd.DataFrame) -> tuple[dict, dict]:
    saved_r = json.loads(reg.NUMBERS_JSON.read_text())
    saved_a = json.loads(pta.NUMBERS_JSON.read_text())
    saved_i = json.loads(ia.NUMBERS_JSON.read_text())
    P = B[B.event_id.isin(E.loc[E.primary_sample, "event_id"])].copy()
    P["age_c"] = P.age - P.age.mean()
    P["age_c2"] = P.age_c ** 2
    Co = P.dropna(subset=NEED_OLD).copy()
    Ca = B.dropna(subset=pta.NEED).copy()
    Ca["age_c"] = Ca.age - Ca.age.mean()
    Ca["age_c2"] = Ca.age_c ** 2
    Ca["age_years"] = np.floor(Ca.age)
    Cg = pta.add_groups(Ca, pta.CUTOFF)
    old, rev, inj = m3(Co), m3(Ca), m3(Cinj, "current_share")
    injf = m3(Cinj, "fit_adjusted_share")
    age = pta.age_model(Cg)
    tol = 1e-9
    assert abs(old["coef"] - saved_r["models"]["M3"]["coef"]) < tol and old["N"] == saved_r["models"]["M3"]["N"]
    sv = saved_a["composition"]["preferred model (age, age²) on the revised sample"]
    assert abs(rev["coef"] - sv["coef"]) < tol and rev["N"] == sv["N"]
    sa = saved_a["cuts"][str(pta.CUTOFF)]["adjusted"]
    for t in ("young", "older", "interaction"):
        assert abs(age[t]["coef"] - sa[t]["coef"]) < tol
    assert abs(inj["coef"] - saved_i["preferred"]["current"]["coef"]) < tol
    assert abs(injf["coef"] - saved_i["preferred"]["fit"]["coef"]) < tol
    out = {
        "old": {"sample": ints(P.kind.value_counts()), "model": ints(Co.kind.value_counts()),
                "raw_gap_sample": raw_gap(P, "share"), "raw_gap_model": raw_gap(Co, "share"), "m3": old,
                "age_center": float(P.age.mean())},
        "revised": {"sample": ints(B.kind.value_counts()), "model": ints(Ca.kind.value_counts()),
                    "raw_gap_sample": raw_gap(B, "share"), "raw_gap_model": raw_gap(Ca, "share"), "m3": rev,
                    "age_center": float(Ca.age.mean()),
                    "age": {t: {k: float(age[t][k]) for k in ("coef", "lo", "hi")} for t in ("young", "older", "interaction")},
                    "age_cells": {f"{k}, {'older' if o == 1 else 'young'}": int(n)
                                  for (k, o), n in Cg.groupby(["kind", "older"]).size().items()}},
        "injury": {"model": ints(Cinj.kind.value_counts()), "raw_gap_model_current": raw_gap(Cinj, "current_share"),
                   "raw_gap_model_fit": raw_gap(Cinj, "fit_adjusted_share"), "m3_current": inj, "m3_fit": injf,
                   "age_center": float(Cinj.age.mean())},
        "reproduces_saved": True}
    return out, {"old": set(Co.event_id), "age": set(Ca.event_id), "injury": set(Cinj.event_id)}


# ---------------------------------------------------------------------------
# Calendar-first sequence (loans gated on their calendar status before playing time)
# ---------------------------------------------------------------------------

CF_STATUS_OK = ("normal", "off_calendar")
CF_VARIANTS = {
    "current": ("no loan-calendar gate (Week 5/6 sequence)", ()),
    "realised": ("loan ending realised (any calendar status)", ("ending_not_realised",)),
    "classifiable": ("classifiable realised ending: calendar-normal or off-calendar (recommended)",
                     ("ending_not_realised", "unclassified")),
    "normal_only": ("calendar-normal endings only", ("ending_not_realised", "unclassified", "off_calendar")),
}
CF_GATE_LABEL = {"ending_not_realised": "loan has no realised ending (still open, or ending flagged as a future move)",
                 "unclassified": "ending not classifiable (club country unknown, or country below the history threshold)",
                 "off_calendar": "ending is off-calendar"}
COVARIATE_CHECKS = [("age missing (no date of birth)", lambda F: F.age_exact.isna()),
                    ("position missing", lambda F: F.position_clean.isna()),
                    ("no player valuation strictly before the move", lambda F: F.player_value.isna()),
                    ("player value not positive", lambda F: F.player_value.le(0)),
                    (f"squad value missing (fewer than {reg.MIN_SQUAD_PLAYERS} valued players)",
                     lambda F: F.squad_value.isna()),
                    ("squad value not positive", lambda F: F.squad_value.le(0))]


def loan_calendar_status(M: pd.DataFrame) -> pd.Series:
    st = M.calendar_state.map({"calendar-normal": "normal", "off-calendar": "off_calendar"})
    st = st.where(st.notna() | M.calendar_state.isna(), "unclassified")
    st = st.where(st.notna() | ~M.kind.eq("loan"), "ending_not_realised")
    return st.where(M.kind.eq("loan"), "not_applicable")


def _seq(F: pd.DataFrame, checks: list) -> tuple[list, pd.Series]:
    """Sequential removals: each row is counted under the first check it fails."""
    out, rem = [], pd.Series(True, index=F.index)
    for lab, m in checks:
        n = int((m & rem).sum())
        out.append({"reason": lab, "removed": n, "n": int((rem & ~m).sum())})
        rem &= ~m
    return out, rem


def _branch(F: pd.DataFrame) -> dict:
    """Playing-time → revised → main regression → injury, for one kind, after any loan-calendar gate."""
    pt = [(ptj.REASONS[r], F.pt_exclusion_reason.eq(r)) for r in ptj.REASONS]
    pt_steps, rel = _seq(F, pt)
    assert rel.equals(F.pt_reliable)
    rv = [("loan did not end with a return to the lender", F.kind.eq("loan") & F.pt_loan_ending.ne("return to lender")),
          ("repeat spell at the same club (first spell kept)", F.prior_spell_at_b.astype(bool))]
    R = F[rel]
    rev_steps, rev = _seq(R, [(a, b[rel]) for a, b in rv])
    assert rev.equals(R.in_revised_sample)
    V = R[rev]
    cov_steps, main = _seq(V, [(a, f(V)) for a, f in COVARIATE_CHECKS])
    assert main.equals(V.in_age_model_sample)
    flags = [r for r in V.injury_data_quality_flag.dropna().unique() if r != "usable"]
    order = ["player not in the external datalake", "ambiguous player id", "window reaches the 2025-09-15 snapshot",
             "injury history truncated before the window", "open injury (no end date) could cover the window",
             "player has a medical row without a start date", "no fit games left"]
    assert set(flags) <= set(order), set(flags) - set(order)
    inj_checks = [(r, V.injury_data_quality_flag.eq(r)) for r in order]
    inj_steps, usable = _seq(V, inj_checks)
    assert usable.equals(V.in_injury_usable)
    U = V[usable]
    icov_steps, imod = _seq(U, [(a, f(U)) for a, f in COVARIATE_CHECKS])
    assert imod.equals(U.in_injury_model_sample)
    return {"start": len(F), "pt_steps": pt_steps, "reliable": int(rel.sum()), "revised_steps": rev_steps,
            "revised": int(rev.sum()), "control_steps": cov_steps, "main": int(main.sum()), "injury_steps": inj_steps,
            "injury_usable": int(usable.sum()), "injury_control_steps": icov_steps, "injury_model": int(imod.sum()),
            "ids": {"revised": set(V.event_id[rev]), "main": set(V.event_id[main]), "usable": set(U.event_id),
                    "injury_model": set(U.event_id[imod])}}


def _estimates(B: pd.DataFrame, Cinj: pd.DataFrame, ids: dict) -> dict:
    Ca = B[B.event_id.isin(ids["main"])].dropna(subset=pta.NEED).copy()
    assert len(Ca) == len(ids["main"])
    Ca["age_c"] = Ca.age - Ca.age.mean()
    Ca["age_c2"] = Ca.age_c ** 2
    Ca["age_years"] = np.floor(Ca.age)
    Cg = pta.add_groups(Ca, pta.CUTOFF)
    Ci = Cinj[Cinj.event_id.isin(ids["injury_model"])].copy()
    assert len(Ci) == len(ids["injury_model"])
    Ci["age_c"] = Ci.age - Ci.age.mean()
    Ci["age_c2"] = Ci.age_c ** 2
    Bv = B[B.event_id.isin(ids["revised"])]
    age = pta.age_model(Cg)
    return {"raw_gap_revised": raw_gap(Bv, "share"), "raw_gap_main": raw_gap(Ca, "share"), "m3": m3(Ca),
            "age": {t: {k: float(age[t][k]) for k in ("coef", "lo", "hi")} for t in ("young", "older", "interaction")},
            "injury_raw_gap_current": raw_gap(Ci, "current_share"), "injury_raw_gap_fit": raw_gap(Ci, "fit_adjusted_share"),
            "injury_m3_current": m3(Ci, "current_share"), "injury_m3_fit": m3(Ci, "fit_adjusted_share")}


def calendar_first(M: pd.DataFrame, B: pd.DataFrame, Cinj: pd.DataFrame, reg_saved: dict) -> dict:
    L, P = M[M.kind.eq("loan")], M[M.kind.eq("permanent")]
    st = L.loan_calendar_status
    perm = _branch(P)
    out = {"loans": len(L), "permanent": len(P), "unknown_type": int(M.kind.isna().sum()),
           "loan_status": ints(st.value_counts()), "permanent_branch": {k: v for k, v in perm.items() if k != "ids"},
           "unclassified_not_covered": int((st.eq("unclassified") & L.pt_exclusion_reason.eq("not_covered")).sum()),
           "variants": {}}
    for key, (label, drop) in CF_VARIANTS.items():
        gate, rem = [], pd.Series(True, index=L.index)
        for s in ("ending_not_realised", "unclassified", "off_calendar"):
            if s in drop:
                m = st.eq(s)
                gate.append({"reason": CF_GATE_LABEL[s], "removed": int((m & rem).sum()), "n": int((rem & ~m).sum())})
                rem &= ~m
        lb = _branch(L[rem])
        ids = {k: lb["ids"][k] | perm["ids"][k] for k in lb["ids"]}
        est = _estimates(B, Cinj, ids)
        out["variants"][key] = {"label": label, "gate_steps": gate, "after_gate": int(rem.sum()),
                                "after_gate_by_status": ints(st[rem].value_counts()),
                                "loan_branch": {k: v for k, v in lb.items() if k != "ids"},
                                "totals": {s: {"loan": lb[s], "permanent": perm[s], "total": lb[s] + perm[s]}
                                           for s in ("reliable", "revised", "main", "injury_usable", "injury_model")},
                                "revised_loans_by_status": ints(L[L.event_id.isin(lb["ids"]["revised"])]
                                                                .loan_calendar_status.value_counts()),
                                "estimates": est}
        if key == "classifiable":
            for c, s in (("in_cf_revised_sample", "revised"), ("in_cf_main_model_sample", "main"),
                         ("in_cf_injury_usable", "usable"), ("in_cf_injury_model_sample", "injury_model")):
                assert set(M.event_id[M[c]]) == ids[s], c
    cur = out["variants"]["current"]
    assert cur["totals"]["revised"]["total"] == int(M.in_revised_sample.sum())
    assert cur["totals"]["main"]["total"] == reg_saved["revised"]["m3"]["N"]
    assert abs(cur["estimates"]["m3"]["coef"] - reg_saved["revised"]["m3"]["coef"]) < 1e-9
    assert abs(cur["estimates"]["injury_m3_fit"]["coef"] - reg_saved["injury"]["m3_fit"]["coef"]) < 1e-9
    assert cur["totals"]["injury_model"]["total"] == reg_saved["injury"]["m3_current"]["N"]
    return out


# ---------------------------------------------------------------------------
# Academy/reserve promotions (sensitivity on the calendar-first sample; the main sample is not replaced)
# ---------------------------------------------------------------------------

ACADEMY_VARIANTS = {
    "A": ("current calendar-first sample, unchanged", None),
    "B": ("excluding high-confidence same-organisation moves (same_organisation_move_v2)", "same_organisation_move_v2"),
    "C": ("excluding every academy-promotion candidate, any confidence (academy_promotion_candidate)",
          "academy_promotion_candidate"),
    "D": ("excluding every arrival from any youth/reserve side, same organisation or not (extreme bound)", "youth_origin"),
}


def academy_sensitivity(M: pd.DataFrame, B: pd.DataFrame, Cinj: pd.DataFrame, CF: dict) -> dict:
    L = M[M.kind.eq("loan") & M.loan_calendar_status.isin(CF_STATUS_OK)]
    P = M[M.kind.eq("permanent")]
    youth = co.youth_side_ids(co.load_map())
    flags = {"same_organisation_move_v2": M.same_organisation_move_v2,
             "academy_promotion_candidate": M.academy_promotion_candidate,
             "youth_origin": M.from_club_id.astype("Int64").isin(youth).astype(bool)}
    out, base = {}, None
    for key, (label, flag) in ACADEMY_VARIANTS.items():
        drop = flags[flag] if flag else pd.Series(False, index=M.index)
        lb, pb = _branch(L[~drop[L.index]]), _branch(P[~drop[P.index]])
        ids = {k: lb["ids"][k] | pb["ids"][k] for k in lb["ids"]}
        est = _estimates(B, Cinj, ids)
        Ca = B[B.event_id.isin(ids["main"])].dropna(subset=pta.NEED).copy()
        Ca["age_c"] = Ca.age - Ca.age.mean()
        Ca["age_c2"] = Ca.age_c ** 2
        ladder = {nm: reg.coef(reg.fit(Ca, x, fe="league_season", cats=["position_clean"]))
                  for nm, x in (("M1", reg.M1), ("M2", reg.M2), ("M3", reg.M3))}
        assert abs(ladder["M3"]["coef"] - est["m3"]["coef"]) < 1e-12
        tot = {s: {"loan": lb[s], "permanent": pb[s], "total": lb[s] + pb[s]}
               for s in ("reliable", "revised", "main", "injury_usable", "injury_model")}
        out[key] = {"label": label, "flagged": {k: int((drop & M.kind.eq(k)).sum()) for k in KINDS},
                    "totals": tot, "estimates": est, "ladder": ladder,
                    "flagged_in_revised": {k: int((drop & M.in_cf_revised_sample & M.kind.eq(k)).sum()) for k in KINDS}}
        if base is None:
            base = tot
        out[key]["removed"] = {s: {k: base[s][k] - tot[s][k] for k in ("loan", "permanent", "total")}
                               for s in ("revised", "main", "injury_model")}
    cf = CF["variants"]["classifiable"]
    assert out["A"]["totals"] == cf["totals"]
    assert abs(out["A"]["estimates"]["m3"]["coef"] - cf["estimates"]["m3"]["coef"]) < 1e-12
    # removing rows before the playing-time steps must remove exactly the flagged rows of each later sample
    for key in ("B", "C"):
        f = M[ACADEMY_VARIANTS[key][1]]
        for s, col in (("revised", "in_cf_revised_sample"), ("main", "in_cf_main_model_sample"),
                       ("injury_model", "in_cf_injury_model_sample")):
            assert out[key]["removed"][s]["total"] == int((f & M[col]).sum()), (key, s)
    R = M[M.in_cf_revised_sample & M.kind.eq("permanent")]
    grp = {"high-confidence same organisation": R.same_organisation_move_v2,
           "medium/low candidate only": R.academy_promotion_candidate & ~R.same_organisation_move_v2,
           "other permanent arrivals": ~R.academy_promotion_candidate}
    desc = {k: {"n": int(m.sum()), "mean_share": float(R.loc[m, "minutes_share"].mean()),
                "true_zero": float(R.loc[m, "true_zero"].astype(bool).mean()),
                "no_fee_shown": float(R.loc[m, "transfer_type_raw"].eq("no_fee_shown").mean()),
                "mean_age": float(R.loc[m, "age_exact"].mean()),
                "mean_log_relative_value_main": float(R.loc[m & R.in_cf_main_model_sample, "log_relative_value"].mean())}
            for k, m in grp.items()}
    status = {s: {st: ints(g.kind.value_counts()) for st, g in M.loc[M[c]].groupby("same_org_confidence")}
              for s, c in (("revised", "in_cf_revised_sample"), ("main", "in_cf_main_model_sample"),
                           ("injury_model", "in_cf_injury_model_sample"))}
    rough = co.rough_screen(M)
    agree = {s: {"rough_screen": int((rough & M[c]).sum()),
                 "rough_and_v2": int((rough & M[c] & M.academy_promotion_candidate).sum()),
                 "rough_only": int((rough & M[c] & ~M.academy_promotion_candidate).sum()),
                 "v2_only": int((~rough & M[c] & M.academy_promotion_candidate).sum())}
             for s, c in (("revised", "in_cf_revised_sample"), ("main", "in_cf_main_model_sample"))}
    labels = {k: int(v) for k, v in M.loc[M.academy_promotion_candidate, "transfer_type_raw"].value_counts().items()}
    return {"variants": out, "permanent_groups": desc, "status_by_sample": status, "rough_vs_v2": agree,
            "candidate_labels": labels,
            "loans_flagged": int((M.academy_promotion_candidate & M.kind.eq("loan")).sum()),
            "no_fee_shown_permanent_unflagged": int((M.kind.eq("permanent") & M.transfer_type_raw.eq("no_fee_shown")
                                                     & ~M.academy_promotion_candidate).sum())}


def market_value_facts(B: pd.DataFrame) -> dict:
    out = {}
    for k in KINDS + ("all",):
        g = B if k == "all" else B[B.kind == k]
        out[k] = {"N": len(g), "player_value": int(g.player_value.notna().sum()),
                  "only_post_move": int((g.player_value.isna() & g.next_vdate.notna()).sum()),
                  "no_valuation_at_all": int((g.player_value.isna() & g.next_vdate.isna()).sum()),
                  "gap_median": float(g.player_value_gap_days.median()), "gap_p90": float(g.player_value_gap_days.quantile(.9)),
                  "stale": int(g.stale_player_value.sum()), "squad_value": int(g.squad_value.notna().sum()),
                  "squad_players_median": float(g.squad_players.median()), "focal_in_squad": int(g.focal_in_squad.sum()),
                  "squad_gap_median": float(g.squad_value_gap_days.median())}
    s = B[B.player_value.notna() & B.market_value_eur.notna()]
    out["stage1c_market_value_vs_player_value"] = {"compared": len(s), "equal": int((s.market_value_eur == s.player_value).sum())}
    out["same_day_valuations_excluded"] = int(B.next_vdate.eq(B.move_date).sum())
    return out


# ---------------------------------------------------------------------------
# Parts 10-12: injury
# ---------------------------------------------------------------------------

def injury_facts(Nn: dict, tb: dict) -> dict:
    E, VAL, INJ = tb["E"], tb["VAL"], tb["INJ"]
    saved = pd.read_csv(ia.EPISODES_CSV).set_index("event_id")
    s = saved.loc[E.event_id]
    assert np.allclose(s.current_share.values, E.current_share.values)
    assert np.allclose(s.fit_adjusted_share.fillna(-1).values, E.fit_adjusted_share.fillna(-1).values)
    assert (s.data_quality_flag.values == E.data_quality_flag.values).all()
    sv = json.loads(ia.NUMBERS_JSON.read_text())
    for k in ("usable", "model_sample", "revised", "join_players"):
        assert sv[k] == Nn[k], k
    raw = pd.read_csv(ia.INJ_FILE, dtype=str, keep_default_na=False)
    per = raw.groupby("player_id").size()
    order = ["player not in the external datalake", "ambiguous player id", "window reaches the 2025-09-15 snapshot",
             "injury history truncated before the window", "open injury (no end date) could cover the window",
             "player has a medical row without a start date", "no fit games left"]
    use = []
    for k in KINDS:
        g = E[E.kind == k]
        cur, st = len(g), []
        for r in order:
            n = int(g.data_quality_flag.eq(r).sum())
            st.append({"reason": r, "removed": n, "n": cur - n})
            cur -= n
        assert cur == int(g.usable.sum())
        use.append((k, st))
    cats = INJ.groupby("category").agg(rows=("source_row", "size"), primary=("counts_primary", "first"),
                                       strict=("counts_strict", "first"))
    V = VAL
    cons = {c: {"events": int((V.after_event == c).sum()), **{f"match_{d}d": int((V.after_event.eq(c) & V[f"match_{d}d"]).sum())
                                                              for d in (0, 1, 3, 7)}} for c in ia.CONSEQ}
    per_m2 = {p: {"events": int(len(g)), "match_7d": int(g.match_7d.sum())}
              for p, g in V[V.after_event == ia.CONSEQ[0]].groupby("period")}
    return {
        "raw_rows": len(raw), "raw_columns": list(raw.columns), "raw_players": int(per.size),
        "rows_per_player_max": int(per.max()), "players_at_15": int((per == 15).sum()), "players_at_14": int((per == 14).sum()),
        "players_at_13": int((per == 13).sum()), "normalize": Nn["normalize"], "profiles": Nn["profiles"],
        "categories": {c: {"rows": int(r.rows), "primary": bool(r.primary), "strict": bool(r.strict)} for c, r in cats.iterrows()},
        "labels": int(tb["labels"].shape[0]),
        "status_players": Nn["join_players"],
        "status_episodes": {k: ints(E[E.kind == k].status.value_counts()) for k in KINDS},
        "revised": Nn["revised"], "revised_players": Nn["revised_players"], "usability_steps": dict(use),
        "usable": Nn["usable"], "usable_players": nplayers(E[E.usable]), "model_sample": Nn["model_sample"],
        "appearance_conflicts": Nn["appearance_conflicts"], "id_check": {k: Nn["id_check"][k] for k in ("compared", "dob_equal")},
        "validation": {"events": int(len(V)), "players": int(V.player_id.nunique()),
                       **{f"match_{d}d": int(V[f"match_{d}d"].sum()) for d in (0, 1, 3, 7)},
                       "covered_by_interval": int(V.covered_by_interval.sum()),
                       "events_players_without_rows": int(V.player_status.eq("B").sum()),
                       "by_consequence": cons, "missed_two_by_period": per_m2},
        "usable_delta_zero": {k: int(((E.kind == k) & E.usable & E.confirmed_injury_games.eq(0)).sum()) for k in KINDS},
        "age_group_missing_age_labelled_older": int(E.age.isna().sum()),
    }


def anton_trace(con, Einj: pd.DataFrame, Ept: pd.DataFrame, B: pd.DataFrame, in_model: set) -> dict:
    ep = pd.read_csv(ia.EPISODES_CSV)
    ep = ep[(ep.player_id == ANTON) & ep.usable & ep.kind.eq("permanent")].sort_values("arrival_season").iloc[-1]
    J = pd.read_csv(ime.JOIN_EPISODES, usecols=["event_id", "player_name", "a_club", "b_club", "b_club_id",
                                                "move_date"]).set_index("event_id")
    INJ = pd.read_csv(ia.PROCESSED, parse_dates=["from_date", "end_date"])
    rec = ime.rebuild(ep, J.loc[ep.event_id], INJ, con)
    e = Ept.set_index("event_id").loc[ep.event_id]
    b = B.set_index("event_id").loc[ep.event_id]
    g = rec["games_table"]
    spells = ia.spells(INJ, "primary")
    spells = spells[(spells.player_id == ANTON) & (spells.from_date <= pd.Timestamp(rec["window_end"]))
                    & (spells.end_date >= pd.Timestamp(rec["window_start"]))]
    return {
        "event_id": ep.event_id, "player": rec["player_name"], "player_id": ANTON, "a_club": rec["from_club"],
        "b_club": rec["to_club"], "b_club_id": int(e.b_club_id), "move_date": str(e.move_date.date()),
        "transfer_type_raw": e.transfer_type_raw, "fee_display_raw": e.fee_display_raw,
        "league": rec["league"], "arrival_season": rec["season"], "window_start": rec["window_start"],
        "window_end": rec["window_end"], "departure_date": None if pd.isna(e.departure_date) else str(e.departure_date.date()),
        "season_games": int(e.season_games), "games": rec["games"], "complete_games": int(e.complete_games),
        "appearances": int(e.appearances), "minutes": rec["minutes"], "removed": rec["removed"],
        "conflicts": rec["conflicts"], "fit_games": rec["fit_games"], "current_share": rec["current_share"],
        "fit_share": rec["fit_share"],
        "spell_rows": [{"reason": s.injury_reason_raw, "from": str(s.from_date.date()),
                        "until": "open" if pd.isna(s.end_date) else str(s.end_date.date()),
                        "counts": bool(s.counts_primary), "category": s.category} for s in rec["spell_rows"].itertuples()],
        "primary_spells_in_window": len(spells),
        "spell_games": [{"date": str(x.date.date()), "opponent": x.opponent, "played": bool(x.played),
                         "minutes": None if pd.isna(x.minutes_played) else int(x.minutes_played),
                         "covering_spells": int(x.covering_spells), "removed": bool(x.medical_unavailable)}
                        for x in g[g.spell_covers].itertuples()],
        "status": Einj.set_index("event_id").loc[ep.event_id, "status"],
        "flag": ep.data_quality_flag, "player_value": float(b.player_value), "player_value_date": str(b.player_value_date.date()),
        "squad_value": float(b.squad_value), "squad_players": int(b.squad_players), "relative_value": float(b.relative_value),
        "age": float(b.age), "position": b.position_clean, "remain": float(b.remain), "league_season": b.league_season,
        "in_model": ep.event_id in in_model}


# ---------------------------------------------------------------------------
# Part 13: master table
# ---------------------------------------------------------------------------

def master_table(u, T: pd.DataFrame, R2: pd.DataFrame, G2: pd.DataFrame, E: pd.DataFrame, B: pd.DataFrame,
                 EI: pd.DataFrame, models: dict) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)   # column-by-column assembly; copied at the end
        return _master_table(u, T, R2, G2, E, B, EI, models).copy()


def _master_table(u, T, R2, G2, E, B, EI, models) -> pd.DataFrame:
    ep = u.episodes.reset_index(drop=True)
    assert ep.event_id.is_unique
    M = pd.DataFrame({"event_id": ep.event_id})
    for c in ("transfermarkt_transfer_id", "event_chain_id", "player_id", "player_name", "transfer_date", "transfer_season",
              "season", "episode_type"):
        M[c] = ep[c]
    M["kind"] = ep.episode_type.map(ptj.KINDS)
    for c in ("transfer_type_raw", "transfer_type_normalized", "fee_display_raw", "fee_disclosure_status",
              "permanent_transfer_fee_eur", "loan_fee_eur", "from_club_id", "from_club_name", "to_club_id", "to_club_name",
              "is_youth_or_reserve_side", "is_senior_move", "transfermarkt_future_transfer", "is_research_target"):
        M[c] = ep[c]
    org = co.episode_flags(M, co.load_map())
    for c in co.MASTER_COLUMNS:
        M[c] = org[c]
    loan = ep.episode_type.eq("loan")
    for c in ("match_reason", "has_matched_return", "terminal_event_id", "terminal_date", "terminal_type",
              "terminal_raw_label", "raw_label_class", "terminal_scheduled_future"):
        M[c] = ep[c].where(loan)
    Tk = T.set_index("loan_event_id")
    nc = pd.Series(lc.next_category(T).values, index=T.loan_event_id)
    M["ending_observed"] = M.event_id.map(Tk.ending_observed)
    M["ending_realised"] = M.event_id.map(Tk.ending_realised)
    for c in ("duration_days", "ending_fee_on_return_eur", "return_boundary_days", "follow_on_event_id",
              "follow_on_gap_days", "follow_on_type", "follow_on_to_club", "economic_ending", "ending_detail"):
        M[c] = ep[c].where(loan)
    for c in ("next_move_event_id", "next_move_gap_days", "next_move_type"):
        M[c] = M.event_id.map(Tk[c])
    M["next_category"] = M.event_id.map(nc)
    M["from_country_v1"], M["to_country_v1"] = ep.from_country, ep.to_country
    gs = G2.assign(club_id=G2.club_id.astype("int64")).set_index("club_id")
    for side, col in (("from", "from_club_id"), ("to", "to_club_id")):
        M[f"{side}_country_v2"] = ep[col].map(gs.calendar_country.replace("", np.nan))
        M[f"{side}_geography_source_v2"] = ep[col].map(gs.source.replace("", np.nan))
    Rk = R2.set_index("loan_event_id")
    st = pd.Series(lcv2.state(R2).values, index=R2.loan_event_id)
    M["calendar_context"] = M.event_id.map(Rk.context)
    M["calendar_key"] = M.event_id.map(Rk.calendar_key.replace("", "(none: club country unknown)"))
    M["ending_in_covid_window"] = M.event_id.map(Rk.covid_period)
    M["calendar_class"] = M.event_id.map(Rk.calendar_class)
    M["calendar_state"] = M.event_id.map(st)
    inc = M.calendar_class.notna()
    M["calendar_classifiable"] = M.calendar_class.ne("geography_unclassified").where(inc)
    M["off_calendar"] = M.calendar_class.str.startswith("off_calendar").where(M.calendar_classifiable.eq(True))
    for c in ("matched_standard_mmdd", "nearest_standard_mmdd", "days_to_nearest_standard"):
        M[c] = M.event_id.map(Rk[c])
    M["unclassified_reason"] = M.event_id.map(Rk.unclassified_reason.replace("", "none (classifiable)"))
    M["loan_calendar_status"] = loan_calendar_status(M)
    assert not M.loc[M.loan_calendar_status.eq("ending_not_realised"), "ending_realised"].astype(bool).any()
    M["off_calendar_flag"] = M.loan_calendar_status.map({"off_calendar": 1, "normal": 0}).astype("Int64")
    assert (M.off_calendar_flag.eq(1).fillna(False).astype(bool) == M.off_calendar.eq(True).fillna(False).astype(bool)).all()
    assert M.off_calendar_flag.notna().sum() == M.off_calendar.notna().sum()
    Ek = E.set_index("event_id")
    for c in ("receiving_league", "arrival_season", "departure_date", "departed_in_window", "window_start", "window_end",
              "season_games", "games_remaining_at_arrival", "available_games", "complete_games", "appearances", "minutes",
              "available_minutes", "minutes_share", "true_zero"):
        M[c] = M.event_id.map(Ek[c])
    M["pt_exclusion_reason"] = M.event_id.map(Ek.exclusion_reason.replace("", "none (reliable)"))
    M["prior_spell_at_b"] = M.event_id.map(Ek.prior_spell_at_b)
    M["pt_loan_ending"] = M.event_id.map(Ek.loan_ending)
    M["arrived_full_season"] = M.event_id.map(Ek.arrived_full_season)
    M["remain"] = (M.games_remaining_at_arrival / M.season_games).astype(float)
    M["date_of_birth"], M["age_at_transfer"] = ep.date_of_birth, ep.age_at_transfer
    M["age_exact"] = (ep._date - pd.to_datetime(ep.date_of_birth, errors="coerce")).dt.days / 365.25
    M["age_years"] = np.floor(M.age_exact)
    M["older"] = (M.age_years > pta.CUTOFF).astype(float).where(M.age_years.notna())
    M["position"] = ep.position
    M["position_clean"] = ep.position.where(ep.position.isin(reg.POSITIONS))
    M["country_of_citizenship"] = ep.country_of_citizenship
    M["market_value_eur"] = ep.market_value_eur
    Bk = B.set_index("event_id")
    for c in ("player_value", "player_value_date", "player_value_gap_days", "stale_player_value", "squad_value",
              "squad_players", "focal_in_squad", "relative_value", "log_player_value", "log_squad_value",
              "log_relative_value", "league_season"):
        M[c] = M.event_id.map(Bk[c])
    # the master's own age and remain must equal the regression code's on every revised episode
    rv = M.event_id.isin(Bk.index)
    assert np.allclose(M.loc[rv, "age_exact"].fillna(-1), M.loc[rv, "event_id"].map(Bk.age).fillna(-1))
    assert np.allclose(M.loc[rv, "remain"], M.loc[rv, "event_id"].map(Bk.remain))
    Ik = EI.set_index("event_id")
    M["injury_join_status"] = M.event_id.map(Ik.status)
    for c in ("available_games_current", "confirmed_injury_games", "available_games_fit", "current_share",
              "fit_adjusted_share", "injury_days_overlap", "injury_spell_count", "appearance_conflicts"):
        M[c] = M.event_id.map(Ik[c])
    M["injury_data_quality_flag"] = M.event_id.map(Ik.data_quality_flag)
    for c in ("injury_games_strict", "fit_share_strict", "injury_games_severe3", "fit_share_severe3",
              "injury_games_long5", "fit_share_long5"):
        M[c] = M.event_id.map(Ik[c])
    M["in_calendar_population"] = M.event_id.isin(set(R2.loan_event_id))
    M["in_playing_time_candidates"] = M.event_id.isin(Ek.index)
    for c, s in (("pt_realised", "realised"), ("pt_covered_league_season", "covered_league_season"),
                 ("pt_matched", "matched"), ("pt_reliable", "reliable"), ("in_old_primary_sample", "primary_sample")):
        M[c] = M.event_id.map(Ek[s]).astype("boolean").fillna(False).astype(bool)
    import duckdb
    from ..config import DEFAULT_DATABASE
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        Q = aud.episode_check(con, M[M.pt_reliable])
    finally:
        con.close()
    rel = M.pt_reliable
    M["pt_lineup_games_checked"] = M.event_id.map(Q.window_games_with_lineups).where(rel).astype("Int64")
    M["pt_games_played_without_row"] = M.event_id.map(Q.games_played_without_row).where(rel).astype("Int64")
    assert M.loc[rel, "available_games"].astype(int).equals(M.loc[rel, "event_id"].map(Q.window_games).astype(int))
    contra = rel & M.pt_games_played_without_row.fillna(0).gt(0)
    M["pt_contradiction_type"] = np.select([contra & M.appearances.eq(0), contra], ["false_zero", "partial_undercount"],
                                           "none")
    M["pt_contradiction_type"] = M.pt_contradiction_type.where(rel)
    M["pt_lineup_contradicted"] = contra.astype(bool)
    M["pt_reliable_v2"] = (rel & ~contra).astype(bool)
    M["in_revised_sample"] = M.event_id.isin(Bk.index)
    M["in_injury_usable"] = M.event_id.isin(set(EI.loc[EI.usable, "event_id"]))
    M["in_old_regression_sample"] = M.event_id.isin(models["old"])
    M["in_age_model_sample"] = M.event_id.isin(models["age"])
    M["in_injury_model_sample"] = M.event_id.isin(models["injury"])
    M["calendar_first_eligible"] = M.kind.eq("permanent") | M.loan_calendar_status.isin(CF_STATUS_OK)
    for c, s in (("in_cf_revised_sample", "in_revised_sample"), ("in_cf_injury_usable", "in_injury_usable"),
                 ("in_cf_main_model_sample", "in_age_model_sample"),
                 ("in_cf_injury_model_sample", "in_injury_model_sample")):
        M[c] = M[s] & M.calendar_first_eligible
    for c in ("transfer_date", "terminal_date", "departure_date", "window_start", "window_end", "player_value_date"):
        M[c] = pd.to_datetime(M[c]).dt.strftime("%Y-%m-%d")
    M["date_of_birth"] = pd.to_datetime(M.date_of_birth, errors="coerce").dt.strftime("%Y-%m-%d")
    M["arrival_season"] = M.arrival_season.astype("Int64")
    order = [x.variable for x in VARS]
    assert sorted(order) == sorted(M.columns) and len(order) == len(set(order)), \
        set(order) ^ set(M.columns)
    M = M[order]
    assert M.event_id.is_unique and len(M) == len(u.episodes)
    return M


def num(x, plain: bool) -> str:
    if float(x).is_integer():
        return f"{int(x)}" if plain else f"{int(x):,}"
    return f"{x:.4g}"


def possible_values(s: pd.Series) -> str:
    n = len(s)
    blank = int(s.isna().sum())
    nn = s.dropna()
    if nn.empty:
        return f"all blank ({n:,})"
    vc = nn.astype(str).value_counts()
    if len(vc) <= 12:
        txt = "; ".join(f"{k} ({v:,})" for k, v in vc.items())
    elif pd.api.types.is_numeric_dtype(nn):
        plain = s.name.endswith(("_id", "_season"))
        txt = f"{num(nn.min(), plain)} … {num(nn.max(), plain)}; {len(vc):,} distinct"
    elif nn.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        txt = f"{nn.min()} … {nn.max()}"
    else:
        txt = f"{len(vc):,} distinct values"
    return txt + (f"; blank ({blank:,})" if blank else "")


def valid_fields(M: pd.DataFrame, c: str) -> dict:
    """When the column can be interpreted, on how many rows, and how many values sit outside that domain."""
    vw = VALID_WHEN[c]
    label, mask = DOMAINS[vw.domain]
    dom = mask(M).astype(bool)
    outside = int((M[c].notna() & ~dom).sum())
    assert (vw.outside == "blank") == (outside == 0), (c, vw.outside, outside)
    text = label + (f"; {vw.note}" if vw.note else "")
    if vw.outside == "filled":
        text += f" (the other {outside:,} rows carry a placeholder value: do not interpret)"
    return {"valid_when": text, "valid_domain": vw.domain, "rows_valid": int(dom.sum()),
            "values_outside_valid": outside}


def variable_dictionary(M: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x in VARS:
        s = M[x.variable]
        rows.append({"variable": x.variable, "section": x.section, "dtype": str(s.dtype), "raw_or_derived": x.raw_or_derived,
                     "source_file_or_table": x.source, "source_columns": x.source_columns, "join_keys": x.join_keys,
                     "formula_or_rule": x.rule, "possible_values": possible_values(s), "n_non_blank": int(s.notna().sum()),
                     "missing_value_meaning": x.missing, "is_filter_or_flag": x.kind, "used_in_analysis": x.used,
                     "plain_english_definition": x.definition, "important_caveat": x.caveat,
                     "lineage_confidence": x.confidence, "origin": ORIGINS[MASTER_GUIDE[x.variable].origin],
                     "usefulness": USEFULNESS[MASTER_GUIDE[x.variable].usefulness], **valid_fields(M, x.variable)})
    D = pd.DataFrame(rows)
    assert list(D.variable) == list(M.columns) and set(MASTER_GUIDE) == set(M.columns), set(MASTER_GUIDE) ^ set(M.columns)
    assert set(VALID_WHEN) == set(M.columns), set(VALID_WHEN) ^ set(M.columns)
    der = D[D.raw_or_derived.str.startswith("derived")]
    assert der.source_columns.str.len().gt(0).all() and der.formula_or_rule.str.len().gt(0).all()
    return D


# ---------------------------------------------------------------------------
# Part 2: source inventory
# ---------------------------------------------------------------------------

def inventory(con) -> pd.DataFrame:
    rows = []
    for s in SOURCES:
        if s.kind == "duckdb":
            n = con.sql(f'select count(*) from "{s.name}"').fetchone()[0]
            cols = [c for c, in con.sql(f"select column_name from information_schema.columns where table_name = '{s.name}' "
                                        "order by ordinal_position").fetchall()]
        elif s.kind == "jsonl":
            n, cols = 0, ["player_id", "response.transfers[]: fee, date, dateUnformatted, from{clubName, href, "
                          "countryFlag}, to{...}, marketValue, url, futureTransfer, upcoming, season"]
            for season in ("2023", "2024", "2025"):
                with open(ROOT / "data" / "raw" / "transfermarkt-api" / season / "transfers.json", "rb") as fh:
                    n += sum(1 for line in fh if line.strip())
            meta = json.loads((OUT / "stage1c_transfermarkt_enrichment_meta.json").read_text())
            s = s._replace(limitations=s.limitations + f" Lines across the three files: {n:,}; distinct players read "
                           f"(most recent file per player): {meta['raw_players_read']:,} "
                           f"({', '.join(f'{k}: {v:,}' for k, v in meta['new_players_per_season_file'].items())}).")
        elif s.kind == "json":
            d = json.loads((ROOT / s.path).read_text())
            n, cols = len(d.get("queried", {})), ["queried{club_id: [{item, country}]}", "method"]
        else:
            F = pd.read_csv(ROOT / s.path, dtype=str, keep_default_na=False)
            n, cols = len(F), list(F.columns)
        rows.append({"source": s.name, "kind": s.kind, "path": s.path, "origin": s.origin, "rows": int(n),
                     "columns": len(cols) if s.kind not in ("jsonl", "json") else "", "grain": s.grain,
                     "key_identifiers": s.keys, "exact_columns": ", ".join(cols), "columns_we_use": s.used_columns,
                     "used_by": s.used_by, "coverage_limitations": s.limitations})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute() -> tuple[dict, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    assert g2.sha256(CANONICAL_CSV) == g2.STAGE1C_SHA256
    N: dict = {"stage1c_sha256": g2.STAGE1C_SHA256}
    S = pd.read_csv(CANONICAL_CSV, low_memory=False)
    Sr = pd.read_csv(CANONICAL_CSV, dtype=str, keep_default_na=False)
    N["stage1c"] = stage1c_facts(S)
    D1 = stage1c_dictionary(S, Sr)

    u = loan_scope.universe()
    N["episodes"] = episode_waterfall(u)
    N["loan_example"] = loan_example(u)
    nm = lte.next_moves(u)
    T = lte.episode_table(u, nm)
    N["loans"] = loan_facts(u, T, nm)
    N["et_example"] = et_example(u, T)

    G2 = pd.read_csv(g2.GEO_V2_CSV, keep_default_na=False)
    u2 = build_universe(load_canonical(), g2.geography_frame(G2))
    N1, T1 = lc.compute(u)
    N2, T2 = lc.compute(u2)
    R1, R2 = T1["R"], T2["R"]
    N["calendar"] = calendar_facts(R1, R2, N2, T, G2)
    N["code_check"] = code_check()
    assert not any(N["code_check"].values()), N["code_check"]
    geo = json.loads(g2.NUMBERS_JSON.read_text())
    N["geography"] = {"v1_clubs": geo["geo1_clubs"], "stage1c_clubs": geo["clubs"]["stage1c"],
                      "real_clubs": geo["clubs"]["real"], "with_country": geo["clubs"]["with_country"],
                      "special": geo["clubs"]["special"], "by_status": geo["clubs"]["by_status"],
                      "sources_used": geo["sources_used"], "unresolved": geo["residual"]["unresolved_list"]}

    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        J, E, _ = ptj.compute(u, con)
        pv = reg.valuations(con)
        N["pt_examples"] = {
            "true_zero": pt_example(E, con, E.reliable & E.true_zero & E.kind.eq("loan") & ~E.prior_spell_at_b
                                    & E.loan_ending.eq("return to lender") & E.available_games.ge(10),
                                    "revised-sample loan with zero minutes and at least 10 window games; latest arrival "
                                    "season, then lowest event id"),
            "absence_not_zero": pt_example(E, con, E.exclusion_reason.eq("incomplete_club_games") & E.appearances_any.eq(0),
                                           "candidate excluded for an incomplete club-game record whose player has no "
                                           "appearance row; latest arrival season, then lowest event id")}
        INV = inventory(con)
        mx = con.sql("select max(minutes_played) from appearances").fetchone()[0]
    finally:
        con.close()
    N["playing_time"] = {"covered_leagues": J["covered_leagues"], "covered_seasons": J["covered_seasons"],
                         "club_games": J["club_games"], "club_games_complete": J["club_games_complete"],
                         "club_games_zero_rows": J["club_games_zero_rows"], "candidates": J["candidates"],
                         "waterfall": pt_waterfall(E, J), "max_minutes_played": int(mx),
                         "max_reliable_share": float(E.loc[E.reliable, "minutes_share"].max()),
                         "complete_rows": ptj.COMPLETE_ROWS}
    saved_red = json.loads((OUT / "playing_time_sample_redesign_numbers.json").read_text())
    B = reg.build(E, pv)
    assert ints(B.kind.value_counts()) == saved_red["revised"]
    N["market_value"] = market_value_facts(B)

    Nn, tb = ia.compute(u)
    Einj, Cinj = tb["E"], tb["C"]
    assert set(Einj.event_id) == set(B.event_id)
    N["injury"] = injury_facts(Nn, tb)
    N["regression"], models = regression_samples(E, B, Cinj)
    Bi = B[B.event_id.isin(Einj.loc[Einj.usable, "event_id"])]
    N["injury_model_missing"] = missing_breakdown(Bi, NEED_OLD[2:])
    assert N["injury_model_missing"]["complete"] == len(Cinj)
    N["age_model_missing"] = missing_breakdown(B, pta.NEED[2:])
    N["old_model_missing"] = missing_breakdown(B[B.event_id.isin(E.loc[E.primary_sample, "event_id"])], NEED_OLD[2:])
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        N["anton"] = anton_trace(con, Einj, E, B, models["injury"])
    finally:
        con.close()

    M = master_table(u, T, R2, G2, E, B, Einj, models)
    D2 = variable_dictionary(M)
    N["master"] = {"rows": len(M), "columns": M.shape[1], "players": nplayers(M),
                   "sections": ints(D2.section.value_counts().sort_index()),
                   "membership": {c: int(M[c].sum()) for c in M.columns if c.startswith(("in_", "pt_")) and M[c].dtype == bool}}
    mem = N["master"]["membership"]
    assert mem["in_revised_sample"] == sum(saved_red["revised"].values())
    assert mem["in_injury_usable"] == sum(Nn["usable"].values())
    assert mem["in_injury_model_sample"] == sum(Nn["model_sample"].values())
    assert mem["in_calendar_population"] == N["calendar"]["v2"]["realised"]
    rv = M[M.in_revised_sample & M.kind.eq("loan")]
    N["revised_loans_by_calendar_state"] = ints(rv.calendar_state.fillna("not in the calendar population").value_counts())
    N["calendar_first"] = calendar_first(M, B, Cinj, N["regression"])
    N["academy"] = academy_sensitivity(M, B, Cinj, N["calendar_first"])
    N["club_organisation"] = json.loads(co.NUMBERS_JSON.read_text())
    MAP = co.load_map()
    AC = co.candidate_table(M, MAP)
    AR = co.review_sample(AC)
    N["academy"]["candidates"] = {"rows": len(AC), "by_status": ints(AC.same_org_confidence.value_counts()),
                                  "academy_promotion_candidate": int(AC.academy_promotion_candidate.sum())}
    assert N["academy"]["candidates"]["academy_promotion_candidate"] == int(M.academy_promotion_candidate.sum())
    N["academy"]["review"] = co.review_summary(AR) if co.MANUAL_REVIEW else None
    outside = rv[~rv.in_calendar_population]
    N["revised_loans_outside_calendar_scheduled_return"] = int(outside.terminal_scheduled_future.eq(True).sum())
    assert N["revised_loans_outside_calendar_scheduled_return"] == len(outside)
    N["inventory"] = INV[["source", "rows", "columns"]].to_dict("records")
    return N, {"D1": D1, "INV": INV, "M": M, "D2": D2, "AC": AC, "AR": AR}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def t(head, rows, right=None) -> list[str]:
    return md_table(head, [[cell(c) for c in r] for r in rows], right=right) + [""]


def render_inventory(INV: pd.DataFrame) -> str:
    W = ["# Data source inventory", "",
         "*Generated by `python -m src.analysis.data_lineage`. Row and column counts are read from the current files; "
         "the rest is traced from the code. Full table: `data_source_inventory.csv`.*", ""]
    W += t(["Source", "Rows", "Columns", "One row is", "Keys", "Used by"],
           [[r.source, f0(r.rows), r.columns, r.grain, r.key_identifiers, r.used_by] for r in INV.itertuples()],
           right={1, 2})
    for r in INV.itertuples():
        W += [f"## {r.source}", "", f"- **Where it came from:** {r.origin}", f"- **Path:** `{r.path}`",
              f"- **Rows:** {f0(r.rows)}" + (f"; **columns:** {r.columns}" if r.columns != "" else ""),
              f"- **One row is:** {r.grain}", f"- **Keys:** {r.key_identifiers}",
              f"- **Exact columns:** {r.exact_columns}", f"- **Columns we use:** {r.columns_we_use}",
              f"- **Used by:** {r.used_by}"]
        if r.coverage_limitations:
            W += [f"- **Coverage limitations:** {r.coverage_limitations}"]
        W += [""]
    return "\n".join(W)


def calendar_first_rows(CF: dict) -> pd.DataFrame:
    """Long table: every step of every calendar-first variant, loans and permanents separately."""
    rows = []

    def add(var, branch, stage, step, removed, n):
        rows.append({"variant": var, "branch": branch, "stage": stage, "step": step, "removed": removed, "left": n})

    pb = CF["permanent_branch"]
    for var, V in CF["variants"].items():
        add(var, "loan", "start", "loan episodes", "", CF["loans"])
        for s in V["gate_steps"]:
            add(var, "loan", "loan-calendar gate", s["reason"], s["removed"], s["n"])
        for br, B in (("loan", V["loan_branch"]), ("permanent", pb)):
            if br == "permanent":
                add(var, br, "start", "permanent episodes", "", CF["permanent"])
            for st, key, end, endkey in (("playing time", "pt_steps", "reliable outcome", "reliable"),
                                         ("revised sample", "revised_steps", "revised playing-time sample", "revised"),
                                         ("controls", "control_steps", "MAIN REGRESSION SAMPLE", "main"),
                                         ("injury usability", "injury_steps", "injury-usable", "injury_usable"),
                                         ("injury controls", "injury_control_steps", "INJURY REGRESSION SAMPLE",
                                          "injury_model")):
                for s in B[key]:
                    if s["removed"]:
                        add(var, br, st, s["reason"], s["removed"], s["n"])
                add(var, br, st, f"= {end}", "", B[endkey])
        for s, T in V["totals"].items():
            add(var, "total", "total", s, "", T["total"])
    return pd.DataFrame(rows)


def render_calendar_first(CF: dict) -> str:
    V = CF["variants"]
    cur, cl, no = V["current"], V["classifiable"], V["normal_only"]
    W = ["# Calendar-first sample construction", "",
         "Generated by `python -m src.analysis.data_lineage`. Every count below is computed from the master file's flags; "
         "no definition is changed. The **current** variant reproduces the Week 5/6 samples and coefficients exactly "
         "(asserted in code).", "",
         "**Sequence.** Stage 1C → master → loans: loan-ending calendar classification (the existing country-calendar "
         "rule, unchanged) → playing-time eligibility → controls → main regression → injury robustness. Permanent "
         "arrivals have no loan ending: they skip the calendar step and enter the playing-time branch directly.", "",
         "## Loan calendar status (all loans)", ""]
    ls = CF["loan_status"]
    W += t(["`loan_calendar_status`", "Loans"], [[k, f0(ls.get(k, 0))] for k in
                                              ("normal", "off_calendar", "unclassified", "ending_not_realised")]
           + [["**total**", f"**{f0(CF['loans'])}**"]], right={1})
    W += ["## The four variants", ""]
    stages = [("reliable", "Reliable playing-time outcome"), ("revised", "Revised playing-time sample"),
              ("main", "**Main regression sample**"), ("injury_usable", "Injury-usable sample"),
              ("injury_model", "**Injury regression sample**")]
    head = ["Stage"] + [f"{k}" for k in V]
    rows = [["Loans after the calendar gate"] + [f0(v["after_gate"]) for v in V.values()]]
    for s, lab in stages:
        rows.append([lab] + [f"{f0(v['totals'][s]['total'])} ({f0(v['totals'][s]['loan'])} / "
                             f"{f0(v['totals'][s]['permanent'])})" for v in V.values()])
    W += t(head, rows, right=set(range(1, len(head))))
    W += ["Cells: total (loans / permanent). Variants: " + "; ".join(f"**{k}** = {v['label']}" for k, v in V.items())
          + ".", ""]
    W += ["## Step by step: loans (classifiable variant)", ""]
    rows = [["Loan episodes", "", f0(CF["loans"])]]
    rows += [[s["reason"], f0(s["removed"]), f0(s["n"])] for s in cl["gate_steps"]]
    lb = cl["loan_branch"]
    for key, end, ek in (("pt_steps", "reliable outcome", "reliable"), ("revised_steps", "revised sample", "revised"),
                         ("control_steps", "MAIN REGRESSION", "main"), ("injury_steps", "injury-usable", "injury_usable"),
                         ("injury_control_steps", "INJURY REGRESSION", "injury_model")):
        rows += [[s["reason"], f0(s["removed"]), f0(s["n"])] for s in lb[key] if s["removed"]]
        rows.append([f"**= {end}**", "", f"**{f0(lb[ek])}**"])
    W += t(["Step", "Removed", "Left"], rows, right={1, 2})
    pb = CF["permanent_branch"]
    W += ["## Step by step: permanent arrivals (identical in every variant)", ""]
    rows = [["Permanent episodes", "", f0(CF["permanent"])]]
    for key, end, ek in (("pt_steps", "reliable outcome", "reliable"), ("revised_steps", "revised sample", "revised"),
                         ("control_steps", "MAIN REGRESSION", "main"), ("injury_steps", "injury-usable", "injury_usable"),
                         ("injury_control_steps", "INJURY REGRESSION", "injury_model")):
        rows += [[s["reason"], f0(s["removed"]), f0(s["n"])] for s in pb[key] if s["removed"]]
        rows.append([f"**= {end}**", "", f"**{f0(pb[ek])}**"])
    W += t(["Step", "Removed", "Left"], rows, right={1, 2})
    W += ["## What the gate removes, and why it matters", "",
          f"- **Realised vs classifiable give the same samples.** All {f0(ls.get('unclassified', 0))} loans with an "
          f"unclassified ending ({f0(CF['unclassified_not_covered'])} of them) are removed anyway at the covered-league "
          "step: the receiving club is not in a covered league that season. Requiring a classifiable ending removes "
          "nothing beyond requiring a realised one.",
          f"- **The loans the gate actually removes** from the revised sample are the "
          f"{f0(cur['revised_loans_by_status'].get('ending_not_realised', 0))} revised loans whose ending Transfermarkt "
          "flags as a future move: loans whose season in the playing-time window is over but whose return (mostly "
          "30 June 2026) had not happened when the history was captured.",
          f"- **Calendar-normal only** also removes the {f0(cur['revised_loans_by_status'].get('off_calendar', 0))} "
          "off-calendar revised loans. An off-calendar ending can itself be a consequence of low playing time (a loan "
          "cut short), so restricting to normal endings selects on the outcome. Treat it as a sensitivity check, not "
          "a cleaner sample.", ""]
    W += ["## Sensitivity: estimates on each variant (same model, same code)", "",
          "M3 = share on loan, arrival timing, age, age², log relative value, log squad value, position, league × season "
          "fixed effects, clustered by receiving club. Injury models: same specification.", ""]
    rows = []
    for k, v in V.items():
        e = v["estimates"]
        sg = lambda x: f"{x:+.4f}".replace("-0.0000", "0.0000").replace("+0.0000", "0.0000")
        ci = lambda m: f"{sg(m['coef'])} [{sg(m['lo'])}, {sg(m['hi'])}]"
        rows.append([k, f"{e['raw_gap_revised'] * 100:+.1f} pp", f"{e['raw_gap_main'] * 100:+.1f} pp", ci(e["m3"]),
                     f0(e["m3"]["N"]), ci(e["injury_m3_current"]), ci(e["injury_m3_fit"]), f0(e["injury_m3_fit"]["N"])])
    W += t(["Variant", "Raw gap (revised)", "Raw gap (main)", "M3 loan coefficient [95% CI]", "N", "Injury: current",
            "Injury: fit-adjusted", "N"], rows, right={1, 2, 4, 7})
    return "\n".join(W) + "\n"


def render_core(D2: pd.DataFrame) -> str:
    Dk = D2.set_index("variable")
    W = ["# The most useful master columns", "",
         "`data/outputs/rebuild/analysis_episode_master.csv`: one row per transfer episode. These are the columns to "
         "reach for first; `analysis_variable_dictionary.csv` documents all of them (`usefulness`, `valid_when`, "
         "`rows_valid` and `values_outside_valid` columns).", "",
         "**Valid when** says on which rows a value can be interpreted. Where a column carries a placeholder outside "
         "that domain (for example `true_zero = False` on episodes without a reliable outcome), filter on the flag "
         "shown before using it. Analysis samples: filter on the `in_cf_*` flags.", ""]
    for group, cols in CORE_SHORTLIST:
        W += [f"## {group}", ""]
        W += t(["Column", "What it is", "Valid when (filter first)", "Rows valid", "Origin"],
               [[f"`{c}`", Dk.loc[c, "plain_english_definition"], Dk.loc[c, "valid_when"],
                 f0(Dk.loc[c, "rows_valid"]), Dk.loc[c, "origin"]] for c in cols], right={3})
    cnt = D2.usefulness.value_counts()
    W += ["## All columns by usefulness", ""]
    W += t(["Usefulness", "Columns", "Names"], [[u, str(int(cnt.get(u, 0))), ", ".join(f"`{c}`" for c in
                                                D2.loc[D2.usefulness == u, "variable"])] for u in USEFULNESS.values()],
           right={1})
    return "\n".join(W) + "\n"


def render(N: dict, tb: dict) -> str:
    s1, ep, lo, ca, pt = N["stage1c"], N["episodes"], N["loans"], N["calendar"], N["playing_time"]
    mv, rg, inj, an = N["market_value"], N["regression"], N["injury"], N["anton"]
    v1, v2 = ca["v1"], ca["v2"]
    W = ["# Data lineage, data dictionary and sample construction", "",
         "*Generated by `python -m src.analysis.data_lineage`. Documentation only: no analysis definition was changed. "
         "Every sample below is rebuilt in memory with the project's own functions and checked against its saved "
         "output; every number is computed in that run. Stage 1C SHA-256 "
         f"`{N['stage1c_sha256'][:16]}…` (unchanged).*", "",
         "**Companion files** (all new, in `data/outputs/rebuild/`): `stage1c_column_dictionary.csv` (every Stage 1C "
         "column), `data_source_inventory.csv` / `.md` (every source), `analysis_episode_master.csv` (one row per "
         "transfer episode), `analysis_variable_dictionary.csv` (one row per master variable), "
         "`data_lineage_numbers.json` (every number here).", "",
         "**How to trace any result:** raw table → raw columns → join → derived variable → flag / filter → sample count "
         "→ final analysis. Example, the injury-model loan coefficient: `transfers` row → Stage 1C `event_id`, "
         "`to_club_id`, `transfer_date` → episode (§3) → `games` + `appearances` joined on B and the window → "
         "`minutes`, `available_games` → `minutes_share` (§6) → reliable, first spell, loan return (§7) → Kaggle "
         "`player_injuries` joined on `player_id` → `fit_adjusted_share` (§11) → usable (§11) → covariates complete "
         f"(§8) → {f0(sum(rg['injury']['model'].values()))} observations → OLS with league × season FE (§9).", "",
         "**Contents.** 1 Stage 1C · 2 Sources · 3 Rows → episodes · 4 Loan endings and `early_termination` · "
         "5 Calendar · 6 Playing-time outcome · 7 Playing-time samples · 8 Market value · 9 Regression variables · "
         "10 External injury data · 11 Injury adjustment · 12 Validation · 13 Master table · 14 Lineage diagrams · "
         "15 QA and open points · 16 Questions you should be able to answer", ""]

    # ---- 1 Stage 1C
    W += ["## 1. Stage 1C: the foundation", ""]
    W += t(["", ""], [
        ["File", "`data/outputs/rebuild/stage1c_canonical_transfers.csv` (frozen; every phase checks its SHA-256)"],
        ["Rows", f0(s1["rows"])], ["Columns", f0(s1["columns"])],
        ["One row is", "one **registration movement** of one player between two Transfermarkt entries (a club, a "
         "youth/reserve side, or a placeholder such as 'Without Club'), exactly one row of the upstream `transfers` "
         "table"],
        ["Distinct players", f0(s1["players"])],
        ["Rows per player", f"median {s1['rows_per_player_median']:.0f}, max {s1['rows_per_player_max']}"],
        ["Dates", f"{s1['date_min']} to {s1['date_max']} (seasons {s1['season_min']} to {s1['season_max']}); "
         f"{f0(s1['rows_before_2000'])} rows before 2000, {f0(s1['rows_from_2010'])} from 2010; "
         f"{f0(s1['future_flagged'])} rows flagged by Transfermarkt as future moves"],
        ["Upstream", "`dcaribou/transfermarkt-datasets` DuckDB snapshot, table `transfers` (175,165 rows, 10 columns), "
         "plus that project's raw transfer-history files for the original fee text. No request to transfermarkt.com."],
        ["Population", "every recorded movement of the players whose transfer history upstream captured in its "
         "2023, 2024 and 2025 season files (players on recent squads). It is **not** a census of all transfers."]], right=set())
    W += ["### What was standardised before Stage 1C", "",
          "1. **Stage 1 backbone** (`src/stage1/backbone.py`): reads the 10 raw `transfers` columns; orders each player's "
          "rows chain-aware (a same-day return before the next loan); joins `players` (date of birth, position, "
          "citizenship) on `player_id`; computes `event_id` and `age_at_transfer`; links continuing movements "
          "(`event_chain_id`); keeps a timeline heuristic of the movement type (`stage1_heuristic_class`).",
          "2. **Stage 1B parse** (`src/stage1b/parse.py`): the upstream DuckDB turns every non-numeric fee into 0. The "
          "raw fee text is therefore re-read from the upstream raw files and mapped to a label by exact text rules "
          "('-' → no_fee_shown, '?' → undisclosed, 'free transfer', 'End of loan', 'loan transfer', 'Loan fee: €x', "
          "'End of loan €x', 'draft', '€x' → paid_transfer).",
          "3. **Stage 1C policy** (`src/stage1c/policy.py`): label → `transfer_type_normalized`; the amount is routed to "
          "exactly one of `permanent_transfer_fee_eur`, `loan_fee_eur`, `fee_on_return_eur`; club-shape flags "
          "(placeholder, youth/reserve, internal); the Stage 2 research decision.", "",
          "Label → normalised type (rows):", ""]
    W += t(["transfer_type_raw → transfer_type_normalized", "Rows"],
           [[k, f0(n)] for k, n in sorted(s1["label_to_type"].items(), key=lambda kv: -kv[1])])
    W += ["### What Stage 1C deliberately does not do", "",
          "- It does **not** pair a loan with its return, and does not decide how a loan ended (that is `loan_episodes`, §3–4).",
          "- It does **not** say whether an option, obligation or buy-back existed. No contract term is inferred.",
          "- It does **not** impute fees: '-', '?' and a bare 'loan transfer' stay blank, never 0. The one written zero "
          "is Transfermarkt's own 'free transfer'.",
          "- It does **not** assign club countries, league seasons, playing time or injuries.",
          "- It does **not** drop rows: every upstream row is kept; later stages give each row a role.", "",
          "### Why one loan produces several rows", "",
          "Transfermarkt records the registration moving out (`loan transfer` or `Loan fee: €x`) and moving back "
          "(`End of loan`), each as its own row, often with the scheduled end date. A loan that is extended, cut short, "
          "converted or followed by a sub-loan produces more rows. Example (fixed rule: the 2022/23 loan with the lowest "
          "event id that ended with an End-of-loan row on 30 June 2023 and no follow-on):", ""]
    ex = N["loan_example"]
    W += t(["event_id", "Date", "From → to", "Fee cell", "Normalised type", "Row role"],
           [[r["event_id"], r["transfer_date"], f"{r['from_club_name']} → {r['to_club_name']}", r["fee_display_raw"],
             r["transfer_type_normalized"], r["row_role"]] for r in ex["rows"]], right=set())
    W += [f"Two Stage 1C rows, one loan episode for {ex['player']}: the episode keeps the first row's `event_id`; the "
          f"second becomes its `terminal_event_id` (match reason `{ex['episode']['match_reason']}`, "
          f"{ex['episode']['duration_days']} days, ending `{ex['episode']['economic_ending']}`).", "",
          "### Why a row is not one economic contract", "",
          "A row is a registration event. An End-of-loan row is the administrative unwinding of a deal agreed on the "
          "outbound row. A youth-to-senior promotion inside one club is a row with no counterparty. A release to 'Without "
          "Club' is a row with no buyer. A fee on a return row may settle an option agreed elsewhere. One contract can "
          "therefore span several rows, and some rows carry no contract at all.", "",
          "### If Professor Freund asks what Stage 1C is, say this", "",
          f"> Stage 1C is our frozen copy of every Transfermarkt registration movement for the {f0(s1['players'])} players "
          f"in the upstream snapshot: {f0(s1['rows'])} rows, one per movement, with Transfermarkt's own fee label "
          "recovered from the raw pages' text and the fee split into permanent fee, loan fee and fee on return. It "
          "does not pair loans with returns, infer contract terms, impute fees or assign countries; it only records "
          "what Transfermarkt shows, so every later step can be checked against it. A loan is two rows in it (out and "
          "back); our episode code turns those into one loan.", "",
          "### Column dictionary (all 42; full detail in `stage1c_column_dictionary.csv`)", ""]
    D1 = tb["D1"]
    W += t(["#", "Column", "dtype", "Raw / derived", "Meaning", "Blank", "Special values", "Used in analysis"],
           [[r.position, f"`{r.column}`", r.dtype, r.raw_or_derived, r.plain_english_meaning, f0(r.n_blank),
             r.special_tokens if r.special_tokens != "none" else "", f"{r.used_in_analysis}: {r.downstream_use}"]
            for r in D1.itertuples()], right={0, 5})

    # ---- 2 sources
    INV = tb["INV"]
    W += ["## 2. Raw and source data", "", "Full detail per source (exact columns, columns used, limitations) in "
          "`data_source_inventory.md`.", ""]
    W += t(["Source", "Rows", "One row is", "Used by"], [[r.source, f0(r.rows), r.grain, r.used_by] for r in INV.itertuples()],
           right={1})

    # ---- 3 episodes
    W += ["## 3. From Stage 1C rows to transfer episodes", "",
          "Code: `src/analysis/loan_episodes.py` — `pair_loans` (pairing), `classify_spells` (endings), `assign_row_roles` "
          "(one role per row; a later rule overrides an earlier one, so the priority is internal > placeholder > other > "
          "matched return > unmatched return > episode), `build_universe` (episodes). No row is deleted: every row gets "
          "exactly one role.", ""]
    rows = []
    for s in ep["steps"]:
        cond, what = ROLE_TEXT[s["role"]]
        rows.append([f"`{s['role']}`", cond, what, f0(s["before"]), f0(s["removed"]), f0(s["after"]),
                     ", ".join(f"{k} {f0(v)}" for k, v in s["by_type"].items())])
    W += t(["Row role", "Condition", "What happens", "Before", "Rows with this role", "After", "By Stage 1C type"], rows,
           right={3, 4, 5})
    W += [f"**{f0(ep['episodes'])} transfer episodes** ({f0(ep['episode_players'])} distinct players) remain: every row "
          "whose role is `episode`. By type:", ""]
    W += t(["episode_type", "Episodes", "Players"], [[k, f0(v), f0(ep["episode_type_players"][k])]
                                                     for k, v in ep["episode_type"].items()])
    W += [f"In the playing-time analysis `permanent` pools permanent_paid, free_transfer, undisclosed_fee and "
          f"no_fee_shown: {f0(ep['permanent_pooled'])} episodes ({f0(ep['permanent_players'])} players). The "
          f"{ep['episode_type'].get('unknown_type', 0)} unknown_type episodes are in no analysis.", "",
          "### How an outbound loan and its return become one loan spell", "",
          f"Stage 1C has {f0(ep['outbound_loan_rows'])} outbound loan rows; each opens a spell ({f0(ep['spells'])} "
          f"spells). {ep['spells_internal']} are internal and {ep['spells_placeholder']} involve a placeholder, so "
          f"**{f0(ep['loans'])} loans** are real episodes ({f0(ep['loan_players'])} players). `pair_loans` walks each "
          "player's rows in date order (a return sorts before anything else on the same day, then by Transfermarkt "
          "transfer id) with a stack of open spells:", "",
          "1. A return **from** club X closes the open spell whose borrower is X: best exact reversed club ids, then same "
          "organisation, then any spell whose borrower matches (flagged: the player went back to another side of the lender).",
          "2. A loan from the current borrower is a **sub-loan**: the outer spell stays open.",
          "3. A loan between the same two clubs as an open spell **renews** it (the earlier spell ends there).",
          "4. A permanent move lender → borrower **converts** the spell.",
          "5. A non-loan move out of the borrower ends the spell without a return; a move out of the lender to a third "
          "club ends it (sold while on loan).", "6. Internal registrations never end a spell.",
          "7. Anything else is noted; if no return follows, the spell ends there as `next_movement_inconsistent`; with no "
          "later movement at all it stays open (`open_no_later_movement`).", "",
          "Match reason of the real loans:", ""]
    W += t(["match_reason", "Loans"], [[f"`{k}`", f0(v)] for k, v in ep["loan_match_reason"].items()])
    rr = ep["returns_matched_roles"]
    W += [f"**Return rows.** Stage 1C has {f0(ep['return_rows'])} End-of-loan rows. {f0(ep['returns_matched'])} closed "
          f"some spell; {f0(ep['returns_closing_real_loans'])} closed a real loan. Of those, "
          f"{f0(ep['returns_closing_real_loans_roles'].get('loan_return_collapsed', 0))} are `loan_return_collapsed` and "
          f"{ep['returns_closing_real_loans_roles'].get('internal_registration', 0)} keep the higher-priority role "
          "`internal_registration` (the return row itself is a youth-to-senior move inside one club, e.g. Kansas City II "
          "→ Kansas City) while still closing the loan, which is why loans with a matched return "
          f"({f0(ep['returns_closing_real_loans'])}) exceed collapsed return rows "
          f"({f0(ep['steps'][3]['removed'])}). {f0(ep['returns_unmatched'])} returns found no open loan "
          f"({', '.join(f'{k} {v}' for k, v in ep['returns_unmatched_roles'].items())}): typically the outbound loan "
          "predates the player's recorded history.", "",
          "### Terms", "",
          "- **Internal registration:** both sides are one organisation after stripping youth/reserve suffixes "
          "(regex on names, Stage 1C `is_internal_move`). Not a transfer.",
          "- **Placeholder / non-club movement:** a side is 'Without Club', 'Retired', 'Career break', 'Ban' or 'Unknown'.",
          "- **Loan return row:** a Stage 1C row with `transfer_type_normalized == 'loan_return'` (label 'End of loan', "
          "with or without a fee).",
          "- **Unmatched loan return:** a return row with no open loan from that club to close.",
          "- **Real episode row:** a row whose role is `episode`; one per loan (its outbound row) or permanent move.",
          "- **Recorded return:** the loan's terminal row is a return row (`has_matched_return`). It may be dated in the "
          "future: Transfermarkt files scheduled returns.",
          "- **Open loan:** no later movement of the player at all (`open_no_later_movement`); no ending date.",
          "- **Follow-on movement:** the player's first substantive row after a return (internal, youth_or_internal, "
          "return and draft rows skipped), if within 60 days and not a release. The unwindowed version is `next_move_*`.", ""]

    # ---- 4 loan endings
    W += ["## 4. Loan-ending variables and the early-termination flag", "",
          f"Of {f0(lo['loans'])} loans: {f0(lo['ending_observed'])} have an ending row ({f0(lo['ended_by_return'])} a "
          f"return, {f0(lo['ended_by_other_movement'])} another movement), {f0(lo['open'])} are open. "
          f"{f0(lo['scheduled_endings'])} endings are flagged by Transfermarkt as future (scheduled); "
          f"{f0(lo['realised'])} are realised (ending row exists and is not flagged future). The ending date is the "
          "Transfermarkt date on the terminal row.", "", "### Fields of the loan-spell table (`loan_episodes.SPELL_COLUMNS`)", ""]
    W += t(["Field", "Source columns", "Rule"], [[f"`{k}`", a, b] for k, (a, b) in LOAN_FIELDS.items()], right=set())
    W += ["Added by `loan_timing_exploration.episode_table`: `ending_observed` (terminal_date not blank), "
          "`ending_realised` (observed and not terminal_scheduled_future), `next_move_*` (the first substantive row "
          "after the return at any distance, releases included).", "",
          "### `economic_ending`: the decision order (`loan_episodes._economic`, first rule that applies)", ""]
    ec = lo["economic_ending"]
    W += t(["#", "Condition", "economic_ending", "Loans"], [
        [1, "no later movement, or the next movement involves neither club", "unresolved", f0(ec.get("unresolved", 0))],
        [2, "match_reason converted_to_permanent_same_clubs; or a return then a permanent move to the borrower within "
            "60 days", "purchase_option_or_permanent_conversion", f0(ec.get("purchase_option_or_permanent_conversion", 0))],
        [3, "lender transferred the player while on loan; or a return then a paid or undisclosed-fee move to a third "
            "club within 60 days", "third_party_sale_related", f0(ec.get("third_party_sale_related", 0))],
        [4, "renewed / re-loaned elsewhere / left the borrower without a return row", "other_nonstandard",
         f0(ec.get("other_nonstandard", 0))],
        [5, "a return, follow-on within 21 days, return more than 10 days from 30 June and 31 December",
         "**early_termination**", f0(ec.get("early_termination", 0))],
        [6, "a return, follow-on within 21 days (or a new loan within 60)", "immediate_follow_on_transfer",
         f0(ec.get("immediate_follow_on_transfer", 0))],
        [7, "a return with a fee and none of the above", "fee_bearing_return", f0(ec.get("fee_bearing_return", 0))],
        [8, "a plain return and none of the above", "ordinary_end_of_loan", f0(ec.get("ordinary_end_of_loan", 0))]],
        right={0, 3})
    W += ["### `early_termination`, exactly", "",
          "- **Provided by Transfermarkt?** No. Transfermarkt has no such field; no raw column carries it.",
          "- **Constructed by us** in `src/analysis/loan_episodes.py`, `_economic`, lines `if gap <= FOLLOW_ON_IMMEDIATE_DAYS "
          "and rec[\"return_boundary_days\"] > SEASON_END_TOLERANCE_DAYS`.",
          "- **Inputs:** the terminal row (must be an End-of-loan row), its date, and the follow-on row (type, destination, "
          "gap). Constants from `src/stage2/event_family.py`: `FOLLOW_ON_WINDOW_DAYS = 60`, `FOLLOW_ON_IMMEDIATE_DAYS = 21`, "
          "`SEASON_END_TOLERANCE_DAYS = 10`.",
          "- **Date rule:** `return_boundary_days` > 10, where `return_boundary_days` is the distance to the nearest **30 June "
          "or 31 December** (previous, same or next year). The current loan-episode code uses both dates.",
          "- **Follow-on rule:** a follow-on exists (next substantive row within 60 days, not a release) and is within "
          "**21 days** of the return.",
          "- **Priority:** checked only after a permanent move back to the borrower (→ conversion) and a paid or "
          "undisclosed-fee move to a third club (→ third-party sale). It wins over `immediate_follow_on_transfer`.",
          "- **Meaning:** the End-of-loan row is dated away from the usual season ends and the player moves on within "
          "three weeks. The sequence looks like a loan cut short.",
          "- **What it does NOT prove:** that any contract was terminated early, that a recall or break clause existed, "
          "or who initiated it. Given a quick follow-on, a loan scheduled to end on 31 May or 31 January (normal dates "
          "in some countries) is flagged too; a loan really cut short on 25 June is not.", "",
          f"How the {f0(lo['early_termination'])} arise:", ""]
    W += t(["Step", "Loans"], [[k, f0(v)] for k, v in lo["funnel"].items()])
    W += [f"**Two historical implementations differ.** `src/stage2/event_family.py` (`_days_from_season_end`) measures "
          "distance to **30 June only**, and treats any permanent move to a third club (free or no-fee included) as a "
          "third-party sale. `loan_episodes` adds 31 December and narrows the sale to paid or undisclosed fees. With the "
          f"current priority order, measuring to 30 June only would give {f0(lo['early_termination_if_30_june_only'])} "
          f"early terminations instead of {f0(lo['early_termination'])}; {f0(lo['only_far_from_30_june_not_31_dec'])} "
          "returns near 31 December are cleared only by the second date. The Stage 2 family classes are a separate layer "
          "and are not used by any later analysis.", ""]
    e = N["et_example"]
    W += ["### Worked example", "", f"Fixed rule: realised early_termination loans of 2023/24, lowest loan event id: "
          f"**{e['player']}** (id {e['player_id']}).", ""]
    W += t(["Row", "event_id", "Date", "From → to", "Fee cell", "Type"],
           [[r["role"], r["event_id"], r["transfer_date"], f"{r['from']} → {r['to']}", r["fee_display_raw"],
             r["transfer_type_normalized"]] for r in e["rows"]], right=set())
    W += [f"Duration {e['duration_days']} days. The return is {e['return_boundary_days']} days from the nearest 30 June / "
          f"31 December ({e['days_from_30_june']} from 30 June). The follow-on ({e['follow_on_type']}) comes "
          f"{e['follow_on_gap_days']} days later: within 21 days, and > 10 days from both dates → `{e['economic_ending']}` "
          f"(\"{e['ending_detail']}\").", "",
          "### Do not conflate it with off-calendar", "",
          "1. `economic_ending == 'early_termination'` is the old sequence heuristic above: fixed dates (30 Jun, 31 Dec) "
          "and a required quick follow-on.",
          "2. Calendar-normal / off-calendar (§5) is a separate classification of the ending date against dates learned "
          "per country from the data (±3 days), with no follow-on condition.", "",
          "They overlap partly by construction: both depend on the ending date's distance from common dates.", ""]
    eb = ca["et_by_state"]
    W += t(["Calendar state (v2, realised, classifiable)", "Endings", "early_termination", "Share"],
           [[s, f0(v["N"]), f0(v["early_termination"]), pc(v["early_termination"], v["N"])] for s, v in eb.items()])
    W += ["Calendar-normal endings can carry the label because their calendar date is not 30 June or 31 December. "
          "Matched dates of those cases: " + "; ".join(f"{k} {v}" for k, v in ca["et_normal_matched_dates"].items()) + ".", ""]

    # ---- 5 calendar
    gv = N["geography"]
    W += ["## 5. Calendar analysis lineage", "", "### Geography", "",
          f"- **Source:** club ids in Stage 1C, matched to countries. **v1** (`loan_episodes.club_geography`): "
          "`clubs.domestic_competition_id` → `competitions.country_name`, else participation in a country's domestic "
          f"league / cup (`games` × `competitions`). {f0(gv['v1_clubs'])} clubs mapped; clubs placed in two countries are "
          "dropped, never guessed.",
          "- **v2** (`club_geography_v2.py`, `club_geography_v2.csv`): precedence 1 = v1 country; 2 = the raw Transfermarkt "
          "`countryFlag` on the club's side of every raw transfer row, when all its rows carry one flag; 3 = fallbacks for "
          "the rest: A competition membership, B a validated parent-club link, C Wikidata (P7223 → P17). No country from a "
          f"club name. Of {f0(gv['real_clubs'])} real Stage 1C clubs, {f0(gv['with_country'])} have a country "
          f"({', '.join(f'{k} {f0(v)}' for k, v in gv['sources_used'].items())}); unresolved: "
          + "; ".join(gv["unresolved"]) + ".", "",
          "### The rule, operationally (`loan_calendar_rule.py`; v2 = same code with v2 countries)", "",
          f"- **Population:** realised loan endings (one per loan; date = terminal row date).",
          "- **Grouping:** `domestic` when lender and borrower have the same country → calendar key 'domestic: <country>'; "
          "`cross-border` when both are mapped and differ → one pooled 'cross-border' calendar; otherwise unmapped (no key).",
          f"- **Minimum history:** a domestic country gets its own calendar only with ≥ {lcv2.PARAMS['N_MIN']} realised "
          "non-COVID domestic endings; below that its domestic endings stay unclassified (they are not pooled elsewhere).",
          f"- **COVID:** endings dated {ca['covid_window'][0]} to {ca['covid_window'][1]} are excluded from **learning** "
          "only; the learned calendar is applied to them like any other.",
          "- **Clustering:** month-day on a 366-day circle (31 Dec and 1 Jan are 1 day apart). Greedy: the most frequent "
          f"unassigned day becomes a centre; every unassigned day within ±{lcv2.PARAMS['TOL']} joins it; repeat.",
          f"- **Selection:** clusters in frequency order; skip a cluster seen in fewer than {lcv2.PARAMS['Y_MIN']} calendar "
          f"years; stop when {lcv2.PARAMS['TARGET']:.0%} of the calendar's training endings are covered, at the first cluster "
          f"below {lcv2.PARAMS['S_MIN']:.0%} share, or at {lcv2.PARAMS['K_MAX']} clusters.",
          f"- **Classification:** an ending within ±{lcv2.PARAMS['TOL']} days of one of its calendar's dates is "
          "calendar-normal, otherwise off-calendar; no calendar → unclassified.", "",
          "### Waterfall", ""]
    W += t(["Step", "Rule", "v1 geography", "v2 geography (current)"], [
        ["All loan episodes", "episode_type == loan", f0(lo["loans"]), f0(lo["loans"])],
        ["Loans with a recorded ending", f"terminal row exists (removes {f0(lo['open'])} open)", f0(lo["ending_observed"]),
         f0(lo["ending_observed"])],
        ["Realised endings (calendar population)", f"ending not flagged future (removes {f0(lo['scheduled_endings'])} "
         "scheduled)", f0(v1["realised"]), f0(v2["realised"])],
        ["Geography mapped (both clubs)", "lender and borrower country known", f0(v1["mapped"]), f0(v2["mapped"])],
        ["… domestic / cross-border", "same / different country", f"{f0(v1['domestic'])} / {f0(v1['cross_border'])}",
         f"{f0(v2['domestic'])} / {f0(v2['cross_border'])}"],
        ["Classifiable", f"has a calendar (domestic country with ≥ {lcv2.PARAMS['N_MIN']} training endings, or cross-border)",
         f0(v1["classifiable"]), f0(v2["classifiable"])],
        ["Calendar-normal", f"within ±{lcv2.PARAMS['TOL']} days of a calendar date", f0(v1["normal"]), f0(v2["normal"])],
        ["Off-calendar", "classifiable, not normal", f0(v1["off"]), f0(v2["off"])],
        ["Unclassified: club country unknown", "a club has no country", f0(v1["country_unknown"]), f0(v2["country_unknown"])],
        ["Unclassified: below history threshold", "domestic, country below the minimum", f0(v1["below_threshold"]),
         f0(v2["below_threshold"])]], right={2, 3})
    W += [f"v2 off-calendar share of classifiable endings: {pc(v2['off'], v2['classifiable'])}. The "
          f"{f0(v2['below_threshold'])} below-threshold endings are all domestic "
          f"({'confirmed' if ca['below_threshold_all_domestic'] else 'NOT all'}) loans in {ca['below_threshold_countries']} "
          "countries, led by " + ", ".join(f"{k} ({v})" for k, v in list(ca["below_threshold_top"].items())[:8])
          + f"; the largest is {ca['largest_country_without_calendar']['country']} with "
          f"{ca['largest_country_without_calendar']['n']} non-COVID domestic endings. The "
          f"{v2['country_unknown']} country-unknown endings involve: " + "; ".join(ca["country_unknown_clubs"]) + ". "
          f"Learning used {f0(ca['training_endings'])} non-COVID endings.", "",
          "### The v2 calendars", ""]
    W += t(["Calendar", "Training endings", "Dates (centres, ±3 days)", "Coverage", "Stopped because"],
           [[k, f0(v["N"]), ", ".join(v["centers"]), f"{v['coverage']:.1%}", v["stopped"]]
            for k, v in sorted(ca["calendars_v2"].items(), key=lambda kv: -kv[1]["N"])], right={1, 3})
    W += ["### Field-level explanation (columns of `loan_calendar_classified_v2.csv`)", ""]
    W += t(["Field", "Meaning"], [
        ["`context`", "domestic / cross-border / one or both clubs unmapped"],
        ["`calendar_key`", "'domestic: <country>', 'cross-border', or '' (unmapped)"],
        ["`covid_period`", "ending in the COVID window (excluded from learning only)"],
        ["`calendar_class`", "calendar_normal_domestic, calendar_normal_cross_border, off_calendar_domestic, "
                             "off_calendar_cross_border, geography_unclassified"],
        ["`v2_state`", "calendar-normal, off-calendar, unclassified: club country unknown, unclassified: below history threshold"],
        ["`matched_standard_mmdd`", "the calendar date matched (normal endings only)"],
        ["`nearest_standard_mmdd`, `days_to_nearest_standard`", "closest calendar date and its distance (any distance)"],
        ["`unclassified_reason`", "'' / club country unknown / country below the history threshold"],
        ["(no column) calendar_classifiable", "derived in the master table: calendar_class != geography_unclassified"],
        ["(no column) off_calendar", "derived in the master table: calendar_class starts with off_calendar"]], right=set())
    cc = N["code_check"]
    W += ["### Is off-calendar a filter?", "",
          "**No.** `calendar_class` is a **category** (with off-calendar as one value; the master table adds Boolean "
          "helpers). It is used only in the calendar tables. No playing-time or injury module reads it, nor "
          "`economic_ending`, nor geography v2: a scan of their code for "
          + ", ".join(f"`{x}`" for x in CODE_TOKENS) + " finds "
          + ("nothing" if not any(cc.values()) else "matches") + " in " + ", ".join(f"`{m}`" for m in cc) + ". "
          "The only loan-ending input to playing time is whether the loan ended with a return to the lender. Revised-sample "
          "loans by calendar state: " + "; ".join(f"{k} {f0(v)}" for k, v in N["revised_loans_by_calendar_state"].items())
          + f": all are kept. The {f0(N['revised_loans_outside_calendar_scheduled_return'])} outside the calendar "
          "population all have an End-of-loan row that Transfermarkt flags as a future move (so the ending is not "
          "'realised'), while their playing-time window, which ends at the season's last game, is complete.", ""]

    # ---- 6 playing time
    tz, az = N["pt_examples"]["true_zero"], N["pt_examples"]["absence_not_zero"]
    W += ["## 6. Playing time: column-to-outcome lineage", "",
          "Code: `playing_time_join.py` (`candidate_episodes`, `arrival_season`, `measure`, `classify`).", ""]
    W += t(["Item", "Source table", "Columns", "Rule"], [
        ["Player", "Stage 1C", "player_id", "copied"],
        ["Receiving club B", "Stage 1C", "to_club_id (`b_club_id`)", "copied"],
        ["Move date", "Stage 1C", "transfer_date (`move_date`)", "window start"],
        ["Loan or permanent", "episode", "episode_type → kind", "loan / permanent (paid, free, undisclosed, no fee shown)"],
        ["Scheduled move", "Stage 1C", "transfermarkt_future_transfer", "excluded"],
        ["Departure", "Stage 1C", "the player's next substantive row", "internal and draft rows skipped"],
        ["Arrival season, league", "games", "competition_type, competition_id, season, date, home/away_club_id",
         "B's covered league season containing the move date"],
        ["Window games", "games", "date, home/away_club_id", "B's league games from move date to window end"],
        ["Club-game completeness", "appearances", "game_id, player_club_id", f"≥ {pt['complete_rows']} rows for B"],
        ["Minutes", "appearances", "game_id, player_id, player_club_id, minutes_played", "sum over window games for B"],
        ["Player exists", "players", "player_id", "else excluded"],
        ["History capture date", "raw season files", "last realised date per file", "window must end before it"],
        ["Other-club appearances", "appearances", "player_club_id ≠ B in the window", "excluded (a move is missing)"]],
        right=set())
    W += ["### The window", "",
          "- **Start:** the move date (games on that day count).",
          "- **End:** B's last league game of the arrival season, or the day before the player's next recorded move "
          "(any type) if that comes first. An early departure therefore shortens the window; it never excludes the episode.",
          "- **Arrival season:** the covered season s ∈ {year − 1, year} of the move such that the previous season's last "
          "league game < move date ≤ season s's last league game. A summer arrival therefore belongs to the coming season.",
          "", "### Competitions and the 14 leagues", "",
          f"Only `competition_type = 'domestic_league'` games count (no cups, no Europe). A league is **covered** if the "
          "snapshot has appearance rows for it in every season: "
          + ", ".join(pt["covered_leagues"]) + f" ({len(pt['covered_leagues'])} leagues, seasons {pt['covered_seasons'][0]}–"
          f"{pt['covered_seasons'][1]}). Other leagues (lower divisions, calendar-year leagues) have no appearance data, so "
          f"they cannot enter. Of {f0(pt['club_games'])} covered club-games, {f0(pt['club_games_complete'])} have a complete "
          f"record ({pc(pt['club_games_complete'], pt['club_games'])}).", "",
          "### Reliable outcome", "",
          f"A **complete appearance record** for a club-game means B has at least {pt['complete_rows']} appearance rows in "
          "it (a starting XI). An outcome is **reliable** if none of these holds (first failing reason recorded): scheduled "
          "move; B not in a covered league season; player not in `players`; B has no appearance rows that season; no "
          "league game of B in the window; window ends after the player's history was captured; **any** window game "
          "lacks a complete record; the player appears for another club in the window.", "",
          "### The outcome", "",
          "```", "minutes_share = player's league minutes for B in the window / (90 × B's league games in the window)",
          "minutes          = Σ appearances.minutes_played  (player_id, player_club_id = B, window league games)",
          "available_games  = count of B's domestic-league games with window_start ≤ date ≤ window_end",
          "available_minutes = 90 × available_games", "```", "",
          f"Minutes and appearances are written only when the outcome is reliable. The largest reliable share is "
          f"{pt['max_reliable_share']:.3f} (largest `minutes_played` in the snapshot: {pt['max_minutes_played']}).", "",
          "### How a true zero is built, and when absence is not a zero", "",
          "`appearances` has rows only for players who appeared. A player with no row in a game played 0 minutes **only "
          "if** B's record for that game is complete. So: every window game complete → the player's missing rows are "
          "genuine non-appearances → minutes = 0, denominator > 0, share = 0 (`true_zero`). If any window game is "
          "incomplete, the episode is excluded (`incomplete_club_games`) and no zero is written.", "",
          f"**Worked example, true zero** ({tz['rule']}): {tz['player']} (id {tz['player_id']}), {tz['kind']} "
          f"{tz['a_club']} → {tz['b_club']} ({tz['league']} {tz['arrival_season']}), moved {tz['move_date']}; window "
          f"{tz['window_start']} to {tz['window_end']}"
          + (f" (next move {tz['departure_date']})" if tz["departure_date"] else "") + f". {tz['available_games']} window "
          f"games, {tz['complete_games']} complete (B has ≥ 11 rows in each), {tz['player_rows']} appearance rows for the "
          f"player → minutes {tz['minutes']}, share {tz['minutes']} / (90 × {tz['available_games']}) = "
          f"{tz['minutes_share']:.3f}.", ""]
    W += t(["Date", "Opponent", "B appearance rows", "Complete", "Player minutes"],
           [[g["date"], g["opponent"], g["b_rows"], "yes" if g["complete"] else "no",
             "" if g["player_minutes"] is None else g["player_minutes"]] for g in tz["games"]], right={2, 4})
    inc = [g for g in az["games"] if not g["complete"]]
    W += [f"**Worked example, absence that is NOT a zero** ({az['rule']}): {az['player']} (id {az['player_id']}), "
          f"{az['kind']} {az['a_club']} → {az['b_club']} ({az['league']} {az['arrival_season']}); "
          f"{az['available_games']} window games, only {az['complete_games']} complete; the player has "
          f"{az['player_rows']} appearance rows. Because {len(inc)} game(s) have fewer than 11 rows for B "
          f"(e.g. {inc[0]['date']} vs {inc[0]['opponent']}: {inc[0]['b_rows']} rows), his missing rows could be missing "
          f"data → excluded (`{az['exclusion_reason']}`), minutes left blank.", ""]

    # ---- 7 samples
    wf = pt["waterfall"]
    W += ["## 7. Playing-time sample waterfall", "", "Episodes, then distinct players in brackets. Each exclusion is the "
          "first failing reason (`exclusion_reason`), applied in this order.", ""]
    rows = []
    for i, s in enumerate(wf["loan"]["steps"]):
        p = wf["permanent"]["steps"][i]
        rows.append([s["step"], f0(s.get("removed", 0)) if i else "", f"{f0(s['n'])} ({f0(s['players'])})",
                     f0(p.get("removed", 0)) if i else "", f"{f0(p['n'])} ({f0(p['players'])})"])
    rows.append(["**Reliable outcome**", "", f"**{f0(wf['loan']['reliable'])}** ({f0(wf['loan']['reliable_players'])})", "",
                 f"**{f0(wf['permanent']['reliable'])}** ({f0(wf['permanent']['reliable_players'])})"])
    for i, s in enumerate(wf["loan"]["revised_steps"]):
        p = wf["permanent"]["revised_steps"][i]
        rows.append([s["step"], f0(s["removed"]), f"{f0(s['n'])} ({f0(s['players'])})", f0(p["removed"]),
                     f"{f0(p['n'])} ({f0(p['players'])})"])
    rows.append(["**REVISED PRIMARY SAMPLE**", "", f"**{f0(wf['loan']['revised'])}** ({f0(wf['loan']['revised_players'])})", "",
                 f"**{f0(wf['permanent']['revised'])}** ({f0(wf['permanent']['revised_players'])})"])
    W += t(["Step", "Loans removed", "Loans left (players)", "Permanents removed", "Permanents left (players)"], rows,
           right={1, 2, 3, 4})
    tot_players = inj["revised_players"]
    W += [f"The revised sample has {f0(wf['loan']['revised'] + wf['permanent']['revised'])} episodes and "
          f"{f0(tot_players)} distinct players (a player can be in it as a loan and as a permanent arrival, or several "
          "times at different clubs). There is no ambiguous-loan-ending step for permanents and no 'no league match "
          "opportunity' step after reliability (that is upstream).", "",
          "**The old two-thirds rule** (old primary sample, first regression):", ""]
    W += t(["Old step (from reliable)", "Loans removed", "Loans left", "Permanents removed", "Permanents left"],
           [[s["step"], f0(s["removed"]), f0(s["n"]), f0(p["removed"]), f0(p["n"])]
            for s, p in zip(wf["loan"]["old_steps"], wf["permanent"]["old_steps"])], right={1, 2, 3, 4})
    W += [f"It required at least two-thirds of B's league games to remain at arrival; it removed mostly January and "
          f"February arrivals ({f0(wf['loan']['two_thirds_jan_feb'])} of the loans it removed). It was dropped after "
          "Professor Freund's feedback because the outcome already divides by the games available while the player is "
          "at the club, so a January arrival is compared on the games actually left (`playing_time_sample_redesign.md`).", ""]

    # ---- 8 market value
    a = mv["all"]
    W += ["## 8. Market value and relative value", "",
          "Code: `playing_time_regression.py` (`valuations`, `player_value`, `squad_value`, `build`). Computed for the "
          f"revised sample ({f0(a['N'])} episodes).", "",
          "**Player value.** Table `player_valuations` (player_id, date, market_value_in_eur). The latest valuation dated "
          "**strictly before** the move date (`merge_asof`, backward, exact matches not allowed: a valuation on the move "
          f"day may already reflect the move; {f0(mv['same_day_valuations_excluded'])} episodes have one and it is not "
          "used). No maximum age: valuations older than 365 days are flagged `stale_player_value` "
          f"({f0(a['stale'])}). None before the move → `player_value` blank ({f0(a['N'] - a['player_value'])}: "
          f"{f0(a['only_post_move'])} have only later valuations, {f0(a['no_valuation_at_all'])} none at all). Median gap "
          f"{a['gap_median']:.0f} days (p90 {a['gap_p90']:.0f}). The Stage 1C `market_value_eur` (shown on the transfer row) "
          f"equals the rule's value in {f0(mv['stage1c_market_value_vs_player_value']['equal'])} of "
          f"{f0(mv['stage1c_market_value_vs_player_value']['compared'])} comparable episodes; it is not used.", "",
          "**Squad value.** Also from `player_valuations`: for each (B, move date), every player whose latest valuation "
          "strictly before the move, and within 365 days, lists `current_club_id` = B; their values are summed, the focal "
          f"player excluded (he was already listed at B in {f0(a['focal_in_squad'])} episodes). Fewer than "
          f"{reg.MIN_SQUAD_PLAYERS} such players → blank. Median {a['squad_players_median']:.0f} valued players. The "
          "`clubs.total_market_value` column cannot be used: it is a current snapshot, not a value at the move, and it is "
          "NULL for every club in this snapshot.", "",
          "**Relative value** = player_value / squad_value. Regression form `log_relative_value` = ln(player_value) − "
          "ln(squad_value), with `log_squad_value` = ln(squad_value); no constant is added (all values are positive).", "",
          "### Why episodes fall out of the controlled regressions", ""]
    for lab, key, nn in (("Injury-model sample (§11)", "injury_model_missing", None), ("Age / revised-sample models",
                                                                                         "age_model_missing", None),
                         ("Old primary (first regression)", "old_model_missing", None)):
        m = N[key]
        W += [f"**{lab}: {f0(m['N'])} → {f0(m['complete'])}** ({f0(m['dropped'])} dropped):", ""]
        W += t(["Reason, checked in order", "Episodes with it", "Dropped at this step", "Loans", "Permanents"],
               [[s["reason"], f0(s["any"]), f0(s["sequential"]), f0(s["loan_sequential"]), f0(s["permanent_sequential"])]
                for s in m["steps"]], right={1, 2, 3, 4})
        W += [f"Of the episodes with no prior valuation, {f0(m['no_pv_only_post_move'])} have only later valuations and "
              f"{f0(m['no_pv_none_at_all'])} none at all.", ""]

    # ---- 9 regression
    W += ["## 9. Regression variables (PRELIMINARY REGRESSION ANALYSES)", "",
          "All models are OLS of the playing-time share (statsmodels), league × season fixed effects as dummies, position "
          "dummies, standard errors clustered by receiving club `b_club_id`. Observational, not causal. **Raw** = difference "
          "in mean share, loan minus permanent; **adjusted** = the loan coefficient conditional on the controls.", ""]
    W += t(["Variable", "Source columns", "Derivation", "Unit", "Missing", "Interpretation"], [
        ["share (y)", "minutes, available_games", "minutes / (90 × available_games)", "0–1", "never in a sample", "playing-time share"],
        ["loan", "episode_type", "1 if loan, 0 if permanent", "0/1", "never", "loan arrival"],
        ["age_c, age_c²", "date_of_birth, transfer_date", "age = days/365.25; centred on the model sample's mean",
         "years", "no date of birth → dropped", "age (quadratic)"],
        ["older", "age", f"1 if floor(age) ≥ {pta.CUTOFF + 1}", "0/1", "as age", f"older arrival (young ≤ {pta.CUTOFF})"],
        ["loan × older", "loan, older", "product", "0/1", "as age", "how much the loan gap differs for older arrivals"],
        ["remain", "games_remaining_at_arrival, season_games", "ratio", "0–1", "never", "arrival timing"],
        ["log_relative_value", "player_value, squad_value", "ln(pv) − ln(sv)", "log points", "no valuation / squad value → dropped",
         "player value relative to the receiving squad"],
        ["log_squad_value", "squad_value", "ln(sv)", "log €", "as above", "receiving squad's value"],
        ["position dummies", "players.position", "Goalkeeper/Defender/Midfield/Attack", "categories", "'Missing' → dropped",
         "position"],
        ["league × season FE", "receiving_league, arrival_season", "dummies for `league_season`", "", "never",
         "compares arrivals within the same league season"],
        ["clustering", "b_club_id", "cluster-robust covariance", "", "never", "allows correlated errors within a club"]],
        right=set())
    ag = rg["revised"]["age"]
    W += ["**`loan × older` in plain English:** the main `loan` coefficient is the loan − permanent gap for young "
          f"arrivals (≤ {pta.CUTOFF}); `loan × older` is how much that gap changes for arrivals aged {pta.CUTOFF + 1}+; "
          "their gap is `loan` + `loan × older`. On the revised model sample: young "
          f"{100 * ag['young']['coef']:+.1f} pp, older {100 * ag['older']['coef']:+.1f} pp, interaction "
          f"{100 * ag['interaction']['coef']:+.1f} pp.", "", "### Raw versus adjusted, by sample", ""]
    o, r_, i_ = rg["old"], rg["revised"], rg["injury"]
    W += t(["Sample", "Episodes (loans / permanents)", "Raw gap, whole sample", "Model observations", "Raw gap, model sample",
            "Adjusted loan coefficient (M3) [95% CI]"], [
        ["Old primary (two-thirds rule)", f"{f0(o['sample']['loan'])} / {f0(o['sample']['permanent'])}",
         f"{100 * o['raw_gap_sample']:+.1f} pp", f"{f0(o['model']['loan'])} / {f0(o['model']['permanent'])}",
         f"{100 * o['raw_gap_model']:+.1f} pp", f"{100 * o['m3']['coef']:+.1f} pp [{100 * o['m3']['lo']:+.1f}, {100 * o['m3']['hi']:+.1f}]"],
        ["Revised primary", f"{f0(r_['sample']['loan'])} / {f0(r_['sample']['permanent'])}",
         f"{100 * r_['raw_gap_sample']:+.1f} pp", f"{f0(r_['model']['loan'])} / {f0(r_['model']['permanent'])}",
         f"{100 * r_['raw_gap_model']:+.1f} pp", f"{100 * r_['m3']['coef']:+.1f} pp [{100 * r_['m3']['lo']:+.1f}, {100 * r_['m3']['hi']:+.1f}]"],
        ["Injury model, current outcome", "", "", f"{f0(i_['model']['loan'])} / {f0(i_['model']['permanent'])}",
         f"{100 * i_['raw_gap_model_current']:+.1f} pp",
         f"{100 * i_['m3_current']['coef']:+.1f} pp [{100 * i_['m3_current']['lo']:+.1f}, {100 * i_['m3_current']['hi']:+.1f}]"],
        ["Injury model, fit-adjusted outcome", "", "", f"{f0(i_['model']['loan'])} / {f0(i_['model']['permanent'])}",
         f"{100 * i_['raw_gap_model_fit']:+.1f} pp",
         f"{100 * i_['m3_fit']['coef']:+.1f} pp [{100 * i_['m3_fit']['lo']:+.1f}, {100 * i_['m3_fit']['hi']:+.1f}]"]],
        right={2, 4, 5})
    W += ["M3 = loan, remain, age_c, age_c², log_relative_value, log_squad_value, position dummies, league × season FE. "
          f"age_c is centred on each sample's own mean ({o['age_center']:.2f}, {r_['age_center']:.2f}, "
          f"{i_['age_center']:.2f} years), which leaves the loan coefficient unchanged. Re-estimated here and equal to the "
          "saved values of `playing_time_regression`, `playing_time_age` and `playing_time_injury_adjusted`.", ""]

    # ---- 10 external injury
    nz = inj["normalize"]
    W += ["## 10. External injury data", "",
          "Files (read-only, SHA-256 verified): `data/raw/external_injuries/salimt_kaggle/player_injuries/player_injuries.csv` "
          f"and `…/player_profiles/player_profiles.csv`; Kaggle `xfkzujqjvx97n/football-datasets` version 2 (CC0), "
          f"snapshot **{ia.SNAPSHOT.date()}**, downloaded manually by the user; its author states Transfermarkt as the source.", ""]
    W += t(["", "player_injuries.csv", "player_profiles.csv"], [
        ["Rows", f0(inj["raw_rows"]), f0(inj["profiles"]["rows"])],
        ["Columns", f"{len(inj['raw_columns'])}: " + ", ".join(inj["raw_columns"]), "34 (we use player_id, player_name, date_of_birth)"],
        ["One row is", "one listed injury or absence spell of one player", "one player in the external datalake"],
        ["Players", f0(inj["raw_players"]), f0(inj["profiles"]["unique_ids"])],
        ["Role", "spells (from_date, end_date, injury_reason)", "distinguishes 'in the datalake, no injury listed' from 'absent'"]],
        right=set())
    W += [f"- **Join:** exact integer `player_id` = Transfermarkt player id, the same id as our `players` / `transfers` "
          f"(both datasets come from Transfermarkt). No name matching. Dates of birth agree for "
          f"{f0(inj['id_check']['dob_equal'])} of {f0(inj['id_check']['compared'])} players compared.",
          f"- **Duplicates:** {f0(nz['exact_duplicates_dropped'])} exact duplicate rows dropped; {f0(nz['conflicting_duplicate_rows'])} "
          "rows share a player and start date but differ (kept and flagged; overlapping spells count a game once).",
          f"- **Problems flagged, never fixed:** {nz['missing_start']} rows without a start date, {f0(nz['open'])} open "
          f"(no end date), {nz['end_before_start']} ending before they start, {nz['starts_after_snapshot']} starting after the snapshot.",
          f"- **The 15-row ceiling is inferred, not documented.** No player has more than {inj['rows_per_player_max']} rows; "
          f"{f0(inj['players_at_15'])} have exactly 15 against {f0(inj['players_at_14'])} with 14 and "
          f"{f0(inj['players_at_13'])} with 13. Neither the Kaggle description nor the README states a cap; the spike says "
          "the source lists only the latest 15. Handling: for those players, an episode whose window starts before their "
          "oldest listed injury gets no adjusted value (`injury history truncated before the window`).",
          f"- **Snapshot:** injuries after {ia.SNAPSHOT.date()} are not in the file, so a window reaching that date is not "
          "usable. 2025/26 is incomplete. This is the injury file's snapshot, not the appearance data's.", "",
          "**Player-level status** (revised sample):", ""]
    se = inj["status_episodes"]
    W += t(["Status", "Meaning", "Players", "Loan episodes", "Permanent episodes"], [
        [k, m, f0(inj["status_players"].get(k, 0)), f0(se["loan"].get(k, 0)), f0(se["permanent"].get(k, 0))]
        for k, m in (("A", "found with injury rows"), ("B", "profile found, no injury rows (no listed injury, not proven health)"),
                     ("C", "absent from the external dataset"))], right={2, 3, 4})
    W += ["Injury labels are classified by `classify_reason` (every label in `injury_reason_classification.csv`):", ""]
    W += t(["Category", "Rows", "Counts (primary)", "Counts (strict)"],
           [[c, f0(v["rows"]), "yes" if v["primary"] else "no", "yes" if v["strict"] else "no"]
            for c, v in sorted(inj["categories"].items(), key=lambda kv: -kv[1]["rows"])], right={1})

    # ---- 11 injury adjustment
    W += ["## 11. Injury-adjusted playing time", "", "Code: `playing_time_injury_adjusted.py` (`spells`, `game_flags`, "
          "`episode_outcomes`, `usability`). Same episodes, windows and minutes as §6.", "",
          "```", "available_games_current = B's league games in the window   (= available_games)",
          "confirmed_injury_games  = window games covered by ≥ 1 primary medical spell in which the player has no appearance",
          "available_games_fit     = available_games_current − confirmed_injury_games",
          "current_share           = minutes / (90 × available_games_current)   (= minutes_share)",
          "fit_adjusted_share      = minutes / (90 × available_games_fit)       (blank if available_games_fit = 0)", "```", "",
          "- **Spell overlap:** a spell covers a game if from_date ≤ game date ≤ end_date (both inclusive).",
          "- **Appearance override:** if the player has an appearance row for B in that game, the game stays in the "
          f"denominator (`appearance_conflicts`: {f0(inj['appearance_conflicts']['games'])} games in "
          f"{f0(inj['appearance_conflicts']['episodes'])} episodes).",
          "- **Open spells** (no end date): never given an end date. If one for the player starts on or before the window end, "
          "the episode is not usable.",
          "- **Missing start date:** the row cannot be placed; if the player has such a medical row, the episode is not usable.",
          "- **Non-medical reasons:** 'Rest', 'Fitness', 'Quarantine' never count. 'unknown injury' and minor muscular "
          "complaints count in PRIMARY only, not STRICT.",
          "- **Overlapping spells:** a game covered by several spells is removed once (union).",
          "- **Snapshot / truncation:** windows reaching 2025-09-15, and windows starting before a truncated player's oldest "
          "listed injury, are not usable.",
          "- **Calendar classification:** not involved anywhere (§5 code check).", "",
          "**Usability waterfall** (first failing reason, in the code's order):", ""]
    us = inj["usability_steps"]
    rows = [["revised sample", "", f0(inj["revised"]["loan"]), "", f0(inj["revised"]["permanent"])]]
    for s, p in zip(us["loan"], us["permanent"]):
        rows.append([s["reason"], f0(s["removed"]), f0(s["n"]), f0(p["removed"]), f0(p["n"])])
    rows.append(["**usable**", "", f"**{f0(inj['usable']['loan'])}**", "", f"**{f0(inj['usable']['permanent'])}**"])
    W += t(["Step", "Loans removed", "Loans left", "Permanents removed", "Permanents left"], rows, right={1, 2, 3, 4})
    W += [f"Usable: {f0(sum(inj['usable'].values()))} episodes ({f0(inj['usable_players'])} players); of these "
          f"{f0(sum(inj['usable_delta_zero'].values()))} have no game removed (fit share = current share). Regression "
          f"observations: {f0(sum(inj['model_sample'].values()))} (§8).", "",
          f"### Worked example: {an['player']}, column by column", ""]
    W += t(["Step", "Table", "Columns", "Value"], [
        ["Episode row", "Stage 1C", "event_id, player_id, from/to_club_name, transfer_date, transfer_type_raw",
         f"{an['event_id']}; {an['player_id']}; {an['a_club']} → {an['b_club']}; {an['move_date']}; {an['transfer_type_raw']} "
         f"({an['fee_display_raw']})"],
        ["Kind", "episode", "episode_type", "permanent"],
        ["Arrival season, league", "games", "competition_id, season, date, home/away_club_id",
         f"{an['league']} {an['arrival_season']} (B = club {an['b_club_id']})"],
        ["Window", "games, Stage 1C", "B's last league game; next move", f"{an['window_start']} to {an['window_end']}"
         + (f" (next move {an['departure_date']})" if an["departure_date"] else " (no earlier departure)")],
        ["Window games / complete", "games, appearances", "date; rows per (game, B)", f"{an['games']} / {an['complete_games']}"],
        ["Minutes", "appearances", "minutes_played (player, B, window games)", f"{f0(an['minutes'])} in {an['appearances']} appearances"],
        ["Current share", "derived", "minutes / (90 × games)", f"{f0(an['minutes'])} / (90 × {an['games']}) = {an['current_share']:.3f}"],
        ["Injury status", "Kaggle player_injuries", "player_id", f"{an['status']} (has injury rows)"],
        ["Spells touching the window", "Kaggle player_injuries", "injury_reason, from_date, end_date",
         "; ".join(f"{s['reason']} {s['from']}–{s['until']} ({'counts' if s['counts'] else 'not medical'})" for s in an["spell_rows"])],
        ["Games removed", "derived", "covered & no appearance", f"{an['removed']} (kept because he appeared: {an['conflicts']})"],
        ["Fit-adjusted share", "derived", "minutes / (90 × fit games)",
         f"{f0(an['minutes'])} / (90 × {an['fit_games']}) = {an['fit_share']:.3f}"],
        ["Usability", "derived", "data_quality_flag", an["flag"]],
        ["Player value", "player_valuations", "latest before the move", f"€{an['player_value']:,.0f} ({an['player_value_date']})"],
        ["Squad value", "player_valuations", "valuations listing B, 365 days", f"€{an['squad_value']:,.0f} ({an['squad_players']} players)"],
        ["Relative value / age / position / remain", "derived", "", f"{an['relative_value']:.4f} / {an['age']:.2f} / "
                                                                     f"{an['position']} / {an['remain']:.3f}"],
        ["In the injury regression", "derived", "all covariates present", "yes" if an["in_model"] else "no"]], right=set())
    W += t(["Game inside a spell", "Opponent", "Played?", "Minutes", "Spells covering", "Denominator"],
           [[g["date"], g["opponent"], "yes" if g["played"] else "no", "" if g["minutes"] is None else g["minutes"],
             g["covering_spells"], "removed" if g["removed"] else "kept (appeared)"] for g in an["spell_games"]],
           right={3, 4})

    # ---- 12 validation
    va = inj["validation"]
    W += ["## 12. Validation of the external injury data", "",
          "- **Events:** `game_events` rows with `type = 'Substitutions'` and 'Injury' in `description` (e.g. ', Injury'). "
          "`player_id` is the player going **off** (checked: his `minutes_played` equals the event minute on average to "
          "within a minute; `player_in_id` is the replacement).",
          "- **Scope:** league games inside a revised-sample window, for players with status A or B, dated before "
          f"{ia.SNAPSHOT.date()}, one event per (player, game).",
          "- **Match:** the external injury row (any label, start date known) nearest in start date; matched within k days "
          "if |start − event date| ≤ k.",
          "- **Stronger subset:** the player missed B's next two league games in the window.", ""]
    W += t(["Group", "Events", "±0 days", "±1", "±3", "±7"],
           [["All injury substitutions", f0(va["events"]), f0(va["match_0d"]), f0(va["match_1d"]), f0(va["match_3d"]),
             f0(va["match_7d"])]]
           + [[c, f0(v["events"]), f0(v["match_0d"]), f0(v["match_1d"]), f0(v["match_3d"]), f0(v["match_7d"])]
              for c, v in va["by_consequence"].items()], right={1, 2, 3, 4, 5})
    m2 = va["by_consequence"][ia.CONSEQ[0]]
    W += [f"Missed the next two league games: {f0(m2['match_7d'])} / {f0(m2['events'])} = {pc(m2['match_7d'], m2['events'])} "
          "matched within ±7 days; by period: " + "; ".join(f"{p} {f0(v['match_7d'])} / {f0(v['events'])} = "
                                                          f"{pc(v['match_7d'], v['events'])}"
                                                          for p, v in va["missed_two_by_period"].items()) + ".", "",
          "**Why this is coverage validation, not proof of completeness:** it tests only injuries visible as an in-match "
          "substitution, a minority (training injuries and illness never produce one). A match shows the source records "
          "that kind of injury; a miss may be a missing row, a short knock, or a tactical label. Unmatched injuries are "
          "not added and are never inferred from absence.", ""]

    # ---- 13 master
    mm = N["master"]
    W += ["## 13. Master episode-level table", "",
          f"`analysis_episode_master.csv`: **{f0(mm['rows'])} rows × {mm['columns']} columns**, one row per transfer episode "
          f"(`event_id` unique; {f0(mm['players'])} players). Built only by one-to-one joins on `event_id` (loan fields "
          "on `loan_event_id` = `event_id`; club countries on club id). Every column has a row in "
          "`analysis_variable_dictionary.csv`.", ""]
    W += t(["Group", "Columns"], [[k, v] for k, v in mm["sections"].items()])
    W += t(["Membership flag", "Episodes"], [[f"`{k}`", f0(v)] for k, v in mm["membership"].items()])
    W += ["**Kept out (cannot be reduced to one row per episode without duplication):** the window's game-by-game rows "
          "(`playing_time_injury_feasibility.episode_games`, rebuilt by `injury_method_example.py`), injury spells "
          "(`data/processed/external_injuries/salimt_kaggle_injuries_normalized.csv`), validation events "
          "(`injury_external_validation.csv`), the calendar clusters (`loan_calendar_rules_v2.csv`), the return rows "
          "(Stage 1C), and the club geography table (`club_geography_v2.csv`). Market value and injury columns are "
          "computed only for the revised sample; blank elsewhere means 'not computed', recorded per column.", ""]

    # ---- 14 diagrams
    W += ["## 14. Lineage diagrams", "", "### 1. Transfer / loan ending / calendar", "", "```",
          f"{f0(s1['rows'])} Stage 1C registration rows   (DuckDB transfers + raw fee labels)",
          "        │ role per row: internal " + f0(ep['steps'][0]['removed']) + ", placeholder " + f0(ep['steps'][1]['removed'])
          + ", draft " + f0(ep['steps'][2]['removed']),
          "        │ collapse " + f0(ep['steps'][3]['removed']) + " matched returns; set aside " + f0(ep['steps'][4]['removed'])
          + " unmatched", "        ▼",
          f"{f0(ep['episodes'])} transfer episodes ({f0(ep['episode_players'])} players)",
          "        │ episode_type == loan", "        ▼", f"{f0(lo['loans'])} loan episodes",
          f"        │ ending row exists (−{f0(lo['open'])} open)", "        ▼", f"{f0(lo['ending_observed'])} with a recorded ending",
          f"        │ not flagged future (−{f0(lo['scheduled_endings'])} scheduled)", "        ▼",
          f"{f0(v2['realised'])} realised endings          ──►  economic_ending (all loans; early_termination {f0(lo['early_termination'])})",
          f"        │ both club countries (geography v2; −{f0(v2['unmapped'])})", "        ▼", f"{f0(v2['mapped'])} geography-mapped",
          f"        │ has a calendar (−{f0(v2['below_threshold'])} below the {lcv2.PARAMS['N_MIN']}-ending threshold)",
          "        ▼", f"{f0(v2['classifiable'])} classifiable  ──►  {f0(v2['normal'])} calendar-normal  +  {f0(v2['off'])} off-calendar",
          "```", "", "### 2. Playing time", "", "```",
          f"{f0(ep['episodes'])} episodes  ──►  {f0(sum(pt['candidates'].values()))} candidates "
          f"({f0(pt['candidates']['loan'])} loans, {f0(pt['candidates']['permanent'])} permanents; −{ep['episode_type'].get('unknown_type', 0)} unknown type)"]
    for k in KINDS:
        st = wf[k]["steps"]
        W += [f"  {k}: {f0(st[0]['n'])} │ not scheduled ▼ {f0(st[1]['n'])} │ B in covered league season ▼ {f0(st[2]['n'])} "
              f"│ player and club rows exist ▼ {f0(st[4]['n'])} │ window has games, ends before capture, every game "
              f"complete, no other club ▼ {f0(wf[k]['reliable'])} reliable │ first spell"
              + (", loan return" if k == "loan" else "") + f" ▼ {f0(wf[k]['revised'])} revised"]
    W += ["  outcome: Σ appearances.minutes_played / (90 × B's league games in the window)", "```", "",
          "### 3. Market value / regression", "", "```",
          f"{f0(sum(rg['revised']['sample'].values()))} revised episodes",
          "        │ + player_valuations: player value (strictly before move), squad value (≥ 11 valued players at B, 365 days)",
          f"        │ drop missing age / position / player value / squad value (−{f0(N['age_model_missing']['dropped'])})",
          "        ▼", f"{f0(sum(rg['revised']['model'].values()))} regression observations (age models)",
          "        │ OLS share ~ loan [+ older + loan×older] + remain + age + log relative value + log squad value",
          "        │     + position + league×season FE; SE clustered by receiving club", "```", "",
          "### 4. Injury", "", "```",
          f"{f0(sum(inj['revised'].values()))} revised episodes ({f0(inj['revised_players'])} players)",
          "        │ exact player_id join to Kaggle salimt v2: "
          + ", ".join(f"{k} {f0(v)} players" for k, v in inj["status_players"].items()),
          "        │ window games × primary medical spells (inclusive), appearance overrides, union",
          "        │ usability: datalake, snapshot, truncation, open spell, missing start, fit games", "        ▼",
          f"{f0(sum(inj['usable'].values()))} injury-usable episodes",
          f"        │ covariates complete (−{f0(N['injury_model_missing']['dropped'])}, mostly no prior player valuation)",
          "        ▼", f"{f0(sum(inj['model_sample'].values()))} regression observations: current vs fit-adjusted share, same rows",
          "```", ""]

    # ---- 15 QA
    W += ["## 15. QA, definitions that differ from older notes, open lineage questions", "",
          "**Checks run by this module (the run fails if any fails):** Stage 1C SHA-256 before and after; every other file "
          "under `data/` hashed before and after (unchanged); row-role waterfall sums to the Stage 1C rows; loans with a "
          "matched return = return rows that closed a real loan; `reclassify` reproduces `economic_ending`; the windowed "
          "next move reproduces the production follow-on; the early-termination funnel reproduces the label; v1 and v2 "
          "calendar classes equal the saved files; the playing-time waterfall equals `playing_time_join_numbers.json`; the "
          "revised sample equals `playing_time_sample_redesign_numbers.json`; injury shares, flags, usable and model counts "
          "equal the saved injury outputs; the four preferred-model coefficients equal the saved ones; worked examples "
          "rebuild game by game; master `event_id` unique, one row per episode, every column in the dictionary, every "
          "derived variable has source columns and a rule.", "",
          "**Definitions that differ from older notes or comments:**", ""]
    diffs = [
        "`early_termination` uses 30 June **and** 31 December in the current loan code; Stage 2's `event_family` uses 30 "
        f"June only (§4: {f0(lo['early_termination_if_30_june_only'])} vs {f0(lo['early_termination'])}). The constant's "
        "comment in `event_family.py` still says 'from 30 June'.",
        "Two constants share the name `MAX_LOAN_DAYS`: 550 in the Stage 1 heuristic (`stage1/movement.py`) and 950 in "
        "Stage 2 / `loan_episodes` (used there only to stop trusting a spell after an unexplained move).",
        "The calendar history threshold counts **non-COVID** domestic endings (≥ 100). The older country-date analysis "
        "(`loan_country_dates.py`) used ≥ 100 realised domestic endings including COVID ones.",
        "'Next move' in the calendar tables (≤ 1 / 21 / 60 days, permanent to borrower, loan to a different club) is the "
        "unwindowed `next_move_*`, not the 60-day `follow_on_*` behind `economic_ending`.",
        "'Reliable' requires **every** league game in the window to have a complete record (and at least one game). The "
        "redesign summary's wording 'at least one complete covered league game' understates this.",
        "Age: Stage 1C `age_at_transfer` is completed years by birthday; the regression age is (days / 365.25) and the "
        "young/older split uses its floor. They can differ by one year within a day or two of a birthday.",
        "`age_c` is centred on a different sample mean in each module (old primary, revised model sample, injury model "
        "sample). The loan coefficient is unaffected.",
        "The 15-injury cap is inferred from the data, not documented; the code comment 'the source shows only the latest "
        "15' states it as fact.",
        f"In `playing_time_injury_adjusted_episodes.csv`, `age_group` labels the {inj['age_group_missing_age_labelled_older']} "
        "episodes without a date of birth as 'older (≥24)' (np.where on a missing age). They are excluded from every model; "
        "the master table's `older` is blank for them.",
        "The flag 'window reaches the 2025-09-15 snapshot' refers to the Kaggle injury file's date; the appearance data "
        "run later, so those windows have a valid playing-time outcome but no complete injury history.",
        "The Stage 1C build script's comment calls the canonical record '43-column'; the file has 42 columns.",
        "`clubs.total_market_value` is not only a current snapshot (the reason given in the descriptives) but NULL for "
        "every club in this snapshot."]
    W += [f"{i}. {d}" for i, d in enumerate(diffs, 1)] + [""]
    W += ["**Lineage that could not be fully established (marked `medium` in the dictionary):**", "",
          "- `transfermarkt_future_transfer`: upstream does not document the flag; we read it as 'not yet happened at capture'.",
          "- `market_value_eur` (Stage 1C): how upstream sets the transfer-row value is undocumented; it is not the "
          "regression player value.",
          "- `arrival_season` for a league's first snapshot season approximates the previous season's end as one year earlier.",
          "- The Kaggle injury rows' own collection method and the 15-row cap are undocumented by their author.", ""]

    # ---- 16 Q&A
    tzn = wf["loan"]["revised_true_zero"] + wf["permanent"]["revised_true_zero"]
    W += ["## 16. Questions you should be able to answer", ""]
    qa = [
        ("What is Stage 1C?", f"Our frozen table of all {f0(s1['rows'])} Transfermarkt registration movements of the "
         f"{f0(s1['players'])} players in the upstream snapshot, with Transfermarkt's fee labels and amounts restored."),
        ("What is one Stage 1C row?", "One registration movement of one player from one entry to another (club, youth or "
         "reserve side, or a placeholder like 'Without Club'), on one date."),
        ("What is one transfer episode?", "One real club-to-club move: a loan (its outbound row, with its return folded in) "
         "or a permanent move. Internal, placeholder, draft and return rows are not episodes."),
        ("Why are there fewer episodes than Stage 1C rows?", f"{f0(ep['steps'][3]['removed'])} return rows are folded into "
         f"their loans, and {f0(ep['steps'][0]['removed'] + ep['steps'][1]['removed'] + ep['steps'][2]['removed'] + ep['steps'][4]['removed'])} "
         f"rows are internal registrations, placeholder movements, drafts or unmatched returns: {f0(s1['rows'])} → "
         f"{f0(ep['episodes'])}."),
        ("What is one loan episode?", "One outbound loan from a lender to a borrower, paired one-to-one with the row that ended "
         "it (usually its End-of-loan row)."),
        ("Where does the loan ending date come from?", "From the Transfermarkt date on the terminal row (`terminal_date`), "
         "usually the End-of-loan row. It is often the scheduled end date, and may be in the future."),
        ("What is the early-termination flag?", "`economic_ending == 'early_termination'`: a return dated more than 10 days "
         "from 30 June and 31 December, followed within 21 days by another move that is not a permanent move back to the "
         "borrower or a paid/undisclosed sale."),
        ("Did Transfermarkt provide that flag?", "No. We built it from the row sequence. It proves no contract term."),
        ("What is off-calendar?", "A realised loan ending whose date is not within ±3 days of any date in its country's "
         "learned calendar (or the cross-border calendar)."),
        ("Is off-calendar a filter?", "No. It is a classification used in the calendar tables only; no sample is cut by it."),
        ("How do we calculate playing-time share?", "League minutes for the receiving club in the window divided by 90 × the "
         "receiving club's league games in the window."),
        ("How can playing-time share equal zero?", f"When every window game has a complete appearance record and the player "
         f"has no appearance row in any of them ({f0(tzn)} revised episodes)."),
        ("What makes an observation unreliable?", "A scheduled move, a club outside the 14 covered leagues, no window game, "
         "a window ending after the history capture, any incomplete club-game in the window, or appearances for another club."),
        ("Where do minutes come from?", "`appearances.minutes_played`, for the player and the receiving club, in its league "
         "games inside the window."),
        ("Where do available games come from?", "`games`: the receiving club's domestic-league games dated inside the window."),
        ("Where does age come from?", "`players.date_of_birth` (via Stage 1C) and the move date: days / 365.25."),
        ("Where does player market value come from?", "`player_valuations`: the latest valuation strictly before the move."),
        ("How is squad value constructed?", "Sum of the latest pre-move valuations (within 365 days) of every player whose "
         "valuation lists the receiving club, excluding the player; blank if fewer than 11."),
        ("What is relative value?", "Player value ÷ squad value; the models use its log."),
        ("What is an observation in the playing-time regression?", "One transfer episode (a loan or permanent arrival at one "
         "club), not a player and not a season."),
        ("What is loan × older?", f"How much the loan − permanent gap differs for arrivals aged {pta.CUTOFF + 1}+ compared "
         f"with those ≤ {pta.CUTOFF}."),
        ("Where do injury spells come from?", "The Kaggle salimt v2 file `player_injuries.csv` (from_date, end_date, "
         "injury_reason), joined on Transfermarkt player id."),
        ("How do we decide a game is injury-unavailable?", "A primary medical spell covers its date (inclusive) and the "
         "player has no appearance in it."),
        ("Does calendar classification affect the injury sample?", "No. No injury or playing-time module reads it."),
        ("Why does the injury-usable sample shrink further for regression?",
         f"{f0(N['injury_model_missing']['dropped'])} usable episodes lack a covariate, almost all a player valuation "
         f"before the move: {f0(sum(inj['usable'].values()))} → {f0(sum(inj['model_sample'].values()))}."),
        ("What is a unique player versus a transfer episode?", f"A player can have many episodes: the revised sample is "
         f"{f0(sum(inj['revised'].values()))} episodes of {f0(inj['revised_players'])} players. Count players with "
         "distinct player_id, episodes with event_id."),
    ]
    for q, a_ in qa:
        W += [f"**{q}** {a_}", ""]
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
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    return x


def main() -> dict:
    before = watched()
    N, tb = compute()
    N["watched_files"] = len(before)
    N["old_output_sha256"] = {Path(k).name: v for k, v in before.items() if k.startswith("data/outputs/rebuild/")
                              and "/" not in k[len("data/outputs/rebuild/"):]}
    tb["D1"].to_csv(STAGE1C_DICT_CSV, index=False)
    tb["INV"].to_csv(INVENTORY_CSV, index=False)
    INVENTORY_MD.write_text(render_inventory(tb["INV"]))
    tb["M"].to_csv(MASTER_CSV, index=False)
    tb["D2"].to_csv(VARDICT_CSV, index=False)
    LINEAGE_MD.write_text(render(N, tb))
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    calendar_first_rows(N["calendar_first"]).to_csv(CALFIRST_CSV, index=False)
    CALFIRST_MD.write_text(render_calendar_first(N["calendar_first"]))
    CORE_MD.write_text(render_core(tb["D2"]))
    tb["AC"].to_csv(ACADEMY_CSV, index=False)
    tb["AR"].to_csv(ACADEMY_REVIEW_CSV, index=False)
    if N["academy"]["review"] is not None:
        ACADEMY_MD.write_text(co.render(N["academy"], N["club_organisation"], tb["AR"]))
    after = watched()
    changed = sorted(k for k in before if before[k] != after.get(k))
    assert not changed, f"files changed: {changed}"
    assert g2.sha256(CANONICAL_CSV) == g2.STAGE1C_SHA256
    print(f"Stage 1C: {N['stage1c']['rows']:,} rows x {N['stage1c']['columns']} columns (hash unchanged)")
    print(f"master: {N['master']['rows']:,} rows x {N['master']['columns']} columns; {len(before)} other files unchanged")
    return N


if __name__ == "__main__":
    main()
