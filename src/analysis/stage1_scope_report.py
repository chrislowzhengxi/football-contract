"""Stage 1 dataset scoping report: python -m src.analysis.stage1_scope_report

Deterministic. Reads the frozen Stage 1C table and the upstream DuckDB
snapshot (read-only), writes CSVs and a Markdown summary to
data/outputs/rebuild/, and prints a terminal summary. No network, no LLM.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from ..config import DEFAULT_DATABASE
from ..stage2.event_family import (FOLLOW_ON_IMMEDIATE_DAYS, FOLLOW_ON_WINDOW_DAYS,
                                   SEASON_END_TOLERANCE_DAYS, _days_from_season_end)
from .loan_episodes import (CANONICAL_CSV, ECONOMIC_ENDINGS, EXPECTED_ROWS, ROOT,
                            build_universe, club_geography, load_canonical)
from .loan_scope import NONSTANDARD, scope, token_profile, workload

OUT = ROOT / "data" / "outputs" / "rebuild"
AUDIT_SEED = 20260924


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(n, d) -> str:
    return f"{n:,} / {d:,} = {100 * n / d:.1f}%" if d else f"{n:,} / 0"


def p(n, d) -> float:
    return round(100 * n / d, 4) if d else float("nan")


# ---------------------------------------------------------------------------

def upstream_facts() -> dict:
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        q = lambda s: con.sql(s).fetchall()
        comps = con.sql("""select competition_id, name, type, country_name, confederation
                           from competitions order by type, country_name""").df()
        seasons = con.sql("""select c.type, min(g.season) first_season, max(g.season) last_season
                             from games g join competitions c using(competition_id)
                             group by 1""").df()
        return {
            "commit": q("select commit_hash from version")[0][0],
            "players": q("select count(*) from players")[0][0],
            "players_with_transfers": q("select count(distinct player_id) from transfers")[0][0],
            "transfer_rows": q("select count(*) from transfers")[0][0],
            "clubs_table": q("select count(*) from clubs")[0][0],
            "games": q("select count(*) from games")[0][0],
            "max_game_date": str(q("select max(date) from games")[0][0]),
            "competitions": comps, "game_seasons": seasons,
        }
    finally:
        con.close()


def waterfall(u) -> pd.DataFrame:
    r = u.rows
    role = r.row_role
    raw = len(r)
    internal = r[role == "internal_registration"]
    placeholder = r[role == "placeholder_non_club_movement"]
    rows = [
        ("raw Stage 1C rows", raw, "every registration movement in the frozen table"),
        ("- internal registration moves", -len(internal),
         "same organisation on both sides (Stage 1C `is_internal_move`): "
         f"{int((internal.transfer_type_normalized == 'youth_or_internal').sum()):,} unlabelled youth/reserve moves, "
         f"{int((internal.transfer_type_normalized == 'loan').sum())} internal loans, "
         f"{int((internal.transfer_type_normalized == 'loan_return').sum())} internal returns, "
         f"{int((internal.transfer_type_normalized == 'free_transfer').sum())} other"),
        ("- placeholder-club movements", -len(placeholder),
         "one side is Without Club / Retired / Career break / Ban / Unknown, so there is no "
         f"counterparty club: {int(placeholder.from_club_name.isin(['Without Club','Retired','Career break','Ban','Unknown']).sum()):,} "
         f"arrivals from, {int(placeholder.to_club_name.isin(['Without Club','Retired','Career break','Ban','Unknown']).sum()):,} "
         "departures to a placeholder (20 rows have one on both sides)"),
        ("- draft rows", -int((role == "non_transfer_other").sum()),
         "Transfermarkt label `draft` (MLS SuperDraft etc.): an allocation, not a club-to-club transfer"),
        ("- loan returns paired with their loan and collapsed", -int((role == "loan_return_collapsed").sum()),
         "each return is folded into the outbound loan it closes; the loan is the episode"),
        ("- loan returns with no outbound loan to close", -int((role == "loan_return_unmatched").sum()),
         "kept aside, not counted as transfers (see pairing audit)"),
    ]
    ep = int((role == "episode").sum())
    rows.append(("= real transfer episodes", ep, "one per remaining row"))
    df = pd.DataFrame(rows, columns=["step", "rows", "definition"])
    assert raw + df.rows.iloc[1:-1].sum() == ep, "waterfall does not reconcile"
    return df


def episode_type_counts(u) -> pd.DataFrame:
    ep = u.episodes
    t = ep.episode_type.value_counts().rename_axis("episode_type").reset_index(name="episodes")
    t["pct_of_episodes"] = (100 * t.episodes / len(ep)).round(2)
    return t


def sensitivities(u) -> dict:
    ep, L = u.episodes, u.loans
    youth = ep.is_youth_or_reserve_side.fillna(False)
    conv_follow = set(L.loc[L.economic_ending == "purchase_option_or_permanent_conversion",
                            "follow_on_event_id"].dropna())
    return {
        "episodes_involving_youth_or_reserve_side": int(youth.sum()),
        "episodes_senior_only": int((~youth).sum()),
        "permanent_episodes_that_follow_a_loan_conversion": int(ep.event_id.isin(conv_follow).sum()),
        "episodes_flagged_future_by_transfermarkt": int(ep.transfermarkt_future_transfer.sum()),
    }


def loan_outcomes(u) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    L = u.loans
    n = len(L)
    ended = L[L.terminal_event_id.notna()]
    # A. raw label
    a = ended.raw_label_class.value_counts().rename_axis("raw_label_class").reset_index(name="loans")
    a["denominator"] = len(ended)
    a["pct"] = (100 * a.loans / len(ended)).round(2)
    a["grouping"] = a.raw_label_class.map(
        lambda c: "ordinary End of loan" if c == "End of loan"
        else ("missing/unknown label" if c in ("missing",) else "non-End-of-loan"))
    a.insert(0, "answer", "A_raw_label")
    # B. economic
    plain = L.raw_label_class == "End of loan"
    near = L.return_boundary_days.le(SEASON_END_TOLERANCE_DAYS)
    b = []
    for cat in ECONOMIC_ENDINGS:
        s = L.economic_ending == cat
        b.append({"answer": "B_economic", "economic_ending": cat, "loans": int(s.sum()),
                  "denominator": n, "pct": p(int(s.sum()), n),
                  "of_which_plain_end_of_loan_row": int((s & plain).sum()),
                  "of_which_plain_end_of_loan_within_10d_of_30jun_31dec": int((s & plain & near).sum()),
                  "of_which_fee_bearing_return": int((s & L.ending_fee_on_return_eur.notna()).sum()),
                  "of_which_scheduled_future_ending": int((s & L.terminal_scheduled_future).sum())})
    b = pd.DataFrame(b)
    fees = L.ending_fee_on_return_eur.dropna()
    fee = {"count": int(len(fees)), "median": fees.median(), "mean": fees.mean(),
           "p25": fees.quantile(.25), "p75": fees.quantile(.75), "p90": fees.quantile(.9),
           "max": fees.max(), "min": fees.min()}
    return a, b, fee


def stage2_rule_sensitivity(u) -> dict:
    L = u.loans
    ret = L[L.terminal_is_return]
    near_jun_only = ret.terminal_date.map(_days_from_season_end) <= SEASON_END_TOLERANCE_DAYS
    near_both = ret.return_boundary_days <= SEASON_END_TOLERANCE_DAYS
    follow_free_third = (L.follow_on_type.isin(["free_transfer", "no_fee_shown"])
                         & (L.follow_on_to_club != L.borrower_club)
                         & L.follow_on_gap_days.le(FOLLOW_ON_WINDOW_DAYS))
    early_under_jun_only = ret[(~near_jun_only) & near_both & ret.follow_on_gap_days.le(FOLLOW_ON_IMMEDIATE_DAYS)
                               & ~ret.economic_ending.isin(["purchase_option_or_permanent_conversion",
                                                             "third_party_sale_related"])]
    return {
        "returns_near_31dec_not_30jun_with_immediate_follow_on": int(len(early_under_jun_only)),
        "free_or_no_fee_moves_to_third_club_after_return": int(follow_free_third.sum()),
    }


# ---------------------------------------------------------------------------

def date_buckets(frame: pd.DataFrame) -> dict:
    md = frame.terminal_date.dt.strftime("%m-%d")
    n = len(frame)
    j, d = int((md == "06-30").sum()), int((md == "12-31").sum())
    m = int((md == "05-31").sum())
    return {"N": n, "jun30_n": j, "jun30_pct": p(j, n), "dec31_n": d, "dec31_pct": p(d, n),
            "jun30_or_dec31_n": j + d, "jun30_or_dec31_pct": p(j + d, n),
            "other_dates_n": n - j - d, "other_dates_pct": p(n - j - d, n),
            # Not asked for, but it is the second most common date: British
            # loans are recorded as ending on 31 May.
            "of_other_may31_n": m, "of_other_may31_pct": p(m, n)}


def europe_end_dates(u) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    L = u.loans
    eu_both = (L.from_confederation == "UEFA") & (L.to_confederation == "UEFA")
    E = L[eu_both]
    ended = E[E.terminal_date.notna()]
    coverage = {
        "all_loan_episodes": int(len(L)),
        "loans_with_a_club_of_unknown_country": int((L.from_country.isna() | L.to_country.isna()).sum()),
        "loans_both_clubs_mapped": int((L.from_country.notna() & L.to_country.notna()).sum()),
        "loans_both_clubs_uefa": int(len(E)),
        "uefa_loans_with_matched_ending": int(len(ended)),
        "uefa_loans_without_ending": int(len(E) - len(ended)),
        "uefa_loans_ending_scheduled_future": int(ended.terminal_scheduled_future.sum()),
    }
    pops = [
        ("all matched endings", ended),
        ("raw label: plain End of loan", ended[ended.raw_label_class == "End of loan"]),
        ("raw label: not plain End of loan", ended[ended.raw_label_class != "End of loan"]),
        ("fee-bearing returns", ended[ended.ending_fee_on_return_eur.notna()]),
        ("economic: ordinary_end_of_loan", ended[ended.economic_ending == "ordinary_end_of_loan"]),
        ("economic: any nonstandard class", ended[ended.economic_ending.isin(NONSTANDARD)]),
    ] + [(f"economic: {c}", ended[ended.economic_ending == c]) for c in NONSTANDARD]
    rows = []
    for name, fr in pops:
        for sl, sub in (("all", fr), ("realised", fr[~fr.terminal_scheduled_future]),
                        ("scheduled (future-flagged)", fr[fr.terminal_scheduled_future])):
            rows.append({"population": name, "slice": sl, **date_buckets(sub)})
    by_season = []
    for s, g in ended.groupby("loan_season"):
        by_season.append({"loan_season": s, **date_buckets(g),
                          "nonstandard_N": int(g.economic_ending.isin(NONSTANDARD).sum()),
                          **{f"nonstandard_{k}": v for k, v in
                             date_buckets(g[g.economic_ending.isin(NONSTANDARD)]).items() if k != "N"}})
    by_country = []
    for c, g in ended.groupby("from_country"):
        if len(g) >= 50:
            by_country.append({"lender_country": c, **date_buckets(g)})
    coverage["by_lender_country"] = (pd.DataFrame(by_country)
                                     .sort_values("N", ascending=False).reset_index(drop=True))
    top_dates = ended.terminal_date.dt.strftime("%m-%d").value_counts().head(8)
    coverage["top_dates"] = top_dates
    return pd.DataFrame(rows), coverage, pd.DataFrame(by_season)


# ---------------------------------------------------------------------------

def coverage_tables(u) -> pd.DataFrame:
    ep = u.episodes
    frames = []
    s = ep.groupby("season").size().rename("episodes").reset_index()
    s["loans"] = s.season.map(ep[ep.episode_type == "loan"].groupby("season").size()).fillna(0).astype(int)
    frames.append(s.assign(breakdown="season"))
    for side in ("from", "to"):
        c = ep.groupby(ep[f"{side}_country"].fillna("(unmapped)")).size().rename("episodes").reset_index()
        c = c.rename(columns={f"{side}_country": "country"})
        frames.append(c.assign(breakdown="country_of_selling_or_lending_club" if side == "from"
                               else "country_of_buying_or_borrowing_club"))
    dom = ep[ep.from_country.notna() & (ep.from_country == ep.to_country)]
    frames.append(dom.groupby("from_country").size().rename("episodes").reset_index()
                  .rename(columns={"from_country": "country"}).assign(breakdown="country_domestic_both_clubs"))
    cs = ep.assign(country=ep.from_country.fillna("(unmapped)")).groupby(["country", "season"]).size()
    frames.append(cs.rename("episodes").reset_index().assign(breakdown="country_x_season_by_from_club"))
    ts = ep.groupby(["episode_type", "season"]).size().rename("episodes").reset_index()
    frames.append(ts.assign(breakdown="transfer_type_x_season"))
    out = pd.concat(frames, ignore_index=True)
    cols = ["breakdown", "country", "season", "episode_type", "episodes", "loans"]
    return out.reindex(columns=cols)


def loans_by_season(u) -> pd.DataFrame:
    L = u.loans
    g = L.groupby("loan_season")
    out = pd.DataFrame({
        "loans": g.size(),
        "with_matched_return": g.has_matched_return.sum(),
        "with_other_ending_event": g.apply(lambda x: int((x.terminal_event_id.notna() & ~x.has_matched_return).sum())),
        "open_or_unmatched": g.apply(lambda x: int(x.terminal_event_id.isna().sum())),
        "raw_label_not_plain_end_of_loan": g.apply(lambda x: int(((x.raw_label_class != "End of loan")
                                                                   & x.terminal_event_id.notna()).sum())),
        "fee_bearing_returns": g.ending_fee_on_return_eur.count(),
        "scheduled_future_endings": g.terminal_scheduled_future.sum(),
        "uefa_both_clubs": g.apply(lambda x: int(((x.from_confederation == "UEFA")
                                                  & (x.to_confederation == "UEFA")).sum())),
    })
    for cat in ECONOMIC_ENDINGS:
        out[cat] = g.economic_ending.apply(lambda x, c=cat: int((x == c).sum()))
    out["nonstandard_any"] = out[list(NONSTANDARD)].sum(axis=1)
    out["nonstandard_pct"] = (100 * out.nonstandard_any / out.loans).round(2)
    return out.reset_index()


def loans_by_country(u) -> pd.DataFrame:
    L = u.loans
    rows = []
    for side, col in (("lender", "from_country"), ("borrower", "to_country")):
        for c, g in L.groupby(L[col].fillna("(unmapped)")):
            rows.append({"country": c, "role": side, "loans": len(g),
                         "domestic_both_clubs_same_country": int((g.from_country == g.to_country).sum()),
                         "ordinary_end_of_loan": int((g.economic_ending == "ordinary_end_of_loan").sum()),
                         "nonstandard_any": int(g.economic_ending.isin(NONSTANDARD).sum()),
                         "purchase_option_or_permanent_conversion":
                             int((g.economic_ending == "purchase_option_or_permanent_conversion").sum()),
                         "fee_bearing_returns": int(g.ending_fee_on_return_eur.notna().sum()),
                         "open_or_unresolved": int((g.economic_ending == "unresolved").sum())})
    out = pd.DataFrame(rows)
    out["nonstandard_pct"] = (100 * out.nonstandard_any / out.loans).round(2)
    return out.sort_values(["role", "loans"], ascending=[True, False])


def loans_by_country_season(u) -> pd.DataFrame:
    L = u.loans
    rows = []
    for side, col in (("lender", "from_country"), ("borrower", "to_country")):
        for (c, s), g in L.groupby([L[col].fillna("(unmapped)"), "loan_season"]):
            rows.append({"country": c, "role": side, "loan_season": s, "loans": len(g),
                         "nonstandard_any": int(g.economic_ending.isin(NONSTANDARD).sum()),
                         "fee_bearing_returns": int(g.ending_fee_on_return_eur.notna().sum())})
    return pd.DataFrame(rows)


def country_flows(u) -> pd.DataFrame:
    L = u.loans
    f = L.assign(lender_country=L.from_country.fillna("(unmapped)"),
                 borrower_country=L.to_country.fillna("(unmapped)"))
    g = f.groupby(["lender_country", "borrower_country"])
    out = pd.DataFrame({"loans": g.size(),
                        "nonstandard_any": g.economic_ending.apply(lambda x: int(x.isin(NONSTANDARD).sum())),
                        "fee_bearing_returns": g.ending_fee_on_return_eur.count()}).reset_index()
    return out.sort_values("loans", ascending=False)


# ---------------------------------------------------------------------------

AUDIT_COLS = ["audit_stratum", "row_kind", "player", "player_id", "loan_event_id", "outbound_date",
              "loan_season", "lending_club", "lending_country", "borrowing_club", "borrowing_country",
              "return_or_end_date", "return_from", "return_to", "raw_outbound_label",
              "raw_ending_label", "raw_label_class", "fee_on_return_eur", "duration_days",
              "matching_reason", "endpoint_classification", "ending_detail",
              "follow_on_to_club", "follow_on_gap_days", "ending_scheduled_future"]


def pairing_audit(u) -> pd.DataFrame:
    L = u.loans.copy()
    L["_bucket"] = (L.loan_season.str[:4].astype(int) // 5) * 5
    picks = []

    def take(stratum, frame, n):
        if len(frame):
            picks.append(frame.sample(min(n, len(frame)), random_state=AUDIT_SEED).assign(audit_stratum=stratum))

    for cat, n in (("ordinary_end_of_loan", 14), ("purchase_option_or_permanent_conversion", 12),
                   ("third_party_sale_related", 12), ("early_termination", 12),
                   ("immediate_follow_on_transfer", 12), ("fee_bearing_return", 18),
                   ("other_nonstandard", 12), ("unresolved", 16)):
        take(f"economic:{cat}", L[L.economic_ending == cat], n)
    take("fee-bearing return (any class)", L[L.ending_fee_on_return_eur.notna()
                                             & (L.economic_ending != "fee_bearing_return")], 10)
    take("match: same organisation", L[L.match_reason == "reversed_same_organisation"], 8)
    take("match: returned to other side of lender",
         L[L.match_reason == "returned_from_borrower_to_other_side_of_lender"], 8)
    take("raw label not plain End of loan", L[(L.raw_label_class != "End of loan")
                                              & L.terminal_event_id.notna()], 10)
    for b, g in L.groupby("_bucket"):
        take(f"season bucket {b}-{b + 4}", g, 3)
    take("country unmapped", L[L.from_country.isna() | L.to_country.isna()], 6)
    take("non-UEFA mapped", L[L.from_country.notna() & L.to_country.notna()
                              & ((L.from_confederation != "UEFA") | (L.to_confederation != "UEFA"))], 6)
    take("scheduled future ending", L[L.terminal_scheduled_future], 6)
    s = pd.concat(picks).drop_duplicates("loan_event_id")
    audit = pd.DataFrame({
        "audit_stratum": s.audit_stratum, "row_kind": "loan_episode", "player": s.player_name,
        "player_id": s.player_id, "loan_event_id": s.loan_event_id,
        "outbound_date": s.loan_date.dt.date, "loan_season": s.loan_season,
        "lending_club": s.lender_club, "lending_country": s.from_country,
        "borrowing_club": s.borrower_club, "borrowing_country": s.to_country,
        "return_or_end_date": s.terminal_date.dt.date, "return_from": s.terminal_from_club,
        "return_to": s.terminal_to_club, "raw_outbound_label": s.loan_raw_label,
        "raw_ending_label": s.terminal_raw_label, "raw_label_class": s.raw_label_class,
        "fee_on_return_eur": s.ending_fee_on_return_eur, "duration_days": s.duration_days,
        "matching_reason": s.match_reason, "endpoint_classification": s.economic_ending,
        "ending_detail": s.ending_detail, "follow_on_to_club": s.follow_on_to_club,
        "follow_on_gap_days": s.follow_on_gap_days, "ending_scheduled_future": s.terminal_scheduled_future,
    })
    # Unmatched returns are the ambiguous cases the pairing refused to force.
    r = u.rows[u.rows.row_role == "loan_return_unmatched"]
    um = pd.DataFrame({
        "audit_stratum": "unmatched return (not forced onto a loan)", "row_kind": "unmatched_return",
        "player": r.player_name, "player_id": r.player_id, "loan_event_id": None,
        "outbound_date": None, "loan_season": r.season, "lending_club": None, "lending_country": None,
        "borrowing_club": None, "borrowing_country": None, "return_or_end_date": r._date.dt.date,
        "return_from": r.from_club_name, "return_to": r.to_club_name, "raw_outbound_label": None,
        "raw_ending_label": r.fee_display_raw, "raw_label_class": None,
        "fee_on_return_eur": r.fee_on_return_eur, "duration_days": None,
        "matching_reason": "no open loan from this club", "endpoint_classification": "unmatched_return",
        "ending_detail": None, "follow_on_to_club": None, "follow_on_gap_days": None,
        "ending_scheduled_future": r.transfermarkt_future_transfer,
    })
    return pd.concat([audit, um], ignore_index=True)[AUDIT_COLS]


# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame, cols=None) -> str:
    df = df if cols is None else df[cols]
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---"
                            for c in df.columns) + " |"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float) and not pd.isna(v) and float(v).is_integer() and abs(v) > 99:
                v = f"{int(v):,}"
            elif isinstance(v, (int,)) and abs(v) > 999:
                v = f"{v:,}"
            elif isinstance(v, float) and pd.isna(v):
                v = ""
            cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def eur(x) -> str:
    return "" if pd.isna(x) else (f"€{x / 1e6:,.2f}m" if x >= 1e6 else f"€{x / 1e3:,.0f}k")


def write_report(ctx: dict) -> str:
    u, up, wf = ctx["u"], ctx["upstream"], ctx["waterfall"]
    ep, L = u.episodes, u.loans
    n_ep, n_loan = len(ep), len(L)
    ended = L.terminal_event_id.notna()
    ret = L.has_matched_return
    a, b, fee = ctx["outcomes"]
    eu, eucov, eu_season = ctx["europe"]
    eall = eu[(eu.population == "all matched endings") & (eu.slice == "all")].iloc[0]
    eal_real = eu[(eu.population == "all matched endings") & (eu.slice == "realised")].iloc[0]
    ens = eu[(eu.population == "economic: any nonstandard class") & (eu.slice == "all")].iloc[0]
    eraw = eu[(eu.population == "raw label: not plain End of loan") & (eu.slice == "all")].iloc[0]
    efee = eu[(eu.population == "fee-bearing returns") & (eu.slice == "all")].iloc[0]
    eord = eu[(eu.population == "raw label: plain End of loan") & (eu.slice == "all")].iloc[0]
    plain = int((a[a.raw_label_class == "End of loan"].loans).sum())
    nonplain = int(ended.sum()) - plain
    econ_ns = int(L.economic_ending.isin(NONSTANDARD).sum())
    types = ep.episode_type.value_counts()
    sens, s2 = ctx["sensitivities"], ctx["stage2_sensitivity"]
    wk = ctx["workload_all"]
    tp = wk["observed"]["current_prompt"] if wk.get("available") else {}
    dates = ep._date
    geo = ctx["geo"]
    mapped_clubs = set(geo.club_id)
    ep_clubs = set(ep.from_club_id) | set(ep.to_club_id)
    W = []
    w = W.append

    w("# Stage 1 dataset scope: what the Transfermarkt data contains")
    w("")
    w("*Deterministic analysis of the frozen Stage 1C table "
      f"(`stage1c_canonical_transfers.csv`, {EXPECTED_ROWS:,} rows, SHA-256 "
      f"`{ctx['sha'][:16]}…`, unchanged by this analysis). No web search, no LLM.*")
    w("")
    w("## Answers at a glance")
    w("")
    w("| Question | Answer | Numerator / denominator |")
    w("| --- | --- | --- |")
    w(f"| Raw movement rows | **{EXPECTED_ROWS:,}** | Transfermarkt registration movements |")
    w(f"| Real transfer episodes | **{n_ep:,}** | rows minus internal, placeholder, draft and loan-return rows (§1) |")
    w(f"| Loans | **{n_loan:,}** | {pct(n_loan, n_ep)} of real transfer episodes |")
    w(f"| Loans with a matched return row | **{int(ret.sum()):,}** | {pct(int(ret.sum()), n_loan)} of loans |")
    w(f"| Loans with no recorded ending | **{int((~ended).sum()):,}** | {pct(int((~ended).sum()), n_loan)} of loans |")
    w(f"| Loans whose ending row is not a plain `End of loan` (raw label) | **{nonplain:,}** | "
      f"{pct(nonplain, int(ended.sum()))} of loans with an ending |")
    w(f"| … of which `End of loan` + a fee | **{fee['count']}** | {pct(fee['count'], int(ended.sum()))} |")
    w(f"| Loans in a non-ordinary sequence pattern (economic, broad) | **{econ_ns:,}** | "
      f"{pct(econ_ns, n_loan)} of loans — mostly a plain return followed by a new move (§3B) |")
    w(f"| European loans (both clubs UEFA) with an ending, N | **{int(eall.N):,}** | "
      f"{pct(int(eall.N), eucov['all_loan_episodes'])} of all loans; see §4 for coverage |")
    w(f"| … ending 30 June | **{eall.jun30_pct:.1f}%** | {int(eall.jun30_n):,} / {int(eall.N):,} |")
    w(f"| … ending 31 December | **{eall.dec31_pct:.1f}%** | {int(eall.dec31_n):,} / {int(eall.N):,} |")
    w(f"| … ending 30 June or 31 December | **{eall.jun30_or_dec31_pct:.1f}%** | "
      f"{int(eall.jun30_or_dec31_n):,} / {int(eall.N):,} |")
    w(f"| … same, non-ordinary sequence endings only | **{ens.jun30_pct:.1f}% / {ens.dec31_pct:.1f}% / "
      f"{ens.jun30_or_dec31_pct:.1f}%** | 30 Jun / 31 Dec / either, N = {int(ens.N):,} |")
    w(f"| Players / clubs / seasons | **{ep.player_id.nunique():,} / {len(ep_clubs):,} / "
      f"{ep.season.nunique()}** | in real transfer episodes |")
    w(f"| Date range | **{dates.min().date()} to {dates.max().date()}** | includes "
      f"{sens['episodes_flagged_future_by_transfermarkt']:,} episodes Transfermarkt flags as future |")
    w("")
    w("Two things to hold in mind while reading the numbers:")
    w("")
    w(f"1. **The dataset is career histories, not a census of any league.** Every row belongs to one of "
      f"{up['players_with_transfers']:,} players who appeared in a competition the upstream snapshot covers "
      f"between 2012/13 and 2025/26; transfers back to 1993 are those players' earlier careers. Counts "
      f"before about 2012 are therefore survivor samples and rise over time for that reason, not because "
      f"the market grew.")
    w(f"2. **Club country is known only for {len(mapped_clubs & ep_clubs):,} of {len(ep_clubs):,} clubs** "
      f"(those in a covered first-tier league or that played a covered national cup). Both clubs' "
      f"countries are known for {pct(int((ep.from_country.notna() & ep.to_country.notna()).sum()), n_ep)} "
      f"of episodes and {pct(int((L.from_country.notna() & L.to_country.notna()).sum()), n_loan)} of loans. "
      f"Lower-league and youth sides are the usual gaps, so any \"both clubs in country X\" filter "
      f"under-counts domestic lower-league loans.")
    w("")

    # ---- 1
    w("## 1. How many transfers are in the dataset?")
    w("")
    r = u.rows
    all_clubs = set(r.from_club_id) | set(r.to_club_id)
    w(f"- Raw Stage 1C rows: **{len(r):,}**")
    w(f"- Unique players: **{r.player_id.nunique():,}** (all rows) / {ep.player_id.nunique():,} (in real transfer episodes)")
    w(f"- Unique clubs (Transfermarkt club ids): **{len(all_clubs):,}** (all rows) / {len(ep_clubs):,} (in episodes)")
    extra = sorted(set(r.season) - set(ep.season))
    w(f"- Seasons: **{r.season.nunique()}**, from {min(r.season)} to {max(r.season)}"
      + (f" ({', '.join(extra)} appears only on scheduled future rows, so episodes span "
         f"{ep.season.nunique()})" if extra else ""))
    w(f"- Transfer dates: **{r._date.min().date()}** to **{r._date.max().date()}**. "
      f"{int(r.transfermarkt_future_transfer.sum()):,} rows carry Transfermarkt's `futureTransfer` flag — "
      f"{int((r.transfermarkt_future_transfer & (r.transfer_type_normalized == 'loan_return')).sum()):,} "
      "of them are loan returns, i.e. the *scheduled* end of a loan as recorded when the page was scraped. "
      f"The snapshot's last match is {up['max_game_date']}.")
    w("")
    w("### Waterfall from movement rows to real transfer episodes")
    w("")
    w(md_table(wf))
    w("")
    w("Every exclusion uses a flag Stage 1C already carries; none was invented for this count. "
      "Loan returns are not transfers: each is either folded into the loan it ends or, "
      "if no loan can be found for it, set aside and listed in the pairing audit.")
    w("")
    w("Real transfer episodes by type:")
    w("")
    t = ctx["types"]
    w(md_table(t))
    w("")
    w("`no_fee_shown` is Transfermarkt's `-`: the fee cell is empty. It is kept as its own type, not "
      "treated as a free transfer or a zero fee.")
    w("")
    w("Sensitivity lines, for alternative definitions of a \"real transfer\":")
    w("")
    w(f"- {sens['episodes_involving_youth_or_reserve_side']:,} episodes involve a youth or reserve side at a "
      f"*different* organisation (e.g. Man City U18 → Dortmund). They are counted, because they are genuine "
      f"inter-club moves; senior-only episodes number {sens['episodes_senior_only']:,}.")
    w(f"- {sens['permanent_episodes_that_follow_a_loan_conversion']:,} permanent episodes are the second half "
      "of a loan-then-permanent sequence to the same club (§3B). They are counted as their own episode; "
      "subtract them if a loan and its conversion should be one episode.")
    w("")

    # ---- 2
    w("## 2. How many real transfers are loans?")
    w("")
    other_end = int((ended & ~ret).sum())
    w(f"- Real transfer episodes: **{n_ep:,}**")
    w(f"- Loan episodes: **{n_loan:,}** — {pct(n_loan, n_ep)}. A loan counts once, on its outbound row.")
    w(f"- Matched to an `End of loan` return row: **{int(ret.sum()):,}** — {pct(int(ret.sum()), n_loan)}")
    w(f"- Ended by a different movement (conversion, re-loan, departure from the borrower, sale by the "
      f"lender): **{other_end:,}** — {pct(other_end, n_loan)}")
    w(f"- No ending recorded (open or unresolved): **{int((~ended).sum()):,}** — {pct(int((~ended).sum()), n_loan)}")
    w(f"- Of the matched endings, {int(L.terminal_scheduled_future.sum()):,} are scheduled (future-flagged) "
      "returns rather than realised ones.")
    w("")
    mr = L.match_reason.value_counts().rename_axis("how the ending was matched").reset_index(name="loans")
    mr["pct of loans"] = (100 * mr.loans / n_loan).round(2)
    w(md_table(mr))
    w("")
    w("A return is matched to the open loan whose borrowing club it leaves, preferring exact reversed "
      "club ids, then the same organisation (e.g. a youth-side loan returning to the senior side). "
      "Sub-loans (A→B, B→C, C→B, B→A) are tracked as nested spells. One return closes at most one loan "
      "and each loan is counted once; both are tested.")
    w("")

    # ---- 3
    w("## 3. How many loans end with something other than an ordinary `End of loan`?")
    w("")
    w("### A. Strict raw-label answer")
    w("")
    w(f"Transfermarkt's own fee cell on the row that ended the loan. Denominator: the "
      f"{int(ended.sum()):,} loans with an ending (the {int((~ended).sum())} open loans have no ending row).")
    w("")
    w(md_table(a, ["raw_label_class", "grouping", "loans", "pct"]))
    w("")
    w(f"**Ordinary `End of loan`: {pct(plain, int(ended.sum()))}. Non-`End of loan`: "
      f"{pct(nonplain, int(ended.sum()))}. Missing label: 0.** Transfermarkt return rows are only ever "
      "labelled `End of loan`, with or without a fee, so a different label can only appear when the loan "
      "was ended by a different kind of movement. `?`, `-`, `0`, a stated amount and `free transfer` are "
      "kept separate; none is converted to zero.")
    w("")
    w("### B. Economic / event-family answer")
    w("")
    w(f"The sequence around the ending, using Stage 2's event-family windows (follow-on within "
      f"{FOLLOW_ON_IMMEDIATE_DAYS} days is immediate, within {FOLLOW_ON_WINDOW_DAYS} days is related; a "
      f"return within {SEASON_END_TOLERANCE_DAYS} days of a season boundary is a scheduled expiry). "
      f"Denominator: all {n_loan:,} loans. **These are registration patterns, not contract terms**: "
      "a return followed by a permanent move to the borrower is *consistent with* an option or obligation, "
      "but no mechanism is inferred.")
    w("")
    bb = b.rename(columns={"economic_ending": "class", "of_which_plain_end_of_loan_row": "ended by plain End of loan",
                           "of_which_plain_end_of_loan_within_10d_of_30jun_31dec": "… within 10d of 30 Jun/31 Dec",
                           "of_which_fee_bearing_return": "fee-bearing",
                           "of_which_scheduled_future_ending": "scheduled"})
    w(md_table(bb, ["class", "loans", "pct", "ended by plain End of loan", "… within 10d of 30 Jun/31 Dec",
                    "fee-bearing", "scheduled"]))
    w("")
    w("How each class is defined:")
    w("")
    w("| Class | Rule |")
    w("| --- | --- |")
    w("| ordinary_end_of_loan | plain `End of loan` return, no follow-on pattern below |")
    w("| fee_bearing_return | `End of loan` + fee, no follow-on pattern below |")
    w(f"| early_termination | return more than {SEASON_END_TOLERANCE_DAYS}d from 30 Jun/31 Dec and another move within {FOLLOW_ON_IMMEDIATE_DAYS}d |")
    w(f"| purchase_option_or_permanent_conversion | permanent move lender → borrower within {FOLLOW_ON_WINDOW_DAYS}d of the return, or instead of one |")
    w(f"| immediate_follow_on_transfer | another move within {FOLLOW_ON_IMMEDIATE_DAYS}d of a season-boundary return, or a new loan within {FOLLOW_ON_WINDOW_DAYS}d |")
    w(f"| third_party_sale_related | paid or undisclosed-fee move to a third club within {FOLLOW_ON_WINDOW_DAYS}d, or the lender transferring the player while on loan |")
    w("| other_nonstandard | re-loaned to the same club, left the borrower with no return row, or re-loaned elsewhere by the lender |")
    w("| unresolved | no ending recorded, or the next movement involves neither club |")
    w("")
    ord_ = int((L.economic_ending == "ordinary_end_of_loan").sum())
    fu = int(b[b.economic_ending.isin(["immediate_follow_on_transfer", "third_party_sale_related"])].loans.sum())
    fu_plain = int(b[b.economic_ending.isin(["immediate_follow_on_transfer", "third_party_sale_related"])]
                   ["of_which_plain_end_of_loan_row"].sum())
    w(f"The broad non-ordinary count ({pct(econ_ns, n_loan)}) is dominated by loans that ended with a "
      f"plain `End of loan` and were followed by a new move: {fu:,} loans are follow-on or third-party "
      f"patterns, and {fu_plain:,} of them ended with a plain `End of loan` row. Whether those count as "
      "non-standard endings is a definitional choice; the raw-label answer (A) and the broad sequence "
      "answer (B) bracket it.")
    w("")
    sched_ord = int(b.loc[b.economic_ending == "ordinary_end_of_loan", "of_which_scheduled_future_ending"].iloc[0])
    bsi = ctx["by_season"].set_index("loan_season")
    last = bsi.index.max() if "2026/27" not in bsi.index else "2025/26"
    prior = bsi.loc[[x for x in bsi.index if "2021/22" <= x <= "2024/25"]]
    w(f"**Recent loans are biased toward \"ordinary\".** {sched_ord:,} of the "
      f"{int((L.economic_ending == 'ordinary_end_of_loan').sum()):,} ordinary endings are *scheduled* returns "
      "— loans still running when the page was scraped, which cannot yet show a conversion or follow-on "
      f"move. In {last}, {bsi.loc[last, 'nonstandard_pct']:.1f}% of loans are non-ordinary, against "
      f"{prior.nonstandard_any.sum() / prior.loans.sum() * 100:.1f}% across 2021/22–2024/25. "
      "That is censoring, not a change in behaviour.")
    w("")
    w("Two deliberate departures from the Stage 2 rules, with their effect:")
    w("")
    w(f"- A free or `-` move to a third club after a return is **not** counted as a sale (most often the "
      f"parent contract expired with the loan). Stage 2 counted it; doing so here would move "
      f"{s2['free_or_no_fee_moves_to_third_club_after_return']:,} loans into third_party_sale_related.")
    w(f"- 31 December is treated as a season boundary as well as 30 June, because calendar-year leagues and "
      f"half-season loans end there. Under Stage 2's 30-June-only rule, "
      f"{s2['returns_near_31dec_not_30jun_with_immediate_follow_on']:,} more returns would be classed as "
      "early terminations.")
    w("")
    w("### `fee_on_return_eur` for fee-bearing endings")
    w("")
    w(f"All {fee['count']} returns carrying a fee (`End of loan` + amount). The fee is a Stage 1 "
      "registration fact; it is not evidence of a particular mechanism.")
    w("")
    w("| count | median | mean | p25 | p75 | p90 | max |")
    w("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    w(f"| {fee['count']} | {eur(fee['median'])} | {eur(fee['mean'])} | {eur(fee['p25'])} | "
      f"{eur(fee['p75'])} | {eur(fee['p90'])} | {eur(fee['max'])} |")
    w("")

    # ---- 4
    w("## 4. European loan end dates")
    w("")
    w("**Population: loans where both the lending and the borrowing club are in a UEFA country, with a "
      "matched ending.** Country and confederation come only from the upstream snapshot (§5). The end "
      "date is the date of the row that ended the loan.")
    w("")
    w("Denominator coverage:")
    w("")
    w(f"- All loan episodes: {eucov['all_loan_episodes']:,}")
    w(f"- … at least one club of unknown country: {pct(eucov['loans_with_a_club_of_unknown_country'], eucov['all_loan_episodes'])} — **excluded, not assumed European**")
    w(f"- … both clubs in UEFA countries: {eucov['loans_both_clubs_uefa']:,}")
    w(f"- … of which with a matched ending (**N**): {eucov['uefa_loans_with_matched_ending']:,}; "
      f"without an ending: {eucov['uefa_loans_without_ending']:,}")
    w(f"- … of the N, scheduled (future-flagged) rather than realised: {eucov['uefa_loans_ending_scheduled_future']:,}")
    w("")
    show = eu[(eu.N > 0) & eu.population.isin(["all matched endings", "raw label: plain End of loan",
                                  "raw label: not plain End of loan", "fee-bearing returns",
                                  "economic: ordinary_end_of_loan", "economic: any nonstandard class"])]
    show = show.assign(**{
        "30 Jun": show.apply(lambda x: f"{int(x.jun30_n):,} ({x.jun30_pct:.1f}%)", axis=1),
        "31 Dec": show.apply(lambda x: f"{int(x.dec31_n):,} ({x.dec31_pct:.1f}%)", axis=1),
        "either": show.apply(lambda x: f"{int(x.jun30_or_dec31_n):,} ({x.jun30_or_dec31_pct:.1f}%)", axis=1),
        "other dates": show.apply(lambda x: f"{int(x.other_dates_n):,} ({x.other_dates_pct:.1f}%)", axis=1)})
    w(md_table(show, ["population", "slice", "N", "30 Jun", "31 Dec", "either", "other dates"]))
    w("")
    w(f"Realised endings only (excluding scheduled ones): {eal_real.jun30_pct:.1f}% on 30 June, "
      f"{eal_real.dec31_pct:.1f}% on 31 December, {eal_real.jun30_or_dec31_pct:.1f}% either "
      f"(N = {int(eal_real.N):,}). Scheduled endings sit on 30 June almost by construction, so "
      "pooling them inflates the 30 June share.")
    w("")
    td = eucov["top_dates"]
    w("The most common end dates (all European endings): " + ", ".join(
        f"{k.replace('06-30', '30 Jun').replace('05-31', '31 May').replace('12-31', '31 Dec')} "
        f"{v:,} ({100 * v / int(eall.N):.1f}%)" for k, v in td.items()) + ".")
    w("")
    bc = eucov["by_lender_country"]
    w(f"**31 May is the second most common end date, and it is almost entirely British.** "
      "The share falling on 30 June or 31 December varies sharply by the lending club's country "
      "(countries with at least 50 endings):")
    w("")
    bc2 = bc.assign(**{
        "30 Jun": bc.apply(lambda x: f"{x.jun30_pct:.1f}%", axis=1),
        "31 Dec": bc.apply(lambda x: f"{x.dec31_pct:.1f}%", axis=1),
        "30 Jun or 31 Dec": bc.apply(lambda x: f"{x.jun30_or_dec31_pct:.1f}%", axis=1),
        "31 May": bc.apply(lambda x: f"{x.of_other_may31_pct:.1f}%", axis=1)})
    w(md_table(bc2.head(14), ["lender_country", "N", "30 Jun", "31 Dec", "30 Jun or 31 Dec", "31 May"]))
    w("")
    w(f"Each economic class separately, every lending country, and a by-season appendix are in "
      f"`stage1_europe_loan_end_dates.csv`.")
    w("")

    # ---- 5
    w("## 5. Dataset coverage")
    w("")
    countries = pd.concat([ep.from_country, ep.to_country]).dropna()
    confs = pd.concat([ep.from_confederation, ep.to_confederation]).dropna()
    w("| Item | Count |")
    w("| --- | ---: |")
    for k, v in [("raw rows", f"{EXPECTED_ROWS:,}"), ("real transfer episodes", f"{n_ep:,}"),
                 ("loans", f"{n_loan:,}"), ("permanent transfers with a stated fee", f"{int(types.get('permanent_paid', 0)):,}"),
                 ("free transfers", f"{int(types.get('free_transfer', 0)):,}"),
                 ("undisclosed-fee transfers (`?`)", f"{int(types.get('undisclosed_fee', 0)):,}"),
                 ("no fee shown (`-`)", f"{int(types.get('no_fee_shown', 0)):,}"),
                 ("players", f"{ep.player_id.nunique():,}"), ("clubs", f"{len(ep_clubs):,}"),
                 ("clubs with a known country", f"{len(mapped_clubs & ep_clubs):,}"),
                 ("countries (of mapped clubs)", f"{countries.nunique()}"),
                 ("confederations", f"{confs.nunique()} ({', '.join(sorted(confs.unique()))})"),
                 ("seasons", f"{ep.season.nunique()} ({min(ep.season)} to {max(ep.season)})"),
                 ("date range", f"{dates.min().date()} to {dates.max().date()}")]:
        w(f"| {k} | {v} |")
    w("")
    both_mapped = ep.from_country.notna() & ep.to_country.notna()
    w(f"Episodes with both clubs' country known: {pct(int(both_mapped.sum()), n_ep)}; with at least one: "
      f"{pct(int((ep.from_country.notna() | ep.to_country.notna()).sum()), n_ep)}.")
    w("")
    w("### What the upstream snapshot covers")
    w("")
    comps = up["competitions"]
    leagues = comps[comps.type == "domestic_league"]
    cups = comps[comps.type == "domestic_cup"]
    w(f"The data comes from the public `dcaribou/transfermarkt-datasets` DuckDB snapshot (commit "
      f"`{up['commit'][:10]}`), which is scraped from Transfermarkt and also published on Kaggle. "
      "Everything below is read from that file; nothing was looked up online.")
    w("")
    w(f"- **{len(leagues)} first-tier domestic leagues**: " + ", ".join(
        f"{r.country_name}" for r in leagues.itertuples()) + ".")
    w(f"- **{len(cups)} national cups**: " + ", ".join(f"{r.country_name}" for r in cups.itertuples()) +
      "; plus domestic super cups and the UEFA club competitions.")
    gs = up["game_seasons"].set_index("type")
    w(f"- Match data runs from the {gs.loc['domestic_league', 'first_season']} to the "
      f"{gs.loc['domestic_league', 'last_season']} season (last match {up['max_game_date']}).")
    w(f"- {up['players']:,} players appear in those competitions; {up['players_with_transfers']:,} of them "
      f"have transfer histories, which make up all {up['transfer_rows']:,} transfer rows. Transfers are "
      "therefore complete *careers of players who reached a covered league*, including moves between "
      "clubs the snapshot does not otherwise cover — not all transfers made by covered clubs.")
    w(f"- Club country: {int((geo.geo_source == 'clubs_table_domestic_league').sum()):,} clubs from the "
      f"snapshot's `clubs` table (domestic league → country), plus "
      f"{int((geo.geo_source == 'domestic_competition_participation').sum()):,} clubs that played in a "
      "covered country's league or national cup. The two sources never disagree where both apply, and "
      "no club is placed in two countries. Other clubs are left unmapped rather than guessed from their names.")
    w("")
    top = ctx["by_country"]
    top = top[(top.role == "lender")].head(15)
    w("Loans by lending-club country (top 15; full table and borrower side in `stage1_loans_by_country.csv`):")
    w("")
    w(md_table(top, ["country", "loans", "domestic_both_clubs_same_country", "nonstandard_any",
                     "nonstandard_pct", "fee_bearing_returns"]))
    w("")
    bs = ctx["by_season"]
    recent = bs[bs.loan_season >= "2012/13"]
    w("Loans by season from 2012/13 (full table from 1993/94 in `stage1_loans_by_season.csv`):")
    w("")
    w(md_table(recent, ["loan_season", "loans", "with_matched_return", "open_or_unmatched",
                        "raw_label_not_plain_end_of_loan", "fee_bearing_returns", "nonstandard_any",
                        "nonstandard_pct", "uefa_both_clubs"]))
    w("")
    w("Counts by season, country, country × season and transfer type × season for all episodes are in "
      "`stage1_dataset_coverage.csv`.")
    w("")

    # ---- 6
    w("## 6. Scope estimator")
    w("")
    w("`src/analysis/loan_scope.py` counts any candidate sample:")
    w("")
    w("```")
    w("python -m src.analysis.loan_scope --countries Italy,England,Spain \\")
    w("    --seasons 2021/22,2022/23,2023/24 --country-mode both")
    w("```")
    w("")
    w("`--country-mode both` requires both clubs in the selected countries; `either` requires one. "
      "`--confederation UEFA` filters by confederation instead. It is also callable as "
      "`scope(countries=[...], seasons=[...], country_mode='both')`.")
    w("")
    w(md_table(ctx["scenarios"]))
    w("")

    # ---- 7
    w("## 7. Stage 2 workload")
    w("")
    if tp:
        w(f"Observed from the Stage 2 extraction cache, current prompt: **{tp['calls']} extraction calls "
          f"across {tp['families']} researched families**, {tp['calls_per_family']:.2f} calls per family. "
          "Parley's `prompt_tokens` field is unreliable (it reports 3 on calls whose total is over 13,000), "
          "so input is taken as total minus completion.")
        w("")
        w("| per call | input | completion | total | cost |")
        w("| --- | ---: | ---: | ---: | ---: |")
        w(f"| median | {tp['input_tokens_median']:,.0f} | {tp['completion_tokens_median']:,.0f} | "
          f"{tp['total_tokens_median']:,.0f} | |")
        w(f"| mean | {tp['input_tokens_mean']:,.0f} | {tp['completion_tokens_mean']:,.0f} | "
          f"{tp['total_tokens_mean']:,.0f} | ${tp['cost_usd_mean_per_call']:.3f} |")
        w(f"| p25 / p90 total | | | {tp['total_tokens_p25']:,.0f} / {tp['total_tokens_p90']:,.0f} | "
          f"p90 ${tp['cost_usd_p90_per_call']:.3f} |")
        w("")
        w("`tokens = loans × share reaching an extraction call × calls per family × tokens per call`")
        w("")
        w("| Scenario | tokens per call | calls per family |")
        w("| --- | ---: | ---: |")
        for k, v in wk["scenarios"].items():
            w(f"| {k} | {v['tokens_per_call']:,.0f} | {v['calls_per_family']:.2f} |")
        w("")
        w("LOW uses the 25th-percentile call; BASE the mean call and observed calls per family; HIGH the "
          "90th-percentile call and the calls per family seen with the earlier prompt, which includes "
          "development re-runs.")
        w("")
        w(f"**The share reaching an extraction call is not observed for ordinary loans.** "
          f"{wk['share_reaching_llm_note']} The table in §6 therefore shows three shares. Evidence "
          "acquisition (search and page retrieval) is excluded: the researched families ran entirely "
          "from cache, so its cost is also unobserved.")
    else:
        w("No usable token logs were found; the formula above applies with measured inputs.")
    w("")

    # ---- implications
    w("## Implications for Stage 2 scope")
    w("")
    w("This section quantifies trade-offs; it does not choose a design.")
    w("")
    w(f"- **Non-standard endings are rare by label and common by sequence.** By Transfermarkt's own label, "
      f"{pct(nonplain, int(ended.sum()))} of loan endings are not a plain `End of loan`, and only "
      f"{fee['count']} carry a fee. Excluding them would change almost nothing about sample size — but "
      f"they are the only endings whose own label records something other than expiry. By sequence "
      f"pattern, {pct(econ_ns, n_loan)} of loans are non-ordinary, including "
      f"{int((L.economic_ending == 'purchase_option_or_permanent_conversion').sum()):,} loans followed by a "
      "permanent move to the borrower — the pattern most consistent with an option or obligation.")
    eo = eu[(eu.population == "economic: ordinary_end_of_loan") & (eu.slice == "all")].iloc[0]
    w(f"- **A \"30 June or 31 December\" rule keeps about seven in ten European endings, and it does not "
      f"selectively drop non-ordinary ones.** {pct(int(eall.jun30_or_dec31_n), int(eall.N))} of European "
      f"loan endings fall on one of the two dates. Non-ordinary sequence endings are, if anything, more "
      f"concentrated there ({ens.jun30_or_dec31_pct:.1f}%, N = {int(ens.N):,}) than ordinary ones "
      f"({eo.jun30_or_dec31_pct:.1f}%, N = {int(eo.N):,}). Only the small raw-label tail is less "
      f"concentrated: {eraw.jun30_or_dec31_pct:.1f}% of endings with a non-`End of loan` label "
      f"(N = {int(eraw.N)}) and {efee.jun30_or_dec31_pct:.1f}% of fee-bearing returns (N = {int(efee.N)}) — "
      "samples too small to rest a rule on.")
    bc = eucov["by_lender_country"].set_index("lender_country")
    if {"England", "Italy"} <= set(bc.index):
        w(f"- **That date rule is not neutral across countries.** It keeps "
          f"{bc.loc['Italy', 'jun30_or_dec31_pct']:.0f}% of Italian-lent loans but only "
          f"{bc.loc['England', 'jun30_or_dec31_pct']:.0f}% of English-lent ones"
          + (f" and {bc.loc['Scotland', 'jun30_or_dec31_pct']:.0f}% of Scottish-lent ones"
             if "Scotland" in bc.index else "")
          + ", because Transfermarkt records British loans as ending on 31 May. Adding 31 May "
          f"would raise the European total from {eall.jun30_or_dec31_pct:.1f}% to "
          f"{eall.jun30_or_dec31_pct + eall.of_other_may31_pct:.1f}%.")
    sc = ctx["scenarios"]
    w(f"- **Country and season filters shrink the research volume sharply.** From {n_loan:,} loans in "
      "total: " + "; ".join(f"{r.sample}: {r.loan_episodes:,} loans ({r.nonstandard_loan_endings:,} "
                             f"non-ordinary, {r.fee_bearing_returns} fee-bearing)" for r in sc.itertuples()) + ".")
    w(f"- **The most recent season is not yet observable.** {last} loans are mostly still running "
      f"({int(bsi.loc[last, 'scheduled_future_endings']):,} of {int(bsi.loc[last, 'loans']):,} have only a "
      "scheduled end), so their endings cannot be classified yet. Restricting to completed seasons "
      "avoids that censoring.")
    w("- **Country filters are conservative by construction.** With `both` clubs required, a loan to an "
      "unmapped lower-league or youth club drops out, so the count is a floor on domestic loans; `either` "
      "gives the corresponding ceiling.")
    w("- **The volume that drives cost is the number of families that reach an extraction call.** At "
      f"BASE token use, every 1,000 loans that reach one cost about "
      f"{1000 * tp.get('total_tokens_mean', 0) * tp.get('calls_per_family', 1) / 1e6:,.1f}M tokens"
      f" (≈${1000 * tp.get('cost_usd_mean_per_call', 0) * tp.get('calls_per_family', 1):,.0f}). How many "
      "ordinary loans reach one is unknown until a random sample of them is run.")
    w("")
    w("## Files")
    w("")
    for f, d in ctx["files"]:
        w(f"- `{f}` — {d}")
    w("")
    return "\n".join(W) + "\n"


# ---------------------------------------------------------------------------

SCENARIOS = [
    ("all loans", dict()),
    ("UEFA, both clubs", dict(confederation="UEFA", country_mode="both")),
    ("UEFA, both clubs, 2021/22-2023/24", dict(confederation="UEFA", country_mode="both",
                                               seasons=["2021/22", "2022/23", "2023/24"])),
    ("Italy+England+Spain, both, 2021/22-2023/24", dict(countries=["Italy", "England", "Spain"],
                                                        seasons=["2021/22", "2022/23", "2023/24"],
                                                        country_mode="both")),
    ("Italy+England+Spain, either, 2021/22-2023/24", dict(countries=["Italy", "England", "Spain"],
                                                          seasons=["2021/22", "2022/23", "2023/24"],
                                                          country_mode="either")),
    ("Italy, both, all seasons", dict(countries=["Italy"], country_mode="both")),
]


def main() -> dict:
    sha_before = sha256(CANONICAL_CSV)
    df = load_canonical()
    assert len(df) == EXPECTED_ROWS, f"Stage 1C has {len(df):,} rows, expected {EXPECTED_ROWS:,}"
    geo = club_geography()
    u = build_universe(df, geo)
    # Scenarios reuse this universe rather than building a second one.
    from . import loan_scope
    loan_scope.universe = lambda: u

    wf = waterfall(u)
    outcomes = loan_outcomes(u)
    europe = europe_end_dates(u)
    by_season = loans_by_season(u)
    by_country = loans_by_country(u)

    scen_rows = []
    for name, kw in SCENARIOS:
        r = scope(**kw)
        scen_rows.append({"sample": name, "real_transfer_episodes": r["real_transfer_episodes"],
                          "loan_episodes": r["loan_episodes"],
                          "ordinary_loan_endings": r["ordinary_loan_endings"],
                          "nonstandard_loan_endings": r["nonstandard_loan_endings"],
                          "fee_bearing_returns": r["fee_bearing_returns"],
                          "unresolved_or_open": r["unresolved_or_open_loans"],
                          **({f"BASE tokens, share {sh}": f"{r['workload']['grid'][f'BASE|share={sh}|all_loans'] / 1e6:,.1f}M"
                              for sh in (0.1, 0.5, 0.9)} if r["workload"].get("available") else {})})
    scenarios = pd.DataFrame(scen_rows)

    out = {
        "stage1_transfer_universe_counts.csv": (
            pd.concat([wf.assign(section="waterfall"),
                       ctx_types(u).assign(section="episode_type")], ignore_index=True),
            "waterfall from rows to real transfer episodes, and episodes by type"),
        "stage1_dataset_coverage.csv": (coverage_tables(u),
                                        "episodes by season, country, country × season, type × season"),
        "stage1_loan_outcomes.csv": (pd.concat([outcomes[0], outcomes[1]], ignore_index=True),
                                     "loan endings: raw-label answer (A) and economic answer (B)"),
        "stage1_europe_loan_end_dates.csv": (
            pd.concat([europe[0].assign(table="by_population"),
                       europe[1]["by_lender_country"].assign(table="by_lender_country",
                                                            population="all matched endings", slice="all"),
                       europe[2].assign(table="by_loan_season", population="all matched endings",
                                        slice="all")], ignore_index=True),
            "European loan end dates by population, realised vs scheduled, and by season"),
        "stage1_loans_by_season.csv": (by_season, "loans and their endings by season"),
        "stage1_loans_by_country.csv": (by_country, "loans by lending- and borrowing-club country"),
        "stage1_loans_by_country_season.csv": (loans_by_country_season(u), "loans by country × season"),
        "stage1_loan_country_flows.csv": (country_flows(u), "loans by lender country → borrower country"),
        "loan_episode_pairing_audit.csv": (pairing_audit(u),
                                           "stratified sample of loan pairings, plus every unmatched return"),
    }
    for name, (frame, _) in out.items():
        frame.to_csv(OUT / name, index=False)

    ctx = {
        "u": u, "upstream": upstream_facts(), "waterfall": wf, "outcomes": outcomes,
        "europe": europe, "types": ctx_types(u), "sensitivities": sensitivities(u),
        "stage2_sensitivity": stage2_rule_sensitivity(u), "by_country": by_country,
        "by_season": by_season, "scenarios": scenarios, "geo": geo, "sha": sha_before,
        "workload_all": workload(len(u.loans), int(u.loans.economic_ending.isin(NONSTANDARD).sum())),
        "files": [("stage1_scope_summary.md", "this report")] + [(k, v[1]) for k, v in out.items()],
    }
    (OUT / "stage1_scope_summary.md").write_text(write_report(ctx))

    sha_after = sha256(CANONICAL_CSV)
    ctx["unchanged"] = sha_before == sha_after
    ctx["written"] = ["stage1_scope_summary.md"] + list(out)
    return ctx


def ctx_types(u) -> pd.DataFrame:
    return episode_type_counts(u)


def terminal_summary(ctx: dict, tests: str) -> None:
    u = ctx["u"]; L = u.loans; ep = u.episodes
    a, b, fee = ctx["outcomes"]
    eu = ctx["europe"][0]
    g = lambda pop: eu[(eu.population == pop) & (eu.slice == "all")].iloc[0]
    e_all, e_ns = g("all matched endings"), g("economic: any nonstandard class")
    e_raw = g("raw label: not plain End of loan")
    ended = int(L.terminal_event_id.notna().sum())
    plain = int(a[a.raw_label_class == "End of loan"].loans.sum())
    wk = ctx["workload_all"]
    clubs = set(ep.from_club_id) | set(ep.to_club_id)
    countries = pd.concat([ep.from_country, ep.to_country]).dropna().nunique()
    lines = [
        ("raw Stage 1C rows", f"{len(u.rows):,}"),
        ("real transfer episodes", f"{len(ep):,}"),
        ("loans", f"{len(L):,}"),
        ("loans as % of real transfers", f"{100 * len(L) / len(ep):.1f}%"),
        ("matched loan endings (any ending event)", f"{ended:,}  (End-of-loan return rows: {int(L.has_matched_return.sum()):,})"),
        ("ordinary End-of-loan (raw label)", f"{plain:,}"),
        ("non-End-of-loan (raw label)", f"{ended - plain:,}"),
        ("non-ordinary (economic, broad)", f"{int(L.economic_ending.isin(NONSTANDARD).sum()):,}"),
        ("fee-bearing returns", f"{fee['count']}"),
        ("Europe June 30 fraction", f"{e_all.jun30_pct:.1f}%  (N={int(e_all.N):,})"),
        ("Europe December 31 fraction", f"{e_all.dec31_pct:.1f}%"),
        ("Europe June 30 OR December 31", f"{e_all.jun30_or_dec31_pct:.1f}%"),
        ("  nonstandard (economic) Jun30/Dec31/either", f"{e_ns.jun30_pct:.1f}% / {e_ns.dec31_pct:.1f}% / {e_ns.jun30_or_dec31_pct:.1f}%  (N={int(e_ns.N):,})"),
        ("  non-End-of-loan label Jun30/Dec31/either", f"{e_raw.jun30_pct:.1f}% / {e_raw.dec31_pct:.1f}% / {e_raw.jun30_or_dec31_pct:.1f}%  (N={int(e_raw.N):,})"),
        ("unique players", f"{ep.player_id.nunique():,}"),
        ("unique clubs", f"{len(clubs):,}"),
        ("countries (mapped clubs)", f"{countries}"),
        ("seasons", f"{ep.season.nunique()}"),
        ("date range", f"{ep._date.min().date()} to {ep._date.max().date()}"),
    ]
    if wk.get("available"):
        tp = wk["observed"]["current_prompt"]
        lines.append(("observed Stage 2 tokens / call", f"median {tp['total_tokens_median']:,.0f}, mean {tp['total_tokens_mean']:,.0f}, p90 {tp['total_tokens_p90']:,.0f} ({tp['calls']} calls, {tp['families']} families)"))
        lines.append(("all loans, BASE, share 0.1 / 0.5 / 0.9",
                      " / ".join(f"{wk['grid'][f'BASE|share={s}|all_loans'] / 1e6:,.0f}M" for s in (0.1, 0.5, 0.9))))
    lines += [("tests", tests), ("Stage 1C unchanged", "YES" if ctx["unchanged"] else "NO"),
              ("Tavily calls", "0"), ("Parley calls", "0"),
              ("files written", ", ".join(ctx["written"]))]
    w = max(len(k) for k, _ in lines)
    for k, v in lines:
        print(f"{k:<{w}}  {v}")


def run_tests() -> str:
    import re
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_loan_episodes.py", "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().splitlines() if re.search(r"passed|failed|error", l)]
    return tail[-1] if tail else f"pytest exit {r.returncode}"


if __name__ == "__main__":
    c = main()
    terminal_summary(c, run_tests())
