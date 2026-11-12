"""Week 10 LaTeX report (every number read from generated, audited outputs).

    python -m src.analysis.week10_latex [--week9 path/to/edited_week9_guide.tex]

Writes data/outputs/rebuild/professor_feedback_week10/report/:
  week10_main.tex        Week 10 section + Open Methodological Decisions (\\input-able into the Week 9 guide)
  week10_appendix.tex    Week 10 appendices (\\input-able after the Week 9 appendices)
  week10_report.tex/pdf  standalone report (Week 9 guide preamble + the two files above)
  week9_week10_guide.tex/pdf  only with --week9 PATH: the user's edited Week 9 source with Week 10 added
  report_crosscheck.csv  headline numbers recomputed independently from the episode files
  report_number_scan.csv numbers appearing in the hand-written template text (must all be definitional constants)

Prose lives in templates with <<key>> placeholders; every <<key>> is filled from numbers.json / CSV outputs of
week10_followup, week10_audit, week10_squad_value, week10_regression_audit (Week 10) and feedback_nov2026_report
(Week 9, Question 5 only). Nothing upstream is modified. Run after week10_meeting.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import league_tiers_wiki as ltw
from . import week10_followup as wk
from .week10_followup import FOURTEEN, MASTER, OUT, V2

REP = OUT / "report"
AUD = OUT / "audit"
W9 = OUT.parent / "professor_feedback_nov_2026"
W9_TEX = W9 / "data_and_sample_guide_v5.tex"
PDFLATEX = "/usr/local/bin/pdflatex"

# numbers allowed to appear literally in template prose: definitions, not results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "8", "10", "11", "14", "28", "29", "90", "95", "365", "548", "730"}


# --------------------------------------------------------------------------------------------- formatting
def n(x) -> str:
    return f"{int(round(float(x))):,}"


def p(x, d: int = 1) -> str:
    """x already in percent."""
    return f"{float(x):.{d}f}\\%"


def r(a, b, d: int = 1) -> str:
    return p(100 * float(a) / float(b), d)


def c(x, d: int = 2) -> str:
    """signed coefficient; no '-0.00'."""
    v = round(float(x), d) + 0.0
    if v == 0:
        return f"{0:.{d}f}"
    return f"$-{abs(v):.{d}f}$" if v < 0 else f"+{v:.{d}f}"


def pv(x) -> str:
    return f"{float(x):.3f}"


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    s = s.replace("->", r"$\rightarrow$").replace("→", r"$\rightarrow$").replace("<", r"$<$").replace(">", r"$>$")
    return s.replace("—", ", ").replace("`", "")


def fill(tpl: str, V: dict) -> str:
    keys = re.findall(r"<<([A-Za-z0-9_]+)>>", tpl)
    missing = sorted({k for k in keys if k not in V})
    if missing:
        raise KeyError(f"unfilled placeholders: {missing}")
    return re.sub(r"<<([A-Za-z0-9_]+)>>", lambda m: str(V[m.group(1)]), tpl)


# --------------------------------------------------------------------------------------------- data
def _j(p_: Path) -> dict:
    return json.loads(p_.read_text())


def load() -> dict:
    q6dir = OUT / "q6_squad_value"
    d = {"now": _j(OUT / "numbers.json"), "lt": _j(V2 / "league_tiers_v2_numbers.json"),
         "aud": _j(AUD / "audit_numbers.json"), "q6": _j(q6dir / "numbers.json"),
         "ra": _j(q6dir / "regression_audit" / "numbers.json"), "w9": _j(W9 / "numbers.json")}
    rd = lambda *a, **k: pd.read_csv(OUT.joinpath(*a), **k)
    d["mv"] = rd("q3_tier_movement", "movement_summary.csv")
    d["bb"] = rd("q3_tier_movement", "direction_by_domestic_crossborder.csv")
    d["period"] = rd("q3_tier_movement", "direction_by_period.csv")
    dt = pd.read_csv(OUT / "q3_tier_movement" / "down_share_from_tier1_by_origin_country.csv", header=[0, 1],
                     skiprows=[2], index_col=0)
    dt.columns = [f"{a} {b}" for a, b in dt.columns]
    dt.index.name = "origin country (tier 1)"
    d["down_t1"] = dt.reset_index()
    d["dest_t1"] = rd("q3_tier_movement", "destinations_from_14country_tier1_pct.csv")
    d["t2c"] = rd("q1_tier2_playing_time", "potential_tier2_episodes_by_country.csv")
    d["kind_out"] = rd("q2_origin_tiers", "kind_by_outcome.csv")
    d["V1"] = rd("q4_heatmaps", "V1_loan_flow_matrix_29x29.csv", index_col=0)
    d["V2"] = rd("q4_heatmaps", "V2_permanent_flow_matrix_29x29.csv", index_col=0)
    d["omit"] = rd("q4_heatmaps", "omitted_episodes_by_reason.csv")
    d["catcov"] = rd("q4_heatmaps", "category_list_coverage.csv")
    d["miss"] = rd("q6_squad_value", "missingness_reasons.csv")
    d["sens6"] = rd("q6_squad_value", "sensitivity_summary.csv")
    d["spec"] = rd("q6_squad_value", "regression_audit", "specifications.csv")
    d["dec"] = rd("q6_squad_value", "regression_audit", "decomposition.csv")
    d["prof"] = rd("q6_squad_value", "regression_audit", "excluded_vs_retained_profile.csv")
    d["means"] = rd("q6_squad_value", "regression_audit", "excluded_vs_retained_means.csv")
    d["sens"] = pd.read_csv(AUD / "tier_rule_sensitivity.csv", index_col=0)
    d["mvv"] = pd.read_csv(AUD / "match_audit_verdicts.csv")
    d["exv"] = pd.read_csv(AUD / "exclusion_audit_verdicts.csv")
    d["msum"] = pd.read_csv(AUD / "match_audit_summary.csv")
    d["cons"] = pd.read_csv(AUD / "consistency_checks.csv")
    d["ba"] = pd.read_csv(AUD / "before_after_headline_numbers.csv")
    d["dict"] = rd("week10_episode_supplement_dictionary.csv")
    d["q5cov"] = pd.read_csv(W9 / "q5_prior_season_coverage.csv")
    return d


def mvs(d, kind, cat, col="pct"):
    S = d["mv"]
    return float(S[(S.kind == kind) & (S.category == cat)].iloc[0][col])


def spec(d, prefix):
    x = d["spec"][d["spec"]["sample"].str.startswith(prefix)]
    return x.set_index(x.model.str[:1])


def values(d: dict) -> dict:
    V: dict[str, str] = {}
    now, lt, q1, q2, q4 = d["now"], d["lt"], d["now"]["q1"], d["now"]["q2"], d["now"]["q4"]
    w9 = d["w9"]
    # samples
    V.update(n_main=n(q2["n_main"]), n_main_val=n(now["checks"]["main_validated"]),
             n_inj=n(now["checks"]["injury_sample"]), n_inj_val=n(now["checks"]["injury_validated"]),
             master_rows=n(now["checks"]["master_shape"][0]), master_cols=n(now["checks"]["master_shape"][1]),
             loans_period=n(q4["loan"]["episodes_in_period"]), perm_period=n(q4["permanent"]["episodes_in_period"]))
    # Q1
    t1, t2 = q1["tier1_episodes_14"], q1["tier2_episodes_14"]
    V.update(snap_leagues=n(q1["snapshot_domestic_league_competitions"]),
             t2_total=n(sum(t2.values())), t2_loan=n(t2["loan"]), t2_perm=n(t2["permanent"]),
             t1_total=n(sum(t1.values())), t1_loan=n(t1["loan"]), t1_perm=n(t1["permanent"]),
             t2_loan_share=p(100 * q1["loan_share_tier2_episodes"]),
             t1_loan_share=p(100 * q1["loan_share_tier1_episodes"]),
             t2_lists=n(q1["tier2_list_available_country_seasons"]), country_seasons=n(q1["country_seasons"]),
             pass_loan=p(100 * q1["tier1_pass_rate_to_revised"]["loan"]),
             pass_perm=p(100 * q1["tier1_pass_rate_to_revised"]["permanent"]),
             proj_loan=n(q1["projected_tier2_additions"]["loan"]),
             proj_perm=n(q1["projected_tier2_additions"]["permanent"]),
             revised_total=n(sum(q1["revised_sample"].values())),
             share_revised=p(100 * q1["loan_share_revised"]),
             share_combined=p(100 * q1["loan_share_projected_combined"]))
    # Q2 / lookup
    series_t1 = sum(1 for s in ltw.SERIES if s["tier"] == 1)
    series_t2 = sum(1 for s in ltw.SERIES if s["tier"] == 2)
    c14 = set(FOURTEEN)
    oth1 = sorted({s["country"] for s in ltw.SERIES if s["tier"] == 1 and s["country"] not in c14})
    oth2 = sorted({s["country"] for s in ltw.SERIES if s["tier"] == 2 and s["country"] not in c14})
    V.update(series=n(lt["series"]), series_t1=n(series_t1), series_t2=n(series_t2),
             series_oth1=n(len(oth1)), series_oth2=n(len(oth2)),
             oth1_list=esc(", ".join(oth1)), oth2_list=esc(", ".join(oth2)),
             pages=n(lt["season_pages_parsed"]), team_rows=n(lt["team_rows"]),
             team_matched=n(lt["team_rows_matched"]), team_matched_pct=r(lt["team_rows_matched"], lt["team_rows"]),
             overlap=n(lt["overlap_rows"]), overlap_agree=n(lt["overlap_agree_tier"]),
             conflicts=n(lt["wiki_internal_conflict_rows"]), rows_w9=n(lt["table_rows_week9"]),
             rows_added=n(lt["table_rows_added"]),
             rows_total=n(lt["table_rows_week9"] + lt["table_rows_added"]),
             rejected_end=n(lt["rejected_after_club_end_year"]),
             fb_unique=n(lt["fallback_precision_check"]["unique_candidate"]),
             fb_correct=n(lt["fallback_precision_check"]["unique_correct"]),
             sim_max=f"{wk.SIM_MAX:.2f}")
    V.update(w9_verified=n(q2["verified_week9"]), w9_pct=r(q2["verified_week9"], q2["n_main"]),
             missing=n(q2["missing_week9"]), rec_exact=n(q2["recovered_exact"]),
             rec_not12=n(q2["recovered_not_1_2"]), rec_yr=n(q2["recovered_youth_reserve"]),
             part_not1=n(q2["partly_not_1"]), still_unknown=n(q2["still_unknown"]),
             exact_v2=n(q2["exact_tier_v2"]), exact_v2_pct=r(q2["exact_tier_v2"], q2["n_main"]),
             band_v2=n(q2["band_known_v2"]), band_v2_pct=r(q2["band_known_v2"], q2["n_main"]),
             resolved=n(q2["recovered_exact"] + q2["recovered_not_1_2"] + q2["recovered_youth_reserve"]),
             inferred=n(q2["recovered_not_1_2"] + q2["recovered_youth_reserve"] + q2["partly_not_1"]))
    for k, v in q2["exact_tier_counts_v2"].items():
        V[f"tier{k}"] = n(v)
    su = q2["still_unknown_reason"]
    V.update(su_nolist=n(su["no league list for the club's country in that season"]),
             su_token=n(su["lists exist, but the club may be one of their unmatched names (shared name token)"]))
    for kind in ("loan", "permanent"):
        b = q2["by_kind"][kind]
        V[f"{kind}_main"] = n(b["main"])
        V[f"{kind}_w9"], V[f"{kind}_w9_pct"] = n(b["verified_week9"]), r(b["verified_week9"], b["main"])
        V[f"{kind}_ex"], V[f"{kind}_ex_pct"] = n(b["exact_v2"]), r(b["exact_v2"], b["main"])
        V[f"{kind}_band"], V[f"{kind}_band_pct"] = n(b["band_v2"]), r(b["band_v2"], b["main"])
    wr = q2["week9_reason"]
    V.update(wr_out_senior=n(wr["outside the 14 countries: senior club"]),
             wr_in_yr=n(wr["within the 14 countries: youth or reserve side"]),
             wr_in_senior=n(wr["within the 14 countries: senior club, lower-division list missing"]),
             wr_out_yr=n(wr["outside the 14 countries: youth or reserve side"]))
    # audits
    mvv, exv = d["mvv"], d["exv"]
    kept = mvv[mvv.in_final_lookup]
    fb = mvv[mvv.method == "token_subset_country"]
    V.update(ma_n=n(len(mvv)), ma_acc=n((mvv.verdict == "accept").sum()), ma_rej=n((mvv.verdict == "reject").sum()),
             ma_unc=n((mvv.verdict == "uncertain").sum()), ma_kept=n(len(kept)),
             ma_kept_acc=n((kept.verdict == "accept").sum()), ma_kept_rej=n((kept.verdict == "reject").sum()),
             ma_bound=p(100 * (1 - 0.05 ** (1 / len(kept)))),
             fb_n=n(len(fb)), fb_wrong=n((fb.verdict == "reject").sum()),
             fb_notacc=n((fb.verdict != "accept").sum()),
             ex_n=n(len(exv)), ex_acc=n((exv.verdict == "accept").sum()), ex_rej=n((exv.verdict == "reject").sum()),
             ex_unc=n((exv.verdict == "uncertain").sum()),
             ex_unc_list=esc("; ".join(f"{a} ({s}/{(s + 1) % 100:02d})" for a, s in
                                       exv[exv.verdict == "uncertain"][["from_club_name", "tier_reference_season"]]
                                       .itertuples(index=False))),
             n_cons=n(len(d["cons"])), cons_ok=n(d["cons"].ok.sum()))
    s = d["sens"]
    g = lambda row, col: float(s.loc[row].iloc[col])
    V.update(st_band=n(g("origin tier or band known (main)", 1)),
             st_band_diff=n(abs(g("origin tier or band known (main)", 2))),
             st_not12=n(g("recovered: senior not tier 1 or 2", 1)),
             st_yr=n(g("recovered: youth/reserve not tier 1 or 2", 1)),
             st_not1=n(g("partly recovered: not tier 1", 1)), st_unknown=n(g("still unknown", 1)),
             st_loan_down=p(g("loan: down %", 1)), st_loan_up=p(g("loan: up %", 1)),
             st_perm_down=p(g("permanent: down %", 1)), st_perm_up=p(g("permanent: up %", 1)),
             st_loan_dom=p(g("loan: domestic down %", 1)), st_perm_dom=p(g("permanent: domestic down %", 1)),
             st_loan_dec=n(g("loan: decidable n", 1)), st_perm_dec=n(g("permanent: decidable n", 1)),
             st_loan_om=n(g("loan: heatmap omitted", 1)), st_perm_om=n(g("permanent: heatmap omitted", 1)))
    # Q3
    D, U, SM = "Higher division -> lower division", "Lower division -> higher division", "Same level"
    bb = d["bb"].set_index(["kind", "border"])
    for kind, k in (("loan", "l"), ("permanent", "pm")):
        V[f"{k}_down"], V[f"{k}_up"], V[f"{k}_same"] = (p(mvs(d, kind, D)), p(mvs(d, kind, U)), p(mvs(d, kind, SM)))
        V[f"{k}_down_n"], V[f"{k}_up_n"] = n(mvs(d, kind, D, "count")), n(mvs(d, kind, U, "count"))
        V[f"{k}_dec"] = n(mvs(d, kind, D, "denominator"))
        V[f"{k}_dec_pct"] = p(mvs(d, kind, "direction decidable"))
        V[f"{k}_both"] = p(mvs(d, kind, "both tiers known"))
        V[f"{k}_both_n"] = n(mvs(d, kind, "both tiers known", "count"))
        V[f"{k}_t1t2"] = p(mvs(d, kind, "Tier 1 -> Tier 2"))
        V[f"{k}_t2t1"] = p(mvs(d, kind, "Tier 2 -> Tier 1"))
        V[f"{k}_xb"] = p(mvs(d, kind, "cross-border"))
        for border, b in (("domestic", "dom"), ("cross-border", "xb")):
            row = bb.loc[(kind, border)]
            V[f"{k}_{b}_n"] = n(row["n"])
            V[f"{k}_{b}_down"], V[f"{k}_{b}_up"] = p(row["down_pct"]), p(row["up_pct"])
            V[f"{k}_{b}_same"] = p(row["same level_pct"])
    per = d["period"].set_index(["kind", "period"])
    V.update(per_first=esc(per.index.get_level_values(1)[0]), per_last=esc(per.index.get_level_values(1)[-1]),
             l_down_first=p(per.loc[("loan", per.index.get_level_values(1)[0]), "down_pct"]),
             l_down_last=p(per.loc[("loan", per.index.get_level_values(1)[-1]), "down_pct"]))
    dt = d["down_t1"].set_index("origin country (tier 1)")
    hi = dt["down_pct loan"].idxmax()
    lo = dt["down_pct loan"].idxmin()
    above = (dt["down_pct loan"] > dt["down_pct permanent"]).sum()
    exc = dt.index[dt["down_pct loan"] <= dt["down_pct permanent"]]
    assert len(exc) == 1, "text assumes exactly one exception"
    hi = (dt["down_pct loan"] - dt["down_pct permanent"]).idxmax()
    V["dt_exc"] = f"{esc(exc[0])} ({p(dt.loc[exc[0], 'down_pct loan'])} against {p(dt.loc[exc[0], 'down_pct permanent'])})"
    V.update(dt_hi=esc(hi), dt_hi_l=p(dt.loc[hi, "down_pct loan"]), dt_hi_p=p(dt.loc[hi, "down_pct permanent"]),
             dt_lo=esc(lo), dt_lo_l=p(dt.loc[lo, "down_pct loan"]), dt_lo_p=p(dt.loc[lo, "down_pct permanent"]),
             dt_above=n(above), dt_total=n(len(dt)))
    # Q4
    for kind, M in (("loan", d["V1"]), ("permanent", d["V2"])):
        k = "l" if kind == "loan" else "pm"
        x = q4[kind]
        V[f"{k}_in"], V[f"{k}_om"], V[f"{k}_core"] = n(x["in_matrix"]), n(x["omitted"]), n(x["core_28x28"])
        V[f"{k}_in_pct"] = r(x["in_matrix"], x["episodes_in_period"])
        V[f"{k}_oo"], V[f"{k}_oo_pct"] = n(x["other_to_other"]), r(x["other_to_other"], x["in_matrix"])
        V[f"{k}_core_pct"] = r(x["core_28x28"], x["in_matrix"])
        core = M.drop(index="Other", columns="Other")
        cty = lambda lab: lab.rsplit(" ", 1)[0]
        dom = sum(core.loc[a, b] for a in core.index for b in core.columns if cty(a) == cty(b))
        V[f"{k}_core_dom"] = r(dom, core.to_numpy().sum())
        t1t2 = sum(core.loc[f"{cc} T1", f"{cc} T2"] for cc in {cty(a) for a in core.index})
        t2t1 = sum(core.loc[f"{cc} T2", f"{cc} T1"] for cc in {cty(a) for a in core.index})
        V[f"{k}_core_t1t2"], V[f"{k}_core_t2t1"] = n(t1t2), n(t2t1)
        assert (t1t2 > t2t1) if kind == "loan" else (t2t1 > t1t2), "text states the direction of the larger flow"
        st = core.stack().sort_values(ascending=False)
        V[f"{k}_top"] = "; ".join(f"{esc(a)} $\\rightarrow$ {esc(b)} {n(v)}" for (a, b), v in st.head(4).items())
    om = d["omit"]
    cnt = lambda side, reason, kind: int(om[(om.side == side) & om.reason.str.contains(reason)
                                            & (om.kind == kind)].episodes.sum())
    V.update(om_l_from_tier=n(cnt("from", "tier 1 or 2 cannot", "loan")),
             om_l_to_tier=n(cnt("to", "tier 1 or 2 cannot", "loan")),
             om_p_from_tier=n(cnt("from", "tier 1 or 2 cannot", "permanent")),
             om_p_to_tier=n(cnt("to", "tier 1 or 2 cannot", "permanent")),
             om_l_country=n(cnt("from", "country unknown", "loan") + cnt("to", "country unknown", "loan")),
             om_p_country=n(cnt("from", "country unknown", "permanent") + cnt("to", "country unknown", "permanent")))
    cc_ = d["catcov"]
    partial = cc_[cc_.seasons_with_list < cc_.seasons_in_period]
    V.update(cat_partial=esc(", ".join(f"{a} ({n(b)} of {n(c2)} seasons)" for a, b, c2 in
                                       partial[["category", "seasons_with_list", "seasons_in_period"]]
                                       .itertuples(index=False))),
             cat_seasons=n(cc_.seasons_in_period.max()))
    t = d["t2c"].assign(share=lambda x: 100 * x.loan / x.total).set_index("to_country_for_flows")
    V.update(t2c_hi=p(t.share.max()), t2c_hi_c=esc(t.share.idxmax()), t2c_lo=p(t.share.min()),
             t2c_lo_c=esc(t.share.idxmin()))
    bk = q2["by_kind"]
    gap = lambda key: 100 * (bk["loan"][key] / bk["loan"]["main"] - bk["permanent"][key] / bk["permanent"]["main"])
    V.update(gap_w9=f"{gap('verified_week9'):.1f}", gap_w10=f"{gap('exact_v2'):.1f}",
             rec_exact_perm=n(d["kind_out"].set_index("kind").loc["permanent",
                                                                    "recovered: exact tier verified (Wikipedia list)"]))
    ba = d["ba"].set_index("number")
    V.update(ba_exact_before=n(ba.loc["Main sample: exact origin tier", "before audit (10 Nov)"]),
             ba_exact_after=n(ba.loc["Main sample: exact origin tier", "after audit (12 Nov)"]),
             inferred_full=n(q2["recovered_not_1_2"] + q2["recovered_youth_reserve"]))
    labm = {"exact_normalised": "exact names", "alias_learned": "learned aliases", "alias_manual": "manual aliases"}
    km = kept.groupby("method").size()
    assert set(km.index) <= set(labm), km
    V["ma_strata"] = ", ".join(f"{labm[m]} {n(km[m])}" for m in labm if m in km.index)
    rc = pd.read_csv(W9 / "regression_comparison.csv")
    rc = rc[(rc.outcome == "Main") & rc.version.str.startswith("1 stored")].set_index("calendar_key")
    V.update(cf_N=n(rc.loc["calendar_first", "N"]), cf_b=c(100 * rc.loc["calendar_first", "loan_coef"]),
             cf_p=pv(rc.loc["calendar_first", "p"]), nc_N=n(rc.loc["no_calendar", "N"]),
             nc_b=c(100 * rc.loc["no_calendar", "loan_coef"]), nc_p=pv(rc.loc["no_calendar", "p"]))
    assert V["cf_N"] == V["n_main"]
    # Q5 (Week 9, unchanged)
    q5 = w9["q5"]
    V.update(q5_def=n(q5["main_defined"]), q5_def_pct=r(q5["main_defined"], q5["main"]),
             q5_fit=n(q5["main_fit_defined"]), q5_fit_pct=r(q5["main_fit_defined"], q5["main"]),
             q5_whole=n(q5["main_whole"]),
             q5_loan=n(q5["main_by_kind"]["loan"]["share"]), q5_loan_pct=r(q5["main_by_kind"]["loan"]["share"],
                                                                         q5["main_by_kind"]["loan"]["n"]),
             q5_perm=n(q5["main_by_kind"]["permanent"]["share"]),
             q5_perm_pct=r(q5["main_by_kind"]["permanent"]["share"], q5["main_by_kind"]["permanent"]["n"]),
             q5_nocov=n(q5["reasons_main"]["no_covered_club"]),
             q5_incomplete=n(q5["reasons_main"]["incomplete_club_games"]),
             q5_nogames=n(q5["reasons_main"]["no_league_games_before_move"]),
             q5_outside=n(q5["reasons_main"]["prior_season_outside_snapshot"]),
             q5_sql=n(q5["sql_equal"]), q5_sql_n=n(q5["sql_checked"]),
             q5_multi=n(q5["multi_equal"]), q5_multi_n=n(q5["multi_checked"]),
             q5_trunc=n(q5["fit_flag_main"]["injury history truncated before the window"]),
             q5_notlake=n(q5["fit_flag_main"]["player not in the external datalake"]),
             q5_corr=f"{q5['corr_current']:.2f}")
    qc = d["q5cov"].set_index(["sample", "kind"])
    V.update(q5_nocal=n(qc.loc[("no-calendar main", "all"), "prior_share"]),
             q5_nocal_n=n(qc.loc[("no-calendar main", "all"), "episodes"]),
             w9_q4_t1=n(w9["q4"]["main_from_tiers"]["1.0"]), w9_q4_t2=n(w9["q4"]["main_from_tiers"]["2.0"]))
    # Q6
    q6 = d["q6"]
    dec = q6["decomposition"]
    V.update(sv_stored=n(q6["reproduction"]["stored_non_missing"]),
             sv_cov=p(100 * q6["main_coverage"]["coverage"]), sv_cov_n=n(q6["main_coverage"]["covered"]),
             sv_loan=p(100 * q6["loan_coverage"]["coverage"]), sv_perm=p(100 * q6["perm_coverage"]["coverage"]),
             sv_loan_n=n(q6["loan_coverage"]["covered"]), sv_perm_n=n(q6["perm_coverage"]["covered"]),
             sv_miss=n(q6["main_coverage"]["N"] - q6["main_coverage"]["covered"]),
             sv_all=p(100 * q6["all_typed_coverage"]["coverage"]),
             sv_top=p(100 * dec["coverage_origin_covered_top_flight"]),
             sv_nottop=p(100 * dec["coverage_origin_not_covered_top_flight"]),
             sv_loan_top=p(100 * dec["loan_share_origin_covered_top_flight"]),
             sv_perm_top=p(100 * dec["perm_share_origin_covered_top_flight"]),
             sv_perm_mix=p(100 * dec["perm_coverage_with_loan_mix"]),
             sv_gap=p(100 * dec["share_of_gap_explained"], 0),
             sv_thin=n(q6["thin"]["thin_main"]), sv_thin_8=n(q6["thin"]["thin_8_to_10"]),
             sv_thin_730=n(q6["thin"]["thin_reach_11_at_730"]), sv_stale=n(q6["thin"]["stale_main"]),
             sv_stale_730=n(q6["thin"]["stale_reach_11_at_730"]),
             sv_players=n(q6["hypothesis"]["players_in_snapshot_players_table"]),
             sv_players_val=n(q6["hypothesis"]["players_with_any_valuation"]),
             sv_vals=n(q6["hypothesis"]["valuations_total"]))
    s6 = d["sens6"].set_index("variant")
    for var in ("lb730_min11", "lb365_min8", "union730_min11", "lb548_min11"):
        V[f"{var}_cov"] = p(100 * s6.loc[var, "coverage_all"])
        V[f"{var}_new"] = n(s6.loc[var, "new_episodes_all"])
        V[f"{var}_gain"] = f"{100 * s6.loc[var, 'gain_pp_all']:.1f}"
    V["lb730_corr"] = f"{s6.loc['lb730_min11', 'corr_logs']:.3f}"
    V["lb730_mad"] = p(100 * s6.loc["lb730_min11", "median_abs_pct_diff"])
    mi = d["miss"]
    mm = mi[(mi.population.str.startswith("main")) & (mi.kind == "all")].set_index("reason")
    V.update(mr_yr=n(mm.loc["youth_or_reserve_origin", "episodes"]), mr_thin=n(mm.loc["thin_1_to_10_players", "episodes"]))
    sp, sv = spec(d, "main (calendar"), spec(d, "main, fully")
    for lab, S_ in (("s", sp), ("v", sv)):
        for m in "123456":
            V[f"{lab}{m}_N"] = n(S_.loc[m, "N"])
            V[f"{lab}{m}_b"] = c(S_.loc[m, "loan_coef_pp"])
            V[f"{lab}{m}_p"] = pv(S_.loc[m, "p"])
    V.update(s2_share=r(sp.loc["2", "N"], sp.loc["1", "N"]),
             loanshare_full=r(sp.loc["1", "loans"], sp.loc["1", "N"]),
             loanshare_cc=r(sp.loc["2", "loans"], sp.loc["2", "N"]),
             theta=f"{100 * sp.loc['3', 'origin_control_coef']:.2f}", theta_p=pv(sp.loc["3", "origin_control_p"]),
             clusters=n(sp.loc["1", "clusters"]))
    de = d["dec"].set_index("sample")
    for lab, key in (("s", "main (calendar-first, stored)"), ("v", "main, fully validated")):
        V[f"{lab}_sel"] = c(de.loc[key, "selection_change"])
        V[f"{lab}_ctl"] = c(de.loc[key, "control_change"])
        V[f"{lab}_sel730"] = c(de.loc[key, "selection_change_730"])
        V[f"{lab}_ctl730"] = c(de.loc[key, "control_change_730"])
    rel = d["ra"]["log_origin_value_by_kind_complete_cases"]
    V.update(rel_loan=f"{rel['loan']['mean']:.2f}", rel_perm=f"{rel['permanent']['mean']:.2f}")
    pr = d["prof"]
    EX, RT = "excluded (origin value missing) %", "retained (origin value present) %"
    q = lambda dim, val, col: float(pr[(pr.dimension == dim) & (pr.value == val)][col].iloc[0])
    V.update(ex_perm=p(q("kind", "permanent", EX)), rt_perm=p(q("kind", "permanent", RT)),
             ex_nottop=p(q("origin_in_covered_top_flight", "no", EX)),
             rt_nottop=p(q("origin_in_covered_top_flight", "no", RT)),
             ex_yr=p(q("from_tier_band_v2", "youth/reserve", EX)), rt_yr=p(q("from_tier_band_v2", "youth/reserve", RT)),
             ex_other=p(q("origin_country_group", "other country", EX)),
             rt_other=p(q("origin_country_group", "other country", RT)))
    mn = d["means"].set_index(["group", "kind"])
    for g_, lab in (("excluded (origin value missing)", "ex"), ("retained (origin value present)", "rt")):
        for kind, k in (("loan", "l"), ("permanent", "pm")):
            V[f"ms_{lab}_{k}"] = p(100 * mn.loc[(g_, kind), "minutes_share"])
    return V



# --------------------------------------------------------------------------------------------- print heatmaps
def print_heatmap(mat: pd.DataFrame, cov: pd.DataFrame, vmax: float, path: Path, with_other: bool) -> None:
    """Re-render a stored flow matrix at true print size for a landscape page (fonts stay >= 5 pt at 100% zoom).
    Same colour ramp, log(1 + count) scale (vmax shared by V1 and V2), zero/Other conventions as
    week10_followup.plot_heatmap; titles live in the LaTeX caption instead of the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue_seq", ramp)
    ink, muted, other_fill = "#1f1f1e", "#6b6a66", "#e8e7e3"
    cats = list(wk.CATS) if with_other else [c_ for c_ in wk.CATS if c_ != "Other"]
    assert list(mat.index) == list(wk.CATS) and list(mat.columns) == list(wk.CATS)
    partial = set(cov.loc[(cov.seasons_with_list > 0) & (cov.seasons_with_list < cov.seasons_in_period), "category"])
    norm = Normalize(vmin=0, vmax=vmax)
    k = len(cats)
    fig = plt.figure(figsize=(9.6, 6.25))
    ax = fig.add_axes([0.105, 0.015, 0.735, 0.80])
    for i, r_ in enumerate(cats):
        for j, c_ in enumerate(cats):
            v = int(mat.loc[r_, c_])
            is_other = r_ == "Other" or c_ == "Other"
            fc = other_fill if is_other else ("#ffffff" if v == 0 else cmap(norm(np.log1p(v))))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="#ffffff", linewidth=0.5))
            col = muted if v == 0 else ink
            if not is_other and v > 0:
                rgb = matplotlib.colors.to_rgb(fc)
                if 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.45:
                    col = "#ffffff"
            fs = 5.4 if v < 1000 else (4.7 if v < 10000 else 4.3)
            ax.text(j + .5, i + .52, f"{v:,}", ha="center", va="center", fontsize=fs, color=col,
                    fontweight="bold" if is_other else "normal")
    for b in range(0, 29, 2):
        ax.axhline(b, color="#9a9893", lw=0.45)
        ax.axvline(b, color="#9a9893", lw=0.45)
    if with_other:
        ax.axhline(28, color=ink, lw=1.0)
        ax.axvline(28, color=ink, lw=1.0)
    ax.set_xlim(0, k)
    ax.set_ylim(k, 0)
    lab = [c_ + (" *" if c_ in partial else "") for c_ in cats]
    ax.set_xticks(np.arange(k) + .5)
    ax.set_yticks(np.arange(k) + .5)
    ax.xaxis.tick_top()
    ax.set_xticklabels(lab, rotation=55, ha="left", rotation_mode="anchor", fontsize=6.2, color=ink)
    ax.set_yticklabels(lab, fontsize=6.2, color=ink)
    ax.tick_params(length=0, pad=2)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("TO (receiving club)", fontsize=7.5, color=ink, labelpad=4)
    ax.set_ylabel("FROM (origin club)", fontsize=7.5, color=ink, labelpad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # legend on the right
    cax = fig.add_axes([0.865, 0.42, 0.018, 0.36])
    grad = np.linspace(0, vmax, 256)[:, None]
    cax.imshow(grad, aspect="auto", cmap=cmap, extent=[0, 1, 0, vmax], origin="lower")
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000) if np.log1p(t) <= vmax]
    cax.set_yticks([np.log1p(t) for t in ticks])
    cax.set_yticklabels([f"{t:,}" for t in ticks], fontsize=6, color=ink)
    cax.yaxis.tick_right()
    cax.set_xticks([])
    cax.tick_params(length=2, width=0.5)
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.text(0.858, 0.80, "Colour:\nlog(1 + count)\n(tick labels are\nraw counts)", fontsize=6, color=ink, va="bottom")
    items = [("#ffffff", "0 = no transfers")]
    if with_other:
        items.append((other_fill, "Other: raw counts,\nnot on the colour scale"))
    y = 0.33
    for fc, txt in items:
        bx = fig.add_axes([0.865, y, 0.014, 0.022])
        bx.add_patch(Rectangle((0, 0), 1, 1, facecolor=fc, edgecolor="#9a9893", lw=0.5))
        bx.set_axis_off()
        fig.text(0.885, y + 0.011, txt, fontsize=6, color=ink, va="center")
        y -= 0.07
    fig.text(0.865, y + 0.01, "* division list for only\nsome seasons", fontsize=6, color=ink, va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def print_heatmaps(d: dict) -> dict:
    vmax = max(np.log1p(m.iloc[:28, :28].to_numpy().max()) for m in (d["V1"], d["V2"]))
    out = {}
    for M_, name in ((d["V1"], "V1_loan_heatmap"), (d["V2"], "V2_permanent_heatmap")):
        for other, suf in ((True, ""), (False, "_28x28")):
            f = REP / "figures" / f"{name}{suf}_print.pdf"
            print_heatmap(M_, d["catcov"], vmax, f, other)
            out[f.name] = int(M_.to_numpy().sum() if other else M_.iloc[:28, :28].to_numpy().sum())
    return out

# --------------------------------------------------------------------------------------------- tables
def tab_q1(d, V) -> str:
    return fill(r"""
\begin{table}[H]
\centering\footnotesize
\begin{tabular}{l rrr r}
\toprule
Receiving club (14 countries, verified tier) & Loans & Permanent & All & Loan share \\
\midrule
First division & <<t1_loan>> & <<t1_perm>> & <<t1_total>> & <<t1_loan_share>> \\
Second division & <<t2_loan>> & <<t2_perm>> & <<t2_total>> & <<t2_loan_share>> \\
\bottomrule
\end{tabular}
\caption{Arrivals at first- and second-division clubs of the 14 countries. Population: all loans and permanent arrivals in the master with reference season 2012/13 to 2025/26 whose receiving club has a verified tier 1 or 2 in that season (full transfer history, not the regression sample). Denominator of the loan share: the row total. By country: Table~\ref{tab:w10t2country}.}
\label{tab:w10q1}
\end{table}""", V)


def tab_sources() -> str:
    return r"""
\begin{table}[H]
\centering\footnotesize
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{P{3.6cm} L}
\toprule
Category & Sources and status \\
\midrule
Absent from our snapshot & Every second-division game, line-up and appearance, for all 14 countries and all seasons. Second-division clubs appear only in domestic cup games, which playing time excludes. The upstream project (transfermarkt-datasets) collects first divisions only. \\
Possibly available under licence & Commercial event and line-up providers (for example Stats Perform/Opta or Wyscout), or written permission from Transfermarkt. Coverage, cost and terms not yet confirmed. \\
Not collectable under access terms & Transfermarkt (terms of use forbid scraping); FBref and Sofascore (refuse automated requests); worldfootball.net (robots.txt excludes AI agents); FotMob match API (disallowed in robots.txt). \\
Available, but not sufficient & openfootball and Wikipedia: results and league tables, which identify the clubs in each division but contain no minutes. \\
\bottomrule
\end{tabularx}
\caption{Where second-division playing time could come from. Playing time needs every league game of the receiving club and the player's minutes in each; a league table is not enough.}
\label{tab:w10sources}
\end{table}"""


def tab_q2(d, V) -> str:
    ko = d["kind_out"].set_index("kind")
    col = {"exact": "recovered: exact tier verified (Wikipedia list)",
           "not12": "recovered: not tier 1 or 2 (absent from the lists)",
           "yr": "recovered: youth or reserve side, not in tier 1 or 2 (absent from the lists)",
           "not1": "partly recovered: not tier 1 (absent from the top-flight list)", "unk": "still unknown"}
    row = lambda k: " & ".join(n(ko.loc[x, col[k]]) for x in ("loan", "permanent", "Total"))
    tot = " & ".join(n(ko.loc[x, "Total"]) for x in ("loan", "permanent", "Total"))
    return r"""
\begin{table}[H]
\centering\footnotesize
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{P{4.3cm} L c rrr}
\toprule
Week 10 status & Basis & Kind of evidence & Loans & Permanent & All \\
\midrule
Exact tier & The club is named in that season's Wikipedia list of its division. & Verified & """ + row("exact") + r""" \\
Senior club, not tier 1 or 2 & The club is in neither that season's tier-1 nor tier-2 list, and passes the safeguard described in the text. & Inferred from absence & """ + row("not12") + r""" \\
Youth or reserve side, not tier 1 or 2 & As above, for a youth or reserve side. & Inferred from absence & """ + row("yr") + r""" \\
Not tier 1 & Not in that season's tier-1 list; no tier-2 list available. & Inferred, partial & """ + row("not1") + r""" \\
Still unknown & No list for the country-season, or the club may be one of a list's unmatched names. & None & """ + row("unk") + r""" \\
\midrule
Total (unknown in Week 9) & & & """ + tot + r""" \\
\bottomrule
\end{tabularx}
\caption{What happened to the """ + V["missing"] + r""" main-regression-sample episodes whose origin tier was unknown in Week 9. Population: main regression sample (calendar-first, """ + V["n_main"] + r""" episodes). The """ + V["w9_verified"] + r""" episodes verified in Week 9 are unchanged.}
\label{tab:w10q2}
\end{table}"""


def tab_q2cov(V) -> str:
    return fill(r"""
\begin{table}[H]
\centering\footnotesize
\begin{tabular}{l r rr rr rr}
\toprule
 & & \multicolumn{2}{c}{Week 9: exact tier} & \multicolumn{2}{c}{Week 10: exact tier} & \multicolumn{2}{c}{Week 10: tier or band} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
 & Episodes & n & \% & n & \% & n & \% \\
\midrule
Loans & <<loan_main>> & <<loan_w9>> & <<loan_w9_pct>> & <<loan_ex>> & <<loan_ex_pct>> & <<loan_band>> & <<loan_band_pct>> \\
Permanent arrivals & <<permanent_main>> & <<permanent_w9>> & <<permanent_w9_pct>> & <<permanent_ex>> & <<permanent_ex_pct>> & <<permanent_band>> & <<permanent_band_pct>> \\
All & <<n_main>> & <<w9_verified>> & <<w9_pct>> & <<exact_v2>> & <<exact_v2_pct>> & <<band_v2>> & <<band_v2_pct>> \\
\bottomrule
\end{tabular}
\caption{Origin-tier coverage. Population and denominator: main regression sample (calendar-first). ``Tier or band'' adds the inferred bands ``not tier 1 or 2'' and ``youth or reserve side, not tier 1 or 2'' to the exact tiers; the partial ``not tier 1'' cases are not counted as known.}
\label{tab:w10q2cov}
\end{table}""", V).replace("\\%", "\\%")


def tab_q3(V) -> str:
    return fill(r"""
\begin{table}[H]
\centering\footnotesize
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{l rr rr}
\toprule
 & \multicolumn{2}{c}{Loans} & \multicolumn{2}{c}{Permanent arrivals} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & n & \% & n & \% \\
\midrule
\multicolumn{5}{@{}l}{\textit{All decidable moves (denominator: decidable moves)}} \\
Higher division $\rightarrow$ lower division & <<l_down_n>> & <<l_down>> & <<pm_down_n>> & <<pm_down>> \\
Same level & & <<l_same>> & & <<pm_same>> \\
Lower division $\rightarrow$ higher division & <<l_up_n>> & <<l_up>> & <<pm_up_n>> & <<pm_up>> \\
Decidable moves & <<l_dec>> & & <<pm_dec>> & \\
\midrule
\multicolumn{5}{@{}l}{\textit{Domestic moves (denominator: decidable domestic moves)}} \\
Down / same / up (\%) & \multicolumn{2}{r}{<<l_dom_down>> / <<l_dom_same>> / <<l_dom_up>>} & \multicolumn{2}{r}{<<pm_dom_down>> / <<pm_dom_same>> / <<pm_dom_up>>} \\
Decidable domestic moves & <<l_dom_n>> & & <<pm_dom_n>> & \\
\midrule
\multicolumn{5}{@{}l}{\textit{Cross-border moves (denominator: decidable cross-border moves)}} \\
Down / same / up (\%) & \multicolumn{2}{r}{<<l_xb_down>> / <<l_xb_same>> / <<l_xb_up>>} & \multicolumn{2}{r}{<<pm_xb_down>> / <<pm_xb_same>> / <<pm_xb_up>>} \\
Decidable cross-border moves & <<l_xb_n>> & & <<pm_xb_n>> & \\
\midrule
\multicolumn{5}{@{}l}{\textit{Coverage (denominator: all episodes in the period)}} \\
Episodes, 2012/13 to 2025/26 & <<loans_period>> & & <<perm_period>> & \\
Both tiers known (1, 2 or 3 or lower) & <<l_both_n>> & <<l_both>> & <<pm_both_n>> & <<pm_both>> \\
Direction decidable & <<l_dec>> & <<l_dec_pct>> & <<pm_dec>> & <<pm_dec_pct>> \\
\bottomrule
\end{tabular}
\caption{Direction of moves between division levels. Population: all loans and permanent arrivals in the master with reference season 2012/13 to 2025/26 (full transfer history, not the regression sample). A move is decidable when both tier numbers are verified, or when a verified tier lies above the other side's bound (for example tier 1 to ``3 or lower''). Youth and reserve sides are not decidable. The domestic and cross-border rows add up to the decidable total.}
\label{tab:w10q3}
\end{table}""", V)


def tab_q4(d, V) -> str:
    q4 = d["now"]["q4"]
    rows = []
    for kind, lab in (("loan", "V1 loans"), ("permanent", "V2 permanent arrivals")):
        x = q4[kind]
        rows.append(f"{lab} & {n(x['episodes_in_period'])} & {n(x['in_matrix'])} & {n(x['core_28x28'])} & "
                    f"{n(x['other_row_or_col'] - x['other_to_other'])} & {n(x['other_to_other'])} & {n(x['omitted'])} \\\\")
    return r"""
\begin{table}[H]
\centering\footnotesize
\begin{tabular}{l rrrrrr}
\toprule
 & Episodes & In matrix & 28 $\times$ 28 core & Other row or column & Other to Other & Omitted \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Where the episodes of each heatmap go. Population: all loans (V1) and permanent arrivals (V2) of the master with reference season 2012/13 to 2025/26. In matrix = core + Other row or column + Other to Other; Episodes = In matrix + Omitted.}
\label{tab:w10q4}
\end{table}"""


def tab_q6(d, V) -> str:
    return fill(r"""
\begin{table}[H]
\centering\footnotesize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{l rr rr}
\toprule
 & \multicolumn{2}{c}{Main sample, stored} & \multicolumn{2}{c}{Main sample, fully validated} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Specification & N & loan coef. ($p$) & N & loan coef. ($p$) \\
\midrule
(1) Full sample, no origin control & <<s1_N>> & <<s1_b>> (<<s1_p>>) & <<v1_N>> & <<v1_b>> (<<v1_p>>) \\
(2) Complete cases, no origin control & <<s2_N>> & <<s2_b>> (<<s2_p>>) & <<v2_N>> & <<v2_b>> (<<v2_p>>) \\
(3) Same complete cases, with $\log$ origin squad value & <<s2_N>> & <<s3_b>> (<<s3_p>>) & <<v2_N>> & <<v3_b>> (<<v3_p>>) \\
\midrule
\multicolumn{5}{@{}l}{\textit{Supporting estimates}} \\
(4) Excluded episodes only (origin value missing) & <<s4_N>> & <<s4_b>> (<<s4_p>>) & <<v4_N>> & <<v4_b>> (<<v4_p>>) \\
(5) Complete cases, 730-day window, no origin control & <<s5_N>> & <<s5_b>> (<<s5_p>>) & <<v5_N>> & <<v5_b>> (<<v5_p>>) \\
(6) Same, with $\log$ origin squad value (730-day) & <<s5_N>> & <<s6_b>> (<<s6_p>>) & <<v5_N>> & <<v6_b>> (<<v6_p>>) \\
\midrule
Selection: (2) $-$ (1) & & <<s_sel>> & & <<v_sel>> \\
Control: (3) $-$ (2) & & <<s_ctl>> & & <<v_ctl>> \\
\bottomrule
\end{tabular}
\caption{Loan coefficient $\hat\beta$ (percentage points of possible league minutes) in the preferred specification, equation~\eqref{eq:w10reg}. Every model has the same controls, league-season fixed effects, position dummies and standard errors clustered by receiving club; only the sample and the origin control change. Outcome: \col{minutes_share}. Population: the calendar-first main regression sample (stored) and its fully validated subset (Week 9). Complete cases: origin squad value present and positive.}
\label{tab:w10q6}
\end{table}""", V)



# --------------------------------------------------------------------------------------------- appendix tables
def longtab(cols: str, header: str, rows: list[str], caption: str, label: str) -> str:
    return (r"{\footnotesize\setlength{\tabcolsep}{4pt}" + "\n" + rf"\begin{{longtable}}{{{cols}}}" + "\n"
            + rf"\caption{{{caption}}}\label{{{label}}}\\" + "\n\\toprule\n" + header + r" \\" + "\n\\midrule\n\\endfirsthead\n"
            + "\\toprule\n" + header + r" \\" + "\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot\n"
            + "\n".join(rows) + "\n\\end{longtable}}\n")


def app_tables(d, V) -> dict:
    T = {}
    t = d["t2c"].copy()
    t["share"] = 100 * t.loan / t.total
    t = t.sort_values("total", ascending=False)
    rows = [f"{esc(a)} & {n(b)} & {n(c2)} & {n(e)} & {p(f)} \\\\" for a, b, c2, e, f in
            t[["to_country_for_flows", "loan", "permanent", "total", "share"]].itertuples(index=False)]
    T["t2country"] = longtab("l rrrr", "Receiving country & Loans & Permanent & All & Loan share", rows,
                             "Second-division arrivals by receiving country (verified tier 2, reference season "
                             "2012/13 to 2025/26; full transfer history). Denominator of the loan share: the row "
                             "total.", "tab:w10t2country")
    S = d["mv"]
    pr = S[S.group.str.startswith("tier pair")]
    rows = []
    for cat in pr.category.unique():
        a = pr[(pr.category == cat) & (pr.kind == "loan")].iloc[0]
        b = pr[(pr.category == cat) & (pr.kind == "permanent")].iloc[0]
        rows.append(f"{esc(cat)} & {n(a['count'])} & {p(a.pct)} & {n(b['count'])} & {p(b.pct)} \\\\")
    a0 = pr[pr.kind == "loan"].iloc[0]
    b0 = pr[pr.kind == "permanent"].iloc[0]
    T["pairs"] = longtab("l rr rr", "Origin tier $\\rightarrow$ receiving tier & Loans & \\% & Permanent & \\%", rows,
                         f"Tier pairs, moves with both clubs in tier 1, tier 2 or ``3 or lower'' (denominators: {n(a0.denominator)} loans, "
                         f"{n(b0.denominator)} permanent arrivals; reference season 2012/13 to 2025/26, full "
                         "transfer history).", "tab:w10pairs")
    P = d["period"]
    rows = [f"{esc(k)} & {esc(pe)} & {n(nn)} & {p(dn)} & {p(up)} \\\\" for k, pe, nn, dn, up in
            P[["kind", "period", "n", "down_pct", "up_pct"]].itertuples(index=False)]
    T["period"] = longtab("ll rrr", "Kind & Reference season & Decidable moves & Down & Up", rows,
                          "Direction by period (denominator: decidable moves of the kind and period; full "
                          "transfer history).", "tab:w10period")
    DT = d["down_t1"]
    rows = [f"{esc(a)} & {n(b)} & {p(c2)} & {n(e)} & {p(f)} \\\\" for a, b, c2, e, f in
            DT[["origin country (tier 1)", "n loan", "down_pct loan", "n permanent", "down_pct permanent"]]
            .itertuples(index=False)]
    T["downt1"] = longtab("l rr rr", "Origin: first division of & Loans & Down & Permanent & Down", rows,
                          "Share of decidable moves out of a first-division club that go to a lower division, by "
                          "origin country (denominator: the decidable moves in the row; full transfer history).",
                          "tab:w10downt1")
    M = d["msum"]
    rows = [f"{esc(a)} & {esc(b)} & {n(c2)} & {n(e)} & {n(f)} \\\\" for a, b, c2, e, f in
            M[["method", "group14", "accept", "reject", "uncertain"]].itertuples(index=False)]
    T["matchaudit"] = longtab("ll rrr", "Matching method & Countries & Accept & Reject & Uncertain", rows,
                              f"Manual audit of new club-season matches ({V['ma_n']} rows, stratified by method, "
                              "country group, season band and youth/reserve status; fixed random seed). "
                              "\\col{token_subset_country} is the fallback step that was disabled after this audit.",
                              "tab:w10matchaudit")
    s = d["sens"]
    rows = []
    for i, rr in s.iterrows():
        f = (lambda v: f"{v:,.2f}") if "%" in i else (lambda v: n(v))
        diff = rr.iloc[2]
        ds = ("+" if diff > 0 else ("$-$" if diff < 0 else "")) + (f"{abs(diff):,.2f}" if "%" in i else n(abs(diff)))
        rows.append(f"{esc(i)} & {f(rr.iloc[0])} & {f(rr.iloc[1])} & {ds} \\\\")
    T["sens"] = longtab("l rrr", "Quantity & Primary rule & Strict rule & Difference", rows,
                        "Primary (per-club safeguard) versus strict (complete lists required) rule for ``not tier 1 "
                        "or 2''. Origin rows: main regression sample; loan and permanent rows: full transfer history "
                        "2012/13 to 2025/26. Percentages are of decidable moves or of all episodes in the period, as "
                        "in Table~\\ref{tab:w10q3}.", "tab:w10sens")
    B = d["ba"]
    rows = []
    for a, b, c2, e in B[["number", "before audit (10 Nov)", "after audit (12 Nov)", "change"]].itertuples(index=False):
        isint = float(b) == int(float(b)) and float(c2) == int(float(c2)) and "%" not in a
        f = (lambda v: n(v)) if isint else (lambda v: f"{float(v):.1f}")
        de = float(e)
        ds = ("+" if de > 0 else ("$-$" if de < 0 else "")) + (n(abs(de)) if isint else f"{abs(de):.1f}")
        rows.append(f"{esc(a)} & {f(b)} & {f(c2)} & {ds} \\\\")
    T["ba"] = longtab("L rrr".replace("L", "P{8.2cm}"), "Number & Before audit & After audit & Change", rows,
                      "Every headline number affected by the audit corrections, before (10 November) and after "
                      "(12 November). Origin rows: main regression sample; loan and permanent rows: full transfer "
                      "history 2012/13 to 2025/26.", "tab:w10ba")
    C = d["cons"]
    rows = [f"{esc(a)} & {n(b)} & {n(c2)} & {'yes' if ok else 'NO'} \\\\" for a, b, c2, ok in
            C[["check", "a", "b", "ok"]].itertuples(index=False)]
    T["cons"] = longtab("P{9.5cm} rrc", "Consistency check & Value A & Value B & Pass", rows,
                        "Automatic consistency checks between the heatmap matrices, the per-episode file and the "
                        "summary tables.", "tab:w10cons")
    Dd = d["dict"]
    rows = [f"\\col{{{a}}} & {esc(b)} \\\\" for a, b in Dd[["column", "meaning"]].itertuples(index=False)]
    T["dict"] = longtab("P{4.6cm} L".replace("L", "P{11.2cm}"), "Column & Meaning", rows,
                        "Columns of \\col{week10_episode_supplement.csv} (one row per master episode, key "
                        "\\col{event_id}). The \\col{to_} columns mirror the \\col{from_} columns for the "
                        "receiving club.", "tab:w10dict")
    s6 = d["sens6"]
    rows = [f"\\col{{{a}}} & {esc(b)} & {p(100 * c2)} & {n(e)} \\\\" for a, b, c2, e in
            s6[["variant", "definition", "coverage_all", "new_episodes_all"]].itertuples(index=False)]
    T["sv"] = longtab("P{2.6cm} P{9.6cm} rr", "Variant & Definition & Coverage & Added", rows,
                      f"Origin squad-value variants (main regression sample, {V['n_main']} episodes). Coverage: "
                      "share of the sample with a value; Added: episodes with a value that the stored rule leaves "
                      "blank. Stored in \\col{q6_squad_value/sensitivity_values.csv}; the master is unchanged.",
                      "tab:w10sv")
    mi = d["miss"]
    mm = mi[mi.population.str.startswith("main")]
    rows = []
    for reason in mm.reason.unique():
        x = mm[mm.reason == reason].set_index("kind")
        if x.loc["all", "episodes"] == 0:
            continue
        rows.append(f"{esc(x.loc['all', 'description'])} & {n(x.loc['loan', 'episodes'])} & "
                    f"{n(x.loc['permanent', 'episodes'])} & {n(x.loc['all', 'episodes'])} \\\\")
    rows.append(f"\\midrule Total missing & {n(mm[mm.kind == 'loan'].episodes.sum())} & "
                f"{n(mm[mm.kind == 'permanent'].episodes.sum())} & {n(mm[mm.kind == 'all'].episodes.sum())} \\\\")
    T["miss"] = longtab("P{9.8cm} rrr", "Reason the origin squad value is missing & Loans & Permanent & All", rows,
                        "Why \\col{from_squad_value} is missing (main regression sample). Each episode gets the first "
                        "reason that applies, in the order shown.", "tab:w10miss")
    return T


# --------------------------------------------------------------------------------------------- templates
MAIN = r"""
% ============================================================================================================
% Week 10 section. Generated by python -m src.analysis.week10_latex; do not edit numbers by hand.
% ============================================================================================================
\section{Week 10: answers to the six follow-up questions}
\label{sec:w10}

This section answers the six questions raised after the Week 9 guide. It uses the same episodes, samples and definitions as Week 9. Nothing in the master (<<master_rows>> episodes, <<master_cols>> columns), Stage 1C, the Week 9 variables or the stored regressions was changed. The new tier columns are kept in a separate per-episode file (\col{week10_episode_supplement.csv}, Appendix~\ref{app:w10dict}).

\textbf{Two populations.} Every table states which one it uses.
\begin{itemize}
\item The \textbf{main regression sample}: the calendar-first main sample of Week 9, <<n_main>> episodes (<<loan_main>> loans, <<permanent_main>> permanent arrivals); fully validated subset <<n_main_val>>. Questions 2, 5 and 6 report on it.
\item The \textbf{full transfer history}: every loan and permanent arrival in the master with reference season 2012/13 to 2025/26, <<loans_period>> loans and <<perm_period>> permanent arrivals. Questions 1, 3 and 4 describe where players move, which does not require a playing-time outcome, so they use all episodes. (The figures call permanent arrivals ``permanent transfers''.)
\end{itemize}

\textbf{How to read the answers.} Each statement is marked by the kind of evidence behind it:
\kind{Verified} checked against independent data or by hand;
\kind{Descriptive} an association in the data, not a causal effect;
\kind{Sensitivity} the same quantity under an alternative rule;
\kind{Limitation} what the data cannot tell us;
\kind{Decision} a choice for you, collected in Section~\ref{sec:w10open}.

\begin{table}[H]
\centering\footnotesize
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{P{4.9cm} L}
\toprule
Question & Short answer \\
\midrule
1. Second-division playing time? & Not with data we may use. Second divisions matter: loans are <<t2_loan_share>> of arrivals there, against <<t1_loan_share>> in first divisions. \\
2. Missing origin-club tiers? & Exact origin tier now known for <<exact_v2_pct>> of the main sample (Week 9: <<w9_pct>>); tier or band for <<band_v2_pct>>. \\
3. Loans versus permanent moves between tiers? & Loans move down far more often (<<l_down>> against <<pm_down>>); permanent arrivals move up far more often (<<pm_up>> against <<l_up>>). \\
4. 29 $\times$ 29 heatmaps? & Done (Figures~\ref{fig:w10v1} and~\ref{fig:w10v2}); they show <<l_in_pct>> of loans and <<pm_in_pct>> of permanent arrivals. \\
5. Prior-season playing time? & Answered in Week 9; unchanged. \\
6. Origin squad-value coverage? & Coverage (<<sv_cov>>) cannot be raised much without changing the variable. Adding it as a control moves the loan coefficient from <<s1_b>> to <<s3_b>>, about half through sample selection. \\
\bottomrule
\end{tabularx}
\caption{Summary of the Week 10 answers.}
\label{tab:w10summary}
\end{table}

% ------------------------------------------------------------------------------------------------------------
\qsec{Question 1: Can we include second-division playing time?}{q:w10-1}
\question{Can the playing-time analysis include players who move to second-division clubs?}

\kind{Limitation} \textbf{No, not with data we are allowed to use.} Playing time needs every league game of the receiving club and the player's minutes in each game. Our snapshot has <<snap_leagues>> domestic leagues, all of them first divisions, and contains no second-division game, line-up or appearance for any of the 14 countries in any season. This is why every receiving club in the Week 9 playing-time samples is a first-division club. Table~\ref{tab:w10sources} sorts the other possible sources.

<<TAB_SOURCES>>

\textbf{What the missing population is.} League tables do tell us which clubs were in each second division: Question 2 built second-division lists for <<t2_lists>> of the <<country_seasons>> country-seasons (14 countries, 2012/13 to 2025/26). So we can count the transfers we cannot measure, even though we cannot measure their minutes.

<<TAB_Q1>>

\kind{Descriptive} Second divisions are a loan destination. Loans are <<t2_loan_share>> of the <<t2_total>> arrivals at verified second-division clubs, against <<t1_loan_share>> of the <<t1_total>> arrivals at first-division clubs. By receiving country, the second-division loan share ranges from <<t2c_lo>> (<<t2c_lo_c>>) to <<t2c_hi>> (<<t2c_hi_c>>) (Table~\ref{tab:w10t2country}).

\textbf{Why this is a selection concern.} Write $D_i \in \{1, 2\}$ for the division of episode $i$'s receiving club. The playing-time sample conditions on $D_i = 1$, so the loan coefficient $\hat\beta$ of Week 9 estimates the loan/permanent difference among first-division destinations only. Loans to second divisions are a large group, so loans as a whole are under-represented. \kind{Limitation} The direction and size of any bias cannot be determined: it depends on how the loan/permanent difference in minutes behaves in second divisions, which is exactly what we cannot observe. If that difference is larger there, our estimate understates the average difference; if it is smaller, it overstates it.

\textbf{Scale (a projection, not data).} If second-division arrivals passed the playing-time filters at the first-division rate (<<pass_loan>> of loans and <<pass_perm>> of permanent arrivals reach the revised playing-time sample), about <<proj_loan>> loans and <<proj_perm>> permanent arrivals would join the <<revised_total>> episodes of that sample, and its loan share would rise from <<share_revised>> to about <<share_combined>>. These are not observations: no projected episode enters any regression, and the stored sample and results are unchanged.

% ------------------------------------------------------------------------------------------------------------
\qsec{Question 2: Can we identify the missing origin-club tiers?}{q:w10-2}
\question{For the main-sample episodes whose origin tier was unknown in Week 9, can the tier be recovered?}

\kind{Verified} \textbf{Mostly yes.} The exact origin tier is now known for <<exact_v2>> of the <<n_main>> main-sample episodes (<<exact_v2_pct>>), up from <<w9_verified>> (<<w9_pct>>) in Week 9. Including the inferred bands ``not tier 1 or 2'' (senior or youth/reserve side), a tier or band is known for <<band_v2>> (<<band_v2_pct>>). <<still_unknown>> episodes remain unknown.

\textbf{Why tiers were missing in Week 9.} Of the <<missing>> unknown origin tiers, <<wr_out_senior>> were senior clubs outside the 14 countries, <<wr_in_senior>> senior clubs of the 14 countries in a lower division without a list, <<wr_in_yr>> youth or reserve sides of the 14 countries and <<wr_out_yr>> youth or reserve sides elsewhere. The binding constraint was not the country itself but the absence of a league list for the club's country and season: openfootball, the Week 9 source for lower divisions, covers few seasons outside western Europe and few second divisions.

\textbf{The new source and how it is matched.} Wikipedia's league-season articles (licence CC BY-SA), read as ordinary article pages at one request per second (robots.txt allows this; the disallowed \col{/w/} and \col{/api/} endpoints were not used).
\begin{enumerate}
\item \textbf{Lists.} <<series>> league series: <<series_t1>> first divisions (the 14 countries and <<series_oth1>> others) and <<series_t2>> second divisions (the 14 countries and <<series_oth2>> others), <<pages>> season articles from 2005/06 to 2025/26, <<team_rows>> listed teams. Seasons are numbered as in Week 9 (a calendar-year league's season $Y$ is aligned with the European season starting in $Y-1$).
\item \textbf{Matching.} Each listed team is matched to a Transfermarkt club id with the Week 9 rules: exact normalised names, aliases learned from seasons that both sources cover, a short manual alias list, youth and reserve sides kept apart from their senior clubs, and ambiguous names left unmatched. <<team_matched>> teams (<<team_matched_pct>>) are matched.
\item \textbf{Rejections.} A match to a club in a season after Transfermarkt's recorded end of that club is rejected (<<rejected_end>> cases; a refounded club has a new id). A club that Wikipedia places in two divisions in one season is dropped (<<conflicts>> rows).
\item \textbf{Validation.} Week 9 tiers are never overwritten. Where Week 9 already had the club-season, Wikipedia gives the same tier in <<overlap_agree>> of <<overlap>> cases. The tier lookup grows from <<rows_w9>> to <<rows_total>> club-seasons.
\end{enumerate}

\textbf{Exact tiers versus inferred classifications.} A tier number is assigned only when the club is named in that season's list of its division; these are verified. Everything else is inferred from absence, which needs a safeguard because some list entries could not be matched to a Transfermarkt id. For a club $c$ of one of the 14 countries in season $s$, let $L_1$ and $L_2$ be the sets of Transfermarkt clubs matched to that season's tier-1 and tier-2 lists (each list with at least 8 teams), and $U$ the list entries left unmatched (including entries dropped as conflicts). We record ``not tier 1 or 2'' only if
\[
c \notin L_1 \cup L_2 \quad\text{and}\quad \forall u \in U:\;\; \text{no shared name word of 3+ letters} \;\text{ and }\; \operatorname{sim}(c, u) < <<sim_max>>,
\]
where $\operatorname{sim}$ is a string-similarity ratio between 0 and 1 (Python's \col{SequenceMatcher}). The similarity condition catches transliterations such as Metalurg/Metalurh. A youth or reserve side passing the same test gets its own band. When only a tier-1 list exists, the club can be classified only as ``not tier 1''. Being outside our 14 countries or outside the covered leagues never implies a tier.

<<TAB_Q2>>

<<TAB_Q2COV>>

Of the <<missing>> episodes, <<rec_exact>> now rest on a verified list entry, <<inferred_full>> on absence from both lists and <<part_not1>> on absence from the tier-1 list only. The exact tiers are tier 1 for <<tier1>> episodes, tier 2 for <<tier2>>, tier 3 for <<tier3>>, tier 4 for <<tier4>> and tier 5 for <<tier5>>. Of the <<still_unknown>> still unknown, <<su_nolist>> have no league list for the club's country in that season and <<su_token>> could be one of a list's unmatched names. The loan/permanent gap in exact-tier coverage narrows from <<gap_w9>> to <<gap_w10>> percentage points (Table~\ref{tab:w10q2cov}), because most newly verified origins belong to permanent arrivals (<<rec_exact_perm>> of the <<rec_exact>>).

\textbf{Audit.} We drew two stratified samples with a fixed random seed and checked each row by hand against the source pages.
\begin{itemize}
\item \kind{Verified} \textbf{New club-season matches}: <<ma_n>> audited, <<ma_acc>> accepted, <<ma_rej>> rejected, <<ma_unc>> uncertain. Every rejected or uncertain row came from a fallback step that matched names by shared words across a whole country; it was right for <<fb_correct>> of <<fb_unique>> names that other rules also match, but wrong for <<fb_wrong>> of the <<fb_n>> audited club-seasons that only it matched, so it has been disabled. All <<ma_kept>> audited matches that are still in the lookup were correct (<<ma_strata>>). The sample was stratified by matching method, country group, season band and youth/reserve status rather than drawn in proportion to the <<rows_added>> new club-seasons, so this shows that no error was found in any stratum we checked; it is not a formal estimate of the error rate of the whole lookup.
\item \kind{Verified} \textbf{``Not tier 1 or 2'' with an incomplete list}: <<ex_n>> main-sample club-seasons audited with the full team lists: <<ex_acc>> confirmed, <<ex_rej>> wrong (SL16 FC, corrected and now unclassified) and <<ex_unc>> not confirmable from the lists (<<ex_unc_list>>).
\end{itemize}
Four errors were corrected (Appendix~\ref{app:w10audit}): the fallback matched Malatyaspor's 2005/06 to 2008/09 seasons and other predecessor clubs to their successor clubs; the 2024/25 to 2025/26 Lierse and Lokeren entries were matched to the dissolved clubs' ids; transliterated names slipped past the word-based safeguard; and a reserve side that Wikipedia links to its parent club (SL16 FC, Standard Liège) was treated as matched. The corrections changed the main-sample exact origin tier count from <<ba_exact_before>> to <<ba_exact_after>>, and no conclusion (Table~\ref{tab:w10ba}).

\kind{Sensitivity} \textbf{Stricter rule.} If ``not tier 1 or 2'' also requires every list of that country-season to be complete (every unmatched entry shown not to be a club of our data), the exact tiers are identical, but the tier or band is known for <<st_band>> main-sample episodes (<<st_band_diff>> fewer): <<st_not12>> senior and <<st_yr>> youth or reserve ``not tier 1 or 2'', <<st_not1>> ``not tier 1'', <<st_unknown>> unknown. Its effect on Questions 3 and 4 is reported there; the full comparison is in Table~\ref{tab:w10sens}.

% ------------------------------------------------------------------------------------------------------------
\qsec{Question 3: Do loans and permanent transfers move between different league tiers?}{q:w10-3}
\question{Do loans and permanent transfers differ in whether the player moves up or down the league pyramid?}

\kind{Descriptive} \textbf{Yes, clearly.} Among moves whose direction can be decided, <<l_down>> of loans go to a lower division against <<pm_down>> of permanent arrivals, and <<pm_up>> of permanent arrivals go to a higher division against <<l_up>> of loans. The contrast is sharpest within a country: <<l_dom_down>> of domestic loans go down, against <<pm_dom_down>> of domestic permanent arrivals, while <<pm_dom_up>> of domestic permanent arrivals go up.

\textbf{Definitions.} Each club gets a band in the season of the move: tier 1, tier 2, ``3 or lower'' (a verified tier 3 or lower, or a senior club of the 14 countries in neither the tier-1 nor the tier-2 list), youth or reserve side, ``not tier 1'', or unknown (Question 2). With $T^{\text{from}}_i$ and $T^{\text{to}}_i$ the tier numbers (1 is the top),
\[
\text{direction}_i = \begin{cases} \text{down} & \text{if } T^{\text{to}}_i > T^{\text{from}}_i,\\ \text{same level} & \text{if } T^{\text{to}}_i = T^{\text{from}}_i,\\ \text{up} & \text{if } T^{\text{to}}_i < T^{\text{from}}_i. \end{cases}
\]
Two verified tier numbers are compared directly. When one side is only bounded (``3 or lower'', or ``not tier 1'', meaning 2 or lower), the move is decided only if the verified side is above the bound: tier 1 to ``3 or lower'' is down, tier 2 to ``3 or lower'' is down, but tier 2 to ``not tier 1'' is not decidable. Two bounded sides are never compared. Youth and reserve sides are kept out of up and down. Domestic means both clubs are in the same country.

<<TAB_Q3>>

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{presentation/direction_loans_vs_permanent.pdf}
\caption{Direction of moves, loans versus permanent arrivals, all decidable moves and separately for domestic and cross-border moves. Population: full transfer history, reference season 2012/13 to 2025/26. Counts and denominators: Table~\ref{tab:w10q3}.}
\label{fig:w10dir}
\end{figure}

\textbf{The pattern.}
\begin{itemize}
\item \kind{Descriptive} The typical loan with both tiers known goes from a first to a second division (<<l_t1t2>> of such loans, against <<pm_t1t2>> of permanent arrivals); the reverse move, second to first, is <<l_t2t1>> of loans and <<pm_t2t1>> of permanent arrivals (Table~\ref{tab:w10pairs}).
\item \kind{Descriptive} Out of a first-division club, loans go down more often than permanent arrivals in <<dt_above>> of the 14 countries. The gap is largest in <<dt_hi>> (<<dt_hi_l>> of loans against <<dt_hi_p>> of permanent arrivals); the exception is <<dt_exc>> (Table~\ref{tab:w10downt1}).
\item \kind{Descriptive} The share of loans going down falls over time, from <<l_down_first>> in <<per_first>> to <<l_down_last>> in <<per_last>> (Table~\ref{tab:w10period}).
\item About a third of both kinds cross a border (<<l_xb>> of loans, <<pm_xb>> of permanent arrivals). Cross-border moves are mostly ``same level'' for both kinds.
\end{itemize}

\textbf{Caveats.}
\begin{itemize}
\item \kind{Limitation} \textbf{Division numbers are not strength.} A move from a Danish first-division club to an English second-division club counts as ``down'', although the English club may be stronger. Cross-border up and down therefore refer to the pyramid level only. The domestic rows do not have this problem, and the contrast holds there.
\item \kind{Limitation} \textbf{Coverage.} Both tiers are known for <<l_both>> of loans but only <<pm_both>> of permanent arrivals, mostly because many permanent arrivals involve clubs in countries without lists. The shares describe the moves we can classify, which over-represent the 14 countries.
\item \kind{Sensitivity} Under the strict complete-list rule (Question 2), <<st_loan_down>> of loans go down and <<st_loan_up>> up, against <<st_perm_down>> and <<st_perm_up>> of permanent arrivals; domestic loans go down in <<st_loan_dom>> of cases. The contrast is unchanged in sign and similar in size.
\item These are descriptions of where players move. They do not say that a loan causes a player to move down, or that moving down affects playing time.
\end{itemize}

% ------------------------------------------------------------------------------------------------------------
\qsec{Question 4: What do the 29 \texorpdfstring{$\times$}{x} 29 transfer-flow heatmaps show?}{q:w10-4}
\question{Show transfer flows between the 14 countries' first and second divisions, separately for loans (V1) and permanent transfers (V2).}

\kind{Verified} \textbf{Both heatmaps are built} (Figures~\ref{fig:w10v1} and~\ref{fig:w10v2}). They show <<l_in>> of <<loans_period>> loans (<<l_in_pct>>) and <<pm_in>> of <<perm_period>> permanent arrivals (<<pm_in_pct>>), reference season 2012/13 to 2025/26.

\textbf{Structure.} Rows are the origin club (FROM), columns the receiving club (TO). The 29 categories are the 14 countries $\times$ 2 tiers in alphabetical order (tier 1 then tier 2) plus \textbf{Other}, identical in both figures. Each cell prints the \textbf{actual number} of episodes. The colour is $\log(1 + \text{count})$ on one scale shared by V1 and V2, so that cells from single units to hundreds stay distinguishable. White cells are true zeros.

\textbf{Other and omitted episodes.} \emph{Other} is a club outside the 14 countries (any tier), or a club of the 14 countries that is verified in tier 3 or lower or is ``not tier 1 or 2'' under the Question 2 rule (youth and reserve sides included). The Other row and column are grey with raw counts only, because they mix very different clubs and would dominate the colour scale. An episode is \emph{omitted} when a side's tier 1 or 2 cannot be ruled out or its country is unknown; such clubs are never put in Other. A category without a list in some season (<<cat_partial>>; marked * in the figures) loses that season's episodes to Omitted, not to Other.

<<TAB_Q4>>

Omitted sides: for loans, <<om_l_from_tier>> origins and <<om_l_to_tier>> destinations of the 14 countries whose tier 1 or 2 cannot be ruled out, and <<om_l_country>> sides with unknown country; for permanent arrivals, <<om_p_from_tier>>, <<om_p_to_tier>> and <<om_p_country>> (one episode can be omitted for both sides).

\textbf{What the heatmaps show.}
\begin{itemize}
\item \kind{Descriptive} \textbf{Most traffic involves Other.} Other to Other alone is <<l_oo_pct>> of the loans and <<pm_oo_pct>> of the permanent arrivals in the matrix; the 28 $\times$ 28 core holds <<l_core_pct>> and <<pm_core_pct>>.
\item \kind{Descriptive} \textbf{Within the core, flows are mostly domestic}: <<l_core_dom>> of core loans and <<pm_core_dom>> of core permanent arrivals stay in one country, visible as $2 \times 2$ blocks along the diagonal.
\item \kind{Descriptive} \textbf{Loans run from tier 1 to tier 2 within a country.} The largest loan cells are <<l_top>>. Across the 14 countries, <<l_core_t1t2>> core loans go from a first to the same country's second division and <<l_core_t2t1>> the other way.
\item \kind{Descriptive} \textbf{Permanent moves concentrate within first divisions and run upward more than downward}: the largest cells are <<pm_top>>. Same-country tier 2 to tier 1 moves number <<pm_core_t2t1>>, against <<pm_core_t1t2>> from tier 1 to tier 2.
\end{itemize}

\kind{Sensitivity} Under the strict complete-list rule the 28 $\times$ 28 cores are identical; more episodes are omitted (<<st_loan_om>> loans and <<st_perm_om>> permanent arrivals). Versions without the Other row and column (28 $\times$ 28) are in Appendix~\ref{app:w10heat}; the raw-count matrices are in \col{q4_heatmaps/}. The heatmaps are printed on landscape pages so that every cell count is readable at normal zoom.

\begin{landscape}
\begin{figure}[p]
\centering
\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{report/figures/V1_loan_heatmap_print.pdf}
\caption{V1: loans between country-tier categories, reference season 2012/13 to 2025/26 (full transfer history). Rows: origin club; columns: receiving club. Cell numbers are raw counts; colour is $\log(1+\text{count})$ on a scale shared with Figure~\ref{fig:w10v2}; white cells are zeros. The grey Other row and column show raw counts only. * marks a category with a division list for only some seasons (Section~\ref{q:w10-4}). Episodes shown and omitted: Table~\ref{tab:w10q4}. Tier numbers are not comparable across countries.}
\label{fig:w10v1}
\end{figure}
\end{landscape}

\begin{landscape}
\begin{figure}[p]
\centering
\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{report/figures/V2_permanent_heatmap_print.pdf}
\caption{V2: permanent arrivals between country-tier categories, reference season 2012/13 to 2025/26 (full transfer history). Same layout and colour scale as Figure~\ref{fig:w10v1}.}
\label{fig:w10v2}
\end{figure}
\end{landscape}

% ------------------------------------------------------------------------------------------------------------
\qsec{Question 5: Can we measure prior-season playing time?}{q:w10-5}
\question{Can we calculate each player's playing-time share in the season before the transfer?}

\kind{Verified} \textbf{Yes, and this was answered in Week 9} (<<W9Q5>>). No additional recovery was requested in Week 10, and the variables were not recomputed or changed.

\textbf{Method (Week 9).} The prior share uses the same definition as the playing-time share, over the season before the move and cut at the move date:
\[
\text{\col{prior_minutes_share}} = \frac{\text{player's league minutes in the previous season, up to the move}}{90 \times \text{eligible league games of the player's clubs in that period}}.
\]
A player who changed clubs during that season gets one share pooled across clubs. \col{prior_fit_adjusted_share} removes games in which the player was injured or in quarantine and did not play (injury rule v2), as in Week 9's Step 5 (injury adjustment).

\textbf{Coverage (main regression sample).} The ordinary share exists for <<q5_def>> of <<n_main>> episodes (<<q5_def_pct>>): <<q5_loan>> loans (<<q5_loan_pct>>) and <<q5_perm>> permanent arrivals (<<q5_perm_pct>>); <<q5_whole>> cover a whole previous season. The injury-adjusted share exists for <<q5_fit>> (<<q5_fit_pct>>). \kind{Limitation} Most missing values occur because none of the player's previous clubs was in a league of the snapshot (<<q5_def_miss_main>>), followed by incomplete club records (<<q5_incomplete>>), no league game before the move (<<q5_nogames>>) and a previous season outside the snapshot (<<q5_outside>>). Independent recomputations agreed for <<q5_sql>> of <<q5_sql_n>> single-club and <<q5_multi>> of <<q5_multi_n>> multi-club episodes. The variables are stored for future analysis and restrict no current sample.

% ------------------------------------------------------------------------------------------------------------
\qsec{Question 6: Can we improve origin-club squad-value coverage?}{q:w10-6}
\question{Why is the origin club's squad value missing so often, especially for permanent arrivals, and does it matter for the results?}

\kind{Verified} \textbf{Coverage cannot be raised much without changing what the variable measures.} \col{from_squad_value} covers <<sv_cov>> of the main regression sample (<<sv_cov_n>> of <<n_main>>): <<sv_loan>> of loans and <<sv_perm>> of permanent arrivals. We re-derived it from the raw valuations and reproduced all <<sv_stored>> stored values exactly; no implementation error was found.

\textbf{Definition (Week 9, unchanged).} For episode $i$, in which player $p_i$ leaves club $c_i$ on date $t_i$,
\[
\text{\col{from_squad_value}}_i = \sum_{j \in S_i} v_j(t_i), \qquad S_i = S_{c_i}(t_i) \setminus \{p_i\}, \qquad \text{defined only if } |S_i| \ge 11,
\]
where $v_j(t_i)$ is player $j$'s latest market value before $t_i$, at most 365 days old, and $S_c(t)$ is the set of players whose latest valuation before $t$ (within 365 days) lists club $c$. Excluding the moving player makes the origin value comparable with the receiving-club value (built the same way) and avoids counting the player's own value, already a control, twice.

\textbf{Why values are missing.} \kind{Limitation} Of the <<sv_miss>> missing values, <<mr_thin>> have only 1 to 10 valued players and <<mr_yr>> are youth or reserve origins (Table~\ref{tab:w10miss}). The valuation table is not a census: it holds valuations of <<sv_players_val>> players, essentially those who appear in the snapshot's covered competitions. A club outside those leagues therefore shows only the few players who also passed through a covered club. Coverage is <<sv_top>> for origins that played in a covered top flight that season and <<sv_nottop>> for all others.

\textbf{Why loans are covered better.} \kind{Descriptive} <<sv_loan_top>> of loans but only <<sv_perm_top>> of permanent arrivals come from a covered top-flight club. If permanent arrivals had the loans' mix of origins, their coverage would be <<sv_perm_mix>> instead of <<sv_perm>>: the origin mix explains <<sv_gap>> of the gap. The difference is where players come from, not whether the move is a loan.

\textbf{Attempts to recover more values} (Table~\ref{tab:w10sv}). A 730-day look-back adds only <<lb730_min11_new>> episodes (coverage <<lb730_min11_cov>>), because most thin squads are thin for lack of players, not stale valuations: of the <<sv_thin>> thin cases, only <<sv_thin_730>> reach 11 players at 730 days, and none of the <<sv_stale>> stale-only cases does. Where both exist, the 730-day value is close to the original (median absolute difference <<lb730_mad>>, correlation of logs <<lb730_corr>>). Lowering the minimum to 8 players adds <<lb365_min8_new>> (<<lb365_min8_cov>>) but sums fewer than a starting eleven, a different quantity. Adding Stage 1C registrations at 730 days reaches <<union730_min11_cov>>.

\textbf{Regression.} The preferred specification of Week 9 is
\begin{equation}
y_i = \beta\,\text{Loan}_i + \gamma^\top X_i + \alpha_{\ell(i), s(i)} + \delta_{\text{pos}(i)} + \varepsilon_i,
\label{eq:w10reg}
\end{equation}
where $y_i$ is the playing-time share (\col{minutes_share}), $X_i$ holds arrival timing, age and age squared, log relative player value and log receiving squad value, $\alpha_{\ell(i),s(i)}$ is a fixed effect for the receiving club's league $\ell(i)$ and season $s(i)$, $\delta_{\text{pos}(i)}$ a position dummy, and standard errors are clustered by receiving club (<<clusters>> clusters in the main sample). Specification (3) adds $\theta \log(\text{\col{from_squad_value}}_i)$. Table~\ref{tab:w10q6} compares three samples.

<<TAB_Q6>>

\textbf{Selection versus a new control.} Moving from (1) to (3) changes two things at once. The step from (1) to (2) keeps the model and only drops the <<sv_miss>> episodes without an origin value: \kind{Descriptive} the loan coefficient moves from <<s1_b>> to <<s2_b>> ($p = <<s2_p>>$), a pure sample-selection change of <<s_sel>> points. The step from (2) to (3) keeps the sample and adds the control: the coefficient moves to <<s3_b>> ($p = <<s3_p>>$), a further <<s_ctl>> points. So about half of the change comes from which episodes are used.
\begin{itemize}
\item \textbf{The complete cases are a selected sample.} Excluded episodes are <<ex_perm>> permanent arrivals (retained: <<rt_perm>>), <<ex_nottop>> come from clubs outside the covered top flights (retained: <<rt_nottop>>) and <<ex_yr>> from youth or reserve sides (retained: <<rt_yr>>). They also play less: mean share <<ms_ex_l>> for loans and <<ms_ex_pm>> for permanent arrivals, against <<ms_rt_l>> and <<ms_rt_pm>> when retained. Requiring the origin value raises the loan share of the sample from <<loanshare_full>> to <<loanshare_cc>>. In the excluded episodes alone the loan coefficient is <<s4_b>> ($p = <<s4_p>>$).
\item \textbf{Why the control lowers the loan coefficient.} Within complete cases, loans come from richer origin clubs (mean log origin value <<rel_loan>> against <<rel_perm>>), and a richer origin is associated with a higher share at the new club ($\hat\theta$ = <<theta>> points per log unit, $p = <<theta_p>>$). Holding origin value fixed therefore removes part of the loans' apparent advantage.
\item \kind{Sensitivity} The fully validated sample gives the same pattern (<<v1_b>>, <<v2_b>>, <<v3_b>>), and so does the 730-day window (<<s5_b>> without and <<s6_b>> with the control, $p = <<s6_p>>$).
\end{itemize}
\kind{Limitation} The negative coefficient in (3) is an association conditional on a selected sample, not a causal effect of loans. It describes comparable loan and permanent arrivals whose origin club the data record well.

% ============================================================================================================
\section{Open Methodological Decisions}
\label{sec:w10open}

The six questions are answered. The choices below are decisions about how to use these results, and the evidence does not settle them on its own. For each we give the relevant evidence, the trade-off and a cautious suggestion.

\kind{Decision} \textbf{1. Pursue licensed second-division data?}
\emph{Evidence:} loans are <<t2_loan_share>> of second-division arrivals against <<t1_loan_share>> of first-division arrivals (Question 1); the direction of the resulting bias is unknown. \emph{Trade-off:} a licence (or Transfermarkt's written permission) could add second-division episodes on the scale of the Question 1 projection (about <<proj_loan>> loans and <<proj_perm>> permanent arrivals), but costs money and time and needs line-up data of the same quality as ours. Keeping first divisions costs nothing but limits the conclusions to first-division destinations. \emph{Suggestion:} keep first divisions for now and state the limitation explicitly; ask Transfermarkt for permission first, since that is the cheapest route, and pursue a commercial licence only if loans to second divisions are central to the paper.

\kind{Decision} \textbf{2. Require complete division lists for inferred classifications?}
\emph{Evidence:} the per-club rule was confirmed for <<ex_acc>> of <<ex_n>> audited cases, with <<ex_rej>> corrected and <<ex_unc>> unconfirmable; the strict rule leaves exact tiers and the heatmap cores unchanged, lowers loans going down from <<l_down>> to <<st_loan_down>>, and shrinks the decidable moves to <<st_loan_dec>> loans and <<st_perm_dec>> permanent arrivals. \emph{Trade-off:} the strict rule is easier to defend but loses coverage, mostly for permanent arrivals. \emph{Suggestion:} keep the per-club rule as primary and report the strict rule as a sensitivity, as in this report.

\kind{Decision} \textbf{3. How to interpret cross-border movements?}
\emph{Evidence:} <<l_xb>> of loans and <<pm_xb>> of permanent arrivals cross a border, and division numbers are not comparable across countries; the domestic contrast is the strongest (<<l_dom_down>> against <<pm_dom_down>> going down). \emph{Options:} (a) keep division numbers for all moves with the caveat; (b) restrict up and down to domestic moves; (c) build a strength measure such as receiving-minus-origin squad value or UEFA country coefficients. \emph{Suggestion:} lead with domestic moves, report cross-border moves by division number only descriptively, and consider (c) only if direction becomes a regression variable.

\kind{Decision} \textbf{4. Origin squad value: primary control or robustness?}
\emph{Evidence:} complete cases are <<s2_share>> of the main sample and are selected (mostly permanent arrivals from uncovered origins are lost); about half of the change in the loan coefficient is selection. \emph{Trade-off:} as a primary control it answers ``comparable origin clubs'' but changes the population; as a robustness check it keeps the full sample. \emph{Suggestion:} keep the full-sample specification as primary and report (2) and (3) together as a robustness check, always side by side, with the 365-day window as the main definition and 730 days as a sensitivity.

\kind{Decision} \textbf{5. How to handle youth and reserve teams?}
\emph{Evidence:} <<rec_yr>> main-sample episodes come from a youth or reserve side outside tiers 1 and 2; these sides have no meaningful division in the senior pyramid, are kept out of up and down, and are <<ex_yr>> of the episodes lacking an origin squad value. \emph{Options:} a separate origin category in the analysis, pooling them with ``3 or lower'', or excluding them. \emph{Suggestion:} a separate category, because a youth or reserve side has no place in the senior pyramid, and pooling it with lower divisions would mix two different kinds of origin.

\textbf{Carried over from Week 9.}\wninecarriedlabel{} Week 9 also left open whether the playing-time analysis should require the loan ending to be observed and classified first (the calendar step). The calendar-first main sample has <<cf_N>> episodes and a loan coefficient of <<cf_b>> ($p = <<cf_p>>$); without the calendar step the sample has <<nc_N>> episodes and the coefficient is <<nc_b>> ($p = <<nc_p>>$). Nothing in Week 10 changes this evidence.
"""

APPENDIX = r"""
% ============================================================================================================
% Week 10 appendices. Generated by python -m src.analysis.week10_latex.
% ============================================================================================================
\clearpage
\section{Week 10: methods and sources}
\label{app:w10methods}

\textbf{League lists.} <<series>> Wikipedia league series. First divisions: the 14 countries and <<series_oth1>> others (<<oth1_list>>). Second divisions: the 14 countries and <<series_oth2>> others (<<oth2_list>>). <<pages>> season articles, 2005/06 to 2025/26, every one with a standings table; <<team_rows>> listed teams, of which <<team_matched>> are matched to a Transfermarkt id. Week 9 sources (snapshot games, openfootball) take precedence; a Wikipedia row only fills club-seasons Week 9 did not have.

\textbf{Completeness.} A division list counts when it has at least 8 teams. Under the strict rule a list is complete only if every unmatched entry shares no distinctive name word with any club of that country in our data, so it cannot be the club of any episode.

\textbf{Access.} Article pages only, one request per second, with a descriptive user agent; cached locally so that every rerun is offline. No other source was collected.

\section{Week 10: audit statistics}
\label{app:w10audit}

\textbf{Corrections.} (1) The disabled fallback had matched ``Malatyaspor'' to Yeni Malatyaspor for three seasons because the ``Y.'' of Transfermarkt's ``Y. Malatyaspor'' was ignored; the audit found the same failure for Køge and Nafta. (2) Learned aliases sent the 2024/25 to 2025/26 Lierse and Lokeren entries to the dissolved clubs' ids; matches after a club's recorded end are now rejected. (3) Transliterations (Metalurg/Metalurh, Chaika/Chayka, Eginiakos/Aiginiakos) are now caught by the similarity condition. (4) Rows dropped as conflicts now count as unmatched, which fixed SL16 FC.

<<T_matchaudit>>

\textbf{Exclusion audit.} <<ex_n>> main-sample club-seasons classified ``not tier 1 or 2'' from an incomplete list: <<ex_acc>> confirmed, <<ex_rej>> wrong (corrected), <<ex_unc>> not confirmable: <<ex_unc_list>>.

\textbf{Consistency.} <<cons_ok>> of <<n_cons>> automatic checks pass:

<<T_cons>>

<<T_sens>>

<<T_ba>>

\section{Week 10: additional tables}
\label{app:w10tables}

<<T_t2country>>

<<T_pairs>>

<<T_downt1>>

<<T_period>>

<<T_miss>>

<<T_sv>>

\section{Week 10: heatmaps without the Other category}
\label{app:w10heat}

\begin{landscape}
\begin{figure}[p]
\centering
\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{report/figures/V1_loan_heatmap_28x28_print.pdf}
\caption{V1 loans, 28 $\times$ 28 country-tier core only (<<l_core>> episodes; reference season 2012/13 to 2025/26, full transfer history). Same colour scale as Figure~\ref{fig:w10v1}.}
\label{fig:w10v1core}
\end{figure}
\end{landscape}

\begin{landscape}
\begin{figure}[p]
\centering
\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{report/figures/V2_permanent_heatmap_28x28_print.pdf}
\caption{V2 permanent arrivals, 28 $\times$ 28 country-tier core only (<<pm_core>> episodes; reference season 2012/13 to 2025/26, full transfer history). Same colour scale as Figure~\ref{fig:w10v1}.}
\label{fig:w10v2core}
\end{figure}
\end{landscape}

\section{Week 10: variable definitions}
\label{app:w10dict}

The tier columns are not in the master; they are in \col{week10_episode_supplement.csv}, one row per master episode, joined by \col{event_id}. A blank tier means ``not verified'', never a lower tier.

<<T_dict>>
"""

PREAMBLE_EXTRA = r"""
\usepackage{graphicx}
\usepackage{pdflscape}
\graphicspath{{../}}
\providecommand{\colc}[1]{\texttt{\detokenize{#1}}}
\providecommand{\kind}[1]{\colorbox{gray!14}{\scriptsize\sffamily\strut #1}\,}
"""


def number_scan(texts: dict[str, str]) -> pd.DataFrame:
    """Numbers typed into the template prose (placeholders removed). Every one must be a definitional constant."""
    rows = []
    for name, t in texts.items():
        x = re.sub(r"<<[A-Za-z0-9_]+>>", " ", t)
        x = re.sub(r"%.*", " ", x)                                                       # comments
        x = re.sub(r"\\(col|label|ref|eqref|includegraphics|texttt|cmidrule|multicolumn|begin|end)"
                   r"(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", " ", x)
        x = re.sub(r"\\(cmidrule)\(lr\)\{[^}]*\}", " ", x)
        x = re.sub(r"\{[^}]*\}\s*\\\\?\s*$", " ", x, flags=re.M) if False else x
        x = re.sub(r"\d{4}/\d{2}", " ", x)                                               # seasons
        x = re.sub(r"\b(19|20)\d{2}\b", " ", x)                                          # years
        x = re.sub(r"\d*\.?\d+\s*(pt|em|cm|mm|ex)\b", " ", x)                            # dimensions
        x = re.sub(r"\d*\.?\d+\\(text|line)width", " ", x)
        x = re.sub(r"\{[lrcLP0-9.@{}|]+\}", " ", x)                                      # column specs
        x = re.sub(r"gray!\d+", " ", x)
        x = re.sub(r"(Question|Questions|Section|Step|Figure|Table|Appendix)s?~?\s*\d+(\s*(,|and|to)\s*\d+)*", " ", x)
        x = re.sub(r"^\s*\(?\d\)?\.? ", " ", x, flags=re.M)                              # list numbering
        x = re.sub(r"\(\d\)", " ", x)
        x = re.sub(r"Week \d+", " ", x)
        x = re.sub(r"\\textbf\{\d\.", " ", x)
        x = re.sub(r"\{(q|sec|app|tab|fig|eq):[^}]*\}", " ", x)
        for m in re.finditer(r"(?<![A-Za-z0-9\\.])\d[\d,]*(\.\d+)?", x):
            tok = m.group(0).rstrip(",")
            rows.append({"template": name, "number": tok, "allowed": tok in CONSTANTS,
                         "context": x[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")})
    return pd.DataFrame(rows, columns=["template", "number", "allowed", "context"])


def crosscheck(d: dict, V: dict) -> pd.DataFrame:
    """Recompute headline numbers independently from the master and the per-episode supplement."""
    M = pd.read_csv(MASTER, usecols=["event_id", "kind", "in_cf_main_model_sample", "from_league_tier_status",
                                     "from_squad_value"], low_memory=False)
    X = pd.read_csv(OUT / "week10_episode_supplement.csv", low_memory=False)
    E = M.merge(X, on="event_id", how="left")
    main = E[E.in_cf_main_model_sample.astype(bool)]
    ex = main.from_tier_exact_v2.astype(str).str.lower().eq("true")
    band = main.from_tier_band_v2.notna() & ~main.from_tier_band_v2.isin(["unknown", "not 1"])
    per = E[E.tier_reference_season.between(2012, 2025) & E.kind.isin(["loan", "permanent"])]
    known = ["1", "2", "3+"]
    both = per.from_tier_band_v2.astype(str).isin(known) & per.to_tier_band_v2.astype(str).isin(known)
    L = per[both & per.kind.eq("loan")]
    t1t2 = ((L.from_tier_band_v2.astype(str) == "1") & (L.to_tier_band_v2.astype(str) == "2")).sum()
    out = []

    def add(label, text, value):
        out.append({"claim": label, "in_report": text, "recomputed": value,
                    "ok": str(text).replace(",", "").replace("\\%", "") == str(value)})
    add("main regression sample", V["n_main"], str(len(main)))
    add("main: exact origin tier (Week 10)", V["exact_v2"], str(int(ex.sum())))
    add("main: exact origin tier (%)", V["exact_v2_pct"], f"{100 * ex.mean():.1f}")
    add("main: origin tier or band known", V["band_v2"], str(int(band.sum())))
    add("main: Week 9 verified origin tier", V["w9_verified"], str(int(main.from_league_tier_status.eq("verified").sum())))
    add("loans in period", V["loans_period"], str(int(per.kind.eq("loan").sum())))
    add("permanent in period", V["perm_period"], str(int(per.kind.eq("permanent").sum())))
    add("loans with both tiers known", V["l_both_n"], str(int(len(L))))
    add("loan tier pairs: tier 1 -> tier 2 (%)", V["l_t1t2"], f"{100 * t1t2 / len(L):.1f}")
    for kind, k, Mx in (("loan", "l", d["V1"]), ("permanent", "pm", d["V2"])):
        P = per[per.kind.eq(kind)]
        inm = P.from_flow_category.notna() & P.to_flow_category.notna()
        add(f"{kind}: heatmap episodes shown", V[f"{k}_in"], str(int(inm.sum())))
        add(f"{kind}: heatmap matrix total", V[f"{k}_in"], str(int(Mx.to_numpy().sum())))
    sv = main.from_squad_value
    add("main: origin squad value coverage (%)", V["sv_cov"], f"{100 * (sv.notna() & (sv > 0)).mean():.1f}")
    sp = spec(d, "main (calendar")
    add("spec (1) N equals main sample", V["s1_N"], str(len(main)))
    add("spec (2) N equals origin-value complete cases", V["s2_N"], str(int((sv.notna() & (sv > 0)).sum())))
    add("decomposition adds up: (3)-(1) = selection + control",
        f"{float(sp.loc['3', 'loan_coef_pp']) - float(sp.loc['1', 'loan_coef_pp']):.6f}",
        f"{float(d['dec'].iloc[0].selection_change + d['dec'].iloc[0].control_change):.6f}")
    return pd.DataFrame(out)


def build_main(d, V) -> str:
    T = {"TAB_SOURCES": tab_sources(), "TAB_Q1": tab_q1(d, V), "TAB_Q2": tab_q2(d, V), "TAB_Q2COV": tab_q2cov(V),
         "TAB_Q3": tab_q3(V), "TAB_Q4": tab_q4(d, V), "TAB_Q6": tab_q6(d, V)}
    V2_ = dict(V, **T, W9Q5=r"\wnineref{q:5}{Section 10.5}",
               q5_def_miss_main=V["q5_nocov"])
    return fill(MAIN, V2_)


def build_appendix(d, V) -> str:
    A = app_tables(d, V)
    return fill(APPENDIX, dict(V, **{f"T_{k}": v for k, v in A.items()}))


def standalone(v5: str) -> str:
    """Week 10 report on its own, with the Week 9 guide's preamble (layout only; no Week 9 text is used)."""
    pre = v5.split(r"\begin{document}")[0]
    pre = pre.replace(r"\title{\vspace{-1.5em}Football Contracts: Data and Sample Guide}",
                      r"\title{\vspace{-1.5em}Football Contracts: Week 10 Report}")
    pre = re.sub(r"\\date\{[^}]*\}", r"\\date{12 November 2026}", pre)
    return pre + PREAMBLE_EXTRA + r"""
% references to the Week 9 guide (Transfermarkt_Dataset_Week_9.pdf, 8 November 2026) are given as text
\newcommand{\wnineref}[2]{Week 9 guide, #2}
\newcommand{\wninecarriedlabel}{}
\begin{document}
\maketitle
\vspace{-1.2em}

This report continues the Week 9 Data and Sample Guide (8 November 2026) and answers the six questions raised at the Week 10 meeting. Section~\ref{sec:w10} gives the answers, each question opening with a direct answer followed by the method and the evidence; Section~\ref{sec:w10open} collects the decisions we need from you. The appendices give methods, audit statistics, further tables, the heatmaps without the Other category and the definitions of the new columns. Terms such as \emph{episode}, \emph{calendar-first} and \emph{fully validated} are used exactly as in the Week 9 guide. Every number is generated from the audited Week 10 outputs.

{\small\setlength{\parskip}{0pt}\tableofcontents}

\input{week10_main}

\clearpage
\appendix
\input{week10_appendix}

\end{document}
"""


NEEDS = [  # (test substring in the Week 9 source, line to add before \begin{document} if missing)
    (r"{graphicx}", r"\usepackage{graphicx}"), (r"{pdflscape}", r"\usepackage{pdflscape}"),
    (r"{longtable}", r"\usepackage{longtable}"), (r"{booktabs}", r"\usepackage{booktabs}"),
    (r"{tabularx}", r"\usepackage{tabularx}"), (r"{amsmath}", r"\usepackage{amsmath}"),
    (r"{float}", r"\usepackage{float}"), (r"{xcolor}", r"\usepackage{xcolor}"), (r"{url}", r"\usepackage{url}"),
    (r"{needspace}", r"\usepackage{needspace}"),
    (r"\newcolumntype{P}", r"\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}"),
    (r"\newcolumntype{L}", r"\newcolumntype{L}{>{\raggedright\arraybackslash}X}"),
    (r"\DeclareUrlCommand\col", r"\DeclareUrlCommand\col{\urlstyle{tt}}"),
    (r"\newcommand{\qsec}", r"\newcommand{\qsec}[2]{\needspace{5\baselineskip}\subsection{#1}\label{#2}}"),
    (r"\newcommand{\question}", r"\newcommand{\question}[1]{\vspace{-0.3em}{\small\itshape #1}\par}"),
]


def integrate(week9_tex: Path, out: Path) -> dict:
    """Add Week 10 to the user's edited Week 9 LaTeX source. The Week 9 text is copied unchanged; only preamble
    lines that are missing, two \\input lines and (if the Week 9 source has no \\appendix) nothing else are added."""
    s = week9_tex.read_text()
    assert r"\begin{document}" in s and r"\end{document}" in s, "not a complete LaTeX document"
    added = []
    pre_add = []
    for test, line in NEEDS:
        if test not in s.split(r"\begin{document}")[0]:
            pre_add.append(line)
            added.append(line)
    has_label = lambda lab: bool(re.search(r"\\label\{" + re.escape(lab) + r"\}|\\qsec\{.*\}\{" + re.escape(lab) + r"\}", s))
    has_q5 = has_label("q:5")
    has_question = has_label("sec:question")
    defs = PREAMBLE_EXTRA.replace(r"\usepackage{graphicx}", "").replace(r"\usepackage{pdflscape}", "")
    defs += ("\\newcommand{\\wnineref}[2]{Section~\\ref{#1}}\n" if has_q5 else "\\newcommand{\\wnineref}[2]{#2}\n")
    # Week 9 refers to sec:question; if that section was removed, the carried-over paragraph takes the label
    defs += ("\\newcommand{\\wninecarriedlabel}{}\n" if has_question
             else "\\newcommand{\\wninecarriedlabel}{\\label{sec:question}}\n")
    s = s.replace(r"\begin{document}", "\n".join(pre_add) + "\n" + defs + "\\begin{document}", 1)
    # Week 10 main text goes before the appendices
    cands = [m.start() for m in (re.search(r"\\clearpage\s*\n\s*\\appendix", s), re.search(r"\\appendix", s),
                                 re.search(r"\\section\{All 42 Stage 1C columns\}", s)) if m]
    pos = min(cands) if cands else s.rindex(r"\end{document}")
    s = s[:pos] + "% ---- Week 10 (added 12 November 2026) ----\n\\input{week10_main}\n\\clearpage\n\n" + s[pos:]
    s = s.replace(r"\end{document}", "% ---- Week 10 appendices ----\n\\input{week10_appendix}\n\n\\end{document}")
    out.write_text(s)
    return {"source": str(week9_tex), "preamble_lines_added": added, "q5_label_found": has_q5,
            "sec_question_label_found": has_question, "inserted_before": "appendices" if cands else "end"}


def compile_tex(path: Path) -> dict:
    for _ in range(3):
        rr = subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", path.name], cwd=path.parent,
                            capture_output=True, text=True)
        if rr.returncode != 0:
            raise RuntimeError(rr.stdout[-4000:])
    log = path.with_suffix(".log").read_text(errors="ignore")
    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    return {"pages": int(pages.group(1)) if pages else None,
            "undefined_refs": len(re.findall(r"Reference `[^']*' on page \d+ undefined", log)),
            "undefined_list": sorted(set(re.findall(r"Reference `([^']*)' on page \d+ undefined", log))),
            "multiply_defined": len(re.findall(r"multiply defined", log)),
            "overfull": len(re.findall(r"Overfull \\hbox", log)),
            "missing_figures": len(re.findall(r"File `[^']*' not found", log))}


def main(week9: Path | None = None) -> dict:
    REP.mkdir(parents=True, exist_ok=True)
    d = load()
    V = values(d)
    res = {"print_heatmaps_cell_totals": print_heatmaps(d)}
    main_tex, app_tex = build_main(d, V), build_appendix(d, V)
    fixcap = lambda t: "\n".join(ln.replace("\\col{", "\\colc{") if "\\caption{" in ln else ln
                                  for ln in t.split("\n"))
    main_tex, app_tex = fixcap(main_tex), fixcap(app_tex)
    for t in (main_tex, app_tex):
        assert "—" not in t and "<<" not in t
    (REP / "week10_main.tex").write_text(main_tex)
    (REP / "week10_appendix.tex").write_text(app_tex)
    scan = number_scan({"MAIN": MAIN, "APPENDIX": APPENDIX, "sources table": tab_sources()})
    scan.to_csv(REP / "report_number_scan.csv", index=False)
    bad = scan[~scan.allowed]
    if len(bad):
        raise AssertionError("hand-typed numbers in templates:\n" + bad.to_string())
    X = crosscheck(d, V)
    X.to_csv(REP / "report_crosscheck.csv", index=False)
    if not X.ok.all():
        raise AssertionError("cross-check failed:\n" + X[~X.ok].to_string())
    (REP / "week10_report.tex").write_text(standalone(W9_TEX.read_text()))
    res["week10_report"] = compile_tex(REP / "week10_report.tex")
    if week9 is not None:
        res["integration"] = integrate(Path(week9), REP / "week9_week10_guide.tex")
        res["week9_week10_guide"] = compile_tex(REP / "week9_week10_guide.tex")
    for f in REP.glob("*"):
        if f.suffix in (".aux", ".out", ".toc", ".log"):
            f.unlink()
    res["number_scan_tokens"] = int(len(scan))
    res["crosscheck"] = f"{int(X.ok.sum())} of {len(X)}"
    (REP / "build_status.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--week9", help="path to the edited Week 9 guide .tex; adds Week 10 to it")
    a = ap.parse_args()
    print(json.dumps(main(a.week9), indent=1))
