"""Week 10 audit of the origin squad-value regression (Question 6).

    python -m src.analysis.week10_regression_audit

Same preferred specification as the stored results (playing_time_regression M3: loan, arrival timing, age and
age squared, log relative value, log receiving squad value; league-season fixed effects; position dummies; standard
errors clustered by receiving club; outcome minutes_share). Only the sample and the added control change.
Writes data/outputs/rebuild/professor_feedback_week10/q6_squad_value/regression_audit/.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import playing_time_regression as reg
from .week10_followup import FOURTEEN, MASTER, OUT

DIR = OUT / "q6_squad_value" / "regression_audit"


def fit(M: pd.DataFrame, mask: pd.Series, extra: list[str]) -> dict:
    C = M[mask.astype(bool).to_numpy()].copy()
    C["loan"] = C.kind.eq("loan").astype(int)
    C["age_c"] = C.age_exact - C.age_exact.mean()
    C["age_c2"] = C.age_c ** 2
    C["b_club_id"] = C.to_club_id
    m = reg.fit(C, reg.M3 + extra, fe="league_season", cats=["position_clean"], y="minutes_share")
    r = reg.coef(m)
    res = m["res"]
    r.update({"loans": int(C.kind.eq("loan").sum()), "permanent": int(C.kind.eq("permanent").sum()),
              "clusters": m["clusters"], "se": float(res.bse["loan"])})
    for e in extra:
        r[f"coef_{e}"], r[f"p_{e}"] = float(res.params[e]), float(res.pvalues[e])
    return r


def run() -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(MASTER, low_memory=False)
    SV = pd.read_csv(OUT / "q6_squad_value" / "sensitivity_values.csv",
                     usecols=["event_id", "from_squad_value_lb730_min11", "from_squad_value_missing_reason"])
    X = pd.read_csv(OUT / "week10_episode_supplement.csv", usecols=["event_id", "from_tier_band_v2"])
    M = M.merge(SV, on="event_id", how="left").merge(X, on="event_id", how="left")
    v730 = M.from_squad_value_lb730_min11
    M["log_from_sv_730"] = np.log(v730.where(v730 > 0))
    has = M.log_from_squad_value.notna() & np.isfinite(M.log_from_squad_value)
    has730 = M.log_from_sv_730.notna() & np.isfinite(M.log_from_sv_730)
    rows = []
    for samp, base in (("main (calendar-first, stored)", M.in_cf_main_model_sample),
                       ("main, fully validated", M.in_cf_main_model_sample_validated)):
        base = base.astype(bool)
        spec = [("1 full sample", base, []),
                ("2 complete cases (365-day origin value), no origin control", base & has, []),
                ("3 same complete cases, with log origin squad value", base & has, ["log_from_squad_value"]),
                ("4 excluded episodes only (origin value missing)", base & ~has, []),
                ("5 complete cases (730-day origin value), no origin control", base & has730, []),
                ("6 same, with log origin squad value (730-day)", base & has730, ["log_from_sv_730"])]
        for lab, mk, ex in spec:
            r = fit(M, mk, ex)
            rows.append({"sample": samp, "model": lab, "N": r["N"], "loans": r["loans"], "permanent": r["permanent"],
                         "clusters": r["clusters"], "loan_coef_pp": 100 * r["coef"], "se_pp": 100 * r["se"],
                         "p": r["p"], "origin_control_coef": r.get(f"coef_{ex[0]}") if ex else np.nan,
                         "origin_control_p": r.get(f"p_{ex[0]}") if ex else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(DIR / "specifications.csv", index=False)
    dec = []
    for samp, g in R.groupby("sample", sort=False):
        c = g.set_index(g.model.str[:1]).loan_coef_pp
        dec.append({"sample": samp, "full": c["1"], "complete_cases": c["2"], "with_control": c["3"],
                    "selection_change": c["2"] - c["1"], "control_change": c["3"] - c["2"],
                    "total_change": c["3"] - c["1"], "excluded_only": c["4"],
                    "selection_change_730": c["5"] - c["1"], "control_change_730": c["6"] - c["5"]})
    D = pd.DataFrame(dec)
    D.to_csv(DIR / "decomposition.csv", index=False)

    # who is excluded (main sample)
    base = M.in_cf_main_model_sample.astype(bool)
    Z = M[base].assign(group=np.where(has[base], "retained (origin value present)", "excluded (origin value missing)"))
    Z["origin_country_group"] = np.where(Z.from_country_v2.isin(FOURTEEN), "one of the 14 countries",
                                         np.where(Z.from_country_v2.notna(), "other country", "unknown"))
    Z["origin_in_covered_top_flight"] = Z.from_in_covered_top_flight_at_move.map({True: "yes", False: "no"}).fillna(
        "unknown")
    Z["season_bin"] = pd.cut(Z.tier_reference_season.astype(float), [2011, 2015, 2019, 2022, 2025],
                             labels=["2012-15", "2016-19", "2020-22", "2023-25"]).astype(str)
    prof = []
    for dim in ("kind", "origin_country_group", "from_tier_band_v2", "origin_in_covered_top_flight", "season_bin",
                "from_squad_value_missing_reason"):
        t = pd.crosstab(Z[dim].fillna("n/a"), Z.group, normalize="columns") * 100
        n = pd.crosstab(Z[dim].fillna("n/a"), Z.group)
        for idx in t.index:
            prof.append({"dimension": dim, "value": idx, **{f"{c} %": t.loc[idx, c] for c in t.columns},
                         **{f"{c} n": int(n.loc[idx, c]) for c in n.columns}})
    P = pd.DataFrame(prof)
    P.to_csv(DIR / "excluded_vs_retained_profile.csv", index=False)
    means = Z.groupby(["group", "kind"]).agg(n=("event_id", "size"), minutes_share=("minutes_share", "mean"),
                                             log_player_value=("log_player_value", "mean"),
                                             log_relative_value=("log_relative_value", "mean"),
                                             log_squad_value=("log_squad_value", "mean"), age=("age_exact", "mean"))
    means.to_csv(DIR / "excluded_vs_retained_means.csv")
    # how the origin value relates to loan status within complete cases
    cc = Z[Z.group.str.startswith("retained")]
    rel = cc.groupby("kind").log_from_squad_value.agg(["mean", "median"]).to_dict("index")
    out = {"specifications": R.to_dict("records"), "decomposition": D.to_dict("records"),
           "log_origin_value_by_kind_complete_cases": rel}
    (DIR / "numbers.json").write_text(json.dumps(out, indent=1, default=float))
    return {"R": R, "D": D, "P": P, "means": means, "rel": rel}


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 250)
    print(r["R"].round(3).to_string())
    print(r["D"].round(3).to_string())
    print(r["means"].round(3).to_string())
    print(r["rel"])
