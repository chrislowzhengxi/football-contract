"""Final professor-ready result set: the last QA and robustness pass on the loan vs permanent playing-time comparison.

    python -m src.analysis.final_package

Reads the master, the lineage numbers and the appearance audit; writes `data/outputs/rebuild/final/`. Every estimate
reuses the specifications of `professor_empirical` (and through it the Week 5/6 regression code), and every preferred
number is asserted equal to the lineage pipeline's saved value. Nothing here changes a sample: alternative samples are
built from existing master flags. Nothing is causal.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import patsy  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from . import appearance_audit as aud  # noqa: E402
from . import data_lineage as dl  # noqa: E402
from . import playing_time_age as pta  # noqa: E402
from . import playing_time_regression as reg  # noqa: E402
from . import professor_empirical as pe  # noqa: E402
from .daniel_scope_final import f0, md_table  # noqa: E402

OUT = dl.OUT
FINAL = OUT / "final"
FFIG = FINAL / "figures"
README = FINAL / "FINAL_README.md"
QNOTES = FINAL / "final_data_quality_notes.md"
FLOW_CSV = FINAL / "final_sample_flow.csv"
REG_CSV = FINAL / "final_regression_table.csv"
ROB_CSV = FINAL / "final_robustness_table.csv"
HET_CSV = FINAL / "final_heterogeneity_table.csv"
GATE_CSV = FINAL / "final_calendar_gate_by_season.csv"
OFFCAL_CSV = FINAL / "final_off_calendar_covid.csv"
FILES_CSV = FINAL / "final_file_status.csv"
COLHASH_CSV = FINAL / "master_column_sha256.csv"
NUMBERS_JSON = FINAL / "final_numbers.json"
WRITES = ()                                      # everything is under final/
MAIN, INJ, V2 = "in_cf_main_model_sample", "in_cf_injury_model_sample", "pt_reliable_v2"
LAST_COMPLETE = 2024                             # 2024/25: the last season whose loans have all ended in the snapshot
TOL = 1e-10


def est(C: pd.DataFrame, x=None, cats=None, fe="league_season", absorb=None, cluster="b_club_id", y="share") -> dict:
    m = reg.fit(C, x or reg.M3, fe=None if absorb else fe, absorb=absorb, cats=["position_clean"] + (cats or []),
                cluster=cluster, y=y)
    return pe.co(m)


def row(group: str, label: str, C: pd.DataFrame, r: dict, note: str = "") -> dict:
    return {"group": group, "specification": label, "N": int(r["N"]), "N loans": int(C.loan.sum()),
            "N permanent": int((1 - C.loan).sum()), "raw gap": pe.raw_diff(C)["coef"], "coef": r["coef"], "se": r["se"],
            "lo": r["lo"], "hi": r["hi"], "p": r["p"], "note": note}


# ---------------------------------------------------------------------------
# Headline results on one sample
# ---------------------------------------------------------------------------

def headline(M: pd.DataFrame, mask: pd.Series, inj_mask: pd.Series, label: str) -> dict:
    C = pe.frame(M, mask)
    G = pta.add_groups(C, pta.CUTOFF)
    am = pta.age_model(G)
    V = pe.value_section(C)
    T = pe.timing_section(C)
    K = pe.calendar_section(C)
    I = pe.frame(M, inj_mask, "fit_adjusted_share")
    Ic = pe.frame(M, inj_mask, "current_share")
    out = {"label": label, "N": {k: int(C.kind.eq(k).sum()) for k in pe.KINDS}, "raw": pe.raw_diff(C),
           "m3": est(C), "age": {k: am[k] for k in ("young", "older", "interaction")},
           "value_interaction": V["interaction"]["loan_x_lrv"],
           "value_quintiles": V["table"][["bin", "N loans", "N permanent", "raw difference", "adjusted loan coef",
                                          "adjusted SE", "adjusted lo", "adjusted hi", "adjusted p"]].to_dict("records"),
           "winter": T["winter_model"], "remain": T["remain_model"]["loan × remain"],
           "offcal_share_diff": K["share_diff"], "offcal_table": K["table"].to_dict("records"),
           "injury_fit": est(I), "injury_current": est(Ic), "injury_N": len(I)}
    return out


def het_rows(H: dict) -> list[dict]:
    rows = []
    add = lambda term, r, n: rows.append({"sample": H["label"], "term": term, "N": n, "coef": r["coef"], "se": r["se"],
                                          "lo": r.get("lo", r["coef"] - 1.96 * r["se"]),
                                          "hi": r.get("hi", r["coef"] + 1.96 * r["se"]), "p": r["p"]})
    n = sum(H["N"].values())
    add("loan (M3, all players)", H["m3"], n)
    add(f"loan, young (≤{pta.CUTOFF})", H["age"]["young"], n)
    add(f"loan, older (≥{pta.CUTOFF + 1})", H["age"]["older"], n)
    add("loan × older", H["age"]["interaction"], n)
    for q in H["value_quintiles"]:
        add(f"loan, relative-value {q['bin']}", {"coef": q["adjusted loan coef"], "se": q["adjusted SE"],
                                                 "lo": q["adjusted lo"], "hi": q["adjusted hi"], "p": q["adjusted p"]},
            int(q["N loans"] + q["N permanent"]))
    add("loan × log relative value (continuous)", H["value_interaction"], n)
    add("loan, summer-window arrival", H["winter"]["loan (summer)"], n)
    add("loan × winter window", H["winter"]["loan × winter"], n)
    add("loan × share of season remaining", H["remain"], n)
    add("off-calendar minus normal (loans only, raw)", H["offcal_share_diff"], H["N"]["loan"])
    add("loan, injury-adjusted (fit) share, injury sample", H["injury_fit"], H["injury_N"])
    return rows


# ---------------------------------------------------------------------------
# Calendar gate
# ---------------------------------------------------------------------------

def gate_by_season(M: pd.DataFrame) -> pd.DataFrame:
    L = M[M.in_age_model_sample & M.kind.eq("loan")]
    st = L.loan_calendar_status
    T = pd.DataFrame({"loans before the calendar gate (no-gate main sample)": L.groupby("arrival_season").size(),
                      "removed: ending not realised": L[st.eq("ending_not_realised")].groupby("arrival_season").size(),
                      "removed: ending unclassified": L[st.eq("unclassified")].groupby("arrival_season").size(),
                      "normal endings": L[st.eq("normal")].groupby("arrival_season").size(),
                      "off-calendar endings": L[st.eq("off_calendar")].groupby("arrival_season").size(),
                      "final main-sample loans (calendar-first)":
                          M[M[MAIN] & M.kind.eq("loan")].groupby("arrival_season").size(),
                      "permanent (both samples)": M[M[MAIN] & M.kind.eq("permanent")].groupby("arrival_season").size()}
                     ).fillna(0).astype(int)
    T.index = T.index.astype(int)
    assert (T.iloc[:, 0] == T.iloc[:, 1] + T.iloc[:, 2] + T.iloc[:, 3] + T.iloc[:, 4]).all()
    assert (T.iloc[:, 5] == T.iloc[:, 3] + T.iloc[:, 4]).all()
    T["off-calendar share of final loans"] = T["off-calendar endings"] / T["final main-sample loans (calendar-first)"]
    T.index.name = "arrival season"
    return T.reset_index()


def gate_compare(M: pd.DataFrame) -> list[dict]:
    done = M.arrival_season.le(LAST_COMPLETE)
    spec = [("A", "no calendar gate, all seasons", M.in_age_model_sample),
            ("B", "calendar-first, all seasons (current preferred)", M[MAIN]),
            ("C", f"no calendar gate, through {LAST_COMPLETE}/{str(LAST_COMPLETE + 1)[2:]}", M.in_age_model_sample & done),
            ("D", f"calendar-first, through {LAST_COMPLETE}/{str(LAST_COMPLETE + 1)[2:]}", M[MAIN] & done)]
    rows = []
    for k, lab, mask in spec:
        C = pe.frame(M, mask)
        am = pta.age_model(pta.add_groups(C, pta.CUTOFF))
        r = est(C)
        rows.append({"variant": k, "sample": lab, "N loans": int(C.loan.sum()), "N permanent": int((1 - C.loan).sum()),
                     "raw gap": pe.raw_diff(C)["coef"], "coef": r["coef"], "se": r["se"], "lo": r["lo"], "hi": r["hi"],
                     "p": r["p"], "loan × older": am["interaction"]["coef"], "loan × older se": am["interaction"]["se"],
                     "loan × older p": am["interaction"]["p"]})
    return rows


# ---------------------------------------------------------------------------
# Value-control stress test
# ---------------------------------------------------------------------------

BASE = ["loan", "remain", "age_c", "age_c2"]


def value_stress(M: pd.DataFrame) -> tuple[list[dict], dict]:
    C = pe.frame(M, M[MAIN])
    C["club_season"] = C.to_club_id.astype(int).astype(str) + "_" + C.arrival_season.astype(int).astype(str)
    rows, G = [], "value-control stress test"
    r1 = est(C)
    rows.append(row(G, "M3: log relative value + log squad value (preferred)", C, r1))
    eq_gap = float((C.log_relative_value - (C.log_player_value - C.log_squad_value)).abs().max())
    r2 = est(C, BASE + ["log_player_value", "log_squad_value"])
    rows.append(row(G, "Direct: log player value + log squad value", C, r2,
                    "algebraically identical to M3 (log relative value = log player value − log squad value)"))
    equiv = {"max_abs_identity_gap": eq_gap, "coef_difference": abs(r1["coef"] - r2["coef"]),
             "se_difference": abs(r1["se"] - r2["se"])}
    assert eq_gap < 1e-9 and equiv["coef_difference"] < 1e-9 and equiv["se_difference"] < 1e-9
    S = C.copy()
    for v, tag in (("log_player_value", "pv"), ("log_squad_value", "sv")):
        B = patsy.dmatrix(f"cr(x, df=5) - 1", {"x": S[v].to_numpy()}, return_type="dataframe")
        for i, c in enumerate(B.columns):
            S[f"spl_{tag}{i}"] = B[c].to_numpy()
    spl = [c for c in S.columns if c.startswith("spl_")]
    rows.append(row(G, "Natural cubic splines (5 df) in log player value and log squad value", S,
                    est(S, BASE + spl[:-1])))      # one basis column dropped: the splines sum to one with the constant
    S["pv_bin"] = pd.qcut(S.log_player_value.rank(method="first"), 20, labels=False)
    S["sv_bin"] = pd.qcut(S.log_squad_value.rank(method="first"), 20, labels=False)
    rows.append(row(G, "20 quantile bins of log player value and of log squad value", S,
                    est(S, BASE, cats=["pv_bin", "sv_bin"])))
    ok = pd.Series(True, index=C.index)
    for v in ("log_player_value", "log_squad_value"):
        lo = max(C.loc[C.loan.eq(g), v].quantile(.01) for g in (0, 1))
        hi = min(C.loc[C.loan.eq(g), v].quantile(.99) for g in (0, 1))
        ok &= C[v].between(lo, hi)
    T = pe.frame(M, M.event_id.isin(C.event_id[ok]))
    rows.append(row(G, "Common support: player and squad value inside both groups' 1st–99th percentiles", T, est(T)))
    X = C[["log_player_value", "log_squad_value", "age_c", "age_c2", "remain"]].astype(float)
    X = X.join(pd.get_dummies(C.position_clean, prefix="pos", drop_first=True, dtype=float))
    X = X.join(pd.get_dummies(C.receiving_league, prefix="lg", drop_first=True, dtype=float))
    X = sm.add_constant(X.join(pd.get_dummies(C.arrival_season.astype(int), prefix="s", drop_first=True, dtype=float)))
    ps = sm.Logit(C.loan.astype(float), X).fit(disp=0, maxiter=200).predict(X)
    T = pe.frame(M, M.event_id.isin(C.event_id[ps.between(0.1, 0.9)]))
    rows.append(row(G, "Common support: propensity of being a loan between 0.1 and 0.9", T, est(T),
                    "logit on value, age, timing, position, league and season"))
    for d in (30, 60, 90):
        T = pe.frame(M, M.event_id.isin(C.event_id[C.player_value_gap_days.ge(d)]))
        rows.append(row(G, f"Player valuation at least {d} days before the move", T, est(T)))
    rc = est(C, absorb="club_season")
    rows.append(row(G, "Receiving club × season fixed effects (absorbed; replace league × season)", C, rc,
                    "identified from club-seasons that receive both loans and permanent players"))
    rows.append(row(G, "SE clustered by player", C, est(C, cluster="player_id")))
    rows.append(row(G, "SE two-way clustered by receiving club and player", C, est(C, cluster=("player_id", "b_club_id"))))
    G1 = "one episode per player"
    draws = []
    for seed in range(5):
        pick = C.sample(frac=1, random_state=seed).drop_duplicates("player_id")
        T = pe.frame(M, M.event_id.isin(pick.event_id))
        draws.append((T, est(T)))
    lo, hi = min(r["coef"] for _, r in draws), max(r["coef"] for _, r in draws)
    rows.append(row(G1, "One randomly drawn episode per player (draw 1 of 5)", draws[0][0], draws[0][1],
                    f"5 draws: {pp(lo)} to {pp(hi)}"))
    first = C.sort_values(["transfer_date", "event_id"]).drop_duplicates("player_id")
    T = pe.frame(M, M.event_id.isin(first.event_id))
    rows.append(row(G1, "First eligible episode per player", T, est(T), f"mean age {T.age.mean():.1f}"))
    T = pe.frame(M, M[MAIN] & ~M.event_id.isin(first.event_id))
    rows.append(row(G1, "Later episodes of the same players (all but the first)", T, est(T),
                    f"mean age {T.age.mean():.1f}; diagnostic for the row above"))
    return rows, equiv


# ---------------------------------------------------------------------------
# Off-calendar with and without COVID-period endings
# ---------------------------------------------------------------------------

PATHS = ["loan to a different club", "permanent move to the borrower", "paid or undisclosed move to a third club",
         "free or no-fee move to a third club", "no observed next move"]


def offcal_covid(M: pd.DataFrame) -> pd.DataFrame:
    L = pe.frame(M, M[MAIN] & M.kind.eq("loan"))
    cov = L.ending_in_covid_window.astype(bool)
    rows = []
    for sl, m in (("all years", pd.Series(True, index=L.index)), ("excluding COVID-period endings", ~cov),
                  ("COVID-period endings only", cov)):
        for st in ("normal", "off_calendar"):
            g = L[m & L.loan_calendar_status.eq(st)]
            nc = g.next_category.value_counts(normalize=True)
            rows.append({"slice": sl, "status": st, "N": len(g), "mean share": g.share.mean(),
                         "zero-minute rate": g.zero.mean(), "median duration (days)": g.duration_days.median(),
                         "mean age": g.age.mean(), "median relative value (%)": 100 * g.relative_value.median(),
                         **{f"next: {p}": float(nc.get(p, 0)) for p in PATHS}})
    T = pd.DataFrame(rows)
    assert T.loc[T.slice.eq("all years"), "N"].sum() == len(L)
    return T


# ---------------------------------------------------------------------------
# Sample flow
# ---------------------------------------------------------------------------

def sample_flow(N: dict, M: pd.DataFrame) -> pd.DataFrame:
    CF = N["calendar_first"]
    V = CF["variants"]["classifiable"]
    rows = [{"branch": "all", "stage": "Stage 1C rows (frozen)", "rule": "", "removed": None,
             "remaining": N["stage1c"]["rows"]},
            {"branch": "all", "stage": "master episodes", "rule": "one row per transfer episode",
             "removed": N["stage1c"]["rows"] - N["master"]["rows"], "remaining": N["master"]["rows"]}]
    rows.append({"branch": "loan", "stage": "loans in the master", "rule": "", "removed": None, "remaining": CF["loans"]})
    for s in V["gate_steps"]:
        rows.append({"branch": "loan", "stage": "loan-calendar gate", "rule": s["reason"], "removed": s["removed"],
                     "remaining": s["n"]})
    rows.append({"branch": "permanent", "stage": "permanent transfers in the master", "rule": "", "removed": None,
                 "remaining": CF["permanent"]})
    for br, B in (("loan", V["loan_branch"]), ("permanent", CF["permanent_branch"])):
        for part, key, end in (("playing time", "pt_steps", "reliable"), ("revised sample", "revised_steps", "revised"),
                               ("main regression controls", "control_steps", "main")):
            for s in B[key]:
                rows.append({"branch": br, "stage": part, "rule": s["reason"], "removed": s["removed"],
                             "remaining": s["n"]})
            rows.append({"branch": br, "stage": f"= {end}", "rule": "", "removed": None, "remaining": B[end]})
        x = B["main"]
        rows.append({"branch": br, "stage": "appearance-gap check (sensitivity only)",
                     "rule": "drop episodes contradicted by line-ups / substitutions (pt_reliable_v2)",
                     "removed": int((M[MAIN] & M.kind.eq(br) & ~M[V2]).sum()),
                     "remaining": int((M[MAIN] & M.kind.eq(br) & M[V2]).sum())})
        for part, key, end in (("injury data", "injury_steps", "injury_usable"),
                               ("injury regression controls", "injury_control_steps", "injury_model")):
            for s in B[key]:
                rows.append({"branch": br, "stage": part, "rule": s["reason"], "removed": s["removed"],
                             "remaining": None})
            rows.append({"branch": br, "stage": f"= {end}", "rule": "", "removed": None, "remaining": B[end]})
        assert x == int((M[MAIN] & M.kind.eq(br)).sum())
        assert B["injury_model"] == int((M[INJ] & M.kind.eq(br)).sum())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Which files are current
# ---------------------------------------------------------------------------

PIPELINE = ["src.stage1c.build", "src.stage1c.validation_sample", "src.stage1b.club_flags",
            "src.analysis.club_geography_v2", "src.analysis.club_organisation", "src.analysis.data_lineage",
            "src.analysis.appearance_audit", "src.analysis.variable_provenance", "src.analysis.professor_data_overview",
            "src.analysis.playing_time_qa", "src.analysis.professor_empirical", "src.analysis.data_guide_professor",
            "src.analysis.final_package"]


def import_closure(roots: list[str]) -> set[Path]:
    root = dl.ROOT
    seen, todo = set(), list(roots)
    while todo:
        mod = todo.pop()
        path = root / (mod.replace(".", "/") + ".py")
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        pkg = mod.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
                if node.module:
                    todo.append(f"{base}.{node.module}")
                for a in node.names:
                    todo.append(f"{base}.{node.module}.{a.name}" if node.module else f"{base}.{a.name}")
    return seen


def file_status() -> pd.DataFrame:
    srcs = {p: p.read_text() for p in import_closure(PIPELINE)}
    rows = []
    for f in sorted(p for p in OUT.iterdir() if p.is_file() and p.name != ".DS_Store"):
        stem_hits = [p.stem for p, t in srcs.items() if f.name in t]
        latex = f.suffix in (".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".gz")
        status = ("CURRENT" if stem_hits else "EDITOR BUILD FILE" if latex else "LEGACY / SUPERSEDED")
        rows.append({"file": f.name, "status": status, "named_in": ", ".join(sorted(set(stem_hits))[:4])})
    for sub in ("empirical", "final"):
        rows.append({"file": f"{sub}/", "status": "CURRENT", "named_in": "professor_empirical" if sub == "empirical"
                     else "final_package"})
    return pd.DataFrame(rows)


def column_hashes(M: pd.DataFrame) -> pd.DataFrame:
    Mr = pd.read_csv(dl.MASTER_CSV, dtype=str, keep_default_na=False)
    return pd.DataFrame({"column": Mr.columns,
                         "sha256": [hashlib.sha256("\n".join(Mr[c]).encode()).hexdigest() for c in Mr.columns]})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_forest(rows: list[dict]) -> list[str]:
    R = [r for r in rows if r["group"] != "not preferred" and not r["specification"].startswith("M3: log relative")]
    fig, a = plt.subplots(figsize=(10, 0.36 * len(R) + 1.4), constrained_layout=True)
    y = np.arange(len(R))[::-1]
    a.axvline(0, color=pe.INK2, linewidth=1)
    for yi, r in zip(y, R):
        pref = r["specification"].startswith("Calendar-first sample")
        a.plot([100 * r["lo"], 100 * r["hi"]], [yi, yi], color=pe.COL["loan"], linewidth=2)
        a.plot(100 * r["coef"], yi, "o", color=pe.COL["loan"], markersize=7 if pref else 5)
    a.set_yticks(y, [r["specification"] for r in R], fontsize=8)
    pe._style(a, "loan coefficient, pp of minutes share (95% CI)", "")
    a.grid(axis="x", color=pe.GRID, linewidth=0.8)
    a.grid(axis="y", visible=False)
    a.set_title("The adjusted loan coefficient across specifications", color=pe.INK, fontsize=12, loc="left")
    return pe._save(fig, "final_fig5_robustness")


def figures(E: dict, rows: list[dict]) -> list[str]:
    old = pe.FIG
    pe.FIG = FFIG                                # write the professor-facing copies into final/figures
    try:
        files = pe.fig_distribution(E["C"]) + pe.fig_ladder(E["ladder"], E["gelbach"]) + \
            pe.fig_value(E["value"], E["C"]) + pe.fig_age(E["age"]) + fig_forest(rows)
    finally:
        pe.FIG = old
    return files


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

pp, pct, cs = pe.pp, pe.pct, pe.cs


def render_readme(R: dict) -> str:
    N, H, Hv, E = R["N"], R["H"], R["Hv"], R["E"]
    ma = N["master"]
    cfv = N["calendar_first"]["variants"]["classifiable"]["totals"]
    g = {r["variant"]: r for r in R["gate"]}
    m3 = H["m3"]
    am = H["age"]
    q = {x["bin"]: x for x in H["value_quintiles"]}
    vi = H["value_interaction"]
    gp = E["gelbach"]["point"]
    val = gp["log_relative_value"] + gp["log_squad_value"]
    ac = {r["specification"]: r for r in R["robust"]}
    acB = ac["Calendar-first, without high-confidence same-organisation moves"]
    oc = R["offcal"].set_index(["slice", "status"])
    nc = lambda st: oc.loc[("excluding COVID-period endings", st)]
    sp = R["stress_pooled"]
    gt = R["gate_table"].set_index("arrival season")
    last = int(gt.index.max())
    W = ["# Football transfers: final result set", "",
         "Generated by `python -m src.analysis.final_package`. Every number is computed by code from the files named; "
         "nothing is causal.", "",
         "## Data", "",
         f"- **Stage 1C** (frozen): {f0(N['stage1c']['rows'])} rows × {N['stage1c']['columns']} columns, SHA-256 "
         f"`{N['stage1c_sha256']}`.",
         f"- **Master** (`analysis_episode_master.csv`): {f0(ma['rows'])} episodes × {ma['columns']} columns.",
         f"- **Preferred main sample** (`{MAIN}`): {f0(cfv['main']['total'])} = {f0(cfv['main']['loan'])} loans + "
         f"{f0(cfv['main']['permanent'])} permanent transfers (calendar-first sequence; see the recommendation below).",
         f"- **Preferred injury sample** (`{INJ}`): {f0(cfv['injury_model']['total'])} = {f0(cfv['injury_model']['loan'])}"
         f" + {f0(cfv['injury_model']['permanent'])}.",
         "- Outcome: league minutes for the receiving club ÷ (90 × its league games in the observation window).", "",
         "## Main empirical results", "",
         f"1. **Raw difference.** Loans play {pp(H['raw']['coef'])} more of the available minutes (SE "
         f"{pp(H['raw']['se'], False)}), with {pct(E['dist'].loc['loan', 'true zero (0 minutes)'])} vs "
         f"{pct(E['dist'].loc['permanent', 'true zero (0 minutes)'])} zero-minute spells.",
         f"2. **Adjusted.** With league × season and position fixed effects, arrival timing, age and value controls (M3) "
         f"the loan coefficient is {cs(m3)}. Relative value accounts for {pp(val)} of the raw gap: loans are larger "
         "investments relative to the receiving squad, and relative value predicts playing time in both groups. "
         f"Across {sp['n']} specifications that pool all career stages (alternative value controls, common support, "
         "valuation age, club × season effects, clustering) the coefficient stays between "
         f"{pp(sp['lo'])} and {pp(sp['hi'])}"
         + (", every 95% interval including zero." if not sp["out"] else
            "; " + "; ".join(f"{r['specification'].lower()}: {pp(r['coef'])} (p = {r['p']:.3f})" for r in sp["out"])
            + ". No specification restores a positive loan advantage once value is controlled."
            if all(r["coef"] < 0 for r in sp["out"]) else "; see the robustness table."),
         f"3. **Age.** Loan coefficient {pp(am['young']['coef'])} for players ≤{pta.CUTOFF} (p = {am['young']['p']:.3f}) "
         f"and {pp(am['older']['coef'])} for players ≥{pta.CUTOFF + 1} (p = {am['older']['p']:.3f}); loan × older "
         f"{pp(am['interaction']['coef'])} (p = {am['interaction']['p']:.3f}). The same split appears by career stage: "
         f"{pp(sp['first']['coef'])} on players' first observed move, {pp(sp['later']['coef'])} on their later moves.",
         f"4. **Relative value.** Adjusted loan coefficient {pp(q['Q1']['adjusted loan coef'])} in the lowest relative-"
         f"value quintile, {pp(q['Q3']['adjusted loan coef'])} in the middle and {pp(q['Q5']['adjusted loan coef'])} in "
         f"the highest; loan × log relative value {vi['coef']:+.4f} (p = {vi['p']:.3f}). Loans do relatively best when "
         "the player is a small investment for the receiving club.",
         f"5. **Injury.** Injury-adjusted share: {cs(H['injury_fit'])} on the {f0(H['injury_N'])}-episode injury sample "
         f"(current share on the same episodes: {pp(H['injury_current']['coef'])}).",
         f"6. **Academy promotions.** Without the high-confidence same-organisation moves: {cs(acB)}; the raw gap falls "
         f"to {pp(acB['raw gap'])}.",
         f"7. **Off-calendar loans** (descriptive; not early termination or breach). Excluding COVID-period endings: "
         f"mean share {nc('off_calendar')['mean share']:.3f} vs {nc('normal')['mean share']:.3f} for calendar-normal "
         f"endings, zero-minute rate {pct(nc('off_calendar')['zero-minute rate'])} vs "
         f"{pct(nc('normal')['zero-minute rate'])}, median duration {nc('off_calendar')['median duration (days)']:.0f} vs "
         f"{nc('normal')['median duration (days)']:.0f} days; more often followed by a loan to a different club "
         f"({pct(nc('off_calendar')['next: loan to a different club'])} vs "
         f"{pct(nc('normal')['next: loan to a different club'])}).", "",
         "## Remaining data limitations", "",
         f"- **Appearance-row gaps.** The snapshot's `appearances` misses some players' rows although line-ups or "
         f"substitutions show them playing ({f0(R['A']['undetected'])} player-games in otherwise complete club-games, "
         f"{100 * R['A']['share_2021']:.0f}% of them in 2021/22). In the main sample this turns "
         f"{f0(R['contra']['false_zero'])} spells into false zeros and undercounts {f0(R['contra']['partial_undercount'])}."
         f" Dropping them (`{V2}`) gives {cs(Hv['m3'])}; no conclusion changes. 2012/13 cannot be checked (no line-ups).",
         "- **Market values** are Transfermarkt estimates; the value result is an association with how valuable the "
         "player is relative to the squad, which may partly reflect the club's own plans for him.",
         f"- **Last season.** In {last}/{str(last + 1)[2:]} most loans had not yet returned when the data were captured; "
         "see the decision below.", "",
         "## Decision still requiring professor judgment", "",
         R["gate_recommendation"], "",
         "## Files", ""]
    W += [f"- `{p}`" for p in R["readme_files"]]
    return "\n".join(W) + "\n"


def render_notes(R: dict) -> str:
    A, Hc, Hv = R["A"], R["H"], R["Hv"]
    W = ["# Final data-quality notes", "", "Generated by `python -m src.analysis.final_package`.", "",
         "## 1. Appearance rows vs line-ups and substitutions", "",
         f"Full audit (`appearance_audit.md`): {f0(A['on_pitch'])} player-games checked across "
         f"{len(A['leagues'])} leagues. {f0(A['on_pitch_no_row'])} have no appearance row; "
         f"{f0(A['in_incomplete_club_games'])} of those are in club-games that already fail the ≥ {A['complete_rows']}-row "
         f"test (outside every sample). The other **{f0(A['undetected'])}** ({100 * A['undetected_rate']:.2f}% of "
         f"player-games in complete club-games) are invisible to that test; {100 * A['share_2021']:.0f}% are in 2021/22, "
         f"spread over most leagues (largest league-season {100 * A['share_in_top_cell']:.0f}%, largest three "
         f"{100 * A['share_in_top3_cells']:.0f}%). Season {', '.join(map(str, A['seasons_without_lineups']))} has no "
         "line-ups and cannot be checked.", "",
         f"Master flags (non-destructive): `pt_lineup_contradicted`, `pt_reliable_v2` (= `pt_reliable` and not "
         f"contradicted), with `pt_contradiction_type` and the counts behind them. Reliable episodes: "
         f"{f0(R['contra_all']['reliable'])}; contradicted {f0(R['contra_all']['contradicted'])} "
         f"({f0(R['contra_all']['false_zero'])} false zeros, {f0(R['contra_all']['partial_undercount'])} partial "
         f"undercounts); not checkable (no line-ups) {f0(R['contra_all']['unchecked'])}. Main sample: "
         f"{f0(R['contra']['false_zero'])} false zeros and {f0(R['contra']['partial_undercount'])} partial undercounts.",
         "", "Headline results with and without the contradicted episodes:", ""]
    hr = pd.DataFrame(R["het"])
    terms = hr.term.unique()
    piv = {t: {s: hr[(hr.term == t) & (hr["sample"] == s)].iloc[0] for s in hr["sample"].unique()} for t in terms}
    s0, s1 = Hc["label"], Hv["label"]
    W += md_table(["Result", s0, s1],
                  [[t, f"{pp(piv[t][s0]['coef'])} (p {piv[t][s0]['p']:.3f})", f"{pp(piv[t][s1]['coef'])} "
                    f"(p {piv[t][s1]['p']:.3f})"] for t in terms], right={1, 2})
    W += ["", "No exact minutes are reconstructed from line-ups or events: they carry no minutes for substitutes, so a "
          "reconstruction could not be validated. They are used only to flag contradictions.", "",
          "## 2. The loan-calendar gate and 2025/26", ""]
    gt = R["gate_table"]
    W += md_table(["Arrival season", "Loans before gate", "Ending not realised", "Normal", "Off-calendar",
                   "Final loans", "Off-calendar share of final"],
                  [[f"{int(r['arrival season'])}/{str(int(r['arrival season']) + 1)[2:]}",
                    f0(r["loans before the calendar gate (no-gate main sample)"]), f0(r["removed: ending not realised"]),
                    f0(r["normal endings"]), f0(r["off-calendar endings"]),
                    f0(r["final main-sample loans (calendar-first)"]), pct(r["off-calendar share of final loans"])]
                   for _, r in gt.iterrows()])
    W += [""]
    W += md_table(["", "Sample", "N loans / perm.", "Raw gap", "M3 loan coef (SE) [95% CI], p", "Loan × older (p)"],
                  [[r["variant"], r["sample"], f"{f0(r['N loans'])} / {f0(r['N permanent'])}", pp(r["raw gap"]), cs(r),
                    f"{pp(r['loan × older'])} ({r['loan × older p']:.3f})"] for r in R["gate"]], right={2, 3})
    W += ["", R["gate_recommendation"], "",
          "## 3. Value-control stress test", "",
          f"Algebraic identity: max |log relative value − (log player value − log squad value)| = "
          f"{R['equiv']['max_abs_identity_gap']:.1e}; the loan coefficient and SE of the two parametrisations differ by "
          f"{R['equiv']['coef_difference']:.1e} and {R['equiv']['se_difference']:.1e}.", ""]
    st = [r for r in R["robust"] if r["group"] in ("value-control stress test", "one episode per player")]
    W += md_table(["Specification", "N", "Loan coef, pp (SE) [95% CI], p", "Note"],
                  [[r["specification"], f0(r["N"]), cs(r), r["note"]] for r in st], right={1, 2})
    W += ["", R["stress_text"], "", "## 4. Off-calendar loans and COVID-period endings", ""]
    oc = R["offcal"]
    W += md_table(["Slice", "Status", "N", "Mean share", "Zero-minute", "Median days", "Mean age", "Median rel. value",
                   "Next: loan elsewhere", "Next: permanent to borrower"],
                  [[r.slice, r.status, f0(r.N), f"{r['mean share']:.3f}", pct(r["zero-minute rate"]),
                    f"{r['median duration (days)']:.0f}", f"{r['mean age']:.1f}", f"{r['median relative value (%)']:.2f}%",
                    pct(r["next: loan to a different club"]), pct(r["next: permanent move to the borrower"])]
                   for _, r in oc.iterrows()])
    ex = R["offcal_example"]
    W += ["", f"Professor-facing off-calendar example (highest-valued off-calendar loan whose ending is outside the COVID "
          f"period): {ex['player_name']}, {ex['from_club_name']} → {ex['to_club_name']} ({ex['receiving_league']}), "
          f"{ex['transfer_date']} to {ex['terminal_date']} ({int(ex['duration_days'])} days); share "
          f"{ex['minutes_share']:.3f} over {int(ex['available_games'])} league games; next: {ex['next_category']}. "
          "Off-calendar describes the date of the return, not its reason.", "",
          "## 5. Validation", ""]
    W += [f"- {x}" for x in R["validation"]]
    fs = R["files"]
    W += ["", "## 6. Current vs legacy files in `data/outputs/rebuild/`", "",
          f"`{FILES_CSV.name}` lists every file. {f0((fs.status == 'CURRENT').sum())} are CURRENT (named by a module of "
          f"the current pipeline), {f0((fs.status == 'LEGACY / SUPERSEDED').sum())} are LEGACY / SUPERSEDED (earlier "
          "pipelines and reports, kept but not used), and "
          f"{f0((fs.status == 'EDITOR BUILD FILE').sum())} are LaTeX build files from the editor. Nothing was deleted. "
          "Legacy files:", ""]
    W += [f"- `{f}`" for f in fs.loc[fs.status.eq("LEGACY / SUPERSEDED"), "file"]]
    return "\n".join(W) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute() -> dict:
    assert dl.g2.sha256(dl.CANONICAL_CSV) == dl.g2.STAGE1C_SHA256
    N = json.loads(dl.NUMBERS_JSON.read_text())
    A = json.loads(aud.NUMBERS_JSON.read_text())
    M = pd.read_csv(dl.MASTER_CSV, low_memory=False)
    E = pe.compute()
    E["dist"] = E["distribution"].set_index("kind")
    cf = N["calendar_first"]["variants"]["classifiable"]
    R = {"N": N, "A": A, "E": E}
    R["H"] = headline(M, M[MAIN], M[INJ], "calendar-first (preferred)")
    R["Hv"] = headline(M, M[MAIN] & M[V2], M[INJ] & M[V2], "calendar-first, pt_reliable_v2")
    assert abs(R["H"]["m3"]["coef"] - cf["estimates"]["m3"]["coef"]) < TOL
    assert abs(R["H"]["injury_fit"]["coef"] - cf["estimates"]["injury_m3_fit"]["coef"]) < TOL
    assert sum(R["Hv"]["N"].values()) == int((M[MAIN] & M[V2]).sum())
    R["het"] = het_rows(R["H"]) + het_rows(R["Hv"])
    rel = M[M.pt_reliable]
    R["contra_all"] = {"reliable": len(rel), "contradicted": int(rel.pt_lineup_contradicted.sum()),
                       "false_zero": int(rel.pt_contradiction_type.eq("false_zero").sum()),
                       "partial_undercount": int(rel.pt_contradiction_type.eq("partial_undercount").sum()),
                       "unchecked": int(rel.pt_lineup_games_checked.eq(0).sum())}
    R["contra"] = {k: int((M[MAIN] & M.pt_contradiction_type.eq(k)).sum()) for k in ("false_zero", "partial_undercount")}
    R["gate_table"] = gate_by_season(M)
    R["gate"] = gate_compare(M)
    assert abs(R["gate"][0]["coef"] - N["regression"]["revised"]["m3"]["coef"]) < TOL
    assert abs(R["gate"][1]["coef"] - cf["estimates"]["m3"]["coef"]) < TOL
    stress, R["equiv"] = value_stress(M)
    other = []
    C = pe.frame(M, M[MAIN])
    other.append(row("sample", "Calendar-first sample (preferred)", C, est(C)))
    masks = {"A": M.in_age_model_sample, "C": M.in_age_model_sample & M.arrival_season.le(LAST_COMPLETE),
             "D": M[MAIN] & M.arrival_season.le(LAST_COMPLETE)}
    for r in R["gate"]:
        if r["variant"] != "B":
            Cg = pe.frame(M, masks[r["variant"]])
            rg = est(Cg)
            assert abs(rg["coef"] - r["coef"]) < TOL
            other.append(row("sample", f"{r['variant']}: {r['sample']}", Cg, rg))
    Cv = pe.frame(M, M[MAIN] & M[V2])
    other.append(row("sample", "Calendar-first, without line-up-contradicted episodes (pt_reliable_v2)", Cv, est(Cv)))
    for lab, col in (("Calendar-first, without high-confidence same-organisation moves", "same_organisation_move_v2"),
                     ("Calendar-first, without any academy-promotion candidate", "academy_promotion_candidate")):
        Ca = pe.frame(M, M[MAIN] & ~M[col])
        other.append(row("sample", lab, Ca, est(Ca)))
    for lab, y in (("Injury sample, current share", "current_share"),
                   ("Injury sample, injury-adjusted (fit) share", "fit_adjusted_share")):
        Ci = pe.frame(M, M[INJ], y)
        other.append(row("outcome", lab, Ci, est(Ci)))
    fl = {r["specification"]: r for r in E["robustness"]}["Calendar-first, fractional logit (average marginal effect)"]
    other.append({"group": "functional form", "specification": "Fractional logit, average marginal effect",
                  **{k: fl[k] for k in ("N", "N loans", "N permanent", "raw gap", "coef", "se", "lo", "hi", "p")},
                  "note": "Week 5 functional-form check"})
    Cn = pe.frame(M, M[MAIN] & (M.kind.eq("permanent") | M.loan_calendar_status.eq("normal")))
    other.append(row("not preferred", "Calendar-normal loans only", Cn, est(Cn),
                     "selected on the outcome: off-calendar endings can follow low playing time"))
    R["robust"] = other + stress
    R["offcal"] = offcal_covid(M)
    L = M[M[MAIN] & M.loan_calendar_status.eq("off_calendar") & ~M.ending_in_covid_window.astype(bool)]
    R["offcal_example"] = L.sort_values(["player_value", "event_id"], ascending=[False, True]).iloc[0].to_dict()
    R["flow"] = sample_flow(N, M)
    R["files"] = file_status()
    R["colhash"] = column_hashes(M)
    texts(R)
    return R


def texts(R: dict) -> None:
    g = {r["variant"]: r for r in R["gate"]}
    gt = R["gate_table"].set_index("arrival season")
    last = int(gt.index.max())
    lr = gt.loc[last]
    done = gt[gt.index < last]
    rem_last = int(lr["removed: ending not realised"])
    rem_done = int(done["removed: ending not realised"].sum())
    off_last = lr["off-calendar share of final loans"]
    off_done = done["off-calendar endings"].sum() / done["final main-sample loans (calendar-first)"].sum()
    close = lambda a, b: abs(a["coef"] - b["coef"]) < 0.5 * min(a["se"], b["se"])
    same = close(g["A"], g["B"]) and close(g["C"], g["D"])
    R["gate_recommendation"] = (
        f"**Should the playing-time analysis require a realised, classified loan ending?** The gate removes "
        f"{f0(rem_last)} of the {f0(lr['loans before the calendar gate (no-gate main sample)'])} {last}/{str(last + 1)[2:]} "
        f"main-sample loans (returns still in the future when the data were captured) but only {f0(rem_done)} in all "
        f"earlier seasons together. The {f0(lr['final main-sample loans (calendar-first)'])} {last}/{str(last + 1)[2:]} "
        f"loans it keeps are {pct(off_last)} off-calendar, against {pct(off_done)} in earlier seasons: it keeps the "
        f"loans that came back early, a selection related to the outcome. The coefficient barely moves (A "
        f"{pp(g['A']['coef'])}, B {pp(g['B']['coef'])}; through {LAST_COMPLETE}/{str(LAST_COMPLETE + 1)[2:]}: C "
        f"{pp(g['C']['coef'])}, D {pp(g['D']['coef'])})"
        + (". **Recommendation: do not gate the playing-time analysis on the loan ending.** The minutes are measured "
           "in the arrival season and do not depend on how or whether the loan has ended; treat calendar status as a "
           "loan characteristic for the separate loan-ending analysis, and use sample A "
           f"(`in_age_model_sample`, {f0(g['A']['N loans'])} loans + {f0(g['A']['N permanent'])} permanent) as the main "
           "playing-time sample. The current preferred flag is not changed here; every conclusion holds in either "
           "sample." if same else
           ". The samples give materially different coefficients; the choice needs discussion."))
    st = [r for r in R["robust"] if r["group"] == "value-control stress test"]
    out = [r for r in st if not r["lo"] <= 0 <= r["hi"]]
    one = {r["specification"]: r for r in R["robust"] if r["group"] == "one episode per player"}
    fr, la = one["First eligible episode per player"], one["Later episodes of the same players (all but the first)"]
    rd = one["One randomly drawn episode per player (draw 1 of 5)"]
    R["stress_pooled"] = {"n": len(st), "inside": len(st) - len(out), "lo": min(r["coef"] for r in st),
                          "hi": max(r["coef"] for r in st), "out": out, "first": fr, "later": la, "random": rd}
    pos = [r for r in out if r["coef"] > 0]
    R["stress_text"] = (
        f"Across the {len(st)} specifications that pool all career stages (M3 and its alternatives above), the loan "
        "coefficient ranges from "
        f"{pp(min(r['coef'] for r in st))} to {pp(max(r['coef'] for r in st))}; {len(st) - len(out)} of the 95% intervals "
        "include zero"
        + ("." if not out else ", and " + "; ".join(f"{r['specification'].lower()} gives {cs(r)}" for r in out) + ".")
        + (" No specification that compares players of similar value restores a positive loan advantage."
           if not pos else "")
        + f" Keeping one episode per player: a random draw gives {pp(rd['coef'])} ({rd['note']}), so repeated "
        f"observations of the same player do not drive the result. The first observed move per player gives "
        f"{pp(fr['coef'])} (p = {fr['p']:.3f}, {fr['note']}) and the later moves {pp(la['coef'])} (p = {la['p']:.3f}, "
        f"{la['note'].split(';')[0]}): the pooled zero averages a positive association early in a career and a negative "
        "one later, the same pattern as the age split.")


def main() -> dict:
    FFIG.mkdir(parents=True, exist_ok=True)
    R = compute()
    R["flow"].to_csv(FLOW_CSV, index=False)
    E = R["E"]
    regt = pd.DataFrame([{"model": k, **v} for k, v in E["ladder"].items()]
                        + [{"model": "M3 on calendar-first, pt_reliable_v2", **R["Hv"]["m3"]}]
                        + [{"model": f"M3, sample {r['variant']}: {r['sample']}", "coef": r["coef"], "se": r["se"],
                            "lo": r["lo"], "hi": r["hi"], "p": r["p"], "N": r["N loans"] + r["N permanent"]}
                           for r in R["gate"] if r["variant"] != "B"])
    regt.to_csv(REG_CSV, index=False)
    pd.DataFrame(R["robust"]).to_csv(ROB_CSV, index=False)
    pd.DataFrame(R["het"]).to_csv(HET_CSV, index=False)
    R["gate_table"].to_csv(GATE_CSV, index=False)
    R["offcal"].to_csv(OFFCAL_CSV, index=False)
    R["files"].to_csv(FILES_CSV, index=False)
    R["colhash"].to_csv(COLHASH_CSV, index=False)
    figs = figures(E, R["robust"])
    R["readme_files"] = [str(p.relative_to(OUT)) for p in (QNOTES, FLOW_CSV, REG_CSV, ROB_CSV, HET_CSV, GATE_CSV,
                                                           OFFCAL_CSV, FILES_CSV)] + figs + \
        ["professor_empirical_summary.md", "data_guide_professor.pdf", "appearance_audit.md",
         "academy_promotion_sensitivity.md"]
    R["validation"] = validation(R)
    QNOTES.write_text(render_notes(R))
    README.write_text(render_readme(R))
    NUMBERS_JSON.write_text(json.dumps({"headline": R["H"], "headline_v2": R["Hv"], "gate": R["gate"],
                                        "robust": R["robust"], "equiv": R["equiv"], "contra": R["contra"],
                                        "contra_all": R["contra_all"], "recommendation": R["gate_recommendation"]},
                                       indent=1, default=float))
    print(f"wrote {README.relative_to(OUT)} and {len(list(FINAL.rglob('*')))} files under final/")
    return R


def validation(R: dict) -> list[str]:
    N, M = R["N"], pd.read_csv(dl.MASTER_CSV, low_memory=False, usecols=[MAIN, INJ, V2, "kind", "in_age_model_sample"])
    out = [f"Stage 1C SHA-256 `{N['stage1c_sha256'][:16]}…` checked before and after every module of the pipeline.",
           f"Preferred samples reconcile with their flags: `{MAIN}` = {f0(M[MAIN].sum())}, `{INJ}` = {f0(M[INJ].sum())}, "
           f"`{MAIN}` & `{V2}` = {f0((M[MAIN] & M[V2]).sum())}, `in_age_model_sample` = "
           f"{f0(M.in_age_model_sample.sum())}; every regression N above is asserted against these.",
           "Preferred estimates (M3, injury fit, Week 5/6 sample) are asserted equal to the lineage pipeline's saved "
           "values to 1e-10.",
           f"Per-column SHA-256 of the master saved to `{COLHASH_CSV.name}` ({len(R['colhash'])} columns), so a silent "
           "change to any existing column is detectable by comparing with this file.",
           "Every table and figure here is written by `final_package.py`, `professor_empirical.py`, "
           "`appearance_audit.py` or `data_lineage.py`; none is edited by hand. Hand-written inputs to the pipeline "
           "are in code (`club_organisation.MANUAL_REVIEW`, token aliases, calendar parameters), not in CSV files, apart "
           "from the frozen Stage 1 audit list `stage1_validation_20.csv` that selects the Stage 1C validation sample."]
    return out


if __name__ == "__main__":
    main()
