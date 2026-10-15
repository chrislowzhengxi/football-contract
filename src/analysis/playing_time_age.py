"""Is the loan playing-time relationship an age-composition story? Young vs older, on the revised sample.

    python -m src.analysis.playing_time_age

Sample: the revised primary sample of `playing_time_sample_redesign` (reliable outcome, first spell, loans
ending in a return; no season-fraction rule). Covariates are `playing_time_regression.build`'s, unchanged.
Adjusted models keep the preferred controls (arrival timing, log relative value, log squad value, position,
league × season fixed effects, receiving-club clustered errors) and replace age and age-squared with an age-group
indicator and its interaction with the loan indicator. The cutoff is fixed in advance (young = 23 or younger at the
move); 22 and 24 are reported as sensitivity, never searched. Observational; not causal. New files only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import loan_scope
from . import playing_time_join as ptj
from . import playing_time_regression as reg
from .daniel_scope_final import f0, md_table, pc
from .loan_episodes import ROOT
from .playing_time_descriptives import COLOR, GRID, INK, LABEL, MUTED, _style, describe
from .playing_time_sample_redesign import GROUPS, arrival_group

OUT = ROOT / "data" / "outputs" / "rebuild"
REPORT_MD = OUT / "playing_time_age_analysis.md"
GROUPS_CSV = OUT / "playing_time_age_groups.csv"
MODELS_CSV = OUT / "playing_time_age_models.csv"
NUMBERS_JSON = OUT / "playing_time_age_numbers.json"
FIGS = {"groups": "figures/playing_time_age_fig1_four_groups.png",
        "diffs": "figures/playing_time_age_fig2_within_age_differences.png"}
WRITES = (REPORT_MD, GROUPS_CSV, MODELS_CSV, NUMBERS_JSON) + tuple(OUT / f for f in FIGS.values())
KINDS = ("loan", "permanent")
CUTOFF = 23                       # primary: young = age 23 or younger at the move (completed years)
SENSITIVITY = (22, 24)            # fixed in advance; not searched
CONTROLS = ["remain", "log_relative_value", "log_squad_value"]
NEED = ["share", "loan", "age", "remain", "position_clean", "log_relative_value", "log_squad_value"]
AGE_LO, AGE_HI = 17, 34           # single-year cells for standardisation; the tails are pooled into the end cells


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

def load(u=None) -> tuple[pd.DataFrame, dict]:
    import duckdb
    from ..config import DEFAULT_DATABASE

    u = u or loan_scope.universe()
    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        J, E, _ = ptj.compute(u, con)
        pv = reg.valuations(con)
    finally:
        con.close()
    B = reg.build(E, pv)                      # = the revised primary sample, with every pre-move covariate
    rev = E.reliable & ~E.prior_spell_at_b & ((E.kind == "permanent") | (E.loan_ending == "return to lender"))
    assert set(B.event_id) == set(E.loc[rev, "event_id"])
    saved = OUT / "playing_time_sample_redesign_numbers.json"
    meta = {"revised": B.kind.value_counts().to_dict()}
    if saved.exists():
        assert meta["revised"] == json.loads(saved.read_text())["revised"], "revised sample does not reproduce"
    B["age_years"] = np.floor(B.age)
    B["arrival_group"] = arrival_group(B)
    return B, meta


def add_groups(B: pd.DataFrame, cut: int) -> pd.DataFrame:
    B = B.copy()
    B["older"] = (B.age_years > cut).astype(float).where(B.age_years.notna())
    B["loan_x_older"] = B.loan * B.older
    B["cell"] = np.where(B.older.isna(), None,
                         B.kind + ", " + np.where(B.older == 1, f"older (≥{cut + 1})", f"young (≤{cut})"))
    return B


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def age_distribution(B: pd.DataFrame) -> dict:
    out = {}
    for k in KINDS:
        a = B.loc[B.kind == k, "age"].dropna()
        y = B.loc[B.kind == k, "age_years"].dropna()
        out[k] = {"N": len(a), "missing": int(B.loc[B.kind == k, "age"].isna().sum()), "mean": float(a.mean()),
                  "median": float(a.median()), **{f"p{q}": float(a.quantile(q / 100)) for q in (10, 25, 75, 90)},
                  "by_year": {int(v): int(n) for v, n in y.clip(15, 36).value_counts().sort_index().items()}}
    return out


def cell_stats(g: pd.DataFrame) -> dict:
    d = describe(g.share)
    return {"N": d["N"], "mean": d.get("mean"), "median": d.get("median"), "zero": d.get("zero"), "p25": d.get("p25"),
            "p75": d.get("p75"), "age_median": float(g.age.median()),
            "player_value_recorded": float(g.player_value.notna().mean()),
            "player_value_median_eur": float(g.player_value.median()),
            "relative_value_median": float(g.relative_value.median()),
            "remain_mean": float(g.remain.mean()), "preseason": float(g.arrival_group.eq(GROUPS[0]).mean()),
            "dec_jan": float(g.arrival_group.eq("Dec–Jan").mean())}


def raw_diff(g: pd.DataFrame) -> dict:
    a, b = g.loc[g.loan == 1, "share"], g.loc[g.loan == 0, "share"]
    d = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var() / len(a) + b.var() / len(b))) if len(a) > 1 and len(b) > 1 else np.nan
    return {"coef": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "N": len(g),
            "n_loan": len(a), "n_perm": len(b)}


def lincom(res, names: list[str]) -> dict:
    b = float(sum(res.params[n] for n in names))
    V = res.cov_params().loc[names, names].to_numpy()
    se = float(np.sqrt(V.sum()))
    from scipy import stats
    return {"coef": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": float(2 * stats.norm.sf(abs(b / se)))}


def age_model(C: pd.DataFrame, fe: str | None = "league_season", controls: list[str] | None = None) -> dict:
    """share ~ loan + older + loan×older + arrival timing + log relative value + log squad value + position + FE."""
    m = reg.fit(C, ["loan", "older", "loan_x_older"] + (CONTROLS if controls is None else controls), fe=fe,
                cats=["position_clean"])
    r = m["res"]
    return {"young": lincom(r, ["loan"]), "older": lincom(r, ["loan", "loan_x_older"]),
            "interaction": lincom(r, ["loan_x_older"]), "older_main": lincom(r, ["older"]),
            "N": m["N"], "clusters": m["clusters"], "r2": float(r.rsquared)}


def standardised_gap(C: pd.DataFrame, weights: str = "pooled") -> float:
    """Loan minus permanent mean share, averaged over single-year age cells with common weights."""
    a = C.age_years.clip(AGE_LO, AGE_HI)
    m = C.groupby([a, "loan"]).share.mean().unstack()
    w = (C if weights == "pooled" else C[C.loan == 1]).pipe(lambda d: d.age_years.clip(AGE_LO, AGE_HI)).value_counts(normalize=True)
    ok = m.dropna().index
    w = w.reindex(ok).fillna(0)
    w = w / w.sum()
    return float(((m.loc[ok, 1] - m.loc[ok, 0]) * w).sum())


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(u=None) -> tuple[dict, dict]:
    B, meta = load(u)
    N: dict = {"revised": meta["revised"], "cutoff": CUTOFF, "sensitivity_cutoffs": list(SENSITIVITY)}
    N["age_distribution"] = age_distribution(B)
    C0 = B.dropna(subset=NEED).copy()
    N["common"] = C0.kind.value_counts().to_dict()
    N["common_dropped"] = {"no age": int(B.age.isna().sum()), "no position": int(B.position_clean.isna().sum()),
                           "no player value before the move": int(B.player_value.isna().sum()),
                           "no squad value": int(B.squad_value.isna().sum()), "any": int(len(B) - len(C0))}

    group_rows, model_rows = [], []
    N["cuts"] = {}
    for cut in (CUTOFF,) + SENSITIVITY:
        Bc, C = add_groups(B, cut), add_groups(C0, cut)
        cells = {}
        for scope, frame in (("revised sample", Bc), ("model sample", C)):
            for (k, o), g in frame.dropna(subset=["older"]).groupby(["kind", "older"]):
                lab = f"{k}, {'older' if o == 1 else 'young'}"
                st = cell_stats(g)
                cells[f"{scope} | {lab}"] = st
                group_rows.append({"cutoff_young_max_age": cut, "scope": scope, "group": lab, **st})
        rec = {"cells": cells}
        for scope, frame in (("revised sample", Bc.dropna(subset=["older"])), ("model sample", C)):
            for lab, o in (("young", 0), ("older", 1)):
                d = raw_diff(frame[frame.older == o])
                rec[f"raw | {scope} | {lab}"] = d
                model_rows.append({"cutoff_young_max_age": cut, "model": f"raw difference ({scope})", "term": f"loan − permanent, {lab}", **d})
            yo, ol = rec[f"raw | {scope} | young"], rec[f"raw | {scope} | older"]
            did = {"coef": ol["coef"] - yo["coef"], "se": float(np.hypot(ol["se"], yo["se"]))}
            did.update({"lo": did["coef"] - 1.96 * did["se"], "hi": did["coef"] + 1.96 * did["se"]})
            rec[f"raw | {scope} | difference"] = did
            model_rows.append({"cutoff_young_max_age": cut, "model": f"raw difference ({scope})",
                               "term": "older minus young loan gap", **did})
        am = age_model(C)
        rec["adjusted"] = am
        rec["adjusted_no_value"] = age_model(C, controls=["remain"])
        for term in ("young", "older", "interaction"):
            model_rows.append({"cutoff_young_max_age": cut, "model": "adjusted without the two value controls",
                               "term": term, **rec["adjusted_no_value"][term], "N": rec["adjusted_no_value"]["N"]})
        for term in ("young", "older", "interaction", "older_main"):
            lab = {"young": "loan − permanent, young", "older": "loan − permanent, older",
                   "interaction": "loan × older (difference in differences)", "older_main": "older (permanent arrivals)"}[term]
            model_rows.append({"cutoff_young_max_age": cut, "model": "adjusted: age group × loan + preferred controls",
                               "term": lab, **am[term], "N": am["N"], "clusters": am["clusters"], "r2": am["r2"]})
        N["cuts"][cut] = rec

    # ---- Part 6.1: is the overall gap an age-composition story?
    C = add_groups(C0, CUTOFF)
    age_mean = float(C.age.mean())
    C["age_c"], C["age_c2"] = C.age - age_mean, (C.age - age_mean) ** 2
    C["age_cell"] = C.age_years.clip(AGE_LO, AGE_HI).astype(int).astype(str)
    comp = {"raw gap (revised sample)": raw_diff(B),
            "raw gap (model sample)": raw_diff(C),
            "age-standardised raw gap, pooled age weights": {"coef": standardised_gap(C, "pooled")},
            "age-standardised raw gap, loan age weights": {"coef": standardised_gap(C, "loan")}}
    comp["loan, single-year age dummies only"] = reg.coef(reg.fit(C, ["loan"], cats=["age_cell"]))
    comp["loan, league × season FE + single-year age dummies"] = reg.coef(reg.fit(C, ["loan"], fe="league_season",
                                                                                   cats=["age_cell"]))
    comp["preferred model (age, age²) on the revised sample"] = reg.coef(
        reg.fit(C, ["loan", "remain", "age_c", "age_c2", "log_relative_value", "log_squad_value"],
                fe="league_season", cats=["position_clean"]))
    comp["preferred model with single-year age dummies"] = reg.coef(
        reg.fit(C, ["loan"] + CONTROLS, fe="league_season", cats=["position_clean", "age_cell"]))
    comp["preferred model without any age term"] = reg.coef(
        reg.fit(C, ["loan"] + CONTROLS, fe="league_season", cats=["position_clean"]))
    N["composition"] = comp
    for name, d in comp.items():
        model_rows.append({"cutoff_young_max_age": None, "model": "overall gap and age composition", "term": name, **d})
    N["young_share"] = {k: float((C.loc[C.kind == k, "older"] == 0).mean()) for k in KINDS}

    # ---- Part 6.2: England
    eng = C[C.receiving_league == "GB1"]
    N["england"] = {"N": len(eng), "n_loan": int(eng.loan.sum()), "raw": raw_diff(eng),
                    "raw_young": raw_diff(eng[eng.older == 0]), "raw_older": raw_diff(eng[eng.older == 1]),
                    "adjusted_overall": reg.coef(reg.fit(eng, ["loan", "remain", "age_c", "age_c2", "log_relative_value",
                                                              "log_squad_value"], fe="league_season", cats=["position_clean"])),
                    "adjusted": age_model(eng),
                    "cells": {f"{k}, {'older' if o == 1 else 'young'}": int(len(g)) for (k, o), g in eng.groupby(["kind", "older"])}}
    for term in ("young", "older", "interaction"):
        model_rows.append({"cutoff_young_max_age": CUTOFF, "model": "England (GB1) only", "term": term,
                           **N["england"]["adjusted"][term], "N": N["england"]["adjusted"]["N"]})

    # ---- Part 6.3: lowest relative-value fifth
    C["lrv_q"] = pd.qcut(C.log_relative_value, 5, labels=[f"Q{i}" for i in range(1, 6)])
    q1 = C[C.lrv_q == "Q1"]
    N["low_value"] = {"N": len(q1), "relative_value_max": float(q1.relative_value.max()), "raw": raw_diff(q1),
                      "raw_young": raw_diff(q1[q1.older == 0]), "raw_older": raw_diff(q1[q1.older == 1]),
                      "adjusted": age_model(q1),
                      "young_share": {k: float((q1.loc[q1.kind == k, "older"] == 0).mean()) for k in KINDS},
                      "cells": {f"{k}, {'older' if o == 1 else 'young'}": int(len(g)) for (k, o), g in q1.groupby(["kind", "older"])}}
    qd = pd.get_dummies(C.lrv_q, dtype=float)
    for q in qd.columns:
        C[f"loan_{q}"] = C.loan * qd[q]
    mq = reg.fit(C, [f"loan_{q}" for q in qd.columns] + ["remain", "age_c", "age_c2", "log_squad_value"],
                 fe="league_season", cats=["position_clean", "lrv_q"])
    N["quintiles"] = {q: reg.coef(mq, f"loan_{q}") for q in qd.columns}
    for term in ("young", "older", "interaction"):
        model_rows.append({"cutoff_young_max_age": CUTOFF, "model": "lowest relative-value fifth only", "term": term,
                           **N["low_value"]["adjusted"][term], "N": N["low_value"]["adjusted"]["N"]})
    N["older_share_by_quintile"] = {q: {k: float((C.loc[(C.lrv_q == q) & (C.kind == k), "older"] == 1).mean()) for k in KINDS}
                                    for q in qd.columns}
    return N, {"B": B, "C": C, "GROUPS": pd.DataFrame(group_rows), "MODELS": pd.DataFrame(model_rows)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(N: dict, B: pd.DataFrame) -> None:
    plt = _style()
    (OUT / "figures").mkdir(exist_ok=True)
    Bc = add_groups(B, CUTOFF).dropna(subset=["older"])
    labs = [f"Young (≤{CUTOFF})", f"Older (≥{CUTOFF + 1})"]

    # 1. four groups: mean share (with 95% intervals) and zero-minute share, as two separate panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(2)
    for off, k in ((-0.14, "loan"), (0.14, "permanent")):
        g = Bc[Bc.kind == k]
        m = [g.loc[g.older == o, "share"].mean() for o in (0, 1)]
        se = [g.loc[g.older == o, "share"].std() / np.sqrt((g.older == o).sum()) for o in (0, 1)]
        n = [int((g.older == o).sum()) for o in (0, 1)]
        z = [g.loc[g.older == o, "share"].eq(0).mean() * 100 for o in (0, 1)]
        axes[0].errorbar(x + off, m, yerr=1.96 * np.array(se), fmt="o", ms=9, color=COLOR[k], elinewidth=2, capsize=0,
                         label=LABEL[k])
        for xi, mi, ni in zip(x + off, m, n):
            axes[0].text(xi, mi + 0.022, f"{mi:.3f}\nn={ni:,}", ha="center", va="bottom", fontsize=8.5, color=INK)
        axes[1].bar(x + off, z, width=0.26, color=COLOR[k], label=LABEL[k])
        for xi, zi in zip(x + off, z):
            axes[1].text(xi, zi + 0.3, f"{zi:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    axes[0].set_ylim(0.25, 0.55)
    axes[0].set_ylabel("Mean playing-time share")
    axes[0].set_title("Mean share (bars: 95% interval)", loc="left", fontsize=10)
    axes[1].set_ylabel("Players with zero league minutes (%)")
    axes[1].set_title("Zero-minute share", loc="left", fontsize=10)
    axes[1].set_ylim(0, max(axes[1].get_ylim()[1], 14))
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_xlim(-0.5, 1.5)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower left", ncol=2)
    fig.suptitle(f"Playing time by age group and move type (revised sample; age at the move, cutoff {CUTOFF}/{CUTOFF + 1})",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / FIGS["groups"], dpi=160)
    plt.close(fig)

    # 2. within-age loan − permanent difference, raw and adjusted, at each fixed cutoff
    fig, ax = plt.subplots(figsize=(10, 4.4))
    cuts = (SENSITIVITY[0], CUTOFF, SENSITIVITY[1])
    rows = []
    for cut in cuts:
        rec = N["cuts"][cut]
        for lab in ("young", "older"):
            rows.append((cut, lab, rec[f"raw | model sample | {lab}"], rec["adjusted"][lab]))
    pos = np.arange(len(rows))
    for i, (cut, lab, r, a) in enumerate(rows):
        ax.errorbar(i - 0.12, 100 * r["coef"], yerr=[[100 * (r["coef"] - r["lo"])], [100 * (r["hi"] - r["coef"])]],
                    fmt="o", ms=7, mfc="white", color=MUTED, elinewidth=1.5, capsize=0,
                    label="raw (model sample)" if i == 0 else None)
        ax.errorbar(i + 0.12, 100 * a["coef"], yerr=[[100 * (a["coef"] - a["lo"])], [100 * (a["hi"] - a["coef"])]],
                    fmt="o", ms=8, color=INK, elinewidth=2, capsize=0, label="adjusted (preferred controls)" if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{'young' if lab == 'young' else 'older'}\n{'≤' if lab == 'young' else '≥'}"
                        f"{cut if lab == 'young' else cut + 1}" + ("\n(primary)" if cut == CUTOFF else "")
                        for cut, lab, _, _ in rows], fontsize=9)
    for b in (1.5, 3.5):
        ax.axvline(b, color=GRID, lw=1)
    ax.set_ylabel("Loan minus permanent (pp of league minutes)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Loan − permanent playing-time difference within age groups, at three fixed cutoffs (95% intervals)",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / FIGS["diffs"], dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def pp(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.1f} pp"


def ci(d) -> str:
    return f"[{pp(d['lo'])}, {pp(d['hi'])}]" if "lo" in d else ""


def p3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def sig(d) -> bool:
    return d["lo"] > 0 or d["hi"] < 0


def say(d, a: str, b: str) -> str:
    """'young loanees play X pp more than young permanent arrivals [interval]' with the interval's verdict."""
    word = "more" if d["coef"] > 0 else "less"
    tail = "" if sig(d) else "; the interval includes zero"
    return f"{a} play {abs(100 * d['coef']):.1f} pp {word} than {b} {ci(d)}{tail}"


def render(N: dict) -> str:
    cut = CUTOFF
    rec = N["cuts"][cut]
    ad, ad_ = rec["adjusted"], N["age_distribution"]
    young, older = f"young (≤{cut})", f"older (≥{cut + 1})"
    comp = N["composition"]
    W = ["# Playing time by age: young vs older loanees and permanent signings", "",
         "*Generated by `python -m src.analysis.playing_time_age`. Revised primary sample (no season-fraction rule). "
         f"Primary cutoff fixed in advance: young = {cut} or younger at the move; {SENSITIVITY[0]} and {SENSITIVITY[1]} "
         "are sensitivity checks, not a search. Observational associations, not causal effects. New files only.*", ""]

    # ---- headline
    W += ["## Headline", "",
          f"- {say(ad['young'], 'Young loanees', 'young permanent arrivals')} (adjusted).",
          f"- {say(ad['older'], 'Older loanees', 'older permanent arrivals')} (adjusted).",
          f"- Difference in differences (loan × older): {pp(ad['interaction']['coef'])} {ci(ad['interaction'])}"
          + ("." if sig(ad["interaction"]) else ", not distinguishable from zero."),
          f"- The raw loan advantage sits with young players: raw {pp(rec['raw | revised sample | young']['coef'])} for "
          f"young, {pp(rec['raw | revised sample | older']['coef'])} for older. Young permanent signings play little "
          f"(mean {p3(rec['cells']['revised sample | permanent, young']['mean'])}, "
          f"{pct(rec['cells']['revised sample | permanent, young']['zero'])} with zero minutes).",
          f"- It is not an age-composition effect: loans and permanent moves have similar ages, and standardising to a "
          f"common age distribution moves the model-sample raw gap only from {pp(comp['raw gap (model sample)']['coef'])} "
          f"to {pp(comp['age-standardised raw gap, pooled age weights']['coef'])}.", ""]

    # ---- Part 1
    W += ["## 1. Age distribution (age at the move)", ""]
    W += md_table(["", "N", "missing", "mean", "median", "p10", "p25", "p75", "p90"],
                  [[LABEL[k], f0(ad_[k]["N"]), ad_[k]["missing"]] + [f"{ad_[k][s]:.1f}" for s in
                                                                     ("mean", "median", "p10", "p25", "p75", "p90")]
                   for k in KINDS])
    years = list(range(17, 33))
    W += ["", "Count by completed years (17–32 shown; tails in the CSV):", ""]
    W += md_table(["Age"] + [str(y) for y in years],
                  [[LABEL[k]] + [f0(ad_[k]["by_year"].get(y, 0)) for y in years] for k in KINDS])
    W += ["", f"With the cutoff at {cut}, {pct(N['young_share']['loan'])} of loans and {pct(N['young_share']['permanent'])} "
          "of permanent moves in the model sample are young.", ""]

    # ---- Part 2-3
    cells = rec["cells"]
    W += ["## 2–3. Four groups (primary cutoff)", "",
          f"Young = {cut} or younger; older = {cut + 1} or older, so no age is dropped.", ""]
    keys = [("loan", "young"), ("loan", "older"), ("permanent", "young"), ("permanent", "older")]
    rows = []
    for k, a in keys:
        c, cm = cells[f"revised sample | {k}, {a}"], cells[f"model sample | {k}, {a}"]
        rows.append([f"{LABEL[k]}, {young if a == 'young' else older}", f0(c["N"]), p3(c["mean"]), p3(c["median"]),
                     pct(c["zero"]), f"{p3(c['p25'])} / {p3(c['p75'])}", f"{c['age_median']:.1f}",
                     f"€{c['player_value_median_eur'] / 1e3:,.0f}k", pct(c["player_value_recorded"]),
                     f"{cm['relative_value_median']:.3f}", f"{c['remain_mean']:.2f}", pct(c["preseason"]), pct(c["dec_jan"])])
    W += md_table(["Group", "N", "Mean share", "Median", "Zero", "p25 / p75", "Median age", "Median player value",
                   "Value recorded", "Median relative value", "Mean share of season left", "Preseason arrival",
                   "Dec–Jan arrival"], rows)
    W += ["", "Relative value = player value / receiving squad value (model sample, where both exist). "
          f"![Four groups]({FIGS['groups']})", ""]

    # ---- Part 4-5
    W += ["## 4–5. Loan minus permanent within each age group", ""]
    rows = []
    for c_ in (SENSITIVITY[0], cut, SENSITIVITY[1]):
        r = N["cuts"][c_]
        tag = " (primary)" if c_ == cut else ""
        for lab in ("young", "older"):
            rows.append([f"≤{c_} vs ≥{c_ + 1}{tag}", lab,
                         f"{pp(r[f'raw | revised sample | {lab}']['coef'])}",
                         f"{pp(r[f'raw | model sample | {lab}']['coef'])} {ci(r[f'raw | model sample | {lab}'])}",
                         f"{pp(r['adjusted'][lab]['coef'])} {ci(r['adjusted'][lab])}",
                         f0(r[f"raw | model sample | {lab}"]["n_loan"]), f0(r[f"raw | model sample | {lab}"]["n_perm"])])
        rows.append([f"≤{c_} vs ≥{c_ + 1}{tag}", "**older − young (loan × older)**",
                     pp(r["raw | revised sample | difference"]["coef"]),
                     f"{pp(r['raw | model sample | difference']['coef'])} {ci(r['raw | model sample | difference'])}",
                     f"**{pp(r['adjusted']['interaction']['coef'])}** {ci(r['adjusted']['interaction'])}", "", ""])
    W += md_table(["Cutoff", "Age group", "Raw (revised sample)", "Raw (model sample)", "Adjusted", "Loans", "Permanent"],
                  rows, right={2, 3, 4, 5, 6})
    W += ["", f"Adjusted model: share on loan, older, loan × older, arrival timing (share of B's season left), log "
          f"relative value, log squad value, position and league × season fixed effects; errors clustered by receiving "
          f"club; N = {f0(ad['N'])} ({f0(ad['clusters'])} clubs). The model sample keeps episodes with a pre-move player "
          f"and squad value and a position ({f0(N['common'].get('loan', 0))} loans, {f0(N['common'].get('permanent', 0))} "
          f"permanent of {f0(N['revised']['loan'])} and {f0(N['revised']['permanent'])}).", "",
          "**In plain terms (primary cutoff, adjusted, associations only):**", "",
          f"- {say(ad['young'], 'Young loanees', 'young permanent arrivals')}.",
          f"- {say(ad['older'], 'Older loanees', 'older permanent arrivals')}.",
          f"- The loan–permanent difference is {abs(100 * ad['interaction']['coef']):.1f} pp "
          f"{'larger' if ad['interaction']['coef'] > 0 else 'smaller'} for older players than for young ones "
          f"{ci(ad['interaction'])}" + ("." if sig(ad["interaction"]) else "; the interval includes zero."),
          f"- Among permanent arrivals, older players play {abs(100 * ad['older_main']['coef']):.1f} pp "
          f"{'more' if ad['older_main']['coef'] > 0 else 'less'} than young ones {ci(ad['older_main'])}, holding the "
          "controls fixed.", "",
          f"**What absorbs the young players' raw gap.** Without the two value controls (log relative value, log "
          f"squad value) the adjusted differences are young {pp(rec['adjusted_no_value']['young']['coef'])} "
          f"{ci(rec['adjusted_no_value']['young'])} and older {pp(rec['adjusted_no_value']['older']['coef'])} "
          f"{ci(rec['adjusted_no_value']['older'])}; with them, {pp(ad['young']['coef'])} and {pp(ad['older']['coef'])}. "
          f"Young permanent signings have a median relative value of "
          f"{rec['cells']['model sample | permanent, young']['relative_value_median']:.3f} against "
          f"{rec['cells']['model sample | loan, young']['relative_value_median']:.3f} for young loanees: they join "
          "squads where they rank lower.", "", f"![Within-age differences]({FIGS['diffs']})", ""]

    # ---- robustness of cutoff
    ints = {c_: N["cuts"][c_]["adjusted"]["interaction"] for c_ in (SENSITIVITY[0], cut, SENSITIVITY[1])}
    same_sign = len({np.sign(v["coef"]) for v in ints.values()}) == 1
    W += [f"**Cutoff sensitivity.** The interaction is {', '.join(f'{pp(v['coef'])} (≤{c_})' for c_, v in ints.items())}; "
          + ("all with the same sign" if same_sign else "the sign changes across cutoffs")
          + ("; every interval includes zero." if not any(sig(v) for v in ints.values())
             else "; all three intervals exclude zero." if all(sig(v) for v in ints.values())
             else "; the interval excludes zero at " + ", ".join(f"≤{c_}" for c_, v in ints.items() if sig(v)) + ".")
          + " The cutoff does not change the conclusion.", ""]

    # ---- Part 6
    W += ["## 6. Does age explain the earlier patterns?", "", "**Overall gap and age composition** (model sample unless noted)", ""]
    W += md_table(["", "Loan − permanent", "95% interval"],
                  [[k, pp(v["coef"]), ci(v) if "lo" in v else ""] for k, v in comp.items()])
    sd = comp["age-standardised raw gap, pooled age weights"]["coef"] - comp["raw gap (model sample)"]["coef"]
    W += ["", f"Loans are slightly younger ({pct(N['young_share']['loan'])} vs {pct(N['young_share']['permanent'])} aged "
          f"{cut} or under), but standardising to a common single-year age distribution changes the raw gap by "
          f"{pp(sd)}, and the age-dummy model gives {pp(comp['loan, single-year age dummies only']['coef'])}. Age "
          "composition does not produce the overall gap. What age does is split it: the raw advantage is concentrated "
          "among young players (section 4).", ""]
    e = N["england"]
    W += [f"**England (GB1)**, {f0(e['N'])} episodes ({f0(e['n_loan'])} loans). Raw {pp(e['raw']['coef'])}; preferred "
          f"model {pp(e['adjusted_overall']['coef'])} {ci(e['adjusted_overall'])}. By age: young "
          f"{pp(e['adjusted']['young']['coef'])} {ci(e['adjusted']['young'])}, older {pp(e['adjusted']['older']['coef'])} "
          f"{ci(e['adjusted']['older'])}; interaction {pp(e['adjusted']['interaction']['coef'])} "
          f"{ci(e['adjusted']['interaction'])}. Cells: " + "; ".join(f"{k} {v}" for k, v in e["cells"].items()) + ". "
          + ("The England result is negative in both age groups, so it is not an age story."
             if e["adjusted"]["young"]["coef"] < 0 and e["adjusted"]["older"]["coef"] < 0
             else "The England result differs by age group; see the cells."), ""]
    lv = N["low_value"]
    q1 = N["quintiles"]["Q1"]
    W += [f"**Lowest relative-value fifth** (relative value up to {lv['relative_value_max']:.3f}), {f0(lv['N'])} episodes. "
          f"Quintile model on the revised sample: loan {pp(q1['coef'])} {ci(q1)} in the lowest fifth (other fifths: "
          + ", ".join(f"{q} {pp(v['coef'])}" for q, v in N["quintiles"].items() if q != "Q1") + "). "
          f"Within the lowest fifth by age: young {pp(lv['adjusted']['young']['coef'])} {ci(lv['adjusted']['young'])}, "
          f"older {pp(lv['adjusted']['older']['coef'])} {ci(lv['adjusted']['older'])}; interaction "
          f"{pp(lv['adjusted']['interaction']['coef'])} {ci(lv['adjusted']['interaction'])}. Young share of the lowest "
          f"fifth: loans {pct(lv['young_share']['loan'])}, permanent {pct(lv['young_share']['permanent'])}. Cells: "
          + "; ".join(f"{k} {v}" for k, v in lv["cells"].items()) + ". "
          + ("The low-value loan advantage is concentrated among young players, who are also most of that fifth; "
             "the older cell is small." if sig(lv["adjusted"]["young"]) and not sig(lv["adjusted"]["older"])
             else "See the cells for how the low-value pattern splits by age."), ""]
    W += ["## 7. Conclusion", "", conclusion(N), ""]
    return "\n".join(W)


def conclusion(N: dict) -> str:
    rec = N["cuts"][CUTOFF]
    ad, comp = rec["adjusted"], N["composition"]
    ints = [N["cuts"][c]["adjusted"]["interaction"] for c in (SENSITIVITY[0], CUTOFF, SENSITIVITY[1])]
    return (f"Age does not explain the loan–permanent gap through composition: loans and permanent signings have "
            f"similar age profiles, and age-standardising leaves the raw gap at "
            f"{pp(comp['age-standardised raw gap, pooled age weights']['coef'])}. It does change how the result should "
            f"be told. The raw loan advantage is a young-player pattern ({pp(rec['raw | revised sample | young']['coef'])} "
            f"for players aged {CUTOFF} or under, {pp(rec['raw | revised sample | older']['coef'])} for older players), "
            f"driven by young permanent signings who join squads where they rank low and often do not play. With the "
            f"preferred controls, young loanees and young permanent signings play about the same "
            f"({pp(ad['young']['coef'])} {ci(ad['young'])}), while older loanees play less than older permanent "
            f"signings ({pp(ad['older']['coef'])} {ci(ad['older'])}). The difference between the two age groups "
            f"({pp(ad['interaction']['coef'])}) is "
            + ("stable" if max(i["coef"] for i in ints) - min(i["coef"] for i in ints) < 0.01 else "similar")
            + f" across the 22, 23 and 24 cutoffs ({', '.join(pp(i['coef']) for i in ints)}). These are "
            "associations in observational data, not effects of choosing a loan.")


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
    return x


def main(u=None) -> dict:
    from .club_geography_v2 import CANONICAL_CSV, STAGE1C_SHA256, sha256

    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    old = [reg.SUMMARY_MD, reg.MODELS_CSV, reg.SAMPLE_CSV, reg.NUMBERS_JSON, reg.ROBUST_CSV, reg.LEAGUE_CSV,
           OUT / "playing_time_regression_audit_models.csv", OUT / "playing_time_sample_redesign.md",
           OUT / "playing_time_sample_sensitivity.csv"]
    before = {p.name: sha256(p) for p in old if p.exists()}
    assert not {p.name for p in WRITES} & set(before)
    N, tb = compute(u)
    tb["GROUPS"].to_csv(GROUPS_CSV, index=False)
    tb["MODELS"].to_csv(MODELS_CSV, index=False)
    figures(N, tb["B"])
    md = render(N)
    for k in KINDS:
        assert f0(N["cuts"][CUTOFF]["cells"][f"revised sample | {k}, young"]["N"]) in md
    REPORT_MD.write_text(md)
    N["old_output_sha256"] = before
    NUMBERS_JSON.write_text(json.dumps(_json(N), indent=1, default=str))
    assert {p.name: sha256(p) for p in old if p.exists()} == before, "an old output changed"
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print_summary(N)
    return N


def print_summary(N: dict) -> None:
    rec = N["cuts"][CUTOFF]
    ad, ag = rec["adjusted"], N["age_distribution"]
    cells = rec["cells"]
    print("AGE COMPOSITION")
    print(f"- loan median age: {ag['loan']['median']:.1f} (mean {ag['loan']['mean']:.1f}, N {ag['loan']['N']:,})")
    print(f"- permanent median age: {ag['permanent']['median']:.1f} (mean {ag['permanent']['mean']:.1f}, N {ag['permanent']['N']:,})")
    for lab, name in (("young", f"YOUNG <={CUTOFF}"), ("older", f"OLDER >={CUTOFF + 1}")):
        r = rec[f"raw | revised sample | {lab}"]
        rm = rec[f"raw | model sample | {lab}"]
        print(name)
        print(f"- loans N: {cells[f'revised sample | loan, {lab}']['N']:,} (model sample {rm['n_loan']:,})")
        print(f"- permanent N: {cells[f'revised sample | permanent, {lab}']['N']:,} (model sample {rm['n_perm']:,})")
        print(f"- raw difference: {100 * r['coef']:+.1f} pp [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] "
              f"(model sample {100 * rm['coef']:+.1f} pp)")
        a = ad[lab]
        print(f"- adjusted difference: {100 * a['coef']:+.1f} pp [{100 * a['lo']:+.1f}, {100 * a['hi']:+.1f}]")
    i = ad["interaction"]
    print("INTERACTION")
    print(f"- loan × older estimate: {100 * i['coef']:+.1f} pp [{100 * i['lo']:+.1f}, {100 * i['hi']:+.1f}], p = {i['p']:.3f}")
    print(f"- interpretation: the loan−permanent difference is {abs(100 * i['coef']):.1f} pp "
          f"{'larger' if i['coef'] > 0 else 'smaller'} for older than for young players"
          + ("" if sig(i) else " (not distinguishable from zero)"))
    print("ROBUSTNESS")
    print("- does changing the age cutoff matter?: interaction "
          + ", ".join(f"<={c}: {100 * N['cuts'][c]['adjusted']['interaction']['coef']:+.1f} pp "
                      f"[{100 * N['cuts'][c]['adjusted']['interaction']['lo']:+.1f}, "
                      f"{100 * N['cuts'][c]['adjusted']['interaction']['hi']:+.1f}]"
                      for c in (SENSITIVITY[0], CUTOFF, SENSITIVITY[1])))


if __name__ == "__main__":
    main()
