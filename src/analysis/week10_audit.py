"""Week 10 audit (pre-meeting): tier-rule sensitivity, manual audits of matches and exclusions, consistency checks,
presentation figures, before/after comparison.

    python -m src.analysis.week10_audit samples   # draw the stratified manual-audit samples (fixed seed)
    python -m src.analysis.week10_audit           # everything else (reads the reviewed verdict files)

Writes only data/outputs/rebuild/professor_feedback_week10/audit/ and .../presentation/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import week10_followup as wk

AUD = wk.OUT / "audit"
PRES = wk.OUT / "presentation"
SEED = 20261112
SCR_BEFORE = AUD / "before_numbers.json"     # frozen copy of numbers.json before the audit corrections


# --------------------------------------------------------------------------------------------------------------------
# Manual-audit samples
# --------------------------------------------------------------------------------------------------------------------
def match_sample(n_per: int = 4) -> pd.DataFrame:
    """Stratified sample of club-seasons added by Wikipedia: matching method x 14 countries or not x season band x
    reserve/youth side. At most n_per per stratum."""
    from . import league_tiers as lt
    L = pd.read_csv(wk.V2 / "club_season_league_v2.csv")
    N = L[L.source_week.eq("week10_wikipedia")].copy()
    con = lt.connect()
    try:
        tm = lt.tm_universe(con, lt.load_master())
    finally:
        con.close()
    names = tm["names"].groupby("club_id").name.agg(lambda s: " | ".join(sorted(set(s))[:6]))
    N["method"] = N.match_method.str.split(":").str[0]
    N["group14"] = np.where(N.country.isin(wk.FOURTEEN), "14 countries", "other")
    N["band"] = pd.cut(N.season, [2004, 2011, 2018, 2025], labels=["2005-11", "2012-18", "2019-25"]).astype(str)
    N["reserve_or_youth"] = N.club_id.map(lambda i: bool(tm["reserve"].get(int(i), False)))
    rng = np.random.default_rng(SEED)
    out = []
    for _, g in N.groupby(["method", "group14", "band", "reserve_or_youth"]):
        k = min(n_per, len(g))
        out.append(g.sample(k, random_state=int(rng.integers(1 << 31))))
    S = pd.concat(out).reset_index(drop=True)
    S["tm_names"] = S.club_id.map(names)
    S["tm_countries"] = S.club_id.map(lambda i: " | ".join(sorted(tm["countries"].get(int(i), set()))))
    Lk = L.set_index(["club_id", "season"])
    S["tier_prev"] = [Lk.tier.get((c, s - 1)) for c, s in zip(S.club_id, S.season)]
    S["tier_next"] = [Lk.tier.get((c, s + 1)) for c, s in zip(S.club_id, S.season)]
    S.insert(0, "audit_id", [f"M{i + 1:03d}" for i in range(len(S))])
    return S[["audit_id", "method", "group14", "band", "reserve_or_youth", "country", "season", "league_name", "tier",
              "external_team_name", "club_id", "club_name_tm", "tm_names", "tm_countries", "tier_prev", "tier_next",
              "source_file"]]


def exclusion_sample(n: int = 40) -> pd.DataFrame:
    """Main-sample origin sides classified 'not tier 1 or 2' / 'not tier 1' where a list involved is not marked
    complete: random sample with each list's full team list for checking."""
    X = pd.read_csv(wk.OUT / "week10_episode_supplement.csv", low_memory=False)
    M = pd.read_csv(wk.MASTER, usecols=["event_id", "in_cf_main_model_sample", "from_club_id", "from_club_name",
                                        "from_country_v2"], low_memory=False)
    X = X.merge(M, on="event_id")
    C = pd.read_csv(wk.V2 / "league_list_completeness.csv")
    comp = C.groupby(["country", "season", "tier"]).complete.max()
    X = X[X.in_cf_main_model_sample.astype(bool)]
    st = X.from_league_tier_status_v2
    tiers = np.where(st.eq("not_tier_1_or_2_absent_from_lists"), "1,2", np.where(st.eq("not_tier_1_absent_from_list"),
                                                                                 "1", ""))
    X = X.assign(tiers=tiers)
    X = X[X.tiers != ""]
    X["lists_complete"] = [all(comp.get((k, s, int(t)), False) for t in ts.split(","))
                           for k, s, ts in zip(X.from_country_v2, X.tier_reference_season, X.tiers)]
    E = X[~X.lists_complete].drop_duplicates(["from_club_id", "tier_reference_season"])
    S = E.sample(min(n, len(E)), random_state=SEED).copy()
    A = pd.read_csv(wk.V2 / "wiki_name_match_audit.csv").drop_duplicates(["code", "season", "text", "title"])
    F = pd.read_csv(wk.V2 / "wiki_league_seasons.csv")
    lists, urls = [], []
    for k, s, ts in zip(S.from_country_v2, S.tier_reference_season, S.tiers):
        parts, us = [], []
        for t in ts.split(","):
            a = A[(A.country == k) & (A.season == s) & (A.tier == int(t))]
            f = F[F.code.isin(a.code.unique()) & (F.season == s)]
            us += list(f.url.unique())
            parts.append(f"T{t}: " + "; ".join(f"{x}{'' if pd.notna(c) else ' [UNMATCHED]'}"
                                               for x, c in zip(a.text, a.club_id)))
        lists.append(" || ".join(parts))
        urls.append(" ".join(us))
    S["list_teams"] = lists
    S["source_urls"] = urls
    S.insert(0, "audit_id", [f"E{i + 1:03d}" for i in range(len(S))])
    return S[["audit_id", "from_club_id", "from_club_name", "from_country_v2", "tier_reference_season",
              "from_league_tier_status_v2", "from_tier_band_v2", "lists_complete", "list_teams", "source_urls"]]


# --------------------------------------------------------------------------------------------------------------------
# Rule sensitivity and consistency
# --------------------------------------------------------------------------------------------------------------------
def headline(M, X) -> dict:
    C = pd.read_csv(wk.V2 / "league_list_completeness.csv")
    Q2, Q3, Q4 = wk.q2_origin(M, X, C), wk.q3_movement(M, X), wk.q4_flows(M, X, C)
    S = Q3["summary"]
    g = lambda k, c: S[(S.kind == k) & (S.category == c)].iloc[0]
    h = {"origin exact tier (main)": Q2["N"]["exact_tier_v2"],
         "origin tier or band known (main)": Q2["N"]["band_known_v2"],
         "recovered: senior not tier 1 or 2": Q2["N"]["recovered_not_1_2"],
         "recovered: youth/reserve not tier 1 or 2": Q2["N"]["recovered_youth_reserve"],
         "partly recovered: not tier 1": Q2["N"]["partly_not_1"], "still unknown": Q2["N"]["still_unknown"]}
    for k in ("loan", "permanent"):
        for c, lab in (("Higher division -> lower division", "down"), ("Lower division -> higher division", "up"),
                       ("Same level", "same")):
            x = g(k, c)
            h[f"{k}: {lab} %"] = round(float(x.pct), 2)
            h[f"{k}: {lab} n"] = int(x["count"])
        h[f"{k}: decidable n"] = int(g(k, "Higher division -> lower division").denominator)
        h[f"{k}: both tiers known %"] = round(float(g(k, "both tiers known").pct), 2)
        b = Q3["by_border"].loc[(k, "domestic")]
        h[f"{k}: domestic down %"] = round(float(b.down_pct), 2)
        a = Q4["accounting"][k]
        h[f"{k}: heatmap in matrix"] = a["in_matrix"]
        h[f"{k}: heatmap omitted"] = a["omitted"]
        h[f"{k}: heatmap 28x28 core"] = a["core_28x28"]
    return {"h": h, "Q2": Q2, "Q3": Q3, "Q4": Q4}


def consistency(M, X, Q3, Q4) -> pd.DataFrame:
    """Independent recounts from the supplement against the matrices and summary tables."""
    rows = []
    D = X.assign(kind=M.kind.to_numpy())
    s = D.tier_reference_season.astype(float)
    D = D[D.kind.isin(["loan", "permanent"]) & s.between(*wk.PERIOD)]
    for k, lab in (("loan", "V1"), ("permanent", "V2")):
        d = D[D.kind == k]
        mat = pd.read_csv(wk.OUT / "q4_heatmaps" / f"{lab}_{k}_flow_matrix_29x29.csv", index_col=0)
        both = d.from_flow_category.notna() & d.to_flow_category.notna()
        rows += [
            {"check": f"{lab} matrix total = supplement episodes with both categories", "a": int(mat.to_numpy().sum()),
             "b": int(both.sum())},
            {"check": f"{lab} matrix + omitted = episodes in period", "a": int(mat.to_numpy().sum()) +
             int((~both).sum()), "b": len(d)},
            {"check": f"{lab} matrix shape 29 x 29 and category order", "a": int(mat.shape == (29, 29) and
                                                                              list(mat.index) == wk.CATS), "b": 1},
            {"check": f"{lab} England T1 -> England T2 recount", "a": int(mat.loc["England T1", "England T2"]),
             "b": int(((d.from_flow_category == "England T1") & (d.to_flow_category == "England T2")).sum())},
        ]
        S = Q3["summary"]
        dn = S[(S.kind == k) & (S.category == "Higher division -> lower division")].iloc[0]
        dd = d.assign(dir=wk.direction(d.from_tier_lower_bound_v2.to_numpy(float), d.from_tier_exact_v2.to_numpy(bool),
                                       d.to_tier_lower_bound_v2.to_numpy(float), d.to_tier_exact_v2.to_numpy(bool)))
        rows.append({"check": f"{k} 'down' count recomputed from supplement", "a": int(dn["count"]),
                     "b": int((dd.dir == "down").sum())})
    R = pd.DataFrame(rows)
    R["ok"] = R.a == R.b
    return R


# --------------------------------------------------------------------------------------------------------------------
# Presentation figures
# --------------------------------------------------------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#d9d7d2"
BLUE, ORANGE = "#2a78d6", "#e8862a"


def direction_figure(Q3, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = Q3["summary"]
    b = Q3["by_border"]
    cats = [("Higher -> lower division", "Higher division -> lower division", "down_pct"),
            ("Same level", "Same level", "same level_pct"),
            ("Lower -> higher division", "Lower division -> higher division", "up_pct")]
    panels = [("All decidable moves", None), ("Domestic moves", "domestic"), ("Cross-border moves", "cross-border")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (title, border) in zip(axes, panels):
        vals = {}
        for k in ("loan", "permanent"):
            if border is None:
                vals[k] = [float(S[(S.kind == k) & (S.category == c)].pct.iloc[0]) for _, c, _ in cats]
                n = int(S[(S.kind == k) & (S.category == cats[0][1])].denominator.iloc[0])
            else:
                vals[k] = [float(b.loc[(k, border), col]) for _, _, col in cats]
                n = int(b.loc[(k, border), "n"])
            vals[k + "_n"] = n
        x = np.arange(3)
        w = 0.36
        for off, k, col in ((-w / 2 - 0.01, "loan", BLUE), (w / 2 + 0.01, "permanent", ORANGE)):
            bars = ax.bar(x + off, vals[k], width=w, color=col, edgecolor="white", linewidth=1.2,
                          label="Loans" if k == "loan" else "Permanent transfers")
            for r, v in zip(bars, vals[k]):
                ax.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5,
                        color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in cats], fontsize=9, color=INK)
        ax.set_title(f"{title}\nloans n = {vals['loan_n']:,}; permanent n = {vals['permanent_n']:,}", fontsize=10.5,
                     color=INK, loc="left")
        ax.set_ylim(0, 85)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", colors=MUTED, length=0)

    axes[0].set_ylabel("% of moves whose direction can be determined", color=MUTED, fontsize=9)
    fig.suptitle("Direction of moves between league tiers, loans vs permanent transfers", x=0.01, ha="left",
                 fontsize=13, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.99, 0.985))
    fig.text(0.01, 0.01, "Transfer episodes with reference season 2012/13 to 2025/26 (full master), moves whose direction "
             "can be determined. Direction compares division numbers within each\ncountry's pyramid (1 = top). Division "
             "numbers are not comparable measures of competitive strength across countries.", fontsize=8,
             color=MUTED, ha="left", va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig)


def heatmap_28(Q4, path_base: Path) -> None:
    """28 x 28 country-tier-only heatmaps (Other dropped), raw counts, log(1 + count) colour, same scale as 29 x 29."""
    for k, lab, v in (("loan", "V1", "Loans"), ("permanent", "V2", "Permanent transfers")):
        mat = Q4["matrices"][k]
        core = mat.iloc[:28, :28]
        cats_bak = wk.CATS
        try:
            wk.CATS = cats_bak[:28]
            a = Q4["accounting"][k]
            sub = (f"{v}: the 28 x 28 country-tier block of the 29 x 29 matrix ({a['core_28x28']:,} episodes; reference "
                   f"seasons 2012/13 to 2025/26). Other row and column left out.\nColour scale shared with the 29 x 29 "
                   f"figures. Division numbers are not comparable measures of competitive strength across countries.")
            wk.plot_heatmap(core, f"{lab}: {v}, country-tier categories only (28 x 28)", sub, Q4["coverage"],
                            Q4["vmax"], path_base.parent / f"{lab}_{k}_heatmap_28x28")
        finally:
            wk.CATS = cats_bak


# --------------------------------------------------------------------------------------------------------------------
# Verdict summaries
# --------------------------------------------------------------------------------------------------------------------
def verdict_summary(path: Path, by: list[str]) -> tuple[pd.DataFrame, dict]:
    V = pd.read_csv(path)
    assert V.verdict.isin(["accept", "reject", "uncertain"]).all(), V.verdict.unique()
    assert V.reason.notna().all() and (V.reason.str.len() > 5).all(), "every verdict needs a reason"
    T = V.groupby(by).verdict.value_counts().unstack(fill_value=0)
    n = {"audited": len(V), "accept": int((V.verdict == "accept").sum()), "reject": int((V.verdict == "reject").sum()),
         "uncertain": int((V.verdict == "uncertain").sum())}
    n["error_rate_decided"] = n["reject"] / max(1, n["accept"] + n["reject"])
    return T, n


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    AUD.mkdir(parents=True, exist_ok=True)
    PRES.mkdir(parents=True, exist_ok=True)
    if argv[:1] == ["samples"]:
        # the match sample was drawn once (12 Nov 2026) from the lookup before the fallback was disabled; it is not
        # redrawn so the reviewed verdicts stay attached to it. The exclusion sample is drawn from the final outputs.
        if not (AUD / "match_audit_sample.csv").exists():
            match_sample().to_csv(AUD / "match_audit_sample.csv", index=False)
        exclusion_sample().to_csv(AUD / "exclusion_audit_sample.csv", index=False)
        print("samples written")
        return
    M, sides, L1, L2, C, conf = wk.load()
    X = wk.tiers_v2(M, sides, L1, L2, C, conf, rule="per_club")
    Xs = wk.tiers_v2(M, sides, L1, L2, C, conf, rule="complete_lists")
    P, Ss = headline(M, X), headline(M, Xs)
    sens = pd.DataFrame({"primary: per-club rule": P["h"], "strict: complete lists required": Ss["h"]})
    sens["difference"] = sens.iloc[:, 1] - sens.iloc[:, 0]
    sens.to_csv(AUD / "tier_rule_sensitivity.csv")
    chg = []
    for side in ("from", "to"):
        d = X[f"{side}_league_tier_status_v2"] != Xs[f"{side}_league_tier_status_v2"]
        chg.append(pd.DataFrame({"side": side, "primary": X.loc[d, f"{side}_league_tier_status_v2"],
                                 "strict": Xs.loc[d, f"{side}_league_tier_status_v2"]}).value_counts().rename("episodes"))
    pd.concat(chg).to_csv(AUD / "tier_rule_status_changes.csv")
    K = consistency(M, X, P["Q3"], P["Q4"])
    K.to_csv(AUD / "consistency_checks.csv", index=False)
    assert K.ok.all(), K[~K.ok]
    direction_figure(P["Q3"], PRES / "direction_loans_vs_permanent")
    heatmap_28(P["Q4"], wk.OUT / "q4_heatmaps" / "x")
    out = {"sensitivity": sens.reset_index().to_dict("records"), "consistency_all_ok": bool(K.ok.all())}
    for name, f, by in (("match_audit", AUD / "match_audit_verdicts.csv", ["method", "group14"]),
                        ("exclusion_audit", AUD / "exclusion_audit_verdicts.csv", ["from_league_tier_status_v2"])):
        if f.exists():
            T, n = verdict_summary(f, by)
            T.to_csv(AUD / f"{name}_summary.csv")
            out[name] = n
    (AUD / "audit_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
