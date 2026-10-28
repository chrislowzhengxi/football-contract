"""A readable guide to the project's data, for Professor Freund: LaTeX (Overleaf-ready) and Markdown.

    python -m src.analysis.data_guide_professor

Built only from the lineage and provenance outputs (data_lineage_numbers.json, data_source_inventory.csv,
stage1c_column_dictionary.csv, analysis_variable_dictionary.csv, variable_provenance_matrix.csv). Every number is
interpolated; no analysis is rerun and no definition changes. Writes data_guide_professor.tex and .md only.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import data_lineage as dl
from . import variable_provenance as vp
from .column_guide_specs import ORIGINS, S1C_ROLES
from .daniel_scope_final import f0
from .loan_episodes import ROOT

OUT = dl.OUT
ORIGIN_SHORT = {"tm": "Transfermarkt row", "s1c": "Stage 1C column", "multi": "several Stage 1C rows",
                "master": "other master columns", "geo": "club-country join", "cal": "our calendar rule",
                "games": "games + appearances", "players": "players table", "val": "player valuations",
                "kaggle": "Kaggle injuries", "sample": "our sample rule", "org": "our club-organisation map"}
TEX = OUT / "data_guide_professor.tex"
MD = OUT / "data_guide_professor.md"
PDF = OUT / "data_guide_professor.pdf"
WRITES = (TEX, MD, PDF)        # the PDF is compiled from the .tex

TYPE_LABEL = {vp.RAW: "Transfermarkt raw", vp.JOIN: "Transfermarkt joined", vp.EXT: "External raw",
              vp.MECH: "Created by us -- mechanical", vp.CLASS: "Created by us -- rule",
              vp.FLAG: "Created by us -- sample flag", vp.STAT: "Statistical output"}
SHORT_SRC = {"duckdb": None, "raw": "raw history JSON", "stage1c": "Stage 1C", "master": "master",
             "kaggle_inj": "Kaggle injuries", "kaggle_prof": "Kaggle profiles", "geo_v2": "club geography",
             "flags": "club flags", "wikidata": "Wikidata"}
USED = {"yes": "yes", "audit / display only": "audit only", "Stage 2 only": "Stage 2 only", "no": "no",
        "Stage 1C build only": "builds other columns"}
HOW_S1C = {
    "event_id": "hash of player id, date, from- and to-club ids",
    "event_chain_id": "joins the previous row if it leaves the club the player just joined, within 14 days",
    "age_at_transfer": "completed years from date of birth to transfer date",
    "transfer_type_raw": "exact-text code of the fee cell ('-', '?', 'End of loan', 'loan transfer', ...)",
    "transfer_type_normalized": "fee code -> type; a '-' or '?' row inside one club becomes youth_or_internal",
    "transfer_type_source": "whether the type came from the label or the club-name rule",
    "stage1_heuristic_class": "old guess from the upstream fee number and row order",
    "heuristic_agrees_with_label": "old guess compared with the label",
    "fee_numeric_eur": "euro amount parsed from the fee text; 'free transfer' = 0",
    "fee_role": "which kind of money the amount is",
    "fee_disclosure_status": "lookup from the fee code",
    "permanent_transfer_fee_eur": "amount on a paid transfer; 0 on 'free transfer'",
    "loan_fee_eur": "amount on a 'Loan fee' row",
    "fee_on_return_eur": "amount on an 'End of loan' row with a fee",
    "fee_matches_duckdb": "our fee equals the upstream number? (paid / free rows)",
    "is_loan_related": "fee code is a loan or an end of loan",
    "return_leg_has_fee": "return row with a fee above 0",
    "involves_placeholder_club": "a side is Without Club / Retired / Career break / Ban / Unknown",
    "is_youth_or_reserve_side": "club-name pattern (U19, II, B, Youth, ...)",
    "is_internal_move": "both names equal once youth / reserve suffixes are removed",
    "is_senior_move": "none of the three club flags",
    "has_monetary_amount": "any fee field above 0",
    "enrichment_matched": "a raw history row was found for this row",
    "is_research_target": "Stage 2 contract-research eligibility",
    "research_exclusion_reason": "why a row is not a Stage 2 target",
}

# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------

SYM = {"→": r"$\rightarrow$", "←": r"$\leftarrow$", "×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$",
       "÷": r"$\div$", "−": r"$-$", "≠": r"$\neq$", "Σ": r"$\Sigma$", "…": r"\ldots{}", "—": "---", "–": "--",
       "≈": r"$\approx$", ">": r"$>$", "<": r"$<$", "∈": r"$\in$", "§": r"\S{}", "²": r"$^2$"}


def tex(s: str) -> str:
    s = str(s)
    assert "\\" not in s, s
    for a, b in (("{", r"\{"), ("}", r"\}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$"), ("_", r"\_"),
                 ("^", r"\^{}"), ("~", r"\~{}")):
        s = s.replace(a, b)
    s = re.sub(r"`(.+?)`", lambda m: r"\texttt{" + m.group(1).replace("/", r"/\allowbreak{}") + "}", s)
    s = s.replace(r"\_", r"\_\allowbreak{}")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"(?<![A-Za-z])'([^']+?)'", r"`\1'", s)
    s = re.sub(r'"([^"]+?)"', r"``\1''", s)
    s = s.replace("->", r"$\rightarrow$")
    for a, b in SYM.items():
        s = s.replace(a, b)
    return s


def name_tex(c: str) -> str:
    return r"\texttt{" + c.replace("_", r"\_\allowbreak{}") + "}"


class Doc:
    """Blocks rendered to both LaTeX and Markdown from one source."""

    def __init__(self):
        self.blocks = []

    def h1(self, t): self.blocks.append(("h1", t))
    def h2(self, t): self.blocks.append(("h2", t))
    def p(self, t): self.blocks.append(("p", t))
    def bullets(self, items): self.blocks.append(("ul", items))
    def box(self, lines): self.blocks.append(("box", lines))
    def flow(self, lines): self.blocks.append(("flow", lines))
    def formula(self, latex, plain): self.blocks.append(("math", (latex, plain)))
    def diagram(self, tikz, ascii_lines, title): self.blocks.append(("dia", (tikz, ascii_lines, title)))
    def table(self, head, rows, widths, size="footnotesize", raw_cols=()):
        self.blocks.append(("tab", (head, rows, widths, size, set(raw_cols))))
    def raw(self, latex, md=""): self.blocks.append(("raw", (latex, md)))

    # ---- LaTeX
    def tex(self) -> str:
        W = []
        for kind, x in self.blocks:
            if kind == "h1":
                W += [rf"\section{{{tex(x)}}}", ""]
            elif kind == "h2":
                W += [rf"\subsection*{{{tex(x)}}}", ""]
            elif kind == "p":
                W += [tex(x), ""]
            elif kind == "ul":
                W += [r"\begin{itemize}[nosep,leftmargin=*]"] + [r"\item " + tex(i) for i in x] + [r"\end{itemize}", ""]
            elif kind == "box":
                W += [r"\par\noindent\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\centering"]
                W += [tex(line) + r"\\" for line in x[:-1]] + [tex(x[-1]), "}}", r"\par\medskip", ""]
            elif kind == "flow":
                for line in x:
                    assert all(ord(c) < 128 for c in line), line
                W += [r"{\footnotesize", r"\begin{verbatim}"] + x + [r"\end{verbatim}", "}", ""]
            elif kind == "math":
                W += [r"\[", x[0], r"\]", ""]
            elif kind == "dia":
                W += [r"\begin{center}", x[0], r"\end{center}", ""]
            elif kind == "tab":
                head, rows, widths, size, rawc = x
                spec = "".join(rf">{{\raggedright\arraybackslash}}p{{\dimexpr{w}\linewidth-2\tabcolsep\relax}}" for w in widths)
                hrow = " & ".join(r"\textbf{" + tex(h) + "}" for h in head) + r" \\"
                W += [rf"{{\{size}", r"\setlength{\tabcolsep}{3pt}", r"\setlength{\LTpre}{4pt}\setlength{\LTpost}{6pt}",
                      rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}", r"\toprule", hrow, r"\midrule", r"\endfirsthead",
                      r"\toprule", hrow, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
                for r in rows:
                    W += [" & ".join(c if i in rawc else tex(c) for i, c in enumerate(r)) + r" \\[1pt]"]
                W += [r"\end{longtable}", "}", ""]
            elif kind == "raw":
                W += [x[0], ""]
        return "\n".join(W)

    # ---- Markdown
    def md(self) -> str:
        W = []
        cell = lambda c: str(c).replace("|", "\\|").replace("\n", " ")
        for kind, x in self.blocks:
            if kind == "h1":
                W += [f"## {x}", ""]
            elif kind == "h2":
                W += [f"### {x}", ""]
            elif kind == "p":
                W += [x, ""]
            elif kind == "ul":
                W += [f"- {i}" for i in x] + [""]
            elif kind == "box":
                W += ["> " + "  \n> ".join(x), ""]
            elif kind in ("flow",):
                W += ["```"] + x + ["```", ""]
            elif kind == "math":
                W += [f"`{x[1]}`", ""]
            elif kind == "dia":
                W += [f"**{x[2]}**", "", "```"] + x[1] + ["```", ""]
            elif kind == "tab":
                head, rows, *_ = x
                md_row = lambda r: "| " + " | ".join(cell(re.sub(r"\\texttt\{(.+?)\}", r"`\1`", re.sub(
                    r"\\newline\{\\footnotesize\\emph\{(.+?)\}\}", r" (*\1*)", str(c))
                                                                 .replace(r"\_\allowbreak{}", "_").replace(r"\_", "_")))
                                                     for c in r) + " |"
                W += [md_row(head), "| " + " | ".join("---" for _ in head) + " |"] + [md_row(r) for r in rows] + [""]
            elif kind == "raw":
                if x[1]:
                    W += [x[1], ""]
        return "\n".join(W)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load() -> dict:
    L = json.loads(dl.NUMBERS_JSON.read_text())
    INV = pd.read_csv(dl.INVENTORY_CSV).set_index("source")
    PM = pd.read_csv(vp.MATRIX_CSV, keep_default_na=False)
    D1 = pd.read_csv(dl.STAGE1C_DICT_CSV, keep_default_na=False).set_index("column")
    D2 = pd.read_csv(dl.VARDICT_CSV, keep_default_na=False).set_index("variable")
    head = list(pd.read_csv(dl.MASTER_CSV, nrows=0).columns)
    flags = pd.read_csv(dl.MASTER_CSV, usecols=["event_id", "in_revised_sample", "in_injury_usable",
                                                 "in_age_model_sample", "in_injury_model_sample"])
    nrow = len(flags)
    assert not (flags.in_injury_model_sample & ~flags.in_age_model_sample).any()
    assert not (flags.in_injury_usable & ~flags.in_revised_sample).any()
    assert (nrow, len(head)) == (L["master"]["rows"], L["master"]["columns"])
    assert (PM.scope == "stage1c").sum() == len(D1) == 42 and (PM.scope == "master").sum() == len(D2) == len(head)
    EMPN = json.loads((OUT / "empirical" / "empirical_numbers.json").read_text())
    return {"L": L, "INV": INV, "PM": PM, "D1": D1, "D2": D2, "head": head, "EMP": EMPN}


def short_sources(tokens: str) -> str:
    out = []
    for tk in tokens.split():
        src, rest = tk.split(".", 1)
        lab = f"{rest.split('.')[0]}" if src == "duckdb" else SHORT_SRC[src]
        if lab not in out:
            out.append(lab)
    return ", ".join(out)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

MEDICAL_SHORT = {"physical injury": "physical injury", "illness": "illness",
                 "injury, type unspecified (primary only)": "unspecified injury",
                 "minor / unclear muscular complaint (primary only)": "minor muscular complaint",
                 "fitness / rehabilitation (not counted)": "fitness / rehabilitation",
                 "non-medical: rest": "rest", "unclear: quarantine / isolation (not counted)": "quarantine"}
PT_SHORT = {"scheduled_move": "move still scheduled at capture",
            "not_covered": r"club not in one of our \textbf{14 covered leagues that season}",
            "incomplete_club_games": r"incomplete appearance data (a window game has $<$11 receiving-club rows)",
            "no_league_games_in_window": "no league game before the player left"}


def flow_numbers(L: dict) -> dict:
    """Every count the calendar-first flow diagram shows, from data_lineage_numbers.json, checked to add up."""
    import src.analysis.playing_time_join as ptj
    ep, lo, v2, CF = L["episodes"], L["loans"], L["calendar"]["v2"], L["calendar_first"]
    V, P, cur, no = CF["variants"]["classifiable"], CF["permanent_branch"], CF["variants"]["current"], \
        CF["variants"]["normal_only"]
    lb = V["loan_branch"]
    st = {s["role"]: s["removed"] for s in ep["steps"]}
    code = {txt: c for c, txt in ptj.REASONS.items()}
    pts = lambda B: {code[s["reason"]]: s["removed"] for s in B["pt_steps"]}
    seq = lambda steps: {s["reason"]: s["removed"] for s in steps}
    N = {"s1c": ep["rows"], "st": st, "removed_rows": sum(st.values()), "master": L["master"]["rows"],
         "master_cols": L["master"]["columns"], "loans": ep["loans"], "loan_players": ep["loan_players"],
         "perm": ep["permanent_pooled"], "perm_players": ep["permanent_players"], "etype": ep["episode_type"],
         "unknown": ep["episode_type"].get("unknown_type", 0), "open": lo["open"], "scheduled": lo["scheduled_endings"],
         "realised": v2["realised"], "unmapped": v2["unmapped"], "below": v2["below_threshold"],
         "classifiable": v2["classifiable"], "normal": v2["normal"], "off": v2["off"], "by_return": lo["ended_by_return"],
         "et": lo["early_termination"], "gate": V["gate_steps"], "lpt": pts(lb), "ppt": pts(P), "L": lb, "P": P,
         "lrev": seq(lb["revised_steps"]), "prev": seq(P["revised_steps"]), "lcov": seq(lb["control_steps"]),
         "pcov": seq(P["control_steps"]), "linj": seq(lb["injury_steps"]), "pinj": seq(P["injury_steps"]),
         "licov": seq(lb["injury_control_steps"]), "picov": seq(P["injury_control_steps"]), "T": V["totals"],
         "cur": cur["totals"], "no": no["totals"], "est": {k: v["estimates"] for k, v in CF["variants"].items()},
         "revised_by_status": V["revised_loans_by_status"], "cur_rev_by_status": cur["revised_loans_by_status"],
         "categories": L["injury"]["categories"], "kaggle_cols": L["injury"]["raw_columns"],
         "complete_rows": L["playing_time"]["complete_rows"], "n_leagues": len(L["playing_time"]["covered_leagues"]),
         "unclassified_not_covered": CF["unclassified_not_covered"]}
    assert N["s1c"] - N["removed_rows"] == N["master"] == N["loans"] + N["perm"] + N["unknown"]
    assert N["loans"] - N["open"] - N["scheduled"] == N["realised"] == V["gate_steps"][0]["n"]
    assert N["realised"] - N["unmapped"] - N["below"] == N["classifiable"] == N["normal"] + N["off"] == lb["start"]
    assert N["n_leagues"] == 14 and N["complete_rows"] == 11       # both are written into PT_SHORT
    for B in (lb, P):
        assert B["start"] - sum(s["removed"] for s in B["pt_steps"]) == B["reliable"]
        assert B["reliable"] - sum(s["removed"] for s in B["revised_steps"]) == B["revised"]
        assert B["revised"] - sum(s["removed"] for s in B["control_steps"]) == B["main"]
        assert B["revised"] - sum(s["removed"] for s in B["injury_steps"]) == B["injury_usable"]
        assert B["injury_usable"] - sum(s["removed"] for s in B["injury_control_steps"]) == B["injury_model"]
    for s in ("revised", "main", "injury_usable", "injury_model"):
        assert N["T"][s]["total"] == lb[s] + P[s]
    assert set(N["categories"]) == set(MEDICAL_SHORT)
    for c in ("player_id", "from_date", "end_date", "injury_reason"):
        assert c in N["kaggle_cols"]
    return N


def flow_diagram(N: dict) -> tuple[str, list[str]]:
    """One-page TikZ: master → loans → calendar → playing time; permanents straight to playing time; → regression → injury."""
    n = f0
    lpm = lambda a, b: rf"$-${n(a)} / $-${n(b)}"
    st, lb, P, T = N["st"], N["L"], N["P"], N["T"]
    lpt, ppt = N["lpt"], N["ppt"]
    VAL, SQ = "no player valuation strictly before the move", "squad value missing (fewer than 11 valued players)"
    AGE, POS = "age missing (no date of birth)", "position missing"
    WIN, KAG, TRUNC = "window reaches the 2025-09-15 snapshot", "player not in the external datalake", \
        "injury history truncated before the window"
    l_other = sum(v for c, v in lpt.items() if c not in ("not_covered", "incomplete_club_games", "scheduled_move"))
    p_other = sum(v for c, v in ppt.items() if c not in ("not_covered", "incomplete_club_games", "scheduled_move",
                                                         "no_league_games_in_window"))
    i_other = lambda d: sum(v for r, v in d.items() if r not in (WIN, KAG, TRUNC))
    medical = [MEDICAL_SHORT[k] for k, v in N["categories"].items() if v["primary"]]
    big = lambda x: r"{\normalsize\textbf{" + n(x) + "}}"
    lp = lambda d: rf"{n(d['loan'])} loans $\cdot$ {n(d['permanent'])} permanent"
    hd = lambda s: r"\textcolor{black}{\textbf{" + s + r"}}\\ "
    W = "5.1cm"
    tikz = (r"""\begin{tikzpicture}[font=\scriptsize, node distance=1.9mm,
  bx/.style={draw, rounded corners=2pt, align=center, inner sep=2.5pt, text width=#1},
  full/.style={bx=#1, very thick, fill=black!10},
  sub/.style={bx=#1, fill=blue!8},
  fin/.style={bx=#1, double, double distance=1.3pt, thick, draw=green!35!black, fill=green!14},
  ext/.style={bx=#1, fill=orange!18},
  why/.style={align=left, text width=#1, fill=red!6, text=red!45!black, inner sep=2.5pt},
  arr/.style={-{Stealth[length=1.6mm]}, semithick}]
\coordinate (cc) at (-5.45,0); \coordinate (pc) at (0,0); \coordinate (ic) at (5.45,0); \coordinate (sx) at (0.25,0);
% ---- legend
\foreach \st/\x/\lab in {full/-8.0/full dataset, sub/-5.5/analysis subset, fin/-2.6/final regression sample,
  ext/1.3/external data} {
  \node[\st=0.4cm, minimum height=3mm, inner sep=0pt] (lg) at (\x,0) {};
  \node[right, inner sep=2pt] at (lg.east) {\lab};}
\node[why=0.4cm, minimum height=3mm, inner sep=0pt] (lg) at (4.2,0) {};
\node[right, inner sep=2pt, align=left] at (lg.east) {rows removed, and why\\ ($-$loans / $-$permanent)};
% ---- full data
\node[full=7.6cm, anchor=north] (s1) at (-4.1,-0.5) {\textbf{STAGE 1C}\enspace """ + big(N["s1c"]) + r""" recorded player
  movements\\ Transfermarkt transfer-history rows $\cdot$ one row = one movement};
\node[full=7.6cm, below=4.5mm of s1] (m) {\textbf{MASTER DATASET}\enspace \texttt{analysis\_episode\_master.csv}\\ """
    + big(N["master"]) + r""" transfer episodes $\times$ """ + str(N["master_cols"]) + r""" columns\\ one row = one loan or
  one permanent arrival ($+$""" + str(N["unknown"]) + r""" of unknown type, in no analysis)};
\node[why=7.6cm, anchor=north west] (sd) at (s1.north -| sx) {""" + "\\\\ ".join([
        rf"$-${n(st['internal_registration'])}\enspace internal (same club, e.g.\ U19 $\rightarrow$ first team)",
        rf"$-${n(st['placeholder_non_club_movement'])}\enspace to or from Without Club, Retired, Career break",
        rf"$-${n(st['non_transfer_other'])}\enspace drafts",
        rf"$-${n(st['loan_return_collapsed'])}\enspace End-of-loan rows folded into their loan",
        rf"$-${n(st['loan_return_unmatched'])}\enspace End-of-loan rows with no open loan to close"]) + r"""};
\draw[arr] (s1) -- (m) coordinate[pos=0.45] (mid);
\draw[semithick, red!45!black] (mid) -- (mid -| sd.west);
% ---- split
\path (m.south) ++(0,-0.4) coordinate (r1);
\node[sub=""" + W + r""", anchor=north, minimum height=11mm] (lo) at (r1 -| cc) {\textbf{LOANS}\\ """ + big(N["loans"])
    + r""" episodes $\cdot$ """ + n(N["loan_players"]) + r""" players};
\node[sub=""" + W + r""", anchor=north, minimum height=11mm] (pe) at (r1 -| pc) {\textbf{PERMANENT ARRIVALS}\\ """
    + big(N["perm"]) + r""" episodes $\cdot$ """ + n(N["perm_players"]) + r""" players};
\foreach \x in {lo,pe} \draw[arr] (m.south) -- ++(0,-0.2) -| (\x.north);
% ---- loans: calendar classification first
\node[why=""" + W + r""", below=3.5mm of lo] (cd1) {""" + hd("1. LOAN-ENDING CALENDAR") + r"""$-$""" + n(N["open"])
    + r""" still open (no later movement)\\ $-$""" + n(N["scheduled"]) + r""" ending flagged by Transfermarkt as a future
  move};
\node[sub=""" + W + r""", below=of cd1] (c2) {""" + big(N["realised"]) + r""" realised loan endings};
\node[why=""" + W + r""", below=of c2] (cd3) {$-$""" + n(N["unmapped"]) + r""" a club has no country\\ $-$""" + n(N["below"])
    + r""" domestic, country has under 100 past endings};
\node[sub=""" + W + r""", below=of cd3, very thick] (c3) {\textbf{Calendar-classified loans}\enspace """
    + big(N["classifiable"]) + r"""\\ """ + n(N["normal"]) + r""" normal $\cdot$ """ + n(N["off"])
    + r""" off-calendar\\ \textit{both continue}};
\foreach \a/\b in {lo/cd1,cd1/c2,c2/cd3,cd3/c3} \draw[arr] (\a) -- (\b);
% ---- loans: playing time
\node[why=""" + W + r""", below=of c3] (ld1) {""" + hd("2. PLAYING TIME (loans)") + rf"""$-${n(lpt['not_covered'])} """
    + PT_SHORT["not_covered"] + rf"""\\ $-${n(lpt['incomplete_club_games'])} """ + PT_SHORT["incomplete_club_games"]
    + rf"""\\ $-${n(l_other)} no league game in the window or other data gaps}};
\node[sub=""" + W + r""", below=of ld1] (l2) {""" + big(lb["reliable"]) + r""" loans with a reliable outcome};
\node[why=""" + W + rf""", below=of l2] (ld3) {{$-${n(N['lrev']['repeat spell at the same club (first spell kept)'])} repeat
  spell at the same club\\ $-${n(N['lrev']['loan did not end with a return to the lender'])} loan not ended by a return}};
\node[sub=""" + W + r""", below=of ld3] (l3) {""" + big(lb["revised"]) + r""" revised loans\\ """
    + n(N["revised_by_status"].get("normal", 0)) + r""" normal $\cdot$ """ + n(N["revised_by_status"].get("off_calendar", 0))
    + r""" off-calendar};
\foreach \a/\b in {c3/ld1,ld1/l2,l2/ld3,ld3/l3} \draw[arr] (\a) -- (\b);
% ---- permanents: straight to playing time
\node[why=""" + W + r""", anchor=north] (pd1) at (ld1.north -| pc) {""" + hd("2. PLAYING TIME (permanent)")
    + rf"""$-${n(ppt['scheduled_move'])} {PT_SHORT['scheduled_move']}\\ $-${n(ppt['not_covered'])} """
    + PT_SHORT["not_covered"] + rf"""\\ $-${n(ppt['incomplete_club_games'])} incomplete appearance data\\
  $-${n(ppt['no_league_games_in_window'])} no league game before the player left\\ $-${n(p_other)} other data gaps}};
\node[sub=""" + W + r""", below=of pd1] (p2) {""" + big(P["reliable"]) + r""" permanent moves with a reliable outcome};
\node[why=""" + W + rf""", below=of p2] (pd3) {{$-${n(N['prev']['repeat spell at the same club (first spell kept)'])}
  repeat spell at the same club}};
\node[sub=""" + W + r""", below=of pd3] (p3) {""" + big(P["revised"]) + r""" revised permanent moves};
\draw[arr] (pe.south) -- node[fill=white, inner sep=1.5pt, align=center, text width=3.2cm] {\textit{no loan-ending
  calendar step}} (pd1.north);
\foreach \a/\b in {pd1/p2,p2/pd3,pd3/p3} \draw[arr] (\a) -- (\b);
% ---- join: revised sample, controls, main regression
\path (l3.south) ++(0,-0.45) coordinate (j1); \path (p3.south) ++(0,-0.45) coordinate (j2);
\path let \p1 = (j1), \p2 = (j2) in coordinate (j) at (-2.725, {min(\y1, \y2)});
\node[sub=10.55cm, anchor=north, very thick] (rv) at (j) {\textbf{3. REVISED PLAYING-TIME SAMPLE}\\ """
    + big(T["revised"]["total"]) + r""" = """ + lp(T["revised"]) + r"""};
\draw[arr] (l3.south) -- (l3.south |- rv.north); \draw[arr] (p3.south) -- (p3.south |- rv.north);
\node[why=10.55cm, below=of rv] (rd) {controls must be present: """
    + lpm(N["lcov"].get(VAL, 0), N["pcov"].get(VAL, 0)) + r""" no player valuation before the move $\cdot$ """
    + lpm(N["lcov"].get(SQ, 0), N["pcov"].get(SQ, 0)) + r""" squad value missing $\cdot$ """
    + lpm(N["lcov"].get(AGE, 0) + N["lcov"].get(POS, 0), N["pcov"].get(AGE, 0) + N["pcov"].get(POS, 0))
    + r""" age or position missing};
\node[fin=10.55cm, below=of rd] (mn) {\textbf{4. MAIN REGRESSION SAMPLE}\\ """ + big(T["main"]["total"])
    + r""" = """ + lp(T["main"]) + r"""};
\draw[arr] (rv) -- (rd); \draw[arr] (rd) -- (mn);
% ---- right column: legacy check and comparison (detached), then injury
\node[bx=""" + W + r""", dashed, fill=black!4, anchor=north] (et) at (lo.north -| ic) {""" + hd("SEPARATE LEGACY CHECK")
    + r"""not part of the calendar classification or of any sample: from """ + n(N["by_return"]) + r""" loans ended by
  an End-of-loan row $\rightarrow$ """ + big(N["et"]) + r""" legacy early-termination flags};
\node[bx=""" + W + r""", fill=white, below=3mm of et] (cmp) {""" + hd("FOR COMPARISON") + r"""Week 5/6 sequence (no loan
  calendar step): revised """ + n(N["cur"]["revised"]["total"]) + r""", main """ + n(N["cur"]["main"]["total"])
    + r""", injury """ + n(N["cur"]["injury_model"]["total"]) + r"""\\ calendar-normal loans only: revised """
    + n(N["no"]["revised"]["total"]) + r""", main """ + n(N["no"]["main"]["total"]) + r""", injury """
    + n(N["no"]["injury_model"]["total"]) + r"""};
\node[why=""" + W + r""", anchor=north] (id1) at ([yshift=14mm]rv.north -| ic) {""" + lpm(N["linj"].get(WIN, 0),
                                                                                           N["pinj"].get(WIN, 0))
    + r""" window reaches the injury snapshot (2025-09-15)\\ """ + lpm(N["linj"].get(KAG, 0), N["pinj"].get(KAG, 0))
    + r""" player not in Kaggle\\ """ + lpm(N["linj"].get(TRUNC, 0), N["pinj"].get(TRUNC, 0)) + r""" injury history appears
  truncated before the window\\ """ + lpm(i_other(N["linj"]), i_other(N["pinj"])) + r""" open or undated spell; no fit
  game};
\node[ext=""" + W + r""", above=3mm of id1] (kg) {""" + hd("5. INJURY ROBUSTNESS")
    + r"""\textbf{Kaggle injuries} (external): \texttt{player\_id}, \texttt{from\_date}, \texttt{end\_date},
  \texttt{injury\_reason}\\[1pt] spell = medical row, from--end inclusive; injury-unavailable game = window league
  game inside a spell with no appearance; denominator 90 $\times$ (available $-$ injury-unavailable games)};
\node[sub=""" + W + r""", below=of id1] (i1) {\textbf{Injury-usable}\enspace """ + big(T["injury_usable"]["total"])
    + r"""\\ """ + lp(T["injury_usable"]) + r"""};
\node[why=""" + W + r""", below=of i1] (id2) {controls: """ + lpm(N["licov"].get(VAL, 0), N["picov"].get(VAL, 0))
    + r""" valuation, """ + lpm(N["licov"].get(SQ, 0), N["picov"].get(SQ, 0)) + r""" squad value, """
    + lpm(N["licov"].get(AGE, 0) + N["licov"].get(POS, 0), N["picov"].get(AGE, 0) + N["picov"].get(POS, 0))
    + r""" age/position};
\node[fin=""" + W + r""", below=of id2] (i2) {\textbf{6. INJURY REGRESSION}\\ """ + big(T["injury_model"]["total"])
    + r"""\\ """ + lp(T["injury_model"]) + r"""};
\draw[arr] (rv.east) -- (rv.east -| id1.west); \draw[arr] (kg) -- (id1);
\foreach \a/\b in {id1/i1,i1/id2,id2/i2} \draw[arr] (\a) -- (\b);
\end{tikzpicture}""")
    A = lambda a, b: f"-{n(a)} / -{n(b)}"
    row = lambda why, to: f"  {why:<86} -> {to}" if to else f"  {why}"
    ascii_lines = [
        f"STAGE 1C  {n(N['s1c'])} recorded player movements (Transfermarkt transfer-history rows)",
        row(f"-{n(N['removed_rows'])} internal, placeholder, draft rows; End-of-loan rows folded into their loan",
            f"MASTER {n(N['master'])} episodes x {N['master_cols']} columns"),
        f"  MASTER splits into LOANS {n(N['loans'])} | PERMANENT {n(N['perm'])} | {N['unknown']} unknown type (no analysis)",
        "",
        "1. LOAN-ENDING CALENDAR (loans only)",
        row(f"-{n(N['open'])} still open, -{n(N['scheduled'])} ending flagged as a future move",
            f"{n(N['realised'])} realised endings"),
        row(f"-{N['unmapped']} club without country, -{n(N['below'])} country with < 100 past endings",
            f"{n(N['classifiable'])} classified: {n(N['normal'])} normal + {n(N['off'])} off-calendar (both continue)"),
        "2. PLAYING TIME (loans continue from the calendar step; permanents enter directly)",
        row(f"loans: -{n(lpt['not_covered'])} club not in a covered league, -{n(lpt['incomplete_club_games'])} incomplete "
            f"appearance data, -{n(l_other)} other", f"{n(lb['reliable'])} reliable"),
        row(f"loans: -{n(N['lrev']['repeat spell at the same club (first spell kept)'])} repeat spell, "
            f"-{n(N['lrev']['loan did not end with a return to the lender'])} not ended by a return",
            f"{n(lb['revised'])} revised loans"),
        row(f"permanent: -{n(ppt['scheduled_move'])} scheduled, -{n(ppt['not_covered'])} not covered, "
            f"-{n(ppt['incomplete_club_games'] + ppt['no_league_games_in_window'] + p_other)} other gaps",
            f"{n(P['reliable'])} reliable"),
        row(f"permanent: -{n(N['prev']['repeat spell at the same club (first spell kept)'])} repeat spell",
            f"{n(P['revised'])} revised permanent"),
        row("3. REVISED SAMPLE", f"{n(T['revised']['total'])} ({n(T['revised']['loan'])} loans / "
                                  f"{n(T['revised']['permanent'])} permanent)"),
        row(f"controls: {A(N['lcov'].get(VAL, 0), N['pcov'].get(VAL, 0))} no valuation, "
            f"{A(N['lcov'].get(SQ, 0), N['pcov'].get(SQ, 0))} squad value, age/position",
            f"4. MAIN REGRESSION {n(T['main']['total'])} ({n(T['main']['loan'])} / {n(T['main']['permanent'])})"),
        row("injury: Kaggle player_id, from_date, end_date, injury_reason; usability checks",
            f"{n(T['injury_usable']['total'])} injury-usable"),
        row("same control rule", f"6. INJURY REGRESSION {n(T['injury_model']['total'])} "
                                 f"({n(T['injury_model']['loan'])} / {n(T['injury_model']['permanent'])})"),
        "",
        f"SEPARATE LEGACY CHECK (not part of the calendar or any sample): {n(N['by_return'])} loans -> {n(N['et'])} flags",
        f"FOR COMPARISON: Week 5/6 sequence {n(N['cur']['revised']['total'])} / {n(N['cur']['main']['total'])} / "
        f"{n(N['cur']['injury_model']['total'])}; calendar-normal only {n(N['no']['revised']['total'])} / "
        f"{n(N['no']['main']['total'])} / {n(N['no']['injury_model']['total'])} (revised / main / injury regression)"]
    return tikz, ascii_lines


def build(d: dict) -> Doc:
    L, INV, PM, D1, D2 = d["L"], d["INV"], d["PM"], d["D1"], d["D2"]
    s1, ep, lo, ca, pt, inj = L["stage1c"], L["episodes"], L["loans"], L["calendar"], L["playing_time"], L["injury"]
    v2, prm, geo, mv, mm = ca["v2"], ca["params"], L["geography"], L["market_value"]["all"], L["master"]
    st = {s["role"]: s["removed"] for s in ep["steps"]}
    wf = pt["waterfall"]
    ex, tz, az, an = L["loan_example"], L["pt_examples"]["true_zero"], L["pt_examples"]["absence_not_zero"], L["anton"]
    rows_inv = lambda k: f0(INV.loc[k, "rows"])
    cnt = lambda scope: PM[PM.scope == scope].provenance_category.value_counts()
    c1, cm, cr = cnt("stage1c"), cnt("master"), cnt("report")
    n_unknown = int(PM.llm_inference_used_for_value.eq("UNKNOWN").sum())
    n_yes = int(PM.llm_inference_used_for_value.eq("YES").sum())
    revised = wf["loan"]["revised"] + wf["permanent"]["revised"]
    usable = sum(inj["usable"].values())
    model = sum(inj["model_sample"].values())
    seasons = f"{pt['covered_seasons'][0]}/{str(pt['covered_seasons'][0] + 1)[2:]}–{pt['covered_seasons'][1]}/" \
              f"{str(pt['covered_seasons'][1] + 1)[2:]}"
    FN = flow_numbers(L)
    D = Doc()

    # ================================================================ 1
    D.h1("Big picture: what data do we actually have?")
    D.p("Everything starts from Transfermarkt. We take one public snapshot of Transfermarkt data, clean the transfer "
        "rows into Stage 1C, turn those rows into transfer episodes, and then attach matches, market values and injuries "
        "to each episode.")
    D.diagram(r"""\begin{tikzpicture}[
  box/.style={draw, rounded corners=2pt, align=center, font=\footnotesize, inner sep=4pt},
  src/.style={box, fill=black!6}, ours/.style={box, fill=blue!7}, ext/.style={box, fill=orange!15},
  arr/.style={-{Stealth[length=2mm]}, thick}]
\node[src, text width=12.6cm] (raw) at (0,0) {\textbf{TRANSFERMARKT SOURCE DATA}\\
  DuckDB snapshot: transfers, players, clubs, competitions, games, appearances, player valuations\\
  raw transfer histories: Transfermarkt's original fee text and club country flags};
\node[ours, text width=8.2cm] (s1c) at (0,-2.1) {\textbf{STAGE 1C}\quad \texttt{stage1c\_canonical\_transfers.csv}\\
  """ + f0(s1["rows"]) + r""" recorded player movements $\times$ """ + str(s1["columns"]) + r""" columns};
\node[ours, text width=8.2cm] (m) at (0,-4.25) {\textbf{MASTER}\quad \texttt{analysis\_episode\_master.csv}\\
  """ + f0(mm["rows"]) + r""" transfer episodes $\times$ """ + str(mm["columns"]) + r""" columns\\ one row = one transfer episode};
\draw[arr] (raw) -- node[right, font=\scriptsize, xshift=2pt]{restore fee labels; flag club types} (s1c);
\draw[arr] (s1c) -- node[right, font=\scriptsize, xshift=2pt]{pair loans with their returns; set aside non-transfer rows} (m);
\node[ours, text width=3.3cm] (b1) at (-5.85,-6.9) {\textbf{Loan endings}\\ + club countries\\ $\rightarrow$ calendar-normal /
  off-calendar\\ $\rightarrow$ legacy early-termination heuristic};
\node[ours, text width=3.3cm] (b2) at (-1.95,-6.9) {\textbf{Playing time}\\ + games\\ + appearances\\ $\rightarrow$ minutes share,
  true zeros};
\node[ours, text width=3.3cm] (b3) at (1.95,-6.9) {\textbf{Market value}\\ + player valuations\\ $\rightarrow$ player value,
  squad value, relative value};
\node[ext, text width=3.3cm] (b4) at (5.85,-6.9) {\textbf{Injuries}\\ + Kaggle injury data\\ + games, appearances\\
  $\rightarrow$ injury-adjusted playing time};
\foreach \b in {b1,b2,b3,b4} \draw[arr] (m.south) -- (\b.north);
\end{tikzpicture}""",
              [f"TRANSFERMARKT SOURCE DATA (DuckDB snapshot + raw transfer histories)",
               "        |  restore fee labels; flag club types",
               "        v",
               f"STAGE 1C  stage1c_canonical_transfers.csv  {f0(s1['rows'])} recorded player movements x {s1['columns']} columns",
               "        |  pair loans with their returns; set aside non-transfer rows",
               "        v",
               f"MASTER  analysis_episode_master.csv  {f0(mm['rows'])} episodes x {mm['columns']} columns (one row = one episode)",
               "        +--> + club countries          -> calendar-normal / off-calendar; legacy early-termination heuristic",
               "        +--> + games + appearances     -> playing-time share, true zeros",
               "        +--> + player valuations       -> player value, squad value, relative value",
               "        +--> + Kaggle injury data      -> injury-adjusted playing time"],
              "Diagram 1. From source data to the analysis branches")
    D.p("Grey = source data; blue = created by us; orange = external (non-Transfermarkt) data.")
    D.box(["**MAIN MERGED ANALYSIS FILE**", "`data/outputs/rebuild/analysis_episode_master.csv`",
           f"{f0(mm['rows'])} rows × {mm['columns']} columns ({f0(mm['players'])} players)",
           "**ONE ROW = ONE TRANSFER EPISODE** (one loan, or one permanent arrival at a club)"])
    D.p("Some data have many rows per episode: one row per match in the player's window, one row per injury spell, and "
        "the End-of-loan rows themselves. They stay in their own tables. Merging them into the master would repeat each "
        "episode many times, so the master holds only their per-episode results (minutes, games removed for injury, the "
        "loan's ending date).")

    # ================================================================ 2
    D.h1("Source datasets")
    D.p("Every row below is original data: the first ten are Transfermarkt (via the upstream snapshot), the last three "
        "are external. The last two columns separate what the source gives us from what we build out of it.")
    src_rows = [
        ["**transfers** (DuckDB)", "dcaribou transfermarkt-datasets snapshot", "one registration movement of one player",
         rows_inv("transfers"), "player_id, transfer_date, transfer_season, from/to club id and name, market_value_in_eur",
         "Stage 1C rows; event_id; age; loan episodes"],
        ["**raw transfer histories** (JSON)", "same project's raw files (2023–2025)", "one player's full transfer history",
         f"{rows_inv('raw transfer-history season files')} lines", "fee (text), url, futureTransfer, marketValue, club "
         "countryFlag", "fee label and amounts; future flag; club country (flag)"],
        ["**players**", "DuckDB snapshot", "one player", rows_inv("players"),
         "date_of_birth, position, country_of_citizenship", "age, young / older, position controls"],
        ["**clubs**", "DuckDB snapshot", "one club in a covered league", rows_inv("clubs"),
         "club_id, domestic_competition_id", "league country of a club"],
        ["**countries**", "DuckDB snapshot", "one country", rows_inv("countries"), "country_id, country_name",
         "names the club flag ids"],
        ["**competitions**", "DuckDB snapshot", "one competition", rows_inv("competitions"),
         "competition_id, country_name, confederation", "league country; which leagues count"],
        ["**games**", "DuckDB snapshot", "one match", rows_inv("games"),
         "game_id, competition_type, season, date, home_club_id, away_club_id",
         "arrival season; observation window; available games"],
        ["**appearances**", "DuckDB snapshot", "one player in one match for one club (no row if he did not play)",
         rows_inv("appearances"), "game_id, player_id, player_club_id, **minutes_played**",
         "minutes; appearances; completeness check; true zero; injury appearance override"],
        ["**game_events**", "DuckDB snapshot", "one in-match event", rows_inv("game_events"),
         "type, description, player_id, game_id", "injury-substitution validation only"],
        ["**player_valuations**", "DuckDB snapshot", "one market valuation of one player on one date",
         rows_inv("player_valuations"), "player_id, date, market_value_in_eur, current_club_id",
         "player value; squad value; relative value"],
        ["**Kaggle player_injuries**", "Kaggle salimt v2 (CC0), downloaded by hand", "one listed injury spell",
         f"{rows_inv('player_injuries.csv (Kaggle salimt v2)')} ({f0(inj['raw_players'])} players)",
         "player_id, from_date, end_date, injury_reason (days_missed, games_missed not used for the denominator)",
         "medical spells; injury-unavailable games"],
        ["**Kaggle player_profiles**", "same Kaggle dataset", "one player in that dataset",
         rows_inv("player_profiles.csv (Kaggle salimt v2)"), "player_id, date_of_birth",
         "'no injury listed' vs 'not in the data'; id check"],
        ["**Wikidata**", "Wikidata API, queried once", "one queried club", rows_inv("Wikidata cache"), "country (P17)",
         "country of 2 clubs nothing else could place"],
    ]
    D.table(["Source", "Where it came from", "One row means", "Rows", "Original columns we use", "What we build from it"],
            src_rows, [0.14, 0.14, 0.17, 0.105, 0.235, 0.21])
    D.p("`game_lineups` is used only in coverage checks; `club_games` and `national_teams` are not used.")

    # ================================================================ 3
    D.h1("Stage 1C: what it is")
    D.box(["`data/outputs/rebuild/stage1c_canonical_transfers.csv`",
           f"{f0(s1['rows'])} rows × {s1['columns']} columns, {f0(s1['players'])} players, dated {s1['date_min']} to "
           f"{s1['date_max']}",
           "**ONE ROW = ONE TRANSFERMARKT REGISTRATION MOVEMENT** -- not one contract, not one economic deal"])
    D.p("Each row is one line of a player's Transfermarkt transfer history: the player moves from one entry to another "
        "on a date. An entry can be a club, a club's youth or reserve side, or a placeholder such as 'Without Club'. "
        "Stage 1C keeps every upstream row and adds back what the upstream table had lost: Transfermarkt's own fee label "
        "('End of loan', 'loan transfer', '?', '-') and the fee amount, split into permanent fee, loan fee and fee on "
        "return.")
    D.p("It does not pair loans with returns, decide how a loan ended, infer contract terms, impute missing fees or "
        "assign countries. Those are later steps.")
    D.h2("Why a loan is two rows")
    D.p("Transfermarkt records the loan going out and the loan coming back as separate movements:")
    r0, r1 = ex["rows"]
    D.table(["Row", "Date", "From → to", "Fee cell", "What our code does"],
            [["1", r0["transfer_date"], f"{r0['from_club_name']} → {r0['to_club_name']}", r0["fee_display_raw"],
              "becomes the loan episode (keeps this row's event_id)"],
             ["2", r1["transfer_date"], f"{r1['from_club_name']} → {r1['to_club_name']}", r1["fee_display_raw"],
              "folded into row 1 as its ending (loan ending date, ending label)"]],
            [0.06, 0.13, 0.3, 0.14, 0.37])
    D.p(f"Two Stage 1C rows, one loan episode for {ex['player']}: A → B on loan, B → A at the end of the loan. The "
        f"episode lasts {ex['episode']['duration_days']} days. The same logic handles sub-loans, renewals and "
        "conversions to permanent moves.")
    D.h2("All 42 Stage 1C columns")
    D.p("Appendix A explains every column: what it means, where it comes from, how we built it if it is ours, and every "
        "category it can take. By role:")
    role_rows = []
    for role, label in S1C_ROLES.items():
        cols = [c for c in D1.index if D1.loc[c, "role"] == label]
        if cols:
            role_rows.append([label, str(len(cols)), ", ".join(name_tex(c) for c in cols)])
    assert sum(int(r[1]) for r in role_rows) == 42
    D.table(["Role", "Count", "Names"], role_rows, [0.3, 0.08, 0.62], raw_cols=(2,))

    # ================================================================ 4
    D.raw(r"\clearpage", "")
    D.h1("The full dataset and every analysis sample")
    tikz, ascii_lines = flow_diagram(FN)
    D.diagram(tikz, ascii_lines, "Diagram 2. From the full transfer dataset to every analysis sample")
    D.h2(f"A. Stage 1C → master: why {f0(FN['removed_rows'])} rows are not episodes")
    D.p("Every Stage 1C row gets exactly one role. Nothing is deleted from Stage 1C; rows that are not transfers are "
        "simply not episodes.")
    D.table(["Step", "Rows", "Why"], [
        ["Stage 1C recorded player movements", f0(ep["rows"]), ""],
        ["− internal registrations", f0(st["internal_registration"]), "both sides are one club (e.g. U19 → first team)"],
        ["− placeholder movements", f0(st["placeholder_non_club_movement"]), "Without Club, Retired, Career break, Ban, Unknown"],
        ["− drafts", f0(st["non_transfer_other"]), "Transfermarkt 'draft' rows"],
        ["− return rows folded into their loans", f0(st["loan_return_collapsed"]), "an End-of-loan row becomes the ending of its loan"],
        ["− unmatched returns set aside", f0(st["loan_return_unmatched"]), "no open loan to close (history starts mid-loan)"],
        ["**= transfer episodes**", f"**{f0(ep['episodes'])}**", f"{f0(ep['loans'])} loans + {f0(ep['permanent_pooled'])} "
         f"permanent arrivals + {ep['episode_type'].get('unknown_type', 0)} of unknown type"]],
        [0.38, 0.12, 0.5])
    D.p("The internal rule compares the two club names after removing a youth / reserve suffix and needs an exact "
        "match. A promotion whose first team is written differently ('B. Dortmund II' → 'Dortmund', 'RM Castilla' → "
        "'Real Madrid') is not caught and stays a permanent arrival; `playing_time_qa.md` measures how many reach the "
        "samples.")
    D.p("**One master row = one transfer episode**: a loan (its outbound row, with the return attached) or a permanent "
        "arrival (paid, free, undisclosed fee or no fee shown). Columns that only one analysis needs are filled only for "
        f"that analysis's sample: market values and injury columns exist for the {f0(revised)} episodes of the revised "
        "playing-time sample, so a blank elsewhere means 'not computed'.")
    D.h2(f"B. What the {mm['columns']} master columns hold")
    groups = {
        "A identity": "ids of the episode, the Transfermarkt transfer and the player",
        "B transfer": "date, season, clubs, fee label and type, fees, club flags, same-organisation (academy) flags",
        "C loan ending": "the row that ended the loan, its date and label, duration, the player's next move, ending class",
        "D geography/calendar": "club countries (v1, v2), domestic / cross-border, calendar class, loan calendar status",
        "E playing time": "league, window, available games, minutes, minutes share, reliability",
        "F player": "date of birth, ages, young / older, position",
        "G market value": "player value, squad value, relative value, logs, league × season cell",
        "H injury": "join status, injury games removed, fit-adjusted shares, usability",
        "I sample membership": "which sample each episode belongs to (Week 5/6 and calendar-first)",
        "J regression-complete": "which episodes enter each regression",
    }
    secs = D2.section.value_counts()
    D.table(["Group", "Columns", "What is in it"],
            [[g, str(int(secs.get(g, 0))), v] for g, v in groups.items()] + [["**Total**", f"**{mm['columns']}**", ""]],
            [0.24, 0.1, 0.66])
    D.p("Regression inputs are spread over groups E–G: arrival timing (`remain`), age, position, log relative value, log "
        "squad value and the league × season cell. Appendix B lists every column; Appendix C says on which rows each "
        "column can be interpreted (for example, `minutes_share` only where `pt_reliable` is True, and `true_zero` is "
        "False, not blank, everywhere else).")

    # ================================================================ 5
    import src.analysis.playing_time_join as ptj
    D.h1("Analysis subsets: why each sample shrinks")
    D.p("The samples are built in this order: Stage 1C → master → **loans: loan-ending calendar** → **playing-time "
        "eligibility** → controls → **main regression** → **injury robustness**. Permanent arrivals have no loan ending, "
        "so they skip the calendar step and enter at playing time. Each drop is labelled **DATA AVAILABILITY** (the data "
        "cannot support the measure) or **OUR RESEARCH CHOICE** (a definition we picked and could change).")
    D.h2("A. Loan-ending calendar (loans only)")
    D.table(["Step", "Left", "Rule", "Type"], [
        ["Loan episodes", f0(lo["loans"]), "", ""],
        ["Recorded ending", f0(lo["ending_observed"]), f"{lo['open']} loans have no later movement at all", "DATA AVAILABILITY"],
        ["Realised ending", f0(v2["realised"]), f"{f0(lo['scheduled_endings'])} endings are flagged by Transfermarkt as "
                                                 "future moves", "DATA AVAILABILITY"],
        ["Both club countries known", f0(v2["mapped"]), f"{v2['unmapped']} endings involve a club with no country",
         "DATA AVAILABILITY"],
        ["Classified", f0(v2["classifiable"]), f"{f0(v2['below_threshold'])} domestic endings in countries with fewer "
                                               f"than {prm['N_MIN']} training endings", "OUR RESEARCH CHOICE"],
        ["of which calendar-normal / off-calendar", f"{f0(v2['normal'])} / {f0(v2['off'])}",
         f"±{prm['TOL']} days of a learned date; **both continue**", "OUR RESEARCH CHOICE"]], [0.27, 0.15, 0.38, 0.2])
    D.p("Master columns: `loan_calendar_status` (normal / off_calendar / unclassified / ending_not_realised; "
        "not_applicable for permanent moves), `off_calendar_flag` (1 / 0, blank where no classification exists) and "
        "`calendar_first_eligible` (permanent moves, and loans with a classified ending).")
    D.h2("B. Playing time (classified loans and all permanent arrivals)")
    lab = {"scheduled_move": ("move still scheduled at capture", "DATA AVAILABILITY"),
           "not_covered": ("receiving club not in one of our 14 covered leagues that season", "DATA AVAILABILITY"),
           "player_not_in_universe": ("player missing from the players table", "DATA AVAILABILITY"),
           "club_season_without_rows": ("club has no appearance rows that season", "DATA AVAILABILITY"),
           "no_league_games_in_window": ("no league game before the player left", "DATA AVAILABILITY"),
           "window_after_capture": ("window ends after his history was captured", "DATA AVAILABILITY"),
           "incomplete_club_games": (f"incomplete appearance data: a window game has fewer than {pt['complete_rows']} "
                                     "rows for the receiving club", "DATA AVAILABILITY"),
           "appearances_for_other_club": ("he appears for another club in the window", "DATA AVAILABILITY")}
    LB, PB = FN["L"], FN["P"]
    rows = [["Start", "", f0(LB["start"]), "", f0(PB["start"]), ""]]
    l_left, p_left = LB["start"], PB["start"]
    for code in ptj.REASONS:
        a, b = FN["lpt"].get(code, 0), FN["ppt"].get(code, 0)
        if a or b:
            l_left, p_left = l_left - a, p_left - b
            rows.append([lab[code][0], f0(a), f0(l_left), f0(b), f0(p_left), lab[code][1]])
    assert (l_left, p_left) == (LB["reliable"], PB["reliable"])
    rows.append(["**Reliable outcome**", "", f"**{f0(LB['reliable'])}**", "", f"**{f0(PB['reliable'])}**", ""])
    for r in ("loan did not end with a return to the lender", "repeat spell at the same club (first spell kept)"):
        a, b = FN["lrev"].get(r, 0), FN["prev"].get(r, 0)
        l_left, p_left = l_left - a, p_left - b
        rows.append([r, f0(a), f0(l_left), f0(b), f0(p_left), "OUR RESEARCH CHOICE"])
    rows.append(["**Revised playing-time sample**", "", f"**{f0(LB['revised'])}**", "", f"**{f0(PB['revised'])}**", ""])
    D.table(["Step", "Loans −", "Loans left", "Perm. −", "Perm. left", "Type"], rows, [0.34, 0.09, 0.1, 0.09, 0.1, 0.28])
    rbs = FN["revised_by_status"]
    D.p(f"The revised sample has {f0(FN['T']['revised']['total'])} episodes. Its {f0(LB['revised'])} loans are "
        f"{f0(rbs.get('normal', 0))} calendar-normal and {f0(rbs.get('off_calendar', 0))} off-calendar: off-calendar "
        "loans are kept. The window and minutes are measured the same way for loans and permanent moves; "
        "`playing_time_qa.md` recomputes all of them independently and finds no disagreement.")
    D.h2("C. Main regression sample")
    D.p("The regression needs every control variable. These are DATA AVAILABILITY drops, except that the 365-day / "
        "11-player squad-value rule is our choice.")
    rows = [["Revised playing-time sample", f0(LB["revised"]), f0(PB["revised"]), f0(FN["T"]["revised"]["total"])]]
    for s in L["age_model_missing"]["steps"]:
        a, b = FN["lcov"].get(s["reason"], 0), FN["pcov"].get(s["reason"], 0)
        if a or b:
            rows.append([f"− {s['reason']}", f0(a), f0(b), f0(a + b)])
    rows.append(["**= main regression sample**", f"**{f0(LB['main'])}**", f"**{f0(PB['main'])}**",
                 f"**{f0(FN['T']['main']['total'])}**"])
    D.table(["Step", "Loans", "Permanent", "Total"], rows, [0.52, 0.16, 0.16, 0.16])
    D.h2("D. Injury robustness sample")
    D.p("The injury branch starts again from the revised playing-time sample, not from the main regression sample.")
    D.bullets(["**Kaggle columns used:** `player_id`, `from_date`, `end_date`, `injury_reason`. Kaggle's `days_missed` "
               "and `games_missed` are not used.",
               "**Injury spell:** one Kaggle row with a medical label, from `from_date` to `end_date`, both days included. "
               "Rest, fitness / rehabilitation and quarantine labels never count.",
               "**Injury-unavailable game:** a league game of the new club in the playing-time window that falls inside a "
               "spell and in which the player has no appearance. A game inside two spells counts once.",
               "**Denominator:** 90 × available games becomes 90 × (available games − injury-unavailable games). Minutes "
               "played do not change."])
    itype = {"player not in the external datalake": "DATA AVAILABILITY", "ambiguous player id": "DATA AVAILABILITY",
             "window reaches the 2025-09-15 snapshot": "DATA AVAILABILITY",
             "injury history truncated before the window": "DATA AVAILABILITY",
             "open injury (no end date) could cover the window": "OUR RESEARCH CHOICE (no imputed end dates)",
             "player has a medical row without a start date": "OUR RESEARCH CHOICE (no imputed dates)",
             "no fit games left": "DATA AVAILABILITY"}
    ilab = {"injury history truncated before the window": "injury history appears truncated before the observation window"}
    rows = [["Revised playing-time sample", f0(LB["revised"]), f0(PB["revised"]), ""]]
    for r in itype:
        a, b = FN["linj"].get(r, 0), FN["pinj"].get(r, 0)
        if a or b:
            rows.append([f"− {ilab.get(r, r)}", f0(a), f0(b), itype[r]])
    rows.append(["**= injury-usable**", f"**{f0(LB['injury_usable'])}**", f"**{f0(PB['injury_usable'])}**", ""])
    for s in L["injury_model_missing"]["steps"]:
        a, b = FN["licov"].get(s["reason"], 0), FN["picov"].get(s["reason"], 0)
        if a or b:
            rows.append([f"− {s['reason']}", f0(a), f0(b), "DATA AVAILABILITY"])
    rows.append(["**= injury regression sample**", f"**{f0(LB['injury_model'])}**", f"**{f0(PB['injury_model'])}**", ""])
    D.table(["Step", "Loans", "Perm.", "Type"], rows, [0.5, 0.12, 0.12, 0.26])
    D.p(f"Both injury regressions (current share and injury-adjusted share) use the same "
        f"{f0(FN['T']['injury_model']['total'])} episodes, so the comparison between them is like for like.")

    D.h2("E. What the calendar step changes")
    V = L["calendar_first"]["variants"]
    lab_v = {"current": "No calendar step (Week 5/6)", "realised": "Realised ending required",
             "classifiable": "**Classified ending required (used above)**", "normal_only": "Calendar-normal only"}
    tl = lambda d: f"{f0(d['total'])} ({f0(d['loan'])} / {f0(d['permanent'])})"
    D.table(["Loan gate", "Revised", "Main regression", "Injury-usable", "Injury regression"],
            [[lab_v[k], tl(v["totals"]["revised"]), tl(v["totals"]["main"]), tl(v["totals"]["injury_usable"]),
              tl(v["totals"]["injury_model"])] for k, v in V.items()], [0.28, 0.18, 0.18, 0.18, 0.18])
    crb = FN["cur_rev_by_status"]
    D.bullets([f"**Realised and classified give the same samples.** All {f0(L['calendar_first']['loan_status']['unclassified'])} "
               "loans with an unclassified ending are removed anyway at the covered-league step: their receiving club is "
               "not in a covered league that season.",
               f"**What the calendar step removes** are the {f0(crb.get('ending_not_realised', 0))} revised loans whose "
               "return Transfermarkt flags as a future move: loans whose season is over in the data but whose return "
               "(mostly 30 June 2026) had not happened when the transfer history was captured.",
               f"**Calendar-normal only** also removes the {f0(crb.get('off_calendar', 0))} off-calendar revised loans. An "
               "off-calendar ending can be a consequence of low playing time (a loan cut short), so this restriction "
               "selects on the outcome. It is a sensitivity check, not a cleaner sample."])
    sg = lambda x: f"{x:+.4f}".replace("-0.0000", "0.0000").replace("+0.0000", "0.0000")
    ci = lambda m: f"{sg(m['coef'])} [{sg(m['lo'])}, {sg(m['hi'])}]"
    D.p("Same model on each sample (M3: loan, arrival timing, age, age², log relative value, log squad value, position, "
        "league × season fixed effects; standard errors clustered by receiving club):")
    D.table(["Loan gate", "Raw gap (main sample)", "Loan coefficient [95% CI]", "Injury model, fit-adjusted share"],
            [[lab_v[k], f"{v['estimates']['raw_gap_main'] * 100:+.1f} pp", ci(v["estimates"]["m3"]),
              ci(v["estimates"]["injury_m3_fit"])] for k, v in V.items()], [0.28, 0.18, 0.27, 0.27])

    D.h2("F. Which N belongs to which analysis")
    tl2 = lambda d: f"{f0(d['loan'])} / {f0(d['permanent'])}"
    cur = V["current"]["totals"]
    D.table(["N", "Loans / permanent", "What it is", "Used for"], [
        [f0(FN["master"]), f"{f0(FN['loans'])} / {f0(FN['perm'])}", f"master dataset: every transfer episode "
         f"(+ {FN['unknown']} of unknown type)", "the full dataset; no analysis uses every row"],
        [f0(FN["realised"]), "loans only", "loans with a realised ending", "calendar population"],
        [f0(FN["classifiable"]), "loans only", f"classified endings: {f0(FN['normal'])} calendar-normal, "
         f"{f0(FN['off'])} off-calendar", "off-calendar shares; gate into playing time"],
        [f0(FN["by_return"]), "loans only", f"loans ended by an End-of-loan row ({f0(FN['et'])} flagged)",
         "legacy early-termination heuristic (separate)"],
        [f0(FN["T"]["revised"]["total"]), tl2(FN["T"]["revised"]), "revised playing-time sample", "descriptive comparison"],
        [f"**{f0(FN['T']['main']['total'])}**", tl2(FN["T"]["main"]), "revised sample with every control",
         "**main loan vs permanent regression**"],
        [f0(FN["T"]["injury_usable"]["total"]), tl2(FN["T"]["injury_usable"]), "injury-usable part of the revised sample",
         "injury-adjusted shares"],
        [f"**{f0(FN['T']['injury_model']['total'])}**", tl2(FN["T"]["injury_model"]), "injury-usable with every control",
         "**injury robustness regression**"],
        [f"{f0(cur['revised']['total'])} / {f0(cur['main']['total'])} / {f0(cur['injury_model']['total'])}",
         "", "Week 5/6 sequence (no calendar step): revised / main / injury regression", "the results reported in Weeks 5–6"]],
        [0.16, 0.18, 0.36, 0.3])
    D.p("Master flags: `in_cf_revised_sample`, `in_cf_main_model_sample`, `in_cf_injury_usable`, "
        "`in_cf_injury_model_sample` (calendar-first, used above); `in_revised_sample`, `in_age_model_sample`, "
        "`in_injury_usable`, `in_injury_model_sample` (Week 5/6 sequence). The legacy early-termination count starts from "
        "a different denominator (loans ended by an End-of-loan row) and is not part of any sample.")

    D.h2("G. Sensitivity: promotions from a club's own academy")
    A, MN = L["academy"], L["club_organisation"]
    AV, PG = A["variants"], A["permanent_groups"]
    hi = PG["high-confidence same organisation"]
    ot = PG["other permanent arrivals"]
    D.p("Stage 1C calls a move internal only when the two club names match exactly once a youth/reserve suffix is "
        "removed, so many promotions such as B. Dortmund II → Dortmund or RM Castilla → Real Madrid are counted as "
        "permanent arrivals. Transfermarkt has no parent-club field. We built a separate club-organisation map "
        "(Stage 1C untouched): a youth/reserve side is linked to a senior club only when the names agree word by word "
        "(no similarity scores), both clubs carry the same country flag, and that club is clearly the most connected by "
        "player moves. Senior clubs named like reserve sides (Willem II, Hamilton Academical) are excluded by a "
        "top-flight rule, and Dynamo Kyiv II → Arsenal Kyiv is correctly not linked.")
    D.p(f"In the calendar-first revised sample, {f0(hi['n'])} permanent arrivals are high-confidence promotions "
        f"(mean minutes share {hi['mean_share']:.3f}, mean age {hi['mean_age']:.1f}) against {f0(ot['n'])} other "
        f"permanent arrivals ({ot['mean_share']:.3f}, {ot['mean_age']:.1f}). No loan in any sample is flagged.")
    vl = {"A": "A. Current (main)", "B": "B. Without high-confidence promotions",
          "C": "C. Without any promotion candidate", "D": "D. Without any youth/reserve-side arrival (bound)"}
    D.table(["Sample", "Revised (loans / perm.)", "Main N", "Injury N", "Raw gap (main)", "M3 loan coef [95% CI]",
             "Injury fit-adj. coef [95% CI]"],
            [[vl[k], tl(v["totals"]["revised"]), f0(v["totals"]["main"]["total"]), f0(v["totals"]["injury_model"]["total"]),
              f"{v['estimates']['raw_gap_main'] * 100:+.1f} pp", ci(v["estimates"]["m3"]),
              ci(v["estimates"]["injury_m3_fit"])] for k, v in AV.items()],
            [0.2, 0.14, 0.08, 0.08, 0.1, 0.2, 0.2])
    lad = lambda k, m: sg(AV[k]["ladder"][m]["coef"])
    rg = lambda k, s: f"{AV[k]['estimates'][s] * 100:+.1f}"
    D.p(f"Removing the high-confidence promotions cuts the raw loan-minus-permanent gap (revised sample "
        f"{rg('A', 'raw_gap_revised')} → {rg('B', 'raw_gap_revised')} pp; main sample {rg('A', 'raw_gap_main')} → "
        f"{rg('B', 'raw_gap_main')} pp) but leaves the adjusted coefficient where it was. "
        f"With only arrival timing and fixed effects (M1) the loan coefficient moves from {lad('A', 'M1')} to "
        f"{lad('B', 'M1')}; once player and squad value are controlled (M3) it moves from {lad('A', 'M3')} to "
        f"{lad('B', 'M3')}. Promoted players have a mean log relative value of {hi['mean_log_relative_value_main']:.2f} "
        f"against {ot['mean_log_relative_value_main']:.2f}, so the value controls already absorb their shortfall.")
    R = A["review"]
    bs = R["by_status"]
    n_flag = sum(sum(bs.get(k, {}).values()) for k in ("high", "medium", "low"))
    fp_text = (f"no false positives were found among the {n_flag} flagged observations in the manual review sample"
               if not R["false_positives"] else f"{len(R['false_positives'])} of the {n_flag} flagged observations in "
               "the manual review sample are false positives")
    D.p(f"Manual check of {R['rows']} stratified rows across {R['leagues']} leagues: {fp_text}; "
        f"{len(R['false_negatives'])} unflagged rows are promotions the names cannot reveal (SVZW U21 → Zulte Waregem, Loko-Kazanka → Lokomotiv "
        "Moscow). The correction is reported as a sensitivity; the main sample is unchanged. Master flags: "
        "`same_organisation_move_v2` (high confidence), `academy_promotion_candidate` (any confidence), "
        "`same_org_confidence`, `same_org_reason`. Full report: `academy_promotion_sensitivity.md`.")

    # ================================================================ 6
    D.h1("Key variable lineage: column → result")
    D.h2("A. Playing time")
    D.diagram(r"""\begin{tikzpicture}[
  box/.style={draw, rounded corners=2pt, align=center, font=\footnotesize, inner sep=4pt},
  src/.style={box, fill=black!6}, ours/.style={box, fill=blue!7},
  arr/.style={-{Stealth[length=2mm]}, thick}]
\node[src, text width=4.3cm] (a) at (-5.1,0) {\textbf{Stage 1C} (source)\\ player\_id\\ to\_club\_id (club B)\\ transfer\_date};
\node[src, text width=4.3cm] (b) at (0,0) {\textbf{games} (source)\\ game\_id, competition\_type, season, date,
  home\_club\_id, away\_club\_id};
\node[src, text width=4.3cm] (c) at (5.1,0) {\textbf{appearances} (source)\\ game\_id, player\_id, player\_club\_id,\\
  \textbf{minutes\_played}};
\node[ours, text width=14.2cm] (w) at (0,-2.15) {\textbf{Observation window} (our rule): from transfer\_date to B's last
  domestic-league game of that season, or to the day before the player's next recorded move if that comes first};
\node[ours, text width=4.3cm] (g) at (-5.1,-4.05) {\textbf{available\_games}\\ number of B's league games in the window};
\node[ours, text width=4.3cm] (k) at (0,-4.05) {\textbf{completeness}\\ every window game has $\geq$ """
              + str(pt["complete_rows"]) + r""" appearance rows for B};
\node[ours, text width=4.3cm] (n) at (5.1,-4.05) {\textbf{minutes}\\ sum of the player's minutes\_played for B in window games};
\node[ours, text width=14.2cm] (s) at (0,-6.05) {\textbf{minutes\_share} = minutes / (90 $\times$ available\_games)\\
  all games complete and no rows $\Rightarrow$ true zero \qquad any game incomplete $\Rightarrow$ unreliable, no value};
\foreach \x in {a,b,c} \draw[arr] (\x.south) -- (\x.south |- w.north);
\foreach \x in {g,k,n} \draw[arr] (\x.north |- w.south) -- (\x.north);
\foreach \x in {g,k,n} \draw[arr] (\x.south) -- (\x.south |- s.north);
\end{tikzpicture}""",
              ["Stage 1C: player_id, to_club_id (club B), transfer_date",
               "games: game_id, competition_type, season, date, home_club_id, away_club_id",
               "appearances: game_id, player_id, player_club_id, minutes_played",
               "        |  observation window (our rule): transfer_date .. B's last league game,",
               "        |  or the day before the player's next recorded move",
               "        v",
               "available_games | completeness (>= 11 rows per game for B) | minutes = sum(minutes_played)",
               "        v",
               "minutes_share = minutes / (90 x available_games)"],
              "Diagram 3. Playing-time construction")
    D.formula(r"\text{minutes\_share} \;=\; \frac{\sum \text{appearances.minutes\_played (player, club B, window league games)}}"
              r"{90 \times \text{number of B's domestic-league games in the window}}",
              "minutes_share = sum(appearances.minutes_played) / (90 x available_games)")
    D.p("`appearances.minutes_played` is the upstream field that holds the minutes actually played. We only sum it. Only "
        f"domestic-league games count, in the {len(pt['covered_leagues'])} leagues that have appearance data in every "
        f"season ({seasons}).")
    D.p("**True zero.** `appearances` has a row only for a player who played. A missing row means zero minutes only if "
        f"every game in the window has a complete record for club B (at least {pt['complete_rows']} appearance rows). "
        "Then minutes = 0 and minutes_share = 0. If even one game is incomplete, we do not assign a zero: the outcome is "
        "marked unreliable and the episode leaves the playing-time samples.")
    D.bullets([f"**Zero:** {tz['player']}, loan to {tz['b_club']} ({tz['league']}), window {tz['window_start']} to "
               f"{tz['window_end']}. {tz['available_games']} league games, all {tz['complete_games']} complete, no "
               f"appearance row → minutes 0, share 0 / (90 × {tz['available_games']}) = 0.",
               f"**Not a zero:** {az['player']}, {az['kind']} move to {az['b_club']} ({az['league']}). "
               f"{az['available_games']} games, only {az['complete_games']} complete, no appearance row → unreliable, no "
               "value.",
               f"**Positive share:** {an['player']}, {an['a_club']} → {an['b_club']}: {f0(an['minutes'])} minutes in "
               f"{an['games']} games → {f0(an['minutes'])} / (90 × {an['games']}) = {an['current_share']:.3f}."])
    lu, lt = d["EMP"]["lineup"], d["EMP"]["lineup_top"]
    rq = next(r for r in d["EMP"]["robustness"] if r["specification"].startswith("Calendar-first, without episodes whose"))
    D.p(f"**Known gap in the upstream appearance data.** Cross-checking against the snapshot's starting line-ups and "
        f"substitution events finds {f0(lu['main_contradicted_zero'])} of the {f0(lu['main_true_zeros'])} true zeros in "
        "the main regression sample where the player did play for club B but has no `appearances` row at all (for "
        f"example {lt['player_name']}, {lt['from_club_name']} → {lt['to_club_name']}: {int(lt['started_no_row'])} starts "
        f"and {int(lt['subbed_on_no_row'])} substitute appearances, recorded as zero). The ≥ "
        f"{pt['complete_rows']}-row test checks the club's rows, so it cannot see one missing player. The samples are "
        f"unchanged; the master marks these episodes (`pt_lineup_contradicted`; `pt_reliable_v2` = reliable and not "
        f"contradicted). Without the {f0(lu['main_contradicted'])} contradicted episodes the M3 loan coefficient is "
        f"{100 * rq['coef']:+.2f} pp (`professor_empirical_summary.md`, sections 9 and 11; full audit by season, league "
        "and club in `appearance_audit.md`).")

    D.h2("B. Country and geography")
    src = geo["sources_used"]
    D.flow(["club_id",
            f"  +--> league / competition country (DuckDB clubs, competitions, games)     {f0(src.get('v1 league/competition geography', 0))} clubs",
            f"  +--> else Transfermarkt's own countryFlag on the club's raw rows (one flag) {f0(src.get('raw Transfermarkt countryFlag', 0))} clubs",
            f"  +--> else fixed fallbacks: parent-club link, Wikidata                       {sum(v for k, v in src.items() if k.startswith('fallback'))} clubs",
            "  v",
            "calendar_country",
            "",
            "lender country + borrower country",
            "  -> same country: domestic; both known, different: cross-border; else unmapped",
            "  -> calendar key: 'domestic: <country>' or 'cross-border'"])
    D.p(f"**No LLM guesses a club country, and no country is inferred from a club name.** {geo['with_country']:,} of "
        f"{geo['real_clubs']:,} real Stage 1C clubs get a country; the rest stay blank.")

    D.h2("C. Off-calendar")
    D.flow(["terminal_date (the date on the loan's ending row, from Transfermarkt)",
            "  + calendar key (from the two club countries)",
            "  + learned recurring ending dates for that calendar",
            "  -> within +/-3 days of a learned date: calendar-normal",
            "  -> otherwise: off-calendar",
            "  -> no calendar for that country: unclassified"])
    D.p(f"Calendars are learned from past non-COVID domestic endings: a country needs at least {prm['N_MIN']} of them; "
        f"dates within ±{prm['TOL']} days are grouped; a date must recur in at least {prm['Y_MIN']} years and hold at least "
        f"{prm['S_MIN']:.0%} of endings; at most {prm['K_MAX']} dates are kept, stopping at {prm['TARGET']:.0%} coverage. "
        "Cross-border loans share one pooled calendar.")
    D.p("Off-calendar is **created by us**. It is a timing classification, not a Transfermarkt field, not proof of early "
        "termination, and **not a filter** for the playing-time or injury samples.")

    D.h2("D. Legacy early-termination heuristic")
    D.p("Source columns: the ending row's date and type; the player's next move after the return (its type, destination "
        "and number of days later); the borrower club; the distance of the return date from 30 June and 31 December "
        "(`terminal_date`, `follow_on_event_id`, `follow_on_gap_days`, `follow_on_type`, borrower club, "
        "`return_boundary_days`).")
    D.table(["Rule (applied in this order)", "Loans left"],
            [[k.replace(" (next deal within 60 days, not a release)", ": a next deal within 60 days, not a release"), f0(v)]
             for k, v in lo["funnel"].items()], [0.8, 0.2])
    D.box(["**CREATED BY US. NOT A TRANSFERMARKT FIELD. NOT PROOF THAT THE CONTRACT WAS TERMINATED EARLY.**",
           "A permanent move back to the borrower (conversion) and a paid or undisclosed-fee move to a third club "
           "(sale) are classified first, so they never count here."])

    D.h2("E. Market value")
    D.flow(["player_valuations.market_value_in_eur (source)",
            "  -> the player's latest valuation dated strictly before the move     => player_value",
            "",
            "player_valuations for every player listed at the receiving club (source)",
            "  -> each player's latest valuation before the move, within 365 days",
            "  -> drop the moving player; need at least 11 players; sum              => squad_value",
            "",
            "player_value / squad_value                                            => relative_value",
            "ln(player_value) - ln(squad_value)                                    => log_relative_value"])
    D.bullets(["`player_value`: an **upstream value, selected by us** (latest before the move; same-day values are not "
               f"used). Missing for {f0(mv['N'] - mv['player_value'])} of the {f0(mv['N'])} revised episodes.",
               "`squad_value`: an **aggregate created by us**. Transfermarkt's own club squad value is a current figure "
               "and is empty in our snapshot, so it cannot be used.",
               "`relative_value`: a **formula created by us**."])

    D.h2("F. Injury")
    D.diagram(r"""\begin{tikzpicture}[
  box/.style={draw, rounded corners=2pt, align=center, font=\footnotesize, inner sep=4pt},
  src/.style={box, fill=black!6}, ours/.style={box, fill=blue!7}, ext/.style={box, fill=orange!15},
  arr/.style={-{Stealth[length=2mm]}, thick}]
\node[ext, text width=4.3cm] (k) at (-5.1,0) {\textbf{Kaggle player\_injuries} (EXTERNAL)\\ player\_id, from\_date,
  end\_date, injury\_reason};
\node[src, text width=4.3cm] (g) at (0,0) {\textbf{games} (source)\\ date of each of B's league games in the window};
\node[src, text width=4.3cm] (a) at (5.1,0) {\textbf{appearances} (source)\\ did the player appear for B in that game?};
\node[ours, text width=14.2cm, align=left] (r) at (0,-2.45) {\textbf{Our rules}\\
  1. injury\_reason $\rightarrow$ medical or not (Rest, Fitness, Quarantine never count)\\
  2. a spell covers a game if from\_date $\leq$ game date $\leq$ end\_date (both inclusive)\\
  3. if the player appeared in that game, it stays in the denominator (appearance overrides)\\
  4. a game covered by several spells is removed once};
\node[ours, text width=14.2cm] (c) at (0,-4.6) {\textbf{confirmed\_injury\_games} = window games covered by a medical
  spell in which the player did not appear};
\node[ours, text width=14.2cm] (f) at (0,-6.0) {\textbf{available\_games\_fit} = available\_games $-$
  confirmed\_injury\_games\\ \textbf{fit\_adjusted\_share} = minutes / (90 $\times$ available\_games\_fit)};
\foreach \x in {k,g,a} \draw[arr] (\x.south) -- (\x.south |- r.north);
\draw[arr] (r) -- (c); \draw[arr] (c) -- (f);
\end{tikzpicture}""",
              ["KAGGLE (external): player_id, from_date, end_date, injury_reason",
               "games (source): date of each of B's league games in the window",
               "appearances (source): did the player appear for B in that game?",
               "        |  OUR RULES: medical or not; from_date <= game date <= end_date;",
               "        |  appearance overrides; overlapping spells counted once",
               "        v",
               "confirmed_injury_games",
               "available_games_fit = available_games - confirmed_injury_games",
               "fit_adjusted_share = minutes / (90 x available_games_fit)"],
              "Diagram 4. Injury-adjusted playing time")
    D.p("Kaggle gives us `player_id`, `from_date`, `end_date`, `injury_reason`, `days_missed` and `games_missed`. **We do "
        "not use Kaggle's `games_missed` to build the denominator.** We take only the player id, the two dates and the "
        "label; the games, the appearances and every rule that turns spells into removed games are ours.")
    D.formula(r"\text{fit\_adjusted\_share} = \frac{\text{minutes}}{90 \times (\text{available\_games} - "
              r"\text{confirmed\_injury\_games})}",
              "fit_adjusted_share = minutes / (90 x (available_games - confirmed_injury_games))")
    D.p(f"Example: {an['player']} had {an['games']} window games and {f0(an['minutes'])} minutes. His 'thigh problems' "
        f"spell starts on the day of a game he played (kept); {an['removed']} later games fall inside medical spells and "
        f"he did not play (removed). Share: {an['current_share']:.3f} → {f0(an['minutes'])} / (90 × {an['fit_games']}) = "
        f"{an['fit_share']:.3f}.")

    D.h2("G. Age, older, loan × older")
    D.flow(["players.date_of_birth + transfer_date  -> age = (transfer_date - date_of_birth) / 365.25",
            "floor(age) >= 24                       -> older = 1   (young = 23 or under)",
            "loan indicator x older                 -> loan x older"])
    D.p("Loan × older measures how much the loan − permanent difference changes for older players relative to younger "
        "players.")

    # ================================================================ 7
    D.h1("Source data vs variables we created")
    eg = lambda cat, scope="master": ", ".join(f"`{v}`" for v in PM[(PM.scope == scope) & (PM.provenance_category == cat)]
                                               .variable.head(3))
    D.table(["Category", "Meaning", "Example"], [
        ["**Transfermarkt raw**", "published by Transfermarkt, same transfer row, unchanged", eg(vp.RAW, "stage1c")],
        ["**Transfermarkt joined**", "Transfermarkt value from another table or row, attached by our join", eg(vp.JOIN)],
        ["**External raw**", "published by a non-Transfermarkt source", "Kaggle injury dates and labels; Wikidata country"],
        ["**Our mechanical derivation**", "arithmetic, counts, dates, exact recodes", eg(vp.MECH)],
        ["**Our classification rule**", "a rule we designed (threshold, window, priority, pattern, learned calendar)",
         eg(vp.CLASS)],
        ["**Our sample flag**", "eligibility, reliability or sample membership", eg(vp.FLAG)],
        ["**Statistical output**", "produced by a model", "regression coefficients"]],
        [0.24, 0.42, 0.34])
    D.box(["**Claude / AI helping write code is not the same as AI-generated data.**",
           f"No Stage 1C column, master column or Week 4–6 report variable has a stored value determined by an LLM "
           f"({n_yes} such values in the provenance audit)."])
    D.p("The only code in the repository that calls an LLM belongs to the Stage 2 contract research. None of the modules "
        "that build Stage 1C, the master table or the report variables imports it, directly or indirectly.")
    D.p(f"{n_unknown} sources are marked UNKNOWN rather than NO: the Kaggle injury rows, the Kaggle player profiles and "
        "the Wikidata country. UNKNOWN means the third party does not document how it compiled its data. It does "
        "**not** mean our code used AI: we copy those values unchanged with ordinary code.")
    D.table(["Category", "Stage 1C columns", "Master columns", "Report variables (Weeks 4–6)"],
            [[TYPE_LABEL[k], str(int(c1.get(k, 0))), str(int(cm.get(k, 0))), str(int(cr.get(k, 0)))] for k in vp.CATEGORIES]
            + [["**Total**", f"**{int(c1.sum())}**", f"**{int(cm.sum())}**", f"**{int(cr.sum())}**"]],
            [0.34, 0.2, 0.2, 0.26])

    # ================================================================ 8
    D.h1("Where do I find everything?")
    R = "data/outputs/rebuild/"
    files = [
        ("The raw Transfermarkt snapshot (all DuckDB tables)", "data/raw/transfermarkt-datasets.duckdb"),
        ("Raw transfer histories (fee text, club flags)", "data/raw/transfermarkt-api/<2023|2024|2025>/transfers.json"),
        ("Raw Kaggle injury rows", "data/raw/external_injuries/salimt_kaggle/player_injuries/player_injuries.csv"),
        ("Raw Kaggle player profiles", "data/raw/external_injuries/salimt_kaggle/player_profiles/player_profiles.csv"),
        ("Cleaned injury rows with our medical labels",
         "data/processed/external_injuries/salimt_kaggle_injuries_normalized.csv"),
        ("How every injury label is classified", R + "injury_reason_classification.csv"),
        ("Wikidata answers", "data/raw/wikidata/transfermarkt_team_p7223.json"),
        ("Canonical transfers (Stage 1C)", R + "stage1c_canonical_transfers.csv"),
        ("Definitions of all Stage 1C fields", R + "stage1c_column_dictionary.csv"),
        ("Master merged episode dataset", R + "analysis_episode_master.csv"),
        ("Definitions of all master fields", R + "analysis_variable_dictionary.csv"),
        ("Raw-vs-derived provenance, every column", R + "variable_provenance_matrix.csv"),
        ("Readable provenance explanation", R + "variable_provenance_overview.md"),
        ("Full lineage, waterfalls, worked examples", R + "data_lineage_overview.md"),
        ("Calendar-first samples: every step, four variants", R + "calendar_first_samples.md"),
        ("…the same steps as a table", R + "calendar_first_samples.csv"),
        ("The most useful master columns (shortlist)", R + "master_core_columns.md"),
        ("Playing-time QA: independent recomputation, examples", R + "playing_time_qa.md"),
        ("…game-level rows of the examples", R + "playing_time_qa_example_games.csv"),
        ("FINAL result set: README, tables, figures, data-quality notes", R + "final/FINAL_README.md"),
        ("Empirical summary: loans vs permanent, ladder, robustness, examples", R + "professor_empirical_summary.md"),
        ("Appearance rows vs line-ups and substitutions (audit)", R + "appearance_audit.md"),
        ("…its tables (CSV) and figures (PNG, PDF)", R + "empirical/"),
        ("Academy promotions: method, sensitivity, manual check", R + "academy_promotion_sensitivity.md"),
        ("Youth/reserve side → senior club map", R + "club_organisation_map.csv"),
        ("Youth/reserve review universe (all episodes touching a side)", R + "youth_reserve_review_universe.csv"),
        ("Every source table: rows, columns, use", R + "data_source_inventory.md"),
        ("Every number in this guide", R + "data_lineage_numbers.json"),
        ("Club countries with their source", R + "club_geography_v2.csv"),
        ("Calendar class of every realised loan ending", R + "loan_calendar_classified_v2.csv"),
        ("The learned calendars", R + "loan_calendar_rules_v2.csv"),
        ("Injury outcome per revised episode", R + "playing_time_injury_adjusted_episodes.csv"),
        ("Game-by-game injury examples", R + "injury_method_example.md"),
        ("Injury validation events", R + "injury_external_validation.csv"),
    ]
    for _, path in files:
        if "<" not in path:
            assert (ROOT / path).exists(), path
    D.table(["If you want…", "Open this file"], [[a, f"`{b}`"] for a, b in files], [0.4, 0.6])

    # ================================================================ 9
    D.h1("Source fact vs our research decision")
    D.p("The left column is what the data say. The right column is what we decided, and is open to challenge.")
    facts = ["Transfermarkt player id", "registration date of each movement", "from club and to club",
             "Transfermarkt's fee label ('End of loan', '?', '-')", "minutes played in each match (minutes_played)",
             "which matches each club played, and when", "injury start and end dates and labels (Kaggle)",
             "player valuation records and dates", "club country flags on transfer rows"]
    decisions = ["how rows are paired into a loan episode", "how fee labels map to transfer types",
                 "the 60-day follow-on window", "the 21-day early-termination follow-on window",
                 "the 10-day distance from 30 June / 31 December", "the ±3-day calendar tolerance",
                 "the 100-ending calendar threshold", "the 5-date calendar cap", "the COVID window excluded from learning",
                 "the playing-time reliability rule (≥ 11 rows per game)", "the 23 / 24 age cutoff",
                 "the squad-value rule (365 days, ≥ 11 players)", "which injury labels count as medical",
                 "the injury appearance override", "which observations enter each sample",
                 "loans need a classified ending before the playing-time step",
                 "which youth/reserve side belongs to which senior club (academy sensitivity only)"]
    n = max(len(facts), len(decisions))
    D.table(["SOURCE FACT", "OUR RESEARCH DECISION"],
            [[facts[i] if i < len(facts) else "", decisions[i] if i < len(decisions) else ""] for i in range(n)], [0.5, 0.5])

    # ================================================================ appendix
    D.raw(r"\appendix", "")
    D.h1("Appendix A: all 42 Stage 1C columns")
    D.p("One row per Stage 1C column, in file order. **Role** says what the column is for. Categories are listed with "
        "their counts in the 175,165 rows and what each means. Full detail: `stage1c_column_dictionary.csv`.")
    rows = []
    for c in D1.index:
        x = D1.loc[c]
        rows.append([name_tex(c) + r"\newline{\footnotesize\emph{" + tex(x.role) + "}}",
                     f"**Meaning.** {x.plain_english_meaning} **From:** {x.comes_from}. **How:** {x.how_created}",
                     x.categories_explained or f"free values; {x.missing_value_meaning}"])
    D.table(["Column and role", "Meaning, source and construction", "Values and what they mean"], rows,
            [0.2, 0.42, 0.38], size="scriptsize", raw_cols=(0,))

    D.h1(f"Appendix B: all {mm['columns']} master columns")
    D.p("**Origin** says where the value comes from:")
    D.bullets([f"**{ORIGIN_SHORT[k]}**: {v}" for k, v in ORIGINS.items()])
    D.p("**Built from** lists the inputs (other master columns, or the source table's columns); **How** is the rule "
        "(blank where the value is copied). **Use**: core = core analysis variable; flag = sample / filter flag; "
        "helper = intermediate step; QA = audit; legacy = superseded or Stage 2; id = identifier; special = currently "
        "unused or highly specialised. Full detail: `analysis_variable_dictionary.csv` (columns `origin`, `usefulness`); "
        "the shortlist of the most useful columns is `master_core_columns.md`.")
    inv = {v: k for k, v in ORIGINS.items()}
    use_short = {"core analysis variable": "core", "sample / filter flag": "flag", "intermediate / helper": "helper",
                 "QA / audit": "QA", "legacy": "legacy", "identifier / linkage": "id",
                 "currently unused or highly specialised": "special"}
    rows = []
    for c in d["head"]:
        x = D2.loc[c]
        o = inv[x.origin]
        copied = o in ("tm", "s1c") or x.formula_or_rule.startswith("copied")
        rows.append([name_tex(c), use_short[x.usefulness], ORIGIN_SHORT[o], x.source_columns,
                     "copied" if copied else x.formula_or_rule])
    D.table(["Column", "Use", "Origin", "Built from", "How"], rows, [0.19, 0.075, 0.15, 0.22, 0.365],
            size="scriptsize", raw_cols=(0,))

    D.h1("Appendix C: when each master column can be interpreted")
    D.p("**Valid when** names the rows on which a value means something; filter on that flag first. **Rows** is how "
        "many master rows satisfy it. **Outside** counts values that exist on other rows: a placeholder (False, or "
        "'(not in a covered league)') or a value computed but not interpretable there (injury columns outside the "
        "injury-usable episodes). Blank outside the domain is checked by code. Columns valid on every row are listed "
        "together at the end. Same content: `analysis_variable_dictionary.csv` (`valid_when`, `rows_valid`, "
        "`values_outside_valid`).")
    rows, everywhere = [], []
    for c in d["head"]:
        x = D2.loc[c]
        if x.valid_domain == "all" and x.valid_when == "every episode":
            everywhere.append(c)
            continue
        out = int(x.values_outside_valid)
        rows.append([name_tex(c), x.valid_when.split(" (the other ")[0], f0(x.rows_valid),
                     f"{f0(out)} (do not interpret)" if out else "blank"])
    D.table(["Column", "Valid when", "Rows", "Outside"], rows, [0.24, 0.5, 0.09, 0.17], size="scriptsize",
            raw_cols=(0,))
    D.p(f"Valid on every row ({len(everywhere)} columns): " + ", ".join(f"`{c}`" for c in everywhere) + ".")
    return D


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{enumitem}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{3em}
\title{Football transfers project: a guide to the data}
\author{Chris Low}
\date{%s}
\begin{document}
\maketitle
\noindent\textit{Built from the project's lineage and provenance files. Every number is generated from them; nothing here
is a new analysis.}
\medskip
"""


def main() -> None:
    before = dl.watched()
    d = load()
    D = build(d)
    date = pd.Timestamp.today().strftime("%B %-d, %Y")
    TEX.write_text(PREAMBLE % date + D.tex() + "\n\\end{document}\n")
    MD.write_text(f"# Football transfers project: a guide to the data\n\n*Chris Low, {date}. Built from the project's "
                  "lineage and provenance files; every number is generated from them.*\n\n" + D.md())
    after = dl.watched()
    changed = [k for k in before if before[k] != after.get(k) and not k.endswith((TEX.name, MD.name))]
    assert not changed, changed
    print(f"wrote {TEX.name}, {MD.name}")


if __name__ == "__main__":
    main()
