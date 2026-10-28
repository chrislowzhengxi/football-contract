"""Two-page professor-facing data overview, rendered from the data-lineage outputs only.

    python -m src.analysis.professor_data_overview

Reads data_lineage_numbers.json, data_source_inventory.csv and the master table; runs no analysis and changes no
definition. Every number is interpolated. Writes professor_data_overview.md and professor_data_overview.tex (the
LaTeX version, to paste before Section 1 of the meeting document; the document itself is not touched).
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import data_lineage as dl
from . import playing_time_regression as reg
from .daniel_scope_final import f0
from ..stage2 import event_family as ef

MD = dl.OUT / "professor_data_overview.md"
TEX = dl.OUT / "professor_data_overview.tex"
WRITES = (MD, TEX)
PDF = dl.ROOT / "data" / "pdf" / "Transfermarkt_Short_Week_6.pdf"


# ---------------------------------------------------------------------------
# Inputs and checks
# ---------------------------------------------------------------------------

def load() -> dict:
    N = json.loads(dl.NUMBERS_JSON.read_text())
    inv = pd.read_csv(dl.INVENTORY_CSV).set_index("source").rows
    head = pd.read_csv(dl.MASTER_CSV, nrows=0).columns
    n_master = len(pd.read_csv(dl.MASTER_CSV, usecols=["event_id"]))
    ep = N["episodes"]
    st = {s["role"]: s["removed"] for s in ep["steps"]}
    unknown = ep["episode_type"].get("unknown_type", 0)
    assert (len(head), n_master) == (N["master"]["columns"], N["master"]["rows"]) == (N["master"]["columns"], ep["episodes"])
    assert ep["rows"] - sum(st.values()) == ep["episodes"]
    assert ep["loans"] + ep["permanent_pooled"] + unknown == ep["episodes"]
    for r in N["inventory"]:
        assert inv[r["source"]] == r["rows"], r["source"]
    return {"N": N, "inv": inv, "st": st, "unknown": unknown}


# ---------------------------------------------------------------------------
# Content (one source for both renderings; **bold**, `code`, *italic* allowed)
# ---------------------------------------------------------------------------

def content(d: dict) -> dict:
    N, inv, st = d["N"], d["inv"], d["st"]
    ep, s1, lo, ca, pt, inj = N["episodes"], N["stage1c"], N["loans"], N["calendar"], N["playing_time"], N["injury"]
    p = ca["params"]
    s0, s1_ = pt["covered_seasons"]
    seasons = f"{s0}/{str(s0 + 1)[2:]}–{s1_}/{str(s1_ + 1)[2:]}"
    revised = N["master"]["membership"]["in_revised_sample"]
    sources = [
        ["Transfers (Stage 1C)", "one registration movement of one player between two clubs (or a club and a "
         "placeholder such as 'Without Club') on one date", f"{f0(inv['stage1c_canonical_transfers.csv'])}",
         "player id, date, from- and to-club ids, fee label", "loans and permanent moves; loan start and end dates"],
        ["Players", "one player", f0(inv["players"]), "date of birth, position", "age, position"],
        ["Club geography (built by us)", "one club", f0(inv["club_geography_v2.csv"]),
         "league country, else club flag", "domestic vs cross-border; country calendars"],
        ["Games", "one match", f0(inv["games"]), "date, season, competition, clubs",
         "the receiving club's league games"],
        ["Appearances", "one player appearing in one match for one club (no row if he did not play)",
         f0(inv["appearances"]), "minutes played, club", "minutes; complete-record check"],
        ["Player valuations", "one market value of one player on one date (with his club)",
         f0(inv["player_valuations"]), "date, value, club", "player value; receiving-squad value"],
        ["Injury data (Kaggle)", "one listed injury or illness spell of one player",
         f0(inv["player_injuries.csv (Kaggle salimt v2)"]), "player id, start and end date, injury type",
         "games a player was medically unavailable"],
    ]
    waterfall = [
        f"{f0(ep['rows'])}  Stage 1C registration movements ({f0(s1['players'])} players)",
        f"   |  - {f0(st['internal_registration']):>6}  internal registrations (e.g. a club's U19 side -> its first team)",
        f"   |  - {f0(st['placeholder_non_club_movement']):>6}  moves to or from a placeholder ('Without Club', 'Retired', ...)",
        f"   |  - {f0(st['non_transfer_other']):>6}  draft rows",
        f"   |  - {f0(st['loan_return_collapsed']):>6}  loan-return rows, attached to the loan they end",
        f"   |  - {f0(st['loan_return_unmatched']):>6}  loan-return rows with no matching loan (set aside)",
        "   v",
        f"{f0(ep['episodes'])}  transfer episodes ({f0(ep['episode_players'])} players) = {f0(ep['loans'])} loans "
        f"+ {f0(ep['permanent_pooled'])} permanent + {d['unknown']} unknown",
        f"   v  analysis_episode_master.csv: {f0(N['master']['rows'])} rows x {N['master']['columns']} columns, "
        "one row per transfer episode"]
    variables = [
        ["**Loan ending date**", "Stage 1C: date of the row ending the loan (normally 'End of loan')",
         "Each loan is paired one-to-one with the movement that ends it. *Realised*: exists and not flagged as a future "
         f"move ({f0(lo['realised'])} of {f0(lo['loans'])} loans).",
         "Transfermarkt's recorded date (often the scheduled end), not match data."],
        ["**Early-termination heuristic** (legacy)", "Ending date; next move after the return (type, club, gap)",
         f"Return > {ef.SEASON_END_TOLERANCE_DAYS} days from both 30 June and 31 December, and next move within "
         f"{ef.FOLLOW_ON_IMMEDIATE_DAYS} days (after ruling out a permanent move back to the borrower or a paid sale "
         f"to a third club). {f0(lo['early_termination'])} loans.",
         "**Built by us; not a Transfermarkt field; not proof** of contractual early termination."],
        ["**Off-calendar**", "Ending date; lender and borrower club → country (club geography)",
         f"Recurring ending dates learned per country from past non-COVID domestic endings (≥ {p['N_MIN']} endings; "
         f"±{p['TOL']}-day clusters; ≥ {p['Y_MIN']} years; ≥ {p['S_MIN']:.0%} share; ≤ {p['K_MAX']} dates or "
         f"{p['TARGET']:.0%} coverage); one pooled cross-border calendar. Off-calendar = not within ±{p['TOL']} days of "
         "a calendar date.",
         "A **timing classification, not a sample filter**. Unclassifiable = too little history, not unusual timing."],
        ["**Playing-time share**", "Stage 1C: player, receiving club, move date. Games: date, season, clubs. Appearances: minutes",
         "minutes ÷ (90 × receiving-club league games in the window). Window: move date to the club's last league game "
         "of the season, or the day before the player's next move.",
         f"League games only; {len(pt['covered_leagues'])} leagues with appearance data in every season ({seasons})."],
        ["**True zero**", "Appearance rows of every window game",
         f"Every window game complete (≥ {pt['complete_rows']} players listed for the club) **and** no appearance row "
         "for the player → minutes = 0 → share = 0.",
         "Any incomplete game → no zero: the outcome is unreliable and the episode is excluded."],
        ["**Player market value**", "Valuations: date, value", "Latest valuation dated strictly before the move.",
         "Same-day valuations excluded; none earlier → missing."],
        ["**Receiving-squad value**", "Valuations: player, date, value, club",
         f"Sum of the latest pre-move valuations (within {reg.LOOKBACK_DAYS} days) of players listed at the receiving "
         f"club, excluding the moving player; missing if fewer than {reg.MIN_SQUAD_PLAYERS} players.",
         "Value as of the move; Transfermarkt's club value is current (and empty in our data)."],
        ["**Relative player value**", "Player value, squad value", "Player value ÷ squad value (log in the models).",
         "The player's size relative to the squad he joins."],
        ["**Injury-unavailable game**", "Injuries: player, start, end, type. Games: date. Appearances",
         "Removed only if a medical spell covers the game date (inclusive) **and** the player did not appear.",
         "Appearance always wins; rest, fitness, quarantine never count; nothing inferred from absence."],
        ["**Injury-adjusted share**", "Minutes; window games; injury-unavailable games",
         "`available_games_fit` = available games − injury-unavailable games; `fit_adjusted_share` = minutes ÷ "
         "(90 × `available_games_fit`).", "Same minutes, smaller denominator: never below the unadjusted share."],
    ]
    diagram = [
        f"TRANSFER EPISODES ({f0(ep['episodes'])}; one row each in the master table)",
        "  +-- + club geography                  -> loan calendar: calendar-normal / off-calendar",
        "  +-- + games + appearances             -> playing-time share, true zeros",
        "  +-- + player valuations               -> player value, squad value, relative value",
        "  +-- + injuries + games + appearances  -> injury-adjusted playing time"]
    text = {
        "intro": "Transfers, matches and market values come from one public snapshot of Transfermarkt data (the "
                 "open-source *transfermarkt-datasets* project), with Transfermarkt's original fee labels restored from "
                 "its raw files. Injuries come from a public Kaggle dataset (CC0), also built from Transfermarkt, linked "
                 f"by exact player id. It lists injuries for {f0(inj['raw_players'])} players; a second file of "
                 f"{f0(inv['player_profiles.csv (Kaggle salimt v2)'])} player profiles separates 'no injury listed' from "
                 "'not in the injury data'.",
        "defs": [
            "**A Stage 1C row is a registration movement, not a contract.** A loan is normally two rows (out, and "
            "'End of loan' back); a youth-to-first-team promotion inside a club is a row with no deal behind it.",
            "**A transfer episode is one real loan or one permanent arrival** (paid, free, undisclosed-fee or no fee "
            "shown).",
            "**For a loan, the outbound row is the episode and its return row is attached to it**, supplying the "
            f"ending date ({f0(ep['returns_closing_real_loans'])} of {f0(ep['loans'])} loans have one)."],
        "outside_intro": "**Kept outside the master table** (several rows per episode, so merging would duplicate "
                         "episodes); each is summarised to one value per episode:",
        "outside": [
            "**game-by-game rows**, one per club league game in the window → minutes, available games;",
            "**injury spells**, several per player and window → games removed for injury;",
            f"**raw loan-return rows** ({f0(st['loan_return_collapsed'])} matched; kept in Stage 1C) → ending date, next move."],
        "partial": f"Market-value and injury columns are filled only for the {f0(revised)} episodes of the revised "
                   "playing-time sample; a blank elsewhere means 'not computed', not 'missing'.",
        "separate": "These branches are separate. In particular, off-calendar classification is not used to select the "
                    "playing-time or injury samples.",
    }
    return {"sources": sources, "waterfall": waterfall, "variables": variables, "diagram": diagram, "text": text}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def md_cell(s: str) -> str:
    return s.replace("|", "\\|")


def render_md(c: dict) -> str:
    t = c["text"]
    W = ["# Data overview", "", "## Data sources", "", t["intro"], "",
         "| Source | What one row means | Rows | Fields we use | What it gives us |", "| --- | --- | ---: | --- | --- |"]
    W += ["| " + " | ".join(md_cell(x) for x in r) + " |" for r in c["sources"]]
    W += ["", "```"] + c["waterfall"] + ["```", ""]
    W += [f"- {x}" for x in t["defs"]] + ["", t["outside_intro"], ""] + [f"- {x}" for x in t["outside"]]
    W += ["", t["partial"], "", "## How the variables in this report are built", "",
          "| Report variable | Raw / source columns | Construction | Important interpretation |", "| --- | --- | --- | --- |"]
    W += ["| " + " | ".join(md_cell(x) for x in r) + " |" for r in c["variables"]]
    W += ["", "```"] + c["diagram"] + ["```", "", f"*{t['separate']}*", ""]
    return "\n".join(W)


# ---------------------------------------------------------------------------
# LaTeX (ASCII only, standard commands only: no extra packages needed)
# ---------------------------------------------------------------------------

UNI = {"×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$", "→": r"$\rightarrow$", "÷": r"$\div$",
       "−": r"$-$", "–": "--", "—": "---", "…": r"\ldots{}", ">": r"$>$", "<": r"$<$"}   # bare < > misprint in OT1


def tex(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
    s = re.sub(r"(?<![A-Za-z])'([^']+?)'", r"`\1'", s)
    for a, b in UNI.items():
        s = s.replace(a, b)
    assert all(ord(ch) < 128 for ch in s), s
    return s


def tex_table(widths: list[str], head: list[str], rows: list[list[str]], chunk: int = 0) -> list[str]:
    """Plain tabular(s). With `chunk`, the rows are split into consecutive tabulars so the table can break across a
    page without the longtable package."""
    spec = "".join(f"p{{{w}\\linewidth}}" for w in widths)
    cell = lambda x: r"\raggedright " + tex(x)
    head_row = " & ".join(r"\raggedright\textbf{" + tex(h) + "}" for h in head) + r" \tabularnewline"
    parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)] if chunk else [rows]
    W = [r"{\footnotesize", r"\setlength{\tabcolsep}{3pt}"]
    for k, part in enumerate(parts):
        W += [r"\noindent" + rf"\begin{{tabular}}{{{spec}}}", r"\hline"]
        if k == 0:
            W += [head_row, r"\hline"]
        W += [" & ".join(cell(x) for x in r) + r" \tabularnewline[1pt]" for r in part]
        W += [r"\hline" if k == len(parts) - 1 else "", r"\end{tabular}"]
        W += [r"\par\nointerlineskip" if k < len(parts) - 1 else ""]
    return W + ["}"]


def tex_verbatim(lines: list[str]) -> list[str]:
    for x in lines:
        assert all(ord(ch) < 128 for ch in x), x
    return [r"{\footnotesize", r"\begin{verbatim}"] + lines + [r"\end{verbatim}", "}"]


def render_tex(c: dict) -> str:
    t = c["text"]
    W = ["% Data overview: generated by `python -m src.analysis.professor_data_overview` from the data-lineage outputs.",
         "% Paste before Section 1. Unnumbered, so the existing section numbers do not change. Standard LaTeX only.",
         r"\section*{Data overview}", r"\addcontentsline{toc}{section}{Data overview}", "",
         r"\subsection*{Data sources}", "", r"\noindent " + tex(t["intro"]), "", r"\medskip"]
    W += tex_table(["0.14", "0.30", "0.09", "0.19", "0.20"],
                   ["Source", "What one row means", "Rows", "Fields we use", "What it gives us"], c["sources"])
    W += [""] + tex_verbatim(c["waterfall"]) + [""]
    W += [r"\begin{itemize}\setlength{\itemsep}{1pt}"] + [r"\item " + tex(x) for x in t["defs"]] + [r"\end{itemize}", ""]
    W += [r"\noindent " + tex(t["outside_intro"]), r"\begin{itemize}\setlength{\itemsep}{1pt}"]
    W += [r"\item " + tex(x) for x in t["outside"]]
    W += [r"\end{itemize}", "", r"\noindent " + tex(t["partial"]), "",
          r"\subsection*{How the variables in this report are built}", ""]
    W += tex_table(["0.12", "0.19", "0.39", "0.24"],
                   ["Report variable", "Raw / source columns", "Construction", "Important interpretation"], c["variables"],
                   chunk=2)
    W += [""] + tex_verbatim(c["diagram"]) + ["", r"\noindent\emph{" + tex(t["separate"]) + "}", "",
                                               "% \\clearpage  % uncomment to start Section 1 on a new page", ""]
    out = "\n".join(W)
    assert out.count(r"\begin{") == out.count(r"\end{")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pdf_consistency(d: dict) -> dict:
    """Counts this section shares with the current meeting PDF must be identical there (reported, not enforced)."""
    if not PDF.exists():
        return {}
    from pypdf import PdfReader
    text = " ".join(p.extract_text() for p in PdfReader(str(PDF)).pages)
    lo = d["N"]["loans"]
    return {f0(n): f0(n) in text for n in (lo["loans"], lo["realised"])}


def main() -> dict:
    d = load()
    c = content(d)
    MD.write_text(render_md(c))
    TEX.write_text(render_tex(c))
    chk = pdf_consistency(d)
    print(f"wrote {MD.name}, {TEX.name}; counts shared with the Week 6 PDF found there: {chk}")
    return chk


if __name__ == "__main__":
    main()
