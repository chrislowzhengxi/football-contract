"""From registration movements to real transfer episodes.

Transfermarkt records registration movements. A loan appears twice - the
outbound loan and its "End of loan" row - but it is one economic arrangement.
This module turns the 175,165 Stage 1C rows into:

  * one role per row (internal registration, placeholder movement, collapsed
    loan return, unmatched return, or a real transfer episode), so the
    waterfall from rows to episodes is exact and nothing is silently dropped;
  * one record per loan spell, paired one-to-one with the movement that ended
    it, and classified two ways (raw Transfermarkt label, and an economic
    sequence pattern).

It is read-only with respect to Stage 1C and never asserts a contractual
mechanism. The economic classes describe what the registration sequence LOOKS
like; whether an option, obligation or buy-back sits behind it is a Stage 2
question that needs evidence.

Why not reuse `stage2.event_family.build_families` directly: that builder
anchors one family per target event and takes the first reversed return within
950 days without tracking which returns are already used, so across the whole
dataset one return could close several loans. The pairing here is one-to-one.
Its windows and tolerances are imported from the Stage 2 module so the ending
classes mean the same thing in both places.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..stage1c.policy import normalize_club_name
from ..stage2.event_family import (FOLLOW_ON_IMMEDIATE_DAYS, FOLLOW_ON_WINDOW_DAYS,
                                   MAX_LOAN_DAYS, SEASON_END_TOLERANCE_DAYS)

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CSV = ROOT / "data" / "outputs" / "rebuild" / "stage1c_canonical_transfers.csv"
EXPECTED_ROWS = 175_165

LOAN, RETURN = "loan", "loan_return"
PERMANENT_TYPES = ("permanent_transfer", "free_transfer", "undisclosed_transfer", "no_fee_shown")

# One role per Stage 1C row. The order is the priority order of the waterfall.
ROW_ROLES = (
    "internal_registration",          # same organisation on both sides (Stage 1C flag)
    "placeholder_non_club_movement",  # Without Club / Retired / Career break / Ban / Unknown
    "non_transfer_other",             # Transfermarkt "draft" label
    "loan_return_collapsed",          # a return paired with its outbound loan
    "loan_return_unmatched",          # a return with no outbound loan to close
    "episode",                        # a real transfer episode
)

FOLLOW_FEE_KIND = {"permanent_transfer": "paid", "undisclosed_transfer": "undisclosed fee",
                   "free_transfer": "free", "no_fee_shown": "no fee shown", "loan": "loan"}

# Fixed schema, so a history with no loans still yields a well-formed table.
SPELL_COLUMNS = [
    "loan_event_id", "player_id", "player_name", "loan_date", "loan_season",
    "lender_club_id", "lender_club", "borrower_club_id", "borrower_club",
    "loan_raw_label", "outbound_loan_fee_eur", "loan_is_internal", "loan_involves_placeholder",
    "loan_is_youth_or_reserve_side", "match_reason", "terminal_event_id", "terminal_date",
    "terminal_from_club", "terminal_to_club", "terminal_type", "terminal_raw_label",
    "terminal_is_return", "terminal_scheduled_future", "ending_fee_on_return_eur",
    "terminal_permanent_fee_eur", "duration_days", "raw_label_class", "has_matched_return",
    "follow_on_event_id", "follow_on_gap_days", "follow_on_type", "follow_on_to_club",
    "return_boundary_days", "economic_ending", "ending_detail",
]

ECONOMIC_ENDINGS = (
    "ordinary_end_of_loan", "fee_bearing_return", "early_termination",
    "purchase_option_or_permanent_conversion", "immediate_follow_on_transfer",
    "third_party_sale_related", "other_nonstandard", "unresolved",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_canonical(path: Path | str = CANONICAL_CSV) -> pd.DataFrame:
    """The frozen Stage 1C table, with derived helper columns added in memory."""
    return prepare(pd.read_csv(path, low_memory=False))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derived helper columns. Never written back to Stage 1C."""
    df = df.copy()
    df["_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    df["_from_org"] = df["from_club_name"].map(normalize_club_name)
    df["_to_org"] = df["to_club_name"].map(normalize_club_name)
    df["season"] = df["transfer_season"].map(season_label)
    return df


def season_label(raw) -> str:
    """Transfermarkt's '02/03' as '2002/03'. The data starts in 1993."""
    s = str(raw)
    if len(s) != 5 or s[2] != "/":
        return s
    yy = int(s[:2])
    return f"{1900 + yy if yy >= 90 else 2000 + yy}/{s[3:]}"


def normalise_season_arg(s: str) -> str:
    """Accept '2021/22', '21/22' or '2021-22'."""
    s = s.strip().replace("-", "/")
    if len(s) == 5:
        return season_label(s)
    if len(s) == 7 and s[4] == "/":
        return s
    raise ValueError(f"unrecognised season {s!r}; use e.g. 2021/22")


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

@dataclass
class _Spell:
    idx: int                 # row index of the outbound loan
    lender_id: int
    borrower_id: int
    lender_org: str
    borrower_org: str
    date: pd.Timestamp
    first_unexplained: int | None = None   # a movement that neither ends nor fits the spell


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    # Within a day, a return closes a spell before anything else opens one;
    # the Transfermarkt transfer id breaks any remaining tie deterministically.
    pri = (df["transfer_type_normalized"] != RETURN).astype(int)
    return df.assign(_pri=pri).sort_values(
        ["player_id", "_date", "_pri", "transfermarkt_transfer_id"])


def pair_loans(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair every outbound loan with the movement that ended it, one-to-one.

    Walks each player's movements in date order, keeping a stack of open loan
    spells. The rules, in words:

      * A return FROM club X closes the open spell whose borrower is X. The
        best candidate is the one whose lender is also the return's
        destination (exact club ids, then same organisation); failing that,
        a spell whose borrower matches is closed and flagged, because the
        return went to a different side of the lender (Porto B vs Porto).
      * A loan FROM the current borrower is a sub-loan: the outer spell stays
        open (Liverpool -> Atletico, Atletico -> Rayo, Rayo -> Atletico,
        Atletico -> Liverpool is two spells and two returns).
      * A loan between the same two clubs as an open spell renews it: the
        earlier spell ends there.
      * A permanent move between the same two clubs converts the spell.
      * A non-loan move out of the borrower ends the spell without a return.
      * A move out of the lender to a third club ends the spell (sold while
        on loan).
      * Internal registration changes never end a spell.
      * Anything else is recorded as unexplained; if no return follows the
        spell ends there as `next_movement_inconsistent`.

    Returns (spells, row_matches): one row per outbound loan, and a mapping
    from return rows to the loan they closed (or none).
    """
    d = _sorted(df)
    cols = d[["player_id", "transfer_type_normalized", "from_club_id", "to_club_id",
              "_from_org", "_to_org", "_date", "is_internal_move"]]
    spells_out, returns_out = [], []

    for _, g in cols.groupby("player_id", sort=False):
        open_: list[_Spell] = []
        for idx, t, fid, tid, forg, torg, dt, internal in (
                g.drop(columns="player_id").itertuples(index=True, name=None)):
            if t == RETURN:
                best, rank = None, 9
                for s in reversed(open_):                      # innermost first
                    if s.borrower_id == fid and s.lender_id == tid:
                        r = 1
                    elif s.borrower_org == forg and s.lender_org == torg:
                        r = 2
                    elif s.borrower_id == fid or s.borrower_org == forg:
                        r = 3
                    else:
                        continue
                    if (dt - s.date).days > MAX_LOAN_DAYS and s.first_unexplained is not None:
                        continue   # too far past an unexplained move to trust
                    if r < rank:
                        best, rank = s, r
                if best is None:
                    returns_out.append((idx, None, "no_open_loan_from_this_club"))
                    continue
                reason = {1: "reversed_club_ids", 2: "reversed_same_organisation",
                          3: "returned_from_borrower_to_other_side_of_lender"}[rank]
                spells_out.append((best.idx, idx, reason, best.first_unexplained))
                returns_out.append((idx, best.idx, reason))
                open_.remove(best)
                continue

            if internal and t != LOAN:
                continue                                        # never ends a spell

            if t == LOAN:
                if any(s.borrower_id == fid or s.borrower_org == forg for s in open_):
                    pass                                        # sub-loan; outer stays open
                else:
                    for s in list(open_):
                        if s.lender_id == fid and s.borrower_id == tid:
                            spells_out.append((s.idx, idx, "re_loaned_same_clubs", s.first_unexplained))
                            open_.remove(s)
                        elif s.lender_id == fid or s.lender_org == forg:
                            spells_out.append((s.idx, idx, "lender_loaned_player_elsewhere", s.first_unexplained))
                            open_.remove(s)
                        elif s.first_unexplained is None:
                            s.first_unexplained = idx
                open_.append(_Spell(idx, fid, tid, forg, torg, dt))
                continue

            for s in list(open_):
                if s.lender_id == fid and s.borrower_id == tid:
                    reason = "converted_to_permanent_same_clubs"
                elif s.borrower_id == fid or s.borrower_org == forg:
                    reason = "left_borrower_without_recorded_return"
                elif s.lender_id == fid or s.lender_org == forg:
                    reason = "lender_transferred_player_while_on_loan"
                else:
                    if s.first_unexplained is None:
                        s.first_unexplained = idx
                    continue
                spells_out.append((s.idx, idx, reason, s.first_unexplained))
                open_.remove(s)

        for s in open_:
            if s.first_unexplained is not None:
                spells_out.append((s.idx, s.first_unexplained, "next_movement_inconsistent", None))
            else:
                spells_out.append((s.idx, None, "open_no_later_movement", None))

    spells = pd.DataFrame(spells_out, columns=["loan_idx", "terminal_idx", "match_reason",
                                               "unexplained_idx"]).set_index("loan_idx")
    returns = pd.DataFrame(returns_out, columns=["return_idx", "loan_idx", "match_reason"]
                           ).set_index("return_idx")
    return spells, returns


# ---------------------------------------------------------------------------
# Ending classification
# ---------------------------------------------------------------------------

def _boundary_days(ts: pd.Timestamp) -> int:
    """Days to the nearest 30 June or 31 December.

    Stage 2 measured distance to 30 June only. 31 December is added here
    because the dataset includes calendar-year leagues (Brazil, Argentina,
    MLS, Scandinavia) and European half-season loans, whose scheduled ends
    fall there; the sensitivity of this choice is reported.
    """
    best = 10**6
    for y in (ts.year - 1, ts.year, ts.year + 1):
        for m, dd in ((6, 30), (12, 31)):
            best = min(best, abs((ts - pd.Timestamp(year=y, month=m, day=dd)).days))
    return best


def raw_label_class(label) -> str:
    """Transfermarkt's own fee cell on the ending row, with distinctions kept."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return "missing"
    s = str(label).strip()
    if s == "End of loan":
        return "End of loan"
    if s.startswith("End of loan"):
        return "End of loan + fee"
    if s == "free transfer":
        return "free transfer"
    if s == "loan transfer":
        return "loan transfer"
    if s.startswith("Loan fee"):
        return "Loan fee: amount"
    if s == "?":
        return "? (undisclosed)"
    if s == "-":
        return "- (no fee shown)"
    if s in ("0", "€0"):
        return "0"
    if s.startswith("€") or s[:1].isdigit():
        return "explicit fee amount"
    if s == "draft":
        return "draft"
    return f"other text: {s[:30]}"


def classify_spells(df: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """Attach raw-label and economic ending classes to every loan spell."""
    d = _sorted(df)
    by_player = {pid: g.index.to_list() for pid, g in d.groupby("player_id", sort=False)}
    pos = {i: k for idxs in by_player.values() for k, i in enumerate(idxs)}

    rows = []
    for loan_idx, sp in spells.iterrows():
        L = df.loc[loan_idx]
        term = sp.terminal_idx
        T = df.loc[int(term)] if pd.notna(term) else None
        reason = sp.match_reason
        is_return = T is not None and T.transfer_type_normalized == RETURN
        rec = {
            "loan_event_id": L.event_id, "player_id": L.player_id, "player_name": L.player_name,
            "loan_date": L["_date"], "loan_season": L.season,
            "lender_club_id": L.from_club_id, "lender_club": L.from_club_name,
            "borrower_club_id": L.to_club_id, "borrower_club": L.to_club_name,
            "loan_raw_label": L.fee_display_raw, "outbound_loan_fee_eur": L.loan_fee_eur,
            "loan_is_internal": bool(L.is_internal_move),
            "loan_involves_placeholder": bool(L.involves_placeholder_club),
            "loan_is_youth_or_reserve_side": bool(L.is_youth_or_reserve_side),
            "match_reason": reason,
            "terminal_event_id": T.event_id if T is not None else None,
            "terminal_date": T["_date"] if T is not None else pd.NaT,
            "terminal_from_club": T.from_club_name if T is not None else None,
            "terminal_to_club": T.to_club_name if T is not None else None,
            "terminal_type": T.transfer_type_normalized if T is not None else None,
            "terminal_raw_label": T.fee_display_raw if T is not None else None,
            "terminal_is_return": is_return,
            "terminal_scheduled_future": bool(T.transfermarkt_future_transfer) if T is not None else False,
            "ending_fee_on_return_eur": T.fee_on_return_eur if is_return else None,
            "terminal_permanent_fee_eur": T.permanent_transfer_fee_eur if T is not None else None,
        }
        rec["duration_days"] = (rec["terminal_date"] - L["_date"]).days if T is not None else None
        rec["raw_label_class"] = raw_label_class(rec["terminal_raw_label"]) if T is not None else "no ending recorded"
        rec["has_matched_return"] = is_return

        # Follow-on: the player's next substantive movement after a return.
        follow = None
        if is_return:
            idxs = by_player[L.player_id]
            for j in idxs[pos[int(term)] + 1:]:
                F = df.loc[j]
                # Internal registrations, and the unwinding of an outer loan
                # after a sub-loan returns, are not new deals.
                if (F.is_internal_move or F.transfer_type_normalized
                        in ("youth_or_internal", RETURN, "other")):
                    continue
                gap = (F["_date"] - T["_date"]).days
                if gap > FOLLOW_ON_WINDOW_DAYS:
                    break
                # A release to "Without Club" is not a transfer; it neither
                # makes the return a conduit nor starts a new deal.
                if F.involves_placeholder_club:
                    break
                follow = (F, gap)
                break
        rec["follow_on_event_id"] = follow[0].event_id if follow else None
        rec["follow_on_gap_days"] = follow[1] if follow else None
        rec["follow_on_type"] = follow[0].transfer_type_normalized if follow else None
        rec["follow_on_to_club"] = follow[0].to_club_name if follow else None
        rec["return_boundary_days"] = _boundary_days(T["_date"]) if is_return else None

        rec["economic_ending"], rec["ending_detail"] = _economic(rec, T, L, follow)
        rows.append(rec)
    return pd.DataFrame(rows, columns=SPELL_COLUMNS)


def _economic(rec, T, L, follow) -> tuple[str, str]:
    """Sequence pattern of the ending. Mirrors Stage 2's return-anchor order."""
    reason = rec["match_reason"]
    if reason == "open_no_later_movement":
        return "unresolved", "no later movement recorded (loan may still be running)"
    if reason == "next_movement_inconsistent":
        return "unresolved", "next movement does not involve the lender or borrower"
    if reason == "converted_to_permanent_same_clubs":
        return ("purchase_option_or_permanent_conversion",
                "permanent move lender->borrower with no return row in between")
    if reason == "lender_transferred_player_while_on_loan":
        return "third_party_sale_related", "lender transferred the player on while he was on loan"
    if reason in ("re_loaned_same_clubs", "lender_loaned_player_elsewhere",
                  "left_borrower_without_recorded_return"):
        return "other_nonstandard", reason.replace("_", " ")

    # A return row ended the spell.
    has_fee = pd.notna(rec["ending_fee_on_return_eur"])
    if follow is not None:
        F, gap = follow
        permanent = F.transfer_type_normalized in PERMANENT_TYPES
        fee_kind = FOLLOW_FEE_KIND.get(F.transfer_type_normalized, "n/a")
        if permanent and F.to_club_id == L.to_club_id:
            return ("purchase_option_or_permanent_conversion",
                    f"returned, then moved permanently back to the borrower {gap}d later ({fee_kind})")
        # A sale needs a consideration. "free transfer" and "-" to a third club
        # after a return is a departure - most often the parent contract
        # expiring with the loan - not the parent selling the player. Stage 2
        # counted all permanent types as a third-party sale; that is narrowed
        # here and the effect is reported.
        if permanent and fee_kind in ("paid", "undisclosed fee"):
            return ("third_party_sale_related",
                    f"returned, then sold to a third club {gap}d later ({fee_kind})")
        if gap <= FOLLOW_ON_IMMEDIATE_DAYS and rec["return_boundary_days"] > SEASON_END_TOLERANCE_DAYS:
            return ("early_termination",
                    f"return {rec['return_boundary_days']}d from 30 Jun/31 Dec, next move {gap}d later")
        if gap <= FOLLOW_ON_IMMEDIATE_DAYS or F.transfer_type_normalized == LOAN:
            return ("immediate_follow_on_transfer",
                    f"returned, then {F.transfer_type_normalized} {gap}d later")
    if has_fee:
        return "fee_bearing_return", "End of loan with a fee and no follow-on pattern"
    return "ordinary_end_of_loan", "plain End of loan with no follow-on pattern"


# ---------------------------------------------------------------------------
# Row roles and episodes
# ---------------------------------------------------------------------------

def assign_row_roles(df: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    role = pd.Series("episode", index=df.index)
    role[df.transfer_type_normalized == RETURN] = "loan_return_unmatched"
    matched = returns.dropna(subset=["loan_idx"]).index
    role[matched] = "loan_return_collapsed"
    role[df.transfer_type_normalized == "other"] = "non_transfer_other"
    role[df.involves_placeholder_club] = "placeholder_non_club_movement"
    role[df.is_internal_move] = "internal_registration"
    return role


def episode_type(t: str) -> str:
    return {"loan": "loan", "permanent_transfer": "permanent_paid",
            "free_transfer": "free_transfer", "undisclosed_transfer": "undisclosed_fee",
            "no_fee_shown": "no_fee_shown", "unknown": "unknown_type"}.get(t, t)


@dataclass
class Universe:
    rows: pd.DataFrame        # Stage 1C rows with `row_role`
    spells: pd.DataFrame      # every loan spell (including internal / placeholder)
    loans: pd.DataFrame       # loan spells that are real transfer episodes
    episodes: pd.DataFrame    # one row per real transfer episode
    returns: pd.DataFrame     # return row -> loan it closed


def build_universe(df: pd.DataFrame | None = None, geography: pd.DataFrame | None = None) -> Universe:
    df = load_canonical() if df is None else df
    spell_idx, returns = pair_loans(df)
    spells = classify_spells(df, spell_idx)
    role = assign_row_roles(df, returns)
    rows = df.assign(row_role=role)

    ep = rows[rows.row_role == "episode"].copy()
    ep["episode_type"] = ep.transfer_type_normalized.map(episode_type)
    ep = ep.merge(spells.drop(columns=["player_id", "player_name"]),
                  how="left", left_on="event_id", right_on="loan_event_id",
                  validate="one_to_one")
    # Non-loan episodes have no spell; their flags are False, not NaN.
    for col in ("terminal_is_return", "terminal_scheduled_future", "has_matched_return",
                "loan_is_internal", "loan_involves_placeholder", "loan_is_youth_or_reserve_side"):
        ep[col] = ep[col].astype("boolean").fillna(False).astype(bool)
    if geography is not None:
        ep = attach_geography(ep, geography)
        spells = attach_geography(spells, geography, "lender_club_id", "borrower_club_id")
    loans = ep[ep.episode_type == "loan"].copy()
    return Universe(rows, spells, loans, ep, returns)


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

def club_geography(db_path: Path | str | None = None) -> pd.DataFrame:
    """club_id -> country, confederation, from the upstream snapshot only.

    Two sources, both inside `dcaribou/transfermarkt-datasets`:
      1. `clubs.domestic_competition_id` -> `competitions.country_name`;
      2. participation in a covered country's domestic league, cup or super
         cup (`games` x `competitions`), which reaches lower-league clubs that
         met a covered club in a national cup.
    The two never disagree where both apply, and no club is placed in two
    countries. Clubs reached by neither are left unmapped, not guessed.
    """
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(db_path or DEFAULT_DATABASE), read_only=True)
    try:
        t1 = con.sql("""
            select cast(c.club_id as bigint) club_id, k.country_name country,
                   k.confederation, 'clubs_table_domestic_league' geo_source
            from clubs c join competitions k on c.domestic_competition_id = k.competition_id
            where k.country_name is not null""").df()
        t2 = con.sql("""
            select distinct cast(x.club_id as bigint) club_id, k.country_name country,
                   k.confederation, 'domestic_competition_participation' geo_source
            from games g join competitions k using (competition_id),
                 lateral (select unnest([g.home_club_id, g.away_club_id]) club_id) x
            where k.country_name is not null""").df()
    finally:
        con.close()
    t2 = t2[~t2.club_id.isin(t1.club_id)]
    geo = pd.concat([t1, t2], ignore_index=True)
    dup = geo.club_id.duplicated(keep=False)
    if dup.any():
        # A club placed in two countries is genuinely ambiguous; do not choose.
        geo = geo[~dup]
    geo["confederation"] = geo.confederation.map(
        {"europa": "UEFA", "amerika": "CONMEBOL/CONCACAF", "asien": "AFC",
         "afrika": "CAF"}).fillna(geo.confederation)
    return geo.reset_index(drop=True)


def attach_geography(df: pd.DataFrame, geo: pd.DataFrame,
                     from_col: str = "from_club_id", to_col: str = "to_club_id") -> pd.DataFrame:
    g = geo.set_index("club_id")
    out = df.copy()
    for side, col in (("from", from_col), ("to", to_col)):
        out[f"{side}_country"] = out[col].map(g.country)
        out[f"{side}_confederation"] = out[col].map(g.confederation)
    return out
