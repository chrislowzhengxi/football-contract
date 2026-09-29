"""Loan pairing and the rows -> real-transfer-episodes waterfall.

Two layers: synthetic player histories that pin each pairing rule, and
invariants checked over the full frozen Stage 1C table.
"""
import hashlib

import pandas as pd
import pytest

from src.analysis.loan_episodes import (CANONICAL_CSV, ECONOMIC_ENDINGS, EXPECTED_ROWS,
                                        build_universe, club_geography, load_canonical,
                                        normalise_season_arg, pair_loans, prepare,
                                        raw_label_class, season_label)

# The frozen Stage 1C file. If this changes, Stage 1C was modified.
STAGE1C_SHA256 = "92f383d72a3ed59925c922f0d1b5f98528bc66ad3a9f49bb990867ed1d1d50dd"

CLUB = {1: "Atalanta", 2: "Torino", 3: "Rangers", 4: "Atalanta U19", 5: "Empoli",
        6: "Porto", 7: "FC Porto B", 8: "Gil Vicente", 9: "Without Club"}


def history(*moves, player_id=7):
    """moves: (date, from_id, to_id, type[, raw label[, fee_on_return]])."""
    rows = []
    for k, m in enumerate(moves):
        date, f, t, typ = m[:4]
        label = m[4] if len(m) > 4 else {"loan": "loan transfer", "loan_return": "End of loan",
                                         "permanent_transfer": "€5.00m",
                                         "free_transfer": "free transfer"}.get(typ, "-")
        rows.append({
            "event_id": f"e{k}", "transfermarkt_transfer_id": k, "player_id": player_id,
            "player_name": "Test Player", "transfer_date": date, "transfer_season": "23/24",
            "from_club_id": f, "to_club_id": t, "from_club_name": CLUB[f], "to_club_name": CLUB[t],
            "transfer_type_normalized": typ, "fee_display_raw": label,
            "fee_on_return_eur": m[5] if len(m) > 5 else None, "loan_fee_eur": None,
            "permanent_transfer_fee_eur": 5e6 if typ == "permanent_transfer" else None,
            "is_internal_move": typ == "youth_or_internal",
            "involves_placeholder_club": 9 in (f, t), "is_youth_or_reserve_side": False,
            "transfermarkt_future_transfer": False,
        })
    return prepare(pd.DataFrame(rows))


def ending(df):
    u = build_universe(df)
    return u.loans.set_index("loan_event_id")


# ---------------------------------------------------------------------------
# Pairing rules
# ---------------------------------------------------------------------------

def test_a_loan_and_its_return_are_one_episode():
    df = history(("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"))
    u = build_universe(df)
    assert len(u.episodes) == 1 and len(u.loans) == 1
    assert u.rows.row_role.tolist() == ["episode", "loan_return_collapsed"]
    e = u.loans.iloc[0]
    assert e.match_reason == "reversed_club_ids"
    assert e.economic_ending == "ordinary_end_of_loan"
    assert e.raw_label_class == "End of loan"


def test_return_then_permanent_move_to_the_borrower_is_a_conversion_pattern():
    """Zapata: Atalanta -> Torino loan, returned, bought by Torino a day later."""
    df = history(("2023-09-01", 1, 2, "loan"), ("2024-06-30", 2, 1, "loan_return"),
                 ("2024-07-01", 1, 2, "permanent_transfer"))
    e = ending(df).loc["e0"]
    assert e.economic_ending == "purchase_option_or_permanent_conversion"
    assert e.raw_label_class == "End of loan"          # the label alone says nothing


def test_a_sale_to_a_third_club_needs_a_consideration():
    paid = history(("2022-08-07", 1, 5, "loan"), ("2023-06-30", 5, 1, "loan_return"),
                   ("2023-07-01", 1, 3, "permanent_transfer"))
    free = history(("2022-08-07", 1, 5, "loan"), ("2023-06-30", 5, 1, "loan_return"),
                   ("2023-07-01", 1, 3, "free_transfer"))
    assert ending(paid).loc["e0"].economic_ending == "third_party_sale_related"
    assert ending(free).loc["e0"].economic_ending != "third_party_sale_related"


def test_sub_loans_are_nested_spells_not_endings():
    """Liverpool -> Atletico, Atletico -> Rayo, Rayo -> Atletico, Atletico -> Liverpool."""
    df = history(("2011-08-24", 1, 2, "loan"), ("2011-08-25", 2, 5, "loan"),
                 ("2012-06-29", 5, 2, "loan_return"), ("2012-06-30", 2, 1, "loan_return"))
    u = build_universe(df)
    L = u.loans.set_index("loan_event_id")
    assert L.loc["e0", "terminal_event_id"] == "e3"
    assert L.loc["e1", "terminal_event_id"] == "e2"
    assert (u.rows.row_role == "loan_return_unmatched").sum() == 0


def test_the_outer_loan_is_not_read_as_a_follow_on_after_a_sub_loan_returns():
    df = history(("2011-08-24", 1, 2, "loan"), ("2011-08-25", 2, 5, "loan"),
                 ("2012-06-29", 5, 2, "loan_return"), ("2012-06-30", 2, 1, "loan_return"))
    assert ending(df).loc["e1"].economic_ending == "ordinary_end_of_loan"


def test_internal_moves_at_the_lender_do_not_end_a_loan():
    """Flamengo U20 -> Flamengo registered while the player was at Boavista."""
    df = history(("2010-01-02", 1, 2, "loan"), ("2011-01-01", 4, 1, "youth_or_internal"),
                 ("2011-04-30", 2, 1, "loan_return"))
    e = ending(df).loc["e0"]
    assert e.terminal_event_id == "e2" and e.match_reason == "reversed_club_ids"


def test_a_return_to_another_side_of_the_lender_is_matched_and_flagged():
    df = history(("2011-07-01", 6, 8, "loan"), ("2012-06-30", 8, 7, "loan_return"))
    e = ending(df).loc["e0"]
    assert e.terminal_event_id == "e1"
    assert e.match_reason == "returned_from_borrower_to_other_side_of_lender"


def test_a_return_with_no_open_loan_is_left_unmatched_not_forced():
    df = history(("2020-06-30", 2, 1, "loan_return"))
    u = build_universe(df)
    assert u.rows.row_role.tolist() == ["loan_return_unmatched"]
    assert len(u.episodes) == 0


def test_one_return_cannot_close_two_loans():
    df = history(("2020-07-01", 1, 2, "loan"), ("2021-07-01", 1, 2, "loan"),
                 ("2022-06-30", 2, 1, "loan_return"))
    L = ending(df)
    assert L.loc["e0", "match_reason"] == "re_loaned_same_clubs"
    assert L.loc["e1", "terminal_event_id"] == "e2"
    assert (L.terminal_event_id == "e2").sum() == 1


def test_a_permanent_move_between_the_same_clubs_converts_the_loan():
    df = history(("2020-07-01", 1, 2, "loan"), ("2021-07-01", 1, 2, "permanent_transfer"))
    e = ending(df).loc["e0"]
    assert e.match_reason == "converted_to_permanent_same_clubs"
    assert e.economic_ending == "purchase_option_or_permanent_conversion"
    assert e.raw_label_class == "explicit fee amount"


def test_a_loan_with_nothing_after_it_is_open_not_ordinary():
    e = ending(history(("2025-08-01", 1, 2, "loan"))).loc["e0"]
    assert e.economic_ending == "unresolved"
    assert pd.isna(e.terminal_event_id)


def test_a_fee_on_the_return_is_recorded_not_interpreted():
    df = history(("2023-09-01", 1, 2, "loan"),
                 ("2024-06-30", 2, 1, "loan_return", "End of loan<br />€500k", 500_000.0))
    e = ending(df).loc["e0"]
    assert e.economic_ending == "fee_bearing_return"
    assert e.raw_label_class == "End of loan + fee"
    assert e.ending_fee_on_return_eur == 500_000


def test_a_mid_season_return_followed_immediately_by_a_move_is_early_termination():
    df = history(("2022-08-25", 1, 2, "loan"), ("2023-01-18", 2, 1, "loan_return"),
                 ("2023-01-30", 1, 5, "loan"))
    assert ending(df).loc["e0"].economic_ending == "early_termination"


def test_a_31_december_return_is_a_scheduled_boundary_not_early():
    df = history(("2025-01-07", 1, 2, "loan"), ("2025-12-31", 2, 1, "loan_return"),
                 ("2026-01-05", 1, 5, "loan"))
    assert ending(df).loc["e0"].economic_ending == "immediate_follow_on_transfer"


def test_placeholder_and_internal_rows_are_not_episodes():
    df = history(("2020-07-01", 9, 1, "free_transfer"), ("2021-07-01", 4, 1, "youth_or_internal"))
    u = build_universe(df)
    assert u.rows.row_role.tolist() == ["placeholder_non_club_movement", "internal_registration"]
    assert len(u.episodes) == 0


@pytest.mark.parametrize("label,expected", [
    ("End of loan", "End of loan"), ("End of loan<br />€1.00m", "End of loan + fee"),
    ("?", "? (undisclosed)"), ("-", "- (no fee shown)"), ("0", "0"), ("€2.00m", "explicit fee amount"),
    ("free transfer", "free transfer"), (None, "missing"), (float("nan"), "missing"),
])
def test_raw_labels_keep_their_distinctions(label, expected):
    assert raw_label_class(label) == expected


def test_seasons():
    assert season_label("02/03") == "2002/03" and season_label("99/00") == "1999/00"
    assert normalise_season_arg("21/22") == normalise_season_arg("2021-22") == "2021/22"


# ---------------------------------------------------------------------------
# Invariants over the real dataset
# ---------------------------------------------------------------------------

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def real():
    if not CANONICAL_CSV.exists():
        pytest.skip("Stage 1C not built")
    before = _sha(CANONICAL_CSV)
    df = load_canonical()
    u = build_universe(df, club_geography())
    return {"df": df, "u": u, "sha_before": before}


def test_stage1c_is_the_frozen_file_and_is_not_modified(real):
    assert real["sha_before"] == STAGE1C_SHA256
    assert len(real["df"]) == EXPECTED_ROWS
    assert _sha(CANONICAL_CSV) == STAGE1C_SHA256


def test_every_row_has_exactly_one_role_and_the_waterfall_reconciles(real):
    r = real["u"].rows
    assert r.row_role.notna().all()
    assert len(r) == EXPECTED_ROWS
    assert (r.row_role == "episode").sum() == len(real["u"].episodes)


def test_no_outbound_loan_is_counted_twice(real):
    u = real["u"]
    assert u.spells.loan_event_id.is_unique
    assert u.episodes.event_id.is_unique
    assert u.loans.loan_event_id.is_unique


def test_no_return_closes_more_than_one_loan(real):
    s = real["u"].spells
    closed_by_return = s[s.terminal_is_return]
    assert closed_by_return.terminal_event_id.is_unique


def test_no_ending_precedes_its_loan(real):
    s = real["u"].spells.dropna(subset=["terminal_date"])
    assert (s.terminal_date >= s.loan_date).all()


def test_endings_belong_to_the_same_player(real):
    df = real["df"].set_index("event_id")
    s = real["u"].spells.dropna(subset=["terminal_event_id"])
    assert (df.loc[s.terminal_event_id, "player_id"].values == s.player_id.values).all()


def test_exact_matches_reverse_the_clubs(real):
    df = real["df"].set_index("event_id")
    s = real["u"].spells
    exact = s[s.match_reason == "reversed_club_ids"]
    t = df.loc[exact.terminal_event_id]
    assert (t.from_club_id.values == exact.borrower_club_id.values).all()
    assert (t.to_club_id.values == exact.lender_club_id.values).all()


def test_every_return_leaves_the_borrowing_club(real):
    df = real["df"].set_index("event_id")
    s = real["u"].spells
    ret = s[s.terminal_is_return]
    t = df.loc[ret.terminal_event_id]
    same_id = t.from_club_id.values == ret.borrower_club_id.values
    same_org = t._from_org.values == df.loc[ret.loan_event_id, "_to_org"].values
    assert (same_id | same_org).all()


def test_collapsed_returns_match_the_pairing(real):
    u = real["u"]
    collapsed = set(u.rows[u.rows.row_role == "loan_return_collapsed"].event_id)
    closed = set(u.spells[u.spells.terminal_is_return].terminal_event_id)
    assert collapsed <= closed


def test_classes_are_from_the_declared_vocabulary(real):
    assert set(real["u"].loans.economic_ending) <= set(ECONOMIC_ENDINGS)


def test_fee_bearing_endings_carry_the_fee_label(real):
    L = real["u"].loans
    fee = L[L.ending_fee_on_return_eur.notna()]
    assert (fee.raw_label_class == "End of loan + fee").all()
    assert not L[L.raw_label_class == "End of loan"].ending_fee_on_return_eur.notna().any()


def test_no_club_is_placed_in_two_countries():
    geo = club_geography()
    assert geo.club_id.is_unique


def test_missing_values_are_not_turned_into_zero(real):
    L = real["u"].loans
    # Only rows that genuinely carry a fee have one; nothing was filled with 0.
    assert (L.ending_fee_on_return_eur.dropna() > 0).all()
    assert L.ending_fee_on_return_eur.isna().sum() == len(L) - L.ending_fee_on_return_eur.notna().sum()


# ---------------------------------------------------------------------------
# Scope estimator
# ---------------------------------------------------------------------------

GEO = pd.DataFrame({"club_id": [1, 2, 3, 5], "country": ["Italy", "Italy", "Scotland", "Italy"],
                    "confederation": ["UEFA"] * 4})


@pytest.fixture
def scoped(monkeypatch):
    """A tiny universe: a realised Italian loan (22/23), a loan with only a
    scheduled return (24/25), an open loan (24/25), and a loan to an unmapped
    club (Atalanta U19, 22/23)."""
    from src.analysis import loan_scope
    moves = [("2022-08-01", 1, 2, "loan"), ("2023-06-30", 2, 1, "loan_return"),
             ("2024-08-01", 1, 5, "loan"), ("2025-06-30", 5, 1, "loan_return"),
             ("2024-09-01", 2, 3, "loan"),
             ("2022-09-01", 5, 4, "loan"), ("2023-06-30", 4, 5, "loan_return")]
    frames = [history(*moves[i:j], player_id=pid)
              for pid, (i, j) in enumerate([(0, 2), (2, 4), (4, 5), (5, 7)], start=1)]
    for f, season in zip(frames, ["2022/23", "2024/25", "2024/25", "2022/23"]):
        f["season"] = season
        f["event_id"] = f.player_id.astype(str) + f.event_id
    frames[1].loc[1, "transfermarkt_future_transfer"] = True     # the scheduled return
    df = pd.concat(frames, ignore_index=True)
    u = build_universe(df, GEO)
    monkeypatch.setattr(loan_scope, "universe", lambda: u)
    return loan_scope


def test_scope_defaults_count_every_loan(scoped):
    r = scoped.scope()
    assert r["loan_episodes"] == 4 and r["loans_with_recorded_ending"] == 3
    assert r["unresolved_or_open_loans"] == 1


def test_open_loans_are_not_counted_as_non_plain_endings(scoped):
    assert scoped.scope()["raw_label_not_plain_end_of_loan"] == 0


def test_realised_only_drops_scheduled_endings_but_keeps_open_loans(scoped):
    r = scoped.scope(realised_only=True)
    assert r["loan_episodes"] == 3
    assert r["unresolved_or_open_loans"] == 1


def test_through_season_stops_at_the_season(scoped):
    assert scoped.scope(through_season="2023/24")["loan_episodes"] == 2
    assert scoped.scope(through_season="24/25")["loan_episodes"] == 4


def test_country_modes_never_guess_an_unmapped_club(scoped):
    # Empoli -> Atalanta U19: the U19 side is unmapped, so `both` excludes it.
    assert scoped.scope(countries=["Italy"], country_mode="both")["loan_episodes"] == 2
    # `either` admits it (one mapped Italian club) and the Torino -> Rangers loan.
    assert scoped.scope(countries=["Italy"], country_mode="either")["loan_episodes"] == 4


def test_new_filters_leave_existing_real_counts_unchanged(real, monkeypatch):
    from src.analysis import loan_scope
    monkeypatch.setattr(loan_scope, "universe", lambda: real["u"])
    kw = dict(countries=["Italy", "England", "Spain"], seasons=["2021/22", "2022/23", "2023/24"])
    assert loan_scope.scope(**kw, country_mode="both")["loan_episodes"] == 1162
    assert loan_scope.scope(**kw, country_mode="either")["loan_episodes"] == 2408
    assert loan_scope.scope()["loan_episodes"] == 29699
    assert loan_scope.scope()["raw_label_not_plain_end_of_loan"] == 94
