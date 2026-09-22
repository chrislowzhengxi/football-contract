"""Event-family classification across the whole Stage 1C population.

The layer was written for loan-return anchors. Pointed at the wider research
population it labelled a standalone permanent transfer `likely_negotiated_return`
and called it a "return" in the notes, so a row could say `permanent_transfer`
in one column and `[loan_return]` in its family summary. These tests pin the
behaviour for each anchor type.
"""
import pandas as pd
import pytest

from src.stage2.event_family import (
    FAMILY_CLASSES, build_families, families_to_rows,
)


def _row(eid, pid, date, typ, frm, to, frm_id, to_id, **kw):
    r = {"event_id": eid, "player_id": pid, "player_name": f"Player {pid}",
         "transfer_date": date, "transfer_type_normalized": typ,
         "from_club_name": frm, "to_club_name": to,
         "from_club_id": frm_id, "to_club_id": to_id,
         "permanent_transfer_fee_eur": None, "loan_fee_eur": None,
         "fee_on_return_eur": None, "is_internal_move": False}
    r.update(kw)
    return r


def _fam(rows, anchor):
    df = pd.DataFrame(rows)
    fams = build_families(df, [anchor])
    return fams[anchor], df


def test_standalone_permanent_transfer_is_not_a_return():
    fam, _ = _fam([_row("e1", 1, "2018-08-08", "permanent_transfer",
                        "Sporting", "Lille", 10, 20)], "e1")
    assert fam.classification == "standalone_permanent_transfer"
    assert fam.roles["e1"] == "permanent_transfer"
    assert "return" not in fam.notes.lower()


def test_standalone_free_transfer_is_not_a_return():
    fam, _ = _fam([_row("e1", 1, "2015-07-01", "free_transfer",
                        "Guimaraes", "Porto", 10, 20)], "e1")
    assert fam.classification == "standalone_permanent_transfer"
    assert fam.roles["e1"] == "permanent_transfer"
    assert "return" not in fam.notes.lower()


def test_undisclosed_transfer_gets_a_permanent_role():
    fam, _ = _fam([_row("e1", 1, "2018-01-01", "undisclosed_transfer",
                        "Tijuana", "America", 10, 20)], "e1")
    assert fam.classification == "standalone_permanent_transfer"
    assert fam.roles["e1"] == "permanent_transfer"


def test_loan_with_no_return_yet_is_standalone_loan():
    fam, _ = _fam([_row("e1", 1, "2025-08-12", "loan", "Inter", "Cagliari", 10, 20)], "e1")
    assert fam.classification == "standalone_loan"
    assert fam.roles["e1"] == "original_loan"
    assert fam.review_required
    assert any("no return leg" in r for r in fam.review_reasons)


def test_ordinary_loan_and_return_is_a_loan_family():
    rows = [_row("e_loan", 1, "2015-08-12", "loan", "Man City", "Roma", 10, 20),
            _row("e_ret", 1, "2016-06-30", "loan_return", "Roma", "Man City", 20, 10)]
    fam, _ = _fam(rows, "e_loan")
    assert fam.classification == "loan_family"
    assert fam.roles["e_loan"] == "original_loan"
    assert fam.roles["e_ret"] == "loan_return"
    assert fam.events == ["e_loan", "e_ret"]


def test_return_anchor_keeps_its_original_classification():
    """Backward compatibility: a return anchor with no follow-on is unchanged."""
    rows = [_row("e_loan", 1, "2020-09-01", "loan", "Parent", "Host", 10, 20),
            _row("e_ret", 1, "2021-06-30", "loan_return", "Host", "Parent", 20, 10)]
    fam, _ = _fam(rows, "e_ret")
    assert fam.classification == "likely_negotiated_return"
    assert fam.roles["e_ret"] == "loan_return"
    assert fam.roles["e_loan"] == "original_loan"


def test_negotiated_return_then_sale_back_to_loan_club():
    rows = [_row("e_loan", 1, "2020-07-01", "loan", "Parent", "Host", 10, 20),
            _row("e_ret", 1, "2021-06-30", "loan_return", "Host", "Parent", 20, 10),
            _row("e_buy", 1, "2021-07-05", "permanent_transfer", "Parent", "Host", 10, 20)]
    fam, _ = _fam(rows, "e_ret")
    assert fam.classification == "likely_negotiated_return"
    assert fam.roles["e_buy"] == "permanent_transfer"


def test_third_party_sale_family():
    rows = [_row("e_loan", 1, "2020-07-01", "loan", "Parent", "Host", 10, 20),
            _row("e_ret", 1, "2021-06-30", "loan_return", "Host", "Parent", 20, 10),
            _row("e_sale", 1, "2021-07-08", "permanent_transfer", "Parent", "Third", 10, 30)]
    fam, _ = _fam(rows, "e_ret")
    assert fam.classification == "third_party_sale_related"
    assert fam.roles["e_ret"] == "administrative_return"
    assert fam.roles["e_sale"] == "third_party_sale"


def test_permanent_after_a_loan_at_the_same_club_is_a_loan_family():
    rows = [_row("e_loan", 1, "2022-07-01", "loan", "Parent", "Host", 10, 20),
            _row("e_perm", 1, "2023-06-20", "permanent_transfer", "Parent", "Host", 10, 20)]
    fam, _ = _fam(rows, "e_perm")
    assert fam.classification == "loan_family"
    assert fam.roles["e_perm"] == "permanent_transfer"
    assert fam.roles["e_loan"] == "original_loan"
    assert "return" not in fam.notes.lower()


def test_loan_anchor_followed_by_third_party_sale():
    rows = [_row("e_loan", 1, "2020-07-01", "loan", "Parent", "Host", 10, 20),
            _row("e_ret", 1, "2021-06-30", "loan_return", "Host", "Parent", 20, 10),
            _row("e_sale", 1, "2021-07-08", "permanent_transfer", "Parent", "Third", 10, 30)]
    fam, _ = _fam(rows, "e_loan")
    assert fam.classification == "third_party_sale_related"
    assert fam.roles["e_loan"] == "original_loan"


@pytest.mark.parametrize("typ,expected_role", [
    ("permanent_transfer", "permanent_transfer"),
    ("free_transfer", "permanent_transfer"),
    ("undisclosed_transfer", "permanent_transfer"),
    ("no_fee_shown", "permanent_transfer"),
    ("loan", "original_loan"),
    ("loan_return", "loan_return"),
    ("youth_or_internal", "internal_registration"),
])
def test_anchor_role_follows_stage1_type(typ, expected_role):
    fam, _ = _fam([_row("e1", 1, "2020-01-01", typ, "A", "B", 10, 20)], "e1")
    assert fam.roles["e1"] == expected_role


def test_every_classification_is_in_the_declared_vocabulary():
    shapes = [
        [_row("e1", 1, "2020-01-01", "permanent_transfer", "A", "B", 1, 2)],
        [_row("e1", 1, "2020-01-01", "loan", "A", "B", 1, 2)],
        [_row("e1", 1, "2020-01-01", "free_transfer", "A", "B", 1, 2)],
        [_row("e1", 1, "2020-01-01", "loan", "A", "B", 1, 2),
         _row("e2", 1, "2021-06-30", "loan_return", "B", "A", 2, 1)],
    ]
    for rows in shapes:
        for anchor in [r["event_id"] for r in rows]:
            fam, _ = _fam(rows, anchor)
            assert fam.classification in FAMILY_CLASSES, fam.classification


def test_family_summary_never_calls_a_permanent_transfer_a_return():
    """The contradiction that motivated this module's rewrite."""
    rows = [_row("e1", 1, "2018-08-08", "permanent_transfer", "Sporting", "Lille", 10, 20)]
    df = pd.DataFrame(rows)
    flat = families_to_rows(build_families(df, ["e1"]), df)
    r = flat.iloc[0]
    assert r.transfer_type_normalized == "permanent_transfer"
    assert r.event_role == "permanent_transfer"
    assert "loan_return" not in r.family_events
    assert "return" not in r.family_notes.lower()
    assert r.family_interpretation_status != "likely_negotiated_return"
