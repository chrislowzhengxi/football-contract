"""Club-organisation map, academy-promotion flags, their sensitivity, and the master's valid_when documentation."""
import json

import pandas as pd
import pytest

from src.analysis import club_organisation as co
from src.analysis import data_lineage as dl
from src.analysis.column_guide_specs import DOMAINS, VALID_WHEN

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


# --- name rule (pure) -------------------------------------------------------------------------------------------

def ev(child, parent):
    e = co.name_evidence(co.base_tokens(child), co.senior_tokens(parent))
    return e[0] if e else None


@pytest.mark.parametrize("child,parent,tier", [
    ("Schalke 04 U19", "FC Schalke 04", 1),
    ("Wolfsburg II", "VfL Wolfsburg", 1),
    ("SC Freiburg U19", "Freiburg", 1),
    ("Braga B", "SC Braga", 1),
    ("Stuttgart II", "VfB Stuttgart", 1),
    ("W. Bremen II", "Werder Bremen", 2),          # initial + exact
    ("B. Dortmund II", "Dortmund", 2),             # the side carries an extra initial
    ("Betis B", "Real Betis", 2),
    ("RM Castilla", "Real Madrid", 2),             # acronym
    ("Olymp. Lyon B", "Lyon", 2),
    ("Dundee Utd U20", "Dundee United", 2),        # abbreviation
    ("Rennes B", "Stade Rennais", 2),              # hand-entered alias
])
def test_names_that_agree(child, parent, tier):
    assert ev(child, parent) == tier


@pytest.mark.parametrize("child,parent", [
    ("Dynamo Kyiv II", "Arsenal Kyiv"),            # two organisations sharing a city
    ("CSKA Moscow II", "FC Moscow"),               # the side's extra word is another club's name
    ("Atl. Madrid B", "Real Madrid"),
    ("Celta Vigo B", "Real Betis"),
])
def test_names_that_do_not_agree(child, parent):
    assert ev(child, parent) is None


@pytest.mark.parametrize("name,marker", [
    ("B. Dortmund II", "stage1c"), ("Schalke 04 U19", "stage1c"), ("RM Castilla", "stage1c"),
    ("RCD Mallorca B ", "stage1c"),                # trailing space: Stage 1C's own pattern misses it
    ("Spartak 2", "v2"), ("Chornomorets-2", "v2"), ("Rennes Jugend", "v2"), ("Jong KVM", "v2"),
    ("CA Osasuna Yout", "v2"), ("Boavista Y19", "v2"),
    ("Dnipro-1", None), ("FC Schalke 04", None), ("1860 München", None), ("Real Madrid", None),
])
def test_side_markers(name, marker):
    assert co.side_marker(name) == marker


# --- map and master flags (need the outputs) --------------------------------------------------------------------

@pytest.fixture(scope="module")
def built():
    if not (co.MAP_CSV.exists() and dl.NUMBERS_JSON.exists()):
        pytest.skip("club organisation map / data lineage not run")
    MAP = co.load_map()
    M = pd.read_csv(dl.MASTER_CSV, low_memory=False)
    N = json.loads(dl.NUMBERS_JSON.read_text())
    return MAP, M, N


def test_stage1c_untouched():
    assert _sha(co.CANONICAL_CSV) == STAGE1C_SHA256 == co.STAGE1C_SHA256


def test_senior_guard_keeps_willem_ii_a_senior_club(built):
    MAP, M, _ = built
    K = MAP.set_index("child_club_id")
    assert K.loc[403, "confidence"] == "rejected_senior_club"          # Willem II
    assert 403 not in co.youth_side_ids(MAP)
    assert not MAP.parent_club_id.eq(403).any() or MAP.loc[MAP.parent_club_id.eq(403), "child_names"] \
        .str.contains("Willem II").all()                                 # only Willem II's own sides point to it
    w = M[M.from_club_id.eq(403) | M.to_club_id.eq(403)]
    own = w.from_club_name.str.startswith("Willem II") & w.to_club_name.eq("Willem II") & w.from_club_id.ne(403)
    assert not w.loc[~own, "academy_promotion_candidate"].any()


def test_dynamo_kyiv_ii_is_not_arsenal_kyiv(built):
    MAP, M, _ = built
    K = MAP.set_index("child_club_id")
    assert K.loc[21218, "parent_club_id"] == 338 and K.loc[21218, "confidence"] == "high"   # Dynamo Kyiv
    g = M[M.from_club_name.eq("Dynamo Kyiv II") & M.to_club_name.eq("Arsenal Kyiv")]
    assert len(g) and not g.academy_promotion_candidate.any() and not g.same_organisation_move_v2.any()
    assert g.same_org_confidence.eq("rejected_name_overlap_only").all()


@pytest.mark.parametrize("a,b", [
    ("B. Dortmund II", "Dortmund"), ("W. Bremen II", "Werder Bremen"), ("Betis B", "Real Betis"),
    ("SC Freiburg U19", "Freiburg"), ("Schalke 04 U19", "FC Schalke 04"), ("Wolfsburg II", "VfL Wolfsburg"),
    ("Braga B", "SC Braga"), ("Stuttgart II", "VfB Stuttgart"), ("RM Castilla", "Real Madrid"),
])
def test_named_promotions_are_high_confidence(built, a, b):
    _, M, _ = built
    g = M[M.from_club_name.eq(a) & M.to_club_name.eq(b)]
    assert len(g) and g.same_organisation_move_v2.all() and g.academy_promotion_candidate.all()


def test_flags_are_consistent(built):
    _, M, _ = built
    assert M.same_organisation_move_v2.eq(M.same_org_confidence.eq("high")).all()
    pc = M[M.academy_promotion_candidate]
    assert pc.from_club_parent_id_v2.astype("Int64").eq(pc.to_club_id.astype("Int64")).all()
    assert pc.same_org_confidence.isin(["high", "medium", "low"]).all()
    assert M.same_org_reason.notna().eq(M.same_org_confidence.ne("no_youth_or_reserve_side")).all()
    for c in ("in_cf_revised_sample", "in_cf_main_model_sample", "in_cf_injury_model_sample"):
        S = M[M[c]]
        assert not S.loc[S.kind.eq("loan"), "academy_promotion_candidate"].any()
        assert not (S.same_organisation_move_v2 & ~S.academy_promotion_candidate).any()   # B is inside C


def test_promotions_carry_transfermarkts_no_fee_label(built):
    _, M, _ = built
    S = M[M.in_cf_revised_sample & M.academy_promotion_candidate]
    assert S.transfer_type_raw.eq("no_fee_shown").all()


def test_sensitivity_arithmetic(built):
    _, M, N = built
    A = N["academy"]["variants"]
    cf = N["calendar_first"]["variants"]["classifiable"]
    assert A["A"]["totals"] == cf["totals"]
    assert A["A"]["estimates"]["m3"]["coef"] == cf["estimates"]["m3"]["coef"]
    for key, col in (("B", "same_organisation_move_v2"), ("C", "academy_promotion_candidate")):
        for s, f in (("revised", "in_cf_revised_sample"), ("main", "in_cf_main_model_sample"),
                     ("injury_model", "in_cf_injury_model_sample")):
            for k in ("loan", "permanent"):
                assert A[key]["removed"][s][k] == int((M[col] & M[f] & M.kind.eq(k)).sum())
            assert A[key]["totals"][s]["total"] == A["A"]["totals"][s]["total"] - A[key]["removed"][s]["total"]
        assert A[key]["ladder"]["M3"]["coef"] == pytest.approx(A[key]["estimates"]["m3"]["coef"], abs=1e-12)


def test_every_reviewed_row_has_a_verdict_and_no_false_positive(built):
    _, _, N = built
    R = N["academy"]["review"]
    S = pd.read_csv(dl.ACADEMY_REVIEW_CSV)
    assert R["rows"] == len(S) and S.manual_verdict.notna().all()
    assert set(S.event_id) == set(co.MANUAL_REVIEW)
    assert R["false_positives"] == []
    assert S.receiving_league.nunique() == 14


def test_candidate_file_lists_every_flagged_episode(built):
    _, M, _ = built
    C = pd.read_csv(dl.ACADEMY_CSV, low_memory=False)
    assert set(C.event_id) >= set(M.event_id[M.academy_promotion_candidate | M.same_org_confidence.ne(
        "no_youth_or_reserve_side")])
    assert C.loc[~C.pt_reliable, "true_zero"].isna().all()       # not interpretable there, so not shown


# --- valid_when -------------------------------------------------------------------------------------------------

def test_valid_when_documents_every_master_column(built):
    _, M, _ = built
    D = pd.read_csv(dl.VARDICT_CSV, keep_default_na=False)
    assert set(VALID_WHEN) == set(M.columns) == set(D.variable)
    assert D.valid_when.ne("").all()
    for r in D.itertuples():
        dom = DOMAINS[r.valid_domain][1](M).astype(bool)
        assert r.rows_valid == int(dom.sum())
        assert r.values_outside_valid == int((M[r.variable].notna() & ~dom).sum())
        assert (VALID_WHEN[r.variable].outside == "blank") == (r.values_outside_valid == 0), r.variable


def test_true_zero_outside_reliable_is_a_placeholder(built):
    _, M, _ = built
    out = M[~M.pt_reliable & M.kind.notna()]
    assert out.true_zero.eq(False).all() and len(out) > 0
    for c in ("minutes", "minutes_share", "appearances"):
        assert M.loc[~M.pt_reliable, c].isna().all()
    u = M[M.in_injury_usable]
    assert u.available_games_fit.gt(0).all() and u.fit_adjusted_share.notna().all()
