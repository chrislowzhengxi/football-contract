"""Country-mapping coverage audit: diagnostic only, no country from a name alone, counts reconcile."""
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import loan_calendar_rule as lc
from src.analysis import loan_country_mapping_audit as cm
from src.stage1c.policy import YOUTH_OR_RESERVE

from tests.test_loan_episodes import STAGE1C_SHA256, _sha, real  # noqa: F401  (fixture)


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

NAMES = ["Man City U18", "Real Madrid Castilla", "Bayern Munich II", "Ajax Yth.", "Benfica B", "Barcelona Atlètic",
         "B. Banja Luka", "Sub-20", "Ajax", "Res. Inter", "Juventus U23", "Sporting CP C", "Flamengo Sub-20"]


def test_youth_and_reserve_halves_partition_the_stage1c_pattern():
    for n in NAMES:
        assert bool(cm.YOUTH.search(n) or cm.RESERVE.search(n)) == bool(YOUTH_OR_RESERVE.search(n)), n
    assert cm.category("Man City U18") == cm.CATEGORIES[1]
    assert cm.category("Benfica B") == cm.CATEGORIES[2]
    assert cm.category("B. Banja Luka") == cm.CATEGORIES[3]
    assert cm.category("Without Club") == cm.CATEGORIES[0]


def _names(rows):
    return pd.DataFrame(rows, columns=["club_id", "name"]).assign(rows=1, first=pd.NaT, last=pd.NaT)


def test_name_links_need_one_country_and_a_recorded_move():
    names = _names([(1, "Alpha"), (2, "Alpha U19"), (3, "Alpha B"), (4, "Nacional"), (5, "Nacional"),
                    (6, "Nacional B"), (7, "Gamma"), (8, "Gamma")])
    geo = pd.DataFrame({"club_id": [1, 4, 5, 7], "country": ["X", "P", "U", "Z"]})
    rows = pd.DataFrame({"from_club_id": [2, 6], "to_club_id": [1, 4]})          # 2<->1 and 6<->4 exchanged a player
    L = cm.name_links(names, geo, rows, {2, 3, 6, 8}).set_index("club_id")
    assert L.loc[2, "name_rule"] == cm.PROPOSED_RULE and L.loc[2, "name_country"] == "X"
    assert L.loc[3, "name_rule"] == cm.REVIEW_RULES[0]                           # same name, never a move
    assert L.loc[6, "name_rule"] == "name shared across countries (ambiguous)"   # a move, but two countries
    assert L.loc[6, "name_country"] == ""
    assert L.loc[8, "name_rule"] == cm.REVIEW_RULES[1]                           # senior homonym: review only


def test_projection_only_fills_missing_countries_and_keeps_calendars_fixed():
    june = int(lc.doy(pd.Series(["06-30"]))[0])
    cal = {"domestic: X": {"centers": [june], "coverage": 1.0, "N": 100, "stopped": ""},
           "cross-border": {"centers": [june], "coverage": 1.0, "N": 100, "stopped": ""}}
    frozen = json.dumps(cal, sort_keys=True)
    R = pd.DataFrame({"loan_event_id": [1, 2, 3], "lender_club_id": [10, 11, 12], "borrower_club_id": [20, 21, 22],
                      "lender_country": ["X", None, "X"], "borrower_country": ["X", None, None],
                      "lender_confederation": None, "borrower_confederation": None,
                      "doy": [june] * 3, "covid": False})
    P = cm.project(R, cal, pd.Series({10: "Q", 11: "X", 21: "X", 22: "Y"}))
    assert P.lender_country.tolist() == ["X", "X", "X"]                          # club 10 already mapped: kept
    assert P.context.tolist() == ["domestic", "domestic", "cross-border"]
    assert P.calendar_class.tolist() == ["calendar_normal_domestic", "calendar_normal_domestic",
                                         "calendar_normal_cross_border"]
    assert json.dumps(cal, sort_keys=True) == frozen


def test_bounds_after_full_recovery():
    cal = {"domestic: X": {}}
    R = pd.DataFrame({"context": ["domestic", "cross-border", cm.UNMAPPED, cm.UNMAPPED, cm.UNMAPPED, "domestic"],
                      "lender_country": ["X", "X", "X", None, "Small", "Small"],
                      "borrower_country": ["X", "Y", None, None, None, "Small"],
                      "calendar_key": ["domestic: X", "cross-border", "", "", "", "domestic: Small"]})
    b = cm.bounds_after_full_recovery(R, cal)
    assert b["classified_now"] == 2 and b["guaranteed"] == 1
    assert b["both_missing"] == 1 and b["one_side_known_no_calendar"] == 1
    assert b["lower"] == 3 and b["upper"] == 5


def test_sample_selection_is_deterministic_and_never_repeats_a_club():
    rng = np.random.default_rng(0)
    n = 300
    C = pd.DataFrame({"club_id": np.arange(n), "primary_category": rng.choice(cm.CATEGORIES[1:], n),
                      "as_lender": rng.integers(0, 30, n), "as_borrower": rng.integers(0, 30, n),
                      "in_both_missing": rng.integers(0, 10, n), "n_names": rng.integers(1, 3, n),
                      "in_snapshot_clubs_table": rng.random(n) < .05,
                      "same_name_mapped_ids": np.where(rng.random(n) < .2, "1", "")})
    C["episodes_affected"] = C.as_lender + C.as_borrower
    a, b = cm.select_sample(C), cm.select_sample(C.sample(frac=1, random_state=1))
    assert a.equals(b) and a.club_id.is_unique
    assert len(a) == sum(k for *_, k in cm.SAMPLE_PLAN)


# ---------------------------------------------------------------------------
# Real data (needs the cached external probe; never makes a web request)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audit(real):  # noqa: F811
    if not cm.PROBE_JSON.exists():
        pytest.skip("external probe not cached; run the module once")
    probe_before = cm.PROBE_JSON.read_text()
    N, tb = cm.compute(real["u"])
    assert cm.PROBE_JSON.read_text() == probe_before or json.loads(probe_before).get("wikidata_error")
    return N, tb


def test_population_reconciles_with_the_calendar_rule(audit):
    N, tb = audit
    assert sum(N["sides"].values()) == N["context"][cm.UNMAPPED]
    assert N["unclassified"] == N["context"][cm.UNMAPPED] + N["below_threshold"]
    assert N["classifiable_now"] + N["unclassified"] == N["realised"]
    if lc.NUMBERS_JSON.exists():
        L = json.loads(lc.NUMBERS_JSON.read_text())
        assert L["population"]["unmapped"] == N["context"][cm.UNMAPPED]
        assert sum(L["unclassified"].values()) == N["unclassified"]
        assert L["unmapped_sides"] == N["sides"]


def test_club_table_reconciles(audit):
    N, tb = audit
    C = tb["C"]
    assert C.club_id.is_unique and len(C) == N["clubs"]["unique_ids"]
    assert C.episodes_affected.sum() == N["clubs"]["slots"]
    assert N["clubs"]["slots"] == N["sides"]["lender only missing"] + N["sides"]["borrower only missing"] \
        + 2 * N["sides"]["both missing"]
    assert sum(v["clubs"] for v in N["categories"].values()) == len(C)
    assert sum(N["clubs"]["endings_per_club"].values()) == len(C)
    from src.analysis.loan_episodes import club_geography
    assert not C.club_id.isin(club_geography().club_id).any()


def test_candidates_never_touch_mapped_clubs_and_respect_the_rules(audit):
    N, tb = audit
    CAND, C = tb["CAND"], tb["C"].set_index("club_id")
    from src.analysis.loan_episodes import club_geography
    assert not CAND.club_id.isin(club_geography().club_id).any()
    prop = CAND[CAND.status.eq("proposed")]
    assert prop.club_id.is_unique and set(prop.tier) <= {cm.PROPOSED_RULE, cm.MEMBERSHIP_RULE}
    assert not prop.proposed_country.str.contains(r"\|").any()
    par = prop[prop.tier.eq(cm.PROPOSED_RULE)]
    assert C.loc[par.club_id, "direct_moves_with_same_name_club"].gt(0).all()
    assert C.loc[par.club_id, "n_same_name_countries"].eq(1).all()
    assert C.loc[par.club_id, "has_marker"].all()
    assert (C.loc[par.club_id, "name_country"].values == par.proposed_country.values).all()
    mem = prop[prop.tier.eq(cm.MEMBERSHIP_RULE)]
    assert (C.loc[mem.club_id, "membership_country"].values == mem.proposed_country.values).all()
    for codes, country in zip(C.loc[mem.club_id, "orphan_competitions"], mem.proposed_country):
        implied = {N["orphan_codes"][c]["membership_country"] for c in codes.split(",")} - {""}
        assert implied == {country}                       # every code that gives a country gives this one
        for c in codes.split(","):
            assert N["orphan_codes"][c]["countries"] in ("", country) or N["orphan_codes"][c]["membership_country"] == ""
    assert len(prop) == N["internal_recoverable"]["clubs"]
    senior = CAND[CAND.tier.eq(cm.REVIEW_RULES[1])]
    assert senior.status.eq("review only").all()


def test_coverage_scenarios(audit):
    N, tb = audit
    COV = tb["COV"].set_index("scenario")
    cur = COV.loc["current mapping"]
    assert cur.classifiable == N["classifiable_now"] and cur.newly_classifiable == 0
    assert (COV.classifiable >= cur.classifiable).all()
    b = N["bounds"]
    assert b["lower"] <= b["upper"] == N["classifiable_now"] + N["context"][cm.UNMAPPED]
    assert b["guaranteed"] + b["depends_on_country"] == N["context"][cm.UNMAPPED]


def test_sample_audit_is_reproducible_and_uses_the_cached_probe_only(audit):
    N, tb = audit
    A = tb["A"]
    assert 20 <= len(A) <= 30 and A.club_id.is_unique
    assert A.club_id.tolist() == cm.select_sample(tb["C"]).club_id.tolist()
    P = json.loads(cm.PROBE_JSON.read_text())
    assert sorted(P["sample_club_ids"]) == sorted(A.club_id.tolist())
    assert len(P["players"]) <= cm.PLAYERS_PER_CLUB * len(A)
    assert set(A.recoverable_from_repo_yes_no) <= {"yes", "no"}
    yes = A[A.recoverable_from_repo_yes_no.eq("yes")]
    assert (yes.internal_country_evidence | yes.name_rule.eq(cm.PROPOSED_RULE)).all()


# ---------------------------------------------------------------------------
# Outputs and invariants
# ---------------------------------------------------------------------------

def test_outputs_exist_and_markdown_numbers_are_generated():
    if not cm.NUMBERS_JSON.exists():
        pytest.skip("module not run")
    N = json.loads(cm.NUMBERS_JSON.read_text())
    md = cm.AUDIT_MD.read_text()
    from src.analysis.daniel_scope_final import f0
    for v in (N["realised"], N["context"][cm.UNMAPPED], N["clubs"]["unique_ids"], N["below_threshold"],
              N["internal_recoverable"]["clubs"], N["classifiable_now"], N["internal_recoverable"]["endings_touched"]):
        assert f0(v) in md, v
    S = pd.read_csv(cm.SAMPLE_CSV)
    for col in ("club_id", "club_name", "episodes_affected", "current_country", "evidence_source",
                "recoverable_from_repo_yes_no", "proposed_country_if_deterministically_supported", "reason"):
        assert col in S.columns
    assert len(pd.read_csv(cm.CLUBS_CSV)) == N["clubs"]["unique_ids"]
    cov = pd.read_csv(cm.COVERAGE_CSV)
    assert cov.scenario.iloc[0] == "current mapping" and cov.classifiable.iloc[0] == N["classifiable_now"]


def test_outputs_are_new_files_and_stage1c_is_untouched():
    from src.analysis.loan_episodes import ROOT
    stage1c = ROOT / "data" / "outputs" / "rebuild" / "stage1c_canonical_transfers.csv"
    if stage1c.exists():
        assert _sha(stage1c) == STAGE1C_SHA256
    old = {p.name for p in (lc.SUMMARY_MD, lc.RULES_CSV, lc.CLASSIFIED_CSV, lc.EDGE_CSV, lc.SENS_CSV, lc.XB_CSV,
                            lc.NUMBERS_JSON)}
    assert not old & {p.name for p in cm.WRITES}
    assert all(p.name.startswith("loan_country_mapping_") for p in cm.WRITES)
