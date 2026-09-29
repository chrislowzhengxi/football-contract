"""Final dataset-scope outputs: the squad-coverage audit, the scope cube,
the token estimator, and stale provenance wording in generated documents."""
import re

import pandas as pd
import pytest

from src.analysis import loan_scope
from src.analysis.daniel_scope_final import (FINAL_MD, SQUAD_CSV, cube_count, scope_cube, wilson)
from src.analysis.stage1_scope_report import OUT

from tests.test_loan_episodes import real, scoped  # noqa: F401  (fixtures)


@pytest.fixture(scope="module")
def squads():
    if not SQUAD_CSV.exists():
        pytest.skip("run `python -m src.analysis.provenance_audit --upstream` first")
    return pd.read_csv(SQUAD_CSV)


def test_every_club_squad_season_is_audited(squads):
    assert squads.squad_season.tolist() == list(range(2012, 2026))


def test_transfer_files_exist_only_for_the_selection_seasons(squads):
    assert squads.loc[squads.transfers_json_exists, "squad_season"].tolist() == [2023, 2024, 2025]
    no_file = squads[~squads.transfers_json_exists]
    assert (no_file.squad_players_in_this_seasons_transfers_json == 0).all()
    assert (no_file.history_fetched_in_this_season == 0).all()


def test_squad_coverage_rows_add_up(squads):
    s = squads
    assert (s.club_squad_players + s.national_team_only_players == s.squad_players).all()
    assert (s.with_transfer_history + s.without_transfer_history == s.squad_players).all()
    assert (s.history_fetched_in_this_season + s.history_only_via_other_season_fetch
            == s.with_transfer_history).all()
    assert (s.of_which_via_later_season_fetch + s.of_which_via_earlier_season_fetch_only
            == s.history_only_via_other_season_fetch).all()
    assert (s.club_squad_with_history + s.national_team_only_with_history == s.with_transfer_history).all()


def test_before_the_selection_window_every_history_comes_from_a_later_fetch(squads):
    early = squads[squads.squad_season < 2023]
    assert (early.of_which_via_later_season_fetch == early.with_transfer_history).all()


# ---------------------------------------------------------------------------

def test_cube_reproduces_scope_on_a_small_universe(scoped):
    cube = scope_cube(scoped.universe().loans)
    for mode in ("both", "either"):
        want = scoped.scope(countries=["Italy"], country_mode=mode)["loan_episodes"]
        assert cube_count(cube, ["Italy"], set(cube.loan_season), mode) == want


def test_cube_covers_every_loan_and_reproduces_scope(real, monkeypatch):  # noqa: F811
    u = real["u"]
    monkeypatch.setattr(loan_scope, "universe", lambda: u)
    cube = scope_cube(u.loans)
    assert cube.loans.sum() == len(u.loans) == 29699
    for countries, seasons in ((["Italy", "England", "Spain"], ["2021/22", "2022/23", "2023/24"]),
                               (["Portugal", "Netherlands"], ["2018/19"])):
        for mode in ("both", "either"):
            r = loan_scope.scope(countries=countries, seasons=seasons, country_mode=mode)
            for col, key in (("loans", "loan_episodes"), ("fee_bearing_returns", "fee_bearing_returns"),
                             ("not_plain_end_of_loan", "raw_label_not_plain_end_of_loan")):
                assert cube_count(cube, countries, seasons, mode, col) == r[key], (countries, mode, col)


def test_token_estimate_is_the_stated_formula():
    scen = loan_scope.scenarios()
    if not scen:
        pytest.skip("no extraction cache")
    tok, cpf = scen["BASE"]
    assert loan_scope.extraction_tokens(1000, 0.5, "BASE") == round(1000 * 0.5 * cpf * tok)
    assert loan_scope.extraction_tokens(0, 0.9, "HIGH") == 0


def test_researched_cases_are_counted_from_the_log_not_typed_in():
    tp = loan_scope.token_profile()
    if not tp.get("calls") or not loan_scope.RESEARCHED_CASES_LOG.exists():
        pytest.skip("no extraction cache")
    assert tp["researched_cases"] == pd.read_csv(loan_scope.RESEARCHED_CASES_LOG).event_id.nunique()
    assert tp["researched_cases_reaching_extraction"] <= tp["researched_cases"]


def test_wilson_interval():
    lo, hi = wilson(24, 38)
    assert round(100 * lo) == 47 and round(100 * hi) == 77


# ---------------------------------------------------------------------------

STALE = [r"between 2012/13 and 2025/26", r"appeared in a competition", r"survivor samples",
         r"reached a covered league", r"captured (on )?2026-06-12", r"^- \*\*31 first-tier domestic leagues\*\*:"]


@pytest.mark.parametrize("name", ["stage1_scope_summary.md", FINAL_MD.name])
def test_generated_documents_carry_no_stale_provenance_claims(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    text = path.read_text()
    hits = [p for p in STALE if re.search(p, text, flags=re.M)]
    assert not hits, hits
