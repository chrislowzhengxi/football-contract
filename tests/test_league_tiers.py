"""Tests for the season-specific club -> league/tier lookup (src/analysis/league_tiers.py).

Unit tests use synthetic universes; table tests read data/processed/league_tiers/club_season_league.csv
(build it first with `python -m src.analysis.league_tiers --offline`)."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
import pytest

from src.analysis import league_tiers as lt


# ---------------------------------------------------------------- normalisation
def test_normalisation_folds_accents_case_punctuation():
    assert lt.normalise_name("Borussia Mönchengladbach") == "borussia monchengladbach"
    assert lt.normalise_name("Nott'm Forest") == "nottm forest"
    assert lt.normalise_name("Brighton & Hove Albion") == lt.normalise_name("Brighton and Hove Albion")
    assert lt.normalise_name("Bayern Munich") == lt.normalise_name("Bayern München")
    assert lt.normalise_name("Man Utd") == "man united"
    assert lt.normalise_name("Düsseldorf") == "dusseldorf"
    assert lt.normalise_name("Bodø/Glimt") == "bodo glimt"


def test_loose_key_drops_legal_form_and_short_numbers_only():
    assert lt.norm_keys("1. FSV Mainz 05")[1] == "mainz"
    assert lt.norm_keys("FC Schalke 04")[1] == "schalke"
    assert lt.norm_keys("Hannover 96")[1] == "hannover"
    # four-digit years are identity-bearing and kept
    assert lt.norm_keys("TSV 1860 München")[1] == "1860 munchen"
    # 'Real' is not a legal form
    assert lt.norm_keys("Real Sociedad")[1] == "real sociedad"


# ---------------------------------------------------------------- reserve / senior separation
@pytest.mark.parametrize("name,cls", [
    ("Borussia Dortmund II", "res"), ("FC Barcelona B", "res"), ("Jong Ajax", "res"), ("AZ Alkmaar J", "res"),
    ("Real Madrid Castilla", "res"), ("Barcelona Atlètic", "res"), ("Juventus Next Gen", "res"),
    ("Bilbao Athletic", "res"), ("Chelsea U21", "u21"), ("Go Ahead Eagles O19", "u19"), ("Arsenal U-23", "u23"),
    ("Charlton Athletic", None), ("Atlético Madrid", None), ("Club Atlético Huracán", None),
    ("Athletic Bilbao", None), ("Hannover 96", None), ("TSV 1860 München", None),
])
def test_reserve_class(name, cls):
    assert lt.reserve_class(name)[0] == cls


def test_reserve_and_senior_keys_never_equal():
    for senior, res in [("Bayern München", "Bayern München II"), ("Barcelona", "Barcelona B"),
                        ("Ajax", "Jong Ajax"), ("Chelsea", "Chelsea U21")]:
        assert lt.norm_keys(senior)[0] != lt.norm_keys(res)[0]
        assert lt.norm_keys(senior)[1] != lt.norm_keys(res)[1]


def _universe(rows, parent=None):
    """rows: (club_id, name, country)"""
    names = pd.DataFrame([(i, n) for i, n, _ in rows], columns=["club_id", "name"])
    countries = defaultdict(set)
    for i, _, c in rows:
        countries[i].add(c)
    return lt.index_universe(names, countries, parent)


def test_reserve_external_name_not_matched_to_senior_club():
    tm = _universe([(1, "Werder Bremen", "Germany")])
    r = lt.match_name("Werder Bremen II", "de.3", tm, {}, {})
    assert r["club_id"] is None and r["unmatched_reason"].startswith("reserve/senior mismatch")
    tm = _universe([(2, "Werder Bremen II", "Germany")])
    r = lt.match_name("Werder Bremen", "de.2", tm, {}, {})
    assert r["club_id"] is None


def test_reserve_matched_to_reserve_of_same_name():
    tm = _universe([(1, "FC Barcelona", "Spain"), (2, "FC Barcelona B", "Spain")])
    assert lt.match_name("Barcelona B", "es.2", tm, {}, {})["club_id"] == 2
    assert lt.match_name("FC Barcelona", "es.1", tm, {}, {})["club_id"] == 1


def test_reserve_derived_from_learned_senior_and_organisation_map():
    tm = _universe([(16, "Bor. Dortmund", "Germany"), (17, "B. Dortmund II", "Germany"),
                    (99, "Dortmund U19", "Germany")], parent={17: 16, 99: 16})
    learned = {("Germany", "Borussia Dortmund"): 16}
    r = lt.match_name("Borussia Dortmund II", "de.3", tm, {}, learned)
    assert r["club_id"] == 17 and r["match_method"] == "alias_learned"


# ---------------------------------------------------------------- ambiguity and country
def test_ambiguous_name_stays_unmatched():
    tm = _universe([(10, "Halifax Town", "England"), (11, "Halifax Town", "England")])
    r = lt.match_name("Halifax Town", "en.5", tm, {}, {})
    assert r["club_id"] is None and r["unmatched_reason"].startswith("ambiguous")
    assert r["candidates"] == "10 11"


def test_learned_alias_resolves_ambiguity_but_exact_conflict_is_flagged():
    tm = _universe([(10, "Roda JC", "Netherlands"), (11, "Roda JC Kerkrade", "Netherlands"),
                    (12, "Roda JC Kerkrade", "Netherlands")])
    r = lt.match_name("Roda JC Kerkrade", "nl.2", tm, {}, {("Netherlands", "Roda JC Kerkrade"): 11})
    assert r["club_id"] == 11
    tm = _universe([(20, "Twente Enschede", "Netherlands"), (21, "FC Twente", "Netherlands")])
    r = lt.match_name("FC Twente", "nl.1", tm, {}, {("Netherlands", "FC Twente"): 20})
    assert r["club_id"] is None and r["unmatched_reason"].startswith("conflict")


def test_match_restricted_to_league_country_family():
    tm = _universe([(1, "Arsenal", "Ukraine")])
    assert lt.match_name("Arsenal", "en.1", tm, {}, {})["club_id"] is None
    tm = _universe([(2, "Cardiff City", "Wales")])
    assert lt.match_name("Cardiff City", "en.2", tm, {}, {})["club_id"] == 2  # English pyramid includes Wales


def test_learn_aliases_by_elimination():
    tm = _universe([(1, "Man Utd", "England"), (2, "Arsenal FC", "England"), (3, "Spurs", "England")])
    ext = pd.DataFrame({"code": "en.1", "season": 2020, "member": True,
                        "external_team_name": ["Manchester United FC", "Arsenal", "Tottenham Hotspur"]})
    snap = pd.DataFrame({"competition_id": "GB1", "season": 2020, "member": True, "club_id": [1, 2, 3]})
    al, linked = lt.learn_aliases(ext, snap, tm)
    got = dict(zip(al.external_team_name, al.club_id))
    assert got["Arsenal"] == 2  # restricted exact
    # two names left (Manchester United FC, Tottenham Hotspur) vs two ids: no forced pairing
    assert "Tottenham Hotspur" not in got
    tm = _universe([(1, "Manchester United", "England"), (2, "Arsenal FC", "England"), (3, "Spurs", "England")])
    al, linked = lt.learn_aliases(ext, snap, tm)
    got = dict(zip(al.external_team_name, al.club_id))
    assert got["Tottenham Hotspur"] == 3 and linked.fully_paired.all()


# ---------------------------------------------------------------- tiers and seasons
def test_tier_mapping_table():
    assert {c: lt.LEAGUES[c]["tier"] for c in ("en.1", "en.2", "en.3", "en.4", "en.5")} == \
        {"en.1": 1, "en.2": 2, "en.3": 3, "en.4": 4, "en.5": 5}
    assert lt.LEAGUES["de.3"]["tier"] == 3 and lt.LEAGUES["de.4.west"]["tier"] == 4
    assert lt.LEAGUES["es.2"]["tier"] == 2 and lt.LEAGUES["it.3.a"]["tier"] == 3
    assert lt.LEAGUES["se.3s"]["tier"] == 3 and lt.LEAGUES["sco.1"]["tier"] == 1
    assert lt.league_name("sco.1", 2012) == "Scottish Premier League"
    assert lt.league_name("sco.1", 2013) == "Scottish Premiership"
    assert lt.league_name("en.5", 2015) == "National League"
    assert all(isinstance(v["tier"], int) and v["tier"] >= 1 for v in lt.LEAGUES.values())
    for comp, code in lt.SNAP_TO_CODE.items():
        assert lt.LEAGUES[code]["tier"] == 1


def test_season_labels():
    assert lt.season_from_label("2015-16") == 2015
    assert lt.season_from_label("2025") == 2024  # calendar-year league -> Transfermarkt season id


def test_txt_parser_formats():
    txt = """= X 2020/21
# Teams 4
▪ Matchday 1
Sat Aug 6
  15:00  Bolton Wanderers         1-0 (1-0)  Sheffield United
         SC Verl                  -  Türkgücü München         [cancelled]
  Fri Aug 28 2020
    18:45  FC Utrecht J            v FC Eindhoven             1-2 (0-1)
    21:00  1. FC Köln              v Bor. Mönchengladbach
"""
    got = lt.parse_txt_matches(txt)
    assert got == [("Bolton Wanderers", "Sheffield United"), ("SC Verl", "Türkgücü München"),
                   ("FC Utrecht J", "FC Eindhoven"), ("1. FC Köln", "Bor. Mönchengladbach")]


def test_playoff_only_teams_are_not_members():
    m = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B"), ("A", "Z")]
    t = lt.teams_from_matches(m).set_index("external_team_name")
    assert not t.loc["Z", "member"] and t.loc["A", "member"]


# ---------------------------------------------------------------- the built table
@pytest.fixture(scope="module")
def table():
    if not lt.TABLE_CSV.exists():
        pytest.skip("table not built")
    return pd.read_csv(lt.TABLE_CSV)


def test_no_duplicate_club_seasons(table):
    assert not table.duplicated(["club_id", "season"]).any()


def test_tiers_positive_ints(table):
    assert table.tier.dtype.kind == "i" and (table.tier >= 1).all()
    assert table.season.dtype.kind == "i"


def test_sources_and_methods_valid(table):
    assert set(table.source) <= {lt.SRC_SNAPSHOT, lt.SRC_OF}
    assert set(table.match_method) <= {"snapshot_games", "exact_normalised", "alias_manual", "alias_learned"}
    assert (table.n_sources_agreeing >= 1).all()


def test_club_ids_exist_in_master_or_snapshot(table):
    import duckdb
    con = duckdb.connect(str(lt.DUCKDB_PATH), read_only=True)
    snap = set(con.sql("select club_id::int from clubs union select home_club_id::int from games "
                       "union select away_club_id::int from games").df().iloc[:, 0].dropna().astype(int))
    master = lt.load_master()
    mids = set(pd.concat([master.from_club_id, master.to_club_id]).dropna().astype(int))
    assert set(table.club_id) <= (snap | mids)


def test_season_specific_promotion_relegation(table):
    t = table.set_index(["club_id", "season"]).tier
    # Sheffield United (TM 350): Championship 2018/19, promoted -> Premier League 2019/20
    assert t[(350, 2018)] == 2 and t[(350, 2019)] == 1
    # Hamburger SV (TM 41): relegated from the Bundesliga in 2017/18 -> 2. Bundesliga 2018/19
    assert t[(41, 2017)] == 1 and t[(41, 2018)] == 2


def test_conflicting_club_seasons_excluded(table):
    if lt.CONFLICTS_CSV.exists():
        c = pd.read_csv(lt.CONFLICTS_CSV)
        if len(c):
            keys = set(zip(c.club_id, c.season))
            assert not any(k in keys for k in zip(table.club_id, table.season))


def test_spot_checks(table):
    r = lt.run_spot_checks(table)
    assert len(r) >= 10
    assert r.ok.all(), r[~r.ok].to_string()


def test_txt_parser_agrees_with_json():
    v = lt.parser_validation()
    if v.empty:
        pytest.skip("validation files not cached")
    assert v.identical.all(), v[~v.identical].to_string()


def test_no_row_where_no_source(table):
    t = set(zip(table.club_id, table.season))
    # Rangers (TM 124) were in Scottish League Two / League One / Championship 2012/13-2015/16: no verified source
    assert not any((124, s) in t for s in range(2012, 2016))
