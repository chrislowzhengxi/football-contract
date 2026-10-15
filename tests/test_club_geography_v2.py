"""Geography v2: raw club flags, precedence, provenance, coverage, cached and deterministic rebuilds."""
import json
import urllib.request

import pandas as pd
import pytest

from src.analysis import club_geography_v2 as g2
from src.analysis.loan_country_mapping_audit import PROPOSED_RULE
from src.stage1b import club_flags, raw_api
from src.stage1c.policy import YOUTH_OR_RESERVE

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


# ---------------------------------------------------------------------------
# Raw flag parsing (synthetic)
# ---------------------------------------------------------------------------

def side(cid, name, flag=None, special=False):
    return {"href": f"/x/transfers/verein/{cid}/saison_id/2024", "clubName": name, "isSpecial": special,
            "countryFlag": f"https://tmssl.akamaized.net//images/flagge/verysmall/{flag}.png?lm=1" if flag else ""}


def transfer(tid, date, f, t):
    return {"url": f"/p/transfers/spieler/1/transfer_id/{tid}", "dateUnformatted": date, "from": f, "to": t}


def test_side_rows_and_aggregate_statuses():
    r24 = {"transfers": [transfer(1, "2019-07-01", side(10, "Alpha", 189), side(20, "Beta", 191)),
                         transfer(2, "2020-07-01", side(20, "Beta", 189), side(30, "Gamma", 999)),
                         transfer(3, "2021-07-01", side(30, "Gamma", 999), side(515, "Without Club", special=True)),
                         transfer(4, "2022-07-01", side(40, "Delta"), side(10, "Alpha", 189))]}
    r25 = {"transfers": [transfer(1, "2019-07-01", side(10, "Alpha", 189), side(20, "Beta", 191))]}  # same row again
    S = pd.DataFrame(club_flags.side_rows(r24, 1, "2024") + club_flags.side_rows(r25, 1, "2025"))
    S["preferred"] = S.season_file.eq("2025")
    A = club_flags.aggregate(S, {189: "England", 191: "Wales"}).set_index("club_id")
    assert A.loc[10, "mapping_status"] == "single_flag" and A.loc[10, "country_name"] == "England"
    assert A.loc[10, "n_supporting_rows"] == 2                     # transfer 1 counted once across two files
    assert A.loc[20, "mapping_status"] == "conflicting_flags" and A.loc[20, "flag_ids"] == "189|191"
    assert A.loc[30, "mapping_status"] == "single_flag_not_in_snapshot_countries" and A.loc[30, "country_name"] == ""
    assert A.loc[515, "mapping_status"] == "special_entry"
    assert A.loc[40, "mapping_status"] == "no_flag"
    assert set(A.mapping_status) <= set(club_flags.STATUSES)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def G():
    if not g2.GEO_V2_CSV.exists():
        pytest.skip("geography v2 not built")
    return pd.read_csv(g2.GEO_V2_CSV, keep_default_na=False)


@pytest.fixture(scope="module")
def N():
    return json.loads(g2.NUMBERS_JSON.read_text())


def test_stage1c_unchanged():
    assert _sha(g2.CANONICAL_CSV) == STAGE1C_SHA256 == g2.STAGE1C_SHA256


def test_existing_mapped_countries_are_never_overwritten(G):
    lg = G[G.league_country.ne("") & ~G.special_entry]
    assert (lg.calendar_country == lg.league_country).all() and (lg.source_priority == 1).all()
    differs = lg[lg.raw_flag_countries_observed.ne("") & lg.raw_flag_country.ne(lg.league_country)]
    assert differs.conflict.all()                                   # every disagreement is recorded


def test_raw_flag_only_fills_missing_geography(G):
    fl = G[G.source_priority.eq(2)]
    assert fl.league_country.eq("").all()
    assert fl.raw_flag_status.isin(g2.FLAG_USABLE).all()
    assert (fl.calendar_country == fl.raw_flag_country).all()
    keep = G[G.raw_flag_country.ne("") & G.calendar_country.ne("") & G.raw_flag_country.ne(G.calendar_country)]
    assert keep.conflict.all()


def test_every_assignment_has_provenance(G):
    a = G[G.calendar_country.ne("")]
    assert a.source.ne("").all() and a.source_priority.between(1, 5).all() and a.provenance.ne("").all()
    assert G[G.calendar_country.eq("")].source_priority.eq(0).all()
    assert not G.loc[G.special_entry, "calendar_country"].ne("").any()


def test_no_country_from_a_senior_name_match(G):
    par = G[G.source_priority.eq(4)]
    assert par.parent_rule.eq(PROPOSED_RULE).all() and par.parent_direct_moves.gt(0).all()
    assert par.club_name.map(lambda s: bool(YOUTH_OR_RESERVE.search(s))).all()
    assert set(G.source) <= set(g2.SOURCES.values()) | {"", "special entry (not a club)"}
    fb = G[G.source_priority.ge(3)]                                 # a fallback never contradicts the raw rows
    for r in fb.itertuples():
        assert r.raw_flag_countries_observed == "" or r.calendar_country in r.raw_flag_countries_observed.split("|")


def test_coverage_numbers_reconcile(G, N):
    cl = N["clubs"]
    stage = G[G.in_stage1c]
    assert len(stage) == cl["stage1c"] and cl["special"] + cl["real"] == cl["stage1c"]
    assert cl["with_country"] + cl["real_missing"] == cl["real"]
    e = N["endings_v2"]
    assert e["both_mapped"] + e["one_missing"] + e["both_missing"] == e["realised"] == N["realised"]
    ua = N["unmapped_after"]
    assert ua["recovered_raw"] + ua["recovered_fallback"] + ua["special"] + ua["unresolved"] \
        == N["unmapped_before"]["clubs"]
    RA = pd.read_csv(g2.RESIDUAL_CSV, keep_default_na=False)
    assert (RA.final_status == "unresolved").sum() == N["residual"]["unresolved"] == cl["real_missing"]
    assert RA[RA.final_status.eq("unresolved")].reason_unresolved.ne("").all()


def test_outputs_match_recorded_hashes_and_are_new_files(N):
    for name, h in N["outputs_sha256"].items():
        assert _sha(g2.OUT / name) == h
    from src.analysis import loan_calendar_rule as lc
    old = {p.name for p in (lc.SUMMARY_MD, lc.RULES_CSV, lc.CLASSIFIED_CSV, lc.EDGE_CSV, lc.SENS_CSV, lc.XB_CSV,
                            lc.NUMBERS_JSON)}
    assert not old & {p.name for p in g2.WRITES}


# ---------------------------------------------------------------------------
# Cached, deterministic rebuilds (any web request fails the test)
# ---------------------------------------------------------------------------

@pytest.fixture
def offline(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_rebuild_uses_the_cache_only_and_is_deterministic(offline, N):
    if not (raw_api.CACHE_DIR / "2025" / "transfers.json").exists():
        pytest.skip("raw cache not present")
    info = raw_api.cache_season_files()                             # md5-verifies, never downloads
    assert all(v["verified"] and not v["downloaded"] for v in info.values())
    flags = pd.read_csv(g2.RAW_FLAGS_CSV)
    sample = flags.sort_values("club_id").head(1)
    pid = int(sample.example_transfer.iloc[0].split("/spieler/")[1].split("/")[0])
    a = club_flags.read_sides({pid})
    b = club_flags.read_sides({pid})
    assert len(a) and a.equals(b)
    assert int(sample.club_id.iloc[0]) in set(a.club_id.dropna().astype(int))
    cached = json.loads(g2.WIKIDATA_CACHE.read_text())["queried"]
    assert g2.wikidata_countries([int(k) for k in cached]) == {k: cached[k] for k in cached}
