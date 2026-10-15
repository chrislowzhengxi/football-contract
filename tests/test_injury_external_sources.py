"""External injury-source audit: metadata only, nothing downloaded, licence facts read from captured metadata."""
import inspect
import json

import pandas as pd
import pytest

from src.analysis import injury_external_sources as ext

from tests.test_loan_episodes import STAGE1C_SHA256, _sha


@pytest.fixture(scope="module")
def N():
    if not ext.NUMBERS_JSON.exists():
        pytest.skip("audit not run")
    return json.loads(ext.NUMBERS_JSON.read_text())


def test_metadata_capture_is_intact_and_holds_no_injury_data():
    m = ext.load()                                               # raises if a captured file changed
    assert not [p for p in ext.META.iterdir() if p.suffix == ".csv"]
    assert m["capture"]["note"].startswith("metadata and documentation only")


def test_nothing_downloaded_and_conditional_outputs_absent(N):
    assert N["downloaded_injury_files"] == 0
    for name in ext.NOT_WRITTEN:
        assert not (ext.OUT / name).exists()
    src = inspect.getsource(ext)
    assert "urlopen" not in src and "requests" not in src


def test_licence_facts_come_from_metadata(N):
    p, f = N["primary"], N["figshare"]
    assert p["github_license"] is None and not p["license_file_in_repo"]
    assert p["kaggle_license"] == "CC0: Public Domain"
    assert f["license"] == "Creative Commons Attribution 4.0 International"
    assert f["versions"]["concept"]["title"] == "blank"
    assert "web scraping" in f["collection"]
    T = pd.read_csv(ext.COMPARISON_CSV, keep_default_na=False)
    assert T.license_status.ne("").all() and T.viable.ne("").all()


def test_readme_sample_ids_line_up_with_our_snapshot(N):
    pc = N["preview_check"]
    assert pc["player_id"] == 10 and pc["our_name"] == "Miroslav Klose"
    assert all(r["player_appearances_in_spell"] == 0 for r in pc["rows"])


def test_stage1c_and_earlier_injury_outputs_unchanged(N):
    assert _sha(ext.OUT / "stage1c_canonical_transfers.csv") == STAGE1C_SHA256
    for name, h in N["old_output_sha256"].items():
        assert _sha(ext.OUT / name) == h, name
