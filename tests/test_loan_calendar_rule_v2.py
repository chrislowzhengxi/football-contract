"""Calendar v2: frozen rule, reconciliation, COVID-free training, stability, old outputs untouched."""
import json

import pandas as pd
import pytest

from src.analysis import loan_calendar_rule as lc
from src.analysis import loan_calendar_rule_v2 as v2

from tests.test_loan_episodes import _sha


@pytest.fixture(scope="module")
def out():
    if not v2.NUMBERS_JSON.exists():
        pytest.skip("calendar v2 not built")
    N = json.loads(v2.NUMBERS_JSON.read_text())
    C = pd.read_csv(v2.CLASSIFIED_CSV, keep_default_na=False)
    return N, C


def test_parameters_are_the_frozen_v1_rule(out):
    N, _ = out
    assert N["params"] == v2.PARAMS == {k: getattr(lc, k) for k in v2.PARAMS}


def test_all_realised_endings_reconcile(out):
    N, C = out
    assert C.loan_event_id.is_unique and len(C) == N["v2"]["Realised endings"] == N["v1"]["Realised endings"]
    old = pd.read_csv(lc.CLASSIFIED_CSV, usecols=["loan_event_id"])
    assert set(old.loan_event_id) == set(C.loan_event_id)
    for k in ("v1", "v2"):
        o = N[k]
        assert o["Calendar-normal"] + o["Off-calendar"] + o["Geography-unclassified"] \
            + o["Below-history-threshold"] == o["Realised endings"]
        assert o["Classifiable"] == o["Calendar-normal"] + o["Off-calendar"]
    assert C.calendar_class.isin(lc.CLASSES).all()
    counts = C.calendar_class.value_counts()
    assert counts.filter(like="calendar_normal").sum() == N["v2"]["Calendar-normal"]
    assert counts.filter(like="off_calendar").sum() == N["v2"]["Off-calendar"]


def test_covid_endings_never_train_a_calendar(out):
    N, C = out
    C = C.assign(covid_period=C.covid_period.astype(str).eq("True"))
    for key, v in N["calendars_v2"].items():
        assert v["N"] == int(((C.calendar_key == key) & ~C.covid_period).sum())


def test_mapped_v1_countries_unchanged_and_changes_explained(out):
    N, C = out
    tr = N["transitions"]
    assert tr["classified_context_changed"] == 0 and tr["classified_changed_geography_changed"] == 0
    assert tr["classified_changed"] == tr["classified_changed_calendar_relearned"] + tr["classified_changed_unexplained"]
    assert tr["classified_changed_unexplained"] == 0
    assert C.status_changed.astype(str).eq("True").sum() == sum(
        v for k, v in tr["crosstab"].items() if k.split(" -> ")[0] != k.split(" -> ")[1])
    assert N["v1_reproduces_saved"]["identical"]


def test_old_outputs_unchanged(out):
    N, _ = out
    assert N["old_output_sha256"]
    for name, h in N["old_output_sha256"].items():
        assert _sha(v2.OUT / name) == h, name
    assert not {p.name for p in v2.WRITES} & {p.name for p in v2.OLD_OUTPUTS}


def test_summary_numbers_are_generated(out):
    N, _ = out
    md = v2.SUMMARY_MD.read_text()
    from src.analysis.daniel_scope_final import f0
    for k in ("Classifiable", "Calendar-normal", "Off-calendar", "Below-history-threshold"):
        assert f0(N["v2"][k]) in md and f0(N["v1"][k]) in md
    assert f0(N["transitions"]["classified_changed"]) in md
