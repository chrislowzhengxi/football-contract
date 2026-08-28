import pandas as pd

from src.enrich_structured import age_at_transfer, stable_event_id
from src.schemas import validate_events, validate_outcomes


def test_event_id_is_stable():
    assert stable_event_id(1, "2024-07-01", 10, 20) == stable_event_id(1, "2024-07-01", 10, 20)


def test_age_is_exact_before_birthday():
    assert age_at_transfer("2000-08-10", "2024-08-09") == 23


def test_validators_find_obvious_errors():
    events = pd.DataFrame({"event_id": ["x", "x"], "age_at_transfer": [23, 80]})
    outcomes = pd.DataFrame({"minutes_played": [-1], "window_start": [pd.Timestamp("2024-08-01")], "window_end": [pd.Timestamp("2024-07-01")], "missing_appearance_data": [False]})
    assert validate_events(events).errors
    assert validate_outcomes(outcomes).errors