"""Prior-season playing time at the origin club: pre-move only, missing never zero."""
import pandas as pd

from src.analysis import playing_time_regression_audit as aud


def _frames():
    sides = pd.DataFrame({"game_id": ["g1", "g2", "g3", "h1"], "competition_id": "XX1",
                          "competition_type": "domestic_league", "season": 2022,
                          "date": pd.to_datetime(["2022-09-01", "2023-03-01", "2023-05-01", "2023-03-01"]),
                          "club_id": [1, 1, 1, 3], "rows": [14, 14, 14, 5]})
    sides["complete"] = sides.rows >= 11
    app = pd.DataFrame({"game_id": ["g2", "g3"], "club_id": [1, 1], "player_id": [7, 7], "minutes_played": [45, 90],
                        "date": pd.to_datetime(["2023-03-01", "2023-05-01"]), "national_team": False})
    rows = pd.DataFrame({"player_id": [7, 8, 10], "to_club_id": [1, 1, 3],
                         "_date": pd.to_datetime(["2022-12-31", "2023-06-30", "2022-07-01"]),
                         "transfer_type_normalized": ["loan_return", "loan_return", "permanent_transfer"]})
    C = pd.DataFrame({"event_id": ["a", "b", "c", "d"], "player_id": [7, 8, 9, 10], "a_club_id": [1, 1, 2, 3],
                      "move_date": pd.to_datetime(["2023-04-01", "2023-07-01", "2023-07-01", "2023-07-01"]),
                      "arrival_season": [2023, 2023, 2023, 2023]})
    return aud.prior_playing_time(C, rows, sides, app, ["XX1"]).set_index("event_id")


def test_window_starts_at_the_latest_entry_and_stops_before_the_move():
    P = _frames()
    a = P.loc["a"]
    assert a.prior_measurable and a.prior_games_available == 1          # g2 only: g1 before entry, g3 after the move
    assert a.prior_minutes == 45 and a.prior_share == 0.5 and not a.prior_zero
    assert a.prior_window_max_game_date < pd.Timestamp("2023-04-01")


def test_unmeasurable_prior_playing_time_is_missing_not_zero():
    P = _frames()
    assert P.loc["b", "prior_reason"].startswith("no origin-club league game")   # rejoined after the season
    assert P.loc["c", "prior_reason"].startswith("origin club not in a covered league")
    assert P.loc["d", "prior_reason"] == "incomplete appearance record in the window"
    for e in "bcd":
        assert pd.isna(P.loc[e, "prior_share"]) and pd.isna(P.loc[e, "prior_minutes"]) and pd.isna(P.loc[e, "prior_zero"])
