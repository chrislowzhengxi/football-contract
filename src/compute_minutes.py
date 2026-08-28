from __future__ import annotations

import pandas as pd


def _season_start_year(season: str) -> int:
    value = season.split("/", 1)[0]
    year = int(value)
    return year if len(value) == 4 else 2000 + year


def _season_end(season: str) -> pd.Timestamp:
    start_year = _season_start_year(season)
    return pd.Timestamp(year=start_year + 1, month=6, day=30)


def _season_start(season: str) -> pd.Timestamp:
    return pd.Timestamp(year=_season_start_year(season), month=7, day=1)


def outcome_windows(transfer_date: pd.Timestamp, season: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    end = _season_end(season)
    next_start = _season_start_year(season) + 1
    next_season = f"{next_start:04d}/{next_start + 1:04d}"
    return [("remainder_transfer_season", transfer_date, end), ("following_full_season", _season_start(next_season), _season_end(next_season))]


def compute_minutes(connection, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    outcome_columns = ["event_id", "window_name", "window_start", "window_end", "minutes_club_id", "minutes_club_name", "match_date_min", "match_date_max", "minutes_played", "appearance_count", "missing_appearance_data"]
    if events.empty:
        return pd.DataFrame(columns=outcome_columns), pd.Series(dtype="object")
    player_ids = ", ".join(str(int(value)) for value in events["player_id"].dropna().unique())
    query = f"""
         SELECT a.appearance_id, a.player_id, a.player_club_id AS club_id, c.name AS club_name,
               a.minutes_played, g.date AS match_date, g.season, g.competition_id
        FROM appearances a JOIN games g ON CAST(a.game_id AS VARCHAR) = CAST(g.game_id AS VARCHAR)
         LEFT JOIN clubs c ON CAST(a.player_club_id AS VARCHAR) = CAST(c.club_id AS VARCHAR)
        WHERE a.player_id IN ({player_ids})
    """
    appearances = connection.execute(query).fetchdf()
    appearances["match_date"] = pd.to_datetime(appearances["match_date"])
    appearances = appearances.drop_duplicates("appearance_id")
    rows: list[dict] = []
    for event in events.itertuples(index=False):
        for window_name, start, end in outcome_windows(pd.Timestamp(event.transfer_date), event.transfer_season):
            selected = appearances[(appearances.player_id == event.player_id) & (appearances.match_date >= start) & (appearances.match_date <= end)]
            for (club_id, club_name), club_rows in selected.groupby(["club_id", "club_name"], dropna=False):
                rows.append({"event_id": event.event_id, "window_name": window_name, "window_start": start.date(), "window_end": end.date(), "minutes_club_id": club_id, "minutes_club_name": club_name, "match_date_min": club_rows.match_date.min().date(), "match_date_max": club_rows.match_date.max().date(), "minutes_played": int(club_rows.minutes_played.fillna(0).sum()), "appearance_count": int(club_rows.appearance_id.nunique()), "missing_appearance_data": False})
            if selected.empty:
                rows.append({"event_id": event.event_id, "window_name": window_name, "window_start": start.date(), "window_end": end.date(), "minutes_club_id": None, "minutes_club_name": None, "match_date_min": None, "match_date_max": None, "minutes_played": 0, "appearance_count": 0, "missing_appearance_data": True})
    return pd.DataFrame(rows), appearances.appearance_id