from __future__ import annotations

import hashlib

import pandas as pd

from .schemas import CONTRACT_FIELDS


def stable_event_id(player_id, transfer_date, from_club_id, to_club_id) -> str:
    raw = "|".join(str(value) for value in (player_id, transfer_date, from_club_id, to_club_id))
    return "tm_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def age_at_transfer(date_of_birth, transfer_date):
    if pd.isna(date_of_birth) or pd.isna(transfer_date):
        return None
    dob, moved = pd.Timestamp(date_of_birth), pd.Timestamp(transfer_date)
    return moved.year - dob.year - ((moved.month, moved.day) < (dob.month, dob.day))


def add_event_chains(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["player_id", "transfer_date", "event_id"]).copy()
    chain_ids: list[str] = []
    previous = None
    for row in events.itertuples():
        joins_previous = previous is not None and row.player_id == previous["player_id"] and row.from_club_id == previous["to_club_id"] and (pd.Timestamp(row.transfer_date) - pd.Timestamp(previous["transfer_date"])).days <= 14
        chain_id = previous["event_chain_id"] if joins_previous else "chain_" + row.event_id[3:]
        chain_ids.append(chain_id)
        previous = row._asdict() | {"event_chain_id": chain_id}
    events["event_chain_id"] = chain_ids
    return events


def enrich_events(connection, transfers: pd.DataFrame) -> pd.DataFrame:
    if transfers.empty:
        events = transfers.copy()
        for column in ["player_name", "date_of_birth", "age_at_transfer", "reported_transfer_fee", "market_value_nearest_transfer", "event_id", "event_chain_id"] + CONTRACT_FIELDS + [field + "_status" for field in CONTRACT_FIELDS]:
            events[column] = pd.Series(index=events.index, dtype="object")
        return events
    players = connection.execute("SELECT player_id, name, date_of_birth FROM players").fetchdf()
    events = transfers.merge(players, on="player_id", how="left", suffixes=("", "_player"))
    events["player_name"] = events["player_name"].fillna(events["name"])
    events["date_of_birth"] = pd.to_datetime(events["date_of_birth"])
    events["transfer_date"] = pd.to_datetime(events["transfer_date"])
    events["age_at_transfer"] = [age_at_transfer(dob, moved) for dob, moved in zip(events.date_of_birth, events.transfer_date)]
    valuations = connection.execute("SELECT player_id, date, market_value_in_eur FROM player_valuations").fetchdf()
    valuations["date"] = pd.to_datetime(valuations["date"])
    left = events[["player_id", "transfer_date"]].sort_values(["transfer_date", "player_id"])
    right = valuations.sort_values(["date", "player_id"])
    nearest = pd.merge_asof(left, right, left_on="transfer_date", right_on="date", by="player_id", direction="nearest")
    events = events.merge(nearest[["player_id", "transfer_date", "market_value_in_eur"]], on=["player_id", "transfer_date"], how="left", suffixes=("", "_nearest"))
    events["market_value_nearest_transfer"] = events["market_value_in_eur_nearest"].fillna(events["market_value_in_eur"])
    events["reported_transfer_fee"] = events["transfer_fee"] if "transfer_fee" in events else None
    events["event_id"] = [stable_event_id(*values) for values in events[["player_id", "transfer_date", "from_club_id", "to_club_id"]].itertuples(index=False, name=None)]
    for field in CONTRACT_FIELDS:
        events[field] = None
        events[field + "_status"] = None
    return add_event_chains(events.drop(columns=["name", "market_value_in_eur_nearest"], errors="ignore"))