"""Variable provenance: what came from the source, what we joined, what we constructed, and whether any LLM did.

    python -m src.analysis.variable_provenance

Documentation only: reads the lineage outputs, the saved output headers, the raw-source schemas and the source code;
reruns no analysis and changes no definition. The LLM audit is static: every `src` module that sends a request to an
LLM is found by pattern, and the import closure of every module that produces Stage 1C, the master table or a
Week 4-6 report variable is checked against them. Writes variable_provenance_matrix.csv and
variable_provenance_overview.md; fails if any other file under data/ changes.
"""
from __future__ import annotations

import ast
import json
import re
from collections import namedtuple

import pandas as pd

from . import data_lineage as dl
from .column_guide_specs import MASTER_GUIDE, ORIGINS, S1C_ROLES, STAGE1C_GUIDE, USEFULNESS
from .daniel_scope_final import f0
from .loan_episodes import ROOT
from ..stage1c.build import CANONICAL_COLUMNS

OUT = dl.OUT
MATRIX_CSV = OUT / "variable_provenance_matrix.csv"
OVERVIEW_MD = OUT / "variable_provenance_overview.md"
WRITES = (MATRIX_CSV, OVERVIEW_MD)
PDFS = {"W4": "Transfermarkt_Dataset_Week_4", "W5": "Transfermarkt_Dataset_Week_5", "W6": "Transfermarkt_Short_Week_6"}

RAW, JOIN, EXT = "UPSTREAM_RAW", "UPSTREAM_JOINED", "EXTERNAL_RAW"
MECH, CLASS, FLAG, STAT = "OUR_MECHANICAL_DERIVATION", "OUR_CLASSIFICATION_RULE", "OUR_SAMPLE_FLAG", "STATISTICAL_OUTPUT"
CATEGORIES = {
    RAW: "Value present as published in the upstream Transfermarkt data, on the same transfer row.",
    JOIN: "Value present as published in another upstream Transfermarkt table or row, attached by our join key or link.",
    EXT: "Value present as published in a non-upstream source (Kaggle injury files, Wikidata).",
    MECH: "Deterministic arithmetic, date, string, count, recode or join computed by our code.",
    CLASS: "Deterministic classification we designed (a threshold, window, priority order, pattern or learned "
           "calendar whose choice could reasonably differ).",
    FLAG: "Deterministic indicator of eligibility, reliability or sample membership.",
    STAT: "Produced by a statistical model, not an episode characteristic.",
}
CODE = {"S1": "src/stage1/backbone.py", "MOV": "src/stage1/movement.py", "S1B": "src/stage1b/parse.py",
        "POL": "src/stage1c/policy.py", "S1C": "src/stage1c/build.py", "LE": "src/analysis/loan_episodes.py",
        "LTE": "src/analysis/loan_timing_exploration.py", "LTV": "src/analysis/loan_timing_validation.py",
        "CS": "src/analysis/loan_category_c_split.py", "LCD": "src/analysis/loan_country_dates.py",
        "CAL": "src/analysis/loan_calendar_rule.py", "GEO": "src/analysis/club_geography_v2.py",
        "FLG": "src/stage1b/club_flags.py", "PTJ": "src/analysis/playing_time_join.py",
        "RED": "src/analysis/playing_time_sample_redesign.py", "REG": "src/analysis/playing_time_regression.py",
        "RA": "src/analysis/playing_time_regression_audit.py", "AGE": "src/analysis/playing_time_age.py",
        "FEA": "src/analysis/playing_time_injury_feasibility.py", "INJ": "src/analysis/playing_time_injury_adjusted.py",
        "DL": "src/analysis/data_lineage.py", "ORG": "src/analysis/club_organisation.py",
        "AUD": "src/analysis/appearance_audit.py"}
SOURCE_LABEL = {"duckdb": "DuckDB snapshot table", "raw": "raw Transfermarkt transfer history (JSON)",
                "stage1c": "stage1c_canonical_transfers.csv", "master": "analysis_episode_master.csv",
                "kaggle_inj": "Kaggle player_injuries.csv", "kaggle_prof": "Kaggle player_profiles.csv",
                "geo_v2": "club_geography_v2.csv", "flags": "club_country_raw_flags.csv", "wikidata": "Wikidata cache"}

P = namedtuple("P", "cat inputs code note")


def p(cat, inputs, code, note=""):
    return P(cat, inputs.split(), code, note)


T = "duckdb.transfers."
PAIR = ("stage1c.event_id stage1c.transfer_type_normalized stage1c.from_club_id stage1c.to_club_id stage1c.from_club_name "
        "stage1c.to_club_name stage1c.transfer_date stage1c.is_internal_move stage1c.transfermarkt_transfer_id")
ORGIN = ("stage1c.player_id stage1c.from_club_id stage1c.to_club_id stage1c.from_club_name stage1c.to_club_name "
         "flags.club_names flags.country_flag_id duckdb.clubs.name duckdb.games.competition_type "
         "duckdb.games.home_club_id duckdb.games.away_club_id")
NEXT = "stage1c.event_id stage1c.transfer_date stage1c.transfer_type_normalized stage1c.is_internal_move"
GAMES = "duckdb.games.game_id duckdb.games.competition_type duckdb.games.season duckdb.games.date " \
        "duckdb.games.home_club_id duckdb.games.away_club_id"
APPS = "duckdb.appearances.game_id duckdb.appearances.player_id duckdb.appearances.player_club_id"
KINJ = "kaggle_inj.player_id kaggle_inj.injury_reason kaggle_inj.from_date kaggle_inj.end_date"
PV = "duckdb.player_valuations.player_id duckdb.player_valuations.date duckdb.player_valuations.market_value_in_eur"
COV = "master.age_exact master.position_clean master.log_relative_value master.log_squad_value master.remain"

# ---------------------------------------------------------------------------
# Stage 1C (all 42 columns)
# ---------------------------------------------------------------------------

STAGE1C = {
    "event_id": p(MECH, f"{T}player_id {T}transfer_date {T}from_club_id {T}to_club_id", "S1", "sha256 hash of four raw fields"),
    "transfermarkt_transfer_id": p(RAW, "raw.url", "S1B", "the number inside the raw transfer url"),
    "event_chain_id": p(CLASS, "stage1c.player_id stage1c.from_club_id stage1c.to_club_id stage1c.transfer_date", "S1",
                        "14-day continuation rule (legacy)"),
    "player_id": p(RAW, f"{T}player_id", "S1"),
    "player_name": p(RAW, f"{T}player_name", "S1"),
    "date_of_birth": p(JOIN, "duckdb.players.date_of_birth", "S1", "joined on player_id"),
    "age_at_transfer": p(MECH, f"duckdb.players.date_of_birth {T}transfer_date", "S1"),
    "position": p(JOIN, "duckdb.players.position", "S1", "joined on player_id"),
    "country_of_citizenship": p(JOIN, "duckdb.players.country_of_citizenship", "S1", "joined on player_id"),
    "transfer_date": p(RAW, f"{T}transfer_date", "S1"),
    "transfer_season": p(RAW, f"{T}transfer_season", "S1"),
    "from_club_id": p(RAW, f"{T}from_club_id", "S1"),
    "from_club_name": p(RAW, f"{T}from_club_name", "S1"),
    "to_club_id": p(RAW, f"{T}to_club_id", "S1"),
    "to_club_name": p(RAW, f"{T}to_club_name", "S1"),
    "transfer_type_raw": p(MECH, "raw.fee", "S1B", "exact-text recode of Transfermarkt's fee cell"),
    "transfer_type_normalized": p(CLASS, "stage1c.transfer_type_raw stage1c.is_internal_move stage1c.stage1_heuristic_class",
                                  "POL", "label → type, refined by the internal-move name rule"),
    "transfer_type_source": p(MECH, "stage1c.transfer_type_raw stage1c.is_internal_move", "POL"),
    "stage1_heuristic_class": p(CLASS, f"{T}transfer_fee {T}from_club_id {T}to_club_id {T}from_club_name {T}to_club_name "
                                f"{T}transfer_date", "MOV", "timeline heuristic; overruled by the raw label"),
    "heuristic_agrees_with_label": p(MECH, "stage1c.stage1_heuristic_class stage1c.transfer_type_raw", "S1C"),
    "fee_display_raw": p(RAW, "raw.fee", "S1C"),
    "fee_numeric_eur": p(MECH, "raw.fee", "S1B", "amount parsed from the fee text"),
    "fee_role": p(MECH, "stage1c.transfer_type_raw", "S1C"),
    "fee_disclosure_status": p(MECH, "stage1c.transfer_type_raw", "POL"),
    "permanent_transfer_fee_eur": p(MECH, "stage1c.transfer_type_raw stage1c.fee_numeric_eur", "POL"),
    "loan_fee_eur": p(MECH, "stage1c.transfer_type_raw stage1c.fee_numeric_eur", "POL"),
    "fee_on_return_eur": p(MECH, "stage1c.transfer_type_raw stage1c.fee_numeric_eur", "POL"),
    "duckdb_transfer_fee": p(RAW, f"{T}transfer_fee", "S1", "upstream number; writes 0 for non-numeric fee labels"),
    "fee_matches_duckdb": p(MECH, "stage1c.permanent_transfer_fee_eur stage1c.duckdb_transfer_fee stage1c.transfer_type_raw", "S1C"),
    "market_value_eur": p(RAW, f"{T}market_value_in_eur", "S1", "upstream's derivation of this value is undocumented"),
    "market_value_display": p(RAW, "raw.marketValue", "S1C"),
    "is_loan_related": p(MECH, "stage1c.transfer_type_raw", "S1C"),
    "return_leg_has_fee": p(MECH, "stage1c.transfer_type_normalized stage1c.fee_on_return_eur", "S1C"),
    "transfermarkt_future_transfer": p(RAW, "raw.futureTransfer", "S1C", "flag semantics undocumented upstream"),
    "involves_placeholder_club": p(MECH, "stage1c.from_club_name stage1c.to_club_name", "S1C",
                                   "exact match to Transfermarkt's placeholder names"),
    "is_youth_or_reserve_side": p(CLASS, "stage1c.from_club_name stage1c.to_club_name", "POL", "name regex"),
    "is_internal_move": p(CLASS, "stage1c.from_club_name stage1c.to_club_name stage1c.from_club_id stage1c.to_club_id "
                          "stage1c.involves_placeholder_club", "S1C", "name normalisation rule"),
    "is_senior_move": p(CLASS, "stage1c.involves_placeholder_club stage1c.is_youth_or_reserve_side stage1c.is_internal_move", "S1C"),
    "has_monetary_amount": p(MECH, "stage1c.permanent_transfer_fee_eur stage1c.loan_fee_eur stage1c.fee_on_return_eur", "S1C"),
    "enrichment_matched": p(MECH, f"{T}player_id raw.dateUnformatted raw.from.href raw.to.href", "S1C"),
    "is_research_target": p(FLAG, "stage1c.involves_placeholder_club stage1c.is_internal_move stage1c.transfer_type_normalized "
                            "stage1c.return_leg_has_fee stage1c.has_monetary_amount stage1c.is_senior_move", "POL",
                            "Stage 2 eligibility only"),
    "research_exclusion_reason": p(FLAG, "stage1c.involves_placeholder_club stage1c.is_internal_move "
                                   "stage1c.transfer_type_normalized stage1c.return_leg_has_fee stage1c.has_monetary_amount "
                                   "stage1c.is_senior_move", "POL", "reason attached to the Stage 2 eligibility flag"),
}

# ---------------------------------------------------------------------------
# Master table (every column; Stage 1C columns reuse the entries above)
# ---------------------------------------------------------------------------

MASTER_OWN = {
    "season": p(MECH, "stage1c.transfer_season", "LE"),
    "episode_type": p(MECH, "stage1c.transfer_type_normalized", "LE", "which rows are episodes is our row-role rule"),
    "kind": p(MECH, "master.episode_type", "PTJ"),
    "match_reason": p(CLASS, PAIR, "LE", "our one-to-one pairing algorithm"),
    "has_matched_return": p(CLASS, f"master.terminal_event_id {PAIR}", "LE", "result of our pairing"),
    "terminal_event_id": p(CLASS, PAIR, "LE", "which row ends the loan is decided by our pairing"),
    "terminal_date": p(JOIN, "stage1c.transfer_date", "LE", "Transfermarkt date of the ending row; the link is our pairing"),
    "terminal_type": p(CLASS, "stage1c.transfer_type_normalized", "LE", "normalised type of the ending row"),
    "terminal_raw_label": p(JOIN, "stage1c.fee_display_raw", "LE", "Transfermarkt fee cell of the ending row"),
    "raw_label_class": p(MECH, "master.terminal_raw_label", "LE"),
    "terminal_scheduled_future": p(JOIN, "stage1c.transfermarkt_future_transfer", "LE", "future flag of the ending row"),
    "ending_observed": p(MECH, "master.terminal_date", "LTE"),
    "ending_realised": p(FLAG, "master.terminal_date master.terminal_scheduled_future", "LTE", "calendar population"),
    "duration_days": p(MECH, "master.transfer_date master.terminal_date", "LE"),
    "ending_fee_on_return_eur": p(MECH, "stage1c.fee_on_return_eur", "LE", "parsed amount of the ending row"),
    "return_boundary_days": p(MECH, "master.terminal_date", "LE"),
    "follow_on_event_id": p(CLASS, f"{NEXT} stage1c.involves_placeholder_club", "LE", "60-day window and skip rules"),
    "follow_on_gap_days": p(MECH, "stage1c.transfer_date master.terminal_date", "LE"),
    "follow_on_type": p(CLASS, "stage1c.transfer_type_normalized", "LE", "normalised type of the follow-on row"),
    "follow_on_to_club": p(JOIN, "stage1c.to_club_name", "LE", "club name of the follow-on row"),
    "economic_ending": p(CLASS, "master.match_reason master.follow_on_event_id master.follow_on_type master.follow_on_gap_days "
                         "master.return_boundary_days master.ending_fee_on_return_eur stage1c.to_club_id", "LE"),
    "ending_detail": p(CLASS, "master.match_reason master.follow_on_type master.follow_on_gap_days master.return_boundary_days",
                       "LE"),
    "next_move_event_id": p(CLASS, NEXT, "LTE", "first substantive later row, any distance"),
    "next_move_gap_days": p(MECH, "stage1c.transfer_date master.terminal_date", "LTE"),
    "next_move_type": p(CLASS, "stage1c.transfer_type_normalized", "LTE"),
    "next_category": p(CLASS, "master.next_move_type master.next_move_event_id master.to_club_id master.has_matched_return "
                       "stage1c.to_club_id stage1c.involves_placeholder_club", "CAL"),
    "from_country_v1": p(JOIN, f"duckdb.clubs.domestic_competition_id duckdb.competitions.country_name {GAMES} "
                         "duckdb.games.competition_id", "LE", "league country; superseded by v2"),
    "to_country_v1": p(JOIN, f"duckdb.clubs.domestic_competition_id duckdb.competitions.country_name {GAMES} "
                       "duckdb.games.competition_id", "LE", "league country; superseded by v2"),
    "from_country_v2": p(JOIN, "geo_v2.calendar_country duckdb.competitions.country_name raw.from.countryFlag "
                         "flags.flag_ids duckdb.countries.country_name wikidata.country", "GEO",
                         "precedence ours; 4 clubs via our parent-link rule, 2 via Wikidata"),
    "to_country_v2": p(JOIN, "geo_v2.calendar_country duckdb.competitions.country_name raw.to.countryFlag "
                       "flags.flag_ids duckdb.countries.country_name wikidata.country", "GEO",
                       "precedence ours; 4 clubs via our parent-link rule, 2 via Wikidata"),
    "from_geography_source_v2": p(MECH, "geo_v2.source", "GEO"),
    "to_geography_source_v2": p(MECH, "geo_v2.source", "GEO"),
    "calendar_context": p(MECH, "master.from_country_v2 master.to_country_v2", "LCD"),
    "calendar_key": p(MECH, "master.calendar_context master.from_country_v2", "CAL"),
    "ending_in_covid_window": p(CLASS, "master.terminal_date", "CAL", "window dates chosen by us"),
    "calendar_class": p(CLASS, "master.calendar_key master.terminal_date master.ending_in_covid_window", "CAL",
                        "calendar dates learned from the data by deterministic clustering"),
    "calendar_state": p(CLASS, "master.calendar_class master.unclassified_reason", "CAL"),
    "calendar_classifiable": p(CLASS, "master.calendar_class", "DL"),
    "off_calendar": p(CLASS, "master.calendar_class", "DL"),
    "loan_calendar_status": p(CLASS, "master.kind master.calendar_state master.ending_realised", "DL",
                              "re-labels the existing calendar classification"),
    "off_calendar_flag": p(CLASS, "master.loan_calendar_status", "DL"),
    "matched_standard_mmdd": p(CLASS, "master.terminal_date master.calendar_key", "CAL"),
    "nearest_standard_mmdd": p(CLASS, "master.terminal_date master.calendar_key", "CAL"),
    "days_to_nearest_standard": p(MECH, "master.terminal_date master.calendar_key", "CAL", "distance to a learned date"),
    "unclassified_reason": p(CLASS, "master.calendar_key master.calendar_context", "CAL"),
    "receiving_league": p(JOIN, f"duckdb.games.competition_id {GAMES} stage1c.to_club_id stage1c.transfer_date", "PTJ",
                          "upstream league id; the season assignment is our date rule"),
    "arrival_season": p(MECH, f"{GAMES} stage1c.to_club_id stage1c.transfer_date", "PTJ", "date rule"),
    "departure_date": p(JOIN, "stage1c.transfer_date stage1c.transfer_type_normalized stage1c.is_internal_move", "PTJ",
                        "Transfermarkt date of the player's next row"),
    "departed_in_window": p(MECH, "master.departure_date duckdb.games.date", "PTJ"),
    "window_start": p(RAW, "stage1c.transfer_date", "PTJ", "the move date itself"),
    "window_end": p(MECH, "duckdb.games.date master.departure_date", "PTJ"),
    "season_games": p(MECH, GAMES, "PTJ"),
    "games_remaining_at_arrival": p(MECH, f"{GAMES} master.transfer_date", "PTJ"),
    "available_games": p(MECH, f"{GAMES} master.window_start master.window_end", "PTJ"),
    "complete_games": p(MECH, f"{APPS} {GAMES}", "PTJ", "uses our ≥ 11-row completeness threshold"),
    "appearances": p(MECH, f"{APPS} {GAMES}", "PTJ"),
    "minutes": p(MECH, f"duckdb.appearances.minutes_played {APPS} {GAMES}", "PTJ", "sum of upstream minutes_played"),
    "available_minutes": p(MECH, "master.available_games", "PTJ"),
    "minutes_share": p(MECH, "master.minutes master.available_games", "PTJ"),
    "true_zero": p(CLASS, "master.minutes master.complete_games master.available_games", "PTJ",
                   "a zero exists only under our completeness rule"),
    "pt_exclusion_reason": p(FLAG, f"stage1c.transfermarkt_future_transfer master.receiving_league duckdb.players.player_id "
                             f"master.available_games master.complete_games master.window_end {APPS} duckdb.appearances.date",
                             "PTJ"),
    "prior_spell_at_b": p(FLAG, "stage1c.player_id stage1c.to_club_id stage1c.transfer_date", "PTJ"),
    "pt_loan_ending": p(CLASS, "master.terminal_date master.has_matched_return", "PTJ", "used as a sample criterion"),
    "arrived_full_season": p(FLAG, "master.games_remaining_at_arrival master.season_games", "PTJ", "old two-thirds rule"),
    "remain": p(MECH, "master.games_remaining_at_arrival master.season_games", "REG"),
    "age_exact": p(MECH, "stage1c.date_of_birth stage1c.transfer_date", "REG"),
    "age_years": p(MECH, "master.age_exact", "AGE"),
    "older": p(CLASS, "master.age_years", "AGE", "cutoff 23/24 chosen by us"),
    "position_clean": p(MECH, "stage1c.position", "REG"),
    "player_value": p(JOIN, PV, "REG", "upstream valuation; choosing the one strictly before the move is our rule"),
    "player_value_date": p(JOIN, "duckdb.player_valuations.date", "REG"),
    "player_value_gap_days": p(MECH, "master.transfer_date master.player_value_date", "REG"),
    "stale_player_value": p(CLASS, "master.player_value_gap_days", "REG", "365-day threshold"),
    "squad_value": p(MECH, f"{PV} duckdb.player_valuations.current_club_id", "REG",
                     "our aggregate of upstream valuations (365-day window, ≥ 11 players, focal player excluded)"),
    "squad_players": p(MECH, f"{PV} duckdb.player_valuations.current_club_id", "REG"),
    "focal_in_squad": p(MECH, f"{PV} duckdb.player_valuations.current_club_id", "REG"),
    "relative_value": p(MECH, "master.player_value master.squad_value", "REG"),
    "log_player_value": p(MECH, "master.player_value", "REG"),
    "log_squad_value": p(MECH, "master.squad_value", "REG"),
    "log_relative_value": p(MECH, "master.player_value master.squad_value", "REG"),
    "league_season": p(MECH, "master.receiving_league master.arrival_season", "REG"),
    "injury_join_status": p(MECH, "kaggle_inj.player_id kaggle_prof.player_id stage1c.player_id", "INJ", "exact id membership"),
    "available_games_current": p(MECH, "master.available_games", "INJ"),
    "confirmed_injury_games": p(CLASS, f"{KINJ} duckdb.games.date {APPS}", "INJ",
                                "Kaggle dates; medical classification, inclusive overlap and appearance override are ours"),
    "available_games_fit": p(MECH, "master.available_games_current master.confirmed_injury_games", "INJ"),
    "current_share": p(MECH, "master.minutes master.available_games_current", "INJ"),
    "fit_adjusted_share": p(MECH, "master.minutes master.available_games_fit", "INJ"),
    "injury_days_overlap": p(MECH, f"{KINJ} master.window_start master.window_end", "INJ"),
    "injury_spell_count": p(MECH, f"{KINJ} master.window_start master.window_end", "INJ"),
    "appearance_conflicts": p(MECH, f"{KINJ} duckdb.games.date {APPS}", "INJ"),
    "injury_data_quality_flag": p(FLAG, f"master.injury_join_status master.window_start master.window_end {KINJ} "
                                  "master.available_games_fit", "INJ"),
    "injury_games_strict": p(CLASS, f"{KINJ} duckdb.games.date {APPS}", "INJ", "strict medical classification"),
    "fit_share_strict": p(MECH, "master.minutes master.available_games_current master.injury_games_strict", "INJ"),
    "injury_games_severe3": p(CLASS, f"{KINJ} duckdb.games.date {APPS}", "INJ", "only spells removing ≥ 3 games"),
    "fit_share_severe3": p(MECH, "master.minutes master.available_games_current master.injury_games_severe3", "INJ"),
    "injury_games_long5": p(CLASS, f"{KINJ} duckdb.games.date {APPS}", "INJ", "only spells removing ≥ 5 games"),
    "fit_share_long5": p(MECH, "master.minutes master.available_games_current master.injury_games_long5", "INJ"),
    "in_calendar_population": p(FLAG, "master.ending_realised", "LTE"),
    "in_playing_time_candidates": p(FLAG, "master.kind", "PTJ"),
    "pt_realised": p(FLAG, "stage1c.transfermarkt_future_transfer", "PTJ"),
    "pt_covered_league_season": p(FLAG, "master.receiving_league", "PTJ"),
    "pt_matched": p(FLAG, "duckdb.players.player_id duckdb.appearances.player_club_id", "PTJ"),
    "pt_reliable": p(FLAG, "master.pt_exclusion_reason", "PTJ"),
    "pt_lineup_games_checked": p(MECH, f"{GAMES} duckdb.game_lineups.game_id duckdb.game_lineups.club_id "
                                 "master.window_start master.window_end", "AUD"),
    "pt_games_played_without_row": p(MECH, "duckdb.game_lineups.player_id duckdb.game_lineups.type "
                                     "duckdb.game_events.player_in_id duckdb.game_events.type "
                                     "duckdb.appearances.player_id duckdb.appearances.player_club_id", "AUD"),
    "pt_contradiction_type": p(CLASS, "master.pt_games_played_without_row master.appearances", "AUD"),
    "pt_lineup_contradicted": p(FLAG, "master.pt_reliable master.pt_games_played_without_row", "AUD"),
    "pt_reliable_v2": p(FLAG, "master.pt_reliable master.pt_lineup_contradicted", "AUD"),
    "in_old_primary_sample": p(FLAG, "master.pt_reliable master.arrived_full_season master.prior_spell_at_b "
                               "master.pt_loan_ending", "PTJ"),
    "in_revised_sample": p(FLAG, "master.pt_reliable master.prior_spell_at_b master.pt_loan_ending", "RED"),
    "in_injury_usable": p(FLAG, "master.in_revised_sample master.injury_data_quality_flag", "INJ"),
    "in_old_regression_sample": p(FLAG, f"master.in_old_primary_sample {COV}", "REG"),
    "in_age_model_sample": p(FLAG, f"master.in_revised_sample {COV}", "AGE"),
    "in_injury_model_sample": p(FLAG, f"master.in_injury_usable {COV}", "INJ"),
    "calendar_first_eligible": p(FLAG, "master.kind master.loan_calendar_status", "DL"),
    "in_cf_revised_sample": p(FLAG, "master.in_revised_sample master.calendar_first_eligible", "DL"),
    "in_cf_injury_usable": p(FLAG, "master.in_injury_usable master.calendar_first_eligible", "DL"),
    "in_cf_main_model_sample": p(FLAG, "master.in_age_model_sample master.calendar_first_eligible", "DL"),
    "in_cf_injury_model_sample": p(FLAG, "master.in_injury_model_sample master.calendar_first_eligible", "DL"),
    "from_club_parent_id_v2": p(CLASS, f"{ORGIN}", "ORG", "proposed parent of a youth/reserve side: names + player flows"),
    "same_organisation_move_v2": p(CLASS, f"master.from_club_id master.to_club_id {ORGIN}", "ORG",
                                   "high-confidence links only"),
    "academy_promotion_candidate": p(CLASS, f"master.from_club_id master.to_club_id {ORGIN}", "ORG", "any confidence"),
    "same_org_confidence": p(CLASS, f"master.from_club_id master.to_club_id master.from_club_name master.to_club_name "
                             f"master.is_youth_or_reserve_side {ORGIN}", "ORG", "thresholds chosen by us"),
    "same_org_reason": p(MECH, "master.same_org_confidence master.from_club_id", "ORG"),
}

# ---------------------------------------------------------------------------
# Week 4-6 report variables
# ---------------------------------------------------------------------------

R = namedtuple("R", "group variable cat inputs keys rule meaning kind caveat where master code terms")


def r(group, variable, cat, inputs, keys, rule, meaning, kind, caveat, where, master, code, terms):
    return R(group, variable, cat, inputs.split(), keys, rule, meaning, kind, caveat, where, master, code, terms)


ETR = ("OUR HEURISTIC — NOT AN UPSTREAM TRANSFERMARKT FIELD — NOT PROOF OF CONTRACTUAL EARLY TERMINATION.")
REPORT = [
    # transfer / episode
    r("Transfer / episode", "event_id", MECH, f"{T}player_id {T}transfer_date {T}from_club_id {T}to_club_id", "—",
      "'tm_' + sha256('player_id|transfer_date|from_club_id|to_club_id')[:20]", "Identifier of one registration row; an "
      "episode keeps its outbound row's id.", "identifier", "Identifies a row, not a contract.", "internal key (not printed)",
      "event_id", "S1", {}),
    r("Transfer / episode", "player_id", RAW, f"{T}player_id", "join key to players, appearances, valuations, Kaggle",
      "copied", "Transfermarkt player id.", "identifier", "", "W6 §4 (exact-ID injury join)", "player_id", "S1",
      {"W6": ["Transfermarkt player IDs"]}),
    r("Transfer / episode", "transfer date", RAW, f"{T}transfer_date", "—", "copied",
      "Date Transfermarkt records for the movement.", "value", "Registration timing, not negotiation date.",
      "W4 §1 (lag), W5 §4.2 (window start)", "transfer_date", "S1", {"W4": ["dates recorded by Transfermarkt"],
                                                                     "W5": ["recorded transfer date"]}),
    r("Transfer / episode", "from club / to club (lender, borrower, receiving club)", RAW,
      f"{T}from_club_id {T}from_club_name {T}to_club_id {T}to_club_name", "—", "copied",
      "Club the player leaves and joins (lender and borrower for a loan).", "identifier", "", "W4–W6 throughout",
      "from_club_id, to_club_id", "S1", {"W4": ["lender club"], "W5": ["receiving club"]}),
    r("Transfer / episode", "transfer type (fee label, normalised type)", CLASS,
      "raw.fee stage1c.transfer_type_raw stage1c.is_internal_move", "event_id",
      "Transfermarkt's fee cell → exact-text label → type (label wins; an uninformative label on an internal move → "
      "youth_or_internal)", "Loan, End of loan, paid, free, undisclosed, no fee shown, …", "category",
      "The label text is Transfermarkt's; the mapping is ours.", "W4 §1, W5 §4", "transfer_type_raw, transfer_type_normalized",
      "POL", {"W4": ["End of loan"], "W5": ["permanent transfer"]}),
    r("Transfer / episode", "loan episode", CLASS, "stage1c.transfer_type_normalized stage1c.is_internal_move "
      "stage1c.involves_placeholder_club stage1c.from_club_id stage1c.to_club_id stage1c.transfer_date", "event_id",
      "an outbound loan row whose row role is 'episode' (not internal, not placeholder), paired one-to-one with the "
      "movement that ends it", "One real loan.", "category", "", "W5–W6 (29,699 loans)", "episode_type, kind", "LE",
      {"W6": ["loan episodes"], "W5": ["candidate episodes"]}),
    r("Transfer / episode", "permanent-transfer episode", CLASS, "stage1c.transfer_type_normalized stage1c.is_internal_move "
      "stage1c.involves_placeholder_club", "event_id", "episode row of type permanent_transfer, free_transfer, "
      "undisclosed_transfer or no_fee_shown", "One real permanent arrival.", "category",
      "Pools paid, free, undisclosed and no-fee-shown moves.", "W5 §4, W6 §3", "episode_type, kind", "LE",
      {"W5": ["permanent transfers"], "W6": ["permanent-transfer episodes"]}),
    r("Transfer / episode", "recorded loan ending (ending date, realised)", JOIN, "stage1c.transfer_date "
      "stage1c.transfermarkt_future_transfer stage1c.transfer_type_normalized", "terminal_event_id = event_id of the ending row",
      "date of the row our pairing assigns as the loan's end; realised = that row exists and is not flagged future",
      "When and whether the loan ended.", "value (date) + flag", "Transfermarkt's date on the ending row; the pairing is ours.",
      "W4 §4, W5 §2.2, W6 §1", "terminal_date, ending_realised", "LE",
      {"W4": ["realised loan endings"], "W5": ["realised loan endings"], "W6": ["recorded loan endings"]}),
    r("Transfer / episode", "loan duration", MECH, "master.transfer_date master.terminal_date", "event_id",
      "terminal_date − loan date (days)", "Length of the loan.", "value", "", "W5 Table 6, W6 §1", "duration_days", "LE",
      {"W5": ["Median loan duration"], "W6": ["Median duration"]}),
    r("Transfer / episode", "strict A→B→A→B sequence", CLASS, "stage1c.from_club_id stage1c.to_club_id "
      "stage1c.transfer_type_normalized stage1c.transfer_date", "loan_event_id",
      "loan A→B ended by an exact B→A return whose next substantive row (any distance) is a permanent move from exactly "
      "A to exactly B", "Loan converted into a permanent move back to the borrower.", "category",
      "Describes a registration sequence; mechanism (option/obligation) not identified.", "W4 §1–2", "— (loan_strict_sequence_check.csv)",
      "LTV", {"W4": ["exact instances"]}),
    r("Transfer / episode", "return-to-permanent lag", MECH, "stage1c.transfer_date", "return and permanent rows of the sequence",
      "permanent-move date − return date (days)", "Days between the return and the permanent move.", "value",
      "Transfermarkt dates, i.e. registration timing.", "W4 §1", "— (loan_strict_sequence_check.csv: lag_days)", "LTV",
      {"W4": ["lag days"]}),
    r("Transfer / episode", "realised-only sequences", FLAG, "stage1c.transfermarkt_future_transfer", "loan_event_id",
      "both the return and the permanent move not flagged future", "Robustness subset.", "sample filter", "",
      "W4 §1.3", "—", "LTV", {"W4": ["Realised-only"]}),
    r("Transfer / episode", "calendar-immediate conversion", CLASS, "stage1c.transfer_date", "loan_event_id",
      "lag ≤ 1 day, or a 31 May return followed by a 1 July move in the same year", "Immediate conversion, allowing the "
      "British 31 May convention.", "category", "", "W4 §3, §10", "— (loan_strict_sequence_check.csv)", "LTV",
      {"W4": ["calendar-immediate"]}),
    r("Transfer / episode", "follow-on move (60-day)", CLASS, f"{NEXT} stage1c.involves_placeholder_club", "loan_event_id",
      "first later row after a return that is not internal / youth_or_internal / return / draft, within 60 days, not a "
      "release", "The player's next deal after the return.", "identifier + gap", "Window lengths (21, 60 days) are our choices.",
      "W4 §7–8, W5 §2.9, W6 §1", "follow_on_event_id, follow_on_gap_days", "LE",
      {"W4": ["60-day follow-on window"], "W5": ["60-day follow-on rule"]}),
    r("Transfer / episode", "next substantive move (unwindowed) and its category", CLASS, NEXT, "loan_event_id",
      "first later row after a return (any distance, releases included); category: permanent to borrower, re-loan to the "
      "same borrower, loan elsewhere, paid / free move to a third club, release, none", "What the player did next.",
      "category + gap", "Differs from the 60-day follow-on.", "W4 §7, W5 Table 6, W6 §1",
      "next_move_event_id, next_move_gap_days, next_category", "LTE",
      {"W4": ["next substantive move"], "W5": ["Next move within 1 day"], "W6": ["Next move"]}),
    r("Transfer / episode", "economic_ending (production follow-on classes)", CLASS, "master.match_reason "
      "master.follow_on_type master.follow_on_gap_days master.return_boundary_days master.ending_fee_on_return_eur",
      "loan_event_id", "first applicable of: unresolved; conversion; third-party sale; other nonstandard; early_termination; "
      "immediate follow-on; fee-bearing return; ordinary end of loan", "Sequence pattern of the ending.", "category",
      "Describes the registration sequence, never a contract term.", "W4 §7–8", "economic_ending", "LE",
      {"W4": ["production follow-on categories"]}),
    r("Transfer / episode", "sequence category B/C/D (old 'Category C')", CLASS, "master.economic_ending "
      "master.has_matched_return", "loan_event_id", "B = conversion, C = immediate follow-on or early_termination, "
      "D = third-party sale (returns only); E = ended without a return; A otherwise", "Old broad follow-on grouping.",
      "category", "", "W4 §7, W5 §3", "— (loan_follow_on_timing.csv: sequence_category)", "LTE",
      {"W4": ["Another move or re-loan"], "W5": ["Category C"]}),
    r("Transfer / episode", "Category C subgroups (C1–C5)", CLASS, "master.next_move_type master.next_move_event_id "
      "stage1c.to_club_id stage1c.from_club_name stage1c.to_club_name", "loan_event_id",
      "C1 loan to the same borrower; C2 loan to a different club; C3 free move to a third club; C4 no-fee move to a third "
      "club; C5 other (rules applied in order)", "Where the player went after the return.", "category", "",
      "W5 §3, W6 §2", "— (loan_category_c_split.csv: subgroup)", "CS",
      {"W5": ["Re-loan to same borrower"], "W6": ["Re-loan to same borrower"]}),
    r("Transfer / episode", "non-ordinary registration sequence", CLASS, "master.economic_ending", "loan_event_id",
      "economic_ending is any class other than ordinary_end_of_loan and unresolved", "Earlier headline measure.",
      "category", "Window-dependent; not a share of unusual contract clauses.", "W4 §8", "economic_ending", "LE",
      {"W4": ["non-ordinary"]}),
    r("Transfer / episode", "early_termination", CLASS, "master.terminal_date master.has_matched_return "
      "master.follow_on_event_id master.follow_on_gap_days master.follow_on_type stage1c.to_club_id master.return_boundary_days",
      "loan_event_id", "return row; follow-on exists (≤ 60 days, not a release); not a permanent move back to the borrower; "
      "not a paid / undisclosed-fee move to a third club; follow-on ≤ 21 days; return > 10 days from both 30 June and "
      "31 December", "Return away from usual season ends, then a quick move.", "category", ETR, "W5 Table 6, W6 §1",
      "economic_ending == early_termination", "LE", {"W5": ["early-termination label"], "W6": ["early-termination flag"]}),
    # geography / calendar
    r("Geography / calendar", "club calendar country", JOIN, "duckdb.clubs.domestic_competition_id "
      "duckdb.competitions.country_name duckdb.games.competition_id raw.from.countryFlag raw.to.countryFlag flags.flag_ids "
      "duckdb.countries.country_name geo_v2.calendar_country", "club_id",
      "v1 league country, else the club's single raw countryFlag (named via countries), else competition membership / "
      "parent-club link / Wikidata", "Country of each club.", "category",
      "No country from a club name and no LLM guess; conflicting flags are recorded, not resolved by judgement.",
      "W5 §2.2, W6 §1", "from_country_v2, to_country_v2", "GEO", {"W5": ["unknown country"], "W6": ["country flag"]}),
    r("Geography / calendar", "Wikidata country (fallback C)", EXT, "wikidata.country", "Transfermarkt club id = Wikidata P7223",
      "copied when the club has no league country, no usable flag and no other fallback", "Country of 2 clubs.", "value",
      "Used for 2 clubs (Toronto FC II, Puerto Rico FC).", "not printed", "from_country_v2 (2 clubs)", "GEO", {}),
    r("Geography / calendar", "domestic / cross-border", MECH, "master.from_country_v2 master.to_country_v2", "loan_event_id",
      "same country → domestic; both known and different → cross-border; else unmapped", "Loan within or across countries.",
      "category", "", "W5 §2.2, W6 §1", "calendar_context", "LCD", {"W5": ["Cross-border"], "W6": ["cross-border"]}),
    r("Geography / calendar", "calendar key", MECH, "master.calendar_context master.from_country_v2", "loan_event_id",
      "'domestic: <country>' or 'cross-border'", "Which calendar applies.", "category", "", "W6 §1",
      "calendar_key", "CAL", {"W6": ["pooled cross-border calendar"]}),
    r("Geography / calendar", "learned country calendar (standard dates)", CLASS, "master.terminal_date master.calendar_key "
      "master.ending_in_covid_window", "calendar_key", "≥ 100 non-COVID domestic endings; ±3-day clusters; ≥ 3 years; "
      "≥ 2% share; up to 5 dates or 90% coverage; one pooled cross-border calendar",
      "Recurring ending dates per country.", "value (dates)", "Empirical dates, not official transfer windows.",
      "W5 §2.5–2.6, W6 §1", "matched_standard_mmdd, nearest_standard_mmdd", "CAL",
      {"W5": ["Retained dates"], "W6": ["Main recurring ending dates"]}),
    r("Geography / calendar", "calendar-normal / off-calendar", CLASS, "master.terminal_date master.calendar_key",
      "loan_event_id", "within ±3 days of a date of its calendar → calendar-normal, else off-calendar",
      "Timing classification of the ending.", "category", "A classification, not a sample filter; not proof of early "
      "termination.", "W5 §2.7–2.9, W6 §1", "calendar_class, calendar_state, off_calendar", "CAL",
      {"W5": ["off-calendar"], "W6": ["off-calendar"]}),
    r("Geography / calendar", "unclassified (country unknown / below threshold)", CLASS, "master.calendar_key "
      "master.calendar_context", "loan_event_id", "no calendar: a club without a country, or a domestic country with < 100 "
      "training endings", "No calendar to judge against.", "category", "Not unusual timing.", "W5 §2.7, W6 §1",
      "unclassified_reason", "CAL", {"W5": ["unclassified"], "W6": ["unclassifiable"]}),
    r("Geography / calendar", "nearest standard date / days from it", MECH, "master.terminal_date master.calendar_key",
      "loan_event_id", "closest learned date and its circular day distance", "Distance from normal timing.", "value", "",
      "W5 §2.10", "nearest_standard_mmdd, days_to_nearest_standard", "CAL", {"W5": ["nearest standard date"]}),
    r("Geography / calendar", "COVID flag", CLASS, "master.terminal_date", "loan_event_id",
      "2020-03-01 ≤ ending date ≤ 2021-02-28", "Ending in the pandemic window.", "flag",
      "Excluded from learning calendars only.", "W5 §2.4, W6 §1", "ending_in_covid_window", "CAL",
      {"W5": ["COVID-period"], "W6": ["Outside COVID"]}),
    r("Geography / calendar", "calendar × 60-day follow-on cell (A–D)", CLASS, "master.calendar_state master.follow_on_event_id",
      "loan_event_id", "A normal/no move ≤ 60d; B normal/move; C off/no move; D off/move", "Two-way grouping.", "category",
      "", "W5 §2.9, W6 §1", "— (loan_calendar_classified_v2.csv: two_way_cell)", "CAL",
      {"W5": ["60-day follow-on rule"], "W6": ["Timing + subsequent move"]}),
    r("Geography / calendar", "mapped-UEFA sample", FLAG, "duckdb.competitions.confederation duckdb.clubs.domestic_competition_id",
      "loan_event_id", "both clubs mapped (v1) to UEFA countries; loan season ≤ latest completed", "Earlier European sample.",
      "sample filter", "v1 geography.", "W4 §5", "—", "LTE", {"W4": ["mapped-UEFA"]}),
    r("Geography / calendar", "lender country (v1, Week 4)", JOIN, "duckdb.clubs.domestic_competition_id "
      "duckdb.competitions.country_name", "lender_club_id = club_id", "league country of the lender (v1)",
      "Country of the lending club.", "category", "Superseded by v2.", "W4 §6", "from_country_v1", "LE",
      {"W4": ["lender countries"]}),
    # playing time
    r("Playing time", "receiving league / covered league-season", JOIN, f"{GAMES} duckdb.games.competition_id "
      "stage1c.to_club_id stage1c.transfer_date", "to_club_id + season",
      "B's domestic league in the arrival season, if among the 14 leagues with appearance rows in every season",
      "League where playing time is measured.", "category", "", "W5 §4.1, W6 §3", "receiving_league, arrival_season",
      "PTJ", {"W5": ["covered league-season"], "W6": ["long-covered top divisions"]}),
    r("Playing time", "observation window start", RAW, "stage1c.transfer_date", "event_id", "= move date",
      "First day of the window.", "value (date)", "", "W5 §4.2", "window_start", "PTJ",
      {"W5": ["window begins on the recorded transfer date"]}),
    r("Playing time", "observation window end", MECH, "duckdb.games.date stage1c.transfer_date", "to_club_id + season; "
      "player's next row", "B's last league game of the season, or the day before the player's next recorded move",
      "Last day of the window.", "value (date)", "", "W5 §4.2, W6 §3", "window_end", "PTJ",
      {"W5": ["final league match of the season"]}),
    r("Playing time", "available_games", MECH, GAMES, "to_club_id + season + window",
      "count of B's domestic-league games with window_start ≤ date ≤ window_end", "Games the player could play.",
      "value", "", "W5 §4.2, W6 §3", "available_games", "PTJ", {"W5": ["league matches in the observation window"]}),
    r("Playing time", "appearances", MECH, APPS, "player_id + player_club_id + game_id",
      "count of the player's appearance rows for B in window games", "League appearances.", "value",
      "Written only when reliable.", "W4 §11", "appearances", "PTJ", {"W4": ["appearances table"]}),
    r("Playing time", "minutes", MECH, f"duckdb.appearances.minutes_played {APPS}", "player_id + player_club_id + game_id",
      "Σ appearances.minutes_played over window league games for B", "League minutes played.", "value",
      "minutes_played is the upstream field; we only sum it.", "W4 §11, W6 §3–4", "minutes", "PTJ",
      {"W4": ["minutes played"], "W6": ["minutes played"]}),
    r("Playing time", "available_minutes", MECH, "master.available_games", "event_id", "90 × available_games",
      "Minutes the club played in the window.", "value", "Extra time ignored.", "W5 §1", "available_minutes", "PTJ",
      {"W5": ["league minutes available"]}),
    r("Playing time", "minutes_share", MECH, "master.minutes master.available_games", "event_id",
      "minutes / (90 × available_games)", "Share of available league minutes played.", "value", "", "W5 §4.2, W6 §3",
      "minutes_share", "PTJ", {"W5": ["PTShare"], "W6": ["fraction of league minutes"]}),
    r("Playing time", "true zero playing time", CLASS, "master.minutes master.complete_games master.available_games",
      "event_id", "reliable (every window game has ≥ 11 appearance rows for B) and the player has no row → 0",
      "Real non-appearance.", "flag", "Missing coverage is never written as zero.", "W5 §4.2, §9; W6 §3", "true_zero", "PTJ",
      {"W5": ["true zero"], "W6": ["Zero playing time"]}),
    r("Playing time", "reliable outcome", FLAG, f"stage1c.transfermarkt_future_transfer master.receiving_league "
      f"master.available_games master.complete_games master.window_end {APPS}", "event_id",
      "none of: scheduled, not covered, player missing, no club rows, no window game, window after capture, any incomplete "
      "game, other-club appearance", "Outcome can be trusted.", "sample filter", "", "W5 Table 9", "pt_reliable", "PTJ",
      {"W5": ["Reliable playing-time outcome"]}),
    r("Playing time", "borrower appearance coverage during the loan", FLAG, f"{APPS} duckdb.games.date", "loan_event_id",
      "the borrowing club has appearance rows during the loan", "Earlier coverage check.", "flag", "",
      "W4 §11", "— (loan_follow_on_timing.csv)", "LTE", {"W4": ["appearance coverage"]}),
    r("Playing time", "prior_spell_at_b (first spell)", FLAG, "stage1c.player_id stage1c.to_club_id stage1c.transfer_date",
      "player_id + to_club_id", "an earlier loan or permanent candidate episode of the player into the same B",
      "Not the first spell at the club.", "sample filter", "", "W5 §4.3", "prior_spell_at_b", "PTJ",
      {"W5": ["first observed spell"]}),
    r("Playing time", "two-thirds rule / old primary sample", FLAG, "master.games_remaining_at_arrival master.season_games "
      "master.pt_reliable master.prior_spell_at_b master.pt_loan_ending", "event_id",
      "reliable & ≥ 2/3 of B's league games left & first spell & (permanent or return to lender)", "Week 5 sample.",
      "sample filter", "Rule removed in Week 6.", "W5 §4.1–4.3, W6 §3", "arrived_full_season, in_old_primary_sample", "PTJ",
      {"W5": ["two-thirds"], "W6": ["two-thirds"]}),
    r("Playing time", "revised-sample flag", FLAG, "master.pt_reliable master.prior_spell_at_b master.pt_loan_ending",
      "event_id", "reliable & first spell & (permanent or return to lender)", "Week 6 sample.", "sample filter", "",
      "W6 §3–4", "in_revised_sample", "RED", {"W6": ["revised sample"]}),
    r("Playing time", "arrived before club's first league game", MECH, "master.games_remaining_at_arrival "
      "master.season_games", "event_id", "games_remaining_at_arrival == season_games", "Pre-season arrival.", "flag", "",
      "W5 Table 13", "— (playing_time_regression_sample.csv)", "REG", {"W5": ["Arrived before club"]}),
    # market value
    r("Market value", "player_value", JOIN, PV, "player_id; latest date < move date",
      "latest player_valuations.market_value_in_eur dated strictly before the move", "Player value before the move.",
      "value", "Upstream value; the as-of selection is ours. Same-day valuations not used.", "W5 §6, W6 §3", "player_value",
      "REG", {"W5": ["player market value before move"], "W6": ["player market value"]}),
    r("Market value", "valuation age (stale)", MECH, "master.transfer_date master.player_value_date", "event_id",
      "move date − valuation date; stale if > 365 days", "Freshness of the valuation.", "value + flag", "",
      "W5 §12", "player_value_gap_days, stale_player_value", "REG", {"W5": ["older than one year"]}),
    r("Market value", "squad_value", MECH, f"{PV} duckdb.player_valuations.current_club_id",
      "current_club_id = to_club_id; date < move, ≥ move − 365 days",
      "sum of the latest pre-move valuations of players listed at B (within 365 days), excluding the moving player; "
      "missing if < 11 players", "Receiving squad's value.", "value", "OUR AGGREGATE of upstream valuations; "
      "clubs.total_market_value is empty.", "W5 §6, W6 §3", "squad_value", "REG",
      {"W5": ["receiving-club squad value"], "W6": ["receiving-club squad value"]}),
    r("Market value", "relative_value", MECH, "master.player_value master.squad_value", "event_id",
      "player_value / squad_value", "Player's size relative to the squad he joins.", "value", "", "W5 §6.2, W6 §3",
      "relative_value", "REG", {"W5": ["RelativeValue"], "W6": ["RelativeValue"]}),
    r("Market value", "log_relative_value", MECH, "master.player_value master.squad_value", "event_id",
      "ln(player_value) − ln(squad_value)", "Regression form.", "value", "", "W5 §7.1", "log_relative_value", "REG",
      {"W5": ["log relative player value"]}),
    r("Market value", "relative-value fifth", MECH, "master.log_relative_value", "event_id",
      "quintiles of log_relative_value in the model sample", "Groups for heterogeneity.", "category", "",
      "W5 §8.1", "— (regression code: lrv_q)", "REG", {"W5": ["relative-value fifth"]}),
    # player
    r("Player", "date of birth", JOIN, "duckdb.players.date_of_birth", "player_id", "copied", "Date of birth.", "value",
      "Current players-table value.", "W6 §4 (agreement check)", "date_of_birth", "S1", {"W6": ["dates of birth"]}),
    r("Player", "age", MECH, "duckdb.players.date_of_birth stage1c.transfer_date", "event_id",
      "(move date − date of birth) / 365.25; centred on each model sample", "Age at the move.", "value", "",
      "W5 §7.1, W6 §3", "age_exact", "REG", {"W5": ["age squared"], "W6": ["Median age"]}),
    r("Player", "young / older", CLASS, "master.age_exact", "event_id", "older = floor(age) ≥ 24; young ≤ 23",
      "Age group.", "flag", "Cutoff chosen in advance (22, 24 as checks).", "W6 §3", "older", "AGE",
      {"W6": ["Young (≤23)"]}),
    r("Player", "position", JOIN, "duckdb.players.position", "player_id", "copied; 'Missing' → blank",
      "Playing position.", "category", "", "W5 §7.1, W6 §3", "position, position_clean", "REG",
      {"W5": ["broad playing position"], "W6": ["position"]}),
    # injury
    r("Injury", "injury raw fields (from_date, end_date, injury_reason, days/games missed)", EXT,
      "kaggle_inj.player_id kaggle_inj.from_date kaggle_inj.end_date kaggle_inj.injury_reason kaggle_inj.days_missed "
      "kaggle_inj.games_missed", "player_id", "copied; exact duplicates dropped; problems flagged",
      "Listed injury spells.", "value", "Third-party file (Kaggle, CC0); its compilation method is undocumented; at most "
      "15 rows per player (inferred).", "W6 §4", "— (salimt_kaggle_injuries_normalized.csv)", "INJ",
      {"W6": ["143,195 raw rows"]}),
    r("Injury", "injury player profiles", EXT, "kaggle_prof.player_id kaggle_prof.date_of_birth", "player_id", "copied",
      "Players in the external datalake.", "value", "Third-party file; compilation undocumented.", "W6 §4",
      "injury_join_status (B)", "INJ", {"W6": ["player profile"]}),
    r("Injury", "injury source status (A/B/C)", MECH, "kaggle_inj.player_id kaggle_prof.player_id stage1c.player_id",
      "player_id (exact integer)", "A has injury rows; B profile only; C absent", "Coverage of the player.", "category",
      "B means no listed injury, not health.", "W6 §4", "injury_join_status", "INJ", {"W6": ["exact-ID coverage"]}),
    r("Injury", "date-of-birth agreement", MECH, "kaggle_prof.date_of_birth duckdb.players.date_of_birth", "player_id",
      "equal dates where both exist", "Id check.", "flag", "", "W6 §4", "—", "INJ", {"W6": ["dates of birth agree"]}),
    r("Injury", "medical-spell classification (primary / strict)", CLASS, "kaggle_inj.injury_reason", "injury_reason",
      "keyword rules: Rest/Fitness/Quarantine never; 'unknown injury' and minor muscular complaints primary only; "
      "illness and physical injury both", "Which listed spells count as medical.", "category",
      "Every label listed in injury_reason_classification.csv.", "W6 §4", "— (injury_reason_classification.csv)", "INJ",
      {"W6": ["medical injury or illness spell"]}),
    r("Injury", "confirmed_injury_games", CLASS, f"{KINJ} duckdb.games.date {APPS}", "player_id; game date in spell",
      "window games covered (inclusive) by ≥ 1 primary medical spell AND the player has no appearance row for B",
      "Games removed for injury.", "value", "Kaggle supplies the dates; the rule is ours.", "W6 §4", "confirmed_injury_games",
      "INJ", {"W6": ["InjuryGames"]}),
    r("Injury", "available_games_fit", MECH, "master.available_games master.confirmed_injury_games", "event_id",
      "available_games − confirmed_injury_games", "Games the player was fit for.", "value", "", "W6 §4",
      "available_games_fit", "INJ", {"W6": ["available games after injury adjustment"]}),
    r("Injury", "current_share", MECH, "master.minutes master.available_games", "event_id",
      "minutes / (90 × available_games)", "= minutes_share.", "value", "", "W6 §4", "current_share", "INJ",
      {"W6": ["Original share"]}),
    r("Injury", "fit_adjusted_share", MECH, "master.minutes master.available_games_fit", "event_id",
      "minutes / (90 × available_games_fit)", "Share of fit games played.", "value", "Never below current_share.",
      "W6 §4", "fit_adjusted_share", "INJ", {"W6": ["injury-adjusted version"]}),
    r("Injury", "injury_data_quality_flag", FLAG, f"master.injury_join_status master.window_start master.window_end {KINJ}",
      "event_id", "first of: not in datalake; window reaches 2025-09-15; history truncated; open spell; missing start; "
      "no fit games; else usable", "Why the adjustment is or is not computable.", "category", "", "W6 §4",
      "injury_data_quality_flag", "INJ", {"W6": ["15-injury history cap"]}),
    r("Injury", "injury-usable flag", FLAG, "master.injury_data_quality_flag", "event_id", "flag == 'usable'",
      "In the injury analysis.", "sample filter", "", "W6 §4", "in_injury_usable", "INJ", {"W6": ["Usable episodes"]}),
    r("Injury", "strict / ≥ 3-game / ≥ 5-game injury definitions", CLASS, f"{KINJ} duckdb.games.date {APPS}", "event_id",
      "strict labels only; or only spells removing ≥ 3 / ≥ 5 window games", "Sensitivity definitions.", "value", "",
      "W6 §4", "injury_games_strict, injury_games_severe3, injury_games_long5", "INJ", {"W6": ["strict definition"]}),
    r("Injury", "injury substitution event (validation)", JOIN, "duckdb.game_events.type duckdb.game_events.description "
      "duckdb.game_events.player_id duckdb.game_events.game_id", "player_id + game_id",
      "game_events type 'Substitutions' with 'Injury' in description; player_id = player going off",
      "In-match injury substitution.", "value", "Transfermarkt's own substitution reason.", "W6 §4",
      "— (injury_external_validation.csv)", "FEA", {"W6": ["injury-related"]}),
    r("Injury", "validation match within ±k days", MECH, "kaggle_inj.from_date duckdb.games.date", "player_id",
      "abs(nearest injury start − event date) ≤ k days", "Does the external data record the injury?", "flag", "", "W6 §4",
      "— (injury_external_validation.csv)", "INJ", {"W6": ["within±3 days"]}),
    r("Injury", "missed next two league games", CLASS, f"{APPS} duckdb.games.date", "event_id + game order",
      "no appearance in either of B's next two window league games", "Stronger validation subset.", "flag", "",
      "W6 §4", "— (injury_external_validation.csv: after_event)", "INJ", {"W6": ["next two league matches"]}),
    # regression
    r("Regression", "loan indicator", MECH, "master.kind", "event_id", "1 if loan, 0 if permanent", "Loan arrival.", "value",
      "", "W5 §7.1, W6 §3", "kind", "REG", {"W5": ["equals 1 for a loan"]}),
    r("Regression", "arrival timing", MECH, "master.games_remaining_at_arrival master.season_games", "event_id",
      "share of B's league games remaining at arrival", "Control for arrival time.", "value", "", "W5 §7.1, W6 §3",
      "remain", "REG", {"W5": ["arrival timing"], "W6": ["arrival timing"]}),
    r("Regression", "loan × older", MECH, "master.kind master.older", "event_id", "loan × older",
      "How the loan gap differs for older arrivals.", "value", "", "W6 §3–4", "— (product of kind and older)", "AGE",
      {"W6": ["Loan×older"]}),
    r("Regression", "league × season fixed effects", MECH, "master.receiving_league master.arrival_season", "event_id",
      "dummies for league_season", "Compare within league-season.", "value", "", "W5 §7.1, W6 §3", "league_season", "REG",
      {"W5": ["league-by-season"], "W6": ["league×season fixed effects"]}),
    r("Regression", "club × season fixed effects", MECH, "master.to_club_id master.arrival_season", "event_id",
      "absorbed dummies for receiving club × season", "Compare within club-season.", "value", "", "W5 §7.5",
      "— (regression code: club_season)", "REG", {"W5": ["Club×season FE"]}),
    r("Regression", "clustering by receiving club", MECH, "master.to_club_id", "event_id", "cluster-robust covariance by B",
      "Standard errors.", "value", "", "W6 §3", "to_club_id", "REG", {"W6": ["clustered by receiving club"]}),
    r("Regression", "regression-complete flags", FLAG, f"master.in_revised_sample master.in_injury_usable {COV}", "event_id",
      "all covariates present", "Regression observations.", "sample filter", "", "W5 §7.3, W6 §4",
      "in_old_regression_sample, in_age_model_sample, in_injury_model_sample", "REG",
      {"W5": ["market-value sample"], "W6": ["complete control variables"]}),
    r("Regression", "prior playing-time share", MECH, f"duckdb.appearances.minutes_played {APPS} {GAMES} stage1c.from_club_id",
      "from_club_id + previous season", "origin club's league minutes share in the season before, from the player's last "
      "move into it to the day before the move; only where every game is complete", "Pre-move playing time.", "value",
      "Measurable subsample only.", "W5 §10", "— (playing_time_prior_playing_time.csv: prior_share)", "RA",
      {"W5": ["prior playing-time share"]}),
    r("Regression", "nonzero / England subsamples", FLAG, "master.minutes master.receiving_league", "event_id",
      "minutes > 0; receiving_league == GB1", "Robustness subsets.", "sample filter", "", "W5 §9, §11; W6 §3",
      "minutes, receiving_league", "REG", {"W5": ["nonzero players"], "W6": ["England"]}),
    r("Regression", "regression coefficients and intervals", STAT, "master.minutes_share master.kind master.remain "
      "master.age_exact master.log_relative_value master.log_squad_value master.position_clean master.league_season",
      "event_id", "OLS (and logit / fractional-logit AMEs) with clustered standard errors",
      "Adjusted associations.", "model output", "Observational, not causal.", "W5 §7–13, W6 §3–4",
      "— (model output CSVs)", "REG", {"W5": ["Loan coefficient"], "W6": ["Pooled loan coefficient"]}),
]
REQUIRED = ["event_id", "player_id", "transfer date", "from club / to club (lender, borrower, receiving club)",
            "transfer type (fee label, normalised type)", "loan episode", "recorded loan ending (ending date, realised)",
            "loan duration", "follow-on move (60-day)", "economic_ending (production follow-on classes)", "early_termination",
            "club calendar country", "domestic / cross-border", "calendar key", "calendar-normal / off-calendar", "COVID flag",
            "observation window start", "observation window end", "available_games", "appearances", "minutes",
            "available_minutes", "minutes_share", "true zero playing time", "reliable outcome", "prior_spell_at_b (first spell)",
            "revised-sample flag", "player_value", "squad_value", "relative_value", "log_relative_value", "date of birth",
            "age", "young / older", "position", "injury source status (A/B/C)", "medical-spell classification (primary / strict)",
            "confirmed_injury_games", "available_games_fit", "current_share", "fit_adjusted_share", "injury_data_quality_flag",
            "injury-usable flag", "loan indicator", "arrival timing", "loan × older", "league × season fixed effects",
            "regression-complete flags"]


# ---------------------------------------------------------------------------
# LLM audit (static)
# ---------------------------------------------------------------------------

LLM_PATTERNS = {
    "LLM endpoint URL": re.compile(r"https?://[^\s\"']*(openai\.com|parley\.api|anthropic\.com|generativelanguage|mistral\.ai|cohere)"),
    "LLM SDK import": re.compile(r"^\s*(?:import|from)\s+(openai|anthropic|litellm|google\.generativeai|cohere|mistralai)\b", re.M),
    "LLM provider instantiated": re.compile(r"\b(ParleyProvider|OpenAIWebResearchProvider)\s*\("),
}
LLM_CACHE_READERS = re.compile(r"\b(token_profile|workload|scenarios|extraction_tokens)\s*\(|loan_scope\.scope\s*\(")


def module_graph() -> tuple[dict, dict]:
    mods = {}
    for f in (ROOT / "src").rglob("*.py"):
        parts = f.relative_to(ROOT).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        mods[name] = f
    graph = {}
    for name, f in mods.items():
        pkg = name if f.name == "__init__.py" else name.rsplit(".", 1)[0]
        deps = set()
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                deps |= {a.name for a in node.names if a.name.startswith("src")}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")[:len(pkg.split(".")) - (node.level - 1)]
                    mod = ".".join(base + ([node.module] if node.module else []))
                elif (node.module or "").startswith("src"):
                    mod = node.module
                else:
                    continue
                for a in node.names:
                    deps.add(f"{mod}.{a.name}" if f"{mod}.{a.name}" in mods else mod)
        for d in list(deps):                      # importing a module runs its parent packages too
            while "." in d:
                d = d.rsplit(".", 1)[0]
                deps.add(d)
        graph[name] = {d for d in deps if d in mods and d != name}
    return graph, mods


def closure(graph: dict, start: set) -> set:
    seen, todo = set(), list(start)
    while todo:
        m = todo.pop()
        if m not in seen:
            seen.add(m)
            todo += graph.get(m, ())
    return seen


def llm_audit() -> dict:
    graph, mods = module_graph()
    calls = {}
    for name, f in mods.items():
        txt = f.read_text()
        hits = sorted({lab for lab, pat in LLM_PATTERNS.items() if pat.search(txt)})
        if hits:
            calls[name] = hits
    entries = {"src." + c.replace("/", ".")[4:-3] for c in CODE.values()} | {"src.stage1c.enrichment", "src.stage1b.raw_api",
                                                                             "src.stage1b.club_flags", "src.analysis.data_lineage"}
    entries = {e for e in entries if e in mods}
    cl = closure(graph, entries)
    reached = sorted(cl & set(calls))
    importers = sorted(m for m in mods if any(c in graph[m] for c in calls))
    readers = {}
    for m in sorted(cl):
        txt = mods[m].read_text()
        found = sorted({x.group(0).rstrip("(").strip() for x in LLM_CACHE_READERS.finditer(txt)})
        if found and m != "src.analysis.loan_scope":
            readers[m] = found
    callers_in_producers = sorted(m for m in readers if m in entries)
    domains = sorted({d for m in cl for d in re.findall(r"https?://([A-Za-z0-9.-]+)", mods[m].read_text())})
    assert not any(LLM_PATTERNS["LLM endpoint URL"].search("https://" + d) for d in domains), domains
    return {"llm_call_modules": calls, "modules_importing_them": importers, "producing_modules": sorted(entries),
            "endpoint_domains_in_closure": domains,
            "closure_size": len(cl), "closure": sorted(cl), "llm_modules_reached": reached,
            "cache_reader_callers_in_closure": readers, "cache_reader_callers_among_producers": callers_in_producers,
            "network_modules_in_closure": sorted(m for m in cl if re.search(r"urlopen|requests\.(get|post)",
                                                                             mods[m].read_text()))}


# ---------------------------------------------------------------------------
# Source checks
# ---------------------------------------------------------------------------

def source_schemas() -> dict:
    import duckdb
    from ..config import DEFAULT_DATABASE

    con = duckdb.connect(str(DEFAULT_DATABASE), read_only=True)
    try:
        duck = {}
        for t, c in con.sql("select table_name, column_name from information_schema.columns").fetchall():
            duck.setdefault(t, set()).add(c)
    finally:
        con.close()
    with open(ROOT / "data" / "raw" / "transfermarkt-api" / "2025" / "transfers.json", encoding="utf-8") as fh:
        for line in fh:
            tr = (json.loads(line).get("response") or {}).get("transfers") or []
            if tr:
                rec = tr[0]
                break
    hdr = lambda path: set(pd.read_csv(path, nrows=0).columns)
    wiki = json.loads((ROOT / "data" / "raw" / "wikidata" / "transfermarkt_team_p7223.json").read_text())
    return {"duckdb": duck, "raw": rec, "stage1c": set(CANONICAL_COLUMNS), "master": hdr(dl.MASTER_CSV),
            "kaggle_inj": hdr(ROOT / "data/raw/external_injuries/salimt_kaggle/player_injuries/player_injuries.csv"),
            "kaggle_prof": hdr(ROOT / "data/raw/external_injuries/salimt_kaggle/player_profiles/player_profiles.csv"),
            "geo_v2": hdr(OUT / "club_geography_v2.csv"), "flags": hdr(OUT / "club_country_raw_flags.csv"),
            "wikidata": {k for v in wiki["queried"].values() for x in v for k in x}}


def exists(token: str, S: dict) -> bool:
    src, rest = token.split(".", 1)
    if src == "duckdb":
        t, c = rest.split(".", 1)
        return c in S["duckdb"].get(t, set())
    if src == "raw":
        node = S["raw"]
        for k in rest.split("."):
            if not isinstance(node, dict) or k not in node:
                return False
            node = node[k]
        return True
    return rest in S[src]


def render_sources(tokens: list[str]) -> tuple[str, str]:
    groups = {}
    for tk in tokens:
        src, rest = tk.split(".", 1)
        if src == "duckdb":
            t, c = rest.split(".", 1)
            groups.setdefault(f"DuckDB `{t}`", []).append(c)
        else:
            groups.setdefault(SOURCE_LABEL[src], []).append(rest)
    files = "; ".join(groups)
    cols = "; ".join(f"{g}: {', '.join(dict.fromkeys(c))}" for g, c in groups.items())
    return files, cols


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

CONSTRUCT = {RAW: "raw value (as published)", JOIN: "raw value from another table or row, attached by our join",
             EXT: "external raw value", MECH: "our construct", CLASS: "our construct", FLAG: "our construct",
             STAT: "model output"}


def llm_value(cat: str) -> tuple[str, str]:
    if cat == EXT:
        return "UNKNOWN", ("copied unchanged by our deterministic code (no LLM); the third party's own compilation "
                           "method is undocumented")
    if cat in (RAW, JOIN):
        return "NO", ("copied unchanged from Transfermarkt's published fields by deterministic code; no LLM-calling "
                      "module is imported by the producing code")
    return "NO", "computed by deterministic code; no LLM-calling module is imported by the producing code"


def weeks_text() -> dict:
    from pypdf import PdfReader
    norm = lambda s: re.sub(r"\s+", "", s).lower()
    return {w: norm(" ".join(pg.extract_text() for pg in PdfReader(str(ROOT / "data" / "pdf" / f"{f}.pdf")).pages))
            for w, f in PDFS.items()}


def build() -> tuple[pd.DataFrame, dict]:
    S = source_schemas()
    audit = llm_audit()
    assert not audit["llm_modules_reached"], audit["llm_modules_reached"]
    assert not audit["cache_reader_callers_among_producers"], audit["cache_reader_callers_among_producers"]
    master_cols = list(pd.read_csv(dl.MASTER_CSV, nrows=0).columns)
    D1 = pd.read_csv(OUT / "stage1c_column_dictionary.csv", keep_default_na=False).set_index("column")
    D2 = pd.read_csv(dl.VARDICT_CSV, keep_default_na=False).set_index("variable")
    master_spec = {c: MASTER_OWN.get(c) or STAGE1C.get(c) for c in master_cols}
    assert set(STAGE1C) == set(CANONICAL_COLUMNS) and None not in master_spec.values(), \
        [c for c, v in master_spec.items() if v is None]
    assert not set(MASTER_OWN) - set(master_cols), set(MASTER_OWN) - set(master_cols)
    shown = {}
    for x in REPORT:
        for m in re.split(r",\s*", x.master):
            m = m.split(" ")[0]
            if m in master_cols:
                shown.setdefault(m, []).append(x.where)
    rows = []

    def add(scope, var, spec, meaning, rule, keys, kind, caveat, where, master, group=""):
        files, cols = render_sources(spec.inputs)
        llm, basis = llm_value(spec.cat)
        rows.append({"scope": scope, "group": group, "variable": var, "provenance_category": spec.cat,
                     "usefulness_or_role": (USEFULNESS[MASTER_GUIDE[var].usefulness] if scope == "master" else
                                            S1C_ROLES[STAGE1C_GUIDE[var].role] if scope == "stage1c" else ""),
                     "origin": ORIGINS[MASTER_GUIDE[var].origin] if scope == "master" else "",
                     "category_note": spec.note, "actual_source_table_or_file": files, "actual_source_columns": cols,
                     "input_tokens": " ".join(spec.inputs), "join_keys": keys, "exact_transformation_or_rule": rule,
                     "plain_english_meaning": meaning, "raw_value_or_our_construct": CONSTRUCT[spec.cat],
                     "flag_filter_or_value": kind, "llm_inference_used_for_value": llm, "llm_audit_basis": basis,
                     "important_caveat": caveat, "where_used_in_reports": where, "master_column": master,
                     "producing_code": CODE[spec.code]})

    for c in CANONICAL_COLUMNS:
        d = D1.loc[c]
        add("stage1c", c, STAGE1C[c], d.plain_english_meaning, d.original_source, "event_id (row key)",
            "flag" if d.dtype == "bool" else "value", d.missing_value_meaning, "; ".join(shown.get(c, [])) or
            "not shown in Weeks 4–6", c if c in master_cols else "")
    for c in master_cols:
        d = D2.loc[c]
        add("master", c, master_spec[c], d.plain_english_definition, d.formula_or_rule, d.join_keys, d.is_filter_or_flag,
            d.important_caveat, "; ".join(shown.get(c, [])) or "not shown in Weeks 4–6", c, d.section)
    for x in REPORT:
        add("report", x.variable, P(x.cat, x.inputs, x.code, ""), x.meaning, x.rule, x.keys, x.kind, x.caveat, x.where,
            x.master, x.group)
    M = pd.DataFrame(rows)
    return M, {"audit": audit, "S": S}


def qa(M: pd.DataFrame, ctx: dict) -> dict:
    S = ctx["S"]
    out = {}
    s1 = M[M.scope == "stage1c"].variable
    ms = M[M.scope == "master"].variable
    assert list(s1) == CANONICAL_COLUMNS and s1.is_unique
    assert list(ms) == list(pd.read_csv(dl.MASTER_CSV, nrows=0).columns) and ms.is_unique
    rep = M[M.scope == "report"].variable
    assert rep.is_unique and not set(REQUIRED) - set(rep), set(REQUIRED) - set(rep)
    missing = sorted({tk for tks in M.input_tokens for tk in tks.split() if not exists(tk, S)})
    assert not missing, missing
    ours = M.provenance_category.isin([MECH, CLASS, FLAG, STAT, JOIN])
    assert M.loc[ours, "input_tokens"].str.len().gt(0).all()
    for c in set(CODE.values()):
        assert (ROOT / c).exists(), c
    W = weeks_text()
    found, notfound = {}, []
    for x in REPORT:
        for w, terms in x.terms.items():
            hit = [t_ for t_ in terms if re.sub(r"\s+", "", t_).lower() in W[w]]
            if hit:
                found.setdefault(w, []).append(x.variable)
            else:
                notfound.append(f"{x.variable} [{w}]")
    assert not notfound, notfound
    out["report_variables_found"] = {w: len(v) for w, v in found.items()}
    out["report_variables_without_terms"] = [x.variable for x in REPORT if not x.terms]
    out["tokens_checked"] = int(len({tk for tks in M.input_tokens for tk in tks.split()}))
    out["llm_yes"] = int(M.llm_inference_used_for_value.eq("YES").sum())
    return out


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

SHORT = {RAW: "source", JOIN: "joined", EXT: "external", MECH: "derived", CLASS: "our rule", FLAG: "sample flag",
         STAT: "model"}


def counts(M: pd.DataFrame, scope: str) -> dict:
    v = M[M.scope == scope].provenance_category.value_counts()
    return {c: int(v.get(c, 0)) for c in CATEGORIES}


def render(M: pd.DataFrame, ctx: dict, Q: dict) -> str:
    A = ctx["audit"]
    L = json.loads(dl.NUMBERS_JSON.read_text())
    geo, lo = L["geography"], L["loans"]
    c1, cm, cr = counts(M, "stage1c"), counts(M, "master"), counts(M, "report")
    n1, nm, nr = sum(c1.values()), sum(cm.values()), sum(cr.values())
    unk = M[M.llm_inference_used_for_value.eq("UNKNOWN")]
    W = ["# Variable provenance: what came from the source, and what we built", "",
         "*Generated by `python -m src.analysis.variable_provenance` from the lineage files, the raw-source schemas and a "
         "static scan of the code. Full detail for every column: `variable_provenance_matrix.csv`.*", "",
         "**Claude/AI assistance in writing code is not the same as AI-generated data. A value is marked AI-inferred only "
         "if an LLM response directly determined the stored value for an observation.**", "",
         "## The four answers", ""]
    eg = lambda cat, scope="master", k=6: ", ".join(f"`{v}`" for v in M[(M.scope == scope) & (M.provenance_category == cat)]
                                                   .variable.head(k))
    W += [f"1. **What came directly from the source.** Transfermarkt's own published fields, unchanged: "
          f"{c1[RAW]} of the {n1} Stage 1C columns (e.g. {eg(RAW, 'stage1c')}). From non-Transfermarkt sources: the Kaggle "
          "injury rows (start date, end date, injury label) and Wikidata countries for 2 clubs; neither is stored "
          "raw in the master table, they enter only through our rules.",
          f"2. **What we joined from other source tables.** Upstream values copied unchanged from another table or row "
          f"and attached by our key or link: {c1[JOIN]} Stage 1C columns (date of birth, position, citizenship) and "
          f"{cm[JOIN]} master columns (e.g. {eg(JOIN)}). The value is Transfermarkt's; the link is ours.",
          f"3. **What we constructed deterministically.** Everything else: {cm[MECH]} mechanical derivations, "
          f"{cm[CLASS]} classification rules and {cm[FLAG]} sample flags among the {nm} master columns. Classification "
          "rules are where our design choices live (thresholds, windows, priority orders, learned calendars).",
          f"4. **What, if anything, was LLM-inferred.** **Nothing.** No Stage 1C column, no master-table column and none "
          f"of the {nr} report variables has a value determined by an LLM response. The code that calls an LLM ("
          + ", ".join(f"`{m}`" for m in A["llm_call_modules"]) + ") belongs to the Stage 2 contract research and is "
          f"imported by none of the {len(A['producing_modules'])} modules that produce these variables, nor by anything "
          f"they import ({A['closure_size']} modules checked). The only marks other than NO are {len(unk)} rows marked "
          "UNKNOWN: the Kaggle injury fields and the Wikidata country. Our code copies them without any LLM, but the "
          "third parties do not document how they compiled them.", ""]
    W += ["## Counts by category", ""]
    W += ["| Category | Meaning | Stage 1C columns | Master columns | Report variables |", "| --- | --- | ---: | ---: | ---: |"]
    W += [f"| {k} | {v} | {c1[k]} | {cm[k]} | {cr[k]} |" for k, v in CATEGORIES.items()]
    W += [f"| **Total** | | **{n1}** | **{nm}** | **{nr}** |", ""]
    W += ["## Report variables (Weeks 4–6)", "",
          "*source* = published by Transfermarkt; *joined* = Transfermarkt value attached by our link; *external* = Kaggle / "
          "Wikidata; *derived* = our deterministic calculation; *our rule* = our designed classification; *sample flag* = "
          "eligibility. LLM-inferred: none.", ""]
    rep = M[M.scope == "report"]
    for g, gd in rep.groupby("group", sort=False):
        W += [f"**{g}**", "", "| Variable | Type | Comes from | Our step |", "| --- | --- | --- | --- |"]
        for x in gd.itertuples():
            W += [f"| {x.variable} | {SHORT[x.provenance_category]} | {x.actual_source_columns.replace('|', chr(92) + '|')} | "
                  f"{x.exact_transformation_or_rule.replace('|', chr(92) + '|')} |"]
        W += [""]
    sources = geo["sources_used"]
    fun = lo["funnel"]
    W += ["## The chains, column by column", "", "### A. Playing time", "", "```",
          "Stage 1C:     player_id, to_club_id (= receiving club B), transfer_date (= window start)",
          "  + games:    game_id, competition_type = 'domestic_league', season, date, home_club_id, away_club_id",
          "  + appearances: game_id, player_id, player_club_id, minutes_played   <- the upstream minutes field",
          "      |  window = transfer_date .. B's last league game (or day before the player's next move)",
          "      v",
          "available_games = count of B's league games in the window                      [derived]",
          "minutes         = sum of appearances.minutes_played (player, B, window games)  [derived from upstream]",
          "minutes_share   = minutes / (90 x available_games)                             [derived]",
          "true_zero       = reliable AND minutes = 0                                     [our rule]",
          "```", "",
          "`appearances` has a row only for a player who played. A missing row becomes **0** only if **every** window game "
          "has a complete record for B (at least 11 appearance rows), at least one window game exists, the window ends "
          "before the player's history was captured, and he has no appearance for another club in the window. If any "
          "window game is incomplete, the episode is **unreliable**: minutes and share stay blank and the episode is "
          "excluded. Absence is never read as zero outside that rule.", "",
          "### B. Injury", "", "```",
          "KAGGLE (external):  player_injuries.player_id, injury_reason, from_date, end_date",
          "                    player_profiles.player_id (status B = in the data, no injury listed)",
          "  + games (upstream): date of each B league game in the window",
          "  + appearances (upstream): did the player appear for B in that game",
          "      |  OUR RULES: injury_reason -> medical? (Rest/Fitness/Quarantine never; keyword classes)",
          "      |             spell covers the game if from_date <= date <= end_date (inclusive)",
          "      |             an appearance always overrides; overlapping spells count once",
          "      v",
          "confirmed_injury_games = covered by a medical spell AND no appearance            [our rule]",
          "available_games_fit    = available_games - confirmed_injury_games                [derived]",
          "fit_adjusted_share     = minutes / (90 x available_games_fit)                    [derived]",
          "```", "",
          "From Kaggle: the player id, the dates and the injury label, nothing else (the source's own 'games missed' is "
          "not used). Everything that turns them into removed games is our rule.", "",
          "### C. Country", "", "```",
          "club_id",
          "  -> 1. league / competition country (DuckDB clubs.domestic_competition_id -> competitions.country_name,",
          "        or the country of a domestic competition the club played in)          "
          f"[{f0(sources.get('v1 league/competition geography', 0))} clubs]",
          "  -> 2. else Transfermarkt's own countryFlag on the club's raw transfer rows,",
          "        if every row shows one flag (flag id -> countries.country_name)       "
          f"[{f0(sources.get('raw Transfermarkt countryFlag', 0))} clubs]",
          "  -> 3. else fallbacks: competition membership, validated parent-club link,",
          f"        Wikidata P7223 -> P17                                                [{sum(v for k, v in sources.items() if k.startswith('fallback'))} clubs]",
          "  => calendar_country (blank if still unresolved)",
          "",
          "lender calendar_country + borrower calendar_country",
          "  -> same country: domestic; both known and different: cross-border; else unmapped",
          "  -> calendar_key ('domestic: <country>' / 'cross-border')",
          "  -> ending date within +/-3 days of a learned date of that calendar? calendar-normal : off-calendar",
          "```", "", "**No LLM guesses a club country**, and no country is inferred from a club name. Clubs whose raw "
          "rows show conflicting flags keep the conflict on record; only a listed fallback can choose, and only among "
          "the observed flags.", "",
          "### D. Early termination", "",
          f"**{ETR}**", "",
          "Source columns: the End-of-loan row's `transfer_date` and `transfer_type_normalized` (Stage 1C); the player's "
          "next Stage 1C row: `transfer_date`, `transfer_type_normalized`, `to_club_id`, `involves_placeholder_club`, "
          "`is_internal_move`; the loan's `to_club_id` (borrower). Rules, in order, with the loans left after each:", ""]
    W += [f"1. {k}: {f0(v)}" for k, v in fun.items()]
    W += ["", "Priority: a permanent move back to the borrower is classed as a conversion and a paid or undisclosed-fee "
          "move to a third club as a third-party sale **before** early_termination is considered. The 60-, 21- and 10-day "
          "constants are ours (from the Stage 2 event-family module).", "",
          "### E. Market value", "", "```",
          "player_valuations.player_id, date, market_value_in_eur        (upstream values)",
          "  -> latest row with date < move date                          => player_value     [joined: upstream value]",
          "player_valuations.player_id, date, market_value_in_eur, current_club_id   for players listed at B",
          "  -> each player's latest row before the move, within 365 days, listing B; minus the moving player;",
          "     sum; blank if fewer than 11 players                      => squad_value      [OUR AGGREGATE]",
          "player_value / squad_value                                     => relative_value   [derived]",
          "ln(player_value) - ln(squad_value)                             => log_relative_value [derived]",
          "```", "",
          "## Provenance that remains uncertain", "",
          "- `transfermarkt_future_transfer`: Transfermarkt's flag; its exact meaning is undocumented (we read it as "
          "'not yet happened at capture').",
          "- `market_value_eur` (Stage 1C): upstream does not document how the transfer-row value is set; it is not the "
          "regression player value.",
          "- Kaggle injury rows and Wikidata countries: copied unchanged, but the third parties' compilation methods are "
          "undocumented (hence UNKNOWN, not NO). The 15-row injury cap is inferred from the data.",
          "- `arrival_season` approximates the previous season's end for a league's first snapshot season.", "",
          "## QA", "",
          f"- Every Stage 1C column appears exactly once ({n1}); every master-table column exactly once ({nm}).",
          f"- {nr} report variables, including every variable you listed; each one with a search term was found in the "
          "report text it cites (" + ", ".join(f"{w} {n}" for w, n in sorted(Q["report_variables_found"].items())) + " "
          "variables).",
          f"- Every input column named ({Q['tokens_checked']} distinct) exists in its source: the DuckDB schema, a raw "
          "transfer-history record, Stage 1C, the master table, the Kaggle files, the geography files or the Wikidata cache.",
          "- Every derived, joined, rule, flag and model variable names its input columns.",
          f"- LLM audit: modules that call an LLM: {', '.join(A['llm_call_modules'])}; reached from the producing code: "
          f"{len(A['llm_modules_reached'])}. The Stage 2 cache reader `loan_scope.token_profile` (token-cost estimates) "
          "is called only by " + ", ".join(A["cache_reader_callers_in_closure"]) + ", which write Stage 2 scope "
          "documents, not variables. Modules in the producing code that can open a network connection: "
          + ", ".join(A["network_modules_in_closure"]) + ". Every URL anywhere in the producing code points to "
          + ", ".join(f"`{d}`" for d in A["endpoint_domains_in_closure"])
          + " (the upstream project's public file bucket, the Kaggle page cited as the injury source, and the Wikidata "
          "API); none is an LLM service.",
          f"- YES values: {Q['llm_yes']}. No data file or existing output was modified (all files under `data/` hashed "
          "before and after).", ""]
    return "\n".join(W)


def main() -> dict:
    before = dl.watched()
    M, ctx = build()
    Q = qa(M, ctx)
    M.to_csv(MATRIX_CSV, index=False)
    OVERVIEW_MD.write_text(render(M, ctx, Q))
    after = dl.watched()
    changed = sorted(k for k in before if before[k] != after.get(k) and not k.endswith(tuple(p.name for p in WRITES)))
    assert not changed, changed
    A = ctx["audit"]
    print(f"rows: {len(M)} ({', '.join(f'{s} {n}' for s, n in M.scope.value_counts().items())}); "
          f"LLM modules reached: {A['llm_modules_reached']}; YES: {Q['llm_yes']}; "
          f"UNKNOWN: {int(M.llm_inference_used_for_value.eq('UNKNOWN').sum())}")
    for s in ("stage1c", "master", "report"):
        print(s, counts(M, s))
    return {"Q": Q, "audit": A}


if __name__ == "__main__":
    main()
