"""Hand-written definitions for the data-lineage documentation (`data_lineage.py`).

Every definition here was traced from the code named in it. Counts, dtypes, examples and value sets are NOT
written here: `data_lineage.py` computes them from the current files and fills them in.
"""
from __future__ import annotations

from collections import namedtuple

# ---------------------------------------------------------------------------
# Stage 1C columns (all 42, in file order)
# ---------------------------------------------------------------------------

C = namedtuple("C", "raw_or_derived origin meaning missing zero tokens used")

ID = "Transfermarkt id"
S1 = "Stage 1 backbone (src/stage1/backbone.py)"
S1B = "Stage 1B parse of the raw Transfermarkt `fee` text (src/stage1b/parse.py)"
POL = "Stage 1C policy (src/stage1c/policy.py, build.py)"
RAWF = "upstream raw Transfermarkt transfer-history files (data/raw/transfermarkt-api/<season>/transfers.json)"
DUCK = "DuckDB snapshot table"

STAGE1C = {
    "event_id": C("derived", S1 + ": 'tm_' + first 20 hex characters of sha256('player_id|transfer_date|from_club_id|"
                  "to_club_id')", "Stable identifier of one registration row.", "Never missing.", "n/a", "none",
                  "Key of every episode-level output (episode `event_id`, loan `loan_event_id`, terminal and follow-on ids)."),
    "transfermarkt_transfer_id": C("raw", RAWF + ", the `/transfer_id/N` part of the row's `url`",
                                   "Transfermarkt's own id for the transfer row.", "Never missing (all rows matched).",
                                   "n/a", "none", "Tie-breaker in the within-day sort order of `loan_episodes._sorted` only."),
    "event_chain_id": C("derived", S1 + " `add_event_chains`: a row joins the previous row's chain if it is the same "
                        "player, leaves the club the previous row arrived at, and is dated within 14 days",
                        "Groups immediately continuing movements (B→A then A→C).", "Never missing.", "n/a", "none",
                        "Not used by the loan, calendar, playing-time or injury analyses."),
    "player_id": C("raw", DUCK + " `transfers.player_id` (" + ID + ")", "The player.", "Never missing.", "n/a", "none",
                   "Join key to `players`, `appearances`, `player_valuations` and the Kaggle injury files."),
    "player_name": C("raw", DUCK + " `transfers.player_name`", "Player name as on the transfer row.",
                     "Never missing.", "n/a", "none", "Display only. Not unique: never join on it."),
    "date_of_birth": C("derived (joined)", DUCK + " `players.date_of_birth`, joined on player_id", "Date of birth.",
                       "Blank: the `players` table has no date of birth for the player.", "n/a", "blank only",
                       "Regression age `age` = (move date − date_of_birth) / 365.25 (playing_time_regression.build)."),
    "age_at_transfer": C("derived", S1 + " `age_at_transfer`: completed years on transfer_date (birthday rule)",
                         "Age in completed years at the move.", "Blank when date_of_birth is blank.",
                         "n/a (no player is 0)", "blank only",
                         "Descriptive only (calendar v2 selection table, country-mapping bias table). NOT the regression age, which is "
                         "continuous, from date_of_birth."),
    "position": C("derived (joined)", DUCK + " `players.position`, joined on player_id",
                  "Goalkeeper / Defender / Midfield / Attack.", "The literal upstream string 'Missing' (not blank) means "
                  "no position recorded.", "n/a", "the string 'Missing'",
                  "Regression control `position_clean` ('Missing' → blank, so the episode leaves the controlled models)."),
    "country_of_citizenship": C("derived (joined)", DUCK + " `players.country_of_citizenship`", "Citizenship.",
                                "Never blank in this file.", "n/a", "none", "Not used by the analyses documented here."),
    "transfer_date": C("raw", DUCK + " `transfers.transfer_date` (also the enrichment join key)",
                       "Date Transfermarkt records for the registration movement.", "Never missing.", "n/a", "none",
                       "Loan start and end dates, episode move date, playing-time window start. Can lie in the future "
                       "(scheduled moves, mostly end-of-loan rows)."),
    "transfer_season": C("raw", DUCK + " `transfers.transfer_season`", "Transfermarkt season label 'YY/YY' of the move.",
                         "Never missing.", "n/a", "none", "Converted to 'YYYY/YY' (`season` / `loan_season`)."),
    "from_club_id": C("raw", DUCK + " `transfers.from_club_id` (" + ID + ")", "Club the player leaves (lender for a loan).",
                      "Never missing.", "n/a", "none", "Loan pairing, geography, playing-time origin club `a_club_id`."),
    "from_club_name": C("raw", DUCK + " `transfers.from_club_name`", "Name of that club as on the row.", "Never missing.",
                        "n/a", "placeholder names ('Without Club', 'Retired', 'Career break', 'Ban', 'Unknown')",
                        "Youth/reserve and internal-move rules, placeholder detection."),
    "to_club_id": C("raw", DUCK + " `transfers.to_club_id` (" + ID + ")", "Club the player joins (borrower for a loan).",
                    "Never missing.", "n/a", "none", "Receiving club B in the playing-time analysis (`b_club_id`)."),
    "to_club_name": C("raw", DUCK + " `transfers.to_club_name`", "Name of that club as on the row.", "Never missing.",
                      "n/a", "placeholder names as above", "As from_club_name."),
    "transfer_type_raw": C("derived", S1B + " `classify_label`: exact text rules on the raw `fee` string",
                           "Transfermarkt's own label for the row, kept as a stable code.", "Never missing ('unknown' "
                           "when the text matches no rule).", "n/a", "codes no_fee_shown and undisclosed stand for '-' "
                           "and '?'", "Stage 1C build only: source of transfer_type_normalized and of the fee routing."),
    "transfer_type_normalized": C("derived", POL + " `classify_type`: raw label → type; an uninformative label "
                                  "('-', '?', unknown) on an internal move becomes youth_or_internal",
                                  "Movement type used by every later stage.", "Never missing.", "n/a", "none",
                                  "Row roles, loan/return pairing, episode_type, permanent kinds."),
    "transfer_type_source": C("derived", POL, "Whether the type came from the raw label or the club-name rule.",
                              "Never missing.", "n/a", "none", "Provenance only."),
    "stage1_heuristic_class": C("derived", "Stage 1 timeline heuristic (src/stage1/movement.py) on the DuckDB fee and "
                                "the player's row sequence", "Pre-label guess at the movement type.", "Never missing.",
                                "n/a", "none", "Kept for comparison only; overruled by the raw label everywhere."),
    "heuristic_agrees_with_label": C("derived", POL, "Does the heuristic agree with the raw label?",
                                     "Blank: not comparable (placeholder / retirement / same-club classes).", "n/a",
                                     "blank = not comparable", "Audit only."),
    "fee_display_raw": C("raw", RAWF + ", the transfer's `fee` string, byte-exact (may contain HTML)",
                         "What Transfermarkt displays in the fee column.",
                         "Blank (3 rows): the raw row had no fee text (label 'unknown').", "'0' appears twice, label "
                         "'unknown' (not a parsed zero fee)", "'?' = undisclosed fee; '-' = no fee shown; 'free transfer'; "
                         "'End of loan'; 'loan transfer'; 'Loan fee:<br/>…€x'; 'End of loan<br/>…€x'; '€1.50m'",
                         "Loan-ending `raw_label_class`, `terminal_raw_label`, `next_move_raw_label`."),
    "fee_numeric_eur": C("derived", S1B + " `parse_amount_eur`; 0.0 for 'free transfer'",
                         "Euro amount parsed from the fee text.", "Blank: the text states no amount ('-', '?', 'End of "
                         "loan', 'loan transfer', 'draft').", "Yes, but only as Transfermarkt's 'free transfer' assertion",
                         "none", "Stage 1C build only: source of the three routed fee fields."),
    "fee_role": C("derived", POL, "Which kind of money the amount is.", "Blank: no amount role (no fee shown, "
                  "undisclosed, bare loan or return, draft).", "n/a", "none", "Provenance only."),
    "fee_disclosure_status": C("derived", POL + " `FEE_DISCLOSURE`", "What the source said about the fee, separate from "
                               "the amount.", "Never missing ('unknown' = no raw label).", "n/a",
                               "disclosed_free = asserted zero; undisclosed = '?'; no_fee_shown = '-' or bare loan",
                               "Descriptive."),
    "permanent_transfer_fee_eur": C("derived", POL + " `split_fees`", "Permanent transfer fee.",
                                    "Blank: not a paid or free permanent move.", "Yes: 0.0 only for 'free transfer'",
                                    "none", "Follow-on fee kind (`terminal_permanent_fee_eur`), has_monetary_amount."),
    "loan_fee_eur": C("derived", POL + " `split_fees`", "Fee for the outbound loan.",
                      "Blank: no loan fee published. A bare 'loan transfer' is blank, never 0.", "No zeros occur",
                      "none", "`outbound_loan_fee_eur` on the loan spell; fee-bearing-loan share in calendar tables."),
    "fee_on_return_eur": C("derived", POL + " `split_fees`", "Money attached to an End-of-loan row.",
                           "Blank: no amount on the return row.", "No zeros occur", "none",
                           "`ending_fee_on_return_eur`; economic_ending `fee_bearing_return`."),
    "duckdb_transfer_fee": C("raw", DUCK + " `transfers.transfer_fee`", "The upstream (dcaribou) numeric fee.",
                             "Blank: upstream null (all '-' and '?' rows).", "NOT meaningful for loans, returns, drafts "
                             "or free transfers: upstream writes 0 for every non-numeric label", "none",
                             "Audit only (`fee_matches_duckdb`). Never used as a fee."),
    "fee_matches_duckdb": C("derived", POL, "Does the upstream fee equal our parsed permanent fee?",
                            "Blank: not comparable (label is not paid_transfer or free_transfer).", "n/a", "blank",
                            "Audit only."),
    "market_value_eur": C("raw", DUCK + " `transfers.market_value_in_eur` (the value Transfermarkt shows on the "
                          "transfer row)", "Player market value at the transfer, as recorded by upstream.",
                          "Blank: no value shown ('-').", "No zeros occur", "none",
                          "Descriptive and audit only (calendar v2 selection table; compared with the regression player "
                          "value). Never a model input."),
    "market_value_display": C("raw", RAWF + ", the transfer's `marketValue` string", "Displayed market value text.",
                              "Never blank; '-' = no value shown.", "n/a", "'-' = no value", "Display only."),
    "is_loan_related": C("derived", POL, "Label is a loan or an end of loan.", "Never missing.", "n/a", "none",
                         "Descriptive."),
    "return_leg_has_fee": C("derived", POL, "An End-of-loan row with a positive fee.", "Never missing.", "n/a", "none",
                            "Stage 2 research population."),
    "transfermarkt_future_transfer": C("raw", RAWF + ", the transfer's `futureTransfer` flag",
                                       "Transfermarkt marked the move as not yet happened when the history was captured.",
                                       "Never missing (False when the flag is absent).", "n/a", "none",
                                       "`scheduled` (playing time), `terminal_scheduled_future` / `ending_realised` "
                                       "(calendar population)."),
    "involves_placeholder_club": C("derived", POL + ": either club name is Without Club / Retired / Career break / Ban / "
                                   "Unknown", "Not a club-to-club movement.", "Never missing.", "n/a", "none",
                                   "Row role `placeholder_non_club_movement`; follow-on stops at a release."),
    "is_youth_or_reserve_side": C("derived", POL + " regex `YOUTH_OR_RESERVE` on either club name (U17–U23, Youth, "
                                  "Jgd., Academy, Castilla, Res., names ending II / III / B / C)",
                                  "A youth or reserve side is involved.", "Never missing.", "n/a", "none",
                                  "Loan flag `loan_is_youth_or_reserve_side`; calendar selection table."),
    "is_internal_move": C("derived", POL + ": names equal after stripping the youth/reserve suffix, ids differ, no "
                          "placeholder", "Movement between sides of one organisation.", "Never missing.", "n/a", "none",
                          "Row role `internal_registration` (highest priority); never ends a loan spell."),
    "is_senior_move": C("derived", POL + ": not placeholder, not youth/reserve, not internal", "Senior-club move.",
                        "Never missing.", "n/a", "none", "Stage 2 research rule only."),
    "has_monetary_amount": C("derived", POL + ": any routed fee > 0", "Some money is published.", "Never missing.",
                             "n/a", "none", "Stage 2 research rule only."),
    "enrichment_matched": C("derived", "Stage 1C build: the backbone row matched a raw Transfermarkt row on (player, "
                            "date, from club, to club)", "Raw label available.", "Never missing (all True).", "n/a",
                            "none", "QA only."),
    "is_research_target": C("derived", POL + " `research_decision`", "In the Stage 2 contract-research population.",
                            "Never missing.", "n/a", "none", "Stage 2 only; not a filter in any playing-time analysis."),
    "research_exclusion_reason": C("derived", POL + " `research_decision`", "Why a row is not a Stage 2 target.",
                                   "Blank: the row IS a research target.", "n/a", "blank = target", "Stage 2 only."),
}

# ---------------------------------------------------------------------------
# Source inventory
# ---------------------------------------------------------------------------

Src = namedtuple("Src", "name kind path origin grain keys used_columns used_by limitations")

DB = "data/raw/transfermarkt-datasets.duckdb"
UP = ("dcaribou/transfermarkt-datasets DuckDB snapshot (upstream commit in the `version` table), downloaded "
      "2026-08-28; not scraped by us")

SOURCES = [
    Src("transfers", "duckdb", DB, UP, "one registration movement of one player (Transfermarkt transfer-history row)",
        "none unique; (player_id, transfer_date, from_club_id, to_club_id) after Stage 1",
        "all 10 columns", "Stage 1 backbone → Stage 1C (every row)",
        "Only players whose transfer history upstream captured (players on recent squads); not a census of all "
        "transfers. Fee column destroys loan fees (writes 0)."),
    Src("players", "duckdb", DB, UP, "one player", "player_id",
        "player_id, name, date_of_birth, position, country_of_citizenship (sub_position is joined by Stage 1 but "
        "not kept in Stage 1C)",
        "Stage 1 backbone (joined attributes); playing-time join (`player_not_in_universe` check); injury id check "
        "(name, date_of_birth)", "Attributes are current values, not as of the move."),
    Src("clubs", "duckdb", DB, UP, "one club with a current domestic league", "club_id",
        "club_id, domestic_competition_id", "Geography v1 (club → league → country)",
        "Only clubs in covered competitions. `total_market_value` is NULL for every club, so no club squad value can "
        "come from here."),
    Src("countries", "duckdb", DB, UP, "one Transfermarkt country", "country_id",
        "country_id, country_name, confederation", "Geography v2: names the raw countryFlag ids",
        "Ids missing from this table are kept as 'Transfermarkt country N'."),
    Src("competitions", "duckdb", DB, UP, "one competition", "competition_id",
        "competition_id, country_name, confederation, type", "Geography v1 (league country); covered-league logic",
        "International competitions have no country."),
    Src("games", "duckdb", DB, UP, "one match", "game_id",
        "game_id, competition_id, competition_type, season, date, home_club_id, away_club_id (names for the worked "
        "examples)", "Playing-time windows and denominators; geography v1 participation; injury window games",
        "Domestic leagues are complete only where appearance rows exist (14 leagues from 2012/13)."),
    Src("appearances", "duckdb", DB, UP, "one player in one match for one club (only players who appeared)",
        "appearance_id; (game_id, player_id)", "game_id, player_id, player_club_id, minutes_played, date, "
        "competition_id (checks), red_cards (injury feasibility)",
        "Minutes numerator; club-game completeness (>= 11 rows); appearance override in the injury rule",
        "No row for players who did not appear: absence is only a zero where the club-game has a complete record."),
    Src("game_lineups", "duckdb", DB, UP, "one player named for one club in one match", "game_lineups_id",
        "game_id, club_id, player_id, type", "Playing-time join lineup check; injury feasibility (squad evidence)",
        "Not used in the outcome. Absence from a squad is never read as injury."),
    Src("game_events", "duckdb", DB, UP, "one in-match event (goal, card, substitution, shootout)", "game_event_id",
        "game_id, club_id, player_id, type, description", "Injury validation: substitutions with 'Injury' in the "
        "description (`player_id` = the player going off)", "Substitution reasons are often blank or 'Not reported'."),
    Src("player_valuations", "duckdb", DB, UP, "one market valuation of one player on one date",
        "(player_id, date)", "player_id, date, market_value_in_eur, current_club_id",
        "Regression player value (latest strictly before the move) and squad value (valuations listing B)",
        "Valuations are irregular; `current_club_id` is the club on the valuation date."),
    Src("club_games", "duckdb", DB, UP, "one club's side of one match", "(game_id, club_id)", "none",
        "Not used", ""),
    Src("national_teams", "duckdb", DB, UP, "one national team", "national_team_id", "none", "Not used", ""),
    Src("version", "duckdb", DB, UP, "the upstream commit", "commit_hash", "commit_hash", "Provenance only", ""),
    Src("raw transfer-history season files", "jsonl",
        "data/raw/transfermarkt-api/{2023,2024,2025}/transfers.json",
        "upstream raw acquisition files (same R2 bucket as the DuckDB, DVC md5 pinned by its commit), cached once; "
        "no transfermarkt.com requests",
        "one line = one player's full transfer-history response (JSON)", "player_id; inside: transfer url / ids",
        "fee, dateUnformatted, from/to href, url, marketValue, futureTransfer, countryFlag",
        "Stage 1C enrichment (raw fee label, transfer id, future flag); geography v2 raw countryFlag",
        "The most recent season file holding a player is the one used (Stage 1B / 1C rule)."),
    Src("stage1_normalized_transfers.csv", "csv", "data/outputs/rebuild/stage1_normalized_transfers.csv",
        "Stage 1 backbone (src/stage1/backbone.py)", "one DuckDB transfer row, chain-ordered", "event_id",
        "raw_* columns, joined_* attributes, derived_* heuristics", "Input to Stage 1C", ""),
    Src("stage1c_transfermarkt_enrichment.csv", "csv", "data/outputs/rebuild/stage1c_transfermarkt_enrichment.csv",
        "Stage 1C enrichment cache (src/stage1c/enrichment.py) from the raw season files",
        "one raw transfer row", "(raw_player_id, join_date, raw_from_club_id, raw_to_club_id)",
        "every column", "Input to Stage 1C (one-to-one join)", ""),
    Src("stage1c_canonical_transfers.csv", "csv", "data/outputs/rebuild/stage1c_canonical_transfers.csv",
        "Stage 1C build (src/stage1c/build.py); frozen (SHA-256 checked by every phase)",
        "one registration movement (= one DuckDB transfers row)", "event_id",
        "all 42 (see stage1c_column_dictionary.csv)", "Every analysis", "Not one row per contract."),
    Src("club_country_raw_flags.csv", "csv", "data/outputs/rebuild/club_country_raw_flags.csv",
        "src/stage1b/club_flags.py from the raw season files", "one club id seen in the raw files",
        "club_id", "club_id, flag_ids, mapping_status, n_supporting_rows", "Geography v2 precedence 2", ""),
    Src("club_geography_v2.csv", "csv", "data/outputs/rebuild/club_geography_v2.csv",
        "src/analysis/club_geography_v2.py", "one club (Stage 1C clubs plus every v1-mapped club)", "club_id",
        "club_id, calendar_country, source, status, confederation",
        "Calendar v2 (lender / borrower country)", "Not used by any playing-time or injury analysis."),
    Src("Wikidata cache", "json", "data/raw/wikidata/transfermarkt_team_p7223.json",
        "Wikidata Action API (P7223 Transfermarkt team id → P17 country), queried once for residual clubs",
        "one queried club id", "club id", "item, country", "Geography v2 fallback C", "10 clubs queried."),
    Src("player_injuries.csv (Kaggle salimt v2)", "csv",
        "data/raw/external_injuries/salimt_kaggle/player_injuries/player_injuries.csv",
        "Kaggle xfkzujqjvx97n/football-datasets v2 (CC0, 2025-09-15), downloaded manually by the user; sourced "
        "from Transfermarkt by its author", "one listed injury / absence spell of one player",
        "none unique; (player_id, from_date) mostly", "all 7", "Injury-adjusted playing time; validation",
        "Snapshot 2025-09-15; at most 15 rows per player (inferred cap); 2025/26 incomplete."),
    Src("player_profiles.csv (Kaggle salimt v2)", "csv",
        "data/raw/external_injuries/salimt_kaggle/player_profiles/player_profiles.csv", "as above",
        "one player in the external datalake", "player_id",
        "player_id, player_name, date_of_birth", "Injury join status B (in the datalake, no injury rows); id check",
        "Not a list of injury-free players: a profile with no injury rows means no listed injury, not health."),
    Src("salimt_kaggle_injuries_normalized.csv", "csv",
        "data/processed/external_injuries/salimt_kaggle_injuries_normalized.csv",
        "playing_time_injury_adjusted.normalize (exact duplicates dropped, problems flagged)",
        "one injury row after exact-duplicate removal", "source_row", "all", "Injury worked examples", ""),
]

# ---------------------------------------------------------------------------
# Loan-spell fields (loan_episodes.SPELL_COLUMNS, all 35)
# ---------------------------------------------------------------------------

LOAN_FIELDS = {
    "loan_event_id": ("Stage 1C `event_id` of the outbound loan row", "= event_id of the loan episode"),
    "player_id": ("Stage 1C player_id", "copied"),
    "player_name": ("Stage 1C player_name", "copied"),
    "loan_date": ("Stage 1C transfer_date of the loan row", "parsed date"),
    "loan_season": ("Stage 1C transfer_season", "'YY/YY' → 'YYYY/YY' (`season_label`; yy ≥ 90 → 19yy)"),
    "lender_club_id": ("Stage 1C from_club_id of the loan row", "copied"),
    "lender_club": ("Stage 1C from_club_name", "copied"),
    "borrower_club_id": ("Stage 1C to_club_id of the loan row", "copied"),
    "borrower_club": ("Stage 1C to_club_name", "copied"),
    "loan_raw_label": ("Stage 1C fee_display_raw of the loan row", "copied"),
    "outbound_loan_fee_eur": ("Stage 1C loan_fee_eur", "copied; blank = no loan fee published"),
    "loan_is_internal": ("Stage 1C is_internal_move", "copied (internal loans are not episodes)"),
    "loan_involves_placeholder": ("Stage 1C involves_placeholder_club", "copied"),
    "loan_is_youth_or_reserve_side": ("Stage 1C is_youth_or_reserve_side", "copied"),
    "match_reason": ("pairing walk over the player's rows (`pair_loans`)", "how the spell ended; see the pairing rules"),
    "terminal_event_id": ("Stage 1C event_id of the row that ended the spell", "blank for an open loan"),
    "terminal_date": ("Stage 1C transfer_date of the terminal row", "blank for an open loan"),
    "terminal_from_club": ("terminal row from_club_name", "copied"),
    "terminal_to_club": ("terminal row to_club_name", "copied"),
    "terminal_type": ("terminal row transfer_type_normalized", "loan_return for a recorded return"),
    "terminal_raw_label": ("terminal row fee_display_raw", "copied"),
    "terminal_is_return": ("terminal row type", "terminal_type == 'loan_return'"),
    "terminal_scheduled_future": ("terminal row transfermarkt_future_transfer", "True = the ending had not happened at capture"),
    "ending_fee_on_return_eur": ("terminal row fee_on_return_eur", "only when the terminal row is a return"),
    "terminal_permanent_fee_eur": ("terminal row permanent_transfer_fee_eur", "copied"),
    "duration_days": ("loan_date, terminal_date", "terminal_date − loan_date in days; blank if open"),
    "raw_label_class": ("terminal_raw_label", "`raw_label_class`: 'End of loan', 'End of loan + fee', 'free transfer', "
                        "'? (undisclosed)', '- (no fee shown)', 'explicit fee amount', 'loan transfer'; 'no ending "
                        "recorded' if open"),
    "has_matched_return": ("terminal_is_return", "identical to terminal_is_return"),
    "follow_on_event_id": ("Stage 1C rows of the player after the return", "first later row that is not internal, not "
                           "youth_or_internal / loan_return / other; blank if it is more than 60 days later, if it is a "
                           "release to a placeholder, or if the spell did not end in a return"),
    "follow_on_gap_days": ("follow-on transfer_date − terminal_date", "days"),
    "follow_on_type": ("follow-on transfer_type_normalized", "copied"),
    "follow_on_to_club": ("follow-on to_club_name", "copied"),
    "return_boundary_days": ("terminal_date", "days to the nearest 30 June or 31 December (previous, same or next "
                             "year); only for returns"),
    "economic_ending": ("match_reason, follow-on, return_boundary_days, ending fee", "`_economic`, first rule that applies"),
    "ending_detail": ("same", "text explaining the rule that fired"),
}

# ---------------------------------------------------------------------------
# Master-table variables
# ---------------------------------------------------------------------------

V = namedtuple("V", "variable section raw_or_derived source source_columns join_keys rule missing kind used "
                    "definition caveat confidence")


def v(variable, section, rod, source, cols, keys, rule, missing, kind, used, definition, caveat="", confidence="high"):
    return V(variable, section, rod, source, cols, keys, rule, missing, kind, used, definition, caveat, confidence)


S1C = "stage1c_canonical_transfers.csv"
LE = "src/analysis/loan_episodes.py build_universe (in memory)"
LTE = "src/analysis/loan_timing_exploration.py episode_table / next_moves (in memory)"
CAL = "loan_calendar_rule.py applied with club_geography_v2.csv (= loan_calendar_classified_v2.csv)"
PTJ = "src/analysis/playing_time_join.py compute (in memory; = playing_time_join_episodes.csv)"
REG = "src/analysis/playing_time_regression.py build (in memory)"
INJ = "src/analysis/playing_time_injury_adjusted.py compute (= playing_time_injury_adjusted_episodes.csv)"
ORG = "src/analysis/club_organisation.py episode_flags with club_organisation_map.csv"
AUD = "src/analysis/appearance_audit.py episode_check (game_lineups, game_events, appearances)"
EV = "event_id"
LID = "event_id = loan_event_id"
NA_LOAN = "Blank for permanent episodes (not applicable)."
NA_CAL = "Blank: not a realised loan ending (permanent move, open loan, or scheduled ending)."
NA_PT = "Blank: not a playing-time candidate or B not in a covered league season."
NA_REL = "Blank unless the outcome is reliable (missing coverage is never written as zero)."
NA_REV = "Blank outside the revised playing-time sample (only computed there)."
FLAG = "flag (descriptive; never used to drop rows)"
FILT = "sample filter (defines a sample)"
CAT = "category / classification"
VAL = "value"
IDT = "identifier"

VARS = [
    # A. identity
    v("event_id", "A identity", "derived (Stage 1)", S1C, "player_id, transfer_date, from_club_id, to_club_id", "primary key",
      "'tm_' + sha256('player_id|transfer_date|from_club_id|to_club_id')[:20]", "Never missing.", IDT, "all",
      "Stage 1C row that starts the episode (the outbound loan row for a loan).",
      "Identifies a registration row, not a contract. For a loan the End-of-loan row is collapsed into this episode."),
    v("transfermarkt_transfer_id", "A identity", "raw", S1C, "raw transfer url", EV, "copied", "Never missing.", IDT,
      "sort tie-break only", "Transfermarkt's own id of the starting row."),
    v("event_chain_id", "A identity", "derived (Stage 1)", S1C, "player_id, from_club_id, to_club_id, transfer_date", EV,
      "same chain as the previous row if it leaves the club that row arrived at within 14 days", "Never missing.", IDT,
      "not used", "Chain of immediately continuing movements."),
    v("player_id", "A identity", "raw", S1C, "transfers.player_id", "joins players, appearances, player_valuations, Kaggle",
      "copied", "Never missing.", IDT, "all", "Transfermarkt player id.",
      "One player has many episodes: count players with nunique(player_id), never with rows."),
    v("player_name", "A identity", "raw", S1C, "transfers.player_name", EV, "copied", "Never missing.", "label",
      "display", "Player name.", "Not unique."),
    # B. transfer
    v("transfer_date", "B transfer", "raw", S1C, "transfers.transfer_date", EV, "copied (YYYY-MM-DD)", "Never missing.",
      VAL, "all", "Date of the starting row: loan start, or the permanent move. = `move_date` / window start.",
      "Can be in the future for scheduled moves (see transfermarkt_future_transfer)."),
    v("transfer_season", "B transfer", "raw", S1C, "transfers.transfer_season", EV, "copied ('YY/YY')", "Never missing.",
      CAT, "via season", "Transfermarkt season label of the move."),
    v("season", "B transfer", "derived", LE, "transfer_season", EV, "season_label: 'YY/YY' → 'YYYY/YY' (yy ≥ 90 → 19yy)",
      "Never missing.", CAT, "loan_season in calendar outputs", "Season of the starting move, four-digit form."),
    v("episode_type", "B transfer", "derived", LE, "transfer_type_normalized", EV,
      "loan→loan; permanent_transfer→permanent_paid; free_transfer→free_transfer; undisclosed_transfer→undisclosed_fee; "
      "no_fee_shown→no_fee_shown; unknown→unknown_type", "Never missing.", CAT, "all", "Type of the episode."),
    v("kind", "B transfer", "derived", PTJ, "episode_type", EV,
      "loan → 'loan'; permanent_paid, free_transfer, undisclosed_fee, no_fee_shown → 'permanent'",
      "Blank for the unknown_type episodes (not playing-time candidates).", CAT, "playing time, regression, injury",
      "Loan versus permanent arrival.", "'permanent' pools paid, free, undisclosed and no-fee-shown moves."),
    v("transfer_type_raw", "B transfer", "derived (Stage 1B)", S1C, "raw fee text", EV, "Stage 1B classify_label",
      "Never missing.", CAT, "fee routing", "Transfermarkt's label for the starting row."),
    v("transfer_type_normalized", "B transfer", "derived (Stage 1C)", S1C, "transfer_type_raw, is_internal_move", EV,
      "policy.classify_type", "Never missing.", CAT, "all", "Normalised movement type of the starting row."),
    v("fee_display_raw", "B transfer", "raw", S1C, "raw fee text", EV, "copied byte-exact", "Blank only when the raw row "
      "had no fee text.", "label", "loan labels", "Fee cell as displayed by Transfermarkt.", "'?' and '-' are different "
      "statements (undisclosed vs no fee shown); neither is zero."),
    v("fee_disclosure_status", "B transfer", "derived (Stage 1C)", S1C, "transfer_type_raw", EV, "policy.FEE_DISCLOSURE",
      "Never missing.", CAT, "descriptive", "What the source said about the fee."),
    v("permanent_transfer_fee_eur", "B transfer", "derived (Stage 1C)", S1C, "transfer_type_raw, fee_numeric_eur", EV,
      "amount if paid_transfer; 0.0 if free_transfer; else blank", "Blank: not a paid or free permanent move.", VAL,
      "follow-on fee kind", "Permanent fee in euros.", "0 means Transfermarkt states 'free transfer'."),
    v("loan_fee_eur", "B transfer", "derived (Stage 1C)", S1C, "transfer_type_raw, fee_numeric_eur", EV,
      "amount if loan_fee label; else blank", "Blank: no loan fee published (never imputed as 0).", VAL,
      "calendar selection table", "Fee for the outbound loan."),
    v("from_club_id", "B transfer", "raw", S1C, "transfers.from_club_id", "club_id", "copied", "Never missing.", IDT,
      "pairing, geography", "Club A: lender for a loan, origin club for a permanent move."),
    v("from_club_name", "B transfer", "raw", S1C, "transfers.from_club_name", EV, "copied", "Never missing.", "label",
      "display", "Name of club A."),
    v("to_club_id", "B transfer", "raw", S1C, "transfers.to_club_id", "club_id; = b_club_id", "copied", "Never missing.",
      IDT, "playing time (receiving club B), geography", "Club B: borrower for a loan, buyer for a permanent move."),
    v("to_club_name", "B transfer", "raw", S1C, "transfers.to_club_name", EV, "copied", "Never missing.", "label",
      "display", "Name of club B."),
    v("is_youth_or_reserve_side", "B transfer", "derived (Stage 1C)", S1C, "from_club_name, to_club_name", EV,
      "YOUTH_OR_RESERVE regex on either name", "Never missing.", FLAG, "descriptive", "A youth or reserve side is involved."),
    v("is_senior_move", "B transfer", "derived (Stage 1C)", S1C, "placeholder, youth/reserve, internal flags", EV,
      "not any of the three", "Never missing.", FLAG, "Stage 2 only", "Senior-club move."),
    v("transfermarkt_future_transfer", "B transfer", "raw", S1C, "raw futureTransfer flag", EV, "copied",
      "Never missing.", FLAG, "playing-time 'scheduled' exclusion",
      "Transfermarkt marked the starting move as not yet happened at capture.",
      "The flag's exact upstream semantics are not documented; read as 'scheduled at capture'.", "medium"),
    v("is_research_target", "B transfer", "derived (Stage 1C)", S1C, "flags and type", EV, "policy.research_decision",
      "Never missing.", FLAG, "Stage 2 only", "In the Stage 2 contract-research population.",
      "Not a filter in any playing-time, calendar or injury analysis."),
    v("from_club_parent_id_v2", "B transfer", "derived (master helper)", ORG, "from_club_id; club_organisation_map.csv",
      "from_club_id = child_club_id", "parent_club_id of club A in the organisation map when its confidence is high, "
      "medium or low", "Blank: club A is not a youth/reserve side, or its parent is unresolved.", IDT,
      "academy-promotion sensitivity", "Senior club that club A (a youth/reserve side) belongs to.",
      "A proposed link, with the confidence in same_org_confidence; Transfermarkt publishes no parent field.", "medium"),
    v("same_organisation_move_v2", "B transfer", "derived (master helper)", ORG, "from/to club ids; organisation map",
      EV, "both clubs belong to one organisation (each youth/reserve side replaced by its proposed parent) and the "
      "weakest link used is high confidence", "Never missing.", FLAG, "academy-promotion sensitivity B",
      "High-confidence move inside one club organisation (mostly academy/reserve promotions).",
      "Sensitivity flag only: the main sample is unchanged. Stage 1C's own is_internal_move is not altered."),
    v("academy_promotion_candidate", "B transfer", "derived (master helper)", ORG, "from/to club ids; organisation map",
      EV, "club A is a youth/reserve side and club B is its proposed parent, at high, medium or low confidence",
      "Never missing.", FLAG, "academy-promotion sensitivity C (upper bound)",
      "Promotion from the club's own academy or reserve side, any confidence.",
      "Includes medium and low links; use same_organisation_move_v2 for the conservative version."),
    v("same_org_confidence", "B transfer", "derived (master helper)", ORG, "from/to club ids; organisation map", EV,
      "'high' / 'medium' / 'low' (same organisation, weakest link); 'rejected_name_overlap_only' (the earlier rough "
      "name screen matched, but club B is not club A's parent); 'rejected_senior_club' (a senior club named like a "
      "reserve side, e.g. Willem II); 'different_organisation_or_unresolved'; 'no_youth_or_reserve_side'",
      "Never missing.", CAT, "academy-promotion sensitivity; audit",
      "How sure we are that the move stays inside one club organisation."),
    v("same_org_reason", "B transfer", "derived (master helper)", ORG, "organisation map evidence", EV,
      "text: the parent link used and the evidence behind it (names, players connecting the clubs, country)",
      "Blank when no youth/reserve side is involved.", "label", "audit", "Why same_org_confidence has its value."),
    # C. loan ending
    v("match_reason", "C loan ending", "derived", LE, "the player's Stage 1C rows in date order", LID,
      "walking the player's later rows, the first row that ends the loan and which rule matched it: the reversed move (reversed_club_ids), a return to another side of the same organisation, a move by the borrower elsewhere without a recorded return, a permanent move lender → borrower, a re-loan, or no later movement (see data_lineage_overview.md §3)", NA_LOAN, CAT, "economic_ending",
      "How the loan spell ended (which row closed it, and why)."),
    v("has_matched_return", "C loan ending", "derived", LE, "terminal row transfer_type_normalized", LID,
      "terminal row is a loan_return row", NA_LOAN, FLAG, "playing-time loan_ending; calendar next category",
      "The loan has a recorded End-of-loan row.", "Same as terminal_is_return / ending_is_return in other outputs. "
      "Includes returns dated in the future (scheduled)."),
    v("terminal_event_id", "C loan ending", "derived", LE, "Stage 1C event_id of the terminal row", LID, "copied",
      "Blank: permanent episode, or open loan (no later movement).", IDT, "calendar", "Row that ended the loan spell."),
    v("terminal_date", "C loan ending", "derived", LE, "terminal row transfer_date", LID, "copied",
      "Blank: permanent episode, or open loan.", VAL, "calendar (ending date)", "The loan ending date.",
      "The Transfermarkt date on the ending row; not derived from any match or contract document."),
    v("terminal_type", "C loan ending", "derived", LE, "terminal row transfer_type_normalized", LID, "copied",
      "Blank: permanent episode, or open loan.", CAT, "descriptive", "Type of the ending row."),
    v("terminal_raw_label", "C loan ending", "derived", LE, "terminal row fee_display_raw", LID, "copied",
      "Blank: permanent episode, or open loan.", "label", "raw_label_class", "Fee cell of the ending row."),
    v("raw_label_class", "C loan ending", "derived", LE, "terminal_raw_label", LID, "the ending row's Transfermarkt fee label, grouped: End of loan; End of loan + fee; '-'; free transfer; '?'; explicit fee amount; loan transfer; no ending recorded",
      NA_LOAN, CAT, "descriptive", "Transfermarkt's label on the ending row."),
    v("terminal_scheduled_future", "C loan ending", "derived", LE, "terminal row transfermarkt_future_transfer", LID,
      "copied", NA_LOAN, FLAG, "calendar population (ending_realised)", "The ending had not happened at capture."),
    v("ending_observed", "C loan ending", "derived", LTE, "terminal_date", LID, "terminal_date is not blank", NA_LOAN,
      FLAG, "calendar population", "Some movement ended the loan."),
    v("ending_realised", "C loan ending", "derived", LTE, "terminal_date, terminal_scheduled_future", LID,
      "ending_observed and not terminal_scheduled_future", NA_LOAN, FILT, "calendar population",
      "The ending had happened at capture: the population of every calendar analysis."),
    v("duration_days", "C loan ending", "derived", LE, "transfer_date, terminal_date", LID, "terminal_date − loan date",
      "Blank: permanent episode, or open loan.", VAL, "calendar tables", "Length of the loan in days."),
    v("ending_fee_on_return_eur", "C loan ending", "derived", LE, "terminal row fee_on_return_eur", LID,
      "copied when the terminal row is a return", "Blank: no fee on the return (or not a return).", VAL,
      "economic_ending fee_bearing_return", "Money on the End-of-loan row."),
    v("return_boundary_days", "C loan ending", "derived", LE, "terminal_date", LID,
      "min |terminal_date − D| for D in {30 Jun, 31 Dec} of year−1, year, year+1",
      "Blank unless the loan ended with a return row.", VAL, "early_termination rule",
      "Distance of the return date from the nearest 30 June or 31 December.",
      "Stage 2 (event_family) measures 30 June only; the current loan code uses both dates."),
    v("follow_on_event_id", "C loan ending", "derived", LE, "player's later Stage 1C rows", LID,
      "first later row not internal / youth_or_internal / loan_return / other, within 60 days, not a release",
      "Blank: no such movement within 60 days, a release came first, or the loan did not end in a return.", IDT,
      "economic_ending", "The player's next deal after the return (60-day window)."),
    v("follow_on_gap_days", "C loan ending", "derived", LE, "follow-on date, terminal_date", LID, "difference in days",
      "Blank when there is no follow-on.", VAL, "economic_ending (21-day immediate window)", "Days from return to follow-on."),
    v("follow_on_type", "C loan ending", "derived", LE, "follow-on transfer_type_normalized", LID, "copied",
      "Blank when there is no follow-on.", CAT, "economic_ending", "Type of the follow-on movement."),
    v("follow_on_to_club", "C loan ending", "derived", LE, "follow-on to_club_name", LID, "copied",
      "Blank when there is no follow-on.", "label", "descriptive", "Destination of the follow-on movement."),
    v("economic_ending", "C loan ending", "derived", LE, "match_reason, follow-on, return_boundary_days, fee on return",
      LID, "first rule that applies: no later movement → unresolved; permanent move lender → borrower, or a permanent move back to the borrower after the return → conversion; lender sells the player on, or a paid / undisclosed move to a third club after the return → third-party sale; re-loan, loan elsewhere or leaving the borrower without a recorded return → other non-standard; otherwise, with a next deal within 60 days that is not a release: next deal within 21 days and return more than 10 days from 30 June / 31 December → early termination (legacy heuristic); next deal within 21 days or a new loan → immediate follow-on; return with a fee → fee-bearing return; otherwise ordinary end of loan (see §4)", NA_LOAN, CAT, "calendar tables (early_termination "
      "share); Stage 2", "Sequence pattern of how the loan ended.",
      "Describes what the registration sequence looks like; never proves a contract term."),
    v("ending_detail", "C loan ending", "derived", LE, "same as economic_ending", LID, "text of the rule that fired",
      NA_LOAN, "label", "display", "Explanation of economic_ending."),
    v("next_move_event_id", "C loan ending", "derived", LTE, "player's later Stage 1C rows", LID,
      "first later row not internal / youth_or_internal / loan_return / other, at ANY distance, releases included",
      "Blank: loan not ended by a return, or no later movement.", IDT, "calendar characterisation tables",
      "The player's next movement after the return, unwindowed.", "Differs from follow_on_event_id: no 60-day window "
      "and releases count."),
    v("next_move_gap_days", "C loan ending", "derived", LTE, "next move date, terminal_date", LID, "difference in days",
      "Blank when there is no next move.", VAL, "calendar tables ('next move ≤ 1 / 21 / 60 days')", "Days to the next move."),
    v("next_move_type", "C loan ending", "derived", LTE, "next move transfer_type_normalized", LID, "copied",
      "Blank when there is no next move.", CAT, "calendar tables", "Type of the next move."),
    v("next_category", "C loan ending", "derived", "loan_calendar_rule.next_category", "next_move_kind, next_move_to_club_id, "
      "borrower_club_id, has_matched_return", LID, "permanent move to the borrower / re-loan to the same borrower / loan to "
      "a different club / paid or undisclosed move to a third club / free or no-fee move to a third club / release / "
      "no observed next move / ended without a return row", NA_LOAN, CAT, "calendar tables",
      "What the player did next after the loan."),
    # D. geography / calendar
    v("from_country_v1", "D geography/calendar", "derived", LE + " club_geography", "clubs.domestic_competition_id → "
      "competitions.country_name; else games × competitions participation", "from_club_id = club_id", "lookup",
      "Blank: club not reached by the snapshot's league geography.", CAT, "calendar v1 (superseded)", "Club A country, v1."),
    v("to_country_v1", "D geography/calendar", "derived", LE + " club_geography", "as from_country_v1",
      "to_club_id = club_id", "lookup", "Blank: club not reached by the snapshot's league geography.", CAT,
      "calendar v1 (superseded); playing-time join audit only", "Club B country, v1."),
    v("from_country_v2", "D geography/calendar", "derived", "club_geography_v2.csv", "calendar_country",
      "from_club_id = club_id", "v1 league country, else single raw countryFlag, else fallbacks A/B/C",
      "Blank: special entry or unresolved club.", CAT, "calendar v2", "Club A country, v2."),
    v("to_country_v2", "D geography/calendar", "derived", "club_geography_v2.csv", "calendar_country",
      "to_club_id = club_id", "as from_country_v2", "Blank: special entry or unresolved club.", CAT, "calendar v2",
      "Club B country, v2.", "Ids the snapshot cannot name appear as 'Transfermarkt country N'."),
    v("from_geography_source_v2", "D geography/calendar", "derived", "club_geography_v2.csv", "source", "from_club_id",
      "copied", "Blank: no source.", CAT, "provenance", "Which source gave club A its v2 country."),
    v("to_geography_source_v2", "D geography/calendar", "derived", "club_geography_v2.csv", "source", "to_club_id",
      "copied", "Blank: no source.", CAT, "provenance", "Which source gave club B its v2 country."),
    v("calendar_context", "D geography/calendar", "derived", CAL, "from_country_v2, to_country_v2", LID,
      "domestic (same country) / cross-border (both mapped, different) / one or both clubs unmapped", NA_CAL, CAT,
      "calendar key", "Domestic or cross-border loan."),
    v("calendar_key", "D geography/calendar", "derived", CAL, "calendar_context, lender country", LID,
      "'domestic: <country>' / 'cross-border' / '' (unmapped)", NA_CAL, CAT, "calendar learning and application",
      "Which calendar the ending is judged against."),
    v("ending_in_covid_window", "D geography/calendar", "derived", CAL, "terminal_date", LID,
      "2020-03-01 ≤ terminal_date ≤ 2021-02-28", NA_CAL, FLAG,
      "excluded from LEARNING calendars only; classified like any other ending", "Ending falls in the COVID window."),
    v("calendar_class", "D geography/calendar", "derived", CAL, "calendar_key, terminal_date (month-day)", LID,
      "no calendar for the ending's country → geography_unclassified; within ±3 days of a calendar date → "
      "calendar_normal_domestic / _cross_border; else off_calendar_domestic / _cross_border", NA_CAL, CAT,
      "calendar tables", "Calendar classification of the ending date.",
      "A classification, not a filter: no later analysis drops or keeps episodes by it."),
    v("calendar_state", "D geography/calendar", "derived", CAL, "calendar_class, unclassified_reason", LID,
      "calendar-normal / off-calendar / unclassified: club country unknown / unclassified: below history threshold",
      NA_CAL, CAT, "calendar tables", "Four-way state of the ending."),
    v("calendar_classifiable", "D geography/calendar", "derived (master helper)", CAL, "calendar_class", LID,
      "calendar_class != 'geography_unclassified'", NA_CAL, FLAG, "the 25,818 'classifiable' count",
      "The ending has a calendar to be judged against.", "Created for this table; the saved calendar file has no such column."),
    v("off_calendar", "D geography/calendar", "derived (master helper)", CAL, "calendar_class", LID,
      "calendar_class starts with 'off_calendar'", "Blank: not a realised ending, or not classifiable.", FLAG,
      "calendar tables only", "Ending date is not within ±3 days of any date of its calendar.",
      "Created for this table from calendar_class. Not used to restrict any playing-time or injury sample."),
    v("matched_standard_mmdd", "D geography/calendar", "derived", CAL, "terminal_date, calendar centres", LID,
      "nearest calendar date when within ±3 days", "Blank: off-calendar or unclassified (or not a realised ending).", CAT,
      "calendar tables", "The calendar date the ending matched."),
    v("nearest_standard_mmdd", "D geography/calendar", "derived", CAL, "terminal_date, calendar centres", LID,
      "nearest calendar date (any distance)", "Blank: unclassified or not a realised ending.", CAT, "calendar tables",
      "Closest calendar date."),
    v("days_to_nearest_standard", "D geography/calendar", "derived", CAL, "terminal_date, calendar centres", LID,
      "circular day distance on a 366-day year", "Blank: unclassified or not a realised ending.", VAL, "calendar tables",
      "Days from the ending to the closest calendar date."),
    v("unclassified_reason", "D geography/calendar", "derived", CAL, "calendar_key, calendar_context", LID,
      "'' if classifiable; 'club country unknown'; 'country below the history threshold'", NA_CAL, CAT,
      "calendar tables", "Why an ending has no calendar."),
    v("loan_calendar_status", "D geography/calendar", "derived (master helper)", CAL,
      "kind, calendar_state, ending_realised", EV,
      "loans: calendar_state calendar-normal → 'normal', off-calendar → 'off_calendar', unclassified → 'unclassified'; "
      "loan without a realised ending (open, or ending flagged as a future move) → 'ending_not_realised'; permanent or "
      "unknown-type episode → 'not_applicable'", "Never missing.", CAT, "calendar-first samples; filtering loans by "
      "calendar status", "One readable calendar status for every episode.",
      "Re-labels the existing classification; no new rule. Permanent moves are never given a loan-ending status."),
    v("off_calendar_flag", "D geography/calendar", "derived (master helper)", CAL, "loan_calendar_status", EV,
      "1 if loan_calendar_status == 'off_calendar'; 0 if 'normal'; blank otherwise", "Blank: permanent move, loan "
      "without a realised ending, or an unclassified ending.", FLAG, "filtering loans by calendar status",
      "Off-calendar indicator (1/0) for classifiable loan endings.",
      "Same content as `off_calendar`, coded 1/0 for filtering and regressions."),
    # E. playing time
    v("receiving_league", "E playing time", "derived", PTJ, "games.competition_id, season, date; to_club_id; transfer_date",
      "b_club_id + season", "covered domestic league of B in the arrival season", NA_PT.replace("Blank", "Blank or '(not in "
      "a covered league)'"), CAT, "playing time, injury, FE", "B's league in the arrival season."),
    v("arrival_season", "E playing time", "derived", PTJ, "games.season/date of B's league; transfer_date", "b_club_id",
      "covered season s ∈ {year−1, year} with previous season's last league game < move date ≤ season s's last game",
      NA_PT, CAT, "playing time, injury, FE", "B's league season the player joins (start year).",
      "For a league's first season the previous season end is approximated as one year before.", "medium"),
    v("departure_date", "E playing time", "derived", PTJ, "player's later Stage 1C rows", EV,
      "date of the next row not internal / youth_or_internal / other (any type, any distance)",
      "Blank: no later movement recorded, or not a candidate.", VAL, "window end",
      "When the player next moves (ends the window if before season end)."),
    v("departed_in_window", "E playing time", "derived", PTJ, "departure_date, B's last league game", EV,
      "departure_date ≤ B's last league game of the arrival season", "False when not covered.", FLAG, "descriptive",
      "The player left B before the season ended."),
    v("window_start", "E playing time", "derived", PTJ, "transfer_date", EV, "= move date", NA_PT, VAL,
      "playing time, injury", "First day of the observation window (inclusive)."),
    v("window_end", "E playing time", "derived", PTJ, "games.date of B, departure_date", EV,
      "B's last league game of the arrival season, or departure_date − 1 day if earlier", NA_PT, VAL,
      "playing time, injury", "Last day of the observation window (inclusive)."),
    v("season_games", "E playing time", "derived", PTJ, "games (domestic_league, B, arrival season)", "b_club_id, season",
      "count", NA_PT, VAL, "remain, two-thirds rule", "B's league games in the arrival season."),
    v("games_remaining_at_arrival", "E playing time", "derived", PTJ, "games.date ≥ move date", "b_club_id, season",
      "count", NA_PT, VAL, "remain", "B's league games on or after the move date."),
    v("available_games", "E playing time", "derived", PTJ, "games (B, arrival season, window_start ≤ date ≤ window_end)",
      "b_club_id, season", "count", NA_PT, VAL, "denominator", "B's league games inside the window."),
    v("complete_games", "E playing time", "derived", PTJ, "appearances rows per (game, B)", "game_id, player_club_id",
      "window games where B has ≥ 11 appearance rows", NA_PT, VAL, "reliability", "Window games with a complete B record."),
    v("appearances", "E playing time", "derived", PTJ, "appearances (player, B, window games)", "player_id, game_id, "
      "player_club_id", "count of the player's appearance rows for B in window league games", NA_REL, VAL, "descriptive",
      "League appearances for B in the window."),
    v("minutes", "E playing time", "derived", PTJ, "appearances.minutes_played", "player_id, game_id, player_club_id",
      "sum of minutes_played over those rows (0 if none)", NA_REL, VAL, "numerator", "League minutes for B in the window."),
    v("available_minutes", "E playing time", "derived", PTJ, "available_games", EV, "90 × available_games", NA_REL, VAL,
      "denominator", "Minutes B played in the window, at 90 per game."),
    v("minutes_share", "E playing time", "derived", PTJ, "minutes, available_games", EV,
      "minutes / (90 × available_games)", NA_REL, VAL, "main outcome (= current_share)",
      "Share of B's league minutes the player played while at B.", "Extra time is ignored (90 per game), so a value "
      "can in principle exceed 1 only if minutes over 90 are recorded; none do in the samples."),
    v("true_zero", "E playing time", "derived", PTJ, "reliable, minutes", EV, "reliable and minutes == 0",
      "False when not reliable.", FLAG, "descriptive", "Reliable outcome with no league minute played."),
    v("pt_exclusion_reason", "E playing time", "derived", PTJ, "see rule", EV,
      "first failing of: scheduled_move, not_covered, player_not_in_universe, club_season_without_rows, "
      "no_league_games_in_window, window_after_capture, incomplete_club_games, appearances_for_other_club",
      "'' = reliable; blank for non-candidates.", CAT, "reliability", "Why the outcome is not reliable (`exclusion_reason`)."),
    v("prior_spell_at_b", "E playing time", "derived", PTJ, "player_id, to_club_id, transfer_date over all candidates",
      "player_id, b_club_id", "an earlier loan or permanent candidate episode of the player into the same B",
      "Blank for non-candidates.", FILT, "revised and old samples (first spell only)", "Not the player's first spell at B."),
    v("pt_loan_ending", "E playing time", "derived", PTJ, "terminal_date, has_matched_return", LID,
      "'return to lender' / 'other movement' / 'no ending recorded'", NA_LOAN, FILT,
      "revised and old samples (loans need 'return to lender')", "How the loan ended, coarse (`loan_ending`).",
      "'return to lender' includes returns dated in the future (scheduled)."),
    v("arrived_full_season", "E playing time", "derived", PTJ, "games_remaining_at_arrival, season_games", EV,
      "games_remaining_at_arrival / season_games ≥ 2/3", "False when not covered.", FILT, "OLD primary sample only "
      "(rule removed in the revised sample)", "Arrived with at least two-thirds of B's league games left."),
    v("remain", "E playing time", "derived", REG, "games_remaining_at_arrival, season_games", EV,
      "games_remaining_at_arrival / season_games", NA_PT, VAL, "regression control (arrival timing)",
      "Share of B's league season left at arrival."),
    # F. player attributes
    v("date_of_birth", "F player", "derived (joined)", S1C, "players.date_of_birth", "player_id", "copied",
      "Blank: no date of birth in `players`.", VAL, "age", "Date of birth."),
    v("age_at_transfer", "F player", "derived (Stage 1)", S1C, "date_of_birth, transfer_date", EV,
      "completed years (birthday rule)", "Blank when date_of_birth is blank.", VAL, "calendar selection table",
      "Age in completed years at the move.", "Not the regression age."),
    v("age_exact", "F player", "derived", REG, "date_of_birth, transfer_date", EV, "(transfer_date − date_of_birth) in "
      "days / 365.25", "Blank when date_of_birth is blank.", VAL, "regression (age_c, age_c²)", "Continuous age at the "
      "move (`age` in the regression code).", "age_c is centred on the mean of each model's own sample, so the centre "
      "differs between modules."),
    v("age_years", "F player", "derived", "src/analysis/playing_time_age.py load", "age_exact", EV, "floor(age_exact)",
      "Blank when date_of_birth is blank.", VAL, "young/older split", "Whole years from the continuous age.",
      "Can differ from age_at_transfer by one year within a day or two of a birthday (365.25-day years)."),
    v("older", "F player", "derived", "playing_time_age.add_groups / injury compute", "age_years", EV,
      "1 if age_years > 23 (24 or older), 0 if ≤ 23", "Blank when age is missing.", FLAG,
      "age models (older, loan × older)", "Older arrival (≥ 24)."),
    v("position", "F player", "derived (joined)", S1C, "players.position", "player_id", "copied",
      "Never blank; the string 'Missing' = no position.", CAT, "position_clean", "Playing position."),
    v("position_clean", "F player", "derived", REG, "position", EV, "position if Goalkeeper/Defender/Midfield/Attack",
      "Blank: position 'Missing'.", CAT, "regression position dummies", "Position used in the models."),
    v("country_of_citizenship", "F player", "derived (joined)", S1C, "players.country_of_citizenship", "player_id",
      "copied", "Never blank.", CAT, "not used", "Citizenship."),
    # G. market value
    v("market_value_eur", "G market value", "raw", S1C, "transfers.market_value_in_eur", EV, "copied",
      "Blank: no value on the transfer row.", VAL, "calendar selection table; compared with player_value",
      "Market value shown on the Transfermarkt transfer row.", "Not the regression player value.", "medium"),
    v("player_value", "G market value", "derived", REG, "player_valuations.market_value_in_eur, date", "player_id",
      "latest valuation dated strictly before transfer_date (merge_asof backward, exact match not allowed)",
      NA_REV + " Inside it, blank = no valuation before the move.", VAL, "relative value", "Player market value just "
      "before the move.", "No age limit on the valuation (see stale_player_value)."),
    v("player_value_date", "G market value", "derived", REG, "player_valuations.date", "player_id", "date of that valuation",
      NA_REV, VAL, "audit", "Date of the valuation used."),
    v("player_value_gap_days", "G market value", "derived", REG, "transfer_date, player_value_date", EV, "difference in days",
      NA_REV, VAL, "stale flag", "Age of the valuation at the move."),
    v("stale_player_value", "G market value", "derived", REG, "player_value_gap_days", EV, "gap > 365 days",
      NA_REV, FLAG, "robustness only", "Valuation older than a year."),
    v("squad_value", "G market value", "derived", REG, "player_valuations (player_id, date, value, current_club_id)",
      "current_club_id = to_club_id", "sum over players whose latest valuation strictly before the move, within 365 days, "
      "lists B as their club, excluding the focal player; blank if fewer than 11 such players",
      NA_REV + " Inside it, blank = fewer than 11 valued players.", VAL, "relative value, log_squad_value",
      "Receiving squad's market value just before the move.", "Reconstructed from player valuations: the clubs table "
      "has no usable squad value (NULL for every club)."),
    v("squad_players", "G market value", "derived", REG, "same as squad_value", "b_club_id", "number of those players",
      NA_REV, VAL, "minimum 11 rule", "Valued players in the squad value."),
    v("focal_in_squad", "G market value", "derived", REG, "same as squad_value", "player_id", "the player himself was "
      "listed at B before the move (then removed from the sum)", NA_REV, FLAG, "audit", "Player already valued at B."),
    v("relative_value", "G market value", "derived", REG, "player_value, squad_value", EV, "player_value / squad_value",
      NA_REV, VAL, "descriptive", "Player value as a share of the receiving squad's value."),
    v("log_player_value", "G market value", "derived", REG, "player_value", EV, "ln(player_value) if > 0", NA_REV, VAL,
      "M4 only", "Log player value."),
    v("log_squad_value", "G market value", "derived", REG, "squad_value", EV, "ln(squad_value) if > 0", NA_REV, VAL,
      "regression control", "Log squad value."),
    v("log_relative_value", "G market value", "derived", REG, "player_value, squad_value", EV,
      "ln(player_value) − ln(squad_value)", NA_REV, VAL, "regression control", "Log relative value."),
    v("league_season", "G market value", "derived", REG, "receiving_league, arrival_season", EV,
      "receiving_league + '_' + arrival_season", NA_REV, CAT, "league × season fixed effects", "Fixed-effect cell."),
    # H. injury
    v("injury_join_status", "H injury", "derived", INJ, "player_id; Kaggle player_injuries.player_id, "
      "player_profiles.player_id", "player_id (exact integer)", "A = has injury rows; B = in profiles, no injury rows; "
      "C = not in the external datalake", NA_REV, CAT, "usability", "Is the player in the external injury data?",
      "B means no listed injury, not proven health."),
    v("available_games_current", "H injury", "derived", INJ, "B's league games in the window (games)", EV,
      "count of window games (= available_games)", NA_REV, VAL, "denominator (current)", "Games in the current denominator."),
    v("confirmed_injury_games", "H injury", "derived", INJ, "window games; Kaggle from_date, end_date, injury_reason; "
      "appearances", "player_id, date", "window games covered (inclusive) by ≥ 1 PRIMARY medical spell in which the "
      "player has no appearance row for B", NA_REV, VAL, "fit denominator", "Games removed for injury or illness."),
    v("available_games_fit", "H injury", "derived", INJ, "available_games_current, confirmed_injury_games", EV,
      "available_games_current − confirmed_injury_games", NA_REV, VAL, "fit denominator", "Games the player was fit for."),
    v("current_share", "H injury", "derived", INJ, "minutes, available_games_current", EV,
      "minutes / (90 × available_games_current)", NA_REV, VAL, "injury models (current outcome)",
      "Current playing-time share (identical to minutes_share)."),
    v("fit_adjusted_share", "H injury", "derived", INJ, "minutes, available_games_fit", EV,
      "minutes / (90 × available_games_fit); blank if available_games_fit = 0", NA_REV + " Inside it, blank = no fit "
      "games left.", VAL, "injury models (fit outcome)", "Playing-time share of the games the player was fit for.",
      "Only meaningful where injury_data_quality_flag = 'usable'."),
    v("injury_days_overlap", "H injury", "derived", INJ, "primary spells, window", EV,
      "days of the union of primary spells clipped to the window", NA_REV, VAL, "descriptive", "Injured days in the window."),
    v("injury_spell_count", "H injury", "derived", INJ, "primary spells, window", EV,
      "number of primary spells overlapping the window", NA_REV, VAL, "worked examples", "Spells touching the window."),
    v("appearance_conflicts", "H injury", "derived", INJ, "spells, appearances", EV,
      "window games inside a spell in which the player did appear (kept in the denominator)", NA_REV, VAL, "audit",
      "Appearances during a recorded spell."),
    v("injury_data_quality_flag", "H injury", "derived", INJ, "status, window, Kaggle rows", EV,
      "first failing of: not in datalake; window reaches 2025-09-15; history truncated before window; open primary "
      "spell starting ≤ window end; primary row without start date; no fit games left; else 'usable'", NA_REV, CAT,
      "in_injury_usable", "Why the injury adjustment can or cannot be computed for this episode."),
    v("injury_games_strict", "H injury", "derived", INJ, "as confirmed_injury_games, STRICT spells", EV,
      "same rule with STRICT labels (no unknown injury, no minor muscular complaints)", NA_REV, VAL, "sensitivity B",
      "Games removed under the strict rule."),
    v("fit_share_strict", "H injury", "derived", INJ, "minutes, injury_games_strict", EV,
      "minutes / (90 × (available_games_current − injury_games_strict))", NA_REV, VAL, "sensitivity B", "Strict fit share."),
    v("injury_games_severe3", "H injury", "derived", INJ, "primary spells", EV,
      "games removed only by spells that remove ≥ 3 window games", NA_REV, VAL, "sensitivity C", "Severe-spell games."),
    v("fit_share_severe3", "H injury", "derived", INJ, "minutes, injury_games_severe3", EV, "as fit share", NA_REV, VAL,
      "sensitivity C", "Fit share, severe spells only."),
    v("injury_games_long5", "H injury", "derived", INJ, "primary spells", EV,
      "games removed only by spells that remove ≥ 5 window games", NA_REV, VAL, "sensitivity D", "Long-spell games."),
    v("fit_share_long5", "H injury", "derived", INJ, "minutes, injury_games_long5", EV, "as fit share", NA_REV, VAL,
      "sensitivity D", "Fit share, long spells only."),
    # I. sample membership
    v("in_calendar_population", "I sample membership", "derived", LTE, "ending_realised", LID,
      "loan episode with a realised ending", "Never missing (False for permanents).", FILT, "every calendar analysis",
      "In the 27,402 realised loan endings."),
    v("in_playing_time_candidates", "I sample membership", "derived", PTJ, "episode_type", EV, "kind is loan or permanent",
      "Never missing.", FILT, "playing-time waterfall", "Candidate for the playing-time join."),
    v("pt_realised", "I sample membership", "derived", PTJ, "transfermarkt_future_transfer", EV,
      "candidate and not scheduled", "Never missing.", FILT, "playing-time waterfall", "Move had happened at capture."),
    v("pt_covered_league_season", "I sample membership", "derived", PTJ, "arrival_season", EV,
      "pt_realised and B in a covered league season", "Never missing.", FILT, "playing-time waterfall",
      "B plays in one of the 14 covered leagues that season."),
    v("pt_matched", "I sample membership", "derived", PTJ, "players, appearances", EV,
      "covered and player in `players` and B has appearance rows that season", "Never missing.", FILT,
      "playing-time waterfall", "Join possible."),
    v("pt_reliable", "I sample membership", "derived", PTJ, "pt_exclusion_reason", EV, "pt_exclusion_reason == ''",
      "Never missing.", FILT, "every playing-time sample", "Outcome reliably calculable."),
    v("pt_lineup_games_checked", "I sample membership", "derived (master helper)", AUD,
      "window games of club B; game_lineups", EV, "window league games of B for which the snapshot has B's line-up",
      "Blank unless pt_reliable.", VAL, "appearance-gap check",
      "How many window games could be checked against line-ups.",
      "0 for 2012/13 arrivals: the snapshot has no line-ups for that season, so those episodes cannot be checked."),
    v("pt_games_played_without_row", "I sample membership", "derived (master helper)", AUD,
      "game_lineups (starting_lineup), game_events (Substitutions), appearances", EV,
      "window games of B in which the player started or came on for B but has no appearances row for B",
      "Blank unless pt_reliable.", VAL, "appearance-gap check",
      "Games the player demonstrably played that the minutes do not include."),
    v("pt_contradiction_type", "I sample membership", "derived (master helper)", AUD,
      "pt_games_played_without_row, appearances", EV,
      "'false_zero' if games are contradicted and the recorded appearances are 0; 'partial_undercount' if contradicted "
      "with some appearances; 'none' otherwise", "Blank unless pt_reliable.", CAT, "appearance-gap check",
      "Kind of appearance-row gap behind a reliable outcome."),
    v("pt_lineup_contradicted", "I sample membership", "derived (master helper)", AUD, "pt_games_played_without_row",
      EV, "pt_reliable and pt_games_played_without_row > 0", "Never missing.", FLAG, "appearance-gap sensitivity",
      "Reliable outcome contradicted by line-ups or substitutions (minutes undercounted, often to zero).",
      "False outside reliable episodes means 'not checked', not 'clean'. The original pt_* columns are unchanged."),
    v("pt_reliable_v2", "I sample membership", "derived (master helper)", AUD, "pt_reliable, pt_lineup_contradicted",
      EV, "pt_reliable and not pt_lineup_contradicted", "Never missing.", FILT, "appearance-gap sensitivity",
      "Reliable and not contradicted by line-ups or substitutions.",
      "Non-destructive: pt_reliable and every in_* sample flag are unchanged; combine pt_reliable_v2 with a sample "
      "flag to drop contradicted episodes."),
    v("in_old_primary_sample", "I sample membership", "derived", PTJ, "reliable, arrived_full_season, prior_spell_at_b, "
      "pt_loan_ending", EV, "reliable & two-thirds rule & first spell & (permanent or return to lender)", "Never missing.",
      FILT, "first regression (superseded)", "Old primary sample (`primary_sample`)."),
    v("in_revised_sample", "I sample membership", "derived", "playing_time_sample_redesign.masks", "reliable, "
      "prior_spell_at_b, pt_loan_ending", EV, "reliable & first spell & (permanent or return to lender)",
      "Never missing.", FILT, "age, injury analyses", "Revised primary playing-time sample."),
    v("in_injury_usable", "I sample membership", "derived", INJ, "injury_data_quality_flag", EV,
      "in_revised_sample and flag == 'usable'", "Never missing.", FILT, "injury descriptives and models",
      "Injury adjustment computable."),
    v("calendar_first_eligible", "I sample membership", "derived (master helper)", CAL, "kind, loan_calendar_status", EV,
      "permanent → True; loan → True if loan_calendar_status is 'normal' or 'off_calendar' (a classifiable realised "
      "ending), else False; unknown type → False", "Never missing.", FILT, "calendar-first samples",
      "Passes the loan-calendar gate of the calendar-first sequence.",
      "Off-calendar loans pass: the gate requires a classifiable ending, not a normal one."),
    v("in_cf_revised_sample", "I sample membership", "derived (master helper)", "data_lineage.calendar_first",
      "in_revised_sample, calendar_first_eligible", EV, "in_revised_sample and calendar_first_eligible", "Never missing.",
      FILT, "calendar-first samples", "Revised playing-time sample under the calendar-first sequence."),
    v("in_cf_injury_usable", "I sample membership", "derived (master helper)", "data_lineage.calendar_first",
      "in_injury_usable, calendar_first_eligible", EV, "in_injury_usable and calendar_first_eligible", "Never missing.",
      FILT, "calendar-first samples", "Injury-usable sample under the calendar-first sequence."),
    # J. regression-complete
    v("in_old_regression_sample", "J regression-complete", "derived", "playing_time_regression.compute",
      "in_old_primary_sample, covariates", EV, "old primary with share, remain, age, position_clean, log_relative_value, "
      "log_squad_value all present", "Never missing.", FILT, "first preferred model (M3)",
      "Old common regression sample."),
    v("in_age_model_sample", "J regression-complete", "derived", "playing_time_age.compute", "in_revised_sample, "
      "covariates", EV, "revised with share, age, remain, position_clean, log_relative_value, log_squad_value present",
      "Never missing.", FILT, "age models", "Revised-sample regression observations."),
    v("in_injury_model_sample", "J regression-complete", "derived", INJ, "in_injury_usable, covariates", EV,
      "injury-usable with current and fit share, remain, age, position_clean, log_relative_value, log_squad_value present",
      "Never missing.", FILT, "injury models (current vs fit-adjusted)", "Injury regression observations."),
    v("in_cf_main_model_sample", "J regression-complete", "derived (master helper)", "data_lineage.calendar_first",
      "in_age_model_sample, calendar_first_eligible", EV, "in_age_model_sample and calendar_first_eligible",
      "Never missing.", FILT, "calendar-first samples", "Main regression observations under the calendar-first sequence."),
    v("in_cf_injury_model_sample", "J regression-complete", "derived (master helper)", "data_lineage.calendar_first",
      "in_injury_model_sample, calendar_first_eligible", EV, "in_injury_model_sample and calendar_first_eligible",
      "Never missing.", FILT, "calendar-first samples",
      "Injury regression observations under the calendar-first sequence."),
]
