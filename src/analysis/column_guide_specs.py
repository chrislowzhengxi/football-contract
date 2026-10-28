"""Reader-facing column guides: what each Stage 1C and master column means, where it comes from, and what it is for.

Hand-written from the code that produces each column (named in the text). Value counts are NOT written here;
`data_lineage.py` attaches them from the current files and checks that every category listed here occurs in the data
and every category in the data is listed.
"""
from __future__ import annotations

from collections import namedtuple

# ---------------------------------------------------------------------------
# Roles and usefulness classes
# ---------------------------------------------------------------------------

S1C_ROLES = {
    "id": "identifier / linkage",
    "core": "analysis input (feeds the episodes, samples or regressions)",
    "filter": "used to set rows aside or define samples",
    "desc": "descriptive only",
    "qa": "QA / audit",
    "legacy": "legacy (Stage 1 heuristic or Stage 2 contract research; not used in Weeks 4–6)",
}

USEFULNESS = {
    "core": "core analysis variable",
    "filter": "sample / filter flag",
    "helper": "intermediate / helper",
    "qa": "QA / audit",
    "legacy": "legacy",
    "id": "identifier / linkage",
    "special": "currently unused or highly specialised",
}

ORIGINS = {
    "tm": "copied from Transfermarkt (the transfer row, via Stage 1C)",
    "s1c": "copied from a Stage 1C column we built from the Transfermarkt row",
    "multi": "built from several Stage 1C rows of the same player (loan pairing, next move, earlier spells)",
    "master": "derived from other master columns of the same row",
    "geo": "joined: club country (league country, Transfermarkt flag, fixed fallbacks)",
    "cal": "our calendar rule (dates learned from all loan endings, applied to this ending)",
    "games": "joined: DuckDB games + appearances of the receiving club",
    "players": "joined: DuckDB players table",
    "val": "joined: DuckDB player_valuations",
    "kaggle": "joined: Kaggle injury data (+ games / appearances)",
    "sample": "our sample rule (combines flags above)",
    "org": "our club-organisation map (club names + player flows + country; club_organisation.py)",
}

# ---------------------------------------------------------------------------
# Stage 1C: all 42 columns
# ---------------------------------------------------------------------------

G = namedtuple("G", "role source how values")
TM = "Transfermarkt transfer row (DuckDB `transfers`)"
TMR = "Transfermarkt raw transfer-history file (the row as shown on the website)"
DP = "DuckDB `players` table, joined on player_id"
OURS = "created by us in Stage 1C"
OURS1 = "created by us in Stage 1 (row backbone)"
PH = "Without Club, Retired, Career break, Ban, Unknown"

STAGE1C_GUIDE = {
    "event_id": G("id", OURS1, "'tm_' + the first 20 characters of a SHA-256 hash of player id, transfer date, from-club id "
                  "and to-club id. One per row; the key of every later table.", None),
    "transfermarkt_transfer_id": G("id", TMR + ": the number in the row's URL", "copied", None),
    "event_chain_id": G("legacy", OURS1, "A row joins the previous row's chain if it is the same player, it leaves the club "
                        "the previous row arrived at, and it is dated within 14 days of it (B→A then A→C). Not used by "
                        "any Week 4–6 analysis.", None),
    "player_id": G("id", TM, "copied (Transfermarkt player id; the join key to players, appearances, valuations and the "
                   "Kaggle injury files)", None),
    "player_name": G("desc", TM, "copied; display only (names are not unique)", None),
    "date_of_birth": G("core", DP, "copied; feeds age", None),
    "age_at_transfer": G("desc", OURS1, "completed years between date_of_birth and transfer_date (birthday rule). NOT the "
                         "regression age, which is continuous (master `age_exact`).", None),
    "position": G("core", DP, "copied; becomes the position control (master `position_clean`)",
                  {"Goalkeeper": "goalkeeper", "Defender": "defender", "Midfield": "midfielder", "Attack": "forward",
                   "Missing": "no position recorded upstream (the episode leaves the controlled models)"}),
    "country_of_citizenship": G("desc", DP, "copied; not used", None),
    "transfer_date": G("core", TM, "copied. Can lie in the future (moves Transfermarkt already lists, mostly loan returns).",
                       None),
    "transfer_season": G("core", TM, "copied, 'YY/YY'", None),
    "from_club_id": G("core", TM, "copied; the club the player leaves (the lender, for a loan)", None),
    "from_club_name": G("core", TM, "copied; read by the placeholder, youth/reserve and internal-move rules", None),
    "to_club_id": G("core", TM, "copied; the receiving club B (the borrower, for a loan)", None),
    "to_club_name": G("core", TM, "copied; read by the same name rules", None),
    "transfer_type_raw": G("core", OURS + " from " + TMR + " (the fee cell text)",
                           "the exact text of Transfermarkt's fee cell, mapped one-to-one to a label. No guessing: text "
                           "that matches no rule is 'unknown'.",
                           {"no_fee_shown": "cell shows '-': no fee given at all",
                            "free_transfer": "cell says 'free transfer'",
                            "end_of_loan": "cell says 'End of loan': the player returns from a loan",
                            "loan_transfer": "cell says 'loan transfer': a loan, no fee shown",
                            "paid_transfer": "cell shows an amount (e.g. '€1.50m'): a paid permanent transfer",
                            "undisclosed": "cell shows '?': a deal happened, fee not published",
                            "loan_fee": "cell says 'Loan fee: €…': a loan with a published fee",
                            "draft": "cell says 'draft' (draft systems, mostly North America)",
                            "end_of_loan_with_fee": "'End of loan' with an amount: money paid on the return",
                            "unknown": "cell text matches none of the rules"}),
    "transfer_type_normalized": G("core", OURS,
                                  "transfer_type_raw mapped to a type. One exception: where the label says nothing about "
                                  "the move ('-', '?' or unknown) and both clubs are one organisation (e.g. U19 → first "
                                  "team), the type becomes youth_or_internal.",
                                  {"free_transfer": "permanent move, Transfermarkt says free",
                                   "loan": "loan out (loan transfer or loan fee)",
                                   "loan_return": "end of a loan (with or without a fee)",
                                   "no_fee_shown": "permanent move with '-' in the fee cell",
                                   "youth_or_internal": "move inside one organisation (academy, reserve, first team)",
                                   "permanent_transfer": "paid permanent transfer",
                                   "undisclosed_transfer": "permanent move with '?' (fee withheld)",
                                   "other": "draft row",
                                   "unknown": "type could not be established"}),
    "transfer_type_source": G("qa", OURS, "which rule set transfer_type_normalized",
                              {"transfermarkt_raw_label": "Transfermarkt's own fee label",
                               "club_name_pattern": "the internal-move rule (club names), because the label said nothing"}),
    "stage1_heuristic_class": G("legacy", OURS1 + ", from the DuckDB fee number and the order of the player's rows",
                                "the old guess at the row type made before the raw fee labels were restored. Kept only "
                                "to measure how wrong it was (heuristic_agrees_with_label).",
                                {"no_fee_recorded": "DuckDB fee is empty",
                                 "zero_fee_move": "DuckDB fee is 0",
                                 "loan_out": "the next row reverses this move within 550 days, both without fee",
                                 "loan_return": "reverses the preceding loan_out row",
                                 "permanent_with_fee": "DuckDB fee above 0",
                                 "exit_to_no_club": "moves to a placeholder (" + PH + ")",
                                 "arrival_from_no_club": "comes from a placeholder",
                                 "retirement": "moves to 'Retired'",
                                 "same_club_record_artifact": "from-club id equals to-club id"}),
    "heuristic_agrees_with_label": G("qa", OURS, "does the old heuristic class match the restored Transfermarkt label?",
                                     {"True": "agrees", "False": "disagrees",
                                      "(blank)": "not comparable (placeholder or same-club rows)"}),
    "fee_display_raw": G("qa", TMR, "copied byte-for-byte (the source of every fee field below)", None),
    "fee_numeric_eur": G("core", OURS, "the euro amount parsed from the fee text ('€1.50m' → 1,500,000; 'free transfer' → "
                         "0); blank when the text carries no amount", None),
    "fee_role": G("core", OURS, "what kind of money the parsed amount is",
                  {"permanent_fee": "fee of a paid permanent transfer", "loan_fee": "fee paid for a loan",
                   "fee_on_return": "amount on an end-of-loan row", "none_free": "free transfer (amount 0)",
                   "(blank)": "no amount on the row"}),
    "fee_disclosure_status": G("core", OURS, "what Transfermarkt told us about the fee, separate from the amount",
                               {"no_fee_shown": "'-' or a bare 'loan transfer': no fee cell, so no zero is assumed",
                                "disclosed_free": "Transfermarkt asserts 'free transfer' (a stated zero)",
                                "not_applicable": "end of loan or draft: no fee concept",
                                "disclosed_amount": "an amount was published",
                                "undisclosed": "'?': a deal happened, terms withheld",
                                "unknown": "label could not be read"}),
    "permanent_transfer_fee_eur": G("core", OURS, "the amount on a paid transfer; 0 on 'free transfer'; blank otherwise",
                                    None),
    "loan_fee_eur": G("core", OURS, "the amount on a 'Loan fee' row; blank otherwise", None),
    "fee_on_return_eur": G("core", OURS, "the amount on an 'End of loan' row with a fee; blank otherwise", None),
    "duckdb_transfer_fee": G("qa", TM + " `transfer_fee`", "copied. Not used: the DuckDB writes 0 for many rows that have "
                             "no fee at all, which is why the raw labels were restored.", None),
    "fee_matches_duckdb": G("qa", OURS, "on paid and free rows: does our fee equal the DuckDB number?",
                            {"True": "equal", "(blank)": "not comparable (any other label)"}),
    "market_value_eur": G("desc", TM + " `market_value_in_eur`", "copied. Not the regression value: the regressions use "
                          "the player's latest valuation strictly before the move (master `player_value`).", None),
    "market_value_display": G("qa", TMR, "copied display text (e.g. '€100k', '-')", None),
    "is_loan_related": G("core", OURS, "the label is a loan or an end-of-loan label",
                         {"True": "loan_transfer, loan_fee, end_of_loan or end_of_loan_with_fee", "False": "any other"}),
    "return_leg_has_fee": G("legacy", OURS, "an end-of-loan row with an amount above 0 (option or obligation exercised); "
                            "used by the Stage 2 contract research", {"True": "fee on the return", "False": "no fee"}),
    "transfermarkt_future_transfer": G("filter", TMR + " `futureTransfer`",
                                       "copied. True on moves Transfermarkt lists as upcoming when the history was "
                                       "captured; used to set aside scheduled moves and unrealised loan endings. The exact "
                                       "upstream definition is not documented.",
                                       {"True": "listed as an upcoming move", "False": "already happened at capture"}),
    "involves_placeholder_club": G("filter", OURS, "either side is " + PH + "; such rows are not club-to-club transfers "
                                   "and become no episode", {"True": "a side is a placeholder", "False": "two real clubs"}),
    "is_youth_or_reserve_side": G("filter", OURS, "either club name matches the youth/reserve pattern (U19, U23, Youth, "
                                  "Academy, Castilla, ' II', ' B', …)", {"True": "a youth or reserve side",
                                                                        "False": "two senior names"}),
    "is_internal_move": G("filter", OURS, "both names are equal once youth/reserve suffixes are removed, the club ids "
                          "differ and no side is a placeholder; such rows become no episode",
                          {"True": "move inside one organisation", "False": "between organisations"}),
    "is_senior_move": G("legacy", OURS, "none of the three flags above; used by the Stage 2 research rule",
                        {"True": "senior club-to-club move", "False": "otherwise"}),
    "has_monetary_amount": G("legacy", OURS, "any of the three fee fields above 0; used by the Stage 2 research rule",
                             {"True": "money stated", "False": "no amount"}),
    "enrichment_matched": G("qa", OURS, "a raw transfer-history row was found for this DuckDB row",
                            {"True": "matched (every row)"}),
    "is_research_target": G("legacy", OURS, "eligible for the Stage 2 contract research (an amount, or a senior move of a "
                            "researchable type)", {"True": "Stage 2 target", "False": "not a target"}),
    "research_exclusion_reason": G("legacy", OURS, "why a row is not a Stage 2 target (first rule that applies)",
                                   {"(blank)": "is a target",
                                    "loan_return_bookkeeping_row_no_deal_terms": "plain end-of-loan row",
                                    "internal_registration_change_within_one_organisation": "internal move",
                                    "youth_or_reserve_side_and_no_disclosed_amount": "youth/reserve side without an amount",
                                    "no_fee_shown_no_evidence_of_a_negotiated_deal": "'-' row",
                                    "placeholder_club_not_a_club_to_club_transfer": "placeholder side",
                                    "not_a_standard_club_to_club_transfer": "draft row",
                                    "transfer_type_could_not_be_established": "unknown type"}),
}

# ---------------------------------------------------------------------------
# Master: origin and usefulness of every column
# ---------------------------------------------------------------------------

MG = namedtuple("MG", "origin usefulness")

MASTER_GUIDE = {
    # A identity
    "event_id": MG("s1c", "id"), "transfermarkt_transfer_id": MG("tm", "id"), "event_chain_id": MG("s1c", "legacy"),
    "player_id": MG("tm", "id"), "player_name": MG("tm", "id"),
    # B transfer
    "transfer_date": MG("tm", "core"), "transfer_season": MG("tm", "core"), "season": MG("master", "core"),
    "episode_type": MG("s1c", "core"), "kind": MG("master", "core"), "transfer_type_raw": MG("s1c", "core"),
    "transfer_type_normalized": MG("s1c", "helper"), "fee_display_raw": MG("tm", "qa"),
    "fee_disclosure_status": MG("s1c", "core"), "permanent_transfer_fee_eur": MG("s1c", "core"),
    "loan_fee_eur": MG("s1c", "core"), "from_club_id": MG("tm", "core"), "from_club_name": MG("tm", "core"),
    "to_club_id": MG("tm", "core"), "to_club_name": MG("tm", "core"), "is_youth_or_reserve_side": MG("s1c", "filter"),
    "is_senior_move": MG("s1c", "legacy"), "transfermarkt_future_transfer": MG("tm", "filter"),
    "is_research_target": MG("s1c", "legacy"),
    "from_club_parent_id_v2": MG("org", "helper"), "same_organisation_move_v2": MG("org", "filter"),
    "academy_promotion_candidate": MG("org", "filter"), "same_org_confidence": MG("org", "qa"),
    "same_org_reason": MG("org", "qa"),
    # C loan ending
    "match_reason": MG("multi", "qa"), "has_matched_return": MG("multi", "helper"), "terminal_event_id": MG("multi", "id"),
    "terminal_date": MG("multi", "core"), "terminal_type": MG("multi", "helper"), "terminal_raw_label": MG("multi", "qa"),
    "raw_label_class": MG("multi", "qa"), "terminal_scheduled_future": MG("multi", "filter"),
    "ending_observed": MG("master", "helper"), "ending_realised": MG("master", "filter"),
    "duration_days": MG("master", "core"), "ending_fee_on_return_eur": MG("multi", "special"),
    "return_boundary_days": MG("master", "helper"), "follow_on_event_id": MG("multi", "helper"),
    "follow_on_gap_days": MG("multi", "helper"), "follow_on_type": MG("multi", "helper"),
    "follow_on_to_club": MG("multi", "helper"), "economic_ending": MG("multi", "legacy"),
    "ending_detail": MG("multi", "legacy"), "next_move_event_id": MG("multi", "id"),
    "next_move_gap_days": MG("multi", "helper"), "next_move_type": MG("multi", "helper"),
    "next_category": MG("multi", "core"),
    # D geography / calendar
    "from_country_v1": MG("geo", "legacy"), "to_country_v1": MG("geo", "legacy"), "from_country_v2": MG("geo", "core"),
    "to_country_v2": MG("geo", "core"), "from_geography_source_v2": MG("geo", "qa"),
    "to_geography_source_v2": MG("geo", "qa"), "calendar_context": MG("master", "core"), "calendar_key": MG("master", "helper"),
    "ending_in_covid_window": MG("master", "helper"), "calendar_class": MG("cal", "core"), "calendar_state": MG("cal", "core"),
    "calendar_classifiable": MG("cal", "filter"), "off_calendar": MG("cal", "core"),
    "matched_standard_mmdd": MG("cal", "helper"), "nearest_standard_mmdd": MG("cal", "helper"),
    "days_to_nearest_standard": MG("cal", "helper"), "unclassified_reason": MG("cal", "qa"),
    "loan_calendar_status": MG("cal", "core"), "off_calendar_flag": MG("cal", "core"),
    # E playing time
    "receiving_league": MG("games", "core"), "arrival_season": MG("games", "core"), "departure_date": MG("multi", "helper"),
    "departed_in_window": MG("games", "helper"), "window_start": MG("master", "helper"), "window_end": MG("games", "helper"),
    "season_games": MG("games", "helper"), "games_remaining_at_arrival": MG("games", "helper"),
    "available_games": MG("games", "core"), "complete_games": MG("games", "qa"), "appearances": MG("games", "core"),
    "minutes": MG("games", "core"), "available_minutes": MG("master", "helper"), "minutes_share": MG("master", "core"),
    "true_zero": MG("master", "core"), "pt_exclusion_reason": MG("sample", "qa"), "prior_spell_at_b": MG("multi", "filter"),
    "pt_loan_ending": MG("multi", "filter"), "arrived_full_season": MG("master", "legacy"), "remain": MG("master", "core"),
    # F player
    "date_of_birth": MG("players", "helper"), "age_at_transfer": MG("s1c", "special"), "age_exact": MG("master", "core"),
    "age_years": MG("master", "helper"), "older": MG("master", "core"), "position": MG("players", "helper"),
    "position_clean": MG("master", "core"), "country_of_citizenship": MG("players", "special"),
    # G market value
    "market_value_eur": MG("tm", "special"), "player_value": MG("val", "core"), "player_value_date": MG("val", "qa"),
    "player_value_gap_days": MG("master", "qa"), "stale_player_value": MG("master", "filter"),
    "squad_value": MG("val", "core"), "squad_players": MG("val", "qa"), "focal_in_squad": MG("val", "qa"),
    "relative_value": MG("master", "core"), "log_player_value": MG("master", "core"), "log_squad_value": MG("master", "core"),
    "log_relative_value": MG("master", "core"), "league_season": MG("master", "core"),
    # H injury
    "injury_join_status": MG("kaggle", "qa"), "available_games_current": MG("master", "helper"),
    "confirmed_injury_games": MG("kaggle", "core"), "available_games_fit": MG("master", "core"),
    "current_share": MG("master", "core"), "fit_adjusted_share": MG("master", "core"),
    "injury_days_overlap": MG("kaggle", "special"), "injury_spell_count": MG("kaggle", "special"),
    "appearance_conflicts": MG("kaggle", "qa"), "injury_data_quality_flag": MG("kaggle", "filter"),
    "injury_games_strict": MG("kaggle", "special"), "fit_share_strict": MG("master", "special"),
    "injury_games_severe3": MG("kaggle", "special"), "fit_share_severe3": MG("master", "special"),
    "injury_games_long5": MG("kaggle", "special"), "fit_share_long5": MG("master", "special"),
    # I / J samples
    "in_calendar_population": MG("sample", "filter"), "in_playing_time_candidates": MG("sample", "filter"),
    "pt_realised": MG("sample", "filter"), "pt_covered_league_season": MG("sample", "filter"),
    "pt_matched": MG("sample", "filter"), "pt_reliable": MG("sample", "filter"),
    "pt_lineup_games_checked": MG("games", "qa"), "pt_games_played_without_row": MG("games", "qa"),
    "pt_contradiction_type": MG("games", "qa"), "pt_lineup_contradicted": MG("games", "filter"),
    "pt_reliable_v2": MG("sample", "filter"),
    "in_old_primary_sample": MG("sample", "legacy"), "in_revised_sample": MG("sample", "filter"),
    "in_injury_usable": MG("sample", "filter"), "calendar_first_eligible": MG("sample", "filter"),
    "in_cf_revised_sample": MG("sample", "filter"), "in_cf_injury_usable": MG("sample", "filter"),
    "in_old_regression_sample": MG("sample", "legacy"), "in_age_model_sample": MG("sample", "filter"),
    "in_injury_model_sample": MG("sample", "filter"), "in_cf_main_model_sample": MG("sample", "filter"),
    "in_cf_injury_model_sample": MG("sample", "filter"),
}

# The columns to reach for first; the short guide (master_core_columns.md) lists exactly these, in this order.
CORE_SHORTLIST = [
    ("Who and when", ["player_id", "player_name", "transfer_date", "season", "kind", "episode_type"]),
    ("Clubs and fee", ["from_club_id", "from_club_name", "to_club_id", "to_club_name", "fee_disclosure_status",
                       "permanent_transfer_fee_eur", "loan_fee_eur"]),
    ("Loan ending and calendar", ["terminal_date", "duration_days", "next_category", "from_country_v2", "to_country_v2",
                                  "calendar_context", "loan_calendar_status", "off_calendar_flag", "calendar_state"]),
    ("Playing time", ["receiving_league", "arrival_season", "available_games", "appearances", "minutes", "minutes_share",
                      "true_zero", "remain"]),
    ("Player", ["age_exact", "older", "position_clean"]),
    ("Market value", ["player_value", "squad_value", "relative_value", "log_relative_value", "log_squad_value",
                      "league_season"]),
    ("Injury", ["confirmed_injury_games", "available_games_fit", "current_share", "fit_adjusted_share"]),
    ("Samples (filter on these)", ["in_revised_sample", "in_age_model_sample", "in_injury_usable", "in_injury_model_sample",
                                   "calendar_first_eligible", "in_cf_revised_sample", "in_cf_main_model_sample",
                                   "in_cf_injury_usable", "in_cf_injury_model_sample"]),
    ("Academy-promotion sensitivity", ["same_organisation_move_v2", "academy_promotion_candidate", "same_org_confidence"]),
]

# ---------------------------------------------------------------------------
# Master: when each column can be interpreted
# ---------------------------------------------------------------------------

# domain key -> (the rows on which the column means something, mask over the master)
DOMAINS = {
    "all": ("every episode", lambda M: M.event_id.notna()),
    "typed": ("every episode of known type (kind not blank)", lambda M: M.kind.notna()),
    "loan": ("loans only (kind = loan)", lambda M: M.kind.eq("loan")),
    "perm_fee": ("permanent moves labelled paid or free transfer (transfer_type_raw = paid_transfer or free_transfer)",
                 lambda M: M.transfer_type_raw.isin(["paid_transfer", "free_transfer"])),
    "loan_fee": ("loans with a published loan fee (transfer_type_raw = loan_fee)",
                 lambda M: M.transfer_type_raw.eq("loan_fee")),
    "loan_end": ("loans with a recorded ending (terminal_event_id present)", lambda M: M.terminal_event_id.notna()),
    "realised": ("loans with a realised ending (in_calendar_population = True)", lambda M: M.in_calendar_population),
    "classifiable": ("classifiable realised loan endings (loan_calendar_status = normal or off_calendar)",
                     lambda M: M.loan_calendar_status.isin(["normal", "off_calendar"])),
    "normal": ("calendar-normal loan endings (loan_calendar_status = normal)",
               lambda M: M.loan_calendar_status.eq("normal")),
    "covered": ("receiving club in a covered league season (pt_covered_league_season = True)",
                lambda M: M.pt_covered_league_season),
    "reliable": ("reliable playing-time outcome (pt_reliable = True)", lambda M: M.pt_reliable),
    "revised": ("revised playing-time sample (in_revised_sample = True)", lambda M: M.in_revised_sample),
    "usable": ("injury data usable (in_injury_usable = True; every such row has at least one fit game)",
               lambda M: M.in_injury_usable),
    "has_dob": ("date of birth known", lambda M: M.date_of_birth.notna()),
}

# column -> (domain, what to watch for, outside): outside = "blank" means the column is empty off its domain (checked);
# "filled" means it carries a placeholder value off its domain that must NOT be interpreted (checked: count > 0).
VW = namedtuple("VW", "domain note outside")
_B, _F = "blank", "filled"
VALID_WHEN = {
    **{c: VW("all", "", _B) for c in (
        "event_id", "transfermarkt_transfer_id", "event_chain_id", "player_id", "player_name", "transfer_date",
        "transfer_season", "season", "episode_type", "transfer_type_raw", "transfer_type_normalized", "fee_display_raw",
        "fee_disclosure_status", "from_club_id", "from_club_name", "to_club_id", "to_club_name", "is_senior_move",
        "transfermarkt_future_transfer", "is_research_target", "from_country_v1", "to_country_v1", "from_country_v2",
        "to_country_v2", "from_geography_source_v2", "to_geography_source_v2", "date_of_birth", "position",
        "position_clean", "country_of_citizenship", "same_organisation_move_v2", "academy_promotion_candidate",
        "same_org_confidence", "same_org_reason", "in_calendar_population", "in_playing_time_candidates", "pt_realised",
        "pt_covered_league_season", "pt_matched", "pt_reliable", "in_old_primary_sample", "in_revised_sample",
        "in_injury_usable", "calendar_first_eligible", "in_cf_revised_sample", "in_cf_injury_usable",
        "in_old_regression_sample", "in_age_model_sample", "in_injury_model_sample", "in_cf_main_model_sample",
        "in_cf_injury_model_sample")},
    "kind": VW("all", "blank for the 4 unknown-type episodes", _B),
    "is_youth_or_reserve_side": VW("all", "Stage 1C name pattern; also True for senior clubs named '… II' (Willem II): "
                                          "use same_org_confidence for organisation questions", _B),
    "market_value_eur": VW("all", "Transfermarkt's value shown on the transfer row; blank when none was shown", _B),
    "from_club_parent_id_v2": VW("all", "blank unless club A is a youth/reserve side with a resolved parent", _B),
    "loan_calendar_status": VW("all", "permanent and unknown-type episodes are 'not_applicable'", _B),
    "permanent_transfer_fee_eur": VW("perm_fee", "0 = 'free transfer'; blank for undisclosed, no-fee-shown and loans", _B),
    "loan_fee_eur": VW("loan_fee", "blank for every other loan: no published fee is not a zero fee", _B),
    **{c: VW("loan", "", _B) for c in (
        "match_reason", "has_matched_return", "raw_label_class", "terminal_scheduled_future", "ending_observed",
        "ending_realised", "economic_ending", "ending_detail", "next_category", "pt_loan_ending")},
    **{c: VW("loan", "blank when the player has no later movement", _B) for c in (
        "next_move_event_id", "next_move_gap_days", "next_move_type")},
    **{c: VW("loan_end", "", _B) for c in ("terminal_event_id", "terminal_date", "terminal_type", "terminal_raw_label",
                                          "duration_days")},
    "return_boundary_days": VW("loan_end", "blank unless the ending is a return row", _B),
    "ending_fee_on_return_eur": VW("loan_end", "blank unless the return row carries a fee", _B),
    **{c: VW("loan_end", "blank when no follow-on deal within 60 days of the return", _B) for c in (
        "follow_on_event_id", "follow_on_gap_days", "follow_on_type", "follow_on_to_club")},
    **{c: VW("realised", "", _B) for c in ("calendar_context", "calendar_key", "ending_in_covid_window", "calendar_class",
                                          "calendar_state", "calendar_classifiable", "unclassified_reason")},
    **{c: VW("classifiable", "", _B) for c in ("off_calendar", "off_calendar_flag", "nearest_standard_mmdd",
                                              "days_to_nearest_standard")},
    "matched_standard_mmdd": VW("normal", "", _B),
    "receiving_league": VW("covered", "elsewhere '(not in a covered league)'", _F),
    **{c: VW("covered", "", _B) for c in ("arrival_season", "window_start", "window_end", "season_games",
                                         "games_remaining_at_arrival", "available_games", "complete_games", "remain")},
    "departed_in_window": VW("covered", "False elsewhere is a placeholder, not 'stayed'", _F),
    "arrived_full_season": VW("covered", "False elsewhere is a placeholder (legacy two-thirds rule)", _F),
    **{c: VW("typed", "", _B) for c in ("departure_date", "pt_exclusion_reason", "prior_spell_at_b")},
    **{c: VW("reliable", "", _B) for c in ("appearances", "minutes", "available_minutes", "minutes_share")},
    "true_zero": VW("reliable", "False outside reliable episodes is a placeholder: never read it as 'played'", _F),
    **{c: VW("reliable", "0 checked games for 2012/13 arrivals (no line-ups in the snapshot)", _B)
       for c in ("pt_lineup_games_checked", "pt_games_played_without_row", "pt_contradiction_type")},
    "pt_lineup_contradicted": VW("reliable", "False outside reliable episodes means 'not checked'", _F),
    "pt_reliable_v2": VW("all", "", _B),
    **{c: VW("has_dob", "", _B) for c in ("age_at_transfer", "age_exact", "age_years", "older")},
    **{c: VW("revised", "blank when no valuation strictly before the move", _B) for c in (
        "player_value", "player_value_date", "player_value_gap_days", "log_player_value")},
    **{c: VW("revised", "blank when fewer than the minimum number of valued squad players", _B) for c in (
        "squad_value", "squad_players", "log_squad_value")},
    **{c: VW("revised", "blank when player or squad value is missing", _B) for c in ("relative_value",
                                                                                     "log_relative_value")},
    **{c: VW("revised", "", _B) for c in ("stale_player_value", "focal_in_squad", "league_season", "injury_join_status",
                                         "injury_data_quality_flag", "available_games_current", "current_share")},
    **{c: VW("usable", "computed for every revised episode, but only interpretable where the injury history is usable",
             _F) for c in ("confirmed_injury_games", "available_games_fit", "fit_adjusted_share", "injury_days_overlap",
                           "injury_spell_count", "appearance_conflicts", "injury_games_strict", "fit_share_strict",
                           "injury_games_severe3", "fit_share_severe3", "injury_games_long5", "fit_share_long5")},
}
