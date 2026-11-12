"""Definitions of the master columns added for the November 2026 professor feedback (`feedback_nov2026.py`).

One record per column feeds every documentation table: the variable dictionary (`data_lineage_specs.VARS`), the column
guide (`column_guide_specs.MASTER_GUIDE`, `VALID_WHEN`, `DOMAINS`) and the provenance matrix
(`variable_provenance.MASTER_OWN`). Counts and value sets are computed from the data, never written here.
"""
from __future__ import annotations

from collections import namedtuple

FB = "src/analysis/feedback_nov2026.py extend (in memory)"
EV = "event_id"
FLAG = "flag (descriptive; never used to drop rows)"
FILT = "sample filter (defines a sample)"
CAT = "category / classification"
VAL = "value"
IDT = "identifier"
MECH, CLASS, SFLAG, JOIN, EXT = ("OUR_MECHANICAL_DERIVATION", "OUR_CLASSIFICATION_RULE", "OUR_SAMPLE_FLAG",
                                 "UPSTREAM_JOINED", "EXTERNAL_RAW")
NEXT = ("stage1c.event_id stage1c.transfer_date stage1c.transfer_type_normalized stage1c.is_internal_move "
        "stage1c.from_club_id stage1c.to_club_id stage1c.involves_placeholder_club stage1c.transfermarkt_future_transfer")
GAMES = ("duckdb.games.game_id duckdb.games.competition_type duckdb.games.season duckdb.games.date "
         "duckdb.games.home_club_id duckdb.games.away_club_id")
APPS = "duckdb.appearances.game_id duckdb.appearances.player_id duckdb.appearances.player_club_id"
EVENTS = ("duckdb.game_lineups.type duckdb.game_events.type duckdb.game_events.minute duckdb.game_events.player_in_id "
          "duckdb.game_events.description duckdb.appearances.minutes_played")
KINJ = "kaggle_inj.player_id kaggle_inj.injury_reason kaggle_inj.from_date kaggle_inj.end_date"
PV = ("duckdb.player_valuations.player_id duckdb.player_valuations.date duckdb.player_valuations.market_value_in_eur "
      "duckdb.player_valuations.current_club_id")
TIERS = "tiers.club_id tiers.season tiers.league_name tiers.tier tiers.country tiers.source"

C = namedtuple("C", "variable section origin use domain outside note rod source cols keys rule missing kind used "
                    "definition caveat cat inputs code")


def c(variable, section, origin, use, domain, outside, note, rod, cols, rule, missing, kind, used, definition,
      caveat, cat, inputs, keys=EV, code="FB", source=FB):
    return C(variable, section, origin, use, domain, outside, note, rod, source, cols, keys, rule, missing, kind, used,
             definition, caveat, cat, inputs, code)


K = "K departure from the receiving club (Nov 2026)"
L = "L substitute-minute validation (Nov 2026)"
Mq = "M injury rule v2: quarantine counted (Nov 2026)"
N = "N league tiers (Nov 2026)"
O = "O prior-season playing time (Nov 2026)"
P = "P origin-club squad value (Nov 2026)"
Qv = "Q combined minute validation (Nov 2026 correction)"
NA_CAND = "Blank for the 4 unknown-type episodes (not playing-time candidates)."
NODEP = "Blank when no realised departure is recorded (none, still scheduled, or the next row does not leave the club)."
SCHED = ("Blank when the receiving club is not in a league of the snapshot in its arrival season, the move itself is "
         "scheduled, or the next recorded move does not leave the club.")

SPECS = [
    # ---------------- K. departure from the receiving club (Freund)
    c("next_departure_event_id", K, "multi", "id", "candidates", "blank", "blank when no later movement", "derived",
      "Stage 1C rows of the player", "first later Stage 1C row of the player, skipping internal registrations and "
      "draft/other rows (the playing_time_join departure rule)", "Blank: no later movement, or not a candidate.", IDT,
      "departure flags", "The movement that ended the stay at the receiving club.", "", MECH, NEXT),
    c("next_departure_date", K, "multi", "helper", "candidates", "blank", "", "derived", "next row transfer_date",
      "transfer_date of next_departure_event_id", "Blank: no later movement, or not a candidate.", VAL,
      "days_at_receiving_club", "Date of the next recorded movement.",
      "Can be a scheduled future date: read with next_departure_type.", MECH, NEXT),
    c("next_departure_type", K, "multi", "core", "candidates", "blank", "", "derived",
      "next row type, clubs, future flag", "first matching of: none_recorded; scheduled_not_realised (future row); "
      "inconsistent_history (the row's origin organisation is not the receiving club's); released_or_retired "
      "(placeholder destination); loan_out (type loan); permanent_transfer (paid, free, undisclosed, no fee shown); "
      "loan_return (end-of-loan row); unknown_type", NA_CAND, CAT, "departure flags",
      "How the player left the receiving club.", "", CLASS, NEXT),
    c("next_departure_to_club_id", K, "multi", "helper", "candidates", "blank", "", "derived", "next row to_club_id",
      "to_club_id of next_departure_event_id", "Blank: no later movement.", IDT, "examples",
      "Club the player moved to next.", "", MECH, NEXT),
    c("next_departure_to_club_name", K, "multi", "helper", "candidates", "blank", "", "derived",
      "next row to_club_name", "to_club_name of next_departure_event_id", "Blank: no later movement.", VAL,
      "examples", "Name of that club.", "", MECH, NEXT + " stage1c.to_club_name"),
    c("days_at_receiving_club", K, "multi", "core", "candidates", "blank", "blank unless the departure is realised",
      "derived", "transfer_date, next_departure_date", "next_departure_date - transfer_date in days, only when the next "
      "move is realised and leaves the club (types loan_out, permanent_transfer, loan_return, released_or_retired, "
      "unknown_type)", NODEP, VAL, "departure QA", "Days between arrival and the next departure.",
      "0 or 1 = moved on the same or next day (e.g. bought and loaned out at once).", MECH,
      NEXT + " master.transfer_date"),
    c("next_departure_is_loan", K, "multi", "filter", "candidates", "blank", "blank when no realised departure of "
      "known type", "derived", "next_departure_type", "True if next_departure_type = loan_out; False for another "
      "realised known type", NODEP, FLAG, "departure flags", "Next departure was a loan out.", "", MECH,
      "master.next_departure_type"),
    c("next_departure_is_permanent", K, "multi", "filter", "candidates", "blank", "blank when no realised departure of "
      "known type", "derived", "next_departure_type", "True if next_departure_type = permanent_transfer; False for "
      "another realised known type", NODEP, FLAG, "departure flags", "Next departure was a permanent transfer.",
      "A release to Without Club is neither a loan nor a permanent transfer (both False).", MECH,
      "master.next_departure_type"),
    c("receiving_first_league_game_date", K, "games", "helper", "candidates", "blank", "blank when the club's league "
      "schedule is not in the snapshot", "derived (joined)", "games (domestic league, all 31 leagues)",
      "first league game of the receiving club on or after the move, in its arrival season (playing_time_join's "
      "arrival-season rule applied to every league of the snapshot)", SCHED, VAL, "left_before_first_…",
      "Receiving club's first league game after the arrival.", "", JOIN, GAMES),
    c("receiving_last_league_game_date", K, "games", "helper", "candidates", "blank", "", "derived (joined)",
      "games (domestic league)", "last league game of the receiving club in the arrival season", SCHED, VAL,
      "stayed_through_…", "Receiving club's last league game of the arrival season.", "", JOIN, GAMES),
    c("left_before_first_receiving_league_game", K, "games", "core", "candidates", "blank", "blank = unknown",
      "derived", "next_departure_date, receiving_first_league_game_date, history capture date",
      "True if the realised departure is on or before the first league game (the playing-time window ends the day "
      "before the departure, so that game is not counted); False if later, or if no departure is recorded and the "
      "history was captured after that game; blank otherwise", SCHED, FLAG, "departure flags",
      "The player left before the receiving club played a league game.",
      "Equals 'no league game in the window' for every covered episode except one with an inconsistent next move.",
      CLASS, "master.next_departure_type master.next_departure_date master.receiving_first_league_game_date"),
    c("stayed_through_arrival_season_end", K, "games", "core", "candidates", "blank", "blank = unknown", "derived",
      "next_departure_date, receiving_last_league_game_date, history capture date",
      "True if the realised departure is after the club's last league game of the arrival season, or none is recorded "
      "and the history was captured after that game; False if the departure is on or before it; blank otherwise",
      SCHED, FLAG, "departure flags", "The player was still at the club when its league season ended.", "",
      CLASS, "master.next_departure_type master.next_departure_date master.receiving_last_league_game_date"),
    c("arrival_season_presence", K, "games", "core", "candidates", "blank", "", "derived",
      "the two flags above, schedule, next_departure_type", "left_before_first_league_game / left_during_season / "
      "stayed_through_season_end / unknown_next_move_inconsistent / unknown_season_not_finished / "
      "unknown_no_league_schedule / not_applicable_move_scheduled", NA_CAND, CAT, "departure flags",
      "Presence at the receiving club over its arrival season, from dates only.",
      "Dates only: combine with appearances / minutes to separate 'stayed but did not play' from 'played'.", CLASS,
      "master.left_before_first_receiving_league_game master.stayed_through_arrival_season_end master.next_departure_type"),
    c("perm_moved_on_before_first_league_game", K, "games", "filter", "permanent", "blank", "blank = unknown",
      "derived (master helper)", "kind, left_before_first_receiving_league_game",
      "left_before_first_receiving_league_game for permanent arrivals; blank for loans", "Blank for loans, and for "
      "permanent arrivals whose status is unknown.", FLAG, "departure flags (Freund's question)",
      "Permanent arrival that moved on before the club's first league game (e.g. bought and loaned out at once).",
      "Never used to drop rows from the master: such episodes are already outside the playing-time sample (no league "
      "game in the window).", SFLAG, "master.kind master.left_before_first_receiving_league_game"),
    # ---------------- L. substitute-minute validation (Fernandes)
    c("pt_minutes_rows_verified", L, "games", "qa", "reliable", "blank", "", "derived (joined)",
      "appearances, game_lineups, game_events", "league appearance rows in the window whose recorded minutes equal the "
      "line-up/event minutes within 2 (conventions) or match a shortened game's length", "Blank unless pt_reliable.",
      VAL, "minutes QA", "Window appearances confirmed by line-ups and events.", "", CLASS, EVENTS + " " + APPS),
    c("pt_minutes_rows_unverifiable", L, "games", "qa", "reliable", "blank", "", "derived (joined)",
      "appearances, game_lineups, game_events", "rows with no line-up for the club-game, player not listed, missing "
      "entry event, unknown event minute (-1) or a line-up/event conflict", "Blank unless pt_reliable.", VAL,
      "minutes QA", "Window appearances that cannot be checked.", "Kept: unverifiable is not evidence of an error.",
      CLASS, EVENTS + " " + APPS),
    c("pt_minutes_rows_contradicted", L, "games", "qa", "reliable", "blank", "", "derived (joined)",
      "appearances, game_lineups, game_events", "rows whose recorded minutes differ by more than 2 from the line-up "
      "and events (not explained by a shortened game) or exceed 90", "Blank unless pt_reliable.", VAL, "minutes QA",
      "Window appearances contradicted by line-ups and events.", "", CLASS, EVENTS + " " + APPS),
    c("pt_minutes_validation_status", L, "games", "qa", "reliable", "blank", "", "derived",
      "the three counts", "contradicted_record if any contradicted row; else verified_except_unverifiable_rows if any "
      "unverifiable; else all_rows_verified; no_appearances_to_check when the player has no window appearance",
      "Blank unless pt_reliable.", CAT, "minutes QA", "Outcome of the minute check for the episode.", "", CLASS,
      "master.pt_minutes_rows_verified master.pt_minutes_rows_unverifiable master.pt_minutes_rows_contradicted"),
    c("pt_minutes_validated", L, "sample", "filter", "all", "blank", "False outside reliable episodes", "derived",
      "pt_reliable, pt_minutes_validation_status", "pt_reliable and status != contradicted_record", "Never missing.",
      FILT, "validated outcome", "Reliable outcome whose minutes are not contradicted.", "", SFLAG,
      "master.pt_reliable master.pt_minutes_validation_status"),
    c("minutes_share_validated", L, "games", "robustness", "reliable", "blank", "blank for contradicted records",
      "derived", "minutes_share, pt_minutes_validated", "minutes_share where pt_minutes_validated, else blank",
      "Blank unless pt_minutes_validated. No minutes are edited: a contradicted episode is left out, not corrected.",
      VAL, "validated regressions", "Minutes share with contradicted records removed.", "", MECH,
      "master.minutes_share master.pt_minutes_validated"),
    c("in_age_model_sample_v2", L, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_age_model_sample, pt_minutes_validated", "in_age_model_sample and pt_minutes_validated", "Never missing.",
      FILT, "no-calendar main regression (validated minutes)", "Main regression sample without the calendar gate, "
      "contradicted minutes removed.", "", SFLAG, "master.in_age_model_sample master.pt_minutes_validated"),
    c("in_cf_main_model_sample_v2", L, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_cf_main_model_sample, pt_minutes_validated", "in_cf_main_model_sample and pt_minutes_validated",
      "Never missing.", FILT, "calendar-first main regression (validated minutes)",
      "Calendar-first main regression sample, contradicted minutes removed.", "", SFLAG,
      "master.in_cf_main_model_sample master.pt_minutes_validated"),
    # ---------------- M. injury rule v2 (Freund)
    c("quarantine_unavailable_games", Mq, "kaggle", "qa", "usable_v2", "filled",
      "computed for every revised episode, but only interpretable where in_injury_usable_v2", "derived",
      "Kaggle injuries, window games", "window games newly unavailable under rule v2 (covered only by a Quarantine "
      "spell, player did not appear)", "Blank outside the revised sample.", VAL, "quarantine comparison",
      "Games added by counting quarantine.", "", CLASS, KINJ),
    c("confirmed_injury_games_v2", Mq, "kaggle", "core", "usable_v2", "filled",
      "computed for every revised episode, but only interpretable where in_injury_usable_v2", "derived",
      "Kaggle injuries, window games", "as confirmed_injury_games, with Quarantine spells counted (rule v2); a game "
      "is counted once whatever the number of covering spells; an appearance always overrides",
      "Blank outside the revised sample.", VAL, "injury models v2", "Window games missed for medical reasons or "
      "quarantine.", "Quarantine means unavailable (isolation), not a confirmed infection.", CLASS, KINJ),
    c("available_games_fit_v2", Mq, "kaggle", "core", "usable_v2", "filled",
      "computed for every revised episode, but only interpretable where in_injury_usable_v2", "derived",
      "available_games_current, confirmed_injury_games_v2", "available_games_current - confirmed_injury_games_v2",
      "Blank outside the revised sample.", VAL, "injury models v2", "Window games the player was available for.", "",
      MECH, "master.available_games_current master.confirmed_injury_games_v2"),
    c("fit_adjusted_share_v2", Mq, "kaggle", "core", "usable_v2", "filled",
      "computed for every revised episode, but only interpretable where in_injury_usable_v2", "derived",
      "minutes, available_games_fit_v2", "minutes / (90 × available_games_fit_v2); blank if no fit game",
      "Blank outside the revised sample, or no fit game.", VAL, "injury models v2 (preferred injury outcome)",
      "Playing-time share of the games the player was available for (rule v2).",
      "The minutes (numerator) are unchanged; only the denominator differs from fit_adjusted_share.", MECH,
      "master.minutes master.available_games_fit_v2"),
    c("injury_data_quality_flag_v2", Mq, "kaggle", "qa", "revised", "blank", "", "derived",
      "Kaggle injuries, profiles", "injury_adjusted.usability with rule v2 spells (open / missing-start checks now also "
      "see Quarantine rows)", "Blank outside the revised sample.", CAT, "injury sample v2",
      "Why the injury adjustment is or is not usable (rule v2).", "", CLASS, KINJ),
    c("in_injury_usable_v2", Mq, "sample", "filter", "all", "blank", "", "derived", "injury_data_quality_flag_v2",
      "in_revised_sample and injury_data_quality_flag_v2 = 'usable'", "Never missing.", FILT, "injury sample v2",
      "Injury adjustment v2 computable.", "", SFLAG, "master.injury_data_quality_flag_v2"),
    c("in_injury_model_sample_v2", Mq, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_injury_usable_v2, in_age_model_sample, pt_minutes_validated",
      "in_injury_usable_v2 and in_age_model_sample and pt_minutes_validated (the v1 identity in_injury_model_sample = "
      "in_injury_usable and in_age_model_sample is checked)", "Never missing.", FILT, "injury models v2",
      "Injury regression sample v2 (quarantine counted, contradicted minutes removed), no calendar gate.", "", SFLAG,
      "master.in_injury_usable_v2 master.in_age_model_sample master.pt_minutes_validated"),
    c("in_cf_injury_usable_v2", Mq, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_injury_usable_v2, calendar_first_eligible", "in_injury_usable_v2 and calendar_first_eligible",
      "Never missing.", FILT, "calendar-first injury sample v2", "Injury-usable v2 under the calendar-first sequence.",
      "", SFLAG, "master.in_injury_usable_v2 master.calendar_first_eligible"),
    c("in_cf_injury_model_sample_v2", Mq, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_injury_model_sample_v2, calendar_first_eligible", "in_injury_model_sample_v2 and calendar_first_eligible",
      "Never missing.", FILT, "calendar-first injury regression v2", "Calendar-first injury regression sample v2.", "",
      SFLAG, "master.in_injury_model_sample_v2 master.calendar_first_eligible"),
]

_TIER_SRC = "data/processed/league_tiers/club_season_league.csv (league_tiers.py)"
for side, club, who in (("from", "from_club_id", "origin club A"), ("to", "to_club_id", "receiving club B")):
    if side == "from":
        SPECS.append(c("tier_reference_season", N, "master", "helper", "all", "blank", "", "derived",
                       "arrival_season, season", "arrival_season where known, else the start year of the Transfermarkt "
                       "season label (2024 = 2024/25)", "Never missing.", VAL, "league tiers, prior season",
                       "Season used for the league and tier columns.", "", MECH, "master.arrival_season master.season"))
    SPECS += [
        c(f"{side}_league_name_at_move", N, "tiers", "core", "all", "blank", "blank = not verified", "derived (joined)",
          _TIER_SRC, f"league of the {who} in tier_reference_season from the season-specific lookup (snapshot league "
          "games; openfootball / football-data.co.uk league files matched by name within country)",
          "Blank: club not verified in any league table of that season (see the status column).", VAL, "league tiers",
          f"League of the {who} in the season of the move.", "Unmatched clubs are blank, never assumed lower-tier.",
          EXT, TIERS + f" master.{club} master.tier_reference_season", source=_TIER_SRC, code="LT"),
        c(f"{side}_league_tier_at_move", N, "tiers", "core", "all", "blank", "blank = not verified", "derived (joined)",
          _TIER_SRC, "tier of that league in its country's pyramid that season (1 = top division)",
          "Blank: not verified.", VAL, "league tiers", f"Division level of the {who} (1 = first division).",
          "Season-specific: a promoted or relegated club changes tier.", EXT,
          TIERS + f" master.{club} master.tier_reference_season", source=_TIER_SRC, code="LT"),
        c(f"{side}_league_country_at_move", N, "tiers", "helper", "all", "blank", "", "derived (joined)", _TIER_SRC,
          "country (league system) of that league", "Blank: not verified.", VAL, "league tiers",
          "Country of the league system.", "", EXT, TIERS + f" master.{club}", source=_TIER_SRC, code="LT"),
        c(f"{side}_league_tier_source", N, "tiers", "qa", "all", "blank", "", "derived (joined)", _TIER_SRC,
          "source of the assignment (transfermarkt_snapshot_games / openfootball / football-data.co.uk)",
          "Blank: not verified.", CAT, "league tiers QA", "Where the league and tier come from.", "", EXT,
          TIERS + f" master.{club}", source=_TIER_SRC, code="LT"),
        c(f"{side}_league_tier_status", N, "tiers", "qa", "all", "blank", "", "derived", _TIER_SRC,
          "verified / placeholder_not_a_club / youth_or_reserve_side_division_unknown / "
          "unknown_season_outside_sources / unknown_division_not_in_sources", "Never missing.", CAT,
          "league tiers QA", "Coverage status of the tier assignment.",
          "Only 'verified' carries a tier: outside a covered top flight is not evidence of tier 2.", CLASS,
          TIERS + f" master.{club} master.tier_reference_season", code="LT"),
        c(f"{side}_in_covered_top_flight_at_move", N, "games", "filter", "all", "blank", "", "derived (joined)",
          "games (14 covered leagues)", f"the {who} played a league game of one of the 14 covered top divisions in "
          "tier_reference_season", "Blank only if no reference season.", FLAG, "league tiers",
          f"{who.capitalize()} was in one of our 14 covered leagues that season.",
          "False is not a tier: the club may be in another country's top flight, a lower division or a youth league.",
          JOIN, GAMES),
    ]
    if side == "from":
        SPECS += [
            c("from_league_name_prior_season", N, "tiers", "helper", "all", "blank", "blank = not verified",
              "derived (joined)", _TIER_SRC, "league of the origin club in tier_reference_season - 1",
              "Blank: not verified.", VAL, "league tiers", "Origin club's league in the season before the move.",
              "For a summer move this is the season the player has just played.", EXT,
              TIERS + " master.from_club_id master.tier_reference_season", source=_TIER_SRC, code="LT"),
            c("from_league_tier_prior_season", N, "tiers", "helper", "all", "blank", "blank = not verified",
              "derived (joined)", _TIER_SRC, "tier of that league", "Blank: not verified.", VAL, "league tiers",
              "Origin club's tier in the season before the move.", "", EXT,
              TIERS + " master.from_club_id master.tier_reference_season", source=_TIER_SRC, code="LT"),
        ]

PRI = ("window 1 July p - 30 June p+1 (p = tier_reference_season - 1), cut at the day before the move; league games of "
       "every snapshot-league club the player was registered with (Stage 1C rows, internal registrations included), "
       "while registered")
SPECS += [
    c("prior_season", O, "master", "helper", "candidates", "blank", "", "derived", "tier_reference_season",
      "tier_reference_season - 1", NA_CAND, VAL, "prior season", "The season before the move (start year).", "", MECH,
      "master.tier_reference_season"),
    c("prior_league_games", O, "games", "core", "prior_reliable", "blank", "", "derived (joined)", "games, Stage 1C",
      f"number of league games in the {PRI}", "Blank unless prior_reliable.", VAL, "prior share",
      "League games the player could have played in the prior season.", "", CLASS, GAMES + " " + NEXT),
    c("prior_appearances", O, "games", "core", "prior_reliable", "blank", "", "derived (joined)", "appearances",
      "league appearances in those games for the registered club", "Blank unless prior_reliable.", VAL, "prior share",
      "Prior-season league appearances.", "", JOIN, APPS),
    c("prior_minutes", O, "games", "core", "prior_reliable", "blank", "", "derived (joined)", "appearances",
      "sum of minutes_played over those appearances", "Blank unless prior_reliable.", VAL, "prior share",
      "Prior-season league minutes.", "", JOIN, APPS + " duckdb.appearances.minutes_played"),
    c("prior_minutes_share", O, "games", "core", "prior_reliable", "blank", "", "derived",
      "prior_minutes, prior_league_games", "prior_minutes / (90 × prior_league_games), aggregated over all spells "
      "(not an average of percentages)", "Blank unless prior_reliable: missing coverage is never written as zero.",
      VAL, "prior share (control)", "Share of available league minutes played in the prior season.",
      "Covers only the covered part of the season: see prior_season_coverage.", MECH,
      "master.prior_minutes master.prior_league_games"),
    c("prior_injury_unavailable_games", O, "kaggle", "core", "prior_reliable", "blank", "", "derived",
      "Kaggle injuries, prior games", "prior games inside a rule-v2 medical or quarantine spell in which the player did "
      "not appear (once per game)", "Blank unless prior_reliable.", VAL, "prior fit share",
      "Prior-season games missed through injury, illness or quarantine.",
      "Only interpretable where prior_injury_data_quality_flag = usable.", CLASS, KINJ),
    c("prior_fit_games", O, "kaggle", "core", "prior_reliable", "blank", "", "derived",
      "prior_league_games, prior_injury_unavailable_games", "prior_league_games - prior_injury_unavailable_games",
      "Blank unless prior_reliable.", VAL, "prior fit share", "Prior-season games the player was available for.",
      "Only interpretable where prior_injury_data_quality_flag = usable.", MECH,
      "master.prior_league_games master.prior_injury_unavailable_games"),
    c("prior_fit_adjusted_share", O, "kaggle", "core", "prior_inj_usable", "blank", "", "derived",
      "prior_minutes, prior_fit_games", "prior_minutes / (90 × prior_fit_games) where the injury history is usable",
      "Blank unless the prior injury history is usable and at least one fit game is left.", VAL,
      "prior fit share (control)", "Prior-season share of the games the player was available for.",
      "Missing injury coverage leaves this blank but keeps prior_minutes_share.", MECH,
      "master.prior_minutes master.prior_fit_games"),
    c("prior_injury_data_quality_flag", O, "kaggle", "qa", "prior_reliable", "blank", "", "derived",
      "Kaggle injuries, profiles", "injury_adjusted.usability with rule v2, window = first to last prior game",
      "Blank unless prior_reliable.", CAT, "prior fit share", "Why the prior injury adjustment is or is not usable.",
      "", CLASS, KINJ),
    c("prior_reliable", O, "sample", "filter", "candidates", "blank", "", "derived", "prior_exclusion_reason",
      "prior_exclusion_reason = 'none (reliable)'", NA_CAND, FILT, "prior share",
      "Prior-season share computable and reliable.", "Not a sample restriction of the main analyses.", SFLAG,
      "master.prior_exclusion_reason"),
    c("prior_exclusion_reason", O, "games", "qa", "candidates", "blank", "", "derived", "spells, games, appearances",
      "first failing of: no_reference_season, prior_season_outside_snapshot, no_registration_history, "
      "no_covered_club, no_league_games_before_move, incomplete_club_games (a game with fewer than 11 appearance "
      "rows), appearances_for_unregistered_club, contradicted_minutes; else 'none (reliable)'", NA_CAND, CAT,
      "prior share QA", "Why the prior share is missing.", "", CLASS, GAMES + " " + APPS + " " + NEXT),
    c("prior_season_coverage", O, "games", "qa", "candidates", "blank", "", "derived",
      "prior_covered_share_of_days", "not_reliable / whole_season_at_covered_clubs (every registered day of the "
      "window at snapshot-league clubs) / part_of_season_at_covered_clubs", NA_CAND, CAT, "prior share QA",
      "Whether the prior share covers the whole season.", "", CLASS, "master.prior_covered_share_of_days"),
    c("prior_covered_share_of_days", O, "games", "qa", "candidates", "blank", "blank when no registered club days",
      "derived", "registration spells", "days registered at snapshot-league clubs / days registered at any club in "
      "the window", "Blank when the player has no registered club day in the window.", VAL, "prior share QA",
      "Fraction of the prior season spent at covered clubs.", "", MECH, NEXT + " " + GAMES),
    c("prior_clubs", O, "multi", "qa", "candidates", "blank", "", "derived", "registration spells",
      "distinct clubs (not placeholders) registered in the window", NA_CAND, VAL, "prior share QA",
      "Clubs in the prior season.", "", MECH, NEXT),
    c("prior_covered_clubs", O, "games", "qa", "candidates", "blank", "", "derived", "registration spells, games",
      "of those, clubs in a snapshot league that season", NA_CAND, VAL, "prior share QA",
      "Covered clubs in the prior season.", "", MECH, NEXT + " " + GAMES),
    c("prior_leagues", O, "games", "helper", "prior_reliable", "blank", "", "derived", "games",
      "competition ids of the prior games, '|'-joined", "Blank unless prior_reliable.", VAL, "prior share QA",
      "Leagues behind the prior share.", "", MECH, GAMES),
    c("prior_unregistered_club_apps", O, "games", "qa", "candidates", "blank", "", "derived", "appearances, spells",
      "league games in the window in which the player appears for a club he was not registered with",
      NA_CAND, VAL, "prior share QA", "Appearances that contradict the registration history.",
      "Positive values make the prior share unreliable (a move is missing).", CLASS, APPS + " " + NEXT),
    # ---------------- P. origin-club squad value (Fernandes)
    c("from_squad_value", P, "val", "core", "typed", "blank", "blank when fewer than 11 valued players", "derived",
      "player_valuations", "playing_time_regression.squad_value applied to the origin club: sum of the latest "
      "valuation strictly before the move (within 365 days) of every player whose valuation lists club A, excluding "
      "the moving player; blank if fewer than 11 such players", "Blank: fewer than 11 valued players (or unknown "
      "type).", VAL, "origin club size", "Origin club's squad market value just before the move, without the player.",
      "Same rule as squad_value (receiving club), which also excludes the moving player.", MECH, PV),
    c("from_squad_value_incl_player", P, "val", "helper", "typed", "blank", "", "derived",
      "from_squad_value, player valuation", "from_squad_value + the player's own latest pre-move valuation when it lists "
      "club A (defined exactly when from_squad_value is)", "Blank when from_squad_value is.", VAL, "origin club size",
      "Origin squad value including the departing player.", "", MECH, PV),
    c("from_squad_players", P, "val", "qa", "typed", "blank", "", "derived", "player_valuations",
      "number of valued players behind from_squad_value (moving player excluded)", "Blank when from_squad_value is.",
      VAL, "minimum 11 rule", "Valued players in the origin squad value.", "", MECH, PV),
    c("from_focal_in_squad", P, "val", "qa", "typed", "blank", "", "derived", "player_valuations",
      "the player's latest pre-move valuation (within 365 days) lists club A", "Blank for unknown type.", FLAG,
      "origin club size", "The player himself was in the origin squad's valuations.", "", MECH, PV),
    c("from_squad_value_gap_days", P, "val", "qa", "typed", "blank", "", "derived", "player_valuations",
      "move date - most recent valuation date used", "Blank when from_squad_value is.", VAL, "valuation timing",
      "Age of the newest valuation in the origin squad value.", "", MECH, PV),
    c("log_from_squad_value", P, "val", "core", "typed", "blank", "", "derived", "from_squad_value",
      "ln(from_squad_value) when positive", "Blank when from_squad_value is blank or not positive.", VAL,
      "origin club size", "Log origin squad value.", "", MECH, "master.from_squad_value"),
    c("to_squad_value", P, "val", "helper", "typed", "blank", "", "derived", "player_valuations",
      "the stored squad_value rule applied to every typed episode (equal to squad_value wherever that is filled)",
      "Blank: fewer than 11 valued players.", VAL, "origin / receiving ratio",
      "Receiving club's squad value for every episode.", "squad_value itself stays restricted to the revised sample.",
      MECH, PV),
    c("from_to_squad_value_ratio", P, "val", "core", "typed", "blank", "", "derived",
      "from_squad_value, to_squad_value", "from_squad_value / to_squad_value", "Blank when either is blank.", VAL,
      "origin vs receiving size", "Origin squad value relative to the receiving squad (>1: moving to a smaller squad).",
      "", MECH, "master.from_squad_value master.to_squad_value"),
    c("log_from_to_squad_value_ratio", P, "val", "core", "typed", "blank", "", "derived", "from_to_squad_value_ratio",
      "ln(from_to_squad_value_ratio)", "Blank when the ratio is blank.", VAL, "origin vs receiving size",
      "Log of that ratio.", "", MECH, "master.from_to_squad_value_ratio"),
    # ---------------- Q. combined validation (correction, 8 Nov 2026)
    c("pt_fully_validated", Qv, "sample", "filter", "all", "blank", "False outside reliable episodes", "derived",
      "pt_reliable_v2, pt_minutes_validated", "pt_reliable_v2 and pt_minutes_validated", "Never missing.", FILT,
      "fully validated outcome", "Reliable outcome passing both minute checks: no line-up contradiction "
      "(pt_lineup_contradicted) and no contradicted recorded minutes (pt_minutes_validation_status).",
      "The two checks catch different problems (games played without an appearance row vs recorded minutes that "
      "disagree with line-ups and events); no episode fails both.", SFLAG,
      "master.pt_reliable_v2 master.pt_minutes_validated"),
    c("in_age_model_sample_validated", Qv, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_age_model_sample_v2, pt_fully_validated", "in_age_model_sample_v2 and pt_fully_validated", "Never missing.",
      FILT, "no-calendar main regression (fully validated)", "Main regression sample without the calendar gate, both "
      "minute checks passed.", "", SFLAG, "master.in_age_model_sample_v2 master.pt_fully_validated"),
    c("in_cf_main_model_sample_validated", Qv, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_cf_main_model_sample_v2, pt_fully_validated", "in_cf_main_model_sample_v2 and pt_fully_validated",
      "Never missing.", FILT, "calendar-first main regression (fully validated)",
      "Calendar-first main regression sample, both minute checks passed.", "", SFLAG,
      "master.in_cf_main_model_sample_v2 master.pt_fully_validated"),
    c("in_injury_model_sample_validated", Qv, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_injury_model_sample_v2, pt_fully_validated", "in_injury_model_sample_v2 and pt_fully_validated",
      "Never missing.", FILT, "injury models v2 (fully validated)", "Injury regression sample v2 (quarantine counted), "
      "both minute checks passed, no calendar gate.", "", SFLAG,
      "master.in_injury_model_sample_v2 master.pt_fully_validated"),
    c("in_cf_injury_model_sample_validated", Qv, "sample", "filter", "all", "blank", "", "derived (master helper)",
      "in_cf_injury_model_sample_v2, pt_fully_validated", "in_cf_injury_model_sample_v2 and pt_fully_validated",
      "Never missing.", FILT, "calendar-first injury regression v2 (fully validated)",
      "Calendar-first injury regression sample v2, both minute checks passed.", "", SFLAG,
      "master.in_cf_injury_model_sample_v2 master.pt_fully_validated"),
]

NEW_COLUMNS = [x.variable for x in SPECS]
assert len(NEW_COLUMNS) == len(set(NEW_COLUMNS))

NEW_DOMAINS = {
    "candidates": ("loan and permanent episodes (in_playing_time_candidates = True)",
                   lambda M: M.in_playing_time_candidates.astype(bool)),
    "permanent": ("permanent arrivals (kind = permanent)", lambda M: M.kind.eq("permanent")),
    "usable_v2": ("injury data usable under rule v2 (in_injury_usable_v2 = True)",
                  lambda M: M.in_injury_usable_v2.astype(bool)),
    "prior_reliable": ("reliable prior-season share (prior_reliable = True)", lambda M: M.prior_reliable.eq(True).fillna(False).astype(bool)),
    "prior_inj_usable": ("prior-season injury history usable (prior_injury_data_quality_flag = usable)",
                         lambda M: M.prior_injury_data_quality_flag.eq("usable").fillna(False).astype(bool)),
}
NEW_ORIGINS = {"tiers": "joined: season-specific league tables (snapshot league games, openfootball, "
                        "football-data.co.uk), matched to our club ids"}
NEW_USEFULNESS = {"robustness": "robustness / sensitivity variant"}
CORE_GROUP = ("November 2026 additions", ["days_at_receiving_club", "next_departure_type",
                                          "perm_moved_on_before_first_league_game", "arrival_season_presence",
                                          "fit_adjusted_share_v2", "in_cf_injury_model_sample_v2",
                                          "minutes_share_validated", "in_cf_main_model_sample_v2",
                                          "from_league_name_at_move", "from_league_tier_at_move",
                                          "to_league_name_at_move", "to_league_tier_at_move", "prior_minutes_share",
                                          "prior_fit_adjusted_share", "from_squad_value", "from_to_squad_value_ratio"])
