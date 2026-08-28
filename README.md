# Football Contracts Research Pipeline

Structured-data backbone for research on contractual incentives in European football transfers. This first stage is deliberately limited to deterministic processing of the published [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) DuckDB snapshot. Web and LLM contract research will be added later without changing the structured event records.

## Source schema and joins

The source snapshot currently contains 13 tables. The pipeline introspects the database at runtime and uses these tables when available:

| Source table | Columns used | Role |
| --- | --- | --- |
| `transfers` | `player_id`, `transfer_date`, `transfer_season`, `from_club_id`, `to_club_id`, club names, `transfer_fee`, `market_value_in_eur`, `player_name` | One row per transfer event |
| `players` | `player_id`, `name`, `date_of_birth` | Stable player identity and age |
| `player_valuations` | `player_id`, `date`, `market_value_in_eur` | Nearest market value to the transfer date |
| `appearances` | `appearance_id`, `game_id`, `player_id`, `player_club_id`, `minutes_played` | Player minutes |
| `games` | `game_id`, `date`, `season` | Match date and season windows |
| `clubs` | `club_id`, `name` | Optional club reference |
| `competitions` | `competition_id`, `name` | Optional competition filter |

The pipeline does not infer a contractual clause from a missing value. Reserved research fields use nullable values and companion status fields such as `undisclosed`, `not_found`, or `conflicting_sources` for the later evidence layer.

## Architecture

```text
data/raw/          Downloaded DuckDB snapshots (ignored by git)
data/processed/    Intermediate data, if needed
data/outputs/      Canonical event and playing-time CSV/DuckDB outputs
src/load_data.py   Download, connect, and runtime schema inspection
src/select_transfers.py  Pilot selection and event construction CLI
src/enrich_structured.py Canonical enrichment orchestration
src/compute_minutes.py    Date-window playing-time calculations
src/schemas.py       Canonical fields and validation checks
tests/               Focused unit tests
```

Each transfer remains a separate event. `event_chain_id` links a nearby onward movement, such as B -> A followed immediately by A -> C, while preserving both events.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run a pilot

The command downloads the current snapshot if it is absent, inspects the source schema, selects matching transfers, enriches deterministic fields, computes two explicit playing-time windows, validates the result, and writes CSV plus DuckDB outputs.

```bash
python -m src.select_transfers --club "FC Porto" --season "2024/25" --limit 25
python -m src.select_transfers --club "FC Porto" --club "Benfica" --season "2024/25" --competition "Primeira Liga"
```

Useful options include `--db PATH`, `--source-url URL`, `--output-dir PATH`, and `--no-download`.

## Output fields

`structured_transfers.csv` currently populates event ID, player identity, transfer parties/date/season, age, `reported_transfer_fee`, and nearest market value. It also includes nullable placeholders for transfer type, add-ons, sell-on, buy-back, purchase option/obligation, trigger, release clause, historical contract expiry, source evidence, and contract summary. Missing values mean “not yet researched”, never “no clause”.

`playing_time_outcomes.csv` is intentionally long-form. Each event can have multiple rows identified by `window_name` and `minutes_club_id`/`minutes_club_name`, with `window_start`, `window_end`, `match_date_min`, `match_date_max`, `minutes_played`, `appearance_count`, and `missing_appearance_data`. This avoids collapsing remainder-season and following-season outcomes into one ambiguous measure.

## Validation

The pipeline checks duplicate event IDs, impossible ages, negative minutes, invalid windows, duplicate appearance IDs, and suspiciously absent appearance data. Warnings are retained in `validation_report.json`; hard structural failures stop the command.

## Data-quality notes

Transfer fee is numeric but does not reliably distinguish permanent, free, or loan terms. The current player contract expiry is not a historical expiry at the transfer date, so it is intentionally not used. Appearance coverage can be incomplete, especially for competitions or dates outside the dataset; the output flags this rather than treating zero rows as zero minutes.
