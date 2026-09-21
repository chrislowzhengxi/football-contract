# Football Contracts Research Pipeline

Structured-data backbone for research on contractual incentives in European football transfers. The deterministic stage uses the public [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) project and its published DuckDB snapshot. A separate one-event contract-research stage adds source-backed findings without changing deterministic Transfermarkt values.

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
src/contract_schemas.py  Research result, evidence, and review schemas
src/research_contract.py One-event provider interface and CLI
prompts/contract_research.md  Web-research extraction instructions
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

## Contract research pilot

The research stage reads exactly one event from `structured_transfers.csv`, sends the event identity and transfer details to a web-search-capable provider, validates the structured response, and writes an independent JSON artifact under `data/outputs/contract_research/`. It never overwrites deterministic fields.

The live adapter uses the OpenAI Responses API with its `web_search_preview` tool. No additional Python SDK is required, but live research requires an API key:

```bash
export OPENAI_API_KEY="..."
python -m src.research_contract --event-id tm_32efe6f99f407ed52bf3
```

Use `OPENAI_MODEL` or `--model` to select a model. The provider interface is isolated in `ResearchProvider`, so another search/model service can be added without changing the schema or CLI orchestration. Offline provider responses can be validated with:

```bash
python -m src.research_contract \
	--event-id tm_32efe6f99f407ed52bf3 \
	--fixture path/to/provider-result.json
```

The output contains one object per researched field, plus source records. Each field includes `status`, `confidence`, and `evidence_ids`; evidence includes URL, title, publisher, source type, publication/retrieval dates, excerpt, and language.

Research status is deliberately conservative:

- `disclosed_yes`: a reliable source establishes the term exists.
- `disclosed_no`: a reliable source explicitly establishes the term does not exist.
- `partially_disclosed`: the term is known but an amount, percentage, or condition is unknown.
- `undisclosed`: a source explicitly says the amount or terms were undisclosed.
- `not_found`: the search did not locate reliable evidence; this does not mean the term was absent.
- `conflicting_sources`: credible sources disagree.
- `not_applicable`: the field does not apply to the deal.

Validation automatically sets `review_required` for conflicting sources, purchase obligations, obligation triggers, low-confidence findings, and other explicitly supplied review reasons. It also rejects unknown statuses, out-of-range confidence, broken evidence references, and false values paired with `not_found` or `partially_disclosed`.

## Output fields

`structured_transfers.csv` currently populates event ID, player identity, transfer parties/date/season, age, `reported_transfer_fee`, and nearest market value. It also includes nullable placeholders for transfer type, add-ons, sell-on, buy-back, purchase option/obligation, trigger, release clause, historical contract expiry, source evidence, and contract summary. Missing values mean “not yet researched”, never “no clause”.

`playing_time_outcomes.csv` is intentionally long-form. Each event can have multiple rows identified by `window_name` and `minutes_club_id`/`minutes_club_name`, with `window_start`, `window_end`, `match_date_min`, `match_date_max`, `minutes_played`, `appearance_count`, and `missing_appearance_data`. This avoids collapsing remainder-season and following-season outcomes into one ambiguous measure.

## Validation

The pipeline checks duplicate event IDs, impossible ages, negative minutes, invalid windows, duplicate appearance IDs, and suspiciously absent appearance data. Warnings are retained in `validation_report.json`; hard structural failures stop the command.

## Data-quality notes

Transfer fee is numeric but does not reliably distinguish permanent, free, or loan terms. The current player contract expiry is not a historical expiry at the transfer date, so it is intentionally not used. Appearance coverage can be incomplete, especially for competitions or dates outside the dataset; the output flags this rather than treating zero rows as zero minutes.

## Stage 1 rebuild (`src/stage1/`)

The pipeline is being rebuilt in independently inspectable stages. Stage 1 is the
Transfermarkt backbone and nothing else: raw snapshot → normalized transfer events.
It performs no web search, no retrieval, and no LLM inference.

```bash
python -m src.stage1.backbone                         # all 175,165 events
python -m src.stage1.backbone --club Porto --club Benfica --season 24/25 --season 25/26 \
       --name stage1_normalized_transfers_porto_benfica
python -m src.stage1.profile                          # profile report
python -m src.stage1.audit_sample                     # 20-event audit sample
```

Everything is written under `data/outputs/rebuild/` and nothing under `data/raw/` or
`data/processed/` is modified.

Column prefixes carry the lineage: `raw_*` is a verbatim Transfermarkt value,
`joined_*` comes from another raw table by key, `derived_*` was computed here.
`derived_movement_basis` says whether a movement class rests on an explicit
Transfermarkt marker, a bare restatement of the fee, or our own heuristic.

Read these before trusting any Stage 1 value:

| Artifact | What it answers |
| --- | --- |
| `stage1_transfermarkt_data_lineage.md` | Where every field comes from and what the snapshot cannot provide |
| `stage1_transfer_type_rules.md` | What the data does and does not let us say about loans, free transfers and fees |
| `stage1_parser_audit.csv` | Field-by-field audit of the older `src/enrich_structured.py` parser |
| `stage1_structured_profile.md` | Counts, missingness and anomalies |
| `stage1_validation_20.md` / `.csv` | 20 stress-test events explained end to end for manual verification |

The legacy deterministic path (`src/select_transfers.py`, `src/enrich_structured.py`)
is left untouched so earlier outputs stay reproducible. Its known defects are recorded
in `stage1_parser_audit.csv`; the most consequential are a many-to-many market-value
merge that duplicates 236 rows, a look-ahead market-value join, and `transfer_type`
being unconditionally set to `None` because it is listed in `schemas.CONTRACT_FIELDS`.

## Stage 1B — Transfermarkt label enrichment (`src/stage1b/`)

Stage 1 found that the DuckDB loses loan fees and every semantic transfer label.
Stage 1B found *where the loss happens*: a single `else 0` branch in the upstream
project's `dbt/models/base/transfermarkt_api/base_transfers.sql`. The labels
survive intact in the upstream project's own **raw** acquisition files, which are
published on the same public R2 bucket we already download the DuckDB from.

```bash
python -m src.stage1b.build          # parse + compare on the Stage 1 audit players
python -m src.stage1b.label_census   # label census across the whole backbone
```

Measured: **100% of the 175,165 backbone events and all 23,379 players are
recoverable from three published raw files, with zero requests to
transfermarkt.com.** This recovers 2,053 loan fees, 16,642 explicitly
undisclosed (`?`) fees, and a transfer-type label for every event.

The page layer enriches the backbone on `event_id`; it does not replace it.
Permanent fees, clubs, dates and market values agree 100% with the DuckDB, so
the enrichment adds nothing there and should not overwrite them.

See `stage1b_access_and_scale.md` for the provenance chain, robots.txt findings
and scale arithmetic.

## Stage 1C — the canonical backbone (`src/stage1c/`) — FROZEN

**Frozen 2026-09-21. Downstream stages read this and nothing else.**
175,165 events, 42 columns, 145 tests, 21 named regression checks.

```bash
python -m src.stage1c.build        # data/outputs/rebuild/stage1c_canonical_transfers.csv
python -m src.stage1c.regression   # 20 named checks against hand-verified cases
```

Stage 1C joins the Stage 1 DuckDB backbone to the raw Transfermarkt labels
recovered in Stage 1B. 175,165 events, 100% enrichment coverage, zero requests
to transfermarkt.com.

Four rules the layer enforces, each covered by tests:

1. A raw Transfermarkt label always beats a Stage 1 heuristic (87.2% of events
   take their type from a label; the heuristic now decides **none**).
2. A loan fee and a permanent fee never share a column — `loan_fee_eur`
   recovers €2.08bn across 2,050 events that the DuckDB records as `0`.
3. `?` (undisclosed) and `-` (no fee shown) never collapse into each other, and
   neither becomes zero.
4. Nothing is deleted. Loan returns stay in the table, flagged, so event chains
   stay reconstructable — including the 43 return legs that carry a fee.

5. A name is never an identity. 267 names are shared by more than one player
   (six different footballers are called "Vitinho"), so every key, join,
   grouping and event chain uses `player_id`.

`is_research_target` (76,701 events, 43.8%) defines the Stage 2 population, and
every excluded row carries a `research_exclusion_reason`.

Read `stage1c_canonical_data_dictionary.md` for the full column reference, the
exhaustive fee rules, and the known limitations.
