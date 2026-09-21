"""Stage 1B: Transfermarkt page/label enrichment feasibility.

Stage 1 established that the dcaribou DuckDB loses loan fees and every semantic
transfer label. Stage 1B establishes where that information still exists and
whether we can recover it.

The answer, found by reading the upstream project rather than by scraping: the
labels survive in the upstream project's own RAW acquisition files, which are
published on the same public R2 bucket we already download the DuckDB from. The
loss happens in a single `else 0` branch of the upstream dbt model. We can
recover everything by re-parsing raw data we are already entitled to download,
with no requests to transfermarkt.com at all.

This package does NOT replace the Stage 1 backbone. It produces an enrichment
layer keyed on the same event_id.
"""
