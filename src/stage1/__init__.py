"""Stage 1 of the rebuilt pipeline: the Transfermarkt backbone.

Stage 1 turns raw Transfermarkt records into normalized transfer events and
nothing else. It performs no web search, no retrieval, and no LLM inference.

Every output column is either a verbatim raw value (``raw_*``), a value joined
from another raw table (``joined_*``), or something this stage computed
(``derived_*``). The prefix is the lineage.
"""
