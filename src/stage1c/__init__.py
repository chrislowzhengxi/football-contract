"""Stage 1C: the canonical, frozen Transfermarkt backbone.

Stage 1  established a faithful but semantically thin event table.
Stage 1B established that the missing semantics survive in the upstream
         project's published raw files.
Stage 1C joins the two into the one dataset downstream research uses.

Rules of the layer:
  * a raw Transfermarkt label always beats a Stage 1 heuristic;
  * a loan fee and a permanent fee never share a column;
  * "?" (undisclosed) and "-" (no fee shown) never collapse into each other,
    and neither becomes zero;
  * nothing is deleted - loan returns stay in the table and are flagged.
"""
