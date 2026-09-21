"""Stage 2: recover contractual terms that Stage 1C cannot determine.

Stage 1C is the deterministic backbone and is FROZEN. Stage 2 never rewrites a
Stage 1C fact; it produces a separate, evidence-backed observation layer keyed
on `event_id`, which the existing fusion architecture combines downstream.

Stage 2 researches ONLY what Transfermarkt cannot tell us:
purchase options, purchase obligations and their triggers, add-ons, sell-on
clauses, buy-back clauses, contract expiry at the transfer date, whether a
release/purchase clause was exercised, and what a fee on a loan-return leg
actually represents.

It does NOT re-research player identity, clubs, dates, transfer type, the
permanent fee, the loan fee, free-transfer status or market value. Those are
settled.
"""
