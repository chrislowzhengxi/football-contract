"""Stage 2 economic-mechanism vocabulary.

Extends what Stage 2 can record WITHOUT touching Stage 1C. The diagnostic found
two economically well-evidenced outcomes that the original eight-field
vocabulary could not hold, so both were scored as research failures:

  * Morata - AC Milan paid Galatasaray ~EUR 5m to end a loan early. Public,
    specific, direction established. No field existed for it.
  * Sorloth - Trabzonspor received EUR 10m against a EUR 20m onward sale to
    Leipzig. A share of a third party's payment, not a transfer fee.

Every mechanism carries provenance. A value without an evidence span, a source
and an event scope is not admissible, because the diagnostic's false positives
all came from values that looked fine until you asked which transaction they
described.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STATUS_VALUES = ("disclosed_yes", "disclosed_no", "partially_disclosed", "undisclosed",
                 "not_found", "conflicting_sources", "not_applicable")


@dataclass
class MechanismField:
    name: str
    description: str
    value_keys: tuple = ()
    requires_direction: bool = False
    requires_quote: bool = False
    attaches_to_roles: tuple = ()       # event roles this term can sensibly attach to
    stage: str = "stage2"


# Provenance required on EVERY populated mechanism.
PROVENANCE_KEYS = (
    "status",                # STATUS_VALUES
    "evidence_ids",          # non-empty unless not_found / not_applicable
    "evidence_span",         # exact supporting text
    "source_url",
    "source_class",
    "statement_type",        # direct | retrospective
    "which_transaction",     # plain-English scope of the evidence
    "scoped_to_event_id",    # the family member the term attaches to
    "confidence",
)

# --- carried forward from the original Stage 2 vocabulary ---
CARRIED_FORWARD = [
    MechanismField("purchase_option", "Right, not duty, to buy the player",
                   ("price", "currency", "exercised", "deadline"), False, True,
                   ("original_loan",)),
    MechanismField("purchase_obligation", "Duty to buy once a condition is met",
                   ("price", "currency", "triggered"), False, True, ("original_loan",)),
    MechanismField("obligation_trigger", "The condition that converts option to duty",
                   ("metric", "threshold", "unit", "additional_condition"), False, True,
                   ("original_loan", "loan_return")),
    MechanismField("sell_on", "Share of a future onward sale retained by a former club",
                   ("percentage", "basis", "cap"), True, True,
                   ("permanent_transfer", "loan_return", "third_party_sale")),
    MechanismField("buy_back", "Right of a selling club to repurchase",
                   ("price", "currency", "window", "exercised"), False, True,
                   ("permanent_transfer", "loan_return")),
    MechanismField("add_ons", "Contingent payments on top of a base fee",
                   ("amount", "currency", "conditions"), True, False,
                   ("permanent_transfer", "original_loan", "third_party_sale")),
    MechanismField("parent_contract_expiry", "Expiry of the player's contract with the parent club",
                   ("date",), False, False,
                   ("original_loan", "loan_return", "permanent_transfer")),
    MechanismField("release_or_purchase_clause", "Fixed price at which the club must let the player go",
                   ("price", "currency"), False, True, ("permanent_transfer",)),
]

# --- new, from the diagnostic ---
NEW_MECHANISMS = [
    MechanismField(
        "termination_compensation",
        "Payment to end a loan or contract before its scheduled expiry. Distinct "
        "from a transfer fee and from a loan fee.",
        ("amount", "currency"), True, True,
        ("early_termination", "loan_return")),
    MechanismField(
        "third_party_sale_share",
        "A club's entitlement to part of the consideration paid by a THIRD club. "
        "The money originates outside the two clubs named on the row, which is why "
        "it must never be written to a bilateral fee column.",
        ("percentage", "amount", "currency", "basis"), True, True,
        ("administrative_return", "third_party_sale", "loan_return")),
    MechanismField(
        "economic_mechanism",
        "The single best-supported description of what the money on this leg IS. "
        "Set only when evidence names it; otherwise not_found.",
        ("value",), False, True,
        ("loan_return", "administrative_return", "early_termination",
         "permanent_transfer", "third_party_sale")),
]

# Companion scalar columns the caller asked for, flattened for CSV output.
FLAT_COLUMNS = (
    "termination_compensation", "termination_compensation_payer",
    "termination_compensation_recipient",
    "third_party_sale_share", "third_party_sale_share_percentage",
    "third_party_sale_share_amount", "third_party_sale_buyer", "third_party_sale_seller",
    "economic_mechanism", "economic_mechanism_status",
)

# Controlled vocabulary for `economic_mechanism`. Never inferred from sequence.
ECONOMIC_MECHANISM_VALUES = (
    "transfer_fee",                  # price of a permanent acquisition
    "loan_fee",                      # price of the temporary use of a player
    "option_exercise",               # a purchase option was taken up
    "obligation_settlement",         # a purchase obligation was triggered and paid
    "buy_back_exercise",             # a retained repurchase right was taken up
    "net_of_offsetting_options",     # two option exercises settled against each other
    "termination_compensation",      # payment to end an arrangement early
    "third_party_sale_share",        # a share of another club's payment
    "sell_on_settlement",            # payment under a previously agreed sell-on
    "unknown",
)

ALL_MECHANISMS = {m.name: m for m in CARRIED_FORWARD + NEW_MECHANISMS}


def admissible(field_name: str, payload: dict, event_role: str) -> tuple[bool, list[str]]:
    """Would this populated mechanism survive Stage 2's evidence rules?

    Checks provenance and event-role fit. Does NOT judge truth - only whether the
    claim is scoped and supported well enough to be recorded.
    """
    spec = ALL_MECHANISMS.get(field_name)
    problems: list[str] = []
    if spec is None:
        return False, [f"unknown mechanism '{field_name}'"]
    status = payload.get("status")
    if status not in STATUS_VALUES:
        problems.append(f"status '{status}' not in controlled vocabulary")
    if status in ("not_found", "not_applicable"):
        return not problems, problems
    if not payload.get("evidence_ids"):
        problems.append("no evidence_ids")
    if not payload.get("evidence_span"):
        problems.append("no evidence_span")
    if not payload.get("which_transaction"):
        problems.append("evidence scope (which_transaction) not declared")
    if spec.requires_quote and not (payload.get("quote") or payload.get("evidence_span")):
        problems.append("mechanism requires an explicit quoted term")
    if spec.requires_direction and not payload.get("direction_established"):
        problems.append("amount carried without an established payer/recipient")
    if spec.attaches_to_roles and event_role not in spec.attaches_to_roles:
        problems.append(
            f"'{field_name}' does not sensibly attach to a leg with role "
            f"'{event_role}' (expected one of {list(spec.attaches_to_roles)})")
    return not problems, problems
