"""
Four ablation arms used to compare KG-augmented LLM compliance reasoning
against weaker grounding modes.

  Arm A — vanilla LLM. Transaction + entity attributes only, no rule
          definitions, no KG-derived relations. Tests the LLM's parametric
          financial-compliance knowledge.
  Arm B — rule-text grounding. Base + the natural-language rule
          definitions. The LLM must deduce which rule applies.
  Arm C — per-rule relations stated. Base + rule definitions + the KG's
          "Tx X violates Rule_Y / is compliant with Rule_Y" relations.
          The current default; tests synthesis from explicit relations.
  Arm D — full compliance flag. Base + rule definitions + KG relations +
          the boolean ground-truth label. Upper-bound sanity check.

The four arms share the same "base" context (transaction row, client
attrs, account attrs, counterparty attrs) so that what they differ on
is exactly the grounding mode under test — not the surface features.

`build_facts(ctx)` returns the `context_facts` string passed to
`FinancialLLM.ask_compliance_json(user_message, context_facts)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Per-tx context payload
# --------------------------------------------------------------------------- #


@dataclass
class TxContext:
    """All facts about a single transaction needed to build any arm's prompt."""
    tx_id: str

    # Transaction attributes
    amount: float
    currency: str
    amount_usd: float
    date: str
    status: str
    tx_type: str
    description: str

    # Counterparty attributes
    counterparty_id: str
    counterparty_name: str
    counterparty_country: str
    counterparty_on_sanctions_list: bool

    # Client attributes
    client_id: str
    client_name: str
    client_risk_level: str
    client_kyc_status: str
    client_kyc_expiry_date: str
    client_country: str
    client_pep_flag: bool

    # Account attributes
    account_id: str
    account_type: str
    account_default_currency: str
    account_open_date: str
    account_last_active_date: str

    # Grounding inputs (only consumed by arms that use them)
    rule_definitions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    kg_relations: List[Dict[str, str]] = field(default_factory=list)
    ground_truth_is_compliant: Optional[bool] = None

    # Bookkeeping (recorded in the run log, NOT in the LLM prompt)
    scenario_rule_id: Optional[str] = None


def _base_facts(ctx: TxContext) -> str:
    """The shared transaction + entity description used by every arm."""
    return (
        f"TRANSACTION {ctx.tx_id}\n"
        f"  amount         : {ctx.amount:.2f} {ctx.currency} "
        f"({ctx.amount_usd:.2f} USD-equivalent at fixed FX)\n"
        f"  date           : {ctx.date}\n"
        f"  type           : {ctx.tx_type}\n"
        f"  status         : {ctx.status}\n"
        f"  description    : {ctx.description}\n"
        f"\n"
        f"COUNTERPARTY {ctx.counterparty_id}\n"
        f"  name           : {ctx.counterparty_name}\n"
        f"  country        : {ctx.counterparty_country}\n"
        f"\n"
        f"CLIENT {ctx.client_id}\n"
        f"  name           : {ctx.client_name}\n"
        f"  risk level     : {ctx.client_risk_level}\n"
        f"  KYC status     : {ctx.client_kyc_status} "
        f"(expiry {ctx.client_kyc_expiry_date})\n"
        f"  country        : {ctx.client_country}\n"
        f"  PEP flag       : {'yes' if ctx.client_pep_flag else 'no'}\n"
        f"\n"
        f"ACCOUNT {ctx.account_id}\n"
        f"  type             : {ctx.account_type}\n"
        f"  default currency : {ctx.account_default_currency}\n"
        f"  open date        : {ctx.account_open_date}\n"
        f"  last active date : {ctx.account_last_active_date}\n"
    )


def _rules_block(rule_definitions: Dict[str, Dict[str, str]]) -> str:
    """Render the formal rule definitions as a single block of bullet text."""
    if not rule_definitions:
        return ""
    lines = ["COMPLIANCE RULES (apply these definitions strictly):"]
    for rid, info in rule_definitions.items():
        desc = info.get("description", "").strip()
        cat = info.get("category", "")
        sev = info.get("severity", "")
        lines.append(f"  - {rid} [{cat}, severity={sev}]: {desc}")
    return "\n".join(lines) + "\n"


def _kg_relations_block(relations: List[Dict[str, str]]) -> str:
    """
    Render the per-tx rule firings from the KG.

    The phrasing matters: an earlier benchmark (finding F-001 in
    THESIS_FINDINGS.md) showed that the LLM lexically hijacked loaded
    rule names (e.g. "is compliant with rule KYC_EXPIRED" got read as
    "KYC is expired"). The current phrasing makes the predicate
    capitalized and unambiguous about whether the rule fired.
    """
    if not relations:
        return "KG RELATIONS: no explicit relations recorded for this transaction.\n"
    lines = ["KG RELATIONS (verified ground-truth-adjacent facts):"]
    for r in relations:
        rid = r.get("rule_id", "?")
        rel = r.get("relation", "?")
        if rel == "compliantWith" or rel == "compliant":
            lines.append(f"  - the {rid} rule does NOT fire on this transaction")
        elif rel == "violatesRule" or rel == "violates":
            lines.append(f"  - the {rid} rule FIRES on this transaction")
        else:
            lines.append(f"  - related to rule {rid} via {rel}")
    return "\n".join(lines) + "\n"


def _ground_truth_block(gt: Optional[bool]) -> str:
    if gt is None:
        return "GROUND-TRUTH FLAG: unknown\n"
    return (
        f"GROUND-TRUTH FLAG: this transaction is_compliant = "
        f"{'true' if gt else 'false'}\n"
    )


# --------------------------------------------------------------------------- #
# Arm definitions
# --------------------------------------------------------------------------- #


class Arm:
    """Base class. Subclasses set `name` and override `build_facts`."""
    name: str = ""
    short_description: str = ""

    def build_facts(self, ctx: TxContext) -> str:
        raise NotImplementedError


class ArmA_Vanilla(Arm):
    name = "A"
    short_description = "vanilla LLM, transaction + entities only, no rules, no KG"

    def build_facts(self, ctx: TxContext) -> str:
        return _base_facts(ctx)


class ArmB_RuleText(Arm):
    name = "B"
    short_description = "rule definitions in plain language + transaction"

    def build_facts(self, ctx: TxContext) -> str:
        return (
            _base_facts(ctx)
            + "\n"
            + _rules_block(ctx.rule_definitions)
        )


class ArmC_KGRelations(Arm):
    name = "C"
    short_description = (
        "rule definitions + KG-stated per-rule relations for this tx"
    )

    def build_facts(self, ctx: TxContext) -> str:
        return (
            _base_facts(ctx)
            + "\n"
            + _rules_block(ctx.rule_definitions)
            + "\n"
            + _kg_relations_block(ctx.kg_relations)
        )


class ArmD_GroundTruth(Arm):
    name = "D"
    short_description = "rules + KG relations + ground-truth flag (upper bound)"

    def build_facts(self, ctx: TxContext) -> str:
        return (
            _base_facts(ctx)
            + "\n"
            + _rules_block(ctx.rule_definitions)
            + "\n"
            + _kg_relations_block(ctx.kg_relations)
            + "\n"
            + _ground_truth_block(ctx.ground_truth_is_compliant)
        )


ARMS: Dict[str, Arm] = {
    "A": ArmA_Vanilla(),
    "B": ArmB_RuleText(),
    "C": ArmC_KGRelations(),
    "D": ArmD_GroundTruth(),
}


def get_arms(names: Optional[List[str]] = None) -> List[Arm]:
    """Return arm instances by name; preserves declaration order if names=None."""
    if not names:
        return [ARMS[k] for k in ("A", "B", "C", "D")]
    out: List[Arm] = []
    for n in names:
        key = n.strip().upper()
        if key not in ARMS:
            raise ValueError(f"Unknown arm: {n!r}. Valid: {list(ARMS)}")
        out.append(ARMS[key])
    return out
