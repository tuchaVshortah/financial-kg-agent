"""
Deterministic synthetic dataset generator for the dissertation evaluation.

Produces a labelled multi-rule transaction dataset that exercises the eight
compliance scenarios described in `docs/dataset_generation.md`.

Reproducibility: every run with the same `--seed` produces byte-identical CSVs
and metadata. No network or LLM calls are made.

Usage:
    python -m src.generate_data --n 500 --seed 42 --out data/

Outputs (in `--out`):
    clients.csv             extended schema (kyc_*, country, pep_flag)
    accounts.csv            extended schema (open_date, last_active_date, default_currency)
    transactions.csv        extended schema (counterparty_id, tx_type, amount_usd, ...)
    rules.csv               eight rules + their categories
    tx_rules.csv            per-transaction rule relations
    counterparties.csv      NEW: counterparty registry incl. sanctions flag
    generation_metadata.json self-documenting record of the run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from faker import Faker

# --------------------------------------------------------------------------- #
# Constants & rule definitions
# --------------------------------------------------------------------------- #

GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 42
DEFAULT_N = 500
DEFAULT_N_CLIENTS = 50
DEFAULT_START_DATE = "2024-06-01"
DEFAULT_END_DATE = "2024-12-31"

# Synthetic sanctions list: 10 counterparty IDs that are always non-compliant
# when a transaction touches them. Embedded directly here for reproducibility
# rather than loaded from an external feed.
SANCTIONED_COUNTERPARTY_IDS = [
    f"CP_SAN_{i:02d}" for i in range(1, 11)
]

# Synthetic high-risk jurisdiction list (FATF "gray list"-style placeholders).
HIGH_RISK_COUNTRIES = ["AF", "MM", "KP", "IR", "SY", "VE"]

# Pool of "normal" counterparty countries. Includes a few that overlap nothing
# special so the LLM can't trivially key on country.
NORMAL_COUNTRIES = [
    "US", "GB", "DE", "FR", "JP", "KZ", "RU", "TR", "AE", "SG",
    "CA", "AU", "NL", "ES", "IT", "PL", "BR",
]

# Currencies + fixed FX rates -> USD. Kept synthetic & fixed so AML threshold
# logic is mechanical and reproducible.
FX_RATES_TO_USD: Dict[str, Decimal] = {
    "USD": Decimal("1.00"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "KZT": Decimal("0.0022"),
}

TX_TYPES = ["wire", "transfer", "cash_deposit", "cash_withdrawal", "card_payment"]

# Formal rule definitions. Used by the generator AND surfaced in
# generation_metadata.json so the eval pipeline (arm B) can load
# the human-readable text directly.
RULE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "AML_THRESHOLD": {
        "description": (
            "Any single transaction at or above 10,000 USD-equivalent must be "
            "reported to the relevant financial-intelligence unit."
        ),
        "category": "AML",
        "severity": "high",
    },
    "STRUCTURING": {
        "description": (
            "Three or more transactions in the 9,000-9,999 USD-equivalent band "
            "from the same client within a 5-day window suggest deliberate "
            "structuring to evade the AML reporting threshold."
        ),
        "category": "AML",
        "severity": "high",
    },
    "SANCTIONS": {
        "description": (
            "Transactions whose counterparty appears on the sanctions list are "
            "prohibited regardless of amount or jurisdiction."
        ),
        "category": "Sanctions",
        "severity": "high",
    },
    "KYC_VALIDITY": {
        "description": (
            "Every transaction requires the client to hold a currently-valid "
            "Know-Your-Customer record. A transaction violates this rule if "
            "the client's KYC is missing, marked expired, or has an expiry "
            "date earlier than the transaction date."
        ),
        "category": "KYC",
        "severity": "high",
    },
    "HIGH_RISK_JURISDICTION": {
        "description": (
            "Transactions involving counterparties registered in high-risk "
            "jurisdictions require enhanced due diligence; absence of EDD "
            "marks the transaction non-compliant."
        ),
        "category": "AML",
        "severity": "medium",
    },
    "VELOCITY": {
        "description": (
            "More than 10 transactions in any rolling 24-hour window for a "
            "medium-risk client (or more than 5 for a high-risk client) "
            "constitutes suspicious velocity."
        ),
        "category": "Behavioral",
        "severity": "medium",
    },
    "DORMANT_ACCOUNT_RULE": {
        "description": (
            "An account that has been inactive for at least 180 days and then "
            "transacts at 5,000 USD-equivalent or above triggers a dormant-"
            "account-reactivation alert."
        ),
        "category": "Behavioral",
        "severity": "medium",
    },
    "ROUND_NUMBER_ANOMALY": {
        "description": (
            "Clusters of round-amount transactions (multiples of 1,000 USD) "
            "for small-business clients indicate deliberate rounding intended "
            "to obscure activity patterns."
        ),
        "category": "Behavioral",
        "severity": "low",
    },
}

# Rules that can be evaluated against a single transaction in isolation,
# without needing cross-tx context. The cross-rule labeling pass walks every
# generated tx and applies all of these so the dataset reflects real-world
# AML labeling (any rule fires -> tx is non-compliant), not just the
# rule-of-the-scenario.
PER_TX_RULES = (
    "AML_THRESHOLD",
    "SANCTIONS",
    "KYC_VALIDITY",
    "HIGH_RISK_JURISDICTION",
    "DORMANT_ACCOUNT_RULE",
)

# Scenario distribution at default --n 500.
# Each entry: (rule_id, total_tx, non_compliant_tx).
# rule_id == "NORMAL" denotes the baseline negative class (no rule triggers).
SCENARIO_DISTRIBUTION: List[Tuple[str, int, int]] = [
    ("AML_THRESHOLD", 80, 40),
    ("STRUCTURING", 60, 30),
    ("SANCTIONS", 50, 30),
    ("KYC_VALIDITY", 60, 30),
    ("HIGH_RISK_JURISDICTION", 50, 25),
    ("VELOCITY", 50, 25),
    ("DORMANT_ACCOUNT_RULE", 40, 20),
    ("ROUND_NUMBER_ANOMALY", 30, 15),
    ("NORMAL", 80, 0),
]
assert sum(t for _, t, _ in SCENARIO_DISTRIBUTION) == DEFAULT_N


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class GenClient:
    client_id: str
    name: str
    risk_level: str            # low | medium | high
    kyc_status: str            # valid | expired | missing
    kyc_expiry_date: str       # ISO date
    country: str               # ISO 3166-1 alpha-2
    pep_flag: bool
    is_small_business: bool    # used only by ROUND_NUMBER_ANOMALY scenario


@dataclass
class GenAccount:
    account_id: str
    client_id: str
    account_type: str          # checking | savings | business
    status: str                # active | closed
    open_date: str
    last_active_date: str
    default_currency: str


@dataclass
class GenCounterparty:
    counterparty_id: str
    name: str
    country: str
    on_sanctions_list: bool


@dataclass
class GenTransaction:
    tx_id: str
    account_id: str
    amount: Decimal
    currency: str
    amount_usd: Decimal
    date: str
    status: str                # completed | pending
    counterparty_id: str
    counterparty_country: str
    tx_type: str
    description: str
    is_compliant: bool
    # rule_id -> "violates" | "compliant".
    # Scenario functions populate the entry for their own scenario rule;
    # cross_rule_pass() additionally merges all PER_TX_RULES evaluations.
    rule_relations: Dict[str, str] = field(default_factory=dict)

    @property
    def rule_ids(self) -> List[str]:
        """Sorted list of all rules in rule_relations (preserves CSV column)."""
        return sorted(self.rule_relations.keys())

    @property
    def firing_rules(self) -> List[str]:
        return sorted(
            r for r, rel in self.rule_relations.items() if rel == "violates"
        )

    def recompute_is_compliant(self) -> None:
        self.is_compliant = not any(
            rel == "violates" for rel in self.rule_relations.values()
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _money(d: Decimal) -> Decimal:
    """Quantise a money amount to two decimal places."""
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _to_usd(amount: Decimal, currency: str) -> Decimal:
    """Convert local-currency amount to USD using the fixed FX table."""
    rate = FX_RATES_TO_USD[currency]
    return _money(amount * rate)


def _from_usd(usd_amount: Decimal, currency: str) -> Decimal:
    """Convert a target USD-equivalent amount into the chosen currency."""
    rate = FX_RATES_TO_USD[currency]
    return _money(usd_amount / rate)


def _date_in_range(rng: random.Random, start: date, end: date) -> date:
    """Pick a uniformly-random date in [start, end]."""
    delta_days = (end - start).days
    if delta_days <= 0:
        return start
    return start + timedelta(days=rng.randint(0, delta_days))


def _datestr(d: date) -> str:
    return d.isoformat()


# --------------------------------------------------------------------------- #
# Population generation
# --------------------------------------------------------------------------- #


def make_clients(rng: random.Random, faker: Faker, n_clients: int) -> List[GenClient]:
    """Generate the client pool with a 30/15/5 low/medium/high risk split."""
    n_low = int(round(n_clients * 0.6))
    n_med = int(round(n_clients * 0.3))
    n_high = n_clients - n_low - n_med

    risk_levels = (
        ["low"] * n_low + ["medium"] * n_med + ["high"] * n_high
    )
    rng.shuffle(risk_levels)

    clients: List[GenClient] = []
    for i, risk in enumerate(risk_levels, start=1):
        # KYC status weighted toward "valid" but with a non-trivial expired tail
        kyc_status = rng.choices(
            ["valid", "expired", "missing"],
            weights=[0.80, 0.15, 0.05],
            k=1,
        )[0]
        if kyc_status == "valid":
            # KYC expires somewhere in the next 1-3 years
            expiry = date(2025, 1, 1) + timedelta(days=rng.randint(180, 1095))
        elif kyc_status == "expired":
            # Expired sometime in the past 6 months -> 2 years
            expiry = date(2024, 1, 1) - timedelta(days=rng.randint(0, 720))
        else:  # missing
            # Use 1970-01-01 as a sentinel; loaders treat it as missing
            expiry = date(1970, 1, 1)

        clients.append(
            GenClient(
                client_id=f"C{i:03d}",
                name=faker.company() if rng.random() < 0.3 else faker.name(),
                risk_level=risk,
                kyc_status=kyc_status,
                kyc_expiry_date=_datestr(expiry),
                country=rng.choice(NORMAL_COUNTRIES + HIGH_RISK_COUNTRIES),
                pep_flag=rng.random() < 0.05,
                # ~20% of clients are small-business — pool used for round-number scenario
                is_small_business=rng.random() < 0.20,
            )
        )
    return clients


def make_accounts(
    rng: random.Random,
    clients: Sequence[GenClient],
    start_date: date,
) -> List[GenAccount]:
    """1-3 accounts per client, with realistic open/last-active dates."""
    accounts: List[GenAccount] = []
    counter = 1
    for client in clients:
        n_acc = rng.randint(1, 3)
        for j in range(n_acc):
            # Most accounts opened well before the simulation window
            open_d = start_date - timedelta(days=rng.randint(180, 2000))
            # last_active_date will be tightened later if scenario needs dormancy
            last_active = open_d + timedelta(
                days=rng.randint(30, max(31, (start_date - open_d).days))
            )
            accounts.append(
                GenAccount(
                    account_id=f"A{counter:04d}",
                    client_id=client.client_id,
                    account_type=rng.choices(
                        ["checking", "savings", "business"],
                        weights=[0.55, 0.25, 0.20],
                        k=1,
                    )[0],
                    status="active",
                    open_date=_datestr(open_d),
                    last_active_date=_datestr(last_active),
                    default_currency=rng.choices(
                        list(FX_RATES_TO_USD.keys()),
                        weights=[0.55, 0.20, 0.15, 0.10],  # USD/EUR/GBP/KZT
                        k=1,
                    )[0],
                )
            )
            counter += 1
    return accounts


def make_counterparties(
    rng: random.Random, faker: Faker, n_normal: int = 90
) -> List[GenCounterparty]:
    """
    Build the counterparty pool: 10 sanctioned + n_normal non-sanctioned.

    The 10 sanctioned IDs are deterministic (CP_SAN_01..CP_SAN_10) so that
    cross-references with the embedded sanctions list are stable.
    """
    cps: List[GenCounterparty] = []
    for i, sid in enumerate(SANCTIONED_COUNTERPARTY_IDS, start=1):
        cps.append(
            GenCounterparty(
                counterparty_id=sid,
                name=f"Sanctioned Entity {i}",
                country=rng.choice(HIGH_RISK_COUNTRIES + ["RU", "BY"]),
                on_sanctions_list=True,
            )
        )
    for i in range(1, n_normal + 1):
        country = rng.choices(
            NORMAL_COUNTRIES + HIGH_RISK_COUNTRIES,
            weights=[1.0] * len(NORMAL_COUNTRIES) + [0.30] * len(HIGH_RISK_COUNTRIES),
            k=1,
        )[0]
        cps.append(
            GenCounterparty(
                counterparty_id=f"CP_{i:03d}",
                name=faker.company(),
                country=country,
                on_sanctions_list=False,
            )
        )
    return cps


# --------------------------------------------------------------------------- #
# Scenario generators
# --------------------------------------------------------------------------- #
#
# Each scenario function takes:
#   rng, faker, accounts, counterparties, clients, start_date, end_date,
#   total, n_noncompliant, tx_id_seq
# and returns a list of GenTransaction.
#
# tx_id_seq is a callable returning the next "T0001" id.


TxIdSeq = "callable"  # for type-hint readability only


def _pick_account(rng: random.Random, accounts: Sequence[GenAccount]) -> GenAccount:
    return rng.choice(list(accounts))


def _pick_normal_counterparty(
    rng: random.Random, counterparties: Sequence[GenCounterparty]
) -> GenCounterparty:
    pool = [c for c in counterparties if not c.on_sanctions_list and c.country in NORMAL_COUNTRIES]
    return rng.choice(pool)


def _pick_sanctioned_counterparty(
    rng: random.Random, counterparties: Sequence[GenCounterparty]
) -> GenCounterparty:
    pool = [c for c in counterparties if c.on_sanctions_list]
    return rng.choice(pool)


def _pick_high_risk_counterparty(
    rng: random.Random, counterparties: Sequence[GenCounterparty]
) -> GenCounterparty:
    pool = [
        c for c in counterparties
        if not c.on_sanctions_list and c.country in HIGH_RISK_COUNTRIES
    ]
    return rng.choice(pool)


def _client_for_account(
    accounts_index: Dict[str, GenAccount],
    clients_index: Dict[str, GenClient],
    account_id: str,
) -> GenClient:
    return clients_index[accounts_index[account_id].client_id]


def gen_aml_threshold(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    AML_THRESHOLD: amount >= $10,000 USD-equivalent must be reported.
    Non-compliant: amount well above threshold and the transaction is *not*
    flagged ("status: completed" with no follow-up). Compliant: amount under
    threshold (so the rule does not fire) — we record the rule as
    "compliant with AML_THRESHOLD" because the threshold check passed.
    """
    out: List[GenTransaction] = []
    for i in range(total):
        is_violation = i < n_noncompliant
        # Pick an account whose client has VALID kyc to isolate the rule signal
        valid_clients = [c for c in clients if c.kyc_status == "valid"]
        valid_accounts = [a for a in accounts if accounts_index[a.account_id].client_id in {vc.client_id for vc in valid_clients}]
        # Defensive fallback (should not trigger with default population sizes)
        acct = rng.choice(valid_accounts) if valid_accounts else _pick_account(rng, accounts)
        cp = _pick_normal_counterparty(rng, counterparties)
        currency = acct.default_currency

        if is_violation:
            usd = Decimal(str(rng.uniform(10_000, 75_000))).quantize(Decimal("0.01"))
        else:
            usd = Decimal(str(rng.uniform(100, 9_500))).quantize(Decimal("0.01"))

        amount = _from_usd(usd, currency)
        out.append(
            GenTransaction(
                tx_id=tx_id_seq(),
                account_id=acct.account_id,
                amount=amount,
                currency=currency,
                amount_usd=usd,
                date=_datestr(_date_in_range(rng, start_date, end_date)),
                status="completed",
                counterparty_id=cp.counterparty_id,
                counterparty_country=cp.country,
                tx_type=rng.choice(TX_TYPES),
                description=f"AML-scenario tx (target USD={usd})",
                is_compliant=not is_violation,
                rule_relations={
                    "AML_THRESHOLD": "violates" if is_violation else "compliant"
                },
            )
        )
    return out


def gen_structuring(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    STRUCTURING: clusters of >=3 tx in the $9,000-$9,999 band from the same
    client within 5 days. Non-compliant clusters trigger the rule on each
    member tx; compliant tx in this scenario are isolated $9,000-$9,999 tx
    with no nearby same-client neighbours.
    """
    out: List[GenTransaction] = []
    # Build clusters of 3-5 tx each, one cluster per ~3.5 violation tx
    n_clusters = max(1, n_noncompliant // 3)
    cluster_sizes = []
    remaining = n_noncompliant
    for _ in range(n_clusters):
        sz = min(rng.randint(3, 5), remaining)
        if sz <= 0:
            break
        cluster_sizes.append(sz)
        remaining -= sz
    if remaining > 0 and cluster_sizes:
        cluster_sizes[-1] += remaining

    for sz in cluster_sizes:
        acct = _pick_account(rng, accounts)
        cp = _pick_normal_counterparty(rng, counterparties)
        cluster_start = _date_in_range(rng, start_date, end_date - timedelta(days=5))
        for k in range(sz):
            usd = Decimal(str(rng.uniform(9_000, 9_999))).quantize(Decimal("0.01"))
            amount = _from_usd(usd, acct.default_currency)
            out.append(
                GenTransaction(
                    tx_id=tx_id_seq(),
                    account_id=acct.account_id,
                    amount=amount,
                    currency=acct.default_currency,
                    amount_usd=usd,
                    date=_datestr(cluster_start + timedelta(days=rng.randint(0, 5))),
                    status="completed",
                    counterparty_id=cp.counterparty_id,
                    counterparty_country=cp.country,
                    tx_type=rng.choice(TX_TYPES),
                    description=f"Structuring cluster member ({k+1}/{sz})",
                    is_compliant=False,
                    rule_relations={"STRUCTURING": "violates"},
                )
            )

    # Compliant: lone $9,000-$9,999 tx
    n_compliant = total - n_noncompliant
    for _ in range(n_compliant):
        acct = _pick_account(rng, accounts)
        cp = _pick_normal_counterparty(rng, counterparties)
        usd = Decimal(str(rng.uniform(9_000, 9_999))).quantize(Decimal("0.01"))
        amount = _from_usd(usd, acct.default_currency)
        out.append(
            GenTransaction(
                tx_id=tx_id_seq(),
                account_id=acct.account_id,
                amount=amount,
                currency=acct.default_currency,
                amount_usd=usd,
                date=_datestr(_date_in_range(rng, start_date, end_date)),
                status="completed",
                counterparty_id=cp.counterparty_id,
                counterparty_country=cp.country,
                tx_type=rng.choice(TX_TYPES),
                description="Isolated near-threshold tx (not structuring)",
                is_compliant=True,
                rule_relations={"STRUCTURING": "compliant"},
            )
        )
    return out


def gen_sanctions(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    SANCTIONS: counterparty appears on the embedded sanctions list.
    Non-compliant tx use a sanctioned counterparty; compliant "distractor" tx
    use a normal counterparty whose name superficially resembles a sanctioned
    one (handled by random selection of normal CPs in the pool).
    """
    out: List[GenTransaction] = []
    for i in range(total):
        is_violation = i < n_noncompliant
        acct = _pick_account(rng, accounts)
        cp = (_pick_sanctioned_counterparty(rng, counterparties)
              if is_violation else _pick_normal_counterparty(rng, counterparties))
        usd = Decimal(str(rng.uniform(500, 25_000))).quantize(Decimal("0.01"))
        amount = _from_usd(usd, acct.default_currency)
        out.append(
            GenTransaction(
                tx_id=tx_id_seq(),
                account_id=acct.account_id,
                amount=amount,
                currency=acct.default_currency,
                amount_usd=usd,
                date=_datestr(_date_in_range(rng, start_date, end_date)),
                status="completed",
                counterparty_id=cp.counterparty_id,
                counterparty_country=cp.country,
                tx_type=rng.choice(TX_TYPES),
                description=("Sanctioned-CP tx" if is_violation
                             else "Normal-CP tx in sanctions scenario"),
                is_compliant=not is_violation,
                rule_relations={
                    "SANCTIONS": "violates" if is_violation else "compliant"
                },
            )
        )
    return out


def gen_kyc_expired(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    KYC_EXPIRED: tx_date > client.kyc_expiry_date.
    Non-compliant tx are routed through clients with kyc_status in
    {expired, missing}; compliant tx through clients with valid KYC and
    a tx_date safely before expiry.
    """
    expired_clients = [c for c in clients if c.kyc_status in ("expired", "missing")]
    valid_clients = [c for c in clients if c.kyc_status == "valid"]
    expired_accts = [a for a in accounts
                     if accounts_index[a.account_id].client_id in {c.client_id for c in expired_clients}]
    valid_accts = [a for a in accounts
                   if accounts_index[a.account_id].client_id in {c.client_id for c in valid_clients}]

    out: List[GenTransaction] = []
    for i in range(total):
        is_violation = i < n_noncompliant
        if is_violation and expired_accts:
            acct = rng.choice(expired_accts)
        elif valid_accts:
            acct = rng.choice(valid_accts)
        else:
            acct = _pick_account(rng, accounts)
        cp = _pick_normal_counterparty(rng, counterparties)
        usd = Decimal(str(rng.uniform(100, 9_500))).quantize(Decimal("0.01"))
        amount = _from_usd(usd, acct.default_currency)
        out.append(
            GenTransaction(
                tx_id=tx_id_seq(),
                account_id=acct.account_id,
                amount=amount,
                currency=acct.default_currency,
                amount_usd=usd,
                date=_datestr(_date_in_range(rng, start_date, end_date)),
                status="completed",
                counterparty_id=cp.counterparty_id,
                counterparty_country=cp.country,
                tx_type=rng.choice(TX_TYPES),
                description=("Tx after KYC expiry" if is_violation
                             else "Tx within KYC validity window"),
                is_compliant=not is_violation,
                rule_relations={
                    "KYC_VALIDITY": "violates" if is_violation else "compliant"
                },
            )
        )
    return out


def gen_high_risk_jurisdiction(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    HIGH_RISK_JURISDICTION: counterparty.country in HIGH_RISK_COUNTRIES.
    Non-compliant: high-risk CP, no enhanced due diligence (modeled as
    description == "no EDD applied"). Compliant: same high-risk CP but
    description marks EDD applied; OR a normal-jurisdiction CP.
    """
    out: List[GenTransaction] = []
    for i in range(total):
        is_violation = i < n_noncompliant
        acct = _pick_account(rng, accounts)
        if is_violation:
            cp = _pick_high_risk_counterparty(rng, counterparties)
            desc = "no EDD applied"
        else:
            # Half compliant cases use high-risk CP w/ EDD, half normal CP
            if rng.random() < 0.5:
                cp = _pick_high_risk_counterparty(rng, counterparties)
                desc = "EDD applied: enhanced verification complete"
            else:
                cp = _pick_normal_counterparty(rng, counterparties)
                desc = "Normal-jurisdiction tx"
        usd = Decimal(str(rng.uniform(500, 20_000))).quantize(Decimal("0.01"))
        amount = _from_usd(usd, acct.default_currency)
        out.append(
            GenTransaction(
                tx_id=tx_id_seq(),
                account_id=acct.account_id,
                amount=amount,
                currency=acct.default_currency,
                amount_usd=usd,
                date=_datestr(_date_in_range(rng, start_date, end_date)),
                status="completed",
                counterparty_id=cp.counterparty_id,
                counterparty_country=cp.country,
                tx_type=rng.choice(TX_TYPES),
                description=desc,
                is_compliant=not is_violation,
                rule_relations={
                    "HIGH_RISK_JURISDICTION":
                        "violates" if is_violation else "compliant"
                },
            )
        )
    return out


def gen_velocity(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    VELOCITY: bursts of tx within 24h. Non-compliant bursts: >10 tx for
    medium-risk client or >5 for high-risk client, all on the same day.
    Compliant in this scenario: a small burst (3-4 tx) on the same day —
    well under threshold.
    """
    out: List[GenTransaction] = []
    n_burst = max(1, n_noncompliant // 11)  # avg 11 tx per non-compliant burst
    burst_sizes: List[Tuple[GenAccount, int]] = []
    remaining = n_noncompliant
    for _ in range(n_burst):
        acct = _pick_account(rng, accounts)
        client = _client_for_account(accounts_index, clients_index, acct.account_id)
        if client.risk_level == "high":
            sz = rng.randint(6, 10)
        else:
            sz = rng.randint(11, 18)
        sz = min(sz, remaining)
        if sz <= 0:
            break
        burst_sizes.append((acct, sz))
        remaining -= sz
    if remaining > 0 and burst_sizes:
        a, s = burst_sizes[-1]
        burst_sizes[-1] = (a, s + remaining)

    for acct, sz in burst_sizes:
        burst_day = _date_in_range(rng, start_date, end_date)
        for k in range(sz):
            cp = _pick_normal_counterparty(rng, counterparties)
            usd = Decimal(str(rng.uniform(50, 2_000))).quantize(Decimal("0.01"))
            amount = _from_usd(usd, acct.default_currency)
            out.append(
                GenTransaction(
                    tx_id=tx_id_seq(),
                    account_id=acct.account_id,
                    amount=amount,
                    currency=acct.default_currency,
                    amount_usd=usd,
                    date=_datestr(burst_day),
                    status="completed",
                    counterparty_id=cp.counterparty_id,
                    counterparty_country=cp.country,
                    tx_type=rng.choice(TX_TYPES),
                    description=f"Velocity burst tx ({k+1}/{sz})",
                    is_compliant=False,
                    rule_relations={"VELOCITY": "violates"},
                )
            )

    # Compliant: small same-day clusters
    n_compliant = total - n_noncompliant
    while n_compliant > 0:
        sz = min(rng.randint(2, 4), n_compliant)
        acct = _pick_account(rng, accounts)
        burst_day = _date_in_range(rng, start_date, end_date)
        for k in range(sz):
            cp = _pick_normal_counterparty(rng, counterparties)
            usd = Decimal(str(rng.uniform(50, 2_000))).quantize(Decimal("0.01"))
            amount = _from_usd(usd, acct.default_currency)
            out.append(
                GenTransaction(
                    tx_id=tx_id_seq(),
                    account_id=acct.account_id,
                    amount=amount,
                    currency=acct.default_currency,
                    amount_usd=usd,
                    date=_datestr(burst_day),
                    status="completed",
                    counterparty_id=cp.counterparty_id,
                    counterparty_country=cp.country,
                    tx_type=rng.choice(TX_TYPES),
                    description=f"Compliant low-velocity cluster ({k+1}/{sz})",
                    is_compliant=True,
                    rule_relations={"VELOCITY": "compliant"},
                )
            )
        n_compliant -= sz
    return out


def gen_dormant_reactivation(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    DORMANT_REACTIVATION: account inactive >=180 days then a tx >= $5,000 USD.
    Non-compliant: rewrite chosen account's last_active_date to >=180 days
    before tx_date, with amount above the trigger. Compliant: short dormancy
    or below-trigger amount.
    """
    out: List[GenTransaction] = []
    for i in range(total):
        is_violation = i < n_noncompliant
        acct = _pick_account(rng, accounts)
        cp = _pick_normal_counterparty(rng, counterparties)
        tx_d = _date_in_range(rng, start_date, end_date)
        if is_violation:
            # Force dormancy >=180 days
            new_last_active = tx_d - timedelta(days=rng.randint(180, 720))
            if new_last_active.isoformat() < acct.last_active_date:
                acct.last_active_date = _datestr(new_last_active)
            usd = Decimal(str(rng.uniform(5_000, 30_000))).quantize(Decimal("0.01"))
        else:
            # Either short dormancy or low amount
            if rng.random() < 0.5:
                # short dormancy, any amount
                usd = Decimal(str(rng.uniform(500, 9_000))).quantize(Decimal("0.01"))
            else:
                # below-trigger amount even after long dormancy
                usd = Decimal(str(rng.uniform(100, 4_900))).quantize(Decimal("0.01"))
        amount = _from_usd(usd, acct.default_currency)
        out.append(
            GenTransaction(
                tx_id=tx_id_seq(),
                account_id=acct.account_id,
                amount=amount,
                currency=acct.default_currency,
                amount_usd=usd,
                date=_datestr(tx_d),
                status="completed",
                counterparty_id=cp.counterparty_id,
                counterparty_country=cp.country,
                tx_type=rng.choice(TX_TYPES),
                description=("Reactivation after long dormancy" if is_violation
                             else "Tx within normal activity window"),
                is_compliant=not is_violation,
                rule_relations={
                    "DORMANT_ACCOUNT_RULE":
                        "violates" if is_violation else "compliant"
                },
            )
        )
    return out


def gen_round_number_anomaly(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    ROUND_NUMBER_ANOMALY: small-business clients producing clusters of
    round-amount tx. Non-compliant: cluster of 3+ round-USD-thousand tx
    from same small-business account in 7 days. Compliant: isolated
    round-amount tx, or non-business clients.
    """
    sb_clients = [c for c in clients if c.is_small_business]
    sb_accts = [a for a in accounts
                if accounts_index[a.account_id].client_id in {c.client_id for c in sb_clients}]
    fallback_accts = sb_accts or accounts

    out: List[GenTransaction] = []
    n_clusters = max(1, n_noncompliant // 3)
    cluster_sizes: List[int] = []
    remaining = n_noncompliant
    for _ in range(n_clusters):
        sz = min(rng.randint(3, 5), remaining)
        if sz <= 0:
            break
        cluster_sizes.append(sz)
        remaining -= sz
    if remaining > 0 and cluster_sizes:
        cluster_sizes[-1] += remaining

    for sz in cluster_sizes:
        acct = rng.choice(fallback_accts)
        cluster_start = _date_in_range(rng, start_date, end_date - timedelta(days=7))
        for k in range(sz):
            usd_round = Decimal(rng.choice([1_000, 2_000, 3_000, 5_000, 10_000]))
            amount = _from_usd(usd_round, acct.default_currency)
            cp = _pick_normal_counterparty(rng, counterparties)
            out.append(
                GenTransaction(
                    tx_id=tx_id_seq(),
                    account_id=acct.account_id,
                    amount=amount,
                    currency=acct.default_currency,
                    amount_usd=usd_round,
                    date=_datestr(cluster_start + timedelta(days=rng.randint(0, 7))),
                    status="completed",
                    counterparty_id=cp.counterparty_id,
                    counterparty_country=cp.country,
                    tx_type=rng.choice(TX_TYPES),
                    description=f"Round-amount SB cluster ({k+1}/{sz})",
                    is_compliant=False,
                    rule_relations={"ROUND_NUMBER_ANOMALY": "violates"},
                )
            )

    n_compliant = total - n_noncompliant
    for _ in range(n_compliant):
        acct = _pick_account(rng, accounts)
        cp = _pick_normal_counterparty(rng, counterparties)
        usd_round = Decimal(rng.choice([1_000, 2_000, 5_000]))
        amount = _from_usd(usd_round, acct.default_currency)
        out.append(
            GenTransaction(
                tx_id=tx_id_seq(),
                account_id=acct.account_id,
                amount=amount,
                currency=acct.default_currency,
                amount_usd=usd_round,
                date=_datestr(_date_in_range(rng, start_date, end_date)),
                status="completed",
                counterparty_id=cp.counterparty_id,
                counterparty_country=cp.country,
                tx_type=rng.choice(TX_TYPES),
                description="Isolated round-amount tx (not SB cluster)",
                is_compliant=True,
                rule_relations={"ROUND_NUMBER_ANOMALY": "compliant"},
            )
        )
    return out


def gen_normal(
    rng, faker, accounts, counterparties, clients, accounts_index, clients_index,
    start_date, end_date, total, n_noncompliant, tx_id_seq,
) -> List[GenTransaction]:
    """
    Negative-class baseline: clearly compliant tx with no rule triggers.
    All amounts under $5,000 USD, normal CP, valid-KYC client, no clustering.
    `n_noncompliant` is always 0 by design.
    """
    assert n_noncompliant == 0, "NORMAL class must have n_noncompliant=0"
    out: List[GenTransaction] = []
    valid_clients = [c for c in clients if c.kyc_status == "valid"]
    valid_accts = [a for a in accounts
                   if accounts_index[a.account_id].client_id in {c.client_id for c in valid_clients}]
    pool = valid_accts or accounts
    for _ in range(total):
        acct = rng.choice(pool)
        cp = _pick_normal_counterparty(rng, counterparties)
        usd = Decimal(str(rng.uniform(20, 4_900))).quantize(Decimal("0.01"))
        amount = _from_usd(usd, acct.default_currency)
        out.append(
            GenTransaction(
                tx_id=tx_id_seq(),
                account_id=acct.account_id,
                amount=amount,
                currency=acct.default_currency,
                amount_usd=usd,
                date=_datestr(_date_in_range(rng, start_date, end_date)),
                status="completed",
                counterparty_id=cp.counterparty_id,
                counterparty_country=cp.country,
                tx_type=rng.choice(TX_TYPES),
                description="Normal compliant tx (negative class)",
                is_compliant=True,
                rule_relations={},
            )
        )
    return out


SCENARIO_FUNCS = {
    "AML_THRESHOLD": gen_aml_threshold,
    "STRUCTURING": gen_structuring,
    "SANCTIONS": gen_sanctions,
    "KYC_VALIDITY": gen_kyc_expired,
    "HIGH_RISK_JURISDICTION": gen_high_risk_jurisdiction,
    "VELOCITY": gen_velocity,
    "DORMANT_ACCOUNT_RULE": gen_dormant_reactivation,
    "ROUND_NUMBER_ANOMALY": gen_round_number_anomaly,
    "NORMAL": gen_normal,
}


# --------------------------------------------------------------------------- #
# Cross-rule labeling pass
# --------------------------------------------------------------------------- #


def _check_aml_threshold(tx: GenTransaction) -> bool:
    """Fires if amount_usd >= 10,000."""
    return tx.amount_usd >= Decimal("10000")


def _check_sanctions(tx: GenTransaction, cp: GenCounterparty) -> bool:
    """Fires if counterparty is on the embedded sanctions list."""
    return cp.on_sanctions_list


def _check_kyc_validity(tx: GenTransaction, client: GenClient) -> bool:
    """
    Fires if the client's KYC is missing, marked expired, OR the recorded
    expiry date is strictly earlier than the transaction date.
    """
    if client.kyc_status in ("expired", "missing"):
        return True
    if client.kyc_status == "valid":
        return client.kyc_expiry_date < tx.date
    return False


def _check_high_risk_jurisdiction(tx: GenTransaction) -> bool:
    """
    Fires if the counterparty country is in HIGH_RISK_COUNTRIES AND the
    transaction description does not contain an explicit 'EDD applied'
    marker (which signals enhanced due diligence was performed).
    """
    if tx.counterparty_country not in HIGH_RISK_COUNTRIES:
        return False
    desc = tx.description.lower()
    # "no edd applied" should NOT count as EDD applied — handle it explicitly
    return not ("edd applied" in desc and "no edd applied" not in desc)


def _check_dormant_account_rule(tx: GenTransaction, account: GenAccount) -> bool:
    """
    Fires if the account was inactive (last_active_date >= 180 days before
    the tx date) AND the transaction is >= 5,000 USD-equivalent.
    """
    last_active = date.fromisoformat(account.last_active_date)
    tx_d = date.fromisoformat(tx.date)
    dormancy_days = (tx_d - last_active).days
    return dormancy_days >= 180 and tx.amount_usd >= Decimal("5000")


def cross_rule_pass(
    transactions: List[GenTransaction],
    accounts_index: Dict[str, GenAccount],
    clients_index: Dict[str, GenClient],
    counterparties_index: Dict[str, GenCounterparty],
) -> Dict[str, Any]:
    """
    Walk every generated transaction and evaluate every PER_TX_RULE against
    it. For each rule, append the (compliant | violates) relation to
    `rule_relations`. Recompute `is_compliant` from the merged relations.

    Returns a small report dict capturing how the dataset moved:
      * label_flips : number of tx that flipped from compliant to non-compliant
                      because of a newly-detected violation
      * cross_rule_fire_counts : per per-tx rule, how many additional
                                 firings the pass discovered (i.e., rules
                                 firing on tx whose scenario was a different
                                 rule)
    """
    label_flips = 0
    cross_rule_fire_counts: Dict[str, int] = {r: 0 for r in PER_TX_RULES}

    for tx in transactions:
        acct = accounts_index[tx.account_id]
        client = clients_index[acct.client_id]
        cp = counterparties_index[tx.counterparty_id]

        was_compliant = tx.is_compliant
        # Evaluate every per-tx rule and merge results into rule_relations.
        # If a rule was already evaluated by the scenario function, the
        # scenario's relation is preserved (we don't double-write).
        for rule_id, fires in (
            ("AML_THRESHOLD", _check_aml_threshold(tx)),
            ("SANCTIONS", _check_sanctions(tx, cp)),
            ("KYC_VALIDITY", _check_kyc_validity(tx, client)),
            ("HIGH_RISK_JURISDICTION", _check_high_risk_jurisdiction(tx)),
            ("DORMANT_ACCOUNT_RULE", _check_dormant_account_rule(tx, acct)),
        ):
            if rule_id in tx.rule_relations:
                continue
            tx.rule_relations[rule_id] = "violates" if fires else "compliant"
            if fires:
                cross_rule_fire_counts[rule_id] += 1

        tx.recompute_is_compliant()
        if was_compliant and not tx.is_compliant:
            label_flips += 1

    return {
        "label_flips": label_flips,
        "cross_rule_fire_counts": cross_rule_fire_counts,
    }


# --------------------------------------------------------------------------- #
# Top-level generation
# --------------------------------------------------------------------------- #


def generate(
    n: int,
    seed: int,
    out_dir: Path,
    n_clients: int,
    start_date: date,
    end_date: date,
) -> Dict:
    """Generate the full dataset and write it to `out_dir`."""
    rng = random.Random(seed)
    faker = Faker()
    Faker.seed(seed)

    # If --n != 500, scale every scenario count proportionally and round.
    scale = n / DEFAULT_N
    distribution: List[Tuple[str, int, int]] = []
    cumulative = 0
    for i, (rule, total, nc) in enumerate(SCENARIO_DISTRIBUTION):
        if i == len(SCENARIO_DISTRIBUTION) - 1:
            # Last scenario absorbs any rounding drift to hit exactly `n`.
            scaled_total = n - cumulative
            scaled_nc = max(0, min(int(round(nc * scale)), scaled_total))
        else:
            scaled_total = int(round(total * scale))
            scaled_nc = int(round(nc * scale))
            cumulative += scaled_total
        distribution.append((rule, scaled_total, scaled_nc))

    # Generate population
    clients = make_clients(rng, faker, n_clients)
    accounts = make_accounts(rng, clients, start_date)
    counterparties = make_counterparties(rng, faker, n_normal=90)
    accounts_index = {a.account_id: a for a in accounts}
    clients_index = {c.client_id: c for c in clients}

    # tx_id sequencer
    counter = {"i": 0}

    def next_tx_id() -> str:
        counter["i"] += 1
        return f"T{counter['i']:04d}"

    transactions: List[GenTransaction] = []
    per_rule_counts = {}
    for rule, total, nc in distribution:
        fn = SCENARIO_FUNCS[rule]
        scenario_txs = fn(
            rng, faker, accounts, counterparties, clients,
            accounts_index, clients_index,
            start_date, end_date, total, nc, next_tx_id,
        )
        per_rule_counts[rule] = {
            "total": len(scenario_txs),
            "non_compliant": sum(1 for t in scenario_txs if not t.is_compliant),
            "compliant": sum(1 for t in scenario_txs if t.is_compliant),
        }
        transactions.extend(scenario_txs)

    # Pre-cross-rule snapshot for the methodology / findings record.
    pre_cross_compliant = sum(1 for t in transactions if t.is_compliant)
    pre_cross_non = len(transactions) - pre_cross_compliant

    # Cross-rule labeling pass: evaluate every PER_TX_RULE against every tx,
    # flip is_compliant to False if any rule fires that wasn't already
    # surfaced by the scenario. This is the F-002 fix.
    cross_rule_report = cross_rule_pass(
        transactions, accounts_index, clients_index,
        {c.counterparty_id: c for c in counterparties},
    )

    # Sort transactions by date for deterministic, scannable CSV ordering.
    transactions.sort(key=lambda t: (t.date, t.tx_id))

    # ----- Write CSVs ------------------------------------------------------- #
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_clients_csv(out_dir / "clients.csv", clients)
    _write_accounts_csv(out_dir / "accounts.csv", accounts)
    _write_counterparties_csv(out_dir / "counterparties.csv", counterparties)
    _write_transactions_csv(out_dir / "transactions.csv", transactions)
    _write_rules_csv(out_dir / "rules.csv")
    _write_tx_rules_csv(out_dir / "tx_rules.csv", transactions)

    # ----- Metadata --------------------------------------------------------- #
    total_compliant = sum(1 for t in transactions if t.is_compliant)
    total_noncompliant = len(transactions) - total_compliant
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_transactions": len(transactions),
        "compliant": total_compliant,
        "non_compliant": total_noncompliant,
        "compliant_pct": round(100.0 * total_compliant / len(transactions), 2),
        "n_clients": len(clients),
        "n_accounts": len(accounts),
        "n_counterparties": len(counterparties),
        "n_sanctioned_counterparties": len(SANCTIONED_COUNTERPARTY_IDS),
        "high_risk_countries": HIGH_RISK_COUNTRIES,
        "fx_rates_to_usd": {k: str(v) for k, v in FX_RATES_TO_USD.items()},
        "scenario_distribution": [
            {"rule": rule, "total": total, "non_compliant": nc}
            for rule, total, nc in distribution
        ],
        "per_rule_counts": per_rule_counts,
        "rule_definitions": RULE_DEFINITIONS,
        "cross_rule_labeling": {
            "enabled": True,
            "per_tx_rules": list(PER_TX_RULES),
            "pre_cross_compliant": pre_cross_compliant,
            "pre_cross_non_compliant": pre_cross_non,
            "label_flips_compliant_to_non": cross_rule_report["label_flips"],
            "additional_firings_by_rule":
                cross_rule_report["cross_rule_fire_counts"],
            "notes": (
                "After scenario-driven generation, each transaction is "
                "evaluated against every PER_TX_RULE. If any rule fires that "
                "the scenario didn't already mark, the firing is added to the "
                "tx's rule_relations and is_compliant is flipped to false. "
                "This makes the dataset labels reflect real-world AML logic "
                "(any rule violation -> overall non-compliant), addressing "
                "finding F-002 in THESIS_FINDINGS.md."
            ),
        },
    }
    (out_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    return metadata


# --------------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------------- #


def _write_clients_csv(path: Path, clients: Sequence[GenClient]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "client_id", "name", "risk_level",
            "kyc_status", "kyc_expiry_date",
            "country", "pep_flag",
        ])
        for c in clients:
            w.writerow([
                c.client_id, c.name, c.risk_level,
                c.kyc_status, c.kyc_expiry_date,
                c.country, "true" if c.pep_flag else "false",
            ])


def _write_accounts_csv(path: Path, accounts: Sequence[GenAccount]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "account_id", "client_id", "account_type", "status",
            "open_date", "last_active_date", "default_currency",
        ])
        for a in accounts:
            w.writerow([
                a.account_id, a.client_id, a.account_type, a.status,
                a.open_date, a.last_active_date, a.default_currency,
            ])


def _write_counterparties_csv(path: Path, cps: Sequence[GenCounterparty]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["counterparty_id", "name", "country", "on_sanctions_list"])
        for c in cps:
            w.writerow([
                c.counterparty_id, c.name, c.country,
                "true" if c.on_sanctions_list else "false",
            ])


def _write_transactions_csv(path: Path, txs: Sequence[GenTransaction]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tx_id", "account_id", "amount", "currency", "amount_usd",
            "date", "status", "is_compliant", "rule_ids",
            "counterparty_id", "counterparty_country", "tx_type", "description",
        ])
        for t in txs:
            w.writerow([
                t.tx_id, t.account_id,
                f"{t.amount:.2f}", t.currency, f"{t.amount_usd:.2f}",
                t.date, t.status,
                "true" if t.is_compliant else "false",
                ",".join(t.rule_ids),
                t.counterparty_id, t.counterparty_country, t.tx_type, t.description,
            ])


def _write_rules_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule_id", "description", "severity", "category"])
        for rule_id, info in RULE_DEFINITIONS.items():
            w.writerow([
                rule_id, info["description"], info["severity"], info["category"],
            ])


def _write_tx_rules_csv(path: Path, txs: Sequence[GenTransaction]) -> None:
    """
    Write one row per (tx, rule) pair the tx has been evaluated against,
    using the per-rule relation captured in `rule_relations`. After the
    cross-rule labeling pass each tx carries a relation for every
    PER_TX_RULE, so a non-compliant tx can carry mixed
    compliant/violates entries (e.g., AML didn't fire, KYC did).
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tx_id", "rule_id", "relation"])
        for t in txs:
            for rid in t.rule_ids:
                w.writerow([t.tx_id, rid, t.rule_relations[rid]])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic financial-tx dataset generator")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"Total number of transactions (default {DEFAULT_N})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"RNG seed for reproducibility (default {DEFAULT_SEED})")
    p.add_argument("--out", type=Path, default=Path("data"),
                   help="Output directory (default ./data)")
    p.add_argument("--n-clients", type=int, default=DEFAULT_N_CLIENTS,
                   help=f"Number of clients (default {DEFAULT_N_CLIENTS})")
    p.add_argument("--start-date", default=DEFAULT_START_DATE,
                   help=f"Simulation window start (default {DEFAULT_START_DATE})")
    p.add_argument("--end-date", default=DEFAULT_END_DATE,
                   help=f"Simulation window end (default {DEFAULT_END_DATE})")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    metadata = generate(
        n=args.n,
        seed=args.seed,
        out_dir=args.out,
        n_clients=args.n_clients,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
    )
    print("Wrote dataset to", args.out.resolve())
    print(f"  total transactions   : {metadata['total_transactions']}")
    print(f"  compliant            : {metadata['compliant']} "
          f"({metadata['compliant_pct']}%)")
    print(f"  non-compliant        : {metadata['non_compliant']}")
    print(f"  clients / accounts   : {metadata['n_clients']} / {metadata['n_accounts']}")
    print(f"  counterparties       : {metadata['n_counterparties']} "
          f"({metadata['n_sanctioned_counterparties']} sanctioned)")
    print(f"  metadata             : {args.out / 'generation_metadata.json'}")


if __name__ == "__main__":
    main()
