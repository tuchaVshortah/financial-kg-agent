# Synthetic dataset generation

This document describes how the labelled transaction dataset used in the
dissertation evaluation is produced, why it is generated this way, and what
its limitations are.

The generator lives at `src/generate_data.py`. A single command reproduces the
exact dataset used in the thesis figures:

```bash
python -m src.generate_data --n 500 --seed 42 --out data/
```

Every CSV is byte-identical across runs with the same seed.

## Why deterministic generation, not LLM generation

We considered three options for producing labelled transaction data:

1. **Anonymised real-world data** — out of scope for legal reasons; access to a
   bank's transaction stream was not available for the dissertation.
2. **LLM-generated synthetic data** — non-reproducible, expensive (each tx
   requires an API call), and the generator and the system under test would
   share a model family, contaminating any comparison.
3. **Deterministic Python generation with seeded RNG and Faker** — chosen.

Option 3 makes the ground-truth label a *mechanical consequence* of the
generation logic, not an interpretation. If a transaction is in the
$10\,000$+ USD-equivalent band, it violates `AML_THRESHOLD`; nothing else
can change that. This separation between data and reasoning is what makes
the four-arm ablation in `README.md` interpretable.

## Schema

The generator extends the original five-CSV schema (preserved
backward-compatibly so the in-code `seed_demo_data()` and any prior tooling
that read the CSV still work). New columns are additive; existing columns
keep their positions and meanings.

### `clients.csv`

| Column | Type | Description |
| ------ | ---- | ----------- |
| `client_id` | str | `Cnnn` |
| `name` | str | Faker-generated person or company name |
| `risk_level` | enum | `low` (60 %) / `medium` (30 %) / `high` (10 %) |
| `kyc_status` | enum | `valid` (80 %) / `expired` (15 %) / `missing` (5 %) **— NEW** |
| `kyc_expiry_date` | ISO date | `1970-01-01` is a sentinel for missing **— NEW** |
| `country` | ISO-3166 α-2 | client's country of residence **— NEW** |
| `pep_flag` | bool | politically-exposed-person flag, ~5 % positive **— NEW** |

### `accounts.csv`

| Column | Type | Description |
| ------ | ---- | ----------- |
| `account_id` | str | `Annnn` |
| `client_id` | str | foreign key → `clients.client_id` |
| `account_type` | enum | `checking` / `savings` / `business` |
| `status` | enum | `active` / `closed` |
| `open_date` | ISO date | well before the simulation window **— NEW** |
| `last_active_date` | ISO date | tightened by the dormancy scenario **— NEW** |
| `default_currency` | enum | `USD` / `EUR` / `GBP` / `KZT` **— NEW** |

### `transactions.csv`

| Column | Type | Description |
| ------ | ---- | ----------- |
| `tx_id` | str | `Tnnnn` |
| `account_id` | str | foreign key → `accounts.account_id` |
| `amount` | decimal | local-currency amount |
| `currency` | str | account's `default_currency` |
| `amount_usd` | decimal | USD-equivalent at the fixed FX table **— NEW** |
| `date` | ISO date | sorted ascending in the file |
| `status` | enum | `completed` / `pending` |
| `is_compliant` | bool | ground-truth label |
| `rule_ids` | comma-list | which rule scenarios this tx exercises (may be empty for `NORMAL`) |
| `counterparty_id` | str | foreign key → `counterparties.counterparty_id` **— NEW** |
| `counterparty_country` | ISO-3166 α-2 | denormalised for retrieval convenience **— NEW** |
| `tx_type` | enum | `wire` / `transfer` / `cash_deposit` / `cash_withdrawal` / `card_payment` **— NEW** |
| `description` | str | free-text memo, sometimes carries rule-specific signal **— NEW** |

### `rules.csv`

| Column | Description |
| ------ | ----------- |
| `rule_id` | one of the eight rule IDs below |
| `description` | full natural-language rule definition (used by ablation arm B) |
| `severity` | `low` / `medium` / `high` |
| `category` | `AML` / `KYC` / `Sanctions` / `Behavioral` **— NEW** |

### `tx_rules.csv`

Per-transaction rule relations. Same schema as before
(`tx_id, rule_id, relation`). `relation ∈ {compliant, violates}`.
A single tx may appear with multiple rule_ids when more than one rule fires.

### `counterparties.csv` *(new file)*

| Column | Description |
| ------ | ----------- |
| `counterparty_id` | `CP_nnn` for normal CPs, `CP_SAN_nn` for sanctioned ones |
| `name` | Faker-generated company name (or `Sanctioned Entity n` for sanctioned) |
| `country` | ISO-3166 α-2 |
| `on_sanctions_list` | bool — `true` for the ten embedded sanctioned IDs |

### `generation_metadata.json` *(new file)*

A self-documenting summary of the run, including: generator version, RNG
seed, generation timestamp, total tx count, compliant/non-compliant split,
per-rule counts, FX rates used, the embedded sanctions / high-risk lists,
and the full natural-language rule definitions. The eval pipeline reads this
file directly when constructing arm B prompts.

## Rule scenarios

Eight rules + one negative-class baseline.

| Rule | Logic | Default total / non-compliant |
| ---- | ----- | -----------------------------: |
| `AML_THRESHOLD` | single tx ≥ $10\,000$ USD-equivalent | 80 / 40 |
| `STRUCTURING` | ≥3 same-client tx in $9\,000\text{–}9\,999$ USD within 5 days | 60 / 30 |
| `SANCTIONS` | counterparty appears on the embedded sanctions list | 50 / 30 |
| `KYC_EXPIRED` | `tx_date > client.kyc_expiry_date` | 60 / 30 |
| `HIGH_RISK_JURISDICTION` | counterparty in synthetic FATF gray list **and** description ≠ `EDD applied` | 50 / 25 |
| `VELOCITY` | bursts > 10 tx/24h (medium-risk) or > 5 tx/24h (high-risk) | 50 / 25 |
| `DORMANT_REACTIVATION` | account inactive ≥ 180 days, then tx ≥ $5\,000$ USD | 40 / 20 |
| `ROUND_NUMBER_ANOMALY` | small-business-client clusters of round-USD-thousand tx | 30 / 15 |
| `NORMAL` | clean tx, no rule triggers — negative class | 80 / 0 |
| **Total** |   | **500 / 215 (43 %)** |

Compliant cases inside a rule scenario are *deliberate distractors*: they
exercise the same surface features the rule keys on, but fail the trigger
condition. Examples: a $9\,200$ USD tx that is *not* part of a structuring
cluster; a high-risk-jurisdiction tx where `description == "EDD applied"`;
a $30\,000$ USD tx after 200 days of dormancy that is correctly recorded
versus one buried in normal activity. This is what stops the ablation arms
from reducing to a single keyword check.

## Population

* **50 clients**, split 60 % low / 30 % medium / 10 % high risk.
  ~20 % flagged as small-business (used by `ROUND_NUMBER_ANOMALY`).
* **~99 accounts** (1–3 per client), with realistic open-/last-active dates.
* **100 counterparties** = 10 sanctioned (deterministic IDs `CP_SAN_01`…
  `CP_SAN_10`) + 90 normal CPs across 17 normal countries and 6 high-risk
  countries.
* Date range: **2024-06-01 → 2024-12-31** (six months).
* Currencies: USD / EUR / GBP / KZT, with a fixed FX table to USD.

## Limitations

We are honest about what synthetic data does and does not give us:

1. **No adversarial signal.** The generator does not model an evader who
   adapts to detection. Real structuring, sanctions evasion, and velocity
   attacks evolve to look like noise; this dataset does not.
2. **Simplified rule logic.** Each rule is mechanical. A real bank's AML
   pipeline considers dozens of co-occurring features and produces
   probabilistic scores, not booleans.
3. **No temporal drift.** Risk levels, KYC validity, and counterparty
   relationships are static across the six-month window.
4. **Counterparty / client name realism is shallow.** Faker generates
   plausible Western-style names; we do not model name variants,
   transliterations, or legal-entity hierarchies — all of which are
   significant in real sanctions screening.
5. **Currency conversion is fixed.** Real FX rates change daily; we use a
   single rate table for reproducibility.

These limitations are accepted because the dissertation's research question
is about *how the KG-augmented LLM reasons over a labelled dataset*, not
about how it would behave on unfiltered production data. The synthetic
dataset is intentionally a clean test bed.

## Prompts

The dissertation's prompt artifact is the set of system prompts hard-coded
in `src/financial_llm.py` (`FinancialLLM.system_prompt` and
`ask_compliance_json`'s instruction text). The original draft prompts were
iteratively refined with ChatGPT during the prior development pass (see
the GitHub commit history for `src/financial_llm.py`); the on-disk version
is the artifact that the evaluation runs use.
