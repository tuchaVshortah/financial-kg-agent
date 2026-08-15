# financial-kg-agent

Hybrid Knowledge-Graph + Large-Language-Model agent for compliance reasoning over
synthetic financial transactions. Companion code for the M.Sc. dissertation
*"Investigating Knowledge Graphs for Context-Aware Search in Financial
Transactional AI Agents"* (SDU University, 2026) and the corresponding ICECCO 2026
publication.

## Architecture

```
┌──────────────┐   facts    ┌──────────────┐   prompt   ┌──────────────┐
│ FinancialKG  │ ─────────▶ │  Retriever   │ ─────────▶ │ FinancialLLM │
│  (rdflib)    │            │  (text view) │            │  (OpenAI)    │
└──────────────┘            └──────────────┘            └──────────────┘
        ▲                                                       │
        └─────────────── FinancialController ───────────────────┘
```

* `src/financial_kg.py` — RDF graph (clients, accounts, transactions, rules) +
  CSV loader + ground-truth queries.
* `src/retriever.py` — turns SPARQL results into compact natural-language fact
  blocks for the LLM.
* `src/financial_llm.py` — thin OpenAI Chat Completions wrapper with retries
  and JSON-mode helpers.
* `src/controller.py` — orchestrator with the high-level workflows
  (summary, per-transaction explanation, JSON evaluation).
* `src/demo_scenarios.py` — CLI entry point for the demo / evaluation runs.

## Requirements

* Python **3.14** (see `.python-version`)
* macOS / Linux
* An OpenAI API key

## Setup

```bash
git clone https://github.com/tuchaVshortah/financial-kg-agent.git
cd financial-kg-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

## Running the demo

All entry points must be launched as modules from the project root so that the
package imports resolve correctly:

```bash
# Default: in-code seed (3 transactions, client A only)
python -m src.demo_scenarios --scenario all

# Use the CSV dataset under data/ instead of the in-code seed
python -m src.demo_scenarios --scenario all --use-csv

# Just the JSON-mode compliance evaluation, write JSONL log
python -m src.demo_scenarios --scenario eval --use-csv \
    --log-file logs/eval_run.jsonl
```

CLI flags:

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--scenario` | `all` | One of `summary`, `compliance`, `eval`, `all`. |
| `--client-id` | `A` | Client to use in the summary scenario. |
| `--tx-id` | `T002` | Transaction to use in the compliance scenario. |
| `--log-file` | _(none)_ | If set, append run records as JSONL to this path. |
| `--use-csv` | `false` | Load the KG from `data/*.csv` instead of `seed_demo_data()`. |
| `--data-dir` | `./data` | Override the CSV directory. |

## Data

The repository ships with the synthetic dataset used in the dissertation
evaluation under `data/` (500 transactions across 8 rule scenarios plus a
clean negative class). The dataset is reproducibly regenerated with:

```bash
python -m src.generate_data --n 500 --seed 42 --out data/
```

Methodology, schema, and rule definitions are documented in
`docs/dataset_generation.md`.

## Evaluation: four-arm ablation

The dissertation comparison runs each transaction through four grounding
modes, all sharing the same base context (transaction row + counterparty +
client + account) so the only thing that varies is what grounding gets
appended:

| Arm | Context given to the LLM | Purpose |
| --- | ------------------------ | ------- |
| **A** — vanilla LLM | Base only, no rules, no KG | LLM-only baseline (parametric knowledge) |
| **B** — rule-text grounding | Base + natural-language rule definitions | Tests rule synthesis |
| **C** — per-rule relations | Base + rule defs + KG facts ("Tx T002 violates Rule_KYC") | The KG-augmented system under test |
| **D** — full compliance flag | Base + rule defs + KG facts + ground-truth flag | Upper-bound sanity check |

`src/eval_arms.py` defines the four arms declaratively;
`src/eval_runner.py` is the CLI runner.

```bash
# Offline smoke test — verifies the harness end-to-end without API spend
python -m src.eval_runner --dry-run --limit 20 \
    --data-dir data --out runs/smoke

# Live run (uses gpt-4o-mini via OPENAI_API_KEY; costs API credits)
python -m src.eval_runner --data-dir data --out runs/full \
    --arms A,B,C,D --log-file runs/full/per_tx.jsonl
```

The runner writes `results.csv` (one row per `(tx_id, arm)` with prediction,
correctness, and explanation) and `summary.json` (per-arm accuracy /
precision / recall / F1 / confusion matrix + per-rule accuracy breakdown).
`runs/` is git-ignored so output never accidentally lands in commits.

## License

MIT — see `LICENSE`.
