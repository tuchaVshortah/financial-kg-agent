# src/demo_scenarios.py

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from financial_kg import FinancialKG
from financial_llm import FinancialLLM
from retriever import FinancialRetriever
from controller import FinancialController


def build_controller(use_csv: bool, data_dir: Optional[Path]) -> FinancialController:
    """
    Build a controller with either code-seeded demo data or CSV-seeded data.
    """
    kg = FinancialKG()

    if use_csv:
        # Default to ../data relative to this file if not provided
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        kg.load_from_csv(data_dir)
    else:
        kg.seed_demo_data()

    retriever = FinancialRetriever(kg)
    llm = FinancialLLM()
    return FinancialController(kg=kg, retriever=retriever, llm=llm)


def log_entry(log_file: Optional[Path], entry: dict) -> None:
    if not log_file:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        **entry,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_summary_scenario(controller: FinancialController, client_id: str, log_file: Optional[Path]) -> None:
    print("=== Scenario: Client transaction summary ===")
    print(f"Client ID: {client_id}\n")

    facts = controller.retriever.get_client_transactions_facts(client_id)
    print("Facts passed to the LLM:")
    print(facts)
    print()

    response = controller.answer_client_transaction_question(
        client_id=client_id,
        user_question="Summarize this client's recent transactions and highlight any that might be risky.",
    )
    print("LLM response:")
    print(response)

    log_entry(
        log_file,
        {
            "scenario": "summary",
            "client_id": client_id,
            "facts": facts,
            "llm_response": response,
        },
    )


def run_compliance_scenario(controller: FinancialController, tx_id: str, log_file: Optional[Path]) -> None:
    print("=== Scenario: Transaction compliance explanation ===")
    print(f"Transaction ID: {tx_id}\n")

    facts = controller.retriever.get_transaction_compliance_facts(tx_id)
    print("Facts passed to the LLM:")
    print(facts)
    print()

    response = controller.explain_transaction_compliance(tx_id=tx_id)
    print("LLM response:")
    print(response)

    log_entry(
        log_file,
        {
            "scenario": "compliance",
            "tx_id": tx_id,
            "facts": facts,
            "llm_response": response,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial KG + LLM demo scenarios")
    parser.add_argument(
        "--scenario",
        choices=["summary", "compliance", "all"],
        default="all",
        help="Which demo scenario to run",
    )
    parser.add_argument(
        "--client-id",
        default="A",
        help="Client ID to use for summary scenario",
    )
    parser.add_argument(
        "--tx-id",
        default="T002",
        help="Transaction ID to use for compliance scenario",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to JSONL log file (e.g. logs/demo_runs.jsonl)",
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Load KG demo data from CSV files in ../data instead of code-seeded demo data",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the default ../data path for CSV loading",
    )

    args = parser.parse_args()

    controller = build_controller(use_csv=args.use_csv, data_dir=args.data_dir)

    if args.scenario in ("summary", "all"):
        run_summary_scenario(controller, client_id=args.client_id, log_file=args.log_file)

    if args.scenario in ("compliance", "all"):
        print()
        run_compliance_scenario(controller, tx_id=args.tx_id, log_file=args.log_file)


if __name__ == "__main__":
    main()
