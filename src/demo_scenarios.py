# src/demo_scenarios.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from controller import FinancialController


def _build_controller(seed_demo: bool = True) -> FinancialController:
    """
    Create a FinancialController and optionally seed the KG with demo data.
    """
    controller = FinancialController()
    if seed_demo:
        controller.kg.seed_demo_data()
    return controller


def _log_entry(
    log_path: Optional[Path],
    entry: Dict[str, Any],
) -> None:
    """
    Append a single JSON entry to a .jsonl log file (if log_path is provided).
    """
    if not log_path:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Demo scenarios
# --------------------------------------------------------------------------- #


def scenario_summary(
    controller: FinancialController,
    client_id: str,
    log_path: Optional[Path] = None,
) -> None:
    """
    Scenario 1:
      - Summarize recent transactions for a client.
      - Highlight any risky / notable ones.
    """
    question = (
        f"Summarize client {client_id}'s recent transactions and highlight any that "
        f"might be risky or unusual."
    )

    # Get facts explicitly for the log & for transparency
    facts_text = controller.retriever.get_client_transactions_facts(client_id)
    response = controller.llm.ask(
        user_message=question,
        context_facts=facts_text,
    )

    print("=== Scenario: Client transaction summary ===")
    print(f"Client ID: {client_id}\n")
    print("Facts passed to the LLM:")
    print(facts_text)
    print("\nLLM response:")
    print(response)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "scenario": "summary",
        "client_id": client_id,
        "question": question,
        "facts": facts_text,
        "llm_response": response,
    }
    _log_entry(log_path, entry)


def scenario_compliance(
    controller: FinancialController,
    tx_id: str,
    log_path: Optional[Path] = None,
) -> None:
    """
    Scenario 2:
      - Explain whether a given transaction is compliant and why.
    """
    default_question = (
        f"Based on the facts, explain whether transaction {tx_id} is compliant "
        f"or non-compliant, and why. Be explicit about the rules."
    )

    facts_text = controller.retriever.get_transaction_compliance_facts(tx_id)
    response = controller.llm.ask(
        user_message=default_question,
        context_facts=facts_text,
    )

    print("=== Scenario: Transaction compliance explanation ===")
    print(f"Transaction ID: {tx_id}\n")
    print("Facts passed to the LLM:")
    print(facts_text)
    print("\nLLM response:")
    print(response)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "scenario": "compliance",
        "tx_id": tx_id,
        "question": default_question,
        "facts": facts_text,
        "llm_response": response,
    }
    _log_entry(log_path, entry)


def scenario_raw_facts(
    controller: FinancialController,
    client_id: str,
    log_path: Optional[Path] = None,
) -> None:
    """
    Scenario 3:
      - Print the raw KG facts for a client (no LLM).
      - Useful to show how symbolic memory looks on its own.
    """
    facts_text = controller.retriever.get_client_transactions_facts(client_id)

    print("=== Scenario: Raw KG facts for client ===")
    print(f"Client ID: {client_id}\n")
    print(facts_text)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "scenario": "raw-facts",
        "client_id": client_id,
        "facts": facts_text,
        "llm_response": None,
    }
    _log_entry(log_path, entry)


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo scenarios for the financial KG + LLM agent."
    )
    parser.add_argument(
        "--scenario",
        choices=["summary", "compliance", "raw-facts"],
        default="summary",
        help="Which scenario to run.",
    )
    parser.add_argument(
        "--client-id",
        default="A",
        help="Client ID to use (for summary/raw-facts scenarios). Default: A",
    )
    parser.add_argument(
        "--tx-id",
        default="T002",
        help="Transaction ID to use (for compliance scenario). Default: T002",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to a JSONL log file (e.g., logs/demo_runs.jsonl).",
    )

    args = parser.parse_args()
    log_path = Path(args.log_file) if args.log_file else None

    controller = _build_controller(seed_demo=True)

    if args.scenario == "summary":
        scenario_summary(controller, client_id=args.client_id, log_path=log_path)
    elif args.scenario == "compliance":
        scenario_compliance(controller, tx_id=args.tx_id, log_path=log_path)
    elif args.scenario == "raw-facts":
        scenario_raw_facts(controller, client_id=args.client_id, log_path=log_path)
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")


if __name__ == "__main__":
    main()
