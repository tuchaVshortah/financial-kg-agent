# src/controller.py

from __future__ import annotations

from typing import Optional, Dict, Any
import json

from financial_kg import FinancialKG
from financial_llm import FinancialLLM
from retriever import FinancialRetriever


class FinancialController:
    """
    Simple orchestrator that connects:
      - FinancialKG  (symbolic memory)
      - FinancialRetriever (formats KG data as facts)
      - FinancialLLM (reasoning + explanation)

    Option A: a small, single-agent controller with a few
    explicit workflows rather than a generic routing engine.
    """

    def __init__(
        self,
        kg: Optional[FinancialKG] = None,
        retriever: Optional[FinancialRetriever] = None,
        llm: Optional[FinancialLLM] = None,
    ) -> None:
        # Allow dependency injection, but provide sensible defaults
        self.kg = kg or FinancialKG()
        self.retriever = retriever or FinancialRetriever(self.kg)
        self.llm = llm or FinancialLLM()

    # ------------------------------------------------------------------ Workflows

    def answer_client_transaction_question(
        self,
        client_id: str,
        user_question: str,
    ) -> str:
        """
        High-level workflow:
          1. Get symbolic facts (transactions) for a given client.
          2. Feed them as context to the LLM.
          3. Return the model's answer.

        Example question: "Is client A engaged in suspicious activity?"
        """
        facts = self.retriever.get_client_transactions_facts(client_id)
        return self.llm.ask(
            user_message=user_question,
            context_facts=facts,
        )

    def explain_transaction_compliance(
        self,
        tx_id: str,
        user_question: Optional[str] = None,
    ) -> str:
        """
        High-level workflow:
          1. Get compliance-related facts for a single transaction.
          2. Ask the LLM to explain whether it is compliant and why.

        If user_question is not provided, a default one is used.
        """
        facts = self.retriever.get_transaction_compliance_facts(tx_id)

        question = (
            user_question
            or f"Based on the facts, explain whether transaction {tx_id} "
               f"is compliant or non-compliant, and why."
        )

        return self.llm.ask(
            user_message=question,
            context_facts=facts,
        )
    
    def evaluate_transaction_compliance_json(self, tx_id: str) -> Dict[str, Any]:
        """
        Evaluation helper:
          - Retrieves KG facts about a transaction's compliance.
          - Asks the LLM for a JSON decision.
          - Compares model decision vs. KG ground truth.

        Returns a dict summarizing the evaluation.
        """
        facts = self.retriever.get_transaction_compliance_facts(tx_id)
        ground_truth = self.kg.get_transaction_compliance_label(tx_id)

        user_msg = (
            f"Decide whether transaction {tx_id} is compliant based on the facts. "
            "Remember: respond ONLY with the requested JSON fields."
        )

        parsed, raw = self.llm.ask_compliance_json(
            user_message=user_msg,
            context_facts=facts,
        )

        model_label = None
        explanation = None
        if isinstance(parsed, dict):
            model_label = parsed.get("is_compliant")
            explanation = parsed.get("explanation")

        correct = None
        if ground_truth is not None and isinstance(model_label, bool):
            correct = (ground_truth == model_label)

        return {
            "tx_id": tx_id,
            "facts": facts,
            "ground_truth": ground_truth,
            "model_label": model_label,
            "correct": correct,
            "explanation": explanation,
            "raw_response": raw,
        }


    # ------------------------------------------------------------------ Demo helper

    def run_demo(self) -> None:
        """
        Very small, deterministic demo:
          - Seeds the KG
          - Asks about client A
          - Asks for an explanation of a specific transaction
        """
        # Seed symbolic memory
        self.kg.seed_demo_data()

        print("=== Demo: Client A transaction overview ===")
        answer1 = self.answer_client_transaction_question(
            client_id="A",
            user_question="Summarize Client A's recent transactions and highlight any that might be risky.",
        )
        print(answer1)
        print()

        print("=== Demo: Compliance explanation for T002 ===")
        answer2 = self.explain_transaction_compliance(tx_id="T002")
        print(answer2)


# --------------------------------------------------------------------------- #
# Manual run
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    controller = FinancialController()
    controller.run_demo()
