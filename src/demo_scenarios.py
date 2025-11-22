# src/demo_scenarios.py

from controller import FinancialController
from financial_kg import FinancialKG


def run_demo():
    """
    Run the end-to-end pipeline:
      - Seed KG
      - Ask a question about a client's transactions
      - Ask for compliance explanation
    """

    # Build the controller (this also builds KG, retriever, LLM)
    controller = FinancialController()

    # Seed symbolic memory
    controller.kg.seed_demo_data()

    print("=== Demo: Client A transaction summary ===")
    answer1 = controller.answer_client_transaction_question(
        client_id="A",
        user_question="Summarize Client A's recent transactions and highlight any risky ones."
    )
    print(answer1)
    print()

    print("=== Demo: Compliance explanation for T002 ===")
    answer2 = controller.explain_transaction_compliance("T002")
    print(answer2)
    print()


if __name__ == "__main__":
    run_demo()
