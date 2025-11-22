# src/retriever.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from financial_kg import FinancialKG


@dataclass
class TransactionFact:
    """Lightweight view of a transaction for LLM consumption."""
    tx_id: str
    account_id: str
    amount: Optional[float]
    currency: Optional[str]
    date: Optional[str]
    status: Optional[str]
    is_compliant: Optional[bool]


class FinancialRetriever:
    """
    Symbolic retrieval layer over the FinancialKG.

    Responsibilities:
    - Query the KG for client- and transaction-level information.
    - Convert structured KG results into concise, human-readable fact strings.
    - Provide data that the controller can pass to FinancialLLM.ask().
    """

    def __init__(self, kg: FinancialKG) -> None:
        self.kg = kg

    # ------------------------------------------------------------------ Helpers

    @staticmethod
    def _shorten_uri(uri: str) -> str:
        """
        Convert a full URI into a shorter identifier, assuming the local
        name is after the last '#' or '/'.
        """
        if "#" in uri:
            return uri.rsplit("#", 1)[-1]
        if "/" in uri:
            return uri.rsplit("/", 1)[-1]
        return uri

    @staticmethod
    def _tx_dict_to_fact(tx: Dict[str, Any]) -> TransactionFact:
        """Convert the dict from FinancialKG into a typed TransactionFact."""
        return TransactionFact(
            tx_id=FinancialRetriever._shorten_uri(tx["tx_uri"]),
            account_id=FinancialRetriever._shorten_uri(tx["account_uri"]),
            amount=tx.get("amount"),
            currency=tx.get("currency"),
            date=tx.get("date"),
            status=tx.get("status"),
            is_compliant=tx.get("is_compliant"),
        )

    # ------------------------------------------------------ High-level retrieval

    def get_client_transactions(
        self, client_id: str
    ) -> List[TransactionFact]:
        """
        Return a list of TransactionFact objects for a given client.
        Thin wrapper over FinancialKG.get_transactions_for_client().
        """
        raw = self.kg.get_transactions_for_client(client_id)
        return [self._tx_dict_to_fact(t) for t in raw]

    def get_client_transactions_facts(
        self, client_id: str
    ) -> str:
        """
        Return a human-readable summary of transactions for a client,
        suitable for injection into the LLM as contextual facts.
        """
        tx_facts = self.get_client_transactions(client_id)

        if not tx_facts:
            return f"No transactions found for client '{client_id}'."

        lines: List[str] = []
        lines.append(f"Known transactions for client '{client_id}':")

        for tx in tx_facts:
            # Build a compact descriptive sentence
            parts = []

            if tx.date:
                parts.append(f"on {tx.date}")
            parts.append(f"transaction {tx.tx_id}")

            if tx.amount is not None and tx.currency:
                parts.append(f"of {tx.amount:.2f} {tx.currency}")

            if tx.status:
                parts.append(f"status: {tx.status}")

            if tx.is_compliant is not None:
                parts.append(
                    f"compliance: {'compliant' if tx.is_compliant else 'non-compliant'}"
                )

            line = "- " + ", ".join(parts)
            lines.append(line)

        return "\n".join(lines)

    def get_transaction_compliance_facts(
        self, tx_id: str
    ) -> str:
        """
        Return a human-readable explanation of which rules a transaction
        is compliant with or violates, based on KG content.
        """
        data = self.kg.explain_transaction_compliance(tx_id)
        rules = data.get("rules", [])

        if not rules:
            return (
                f"No explicit compliance or violation rules were found "
                f"for transaction '{tx_id}'."
            )

        lines: List[str] = [f"Compliance-related facts for transaction '{tx_id}':"]
        for r in rules:
            rule_short = self._shorten_uri(r["rule_uri"])
            relation = r["relation"]

            if relation == "compliantWith":
                lines.append(f"- Transaction {tx_id} is compliant with rule {rule_short}.")
            elif relation == "violatesRule":
                lines.append(f"- Transaction {tx_id} violates rule {rule_short}.")
            else:
                lines.append(
                    f"- Transaction {tx_id} has relation '{relation}' with rule {rule_short}."
                )

        return "\n".join(lines)

    def build_context_for_client_and_tx(
        self,
        client_id: str,
        tx_id: Optional[str] = None,
    ) -> str:
        """
        Convenience method that combines client-level transaction history
        and (optionally) a specific transaction's compliance explanation
        into a single context string.
        """
        sections: List[str] = []

        sections.append(self.get_client_transactions_facts(client_id))

        if tx_id is not None:
            sections.append("")  # blank line
            sections.append(self.get_transaction_compliance_facts(tx_id))

        return "\n".join(sections)


# --------------------------------------------------------------------------- #
# Quick manual test when running this file directly
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from decimal import Decimal

    # Minimal smoke test
    kg = FinancialKG()
    kg.seed_demo_data()

    retriever = FinancialRetriever(kg)

    print("=== Context for client A ===")
    print(retriever.get_client_transactions_facts("A"))
    print("\n=== Context for client A + tx T002 ===")
    print(retriever.build_context_for_client_and_tx("A", "T002"))
