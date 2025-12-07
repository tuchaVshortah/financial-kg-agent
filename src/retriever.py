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
        self,
        client_id: str,
        include_compliance_flag: bool = False,
    ) -> str:
        """
        Returns a textual summary of all transactions for a given client.

        If include_compliance_flag is True, the textual summary will contain
        the `compliance: true / false / unknown` field for each transaction.
        Otherwise, that field is omitted, and the LLM must infer risk/compliance
        only from amounts, dates, etc.
        """
        txs = self.kg.get_transactions_for_client(client_id)

        if not txs:
            return f"No known transactions for client '{client_id}'."

        lines = [f"Known transactions for client '{client_id}':"]
        for tx in txs:
            tx_id = tx["tx_uri"].split("#")[-1]
            amount = tx.get("amount")
            currency = tx.get("currency")
            date = tx.get("date")
            status = tx.get("status") or "unknown"

            # Ground truth, but *optionally* shown to the LLM
            is_compliant = tx.get("is_compliant")
            if is_compliant is None:
                compliance_str = "unknown"
            else:
                compliance_str = "true" if is_compliant else "false"

            # Build the base line
            base = (
                f"- on {date}, transaction {tx_id}, "
                f"of {amount:.2f} {currency}, status: {status}"
            )

            if include_compliance_flag:
                base += f", compliance: {compliance_str}"

            lines.append(base)

        return "\n".join(lines)

    def get_transaction_compliance_facts(self, tx_id: str) -> str:
        """
        Return a human-readable description of which rules a transaction
        is compliant with or violates, based on KG relations.
        """
        info = self.kg.explain_transaction_compliance(tx_id)
        rules = info.get("rules", [])

        lines = [f"Compliance-related facts for transaction '{tx_id}':"]
        if not rules:
            lines.append(f"- No explicit compliance or violation rules are recorded for transaction {tx_id}.")
        else:
            for r in rules:
                rule_uri = r.get("rule_uri")
                relation = r.get("relation")  # "compliantWith" or "violatesRule"
                rule_id = rule_uri.split("#")[-1] if rule_uri else "UNKNOWN_RULE"

                if relation == "compliantWith":
                    lines.append(f"- Transaction {tx_id} is compliant with rule {rule_id}.")
                elif relation == "violatesRule":
                    lines.append(f"- Transaction {tx_id} violates rule {rule_id}.")
                else:
                    lines.append(f"- Transaction {tx_id} is related to rule {rule_id} (relation: {relation}).")

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
