# src/retriever.py
from typing import List
from .financial_kg import FinancialKG, KGFact

class DataRetriever:
    """
    Thin layer that decides *how* to query the KG for a given user query.
    For now it's very simple and rule-based.
    """

    def __init__(self, kg: FinancialKG):
        self.kg = kg

    def retrieve_for_query(self, user_query: str) -> List[KGFact]:
        user_query_lower = user_query.lower()

        # Very naive routing — can be improved later
        if "client a" in user_query_lower or "client a" in user_query:
            return self.kg.query_transactions_for_client("ClientA")

        # Default: no facts found
        return []
