# src/financial_kg.py
from dataclasses import dataclass
from pathlib import Path
from typing import List
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, XSD

EX = Namespace("http://example.org/finance/")

@dataclass
class KGFact:
    subject: str
    predicate: str
    obj: str

class FinancialKG:
    """
    Simple financial knowledge graph using rdflib.
    """

    def __init__(self, ttl_path: Path | None = None):
        self.graph = Graph()
        self.graph.bind("ex", EX)
        if ttl_path and ttl_path.exists():
            self.graph.parse(str(ttl_path), format="turtle")

    def add_mock_data(self) -> None:
        """
        Populate the KG with some synthetic clients, accounts, and transactions.
        """
        g = self.graph

        client_a = EX.ClientA
        account_a1 = EX.AccountA1
        tx_123 = EX.Transaction123

        g.add((client_a, RDF.type, EX.Client))
        g.add((account_a1, RDF.type, EX.Account))
        g.add((tx_123, RDF.type, EX.Transaction))

        g.add((client_a, EX.hasAccount, account_a1))
        g.add((account_a1, EX.hasTransaction, tx_123))

        g.add((tx_123, EX.amount, Literal("9000", datatype=XSD.decimal)))
        g.add((tx_123, EX.currency, Literal("USD")))
        g.add((tx_123, EX.date, Literal("2025-01-10", datatype=XSD.date)))
        g.add((tx_123, EX.isCompliantWith, EX.KYC))

    def save_turtle(self, path: Path) -> None:
        self.graph.serialize(destination=str(path), format="turtle")

    def query_transactions_for_client(self, client_iri: str) -> List[KGFact]:
        """
        Simple example: get transactions linked to a client.
        """
        g = self.graph
        client = EX[client_iri]
        q = """
        PREFIX ex: <http://example.org/finance/>

        SELECT ?tx ?amount ?currency ?date ?rule
        WHERE {
          ?client ex:hasAccount ?acc .
          ?acc ex:hasTransaction ?tx .
          OPTIONAL { ?tx ex:amount ?amount . }
          OPTIONAL { ?tx ex:currency ?currency . }
          OPTIONAL { ?tx ex:date ?date . }
          OPTIONAL { ?tx ex:isCompliantWith ?rule . }
        }
        """
        results = g.query(q, initBindings={"client": client})

        facts: List[KGFact] = []
        for row in results:
            tx = str(row.tx)
            if row.amount:
                facts.append(KGFact(tx, "amount", str(row.amount)))
            if row.currency:
                facts.append(KGFact(tx, "currency", str(row.currency)))
            if row.date:
                facts.append(KGFact(tx, "date", str(row.date)))
            if row.rule:
                facts.append(KGFact(tx, "isCompliantWith", str(row.rule)))
        return facts

    @staticmethod
    def format_facts_for_llm(facts: List[KGFact]) -> str:
        lines = []
        for f in facts:
            lines.append(f"- {f.subject} {f.predicate} {f.obj}")
        return "\n".join(lines) if lines else "No relevant facts were found."
