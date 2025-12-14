# src/financial_kg.py

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import csv  # <-- NEW

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

# --- Domain dataclasses -----------------------------------------------------


@dataclass
class Client:
    """Simple in-memory representation of a client entity."""
    client_id: str
    name: Optional[str] = None
    risk_level: Optional[str] = None  # e.g. "low", "medium", "high"


@dataclass
class Account:
    """Simple in-memory representation of an account."""
    account_id: str
    client_id: str
    account_type: Optional[str] = None  # e.g. "checking", "savings"
    status: Optional[str] = None        # e.g. "active", "closed"


@dataclass
class Transaction:
    """Simple in-memory representation of a transaction."""
    tx_id: str
    account_id: str
    amount: Decimal
    currency: str
    date: str  # ISO date string "YYYY-MM-DD" for simplicity
    status: Optional[str] = None       # e.g. "completed", "pending"
    is_compliant: Optional[bool] = None
    rule_ids: Optional[List[str]] = None  # Compliance rules that apply


# --- Core Knowledge Graph wrapper ------------------------------------------


class FinancialKG:
    """
    Wrapper around an RDFLib Graph representing a small financial domain
    (clients, accounts, transactions, regulations and compliance rules).

    This class is intentionally independent of any LLM logic. It is used
    by the retriever / controller as the symbolic memory layer.
    """

    def __init__(
        self,
        base_iri: str = "http://example.org/finance#",
        ttl_path: Optional[Path] = None,
    ) -> None:
        self.graph = Graph()
        self.base_iri = base_iri
        self.EX = Namespace(base_iri)

        # Bind prefixes for nicer serialization / debugging
        self.graph.bind("ex", self.EX)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)

        # Load from an existing TTL file if provided
        if ttl_path is not None and ttl_path.exists():
            self.graph.parse(str(ttl_path), format="turtle")

        # Ensure core classes & properties exist in the graph
        self._ensure_schema()

    # ------------------------------------------------------------------ Schema

    def _ensure_schema(self) -> None:
        """Add lightweight schema triples if they are not already present."""
        # Classes
        for cls in ["Client", "Account", "Transaction", "Regulation", "ComplianceRule"]:
            cls_uri = self.EX[cls]
            self.graph.add((cls_uri, RDF.type, RDFS.Class))

        # Object properties
        obj_props = {
            "hasAccount": ("Client", "Account"),
            "hasTransaction": ("Account", "Transaction"),
            "isCompliantWith": ("Transaction", "ComplianceRule"),
            "governedBy": ("Transaction", "Regulation"),
            "violatesRule": ("Transaction", "ComplianceRule"),
        }
        for prop, (domain_cls, range_cls) in obj_props.items():
            prop_uri = self.EX[prop]
            self.graph.add((prop_uri, RDF.type, RDF.Property))
            self.graph.add((prop_uri, RDFS.domain, self.EX[domain_cls]))
            self.graph.add((prop_uri, RDFS.range, self.EX[range_cls]))

        # Data properties (we just declare them as properties)
        for prop in ["amount", "currency", "date", "status", "riskLevel", "name"]:
            prop_uri = self.EX[prop]
            self.graph.add((prop_uri, RDF.type, RDF.Property))

    # ---------------------------------------------------------- Helper builders

    def client_uri(self, client_id: str) -> URIRef:
        return self.EX[f"Client_{client_id}"]

    def account_uri(self, account_id: str) -> URIRef:
        return self.EX[f"Account_{account_id}"]

    def tx_uri(self, tx_id: str) -> URIRef:
        return self.EX[f"Transaction_{tx_id}"]

    def rule_uri(self, rule_id: str) -> URIRef:
        return self.EX[f"Rule_{rule_id}"]

    # --------------------------------------------------------------- Add / Upsert

    def add_client(self, client: Client) -> None:
        """Insert or update a client entity in the KG."""
        c_uri = self.client_uri(client.client_id)
        self.graph.add((c_uri, RDF.type, self.EX.Client))

        if client.name:
            self.graph.set((c_uri, self.EX.name, Literal(client.name)))
        if client.risk_level:
            self.graph.set((c_uri, self.EX.riskLevel,
                           Literal(client.risk_level)))

    def add_account(self, account: Account) -> None:
        """Insert or update an account and its link to a client."""
        a_uri = self.account_uri(account.account_id)
        c_uri = self.client_uri(account.client_id)

        self.graph.add((a_uri, RDF.type, self.EX.Account))
        self.graph.add((c_uri, self.EX.hasAccount, a_uri))

        if account.account_type:
            self.graph.set((a_uri, self.EX.accountType,
                           Literal(account.account_type)))
        if account.status:
            self.graph.set((a_uri, self.EX.status, Literal(account.status)))

    def add_transaction(self, tx: Transaction) -> None:
        """Insert or update a transaction and its links to accounts and rules."""
        t_uri = self.tx_uri(tx.tx_id)
        a_uri = self.account_uri(tx.account_id)

        self.graph.add((t_uri, RDF.type, self.EX.Transaction))
        self.graph.add((a_uri, self.EX.hasTransaction, t_uri))

        self.graph.set((t_uri, self.EX.amount, Literal(
            tx.amount, datatype=XSD.decimal)))
        self.graph.set((t_uri, self.EX.currency, Literal(tx.currency)))
        self.graph.set(
            (t_uri, self.EX.date, Literal(tx.date, datatype=XSD.date)))
        if tx.status:
            self.graph.set((t_uri, self.EX.status, Literal(tx.status)))

        if tx.is_compliant is not None:
            self.graph.set(
                (t_uri, self.EX.isCompliant, Literal(
                    tx.is_compliant, datatype=XSD.boolean))
            )

        # Link to rules: compliant → isCompliantWith, non-compliant → violatesRule
        if tx.rule_ids:
            for rule_id in tx.rule_ids:
                r_uri = self.rule_uri(rule_id)
                self.graph.add((r_uri, RDF.type, self.EX.ComplianceRule))

                if tx.is_compliant is True:
                    self.graph.add((t_uri, self.EX.isCompliantWith, r_uri))
                elif tx.is_compliant is False:
                    self.graph.add((t_uri, self.EX.violatesRule, r_uri))

    # --------------------------------------------------------------- Query helpers

    def get_transactions_for_client(self, client_id: str) -> List[Dict[str, Any]]:
        """
        Return a list of transaction dicts for a given client_id.
        This is intended for feeding into the retriever / LLM as context.
        """
        c_uri = self.client_uri(client_id)
        query = f"""
        PREFIX ex: <{self.base_iri}>
        PREFIX xsd: <{XSD}>

        SELECT ?tx ?account ?amount ?currency ?date ?status ?isCompliant
        WHERE {{
            <{c_uri}> ex:hasAccount ?account .
            ?account ex:hasTransaction ?tx .
            OPTIONAL {{ ?tx ex:amount ?amount . }}
            OPTIONAL {{ ?tx ex:currency ?currency . }}
            OPTIONAL {{ ?tx ex:date ?date . }}
            OPTIONAL {{ ?tx ex:status ?status . }}
            OPTIONAL {{ ?tx ex:isCompliant ?isCompliant . }}
        }}
        ORDER BY ?date
        """
        results = self.graph.query(query)

        txs: List[Dict[str, Any]] = []
        for row in results:
            txs.append(
                {
                    "tx_uri": str(row.tx),
                    "account_uri": str(row.account),
                    "amount": float(row.amount) if row.amount else None,
                    "currency": str(row.currency) if row.currency else None,
                    "date": str(row.date) if row.date else None,
                    "status": str(row.status) if row.status else None,
                    "is_compliant": bool(row.isCompliant) if row.isCompliant else None,
                }
            )
        return txs

    def explain_transaction_compliance(self, tx_id: str) -> Dict[str, Any]:
        """
        Return a structured view of which rules a transaction is compliant with
        (or violates), suitable for conversion into natural language by the LLM.
        """
        t_uri = self.tx_uri(tx_id)
        query = f"""
        PREFIX ex: <{self.base_iri}>
        SELECT ?rule ?rel
        WHERE {{
            OPTIONAL {{ <{t_uri}> ex:isCompliantWith ?rule . BIND("compliantWith" AS ?rel) }}
            OPTIONAL {{ <{t_uri}> ex:violatesRule ?rule . BIND("violatesRule" AS ?rel) }}
        }}
        """
        rules = []
        for row in self.graph.query(query):
            rules.append(
                {
                    "rule_uri": str(row.rule),
                    "relation": str(row.rel),
                }
            )

        return {
            "tx_uri": str(t_uri),
            "rules": rules,
        }

    def get_transaction_compliance_label(self, tx_id: str) -> Optional[bool]:
        """
        Return the ground-truth compliance label for a transaction based on the KG.

        Returns
        -------
        Optional[bool]
            True if compliant, False if non-compliant, or None if not specified.
        """
        t_uri = self.tx_uri(tx_id)
        query = f"""
        PREFIX ex: <{self.base_iri}>
        SELECT ?isCompliant
        WHERE {{
            <{t_uri}> ex:isCompliant ?isCompliant .
        }}
        """
        results = list(self.graph.query(query))
        if not results:
            return None

        # We assume at most one isCompliant triple
        row = results[0]
        # RDFLib already maps xsd:boolean to Python bool in most cases,
        # but we guard with a simple cast fallback.
        val = row.isCompliant.toPython() if hasattr(
            row.isCompliant, "toPython") else row.isCompliant
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return None

    # ---------------------------------------------------------- CSV data loader

    def load_from_csv(self, data_dir: Path) -> None:
        """
        Load clients, accounts, and transactions from CSV files in `data_dir`.

        Expected files (all optional, but processed in this order if present):
          - clients.csv
          - accounts.csv
          - transactions.csv

        Each file is expected to have a header row. See repository docs or
        README for column definitions.
        """
        data_dir = Path(data_dir)

        clients_path = data_dir / "clients.csv"
        accounts_path = data_dir / "accounts.csv"
        tx_path = data_dir / "transactions.csv"
        rules_path = data_dir / "rules.csv"
        tx_rules_path = data_dir / "tx_rules.csv"

        # --- Load clients -----------------------------------------------------
        if clients_path.exists():
            with clients_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    client = Client(
                        client_id=row["client_id"],
                        name=row.get("name") or None,
                        risk_level=row.get("risk_level") or None,
                    )
                    self.add_client(client)

        # --- Load accounts ----------------------------------------------------
        if accounts_path.exists():
            with accounts_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    account = Account(
                        account_id=row["account_id"],
                        client_id=row["client_id"],
                        account_type=row.get("account_type") or None,
                        status=row.get("status") or None,
                    )
                    self.add_account(account)

        # --- Load transactions -----------------------------------------------
        if tx_path.exists():
            with tx_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    amount = self._parse_decimal(row.get("amount"))
                    is_compliant = self._parse_bool(row.get("is_compliant"))
                    rule_ids = self._parse_rule_ids(row.get("rule_ids"))

                    tx = Transaction(
                        tx_id=row["tx_id"],
                        account_id=row["account_id"],
                        amount=amount if amount is not None else Decimal(
                            "0.0"),
                        currency=row.get("currency") or "USD",
                        date=row.get("date") or "1970-01-01",
                        status=row.get("status") or None,
                        is_compliant=is_compliant,
                        rule_ids=rule_ids,
                    )
                    self.add_transaction(tx)

        if rules_path.exists():
            with rules_path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rule_id = row["rule_id"].strip()
                    r_uri = self.rule_uri(rule_id)
                    self.graph.add((r_uri, RDF.type, self.EX.ComplianceRule))
                    desc = (row.get("description") or "").strip()
                    if desc:
                        self.graph.set(
                            (r_uri, self.EX.description, Literal(desc)))
                    sev = (row.get("severity") or "").strip()
                    if sev:
                        self.graph.set((r_uri, self.EX.severity, Literal(sev)))

        if tx_rules_path.exists():
            with tx_rules_path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    tx_id = row["tx_id"].strip()
                    rule_id = row["rule_id"].strip()
                    rel = (row.get("relation") or "").strip().lower()

                    t_uri = self.tx_uri(tx_id)
                    r_uri = self.rule_uri(rule_id)
                    self.graph.add((r_uri, RDF.type, self.EX.ComplianceRule))

                    if rel == "compliant":
                        self.graph.add((t_uri, self.EX.isCompliantWith, r_uri))
                    elif rel == "violates":
                        self.graph.add((t_uri, self.EX.violatesRule, r_uri))

    # ---------------------------------------------------------- CSV helpers

    @staticmethod
    def _parse_decimal(value: Optional[str]) -> Optional[Decimal]:
        if value is None or value == "":
            return None
        try:
            return Decimal(value)
        except (InvalidOperation, ValueError):
            return None

    @staticmethod
    def _parse_bool(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
        return None

    @staticmethod
    def _parse_rule_ids(value: Optional[str]) -> Optional[List[str]]:
        if not value:
            return None
        # Expect comma-separated rule IDs, e.g. "KYC,AML_THRESHOLD"
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts or None

    # -------------------------------------------------------------- Serialization

    def save_turtle(self, path: Path) -> None:
        """Serialize the graph to Turtle format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(path), format="turtle")

    # ----------------------------------------------------------- Demo / seeding

    def seed_demo_data(self) -> None:
        """
        Seed the KG with a small, deterministic demo dataset that can be used in
        notebooks and evaluation scenarios.
        """
        # One client with two accounts and a few transactions
        client = Client(client_id="A", name="Client A", risk_level="medium")
        self.add_client(client)

        account1 = Account(account_id="A1", client_id="A",
                           account_type="checking", status="active")
        account2 = Account(account_id="A2", client_id="A",
                           account_type="savings", status="active")
        self.add_account(account1)
        self.add_account(account2)

        tx1 = Transaction(
            tx_id="T001",
            account_id="A1",
            amount=Decimal("9500.00"),
            currency="USD",
            date="2024-05-10",
            status="completed",
            is_compliant=True,
            rule_ids=["KYC"],
        )

        tx2 = Transaction(
            tx_id="T002",
            account_id="A1",
            amount=Decimal("15000.00"),
            currency="USD",
            date="2024-05-12",
            status="completed",
            is_compliant=False,
            rule_ids=["KYC", "AML_THRESHOLD"],
        )

        tx3 = Transaction(
            tx_id="T003",
            account_id="A2",
            amount=Decimal("500.00"),
            currency="EUR",
            date="2024-05-15",
            status="completed",
            is_compliant=True,
            rule_ids=["KYC"],
        )

        for tx in (tx1, tx2, tx3):
            self.add_transaction(tx)


# --------------------------------------------------------------------------- #
# Optional: quick manual test when running this file directly
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    kg = FinancialKG()
    kg.seed_demo_data()

    print("=== Transactions for client A ===")
    txs = kg.get_transactions_for_client("A")
    for tx in txs:
        print(tx)

    print("\n=== Compliance explanation for T002 ===")
    print(kg.explain_transaction_compliance("T002"))

    # Save to a demo TTL file
    kg.save_turtle(Path("financial_kg_demo.ttl"))
    print("\nGraph saved to financial_kg_demo.ttl")
