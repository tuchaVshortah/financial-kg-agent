# src/controller.py
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any, Dict

from .financial_llm import FinancialLLM
from .financial_kg import FinancialKG
from .retriever import DataRetriever

@dataclass
class TraceEntry:
    user_query: str
    facts_text: str
    llm_response: str

class LLMController:
    """
    Orchestrates the flow:
    user → retriever → KG → LLM → answer + trace log.
    """

    def __init__(
        self,
        llm: FinancialLLM,
        kg: FinancialKG,
        retriever: DataRetriever,
        trace_log_path: Path | None = None,
    ):
        self.llm = llm
        self.kg = kg
        self.retriever = retriever
        self.trace_log_path = trace_log_path or Path("trace_log.jsonl")

    def handle_query(self, user_query: str) -> TraceEntry:
        facts = self.retriever.retrieve_for_query(user_query)
        facts_text = self.kg.format_facts_for_llm(facts)

        llm_response = self.llm.ask_with_facts(user_query, facts_text)

        entry = TraceEntry(
            user_query=user_query,
            facts_text=facts_text,
            llm_response=llm_response,
        )
        self._append_trace(entry)
        return entry

    def _append_trace(self, entry: TraceEntry) -> None:
        record: Dict[str, Any] = asdict(entry)
        with self.trace_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
