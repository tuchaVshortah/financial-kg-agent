# src/demo_scenarios.py
from pathlib import Path
from .financial_llm import FinancialLLM
from .financial_kg import FinancialKG
from .retriever import DataRetriever
from .controller import LLMController

def run_demo():
    kg = FinancialKG()
    kg.add_mock_data()
    kg.save_turtle(Path("financial_kg.ttl"))

    retriever = DataRetriever(kg)
    llm = FinancialLLM()
    controller = LLMController(llm=llm, kg=kg, retriever=retriever)

    query = "Was the last transaction by client A compliant with KYC rules?"
    entry = controller.handle_query(query)

    print("User query:")
    print(query)
    print("\nFacts:")
    print(entry.facts_text)
    print("\nLLM response:")
    print(entry.llm_response)

if __name__ == "__main__":
    run_demo()
