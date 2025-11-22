# src/financial_llm.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .config import OPENAI_API_KEY

@dataclass
class LLMMessage:
    role: str   # "system", "user", "assistant"
    content: str

class FinancialLLM:
    """
    Thin wrapper around ChatGPT for financial reasoning with KG grounding.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def chat(self, messages: List[LLMMessage]) -> str:
        """
        Send a list of messages to the model and return the assistant's reply text.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[m.__dict__ for m in messages],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def ask_with_facts(self, user_query: str, facts: str) -> str:
        """
        Helper: answer a user query given a textual description of KG facts.
        """
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a cautious financial AI agent. "
                    "You MUST base your answer ONLY on the facts provided. "
                    "If the facts are insufficient, say so explicitly."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"User question:\n{user_query}\n\n"
                    f"Relevant facts from the knowledge graph:\n{facts}\n\n"
                    "Answer the question and explain your reasoning briefly."
                ),
            ),
        ]
        return self.chat(messages)
