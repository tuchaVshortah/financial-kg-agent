# src/financial_llm.py

import os
import time
from typing import Optional, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FinancialLLM:
    """
    Wrapper for interacting with OpenAI's GPT-based models.
    Provides:
    - Standardized system prompt for financial reasoning
    - Retry logic for robustness
    - Optional structured (JSON) responses
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        system_prompt: str = None
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        self.system_prompt = (
            system_prompt
            or "You are a financial reasoning assistant. "
               "You answer strictly based on provided facts or instructions. "
               "If information is missing, state that it is not available."
        )

    def ask(
        self,
        user_message: str,
        context_facts: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Main method to send a prompt to ChatGPT.

        Parameters
        ----------
        user_message : str
            The user's query.
        context_facts : Optional[str]
            Facts injected from the knowledge graph.
        json_mode : bool
            If True, requests model to return structured JSON.
        temperature : Optional[float]
            Overrides default temperature.

        Returns
        -------
        str : Model output
        """

        messages = [{"role": "system", "content": self.system_prompt}]

        if context_facts:
            messages.append({
                "role": "system",
                "content": f"Here are verified facts you MUST use:\n{context_facts}"
            })

        messages.append({"role": "user", "content": user_message})

        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature if temperature is not None else self.temperature,
                    response_format={"type": "json_object"} if json_mode else None,
                    messages=messages,
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM request failed after retries: {e}")
                time.sleep(1.5)  # Back-off before retrying

        return None  # Should not reach here
