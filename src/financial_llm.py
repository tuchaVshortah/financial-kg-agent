# src/financial_llm.py

import os
import time
import json
from typing import Optional, Dict, Any, List, Literal, Tuple

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

JsonAnswerMode = Literal["qa", "evaluation"]


class FinancialLLM:
    """
    Wrapper for interacting with OpenAI's GPT-based models.

    Features:
    - Standardized system prompt for financial reasoning.
    - Strong grounding in provided facts (KG output).
    - Optional JSON-structured responses for evaluation and logging.
    - Simple retry logic for robustness.

    Backwards compatible:
    - `ask(...) -> str` remains the main entry point used by the controller.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Default system prompt emphasizes:
        # - Use of provided facts
        # - No hallucination
        # - Clear reasoning
        self.system_prompt = (
            system_prompt
            or (
                "You are a careful financial reasoning assistant.\n"
                "- You base your answers ONLY on the verified facts and rules that are provided to you.\n"
                "- If the facts are insufficient to answer reliably, you MUST say so explicitly.\n"
                "- You do not invent transactions, clients, or regulations.\n"
                "- When appropriate, you briefly explain your reasoning in clear, concise language."
            )
        )

    # --------------------------------------------------------------------- Core

    def _build_messages(
        self,
        user_message: str,
        context_facts: Optional[str],
        extra_instructions: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build a standard message list for the Chat Completions API.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        if context_facts:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Here are VERIFIED facts and rules from the financial knowledge graph. "
                        "You MUST treat these as ground truth and you MUST NOT contradict them.\n\n"
                        f"{context_facts}"
                    ),
                }
            )

        if extra_instructions:
            messages.append({"role": "system", "content": extra_instructions})

        messages.append({"role": "user", "content": user_message})
        return messages

    def _chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Low-level call to the OpenAI Chat Completions API with retry logic.
        Returns the raw model output string (content).
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=(
                        temperature if temperature is not None else self.temperature
                    ),
                    response_format={"type": "json_object"} if json_mode else None,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM request failed after {self.max_retries} retries: {e}")
                # Simple exponential-ish backoff
                time.sleep(1.5 * (attempt + 1))

        # Should never reach here
        return ""

    # ----------------------------------------------------------------- Public API

    def ask(
        self,
        user_message: str,
        context_facts: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Backwards-compatible main method used by the controller.

        If json_mode=False (default):
            - Returns a natural language answer that includes short reasoning.

        If json_mode=True:
            - Returns a JSON string conforming to the schema of `ask_json(...)`.
              (You can later parse it with `json.loads` if desired.)
        """
        if not json_mode:
            # Plain text answer with a short reasoning section
            extra = (
                "Answer the user's question in a concise way. "
                "If appropriate, include a brief explanation section starting with "
                "'Reasoning:' on a new line.\n"
                "If the facts are insufficient, explicitly say so."
            )
            messages = self._build_messages(user_message, context_facts, extra)
            return self._chat(messages, json_mode=False, temperature=temperature)

        # JSON mode: delegate to the more structured variant and re-dump to string
        data = self.ask_json(
            user_message=user_message,
            context_facts=context_facts,
            mode="qa",
            temperature=temperature,
        )
        return json.dumps(data, ensure_ascii=False, indent=2)

    def ask_json(
        self,
        user_message: str,
        context_facts: Optional[str] = None,
        mode: JsonAnswerMode = "qa",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Structured JSON answer for downstream evaluation / logging.

        Parameters
        ----------
        user_message : str
            The user's query.
        context_facts : Optional[str]
            Facts injected from the knowledge graph.
        mode : {"qa", "evaluation"}
            - "qa": JSON focused on the final answer and reasoning.
            - "evaluation": includes fields for correctness / confidence flags.
        temperature : Optional[float]
            Overrides default temperature.

        Returns
        -------
        dict with at least:
            {
              "answer": str,
              "reasoning": str,
              "used_facts": [str],
              "insufficient_facts": bool,
              "uncertainty": "low" | "medium" | "high"
            }
        """
        # Describe the target JSON format in the prompt
        schema_instruction = (
            "You MUST respond ONLY with a single valid JSON object (no extra text).\n"
            "The JSON MUST have the following keys:\n"
            '  - "answer": a short natural language answer string.\n'
            '  - "reasoning": a brief explanation of how you used the facts.\n'
            '  - "used_facts": a list of short strings referencing which facts were relevant.\n'
            '  - "insufficient_facts": a boolean indicating whether the facts were insufficient.\n'
            '  - "uncertainty": one of "low", "medium", or "high" representing your confidence.\n'
        )

        if mode == "evaluation":
            schema_instruction += (
                'You are in EVALUATION mode. Be especially careful to mark '
                '"insufficient_facts": true if key facts are missing or ambiguous.\n'
            )

        messages = self._build_messages(
            user_message=user_message,
            context_facts=context_facts,
            extra_instructions=schema_instruction,
        )

        raw = self._chat(messages, json_mode=True, temperature=temperature)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: wrap raw content in a generic structure
            return {
                "answer": raw,
                "reasoning": "Failed to parse JSON; returning raw model output.",
                "used_facts": [],
                "insufficient_facts": True,
                "uncertainty": "high",
            }

        # Ensure all keys exist with default values if the model omitted some
        defaults: Dict[str, Any] = {
            "answer": "",
            "reasoning": "",
            "used_facts": [],
            "insufficient_facts": False,
            "uncertainty": "medium",
        }
        for k, v in defaults.items():
            if k not in parsed:
                parsed[k] = v

        # Normalise types a bit
        if not isinstance(parsed.get("used_facts"), list):
            parsed["used_facts"] = [str(parsed.get("used_facts"))]

        if not isinstance(parsed.get("insufficient_facts"), bool):
            parsed["insufficient_facts"] = bool(parsed["insufficient_facts"])

        if parsed.get("uncertainty") not in ("low", "medium", "high"):
            parsed["uncertainty"] = "medium"

        return parsed

    def ask_compliance_json(
        self,
        user_message: str,
        context_facts: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Ask the model for a compliance decision in structured JSON form.

        The model is instructed to return:
        {
          "is_compliant": true/false,
          "explanation": "..."
        }

        Returns
        -------
        (parsed_json, raw_response)
        """
        prompt = (
            "You are a financial compliance assistant.\n"
            "You will be given verified facts about a single transaction.\n"
            "Based ONLY on these facts, decide if the transaction is compliant.\n\n"
            "Respond in strictly valid JSON with the following fields:\n"
            "{\n"
            '  "is_compliant": true or false,\n'
            '  "explanation": "short natural language justification"\n'
            "}\n"
            "Do not include any additional keys or text outside the JSON object."
        )

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "system",
                "content": f"Facts:\n{context_facts}",
            },
            {"role": "user", "content": user_message},
        ]

        # Retry loop, similar to ask()
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                raw = response.choices[0].message.content
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                return parsed, raw
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM compliance JSON request failed: {e}")
                time.sleep(1.5)

        return None, ""
