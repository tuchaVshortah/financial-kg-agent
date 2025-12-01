# src/log_utils.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


@dataclass
class LogEntry:
    """
    Structured representation of a single run log entry.

    Fields are kept generic so the log format can evolve without breaking this
    helper – unknown keys are preserved in `extra`.
    """
    timestamp: datetime
    scenario: Optional[str] = None
    client_id: Optional[str] = None
    tx_id: Optional[str] = None
    user_question: Optional[str] = None
    facts_text: Optional[str] = None
    llm_response: Optional[str] = None
    extra: Dict[str, Any] = None


class LogReader:
    """
    Utility for reading and summarizing JSONL logs.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------ loading

    def load_entries(self) -> List[LogEntry]:
        """Load all log entries from the JSONL file."""
        entries: List[LogEntry] = []

        if not self.log_path.exists():
            return entries

        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines (or you could raise)
                    continue

                ts_raw = raw.pop("timestamp", None)
                ts: datetime
                if isinstance(ts_raw, str):
                    try:
                        # Handles "...Z" suffix produced by isoformat() + "Z"
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    except Exception:
                        ts = datetime.now(timezone.utc).isoformat()
                else:
                    ts = datetime.now(timezone.utc).isoformat()

                entry = LogEntry(
                    timestamp=ts,
                    scenario=raw.pop("scenario", None),
                    client_id=raw.pop("client_id", None),
                    tx_id=raw.pop("tx_id", None),
                    user_question=raw.pop("user_question", None),
                    facts_text=raw.pop("facts_text", None),
                    llm_response=raw.pop("llm_response", None),
                    extra=raw,  # anything left over
                )
                entries.append(entry)

        return entries

    # ------------------------------------------------------------------ helpers

    def get_last_entry(self) -> Optional[LogEntry]:
        """Return the most recent entry, if any."""
        entries = self.load_entries()
        if not entries:
            return None
        return max(entries, key=lambda e: e.timestamp)

    def get_last_entry_for_scenario(self, scenario: str) -> Optional[LogEntry]:
        """Return the most recent entry for a given scenario, if any."""
        entries = [
            e for e in self.load_entries()
            if (e.scenario or "").lower() == scenario.lower()
        ]
        if not entries:
            return None
        return max(entries, key=lambda e: e.timestamp)

    def summarize(self) -> Dict[str, Any]:
        """
        Return a small summary:
          - total number of entries
          - per-scenario counts
          - timestamp of first / last entry
        """
        entries = self.load_entries()
        if not entries:
            return {
                "total_entries": 0,
                "per_scenario": {},
                "first_timestamp": None,
                "last_timestamp": None,
            }

        per_scenario: Dict[str, int] = {}
        for e in entries:
            key = (e.scenario or "unknown").lower()
            per_scenario[key] = per_scenario.get(key, 0) + 1

        first_ts = min(e.timestamp for e in entries)
        last_ts = max(e.timestamp for e in entries)

        return {
            "total_entries": len(entries),
            "per_scenario": per_scenario,
            "first_timestamp": first_ts.isoformat(),
            "last_timestamp": last_ts.isoformat(),
        }

    # ------------------------------------------------------------------ printing

    def print_last_entry(self, scenario: Optional[str] = None) -> None:
        """
        Pretty-print the last log entry (optionally filtered by scenario).
        """
        if scenario:
            entry = self.get_last_entry_for_scenario(scenario)
        else:
            entry = self.get_last_entry()

        if not entry:
            print("No log entries found.")
            return

        print("=== Last log entry ===")
        print(f"Timestamp : {entry.timestamp.isoformat()}")
        if entry.scenario:
            print(f"Scenario  : {entry.scenario}")
        if entry.client_id:
            print(f"Client ID : {entry.client_id}")
        if entry.tx_id:
            print(f"Tx ID     : {entry.tx_id}")
        if entry.user_question:
            print("\nUser question:")
            print(entry.user_question)
        if entry.facts_text:
            print("\nFacts passed to LLM:")
            print(entry.facts_text)
        if entry.llm_response:
            print("\nLLM response:")
            print(entry.llm_response)
        if entry.extra:
            print("\nExtra fields:")
            for k, v in entry.extra.items():
                print(f"- {k}: {v}")

    def print_summary(self) -> None:
        """Pretty-print a small summary of the log file."""
        summary = self.summarize()
        print("=== Log summary ===")
        print(f"Total entries   : {summary['total_entries']}")
        print(f"First timestamp : {summary['first_timestamp']}")
        print(f"Last timestamp  : {summary['last_timestamp']}")
        print("\nEntries per scenario:")
        for scenario, count in summary["per_scenario"].items():
            print(f"- {scenario}: {count}")
