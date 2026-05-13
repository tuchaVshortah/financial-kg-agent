"""
Four-arm ablation runner.

Iterates the synthetic dataset, builds the per-transaction context for each
of the four arms defined in `src.eval_arms`, asks the LLM for a JSON
compliance decision, and writes a per-tx results CSV plus a summary JSON
of accuracy / confusion-matrix metrics per arm.

Quick smoke test (no API calls, no spend):
    python -m src.eval_runner --dry-run --limit 20 \
        --data-dir data --out runs/smoke

Real run (gpt-4o-mini, costs API credits):
    python -m src.eval_runner --data-dir data --out runs/full \
        --arms A,B,C,D --log-file runs/full/per_tx.jsonl

Stop before running on the full 500 tx until the harness has been
smoke-tested and the design is signed off.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from .eval_arms import Arm, TxContext, get_arms
from .financial_kg import FinancialKG
from .retriever import FinancialRetriever


USER_MESSAGE = (
    "Decide whether the transaction described above is compliant. "
    "Respond ONLY with the requested JSON fields."
)


# --------------------------------------------------------------------------- #
# CSV reading helpers
# --------------------------------------------------------------------------- #


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_bool(s: Optional[str]) -> bool:
    return (s or "").strip().lower() in {"true", "1", "yes", "y"}


def load_dataset(data_dir: Path) -> Tuple[
    Dict[str, dict], Dict[str, dict], Dict[str, dict], Dict[str, dict], Dict[str, dict],
]:
    """
    Load the five entity CSVs into per-id dicts.

    Returns (clients, accounts, counterparties, transactions, rules).
    Each dict maps the entity id -> a dict of its columns.
    """
    raw_clients = _read_csv(data_dir / "clients.csv")
    raw_accounts = _read_csv(data_dir / "accounts.csv")
    raw_cps = _read_csv(data_dir / "counterparties.csv")
    raw_txs = _read_csv(data_dir / "transactions.csv")
    raw_rules = _read_csv(data_dir / "rules.csv")

    clients = {r["client_id"]: r for r in raw_clients}
    accounts = {r["account_id"]: r for r in raw_accounts}
    counterparties = {r["counterparty_id"]: r for r in raw_cps}
    transactions = {r["tx_id"]: r for r in raw_txs}
    rules = {r["rule_id"]: r for r in raw_rules}

    return clients, accounts, counterparties, transactions, rules


def load_rule_definitions(data_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Reconstruct the natural-language rule definitions used by arm B+ from
    generation_metadata.json. Falls back to the rules.csv `description`
    column if the metadata file is absent.
    """
    meta_path = data_dir / "generation_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rd = meta.get("rule_definitions") or {}
        if rd:
            return rd

    # Fallback: build from rules.csv
    out: Dict[str, Dict[str, str]] = {}
    for r in _read_csv(data_dir / "rules.csv"):
        out[r["rule_id"]] = {
            "description": r.get("description", ""),
            "category": r.get("category", ""),
            "severity": r.get("severity", ""),
        }
    return out


# --------------------------------------------------------------------------- #
# Per-tx context builder
# --------------------------------------------------------------------------- #


def build_tx_context(
    tx_id: str,
    txs: Dict[str, dict],
    accounts: Dict[str, dict],
    clients: Dict[str, dict],
    counterparties: Dict[str, dict],
    rule_definitions: Dict[str, Dict[str, str]],
    kg: FinancialKG,
) -> TxContext:
    tx = txs[tx_id]
    acct = accounts[tx["account_id"]]
    client = clients[acct["client_id"]]
    cp_id = tx.get("counterparty_id", "")
    cp = counterparties.get(cp_id, {})

    # Pull KG-derived per-rule relations (the heart of arm C / D)
    kg_info = kg.explain_transaction_compliance(tx_id)
    relations: List[Dict[str, str]] = []
    for r in kg_info.get("rules", []):
        rule_uri = r.get("rule_uri", "")
        rule_id = rule_uri.split("#")[-1].replace("Rule_", "")
        relations.append({"rule_id": rule_id, "relation": r.get("relation", "")})

    # Pull ground-truth label from CSV (always available in synthetic data)
    ground_truth = _parse_bool(tx.get("is_compliant"))

    return TxContext(
        tx_id=tx_id,
        amount=float(tx["amount"]),
        currency=tx["currency"],
        amount_usd=float(tx["amount_usd"]),
        date=tx["date"],
        status=tx["status"],
        tx_type=tx.get("tx_type", ""),
        description=tx.get("description", ""),
        counterparty_id=cp_id,
        counterparty_name=cp.get("name", "?"),
        counterparty_country=tx.get("counterparty_country", cp.get("country", "?")),
        counterparty_on_sanctions_list=_parse_bool(cp.get("on_sanctions_list")),
        client_id=client["client_id"],
        client_name=client.get("name", "?"),
        client_risk_level=client.get("risk_level", "?"),
        client_kyc_status=client.get("kyc_status", "?"),
        client_kyc_expiry_date=client.get("kyc_expiry_date", "?"),
        client_country=client.get("country", "?"),
        client_pep_flag=_parse_bool(client.get("pep_flag")),
        account_id=acct["account_id"],
        account_type=acct.get("account_type", "?"),
        account_default_currency=acct.get("default_currency", tx["currency"]),
        account_open_date=acct.get("open_date", "?"),
        account_last_active_date=acct.get("last_active_date", "?"),
        rule_definitions=rule_definitions,
        kg_relations=relations,
        ground_truth_is_compliant=ground_truth,
        # `scenario_rule` is the authoritative tag, written by the generator.
        # Falls back to first rule_id for datasets predating the column
        # (rule_ids are sorted alphabetically post-cross-rule so this is
        # only safe on pre-cross-rule data).
        scenario_rule_id=(
            tx.get("scenario_rule")
            or (tx.get("rule_ids") or "").split(",")[0]
            or "NORMAL"
        ),
    )


# --------------------------------------------------------------------------- #
# LLM interface
# --------------------------------------------------------------------------- #


@runtime_checkable
class _LLMLike(Protocol):
    """
    Minimal structural interface the runner needs from an LLM client:
        ask_compliance_json(user_message, context_facts) -> (parsed_dict, raw_str)

    Both `FinancialLLM` (real OpenAI client) and `DryRunLLM` (offline
    stand-in) satisfy this by shape — no inheritance required. Marked
    `runtime_checkable` so `isinstance(x, _LLMLike)` is also valid.
    """

    def ask_compliance_json(
        self, user_message: str, context_facts: str
    ) -> Tuple[Optional[Dict[str, Any]], str]: ...


class DryRunLLM:
    """
    Offline stand-in for `FinancialLLM`. Returns predictions derived
    *from the facts string itself*, so the runner exercises end-to-end
    without API calls:

      - If the facts contain "GROUND-TRUTH FLAG: ... = true|false"  → use it.
      - Else if a "KG RELATIONS" block lists 'violates' → predict false.
      - Else if a "KG RELATIONS" block lists 'is compliant with' → predict true.
      - Else: deterministic seeded coin flip.

    This is enough to verify (a) the runner CSV format, (b) that arm-D facts
    actually contain the ground-truth flag, (c) that arm-C facts surface
    the KG relations, and (d) that arm-A/B facts do NOT leak either.

    It is NOT a model of LLM behavior. Real benchmarking requires
    FinancialLLM with an OpenAI key.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def ask_compliance_json(
        self, user_message: str, context_facts: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        facts = context_facts or ""
        # 1) Ground-truth flag (arm D)
        if "GROUND-TRUTH FLAG: this transaction is_compliant = true" in facts:
            return {"is_compliant": True, "explanation": "dry-run: read GT flag"}, "{}"
        if "GROUND-TRUTH FLAG: this transaction is_compliant = false" in facts:
            return {"is_compliant": False, "explanation": "dry-run: read GT flag"}, "{}"

        # 2) KG relations (arm C). Any "FIRES" line is a decisive
        # non-compliant signal; otherwise, if every listed rule explicitly
        # "does NOT fire", treat as compliant.
        if "KG RELATIONS" in facts:
            if "rule FIRES on this transaction" in facts:
                return {"is_compliant": False,
                        "explanation": "dry-run: KG says a rule FIRES"}, "{}"
            if "rule does NOT fire on this transaction" in facts:
                return {"is_compliant": True,
                        "explanation": "dry-run: KG says no rule fires"}, "{}"

        # 3) Arm A / B fallback: seeded coin flip
        pred = self._rng.random() < 0.5
        return {"is_compliant": pred,
                "explanation": "dry-run: coin flip"}, "{}"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class ArmMetrics:
    arm: str
    n_total: int = 0
    n_scored: int = 0   # excludes unparseable / unknown predictions
    n_unknown: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    per_rule_acc: Dict[str, Dict[str, int]] = None  # rule -> {correct, total}

    def __post_init__(self) -> None:
        if self.per_rule_acc is None:
            self.per_rule_acc = defaultdict(lambda: {"correct": 0, "total": 0})

    def record(self, gt: bool, pred: Optional[bool], scenario_rule: str) -> None:
        self.n_total += 1
        if pred is None:
            self.n_unknown += 1
            return
        self.n_scored += 1
        if gt and pred:
            self.tp += 1
        elif (not gt) and pred:
            self.fp += 1
        elif (not gt) and (not pred):
            self.tn += 1
        elif gt and (not pred):
            self.fn += 1
        bucket = self.per_rule_acc[scenario_rule or "UNKNOWN"]
        bucket["total"] += 1
        if pred == gt:
            bucket["correct"] += 1

    def accuracy(self) -> Optional[float]:
        if self.n_scored == 0:
            return None
        return (self.tp + self.tn) / self.n_scored

    def precision(self) -> Optional[float]:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    def recall(self) -> Optional[float]:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    def f1(self) -> Optional[float]:
        p, r = self.precision(), self.recall()
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_unknown": self.n_unknown,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "per_rule_accuracy": {
                rule: {
                    "correct": v["correct"], "total": v["total"],
                    "accuracy": (v["correct"] / v["total"]) if v["total"] else None,
                }
                for rule, v in sorted(self.per_rule_acc.items())
            },
        }


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #


class BudgetExceeded(RuntimeError):
    """Raised when the cumulative LLM cost crosses the configured cap."""


def run(
    data_dir: Path,
    out_dir: Path,
    arms: List[Arm],
    llm: _LLMLike,
    limit: Optional[int],
    seed: int,
    log_file: Optional[Path],
    verbose: bool,
    budget_cap_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute the full ablation matrix and write results to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    clients, accounts, counterparties, txs, _rules_csv = load_dataset(data_dir)
    rule_definitions = load_rule_definitions(data_dir)

    # Build the KG from the same CSVs so arm C/D get the relation triples.
    kg = FinancialKG()
    kg.load_from_csv(data_dir)

    # Deterministic tx ordering
    tx_ids = sorted(txs.keys())
    rng = random.Random(seed)
    rng.shuffle(tx_ids)
    if limit is not None:
        tx_ids = tx_ids[:limit]

    # Per-arm metrics + per-tx CSV
    metrics: Dict[str, ArmMetrics] = {a.name: ArmMetrics(arm=a.name) for a in arms}
    # Per-arm token / cost aggregates. Empty if the LLM doesn't expose usage_log.
    arm_usage: Dict[str, Dict[str, float]] = {
        a.name: {"prompt_tokens": 0, "completion_tokens": 0,
                 "total_tokens": 0, "cost_usd": 0.0, "n_calls": 0}
        for a in arms
    }
    budget_hit = False

    results_path = out_dir / "results.csv"
    log_fp = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_file.open("a", encoding="utf-8")

    with results_path.open("w", encoding="utf-8", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerow([
            "tx_id", "scenario_rule", "ground_truth", "arm",
            "predicted", "correct", "explanation",
        ])

        usage_log = getattr(llm, "usage_log", None)
        for i, tx_id in enumerate(tx_ids, start=1):
            if budget_hit:
                break
            ctx = build_tx_context(
                tx_id, txs, accounts, clients, counterparties,
                rule_definitions, kg,
            )
            gt = ctx.ground_truth_is_compliant
            for arm in arms:
                facts = arm.build_facts(ctx)

                # Snapshot usage-log length BEFORE the call so we can attribute
                # this call's usage to the current arm.
                usage_before = len(usage_log) if usage_log is not None else 0
                parsed_failure = False
                try:
                    parsed, raw = llm.ask_compliance_json(USER_MESSAGE, facts)
                except Exception as e:
                    parsed, raw = None, f"<<llm-error: {e!r}>>"
                    parsed_failure = True

                # Attribute newly-appended usage entries to this arm
                call_cost = 0.0
                call_tokens = 0
                if usage_log is not None:
                    for entry in usage_log[usage_before:]:
                        arm_usage[arm.name]["prompt_tokens"] += entry.get(
                            "prompt_tokens", 0)
                        arm_usage[arm.name]["completion_tokens"] += entry.get(
                            "completion_tokens", 0)
                        arm_usage[arm.name]["total_tokens"] += entry.get(
                            "total_tokens", 0)
                        c = entry.get("cost_usd", 0.0)
                        arm_usage[arm.name]["cost_usd"] += c
                        call_cost += c
                        call_tokens += entry.get("total_tokens", 0)
                    arm_usage[arm.name]["n_calls"] += 1

                pred = None
                explanation = ""
                json_decode_failed = False
                if parsed is None:
                    json_decode_failed = not parsed_failure
                elif isinstance(parsed, dict):
                    val = parsed.get("is_compliant")
                    if isinstance(val, bool):
                        pred = val
                    explanation = parsed.get("explanation", "") or ""

                correct: Optional[bool] = None
                if pred is not None and gt is not None:
                    correct = (pred == gt)

                writer.writerow([
                    tx_id, ctx.scenario_rule_id, str(gt).lower(), arm.name,
                    "" if pred is None else str(pred).lower(),
                    "" if correct is None else str(correct).lower(),
                    explanation.replace("\n", " ").strip(),
                ])

                if log_fp is not None:
                    log_fp.write(json.dumps({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "tx_id": tx_id,
                        "scenario_rule": ctx.scenario_rule_id,
                        "ground_truth": gt,
                        "arm": arm.name,
                        "predicted": pred,
                        "correct": correct,
                        "explanation": explanation,
                        "raw": raw,
                        "call_tokens": call_tokens,
                        "call_cost_usd": call_cost,
                        "json_decode_failed": json_decode_failed,
                        "api_error": parsed_failure,
                    }) + "\n")

                if gt is not None:
                    metrics[arm.name].record(gt, pred, ctx.scenario_rule_id or "UNKNOWN")

                # Budget cap check after every call
                if budget_cap_usd is not None and usage_log is not None:
                    cumulative = sum(e.get("cost_usd", 0.0) for e in usage_log)
                    if cumulative > budget_cap_usd:
                        print(
                            f"  !! BUDGET CAP HIT: cumulative ${cumulative:.4f} "
                            f"> cap ${budget_cap_usd:.4f}. Stopping.",
                            file=sys.stderr,
                        )
                        budget_hit = True
                        break

            if verbose and (i % 5 == 0 or i == len(tx_ids)):
                accs = " ".join(
                    f"{a.name}={(metrics[a.name].accuracy() or 0):.3f}"
                    for a in arms
                )
                cum_cost = (sum(e.get("cost_usd", 0.0) for e in usage_log)
                            if usage_log is not None else 0.0)
                print(f"  [{i}/{len(tx_ids)}] accs {accs}  "
                      f"cum_cost=${cum_cost:.4f}", file=sys.stderr)

    if log_fp is not None:
        log_fp.close()

    total_cost = (
        sum(e.get("cost_usd", 0.0) for e in (getattr(llm, "usage_log", None) or []))
    )
    total_tokens = (
        sum(e.get("total_tokens", 0) for e in (getattr(llm, "usage_log", None) or []))
    )
    summary = {
        "data_dir": str(data_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_transactions": len(tx_ids),
        "n_transactions_completed": len({
            tx_id for tx_id in tx_ids
        }) if not budget_hit else metrics[arms[0].name].n_total,
        "arms": [a.name for a in arms],
        "metrics": {a.name: metrics[a.name].to_dict() for a in arms},
        "usage": {
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "per_arm": arm_usage,
            "budget_cap_usd": budget_cap_usd,
            "budget_hit": budget_hit,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Four-arm ablation runner")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory containing the generated CSVs")
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("runs/last"),
                   help="Directory to write results.csv + summary.json")
    p.add_argument("--arms", default="A,B,C,D",
                   help="Comma-separated subset of arms to run (default ABCD)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of transactions evaluated")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for tx-order shuffling (default 42)")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Optional per-tx JSONL audit log")
    p.add_argument("--dry-run", action="store_true",
                   help="Use DryRunLLM (no API calls, no spend)")
    p.add_argument("--verbose", action="store_true",
                   help="Print running accuracy to stderr")
    p.add_argument("--budget-cap-usd", type=float, default=None,
                   help="Abort the run if cumulative LLM cost crosses this cap")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    arms = get_arms(args.arms.split(","))

    if args.dry_run:
        llm: _LLMLike = DryRunLLM(seed=args.seed)
        mode = "DRY-RUN (no API calls)"
    else:
        # Lazy import to avoid requiring an API key for `--dry-run`.
        from .financial_llm import FinancialLLM
        llm = FinancialLLM()
        mode = f"LIVE ({llm.model})"

    print(f"=== eval_runner: {mode} ===")
    print(f"  data_dir : {args.data_dir.resolve()}")
    print(f"  out_dir  : {args.out_dir.resolve()}")
    print(f"  arms     : {','.join(a.name for a in arms)}")
    print(f"  limit    : {args.limit if args.limit is not None else 'all'}")
    print()

    t0 = time.time()
    summary = run(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        arms=arms,
        llm=llm,
        limit=args.limit,
        seed=args.seed,
        log_file=args.log_file,
        verbose=args.verbose,
        budget_cap_usd=args.budget_cap_usd,
    )
    elapsed = time.time() - t0

    print(f"\n=== summary ({elapsed:.1f}s, n={summary['n_transactions']}) ===")
    for arm_name, m in summary["metrics"].items():
        acc = m["accuracy"]
        f1 = m["f1"]
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        f1_s = f"{f1:.3f}" if f1 is not None else "n/a"
        print(f"  arm {arm_name}: acc={acc_s}  f1={f1_s}  "
              f"tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}  "
              f"unknown={m['n_unknown']}")
    u = summary.get("usage", {})
    if u.get("total_cost_usd", 0) > 0:
        print(f"  ---")
        print(f"  total tokens : {u['total_tokens']}")
        print(f"  total cost   : ${u['total_cost_usd']:.4f}")
        if u.get("budget_hit"):
            print(f"  !! budget cap ${u['budget_cap_usd']:.4f} was hit; run aborted")


if __name__ == "__main__":
    main()
