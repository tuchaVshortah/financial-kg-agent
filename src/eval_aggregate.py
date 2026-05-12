"""
Aggregate the per-repeat summary.json + results.csv files produced by
`src.eval_runner` into a single cross-repeat report.

For each arm we report:
  * mean / std of accuracy, precision, recall, F1 across repeats
  * per-rule mean accuracy across repeats
  * a per-tx **consistency** metric: for each (arm, tx_id) tuple, count
    the fraction of repeats whose prediction equals the modal prediction
    for that tuple; average across tx.

Usage:
    python -m src.eval_aggregate runs/full/r1 runs/full/r2 runs/full/r3 \
        --out runs/full/aggregated_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _mean_std(xs: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Mean / std over non-None values; returns (None, None) if empty."""
    vals = [x for x in xs if x is not None]
    if not vals:
        return None, None
    m = statistics.fmean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _round(x: Optional[float], n: int = 4) -> Optional[float]:
    return None if x is None else round(x, n)


def load_summaries(repeat_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in repeat_dirs:
        sp = d / "summary.json"
        if not sp.exists():
            raise FileNotFoundError(f"missing {sp}")
        out.append(json.loads(sp.read_text(encoding="utf-8")))
    return out


def load_results(repeat_dirs: Sequence[Path]) -> List[List[Dict[str, str]]]:
    out: List[List[Dict[str, str]]] = []
    for d in repeat_dirs:
        rp = d / "results.csv"
        if not rp.exists():
            raise FileNotFoundError(f"missing {rp}")
        with rp.open("r", encoding="utf-8") as f:
            out.append(list(csv.DictReader(f)))
    return out


def aggregate_metrics(
    summaries: List[Dict[str, Any]], arms: List[str]
) -> Dict[str, Any]:
    """Mean / std of scalar metrics per arm across repeats."""
    out: Dict[str, Any] = {}
    for arm in arms:
        per_repeat = [s["metrics"][arm] for s in summaries]
        acc_m, acc_s = _mean_std([m.get("accuracy") for m in per_repeat])
        prec_m, prec_s = _mean_std([m.get("precision") for m in per_repeat])
        rec_m, rec_s = _mean_std([m.get("recall") for m in per_repeat])
        f1_m, f1_s = _mean_std([m.get("f1") for m in per_repeat])
        out[arm] = {
            "n_repeats": len(per_repeat),
            "accuracy_mean": _round(acc_m),
            "accuracy_std": _round(acc_s),
            "precision_mean": _round(prec_m),
            "precision_std": _round(prec_s),
            "recall_mean": _round(rec_m),
            "recall_std": _round(rec_s),
            "f1_mean": _round(f1_m),
            "f1_std": _round(f1_s),
            "tp_sum": sum(m.get("tp", 0) for m in per_repeat),
            "fp_sum": sum(m.get("fp", 0) for m in per_repeat),
            "tn_sum": sum(m.get("tn", 0) for m in per_repeat),
            "fn_sum": sum(m.get("fn", 0) for m in per_repeat),
            "n_unknown_sum": sum(m.get("n_unknown", 0) for m in per_repeat),
        }
    return out


def aggregate_per_rule(
    summaries: List[Dict[str, Any]], arms: List[str]
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Mean per-rule accuracy per arm across repeats."""
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for arm in arms:
        per_arm: Dict[str, Dict[str, Optional[float]]] = {}
        # Gather every rule mentioned across repeats
        rules = set()
        for s in summaries:
            for r in s["metrics"][arm]["per_rule_accuracy"]:
                rules.add(r)
        for rule in sorted(rules):
            accs = []
            totals = []
            corrects = []
            for s in summaries:
                v = s["metrics"][arm]["per_rule_accuracy"].get(rule)
                if v is None:
                    continue
                accs.append(v.get("accuracy"))
                totals.append(v.get("total", 0))
                corrects.append(v.get("correct", 0))
            m, sd = _mean_std(accs)
            per_arm[rule] = {
                "accuracy_mean": _round(m),
                "accuracy_std": _round(sd),
                "correct_sum": sum(corrects),
                "total_sum": sum(totals),
            }
        out[arm] = per_arm
    return out


def aggregate_usage(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cost = sum(s.get("usage", {}).get("total_cost_usd", 0.0)
                     for s in summaries)
    total_tokens = sum(s.get("usage", {}).get("total_tokens", 0)
                       for s in summaries)
    # Per-arm totals across repeats
    per_arm: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0,
                 "total_tokens": 0, "cost_usd": 0.0, "n_calls": 0}
    )
    for s in summaries:
        for arm, v in s.get("usage", {}).get("per_arm", {}).items():
            for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                      "cost_usd", "n_calls"):
                per_arm[arm][k] += v.get(k, 0)
    return {
        "total_cost_usd": round(total_cost, 4),
        "total_tokens": total_tokens,
        "per_arm": dict(per_arm),
    }


def compute_consistency(
    results_per_repeat: List[List[Dict[str, str]]],
    arms: List[str],
) -> Dict[str, Any]:
    """
    For each (arm, tx_id) tuple, gather the predictions across repeats.
    Compute fraction of repeats agreeing with the modal prediction.
    Also count "unanimous" (= n_repeats agreement).
    """
    # Build: arm -> tx_id -> [pred per repeat]
    per_arm_tx: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for results in results_per_repeat:
        for row in results:
            per_arm_tx[row["arm"]][row["tx_id"]].append(row.get("predicted", ""))

    n_repeats = len(results_per_repeat)
    out: Dict[str, Any] = {}
    for arm in arms:
        tx_map = per_arm_tx.get(arm, {})
        agreement_fractions = []
        n_unanimous = 0
        n_tx = 0
        for tx_id, preds in tx_map.items():
            if len(preds) < 2:
                continue
            n_tx += 1
            modal_pred, modal_count = Counter(preds).most_common(1)[0]
            agreement_fractions.append(modal_count / len(preds))
            if modal_count == len(preds):
                n_unanimous += 1
        mean_agree = (sum(agreement_fractions) / len(agreement_fractions)
                      if agreement_fractions else None)
        out[arm] = {
            "n_tx": n_tx,
            "n_repeats": n_repeats,
            "mean_agreement_with_mode": _round(mean_agree, 4),
            "n_unanimous": n_unanimous,
            "unanimous_fraction": _round(n_unanimous / n_tx, 4) if n_tx else None,
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Aggregate per-repeat eval results")
    p.add_argument("repeat_dirs", nargs="+", type=Path,
                   help="Directories containing per-repeat summary.json + results.csv")
    p.add_argument("--out", type=Path, default=None,
                   help="Write aggregated_summary.json here (default: stdout)")
    p.add_argument("--arms", default="A,B,C,D",
                   help="Comma-separated arm names (default ABCD)")
    args = p.parse_args(argv)

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    summaries = load_summaries(args.repeat_dirs)
    results = load_results(args.repeat_dirs)

    report = {
        "repeat_dirs": [str(d) for d in args.repeat_dirs],
        "n_repeats": len(summaries),
        "arms": arms,
        "n_transactions_per_repeat": summaries[0].get("n_transactions"),
        "metrics": aggregate_metrics(summaries, arms),
        "per_rule": aggregate_per_rule(summaries, arms),
        "consistency": compute_consistency(results, arms),
        "usage": aggregate_usage(summaries),
    }

    rendered = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.out is None:
        print(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
