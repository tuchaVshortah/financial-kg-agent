"""
Generate the four headline thesis figures from the v2 + v3 + v3b
benchmark runs. Reads `runs/full_v3b/aggregated_summary.json` (which
contains the combined metrics) and writes PNG + SVG to `figures/`.

Style choices (thesis embedding):
  * 300 dpi PNG + scalable SVG.
  * A4-friendly aspect ratios (~6 x 4 inches typical).
  * Neutral colour palette suitable for greyscale fallback —
    distinguishable shapes/linestyles rather than relying on hue.
  * Tight layout; clear axis labels; no chartjunk; minimal grid.

Usage:
    python -m src.make_figures
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


REPORT_PATH = Path("runs/full_v3b/aggregated_summary.json")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    """Write the figure as both PNG (300 dpi) and SVG."""
    png_path = FIGURES_DIR / f"{name}.png"
    svg_path = FIGURES_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}  +  {svg_path}")


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.facecolor": "white",
})


# --------------------------------------------------------------------------- #
# Figure 1 — degradation curve (acc + F1 vs p)
# --------------------------------------------------------------------------- #


def fig_degradation_curve(report: dict) -> None:
    """Arm E accuracy + F1 vs p, with Arms A & B as reference lines."""
    points = report["crossover_analysis"]["E_curve"]
    ps = [pt["p"] for pt in points]
    accs = [pt["acc"] for pt in points]

    # F1 isn't in crossover_analysis; pull from overall
    arm_name_for_p = {
        0.00: "C", 0.25: "E_p25", 0.50: "E_p50", 0.75: "E_p75",
        0.85: "E_p85", 0.95: "E_p95", 1.00: "E_p100",
    }
    f1s = [report["overall"][arm_name_for_p[p]]["f1"]["mean"] for p in ps]

    A_acc = report["overall"]["A"]["acc"]["mean"]
    B_acc = report["overall"]["B"]["acc"]["mean"]
    C_acc = report["overall"]["C"]["acc"]["mean"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ax.plot(ps, accs, marker="o", color="#1f3a93", linewidth=1.8,
            label="Arm E accuracy")
    ax.plot(ps, f1s, marker="s", color="#1f3a93", linewidth=1.4,
            linestyle="--", alpha=0.7, label="Arm E F1")

    ax.axhline(A_acc, color="#888", linestyle=":", linewidth=1.0)
    ax.axhline(B_acc, color="#555", linestyle="-.", linewidth=1.0)
    ax.axhline(C_acc, color="#1f3a93", linestyle="-", linewidth=0.6, alpha=0.4)

    # Reference labels at the right edge
    ax.text(1.01, A_acc, f" Arm A = {A_acc:.3f}",
            va="center", fontsize=8.5, color="#666")
    ax.text(1.01, B_acc, f" Arm B = {B_acc:.3f}",
            va="center", fontsize=8.5, color="#444")
    ax.text(1.01, C_acc, f" Arm C / D = {C_acc:.3f}",
            va="center", fontsize=8.5, color="#1f3a93")

    # Highlight that E_p100 is STILL above B
    ax.annotate(
        f"E_p100 = {accs[-1]:.3f}\n($+{accs[-1] - B_acc:.3f}$ above B)",
        xy=(1.0, accs[-1]), xytext=(0.72, 0.85),
        fontsize=8.5, ha="left", color="#333",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.6),
    )

    ax.set_xlabel("KG-relation dropout probability $p$")
    ax.set_ylabel("Accuracy / F1")
    ax.set_xlim(-0.04, 1.22)
    ax.set_ylim(0.60, 1.04)
    # Use 0.0, 0.25, 0.5, 0.75, 1.0 as major ticks (0.85/0.95 marked by data
    # points but skipped from the tick line to prevent overlap on the right edge).
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([0.85, 0.95], minor=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(axis="x", which="minor", labelsize=7, length=3, pad=12)
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title("KG-relation dropout: Arm E degradation curve")
    ax.legend(loc="lower left")

    _save(fig, "fig_degradation_curve")


# --------------------------------------------------------------------------- #
# Figure 2 — per-rule accuracy heatmap
# --------------------------------------------------------------------------- #


def fig_per_rule_heatmap(report: dict) -> None:
    """Heatmap: rows = rule scenarios, cols = arms ordered by KG coverage."""
    scenarios = [
        "AML_THRESHOLD", "STRUCTURING", "SANCTIONS", "KYC_VALIDITY",
        "HIGH_RISK_JURISDICTION", "VELOCITY", "DORMANT_ACCOUNT_RULE",
        "ROUND_NUMBER_ANOMALY", "NORMAL",
    ]
    arms_order = [
        "A", "B",
        "E_p100", "E_p95", "E_p85", "E_p75", "E_p50", "E_p25",
        "C", "D",
    ]
    arm_labels = [
        "A\nvanilla", "B\nrule-text",
        "$E_{1.00}$", "$E_{0.95}$", "$E_{0.85}$",
        "$E_{0.75}$", "$E_{0.50}$", "$E_{0.25}$",
        "C\nfull KG", "D\nGT flag",
    ]

    grid = []
    for sc in scenarios:
        row = []
        for arm in arms_order:
            v = report["per_rule"].get(arm, {}).get(sc)
            row.append(v["acc"]["mean"] if v and v["acc"] else float("nan"))
        grid.append(row)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(arms_order)))
    ax.set_xticklabels(arm_labels, fontsize=8.5)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8.5)
    ax.set_title("Per-rule accuracy by arm (mean across 3 repeats)")

    # Annotate cells with the accuracy values
    for i in range(len(scenarios)):
        for j in range(len(arms_order)):
            val = grid[i][j]
            if math.isnan(val):
                txt = "—"
                colour = "#999"
            else:
                txt = f"{val:.2f}"
                # White text on dark cells, black on light cells
                colour = "white" if val < 0.75 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=colour)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=9)
    ax.grid(False)
    _save(fig, "fig_per_rule_heatmap")


# --------------------------------------------------------------------------- #
# Figure 3 — firing-count strata
# --------------------------------------------------------------------------- #


def fig_firing_count_strata(report: dict) -> None:
    """
    Accuracy stratified by firing-count for each Arm E variant.
    Each line corresponds to one firing-count bucket (0..5).
    The fc>=3 immunity should jump out: those lines stay at 1.000.
    """
    arms_in_order = ["E_p25", "E_p50", "E_p75", "E_p85", "E_p95", "E_p100"]
    arm_ps = [0.25, 0.50, 0.75, 0.85, 0.95, 1.00]
    fc_values = ["0", "1", "2", "3", "4", "5"]
    fc_labels = {
        "0": "fc=0 (compliant)",
        "1": "fc=1",
        "2": "fc=2",
        "3": "fc=3",
        "4": "fc=4",
        "5": "fc=5",
    }
    # Greyscale-friendly palette: lighter for low fc, darker for high
    fc_colors = {
        "0": "#d62728",  # red — vulnerable
        "1": "#ff7f0e",  # orange
        "2": "#bcbd22",  # olive
        "3": "#2ca02c",  # green
        "4": "#17becf",  # cyan
        "5": "#1f3a93",  # blue
    }
    fc_styles = {
        "0": ("-", "o"),
        "1": ("-", "s"),
        "2": ("-", "^"),
        "3": ("--", "D"),
        "4": ("--", "v"),
        "5": ("--", "P"),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for fc in fc_values:
        ys = []
        for arm in arms_in_order:
            d = report["per_fc"].get(arm, {}).get(fc)
            ys.append(d["acc"] if d and d["acc"] is not None else float("nan"))
        ls, marker = fc_styles[fc]
        ax.plot(arm_ps, ys, color=fc_colors[fc], linestyle=ls,
                marker=marker, markersize=5, linewidth=1.5,
                label=fc_labels[fc])

    ax.axhspan(0.99, 1.005, color="#2ca02c", alpha=0.08)
    ax.text(1.005, 0.995, " fc$\\geq$3 immunity\n region", fontsize=8,
            color="#2ca02c", va="top")

    ax.set_xlabel("KG-relation dropout probability $p$")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.18, 1.22)
    ax.set_ylim(0.50, 1.04)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([0.85, 0.95], minor=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(axis="x", which="minor", labelsize=7, length=3, pad=12)
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title("Arm E accuracy stratified by firing-count")
    ax.legend(loc="lower left", ncol=2, fontsize=8.5)

    # Annotate the fc=1 rebound at high p — a real effect worth explaining
    ax.annotate(
        "fc=1 rebound: at high $p$ the LLM's\n"
        "default-to-non-compliant bias matches\n"
        "the truly-non-compliant fc=1 class",
        xy=(0.95, 0.875), xytext=(0.30, 0.62),
        fontsize=7.5, color="#aa5500",
        arrowprops=dict(arrowstyle="->", color="#aa5500", lw=0.6),
    )

    _save(fig, "fig_firing_count_strata")


# --------------------------------------------------------------------------- #
# Figure 4 — error-type asymmetry (two-panel)
# --------------------------------------------------------------------------- #


def fig_error_asymmetry(report: dict) -> None:
    """
    Two panels:
      Left:  missed-violation rate (1 - precision)
             = FP / (TP + FP)
             = of "compliant" predictions, fraction wrongly cleared.
      Right: spurious-alarm rate
             = FN / (FN + TN)
             = of "non-compliant" predictions, fraction wrongly flagged.

    Plotted across p with Arms A and B as reference lines (their values
    are evaluated identically: FP/(TP+FP) for A and B too).
    """
    arms_p = [("C", 0.00), ("E_p25", 0.25), ("E_p50", 0.50), ("E_p75", 0.75),
              ("E_p85", 0.85), ("E_p95", 0.95), ("E_p100", 1.00)]

    def rates_for(arm):
        m = report["overall"][arm]
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        miss = fp / (tp + fp) if (tp + fp) else float("nan")
        spurious = fn / (fn + tn) if (fn + tn) else float("nan")
        return miss, spurious

    ps = [p for _, p in arms_p]
    miss_es = []
    spur_es = []
    for arm, _ in arms_p:
        m, s = rates_for(arm)
        miss_es.append(m)
        spur_es.append(s)

    miss_A, spur_A = rates_for("A")
    miss_B, spur_B = rates_for("B")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)

    # Panel left
    axL.plot(ps, miss_es, marker="o", color="#b9293c", linewidth=1.8,
             label="Arm E missed-violation rate")
    axL.axhline(miss_A, color="#888", linestyle=":", linewidth=1.0)
    axL.axhline(miss_B, color="#555", linestyle="-.", linewidth=1.0)
    axL.text(1.03, miss_A, f" A={miss_A:.3f}", va="center",
             fontsize=8.5, color="#666")
    axL.text(1.03, miss_B, f" B={miss_B:.3f}", va="center",
             fontsize=8.5, color="#444")
    axL.set_xlabel("KG-relation dropout probability $p$")
    axL.set_ylabel("Error rate")
    axL.set_title("Missed-violation rate\n"
                  "FP/(TP+FP) — operationally worse",
                  pad=8)
    axL.set_xlim(-0.04, 1.22)
    axL.set_ylim(0.0, 0.40)
    axL.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axL.set_xticks([0.85, 0.95], minor=True)
    axL.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    axL.tick_params(axis="x", which="minor", labelsize=7, length=3, pad=12)
    axL.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    axL.legend(loc="upper left", fontsize=8.5)

    # Panel right
    axR.plot(ps, spur_es, marker="s", color="#1f3a93", linewidth=1.8,
             label="Arm E spurious-alarm rate")
    axR.axhline(spur_A, color="#888", linestyle=":", linewidth=1.0)
    axR.axhline(spur_B, color="#555", linestyle="-.", linewidth=1.0)
    axR.text(1.03, spur_A, f" A={spur_A:.3f}", va="center",
             fontsize=8.5, color="#666")
    axR.text(1.03, spur_B, f" B={spur_B:.3f}", va="center",
             fontsize=8.5, color="#444")
    axR.set_xlabel("KG-relation dropout probability $p$")
    axR.set_title("Spurious-alarm rate\n"
                  "FN/(FN+TN) — annoying but safe",
                  pad=8)
    axR.set_xlim(-0.04, 1.22)
    axR.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axR.set_xticks([0.85, 0.95], minor=True)
    axR.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    axR.tick_params(axis="x", which="minor", labelsize=7, length=3, pad=12)
    axR.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    axR.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Arm E error-type asymmetry under KG-relation dropout",
                 fontsize=11.5, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig_error_asymmetry")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    if not REPORT_PATH.exists():
        raise SystemExit(
            f"missing {REPORT_PATH}; run the v2 + v3 + v3b benchmarks first"
        )
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print("Generating figures under", FIGURES_DIR.resolve())
    fig_degradation_curve(report)
    fig_per_rule_heatmap(report)
    fig_firing_count_strata(report)
    fig_error_asymmetry(report)
    print("done.")


if __name__ == "__main__":
    main()
