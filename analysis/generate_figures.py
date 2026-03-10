"""CLI entry point for generating all ORCHID paper figures.

Usage:
    python -m analysis.generate_figures
    python -m analysis.generate_figures --results-dir evaluation/results --out-dir evaluation/figures
    generate-figures  # if installed via pyproject.toml

Outputs:
    evaluation/figures/fig_01_success_heatmap.{pdf,png}
    evaluation/figures/fig_02_fault_overview.{pdf,png}
    ...
    evaluation/figures/fig_10_tool_latency_box.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt

from analysis.load_results import load_traces
from analysis import plots as P


_FIGURES = [
    ("fig_01_success_heatmap",   lambda df, ds: P.fig_success_heatmap(df)),
    ("fig_02_fault_overview",    lambda df, ds: P.fig_fault_overview(df)),
    ("fig_03_fault_degradation", lambda df, ds: P.fig_fault_degradation(df)),
    ("fig_04_cost_vs_success",   lambda df, ds: P.fig_cost_vs_success(df)),
    ("fig_05_task_difficulty",   lambda df, ds: P.fig_task_difficulty(df)),
    ("fig_06_step_distribution", lambda df, ds: P.fig_step_distribution(df)),
    ("fig_07_token_breakdown",   lambda df, ds: P.fig_token_breakdown(df)),
    ("fig_08_retry_heatmap",     lambda df, ds: P.fig_retry_heatmap(df)),
    ("fig_09_latency_cdf",       lambda df, ds: P.fig_latency_cdf(df)),
    ("fig_10_tool_latency_box",  lambda df, ds: P.fig_tool_latency_box(ds, df)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ORCHID paper figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory containing traces/ and summary_*.csv (default: evaluation/results)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation/figures"),
        help="Output directory for figures (default: evaluation/figures)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output formats (default: pdf png)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="N",
        type=int,
        help="Only generate specific figure numbers, e.g. --only 1 3 9",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading traces from {results_dir} …", flush=True)
    df_summary, df_steps = load_traces(results_dir)

    if df_summary.empty:
        print("ERROR: No traces found. Run experiments first.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(df_summary)} runs loaded, {len(df_steps)} steps total")
    print(f"  Orchestrators: {sorted(df_summary['orchestrator'].dropna().unique())}")
    print(f"  Runtimes:      {sorted(df_summary['runtime'].dropna().unique())}")
    print(f"  Fault types:   {sorted(df_summary['fault_type'].dropna().unique())}")
    print()

    only_set = set(args.only) if args.only else None

    for idx, (name, builder) in enumerate(_FIGURES, start=1):
        if only_set and idx not in only_set:
            continue
        print(f"  [{idx:02d}/10] {name} …", end=" ", flush=True)
        try:
            fig: plt.Figure = builder(df_summary, df_steps)
            for fmt in args.formats:
                out_path = out_dir / f"{name}.{fmt}"
                fig.savefig(out_path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
            plt.close(fig)
            print(f"saved ({', '.join(args.formats)})")
        except Exception as exc:
            print(f"FAILED — {exc}")

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
