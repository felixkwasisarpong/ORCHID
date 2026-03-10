"""Publication-quality figure functions for the ORCHID research paper.

Each function accepts a DataFrame (from load_results) and returns a
matplotlib Figure ready to save as PDF or PNG.

Usage example:
    from analysis.load_results import load_traces
    from analysis import plots

    df, df_steps = load_traces()
    fig = plots.fig_success_heatmap(df)
    fig.savefig("fig_1_success_heatmap.pdf", bbox_inches="tight")
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # embeds TrueType fonts for ACM/IEEE submission
    "ps.fonttype": 42,
})

_PALETTE = "tab10"
_ORCHESTRATOR_ORDER = ["langgraph", "crewai", "autogen"]
_FAULT_TYPES = ["permission", "missing", "timeout", "latency"]
_FAULT_SEVERITIES = ["low", "med", "high"]

sns.set_theme(style="whitegrid", palette=_PALETTE)


def _orch_colors() -> dict[str, str]:
    colors = sns.color_palette(_PALETTE, n_colors=len(_ORCHESTRATOR_ORDER))
    return dict(zip(_ORCHESTRATOR_ORDER, [c for c in colors]))


# ---------------------------------------------------------------------------
# Figure 1 — Success Rate Heatmap (orchestrator × runtime, baseline only)
# ---------------------------------------------------------------------------

def fig_success_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of mean success rate per orchestrator × runtime (baseline runs)."""
    baseline = df[df["fault_type"] == "none"].copy()
    if baseline.empty:
        baseline = df.copy()

    pivot = (
        baseline.groupby(["orchestrator", "runtime"])["success"]
        .mean()
        .unstack(fill_value=float("nan"))
    )
    # Reorder rows to canonical orchestrator order if present
    row_order = [o for o in _ORCHESTRATOR_ORDER if o in pivot.index] + [
        o for o in pivot.index if o not in _ORCHESTRATOR_ORDER
    ]
    pivot = pivot.loc[row_order]

    fig, ax = plt.subplots(figsize=(max(4, pivot.shape[1] * 1.4), max(2.5, pivot.shape[0] * 1.0)))
    sns.heatmap(
        pivot * 100,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="YlGn",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "Success rate (%)"},
    )
    ax.set_title("Success Rate by Orchestrator × Runtime (baseline)")
    ax.set_xlabel("Runtime")
    ax.set_ylabel("Orchestrator")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Fault Overview (success rate per fault category, grouped bars)
# ---------------------------------------------------------------------------

def fig_fault_overview(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: success rate per fault type, grouped by orchestrator."""
    data = df[df["fault_type"].isin(_FAULT_TYPES)].copy()
    if data.empty:
        warnings.warn("No fault runs found; fig_fault_overview will be empty.")

    agg = (
        data.groupby(["fault_type", "orchestrator"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate"})
    )
    agg["success_rate"] *= 100

    orch_present = [o for o in _ORCHESTRATOR_ORDER if o in agg["orchestrator"].unique()]
    colors = [_orch_colors()[o] for o in orch_present]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=agg,
        x="fault_type",
        y="success_rate",
        hue="orchestrator",
        hue_order=orch_present,
        palette=colors,
        order=_FAULT_TYPES,
        ax=ax,
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Fault type")
    ax.set_title("Success Rate Under Each Fault Category")
    ax.legend(title="Orchestrator", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Fault Degradation Curves (4-panel, low/med/high per fault type)
# ---------------------------------------------------------------------------

def fig_fault_degradation(df: pd.DataFrame) -> plt.Figure:
    """4-panel line plots: success rate vs fault severity (low→high) per orchestrator."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    colors = _orch_colors()

    for ax, ftype in zip(axes, _FAULT_TYPES):
        subset = df[df["fault_type"] == ftype].copy()
        # Include baseline as severity='none' at position 0
        baseline = df[df["fault_type"] == "none"].copy()
        baseline["fault_severity"] = "none"
        combined = pd.concat([baseline, subset], ignore_index=True)

        sev_order = ["none"] + _FAULT_SEVERITIES
        agg = (
            combined.groupby(["fault_severity", "orchestrator"])["success"]
            .mean()
            .reset_index()
        )
        agg["sev_idx"] = agg["fault_severity"].map({s: i for i, s in enumerate(sev_order)})
        agg = agg.dropna(subset=["sev_idx"])

        for orch in _ORCHESTRATOR_ORDER:
            sub = agg[agg["orchestrator"] == orch].sort_values("sev_idx")
            if sub.empty:
                continue
            ax.plot(
                sub["sev_idx"],
                sub["success"] * 100,
                marker="o",
                label=orch,
                color=colors.get(orch),
            )

        ax.set_title(f"{ftype.capitalize()} fault")
        ax.set_xticks(range(len(sev_order)))
        ax.set_xticklabels(sev_order, fontsize=8)
        ax.set_xlabel("Severity")
        ax.set_ylim(-5, 110)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    axes[0].set_ylabel("Success rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Orchestrator", loc="lower center",
               ncol=len(_ORCHESTRATOR_ORDER), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Performance Degradation vs Fault Severity", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Cost vs Success Rate (scatter, bubble = token usage)
# ---------------------------------------------------------------------------

def fig_cost_vs_success(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot: mean cost vs mean success rate per runtime, bubble = tokens."""
    baseline = df[df["fault_type"] == "none"].copy()
    if baseline.empty:
        baseline = df.copy()

    agg = baseline.groupby("runtime").agg(
        success_rate=("success", "mean"),
        mean_cost=("llm_cost_usd", "mean"),
        mean_tokens=("llm_total_tokens", "mean"),
    ).reset_index()

    agg["success_rate"] *= 100
    # Normalise bubble size
    max_tokens = agg["mean_tokens"].max() or 1
    sizes = (agg["mean_tokens"] / max_tokens * 800).clip(lower=40)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = sns.color_palette(_PALETTE, n_colors=len(agg))
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.scatter(
            row["mean_cost"],
            row["success_rate"],
            s=sizes.iloc[i],
            color=palette[i],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label=row["runtime"],
        )
        ax.annotate(
            row["runtime"],
            (row["mean_cost"], row["success_rate"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlabel("Mean cost per run (USD)")
    ax.set_ylabel("Mean success rate (%)")
    ax.set_title("Cost vs Success Rate by Runtime\n(bubble size proportional to mean token usage)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(title="Runtime", bbox_to_anchor=(1.01, 1), loc="upper left", markerscale=0.7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Task Difficulty Profile (horizontal bars, sorted by overall rate)
# ---------------------------------------------------------------------------

def fig_task_difficulty(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart: success rate per task, grouped by orchestrator."""
    baseline = df[df["fault_type"] == "none"].copy()
    if baseline.empty:
        baseline = df.copy()

    agg = (
        baseline.groupby(["task_id", "orchestrator"])["success"]
        .mean()
        .reset_index()
    )
    agg["success"] *= 100

    overall_order = (
        agg.groupby("task_id")["success"].mean().sort_values().index.tolist()
    )

    # Shorten task labels: "task_01_count_lines" → "count_lines"
    label_map = {t: "_".join(t.split("_")[2:]) for t in overall_order}

    orch_present = [o for o in _ORCHESTRATOR_ORDER if o in agg["orchestrator"].unique()]
    n_orch = len(orch_present)
    bar_height = 0.25
    y_positions = np.arange(len(overall_order))

    fig, ax = plt.subplots(figsize=(8, max(4, len(overall_order) * 0.5)))
    colors = _orch_colors()

    for i, orch in enumerate(orch_present):
        sub = agg[agg["orchestrator"] == orch].set_index("task_id")
        values = [sub.loc[t, "success"] if t in sub.index else 0.0 for t in overall_order]
        offset = (i - n_orch / 2 + 0.5) * bar_height
        ax.barh(
            y_positions + offset,
            values,
            height=bar_height,
            label=orch,
            color=colors.get(orch),
            alpha=0.88,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label_map[t] for t in overall_order])
    ax.set_xlim(0, 115)
    ax.set_xlabel("Success rate (%)")
    ax.set_title("Task Difficulty Profile by Orchestrator")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(title="Orchestrator")
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Step Efficiency Distribution (violin/box, llm_calls per run)
# ---------------------------------------------------------------------------

def fig_step_distribution(df: pd.DataFrame) -> plt.Figure:
    """Violin + box plots: LLM call count distribution per orchestrator (baseline)."""
    baseline = df[df["fault_type"] == "none"].copy()
    if baseline.empty:
        baseline = df.copy()

    orch_present = [o for o in _ORCHESTRATOR_ORDER if o in baseline["orchestrator"].unique()]
    colors = [_orch_colors()[o] for o in orch_present]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=baseline,
        x="orchestrator",
        y="llm_calls",
        hue="orchestrator",
        order=orch_present,
        hue_order=orch_present,
        palette=colors,
        inner="box",
        cut=0,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Orchestrator")
    ax.set_ylabel("LLM calls per run")
    ax.set_title("Step Efficiency: LLM Call Distribution (baseline)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — Token Usage Breakdown (stacked bar, prompt vs completion)
# ---------------------------------------------------------------------------

def fig_token_breakdown(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar: mean prompt vs completion tokens per runtime."""
    baseline = df[df["fault_type"] == "none"].copy()
    if baseline.empty:
        baseline = df.copy()

    agg = baseline.groupby("runtime").agg(
        prompt=("llm_prompt_tokens", "mean"),
        completion=("llm_completion_tokens", "mean"),
    ).reset_index()

    # Sort by total
    agg["total"] = agg["prompt"] + agg["completion"]
    agg = agg.sort_values("total", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agg))
    ax.bar(x, agg["prompt"], label="Prompt tokens", color=sns.color_palette(_PALETTE)[0], alpha=0.85)
    ax.bar(x, agg["completion"], bottom=agg["prompt"], label="Completion tokens",
           color=sns.color_palette(_PALETTE)[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["runtime"], rotation=15, ha="right")
    ax.set_ylabel("Mean tokens per run")
    ax.set_title("Token Usage Breakdown by Runtime")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 8 — Retry Heatmap (mean retries: orchestrator × fault_type)
# ---------------------------------------------------------------------------

def fig_retry_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap: mean retry count per orchestrator × fault type."""
    faulted = df[df["fault_type"] != "none"].copy()
    if faulted.empty:
        warnings.warn("No fault runs found; fig_retry_heatmap may be empty.")
        faulted = df.copy()

    pivot = (
        faulted.groupby(["orchestrator", "fault_type"])["retries"]
        .mean()
        .unstack(fill_value=0.0)
    )
    row_order = [o for o in _ORCHESTRATOR_ORDER if o in pivot.index]
    pivot = pivot.loc[row_order]

    col_order = [c for c in _FAULT_TYPES if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(4, len(col_order) * 1.5), max(2.5, len(row_order) * 1.0)))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="OrRd",
        linewidths=0.5,
        cbar_kws={"label": "Mean retries per run"},
    )
    ax.set_title("Retry Behaviour Under Fault Conditions")
    ax.set_xlabel("Fault type")
    ax.set_ylabel("Orchestrator")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 9 — Latency CDF (total_latency_ms per orchestrator, baseline)
# ---------------------------------------------------------------------------

def fig_latency_cdf(df: pd.DataFrame) -> plt.Figure:
    """CDF of total run latency per orchestrator (baseline runs)."""
    baseline = df[df["fault_type"] == "none"].copy()
    if baseline.empty:
        baseline = df.copy()

    orch_present = [o for o in _ORCHESTRATOR_ORDER if o in baseline["orchestrator"].unique()]
    colors = _orch_colors()

    fig, ax = plt.subplots(figsize=(6, 4))
    for orch in orch_present:
        vals = baseline[baseline["orchestrator"] == orch]["total_latency_ms"].dropna().sort_values()
        if vals.empty:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals / 1000, cdf, label=orch, color=colors.get(orch), linewidth=1.8)

    ax.set_xlabel("Total latency (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF by Orchestrator (baseline)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Orchestrator")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 10 — Tool Call Latency Under Latency Faults (box plot, df_steps)
# ---------------------------------------------------------------------------

def fig_tool_latency_box(df_steps: pd.DataFrame, df_summary: Optional[pd.DataFrame] = None) -> plt.Figure:
    """Box plot: per-tool-call latency comparing baseline vs latency-fault runs."""
    tool_steps = df_steps[df_steps["action_type"] == "tool_call"].copy()
    if tool_steps.empty:
        warnings.warn("No tool_call steps found; fig_tool_latency_box will be empty.")

    # Label as baseline or faulted
    tool_steps["condition"] = tool_steps["fault_type"].apply(
        lambda x: "baseline" if x == "none" else ("latency fault" if x == "latency" else "other fault")
    )
    plot_data = tool_steps[tool_steps["condition"].isin(["baseline", "latency fault"])]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = {"baseline": sns.color_palette(_PALETTE)[0], "latency fault": sns.color_palette(_PALETTE)[3]}
    sns.boxplot(
        data=plot_data,
        x="tool_name",
        y="tool_latency_ms",
        hue="condition",
        palette=palette,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        ax=ax,
    )
    ax.set_xlabel("Tool")
    ax.set_ylabel("Tool call latency (ms)")
    ax.set_title("Tool Call Latency: Baseline vs Latency Fault")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Condition")
    fig.tight_layout()
    return fig
