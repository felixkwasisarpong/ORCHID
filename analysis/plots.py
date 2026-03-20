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

import math
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


def _bootstrap_mean_ci(
    values: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    clean = values.dropna().astype(float).to_numpy()
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, clean.size, size=(n_boot, clean.size))
    boot = clean[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


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


# ---------------------------------------------------------------------------
# Figure 11 — RQ1 Success with 95% CI by Runtime × Severity
# ---------------------------------------------------------------------------

def fig_success_ci_runtime_severity(df: pd.DataFrame) -> plt.Figure:
    """Line plots of success rate with bootstrap 95% CI by runtime and fault severity."""
    data = df.copy()
    data["severity_group"] = np.where(
        data["fault_type"] == "none",
        "none",
        data["fault_severity"].fillna("none"),
    )
    sev_order = ["none", "low", "med", "high"]
    runtimes = sorted(data["runtime"].dropna().unique())
    if not runtimes:
        warnings.warn("No runtime data found; fig_success_ci_runtime_severity may be empty.")
        runtimes = ["runtime"]

    n_cols = min(3, len(runtimes))
    n_rows = int(math.ceil(len(runtimes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()
    colors = _orch_colors()

    for ax, runtime in zip(axes_arr, runtimes):
        sub_rt = data[data["runtime"] == runtime]
        for orch in _ORCHESTRATOR_ORDER:
            sub = sub_rt[sub_rt["orchestrator"] == orch]
            if sub.empty:
                continue
            stats = []
            for sev in sev_order:
                s = sub[sub["severity_group"] == sev]["success"]
                if s.empty:
                    stats.append((np.nan, np.nan, np.nan))
                    continue
                mean = float(s.mean())
                lo, hi = _bootstrap_mean_ci(s)
                stats.append((mean, lo, hi))

            xs = np.arange(len(sev_order), dtype=float)
            means = np.array([t[0] for t in stats], dtype=float)
            lows = np.array([t[1] for t in stats], dtype=float)
            highs = np.array([t[2] for t in stats], dtype=float)
            valid = np.isfinite(means)
            if not valid.any():
                continue

            yerr = np.vstack([
                (means[valid] - lows[valid]) * 100.0,
                (highs[valid] - means[valid]) * 100.0,
            ])
            ax.errorbar(
                xs[valid],
                means[valid] * 100.0,
                yerr=yerr,
                marker="o",
                capsize=3,
                linewidth=1.6,
                color=colors.get(orch),
                label=orch,
            )

        ax.set_title(str(runtime))
        ax.set_xticks(np.arange(len(sev_order)))
        ax.set_xticklabels(sev_order)
        ax.set_xlabel("Fault severity group")
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    for ax in axes_arr[len(runtimes):]:
        ax.axis("off")

    axes_arr[0].set_ylabel("Success rate (%)")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Orchestrator", loc="lower center", ncol=len(handles))
    fig.suptitle("RQ1: Success Rate with 95% Bootstrap CI by Runtime and Severity", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 12 — RQ1/RQ3 Failure Mode Distribution (stacked bars)
# ---------------------------------------------------------------------------

def fig_failure_mode_stacked(df: pd.DataFrame) -> plt.Figure:
    """Stacked bars of failure_mode composition by orchestrator and fault type."""
    failed = df[(~df["success"]) & (df["fault_type"].isin(_FAULT_TYPES))].copy()
    if failed.empty:
        warnings.warn("No failed faulted runs found; fig_failure_mode_stacked may be empty.")
        failed = df[(~df["success"])].copy()

    if failed.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No failed runs found", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    failed["failure_mode"] = failed["failure_mode"].fillna("unknown_error")
    mode_counts = failed["failure_mode"].value_counts()
    top_modes = mode_counts.head(6).index.tolist()
    failed["failure_mode_plot"] = failed["failure_mode"].where(
        failed["failure_mode"].isin(top_modes), "other"
    )
    modes = top_modes + (["other"] if (failed["failure_mode_plot"] == "other").any() else [])

    orch_present = [o for o in _ORCHESTRATOR_ORDER if o in failed["orchestrator"].unique()]
    n_cols = max(1, len(orch_present))
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.2), sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()
    mode_palette = dict(zip(modes, sns.color_palette("tab20", n_colors=len(modes))))

    for ax, orch in zip(axes_arr, orch_present):
        sub = failed[failed["orchestrator"] == orch]
        count = (
            sub.groupby(["fault_type", "failure_mode_plot"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=[f for f in _FAULT_TYPES if f in sub["fault_type"].unique()], fill_value=0)
        )
        for mode in modes:
            if mode not in count.columns:
                count[mode] = 0
        count = count[modes]
        denom = count.sum(axis=1).replace(0, np.nan)
        frac = count.div(denom, axis=0).fillna(0.0)

        x = np.arange(len(frac))
        bottom = np.zeros(len(frac), dtype=float)
        for mode in modes:
            vals = frac[mode].to_numpy() * 100.0
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=mode_palette[mode],
                edgecolor="white",
                linewidth=0.5,
                label=mode,
            )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(frac.index.tolist(), rotation=20, ha="right")
        ax.set_title(orch)
        ax.set_xlabel("Fault type")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    axes_arr[0].set_ylabel("Failure mode composition (%)")
    handles, labels = axes_arr[-1].get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        fig.legend(
            uniq.values(),
            uniq.keys(),
            title="Failure mode",
            loc="lower center",
            ncol=min(len(uniq), 4),
        )
    fig.suptitle("RQ1/RQ3: Failure Modes by Orchestrator and Fault Type", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 13 — RQ2 Latency CDF by Runtime (lines = orchestrator)
# ---------------------------------------------------------------------------

def fig_latency_cdf_runtime(df: pd.DataFrame) -> plt.Figure:
    """Small-multiple latency CDFs per runtime, colored by orchestrator."""
    baseline = df[df["fault_type"] == "none"].copy()
    if baseline.empty:
        baseline = df.copy()

    runtimes = sorted(baseline["runtime"].dropna().unique())
    if not runtimes:
        warnings.warn("No runtime data found; fig_latency_cdf_runtime may be empty.")
        runtimes = ["runtime"]
    n_cols = min(3, len(runtimes))
    n_rows = int(math.ceil(len(runtimes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()
    colors = _orch_colors()

    for ax, runtime in zip(axes_arr, runtimes):
        sub_rt = baseline[baseline["runtime"] == runtime]
        for orch in _ORCHESTRATOR_ORDER:
            vals = (
                sub_rt[sub_rt["orchestrator"] == orch]["total_latency_ms"]
                .dropna()
                .sort_values()
                .to_numpy()
            )
            if vals.size == 0:
                continue
            cdf = np.arange(1, vals.size + 1, dtype=float) / vals.size
            ax.plot(vals / 1000.0, cdf, color=colors.get(orch), linewidth=1.6, label=orch)

        ax.set_title(str(runtime))
        ax.set_xlabel("Total latency (s)")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    for ax in axes_arr[len(runtimes):]:
        ax.axis("off")

    axes_arr[0].set_ylabel("CDF")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Orchestrator", loc="lower center", ncol=len(handles))
    fig.suptitle("RQ2: Latency CDF by Runtime and Orchestrator (baseline)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 14 — RQ2 Retry Distribution by Runtime × Orchestrator
# ---------------------------------------------------------------------------

def fig_retry_distribution_runtime(df: pd.DataFrame) -> plt.Figure:
    """Box plots of retry distributions grouped by runtime and orchestrator."""
    runtimes = sorted(df["runtime"].dropna().unique())
    orch_present = [o for o in _ORCHESTRATOR_ORDER if o in df["orchestrator"].unique()]
    if not runtimes or not orch_present:
        warnings.warn("Insufficient data for fig_retry_distribution_runtime.")

    fig, ax = plt.subplots(figsize=(max(7, len(runtimes) * 1.2), 4.5))
    sns.boxplot(
        data=df,
        x="runtime",
        y="retries",
        hue="orchestrator",
        order=runtimes,
        hue_order=orch_present,
        palette=[_orch_colors()[o] for o in orch_present],
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35},
        ax=ax,
    )
    ax.set_xlabel("Runtime")
    ax.set_ylabel("Retries per run")
    ax.set_title("RQ2: Retry Distributions by Runtime and Orchestrator")
    ax.legend(title="Orchestrator", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 15 — RQ2 Retries vs Latency Tradeoff
# ---------------------------------------------------------------------------

def fig_retry_latency_tradeoff(df: pd.DataFrame) -> plt.Figure:
    """Scatter + binned trend of retries vs latency, faceted by runtime."""
    runtimes = sorted(df["runtime"].dropna().unique())
    if not runtimes:
        warnings.warn("No runtime data found; fig_retry_latency_tradeoff may be empty.")
        runtimes = ["runtime"]

    n_cols = min(3, len(runtimes))
    n_rows = int(math.ceil(len(runtimes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()
    colors = _orch_colors()
    rng = np.random.default_rng(11)

    for ax, runtime in zip(axes_arr, runtimes):
        sub_rt = df[df["runtime"] == runtime].copy()
        if sub_rt.empty:
            ax.axis("off")
            continue

        for orch in _ORCHESTRATOR_ORDER:
            sub = sub_rt[sub_rt["orchestrator"] == orch]
            if sub.empty:
                continue
            x = sub["retries"].astype(float).to_numpy()
            y = (sub["total_latency_ms"].astype(float) / 1000.0).to_numpy()
            jitter = rng.normal(0.0, 0.05, size=x.shape)
            ax.scatter(
                x + jitter,
                y,
                s=12,
                alpha=0.2,
                color=colors.get(orch),
            )
            trend = sub.groupby("retries")["total_latency_ms"].mean().reset_index()
            ax.plot(
                trend["retries"],
                trend["total_latency_ms"] / 1000.0,
                marker="o",
                linewidth=1.8,
                color=colors.get(orch),
                label=orch,
            )

        ax.set_title(str(runtime))
        ax.set_xlabel("Retries")

    for ax in axes_arr[len(runtimes):]:
        ax.axis("off")

    axes_arr[0].set_ylabel("Total latency (s)")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Orchestrator", loc="lower center", ncol=len(handles))
    fig.suptitle("RQ2: Retries vs Latency Tradeoff", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 16 — RQ3 Step Error Category Heatmap by Tool × Fault
# ---------------------------------------------------------------------------

def fig_step_error_heatmap(df_steps: pd.DataFrame) -> plt.Figure:
    """Heatmaps of step_error_category counts per tool under each fault family."""
    data = df_steps[
        (df_steps["action_type"] == "tool_call")
        & (df_steps["fault_type"].isin(_FAULT_TYPES))
        & (df_steps["step_error_category"].notna())
    ].copy()
    if data.empty:
        warnings.warn("No categorized step errors found; fig_step_error_heatmap may be empty.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No step errors found", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    top_categories = data["step_error_category"].value_counts().head(8).index.tolist()
    data["cat_plot"] = data["step_error_category"].where(
        data["step_error_category"].isin(top_categories),
        "other",
    )
    categories = top_categories + (["other"] if (data["cat_plot"] == "other").any() else [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes_arr = axes.ravel()

    for ax, ftype in zip(axes_arr, _FAULT_TYPES):
        sub = data[data["fault_type"] == ftype]
        if sub.empty:
            ax.set_title(f"{ftype} fault")
            ax.text(0.5, 0.5, "No errors", ha="center", va="center")
            ax.axis("off")
            continue
        pivot = (
            sub.groupby(["tool_name", "cat_plot"])
            .size()
            .unstack(fill_value=0)
        )
        for c in categories:
            if c not in pivot.columns:
                pivot[c] = 0
        pivot = pivot[categories].sort_index()
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="Reds",
            linewidths=0.4,
            cbar=False,
        )
        ax.set_title(f"{ftype.capitalize()} fault")
        ax.set_xlabel("Error category")
        ax.set_ylabel("Tool")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("RQ3: Step Error Category Heatmaps by Tool and Fault Type", y=1.01)
    fig.tight_layout()
    return fig
