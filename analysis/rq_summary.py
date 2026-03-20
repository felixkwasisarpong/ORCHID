"""Generate RQ-aligned summary tables from ORCHID traces.

Outputs CSV tables for:
- RQ1 reliability + recovery
- RQ2 latency/retry tradeoffs
- RQ3 failure mode distributions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.load_results import load_traces


def _safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def _rq1_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["orchestrator", "runtime", "fault_type", "fault_severity"]
    grouped = df.groupby(keys, dropna=False)

    rows: list[dict] = []
    for key, group in grouped:
        retries_positive = group[group["retries"] > 0]
        rows.append({
            "orchestrator": key[0],
            "runtime": key[1],
            "fault_type": key[2],
            "fault_severity": key[3],
            "runs": int(len(group)),
            "success_rate": float(group["success"].mean()),
            "failure_rate": float(1.0 - group["success"].mean()),
            "mean_retries": float(group["retries"].mean()),
            "recovery_success_rate": float(retries_positive["success"].mean()) if not retries_positive.empty else float("nan"),
            "failure_mode_top": (
                group.loc[~group["success"], "failure_mode"].dropna().value_counts().idxmax()
                if (~group["success"]).any() and not group.loc[~group["success"], "failure_mode"].dropna().empty
                else None
            ),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _rq2_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scope_name, scoped in {
        "all": df,
        "baseline": df[df["fault_type"] == "none"],
    }.items():
        if scoped.empty:
            continue
        grouped = scoped.groupby(["orchestrator", "runtime"], dropna=False)
        for (orch, runtime), group in grouped:
            latency = group["total_latency_ms"]
            retries = group["retries"]
            llm_calls = group["llm_calls"]
            rows.append({
                "scope": scope_name,
                "orchestrator": orch,
                "runtime": runtime,
                "runs": int(len(group)),
                "latency_mean_ms": float(latency.mean()),
                "latency_std_ms": float(latency.std(ddof=0)),
                "latency_cv": float((latency.std(ddof=0) / latency.mean()) if latency.mean() > 0 else 0.0),
                "latency_p50_ms": _safe_quantile(latency, 0.50),
                "latency_p95_ms": _safe_quantile(latency, 0.95),
                "latency_p99_ms": _safe_quantile(latency, 0.99),
                "mean_retries": float(retries.mean()),
                "retry_p95": _safe_quantile(retries, 0.95),
                "llm_calls_mean": float(llm_calls.mean()),
                "tool_calls_mean": float(group["tool_calls"].mean()),
            })
    return pd.DataFrame(rows).sort_values(["scope", "orchestrator", "runtime"]).reset_index(drop=True)


def _rq3_run_modes(df: pd.DataFrame) -> pd.DataFrame:
    failed = df[~df["success"]].copy()
    if failed.empty:
        return pd.DataFrame(
            columns=[
                "orchestrator",
                "runtime",
                "fault_type",
                "fault_severity",
                "failure_mode",
                "count",
                "rate_within_group",
            ]
        )

    keys = ["orchestrator", "runtime", "fault_type", "fault_severity"]
    totals = failed.groupby(keys, dropna=False).size().rename("group_total")
    counts = failed.groupby(keys + ["failure_mode"], dropna=False).size().rename("count").reset_index()
    merged = counts.merge(totals.reset_index(), on=keys, how="left")
    merged["rate_within_group"] = merged["count"] / merged["group_total"]
    return merged.sort_values(keys + ["count"], ascending=[True, True, True, True, False]).reset_index(drop=True)


def _rq3_step_modes(df_steps: pd.DataFrame) -> pd.DataFrame:
    errored = df_steps[df_steps["step_error_category"].notna()].copy()
    if errored.empty:
        return pd.DataFrame(
            columns=[
                "orchestrator",
                "runtime",
                "fault_type",
                "fault_severity",
                "step_error_category",
                "count",
            ]
        )

    keys = ["orchestrator", "runtime", "fault_type", "fault_severity", "step_error_category"]
    return (
        errored.groupby(keys, dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(keys[:-1] + ["count"], ascending=[True, True, True, True, False])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ summary tables from ORCHID traces")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory containing traces/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation/results/analysis"),
        help="Output directory for CSV summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_summary, df_steps = load_traces(args.results_dir)
    if df_summary.empty:
        raise SystemExit("No traces found. Run experiments first.")

    rq1 = _rq1_table(df_summary)
    rq2 = _rq2_table(df_summary)
    rq3_runs = _rq3_run_modes(df_summary)
    rq3_steps = _rq3_step_modes(df_steps)

    rq1.to_csv(out_dir / "rq1_reliability_recovery.csv", index=False)
    rq2.to_csv(out_dir / "rq2_latency_retry_tradeoffs.csv", index=False)
    rq3_runs.to_csv(out_dir / "rq3_failure_modes_runs.csv", index=False)
    rq3_steps.to_csv(out_dir / "rq3_failure_modes_steps.csv", index=False)

    print(f"Wrote {out_dir / 'rq1_reliability_recovery.csv'}")
    print(f"Wrote {out_dir / 'rq2_latency_retry_tradeoffs.csv'}")
    print(f"Wrote {out_dir / 'rq3_failure_modes_runs.csv'}")
    print(f"Wrote {out_dir / 'rq3_failure_modes_steps.csv'}")


if __name__ == "__main__":
    main()
