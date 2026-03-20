"""Compute paper-ready statistics: bootstrap CIs and corrected hypothesis tests.

Outputs:
- stats_group_bootstrap_ci.csv
- stats_pairwise_tests_raw.csv
- stats_pairwise_tests_bh.csv
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.load_results import load_traces


def _stable_seed(*parts: object) -> int:
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="big", signed=False) % (2**31 - 1)
    return seed or 1


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, alpha: float, seed: int) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        v = float(clean[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, clean.size, size=(n_boot, clean.size))
    boot = clean[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def _perm_test_diff_mean(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")

    obs = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    n_x = x.size
    rng = np.random.default_rng(seed)

    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(perm[:n_x].mean() - perm[n_x:].mean())
        if abs(diff) >= abs(obs):
            extreme += 1

    p = (extreme + 1.0) / (n_perm + 1.0)
    return obs, p


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float)
    m = len(p)
    if m == 0:
        return p

    order = np.argsort(p.to_numpy())
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(1, m + 1, dtype=float)
    adj = p.to_numpy() * m / ranks

    # Monotonicity correction in reverse rank order
    adj_ordered = adj[order]
    adj_ordered = np.minimum.accumulate(adj_ordered[::-1])[::-1]
    adj[order] = np.clip(adj_ordered, 0.0, 1.0)

    return pd.Series(adj, index=p.index)


def _group_bootstrap_table(df: pd.DataFrame, n_boot: int, alpha: float) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["orchestrator", "runtime", "fault_type", "fault_severity"]
    grouped = df.groupby(group_cols, dropna=False)

    for key, group in grouped:
        success = group["success"].astype(float).to_numpy()
        latency = group["total_latency_ms"].astype(float).to_numpy()
        retries = group["retries"].astype(float).to_numpy()

        seed_base = _stable_seed(*key)
        s_lo, s_hi = _bootstrap_mean_ci(success, n_boot=n_boot, alpha=alpha, seed=seed_base + 1)
        l_lo, l_hi = _bootstrap_mean_ci(latency, n_boot=n_boot, alpha=alpha, seed=seed_base + 2)
        r_lo, r_hi = _bootstrap_mean_ci(retries, n_boot=n_boot, alpha=alpha, seed=seed_base + 3)

        rows.append({
            "orchestrator": key[0],
            "runtime": key[1],
            "fault_type": key[2],
            "fault_severity": key[3],
            "n": int(len(group)),
            "success_rate": float(np.mean(success)) if success.size else float("nan"),
            "success_ci_low": s_lo,
            "success_ci_high": s_hi,
            "latency_mean_ms": float(np.mean(latency)) if latency.size else float("nan"),
            "latency_ci_low_ms": l_lo,
            "latency_ci_high_ms": l_hi,
            "retries_mean": float(np.mean(retries)) if retries.size else float("nan"),
            "retries_ci_low": r_lo,
            "retries_ci_high": r_hi,
        })

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _pairwise_tests(df: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    rows: list[dict] = []
    cell_cols = ["runtime", "fault_type", "fault_severity"]
    metrics = {
        "success": ("success", True),
        "latency_ms": ("total_latency_ms", False),
        "retries": ("retries", False),
    }

    for cell, group in df.groupby(cell_cols, dropna=False):
        orch_vals = sorted(group["orchestrator"].dropna().unique())
        if len(orch_vals) < 2:
            continue

        for metric_name, (col, is_bool) in metrics.items():
            for left, right in combinations(orch_vals, 2):
                left_vals = group[group["orchestrator"] == left][col].astype(float).to_numpy()
                right_vals = group[group["orchestrator"] == right][col].astype(float).to_numpy()
                if left_vals.size < 2 or right_vals.size < 2:
                    continue

                seed = _stable_seed(*cell, metric_name, left, right)
                effect, p_raw = _perm_test_diff_mean(left_vals, right_vals, n_perm=n_perm, seed=seed)

                rows.append({
                    "runtime": cell[0],
                    "fault_type": cell[1],
                    "fault_severity": cell[2],
                    "metric": metric_name,
                    "left_orchestrator": left,
                    "right_orchestrator": right,
                    "left_n": int(left_vals.size),
                    "right_n": int(right_vals.size),
                    "effect_left_minus_right": effect,
                    "p_raw": p_raw,
                    "metric_is_binary": is_bool,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_bh"] = out.groupby("metric", dropna=False)["p_raw"].transform(_bh_fdr)
    out["significant_0_05"] = out["p_bh"] < 0.05
    return out.sort_values(
        ["metric", "runtime", "fault_type", "fault_severity", "left_orchestrator", "right_orchestrator"]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bootstrap CIs and corrected hypothesis tests")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory containing traces/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation/results/stats"),
        help="Output directory for stats CSVs",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default: 2000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CI alpha (default: 0.05 for 95% CI)",
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=3000,
        help="Permutation samples per pairwise test (default: 3000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_summary, _ = load_traces(args.results_dir)
    if df_summary.empty:
        raise SystemExit("No traces found. Run experiments first.")

    group_ci = _group_bootstrap_table(df_summary, n_boot=args.bootstrap_samples, alpha=args.alpha)
    tests = _pairwise_tests(df_summary, n_perm=args.permutation_samples)

    path_group = out_dir / "stats_group_bootstrap_ci.csv"
    path_tests_raw = out_dir / "stats_pairwise_tests_raw.csv"
    path_tests_bh = out_dir / "stats_pairwise_tests_bh.csv"

    group_ci.to_csv(path_group, index=False)
    tests_raw = tests.drop(columns=["p_bh", "significant_0_05"], errors="ignore")
    tests_raw.to_csv(path_tests_raw, index=False)
    tests.to_csv(path_tests_bh, index=False)

    print(f"Wrote {path_group}")
    print(f"Wrote {path_tests_raw}")
    print(f"Wrote {path_tests_bh}")


if __name__ == "__main__":
    main()
