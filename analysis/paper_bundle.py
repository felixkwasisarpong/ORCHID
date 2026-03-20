"""Run the full paper artifact pipeline into one output directory.

Pipeline steps:
1) rq_summary
2) experiment_report
3) stats_plan
4) generate_figures (optional)

Writes a manifest at <out_dir>/bundle_manifest.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


def _run_module(module: str, args: list[str], allow_fail: bool = False) -> dict[str, Any]:
    cmd = [sys.executable, "-m", module, *args]
    cmd_str = shlex.join(cmd)
    print(f"Running: {cmd_str}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return {
            "module": module,
            "command": cmd_str,
            "status": "ok",
            "allow_fail": allow_fail,
        }
    except subprocess.CalledProcessError as exc:
        status = {
            "module": module,
            "command": cmd_str,
            "status": "failed",
            "allow_fail": allow_fail,
            "returncode": exc.returncode,
        }
        if allow_fail:
            print(f"WARNING: step failed but continuing: {module} (exit={exc.returncode})", flush=True)
            return status
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a full paper bundle from experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory containing traces and summaries",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=Path("configs/matrix"),
        help="Directory containing matrix configs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation/paper_bundle"),
        help="Output root for bundle artifacts",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Figure formats for generate_figures",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for stats_plan",
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=3000,
        help="Permutation samples for stats_plan",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for confidence intervals in stats_plan",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--allow-figure-fail",
        action="store_true",
        help="Continue and write bundle even if figure generation fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    analysis_dir = out_dir / "analysis"
    report_dir = out_dir / "report"
    stats_dir = out_dir / "stats"
    figures_dir = out_dir / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []

    steps.append(_run_module(
        "analysis.rq_summary",
        [
            "--results-dir", str(args.results_dir),
            "--out-dir", str(analysis_dir),
        ],
    ))

    steps.append(_run_module(
        "analysis.experiment_report",
        [
            "--results-dir", str(args.results_dir),
            "--matrix-dir", str(args.matrix_dir),
            "--out-dir", str(report_dir),
        ],
    ))

    steps.append(_run_module(
        "analysis.stats_plan",
        [
            "--results-dir", str(args.results_dir),
            "--out-dir", str(stats_dir),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--permutation-samples", str(args.permutation_samples),
            "--alpha", str(args.alpha),
        ],
    ))

    if args.skip_figures:
        steps.append({
            "module": "analysis.generate_figures",
            "status": "skipped",
            "reason": "--skip-figures",
        })
    else:
        steps.append(_run_module(
            "analysis.generate_figures",
            [
                "--results-dir", str(args.results_dir),
                "--out-dir", str(figures_dir),
                "--formats", *args.formats,
            ],
            allow_fail=args.allow_figure_fail,
        ))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "results_dir": str(args.results_dir),
        "matrix_dir": str(args.matrix_dir),
        "out_dir": str(out_dir),
        "paths": {
            "analysis": str(analysis_dir),
            "report": str(report_dir),
            "stats": str(stats_dir),
            "figures": str(figures_dir),
        },
        "options": {
            "formats": args.formats,
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "alpha": args.alpha,
            "skip_figures": args.skip_figures,
            "allow_figure_fail": args.allow_figure_fail,
        },
        "steps": steps,
    }

    manifest_path = out_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)
    print(f"Bundle ready at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
