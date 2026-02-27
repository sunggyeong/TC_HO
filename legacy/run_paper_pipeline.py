#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_paper_pipeline.py

One-shot runner:
1) Train TC (Transformer+Consistency) with preset (debug epochs supported)
2) Evaluate baselines + DAGs + TC
3) Export paper figures (PNGs)

Run this from your repo root (the folder that contains `paper_pkg/`).

Examples (Windows / PowerShell):
  python run_paper_pipeline.py --scenario paper_stress --preset balanced --epochs 3

  python run_paper_pipeline.py --scenario normal --preset stable --epochs 12 --seeds 8 9 10

Outputs (by default):
  results_paper/weights_tc_<scenario>.pth
  results_eval_<scenario>/exp_paper_runs.csv
  results_eval_<scenario>/exp_paper_summary.csv
  results_eval_<scenario>/paper_figures/*.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["normal", "paper_stress"], default="paper_stress")
    ap.add_argument("--preset", choices=["stable", "balanced", "aggressive"], default="balanced")
    ap.add_argument("--epochs", type=int, default=3, help="debug: 3 epochs")
    ap.add_argument("--results_paper", default="results_paper", help="training outputs dir")
    ap.add_argument("--results_eval", default=None, help="eval outputs dir (default: results_eval_<scenario>)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[8, 9, 10])
    ap.add_argument("--skip_train", action="store_true", help="skip training and use existing weights")
    ap.add_argument("--skip_figures", action="store_true", help="skip png export")
    ap.add_argument("--weights", default=None, help="explicit weights path (overrides default)")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    pkg_dir = repo_root / "paper_pkg"
    if not pkg_dir.exists():
        raise FileNotFoundError(f"`paper_pkg/` not found in {repo_root}. Run from repo root.")

    results_paper = Path(args.results_paper)
    results_paper.mkdir(parents=True, exist_ok=True)

    results_eval = Path(args.results_eval or f"results_eval_{args.scenario}")
    results_eval.mkdir(parents=True, exist_ok=True)

    # Decide weight path
    default_weights = results_paper / f"weights_tc_{args.scenario}.pth"
    weights = Path(args.weights) if args.weights else default_weights

    # 1) Train
    if not args.skip_train:
        run([
            sys.executable, "-m", "paper_pkg.train",
            "--scenario", args.scenario,
            "--results_dir", str(results_paper),
            "--preset", args.preset,
            "--epochs", str(args.epochs),
        ])
    else:
        print("\n[skip] training")

    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Either run training (default) or pass --weights <path> or use --skip_train with an existing weights file."
        )

    # 2) Evaluate (baselines + DAGs + TC)
    run([
        sys.executable, "-m", "paper_pkg.eval",
        "--scenario", args.scenario,
        "--results_dir", str(results_eval),
        "--seeds", *[str(s) for s in args.seeds],
        "--weights", str(weights),
    ])

    # 3) Figures
    if not args.skip_figures:
        summary_csv = results_eval / "exp_paper_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")
        fig_dir = results_eval / "paper_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "-m", "paper_pkg.figures",
            "--summary", str(summary_csv),
            "--outdir", str(fig_dir),
            "--scenario", args.scenario,
        ])
    else:
        print("\n[skip] figures")

    print("\n✅ Pipeline complete.")
    print(f"- weights: {weights}")
    print(f"- eval dir: {results_eval}")
    print(f"- figures: {results_eval / 'paper_figures'}")


if __name__ == "__main__":
    main()
