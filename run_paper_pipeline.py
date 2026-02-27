#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_paper_pipeline.py

One-shot runner (train -> eval -> figures) for the CLEAN paper_pkg.

Run from repo root (where paper_pkg/ exists).
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print("\n>>> " + " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["normal","paper_stress"], default="paper_stress")
    ap.add_argument("--preset", choices=["stable","balanced","aggressive"], default="balanced")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_sats", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[8,9,10])
    ap.add_argument("--results_paper", default="results_paper")
    ap.add_argument("--results_eval", default=None)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_figures", action="store_true")
    args = ap.parse_args()

    if not Path("paper_pkg").exists():
        raise FileNotFoundError("paper_pkg folder not found. Copy paper_pkg from paper_pkg_clean into repo root.")

    results_eval = args.results_eval or f"results_eval_{args.scenario}"

    weights = Path(args.results_paper) / f"weights_tc_{args.scenario}.pth"

    if not args.skip_train:
        cmd = [sys.executable, "-m", "paper_pkg.train",
               "--scenario", args.scenario,
               "--preset", args.preset,
               "--results_dir", args.results_paper,
               "--epochs", str(args.epochs)]
        if args.max_sats is not None:
            cmd += ["--max_sats", str(args.max_sats)]
        run(cmd)
    else:
        print("[skip] train")

    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    cmd = [sys.executable, "-m", "paper_pkg.eval",
           "--scenario", args.scenario,
           "--preset", args.preset,
           "--results_dir", results_eval,
           "--weights", str(weights),
           "--seeds", *[str(s) for s in args.seeds]]
    if args.max_sats is not None:
        cmd += ["--max_sats", str(args.max_sats)]
    run(cmd)

    if not args.skip_figures:
        summary = Path(results_eval) / "exp_paper_summary.csv"
        outdir = Path(results_eval) / "paper_figures"
        cmd = [sys.executable, "-m", "paper_pkg.figures",
               "--summary", str(summary),
               "--outdir", str(outdir),
               "--scenario", args.scenario]
        run(cmd)
    else:
        print("[skip] figures")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
