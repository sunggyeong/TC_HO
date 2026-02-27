#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_pkg.figures

Generate paper-ready PNGs from exp_paper_summary.csv.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS: Dict[str, str] = {
    "Sustainable_DAG_NetworkX": "Oracle DAG (actual)",
    "Predicted_DAG_NetworkX": "Predicted DAG",
    "Reactive_MyopicGreedy": "Reactive greedy",
    "Lookahead_Greedy": "Lookahead greedy",
    "ShootingSearch_DeltaNode": "Shooting search",
    "Learned_TC_Offline": "TC (offline)",
    "Learned_TC_RTCorrected": "TC (realtime)",
}

def _mean(df: pd.DataFrame, metric: str) -> np.ndarray:
    if f"{metric}_mean" in df.columns:
        return df[f"{metric}_mean"].to_numpy(dtype=float)
    if metric in df.columns:
        return df[metric].to_numpy(dtype=float)
    raise KeyError(metric)

def _std(df: pd.DataFrame, metric: str) -> Optional[np.ndarray]:
    c = f"{metric}_std"
    return df[c].to_numpy(dtype=float) if c in df.columns else None

def plot_bar(df: pd.DataFrame, methods: List[str], metric: str, ylabel: str, out_path: Path, title: str, mult: float = 1.0, annfmt: str = "{:.2f}"):
    sub = df[df["Method"].isin(methods)].copy()
    sub["Method"] = pd.Categorical(sub["Method"], categories=methods, ordered=True)
    sub = sub.sort_values("Method")
    y = _mean(sub, metric) * mult
    yerr = _std(sub, metric)
    if yerr is not None: yerr = yerr * mult
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bars = ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in methods], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ylim = ax.get_ylim()
    rng = max(1e-9, ylim[1]-ylim[0])
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.015*rng, annfmt.format(h), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--outdir", default="paper_figures")
    ap.add_argument("--scenario", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)
    if args.scenario and "Scenario" in df.columns:
        df = df[df["Scenario"] == args.scenario].copy()

    order = [
        "Reactive_MyopicGreedy",
        "Lookahead_Greedy",
        "ShootingSearch_DeltaNode",
        "Predicted_DAG_NetworkX",
        "Sustainable_DAG_NetworkX",
        "Learned_TC_Offline",
        "Learned_TC_RTCorrected",
    ]
    methods = [m for m in order if m in set(df["Method"].tolist())]

    plot_bar(df, methods, "Availability", "Availability (%)", outdir/"fig_availability.png", "Availability", mult=100.0)
    plot_bar(df, methods, "EffectiveQoE", "Effective QoE", outdir/"fig_qoe.png", "Effective QoE", annfmt="{:.3f}")
    plot_bar(df, methods, "MeanInterruption_ms", "Mean interruption (ms)", outdir/"fig_interruption.png", "Interruption")
    plot_bar(df, methods, "MeanLatency_ms", "Mean latency (ms)", outdir/"fig_latency.png", "Latency")
    plot_bar(df, methods, "HO_Attempt_Count", "HO attempt count", outdir/"fig_ho_attempts.png", "HO attempts", annfmt="{:.1f}")
    plot_bar(df, methods, "PingPong_Ratio", "Ping-pong ratio (%)", outdir/"fig_pingpong.png", "Ping-pong ratio", mult=100.0)

if __name__ == "__main__":
    main()
