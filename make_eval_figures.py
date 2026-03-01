#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate offline/online comparison figures from exp_paper_summary.csv. "
            "Creates per-metric bar charts with value labels, plus zoomed online charts."
        )
    )
    parser.add_argument(
        "--summary_csv",
        required=True,
        help="Path to exp_paper_summary.csv produced by eval.py",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for split CSVs and figures",
    )
    parser.add_argument(
        "--title_prefix",
        default="",
        help="Optional title prefix, e.g. 'H5 seed20 debug'",
    )
    parser.add_argument(
        "--offline_methods",
        nargs="*",
        default=[
            "Oracle_DAG_NetworkX",
            "Predicted_DAG_NetworkX",
            "Lookahead_Greedy",
            "ShootingSearch_DeltaNode",
            "Learned_TC_Offline",
        ],
        help="Methods to include in offline/planning-only figures, in preferred order.",
    )
    parser.add_argument(
        "--online_methods",
        nargs="*",
        default=[
            "Reactive_MyopicGreedy",
            "Predicted_DAG_MinSafeRT",
            "Predicted_DAG_RTWrapped",
            "Learned_TC_RTCorrected",
        ],
        help="Methods to include in online/system-level figures, in preferred order.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI",
    )
    return parser.parse_args()


def ordered_subset(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    sub = df[df["Method"].isin(methods)].copy()
    if sub.empty:
        return sub
    order_map = {m: i for i, m in enumerate(methods)}
    sub["__ord__"] = sub["Method"].map(order_map).fillna(999)
    sub = sub.sort_values(["__ord__", "Method"]).drop(columns="__ord__")
    return sub


def infer_metric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if not col.endswith("_mean"):
            continue
        if col in {"SamplingSteps_mean"}:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        cols.append(col)
    return cols


def metric_base(metric_col: str) -> str:
    return metric_col[:-5] if metric_col.endswith("_mean") else metric_col


def better_high_metrics() -> set[str]:
    return {
        "Availability",
        "EffectiveQoE",
        "DecisionCompletionRate",
    }


def better_low_metrics() -> set[str]:
    return {
        "MeanInterruptionMs",
        "P95InterruptionMs",
        "MeanLatencyMs",
        "P95LatencyMs",
        "MeanJitterMs",
        "P95JitterMs",
        "HO_Failure_Ratio",
        "PingPong_Ratio",
        "DeadlineMissRatio",
        "HO_Attempt_Count",
        "HO_Failure_Count",
    }


def sort_for_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    base = metric_base(metric_col)
    if base in better_high_metrics():
        ascending = False
    elif base in better_low_metrics():
        ascending = True
    else:
        # Default: larger-is-better metrics descending.
        ascending = False
    return df.sort_values(metric_col, ascending=ascending).copy()


def y_limits_for_zoom(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmax) * 0.05, 1e-3)
        return vmin - pad, vmax + pad
    span = vmax - vmin
    pad = max(span * 0.35, max(abs(vmax), 1.0) * 0.01)
    return vmin - pad, vmax + pad


def safe_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def add_value_labels(bars, values: Iterable[float]) -> None:
    for bar, val in zip(bars, values):
        if not np.isfinite(val):
            label = "nan"
        elif abs(val) >= 100:
            label = f"{val:.1f}"
        elif abs(val) >= 10:
            label = f"{val:.2f}"
        else:
            label = f"{val:.4f}"

        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=8,
            rotation=0,
        )


def make_bar_figure(
    df: pd.DataFrame,
    metric_col: str,
    group_name: str,
    outdir: Path,
    title_prefix: str,
    zoom: bool,
    dpi: int,
) -> None:
    if df.empty:
        return

    metric_name = metric_base(metric_col)
    values = df[metric_col].astype(float).to_numpy()
    methods = df["Method"].astype(str).tolist()
    std_col = f"{metric_name}_std"
    yerr = None
    if std_col in df.columns and pd.api.types.is_numeric_dtype(df[std_col]):
        std_vals = df[std_col].astype(float).to_numpy()
        if np.isfinite(std_vals).any():
            yerr = std_vals

    width = max(8, 1.35 * len(methods))
    plt.figure(figsize=(width, 5.6))
    bars = plt.bar(range(len(methods)), values, yerr=yerr, capsize=4 if yerr is not None else 0)
    plt.xticks(range(len(methods)), methods, rotation=30, ha="right")
    plt.ylabel(metric_name)

    title_parts = []
    if title_prefix:
        title_parts.append(title_prefix)
    title_parts.append(group_name)
    title_parts.append(metric_name)
    if zoom:
        title_parts.append("zoom")
    plt.title(" - ".join(title_parts))

    if zoom:
        ymin, ymax = y_limits_for_zoom(values)
        plt.ylim(ymin, ymax)

    add_value_labels(bars, values)
    plt.tight_layout()

    fname = f"{group_name.lower()}__{safe_filename(metric_name)}"
    if zoom:
        fname += "__zoom"
    plt.savefig(outdir / f"{fname}.png", dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    if "Method" not in df.columns:
        raise ValueError("summary csv must contain a 'Method' column")

    metric_cols = infer_metric_columns(df)
    if not metric_cols:
        raise ValueError("No '*_mean' metric columns found in the summary csv")

    offline_df = ordered_subset(df, args.offline_methods)
    online_df = ordered_subset(df, args.online_methods)

    offline_csv = outdir / "offline_summary_only.csv"
    online_csv = outdir / "online_summary_only.csv"
    offline_df.to_csv(offline_csv, index=False)
    online_df.to_csv(online_csv, index=False)

    fig_dir = outdir / "fig_all_metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric in metric_cols:
        if not offline_df.empty:
            off_sorted = sort_for_metric(offline_df, metric)
            make_bar_figure(
                off_sorted,
                metric,
                group_name="offline",
                outdir=fig_dir,
                title_prefix=args.title_prefix,
                zoom=False,
                dpi=args.dpi,
            )

        if not online_df.empty:
            on_sorted = sort_for_metric(online_df, metric)
            make_bar_figure(
                on_sorted,
                metric,
                group_name="online",
                outdir=fig_dir,
                title_prefix=args.title_prefix,
                zoom=False,
                dpi=args.dpi,
            )
            make_bar_figure(
                on_sorted,
                metric,
                group_name="online",
                outdir=fig_dir,
                title_prefix=args.title_prefix,
                zoom=True,
                dpi=args.dpi,
            )

    readme = outdir / "README_make_eval_figures.txt"
    readme.write_text(
        "Generated from summary CSV:\n"
        f"{summary_csv}\n\n"
        "Outputs:\n"
        "- offline_summary_only.csv\n"
        "- online_summary_only.csv\n"
        "- fig_all_metrics/*.png\n"
        "  * offline: one figure per metric\n"
        "  * online: normal + zoom figure per metric\n",
        encoding="utf-8",
    )

    print(f"[OK] summary: {summary_csv}")
    print(f"[OK] outdir: {outdir}")
    print(f"[OK] metrics: {len(metric_cols)}")
    print(f"[OK] offline rows: {len(offline_df)}")
    print(f"[OK] online rows: {len(online_df)}")
    print(f"[OK] figures: {fig_dir}")


if __name__ == "__main__":
    main()
