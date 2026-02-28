#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_pkg.eval

Clean evaluation runner for the NEW paper experiment design.
Includes:
  - Oracle DAG (actual env)
  - Predicted DAG (pred env)
  - Reactive / Lookahead / Shooting baselines
  - TC (offline / realtime corrected)
"""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch


def _safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write CSV; if target is locked (e.g. open in Excel), write to fallback and warn."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except (PermissionError, OSError):
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        fallback = path.with_stem(path.stem + "_new")
        df.to_csv(fallback, index=False)
        print(f"[WARN] Permission denied on {path} (e.g. open in Excel). Wrote: {fallback.resolve()}")
        return fallback
    return path

from .env_core import ExperimentConfig, build_env, sustainable_dag_plan, execute_plan_on_env, summarize_runs
from .scenarios import apply_scenario
from .presets import PRESETS
from .baselines import GuardrailConfig, ScoreConfig, LookaheadConfig, ShootingSearchConfig, plan_reactive_myopic, plan_lookahead_greedy, plan_shooting_search_delta_node
from . import tc_train_eval as tc

def load_models(weights: str, exp_cfg: ExperimentConfig, outdir: Path):
    if weights is None:
        return None
    if not Path(weights).exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    device = tc.get_device()
    ckpt = torch.load(weights, map_location=device)
    # Use N,L,H from checkpoint so model matches saved weights (avoids size mismatch when env N differs)
    N = ckpt.get("num_nodes")
    if N is None:
        N = int(ckpt["transformer_state_dict"]["embedding.weight"].shape[1])
    L = ckpt.get("L") if ckpt.get("L") is not None else tc.TrainEvalRealtimeConfig().L
    H = ckpt.get("H") if ckpt.get("H") is not None else tc.TrainEvalRealtimeConfig().H
    cfg_te = tc.TrainEvalRealtimeConfig()
    cfg_te.exp_cfg = exp_cfg
    cfg_te.results_dir = str(outdir)
    cfg_te.weight_path = weights
    # L, H를 cfg_te에도 반영해야 plan builder들이 올바른 H로 condition 벡터를 만든다
    cfg_te.L = L
    cfg_te.H = H
    transformer = tc.OnlineTransformerPredictor(num_nodes=N, L=L, H=H).to(device)
    consistency = tc.OnlineConsistencyGenerator(num_nodes=N, H=H).to(device)
    transformer.load_state_dict(ckpt["transformer_state_dict"])
    consistency.load_state_dict(ckpt["consistency_state_dict"])
    transformer.eval(); consistency.eval()
    # Do not override exp_cfg.max_sats; plan builders will pad env history to N when env has fewer nodes
    return cfg_te, transformer, consistency

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["normal","paper_stress"], default="paper_stress")
    ap.add_argument("--preset", choices=["stable","balanced","aggressive"], default="balanced")
    ap.add_argument("--results_dir", default="results_eval")
    ap.add_argument("--seeds", nargs="+", type=int, default=[8,9,10])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--max_sats", type=int, default=None)
    ap.add_argument("--save_timelines", action="store_true")
    ap.add_argument("--rt_debug", action="store_true", help="print RTCorrected debug counters")
    ap.add_argument("--phase_breakdown", action="store_true")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    outdir = Path(args.results_dir); outdir.mkdir(parents=True, exist_ok=True)

    exp_cfg = ExperimentConfig(); exp_cfg.results_dir = str(outdir)
    apply_scenario(exp_cfg, args.scenario)
    if args.max_sats is not None:
        exp_cfg.max_sats = int(args.max_sats)

    # baseline configs from preset
    guard = GuardrailConfig(
        min_dwell=int(preset["baseline_dwell"]),
        hysteresis_ratio=float(preset["baseline_hysteresis"]),
        hysteresis_abs=0.0,
        pingpong_window=int(preset["baseline_pingpong_window"]),
        pingpong_extra_hysteresis_ratio=float(preset["baseline_pingpong_extra"]),
    )
    score = ScoreConfig(alpha_latency=float(preset["alpha_latency"]))
    look = LookaheadConfig(H=int(preset["lookahead_H"]), gamma=float(preset["gamma"]), alpha_latency=float(preset["alpha_latency"]))
    shoot = ShootingSearchConfig(H=int(preset["lookahead_H"]), gamma=float(preset["gamma"]), alpha_latency=float(preset["alpha_latency"]), K=int(preset["shoot_K"]))

    bundle = load_models(args.weights, exp_cfg, outdir) if args.weights else None
    if bundle:
        cfg_te, transformer, consistency = bundle
        print(f"[OK] loaded weights: {args.weights}")
        # Make sure RT uses lookahead fallback by default
        cfg_te.rt_fallback_mode = "lookahead"
        cfg_te.rt_enable_pending = True
        cfg_te.rt_debug_log = bool(args.rt_debug)
    else:
        cfg_te = transformer = consistency = None
        print("ℹ️ baselines only (no TC weights)")

    rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    first_seed = int(args.seeds[0])

    for seed in args.seeds:
        actual = build_env(exp_cfg, seed=seed, predicted_view=False)
        pred = build_env(exp_cfg, seed=seed, predicted_view=True)

        if args.phase_breakdown:
            phase_arr = getattr(actual, "phase", None)
            if phase_arr is not None:
                unique_ph = np.unique(phase_arr)
                for ph in unique_ph:
                    mask = (phase_arr == ph)
                    phase_rows.append({
                        "seed": seed,
                        "phase": str(ph),
                        "slots": int(mask.sum()),
                        "mean_visible_nodes": float(np.asarray(actual.A_final)[mask].sum(axis=1).mean()) if mask.any() else 0.0,
                    })

        A_act = np.asarray(actual.A_final); U_act = np.asarray(actual.utility); L_act = np.asarray(actual.latency_ms)
        A_pred = np.asarray(pred.A_final); U_pred = np.asarray(pred.utility); L_pred = np.asarray(pred.latency_ms)

        # Oracle DAG (actual)
        plan = sustainable_dag_plan(actual, exp_cfg)
        tl, m = execute_plan_on_env(actual, plan, "Sustainable_DAG_NetworkX", "sustainable_dag", 0, exp_cfg)
        m["seed"]=seed; m["Scenario"]=args.scenario; rows.append(m)
        if args.save_timelines and seed==first_seed: tl.to_csv(outdir/f"timeline_oracle_dag_seed{seed}.csv", index=False)

        # Predicted DAG
        plan = sustainable_dag_plan(pred, exp_cfg)
        tl, m = execute_plan_on_env(actual, plan, "Predicted_DAG_NetworkX", "sustainable_dag", 0, exp_cfg)
        m["seed"]=seed; m["Scenario"]=args.scenario; rows.append(m)
        if args.save_timelines and seed==first_seed: tl.to_csv(outdir/f"timeline_pred_dag_seed{seed}.csv", index=False)

        # Baselines
        plan = plan_reactive_myopic(A_act, U_act, L_act, score=score, guard=guard)
        tl, m = execute_plan_on_env(actual, plan, "Reactive_MyopicGreedy", "reactive", 0, exp_cfg)
        m["seed"]=seed; m["Scenario"]=args.scenario; rows.append(m)

        plan = plan_lookahead_greedy(A_pred, U_pred, L_pred, look=look, guard=guard)
        tl, m = execute_plan_on_env(actual, plan, "Lookahead_Greedy", "reactive", 0, exp_cfg)
        m["seed"]=seed; m["Scenario"]=args.scenario; rows.append(m)

        plan = plan_shooting_search_delta_node(A_pred, U_pred, L_pred, search=shoot, guard=guard, seed=seed)
        tl, m = execute_plan_on_env(actual, plan, "ShootingSearch_DeltaNode", "reactive", 0, exp_cfg)
        m["seed"]=seed; m["Scenario"]=args.scenario; rows.append(m)

        # TC
        if cfg_te is not None:
            offline = tc.build_learned_offline_plan_from_pred_env(pred, transformer, consistency, cfg_te, cfg_te.consistency_steps_eval)
            tl, m = execute_plan_on_env(actual, offline, "Learned_TC_Offline", "consistency", cfg_te.consistency_steps_eval, exp_cfg)
            m["seed"]=seed; m["Scenario"]=args.scenario; rows.append(m)

            rt = tc.build_learned_realtime_corrected_plan(actual, pred, transformer, consistency, cfg_te, cfg_te.consistency_steps_eval)
            tl, m = execute_plan_on_env(actual, rt, "Learned_TC_RTCorrected", "consistency", cfg_te.consistency_steps_eval, exp_cfg)
            m["seed"]=seed; m["Scenario"]=args.scenario; rows.append(m)

    df_runs = pd.DataFrame(rows)
    # runs 먼저 저장 (summary 계산/저장에서 오류 나도 runs는 남도록)
    out_runs = _safe_write_csv(df_runs, outdir / "exp_paper_runs.csv")
    print(f"[OK] saved runs: {out_runs.resolve()} (rows={len(df_runs)})")

    df_summary = summarize_runs(df_runs)
    out_summary = _safe_write_csv(df_summary, outdir / "exp_paper_summary.csv")
    print(f"[OK] saved summary: {out_summary.resolve()}")
    if args.phase_breakdown and len(phase_rows) > 0:
        df_phase = pd.DataFrame(phase_rows)
        out_phase = _safe_write_csv(df_phase, outdir / "exp_paper_phase_breakdown.csv")
        print(f"[OK] saved phase breakdown: {out_phase.resolve()}")
    print(f"Elapsed: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
