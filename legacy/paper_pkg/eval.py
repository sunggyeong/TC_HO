#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_pkg.eval

Clean evaluation runner for ONLY two scenarios:
  - normal
  - paper_stress

Runs:
  - DAG baseline (actual)
  - Reactive myopic greedy (actual)
  - Lookahead greedy (pred horizon)
  - Shooting search (pred horizon, slow teacher)
  - Optional: Learned TC offline/realtime (if --weights provided)

Uses experiments_core + tc_core, so you can keep legacy files untouched.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .env_core import ExperimentConfig, build_env, sustainable_dag_plan, execute_plan_on_env, summarize_runs
from .scenarios import apply_scenario
from .baselines import (
    GuardrailConfig, ScoreConfig, LookaheadConfig, ShootingSearchConfig,
    plan_reactive_myopic, plan_lookahead_greedy, plan_shooting_search_delta_node,
)
from . import tc_train_eval as tc


def _load_tc(weights: str, exp_cfg: ExperimentConfig, outdir: Path):
    if weights is None:
        return None
    if not Path(weights).exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    tmp = build_env(exp_cfg, seed=0, predicted_view=False)
    _, N = np.asarray(tmp.A_final).shape

    cfg_te = tc.TrainEvalRealtimeConfig()
    cfg_te.exp_cfg = exp_cfg
    cfg_te.results_dir = str(outdir)
    cfg_te.weight_path = weights

    device = tc.get_device()
    transformer = tc.OnlineTransformerPredictor(num_nodes=N, L=cfg_te.L, H=cfg_te.H).to(device)
    consistency = tc.OnlineConsistencyGenerator(num_nodes=N, H=cfg_te.H).to(device)
    tc.load_weights(transformer, consistency, weights, device=device)
    transformer.eval(); consistency.eval()
    return cfg_te, transformer, consistency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="paper_stress", choices=["normal","paper_stress"])
    ap.add_argument("--results_dir", default="results_paper_eval")
    ap.add_argument("--seeds", nargs="+", type=int, default=[8,9,10])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--save_timelines", action="store_true")

    ap.add_argument("--alpha_latency", type=float, default=0.05)
    ap.add_argument("--dwell", type=int, default=3)
    ap.add_argument("--hysteresis", type=float, default=0.08)
    ap.add_argument("--pingpong_window", type=int, default=4)
    ap.add_argument("--pingpong_extra", type=float, default=0.12)

    ap.add_argument("--lookahead_H", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.97)

    ap.add_argument("--shoot_K", type=int, default=256)
    ap.add_argument("--shoot_switch_penalty", type=float, default=0.12)
    ap.add_argument("--shoot_outage_penalty", type=float, default=0.70)
    args = ap.parse_args()

    outdir = Path(args.results_dir); outdir.mkdir(parents=True, exist_ok=True)
    exp_cfg = ExperimentConfig(); exp_cfg.results_dir = str(outdir)
    apply_scenario(exp_cfg, args.scenario)

    guard = GuardrailConfig(args.dwell, args.hysteresis, 0.0, args.pingpong_window, args.pingpong_extra)
    score = ScoreConfig(args.alpha_latency)
    look = LookaheadConfig(args.lookahead_H, args.gamma, args.alpha_latency)
    shoot = ShootingSearchConfig(args.lookahead_H, args.gamma, args.alpha_latency, args.shoot_K, args.shoot_outage_penalty, args.shoot_switch_penalty)

    tc_bundle = _load_tc(args.weights, exp_cfg, outdir)
    if tc_bundle is not None:
        cfg_te, transformer, consistency = tc_bundle
        print(f"✅ loaded TC weights: {args.weights}")
    else:
        cfg_te = transformer = consistency = None
        print("ℹ️ baselines only (no TC).")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    first_seed = int(args.seeds[0])

    for seed in args.seeds:
        actual = build_env(exp_cfg, seed=seed, predicted_view=False)
        pred = build_env(exp_cfg, seed=seed, predicted_view=True)

        A_act, U_act, L_act = np.asarray(actual.A_final), np.asarray(actual.utility), np.asarray(actual.latency_ms)
        A_pred, U_pred, L_pred = np.asarray(pred.A_final), np.asarray(pred.utility), np.asarray(pred.latency_ms)

        # DAG
        plan = sustainable_dag_plan(actual, exp_cfg)
        tl, m = execute_plan_on_env(actual, plan, "Sustainable_DAG_NetworkX", "sustainable_dag", 0, exp_cfg)
        m["seed"] = seed; m["Scenario"] = args.scenario; rows.append(m)
        if args.save_timelines and seed == first_seed: tl.to_csv(outdir / f"timeline_dag_seed{seed}.csv", index=False)

        # Reactive greedy
        plan = plan_reactive_myopic(A_act, U_act, L_act, score=score, guard=guard)
        tl, m = execute_plan_on_env(actual, plan, "Reactive_MyopicGreedy", "reactive", 0, exp_cfg)
        m["seed"] = seed; m["Scenario"] = args.scenario; rows.append(m)
        if args.save_timelines and seed == first_seed: tl.to_csv(outdir / f"timeline_reactive_greedy_seed{seed}.csv", index=False)

        # Lookahead greedy
        plan = plan_lookahead_greedy(A_pred, U_pred, L_pred, look=look, guard=guard)
        tl, m = execute_plan_on_env(actual, plan, "Lookahead_Greedy", "reactive", 0, exp_cfg)
        m["seed"] = seed; m["Scenario"] = args.scenario; rows.append(m)
        if args.save_timelines and seed == first_seed: tl.to_csv(outdir / f"timeline_lookahead_greedy_seed{seed}.csv", index=False)

        # Shooting search
        plan = plan_shooting_search_delta_node(A_pred, U_pred, L_pred, search=shoot, guard=guard, seed=seed)
        tl, m = execute_plan_on_env(actual, plan, "ShootingSearch_DeltaNode", "reactive", 0, exp_cfg)
        m["seed"] = seed; m["Scenario"] = args.scenario; rows.append(m)
        if args.save_timelines and seed == first_seed: tl.to_csv(outdir / f"timeline_shooting_search_seed{seed}.csv", index=False)

        if cfg_te is not None:
            offline = tc.build_learned_offline_plan_from_pred_env(pred, transformer, consistency, cfg_te, cfg_te.consistency_steps_eval)
            tl, m = execute_plan_on_env(actual, offline, "Learned_TC_Offline", "consistency", cfg_te.consistency_steps_eval, exp_cfg)
            m["seed"] = seed; m["Scenario"] = args.scenario; rows.append(m)
            if args.save_timelines and seed == first_seed: tl.to_csv(outdir / f"timeline_tc_offline_seed{seed}.csv", index=False)

            rt = tc.build_learned_realtime_corrected_plan(actual, pred, transformer, consistency, cfg_te, cfg_te.consistency_steps_eval)
            tl, m = execute_plan_on_env(actual, rt, "Learned_TC_RTCorrected", "consistency", cfg_te.consistency_steps_eval, exp_cfg)
            m["seed"] = seed; m["Scenario"] = args.scenario; rows.append(m)
            if args.save_timelines and seed == first_seed: tl.to_csv(outdir / f"timeline_tc_rt_seed{seed}.csv", index=False)

    df_runs = pd.DataFrame(rows)
    df_summary = summarize_runs(df_runs)
    df_runs.to_csv(outdir / "exp_paper_runs.csv", index=False)
    df_summary.to_csv(outdir / "exp_paper_summary.csv", index=False)
    print(f"✅ saved: {outdir / 'exp_paper_runs.csv'}")
    print(f"✅ saved: {outdir / 'exp_paper_summary.csv'}")
    print(f"Elapsed: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
