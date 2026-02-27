#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_pkg.train

Clean training runner.
- preset controls penalties + RT guardrails
- epochs can be small for debugging
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from .env_core import ExperimentConfig
from .scenarios import apply_scenario
from .presets import PRESETS
from . import tc_train_eval as tc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["normal","paper_stress"], default="paper_stress")
    ap.add_argument("--preset", choices=["stable","balanced","aggressive"], default="balanced")
    ap.add_argument("--results_dir", default="results_paper")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_sats", type=int, default=None, help="override cfg.max_sats")
    ap.add_argument("--train_seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--val_seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    outdir = Path(args.results_dir); outdir.mkdir(parents=True, exist_ok=True)

    exp_cfg = ExperimentConfig(); exp_cfg.results_dir = str(outdir)
    apply_scenario(exp_cfg, args.scenario)
    if args.max_sats is not None:
        exp_cfg.max_sats = int(args.max_sats)

    cfg_te = tc.TrainEvalRealtimeConfig()
    cfg_te.exp_cfg = exp_cfg
    cfg_te.results_dir = str(outdir)
    cfg_te.weight_path = str(outdir / f"weights_tc_{args.scenario}.pth")

    cfg_te.train_seeds = list(args.train_seeds)
    cfg_te.val_seeds = list(args.val_seeds)
    cfg_te.L = int(args.L); cfg_te.H = int(args.H); cfg_te.stride = int(args.stride)
    cfg_te.lr = float(args.lr); cfg_te.epochs = int(args.epochs)

    cfg_te.reward_w_switch = float(preset["w_switch"])
    cfg_te.reward_w_pingpong = float(preset["w_pingpong"])
    cfg_te.reward_w_jitter = float(preset["w_jitter"])

    cfg_te.rt_enable_guardrails = True
    cfg_te.rt_min_dwell = int(preset["rt_dwell"])
    cfg_te.rt_hysteresis_ratio = float(preset["rt_hysteresis"])
    cfg_te.rt_pingpong_window = int(preset["rt_pingpong_window"])
    cfg_te.rt_pingpong_extra_hysteresis_ratio = float(preset["rt_pingpong_extra"])

    # Enable pending + lookahead fallback by default
    cfg_te.rt_enable_pending = True
    cfg_te.rt_fallback_mode = "lookahead"
    cfg_te.rt_debug_log = False

    t0 = time.time()
    train_pairs = tc.collect_env_pairs(exp_cfg, cfg_te.train_seeds)
    # val_pairs collected for future use (optional)
    _ = tc.collect_env_pairs(exp_cfg, cfg_te.val_seeds)

    out = tc.train_on_seed_pool(train_pairs, cfg_te)
    if isinstance(out, (tuple, list)):
        transformer, consistency = out[0], out[1]
        hist = out[2] if len(out) > 2 else None
    else:
        raise TypeError("train_on_seed_pool returned unexpected type")

    tc.save_weights(transformer, consistency, cfg_te.weight_path)

    manifest = {
        "scenario": args.scenario,
        "preset": args.preset,
        "epochs": cfg_te.epochs,
        "max_sats": exp_cfg.max_sats,
        "train_seeds": cfg_te.train_seeds,
        "val_seeds": cfg_te.val_seeds,
        "L": cfg_te.L, "H": cfg_te.H, "stride": cfg_te.stride, "lr": cfg_te.lr,
        "penalties": {"w_switch": cfg_te.reward_w_switch, "w_pingpong": cfg_te.reward_w_pingpong, "w_jitter": cfg_te.reward_w_jitter},
        "rt_guardrails": {"dwell": cfg_te.rt_min_dwell, "hysteresis": cfg_te.rt_hysteresis_ratio,
                          "pingpong_window": cfg_te.rt_pingpong_window, "pingpong_extra": cfg_te.rt_pingpong_extra_hysteresis_ratio},
        "elapsed_sec": time.time() - t0,
    }
    (outdir / f"train_manifest_{args.scenario}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if hist is not None and hasattr(hist, "to_csv"):
        hist.to_csv(outdir / f"train_history_{args.scenario}.csv", index=False)

    print(f"✅ saved weights: {cfg_te.weight_path}")
    print(f"✅ manifest: {outdir / f'train_manifest_{args.scenario}.json'}")
    print(f"Elapsed: {manifest['elapsed_sec']:.2f}s")

if __name__ == "__main__":
    main()
