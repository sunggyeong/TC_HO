#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_pkg.train

Clean training runner for TC (Transformer + Consistency).
Scenarios: normal / paper_stress

Adds: --preset {stable,balanced,aggressive}

Preset intent:
- stable     : minimize HO/ping-pong (paper-safe)
- balanced   : best overall trade-off (default)
- aggressive : prioritize continuity/availability (may HO more)

Precedence:
1) Preset provides base values.
2) Any explicitly provided CLI values override the preset.
   (These args default to None so you can rely on preset defaults.)

Outputs:
- weights_tc_<scenario>.pth
- train_manifest_<scenario>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .env_core import ExperimentConfig
from .scenarios import apply_scenario
from . import tc_train_eval as tc


PRESETS = {
    "stable": {
        "w_switch": 0.25,
        "w_pingpong": 0.80,
        "w_jitter": 0.45,
        "rt_dwell": 4,
        "rt_hysteresis": 0.12,
        "rt_pingpong_window": 6,
        "rt_pingpong_extra": 0.18,
    },
    "balanced": {
        "w_switch": 0.18,
        "w_pingpong": 0.60,
        "w_jitter": 0.35,
        "rt_dwell": 3,
        "rt_hysteresis": 0.10,
        "rt_pingpong_window": 5,
        "rt_pingpong_extra": 0.15,
    },
    "aggressive": {
        "w_switch": 0.12,
        "w_pingpong": 0.45,
        "w_jitter": 0.25,
        "rt_dwell": 2,
        "rt_hysteresis": 0.07,
        "rt_pingpong_window": 4,
        "rt_pingpong_extra": 0.10,
    },
}


def _pick(user_val, preset_val):
    return preset_val if user_val is None else user_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="paper_stress", choices=["normal", "paper_stress"])
    ap.add_argument("--results_dir", default="results_paper")
    ap.add_argument("--weights_out", default=None)

    ap.add_argument("--train_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--val_seeds", nargs="+", type=int, default=[6, 7])
    ap.add_argument("--test_seeds", nargs="+", type=int, default=[8, 9, 10])

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)

    ap.add_argument("--preset", choices=["stable", "balanced", "aggressive"], default="balanced")

    # Penalty knobs (default None -> filled by preset)
    ap.add_argument("--w_switch", type=float, default=None)
    ap.add_argument("--w_pingpong", type=float, default=None)
    ap.add_argument("--w_jitter", type=float, default=None)

    # RT guardrails (default None -> filled by preset)
    ap.add_argument("--rt_dwell", type=int, default=None)
    ap.add_argument("--rt_hysteresis", type=float, default=None)
    ap.add_argument("--rt_pingpong_window", type=int, default=None)
    ap.add_argument("--rt_pingpong_extra", type=float, default=None)

    args = ap.parse_args()

    preset = PRESETS[args.preset]

    # Resolve effective hyperparams (preset base -> user overrides)
    eff = {
        "w_switch": _pick(args.w_switch, preset["w_switch"]),
        "w_pingpong": _pick(args.w_pingpong, preset["w_pingpong"]),
        "w_jitter": _pick(args.w_jitter, preset["w_jitter"]),
        "rt_dwell": _pick(args.rt_dwell, preset["rt_dwell"]),
        "rt_hysteresis": _pick(args.rt_hysteresis, preset["rt_hysteresis"]),
        "rt_pingpong_window": _pick(args.rt_pingpong_window, preset["rt_pingpong_window"]),
        "rt_pingpong_extra": _pick(args.rt_pingpong_extra, preset["rt_pingpong_extra"]),
    }

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    weights_out = args.weights_out or str(outdir / f"weights_tc_{args.scenario}.pth")

    exp_cfg = ExperimentConfig()
    exp_cfg.results_dir = str(outdir)
    apply_scenario(exp_cfg, args.scenario)

    cfg_te = tc.TrainEvalRealtimeConfig()
    cfg_te.exp_cfg = exp_cfg
    cfg_te.results_dir = str(outdir)
    cfg_te.weight_path = weights_out

    cfg_te.train_seeds = list(args.train_seeds)
    cfg_te.val_seeds = list(args.val_seeds)
    cfg_te.test_seeds = list(args.test_seeds)

    cfg_te.epochs = int(args.epochs)
    cfg_te.lr = float(args.lr)
    cfg_te.L = int(args.L)
    cfg_te.H = int(args.H)
    cfg_te.stride = int(args.stride)

    # penalties (to reduce excessive HO)
    cfg_te.reward_w_switch = float(eff["w_switch"])
    cfg_te.reward_w_pingpong = float(eff["w_pingpong"])
    cfg_te.reward_w_jitter = float(eff["w_jitter"])

    # realtime guardrails
    cfg_te.rt_enable_guardrails = True
    cfg_te.rt_min_dwell = int(eff["rt_dwell"])
    cfg_te.rt_hysteresis_ratio = float(eff["rt_hysteresis"])
    cfg_te.rt_pingpong_window = int(eff["rt_pingpong_window"])
    cfg_te.rt_pingpong_extra_hysteresis_ratio = float(eff["rt_pingpong_extra"])

    t0 = time.time()
    train_pairs = tc.collect_env_pairs(exp_cfg, cfg_te.train_seeds)
    val_pairs = tc.collect_env_pairs(exp_cfg, cfg_te.val_seeds)

    out = tc.train_on_seed_pool(train_pairs, cfg_te)
    # train_on_seed_pool may return (transformer, consistency, *extras)
    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            raise ValueError(f"train_on_seed_pool returned too few values: {len(out)}")
        transformer, consistency = out[0], out[1]
    elif isinstance(out, dict):
        transformer, consistency = out.get('transformer'), out.get('consistency')
        if transformer is None or consistency is None:
            raise ValueError("train_on_seed_pool returned dict without transformer/consistency")
    else:
        raise TypeError(f"Unexpected return type from train_on_seed_pool: {type(out)}")
    tc.save_weights(transformer, consistency, cfg_te.weight_path)

    manifest = {
        "scenario": args.scenario,
        "preset": args.preset,
        "effective": eff,
        "weights_out": cfg_te.weight_path,
        "seeds": {"train": cfg_te.train_seeds, "val": cfg_te.val_seeds, "test": cfg_te.test_seeds},
        "L": cfg_te.L,
        "H": cfg_te.H,
        "stride": cfg_te.stride,
        "epochs": cfg_te.epochs,
        "lr": cfg_te.lr,
        "elapsed_sec": time.time() - t0,
    }
    (outdir / f"train_manifest_{args.scenario}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"✅ preset={args.preset} effective={eff}")
    print(f"✅ saved weights: {cfg_te.weight_path}")
    print(f"✅ manifest: {outdir / f'train_manifest_{args.scenario}.json'}")
    print(f"Elapsed: {manifest['elapsed_sec']:.2f}s")


if __name__ == "__main__":
    main()
