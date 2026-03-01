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
    ap.add_argument("--epochs", type=int, default=30, help="paper: 30")
    ap.add_argument("--max_sats", type=int, default=None, help="override cfg.max_sats")
    ap.add_argument("--train_seeds", nargs="+", type=int, default=None,
                    help="paper: 0..14 (15). If not set, [0]")
    ap.add_argument("--val_seeds", nargs="+", type=int, default=None,
                    help="paper: 15..19 (5). If not set, [1]")
    ap.add_argument("--test_seeds", nargs="+", type=int, default=None,
                    help="paper eval: 20..29 (10). Stored in manifest only; use eval --seeds for runs")
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reward_learnable", action="store_true", help="learn reward penalty weights during training")

    # --- loss 비중 ---
    ap.add_argument("--alpha_oracle_loss",       type=float, default=0.3,
                    help="oracle point-regression loss weight (default 0.3)")
    ap.add_argument("--beta_pred_action_loss",   type=float, default=1.0,
                    help="expected-reward / soft-action loss weight (default 1.0)")
    ap.add_argument("--rollout_steps",              type=int,   default=2)
    ap.add_argument("--rollout_loss_weight",        type=float, default=0.35)
    ap.add_argument("--reward_time_offset_penalty", type=float, default=None,
                    help="offset 패널티 강도 (default: cfg 기본값 0.02). 올리면 모델이 낮은 offset 선호")
    ap.add_argument("--val_w_avail",     type=float, default=None, help="val_score: availability weight (default 0.50)")
    ap.add_argument("--val_w_latency",   type=float, default=None, help="val_score: latency term weight (default 0.18)")
    ap.add_argument("--val_w_interrupt", type=float, default=None, help="val_score: interruption term weight (default 0.12)")
    ap.add_argument("--val_w_jitter",    type=float, default=None, help="val_score: jitter term weight (default 0.08)")
    ap.add_argument("--val_w_pp",        type=float, default=None, help="val_score: pingpong penalty weight (default 0.25)")
    ap.add_argument("--val_w_hof",       type=float, default=None, help="val_score: HO failure penalty weight (default 0.25)")
    ap.add_argument("--val_w_ho",         type=float, default=None, help="val_score: HO attempt penalty weight (default 0.05)")
    ap.add_argument("--val_w_rt_accept",  type=float, default=None, help="val_score: RT acceptance rate bonus (default 0.05)")
    ap.add_argument("--val_w_rt_fallback",type=float, default=None, help="val_score: RT fallback rate penalty (default 0.03)")
    ap.add_argument("--val_w_rt_block",   type=float, default=None, help="val_score: guardrail block rate penalty (default 0.02)")
    ap.add_argument("--lambda_bc", type=float, default=None,
                    help="behavior cloning auxiliary loss weight")
    ap.add_argument("--rt_fallback_alpha_latency", type=float, default=None,
                    help="runtime fallback latency penalty weight")

    # --- RT guardrail overrides (preset 값 위에 덮어씀) ---
    ap.add_argument("--rt_min_dwell",                     type=int,   default=None,
                    help="override preset rt_dwell (slots)")
    ap.add_argument("--rt_hysteresis_ratio",              type=float, default=None,
                    help="override preset rt_hysteresis ratio")
    ap.add_argument("--rt_pingpong_window",               type=int,   default=None,
                    help="override preset rt_pingpong_window (slots)")
    ap.add_argument("--rt_pingpong_extra_hysteresis_ratio", type=float, default=None,
                    help="override preset rt_pingpong_extra ratio")
    ap.add_argument("--no_pending",  action="store_true",
                    help="disable pending scheduler (revert to immediate semantics)")
    ap.add_argument("--disable_mbb", action="store_true",
                    help="disable Make-Before-Break early execution")
    ap.add_argument("--rt_mbb_lookahead", type=int, default=None,
                    help="MBB lookahead slots (default: cfg 기본값 1)")
    ap.add_argument("--rt_pending_freeze_horizon", type=int,   default=None,
                    help="실행까지 이 슬롯 이하이면 pending 교체 금지 (default 1)")
    ap.add_argument("--rt_pending_advance_margin", type=int,   default=None,
                    help="새 exec_t + margin < 기존 exec_t 일 때만 pending 교체 허용 (default 1)")

    # --- history augmentation ---
    ap.add_argument("--disable_history_augmentation", action="store_true")
    ap.add_argument("--train_aug_flip_prob",    type=float, default=0.02)
    ap.add_argument("--train_aug_dropout_prob", type=float, default=0.06)
    ap.add_argument("--train_aug_phase_extra",  type=float, default=0.1)

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

    # 논문용 기본: 학습 15, 검증 5, 평가 10 (eval 시 --seeds 20 21 ... 29)
    cfg_te.train_seeds = list(args.train_seeds) if args.train_seeds is not None else list(range(15))
    cfg_te.val_seeds = list(args.val_seeds) if args.val_seeds is not None else list(range(15, 20))
    cfg_te.test_seeds = list(args.test_seeds) if args.test_seeds is not None else list(range(20, 30))
    cfg_te.L = int(args.L); cfg_te.H = int(args.H); cfg_te.stride = int(args.stride)
    cfg_te.lr = float(args.lr); cfg_te.epochs = int(args.epochs)

    cfg_te.reward_w_switch = float(preset["w_switch"])
    cfg_te.reward_w_pingpong = float(preset["w_pingpong"])
    cfg_te.reward_w_jitter = float(preset["w_jitter"])
    cfg_te.reward_learnable = getattr(args, "reward_learnable", False)
    cfg_te.alpha_oracle_loss = float(args.alpha_oracle_loss)
    cfg_te.beta_pred_action_loss = float(args.beta_pred_action_loss)
    cfg_te.rollout_steps = int(args.rollout_steps)
    cfg_te.rollout_loss_weight = float(args.rollout_loss_weight)
    if args.reward_time_offset_penalty is not None:
        cfg_te.reward_time_offset_penalty = float(args.reward_time_offset_penalty)
    # val score weights — CLI로 지정 시 cfg 기본값 위에 덮어씀
    if args.val_w_avail       is not None: cfg_te.val_w_avail       = float(args.val_w_avail)
    if args.val_w_latency     is not None: cfg_te.val_w_latency     = float(args.val_w_latency)
    if args.val_w_interrupt   is not None: cfg_te.val_w_interrupt   = float(args.val_w_interrupt)
    if args.val_w_jitter      is not None: cfg_te.val_w_jitter      = float(args.val_w_jitter)
    if args.val_w_pp          is not None: cfg_te.val_w_pp          = float(args.val_w_pp)
    if args.val_w_hof         is not None: cfg_te.val_w_hof         = float(args.val_w_hof)
    if args.val_w_ho          is not None: cfg_te.val_w_ho          = float(args.val_w_ho)
    if getattr(args, 'val_w_rt_accept',  None) is not None: cfg_te.val_w_rt_accept  = float(args.val_w_rt_accept)
    if getattr(args, 'val_w_rt_fallback',None) is not None: cfg_te.val_w_rt_fallback= float(args.val_w_rt_fallback)
    if getattr(args, 'val_w_rt_block',   None) is not None: cfg_te.val_w_rt_block   = float(args.val_w_rt_block)
    if args.lambda_bc is not None:
        cfg_te.lambda_bc = float(args.lambda_bc)
    if args.rt_fallback_alpha_latency is not None:
        cfg_te.rt_fallback_alpha_latency = float(args.rt_fallback_alpha_latency)
    cfg_te.train_use_history_augmentation = not bool(args.disable_history_augmentation)
    cfg_te.train_aug_flip_prob = float(args.train_aug_flip_prob)
    cfg_te.train_aug_dropout_prob = float(args.train_aug_dropout_prob)
    cfg_te.train_aug_phase_extra = float(args.train_aug_phase_extra)

    cfg_te.rt_enable_guardrails = True
    cfg_te.rt_min_dwell = int(preset["rt_dwell"])
    cfg_te.rt_hysteresis_ratio = float(preset["rt_hysteresis"])
    cfg_te.rt_pingpong_window = int(preset["rt_pingpong_window"])
    cfg_te.rt_pingpong_extra_hysteresis_ratio = float(preset["rt_pingpong_extra"])
    # CLI 오버라이드 (지정된 경우에만 preset 값 위에 덮어씀)
    if args.rt_min_dwell is not None:
        cfg_te.rt_min_dwell = int(args.rt_min_dwell)
    if args.rt_hysteresis_ratio is not None:
        cfg_te.rt_hysteresis_ratio = float(args.rt_hysteresis_ratio)
    if args.rt_pingpong_window is not None:
        cfg_te.rt_pingpong_window = int(args.rt_pingpong_window)
    if args.rt_pingpong_extra_hysteresis_ratio is not None:
        cfg_te.rt_pingpong_extra_hysteresis_ratio = float(args.rt_pingpong_extra_hysteresis_ratio)

    # Pending scheduler / MBB
    cfg_te.rt_enable_pending = not bool(args.no_pending)
    cfg_te.rt_enable_mbb = not bool(args.disable_mbb)
    if args.rt_mbb_lookahead is not None:
        cfg_te.rt_mbb_lookahead = int(args.rt_mbb_lookahead)
    if getattr(args, 'rt_pending_freeze_horizon',  None) is not None:
        cfg_te.rt_pending_freeze_horizon  = int(args.rt_pending_freeze_horizon)
    if getattr(args, 'rt_pending_advance_margin',  None) is not None:
        cfg_te.rt_pending_advance_margin  = int(args.rt_pending_advance_margin)
    cfg_te.rt_fallback_mode = "lookahead"
    cfg_te.rt_debug_log = False

    t0 = time.time()
    train_pairs = tc.collect_env_pairs(exp_cfg, cfg_te.train_seeds)
    val_pairs = tc.collect_env_pairs(exp_cfg, cfg_te.val_seeds)

    out = tc.train_on_seed_pool(train_pairs, cfg_te, val_pairs=val_pairs)
    if isinstance(out, (tuple, list)):
        transformer, consistency = out[0], out[1]
        hist = out[2] if len(out) > 2 else None
    else:
        raise TypeError("train_on_seed_pool returned unexpected type")

    # 가중치 저장은 train_on_seed_pool 내부에서 수행 (reward_learnable 시 reward_weights 포함)

    manifest = {
        "scenario": args.scenario,
        "preset": args.preset,
        "epochs": cfg_te.epochs,
        "max_sats": exp_cfg.max_sats,
        "train_seeds": cfg_te.train_seeds,
        "val_seeds": cfg_te.val_seeds,
        "test_seeds": cfg_te.test_seeds,
        "L": cfg_te.L, "H": cfg_te.H, "stride": cfg_te.stride, "lr": cfg_te.lr,
        "reward_learnable": cfg_te.reward_learnable,
        "alpha_oracle_loss": cfg_te.alpha_oracle_loss,
        "beta_pred_action_loss": cfg_te.beta_pred_action_loss,
        "rollout_steps": cfg_te.rollout_steps,
        "rollout_loss_weight": cfg_te.rollout_loss_weight,
        "reward_time_offset_penalty": cfg_te.reward_time_offset_penalty,
        # val_score_weights: 실제 적용된 가중치 전체 기록 (CLI/코드 어느 쪽에서 바꿔도 반영됨)
        "val_score_weights": {
            "w_avail":       cfg_te.val_w_avail,
            "w_latency":     cfg_te.val_w_latency,
            "w_interrupt":   getattr(cfg_te, "val_w_interrupt", 0.12),
            "w_jitter":      cfg_te.val_w_jitter,
            "w_pp":          cfg_te.val_w_pp,
            "w_hof":         cfg_te.val_w_hof,
            "w_ho":          cfg_te.val_w_ho,
            "w_rt_accept":   getattr(cfg_te, "val_w_rt_accept",   0.05),
            "w_rt_fallback": getattr(cfg_te, "val_w_rt_fallback", 0.03),
            "w_rt_block":    getattr(cfg_te, "val_w_rt_block",    0.02),
        },
        "rollout_steps":        cfg_te.rollout_steps,
        "rollout_loss_weight":  cfg_te.rollout_loss_weight,
        "lambda_bc": cfg_te.lambda_bc,
        "rt_fallback_alpha_latency": cfg_te.rt_fallback_alpha_latency,
        "train_use_history_augmentation": cfg_te.train_use_history_augmentation,
        "train_aug_flip_prob": cfg_te.train_aug_flip_prob,
        "train_aug_dropout_prob": cfg_te.train_aug_dropout_prob,
        "train_aug_phase_extra": cfg_te.train_aug_phase_extra,
        "penalties": {"w_switch": cfg_te.reward_w_switch, "w_pingpong": cfg_te.reward_w_pingpong, "w_jitter": cfg_te.reward_w_jitter},
        "rt_guardrails": {"dwell": cfg_te.rt_min_dwell, "hysteresis": cfg_te.rt_hysteresis_ratio,
                          "pingpong_window": cfg_te.rt_pingpong_window, "pingpong_extra": cfg_te.rt_pingpong_extra_hysteresis_ratio},
        "rt_enable_pending": cfg_te.rt_enable_pending,
        "rt_enable_mbb": cfg_te.rt_enable_mbb,
        "rt_mbb_lookahead": cfg_te.rt_mbb_lookahead,
        "rt_pending_freeze_horizon":  getattr(cfg_te, 'rt_pending_freeze_horizon',  1),
        "rt_pending_advance_margin":  getattr(cfg_te, 'rt_pending_advance_margin',  1),
        "elapsed_sec": time.time() - t0,
    }
    (outdir / f"train_manifest_{args.scenario}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if hist is not None and hasattr(hist, "to_csv"):
        hist.to_csv(outdir / f"train_history_{args.scenario}.csv", index=False)

    print(f"[OK] saved weights: {cfg_te.weight_path}")
    print(f"[OK] manifest: {outdir / f'train_manifest_{args.scenario}.json'}")
    print(f"Elapsed: {manifest['elapsed_sec']:.2f}s")

if __name__ == "__main__":
    main()
