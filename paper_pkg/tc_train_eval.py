#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_pkg.tc_train_eval (CLEAN)

Core training/evaluation module for Transformer + Consistency (TC), with realtime guardrails.

This is the import-safe version:
- NO CLI main()
- Imports env/execution from paper_pkg.env_core
- Used by paper_pkg.train and paper_pkg.eval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any
import copy
import json
import time
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# === 기존 실험 파일에서 재사용 ===
from .env_core import (
    ExperimentConfig,
    build_env,
    execute_plan_on_env,
    sustainable_dag_plan,
    summarize_runs,
)

# === 모델 import (프로젝트 구조에 따라 둘 중 하나가 맞을 수 있음) ===
try:
    from models.proposed_tc_planner import OnlineTransformerPredictor, OnlineConsistencyGenerator
except ImportError:
    from proposed_tc_planner import OnlineTransformerPredictor, OnlineConsistencyGenerator


# =========================================================
# Config
# =========================================================

@dataclass
class TrainEvalRealtimeConfig:
    exp_cfg: ExperimentConfig = field(default_factory=ExperimentConfig)

    # 데이터 분할 (예시)
    train_seeds: List[int] = field(default_factory=lambda: [0])
    val_seeds: List[int] = field(default_factory=lambda: [1])
    test_seeds: List[int] = field(default_factory=lambda: [2])

    # 모델/학습
    L: int = 20
    H: int = 30
    epochs: int = 3
    stride: int = 5
    lr: float = 1e-3
    lambda_consistency: float = 1.0
    reward_invalid_penalty: float = -1.0

    # ---- 통합 보상(mini) 하이퍼파라미터 ----
    # (단일 항공기 시나리오이므로 LB는 제외)
    reward_gamma: float = 0.97                 # 할인계수 γ
    reward_w_utility: float = 1.00             # 가용 슬롯 utility 보상
    reward_w_outage: float = 0.70              # 비가용(outage) 패널티
    reward_w_switch: float = 0.05              # 핸드오버(스위치) 패널티
    reward_w_robustness: float = 0.20          # 후보 노드의 미래 가용성(robustness) 보너스
    reward_w_jitter: float = 0.15              # 짧은 유지시간 기반 지터 proxy 패널티
    reward_w_pingpong: float = 0.20            # A->B->A 형태 ping-pong proxy 패널티
    reward_jitter_min_hold_ratio: float = 0.35 # 이 비율보다 짧게 유지되면 지터 패널티 강화
    reward_use_discounted_return: bool = True  # 할인 누적보상 사용 여부
    reward_guided_target_scale: float = 1.0    # RL-loss 가중 스케일 (보상 기반)
    reward_guided_target_min_weight: float = 0.1
    reward_guided_target_max_weight: float = 3.0

    # 평가 시 consistency sampling steps
    consistency_steps_eval: int = 2

    # 실시간 보정 (online correction)
    rt_reliability_alpha: float = 0.90  # EMA: 클수록 느리게 변함
    rt_low_reliability_threshold: float = 0.45
    rt_fp_extra_penalty: float = 0.90    # false positive(예측=1 실제=0) 추가 감쇠
    rt_use_actual_current_slot_fallback: bool = True


    # ---- 실시간 HO 안정화 가드레일(핑퐁/과도한 HO 억제) ----
    # enable_guardrails=False로 두면 기존 동작(매 슬롯 즉시 보정/스위치)을 그대로 유지
    rt_enable_guardrails: bool = True

    # (1) 최소 유지시간: 최근 스위치 후 rt_min_dwell 슬롯 동안은 스위치 금지(현재 연결이 유효한 경우)
    rt_min_dwell: int = 3

    # (2) 히스테리시스: 스위치하려면 현재 대비 충분한 이득이 있어야 함
    # - ratio 기준: score_new >= score_cur * (1 + rt_hysteresis_ratio)
    # - abs 기준: (선택) score_new - score_cur >= rt_hysteresis_abs
    rt_hysteresis_ratio: float = 0.08
    rt_hysteresis_abs: float = 0.0

    # (3) 핑퐁(A->B->A) 억제: 최근 rt_pingpong_window 슬롯 내 되돌림이면 더 큰 히스테리시스 요구
    rt_pingpong_window: int = 4
    rt_pingpong_extra_hysteresis_ratio: float = 0.12

    # ---- Realtime planning enhancements ----
    rt_enable_pending: bool = True
    rt_fallback_mode: str = "lookahead"  # myopic | lookahead
    rt_fallback_alpha_latency: float = 0.05
    rt_fallback_gamma: float = 0.97
    rt_fallback_H: int = 30
    rt_debug_log: bool = False

    # 저장
    results_dir: str = "results"
    weight_path: str = "results/trained_weights_rl_real_env.pth"
    split_manifest_path: str = "results/train_val_test_seed_split.json"


# =========================================================
# Utilities
# =========================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def update_ema_target(online_net, target_net, tau=0.05):
    with torch.no_grad():
        for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


def save_weights(transformer, consistency, weight_path: str):
    Path(weight_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "transformer_state_dict": transformer.state_dict(),
            "consistency_state_dict": consistency.state_dict(),
        },
        weight_path,
    )
    print(f"✅ weights saved: {weight_path}")


def load_weights(transformer, consistency, weight_path: str, device: torch.device):
    ckpt = torch.load(weight_path, map_location=device)
    transformer.load_state_dict(ckpt["transformer_state_dict"])
    consistency.load_state_dict(ckpt["consistency_state_dict"])


# =========================================================
# Seed collection (actual / predicted env pair)
# =========================================================

def collect_env_pairs(cfg: ExperimentConfig, seeds: List[int]) -> List[Dict[str, Any]]:
    pairs = []
    for seed in seeds:
        print(f"[collect] seed={seed}")
        actual_env = build_env(cfg, seed=seed, predicted_view=False)
        pred_env = build_env(cfg, seed=seed, predicted_view=True)

        # 기본 shape 체크
        if actual_env.A_final.shape != pred_env.A_final.shape:
            raise ValueError(f"shape mismatch at seed={seed}: actual {actual_env.A_final.shape} vs pred {pred_env.A_final.shape}")
        if actual_env.A_final.shape[1] != pred_env.A_final.shape[1]:
            raise ValueError(f"N mismatch at seed={seed}")

        pairs.append(
            {
                "seed": seed,
                "actual_env": actual_env,
                "pred_env": pred_env,
            }
        )
    return pairs



# =========================================================
# Reward helpers (mini integrated reward for single-aircraft)
# =========================================================

def _best_available_node(A_row: np.ndarray, U_row: np.ndarray) -> int:
    avail = np.where(A_row == 1)[0]
    if len(avail) == 0:
        return -1
    return int(avail[np.argmax(U_row[avail])])


def _contiguous_available_run_len(future_A_true: np.ndarray, start_i: int, node_idx: int) -> int:
    """start_i부터 연속 가용(1)인 길이. 첫 슬롯이 0이면 0."""
    H, _ = future_A_true.shape
    run_len = 0
    for tau in range(int(start_i), H):
        if int(future_A_true[tau, node_idx]) != 1:
            break
        run_len += 1
    return int(run_len)


def _discounted_integrated_reward_single_aircraft(
    future_A_true: np.ndarray,   # (H, N)
    future_U_true: np.ndarray,   # (H, N)
    target_time_offset: int,
    target_node_idx: int,
    current_node: int,
    cfg_te: TrainEvalRealtimeConfig,
    prev_node: int = -1,
) -> float:
    """
    논문식 통합 보상의 '미니 버전' (단일 항공기):
      - utility 항 (정규화된 utility 사용)
      - outage 패널티
      - switch(핸드오버) 비용
      - robustness 보너스(선택 노드의 미래 가용 비율)
      - jitter proxy 패널티 (짧은 유지시간에 스위치할수록 불리)
      - ping-pong proxy 패널티 (A->B->A)
    ※ Load Balance(LB)는 단일 항공기라 제외
    """
    H, N = future_A_true.shape
    i = int(np.clip(target_time_offset, 0, H - 1))
    n = int(np.clip(target_node_idx, 0, N - 1))

    gamma = float(cfg_te.reward_gamma)
    w_u = float(cfg_te.reward_w_utility)
    w_o = float(cfg_te.reward_w_outage)
    w_s = float(cfg_te.reward_w_switch)
    w_r = float(cfg_te.reward_w_robustness)
    w_j = float(getattr(cfg_te, "reward_w_jitter", 0.0))
    w_p = float(getattr(cfg_te, "reward_w_pingpong", 0.0))
    jitter_thr = float(getattr(cfg_te, "reward_jitter_min_hold_ratio", 0.35))

    if i >= H:
        return float(cfg_te.reward_invalid_penalty)

    robustness = float(np.mean(future_A_true[i:, n])) if i < H else 0.0
    total = 0.0

    # 선택 시점에 1회 스위치 비용
    switch_happens = bool(current_node != -1 and current_node != n)
    if switch_happens:
        total -= w_s

        # ---- Jitter proxy: 스위치했는데 곧바로 끊길수록 불리 ----
        run_len = _contiguous_available_run_len(future_A_true, i, n)
        horizon_left = max(1, H - i)
        hold_ratio = float(run_len) / float(horizon_left)  # 0~1
        # threshold 이하면 선형 패널티, 이상이면 0
        jitter_pen = max(0.0, (jitter_thr - hold_ratio) / max(1e-6, jitter_thr))
        total -= (w_j * jitter_pen)

        # ---- Ping-pong proxy: 직전 노드(prev_node)로 되돌아가는 A->B->A 패턴 ----
        # prev_node --(이전 슬롯)--> current_node --(이번 결정)--> n
        if prev_node != -1 and current_node != -1 and prev_node == n and current_node != n:
            total -= w_p

    for tau in range(i, H):
        a = float(future_A_true[tau, n])
        u = float(future_U_true[tau, n])
        inst = (w_u * u * a) - (w_o * (1.0 - a))
        if tau == i:
            inst += (w_r * robustness)
        if cfg_te.reward_use_discounted_return:
            total += (gamma ** (tau - i)) * inst
        else:
            total += inst

    return float(total)


def _find_best_reward_guided_target(
    future_A_true: np.ndarray,    # (H, N)
    future_U_true: np.ndarray,    # (H, N)
    current_node: int,
    cfg_te: TrainEvalRealtimeConfig,
    prev_node: int = -1,
) -> Tuple[int, int, float]:
    """
    미래 H-step 후보 (time_offset, node)를 스캔해서 통합보상(mini) 기준 최적 후보를 선택.
    유효 후보가 없으면 (0,0, invalid_penalty) 반환.
    """
    H, N = future_A_true.shape
    best_i, best_n = 0, 0
    best_r = -1e18

    any_valid = False
    for i in range(H):
        valid_nodes = np.where(future_A_true[i] == 1)[0]
        if len(valid_nodes) == 0:
            continue
        any_valid = True
        for n in valid_nodes:
            r = _discounted_integrated_reward_single_aircraft(
                future_A_true=future_A_true,
                future_U_true=future_U_true,
                target_time_offset=i,
                target_node_idx=int(n),
                current_node=current_node,
                cfg_te=cfg_te,
            )
            if r > best_r:
                best_r = float(r)
                best_i, best_n = int(i), int(n)

    if not any_valid:
        return 0, 0, float(cfg_te.reward_invalid_penalty)

    return best_i, best_n, float(best_r)


def _reward_to_supervision_weight(reward_val: float, cfg_te: TrainEvalRealtimeConfig) -> float:
    """
    보상값을 안정적인 supervised RL-loss 가중치로 변환.
    """
    w = 1.0 + cfg_te.reward_guided_target_scale * math.tanh(reward_val / 2.0)
    w = float(np.clip(w, cfg_te.reward_guided_target_min_weight, cfg_te.reward_guided_target_max_weight))
    return w


# =========================================================
# Training (predicted history -> actual future target)
# =========================================================

def train_on_seed_pool(
    env_pairs: List[Dict[str, Any]],
    cfg_te: TrainEvalRealtimeConfig,
):
    device = get_device()
    print(f"🚀 training start | device={device}")

    # num_nodes 고정
    first = env_pairs[0]
    _, N = first["actual_env"].A_final.shape

    L, H = cfg_te.L, cfg_te.H
    transformer = OnlineTransformerPredictor(num_nodes=N, L=L, H=H).to(device)
    online_consistency = OnlineConsistencyGenerator(num_nodes=N, H=H).to(device)

    target_consistency = copy.deepcopy(online_consistency).to(device)
    target_consistency.eval()
    for p in target_consistency.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(
        list(transformer.parameters()) + list(online_consistency.parameters()),
        lr=cfg_te.lr,
    )
    bce_loss_fn = nn.BCELoss()

    history_rows = []
    t0 = time.time()

    for epoch in range(cfg_te.epochs):
        transformer.train()
        online_consistency.train()

        epoch_loss = 0.0
        epoch_loss_tf = 0.0
        epoch_loss_rl = 0.0
        epoch_loss_cs = 0.0
        epoch_reward = 0.0
        step_count = 0

        for pair in env_pairs:
            actual_env = pair["actual_env"]
            pred_env = pair["pred_env"]

            A_in = np.asarray(pred_env.A_final, dtype=np.float32)        # 입력 = predicted
            A_target = np.asarray(actual_env.A_final, dtype=np.float32)  # 타깃 = actual
            U_target = np.asarray(actual_env.utility, dtype=np.float32)  # 보상 = actual utility

            T, N2 = A_in.shape
            if N2 != N:
                raise ValueError(f"num_nodes mismatch in training: expected {N}, got {N2}")

            # 슬라이딩 윈도우
            for t in range(L, T - H, cfg_te.stride):
                history_A = A_in[t - L : t, :]
                future_A_true = A_target[t : t + H, :]

                state_tensor = torch.tensor(history_A, dtype=torch.float32, device=device).unsqueeze(0)
                future_tensor_true = torch.tensor(future_A_true, dtype=torch.float32, device=device).unsqueeze(0)

                optimizer.zero_grad()

                # 1) Transformer: predicted history -> actual future coverage
                future_pred = transformer(state_tensor)
                loss_tf = bce_loss_fn(future_pred, future_tensor_true)

                # 2) Consistency action generation (continuous target: [time_ratio, node_ratio])
                condition = future_pred.view(1, -1).detach()
                y_noisy_t2 = torch.randn(1, 2, device=device)
                y_noisy_t1 = y_noisy_t2 * 0.5

                step_2 = torch.tensor([[2.0]], device=device)
                step_1 = torch.tensor([[1.0]], device=device)

                y_curr = online_consistency(y_noisy_t1, condition, step_1)

                # 3) Reward-guided target (mini integrated reward, single-aircraft)
                # 현재/직전 노드 근사: t-1, t-2 슬롯에서 actual utility가 최대인 가용 노드
                if t - 1 >= 0:
                    current_node_approx = _best_available_node(A_target[t - 1], U_target[t - 1])
                else:
                    current_node_approx = -1
                if t - 2 >= 0:
                    prev_node_approx = _best_available_node(A_target[t - 2], U_target[t - 2])
                else:
                    prev_node_approx = -1

                best_i, best_n, reward_val = _find_best_reward_guided_target(
                    future_A_true=future_A_true,
                    future_U_true=U_target[t : t + H, :],
                    current_node=current_node_approx,
                    cfg_te=cfg_te,
                    prev_node=prev_node_approx,
                )
                epoch_reward += reward_val

                # reward-guided normalized target y* = [time_ratio, node_ratio]
                y_star = torch.tensor(
                    [[
                        0.0 if H <= 1 else (best_i / float(H - 1)),
                        0.0 if N <= 1 else (best_n / float(N - 1)),
                    ]],
                    dtype=torch.float32,
                    device=device,
                )
                rl_weight = _reward_to_supervision_weight(reward_val, cfg_te)
                loss_rl = rl_weight * F.mse_loss(y_curr, y_star)

                # 4) Consistency self-consistency loss (EMA target)
                with torch.no_grad():
                    y_target = target_consistency(y_noisy_t2, condition, step_2)
                loss_cs = F.mse_loss(y_curr, y_target)

                total_loss = loss_tf + loss_rl + (cfg_te.lambda_consistency * loss_cs)
                total_loss.backward()
                optimizer.step()
                update_ema_target(online_consistency, target_consistency)

                epoch_loss += float(total_loss.item())
                epoch_loss_tf += float(loss_tf.item())
                epoch_loss_rl += float(loss_rl.item())
                epoch_loss_cs += float(loss_cs.item())
                step_count += 1

        row = {
            "epoch": epoch + 1,
            "steps": step_count,
            "avg_total_loss": epoch_loss / max(1, step_count),
            "avg_tf_loss": epoch_loss_tf / max(1, step_count),
            "avg_rl_loss": epoch_loss_rl / max(1, step_count),
            "avg_cs_loss": epoch_loss_cs / max(1, step_count),
            "avg_reward": epoch_reward / max(1, step_count),
            "elapsed_sec": time.time() - t0,
        }
        history_rows.append(row)
        print(
            f"[train] epoch {epoch+1:02d}/{cfg_te.epochs} | "
            f"loss={row['avg_total_loss']:.4f} (tf={row['avg_tf_loss']:.4f}, rl={row['avg_rl_loss']:.4f}, cs={row['avg_cs_loss']:.4f}) | "
            f"reward={row['avg_reward']:.4f} | elapsed={row['elapsed_sec']:.1f}s"
        )

    transformer.eval()
    online_consistency.eval()

    save_weights(transformer, online_consistency, cfg_te.weight_path)

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(Path(cfg_te.results_dir) / "train_history_real_env.csv", index=False)
    print(f"✅ train history saved: {Path(cfg_te.results_dir) / 'train_history_real_env.csv'}")

    return transformer, online_consistency, hist_df


# =========================================================
# Inference helpers (learned planner)
# =========================================================

def _predict_target_from_history(
    transformer,
    consistency,
    history_A: np.ndarray,
    H: int,
    N: int,
    device: torch.device,
    consistency_steps: int = 2,
) -> Tuple[int, int]:
    """
    history_A: (L, N) binary matrix
    return: (target_time_offset, target_node_idx)
    """
    state_tensor = torch.tensor(history_A, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        future_pred = transformer(state_tensor)
        condition = future_pred.view(1, -1)

        y_curr = torch.randn(1, 2, device=device)

        # configurable few-step consistency
        steps = max(1, int(consistency_steps))
        for s in range(steps, 0, -1):
            step_tensor = torch.tensor([[float(s)]], dtype=torch.float32, device=device)
            y_curr = consistency(y_curr, condition, step_tensor)

        target_time_offset = int(y_curr[0, 0].item() * (H - 1))
        target_node_idx = int(y_curr[0, 1].item() * (N - 1))

        target_time_offset = min(max(target_time_offset, 0), H - 1)
        target_node_idx = min(max(target_node_idx, 0), N - 1)

    return target_time_offset, target_node_idx


def build_learned_offline_plan_from_pred_env(
    pred_env,
    transformer,
    consistency,
    cfg_te: TrainEvalRealtimeConfig,
    consistency_steps: int,
) -> np.ndarray:
    """
    predicted env만 보고 만든 learned plan (offline planning)
    """
    device = next(transformer.parameters()).device
    A_pred = np.asarray(pred_env.A_final)
    U_pred = np.asarray(pred_env.utility)
    T, N = A_pred.shape

    L, H = cfg_te.L, cfg_te.H
    planned_idx = np.full(T, -1, dtype=int)
    current_node = -1

    for t in range(T):
        if t < L:
            available = np.where(A_pred[t] == 1)[0]
            if len(available) > 0:
                current_node = int(available[np.argmax(U_pred[t, available])])
            else:
                current_node = -1
            planned_idx[t] = current_node
            continue

        history_A = A_pred[t - L : t, :]
        target_time_offset, target_node_idx = _predict_target_from_history(
            transformer=transformer,
            consistency=consistency,
            history_A=history_A,
            H=H,
            N=N,
            device=device,
            consistency_steps=consistency_steps,
        )

        # 현재 시점 타이밍이면 switch 제안
        if target_time_offset == 0 and A_pred[t, target_node_idx] == 1:
            current_node = int(target_node_idx)

        # predicted view에서 이미 끊긴 경우
        if current_node != -1 and A_pred[t, current_node] == 0:
            current_node = -1

        planned_idx[t] = current_node

    return planned_idx


def build_learned_realtime_corrected_plan(
    actual_env,
    pred_env,
    transformer,
    consistency,
    cfg_te: TrainEvalRealtimeConfig,
    consistency_steps: int,
) -> np.ndarray:
    """
    실시간 보정 버전 (receding-horizon, MPC 스타일):
    - 모델은 기본적으로 predicted history를 사용
    - 지나간 슬롯은 실제 관측(actual)로 덮어씀 (observed prefix correction)
    - 슬롯마다 node reliability EMA 갱신
    - 저신뢰/비가용 제안은 현재 실제 가용 후보로 fallback
    - (옵션) HO 안정화 가드레일: dwell + hysteresis + ping-pong 억제
      -> 과도한 HO/핑퐁으로 latency/HO attempts가 폭증하는 현상을 완화
    """
    device = next(transformer.parameters()).device

    A_actual = np.asarray(actual_env.A_final)
    U_actual = np.asarray(actual_env.utility)
    A_pred = np.asarray(pred_env.A_final)
    U_pred = np.asarray(pred_env.utility)

    T, N = A_actual.shape
    L, H = cfg_te.L, cfg_te.H

    planned_idx = np.full(T, -1, dtype=int)
    current_node = -1

    # 과거 관측 누적 반영용 (처음엔 predicted로 시작, 지나간 슬롯은 actual로 덮어씀)
    A_obs_corrected = A_pred.copy()
    # node 신뢰도 (0~1)
    reliability = np.ones(N, dtype=np.float32)

    # 마지막 스위치 시각(슬롯 인덱스)
    last_switch_t = -10**9

    def _score(t_: int, n_: int) -> float:
        # utility * reliability (둘 다 0~대략 1 scale 기대)
        return float(U_actual[t_, n_] * reliability[n_])

    for t in range(T):
        # 현재 슬롯까지는 실제 관측값 사용 가능하다고 가정(실시간 측정)
        A_obs_corrected[t] = A_actual[t]

        prev1 = int(planned_idx[t - 1]) if t - 1 >= 0 else -1
        prev2 = int(planned_idx[t - 2]) if t - 2 >= 0 else -1

        # 현재 연결의 유효성(현재 슬롯에서 커버리지/정책상 가능?)
        current_valid_now = (current_node >= 0 and current_node < N and A_actual[t, current_node] == 1)

        # ------------------------ (A) 기본 제안 산출 ------------------------
        if t < L:
            # warmup: 현재 실제 가용 후보 중 신뢰도 반영 점수 최대 선택
            avail = np.where(A_actual[t] == 1)[0]
            if len(avail) > 0:
                score = U_actual[t, avail] * reliability[avail]
                proposed = int(avail[np.argmax(score)])
            else:
                proposed = -1

            corrected = proposed

        else:
            # history는 "과거 actual 관측 반영 + (미래는 여전히 predicted)"
            history_A = A_obs_corrected[t - L : t, :]

            target_time_offset, target_node_idx = _predict_target_from_history(
                transformer=transformer,
                consistency=consistency,
                history_A=history_A,
                H=H,
                N=N,
                device=device,
                consistency_steps=consistency_steps,
            )

            proposed = current_node
            if target_time_offset == 0:
                proposed = int(target_node_idx)

            # 실시간 보정 로직 (기존)
            proposed_valid_now = (proposed is not None and proposed >= 0 and proposed < N and A_actual[t, proposed] == 1)

            low_rel = False
            if proposed is not None and proposed >= 0 and proposed < N:
                low_rel = (float(reliability[proposed]) < cfg_te.rt_low_reliability_threshold)

            corrected = current_node

            if proposed_valid_now and not low_rel:
                corrected = proposed
            else:
                # proposed가 지금 안되거나 / 신뢰도 너무 낮으면 fallback
                if cfg_te.rt_use_actual_current_slot_fallback:
                    avail = np.where(A_actual[t] == 1)[0]
                    if len(avail) > 0:
                        # 신뢰도 반영 + 현재 actual utility 기반 즉시 보정
                        score = U_actual[t, avail] * reliability[avail]
                        fallback = int(avail[np.argmax(score)])

                        # 현재 연결 유지가 유효하고 제안이 low_rel이면 유지를 우선할 수도 있음
                        if current_valid_now and low_rel:
                            corrected = current_node
                        else:
                            corrected = fallback
                    else:
                        corrected = -1 if not current_valid_now else current_node
                else:
                    corrected = current_node if current_valid_now else -1

        # ------------------------ (B) HO 안정화 가드레일 ------------------------
        if cfg_te.rt_enable_guardrails:
            # 최종 후보 유효성
            corrected_valid_now = (corrected is not None and corrected >= 0 and corrected < N and A_actual[t, corrected] == 1)

            # 스위치 시도인지?
            switching = (corrected_valid_now and current_valid_now and corrected != current_node)

            if switching:
                # (1) Dwell: 최근 스위치 후 일정 시간은 유지
                if (t - last_switch_t) < int(cfg_te.rt_min_dwell):
                    corrected = current_node
                else:
                    # (2) Hysteresis: 충분한 이득이 있을 때만 스위치
                    s_cur = _score(t, current_node)
                    s_new = _score(t, corrected)

                    required_ratio = float(cfg_te.rt_hysteresis_ratio)

                    # (3) Ping-pong(A->B->A) 억제: t-2 노드로 되돌아가려 하면 더 큰 히스테리시스 요구
                    is_pingpong = (corrected == prev2 and prev2 != -1 and prev1 == current_node and (t - last_switch_t) <= int(cfg_te.rt_pingpong_window))
                    if is_pingpong:
                        required_ratio += float(cfg_te.rt_pingpong_extra_hysteresis_ratio)

                    ratio_ok = (s_new >= s_cur * (1.0 + required_ratio))
                    if float(cfg_te.rt_hysteresis_abs) > 0.0:
                        abs_ok = ((s_new - s_cur) >= float(cfg_te.rt_hysteresis_abs))
                        ok = ratio_ok and abs_ok
                    else:
                        ok = ratio_ok

                    if not ok:
                        corrected = current_node

            # 현재 연결이 유효한데 corrected가 -1이면 끊김 방지로 유지(단, 실제로 유효할 때만)
            if corrected == -1 and current_valid_now:
                corrected = current_node

        # ------------------------ (C) 최종 적용 & 스위치 기록 ------------------------
        # 최종 유효성 재검사
        if corrected != -1 and (corrected < 0 or corrected >= N or A_actual[t, corrected] != 1):
            corrected = -1

        # 스위치 발생 시각 기록 (연결이 유효한 상태에서 다른 유효 노드로 바뀐 경우)
        if corrected != current_node and corrected != -1:
            last_switch_t = t

        current_node = int(corrected) if corrected is not None else -1
        planned_idx[t] = current_node

        # ---- reliability update (현재 슬롯 관측으로 다음 슬롯 planning 보정에 반영) ----
        match = (A_pred[t].astype(np.int32) == A_actual[t].astype(np.int32)).astype(np.float32)
        reliability = cfg_te.rt_reliability_alpha * reliability + (1.0 - cfg_te.rt_reliability_alpha) * match

        # false positive 예측(예측=1, 실제=0)에는 추가 페널티
        fp_mask = (A_pred[t] == 1) & (A_actual[t] == 0)
        reliability[fp_mask] *= cfg_te.rt_fp_extra_penalty

        reliability = np.clip(reliability, 0.05, 1.0)

    return planned_idx


# =========================================================
# Evaluation on hold-out seeds
# =========================================================

def evaluate_on_test_seeds(
    test_pairs: List[Dict[str, Any]],
    transformer,
    consistency,
    cfg_te: TrainEvalRealtimeConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    results_dir = Path(cfg_te.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Evaluation: Learned planner on held-out seeds ===")
    for pair in test_pairs:
        seed = pair["seed"]
        actual_env = pair["actual_env"]
        pred_env = pair["pred_env"]

        np.random.seed(seed)  # runtime miss stochasticity 재현성 확보

        # 1) Baseline: sustainable DAG on actual env
        dag_plan = sustainable_dag_plan(actual_env, cfg_te.exp_cfg)
        tl_dag, m_dag = execute_plan_on_env(
            build_res=actual_env,
            planned_idx=dag_plan,
            method_name="Sustainable_DAG_NetworkX",
            mode="sustainable_dag",
            sampling_steps=0,
            cfg=cfg_te.exp_cfg,
        )
        m_dag["seed"] = seed
        rows.append(m_dag)

        # 2) Learned offline (predicted env only)
        offline_plan = build_learned_offline_plan_from_pred_env(
            pred_env=pred_env,
            transformer=transformer,
            consistency=consistency,
            cfg_te=cfg_te,
            consistency_steps=cfg_te.consistency_steps_eval,
        )
        tl_off, m_off = execute_plan_on_env(
            build_res=actual_env,
            planned_idx=offline_plan,
            method_name="Learned_TC_Offline",
            mode="consistency",
            sampling_steps=cfg_te.consistency_steps_eval,
            cfg=cfg_te.exp_cfg,
        )
        m_off["seed"] = seed
        rows.append(m_off)

        # 3) Learned + realtime correction
        rt_plan = build_learned_realtime_corrected_plan(
            actual_env=actual_env,
            pred_env=pred_env,
            transformer=transformer,
            consistency=consistency,
            cfg_te=cfg_te,
            consistency_steps=cfg_te.consistency_steps_eval,
        )
        tl_rt, m_rt = execute_plan_on_env(
            build_res=actual_env,
            planned_idx=rt_plan,
            method_name="Learned_TC_RTCorrected",
            mode="consistency",
            sampling_steps=cfg_te.consistency_steps_eval,
            cfg=cfg_te.exp_cfg,
        )
        m_rt["seed"] = seed
        rows.append(m_rt)

        print(
            f"[seed={seed}]\n"
            f" 🔹 [DAG] Avail: {m_dag['Availability']:.4f} | QoE: {m_dag['EffectiveQoE']:.4f} | 핑퐁: {m_dag['PingPong_Ratio']:.4f} | HO실패: {m_dag['HO_Failure_Ratio']:.4f}\n"
            f" 🔸 [ AI] Avail: {m_rt['Availability']:.4f} | QoE: {m_rt['EffectiveQoE']:.4f} | 핑퐁: {m_rt['PingPong_Ratio']:.4f} | HO실패: {m_rt['HO_Failure_Ratio']:.4f}\n"
            f" --------------------------------------------------------"
        )

        # seed0 timeline 저장 (test 첫 seed)
        if seed == test_pairs[0]["seed"]:
            tl_dag.to_csv(results_dir / f"timeline_sustainable_dag_seed{seed}.csv", index=False)
            tl_off.to_csv(results_dir / f"timeline_learned_tc_offline_seed{seed}.csv", index=False)
            tl_rt.to_csv(results_dir / f"timeline_learned_tc_rtcorrected_seed{seed}.csv", index=False)

    df_runs = pd.DataFrame(rows)
    df_summary = summarize_runs(df_runs)

    df_runs.to_csv(results_dir / "exp3_learned_realtime_runs.csv", index=False)
    df_summary.to_csv(results_dir / "exp3_learned_realtime_summary.csv", index=False)

    print(f"✅ saved: {results_dir / 'exp3_learned_realtime_runs.csv'}")
    print(f"✅ saved: {results_dir / 'exp3_learned_realtime_summary.csv'}")
    return df_runs, df_summary


# =========================================================
# Main pipeline
# =========================================================

def main():
    cfg_te = TrainEvalRealtimeConfig()

    results_dir = Path(cfg_te.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # experiments_atg_leo의 env builder가 쓰는 TLE 파일 확인
    if not Path(cfg_te.exp_cfg.tle_path).exists():
        raise FileNotFoundError(
            f"Frozen TLE file not found: {cfg_te.exp_cfg.tle_path}\n"
            "먼저 TLE 파일 경로를 맞춰주세요."
        )

    # split 저장
    split_manifest = {
        "train_seeds": cfg_te.train_seeds,
        "val_seeds": cfg_te.val_seeds,
        "test_seeds": cfg_te.test_seeds,
    }
    with open(cfg_te.split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    print("=== Seed Split ===")
    print(json.dumps(split_manifest, ensure_ascii=False, indent=2))

    t0 = time.time()

    # 1) collect
    train_pairs = collect_env_pairs(cfg_te.exp_cfg, cfg_te.train_seeds)
    val_pairs = collect_env_pairs(cfg_te.exp_cfg, cfg_te.val_seeds)   # 지금은 로깅용/확장용
    test_pairs = collect_env_pairs(cfg_te.exp_cfg, cfg_te.test_seeds)

    print(f"[collect done] train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # 2) train
    transformer, consistency, train_hist = train_on_seed_pool(train_pairs, cfg_te)

    # (선택) val quick sanity metrics를 원하면 여기 추가 가능
    # 지금은 바로 test 평가

    # 3) evaluate (hold-out)
    df_runs, df_summary = evaluate_on_test_seeds(
        test_pairs=test_pairs,
        transformer=transformer,
        consistency=consistency,
        cfg_te=cfg_te,
    )

    elapsed = time.time() - t0
    print("\n=== Done ===")
    print(f"Total elapsed: {elapsed:.2f} sec")
    print("Outputs:")
    print(f" - {cfg_te.weight_path}")
    print(f" - {Path(cfg_te.results_dir) / 'train_history_real_env.csv'}")
    print(f" - {Path(cfg_te.results_dir) / 'exp3_learned_realtime_runs.csv'}")
    print(f" - {Path(cfg_te.results_dir) / 'exp3_learned_realtime_summary.csv'}")
    print(f" - {cfg_te.split_manifest_path}")
