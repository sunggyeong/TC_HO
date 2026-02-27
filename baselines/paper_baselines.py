#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_baselines.py

Paper-ready decision-layer baselines for TN-NTN integrated handover planning.

Outputs:
  planned_idx[t] ∈ {-1, 0..N-1}

Baselines:
  - Reactive_MyopicGreedy (current-slot greedy + guardrails)
  - Lookahead_Greedy      (H-step MPC w/o learning + guardrails)
  - ShootingSearch_DeltaNode (slow teacher; K-shot search over (Δ,node) + guardrails)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class GuardrailConfig:
    min_dwell: int = 3
    hysteresis_ratio: float = 0.08
    hysteresis_abs: float = 0.0
    pingpong_window: int = 4
    pingpong_extra_hysteresis_ratio: float = 0.12

@dataclass
class ScoreConfig:
    alpha_latency: float = 0.05  # score = utility - alpha*latency

@dataclass
class LookaheadConfig:
    H: int = 30
    gamma: float = 0.97
    alpha_latency: float = 0.05

@dataclass
class ShootingSearchConfig:
    H: int = 30
    gamma: float = 0.97
    alpha_latency: float = 0.05
    K: int = 256
    outage_penalty: float = 0.70
    switch_penalty: float = 0.12
    robustness_bonus: float = 0.15
    robustness_window: int = 5

def _guardrail_accept(
    t: int, current_node: int, new_node: int, last_switch_t: int,
    score_cur: float, score_new: float, prev1: int, prev2: int,
    guard: GuardrailConfig
) -> bool:
    if current_node == -1 or new_node == current_node:
        return True
    if (t - last_switch_t) < int(guard.min_dwell):
        return False
    required_ratio = float(guard.hysteresis_ratio)
    is_pingpong = (
        new_node == prev2 and prev2 != -1 and prev1 == current_node and
        (t - last_switch_t) <= int(guard.pingpong_window)
    )
    if is_pingpong:
        required_ratio += float(guard.pingpong_extra_hysteresis_ratio)
    ratio_ok = (score_new >= score_cur * (1.0 + required_ratio))
    if float(guard.hysteresis_abs) > 0.0:
        abs_ok = ((score_new - score_cur) >= float(guard.hysteresis_abs))
        return bool(ratio_ok and abs_ok)
    return bool(ratio_ok)

def plan_reactive_myopic(A: np.ndarray, U: np.ndarray, L_ms: np.ndarray, score: ScoreConfig, guard: GuardrailConfig) -> np.ndarray:
    T, N = A.shape
    plan = np.full(T, -1, dtype=int)
    current = -1
    last_switch_t = -10**9
    for t in range(T):
        avail = np.where(A[t] == 1)[0]
        if len(avail) == 0:
            plan[t] = -1
            current = -1
            continue
        scores = U[t, avail] - score.alpha_latency * L_ms[t, avail]
        cand = int(avail[int(np.argmax(scores))])
        prev1 = int(plan[t-1]) if t-1 >= 0 else -1
        prev2 = int(plan[t-2]) if t-2 >= 0 else -1
        if current != -1 and A[t, current] == 1:
            s_cur = float(U[t, current] - score.alpha_latency * L_ms[t, current])
            s_new = float(U[t, cand] - score.alpha_latency * L_ms[t, cand])
            if _guardrail_accept(t, current, cand, last_switch_t, s_cur, s_new, prev1, prev2, guard):
                if cand != current:
                    last_switch_t = t
                current = cand
        else:
            if cand != current:
                last_switch_t = t
            current = cand
        plan[t] = current
    return plan

def plan_lookahead_greedy(A_pred: np.ndarray, U_pred: np.ndarray, L_pred_ms: np.ndarray, look: LookaheadConfig, guard: GuardrailConfig) -> np.ndarray:
    T, N = A_pred.shape
    H = int(look.H)
    plan = np.full(T, -1, dtype=int)
    current = -1
    last_switch_t = -10**9
    gamma_pows = np.array([float(look.gamma)**k for k in range(H)], dtype=np.float32)
    for t in range(T):
        avail_now = np.where(A_pred[t] == 1)[0]
        if len(avail_now) == 0:
            plan[t] = -1
            current = -1
            continue
        t1 = min(T, t + H)
        def _look_score(n: int) -> float:
            a = A_pred[t:t1, n].astype(np.float32)
            u = U_pred[t:t1, n].astype(np.float32)
            l = L_pred_ms[t:t1, n].astype(np.float32)
            w = gamma_pows[:len(a)]
            return float(np.sum(w * (a*u - look.alpha_latency*l)))
        best_n, best_s = -1, -1e18
        for n in avail_now:
            s = _look_score(int(n))
            if s > best_s:
                best_s, best_n = s, int(n)
        cand = best_n
        prev1 = int(plan[t-1]) if t-1 >= 0 else -1
        prev2 = int(plan[t-2]) if t-2 >= 0 else -1
        if current != -1 and A_pred[t, current] == 1:
            s_cur = _look_score(current)
            s_new = _look_score(cand)
            if _guardrail_accept(t, current, cand, last_switch_t, s_cur, s_new, prev1, prev2, guard):
                if cand != current:
                    last_switch_t = t
                current = cand
        else:
            if cand != current:
                last_switch_t = t
            current = cand
        plan[t] = current
    return plan

def _reward_delta_node(A_pred: np.ndarray, U_pred: np.ndarray, L_pred_ms: np.ndarray, t0: int, H: int, current_node: int, delta: int, target_node: int, cfg: ShootingSearchConfig) -> float:
    T, N = A_pred.shape
    t1 = min(T, t0 + H)
    gam, alpha = float(cfg.gamma), float(cfg.alpha_latency)
    r = 0.0
    for k, t in enumerate(range(t0, t1)):
        use_n = current_node if k < delta else target_node
        if use_n < 0 or use_n >= N or A_pred[t, use_n] != 1:
            step_r = -float(cfg.outage_penalty)
        else:
            step_r = float(U_pred[t, use_n] - alpha*L_pred_ms[t, use_n])
        if k == delta and current_node != -1 and target_node != -1 and target_node != current_node:
            step_r -= float(cfg.switch_penalty)
        if use_n >= 0 and use_n < N and int(cfg.robustness_window) > 0:
            tt1 = min(T, t + int(cfg.robustness_window))
            robust = float(np.mean(A_pred[t:tt1, use_n]))
            step_r += float(cfg.robustness_bonus) * robust
        r += (gam**k) * step_r
    return float(r)

def plan_shooting_search_delta_node(A_pred: np.ndarray, U_pred: np.ndarray, L_pred_ms: np.ndarray, search: ShootingSearchConfig, guard: GuardrailConfig, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(int(seed))
    T, N = A_pred.shape
    H = int(search.H)
    plan = np.full(T, -1, dtype=int)
    current = -1
    last_switch_t = -10**9
    if T > 0:
        avail0 = np.where(A_pred[0] == 1)[0]
        if len(avail0) > 0:
            scores0 = U_pred[0, avail0] - float(search.alpha_latency) * L_pred_ms[0, avail0]
            current = int(avail0[int(np.argmax(scores0))])
    for t in range(T):
        avail_now = np.where(A_pred[t] == 1)[0]
        if len(avail_now) == 0:
            plan[t] = -1
            current = -1
            continue
        K = int(search.K)
        deltas = np.random.randint(0, H, size=K, dtype=np.int32)
        nodes = np.random.choice(avail_now, size=K, replace=True).astype(np.int32)
        best_r, best_delta, best_node = -1e18, 0, int(nodes[0])
        for d, n in zip(deltas, nodes):
            rr = _reward_delta_node(A_pred, U_pred, L_pred_ms, t, H, current, int(d), int(n), search)
            if rr > best_r:
                best_r, best_delta, best_node = rr, int(d), int(n)
        proposed = current if best_delta > 0 else best_node
        prev1 = int(plan[t-1]) if t-1 >= 0 else -1
        prev2 = int(plan[t-2]) if t-2 >= 0 else -1
        if current != -1 and A_pred[t, current] == 1 and proposed != current:
            s_cur = float(U_pred[t, current] - float(search.alpha_latency) * L_pred_ms[t, current])
            s_new = float(U_pred[t, proposed] - float(search.alpha_latency) * L_pred_ms[t, proposed])
            if _guardrail_accept(t, current, proposed, last_switch_t, s_cur, s_new, prev1, prev2, guard):
                last_switch_t = t
                current = proposed
        else:
            if proposed != current and current != -1:
                last_switch_t = t
            current = proposed
        plan[t] = current
    return plan
