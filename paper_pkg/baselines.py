#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_pkg.baselines

Decision-layer baselines with guardrails:
- Reactive_MyopicGreedy (actual slot)
- Lookahead_Greedy (pred horizon)
- ShootingSearch_DeltaNode (slow teacher)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class GuardrailConfig:
    min_dwell: int = 3
    hysteresis_ratio: float = 0.10
    hysteresis_abs: float = 0.0
    pingpong_window: int = 5
    pingpong_extra_hysteresis_ratio: float = 0.15

@dataclass
class ScoreConfig:
    alpha_latency: float = 0.05

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

def _guardrail_accept(t, cur, new, last_sw, s_cur, s_new, prev1, prev2, g: GuardrailConfig) -> bool:
    if cur == -1 or new == cur:
        return True
    if (t - last_sw) < int(g.min_dwell):
        return False
    req = float(g.hysteresis_ratio)
    pingpong = (new == prev2 and prev2 != -1 and prev1 == cur and (t - last_sw) <= int(g.pingpong_window))
    if pingpong:
        req += float(g.pingpong_extra_hysteresis_ratio)
    ratio_ok = (s_new >= s_cur * (1.0 + req))
    if float(g.hysteresis_abs) > 0.0:
        return bool(ratio_ok and ((s_new - s_cur) >= float(g.hysteresis_abs)))
    return bool(ratio_ok)

def plan_reactive_myopic(A, U, L_ms, score: ScoreConfig, guard: GuardrailConfig):
    T, N = A.shape
    plan = np.full(T, -1, dtype=int)
    cur = -1
    last_sw = -10**9
    for t in range(T):
        avail = np.where(A[t] == 1)[0]
        if len(avail) == 0:
            plan[t] = -1
            cur = -1
            continue
        scores = U[t, avail] - score.alpha_latency * L_ms[t, avail]
        cand = int(avail[int(np.argmax(scores))])
        prev1 = int(plan[t-1]) if t-1 >= 0 else -1
        prev2 = int(plan[t-2]) if t-2 >= 0 else -1
        if cur != -1 and A[t, cur] == 1:
            s_cur = float(U[t, cur] - score.alpha_latency * L_ms[t, cur])
            s_new = float(U[t, cand] - score.alpha_latency * L_ms[t, cand])
            if _guardrail_accept(t, cur, cand, last_sw, s_cur, s_new, prev1, prev2, guard):
                if cand != cur:
                    last_sw = t
                cur = cand
        else:
            if cand != cur:
                last_sw = t
            cur = cand
        plan[t] = cur
    return plan

def plan_lookahead_greedy(A_pred, U_pred, L_pred, look: LookaheadConfig, guard: GuardrailConfig):
    T, N = A_pred.shape
    H = int(look.H)
    plan = np.full(T, -1, dtype=int)
    cur = -1
    last_sw = -10**9
    gamma_pows = np.array([float(look.gamma)**k for k in range(H)], dtype=np.float32)
    for t in range(T):
        avail = np.where(A_pred[t] == 1)[0]
        if len(avail) == 0:
            plan[t] = -1
            cur = -1
            continue
        t1 = min(T, t+H)
        def score_n(n: int) -> float:
            a = A_pred[t:t1, n].astype(np.float32)
            u = U_pred[t:t1, n].astype(np.float32)
            l = L_pred[t:t1, n].astype(np.float32)
            w = gamma_pows[:len(a)]
            return float(np.sum(w * (a*u - look.alpha_latency*l)))
        best_n, best_s = -1, -1e18
        for n in avail:
            s = score_n(int(n))
            if s > best_s:
                best_s, best_n = s, int(n)
        cand = best_n
        prev1 = int(plan[t-1]) if t-1 >= 0 else -1
        prev2 = int(plan[t-2]) if t-2 >= 0 else -1
        if cur != -1 and A_pred[t, cur] == 1:
            s_cur = score_n(cur)
            s_new = score_n(cand)
            if _guardrail_accept(t, cur, cand, last_sw, s_cur, s_new, prev1, prev2, guard):
                if cand != cur:
                    last_sw = t
                cur = cand
        else:
            if cand != cur:
                last_sw = t
            cur = cand
        plan[t] = cur
    return plan

def _reward_delta_node(A_pred, U_pred, L_pred, t0, H, cur, delta, node, cfg: ShootingSearchConfig):
    T, N = A_pred.shape
    t1 = min(T, t0+H)
    gam = float(cfg.gamma)
    alpha = float(cfg.alpha_latency)
    r = 0.0
    for k, t in enumerate(range(t0, t1)):
        use = cur if k < delta else node
        if use < 0 or use >= N or A_pred[t, use] != 1:
            step = -float(cfg.outage_penalty)
        else:
            step = float(U_pred[t, use] - alpha * L_pred[t, use])
        if k == delta and cur != -1 and node != -1 and node != cur:
            step -= float(cfg.switch_penalty)
        if use >= 0 and use < N and int(cfg.robustness_window) > 0:
            tt1 = min(T, t + int(cfg.robustness_window))
            robust = float(np.mean(A_pred[t:tt1, use]))
            step += float(cfg.robustness_bonus) * robust
        r += (gam**k) * step
    return float(r)

def plan_shooting_search_delta_node(A_pred, U_pred, L_pred, search: ShootingSearchConfig, guard: GuardrailConfig, seed: Optional[int]=None):
    if seed is not None:
        np.random.seed(int(seed))
    T, N = A_pred.shape
    H = int(search.H)
    plan = np.full(T, -1, dtype=int)
    cur = -1
    last_sw = -10**9
    if T > 0:
        avail0 = np.where(A_pred[0] == 1)[0]
        if len(avail0) > 0:
            scores0 = U_pred[0, avail0] - float(search.alpha_latency)*L_pred[0, avail0]
            cur = int(avail0[int(np.argmax(scores0))])
    for t in range(T):
        avail = np.where(A_pred[t] == 1)[0]
        if len(avail) == 0:
            plan[t] = -1
            cur = -1
            continue
        K = int(search.K)
        deltas = np.random.randint(0, H, size=K, dtype=np.int32)
        nodes = np.random.choice(avail, size=K, replace=True).astype(np.int32)
        best_r, best_d, best_n = -1e18, 0, int(nodes[0])
        for d, n in zip(deltas, nodes):
            rr = _reward_delta_node(A_pred, U_pred, L_pred, t, H, cur, int(d), int(n), search)
            if rr > best_r:
                best_r, best_d, best_n = rr, int(d), int(n)
        proposed = cur if best_d > 0 else best_n
        prev1 = int(plan[t-1]) if t-1 >= 0 else -1
        prev2 = int(plan[t-2]) if t-2 >= 0 else -1
        if cur != -1 and A_pred[t, cur] == 1 and proposed != cur:
            s_cur = float(U_pred[t, cur] - float(search.alpha_latency)*L_pred[t, cur])
            s_new = float(U_pred[t, proposed] - float(search.alpha_latency)*L_pred[t, proposed])
            if _guardrail_accept(t, cur, proposed, last_sw, s_cur, s_new, prev1, prev2, guard):
                last_sw = t
                cur = proposed
        else:
            if proposed != cur and cur != -1:
                last_sw = t
            cur = proposed
        plan[t] = cur
    return plan
