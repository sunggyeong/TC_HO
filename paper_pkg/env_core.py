#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_pkg.env_core

Clean core environment/execution module for the NEW paper experiments only.

Includes:
- ExperimentConfig (+ RuntimeModelConfig)
- build_env (actual/predicted)
- sustainable_dag_plan (with configurable allow_skip_edge)
- execute_plan_on_env
- summarize_runs

Legacy experiment drivers (Exp1/Exp2) are intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math
import time
import copy
import numpy as np
import pandas as pd

# --- providers / env / baseline (이전 답변 코드 기준) ---
from providers.trajectory_provider import Waypoint, WaypointTrajectoryProvider, PerturbedTrajectoryProvider
from providers.tn_atg_provider import TNATGNode, TNATGProvider
from providers.satellite_provider_skyfield import SkyfieldLEOConfig, SkyfieldFrozenTLEProvider
from providers.phase_mask_provider import PhaseMaskProvider
from envs.tn_ntn_env import TN_NTN_Env
from baselines.sustainable_dag import SustainableDAGPlanner, SustainableDAGConfig


# =========================================================
# Configs
# =========================================================

@dataclass
class RuntimeModelConfig:
    # 1초 슬롯 내 의사결정 deadline budget (예: 40ms)
    decision_budget_ms: float = 40.0

    # consistency / diffusion inference latency model (간단 proxy)
    base_ms_consistency: float = 1.8
    per_step_ms_consistency: float = 1.0

    base_ms_diffusion: float = 3.0
    per_step_ms_diffusion: float = 3.2

    # jitter proxy에서 처리/재접속 파라미터 (Generative 참고 구조)
    node_processing_delay_ms: float = 10.0
    reattach_wait_ms: float = 50.0


@dataclass
class ExperimentConfig:
    results_dir: str = "results"
    n_runs: int = 10
    seeds: List[int] = field(default_factory=lambda: list(range(0, 10)))

    # 시나리오
    dt_sec: float = 5.0
    speed_mps: float = 250.0            # 900 km/h
    use_perturbed_predicted: bool = True
    latlon_noise_deg: float = 0.02      # predicted trajectory 오차 모사
    alt_noise_m: float = 80.0

    # skyfield / tle
    tle_path: str = "data/starlink_frozen_20260224_1410Z.tle"
    min_elevation_deg: float = 10.0
    max_sats: int = 16                 # 테스트 시 100~300 권장

    # DAG baseline params
    dag_switch_penalty: float = 0.08
    dag_stay_bonus: float = 0.02
    dag_outage_penalty: float = 1.0
    dag_latency_penalty_scale: float = 0.002
    dag_topk_per_slot: int = 12

    runtime: RuntimeModelConfig = field(default_factory=RuntimeModelConfig)

    # ---------------------------
    # Stress scenario presets (논문용)
    # easy | weather_drift | deadline_crunch | paper_stress
    # ---------------------------
    stress_profile: str = "paper_stress"

    # predicted/actual env에 각각 스트레스 주입 여부
    stress_apply_to_predicted_env: bool = True
    stress_apply_to_actual_env: bool = True

    # (1) 예측 오차/드리프트
    stress_pred_noise_std: float = 0.04
    stress_pred_noise_horizon_gain: float = 0.10
    stress_actual_drift_std: float = 0.03
    stress_actual_noise_std: float = 0.015

    # (2) ATG burst fade (악천후/난기류 느낌)
    stress_atg_burst_prob: float = 0.06
    stress_atg_burst_len_min: int = 2
    stress_atg_burst_len_max: int = 6
    stress_atg_burst_drop: float = 0.45

    # (3) 후보군 급변
    stress_candidate_drop_prob: float = 0.03
    stress_candidate_drop_len_min: int = 1
    stress_candidate_drop_len_max: int = 3
    stress_candidate_drop_frac: float = 0.18

    # (4) deadline crunch (Exp2 차별성 강조)
    stress_decision_budget_ms_override: Optional[float] = 32.0

    # 재현성
    stress_seed_offset_pred: int = 10_000
    stress_seed_offset_actual: int = 20_000



# =========================================================
# Demo Scenario (Roma -> New York gate-to-gate)
# =========================================================

def build_demo_waypoints() -> List[Waypoint]:
    # 로마 -> 뉴욕 (단순화된 gate-to-gate)
    return [
        Waypoint(41.800, 12.250,   0.0),      # gate/taxi
        Waypoint(41.900, 11.500, 2000.0),     # takeoff_climb
        Waypoint(43.000,  4.000, 8000.0),
        Waypoint(46.000, -15.000, 10000.0),   # cruise
        Waypoint(49.000, -35.000, 10000.0),   # transatlantic
        Waypoint(47.000, -55.000, 10000.0),
        Waypoint(42.000, -70.000, 6000.0),    # descent
        Waypoint(40.800, -74.000, 500.0),     # landing
        Waypoint(40.700, -74.000,   0.0),     # gate
    ]


def build_demo_atg_nodes() -> List[TNATGNode]:
    # 데모용 공항/연안/해역 ATG/TN 노드
    return [
        TNATGNode("ATG_ROMA",   41.9,  12.5, kind="ATG", radius_km=60,  base_latency_ms=8.0,  capacity_score=1.00),
        TNATGNode("ATG_W_MED",  40.0,   5.0, kind="ATG", radius_km=90,  base_latency_ms=9.0,  capacity_score=0.95),
        TNATGNode("ATG_E_ATL",  45.0, -15.0, kind="ATG", radius_km=120, base_latency_ms=10.0, capacity_score=0.92),
        TNATGNode("ATG_C_ATL",  48.0, -35.0, kind="ATG", radius_km=120, base_latency_ms=10.5, capacity_score=0.90),
        TNATGNode("ATG_W_ATL",  46.0, -55.0, kind="ATG", radius_km=120, base_latency_ms=10.0, capacity_score=0.92),
        TNATGNode("ATG_NY",     40.7, -74.0, kind="ATG", radius_km=70,  base_latency_ms=8.0,  capacity_score=1.00),
    ]


# =========================================================
# Env Builder (Skyfield + Frozen TLE)
# =========================================================

def build_env(cfg: ExperimentConfig, seed: int, predicted_view: bool = False):
    """
    predicted_view=False : 실제 실행 기준 trajectory (oracle-like path)
    predicted_view=True  : predictor가 보는 perturbed trajectory (예측 오차 모사)
    """
    base_provider = WaypointTrajectoryProvider(
        waypoints=build_demo_waypoints(),
        speed_mps=cfg.speed_mps,
        dt_sec=cfg.dt_sec,
    )

    if predicted_view and cfg.use_perturbed_predicted:
        traj_provider = PerturbedTrajectoryProvider(
            base_provider=base_provider,
            latlon_noise_deg=cfg.latlon_noise_deg,
            alt_noise_m=cfg.alt_noise_m,
            seed=seed,
        )
    else:
        traj_provider = base_provider

    tn_provider = TNATGProvider(build_demo_atg_nodes())

    sat_provider = SkyfieldFrozenTLEProvider(
        SkyfieldLEOConfig(
            tle_path=cfg.tle_path,
            min_elevation_deg=cfg.min_elevation_deg,
            max_sats=cfg.max_sats,
            base_latency_ms=20.0,
            capacity_score=0.95,
        )
    )

    phase_provider = PhaseMaskProvider()

    env = TN_NTN_Env(
        trajectory_provider=traj_provider,
        tn_provider=tn_provider,
        satellite_provider=sat_provider,
        phase_mask_provider=phase_provider,
    )

    build_res = env.build()
    return _stress_mutate_build_result(build_res, cfg, seed=seed, predicted_view=predicted_view)


def _scenario_overrides(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Preset 문자열을 실제 스트레스 파라미터로 변환."""
    p = {
        "pred_noise_std": cfg.stress_pred_noise_std,
        "pred_noise_horizon_gain": cfg.stress_pred_noise_horizon_gain,
        "actual_drift_std": cfg.stress_actual_drift_std,
        "actual_noise_std": cfg.stress_actual_noise_std,
        "atg_burst_prob": cfg.stress_atg_burst_prob,
        "atg_burst_len_min": cfg.stress_atg_burst_len_min,
        "atg_burst_len_max": cfg.stress_atg_burst_len_max,
        "atg_burst_drop": cfg.stress_atg_burst_drop,
        "candidate_drop_prob": cfg.stress_candidate_drop_prob,
        "candidate_drop_len_min": cfg.stress_candidate_drop_len_min,
        "candidate_drop_len_max": cfg.stress_candidate_drop_len_max,
        "candidate_drop_frac": cfg.stress_candidate_drop_frac,
        "decision_budget_ms_override": cfg.stress_decision_budget_ms_override,
    }
    profile = (cfg.stress_profile or "easy").lower()
    if profile == "easy":
        p.update(pred_noise_std=0.0, pred_noise_horizon_gain=0.0,
                 actual_drift_std=0.0, actual_noise_std=0.0,
                 atg_burst_prob=0.0, candidate_drop_prob=0.0,
                 decision_budget_ms_override=None)
    elif profile == "weather_drift":
        # 현실성+변별력 균형형 (all-fail 방지)
        p.update(
            pred_noise_std=max(p["pred_noise_std"], 0.03),
            pred_noise_horizon_gain=max(p["pred_noise_horizon_gain"], 0.08),
            actual_drift_std=max(p["actual_drift_std"], 0.02),
            actual_noise_std=max(p["actual_noise_std"], 0.01),
            atg_burst_prob=max(p["atg_burst_prob"], 0.03),
            atg_burst_drop=max(p["atg_burst_drop"], 0.25),
            candidate_drop_prob=max(p["candidate_drop_prob"], 0.015),
            candidate_drop_frac=max(p["candidate_drop_frac"], 0.12),
            # 이벤트 길이도 짧게
            atg_burst_len_min=min(p["atg_burst_len_min"], 2),
            atg_burst_len_max=min(p["atg_burst_len_max"], 4),
            candidate_drop_len_min=min(p["candidate_drop_len_min"], 1),
            candidate_drop_len_max=min(p["candidate_drop_len_max"], 2),
            decision_budget_ms_override=None,
        )
    elif profile == "deadline_crunch":
        p.update(pred_noise_std=max(p["pred_noise_std"], 0.03),
                 pred_noise_horizon_gain=max(p["pred_noise_horizon_gain"], 0.08),
                 actual_drift_std=max(p["actual_drift_std"], 0.02),
                 actual_noise_std=max(p["actual_noise_std"], 0.015),
                 atg_burst_prob=max(p["atg_burst_prob"], 0.03),
                 candidate_drop_prob=max(p["candidate_drop_prob"], 0.01),
                 decision_budget_ms_override=(p["decision_budget_ms_override"] if p["decision_budget_ms_override"] is not None else 32.0))
    elif profile == "paper_stress":
        # 논문용 추천: 현실성 + 성능 차이 둘 다 드러나는 조합
        p.update(pred_noise_std=max(p["pred_noise_std"], 0.06),
                 pred_noise_horizon_gain=max(p["pred_noise_horizon_gain"], 0.16),
                 actual_drift_std=max(p["actual_drift_std"], 0.05),
                 actual_noise_std=max(p["actual_noise_std"], 0.025),
                 atg_burst_prob=max(p["atg_burst_prob"], 0.10),
                 atg_burst_drop=max(p["atg_burst_drop"], 0.55),
                 candidate_drop_prob=max(p["candidate_drop_prob"], 0.04),
                 candidate_drop_frac=max(p["candidate_drop_frac"], 0.22),
                 decision_budget_ms_override=(p["decision_budget_ms_override"] if p["decision_budget_ms_override"] is not None else 32.0))
    else:
        print(f"[warn] unknown stress_profile={cfg.stress_profile!r}; using manual stress values.")
    return p


def _safe_np(a):
    try:
        return None if a is None else np.array(a, dtype=float, copy=True)
    except Exception:
        return None


def _clamp01(a):
    if a is not None:
        np.clip(a, 0.0, 1.0, out=a)
    return a


def _infer_tn_indices(build_res, n_cols: int) -> np.ndarray:
    meta = getattr(build_res, "meta_all", None)
    if isinstance(meta, pd.DataFrame) and len(meta) == n_cols:
        try:
            joined = meta.astype(str).agg(" | ".join, axis=1).str.lower()
            mask = joined.str.contains(r"\btn\b|atg|terrestrial|ground")
            idx = np.where(mask.to_numpy())[0]
            if len(idx) > 0:
                return idx.astype(int)
        except Exception:
            pass
    return np.arange(n_cols, dtype=int)


def _apply_horizon_noise(A: np.ndarray, rng: np.random.RandomState, base_std: float, gain: float):
    if A is None or (base_std <= 0 and gain <= 0):
        return
    T, N = A.shape
    tnorm = np.linspace(0.0, 1.0, T, dtype=float)[:, None]
    std = base_std + gain * tnorm
    A += rng.normal(0.0, std, size=(T, N))


def _apply_lowfreq_drift(A: np.ndarray, rng: np.random.RandomState, drift_std: float):
    if A is None or drift_std <= 0:
        return
    T, N = A.shape
    t = np.arange(T, dtype=float)[:, None] / max(T, 1)
    freq = rng.uniform(0.5, 1.6, size=(1, N))
    phase = rng.uniform(0.0, 2.0*np.pi, size=(1, N))
    amp = rng.uniform(0.5*drift_std, 1.5*drift_std, size=(1, N))
    A += amp * np.sin(2.0*np.pi*freq*t + phase)


def _apply_candidate_dropouts(A: np.ndarray, rng: np.random.RandomState, p_start: float, len_min: int, len_max: int, drop_frac: float):
    if A is None or p_start <= 0 or drop_frac <= 0:
        return
    T, N = A.shape
    if N <= 0:
        return
    k = max(1, int(round(N * drop_frac)))
    l0 = max(1, int(len_min))
    l1 = max(l0, int(len_max))
    for t0 in range(T):
        if rng.rand() < p_start:
            L = int(rng.randint(l0, l1 + 1))
            t1 = min(T, t0 + L)
            cols = rng.choice(N, size=min(k, N), replace=False)
            A[t0:t1, cols] *= rng.uniform(0.05, 0.25)


def _apply_atg_bursts(A: np.ndarray, build_res, rng: np.random.RandomState, p_start: float, len_min: int, len_max: int, drop: float):
    if A is None or p_start <= 0 or drop <= 0:
        return
    T, N = A.shape
    tn_idx = _infer_tn_indices(build_res, N)
    if len(tn_idx) == 0:
        return
    l0 = max(1, int(len_min))
    l1 = max(l0, int(len_max))
    num_cols = max(1, int(round(0.25 * len(tn_idx))))
    for t0 in range(T):
        if rng.rand() < p_start:
            L = int(rng.randint(l0, l1 + 1))
            t1 = min(T, t0 + L)
            cols = rng.choice(tn_idx, size=min(num_cols, len(tn_idx)), replace=False)
            A[t0:t1, cols] -= drop


def _stress_mutate_build_result(build_res, cfg: ExperimentConfig, seed: int, predicted_view: bool):
    profile = (cfg.stress_profile or "easy").lower()
    if profile == "easy":
        return build_res
    if predicted_view and not cfg.stress_apply_to_predicted_env:
        return build_res
    if (not predicted_view) and not cfg.stress_apply_to_actual_env:
        return build_res

    p = _scenario_overrides(cfg)
    seed_offset = cfg.stress_seed_offset_pred if predicted_view else cfg.stress_seed_offset_actual
    rng = np.random.RandomState(int(seed) + int(seed_offset))

    A_final = _safe_np(getattr(build_res, "A_final", None))
    if A_final is None or A_final.ndim != 2:
        return build_res

    if predicted_view:
        _apply_horizon_noise(A_final, rng, float(p["pred_noise_std"]), float(p["pred_noise_horizon_gain"]))
    else:
        _apply_lowfreq_drift(A_final, rng, float(p["actual_drift_std"]))
        if float(p["actual_noise_std"]) > 0:
            A_final += rng.normal(0.0, float(p["actual_noise_std"]), size=A_final.shape)

    _apply_candidate_dropouts(
        A_final, rng,
        p_start=float(p["candidate_drop_prob"]),
        len_min=int(p["candidate_drop_len_min"]),
        len_max=int(p["candidate_drop_len_max"]),
        drop_frac=float(p["candidate_drop_frac"]),
    )
    _apply_atg_bursts(
        A_final, build_res, rng,
        p_start=float(p["atg_burst_prob"]),
        len_min=int(p["atg_burst_len_min"]),
        len_max=int(p["atg_burst_len_max"]),
        drop=float(p["atg_burst_drop"]),
    )

    # 보조 행렬들도 shape 맞으면 약하게 교란 (A_final과 완전 불일치 방지 목적)
    for attr in ["A_tn", "A_leo", "utility"]:
        aux = _safe_np(getattr(build_res, attr, None))
        if aux is not None and aux.shape == A_final.shape:
            if attr == "utility":
                aux += rng.normal(0.0, 0.25 * float(p["actual_noise_std"] if not predicted_view else p["pred_noise_std"]), size=aux.shape)
            else:
                aux += rng.normal(0.0, 0.5 * float(p["actual_noise_std"] if not predicted_view else p["pred_noise_std"]), size=aux.shape)
            setattr(build_res, attr, _clamp01(aux))

    A_final = _clamp01(A_final)
    # 연속값으로 주입된 경우를 대비해 이진 가용성으로 복구
    A_final = (A_final >= 0.5).astype(np.int8)

    setattr(build_res, "A_final", A_final)
    return build_res


def _apply_deadline_stress(cfg: ExperimentConfig):
    p = _scenario_overrides(cfg)
    override = p.get("decision_budget_ms_override", None)
    if override is None:
        return
    try:
        cfg.runtime.decision_budget_ms = float(override)
    except Exception:
        pass


# =========================================================
# Planner helpers
# =========================================================

def sustainable_dag_plan(build_res, cfg: ExperimentConfig) -> np.ndarray:
    planner = SustainableDAGPlanner(
        SustainableDAGConfig(
            switch_penalty=cfg.dag_switch_penalty,
            stay_bonus=cfg.dag_stay_bonus,
            outage_penalty=cfg.dag_outage_penalty,
            latency_penalty_scale=cfg.dag_latency_penalty_scale,
            topk_per_slot=cfg.dag_topk_per_slot,
            allow_skip_edge=getattr(cfg, 'dag_allow_skip_edge', True),
        )
    )
    out = planner.plan(
        A_final=build_res.A_final,
        utility=build_res.utility,
        latency_ms=build_res.latency_ms,
    )
    return out["planned_idx"]


def predicted_plan_from_predicted_env(predicted_build_res, cfg: ExperimentConfig) -> np.ndarray:
    """
    네 연구 포인트가 consistency 비교이므로:
    'Predicted' 계열은 동일 predicted coverage/utility 기반 계획을 공유하고,
    실행 단계에서 diffusion vs consistency runtime 차이만 반영.
    """
    return sustainable_dag_plan(predicted_build_res, cfg)


# =========================================================
# Runtime / Execution Simulator
# =========================================================

def inference_latency_ms(mode: str, steps: int, runtime_cfg: RuntimeModelConfig) -> float:
    if mode == "consistency":
        return runtime_cfg.base_ms_consistency + runtime_cfg.per_step_ms_consistency * steps
    elif mode == "diffusion":
        return runtime_cfg.base_ms_diffusion + runtime_cfg.per_step_ms_diffusion * steps
    elif mode in ("reactive", "sustainable_dag"):
        return 0.0
    else:
        return 0.0


def execute_plan_on_env(
    build_res,
    planned_idx: np.ndarray,
    method_name: str,
    mode: str,
    sampling_steps: int,
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    계획(planned_idx)을 실제 환경(A_final, latency)에 적용.
    diffusion/consistency는 runtime budget 초과 시 deadline miss -> HO 수행 실패 가능.
    """
    T, N = build_res.A_final.shape
    A = build_res.A_final
    latency_mat = build_res.latency_ms
    meta = build_res.meta_all
    runtime_cfg = cfg.runtime

    inf_ms = inference_latency_ms(mode, sampling_steps, runtime_cfg)
    deadline_miss_global = 1 if inf_ms > runtime_cfg.decision_budget_ms else 0

    # 타임라인 기록
    rows = []
    serving_idx = -1
    prev_serving_idx = -1
    ho_attempt_count = 0
    ho_failure_count = 0
    ho_success_count = 0

    interruption_hist = []
    latency_hist = []
    jitter_hist = []
    outages = []
    deadline_misses = []

    # 핑퐁 계산용
    serving_history = []

    for t in range(T):
        planned = int(planned_idx[t]) if t < len(planned_idx) else -1
        phase_label = str(build_res.traj.phase_label[t])

        # 슬롯별 deadline miss를 약간 확률적으로 (diffusion steps 커질수록 악화)
        if mode == "diffusion":
            # budget 초과량 기반 확률 (step sweep에서 점진적 악화 보이게)
            p_miss = max(0.0, min(1.0, (inf_ms - runtime_cfg.decision_budget_ms) / 20.0))
            dmiss = 1 if np.random.random() < p_miss else 0
        else:
            dmiss = deadline_miss_global

        deadline_misses.append(dmiss)

        ho_attempt = 0
        ho_success = 0
        ho_failure = 0
        outage = 0
        interruption_ms = 0.0

        # 현재 serving 유지 가능 여부 먼저 확인
        current_valid = (serving_idx >= 0 and serving_idx < N and A[t, serving_idx] == 1)

        # 계획 후보 유효 여부
        planned_valid = (planned >= 0 and planned < N and A[t, planned] == 1)

        if planned_valid:
            if serving_idx == -1:
                # 최초 attach 시도
                ho_attempt = 1
                ho_attempt_count += 1
                if dmiss == 0:
                    serving_idx = planned
                    ho_success = 1
                    ho_success_count += 1
                    interruption_ms = 30.0  # attach short interruption
                else:
                    ho_failure = 1
                    ho_failure_count += 1
                    outage = 1
                    interruption_ms = runtime_cfg.reattach_wait_ms
            elif planned != serving_idx:
                # handover 시도
                ho_attempt = 1
                ho_attempt_count += 1
                if dmiss == 0:
                    prev_serving_idx = serving_idx
                    serving_idx = planned
                    ho_success = 1
                    ho_success_count += 1
                    interruption_ms = 40.0  # make-before-break 가정의 짧은 interruption
                else:
                    ho_failure = 1
                    ho_failure_count += 1
                    # 실패 시 기존 링크 유지 가능하면 유지, 아니면 outage
                    if current_valid:
                        interruption_ms = 5.0   # 실패했지만 기존 유지
                    else:
                        outage = 1
                        interruption_ms = runtime_cfg.reattach_wait_ms
            else:
                # 유지
                interruption_ms = 0.0
        else:
            # 계획 후보가 없음/무효 -> 기존 유지 가능하면 유지, 아니면 outage
            if current_valid:
                interruption_ms = 0.0
            else:
                serving_idx = -1
                outage = 1
                interruption_ms = runtime_cfg.reattach_wait_ms

        # 실행 후 serving 유효성 재검사
        if serving_idx >= 0 and A[t, serving_idx] != 1:
            serving_idx = -1
            outage = 1
            interruption_ms = max(interruption_ms, runtime_cfg.reattach_wait_ms)

        # latency / jitter 계산
        if serving_idx >= 0:
            lat_ms = float(latency_mat[t, serving_idx])
        else:
            lat_ms = np.nan

        # jitter proxy = interruption + processing + (latency variation small term)
        if math.isnan(lat_ms):
            jitter_ms = runtime_cfg.node_processing_delay_ms + interruption_ms + 10.0
        else:
            jitter_ms = runtime_cfg.node_processing_delay_ms + interruption_ms + 0.1 * lat_ms

        interruption_hist.append(interruption_ms)
        latency_hist.append(lat_ms)
        jitter_hist.append(jitter_ms)
        outages.append(outage)
        serving_history.append(serving_idx)

        rows.append({
            "t_idx": t,
            "sim_time_sec": t * cfg.dt_sec,
            "phase_label": phase_label,
            "planned_idx": planned,
            "serving_idx": serving_idx,
            "outage": outage,
            "deadline_miss": dmiss,
            "ho_attempt": ho_attempt,
            "ho_success": ho_success,
            "ho_failure": ho_failure,
            "interruption_ms": interruption_ms,
            "latency_ms": lat_ms,
            "inference_latency_ms": inf_ms,
        })

    tl = pd.DataFrame(rows)

    # --- Metrics (summary row schema 맞춤) ---
    outage_arr = np.asarray(outages, dtype=float)
    interrupt_arr = np.asarray(interruption_hist, dtype=float)
    latency_arr = np.asarray(latency_hist, dtype=float)
    jitter_arr = np.asarray(jitter_hist, dtype=float)
    dmiss_arr = np.asarray(deadline_misses, dtype=float)

    availability = float(1.0 - outage_arr.mean())

    valid_lat = latency_arr[~np.isnan(latency_arr)]
    if len(valid_lat) == 0:
        mean_lat = np.nan
        p95_lat = np.nan
    else:
        mean_lat = float(np.mean(valid_lat))
        p95_lat = float(np.percentile(valid_lat, 95))

    mean_int = float(np.mean(interrupt_arr))
    p95_int = float(np.percentile(interrupt_arr, 95))
    mean_jitter = float(np.mean(jitter_arr))
    p95_jitter = float(np.percentile(jitter_arr, 95))

    ho_failure_ratio = float(ho_failure_count / ho_attempt_count) if ho_attempt_count > 0 else 0.0

    # ping-pong ratio (A->B->A within short window)
    pingpong = 0
    transitions = 0
    for i in range(2, len(serving_history)):
        a, b, c = serving_history[i - 2], serving_history[i - 1], serving_history[i]
        if a >= 0 and b >= 0 and c >= 0 and a != b:
            transitions += 1
            if a == c:
                pingpong += 1
    pingpong_ratio = float(pingpong / transitions) if transitions > 0 else 0.0

    deadline_miss_ratio = float(dmiss_arr.mean())
    decision_completion_rate = float(1.0 - deadline_miss_ratio)

    # Effective QoE (0~1 스케일 proxy)
    # - Availability 높을수록 좋음
    # - interruption / jitter / latency 낮을수록 좋음
    lat_term = 0.0 if np.isnan(mean_lat) else np.clip(1.0 - mean_lat / 100.0, 0.0, 1.0)
    int_term = np.clip(1.0 - mean_int / 1000.0, 0.0, 1.0)
    jit_term = np.clip(1.0 - mean_jitter / 1000.0, 0.0, 1.0)
    eff_qoe = float(0.45 * availability + 0.25 * lat_term + 0.15 * int_term + 0.15 * jit_term)

    metrics = {
        "Method": method_name,
        "Mode": mode,
        "SamplingSteps": int(sampling_steps),
        "Scenario": str(getattr(cfg, "stress_profile", "easy")),

        "Availability": availability,
        "MeanInterruption_ms": mean_int,
        "P95Interruption_ms": p95_int,

        "MeanLatency_ms": mean_lat,
        "P95Latency_ms": p95_lat,

        "MeanJitter_ms": mean_jitter,
        "P95Jitter_ms": p95_jitter,

        "HO_Failure_Ratio": ho_failure_ratio,
        "PingPong_Ratio": pingpong_ratio,

        "EffectiveQoE": eff_qoe,

        "DeadlineMissRatio": deadline_miss_ratio,
        "DecisionCompletionRate": decision_completion_rate,

        "HO_Attempt_Count": int(ho_attempt_count),
        "HO_Failure_Count": int(ho_failure_count),
    }

    return tl, metrics


# =========================================================
# Summary helper (기존 plot 스키마 호환용: *_mean, *_std)
# =========================================================

def summarize_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    # 기본 그룹 키
    group_cols = ["Method", "Mode", "SamplingSteps"]

    # 스트레스/시나리오 구분 컬럼이 있으면 그룹 키에 포함 (혼합 집계를 방지)
    for c in ["StressProfile", "stress_profile", "Scenario", "scenario", "profile", "stress"]:
        if c in df_runs.columns and c not in group_cols:
            group_cols.append(c)
            break

    # 숫자형 metric만 집계 대상으로 선택 (문자열 컬럼 mean 집계 에러 방지)
    metric_cols = []
    for c in df_runs.columns:
        if c in group_cols or c == "seed":
            continue
        if pd.api.types.is_numeric_dtype(df_runs[c]):
            metric_cols.append(c)

    agg_map = {c: ["mean", "std"] for c in metric_cols}
    agg_map["seed"] = ["count"]

    g = df_runs.groupby(group_cols, as_index=False).agg(agg_map)

    # MultiIndex 컬럼 평탄화
    new_cols = []
    for col in g.columns:
        if isinstance(col, tuple):
            base, stat = col[0], col[1]
            if base in group_cols and (stat == "" or stat is None):
                new_cols.append(base)
            elif base == "seed" and stat == "count":
                new_cols.append("n_runs")
            else:
                new_cols.append(f"{base}_{stat}")
        else:
            new_cols.append(col)
    g.columns = new_cols

    # groupby(as_index=False) 결과에 따라 Method_/Mode_/SamplingSteps_ 로 들어오는 것 방지
    rename_fix = {}
    for c in g.columns:
        if c.startswith("Method_"):
            rename_fix[c] = "Method"
        elif c.startswith("Mode_"):
            rename_fix[c] = "Mode"
        elif c.startswith("SamplingSteps_"):
            rename_fix[c] = "SamplingSteps"
    if rename_fix:
        g = g.rename(columns=rename_fix)

    return g

# =========================================================
# Experiments
# =========================================================

