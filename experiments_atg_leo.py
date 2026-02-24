# experiments_atg_leo.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
import time
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
    n_runs: int = 3
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

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
    return env.build()


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
            allow_skip_edge=False,
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
    group_cols = ["Method", "Mode", "SamplingSteps"]
    metric_cols = [c for c in df_runs.columns if c not in group_cols + ["seed"]]

    agg_map = {}
    for c in metric_cols:
        agg_map[c] = ["mean", "std"]
    agg_map["seed"] = ["count"]

    g = df_runs.groupby(group_cols, as_index=False).agg(agg_map)

    # MultiIndex 컬럼 평탄화
    new_cols = []
    for col in g.columns:
        if isinstance(col, tuple):
            if col[0] in group_cols and (col[1] == "" or col[1] is None):
                new_cols.append(col[0])
            elif col[0] == "seed" and col[1] == "count":
                new_cols.append("n_runs")
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
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

def run_experiment_1_core_compare(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exp1:
      - Sustainable_DAG_NetworkX (ATG+LEO)
      - Predicted_Diffusion (steps=12)
      - Predicted_Consistency (steps=2)
    """
    methods = [
        {"Method": "Predicted_Consistency", "Mode": "consistency", "SamplingSteps": 2},
        {"Method": "Predicted_Diffusion",   "Mode": "diffusion",   "SamplingSteps": 12},
        {"Method": "Sustainable_DAG_NetworkX", "Mode": "sustainable_dag", "SamplingSteps": 0},
    ]

    all_rows = []
    timeline_saved = False

    print("\n=== Experiment 1: Core Comparison (ATG + LEO, Skyfield+Frozen TLE) ===")
    for seed in cfg.seeds[:cfg.n_runs]:
        np.random.seed(seed)

        # 실행 기준 env (oracle-like actual environment)
        actual_env = build_env(cfg, seed=seed, predicted_view=False)

        # predicted env (예측 오차 반영)
        pred_env = build_env(cfg, seed=seed, predicted_view=True)

        # predicted 계열은 predicted env 기준 동일 계획 공유
        predicted_planned_idx = predicted_plan_from_predicted_env(pred_env, cfg)

        for m in methods:
            name = m["Method"]
            mode = m["Mode"]
            steps = int(m["SamplingSteps"])

            if mode == "sustainable_dag":
                planned_idx = sustainable_dag_plan(actual_env, cfg)
                exec_env = actual_env
            else:
                planned_idx = predicted_planned_idx
                exec_env = actual_env  # 계획은 predicted, 실행평가 는 실제 env

            tl, metrics = execute_plan_on_env(
                build_res=exec_env,
                planned_idx=planned_idx,
                method_name=name,
                mode=mode,
                sampling_steps=steps,
                cfg=cfg,
            )
            metrics["seed"] = seed
            all_rows.append(metrics)

            # seed0 timeline 저장 (predicted diff/cons + sustainable)
            if seed == 0:
                tag = name.lower().replace(" ", "_")
                outp = Path(cfg.results_dir) / f"timeline_{tag}_seed0.csv"
                tl.to_csv(outp, index=False)

                # coverage matrix도 seed0 한 번 저장
                if not timeline_saved:
                    np.save(Path(cfg.results_dir) / "A_atg_seed0.npy", actual_env.A_tn)
                    np.save(Path(cfg.results_dir) / "A_leo_seed0.npy", actual_env.A_leo)
                    np.save(Path(cfg.results_dir) / "M_phase_seed0.npy", actual_env.M_phase)
                    np.save(Path(cfg.results_dir) / "A_final_seed0.npy", actual_env.A_final)
                    
                    meta_df = pd.DataFrame(actual_env.meta_all)
                    #  insert 시 발생할 수 있는 ValueError를 방지하기 위해 'global_idx'가 이미 있다면 안전하게 제거
                    if "global_idx" in meta_df.columns:
                        meta_df = meta_df.drop(columns=["global_idx"])
                    
                    # 'global_idx'를 맨 앞(인덱스 0)에 깔끔하게 삽입
                    meta_df.insert(0, "global_idx", np.arange(len(meta_df), dtype=int))
                    meta_df.to_csv(Path(cfg.results_dir) / "candidate_meta_seed0.csv", index=False)
                    timeline_saved = True

            print(
                f"{name:>26s} | mode={mode:14s} | steps={steps:2d} | "
                f"Avail={metrics['Availability']:.4f} | QoE={metrics['EffectiveQoE']:.4f} | "
                f"DLMiss={metrics['DeadlineMissRatio']:.4f} | Lat(ms)={metrics['MeanLatency_ms']:.2f} | "
                f"Int(ms)={metrics['MeanInterruption_ms']:.2f}"
            )

    df_runs = pd.DataFrame(all_rows)
    df_summary = summarize_runs(df_runs)
    return df_runs, df_summary


def run_experiment_2_step_sweep(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exp2:
      consistency step sweep vs diffusion step sweep
    """
    methods = []
    for s in [1, 2, 4, 8]:
        methods.append({"Method": "Predicted_Consistency", "Mode": "consistency", "SamplingSteps": s})
    for s in [4, 8, 12, 16]:
        methods.append({"Method": "Predicted_Diffusion", "Mode": "diffusion", "SamplingSteps": s})

    all_rows = []

    print("\n=== Experiment 2: Step Budget Sweep (ATG + LEO, Skyfield+Frozen TLE) ===")
    for seed in cfg.seeds[:cfg.n_runs]:
        np.random.seed(seed)

        actual_env = build_env(cfg, seed=seed, predicted_view=False)
        pred_env = build_env(cfg, seed=seed, predicted_view=True)

        predicted_planned_idx = predicted_plan_from_predicted_env(pred_env, cfg)

        for m in methods:
            name = m["Method"]
            mode = m["Mode"]
            steps = int(m["SamplingSteps"])

            tl, metrics = execute_plan_on_env(
                build_res=actual_env,
                planned_idx=predicted_planned_idx,
                method_name=name,
                mode=mode,
                sampling_steps=steps,
                cfg=cfg,
            )
            metrics["seed"] = seed
            all_rows.append(metrics)

            print(
                f"{name:>26s} | mode={mode:12s} | steps={steps:2d} | "
                f"Avail={metrics['Availability']:.4f} | QoE={metrics['EffectiveQoE']:.4f} | "
                f"DLMiss={metrics['DeadlineMissRatio']:.4f} | Lat(ms)={metrics['MeanLatency_ms']:.2f} | "
                f"Int(ms)={metrics['MeanInterruption_ms']:.2f}"
            )

    df_runs = pd.DataFrame(all_rows)
    df_summary = summarize_runs(df_runs)
    return df_runs, df_summary


# =========================================================
# Main
# =========================================================

def main():
    cfg = ExperimentConfig()

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # TLE 파일 체크
    if not Path(cfg.tle_path).exists():
        raise FileNotFoundError(
            f"Frozen TLE 파일이 없습니다: {cfg.tle_path}\n"
            "예) data/starlink_frozen_20260224_1410Z.tle 로 저장 후 실행하세요."
        )

    t0 = time.time()

    exp1_runs, exp1_summary = run_experiment_1_core_compare(cfg)
    exp2_runs, exp2_summary = run_experiment_2_step_sweep(cfg)

    # 저장 (네 기존 파일명 최대한 유지)
    exp1_runs.to_csv(results_dir / "exp1_core_compare_runs.csv", index=False)
    exp1_summary.to_csv(results_dir / "exp1_core_compare_summary.csv", index=False)

    exp2_runs.to_csv(results_dir / "exp2_step_sweep_runs.csv", index=False)
    exp2_summary.to_csv(results_dir / "exp2_step_sweep_summary.csv", index=False)

    dt = time.time() - t0
    print("\nSaved files:")
    print(" - results/exp1_core_compare_runs.csv")
    print(" - results/exp1_core_compare_summary.csv")
    print(" - results/exp2_step_sweep_runs.csv")
    print(" - results/exp2_step_sweep_summary.csv")
    print(" - results/timeline_predicted_consistency_seed0.csv")
    print(" - results/timeline_predicted_diffusion_seed0.csv")
    print(" - results/timeline_sustainable_dag_networkx_seed0.csv")
    print(" - results/A_atg_seed0.npy")
    print(" - results/A_leo_seed0.npy")
    print(" - results/M_phase_seed0.npy")
    print(" - results/A_final_seed0.npy")
    print(" - results/candidate_meta_seed0.csv")
    print(f"\nTotal elapsed: {dt:.2f} sec")


if __name__ == "__main__":
    main()