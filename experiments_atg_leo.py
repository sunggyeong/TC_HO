import os
import csv
import copy
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Tuple, Optional

import numpy as np

from providers import (
    Waypoint,
    WaypointTrajectoryProvider,
    OracleTrajectoryProvider,
    TNATGProvider,
    SimpleLEOSatelliteProvider,
    PhaseMaskProvider,
)
from env import TN_NTN_Env, assert_atg_leo_only
from policies import StickyReactiveGreedyPolicy, PredictedLookaheadPolicy, PredictedPolicyConfig
from runtime_eval import (
    RuntimeConfig,
    HandoverFailureConfig,
    LinkDelayConfig,
    QoEConfig,
    SimulationConfig,
    generate_inference_latency_trace,
    run_policy_and_evaluate,
)

# ============================================================
# Utilities
# ============================================================

CORE_KPI_KEYS = [
    "Availability",
    "MeanInterruption_ms",
    "P95Interruption_ms",
    "MeanLatency_ms",
    "P95Latency_ms",
    "MeanJitter_ms",
    "P95Jitter_ms",
    "HO_Failure_Ratio",
    "PingPong_Ratio",
    "EffectiveQoE",
    "DeadlineMissRatio",
    "DecisionCompletionRate",
    "HO_Attempt_Count",
    "HO_Failure_Count",
]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_rows_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def aggregate_mean_std(rows: List[Dict[str, Any]], group_keys: List[str], numeric_keys: List[str]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in group_keys)].append(r)

    out = []
    for g, items in buckets.items():
        row = {k: g[i] for i, k in enumerate(group_keys)}
        for nk in numeric_keys:
            vals = [float(it[nk]) for it in items if nk in it and isinstance(it[nk], (int, float, np.integer, np.floating)) and np.isfinite(it[nk])]
            row[f"{nk}_mean"] = float(np.mean(vals)) if vals else np.nan
            row[f"{nk}_std"] = float(np.std(vals)) if vals else np.nan
        row["n_runs"] = len(items)
        out.append(row)
    return out

def extract_kpi_row(method: str, seed: int, kpis: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {"Method": method, "Seed": seed}
    if extra:
        row.update(extra)
    for k in CORE_KPI_KEYS:
        row[k] = kpis.get(k, np.nan)
    return row

def print_summary_table(rows_summary: List[Dict[str, Any]], title: str):
    print(f"\n=== {title} ===")
    # stable sort
    rows_summary = sorted(rows_summary, key=lambda r: (str(r.get("Mode","")), float(r.get("SamplingSteps", 0))))
    for r in rows_summary:
        print(
            f"{r['Method']:>24s} | "
            f"mode={r.get('Mode','-'):<11s} | "
            f"steps={int(r.get('SamplingSteps',0)):>2d} | "
            f"Avail={r.get('Availability_mean', np.nan):.4f} | "
            f"QoE={r.get('EffectiveQoE_mean', np.nan):.4f} | "
            f"DLMiss={r.get('DeadlineMissRatio_mean', np.nan):.4f} | "
            f"Lat(ms)={r.get('MeanLatency_ms_mean', np.nan):.2f} | "
            f"Int(ms)={r.get('MeanInterruption_ms_mean', np.nan):.2f}"
        )

# ============================================================
# Experiment Config
# ============================================================

@dataclass
class ExperimentCommonConfig:
    n_seeds: int = 5
    results_dir: str = "results"

    dt_sec: float = 1.0
    # prototype runtime: transatlantic full 7h is heavy; use subset first
    max_duration_sec: int = 1800  # 30 min prototype. For paper-scale, increase gradually.

    slot_deadline_ms: float = 1000.0

    # predictor proxy
    pred_horizon_steps: int = 20
    pred_discount: float = 0.93
    pred_noise_std: float = 1.0
    pred_fn: float = 0.02
    pred_fp: float = 0.01

    # reactive baseline
    reactive_switch_margin: float = 3.0

    # Exp1 step budgets
    diffusion_steps_exp1: int = 12
    consistency_steps_exp1: int = 2

    # LEO provider
    leo_n_sats: int = 96   # prototype; can scale to 256/512/1584 later
    leo_altitude_m: float = 550_000.0
    leo_min_elevation_deg: float = 10.0

def build_sim_cfg(seed: int, common: ExperimentCommonConfig) -> SimulationConfig:
    return SimulationConfig(
        runtime=RuntimeConfig(
            slot_deadline_ms=common.slot_deadline_ms,
            deadline_miss_action="hold",
            fallback_action_if_invalid="hold",
        ),
        failure=HandoverFailureConfig(
            base_fail_prob_tn=0.01,
            base_fail_prob_leo=0.03,
            extra_fail_if_target_not_next_slot=0.08,
            seed=1000 + seed,
        ),
        link=LinkDelayConfig(
            processing_delay_ms=10.0,
            ho_exec_delay_ms=20.0,
            reattach_wait_ms=50.0,
            add_ho_delay_into_latency=True,
        ),
        qoe=QoEConfig(),
    )

# ============================================================
# Env factory (ATG + LEO only)
# ============================================================

def build_env_factory(common: ExperimentCommonConfig) -> Callable[[int], TN_NTN_Env]:
    def env_factory(seed: int) -> TN_NTN_Env:
        # Rome (FCO) -> Atlantic -> NY (JFK) simplified high-lat-ish route
        # (실제 항로 정확재현 아님, 프로토타입 handover dynamics용)
        wps = [
            Waypoint(41.800, 12.238, 0.0),        # FCO gate
            Waypoint(41.8, 12.0, 3_000.0),        # climb start
            Waypoint(43.0, 8.0, 10_000.0),        # climb
            Waypoint(45.0, -5.0, 10_000.0),       # over W Europe
            Waypoint(52.0, -20.0, 10_000.0),      # Atlantic high-lat
            Waypoint(53.0, -40.0, 10_000.0),      # Atlantic
            Waypoint(49.0, -58.0, 10_000.0),      # near Newfoundland
            Waypoint(43.0, -70.0, 8_000.0),       # descent corridor
            Waypoint(40.641, -73.778, 0.0),       # JFK gate
        ]

        traj_provider = OracleTrajectoryProvider(
            WaypointTrajectoryProvider(
                waypoints=wps,
                speed_mps=250.0,  # ~900 km/h
                max_duration_sec=common.max_duration_sec,
            )
        )

        tnatg_provider = TNATGProvider.demo_rome_ny_atg_nodes()

        leo_provider = SimpleLEOSatelliteProvider(
            n_sats=common.leo_n_sats,
            altitude_m=common.leo_altitude_m,
            min_elevation_deg=common.leo_min_elevation_deg,
            seed=seed,
        )

        phase_provider = PhaseMaskProvider(
            gate_frac=0.05,
            takeoff_climb_frac=0.15,
            cruise_frac=0.60,
            descent_landing_frac=0.15,
            arrival_gate_frac=0.05,
        )

        env = TN_NTN_Env(
            trajectory_provider=traj_provider,
            tnatg_provider=tnatg_provider,
            satellite_provider=leo_provider,      # LEO only (GEO 없음)
            phase_mask_provider=phase_provider,
            dt_sec=common.dt_sec,
        )
        return env
    return env_factory


# ============================================================
# Experiment 1: Core comparison
# Sustainable vs Predicted+Diffusion vs Predicted+Consistency
# ============================================================

def run_experiment_1_core_comparison(
    env_factory: Callable[[int], TN_NTN_Env],
    common: ExperimentCommonConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    rows_runs: List[Dict[str, Any]] = []

    for seed in range(common.n_seeds):
        env = env_factory(seed)
        tensors = env.precompute_coverage_matrices()
        assert_atg_leo_only(tensors)
        T = len(tensors["sim_time_sec"])

        sim_cfg = build_sim_cfg(seed, common)

        # 1) Sustainable reactive proxy
        reactive_policy = StickyReactiveGreedyPolicy(
            score_key="Q_proxy",
            switch_margin=common.reactive_switch_margin,
        )
        inf_reactive = generate_inference_latency_trace(T, mode="reactive", step_budget=0, seed=10_000 + seed)
        _, k_reactive = run_policy_and_evaluate(
            tensors=tensors,
            policy=reactive_policy,
            inference_latency_ms=inf_reactive,
            sim_cfg=sim_cfg,
        )
        rows_runs.append(extract_kpi_row(
            method="Sustainable_Reactive",
            seed=seed,
            kpis=k_reactive,
            extra={"Mode": "reactive", "SamplingSteps": 0}
        ))

        # 2) Predicted (same decision quality proxy)
        pred_policy = PredictedLookaheadPolicy(
            PredictedPolicyConfig(
                horizon_steps=common.pred_horizon_steps,
                discount=common.pred_discount,
                prediction_noise_std=common.pred_noise_std,
                availability_false_negative=common.pred_fn,
                availability_false_positive=common.pred_fp,
                seed=seed,
            ),
            score_key="Q_proxy",
        )

        # 2a) Diffusion runtime
        inf_diff = generate_inference_latency_trace(
            T, mode="diffusion", step_budget=common.diffusion_steps_exp1, seed=20_000 + seed
        )
        _, k_diff = run_policy_and_evaluate(
            tensors=tensors,
            policy=pred_policy,
            inference_latency_ms=inf_diff,
            sim_cfg=sim_cfg,
        )
        rows_runs.append(extract_kpi_row(
            method="Predicted_Diffusion",
            seed=seed,
            kpis=k_diff,
            extra={"Mode": "diffusion", "SamplingSteps": common.diffusion_steps_exp1}
        ))

        # 2b) Consistency runtime
        inf_cons = generate_inference_latency_trace(
            T, mode="consistency", step_budget=common.consistency_steps_exp1, seed=30_000 + seed
        )
        _, k_cons = run_policy_and_evaluate(
            tensors=tensors,
            policy=pred_policy,
            inference_latency_ms=inf_cons,
            sim_cfg=sim_cfg,
        )
        rows_runs.append(extract_kpi_row(
            method="Predicted_Consistency",
            seed=seed,
            kpis=k_cons,
            extra={"Mode": "consistency", "SamplingSteps": common.consistency_steps_exp1}
        ))

    rows_summary = aggregate_mean_std(
        rows_runs,
        group_keys=["Method", "Mode", "SamplingSteps"],
        numeric_keys=CORE_KPI_KEYS,
    )

    ensure_dir(common.results_dir)
    save_rows_csv(os.path.join(common.results_dir, "exp1_core_compare_runs.csv"), rows_runs)
    save_rows_csv(os.path.join(common.results_dir, "exp1_core_compare_summary.csv"), rows_summary)

    return rows_runs, rows_summary


# ============================================================
# Experiment 2: Step budget sweep
# ============================================================

def run_experiment_2_step_sweep(
    env_factory: Callable[[int], TN_NTN_Env],
    common: ExperimentCommonConfig,
    diffusion_steps_list: Optional[List[int]] = None,
    consistency_steps_list: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    if diffusion_steps_list is None:
        diffusion_steps_list = [4, 8, 12, 16]
    if consistency_steps_list is None:
        consistency_steps_list = [1, 2, 4, 8]

    rows_runs: List[Dict[str, Any]] = []

    for seed in range(common.n_seeds):
        env = env_factory(seed)
        tensors = env.precompute_coverage_matrices()
        assert_atg_leo_only(tensors)
        T = len(tensors["sim_time_sec"])

        sim_cfg = build_sim_cfg(seed, common)

        pred_policy = PredictedLookaheadPolicy(
            PredictedPolicyConfig(
                horizon_steps=common.pred_horizon_steps,
                discount=common.pred_discount,
                prediction_noise_std=common.pred_noise_std,
                availability_false_negative=common.pred_fn,
                availability_false_positive=common.pred_fp,
                seed=seed,
            ),
            score_key="Q_proxy",
        )

        # Diffusion sweep
        for steps in diffusion_steps_list:
            inf = generate_inference_latency_trace(T, mode="diffusion", step_budget=steps, seed=40_000 + seed * 100 + steps)
            _, k = run_policy_and_evaluate(tensors, pred_policy, inf, sim_cfg)
            rows_runs.append(extract_kpi_row(
                method="Predicted_Diffusion",
                seed=seed,
                kpis=k,
                extra={"Mode": "diffusion", "SamplingSteps": steps}
            ))

        # Consistency sweep
        for steps in consistency_steps_list:
            inf = generate_inference_latency_trace(T, mode="consistency", step_budget=steps, seed=50_000 + seed * 100 + steps)
            _, k = run_policy_and_evaluate(tensors, pred_policy, inf, sim_cfg)
            rows_runs.append(extract_kpi_row(
                method="Predicted_Consistency",
                seed=seed,
                kpis=k,
                extra={"Mode": "consistency", "SamplingSteps": steps}
            ))

    rows_summary = aggregate_mean_std(
        rows_runs,
        group_keys=["Method", "Mode", "SamplingSteps"],
        numeric_keys=CORE_KPI_KEYS,
    )

    ensure_dir(common.results_dir)
    save_rows_csv(os.path.join(common.results_dir, "exp2_step_sweep_runs.csv"), rows_runs)
    save_rows_csv(os.path.join(common.results_dir, "exp2_step_sweep_summary.csv"), rows_summary)

    return rows_runs, rows_summary


# ============================================================
# Optional: save timeline CSV for 1 seed (MATLAB timeline plots)
# ============================================================

def save_timeline_csv(path: str, tensors: Dict[str, Any], sim_result):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t_idx", "sim_time_sec", "phase_label",
            "planned_idx", "serving_idx",
            "outage", "deadline_miss",
            "ho_attempt", "ho_success", "ho_failure",
            "interruption_ms", "latency_ms", "inference_latency_ms"
        ])
        T = len(tensors["sim_time_sec"])
        for t in range(T):
            w.writerow([
                t,
                float(tensors["sim_time_sec"][t]),
                str(tensors["phase_labels"][t]),
                int(sim_result.planned_idx[t]),
                int(sim_result.serving_idx[t]),
                int(sim_result.outage[t]),
                int(sim_result.deadline_miss[t]),
                int(sim_result.ho_attempt[t]),
                int(sim_result.ho_success[t]),
                int(sim_result.ho_failure[t]),
                float(sim_result.interruption_ms[t]),
                float(sim_result.latency_ms[t]) if np.isfinite(sim_result.latency_ms[t]) else np.nan,
                float(sim_result.inference_latency_ms[t]),
            ])


# ============================================================
# Main
# ============================================================

def main():
    common = ExperimentCommonConfig(
        n_seeds=5,
        results_dir="results",
        dt_sec=1.0,
        max_duration_sec=1800,   # 먼저 30분 프로토타입. 나중에 3600/7200으로 확대
        slot_deadline_ms=1000.0,
        pred_horizon_steps=20,
        pred_discount=0.93,
        pred_noise_std=1.0,
        pred_fn=0.02,
        pred_fp=0.01,
        diffusion_steps_exp1=12,
        consistency_steps_exp1=2,
        leo_n_sats=96,           # 프로토타입. 확장시 256/512/1584
        leo_altitude_m=550_000.0,
        leo_min_elevation_deg=10.0,
    )

    env_factory = build_env_factory(common)

    # Experiment 1
    exp1_runs, exp1_summary = run_experiment_1_core_comparison(env_factory, common)
    print_summary_table(exp1_summary, "Experiment 1: Core Comparison (ATG + LEO)")

    # Save one example timeline for MATLAB (seed=0)
    env0 = env_factory(0)
    tensors0 = env0.precompute_coverage_matrices()
    assert_atg_leo_only(tensors0)
    T0 = len(tensors0["sim_time_sec"])
    sim_cfg0 = build_sim_cfg(0, common)

    pred_policy0 = PredictedLookaheadPolicy(
        PredictedPolicyConfig(
            horizon_steps=common.pred_horizon_steps,
            discount=common.pred_discount,
            prediction_noise_std=common.pred_noise_std,
            availability_false_negative=common.pred_fn,
            availability_false_positive=common.pred_fp,
            seed=0,
        ),
        score_key="Q_proxy",
    )

    inf_diff0 = generate_inference_latency_trace(T0, "diffusion", common.diffusion_steps_exp1, seed=20000)
    sim_diff0, _ = run_policy_and_evaluate(tensors0, pred_policy0, inf_diff0, sim_cfg0)
    save_timeline_csv(os.path.join(common.results_dir, "timeline_predicted_diffusion_seed0.csv"), tensors0, sim_diff0)

    inf_cons0 = generate_inference_latency_trace(T0, "consistency", common.consistency_steps_exp1, seed=30000)
    sim_cons0, _ = run_policy_and_evaluate(tensors0, pred_policy0, inf_cons0, sim_cfg0)
    save_timeline_csv(os.path.join(common.results_dir, "timeline_predicted_consistency_seed0.csv"), tensors0, sim_cons0)

    # Experiment 2
    exp2_runs, exp2_summary = run_experiment_2_step_sweep(
        env_factory, common,
        diffusion_steps_list=[4, 8, 12, 16],
        consistency_steps_list=[1, 2, 4, 8],
    )
    print_summary_table(exp2_summary, "Experiment 2: Step Budget Sweep (ATG + LEO)")

    print("\nSaved files:")
    print(" - results/exp1_core_compare_runs.csv")
    print(" - results/exp1_core_compare_summary.csv")
    print(" - results/exp2_step_sweep_runs.csv")
    print(" - results/exp2_step_sweep_summary.csv")
    print(" - results/timeline_predicted_diffusion_seed0.csv")
    print(" - results/timeline_predicted_consistency_seed0.csv")


if __name__ == "__main__":
    main()