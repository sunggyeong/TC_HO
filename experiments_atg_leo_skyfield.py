# experiments_atg_leo_skyfield.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from providers.trajectory_provider import Waypoint, WaypointTrajectoryProvider
from providers.tn_atg_provider import TNATGNode, TNATGProvider
from providers.satellite_provider_skyfield import SkyfieldLEOConfig, SkyfieldFrozenTLEProvider
from providers.phase_mask_provider import PhaseMaskProvider
from envs.tn_ntn_env import TN_NTN_Env
from baselines.sustainable_dag import SustainableDAGPlanner, SustainableDAGConfig


def build_demo_atg_nodes():
    """
    예시: 로마 -> 뉴욕 대서양 항로를 대략 커버하는 ATG/연안 노드 (데모용)
    실제 논문에서는 항공/해상 통신망 가정에 맞게 위치 정교화.
    """
    return [
        TNATGNode("ATG_ROMA",   41.9,  12.5, kind="ATG", radius_km=60, base_latency_ms=8.0, capacity_score=1.00),
        TNATGNode("ATG_W_MED",  40.0,   5.0, kind="ATG", radius_km=90, base_latency_ms=9.0, capacity_score=0.95),
        TNATGNode("ATG_E_ATL",  45.0, -15.0, kind="ATG", radius_km=120, base_latency_ms=10.0, capacity_score=0.92),
        TNATGNode("ATG_C_ATL",  48.0, -35.0, kind="ATG", radius_km=120, base_latency_ms=10.5, capacity_score=0.90),
        TNATGNode("ATG_W_ATL",  46.0, -55.0, kind="ATG", radius_km=120, base_latency_ms=10.0, capacity_score=0.92),
        TNATGNode("ATG_NY",     40.7, -74.0, kind="ATG", radius_km=70, base_latency_ms=8.0, capacity_score=1.00),
    ]


def build_demo_waypoints():
    """
    로마 -> 뉴욕 (고도 포함) 간단 waypoint.
    실제로는 ADS-B / flight plan 기반으로 넣으면 더 좋음.
    """
    return [
        Waypoint(41.800, 12.250,   0.0),      # gate/taxi (Roma 부근)
        Waypoint(41.900, 11.500, 2000.0),     # takeoff/climb
        Waypoint(43.000,  4.000, 8000.0),
        Waypoint(46.000, -15.000, 10000.0),   # cruise
        Waypoint(49.000, -35.000, 10000.0),   # transatlantic
        Waypoint(47.000, -55.000, 10000.0),
        Waypoint(42.000, -70.000, 6000.0),    # descent
        Waypoint(40.800, -74.000, 500.0),     # landing approach
        Waypoint(40.700, -74.000,   0.0),     # gate (NY)
    ]


def summarize_plan(build_res, planned_idx):
    A = build_res.A_final
    lat = build_res.latency_ms
    T = A.shape[0]

    outage = np.zeros(T, dtype=np.uint8)
    valid_choice = np.zeros(T, dtype=np.uint8)
    chosen_latency = np.full(T, np.nan, dtype=float)
    chosen_domain = np.array(["NONE"] * T, dtype=object)

    for t in range(T):
        idx = int(planned_idx[t])
        if idx < 0:
            outage[t] = 1
            continue
        if A[t, idx] == 1:
            valid_choice[t] = 1
            chosen_latency[t] = lat[t, idx]
            chosen_domain[t] = build_res.meta_all[idx]["domain"]
        else:
            outage[t] = 1

    # HO count
    ho_count = 0
    prev = -1
    for t in range(T):
        cur = int(planned_idx[t])
        if cur >= 0 and prev >= 0 and cur != prev:
            ho_count += 1
        if cur >= 0:
            prev = cur

    return {
        "T": T,
        "Availability": float((1 - outage).mean()),
        "MeanLatency_ms": float(np.nanmean(chosen_latency)) if np.any(valid_choice) else np.nan,
        "P95Latency_ms": float(np.nanpercentile(chosen_latency[~np.isnan(chosen_latency)], 95))
            if np.any(valid_choice) else np.nan,
        "HO_Count": int(ho_count),
        "TN_slots": int(np.sum(chosen_domain == "TN")),
        "NTN_slots": int(np.sum(chosen_domain == "NTN")),
    }, outage, chosen_latency, chosen_domain


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 1) providers
    traj_provider = WaypointTrajectoryProvider(
        waypoints=build_demo_waypoints(),
        speed_mps=250.0,          # 900 km/h
        dt_sec=1.0,
    )

    tn_provider = TNATGProvider(build_demo_atg_nodes())

    sat_provider = SkyfieldFrozenTLEProvider(
        SkyfieldLEOConfig(
            tle_path="data/starlink_frozen_20260224.tle",  # <- 실제 frozen TLE 파일
            min_elevation_deg=10.0,
            max_sats=300,        # 처음엔 100~300으로 테스트 후 늘리기
            base_latency_ms=20.0
        )
    )

    phase_provider = PhaseMaskProvider()

    # 2) build environment
    env = TN_NTN_Env(
        trajectory_provider=traj_provider,
        tn_provider=tn_provider,
        satellite_provider=sat_provider,
        phase_mask_provider=phase_provider,
    )
    build_res = env.build()

    print("=== Environment Built ===")
    print(f"T (slots): {build_res.A_final.shape[0]}")
    print(f"TN nodes : {build_res.A_atg.shape[1]}")
    print(f"LEO nodes: {build_res.A_leo.shape[1]}")
    print(f"A_final coverage ratio: {build_res.A_final.mean():.4f}")

    # 3) Sustainable DAG baseline
    planner = SustainableDAGPlanner(
        SustainableDAGConfig(
            switch_penalty=0.08,
            stay_bonus=0.02,
            outage_penalty=1.0,
            latency_penalty_scale=0.002,
            topk_per_slot=12,     # 그래프 크기 조절
            allow_skip_edge=False
        )
    )
    out = planner.plan(
        A_final=build_res.A_final,
        utility=build_res.utility,
        latency_ms=build_res.latency_ms,
    )
    planned_idx = out["planned_idx"]

    summary, outage, chosen_latency, chosen_domain = summarize_plan(build_res, planned_idx)

    print("\n=== Sustainable DAG (NetworkX) Summary ===")
    print(f"Graph nodes={out['graph_nodes']}, edges={out['graph_edges']}")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # 4) save coverage matrices / timeline
    np.save(results_dir / "A_atg.npy", build_res.A_atg)
    np.save(results_dir / "A_leo.npy", build_res.A_leo)
    np.save(results_dir / "M_phase.npy", build_res.M_phase)
    np.save(results_dir / "A_final.npy", build_res.A_final)
    np.save(results_dir / "planned_idx_sustainable_dag.npy", planned_idx)

    timeline = pd.DataFrame({
        "t_idx": np.arange(len(planned_idx)),
        "sim_time_sec": np.arange(len(planned_idx)),  # dt=1s 가정
        "phase_label": build_res.traj.phase_label.astype(str),
        "planned_idx": planned_idx,
        "outage": outage,
        "latency_ms": chosen_latency,
        "serving_domain": chosen_domain,
    })
    timeline.to_csv(results_dir / "timeline_sustainable_dag.csv", index=False)

    # candidate meta 저장
    meta_df = pd.DataFrame(build_res.meta_all)
    meta_df.insert(0, "global_idx", np.arange(len(build_res.meta_all)))
    meta_df.to_csv(results_dir / "candidate_meta.csv", index=False)

    print("\nSaved:")
    print(" - results/A_atg.npy")
    print(" - results/A_leo.npy")
    print(" - results/M_phase.npy")
    print(" - results/A_final.npy")
    print(" - results/planned_idx_sustainable_dag.npy")
    print(" - results/timeline_sustainable_dag.csv")
    print(" - results/candidate_meta.csv")


if __name__ == "__main__":
    main()