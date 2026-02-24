from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import numpy as np

from policies import DecisionPolicy


# ============================================================
# Configs
# ============================================================

@dataclass
class RuntimeConfig:
    slot_deadline_ms: float = 1000.0
    deadline_miss_action: str = "hold"      # "hold" or "drop"
    fallback_action_if_invalid: str = "hold" # "hold" or "drop"

@dataclass
class HandoverFailureConfig:
    base_fail_prob_tn: float = 0.01
    base_fail_prob_leo: float = 0.03
    extra_fail_if_target_not_next_slot: float = 0.08
    seed: int = 1234

@dataclass
class LinkDelayConfig:
    processing_delay_ms: float = 10.0
    ho_exec_delay_ms: float = 20.0
    reattach_wait_ms: float = 50.0
    add_ho_delay_into_latency: bool = True

@dataclass
class QoEConfig:
    qoe_w_latency: float = 0.35
    qoe_w_jitter: float = 0.25
    qoe_w_interrupt: float = 0.25
    qoe_w_availability: float = 0.15

@dataclass
class SimulationConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    failure: HandoverFailureConfig = field(default_factory=HandoverFailureConfig)
    link: LinkDelayConfig = field(default_factory=LinkDelayConfig)
    qoe: QoEConfig = field(default_factory=QoEConfig)


# ============================================================
# Outputs
# ============================================================

@dataclass
class SimulationResult:
    planned_idx: np.ndarray
    serving_idx: np.ndarray
    outage: np.ndarray
    deadline_miss: np.ndarray
    ho_attempt: np.ndarray
    ho_success: np.ndarray
    ho_failure: np.ndarray
    interruption_ms: np.ndarray
    latency_ms: np.ndarray
    inference_latency_ms: np.ndarray


# ============================================================
# Runtime latency trace generators (Diffusion vs Consistency)
# ============================================================

def generate_inference_latency_trace(T: int, mode: str, step_budget: int, seed: int = 0) -> np.ndarray:
    """
    Synthetic control-plane inference latency traces.
    목적: 1초 슬롯에서 Diffusion vs Consistency의 runtime 차이를 재현.
    """
    rng = np.random.default_rng(seed)
    mode = str(mode).lower()

    if mode == "diffusion":
        # step 증가에 따라 지연 크게 증가 (and variance 큼)
        base = 120.0 + 55.0 * step_budget
        sigma = 40.0 + 6.0 * step_budget
        x = rng.normal(base, sigma, size=T)
        # occasional spikes
        spike_mask = rng.random(T) < min(0.20, 0.02 * step_budget)
        x[spike_mask] += rng.uniform(120, 380, size=np.sum(spike_mask))
        x = np.maximum(20.0, x)
    elif mode == "consistency":
        # few-step low-latency
        base = 35.0 + 22.0 * step_budget
        sigma = 8.0 + 2.0 * step_budget
        x = rng.normal(base, sigma, size=T)
        spike_mask = rng.random(T) < 0.02
        x[spike_mask] += rng.uniform(20, 80, size=np.sum(spike_mask))
        x = np.maximum(5.0, x)
    elif mode == "reactive":
        x = np.maximum(1.0, rng.normal(5.0, 1.5, size=T))
    else:
        raise ValueError(f"Unknown mode={mode}")
    return x.astype(float)


# ============================================================
# Simulation / KPI
# ============================================================

def _node_type_orbit(tensors: Dict[str, Any], node_idx: int) -> Tuple[str, str]:
    if node_idx < 0:
        return ("NONE", "NONE")
    return (str(tensors["node_types"][node_idx]), str(tensors["node_orbits"][node_idx]))


def _handover_fail_prob(tensors: Dict[str, Any], target_idx: int, t: int, cfg: HandoverFailureConfig) -> float:
    if target_idx < 0:
        return 0.0
    nt, no = _node_type_orbit(tensors, target_idx)
    if nt == "TN":
        p = cfg.base_fail_prob_tn
    elif nt == "NTN" and no == "LEO":
        p = cfg.base_fail_prob_leo
    else:
        p = max(cfg.base_fail_prob_leo, 0.05)

    A = np.asarray(tensors["A_final"], dtype=int)
    if t + 1 < A.shape[0] and A[t + 1, target_idx] == 0:
        p += cfg.extra_fail_if_target_not_next_slot
    return float(min(0.95, max(0.0, p)))


def _count_pingpong(serving_idx: np.ndarray) -> int:
    # A -> B -> A pattern over consecutive valid states
    cnt = 0
    for t in range(2, len(serving_idx)):
        a = serving_idx[t - 2]
        b = serving_idx[t - 1]
        c = serving_idx[t]
        if a >= 0 and b >= 0 and c >= 0 and a == c and a != b:
            cnt += 1
    return cnt


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)):
        return np.nan
    return float(np.nanmean(x))


def _nanp95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, 95))


def _compute_effective_qoe(
    availability: float,
    mean_latency_ms: float,
    mean_jitter_ms: float,
    mean_interrupt_ms: float,
    cfg: QoEConfig,
) -> float:
    # normalize into [0,1]-ish; tunable
    lat_term = np.exp(-max(0.0, mean_latency_ms if np.isfinite(mean_latency_ms) else 1e3) / 80.0)
    jit_term = np.exp(-max(0.0, mean_jitter_ms if np.isfinite(mean_jitter_ms) else 1e3) / 40.0)
    int_term = np.exp(-max(0.0, mean_interrupt_ms if np.isfinite(mean_interrupt_ms) else 1e3) / 120.0)
    av_term = max(0.0, min(1.0, availability))
    qoe = (
        cfg.qoe_w_latency * lat_term
        + cfg.qoe_w_jitter * jit_term
        + cfg.qoe_w_interrupt * int_term
        + cfg.qoe_w_availability * av_term
    )
    return float(qoe)


def run_policy_and_evaluate(
    tensors: Dict[str, Any],
    policy: DecisionPolicy,
    inference_latency_ms: np.ndarray,
    sim_cfg: SimulationConfig,
) -> Tuple[SimulationResult, Dict[str, float]]:
    A = np.asarray(tensors["A_final"], dtype=int)
    Prop = np.asarray(tensors["PropDelay_ms"], dtype=float)
    T, N = A.shape

    if len(inference_latency_ms) != T:
        raise ValueError("inference_latency_ms length mismatch")

    planned = np.asarray(policy.decide_sequence(tensors), dtype=int)
    if len(planned) != T:
        raise ValueError("policy output length mismatch")

    serving = np.full(T, -1, dtype=int)
    outage = np.zeros(T, dtype=int)
    deadline_miss = np.zeros(T, dtype=int)
    ho_attempt = np.zeros(T, dtype=int)
    ho_success = np.zeros(T, dtype=int)
    ho_failure = np.zeros(T, dtype=int)
    interruption_ms = np.zeros(T, dtype=float)
    latency_ms = np.full(T, np.nan, dtype=float)

    rng = np.random.default_rng(sim_cfg.failure.seed)

    prev = -1
    for t in range(T):
        # runtime deadline check
        if inference_latency_ms[t] > sim_cfg.runtime.slot_deadline_ms:
            deadline_miss[t] = 1
            if sim_cfg.runtime.deadline_miss_action == "drop":
                chosen = -1
            else:
                chosen = prev
        else:
            chosen = int(planned[t])

        # validity at current slot
        if chosen >= 0 and A[t, chosen] == 0:
            if sim_cfg.runtime.fallback_action_if_invalid == "drop":
                chosen = -1
            else:
                if prev >= 0 and A[t, prev] == 1:
                    chosen = prev
                else:
                    chosen = -1

        # no valid current link
        if chosen < 0:
            serving[t] = -1
            outage[t] = 1
            interruption_ms[t] = sim_cfg.link.reattach_wait_ms
            prev = -1
            continue

        # handover event?
        is_ho = (prev != chosen)
        if is_ho:
            ho_attempt[t] = 1
            p_fail = _handover_fail_prob(tensors, chosen, t, sim_cfg.failure)
            fail = (rng.random() < p_fail)
            if fail:
                ho_failure[t] = 1
                outage[t] = 1
                interruption_ms[t] = sim_cfg.link.reattach_wait_ms

                # failed HO -> if prev still valid, remain prev; else disconnect
                if prev >= 0 and A[t, prev] == 1:
                    serving[t] = prev
                    outage[t] = 0
                    # latency still on prev; interruption recorded
                else:
                    serving[t] = -1
                    prev = -1
                    continue
            else:
                ho_success[t] = 1
                serving[t] = chosen
                interruption_ms[t] = sim_cfg.link.ho_exec_delay_ms
        else:
            serving[t] = chosen

        # latency on serving link
        s = serving[t]
        if s >= 0 and A[t, s] == 1 and np.isfinite(Prop[t, s]):
            lat = sim_cfg.link.processing_delay_ms + float(Prop[t, s])
            if sim_cfg.link.add_ho_delay_into_latency and ho_success[t] == 1:
                lat += sim_cfg.link.ho_exec_delay_ms
            latency_ms[t] = lat
        else:
            outage[t] = 1
            latency_ms[t] = np.nan

        prev = serving[t]

    # KPIs
    valid_service_slots = (serving >= 0) & (outage == 0)
    availability = float(np.mean(valid_service_slots.astype(float)))

    # Jitter as |lat[t]-lat[t-1]| for consecutive valid finite samples
    jitter = []
    for t in range(1, T):
        if np.isfinite(latency_ms[t]) and np.isfinite(latency_ms[t - 1]):
            jitter.append(abs(float(latency_ms[t]) - float(latency_ms[t - 1])))
    jitter = np.array(jitter, dtype=float) if len(jitter) > 0 else np.array([], dtype=float)

    ho_attempt_count = int(np.sum(ho_attempt))
    ho_failure_count = int(np.sum(ho_failure))
    ho_failure_ratio = float(ho_failure_count / ho_attempt_count) if ho_attempt_count > 0 else 0.0

    pingpong_count = _count_pingpong(serving)
    pingpong_ratio = float(pingpong_count / ho_attempt_count) if ho_attempt_count > 0 else 0.0

    mean_latency = _nanmean(latency_ms)
    p95_latency = _nanp95(latency_ms)
    mean_jitter = _nanmean(jitter)
    p95_jitter = _nanp95(jitter)

    # interruption metrics on slots with any interruption > 0
    int_samples = interruption_ms[interruption_ms > 0]
    mean_interrupt = float(np.mean(int_samples)) if len(int_samples) > 0 else 0.0
    p95_interrupt = float(np.percentile(int_samples, 95)) if len(int_samples) > 0 else 0.0

    deadline_miss_ratio = float(np.mean(deadline_miss.astype(float)))
    decision_completion_rate = 1.0 - deadline_miss_ratio

    effective_qoe = _compute_effective_qoe(
        availability=availability,
        mean_latency_ms=(mean_latency if np.isfinite(mean_latency) else 1e3),
        mean_jitter_ms=(mean_jitter if np.isfinite(mean_jitter) else 1e3),
        mean_interrupt_ms=mean_interrupt,
        cfg=sim_cfg.qoe,
    )

    kpis = {
        "Availability": availability,
        "MeanInterruption_ms": mean_interrupt,
        "P95Interruption_ms": p95_interrupt,
        "MeanLatency_ms": mean_latency,
        "P95Latency_ms": p95_latency,
        "MeanJitter_ms": mean_jitter,
        "P95Jitter_ms": p95_jitter,
        "HO_Failure_Ratio": ho_failure_ratio,
        "PingPong_Ratio": pingpong_ratio,
        "EffectiveQoE": effective_qoe,
        "DeadlineMissRatio": deadline_miss_ratio,
        "DecisionCompletionRate": decision_completion_rate,
        "HO_Attempt_Count": float(ho_attempt_count),
        "HO_Failure_Count": float(ho_failure_count),
    }

    res = SimulationResult(
        planned_idx=planned,
        serving_idx=serving,
        outage=outage,
        deadline_miss=deadline_miss,
        ho_attempt=ho_attempt,
        ho_success=ho_success,
        ho_failure=ho_failure,
        interruption_ms=interruption_ms,
        latency_ms=latency_ms,
        inference_latency_ms=np.asarray(inference_latency_ms, dtype=float),
    )
    return res, kpis