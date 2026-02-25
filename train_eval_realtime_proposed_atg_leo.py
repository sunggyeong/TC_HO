from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any
import copy
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# === 기존 실험 파일에서 재사용 ===
from experiments_atg_leo import (
    ExperimentConfig,
    build_env,
    execute_plan_on_env,
    sustainable_dag_plan,
    summarize_runs,
    predicted_plan_from_predicted_env,   # 추가
)

# === 모델 import ===
from models.proposed_tc_planner import OnlineTransformerPredictor, OnlineConsistencyGenerator


# =========================================================
# Config
# =========================================================

@dataclass
class TrainEvalRealtimeConfig:
    exp_cfg: ExperimentConfig = field(default_factory=ExperimentConfig)

    # 데이터 분할
    train_seeds: List[int] = field(default_factory=lambda: [0,1,2])
    val_seeds: List[int] = field(default_factory=lambda: [3])
    test_seeds: List[int] = field(default_factory=lambda: [4])

    # 모델/학습
    L: int = 10
    H: int = 10
    epochs: int = 3
    stride: int = 5
    lr: float = 1e-3
    lambda_consistency: float = 1.0
    reward_invalid_penalty: float = -1.0

    # 평가 시 consistency sampling steps
    consistency_steps_eval: int = 2
    diffusion_steps_eval: int = 12  # predicted diffusion baseline용

    # 실시간 보정 (online correction)
    rt_reliability_alpha: float = 0.90
    rt_low_reliability_threshold: float = 0.45
    rt_fp_extra_penalty: float = 0.90
    rt_use_actual_current_slot_fallback: bool = True

    # 평가 baseline 포함 여부
    eval_include_reactive_baseline: bool = True
    eval_include_predicted_runtime_baselines: bool = True

    # 저장
    results_dir: str = "results"
    weight_path: str = "results/trained_weights_rl_real_env.pth"
    split_manifest_path: str = "results/train_val_test_seed_split.json"


# =========================================================
# Utilities
# =========================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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


def infer_num_nodes_from_pair(pair: Dict[str, Any]) -> int:
    return int(pair["actual_env"].A_final.shape[1])


def validate_env_pairs_shapes(pairs: List[Dict[str, Any]], split_name: str):
    if not pairs:
        raise ValueError(f"[{split_name}] env_pairs is empty")

    ref_N = infer_num_nodes_from_pair(pairs[0])
    for pair in pairs:
        seed = pair["seed"]
        actual_env = pair["actual_env"]
        pred_env = pair["pred_env"]

        if actual_env.A_final.shape != pred_env.A_final.shape:
            raise ValueError(
                f"[{split_name}] seed={seed} shape mismatch: "
                f"actual {actual_env.A_final.shape} vs pred {pred_env.A_final.shape}"
            )

        T, N = actual_env.A_final.shape
        if N != ref_N:
            raise ValueError(
                f"[{split_name}] num_nodes mismatch across seeds. "
                f"seed={seed} has N={N}, but ref_N={ref_N}. "
                "모델 입력 차원이 달라져 학습 불가. (meta universe 고정 or padding 필요)"
            )
        if T <= 0:
            raise ValueError(f"[{split_name}] seed={seed} has empty timeline T={T}")


# =========================================================
# Seed collection (actual / predicted env pair)
# =========================================================

def collect_env_pairs(cfg: ExperimentConfig, seeds: List[int]) -> List[Dict[str, Any]]:
    pairs = []
    for seed in seeds:
        print(f"[collect] seed={seed}")
        actual_env = build_env(cfg, seed=seed, predicted_view=False)
        pred_env = build_env(cfg, seed=seed, predicted_view=True)

        if actual_env.A_final.shape != pred_env.A_final.shape:
            raise ValueError(
                f"shape mismatch at seed={seed}: actual {actual_env.A_final.shape} vs pred {pred_env.A_final.shape}"
            )

        pairs.append(
            {
                "seed": seed,
                "actual_env": actual_env,
                "pred_env": pred_env,
            }
        )
    return pairs


# =========================================================
# Training (predicted history -> actual future target)
# =========================================================

def train_on_seed_pool(
    env_pairs: List[Dict[str, Any]],
    cfg_te: TrainEvalRealtimeConfig,
):
    if not env_pairs:
        raise ValueError("train_on_seed_pool: env_pairs is empty")

    validate_env_pairs_shapes(env_pairs, "train")

    device = get_device()
    print(f"🚀 training start | device={device}")

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
        seed_everything(1000 + epoch)  # epoch단위 재현성 보강

        transformer.train()
        online_consistency.train()

        epoch_loss = 0.0
        epoch_loss_tf = 0.0
        epoch_loss_rl = 0.0
        epoch_loss_cs = 0.0
        epoch_reward = 0.0
        step_count = 0
        skipped_short = 0

        for pair in env_pairs:
            actual_env = pair["actual_env"]
            pred_env = pair["pred_env"]

            A_in = np.asarray(pred_env.A_final, dtype=np.float32)        # 입력 = predicted
            A_target = np.asarray(actual_env.A_final, dtype=np.float32)  # 타깃 = actual
            U_target = np.asarray(actual_env.utility, dtype=np.float32)  # 보상 = actual utility

            T, N2 = A_in.shape
            if N2 != N:
                raise ValueError(f"num_nodes mismatch in training: expected {N}, got {N2}")

            if T <= (L + H):
                skipped_short += 1
                continue

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

                step_2 = torch.tensor([[2.0]], dtype=torch.float32, device=device)
                step_1 = torch.tensor([[1.0]], dtype=torch.float32, device=device)

                y_curr = online_consistency(y_noisy_t1, condition, step_1)

                # 3) RL-style reward from actual future
                target_time_offset = int(y_curr[0, 0].item() * (H - 1))
                target_node_idx = int(y_curr[0, 1].item() * (N - 1))

                target_time_offset = min(max(target_time_offset, 0), H - 1)
                target_node_idx = min(max(target_node_idx, 0), N - 1)

                if future_A_true[target_time_offset, target_node_idx] == 1:
                    reward_val = float(U_target[t + target_time_offset, target_node_idx])
                else:
                    reward_val = float(cfg_te.reward_invalid_penalty)

                epoch_reward += reward_val

                reward_tensor = torch.tensor(
                    np.clip(reward_val, -1.0, 1.0),
                    dtype=torch.float32,
                    device=device
                )
                loss_rl = -reward_tensor * y_curr.mean()

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

        if step_count == 0:
            raise RuntimeError(
                "No training windows were generated (step_count=0). "
                "Check timeline length T, or reduce L/H, or adjust stride."
            )

        row = {
            "epoch": epoch + 1,
            "steps": step_count,
            "skipped_short_seeds": skipped_short,
            "avg_total_loss": epoch_loss / step_count,
            "avg_tf_loss": epoch_loss_tf / step_count,
            "avg_rl_loss": epoch_loss_rl / step_count,
            "avg_cs_loss": epoch_loss_cs / step_count,
            "avg_reward": epoch_reward / step_count,
            "elapsed_sec": time.time() - t0,
        }
        history_rows.append(row)

        print(
            f"[train] epoch {epoch+1:02d}/{cfg_te.epochs} | "
            f"loss={row['avg_total_loss']:.4f} "
            f"(tf={row['avg_tf_loss']:.4f}, rl={row['avg_rl_loss']:.4f}, cs={row['avg_cs_loss']:.4f}) | "
            f"reward={row['avg_reward']:.4f} | steps={row['steps']} | "
            f"skipped_short={row['skipped_short_seeds']} | elapsed={row['elapsed_sec']:.1f}s"
        )

    transformer.eval()
    online_consistency.eval()

    Path(cfg_te.results_dir).mkdir(parents=True, exist_ok=True)
    save_weights(transformer, online_consistency, cfg_te.weight_path)

    hist_df = pd.DataFrame(history_rows)
    hist_path = Path(cfg_te.results_dir) / "train_history_real_env.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"✅ train history saved: {hist_path}")

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
    state_tensor = torch.tensor(history_A, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        future_pred = transformer(state_tensor)
        condition = future_pred.view(1, -1)

        y_curr = torch.randn(1, 2, device=device)
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

        if target_time_offset == 0 and A_pred[t, target_node_idx] == 1:
            current_node = int(target_node_idx)

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
    device = next(transformer.parameters()).device

    A_actual = np.asarray(actual_env.A_final)
    U_actual = np.asarray(actual_env.utility)
    A_pred = np.asarray(pred_env.A_final)

    T, N = A_actual.shape
    L, H = cfg_te.L, cfg_te.H

    planned_idx = np.full(T, -1, dtype=int)
    current_node = -1

    A_obs_corrected = A_pred.copy()
    reliability = np.ones(N, dtype=np.float32)

    for t in range(T):
        # 현재 슬롯 실제 관측 반영
        A_obs_corrected[t] = A_actual[t]

        if t < L:
            avail = np.where(A_actual[t] == 1)[0]
            if len(avail) > 0:
                score = U_actual[t, avail] * reliability[avail]
                current_node = int(avail[np.argmax(score)])
            else:
                current_node = -1
            planned_idx[t] = current_node

        else:
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

            proposed_valid_now = (
                proposed is not None and proposed >= 0 and proposed < N and A_actual[t, proposed] == 1
            )
            current_valid_now = (
                current_node >= 0 and current_node < N and A_actual[t, current_node] == 1
            )

            low_rel = False
            if proposed is not None and proposed >= 0 and proposed < N:
                low_rel = float(reliability[proposed]) < cfg_te.rt_low_reliability_threshold

            corrected = current_node

            if proposed_valid_now and not low_rel:
                corrected = proposed
            else:
                if cfg_te.rt_use_actual_current_slot_fallback:
                    avail = np.where(A_actual[t] == 1)[0]
                    if len(avail) > 0:
                        score = U_actual[t, avail] * reliability[avail]
                        fallback = int(avail[np.argmax(score)])

                        if current_valid_now and low_rel:
                            corrected = current_node
                        else:
                            corrected = fallback
                    else:
                        corrected = -1 if not current_valid_now else current_node
                else:
                    corrected = current_node if current_valid_now else -1

            if corrected != -1 and A_actual[t, corrected] != 1:
                corrected = -1

            current_node = int(corrected) if corrected is not None else -1
            planned_idx[t] = current_node

        # reliability update
        match = (A_pred[t].astype(np.int32) == A_actual[t].astype(np.int32)).astype(np.float32)
        reliability = cfg_te.rt_reliability_alpha * reliability + (1.0 - cfg_te.rt_reliability_alpha) * match

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
    if not test_pairs:
        raise ValueError("evaluate_on_test_seeds: test_pairs is empty")

    validate_env_pairs_shapes(test_pairs, "test")

    rows = []
    results_dir = Path(cfg_te.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Evaluation: Learned planner on held-out seeds ===")
    for pair in test_pairs:
        seed = int(pair["seed"])
        actual_env = pair["actual_env"]
        pred_env = pair["pred_env"]

        seed_everything(seed)  # np + torch 모두 고정

        # -------------------------------------------------
        # (A) Reactive baseline
        # -------------------------------------------------
        if cfg_te.eval_include_reactive_baseline:
            T = actual_env.A_final.shape[0]
            reactive_dummy_plan = np.full(T, -1, dtype=int)
            tl_re, m_re = execute_plan_on_env(
                build_res=actual_env,
                planned_idx=reactive_dummy_plan,
                method_name="Reactive_SlotBest",
                mode="reactive",
                sampling_steps=0,
                cfg=cfg_te.exp_cfg,
            )
            m_re["seed"] = seed
            rows.append(m_re)

        # -------------------------------------------------
        # (B) Predicted runtime baselines (same predicted plan)
        # -------------------------------------------------
        if cfg_te.eval_include_predicted_runtime_baselines:
            base_pred_plan = predicted_plan_from_predicted_env(pred_env, cfg_te.exp_cfg)

            pred_plan_cons = base_pred_plan.copy()
            # Predicted_Consistency
            tl_pc, m_pc = execute_plan_on_env(
                build_res=actual_env,
                planned_idx=pred_plan_cons,
                method_name="Predicted_Consistency",
                mode="consistency",
                sampling_steps=cfg_te.consistency_steps_eval,
                cfg=cfg_te.exp_cfg,
            )
            m_pc["seed"] = seed
            rows.append(m_pc)
            
            pred_plan_diff = base_pred_plan.copy()
            
            # Predicted_Diffusion
            tl_pd, m_pd = execute_plan_on_env(
                build_res=actual_env,
                planned_idx=pred_plan_diff,
                method_name="Predicted_Diffusion",
                mode="diffusion",
                sampling_steps=cfg_te.diffusion_steps_eval,
                cfg=cfg_te.exp_cfg,
            )
            m_pd["seed"] = seed
            rows.append(m_pd)

        # -------------------------------------------------
        # (C) Oracle/reference baseline
        # -------------------------------------------------
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

        # -------------------------------------------------
        # (D) Learned offline (predicted env only)
        # -------------------------------------------------
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

        # -------------------------------------------------
        # (E) Learned + realtime correction
        # -------------------------------------------------
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

        # 로그 출력 (존재하는 값만)
        msg = [f"[seed={seed}]"]
        if cfg_te.eval_include_reactive_baseline:
            msg.append(f"RE Avail={m_re['Availability']:.4f}, QoE={m_re['EffectiveQoE']:.4f}")
        if cfg_te.eval_include_predicted_runtime_baselines:
            msg.append(f"PC Avail={m_pc['Availability']:.4f}, QoE={m_pc['EffectiveQoE']:.4f}")
            msg.append(f"PD Avail={m_pd['Availability']:.4f}, QoE={m_pd['EffectiveQoE']:.4f}")
        msg.append(f"DAG Avail={m_dag['Availability']:.4f}, QoE={m_dag['EffectiveQoE']:.4f}")
        msg.append(f"OFF Avail={m_off['Availability']:.4f}, QoE={m_off['EffectiveQoE']:.4f}")
        msg.append(f"RT Avail={m_rt['Availability']:.4f}, QoE={m_rt['EffectiveQoE']:.4f}")
        print(" | ".join(msg))

        # 첫 test seed timeline 저장
        if seed == int(test_pairs[0]["seed"]):
            if cfg_te.eval_include_reactive_baseline:
                tl_re.to_csv(results_dir / f"timeline_reactive_seed{seed}.csv", index=False)
            if cfg_te.eval_include_predicted_runtime_baselines:
                tl_pc.to_csv(results_dir / f"timeline_predicted_consistency_eval_seed{seed}.csv", index=False)
                tl_pd.to_csv(results_dir / f"timeline_predicted_diffusion_eval_seed{seed}.csv", index=False)

            tl_dag.to_csv(results_dir / f"timeline_sustainable_dag_seed{seed}.csv", index=False)
            tl_off.to_csv(results_dir / f"timeline_learned_tc_offline_seed{seed}.csv", index=False)
            tl_rt.to_csv(results_dir / f"timeline_learned_tc_rtcorrected_seed{seed}.csv", index=False)

    df_runs = pd.DataFrame(rows)
    df_summary = summarize_runs(df_runs)

    runs_path = results_dir / "exp3_learned_realtime_runs.csv"
    summary_path = results_dir / "exp3_learned_realtime_summary.csv"
    df_runs.to_csv(runs_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print(f"✅ saved: {runs_path}")
    print(f"✅ saved: {summary_path}")
    return df_runs, df_summary


# =========================================================
# Main pipeline
# =========================================================

def main():
    cfg_te = TrainEvalRealtimeConfig()

    results_dir = Path(cfg_te.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not Path(cfg_te.exp_cfg.tle_path).exists():
        raise FileNotFoundError(
            f"Frozen TLE file not found: {cfg_te.exp_cfg.tle_path}\n"
            "먼저 TLE 파일 경로를 맞춰주세요."
        )

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
    val_pairs = collect_env_pairs(cfg_te.exp_cfg, cfg_te.val_seeds)   # 향후 val tuning용
    test_pairs = collect_env_pairs(cfg_te.exp_cfg, cfg_te.test_seeds)

    validate_env_pairs_shapes(train_pairs, "train")
    validate_env_pairs_shapes(val_pairs, "val")
    validate_env_pairs_shapes(test_pairs, "test")
    print(f"[collect done] train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # train/val/test 간 N 일관성 체크
    N_train = infer_num_nodes_from_pair(train_pairs[0])
    N_val = infer_num_nodes_from_pair(val_pairs[0])
    N_test = infer_num_nodes_from_pair(test_pairs[0])
    if not (N_train == N_val == N_test):
        raise ValueError(
            f"num_nodes mismatch across splits: train={N_train}, val={N_val}, test={N_test}. "
            "모델 입력 차원을 통일해야 합니다."
        )

    # 2) train
    transformer, consistency, train_hist = train_on_seed_pool(train_pairs, cfg_te)

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


if __name__ == "__main__":
    main()