# train_rl_proposed.py
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import time

# 분리된 모델 아키텍처 불러오기
from models.proposed_tc_planner import OnlineTransformerPredictor, OnlineConsistencyGenerator
# 실험 시뮬레이터 재사용
from experiments_atg_leo import ExperimentConfig, build_env

def update_ema_target(online_net, target_net, tau=0.05):
    """지수 이동 평균(EMA)을 통한 타겟 네트워크 업데이트 (Self-Consistency 보장)"""
    with torch.no_grad():
        for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

def collect_env_pairs_from_simulator(exp_cfg, seeds):
    """
    seed마다:
      - pred_env  : predictor 입력(예측 오차 포함)
      - actual_env: 실제 실행/평가 기준
    를 함께 수집
    """
    env_pairs = []
    for seed in seeds:
        print(f"[collect] seed={seed}")
        actual_env = build_env(exp_cfg, seed=seed, predicted_view=False)
        pred_env = build_env(exp_cfg, seed=seed, predicted_view=True)

        if actual_env.A_final.shape != pred_env.A_final.shape:
            raise ValueError(
                f"shape mismatch @ seed={seed}: "
                f"actual={actual_env.A_final.shape}, pred={pred_env.A_final.shape}"
            )

        env_pairs.append({
            "seed": seed,
            "actual_env": actual_env,
            "pred_env": pred_env,
        })
def train_rl_agent(
    env_pairs_list,
    num_nodes,
    L=20,
    H=30,
    epochs=10,
    stride=5,
    weight_path="trained_weights_rl.pth",
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🚀 RL 및 Consistency 학습 시작 (사용 장치: {device})")

    transformer = OnlineTransformerPredictor(num_nodes=num_nodes, L=L, H=H).to(device)
    online_consistency = OnlineConsistencyGenerator(num_nodes=num_nodes, H=H).to(device)

    # Target Network 생성
    target_consistency = copy.deepcopy(online_consistency)
    target_consistency.eval()
    for param in target_consistency.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        list(transformer.parameters()) + list(online_consistency.parameters()),
        lr=1e-3
    )
    bce_loss_fn = nn.BCELoss()
    lambda_consistency = 1.0

    start_time = time.time()

    for epoch in range(epochs):
        epoch_reward = 0.0
        step_count = 0
        epoch_tf_loss = 0.0
        epoch_rl_loss = 0.0
        epoch_cs_loss = 0.0

        for pair in env_pairs_list:
            actual_env = pair["actual_env"]
            pred_env = pair["pred_env"]

            # 입력은 predicted, 타깃/보상은 actual
            A_in = np.asarray(pred_env.A_final, dtype=np.float32)
            A_target = np.asarray(actual_env.A_final, dtype=np.float32)
            utility_target = np.asarray(actual_env.utility, dtype=np.float32)

            T, N = A_in.shape

            for t in range(L, T - H, stride):
                history_A = A_in[t - L:t, :]
                state_tensor = torch.tensor(history_A, dtype=torch.float32).unsqueeze(0).to(device)

                future_A_true = A_target[t:t + H, :]
                future_tensor_true = torch.tensor(future_A_true, dtype=torch.float32).unsqueeze(0).to(device)

                optimizer.zero_grad()

                # 1) Transformer 학습
                future_pred = transformer(state_tensor)
                loss_tf = bce_loss_fn(future_pred, future_tensor_true)

                # 2) Consistency 행동 생성
                condition = future_pred.view(1, -1).detach()
                y_noisy_t2 = torch.randn(1, 2, device=device)
                y_noisy_t1 = y_noisy_t2 * 0.5

                step_2 = torch.tensor([[2.0]], dtype=torch.float32, device=device)
                step_1 = torch.tensor([[1.0]], dtype=torch.float32, device=device)

                y_curr = online_consistency(y_noisy_t1, condition, step_1)

                # 3) actual env 기준 reward 평가
                target_time_offset = int(y_curr[0, 0].item() * (H - 1))
                target_node_idx = int(y_curr[0, 1].item() * (N - 1))

                target_time_offset = min(max(target_time_offset, 0), H - 1)
                target_node_idx = min(max(target_node_idx, 0), N - 1)

                if future_A_true[target_time_offset, target_node_idx] == 1:
                    reward_val = float(utility_target[t + target_time_offset, target_node_idx])
                else:
                    reward_val = -1.0

                epoch_reward += reward_val

                # 4) RL surrogate loss
                reward_tensor = torch.tensor(
                    np.clip(reward_val, -1.0, 1.0),
                    dtype=torch.float32,
                    device=device
                )
                loss_rl = -reward_tensor * y_curr.mean()

                # 5) Consistency loss
                with torch.no_grad():
                    y_target = target_consistency(y_noisy_t2, condition, step_2)
                loss_cs = F.mse_loss(y_curr, y_target)

                total_loss = loss_tf + loss_rl + (lambda_consistency * loss_cs)
                total_loss.backward()
                optimizer.step()

                update_ema_target(online_consistency, target_consistency)

                step_count += 1
                epoch_tf_loss += float(loss_tf.item())
                epoch_rl_loss += float(loss_rl.item())
                epoch_cs_loss += float(loss_cs.item())

        avg_reward = epoch_reward / max(1, step_count)
        avg_tf = epoch_tf_loss / max(1, step_count)
        avg_rl = epoch_rl_loss / max(1, step_count)
        avg_cs = epoch_cs_loss / max(1, step_count)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Reward={avg_reward:.4f} | TF={avg_tf:.4f} | RL={avg_rl:.4f} | CS={avg_cs:.4f} | "
            f"Elapsed={time.time() - start_time:.1f}s"
        )

    Path(weight_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "transformer_state_dict": transformer.state_dict(),
            "consistency_state_dict": online_consistency.state_dict(),
        },
        weight_path
    )
    print(f"✅ 강화학습 완료! '{weight_path}' 저장 완료.")

if __name__ == "__main__":
    # 1) 실험 설정 재사용 (experiments_atg_leo.py와 동일한 환경 생성 로직)
    exp_cfg = ExperimentConfig()

    # TLE 파일 체크
    if not Path(exp_cfg.tle_path).exists():
        raise FileNotFoundError(
            f"Frozen TLE 파일이 없습니다: {exp_cfg.tle_path}\n"
            "먼저 TLE 파일 경로를 확인하세요."
        )

    # 2) 학습/검증/테스트 seed 분할 (예시)
    # 처음엔 작게 시작해도 됨. 예: train=[0,1,2], test=[3]
    train_seeds = [0, 1, 2, 3, 4, 5]
    test_seeds  = [6, 7]

    print("실제 시뮬레이터 seed 환경을 수집합니다...")
    train_pairs = collect_env_pairs_from_simulator(exp_cfg, train_seeds)
    test_pairs = collect_env_pairs_from_simulator(exp_cfg, test_seeds)  # 지금은 수집만, 추후 평가용

    # num_nodes 자동 추론
    num_nodes = train_pairs[0]["actual_env"].A_final.shape[1]
    print(f"[info] num_nodes={num_nodes}, train_seeds={train_seeds}, test_seeds={test_seeds}")

    # 3) 학습 실행
    train_rl_agent(
        env_pairs_list=train_pairs,
        num_nodes=num_nodes,
        L=20,
        H=30,
        epochs=10,
        stride=5,
        weight_path="results/trained_weights_rl_real_env.pth",
    )

    print("✅ 학습 완료 (실제 시뮬레이터 seed 기반)")
    print("ℹ️ 참고: 현재는 test_pairs를 수집만 했고, 평가는 별도 스크립트/함수로 붙이면 됩니다.")