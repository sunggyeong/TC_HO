import torch
import torch.nn as nn
import numpy as np

class ConsistencyGenerator(nn.Module):
    """
    노이즈(y_s), 조건(c_k), 스텝(s)을 입력받아 최적의 핸드오버 시점을 소수 스텝 내에 생성하는 모델
    """
    def __init__(self, condition_dim=180, action_dim=1, step_embed_dim=32, hidden_dim=128):
        super(ConsistencyGenerator, self).__init__()
        # 스텝(s) 정보를 모델이 이해할 수 있도록 임베딩
        self.step_mlp = nn.Sequential(
            nn.Linear(1, step_embed_dim),
            nn.SiLU(),
            nn.Linear(step_embed_dim, step_embed_dim)
        )
        
        # 노이즈(현재 상태의 y)와 조건(Transformer 출력), 스텝 임베딩을 모두 결합하는 MLP 네트워크
        self.net = nn.Sequential(
            nn.Linear(action_dim + condition_dim + step_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid() # 0 ~ 1 사이의 값으로 정규화하여 출력 (이후 0 ~ H 스텝으로 매핑)
        )

    def forward(self, y_noisy, condition, step):
        """
        y_noisy: (Batch, 1) - 노이즈가 낀 핸드오버 시점
        condition: (Batch, 180) - Transformer의 예측 결과 (H=30 * Feature=6차원을 Flatten한 값)
        step: (Batch, 1) - 현재 샘플링 스텝 (K, K-1, ..., 1)
        """
        # 스텝 임베딩 생성
        s_embed = self.step_mlp(step)
        
        # 모든 정보 결합 후 네트워크 통과
        combined_input = torch.cat([y_noisy, condition, s_embed], dim=-1)
        y_0_pred = self.net(combined_input)
        
        return y_0_pred

class HandoverRewardEvaluator:
    """
    Generative 논문을 기반으로 샘플링된 후보들의 통합 보상을 평가하는 모듈
    """
    def __init__(self, H=30, w1=0.4, w2=0.3, w3=0.3, kappa=10.0):
        self.H = H
        self.w1 = w1 # Jitter 가중치
        self.w2 = w2 # Load Difference 가중치
        self.w3 = w3 # Robustness 가중치
        self.kappa = kappa # 실패 페널티
        
    def evaluate(self, y_pred_norm, future_states):
        """
        y_pred_norm: 0~1 사이의 예측값
        future_states: Transformer가 예측한 미래 상태 텐서 (Batch, H, 6)
        """
        # 1. 정규화된 출력을 실제 핸드오버 실행 시점(Delta_t)으로 매핑
        # y_pred_norm 값에 예측 스텝(H)을 곱하고 반올림하여 0 ~ H 사이의 정수로 변환
        delta_t = torch.round(y_pred_norm * self.H).int().item()
        delta_t = min(delta_t, self.H - 1)
        # 2. 해당 시점(delta_t)의 네트워크 상태 추출 (Jitter, Load 등을 추정)
        # (프로토타입을 위해 임의의 보상 계산 로직 적용)
        # 미래 상태에서 해당 시점의 TN 거리와 NTN 앙각 데이터를 기반으로 임의의 품질 수치 계산
        target_state = future_states[:, delta_t, :] 
        tn_dist = target_state[0, 4].item() 
        
        # 3. 평가지표 (Metrics) 계산
        # Jitter: 거리가 멀수록 신호가 약해져 지터가 커진다고 가정
        jitter_J = min(tn_dist / 50.0, 1.0) 
        # Load Difference: 시뮬레이터 상의 다른 에이전트 수 기반 (여기선 0.2로 임의 가정)
        load_LD = 0.2 
        # Robustness: 예상되는 신호 유지 시간 기반 견고성
        robustness_HR = 0.8 
        
        # 실패 여부 판단 (기지국 반경 40km를 벗어났는데 핸드오버를 안 한 경우 실패로 간주)
        I_fail = 1.0 if tn_dist > 40.0 else 0.0
        
        # 4. 통합 보상 함수 r_k(t) 계산 (작성하신 논문 수식 반영)
        # alpha 및 beta는 상황에 따라 조절
        alpha, beta = 0.5, 0.1
        omega = 0.8 # 기본 전송률/유지시간 정규화 효용값
        I_HO = 1.0  # 핸드오버 발생 여부
        
        QoE_term = (-self.w1 * jitter_J) - (self.w2 * load_LD) + (self.w3 * robustness_HR)
        
        reward = (alpha * omega) + ((1 - alpha) * QoE_term) - (beta * I_HO) - (self.kappa * I_fail)
        
        return reward, delta_t

# ==========================================
# 실행 테스트: Multi-step 샘플링 및 최고 후보 선정
# ==========================================
if __name__ == "__main__":
    # 파라미터 세팅
    batch_size = 1
    H = 30
    condition_dim = H * 6 # 180차원
    
    # 모델 및 평가기 인스턴스화
    generator = ConsistencyGenerator(condition_dim=condition_dim)
    evaluator = HandoverRewardEvaluator(H=H)
    
    # 1. Transformer가 뱉어낸 미래 예측값이라 가정 (Condition c_k)
    # Shape: (1, 30, 6)을 Flatten -> (1, 180)
    future_states_pred = torch.rand(batch_size, H, 6)
    c_k = future_states_pred.view(batch_size, -1)
    
    print("🚀 Consistency Model 기반 Few-step 핸드오버 시점 샘플링 시작...\n")
    
    # N개의 후보를 샘플링 (N=5)
    N = 5
    best_reward = -float('inf')
    best_handover_time = 0
    
    for i in range(N):
        # 2. 초기 노이즈 생성 (가우시안 노이즈)
        y_noisy = torch.randn(batch_size, 1)
        
        # 3. Few-step 샘플링 (여기서는 단 2-step 만에 복원한다고 가정)
        steps = [torch.tensor([[2.0]]), torch.tensor([[1.0]])]
        
        y_current = y_noisy
        for s in steps:
            # 상태, 조건, 스텝을 넣고 노이즈를 제거하여 y_0 예측
            y_current = generator(y_current, c_k, s)
            
        # 최종 예측된 정규화된 핸드오버 시점 (0 ~ 1)
        y_0_final = y_current
        
        # 4. 평가기를 통해 보상 계산
        reward, delta_t = evaluator.evaluate(y_0_final, future_states_pred)
        
        print(f"후보 {i+1} | 예측된 전환 시점: 현재로부터 +{delta_t}초 뒤 | 계산된 보상(Reward): {reward:.4f}")
        
        # 최고 보상 업데이트
        if reward > best_reward:
            best_reward = reward
            best_handover_time = delta_t
            
    print("-" * 60)
    print(f"✅ 최종 의사결정: 가장 높은 보상({best_reward:.4f})을 기록한 [ +{best_handover_time}초 뒤 ]에 수직 핸드오버 실행!")