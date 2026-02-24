import torch
import torch.optim as optim
import numpy as np

# 앞서 만든 3개의 모듈 불러오기
from tn_ntn_env_integrated import TN_NTN_Env
from transformer_predictor import TrajectoryPredictor
from consistency_handover import ConsistencyGenerator, HandoverRewardEvaluator

def train_agent():
    print("🚀 6G RAN AI 디지털 트윈 - 모델 학습(Training)을 시작합니다...\n")
    
    # 1. 하이퍼파라미터 설정
    EPOCHS = 10           # 총 비행(학습) 횟수
    FLIGHT_TIME = 35      # 1회 비행당 시뮬레이션 시간 (초)
    LEARNING_RATE = 1e-3  # 학습률 (가중치 업데이트 보폭)
    
    # 2. 모델 인스턴스화
    transformer = TrajectoryPredictor(feature_dim=6, L=20, H=30)
    generator = ConsistencyGenerator(condition_dim=180)
    evaluator = HandoverRewardEvaluator(H=30)
    
    # 3. 옵티마이저(Optimizer) 설정
    # 소프트웨어 최적화에서 가장 널리 쓰이는 Adam 알고리즘 사용
    # 두 모델의 파라미터(가중치)를 모두 업데이트 대상에 포함
    optimizer = optim.Adam(
        list(transformer.parameters()) + list(generator.parameters()), 
        lr=LEARNING_RATE
    )
    
    # 4. 에피소드(비행) 반복 학습 루프
    for epoch in range(1, EPOCHS + 1):
        # 매 비행마다 새로운 환경(디지털 트윈) 초기화
        env = TN_NTN_Env(seq_length=20)
        epoch_rewards = [] # 이번 에피소드에서 획득한 보상 기록
        
        print(f"=== [Epoch {epoch}/{EPOCHS}] 비행 시뮬레이션 시작 ===")
        
        for t in range(1, FLIGHT_TIME + 1):
            raw_state, state_tensor = env.step()
            
            # 버퍼가 차서(20초 이후) AI가 개입할 수 있는 시점
            if state_tensor is not None:
                optimizer.zero_grad() # 이전 스텝의 기울기(Gradient) 초기화
                
                # [순전파] 1. 미래 30초 예측
                future_states = transformer(state_tensor)
                c_k = future_states.view(1, -1)
                
                # [순전파] 2. Consistency Model로 핸드오버 시점 생성 (노이즈 -> 정답)
                y_curr = torch.randn(1, 1)
                steps = [torch.tensor([[2.0]]), torch.tensor([[1.0]])]
                for s in steps:
                    y_curr = generator(y_curr, c_k, s)
                    
                # [평가] 3. 생성된 시점(y_curr)에 대한 보상 계산
                reward, delta_t = evaluator.evaluate(y_curr, future_states)
                epoch_rewards.append(reward)
                
                # [역전파 및 학습] 4. Loss 계산 및 가중치 업데이트
                # 강화학습의 핵심: Reward를 극대화해야 하므로 Loss는 -Reward 로 설정
                # (Gradient Ascent 효과를 내기 위한 PyTorch 대리 손실 함수)
                loss = -torch.tensor(reward, requires_grad=True) * y_curr.mean() 
                
                loss.backward()   # 오차 역전파 (Gradient 계산)
                optimizer.step()  # 모델 가중치 업데이트 (더 똑똑해짐!)
                
        # 한 번의 비행(Epoch)이 끝난 후 평균 보상 출력
        avg_reward = np.mean(epoch_rewards)
        print(f"✈️ 비행 종료 | 평균 Reward 점수: {avg_reward:.4f}")
        print("-" * 50)

    print("✅ 모든 학습이 완료되었습니다! AI가 최적의 핸드오버 전략을 터득했습니다.")
    
    # 학습된 모델 가중치 저장 (선택 사항)
    # torch.save(generator.state_dict(), 'trained_consistency_model.pth')
    # print("💾 학습된 모델이 저장되었습니다.")

if __name__ == "__main__":
    train_agent()