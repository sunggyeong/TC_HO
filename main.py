import torch
# 우리가 만든 3개의 모듈 불러오기
from tn_ntn_env_integrated import TN_NTN_Env
from transformer_predictor import TrajectoryPredictor
from consistency_handover import ConsistencyGenerator, HandoverRewardEvaluator

def run_simulation():
    print("🚀 6G TN-NTN 디지털 트윈 시뮬레이션을 시작합니다...\n")
    
    # 1. 시뮬레이션 모듈들 초기화 (L=20초 과거 관측, H=30초 미래 예측)
    env = TN_NTN_Env(seq_length=20)
    transformer = TrajectoryPredictor(feature_dim=6, L=20, H=30)
    generator = ConsistencyGenerator(condition_dim=180) # 30스텝 * 6차원 = 180
    evaluator = HandoverRewardEvaluator(H=30)
    
    # 2. 비행 시뮬레이션 루프 (예: 25초 동안 비행)
    for t in range(1, 26):
        # [단계 1] 환경에서 1초 이동 및 관측 데이터(텐서) 획득
        raw_state, state_tensor = env.step()
        print(f"[Time {t}초] 현재 위치: {raw_state['aircraft_pos']}")
        
        # 버퍼에 20초 분량의 과거 데이터가 다 찼을 때만 AI 추론 시작
        if state_tensor is not None:
            print(f"  ✅ [AI 추론 시작] 20초 과거 데이터 수집 완료!")
            
            # [단계 2] Transformer: 미래 30초 궤적/네트워크 상태 예측
            future_states = transformer(state_tensor) # Shape: (1, 30, 6)
            c_k = future_states.view(1, -1)           # Shape: (1, 180) - 조건 벡터로 평탄화
            
            # [단계 3] Consistency Model: Few-step(2스텝) 핸드오버 시점 생성
            y_curr = torch.randn(1, 1) # 무작위 노이즈에서 시작
            steps = [torch.tensor([[2.0]]), torch.tensor([[1.0]])] # 2-step 디노이징
            
            for s in steps:
                y_curr = generator(y_curr, c_k, s)
                
            # [단계 4] 생성된 시점의 보상(Reward) 평가 및 최종 결정
            reward, delta_t = evaluator.evaluate(y_curr, future_states)
            
            print(f"  🎯 [결과] 최적의 핸드오버 시점: 현재로부터 +{delta_t}초 뒤")
            print(f"  💰 [보상] 예상 Reward 점수: {reward:.4f}")
            print("=" * 60)
        else:
            print(f"  ⏳ 데이터 버퍼링 중... ({t}/20)")
            print("-" * 60)

if __name__ == "__main__":
    run_simulation()