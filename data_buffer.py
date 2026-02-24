import numpy as np
import torch
from collections import deque

class TrajectoryBuffer:
    def __init__(self, seq_length=20):
        """
        seq_length (L): Transformer 모델이 바라볼 과거 시간 스텝 수 (기본값: 20초)
        """
        self.seq_length = seq_length
        # maxlen을 설정한 deque는 큐가 꽉 차면 가장 오래된 데이터를 자동으로 밀어내고 새 데이터를 넣습니다 (Sliding Window).
        self.buffer = deque(maxlen=seq_length)
        
    def extract_features(self, raw_state):
        """환경(TN_NTN_Env)에서 나온 딕셔너리를 AI 모델이 학습할 수 있는 1차원 숫자 배열로 변환"""
        lat, lon = raw_state['aircraft_pos']
        num_tns = len(raw_state['available_TNs'])
        num_ntns = len(raw_state['available_NTNs'])
        
        # TN 중 가장 가까운 거리 추출 (기지국이 없으면 100km 페널티값 부여)
        if num_tns > 0:
            min_tn_dist = min([tn['distance_km'] for tn in raw_state['available_TNs']])
        else:
            min_tn_dist = 100.0
            
        # NTN 중 가장 높은 앙각 추출 (위성이 없으면 0도 페널티값 부여)
        if num_ntns > 0:
            max_ntn_elev = max([ntn['elevation_deg'] for ntn in raw_state['available_NTNs']])
        else:
            max_ntn_elev = 0.0
            
        # 6차원 특징 벡터(Feature Vector) 구성: [위도, 경도, 가용지상망수, 가용위성수, 최소TN거리, 최대NTN앙각]
        feature_vector = [lat, lon, num_tns, num_ntns, min_tn_dist, max_ntn_elev]
        return np.array(feature_vector, dtype=np.float32)
        
    def add_state(self, raw_state):
        """매 초마다 발생한 상태를 버퍼에 추가"""
        features = self.extract_features(raw_state)
        self.buffer.append(features)
        
    def get_tensor(self):
        """
        버퍼에 L초 분량의 데이터가 다 차면 PyTorch Tensor로 변환하여 반환
        반환 형태(Shape): (Batch, Sequence_Length, Feature_Dim) -> (1, 20, 6)
        """
        if len(self.buffer) < self.seq_length:
            return None # 아직 20초가 채워지지 않음
        
        # (20, 6) 크기의 2차원 Numpy 배열 생성
        seq_array = np.array(self.buffer)
        
        # PyTorch Tensor로 변환 후, Transformer 입력 규격에 맞게 Batch 차원(맨 앞 1차원) 추가
        seq_tensor = torch.tensor(seq_array).unsqueeze(0)
        return seq_tensor

# ==========================================
# 통합 실행 테스트 (환경 + 데이터 버퍼)
# ==========================================
if __name__ == "__main__":
    # 이전 단계의 환경 객체 (만약 파일이 분리되어 있다면 from tn_ntn_env import TN_NTN_Env 처리)
    # env = TN_NTN_Env() 
    
    L = 20 # 20초 (20스텝) 관측 윈도우
    buffer = TrajectoryBuffer(seq_length=L)
    
    print(f"데이터 수집을 시작합니다. (과거 {L}초 분량의 데이터가 찰 때까지 대기...)\n")
    
    # 25초 동안 비행 시뮬레이션
    for t in range(25):
        # 1. 환경에서 1초 진행 및 관측
        # raw_state = env.step() 
        
        # (테스트용 가짜 데이터) 실제 환경을 붙일 때는 위 두 줄의 주석을 해제하세요.
        raw_state = {
            'aircraft_pos': (37.5 + t*0.001, 127.0 + t*0.001),
            'available_TNs': [{'distance_km': 15.0 + t}],
            'available_NTNs': [{'elevation_deg': 45.0 - t}]
        }
        
        # 2. 버퍼에 상태 추가
        buffer.add_state(raw_state)
        
        # 3. 텐서 추출 시도
        state_tensor = buffer.get_tensor()
        
        if state_tensor is not None:
            print(f"[Time {t}초] 텐서 생성 완료! Shape: {state_tensor.shape}")
            # Shape은 torch.Size([1, 20, 6])으로 출력됩니다.
            # 이 state_tensor가 바로 Transformer 모델의 입력(Input)으로 들어갑니다.
        else:
            print(f"[Time {t}초] 데이터 수집 중... (현재 {len(buffer.buffer)}/{L})")