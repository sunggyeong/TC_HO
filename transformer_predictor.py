import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    시간의 흐름(순서)을 Transformer 모델에 알려주기 위한 위치 인코딩 모듈
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # (1, max_len, d_model)

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class TrajectoryPredictor(nn.Module):
    """
    과거 L 스텝의 궤적 및 네트워크 상태를 받아 미래 H 스텝을 예측하는 Transformer 모델
    """
    def __init__(self, feature_dim=6, d_model=64, nhead=4, num_layers=3, L=20, H=30):
        super(TrajectoryPredictor, self).__init__()
        self.L = L
        self.H = H
        self.feature_dim = feature_dim
        self.d_model = d_model

        # 1. 입력 임베딩: 6차원 특징 벡터를 d_model(예: 64) 차원으로 확장
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=L)

        # 2. Transformer Encoder: 과거 데이터 간의 Attention(상관관계) 학습
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 출력 헤드 (Output Head): Encoder의 출력을 미래 H 스텝의 특성으로 매핑
        # L * d_model 차원의 정보를 압축한 뒤, H * feature_dim 크기로 펼침
        self.fc_out = nn.Sequential(
            nn.Linear(L * d_model, 128),
            nn.ReLU(),
            nn.Linear(128, H * feature_dim)
        )

    def forward(self, x):
        """
        x shape: (Batch, L, feature_dim) -> 예: (1, 20, 6)
        """
        batch_size = x.size(0)

        # 임베딩 및 위치 정보 추가
        x = self.embedding(x)         # -> (1, 20, 64)
        x = self.pos_encoder(x)       # -> (1, 20, 64)

        # Transformer 인코더 통과
        out = self.transformer_encoder(x) # -> (1, 20, 64)

        # 다층 퍼셉트론(MLP)을 통과시키기 위해 텐서를 1차원으로 길게 펼침
        out = out.reshape(batch_size, -1) # -> (1, 1280)

        # 미래 H 스텝 예측
        out = self.fc_out(out)            # -> (1, 180) (30스텝 * 6차원)
        
        # 다시 원래의 (Batch, H, feature_dim) 형태로 복구
        out = out.view(batch_size, self.H, self.feature_dim) # -> (1, 30, 6)

        return out

# ==========================================
# 실행 테스트 (데이터 버퍼 연동 가정)
# ==========================================
if __name__ == "__main__":
    # 파라미터 셋업
    L = 20  # 과거 20초 관측
    H = 30  # 미래 30초 예측
    F = 6   # 6차원 특징 벡터
    
    # 모델 인스턴스 생성
    model = TrajectoryPredictor(feature_dim=F, d_model=64, nhead=4, num_layers=3, L=L, H=H)
    
    # 가상의 입력 데이터 텐서 (이전 단계의 Data Buffer에서 나온 값이라고 가정)
    # Shape: (Batch=1, Seq_Len=20, Features=6)
    dummy_input = torch.rand(1, L, F)
    
    # 모델 추론 (Forward Pass)
    predicted_future = model(dummy_input)
    
    print(f"🔹 [입력 데이터 크기]: {dummy_input.shape} -> (Batch, 과거 {L}초, {F}개 특징)")
    print(f"🔸 [예측 데이터 크기]: {predicted_future.shape} -> (Batch, 미래 {H}초, {F}개 특징)")
    print("\n✅ Transformer 예측 모델이 성공적으로 30스텝 앞의 궤적 및 커버리지 상태를 도출했습니다!")