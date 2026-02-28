# models/proposed_tc_planner.py
import numpy as np
import torch
import torch.nn as nn
import os

# =====================================================================
# 1. Transformer Predictor (미래 네트워크 토폴로지 예측)
# =====================================================================
class OnlineTransformerPredictor(nn.Module):
    def __init__(self, num_nodes, d_model=64, nhead=4, num_layers=2, L=20, H=30):
        super().__init__()
        self.L = L
        self.H = H
        self.num_nodes = num_nodes
        
        self.embedding = nn.Linear(num_nodes, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, L, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(L * d_model, 128),
            nn.ReLU(),
            nn.Linear(128, H * num_nodes),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.embedding(x) + self.pos_encoder
        out = self.transformer(out)
        out = out.reshape(batch_size, -1)
        out = self.fc_out(out)
        return out.view(batch_size, self.H, self.num_nodes)


# =====================================================================
# 2. Consistency Generator (Few-step 핸드오버 타겟 생성)
# =====================================================================
class OnlineConsistencyGenerator(nn.Module):
    def __init__(self, num_nodes, H=30, embed_dim=32):
        super().__init__()
        self.H = H
        self.num_nodes = num_nodes
        condition_dim = H * num_nodes
        
        self.step_mlp = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(2 + condition_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid() # [시간_비율(0~1), 노드_비율(0~1)]
        )

    def forward(self, y_noisy, condition, step):
        s_embed = self.step_mlp(step)
        combined = torch.cat([y_noisy, condition, s_embed], dim=-1)
        return self.net(combined)


# =====================================================================
# 3. Proposed TC Planner (실험 연동용 메인 인터페이스)
# =====================================================================
class ProposedTCPlanner:
    def __init__(self, L=20, H=30, device='cpu', weight_path='trained_weights_rl.pth', switch_time_offset_max=1, min_dwell=3, hysteresis_ratio=0.08):
        self.L = L
        self.H = H
        self.device = device
        self.weight_path = weight_path
        self.switch_time_offset_max = switch_time_offset_max
        self.min_dwell = min_dwell
        self.hysteresis_ratio = hysteresis_ratio
        self.num_nodes = None
        self.transformer = None
        self.consistency = None

    def _init_models(self, num_nodes):
        self.num_nodes = num_nodes
        self.transformer = OnlineTransformerPredictor(num_nodes=num_nodes, L=self.L, H=self.H).to(self.device)
        self.consistency = OnlineConsistencyGenerator(num_nodes=num_nodes, H=self.H).to(self.device)
        
        # 학습된 가중치가 있으면 불러오기
        if os.path.exists(self.weight_path):
            checkpoint = torch.load(self.weight_path, map_location=self.device)
            self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
            self.consistency.load_state_dict(checkpoint['consistency_state_dict'])
            print(f"[ProposedTCPlanner] 가중치 로드 완료: {self.weight_path}")
        else:
            print(f"[ProposedTCPlanner] 경고: 가중치 파일({self.weight_path})이 없어 랜덤 초기화 상태로 추론합니다.")
            
        self.transformer.eval()
        self.consistency.eval()

    def plan(self, env_result, consistency_steps: int = 2) -> np.ndarray:
        T, N = env_result.A_final.shape
        if self.transformer is None:
            self._init_models(N)
            
        planned_idx = np.full(T, -1, dtype=int)
        current_node = -1
        last_switch_t = -10**9
        
        for t in range(T):
            if t < self.L:
                # 버퍼가 차기 전에는 휴리스틱하게 utility가 가장 높은 노드 선택
                available = np.where(env_result.A_final[t] == 1)[0]
                if len(available) > 0:
                    current_node = available[np.argmax(env_result.utility[t, available])]
                else:
                    current_node = -1
                planned_idx[t] = current_node
                continue
                
            history_A = env_result.A_final[t - self.L : t, :]
            state_tensor = torch.tensor(history_A, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Make-Before-Break 추론
            with torch.no_grad():
                future_pred = self.transformer(state_tensor) 
                condition = future_pred.view(1, -1)
                
                y_curr = torch.randn(1, 2).to(self.device)
                steps = max(1, int(consistency_steps))
                for s_val in range(steps, 0, -1):
                    s = torch.tensor([[float(s_val)]], dtype=torch.float32, device=self.device)
                    y_curr = self.consistency(y_curr, condition, s)
                
                target_time_offset = int(y_curr[0, 0].item() * (self.H - 1))
                target_node_idx = int(y_curr[0, 1].item() * (N - 1))
                
                target_time_offset = min(max(target_time_offset, 0), self.H - 1)
                target_node_idx = min(max(target_node_idx, 0), N - 1)
                
                # proposal/prior + lightweight guardrail
                can_switch = target_time_offset <= self.switch_time_offset_max and env_result.A_final[t, target_node_idx] == 1
                if can_switch and (t - last_switch_t) >= self.min_dwell:
                    cur_score = float(env_result.utility[t, current_node]) if current_node != -1 and env_result.A_final[t, current_node] == 1 else -1e9
                    new_score = float(env_result.utility[t, target_node_idx])
                    if current_node == -1 or new_score >= cur_score * (1.0 + self.hysteresis_ratio):
                        current_node = target_node_idx
                        last_switch_t = t
            
            # 현재 접속 중인 노드가 커버리지를 벗어나면 fallback 선택
            if current_node != -1 and env_result.A_final[t, current_node] == 0:
                available = np.where(env_result.A_final[t] == 1)[0]
                if len(available) > 0:
                    current_node = int(available[np.argmax(env_result.utility[t, available])])
                else:
                    current_node = -1
                
            planned_idx[t] = current_node
            
        return planned_idx