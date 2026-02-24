# baselines/sustainable_dag.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import networkx as nx


@dataclass
class SustainableDAGConfig:
    switch_penalty: float = 0.08        # HO 페널티
    stay_bonus: float = 0.02            # 유지 보너스
    outage_penalty: float = 1.0         # 해당 슬롯 후보 없음 대응용
    latency_penalty_scale: float = 0.002  # latency(ms) -> utility 차감
    topk_per_slot: Optional[int] = 20   # 그래프 크기 절감용
    allow_skip_edge: bool = False       # True면 t->t+2 edge 허용 가능(선택)


class SustainableDAGPlanner:
    """
    입력:
      A_final[t, n]     : coverage/phase 반영된 최종 후보 (0/1)
      utility[t, n]     : 후보 utility proxy (클수록 좋음)
      latency_ms[t, n]  : 후보 latency (작을수록 좋음)
    출력:
      planned_idx[t]    : 각 슬롯 선택 노드 index (-1 means outage)
    """
    def __init__(self, config: SustainableDAGConfig | None = None):
        self.cfg = config or SustainableDAGConfig()

    def _slot_candidates(self, A_final_t, utility_t):
        idx = np.where(A_final_t > 0)[0]
        if len(idx) == 0:
            return idx
        if self.cfg.topk_per_slot is None or len(idx) <= self.cfg.topk_per_slot:
            return idx
        # utility 높은 top-k만 유지
        top = np.argsort(utility_t[idx])[::-1][: self.cfg.topk_per_slot]
        return idx[top]

    def plan(
        self,
        A_final: np.ndarray,
        utility: np.ndarray,
        latency_ms: np.ndarray,
    ) -> Dict[str, Any]:
        T, N = A_final.shape
        G = nx.DiGraph()

        src = ("src", -1, -1)
        dst = ("dst", -1, -1)
        G.add_node(src)
        G.add_node(dst)

        slot_nodes: List[List[Tuple[int, int]]] = []
        has_any = False

        # 1) 노드 생성 (time-layered)
        for t in range(T):
            cand = self._slot_candidates(A_final[t], utility[t])
            cur_nodes = []
            for n in cand:
                v = (t, int(n))
                G.add_node(v)
                cur_nodes.append(v)
            slot_nodes.append(cur_nodes)
            if len(cur_nodes) > 0:
                has_any = True

        if not has_any:
            return {"planned_idx": np.full(T, -1, dtype=int), "graph_nodes": 0, "graph_edges": 0}

        # 2) source -> t=0
        if len(slot_nodes[0]) > 0:
            for v in slot_nodes[0]:
                t, n = v
                # 첫 슬롯 weight
                score = float(utility[t, n] - self.cfg.latency_penalty_scale * latency_ms[t, n])
                G.add_edge(src, v, weight=score)
        else:
            # 첫 슬롯 후보 없으면 이후 첫 후보 있는 슬롯로 연결
            first_nonempty = next((tt for tt in range(T) if len(slot_nodes[tt]) > 0), None)
            if first_nonempty is not None:
                for v in slot_nodes[first_nonempty]:
                    t, n = v
                    gap_pen = self.cfg.outage_penalty * first_nonempty
                    score = float(utility[t, n] - self.cfg.latency_penalty_scale * latency_ms[t, n] - gap_pen)
                    G.add_edge(src, v, weight=score)

        # 3) 시간 계층 간 edge 생성 (DAG)
        for t in range(T - 1):
            cur = slot_nodes[t]
            nxt = slot_nodes[t + 1]

            if len(cur) == 0 and len(nxt) == 0:
                continue
            if len(cur) == 0 and len(nxt) > 0:
                # outage 구간 후 재진입
                for v2 in nxt:
                    t2, n2 = v2
                    score = float(
                        utility[t2, n2]
                        - self.cfg.latency_penalty_scale * latency_ms[t2, n2]
                        - self.cfg.outage_penalty
                    )
                    # 직전 비어 있더라도 DAG 유지 위해 가상 연결은 src에서 이미 처리 가능
                    # 여기서는 건너뜀
                continue

            if len(cur) > 0 and len(nxt) == 0:
                # 다음 슬롯 outage -> 명시적 노드 없이 넘어감 (실행 단계에서 -1 처리)
                continue

            # cur -> nxt fully connect (top-k로 크기 제어)
            for v1 in cur:
                t1, n1 = v1
                for v2 in nxt:
                    t2, n2 = v2
                    switch = (n1 != n2)
                    score = float(utility[t2, n2] - self.cfg.latency_penalty_scale * latency_ms[t2, n2])
                    if switch:
                        score -= self.cfg.switch_penalty
                    else:
                        score += self.cfg.stay_bonus
                    G.add_edge(v1, v2, weight=score)

            # 선택: t -> t+2 skip edge (짧은 outage/불안정 회피용)
            if self.cfg.allow_skip_edge and t + 2 < T and len(slot_nodes[t + 2]) > 0:
                nxt2 = slot_nodes[t + 2]
                for v1 in cur:
                    t1, n1 = v1
                    for v2 in nxt2:
                        t2, n2 = v2
                        switch = (n1 != n2)
                        score = float(utility[t2, n2] - self.cfg.latency_penalty_scale * latency_ms[t2, n2])
                        score -= 1.5 * self.cfg.outage_penalty  # 1-slot 비는 거 감안
                        if switch:
                            score -= self.cfg.switch_penalty
                        G.add_edge(v1, v2, weight=score)

        # 4) 모든 마지막 후보 -> dst
        last_nonempty = next((tt for tt in range(T - 1, -1, -1) if len(slot_nodes[tt]) > 0), None)
        if last_nonempty is not None:
            for v in slot_nodes[last_nonempty]:
                G.add_edge(v, dst, weight=0.0)

        # 5) Longest path (DAG)
        path = nx.algorithms.dag_longest_path(G, weight="weight")

        # 6) planned_idx 복원 (없는 슬롯은 -1)
        planned_idx = np.full(T, -1, dtype=int)
        for node in path:
            if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], int):
                t, n = node
                planned_idx[t] = int(n)

        # 빈 슬롯 채우기 (forward-fill / backward-fill) -> 후보 있을 때만
        # Sustainable 계열은 경로 최적화 후 연속성 보정이 보통 필요
        for t in range(1, T):
            if planned_idx[t] == -1 and planned_idx[t - 1] != -1 and A_final[t, planned_idx[t - 1]] == 1:
                planned_idx[t] = planned_idx[t - 1]
        for t in range(T - 2, -1, -1):
            if planned_idx[t] == -1 and planned_idx[t + 1] != -1 and A_final[t, planned_idx[t + 1]] == 1:
                planned_idx[t] = planned_idx[t + 1]

        return {
            "planned_idx": planned_idx,
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
        }