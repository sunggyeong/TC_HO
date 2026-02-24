# providers/tn_atg_provider.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class TNATGNode:
    node_id: str
    lat_deg: float
    lon_deg: float
    kind: str = "ATG"   # "ATG" or "TN"
    radius_km: float = 40.0
    base_latency_ms: float = 8.0
    capacity_score: float = 1.0  # 상대적 QoS proxy


class TNATGProvider:
    def __init__(self, nodes: List[TNATGNode], default_aircraft_alt_m: float = 10000.0):
        self.nodes = nodes
        self.default_aircraft_alt_m = float(default_aircraft_alt_m)

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
        return 2.0 * R * np.arcsin(np.sqrt(a))

    def build_matrix(self, traj) -> Dict[str, Any]:
        T = len(traj.lat_deg)
        N = len(self.nodes)

        A = np.zeros((T, N), dtype=np.uint8)           # coverage matrix
        slant_km = np.full((T, N), np.nan, dtype=float)
        latency_ms = np.full((T, N), np.nan, dtype=float)
        utility = np.zeros((T, N), dtype=float)

        for j, node in enumerate(self.nodes):
            horiz_km = self._haversine_km(traj.lat_deg, traj.lon_deg, node.lat_deg, node.lon_deg)
            alt_km = traj.alt_m / 1000.0
            s_km = np.sqrt(horiz_km ** 2 + alt_km ** 2)
            visible = horiz_km <= node.radius_km

            A[:, j] = visible.astype(np.uint8)
            slant_km[:, j] = s_km

            # 단순 latency proxy (연구용; 나중에 더 정교화 가능)
            # propagation + processing
            prop_ms = (s_km / 300000.0) * 1000.0
            lat = node.base_latency_ms + prop_ms
            latency_ms[:, j] = lat

            # utility proxy (0~1 근처): 거리 멀수록 감소
            util = node.capacity_score * np.exp(-horiz_km / max(node.radius_km, 1e-6))
            utility[:, j] = np.where(visible, util, 0.0)

        meta = [
            {
                "node_id": n.node_id,
                "domain": "TN",
                "kind": n.kind,
                "lat_deg": n.lat_deg,
                "lon_deg": n.lon_deg,
                "radius_km": n.radius_km,
            }
            for n in self.nodes
        ]

        return {
            "A": A,
            "slant_km": slant_km,
            "latency_ms": latency_ms,
            "utility": utility,
            "meta": meta,
        }