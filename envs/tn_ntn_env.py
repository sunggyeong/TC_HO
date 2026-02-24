# envs/tn_ntn_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class EnvBuildResult:
    traj: Any
    A_tn: np.ndarray
    A_leo: np.ndarray
    A_raw: np.ndarray
    M_phase: np.ndarray
    A_final: np.ndarray
    utility: np.ndarray
    latency_ms: np.ndarray
    meta_all: List[Dict[str, Any]]


class TN_NTN_Env:
    """
    trajectory_provider / tn_provider / satellite_provider / phase_mask_provider를 조합해
    A_final, utility, latency_ms, meta_all 을 생성하는 통합 Env.
    """

    def __init__(
        self,
        trajectory_provider,
        tn_provider,
        satellite_provider,
        phase_mask_provider=None,
    ):
        self.trajectory_provider = trajectory_provider
        self.tn_provider = tn_provider
        self.satellite_provider = satellite_provider
        self.phase_mask_provider = phase_mask_provider

    # =========================================================
    # Provider output normalization
    # =========================================================
    @staticmethod
    def _pick_first(d: Dict[str, Any], keys: List[str], default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    @staticmethod
    def _infer_domain_from_meta(meta: Dict[str, Any], default_domain: str) -> str:
        if not isinstance(meta, dict):
            return default_domain

        if "domain" in meta and meta["domain"] is not None:
            s = str(meta["domain"]).upper()
            if s in ("TN", "NTN"):
                return s

        merged = " ".join(
            str(meta.get(k, "")).lower()
            for k in ["orbit", "orbit_type", "kind", "node_type", "type", "name", "id", "label"]
        )

        if any(tok in merged for tok in ["leo", "meo", "geo", "sat", "ntn", "starlink"]):
            return "NTN"
        if any(tok in merged for tok in ["tn", "atg", "ground", "bs", "gnb", "airport", "coast"]):
            return "TN"
        return default_domain

    def _normalize_provider_output(
        self,
        out: Dict[str, Any],
        T: int,
        default_domain: str,
        prefix: str,
    ) -> Dict[str, Any]:
        """
        provider 출력이 조금 달라도 아래 표준형으로 맞춤:
        {
          "A": (T,N) uint8,
          "utility": (T,N) float,
          "latency_ms": (T,N) float,
          "meta": [dict] * N
        }
        """
        if not isinstance(out, dict):
            raise TypeError(f"{prefix} provider output은 dict여야 합니다. got={type(out)}")

        # --- A (coverage / availability) ---
        A = self._pick_first(out, ["A", "coverage", "availability", "A_cov", "mask"])
        if A is None:
            raise KeyError(f"{prefix} provider output에 coverage 행렬 키(A/coverage/availability 등)가 없습니다.")
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"{prefix}.A shape 오류: expected 2D, got {A.shape}")
        A = (A > 0).astype(np.uint8)

        if A.shape[0] != T:
            raise ValueError(f"{prefix}.A 시간축 길이 불일치: T={T}, A.shape={A.shape}")

        N = A.shape[1]

        # --- latency_ms ---
        latency = self._pick_first(out, ["latency_ms", "latency", "delay_ms", "lat_ms"])
        if latency is None:
            # 없으면 default latency 추정 (연결된 곳만 20ms, 비연결은 inf)
            latency = np.where(A == 1, 20.0, np.inf)
        latency = np.asarray(latency, dtype=float)
        if latency.shape != A.shape:
            # scalar 또는 (N,) 같은 경우 브로드캐스트 시도
            if latency.ndim == 0:
                latency = np.where(A == 1, float(latency), np.inf)
            elif latency.ndim == 1 and latency.shape[0] == N:
                latency = np.tile(latency[None, :], (T, 1))
                latency = np.where(A == 1, latency, np.inf)
            else:
                raise ValueError(f"{prefix}.latency shape 불일치: A={A.shape}, latency={latency.shape}")

        # --- meta ---
        meta = self._pick_first(out, ["meta", "meta_all", "nodes", "node_meta"], default=None)
        if meta is None:
            meta = []
            for i in range(N):
                meta.append({"id": f"{prefix}_{i}", "name": f"{prefix}_{i}"})
        else:
            meta = list(meta)
            if len(meta) != N:
                raise ValueError(f"{prefix}.meta 길이 불일치: N={N}, len(meta)={len(meta)}")

        meta_norm: List[Dict[str, Any]] = []
        for i, m in enumerate(meta):
            mm = dict(m) if isinstance(m, dict) else {"name": str(m)}
            mm.setdefault("id", mm.get("name", f"{prefix}_{i}"))
            mm.setdefault("name", mm.get("id", f"{prefix}_{i}"))
            mm["domain"] = self._infer_domain_from_meta(mm, default_domain=default_domain)
            meta_norm.append(mm)

        # --- utility ---
        utility = self._pick_first(out, ["utility", "score", "utility_mat", "reward_proxy"])
        if utility is None:
            # latency + capacity_score 기반 proxy 생성
            cap = np.array([float(m.get("capacity_score", 0.8)) for m in meta_norm], dtype=float)
            cap = np.tile(cap[None, :], (T, 1))

            # 연결된 셀만 대상으로 latency 정규화
            lat_finite = np.where(np.isfinite(latency), latency, np.nan)
            lat_min = np.nanmin(lat_finite) if np.any(np.isfinite(latency)) else 10.0
            lat_max = np.nanmax(lat_finite) if np.any(np.isfinite(latency)) else 100.0
            denom = max(1e-6, lat_max - lat_min)
            lat_norm = np.where(np.isfinite(latency), (latency - lat_min) / denom, 1.0)

            # 낮은 latency, 높은 capacity 선호
            utility = 0.6 * cap + 0.4 * (1.0 - lat_norm)
            utility = np.where(A == 1, utility, 0.0)
        else:
            utility = np.asarray(utility, dtype=float)
            if utility.shape != A.shape:
                if utility.ndim == 0:
                    utility = np.full_like(latency, float(utility), dtype=float)
                    utility = np.where(A == 1, utility, 0.0)
                else:
                    raise ValueError(f"{prefix}.utility shape 불일치: A={A.shape}, utility={utility.shape}")

        # 마스킹 일관성
        utility = np.where(A == 1, utility, 0.0)
        latency = np.where(A == 1, latency, np.inf)

        return {
            "A": A,
            "utility": utility,
            "latency_ms": latency,
            "meta": meta_norm,
        }

    # =========================================================
    # Build
    # =========================================================
    def build(self) -> EnvBuildResult:
        traj = self.trajectory_provider.build()
        T = len(traj.lat_deg)

        tn_raw = self.tn_provider.build_matrix(traj)
        leo_raw = self.satellite_provider.build_matrix(traj)

        tn = self._normalize_provider_output(tn_raw, T=T, default_domain="TN", prefix="TN")
        leo = self._normalize_provider_output(leo_raw, T=T, default_domain="NTN", prefix="LEO")

        A_tn = tn["A"]
        A_leo = leo["A"]

        A_raw = np.concatenate([A_tn, A_leo], axis=1).astype(np.uint8)
        utility = np.concatenate([tn["utility"], leo["utility"]], axis=1)
        latency_ms = np.concatenate([tn["latency_ms"], leo["latency_ms"]], axis=1)
        meta_all = tn["meta"] + leo["meta"]

        # phase mask
        if self.phase_mask_provider is not None:
            phase_out = self.phase_mask_provider.build_mask(traj, meta_all)
            M_phase = np.asarray(phase_out["M"], dtype=np.uint8)
            if M_phase.shape != A_raw.shape:
                raise ValueError(f"M_phase shape 불일치: expected {A_raw.shape}, got {M_phase.shape}")
        else:
            M_phase = np.ones_like(A_raw, dtype=np.uint8)

        A_final = ((A_raw > 0) & (M_phase > 0)).astype(np.uint8)

        # final mask 반영
        utility = np.where(A_final == 1, utility, 0.0)
        latency_ms = np.where(A_final == 1, latency_ms, np.inf)

        return EnvBuildResult(
            traj=traj,
            A_tn=A_tn,
            A_leo=A_leo,
            A_raw=A_raw,
            M_phase=M_phase,
            A_final=A_final,
            utility=utility,
            latency_ms=latency_ms,
            meta_all=meta_all,
        )