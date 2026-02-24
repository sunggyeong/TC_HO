# providers/phase_mask_provider.py
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np


class PhaseMaskProvider:
    """
    phase_label 기반으로 TN/NTN 허용 여부를 제어.
    meta_all의 스키마가 provider마다 조금 달라도 domain을 추론해서 동작하도록 만든 버전.
    """

    def __init__(self, allow_rules=None):
        # phase -> dict(TN=True/False, NTN=True/False)
        # 필요하면 여기에 phase alias 더 추가 가능
        self.allow_rules = allow_rules or {
            "gate_or_taxi": {"TN": True,  "NTN": False},
            "gate":         {"TN": True,  "NTN": False},
            "taxi":         {"TN": True,  "NTN": False},

            "takeoff_climb": {"TN": True, "NTN": True},
            "takeoff":       {"TN": True, "NTN": True},
            "climb":         {"TN": True, "NTN": True},

            "cruise":        {"TN": True, "NTN": True},

            "descent":       {"TN": True, "NTN": True},
            "approach":      {"TN": True, "NTN": True},

            "landing":       {"TN": True, "NTN": False},
        }

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _canon_phase(self, phase: str) -> str:
        p = str(phase).strip().lower()

        # 흔한 alias 흡수
        if p in ("gate", "taxi", "gate_taxi", "taxiing"):
            return "gate_or_taxi"
        if p in ("takeoff", "takeoff_climb", "takeoff+climb"):
            return "takeoff_climb"
        if p in ("climb",):
            return "climb"
        if p in ("cruise", "enroute"):
            return "cruise"
        if p in ("descent",):
            return "descent"
        if p in ("approach",):
            return "approach"
        if p in ("landing", "land"):
            return "landing"

        return p

    def _infer_domain(self, meta: Dict[str, Any]) -> str:
        """
        provider meta 스키마가 달라도 TN/NTN 추론.
        우선순위:
        - domain
        - orbit / sat_* / kind / node_type / type / name/id 문자열 패턴
        """
        if not isinstance(meta, dict):
            return "UNKNOWN"

        # 1) 명시적 domain
        d = meta.get("domain", None)
        if d is not None:
            ds = str(d).upper()
            if ds in ("TN", "NTN"):
                return ds

        # 2) 후보 키에서 문자열 모아서 판별
        candidates = []
        for k in [
            "orbit", "orbit_type", "sat_type",
            "kind", "node_kind",
            "type", "node_type",
            "name", "id", "label"
        ]:
            if k in meta and meta[k] is not None:
                candidates.append(str(meta[k]).lower())

        merged = " ".join(candidates)

        # NTN 계열 패턴
        if any(tok in merged for tok in ["leo", "meo", "geo", "sat", "ntn", "starlink"]):
            return "NTN"

        # TN/ATG 계열 패턴
        if any(tok in merged for tok in ["tn", "atg", "ground", "bs", "gnb", "airport", "coast"]):
            return "TN"

        return "UNKNOWN"

    # -----------------------------
    # Public API
    # -----------------------------
    def build_mask(self, traj, meta_all: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not hasattr(traj, "phase_label"):
            raise ValueError("traj.phase_label 이 필요합니다.")

        T = len(traj.phase_label)
        N = len(meta_all)
        M = np.ones((T, N), dtype=np.uint8)

        domains = np.array([self._infer_domain(m) for m in meta_all], dtype=object)

        # 디버깅용: UNKNOWN 있으면 일단 허용(막아버리면 실험이 깨짐)
        # print("[PhaseMask] domain counts:", {k: int(np.sum(domains==k)) for k in np.unique(domains)})

        for t in range(T):
            phase = self._canon_phase(str(traj.phase_label[t]))
            rule = self.allow_rules.get(phase, {"TN": True, "NTN": True})

            if not rule.get("TN", True):
                M[t, domains == "TN"] = 0
            if not rule.get("NTN", True):
                M[t, domains == "NTN"] = 0

            # UNKNOWN은 기본 허용 (provider mismatch 방어)
            # 필요하면 아래처럼 막을 수도 있음:
            # if not rule.get("UNKNOWN", True):
            #     M[t, domains == "UNKNOWN"] = 0

        return {
            "M": M,
            "domains": domains,  # 디버깅에 도움됨
        }