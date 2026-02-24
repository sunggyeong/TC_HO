from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from providers import TrajectoryProvider, TNATGProvider, SimpleLEOSatelliteProvider, PhaseMaskProvider


@dataclass
class TN_NTN_Env:
    """
    Provider-composed simulator environment (ATG+LEO prototype).
    """
    trajectory_provider: TrajectoryProvider
    tnatg_provider: TNATGProvider
    satellite_provider: SimpleLEOSatelliteProvider   # LEO only for Exp1/2
    phase_mask_provider: PhaseMaskProvider
    dt_sec: float = 1.0

    def precompute_coverage_matrices(self) -> Dict[str, Any]:
        traj = self.trajectory_provider.generate(self.dt_sec)

        tn_meta = self.tnatg_provider.metadata()
        leo_meta = self.satellite_provider.metadata()

        tn_m = self.tnatg_provider.compute_matrices(traj)
        leo_m = self.satellite_provider.compute_matrices(traj)

        # concatenate nodes
        node_ids = np.concatenate([tn_meta["node_ids"], leo_meta["node_ids"]], axis=0)
        node_types = np.concatenate([tn_meta["node_types"], leo_meta["node_types"]], axis=0)
        node_orbits = np.concatenate([tn_meta["node_orbits"], leo_meta["node_orbits"]], axis=0)
        node_regions = np.concatenate([tn_meta["node_regions"], leo_meta["node_regions"]], axis=0)

        A_geo = np.concatenate([tn_m["A"], leo_m["A"]], axis=1)
        PropDelay_ms = np.concatenate([tn_m["PropDelay_ms"], leo_m["PropDelay_ms"]], axis=1)
        Q_proxy = np.concatenate([tn_m["Q_proxy"], leo_m["Q_proxy"]], axis=1)

        phase_labels = self.phase_mask_provider.phase_labels(traj)
        M_phase = self.phase_mask_provider.build_mask(phase_labels, node_types, node_orbits)

        A_final = (A_geo.astype(bool) & M_phase.astype(bool)).astype(int)

        # sanitize score/delay where invalid
        PropDelay_ms = np.where(A_geo == 1, PropDelay_ms, np.nan)
        Q_proxy = np.where(A_geo == 1, Q_proxy, np.nan)

        tensors = {
            "sim_time_sec": traj["time_sec"].copy(),
            "lat_deg": traj["lat_deg"].copy(),
            "lon_deg": traj["lon_deg"].copy(),
            "alt_m": traj["alt_m"].copy(),
            "speed_mps": traj["speed_mps"].copy(),
            "progress": traj["progress"].copy(),
            "phase_labels": phase_labels.copy(),
            "node_ids": node_ids,
            "node_types": node_types,
            "node_orbits": node_orbits,
            "node_regions": node_regions,
            "A_geo": A_geo,
            "M_phase": M_phase,
            "A_final": A_final,
            "PropDelay_ms": PropDelay_ms,
            "Q_proxy": Q_proxy,
            "dt_sec": float(self.dt_sec),
        }
        return tensors


def assert_atg_leo_only(tensors: Dict[str, Any]) -> None:
    node_types = np.asarray(tensors["node_types"], dtype=object)
    node_orbits = np.asarray(tensors["node_orbits"], dtype=object)
    bad = []
    for i, (t, o) in enumerate(zip(node_types, node_orbits)):
        if str(t) == "NTN" and str(o) != "LEO":
            bad.append((i, str(o)))
    if bad:
        raise ValueError(f"ATG+LEO experiment인데 LEO 외 NTN 노드 포함: {bad[:10]}")