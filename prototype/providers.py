from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math
import numpy as np

from prototype.geo_utils import (
    haversine_distance_m,
    slant_distance_m,
    elevation_angle_deg,
    propagation_delay_ms_from_distance_m,
    lerp,
    wrap_lon_deg,
)


# ============================================================
# Trajectory Providers
# ============================================================

class TrajectoryProvider:
    def generate(self, dt_sec: float) -> Dict[str, Any]:
        """
        Returns dict with:
          - time_sec: [T]
          - lat_deg: [T]
          - lon_deg: [T]
          - alt_m: [T]
          - speed_mps: [T] or scalar-ish [T]
          - progress: [T] in [0,1]
        """
        raise NotImplementedError


@dataclass
class Waypoint:
    lat_deg: float
    lon_deg: float
    alt_m: float


@dataclass
class WaypointTrajectoryProvider(TrajectoryProvider):
    waypoints: List[Waypoint]
    speed_mps: float = 250.0  # ~900 km/h
    max_duration_sec: Optional[int] = 1800  # prototype default (30 min); set None for full path

    def generate(self, dt_sec: float) -> Dict[str, Any]:
        assert len(self.waypoints) >= 2, "waypoints need >= 2"

        # segment lengths
        seg_surface = []
        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i]
            b = self.waypoints[i + 1]
            d = haversine_distance_m(a.lat_deg, a.lon_deg, b.lat_deg, b.lon_deg)
            seg_surface.append(d)

        seg_times = [max(1e-6, d / max(1e-6, self.speed_mps)) for d in seg_surface]
        total_time = sum(seg_times)
        if self.max_duration_sec is not None:
            total_time = min(total_time, float(self.max_duration_sec))

        T = int(math.floor(total_time / dt_sec)) + 1
        t_arr = np.arange(T, dtype=float) * dt_sec

        lat = np.zeros(T, dtype=float)
        lon = np.zeros(T, dtype=float)
        alt = np.zeros(T, dtype=float)
        spd = np.full(T, float(self.speed_mps), dtype=float)
        prog = np.zeros(T, dtype=float)

        # walk time along segments
        cum_times = [0.0]
        for st in seg_times:
            cum_times.append(cum_times[-1] + st)

        for idx, t in enumerate(t_arr):
            t_clip = min(t, cum_times[-1] - 1e-9)
            # find segment
            seg_idx = 0
            while seg_idx < len(seg_times) - 1 and not (cum_times[seg_idx] <= t_clip < cum_times[seg_idx + 1]):
                seg_idx += 1

            a = self.waypoints[seg_idx]
            b = self.waypoints[seg_idx + 1]
            local_t = (t_clip - cum_times[seg_idx]) / max(1e-9, seg_times[seg_idx])

            # Prototype interpolation (lat/lon linear; enough for env driver)
            lat[idx] = lerp(a.lat_deg, b.lat_deg, local_t)
            # lon wrap-aware interpolation
            lon_a = a.lon_deg
            lon_b = b.lon_deg
            dlon = lon_b - lon_a
            if dlon > 180.0:
                lon_b -= 360.0
            elif dlon < -180.0:
                lon_b += 360.0
            lon[idx] = wrap_lon_deg(lerp(lon_a, lon_b, local_t))
            alt[idx] = lerp(a.alt_m, b.alt_m, local_t)

            prog[idx] = min(1.0, t / max(1e-6, total_time))

        return {
            "time_sec": t_arr,
            "lat_deg": lat,
            "lon_deg": lon,
            "alt_m": alt,
            "speed_mps": spd,
            "progress": prog,
        }


@dataclass
class OracleTrajectoryProvider(TrajectoryProvider):
    base: TrajectoryProvider

    def generate(self, dt_sec: float) -> Dict[str, Any]:
        return self.base.generate(dt_sec)


@dataclass
class PerturbedTrajectoryProvider(TrajectoryProvider):
    base: TrajectoryProvider
    lat_noise_std_deg: float = 0.03
    lon_noise_std_deg: float = 0.03
    alt_noise_std_m: float = 100.0
    seed: int = 0

    def generate(self, dt_sec: float) -> Dict[str, Any]:
        d = self.base.generate(dt_sec)
        rng = np.random.default_rng(self.seed)
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        out["lat_deg"] = out["lat_deg"] + rng.normal(0.0, self.lat_noise_std_deg, size=len(out["lat_deg"]))
        out["lon_deg"] = np.array([wrap_lon_deg(x) for x in (out["lon_deg"] + rng.normal(0.0, self.lon_noise_std_deg, size=len(out["lon_deg"])))])
        out["alt_m"] = np.maximum(0.0, out["alt_m"] + rng.normal(0.0, self.alt_noise_std_m, size=len(out["alt_m"])))
        return out


# ============================================================
# TN/ATG Provider
# ============================================================

@dataclass
class TNATGNode:
    node_id: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    coverage_radius_km: float = 120.0  # ATG-ish broader than terrestrial microcell
    region: str = "ATG"


@dataclass
class TNATGProvider:
    nodes: List[TNATGNode]
    score_bias: float = 15.0  # helps TN near airport/coast when available

    def metadata(self) -> Dict[str, np.ndarray]:
        n = len(self.nodes)
        return {
            "node_ids": np.array([x.node_id for x in self.nodes], dtype=object),
            "node_types": np.array(["TN"] * n, dtype=object),
            "node_orbits": np.array(["TN"] * n, dtype=object),
            "node_regions": np.array([x.region for x in self.nodes], dtype=object),
        }

    def compute_matrices(self, traj: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        T = len(traj["time_sec"])
        N = len(self.nodes)
        A = np.zeros((T, N), dtype=int)
        prop_ms = np.full((T, N), np.nan, dtype=float)
        score = np.full((T, N), np.nan, dtype=float)

        for t in range(T):
            ue = (float(traj["lat_deg"][t]), float(traj["lon_deg"][t]), float(traj["alt_m"][t]))
            for j, nd in enumerate(self.nodes):
                nd_lla = (nd.lat_deg, nd.lon_deg, nd.alt_m)
                surface_d_m = haversine_distance_m(ue[0], ue[1], nd.lat_deg, nd.lon_deg)
                slant_m = slant_distance_m(ue, nd_lla)
                if surface_d_m <= nd.coverage_radius_km * 1000.0:
                    A[t, j] = 1
                    pd = propagation_delay_ms_from_distance_m(slant_m)
                    prop_ms[t, j] = pd
                    # proxy quality (higher is better)
                    # TN is lower propagation than GEO, often competitive with LEO near coast.
                    score[t, j] = self.score_bias + 55.0 - 0.20 * pd - 0.003 * (surface_d_m / 1000.0)
        return {"A": A, "PropDelay_ms": prop_ms, "Q_proxy": score}

    @staticmethod
    def demo_rome_ny_atg_nodes() -> "TNATGProvider":
        """
        Simple ATG/coastal-like nodes for prototype.
        (공항/연안 중심으로 배치해서 해양 한가운데는 LEO가 주가 되도록)
        """
        nodes = [
            TNATGNode("TN_FCO", 41.800, 12.238, region="Rome", coverage_radius_km=160),
            TNATGNode("TN_ItalyWest", 42.0, 10.0, region="ItalyCoast", coverage_radius_km=180),
            TNATGNode("TN_SpainWest", 43.2, -9.0, region="SpainCoast", coverage_radius_km=220),
            TNATGNode("TN_IrelandWest", 53.4, -10.5, region="IrelandCoast", coverage_radius_km=220),
            TNATGNode("TN_Newfoundland", 47.6, -52.7, region="CanadaCoast", coverage_radius_km=220),
            TNATGNode("TN_Halifax", 44.7, -63.6, region="Halifax", coverage_radius_km=220),
            TNATGNode("TN_Boston", 42.36, -71.0, region="Boston", coverage_radius_km=180),
            TNATGNode("TN_JFK", 40.641, -73.778, region="NewYork", coverage_radius_km=180),
        ]
        return TNATGProvider(nodes=nodes)


# ============================================================
# Synthetic LEO Satellite Provider (prototype)
# ============================================================

@dataclass
class SimpleLEOSatelliteProvider:
    n_sats: int = 96         # prototype default; scale up later
    altitude_m: float = 550_000.0
    min_elevation_deg: float = 10.0
    seed: int = 0
    base_score_bias: float = 20.0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # random-ish constellation parameters (prototype)
        self.phase0 = rng.uniform(0, 2 * math.pi, size=self.n_sats)
        self.lon0_deg = rng.uniform(-180, 180, size=self.n_sats)
        self.incl_deg = rng.uniform(45, 75, size=self.n_sats)  # high-lat friendly-ish
        # per-sat ground-track speed proxy (deg/sec in anomaly domain)
        # LEO orbital period ~95 min -> angular speed ~2pi/5700 s
        self.omega_orb = rng.normal(2 * math.pi / 5700.0, 0.00005, size=self.n_sats)
        self.omega_lon_deg_s = rng.normal(0.04, 0.01, size=self.n_sats)  # crude ground drift proxy

    def metadata(self) -> Dict[str, np.ndarray]:
        n = self.n_sats
        return {
            "node_ids": np.array([f"LEO_{i:04d}" for i in range(n)], dtype=object),
            "node_types": np.array(["NTN"] * n, dtype=object),
            "node_orbits": np.array(["LEO"] * n, dtype=object),
            "node_regions": np.array(["LEO"] * n, dtype=object),
        }

    def sat_lla_at(self, sat_idx: int, t_sec: float):
        """
        Synthetic circular-ish ground track (not precise orbital mechanics).
        Enough for coverage dynamics and handover behavior prototyping.
        """
        phase = self.phase0[sat_idx] + self.omega_orb[sat_idx] * t_sec
        incl = math.radians(self.incl_deg[sat_idx])

        # latitude oscillation by inclination
        lat_rad = math.asin(max(-1.0, min(1.0, math.sin(incl) * math.sin(phase))))
        lat_deg = math.degrees(lat_rad)

        # longitude drift
        lon_deg = self.lon0_deg[sat_idx] + self.omega_lon_deg_s[sat_idx] * t_sec + math.degrees(0.15 * math.cos(phase))
        lon_deg = wrap_lon_deg(lon_deg)

        return (lat_deg, lon_deg, self.altitude_m)

    def compute_matrices(self, traj: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        T = len(traj["time_sec"])
        N = self.n_sats
        A = np.zeros((T, N), dtype=int)
        prop_ms = np.full((T, N), np.nan, dtype=float)
        score = np.full((T, N), np.nan, dtype=float)

        for t in range(T):
            ue = (float(traj["lat_deg"][t]), float(traj["lon_deg"][t]), float(traj["alt_m"][t]))
            tt = float(traj["time_sec"][t])

            for j in range(N):
                sat = self.sat_lla_at(j, tt)
                el = elevation_angle_deg(ue, sat)
                if el >= self.min_elevation_deg:
                    A[t, j] = 1
                    sd = slant_distance_m(ue, sat)
                    pd = propagation_delay_ms_from_distance_m(sd)
                    prop_ms[t, j] = pd
                    # proxy quality: lower propagation better, higher elevation better
                    score[t, j] = self.base_score_bias + 65.0 - 0.45 * pd + 0.12 * el

        return {"A": A, "PropDelay_ms": prop_ms, "Q_proxy": score}


# ============================================================
# Phase Mask Provider (gate-to-gate constraints)
# ============================================================

@dataclass
class PhaseMaskProvider:
    gate_frac: float = 0.05
    takeoff_climb_frac: float = 0.15
    cruise_frac: float = 0.60
    descent_landing_frac: float = 0.15
    arrival_gate_frac: float = 0.05

    def _phase_from_progress(self, p: float) -> str:
        # cumulative boundaries
        b1 = self.gate_frac
        b2 = b1 + self.takeoff_climb_frac
        b3 = b2 + self.cruise_frac
        b4 = b3 + self.descent_landing_frac
        if p < b1:
            return "gate_departure"
        elif p < b2:
            return "takeoff_climb"
        elif p < b3:
            return "cruise"
        elif p < b4:
            return "descent_landing"
        else:
            return "gate_arrival"

    def phase_labels(self, traj: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([self._phase_from_progress(float(p)) for p in traj["progress"]], dtype=object)

    def build_mask(
        self,
        phase_labels: np.ndarray,
        node_types: np.ndarray,
        node_orbits: np.ndarray,
    ) -> np.ndarray:
        """
        1 = allowed by operations policy
        Current ATG+LEO experiment policy example:
          - gate: TN only
          - takeoff/climb: TN and LEO both allowed
          - cruise: TN and LEO both allowed (TN likely absent physically over ocean)
          - descent: TN and LEO both allowed
          - arrival gate: TN only
        """
        T = len(phase_labels)
        N = len(node_types)
        M = np.ones((T, N), dtype=int)

        for t in range(T):
            ph = str(phase_labels[t])
            for n in range(N):
                nt = str(node_types[n])
                no = str(node_orbits[n])

                allowed = True
                if ph in ("gate_departure", "gate_arrival"):
                    allowed = (nt == "TN")
                else:
                    # ATG+LEO only experiment
                    if nt == "TN":
                        allowed = True
                    elif nt == "NTN" and no == "LEO":
                        allowed = True
                    else:
                        allowed = False
                M[t, n] = 1 if allowed else 0

        return M