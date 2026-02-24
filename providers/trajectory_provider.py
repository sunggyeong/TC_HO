# providers/trajectory_provider.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np


@dataclass
class Trajectory:
    times_utc: List[datetime]   # length T
    lat_deg: np.ndarray         # shape [T]
    lon_deg: np.ndarray         # shape [T]
    alt_m: np.ndarray           # shape [T]
    phase_label: np.ndarray     # optional, shape [T], dtype=object


@dataclass
class Waypoint:
    lat_deg: float
    lon_deg: float
    alt_m: float


class WaypointTrajectoryProvider:
    """
    waypoint 기반 trajectory 생성기.
    - 1초 타임슬롯 기준 선형 보간
    - 실제 실험에서는 flight plan/ADS-B 경로로 교체 가능
    """
    def __init__(
        self,
        waypoints: List[Waypoint],
        speed_mps: float = 250.0,       # 900 km/h
        dt_sec: float = 1.0,
        start_time_utc: datetime | None = None,
    ):
        if len(waypoints) < 2:
            raise ValueError("waypoints는 최소 2개 이상 필요")
        self.waypoints = waypoints
        self.speed_mps = float(speed_mps)
        self.dt_sec = float(dt_sec)
        self.start_time_utc = start_time_utc or datetime(2026, 2, 24, 0, 0, 0, tzinfo=timezone.utc)

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
        return 2.0 * R * np.arcsin(np.sqrt(a))

    def build(self) -> Trajectory:
        lats, lons, alts = [], [], []
        phases = []

        t_cur = self.start_time_utc
        times = [t_cur]

        # 간단한 gate-to-gate phase 예시 (원하면 별도 provider로 오버라이드)
        def _phase_from_alt(alt_m: float) -> str:
            if alt_m < 1000:
                return "gate_or_taxi"
            elif alt_m < 5000:
                return "takeoff_climb"
            elif alt_m < 9000:
                return "climb"
            else:
                return "cruise"

        # 첫 점
        lats.append(self.waypoints[0].lat_deg)
        lons.append(self.waypoints[0].lon_deg)
        alts.append(self.waypoints[0].alt_m)
        phases.append(_phase_from_alt(self.waypoints[0].alt_m))

        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i]
            b = self.waypoints[i + 1]

            horiz_m = self._haversine_m(a.lat_deg, a.lon_deg, b.lat_deg, b.lon_deg)
            vert_m = b.alt_m - a.alt_m
            dist_m = float(np.sqrt(horiz_m ** 2 + vert_m ** 2))

            if dist_m < 1e-6:
                continue

            seg_sec = dist_m / self.speed_mps
            n_steps = max(1, int(np.ceil(seg_sec / self.dt_sec)))

            for k in range(1, n_steps + 1):
                r = k / n_steps
                lat = (1 - r) * a.lat_deg + r * b.lat_deg
                lon = (1 - r) * a.lon_deg + r * b.lon_deg
                alt = (1 - r) * a.alt_m + r * b.alt_m

                t_cur = t_cur + timedelta(seconds=self.dt_sec)
                times.append(t_cur)
                lats.append(lat)
                lons.append(lon)
                alts.append(alt)
                phases.append(_phase_from_alt(alt))

        return Trajectory(
            times_utc=times,
            lat_deg=np.asarray(lats, dtype=float),
            lon_deg=np.asarray(lons, dtype=float),
            alt_m=np.asarray(alts, dtype=float),
            phase_label=np.asarray(phases, dtype=object),
        )


class PerturbedTrajectoryProvider:
    """
    predicted trajectory 모사 용도:
    base trajectory에 위치 오차를 주어 perturbed trajectory 생성
    """
    def __init__(self, base_provider: WaypointTrajectoryProvider, latlon_noise_deg=0.01, alt_noise_m=50.0, seed=0):
        self.base_provider = base_provider
        self.latlon_noise_deg = float(latlon_noise_deg)
        self.alt_noise_m = float(alt_noise_m)
        self.rng = np.random.default_rng(seed)

    def build(self) -> Trajectory:
        traj = self.base_provider.build()
        lat = traj.lat_deg + self.rng.normal(0.0, self.latlon_noise_deg, size=traj.lat_deg.shape)
        lon = traj.lon_deg + self.rng.normal(0.0, self.latlon_noise_deg, size=traj.lon_deg.shape)
        alt = np.maximum(0.0, traj.alt_m + self.rng.normal(0.0, self.alt_noise_m, size=traj.alt_m.shape))
        return Trajectory(
            times_utc=traj.times_utc,
            lat_deg=lat,
            lon_deg=lon,
            alt_m=alt,
            phase_label=traj.phase_label.copy(),
        )