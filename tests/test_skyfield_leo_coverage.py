# tests/test_skyfield_leo_coverage.py
from __future__ import annotations
import numpy as np
from pathlib import Path

from providers.satellite_provider_skyfield import SkyfieldLEOConfig, SkyfieldFrozenTLEProvider


def build_dummy_aircraft_track(T=300, dt=1.0):
    """
    5분짜리 단순 직선 비행 (데모)
    위도/경도는 아주 단순한 선형 변화 (정밀 항법 X, provider 테스트용)
    """
    sim_t = np.arange(T, dtype=float) * dt

    # 예: 북대서양 일부 구간을 단순 선형 근사
    lat0, lon0, alt0 = 47.0, -20.0, 10000.0
    lat1, lon1, alt1 = 48.0, -22.5, 10000.0

    lat = np.linspace(lat0, lat1, T)
    lon = np.linspace(lon0, lon1, T)
    alt = np.linspace(alt0, alt1, T)
    return sim_t, lat, lon, alt


def main():
    tle_candidates = sorted(Path("data").glob("starlink_frozen_*.tle"))
    if not tle_candidates:
        raise FileNotFoundError("data/starlink_frozen_*.tle 파일이 없습니다.")
    tle_path = str(tle_candidates[-1])

    cfg = SkyfieldLEOConfig(
        tle_path=tle_path,
        min_elevation_deg=10.0,
        max_sats=200,                 # 테스트 속도용
        topk_visible_per_slot=20,     # 슬롯당 후보 수 제한
        epoch_utc="2026-02-24T12:00:00Z",
    )
    provider = SkyfieldFrozenTLEProvider(cfg)

    sim_t, lat, lon, alt = build_dummy_aircraft_track(T=180, dt=1.0)

    out = provider.build_leo_coverage_matrix(
        sim_time_sec_arr=sim_t,
        lat_arr=lat,
        lon_arr=lon,
        alt_m_arr=alt,
        max_global_candidates=80,
    )

    A = out["A_leo"]
    latency = out["latency_ms"]
    elev = out["elevation_deg"]
    meta = out["meta"]

    print("A_leo shape:", A.shape)
    print("Visible ratio:", A.mean())
    print("Num global LEO candidates:", len(meta))

    # 저장
    Path("results").mkdir(exist_ok=True)
    np.save("results/test_A_leo.npy", A)
    np.save("results/test_latency_leo.npy", latency)
    np.save("results/test_elev_leo.npy", elev)
    print("Saved results/test_A_leo.npy, test_latency_leo.npy, test_elev_leo.npy")


if __name__ == "__main__":
    main()