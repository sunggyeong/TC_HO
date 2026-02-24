# providers/satellite_provider_skyfield.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math

import numpy as np

try:
    from skyfield.api import load, wgs84
except ImportError as e:
    raise ImportError(
        "skyfield가 설치되어 있지 않습니다. `pip install skyfield sgp4` 실행하세요."
    ) from e


C_KM_PER_S = 299_792.458  # speed of light [km/s]


@dataclass
class SkyfieldLEOConfig:
    tle_path: str
    min_elevation_deg: float = 10.0
    max_sats: int = 300                    # 로딩 후 상위 N개 위성만 사용(속도 절충)
    base_latency_ms: float = 20.0          # LEO 링크 + 네트워크/처리 기본값 (one-way proxy)
    core_delay_ms: float = 5.0             # 추가 코어 지연 proxy
    capacity_score: float = 0.95
    epoch_utc: str = "2026-02-24T12:00:00Z"  # 시뮬 기준 epoch (frozen 실험 재현성)
    sort_visible_by: str = "elevation"     # "elevation" | "slant"
    topk_visible_per_slot: Optional[int] = None  # 슬롯당 후보 제한 (None이면 전부)


class SkyfieldFrozenTLEProvider:
    """
    frozen TLE(로컬 파일) 기반으로 항공기 위치에서 보이는 LEO 위성을 계산.
    - 최소 앙각 필터
    - slant distance 계산
    - latency proxy 계산
    - 슬롯별 visible candidate 목록 생성
    """

    def __init__(self, cfg: SkyfieldLEOConfig):
        self.cfg = cfg
        self.tle_path = Path(cfg.tle_path)
        if not self.tle_path.exists():
            raise FileNotFoundError(f"Frozen TLE file not found: {self.tle_path}")

        self.ts = load.timescale()
        self._satellites = load.tle_file(str(self.tle_path))  # list[EarthSatellite]

        # 이름 기준 deterministic 정렬 후 max_sats 컷
        self._satellites = sorted(self._satellites, key=lambda s: s.name or "")[: cfg.max_sats]

        self.epoch_dt = self._parse_epoch(cfg.epoch_utc)

    @staticmethod
    def _parse_epoch(epoch_str: str) -> datetime:
        # "2026-02-24T12:00:00Z" -> aware UTC datetime
        if epoch_str.endswith("Z"):
            epoch_str = epoch_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(epoch_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _simsec_to_datetime(self, sim_time_sec: float) -> datetime:
        return self.epoch_dt + timedelta(seconds=float(sim_time_sec))

    def _skyfield_time_from_datetime(self, dt_utc: datetime):
        # Skyfield는 aware datetime도 받지만, 명시적으로 UTC로 고정
        dt_utc = dt_utc.astimezone(timezone.utc)
        return self.ts.from_datetime(dt_utc)

    def _oneway_latency_ms_from_slant_km(self, slant_km: float) -> float:
        # propagation one-way + proxy core/base delay
        prop_ms = (slant_km / C_KM_PER_S) * 1000.0
        return self.cfg.base_latency_ms + self.cfg.core_delay_ms + prop_ms

    def visible_candidates_at(
        self,
        sim_time_sec: float,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
    ) -> List[Dict[str, Any]]:
        """
        한 슬롯에서 visible LEO 후보 리스트 반환.
        각 원소:
          {
            "id": "LEO_xxx",
            "name": "...",
            "kind": "LEO",
            "elevation_deg": ...,
            "slant_km": ...,
            "latency_ms": ...,
            "capacity_score": ...
          }
        """
        dt_utc = self._simsec_to_datetime(sim_time_sec)
        t_sf = self._skyfield_time_from_datetime(dt_utc)

        # 항공기 위치 (WGS84, 고도 포함)
        observer = wgs84.latlon(latitude_degrees=lat_deg,
                                longitude_degrees=lon_deg,
                                elevation_m=alt_m)

        visible = []
        for sat in self._satellites:
            # observer 기준 topocentric
            topocentric = (sat - observer).at(t_sf)
            alt, az, distance = topocentric.altaz()

            elev_deg = float(alt.degrees)
            if elev_deg < self.cfg.min_elevation_deg:
                continue

            slant_km = float(distance.km)
            latency_ms = self._oneway_latency_ms_from_slant_km(slant_km)

            sat_name = sat.name or "UNKNOWN"
            sat_id = f"LEO::{sat_name}"

            visible.append({
                "id": sat_id,
                "name": sat_name,
                "kind": "LEO",
                "elevation_deg": elev_deg,
                "slant_km": slant_km,
                "latency_ms": latency_ms,
                "capacity_score": float(self.cfg.capacity_score),
            })

        # 정렬
        if self.cfg.sort_visible_by == "slant":
            visible.sort(key=lambda x: (x["slant_km"], -x["elevation_deg"], x["name"]))
        else:  # default elevation desc
            visible.sort(key=lambda x: (-x["elevation_deg"], x["slant_km"], x["name"]))

        if self.cfg.topk_visible_per_slot is not None:
            visible = visible[: self.cfg.topk_visible_per_slot]

        return visible

    def visible_candidates_for_trajectory(
        self,
        sim_time_sec_arr: np.ndarray,
        lat_arr: np.ndarray,
        lon_arr: np.ndarray,
        alt_m_arr: np.ndarray,
    ) -> List[List[Dict[str, Any]]]:
        """
        trajectory 전체 슬롯에 대해 slot-wise visible candidate list 생성
        """
        T = len(sim_time_sec_arr)
        out: List[List[Dict[str, Any]]] = []
        for t in range(T):
            out.append(self.visible_candidates_at(
                sim_time_sec=float(sim_time_sec_arr[t]),
                lat_deg=float(lat_arr[t]),
                lon_deg=float(lon_arr[t]),
                alt_m=float(alt_m_arr[t]),
            ))
        return out

    def build_leo_coverage_matrix(
        self,
        sim_time_sec_arr: np.ndarray,
        lat_arr: np.ndarray,
        lon_arr: np.ndarray,
        alt_m_arr: np.ndarray,
        max_global_candidates: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        슬롯별 visible 후보를 전역 candidate index로 pack해서 coverage matrix 생성.

        Returns:
          {
            "A_leo": (T, N) int8,
            "latency_ms": (T, N) float32 (비가시면 nan),
            "elevation_deg": (T, N) float32 (비가시면 nan),
            "slant_km": (T, N) float32 (비가시면 nan),
            "meta": List[dict]   # global candidate meta
            "slot_visible": List[List[dict]]
          }
        """
        slot_visible = self.visible_candidates_for_trajectory(
            sim_time_sec_arr, lat_arr, lon_arr, alt_m_arr
        )

        # dwell count 기반으로 전역 후보 선택
        dwell_count: Dict[str, int] = {}
        first_seen_meta: Dict[str, Dict[str, Any]] = {}

        for slot in slot_visible:
            for c in slot:
                cid = c["id"]
                dwell_count[cid] = dwell_count.get(cid, 0) + 1
                if cid not in first_seen_meta:
                    first_seen_meta[cid] = {
                        "id": c["id"],
                        "name": c["name"],
                        "kind": c["kind"],
                    }

        # dwell_count desc, name asc 정렬
        ranked_ids = sorted(dwell_count.keys(), key=lambda cid: (-dwell_count[cid], cid))

        if max_global_candidates is None:
            max_global_candidates = self.cfg.max_sats
        ranked_ids = ranked_ids[:max_global_candidates]

        id2idx = {cid: i for i, cid in enumerate(ranked_ids)}
        meta = []
        for cid in ranked_ids:
            m = dict(first_seen_meta[cid])
            m["global_idx"] = len(meta)
            m["dwell_count"] = dwell_count[cid]
            meta.append(m)

        T = len(sim_time_sec_arr)
        N = len(meta)

        A = np.zeros((T, N), dtype=np.int8)
        latency = np.full((T, N), np.nan, dtype=np.float32)
        elev = np.full((T, N), np.nan, dtype=np.float32)
        slant = np.full((T, N), np.nan, dtype=np.float32)

        for t, slot in enumerate(slot_visible):
            for c in slot:
                cid = c["id"]
                j = id2idx.get(cid, None)
                if j is None:
                    continue
                A[t, j] = 1
                latency[t, j] = float(c["latency_ms"])
                elev[t, j] = float(c["elevation_deg"])
                slant[t, j] = float(c["slant_km"])

        return {
            "A_leo": A,
            "latency_ms": latency,
            "elevation_deg": elev,
            "slant_km": slant,
            "meta": meta,
            "slot_visible": slot_visible,
        }
    def _safe_attr_names(self, obj):
        try:
            return [n for n in dir(obj) if not n.startswith("_")]
        except Exception:
            return []

    def _to_mapping_like(self, obj):
        """
        dataclass / 일반 객체 / dict를 최대한 dict처럼 다루기 위한 helper
        """
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        try:
            return vars(obj)  # 일반 객체, dataclass 인스턴스 대부분 가능
        except Exception:
            return None

    def _get_candidates_from_obj(self, obj, names):
        """
        obj(dict or object)에서 names 후보 중 하나를 찾아 반환
        """
        if obj is None:
            return None

        # dict-like
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]

        # object-like
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)

        return None

    def _extract_traj_arrays(self, traj):
        """
        traj에서 time/lat/lon/alt 배열을 최대한 유연하게 추출
        - 직접 필드
        - dict 필드
        - 중첩 필드(trajectory/track/path/route 등) 1단계
        """
        time_names = [
            "sim_time_sec", "sim_time_sec_arr", "time_sec", "time_s", "times_sec",
            "t_sec", "t", "time", "times", "times_utc"
        ]
        lat_names = [
            "lat_deg", "lat_arr", "lat", "lats", "latitude", "latitudes", "lat_seq"
        ]
        lon_names = [
            "lon_deg", "lon_arr", "lon", "lons", "longitude", "longitudes", "lon_seq"
        ]
        alt_names = [
            "alt_m", "alt_m_arr", "alt", "alts", "altitude", "altitude_m",
            "altitudes", "alt_seq", "z_m"
        ]

        # 1) traj 자체에서 바로 찾기
        sim_t = self._get_candidates_from_obj(traj, time_names)
        lat = self._get_candidates_from_obj(traj, lat_names)
        lon = self._get_candidates_from_obj(traj, lon_names)
        alt = self._get_candidates_from_obj(traj, alt_names)

        if all(v is not None for v in [sim_t, lat, lon, alt]):
            sim_t = self._normalize_time_array_to_simsec(sim_t)
            return sim_t, lat, lon, alt, "direct"

        # 2) 중첩 객체/딕셔너리 1단계 탐색
        nested_container_names = [
            "trajectory", "track", "path", "route", "mobility", "state", "data"
        ]
        nested = self._get_candidates_from_obj(traj, nested_container_names)

        if nested is not None:
            sim_t2 = self._get_candidates_from_obj(nested, time_names)
            lat2 = self._get_candidates_from_obj(nested, lat_names)
            lon2 = self._get_candidates_from_obj(nested, lon_names)
            alt2 = self._get_candidates_from_obj(nested, alt_names)
            if all(v is not None for v in [sim_t2, lat2, lon2, alt2]):
                sim_t2 = self._normalize_time_array_to_simsec(sim_t2)
                return sim_t2, lat2, lon2, alt2, "nested"

        # 3) 실패 시 디버그 정보 제공
        traj_map = self._to_mapping_like(traj)
        nested_debug = None
        if nested is not None:
            nested_map = self._to_mapping_like(nested)
            if nested_map is not None:
                nested_debug = list(nested_map.keys())
            else:
                nested_debug = self._safe_attr_names(nested)[:60]

        debug_info = {
            "traj_type": type(traj).__name__,
            "traj_keys_or_attrs": (
                list(traj_map.keys()) if traj_map is not None else self._safe_attr_names(traj)[:80]
            ),
            "nested_found_type": (type(nested).__name__ if nested is not None else None),
            "nested_keys_or_attrs": nested_debug,
        }

        raise ValueError(
            "traj에서 필요한 필드를 찾지 못했습니다.\n"
            "필요 필드(예시): time/lat/lon/alt 계열 배열\n"
            f"디버그 정보: {debug_info}"
        )

    def build_matrix(self, traj):
        """
        기존 env 코드와 호환되는 generic interface.
        env가 satellite_provider.build_matrix(traj)를 호출할 때 사용.
        """
        sim_t, lat, lon, alt, source_kind = self._extract_traj_arrays(traj)

        out = self.build_leo_coverage_matrix(
            sim_time_sec_arr=np.asarray(sim_t, dtype=float),
            lat_arr=np.asarray(lat, dtype=float),
            lon_arr=np.asarray(lon, dtype=float),
            alt_m_arr=np.asarray(alt, dtype=float),
            max_global_candidates=self.cfg.max_sats,
        )
                # -------- utility 생성 (env 호환용) --------
        # 우선순위: 높은 앙각(+), 낮은 지연(+)
        # 출력 범위는 대략 [0,1]로 정규화
        A_leo = out["A_leo"].astype(float)

        # 키 이름 유연 대응
        elev = out.get("elev_deg", out.get("elevation_deg", None))
        lat_ms = out.get("latency_ms", out.get("latency", None))

        # 기본값 fallback
        if elev is None:
            elev = np.zeros_like(A_leo, dtype=float)
        else:
            elev = np.asarray(elev, dtype=float)

        if lat_ms is None:
            # LEO 기본 지연값 fallback (대략 30~40ms 수준)
            lat_ms = np.full_like(A_leo, 35.0, dtype=float)
        else:
            lat_ms = np.asarray(lat_ms, dtype=float)

        # 미가시 링크(A=0)는 utility 0이 되도록 마스킹
        # 앙각 정규화 (min_elev ~ 90도)
        min_elev = float(getattr(self.cfg, "min_elevation_deg", 10.0))
        elev_norm = (elev - min_elev) / max(1e-6, (90.0 - min_elev))
        elev_norm = np.clip(elev_norm, 0.0, 1.0)

        # 지연 정규화 (낮을수록 좋음)
        # row/전체 기준 robust 정규화 (값이 거의 같으면 1로 처리)
        lat_valid = lat_ms[A_leo > 0.5]
        if lat_valid.size >= 2:
            lmin = float(np.min(lat_valid))
            lmax = float(np.max(lat_valid))
            if lmax - lmin < 1e-9:
                lat_score = np.ones_like(lat_ms, dtype=float)
            else:
                lat_score = 1.0 - (lat_ms - lmin) / (lmax - lmin)
                lat_score = np.clip(lat_score, 0.0, 1.0)
        else:
            lat_score = np.ones_like(lat_ms, dtype=float)

        # 가중합 (필요하면 나중에 조정)
        w_elev = 0.4
        w_lat = 0.6
        utility = (w_elev * elev_norm + w_lat * lat_score) * A_leo

        out["utility"] = utility

        # env 호환용 generic alias
        out["A"] = out["A_leo"]
        out["node_meta"] = out["meta"]
        out["kind"] = "LEO"
        out["traj_source_kind"] = source_kind
        return out
    def _normalize_time_array_to_simsec(self, time_like):
        """
        traj의 시간 배열을 sim_time_sec(float seconds)로 변환.
        지원:
          - 숫자 배열 (이미 sim sec)
          - datetime 리스트/배열
          - numpy.datetime64 배열
          - ISO 문자열 리스트
        기준 epoch: self.epoch_dt (UTC)
        """
        if time_like is None:
            return None

        # numpy array / list 통일
        if isinstance(time_like, np.ndarray):
            arr = time_like.tolist()
        else:
            arr = list(time_like)

        if len(arr) == 0:
            return np.asarray([], dtype=float)

        first = arr[0]

        # 이미 숫자 시간이면 그대로 float 변환
        if isinstance(first, (int, float, np.integer, np.floating)):
            return np.asarray(arr, dtype=float)

        from datetime import datetime, timezone

        def _to_utc_datetime(x):
            # numpy.datetime64
            if isinstance(x, np.datetime64):
                # ms 정밀도로 변환 후 python datetime
                x = x.astype("datetime64[ms]").tolist()

            # 문자열
            if isinstance(x, str):
                s = x.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                x = datetime.fromisoformat(s)

            # python datetime
            if not isinstance(x, datetime):
                raise TypeError(f"지원하지 않는 시간 타입: {type(x)}")

            if x.tzinfo is None:
                x = x.replace(tzinfo=timezone.utc)
            return x.astimezone(timezone.utc)

        sim_secs = []
        for x in arr:
            dt = _to_utc_datetime(x)
            sim_secs.append((dt - self.epoch_dt).total_seconds())

        return np.asarray(sim_secs, dtype=float)
    