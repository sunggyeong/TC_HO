import math
import numpy as np

EARTH_RADIUS_M = 6371_000.0
C_MPS = 299_792_458.0


def wrap_lon_deg(lon_deg: float) -> float:
    x = (lon_deg + 180.0) % 360.0 - 180.0
    return x


def haversine_distance_m(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Great-circle distance on Earth surface (ignores altitude)."""
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_M * c


def lla_to_ecef_m(lat_deg, lon_deg, alt_m):
    """Spherical Earth ECEF (prototype-level)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = EARTH_RADIUS_M + alt_m
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def slant_distance_m(lla_a, lla_b):
    """3D straight-line distance in ECEF."""
    pa = lla_to_ecef_m(*lla_a)
    pb = lla_to_ecef_m(*lla_b)
    return float(np.linalg.norm(pb - pa))


def elevation_angle_deg(observer_lla, target_lla):
    """
    Elevation angle from observer to target using local zenith.
    > 0 deg means above local horizon.
    """
    po = lla_to_ecef_m(*observer_lla)
    pt = lla_to_ecef_m(*target_lla)
    los = pt - po
    los_norm = np.linalg.norm(los)
    if los_norm <= 1e-9:
        return -90.0

    zenith = po / (np.linalg.norm(po) + 1e-12)
    sin_el = float(np.dot(los / los_norm, zenith))
    sin_el = max(-1.0, min(1.0, sin_el))
    return math.degrees(math.asin(sin_el))


def propagation_delay_ms_from_distance_m(distance_m: float) -> float:
    return (distance_m / C_MPS) * 1000.0


def lerp(a, b, t):
    return a + (b - a) * t