import numpy as np
import torch
from collections import deque
from skyfield.api import load, wgs84
from geopy.distance import geodesic
from datetime import timedelta

class TN_NTN_Env:
    def __init__(self, start_lat=37.5, start_lon=127.0, start_time_utc=(2026, 2, 23, 12, 0, 0), seq_length=20):
        # 1. 시뮬레이션 환경 기본 파라미터 (1초 타임슬롯, 고도 10km, 시속 900km)
        self.time_step = 1  
        self.aircraft_alt_m = 10000  
        self.speed_mps = 250  
        self.min_elevation_deg = 10.0  
        self.tn_coverage_radius_km = 40.0  
        
        # 2. Transformer 입력을 위한 시계열 버퍼 (Sliding Window)
        self.seq_length = seq_length
        self.buffer = deque(maxlen=self.seq_length)
        
        # 3. 시간 및 항공기 초기 위치 설정
        self.ts = load.timescale()
        self.current_time = self.ts.utc(*start_time_utc)
        self.current_lat = start_lat
        self.current_lon = start_lon
        
        # 4. NTN (위성) 데이터 로드 (Celestrak 최신 API)
        print("Loading Satellite TLE data...")
        starlink_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
        self.satellites = load.tle_file(starlink_url, filename='starlink.txt')
        print(f"Loaded {len(self.satellites)} satellites.")
        
        # 5. TN (지상 기지국) 임의 생성
        self.tn_stations = self._generate_tn_stations(start_lat, start_lon, num_stations=50)
        
    def _generate_tn_stations(self, center_lat, center_lon, num_stations):
        """비행기 초기 위치 주변에 지상 기지국 무작위 배치"""
        stations = []
        for i in range(num_stations):
            lat = center_lat + np.random.uniform(-1.0, 1.0)
            lon = center_lon + np.random.uniform(-1.0, 1.0)
            stations.append({'id': f'TN_{i}', 'lat': lat, 'lon': lon})
        return stations

    def _update_aircraft_trajectory(self):
        """1초(250m) 이동에 따른 항공기의 위경도 갱신"""
        lat_movement_deg = (self.speed_mps * 0.707) / 111320.0
        lon_movement_deg = (self.speed_mps * 0.707) / (111320.0 * np.cos(np.radians(self.current_lat)))
        
        self.current_lat += lat_movement_deg
        self.current_lon += lon_movement_deg
        
        self.current_time = self.ts.utc(
            self.current_time.utc_datetime() + timedelta(seconds=self.time_step)
        )

    def get_visible_tn(self):
        """커버리지 반경(40km) 이내의 지상 기지국 탐색"""
        visible_tns = []
        aircraft_coords = (self.current_lat, self.current_lon)
        for tn in self.tn_stations:
            dist_km = geodesic(aircraft_coords, (tn['lat'], tn['lon'])).kilometers
            if dist_km <= self.tn_coverage_radius_km:
                visible_tns.append({'id': tn['id'], 'distance_km': round(dist_km, 2)})
        return visible_tns

    def get_visible_ntn(self):
        """최소 앙각(10도) 이상의 위성 탐색"""
        visible_ntns = []
        aircraft_pos = wgs84.latlon(self.current_lat, self.current_lon, elevation_m=self.aircraft_alt_m)
        
        for sat in self.satellites[:300]: 
            topocentric = (sat - aircraft_pos).at(self.current_time)
            alt, az, distance = topocentric.altaz()
            
            if alt.degrees >= self.min_elevation_deg:
                visible_ntns.append({
                    'id': sat.name, 
                    'elevation_deg': round(alt.degrees, 2),
                    'distance_km': round(distance.km, 2)
                })
        return visible_ntns

    def _extract_features(self, raw_state):
        """원시 상태 데이터를 6차원 특징 벡터(Numpy 배열)로 변환"""
        lat, lon = raw_state['aircraft_pos']
        num_tns = len(raw_state['available_TNs'])
        num_ntns = len(raw_state['available_NTNs'])
        
        min_tn_dist = min([tn['distance_km'] for tn in raw_state['available_TNs']]) if num_tns > 0 else 100.0
        max_ntn_elev = max([ntn['elevation_deg'] for ntn in raw_state['available_NTNs']]) if num_ntns > 0 else 0.0
            
        feature_vector = [lat, lon, num_tns, num_ntns, min_tn_dist, max_ntn_elev]
        return np.array(feature_vector, dtype=np.float32)

    def step(self):
        """
        1초 진행 후, 원시 데이터(Raw State)와 
        Transformer 입력용 PyTorch 텐서(Tensor)를 동시 반환
        """
        self._update_aircraft_trajectory()
        
        # 1. 현재 1초의 원시 데이터 수집
        raw_state = {
            'time_utc': self.current_time.utc_datetime().strftime('%Y-%m-%d %H:%M:%S'),
            'aircraft_pos': (round(self.current_lat, 4), round(self.current_lon, 4)),
            'available_TNs': self.get_visible_tn(),
            'available_NTNs': self.get_visible_ntn()
        }
        
        # 2. 6차원 특징 추출 후 버퍼에 추가
        features = self._extract_features(raw_state)
        self.buffer.append(features)
        
        # 3. 버퍼가 꽉 찼는지 확인 후 텐서 생성
        state_tensor = None
        if len(self.buffer) == self.seq_length:
            seq_array = np.array(self.buffer)
            # Shape: (Batch=1, Seq_Length=20, Features=6)
            state_tensor = torch.tensor(seq_array).unsqueeze(0)
            
        return raw_state, state_tensor

# ==========================================
# 실행 테스트
# ==========================================
if __name__ == "__main__":
    # L=20초 설정으로 환경 초기화
    env = TN_NTN_Env(seq_length=20)
    
    print("Starting Flight Simulation (1 step = 1 second)...\n")
    
    # 25초 동안의 비행 시뮬레이션
    for t in range(1, 26):
        raw_state, state_tensor = env.step()
        
        print(f"[Time: {raw_state['time_utc']}] Pos: {raw_state['aircraft_pos']}")
        
        if state_tensor is not None:
            print(f"  ✅ [버퍼 완성] Transformer 입력 텐서 생성 완료! Shape: {state_tensor.shape}")
        else:
            print(f"  ⏳ [버퍼 수집 중] 현재 데이터 개수: {len(env.buffer)}/20")
            
        print("-" * 60)