import numpy as np
from skyfield.api import load, wgs84
from geopy.distance import geodesic
from datetime import timedelta

class TN_NTN_Env:
    def __init__(self, start_lat=37.5, start_lon=127.0, start_time_utc=(2026, 2, 23, 12, 0, 0)):
        # 1. 시뮬레이션 환경 기본 파라미터 (1초 타임슬롯, 고도 10km, 시속 900km)
        self.time_step = 1  # 1초 단위
        self.aircraft_alt_m = 10000  # 10km 고도
        self.speed_mps = 250  # 900 km/h = 250 m/s
        self.min_elevation_deg = 10.0  # 위성 통신 최소 앙각 (Sustainable 논문 기준)
        self.tn_coverage_radius_km = 40.0  # TN (ATG) 셀 커버리지 반경
        
        # 2. 시간 설정 (Skyfield Timescale)
        self.ts = load.timescale()
        self.current_time = self.ts.utc(*start_time_utc)
        
        # 3. 항공기 초기 위치 설정
        self.current_lat = start_lat
        self.current_lon = start_lon
        
        # 4. NTN (위성) 데이터 로드 (Starlink TLE 데이터 활용)
        # 실제 구동 시 celestrak에서 최신 궤도 데이터를 다운로드하여 캐싱합니다.
        print("Loading Satellite TLE data...")
        starlink_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
        self.satellites = load.tle_file(starlink_url, filename='starlink.txt')
        print(f"Loaded {len(self.satellites)} satellites.")
        
        # 5. TN (지상 기지국) 임의 생성 (비행 경로 주변 50개)
        self.tn_stations = self._generate_tn_stations(start_lat, start_lon, num_stations=50)
        
    def _generate_tn_stations(self, center_lat, center_lon, num_stations):
        """비행기 초기 위치 주변(약 100km 반경 이내)에 지상 기지국을 무작위 배치"""
        stations = []
        for i in range(num_stations):
            # 대략적인 위도/경도 난수 생성 (1도는 약 111km)
            lat = center_lat + np.random.uniform(-1.0, 1.0)
            lon = center_lon + np.random.uniform(-1.0, 1.0)
            stations.append({'id': f'TN_{i}', 'lat': lat, 'lon': lon})
        return stations

    def _update_aircraft_trajectory(self):
        """1초(250m) 이동에 따른 항공기의 위경도 좌표 갱신 (단순 직진 비행 가정)"""
        # 1도는 약 111,320m. 북동쪽으로 비행한다고 가정하여 좌표 업데이트
        lat_movement_deg = (self.speed_mps * 0.707) / 111320.0
        lon_movement_deg = (self.speed_mps * 0.707) / (111320.0 * np.cos(np.radians(self.current_lat)))
        
        self.current_lat += lat_movement_deg
        self.current_lon += lon_movement_deg
        
        # 시간 1초 증가
        self.current_time = self.ts.utc(
            self.current_time.utc_datetime() + timedelta(seconds=self.time_step)
        )

    def get_visible_tn(self):
        """현재 위치에서 커버리지 반경(40km) 이내에 있는 지상 기지국 탐색"""
        visible_tns = []
        aircraft_coords = (self.current_lat, self.current_lon)
        
        for tn in self.tn_stations:
            tn_coords = (tn['lat'], tn['lon'])
            # 항공기와 기지국 간의 2D 지표면 거리 계산 (3D 슬랜트 거리로 확장 가능)
            dist_km = geodesic(aircraft_coords, tn_coords).kilometers
            if dist_km <= self.tn_coverage_radius_km:
                visible_tns.append({'id': tn['id'], 'distance_km': round(dist_km, 2)})
                
        return visible_tns

    def get_visible_ntn(self):
        """현재 위치에서 최소 앙각(10도) 이상으로 보이는 위성 탐색"""
        visible_ntns = []
        # 항공기의 3차원 위치 객체 생성
        aircraft_position = wgs84.latlon(self.current_lat, self.current_lon, elevation_m=self.aircraft_alt_m)
        
        # 연산 속도를 위해 전체 위성 중 일부(예: 300개)만 샘플링하여 검사 (실제 연구에서는 필터링 로직 고도화 필요)
        for sat in self.satellites[:300]: 
            # 위성과 항공기 간의 상대 위치(Topocentric) 계산
            difference = sat - aircraft_position
            topocentric = difference.at(self.current_time)
            alt, az, distance = topocentric.altaz()
            
            if alt.degrees >= self.min_elevation_deg:
                visible_ntns.append({
                    'id': sat.name, 
                    'elevation_deg': round(alt.degrees, 2),
                    'distance_km': round(distance.km, 2)
                })
                
        return visible_ntns

    def step(self):
        """타임슬롯 1틱 진행 후 상태(State) 반환"""
        self._update_aircraft_trajectory()
        
        state = {
            'time_utc': self.current_time.utc_datetime().strftime('%Y-%m-%d %H:%M:%S'),
            'aircraft_pos': (round(self.current_lat, 4), round(self.current_lon, 4)),
            'available_TNs': self.get_visible_tn(),
            'available_NTNs': self.get_visible_ntn()
        }
        return state

# ==========================================
# 실행 테스트
# ==========================================
if __name__ == "__main__":
    env = TN_NTN_Env()
    
    print("Starting Flight Simulation (1 step = 1 second)...\n")
    # 5초 동안의 비행 시뮬레이션 및 가시성 체크
    for t in range(5):
        current_state = env.step()
        print(f"[Time: {current_state['time_utc']}] Pos: {current_state['aircraft_pos']}")
        print(f"  - Visible TNs ({len(current_state['available_TNs'])}): {current_state['available_TNs']}")
        print(f"  - Visible NTNs ({len(current_state['available_NTNs'])}): {current_state['available_NTNs'][:2]} ...") # 출력 길이상 2개만 표기
        print("-" * 60)