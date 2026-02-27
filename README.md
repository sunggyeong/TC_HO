# TC-HO: Transformer + Consistency for Handover Timing Optimization

LEO 위성 핸드오버 시점 최적화를 위한 Transformer + Consistency(TC) 학습/평가 파이프라인입니다.  
예측 궤적 기반으로 (time_offset, node) 계획을 세우고, 실시간 보정(Realtime Corrected) 및 오프라인 계획(Offline)을 지원합니다.

---

## 요구 사항

- Python 3.10+
- PyTorch, numpy, pandas
- skyfield, networkx (환경·DAG 베이스라인)
- 프로젝트 내 `providers`, `envs`, `baselines`, `models`(또는 `proposed_tc_planner`) 모듈

TLE 기반 궤도/가시성 계산을 위해 **`data/starlink_frozen_20260224_1410Z.tle`** 파일이 필요합니다.  
(`env_core.ExperimentConfig.tle_path` 기본값과 동일한 경로에 두면 됩니다.)

---

## 디렉터리 구조

```
TC_HO/
├── paper_pkg/           # 논문용 학습/평가
│   ├── train.py         # 학습 진입점
│   ├── eval.py          # 평가 진입점 (DAG, Greedy, TC Offline/RTCorrected)
│   ├── tc_train_eval.py # TC 학습·추론·보상/가중치 로직
│   ├── env_core.py      # 실험 설정, build_env, 실행 시뮬레이션
│   ├── scenarios.py     # normal / paper_stress 시나리오
│   ├── presets.py       # preset (balanced 등) — 가드레일·베이스라인
│   ├── baselines.py     # Reactive, Lookahead, ShootingSearch
│   └── figures.py       # 결과 시각화 (선택)
├── baselines/           # DAG 베이스라인 (sustainable_dag)
├── providers/           # 궤적·TLE·phase 등
├── envs/                 # TN-NTN 환경
├── data/                 # TLE 파일
├── results_paper/        # 학습 가중치, manifest, train_history_*.csv
└── results_eval/         # 평가 결과 (exp_paper_runs.csv, exp_paper_summary.csv)
```

---

## 시나리오

- **normal**: 스트레스 완화(easy), 일반적인 가용 환경
- **paper_stress**: 논문용 스트레스 설정(예측 오차·outage 등)

학습/평가 시 `--scenario normal` 또는 `--scenario paper_stress`로 지정합니다.

---

## 학습

- **논문용 기본**: 학습 시드 15개(0~14), 검증 5개(15~19), 에포크 30, `reward_learnable` 권장.
- 시드/에포크를 지정하지 않으면 위 논문용 기본값이 적용됩니다.

### 논문용 (두 시나리오, max_sats=100, 가중치 학습 켜기)

```bash
python -m paper_pkg.train --scenario paper_stress --reward_learnable --max_sats 100
python -m paper_pkg.train --scenario normal --reward_learnable --max_sats 100
```

저장 위치: `results_paper/weights_tc_paper_stress.pth`, `results_paper/weights_tc_normal.pth`

### 디버깅용 (에포크 1, 시드 1개)

```bash
python -m paper_pkg.train --scenario paper_stress --reward_learnable --max_sats 100 --epochs 1 --train_seeds 0 --val_seeds 1
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--scenario` | normal / paper_stress | paper_stress |
| `--reward_learnable` | 보상 가중치 6개를 학습 | False |
| `--max_sats` | 최대 위성 수(노드 수 N) | config 기본 |
| `--epochs` | 에포크 수 | 30 |
| `--train_seeds` | 학습 시드 리스트 | 0..14 |
| `--val_seeds` | 검증 시드 리스트 | 15..19 |
| `--results_dir` | 결과 디렉터리 | results_paper |

---

## 평가

- TC(Offline / RTCorrected), Oracle/Predicted DAG, Reactive Myopic, Lookahead Greedy, ShootingSearch를 한 번에 실행합니다.
- 평가 시드는 학습/검증과 겹치지 않게 20~29(10개) 사용을 권장합니다.

### 시나리오별로 결과 폴더 분리

```bash
python -m paper_pkg.eval --scenario paper_stress --max_sats 100 --seeds 20 21 22 23 24 25 26 27 28 29 --weights results_paper/weights_tc_paper_stress.pth --results_dir results_eval/paper_stress

python -m paper_pkg.eval --scenario normal --max_sats 100 --seeds 20 21 22 23 24 25 26 27 28 29 --weights results_paper/weights_tc_normal.pth --results_dir results_eval/normal
```

생성 파일:  
`results_eval/<시나리오>/exp_paper_runs.csv`, `exp_paper_summary.csv`

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--scenario` | normal / paper_stress | paper_stress |
| `--weights` | TC 체크포인트 경로 | None(베이스라인만) |
| `--max_sats` | 환경 최대 위성 수 | config 기본 |
| `--seeds` | 평가 시드 리스트 | [8,9,10] |
| `--results_dir` | CSV 저장 디렉터리 | results_eval |

---

## Colab에서 한 번에 실행

프로젝트 루트가 작업 디렉터리라고 가정:

```python
# 학습
!python -m paper_pkg.train --scenario paper_stress --reward_learnable --max_sats 100
!python -m paper_pkg.train --scenario normal --reward_learnable --max_sats 100

# 평가 (시나리오별 폴더에 저장)
!python -m paper_pkg.eval --scenario paper_stress --max_sats 100 --seeds 20 21 22 23 24 25 26 27 28 29 --weights results_paper/weights_tc_paper_stress.pth --results_dir results_eval/paper_stress
!python -m paper_pkg.eval --scenario normal --max_sats 100 --seeds 20 21 22 23 24 25 26 27 28 29 --weights results_paper/weights_tc_normal.pth --results_dir results_eval/normal
```

---

## 출력 파일 요약

| 경로 | 내용 |
|------|------|
| `results_paper/weights_tc_<scenario>.pth` | 학습된 TC 가중치(transformer, consistency, 선택 시 reward_weights) |
| `results_paper/train_manifest_<scenario>.json` | 학습 설정·시드·preset 요약 |
| `results_paper/train_history_real_env.csv` | 에포크별 loss, reward |
| `results_eval/<scenario>/exp_paper_runs.csv` | 시드·메서드별 런 결과 |
| `results_eval/<scenario>/exp_paper_summary.csv` | 메서드별 집계(평균 등) |

CSV가 다른 프로그램(예: Excel)에서 열려 있으면 쓰기 실패 시 `*_new.csv`로 저장되며, 터미널에 경로가 출력됩니다.

---

## Preset / 보상 가중치

- **Preset**(balanced 등): 실행 시 가드레일(dwell, hysteresis, pingpong) 및 베이스라인 설정에 사용됩니다. 가중치 학습을 켜면 보상 6개는 **초기값**으로만 쓰이고 학습으로 갱신됩니다.
- **reward_learnable**: 보상 식의 6개 스칼라(w_utility, w_outage, w_switch, w_robustness, w_jitter, w_pingpong)를 `nn.Parameter`로 두고, policy와 함께 최적화합니다. TC 알고리즘(transformer + consistency 구조)은 그대로 유지됩니다.
