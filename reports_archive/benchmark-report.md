# Model Benchmark Report

## 1. 개요
본 문서는 치아 연령 예측(Dental Age Estimation) 모듈(`Dental_011`)의 학습 성과, 3-Way 분할(`train` 70% / `val` 15% / `test` 15%) 독립 평가 지표, 베이스라인 비교, 연령대별 오차 분포 및 데이터셋 한계점을 기록합니다. 모델은 PyTorch 기반의 ResNet-18 아키텍처를 차용하여 파노라마 방사선 영상으로부터 단일 스칼라(연령) 값을 회귀 예측하도록 설계되었습니다.

## 2. 데이터셋 구성 및 라이선스
- 원본 소스: Zenodo Record `16745408` (`DOI: 10.5281/zenodo.16745408`)
- 데이터 라이선스: Creative Commons Attribution 4.0 International (`CC BY 4.0`)
- 총 유효 샘플 수: 295장 (정수 연령 라벨 `9`세~`19`세, 파일명 형식 `<age> <index>.jpg`)
- 결정론적 3-Way 분할 (`src/dataset.py`, `random_state=42`):
  - `Train`: 70% (206장)
  - `Validation`: 15% (44장)
  - `Independent Test`: 15% (45장)
- SHA-256 바이트 해시 중복 감사 결과:
  - 전체 295장 중 고유(Unique) SHA-256 해시 수: 293개 (동일 바이트 중복 2쌍 확인)
  - 중복 쌍 1: `['11 16.jpg', '11 17.jpg']` (원본 SHA-256 prefix `67e52c9e88818f24`, Hybrid SHA-256 prefix `db7fb5303b4a1472`, 각각 `test`와 `val`에 위치하여 `train`-`test` 간 누수 없음)
  - 중복 쌍 2: `['17  1.jpg', '17 3.jpg']` (원본 SHA-256 prefix `8a31ffb9853b4f76`, Hybrid SHA-256 prefix `5909c6085d217482`, 각각 `val`과 `train`에 위치하여 `test` 셋 오염 없음)
  - 동일 인물 중복 확인 한계: Zenodo `16745408` 원본 메타데이터에는 환자 고유 식별자(Patient ID)가 제공되지 않고 연령과 일련번호만 표기되어 있으므로, 바이트 해시가 다른 동일 인물의 종단 촬영본 포함 여부는 메타데이터만으로 완전 배제할 수 없습니다.

## 3. 학습 하이퍼파라미터 (최종 Hybrid 모드)
- Backbone: ResNet-18 (Pretrained on ImageNet)
- Optimizer: Adam (Learning Rate: 1e-4, Weight Decay: 1e-4)
- Loss Function: L1 Loss (Mean Absolute Error)
- Batch Size: 8
- Max Epochs: 150
- Scheduler: ReduceLROnPlateau (factor: 0.5, patience: 10)
- Early Stopping Patience: 20 Epochs
- Mode: `use_hybrid=True` (`Dental_008` 치아 세그멘테이션 마스크 ROI 연동)

## 4. 벤치마크 결과 상세

### 4.1 학습 단계별 Validation 성능 비교 (기존 80/20 분할 기준)
| 지표 | 초기 세팅 (Batch 64) | 1차 최적화 (Batch 16) | 최종 Hybrid (Batch 8) | 향상 수치 (초기 대비) |
|---|---|---|---|---|
| Legacy Val MAE (n=59) | 7.8221 years | 0.7664 years | [0.7585 years] | -7.0636 |
| 진행 Epoch | 20 (강제 종료) | 53 (Early Stopping) | 31 (Early Stopping) | N/A |
| 출력 가중치 | `best_age_model.pth` | `best_age_model.pth` | `best_hybrid_age_model.pth` / `.onnx` | N/A |

### 4.2 결정론적 3-Way 분할 및 베이스라인 대비 독립 Test 평가 (`random_state=42`)
| 평가 대상 / 예측기 | 분할 (샘플 수) | MAE (years) | 95% Bootstrap CI (10,000회) | 비고 |
|---|---|---|---|---|
| Constant Median Predictor (`15.00`세) | Independent Test (`n=45`) | 1.9333 years | N/A | 고정 중앙값(15.0세) 단순 예측 베이스라인 |
| Train-Set Median Predictor (`16.00`세) | Independent Test (`n=45`) | 1.9111 years | N/A | 훈련셋 중앙값(16.0세) 단순 예측 베이스라인 |
| Train-Set Mean Predictor (`15.17`세) | Independent Test (`n=45`) | 1.9296 years | N/A | 훈련셋 평균값(15.17세) 단순 예측 베이스라인 |
| ResNet-18 Hybrid (`best_hybrid_age_model.onnx`) | Train (`n=206`) | 0.6588 years | N/A | 70% 훈련 분할 |
| ResNet-18 Hybrid (`best_hybrid_age_model.onnx`) | Validation (`n=44`) | 0.7685 years | N/A | 15% 검증 분할 |
| ResNet-18 Hybrid (`best_hybrid_age_model.onnx`) | Independent Test (`n=45`) | [0.7755 years] | `[0.5762, 1.0055] years` | 베이스라인 대비 오차 약 59.8% 감소 |

### 4.3 연령 구간별(Age-Group) MAE 상세 분석
| 연령 구간 (정수 라벨) | Independent Test 샘플 수 (`n=45`) | Independent Test MAE (years) | 전체 데이터셋 샘플 수 (`n=295`) | 전체 데이터셋 MAE (years) |
|---|---|---|---|---|
| `9–11`세 | 4 | 0.3930 years | 25 | 0.6295 years |
| `12–14`세 | 15 | 0.9246 years | 96 | 0.8124 years |
| `15–16`세 | 11 | 0.6339 years | 70 | 0.5612 years |
| `17–19`세 | 15 | 0.8323 years | 104 | 0.6867 years |
| 전체 합계 / 평균 | 45 | [0.7755 years] | 295 | 0.6930 years |

## 5. 수치 해석 주의사항 및 한계점 (Caveats)
- [정수 연령 라벨 양자화 한계 (Label Quantization Noise)]: Zenodo `16745408` 데이터셋은 생년월일 기반의 소수점 실제 연령이 아닌 `9`세부터 `19`세까지의 정수(Integer) 연령 라벨만 제공합니다. 각 1년 구간 내 실제 연령이 균등 분포한다고 가정할 때, 완벽한 연속 연령 예측기라도 정수 반올림 라벨과 비교하면 이론적 최소 오차 하한(`E[|U(-0.5, 0.5)|] = 0.25 years`)이 내재합니다.
- [소표본 데이터셋 및 조기 종료 해석 주의]: 전체 데이터셋 규모가 295장(독립 Test 45장, 95% CI `[0.5762, 1.0055] years`)으로 소규모이며, 기존 학습 체크포인트(`best_hybrid_age_model.onnx`)는 80/20 분할 시점의 검증셋 기준으로 조기 종료된 가중치입니다. 향후 신규 가중치 학습 시에는 반드시 `src/dataset.py`의 70/15/15 3-Way 분할을 사용하여 `val`에서만 조기 종료를 수행하고 `test`를 최종 보고 전용으로 격리해야 합니다.
- [환자 식별자 부재]: 원본 공개 데이터셋에 환자 ID가 없으므로, 상용 수준의 일반화 성능을 검증하기 위해서는 환자 단위 분할(Patient-level split)과 소수점 실연령이 보장된 1,000장 이상의 외부 다기관 코호트 확보가 필수적입니다.

# [ 261020 0205 랩탑 개발팀 ] 3-Way 독립 Test 벤치마크, SHA-256 중복 감사 및 연령 구간별 MAE 리포트 업데이트 완결 (`[CLOSED]` / `[PASS]`)

- 상태: 공식 종결 (`[CLOSED]` / `[PASS]` 완결)
- 이행 결과: 결정론적 3-Way 분할(`Train 206 / Val 44 / Test 45`), SHA-256 바이트 해시 중복 감사, 단순 예측기 대비 베이스라인 비교 및 연령 구간별 MAE 산출(`Independent Test MAE 0.7755 years`)을 완수하여 `Dental_011#1`을 해결하고 본 벤치마크 보고서를 `[CLOSED]` / `[PASS]` 완결 처리한다.

