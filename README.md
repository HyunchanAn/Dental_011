![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Backend](https://img.shields.io/badge/Backend-ResNet18%20ONNX-red) ![UI](https://img.shields.io/badge/UI-Streamlit-orange) ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github)

# Dental_011 (Dental Age Estimation)

## 개요
이 모듈은 파노라마 방사선 사진을 입력받아 환자의 치아 연령(Dental Age)을 추정하는 딥러닝(ResNet-18 Regression) 기반 분석 시스템입니다. 임상(소아치과, 교정) 및 법의학 분야의 연령 추정 수요를 충족시키며, `Dental_008` 모듈(치아 분할)과 연동하여 치아 ROI 마스킹을 수행하는 Hybrid 파이프라인으로 구성되어 있습니다.

## 디렉토리 구조
- `data/hybrid_dataset/`: Zenodo Record `16745408` (`CC BY 4.0`) 기반 치아 ROI 하이브리드 마스킹 데이터셋 (295장, 결정론적 3-Way 분할 벤치마크 재현을 위해 추적 유지; 원본 압축 아카이브 `*.zip` 및 중복 스테이징 폴더 `data/dataset ++/`는 `.gitignore` 처리)
- `src/`
  - `dataset.py`: 데이터 파싱, SHA-256 중복 점검 및 결정론적 3-Way 분할(`train` 70% / `val` 15% / `test` 15%) PyTorch DataLoader 제공 로직
  - `model.py`: 연령 회귀를 위한 ResNet-18 백본 아키텍처
  - `train.py`: L1 Loss 기반 End-to-End 모델 학습 스크립트
  - `test.py`: 독립 테스트셋 대상 MAE, 베이스라인 비교, 95% 부트스트랩 신뢰구간 및 연령대별 MAE 평가 스크립트
  - `weights/`: 학습 완료된 모델 가중치 저장 공간 (`best_hybrid_age_model.pth`, `best_hybrid_age_model.onnx`)
- `docs/benchmark-report.md`: 상세 벤치마크 결과 및 수치 해석 주의사항 리포트

## 데이터셋 출처 및 라이선스
- 데이터셋 출처: Zenodo Record `16745408` (`DOI: 10.5281/zenodo.16745408`)
- 데이터셋 라이선스: Creative Commons Attribution 4.0 International (`CC BY 4.0`)
- 소스 코드 라이선스: MIT License (`LICENSE`)
- 데이터 구성: 총 295장 (`9`세~`19`세 정수 연령 라벨)
  - SHA-256 바이트 해시 감사: 전체 295장 중 고유 해시 293개 (`['11 16.jpg', '11 17.jpg']` 및 `['17  1.jpg', '17 3.jpg']` 2쌍 중복 확인, `train`과 `test` 간 교차 누수 없음)
  - 결정론적 3-Way 분할 (`random_state=42`): `Train` 206장 (70%) / `Val` 44장 (15%) / `Independent Test` 45장 (15%)

## 벤치마크 요약 및 해석 주의사항

### 1. 베이스라인 대비 모델 성능 (`random_state=42`)
| 예측기 / 분할 | 평가 샘플 수 | MAE (years) | 95% Bootstrap CI | 비고 |
|---|---|---|---|---|
| Constant Median Predictor (`15.00`세) | Independent Test (`n=45`) | 1.9333 years | N/A | 단순 중앙값 예측 베이스라인 |
| Train-Set Median Predictor (`16.00`세) | Independent Test (`n=45`) | 1.9111 years | N/A | 훈련셋 중앙값 예측 베이스라인 |
| Train-Set Mean Predictor (`15.17`세) | Independent Test (`n=45`) | 1.9296 years | N/A | 훈련셋 평균값 예측 베이스라인 |
| ResNet-18 Hybrid (`best_hybrid_age_model.onnx`) | Validation (`n=44`) | 0.7685 years | N/A | 3-Way 검증 분할 (기존 80/20 Val `n=59` 기준 0.7585 years) |
| ResNet-18 Hybrid (`best_hybrid_age_model.onnx`) | Independent Test (`n=45`) | [0.7755 years] | `[0.5762, 1.0055] years` | 베이스라인 대비 오차 약 59.8% 감소 |

### 2. 연령 구간별(Age-Group) MAE 분포
| 연령 구간 | Independent Test (`n=45`) | Independent Test MAE | 전체 데이터셋 (`n=295`) | 전체 데이터셋 MAE |
|---|---|---|---|---|
| `9–11`세 | `n=4` | 0.3930 years | `n=25` | 0.6295 years |
| `12–14`세 | `n=15` | 0.9246 years | `n=96` | 0.8124 years |
| `15–16`세 | `n=11` | 0.6339 years | `n=70` | 0.5612 years |
| `17–19`세 | `n=15` | 0.8323 years | `n=104` | 0.6867 years |

### 3. 수치 해석 주의사항 (Caveats)
- [정수 라벨 양자화 오차]: 원본 데이터셋이 `9`~`19`세의 정수 연령 라벨만 제공하므로, 1년 구간 내 균등 분포 가정 시 양자화 노이즈로 인한 이론적 최소 오차 하한(`~0.25 years`)이 존재합니다.
- [환자 식별자 미제공]: Zenodo `16745408`은 파일명에 환자 고유 ID를 포함하지 않으므로, 바이트 해시 중복(2쌍) 외에 동일 인물의 타 시점 촬영본 포함 여부를 메타데이터만으로 검증할 수 없습니다.
- [소표본 한계]: 295장(독립 Test 45장) 규모의 소표본 데이터셋에서 도출된 수치이므로, 임상적 일반화를 위해서는 환자 단위 분할이 적용된 대규모 다기관 데이터셋 검증이 필요합니다.

## 설치 및 실행 방법

1. [환경 설정 및 의존성 설치]
   ```bash
   pip install torch torchvision numpy scikit-learn matplotlib opencv-python Pillow tqdm onnxruntime
   ```

2. [모델 학습]
   본 레포지토리는 워크스테이션 환경(예: Ryzen 9 9900X, RTX 5080 16GB, 64GB RAM)에서의 고속 학습을 상정하여 작성되었습니다.
   ```bash
   cd src
   python train.py
   ```

3. [모델 검증 및 독립 테스트 평가]
   ```bash
   cd src
   python test.py
   ```

## 주요 특징 및 파이프라인 통합
본 레포지토리는 `Dental_Panoramic_Reader` 메인 시스템 내부에서 `AgePredictorWrapper`를 통해 외부 인터페이스 역할을 수행하도록 통합되어 있습니다.

## 연구용 실행 주의사항
- 기본 실행 주소: 127.0.0.1 (로컬 전용)
- 외부 공개/상용 배포 금지
- API 키·토큰은 환경변수로만 관리
- 오픈 데이터셋 외 민감 의료데이터 사용 금지
- 외부 접근이 필요하면 ALLOW_EXTERNAL_ACCESS=true 로만 명시적 허용

