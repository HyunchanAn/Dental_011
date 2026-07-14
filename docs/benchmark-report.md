# Model Benchmark Report

## 1. 개요
본 문서는 치아 연령 예측(Dental Age Estimation) 모듈(`Dental_011`)의 학습 성과와 벤치마크 지표를 기록하기 위해 작성되었습니다. 모델은 PyTorch 기반의 ResNet-18 아키텍처를 차용하여 단일 스칼라(연령) 값을 회귀 예측하도록 설계되었습니다.

## 2. 데이터셋 구성
- 원본 소스: Zenodo (Record ID: 16745408)
- 총 유효 샘플 수: 295장
- 훈련/검증 분할: 80% (236장) / 20% (59장)

## 3. 학습 하이퍼파라미터 (최종 Hybrid 모드)
- Backborn: ResNet-18 (Pretrained on ImageNet)
- Optimizer: Adam (Learning Rate: 1e-4, Weight Decay: 1e-4)
- Loss Function: L1 Loss (Mean Absolute Error)
- Batch Size: 8
- Max Epochs: 150
- Scheduler: ReduceLROnPlateau (factor: 0.5, patience: 10)
- Early Stopping Patience: 20 Epochs
- Mode: `use_hybrid=True`

## 4. 벤치마크 결과 상세
학습 전후의 성능 격차와 최종 지표는 다음과 같습니다.

### 성능 지표 요약표
| 지표 | 초기 세팅 (Batch 64) | 1차 최적화 (Batch 16) | 최종 Hybrid (Batch 8) | 향상 수치 (초기 대비) |
|---|---|---|---|---|
| Val MAE | 7.8221 years | 0.7664 years | **0.7585 years** | -7.0636 |
| 진행 Epoch | 20 (강제 종료) | 53 (Early Stopping) | 31 (Early Stopping) | N/A |
| 출력 가중치 | best_age_model.pth | best_age_model.pth | best_hybrid_age_model.pth | N/A |

### 학습 추이 분석
- **초기 세팅:** 절대적인 최적화 스텝(가중치 업데이트) 수 부족으로 모델이 수렴하지 못하여 7.8년이라는 매우 높은 오차가 발생했습니다.
- **1차 최적화:** 배치 사이즈 축소 및 Epoch 연장을 통해 옵티마이저가 충분히 수렴하도록 유도한 결과 Val MAE 0.76년을 달성했습니다.
- **최종 Hybrid 모드:** `use_hybrid=True` 및 Batch Size를 8로 더욱 축소하고 최대 Epoch를 150으로 해제하여 스케줄러가 개입할 수 있는 통제된 환경을 조성했습니다. 결과적으로 Epoch 31에서 모델이 최저점을 탐색하여 **0.7585년**이라는 가장 낮은 오차율(Best Val MAE)을 갱신하고 안전하게 조기 종료되었습니다.

## 5. 한계점 및 향후 과제
현재의 0.75년 오차는 295장이라는 극히 제한된 데이터 내에서 하이퍼파라미터 튜닝만으로 달성한 최선의 지표입니다. 딥러닝 모델의 규모를 고려할 때 여전히 과적합(Overfitting) 위험에 노출되어 있습니다. 상용 수준의 견고한 연령 예측 정확도(예: 6개월 미만 오차)를 담보하기 위해서는 1,000장 이상의 추가 데이터 확보 및 재학습이 필수적으로 요구됩니다.
