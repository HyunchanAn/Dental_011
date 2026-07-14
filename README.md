![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Backend](https://img.shields.io/badge/Backend-YOLOv8-red) ![UI](https://img.shields.io/badge/UI-Streamlit-orange) ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github)

# Dental_011 (Dental Age Estimation)

## 개요
이 모듈은 파노라마 방사선 사진을 입력받아 환자의 치아 연령(Dental Age)을 추정하는 딥러닝(ResNet18 Regression) 기반 분석 시스템입니다. 임상(소아치과, 교정) 및 법의학 분야의 연령 추정 수요를 충족시키며, 기존 008 모듈(치아 분할)과 연동하여 정밀성을 높이는 파이프라인으로 구성될 수 있습니다.

## 디렉토리 구조
- `data/`: Zenodo 등 외부 소스로부터 획득한 치아 방사선 데이터 (Gitignore 처리)
- `src/`
  - `dataset.py`: 데이터 파싱 및 PyTorch DataLoader 제공 로직
  - `model.py`: 연령 회귀를 위한 ResNet18 백본 아키텍처
  - `train.py`: MSE/L1 Loss 기반 End-to-End 모델 학습 스크립트
  - `test.py`: 검증 세트 대상 MAE(Mean Absolute Error) 평가 스크립트
  - `weights/`: 학습 완료된 모델 가중치 저장 공간

## 설치 및 실행 방법

1. **환경 설정 및 의존성 설치**
   ```bash
   pip install torch torchvision numpy scikit-learn matplotlib opencv-python Pillow tqdm
   ```

2. **모델 학습**
   본 레포지토리는 워크스테이션 환경(예: Ryzen 9 9900X, RTX 5080 16GB, 64GB RAM)에서의 고속 학습을 상정하여 작성되었습니다.
   ```bash
   cd src
   python train.py
   ```

3. **모델 검증 (성능 평가)**
   ```bash
   python test.py
   ```

## 주요 특징 및 파이프라인 통합
본 레포지토리는 `Dental_Panoramic_Reader` 메인 시스템 내부에서 `AgePredictorWrapper`를 통해 외부 인터페이스 역할을 수행하도록 통합되어 있습니다. 
단독(End-to-End Regression) 모델로도 동작할 수 있으며, 향후 008 모듈과 결합된 하이브리드 연산으로 확장될 예정입니다.
