# Spaceship Titanic - 모델 설명 및 성능 비교

## 목차
1. [개별 모델 설명](#1-개별-모델-설명)
2. [앙상블 모델 설명](#2-앙상블-모델-설명)
3. [Feature Engineering](#3-feature-engineering)
4. [성능 비교](#4-성능-비교)
5. [결론 및 인사이트](#5-결론-및-인사이트)

---

## 1. 개별 모델 설명

### 1.1 Gradient Boosting 계열

#### LightGBM (Light Gradient Boosting Machine)
- **개발**: Microsoft
- **특징**:
  - Leaf-wise 트리 성장 방식 (다른 부스팅은 Level-wise)
  - Histogram 기반 알고리즘으로 빠른 학습 속도
  - 대용량 데이터에서 메모리 효율적
  - Categorical feature 직접 지원
- **주요 하이퍼파라미터**:
  - `n_estimators`: 부스팅 반복 횟수
  - `learning_rate`: 학습률
  - `num_leaves`: 리프 노드 수 (복잡도 제어)
  - `max_depth`: 트리 최대 깊이
- **Optuna 최적 파라미터**:
  ```
  n_estimators: 510
  learning_rate: 0.0148
  max_depth: 6
  num_leaves: 37
  ```

#### XGBoost (eXtreme Gradient Boosting)
- **개발**: University of Washington
- **특징**:
  - Level-wise 트리 성장 방식
  - L1, L2 정규화 내장
  - Sparsity-aware 알고리즘 (결측치 자동 처리)
  - 병렬 처리 지원
- **주요 하이퍼파라미터**:
  - `n_estimators`: 부스팅 반복 횟수
  - `learning_rate`: 학습률
  - `max_depth`: 트리 최대 깊이
  - `min_child_weight`: 리프 노드 최소 가중치
  - `gamma`: 분할 최소 손실 감소량
- **Optuna 최적 파라미터**:
  ```
  n_estimators: 140
  learning_rate: 0.0252
  max_depth: 10
  min_child_weight: 7
  ```

#### CatBoost (Categorical Boosting)
- **개발**: Yandex
- **특징**:
  - Ordered Boosting으로 과적합 방지
  - Categorical feature 최적 처리 (Target Encoding 내장)
  - 대칭 트리 구조로 빠른 예측
  - 결측치 자동 처리
- **주요 하이퍼파라미터**:
  - `iterations`: 부스팅 반복 횟수
  - `learning_rate`: 학습률
  - `depth`: 트리 깊이
  - `l2_leaf_reg`: L2 정규화
- **Optuna 최적 파라미터**:
  ```
  iterations: 822
  learning_rate: 0.0154
  depth: 5
  l2_leaf_reg: 0.0018
  ```

#### HistGradientBoosting (Histogram-based Gradient Boosting)
- **개발**: scikit-learn (LightGBM 영감)
- **특징**:
  - Histogram 기반으로 빠른 학습
  - Native 결측치 처리
  - scikit-learn API와 완벽 호환
  - 메모리 효율적
- **주요 하이퍼파라미터**:
  - `max_iter`: 최대 반복 횟수
  - `learning_rate`: 학습률
  - `max_depth`: 트리 최대 깊이
  - `l2_regularization`: L2 정규화

---

### 1.2 Tree Ensemble 계열

#### RandomForest
- **알고리즘**: Bagging + Random Feature Selection
- **특징**:
  - 여러 결정 트리를 독립적으로 학습 (병렬)
  - Bootstrap Sampling으로 다양성 확보
  - 각 분할에서 랜덤 feature subset 사용
  - 과적합에 강함
- **주요 하이퍼파라미터**:
  - `n_estimators`: 트리 개수
  - `max_depth`: 트리 최대 깊이
  - `min_samples_split`: 분할 최소 샘플 수
  - `max_features`: 분할 시 고려할 feature 수

#### ExtraTrees (Extremely Randomized Trees)
- **알고리즘**: RandomForest보다 더 랜덤한 분할
- **특징**:
  - 분할 임계값도 랜덤 선택
  - RandomForest보다 빠른 학습
  - 더 낮은 분산, 약간 높은 편향
  - 과적합에 더 강함
- **RandomForest와의 차이점**:
  | 항목 | RandomForest | ExtraTrees |
  |------|-------------|------------|
  | 분할 임계값 | 최적값 탐색 | 랜덤 선택 |
  | 샘플링 | Bootstrap | 전체 데이터 |
  | 학습 속도 | 보통 | 빠름 |
  | 분산 | 높음 | 낮음 |

---

## 2. 앙상블 모델 설명

### 2.1 Simple Average (단순 평균)
```
최종 예측 = (모델1 + 모델2 + ... + 모델N) / N
```
- **특징**:
  - 가장 간단한 앙상블 방법
  - 모든 모델에 동일한 가중치 부여
  - 구현이 간단하고 안정적
- **장점**: 간단함, 해석 용이
- **단점**: 성능이 낮은 모델도 동일한 영향력

### 2.2 Optuna Weighted Ensemble (가중 평균)
```
최종 예측 = w1*모델1 + w2*모델2 + ... + wN*모델N
(w1 + w2 + ... + wN = 1)
```
- **특징**:
  - Optuna로 최적 가중치 탐색
  - OOF(Out-of-Fold) 예측으로 가중치 최적화
  - 성능 좋은 모델에 높은 가중치 부여
- **최적화된 가중치**:
  | 모델 | 가중치 |
  |------|--------|
  | CatBoost | 0.3004 |
  | LightGBM | 0.2707 |
  | XGBoost | 0.2010 |
  | HistGradientBoosting | 0.1689 |
  | ExtraTrees | 0.0337 |
  | RandomForest | 0.0254 |

### 2.3 Soft Voting
```
최종 예측 = argmax(평균 확률)
```
- **특징**:
  - 각 모델의 클래스별 확률을 평균
  - 확률 기반으로 더 부드러운 결정
  - Hard Voting보다 일반적으로 좋은 성능
- **Hard vs Soft Voting**:
  | 방식 | 결합 방법 | 특징 |
  |------|----------|------|
  | Hard | 다수결 투표 | 이산적 결정 |
  | Soft | 확률 평균 | 연속적 결정 |

### 2.4 Stacking (스태킹)
```
Level 0: 기본 모델들의 예측
Level 1: 메타 학습기가 기본 모델 예측을 입력으로 학습
```
- **구조**:
  ```
  [원본 데이터] → [6개 기본 모델] → [예측값] → [LightGBM 메타 학습기] → [최종 예측]
  ```
- **특징**:
  - 2단계 학습 구조
  - 메타 학습기가 모델 간 상관관계 학습
  - 과적합 방지를 위해 CV 사용
- **본 프로젝트 구성**:
  - **기본 모델 (Level 0)**: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting
  - **메타 학습기 (Level 1)**: LightGBM (n_estimators=100, max_depth=3)

### 2.5 Threshold 최적화
- **개념**: 기본 분류 임계값 0.5 대신 최적의 임계값 탐색
- **방법**: OOF 예측을 활용하여 0.40 ~ 0.60 범위에서 0.01 단위로 탐색
- **효과**: 일부 모델에서 +0.0005 ~ +0.0006 성능 향상

---

## 3. Feature Engineering

### 3.1 Target Encoding (타겟 인코딩)
- **개념**: 범주형 변수를 해당 범주의 타겟 평균값으로 인코딩
- **Label Encoding과의 차이**:
  | 방식 | 인코딩 값 | 특징 |
  |------|----------|------|
  | Label Encoding | 0, 1, 2, ... | 순서 정보만 부여, 타겟과 무관 |
  | Target Encoding | 타겟 평균값 | 타겟과의 관계 반영, 정보량 높음 |

- **Smoothing 적용**: 샘플 수가 적은 범주의 과적합 방지
  ```
  smoothed_mean = (n * category_mean + m * global_mean) / (n + m)
  ```
  - `n`: 해당 범주의 샘플 수
  - `m`: smoothing 파라미터 (본 프로젝트: 10)

- **적용된 변수**:
  - HomePlanet, Destination, Deck, Side, AgeGroup, DeckSide, Route

### 3.2 시도했으나 제외된 피처

#### 그룹 통계 피처
- **시도한 피처**: Group_MeanExpense, Group_MaxExpense, Group_TotalExpense 등 15개
- **결과**: 성능 하락 (0.8194 → 0.8182)
- **원인 추정**: 과적합 또는 노이즈 증가
- **결론**: 제외하고 Target Encoding만 유지

---

## 4. 성능 비교

### 4.1 전체 모델 성능 순위 (Target Encoding 적용 후)

| 순위 | 모델 | OOF Accuracy | 최적 Threshold | 비고 |
|------|------|-------------|----------------|------|
| 1 | **Optuna Ensemble** | **0.8194** | 0.50 | 최고 성능 |
| 2 | CatBoost (Tuned) | 0.8190 | 0.50 | - |
| 3 | LightGBM (Tuned) | 0.8180 | 0.49 | Threshold 최적화 적용 |
| 4 | HistGradientBoosting | 0.8178 | 0.48 | Threshold 최적화 적용 |
| 5 | Soft Voting (6 Models) | 0.8173 | - | - |
| 6 | Simple Average (6 Models) | 0.8173 | - | - |
| 7 | XGBoost (Tuned) | 0.8155 | 0.51 | Threshold 최적화 적용 |
| 8 | Stacking (LGB Meta) | 0.8142 | - | - |
| 9 | RandomForest | 0.8136 | 0.50 | - |
| 10 | ExtraTrees | 0.8103 | 0.50 | - |

### 4.2 모델 유형별 비교

#### Gradient Boosting 계열
```
CatBoost (0.8190) > LightGBM (0.8180) > HistGradientBoosting (0.8178) > XGBoost (0.8155)
```

#### Tree Ensemble 계열
```
RandomForest (0.8136) > ExtraTrees (0.8103)
```

#### 앙상블 기법
```
Optuna Ensemble (0.8194) > Voting (0.8173) = Simple Avg (0.8173) > Stacking (0.8142)
```

### 4.3 Threshold 최적화 결과

| 모델 | 기본 (0.5) | 최적 Threshold | 개선 |
|------|-----------|----------------|------|
| LightGBM | 0.8174 | 0.8180 (0.49) | +0.0006 |
| HistGradientBoosting | 0.8172 | 0.8178 (0.48) | +0.0006 |
| XGBoost | 0.8150 | 0.8155 (0.51) | +0.0005 |
| CatBoost | 0.8190 | 0.8190 (0.50) | - |
| Optuna Ensemble | 0.8194 | 0.8194 (0.50) | - |

### 4.4 성능 분석

#### 개별 모델 분석
- **HistGradientBoosting**: 가장 높은 단일 모델 성능 (0.8190)
- **CatBoost**: 두 번째로 높은 성능, categorical feature 처리에 강점
- **LightGBM/XGBoost**: 안정적인 성능, 낮은 CV 표준편차
- **RandomForest/ExtraTrees**: 상대적으로 낮은 성능, 높은 분산

#### 앙상블 분석
- **Optuna Ensemble**: 단일 최고 모델과 동일한 성능, 최적 가중치 자동 탐색
- **Voting/Simple Avg**: 동일 성능, 구현 간단
- **Stacking**: 예상보다 낮은 성능 (과적합 가능성)

---

## 5. 결론 및 인사이트

### 5.1 주요 발견사항

1. **Optuna Ensemble의 최고 성능**
   - Target Encoding 적용 후 0.8194 달성
   - 6개 모델의 최적 가중치 조합으로 안정적인 결과

2. **Target Encoding의 효과**
   - Label Encoding 대비 성능 향상
   - Smoothing을 통한 과적합 방지

3. **CatBoost의 우수성**
   - Spaceship Titanic 데이터의 categorical feature가 많아 강점 발휘
   - Ordered Boosting으로 과적합 방지 효과

4. **Threshold 최적화**
   - 일부 모델에서 소폭 성능 향상 (+0.0005 ~ +0.0006)
   - CatBoost, Optuna Ensemble은 기본 0.5가 최적

5. **그룹 통계 피처의 한계**
   - 15개 그룹 통계 피처 시도 → 성능 하락
   - 과적합 또는 노이즈 증가 원인으로 제외

### 5.2 추천 제출 파일

| 우선순위 | 파일명 | OOF Score | 사유 |
|---------|--------|-----------|------|
| 1 | `submission_optuna_ensemble.csv` | 0.8194 | 최고 성능 |
| 2 | `submission_catboost_tuned.csv` | 0.8190 | 두 번째 높은 성능 |
| 3 | `submission_lightgbm_tuned_thresh0.49.csv` | 0.8180 | Threshold 최적화 적용 |

### 5.3 향후 개선 방향

1. **Pseudo Labeling**
   - 높은 확신도의 test 예측을 학습에 활용

2. **Feature Selection**
   - 중요도가 낮은 피처 제거로 노이즈 감소

3. **Blending 전략 다양화**
   - Rank Average
   - Geometric Mean

4. **Neural Network 추가**
   - TabNet, MLP 등 딥러닝 모델 추가

---

## 부록: 하이퍼파라미터 튜닝 결과

### Optuna 튜닝 시도 횟수
| 모델 | 시도 횟수 | 소요 시간 |
|------|----------|----------|
| LightGBM | 50 trials | ~3분 26초 |
| XGBoost | 50 trials | ~1분 18초 |
| CatBoost | 50 trials | ~4분 29초 |
| RandomForest | 30 trials | ~4분 33초 |
| ExtraTrees | 30 trials | ~56초 |
| HistGradientBoosting | 30 trials | ~3분 36초 |
| Ensemble Weights | 100 trials | <1초 |

**총 튜닝 시간**: 약 18분
