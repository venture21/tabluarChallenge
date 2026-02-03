# Spaceship Titanic - Kaggle Competition

## 1. 문제 개요 (Overview)

### 배경
서기 2912년, 우주선 **'스페이스 타이타닉(Spaceship Titanic)'**호가 이주민 약 13,000명을 태우고 가까운 항성계로 향하던 중, **시공간 이상 현상(Spacetime Anomaly)**과 충돌하는 사고가 발생합니다.

### 사건
이 충돌로 인해 승객의 약 절반이 다른 차원으로 **'전송(Transported)'**되어 사라져 버렸습니다.

### 목표
우주선의 손상된 컴퓨터 시스템에서 복구한 승객 데이터를 기반으로, **어떤 승객이 다른 차원으로 전송되었는지(Transported)** 예측하는 것입니다.

### 문제 유형
**이진 분류 (Binary Classification)**
- `True`: 전송됨
- `False`: 전송되지 않음

---

## 2. 데이터 파일 구조

| 파일명 | 설명 |
|--------|------|
| `train.csv` | 학습용 데이터 (Transported 포함) |
| `test.csv` | 테스트용 데이터 (Transported 없음 - 예측 대상) |
| `sample_submission.csv` | 제출 양식 예시 |

---

## 3. Feature 설명 (Data Description)

### 3.1 식별 정보

| Feature | 설명 | 비고 |
|---------|------|------|
| **PassengerId** | 승객 고유 ID | `gggg_pp` 형태 |

- `gggg`: 승객이 속한 **그룹 번호** (같은 그룹 = 가족 또는 동행자)
- `pp`: 그룹 내 개인 번호

### 3.2 출발/도착 정보

| Feature | 설명 | 값 예시 |
|---------|------|---------|
| **HomePlanet** | 출신 행성 (출발지) | Earth, Europa, Mars |
| **Destination** | 목적지 행성 | TRAPPIST-1e, PSO J318.5-22, 55 Cancri e |

### 3.3 탑승 상태 정보

| Feature | 설명 | 값 |
|---------|------|-----|
| **CryoSleep** | 냉동 수면 여부 | True / False |
| **VIP** | VIP 서비스 이용 여부 | True / False |

> **주의**: CryoSleep이 `True`인 승객은 객실에 갇혀 있으므로 **모든 편의시설 이용 금액이 0**일 가능성이 높습니다.

### 3.4 객실 정보

| Feature | 설명 | 형식 |
|---------|------|------|
| **Cabin** | 객실 번호 | `deck/num/side` (예: F/123/P) |

**Cabin 분해 활용:**
- **Deck**: 객실 층 (A, B, C, D, E, F, G, T)
- **Num**: 객실 번호
- **Side**: 선체 위치
  - `P` (Port): 좌현
  - `S` (Starboard): 우현

### 3.5 개인 정보

| Feature | 설명 |
|---------|------|
| **Age** | 승객 나이 |
| **Name** | 승객 이름 (성 이름) |

> **Tip**: Name에서 성(Last Name)을 추출하면 가족 관계를 파악하는 데 활용할 수 있습니다.

### 3.6 편의시설 이용 금액 (Luxury Amenities)

| Feature | 설명 |
|---------|------|
| **RoomService** | 룸서비스 이용 금액 |
| **FoodCourt** | 푸드코트 이용 금액 |
| **ShoppingMall** | 쇼핑몰 이용 금액 |
| **Spa** | 스파 이용 금액 |
| **VRDeck** | VR 데크 이용 금액 |

> **Feature Engineering Tip**: 5개 항목의 합계를 `TotalExpenditure`로 만들어 새로운 변수로 활용 가능

### 3.7 타겟 변수 (Target)

| Feature | 설명 | 값 |
|---------|------|-----|
| **Transported** | 다른 차원으로 전송 여부 | True / False |

---

## 4. 주요 Feature Engineering 아이디어

### 4.1 PassengerId 활용
```python
# 그룹 번호와 그룹 내 번호 분리
df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
```

### 4.2 Cabin 분해
```python
# Cabin을 Deck, Num, Side로 분리
df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
```

### 4.3 총 지출액 계산
```python
expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalExpenditure'] = df[expense_cols].sum(axis=1)
```

### 4.4 Name에서 성(Last Name) 추출
```python
df['LastName'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notna(x) else None)
df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count')
```

### 4.5 나이 그룹 생성
```python
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
```

---

## 5. 데이터 특성 및 분석 포인트

### 5.1 결측치 (Missing Values)
- 대부분의 컬럼에 결측치가 존재함
- CryoSleep과 편의시설 지출 간의 관계를 활용한 결측치 처리 가능

### 5.2 주요 가설
1. **CryoSleep 승객**: 동면 중이므로 편의시설 이용 불가 → 전송 확률에 영향?
2. **그룹/가족 단위**: 같은 그룹은 함께 전송되었을 가능성
3. **객실 위치(Deck, Side)**: 시공간 이상 현상의 위치에 따른 영향
4. **지출 패턴**: 활동적인 승객 vs 비활동적인 승객의 전송 확률 차이

### 5.3 불균형 여부
- 타겟 변수(Transported)는 약 50:50으로 비교적 균형 잡힌 분포

---

## 6. 평가 지표 (Evaluation Metric)

**Classification Accuracy**
```
Accuracy = (정확히 예측한 샘플 수) / (전체 샘플 수)
```

---

## 7. 제출 형식 (Submission Format)

```csv
PassengerId,Transported
0013_01,False
0018_01,False
0019_01,False
...
```

---

## 8. 참고 자료

- [Kaggle Competition Page](https://www.kaggle.com/competitions/spaceship-titanic)
- [Kaggle Notebooks](https://www.kaggle.com/competitions/spaceship-titanic/code)
- [Discussion Forum](https://www.kaggle.com/competitions/spaceship-titanic/discussion)

---

## 9. 적용된 기법 및 최종 결과

### 9.1 적용된 기법

| 기법 | 설명 | 효과 |
|------|------|------|
| **Optuna 튜닝** | 6개 모델 하이퍼파라미터 최적화 | 개별 모델 성능 향상 |
| **Target Encoding** | 범주형 변수를 타겟 평균값으로 인코딩 | +0.004 향상 |
| **Threshold 최적화** | 0.40~0.60 범위에서 최적 임계값 탐색 | +0.0006 향상 |
| **앙상블** | Weighted, Stacking, Voting, Average | 안정적 성능 |

### 9.2 최종 모델 성능

| 순위 | 모델 | OOF Accuracy |
|------|------|-------------|
| 1 | **Optuna Ensemble** | **0.8194** |
| 2 | CatBoost (Tuned) | 0.8190 |
| 3 | LightGBM (Tuned) | 0.8180 |
| 4 | Soft Voting | 0.8173 |
| 5 | HistGradientBoosting | 0.8178 |

### 9.3 추천 제출 파일

1. `submission_optuna_ensemble.csv` - 최고 성능 (0.8194)
2. `submission_catboost_tuned.csv` - 두 번째 (0.8190)
3. `submission_lightgbm_tuned_thresh0.49.csv` - Threshold 최적화 적용

---

## 10. 권장 접근 방법

1. **EDA (탐색적 데이터 분석)**: 각 변수의 분포와 타겟과의 관계 파악
2. **결측치 처리**: 변수 간 관계를 활용한 스마트한 대체
3. **Feature Engineering**: Target Encoding, Log 변환, 파생변수 생성
4. **모델링**: LightGBM, XGBoost, CatBoost 등 Gradient Boosting 계열 + Optuna 튜닝
5. **앙상블**: Optuna 가중치 최적화 앙상블 추천
6. **Threshold 최적화**: OOF 예측으로 최적 임계값 탐색

---

## 11. 디렉토리 구조

```
spaceTitanic/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── processed/          # 전처리된 데이터
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── test_ids.csv
├── notebooks/
│   ├── 01_EDA.ipynb        # 탐색적 데이터 분석
│   ├── 02_Preprocessing.ipynb  # 전처리 및 Feature Engineering
│   └── 03_Modeling.ipynb   # 모델링 및 앙상블
├── models/                 # 저장된 모델
│   ├── optuna_best_params.pkl
│   ├── stacking_lgb_meta.pkl
│   └── voting_all.pkl
├── submissions/            # 제출 파일 (24개)
├── README.md               # 프로젝트 설명
└── Models.md               # 모델 설명 및 성능 비교
```
