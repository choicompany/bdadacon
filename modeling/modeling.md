# BDA Contest - 모델링 전략 정리

## 1. 대회 개요

- **목표**: BDA 학회 지원자의 **합격 여부(completed)** 예측
- **평가 지표**: F1 Score (추정)
- **데이터 특성**: 
  - Train: 748개 샘플
  - Test: 814개 샘플
  - Target 불균형: 0(불합격) 525명 / 1(합격) 223명 (약 7:3 비율)

---

## 2. 핵심 전략 (DACON 1등 솔루션 참고)

### 2.1 Feature Engineering

#### 기본 피처
| 피처 타입 | 설명 | 예시 |
|-----------|------|------|
| **Count Features** | 콤마로 구분된 다중 선택 항목의 개수 | `certificate_acquisition_count`, `desired_job_count` |
| **Text Length** | 서술형 답변의 글자 수 | `whyBDA_len`, `what_to_gain_len` |
| **Null Count** | 비어있는 항목 수 (성실도 지표) | `null_count` |
| **Boolean 변환** | True/False 문자열을 0/1로 변환 | `major_data` |

#### Target Encoding (핵심!)
- **개념**: 범주형 변수를 해당 그룹의 **평균 합격률**로 변환
- **적용 컬럼**: `school1`, `major1_1`, `major_field`, `job`, `inflow_route`, `class1`
- **방법**: **KFold 방식**으로 계산 (Data Leakage 방지)
  
```python
# 예시: school1별 평균 합격률
# 서울대: 0.45, 연세대: 0.38, 고려대: 0.41 ...
# 이 값을 새로운 피처로 추가
train_df['school1_target_mean'] = 0.45  # 해당 학교의 평균 합격률
```

> **왜 효과적인가?**  
> 모델이 "이 학교/전공 출신은 합격률이 높다/낮다"를 직접 학습할 수 있음

---

### 2.2 모델링

#### XGBoost 단일 모델
- AutoGluon이나 복잡한 앙상블보다 **잘 튜닝된 XGBoost 단일 모델**이 효과적
- 소규모 데이터(748개)에서는 단순한 모델이 과적합 방지에 유리

#### 하이퍼파라미터
```python
xgb_params = {
    'n_estimators': 500,
    'max_depth': 4,          # 과적합 방지를 위해 얕게
    'learning_rate': 0.03,   # 천천히 학습
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 2.35, # 클래스 불균형 비율 (525/223)
}
```

#### Class Imbalance 처리
- `scale_pos_weight = (0의 개수) / (1의 개수) = 2.35`
- 소수 클래스(1=합격)에 더 큰 가중치 부여

---

### 2.3 Seed Ensemble

- **개념**: 여러 랜덤 시드로 학습 후 **예측값 평균**
- **사용 시드**: `[0, 1, 2, 42, 2024, 2025]` (6개)
- **효과**: 랜덤성으로 인한 성능 변동 감소, 안정적인 예측

```python
for seed in [0, 1, 2, 42, 2024, 2025]:
    model.fit(X, y, random_state=seed)
    preds += model.predict_proba(X_test)[:, 1] / 6  # 평균
```

---

### 2.4 Threshold 최적화

- **문제**: 기본 임계값 0.5를 사용하면 불균형 데이터에서 성능 저하
- **해결**: OOF(Out-of-Fold) 예측값으로 **F1 최대화 임계값** 탐색

```python
best_f1, best_th = 0, 0.5
for th in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y, (oof_preds >= th).astype(int))
    if f1 > best_f1:
        best_f1, best_th = f1, th
```

> 보통 **0.3~0.4** 정도가 최적 임계값으로 나옴 (데이터 불균형 때문)

---

## 3. 파이프라인 요약

```
[Raw Data]
    ↓
[Feature Engineering]
    - Count Features (다중 선택 항목 개수)
    - Text Length (서술형 답변 길이)
    - Target Encoding (학교/전공별 평균 합격률)
    ↓
[Label Encoding]
    - 범주형 변수를 숫자로 변환
    ↓
[XGBoost Training]
    - 5-Fold Stratified CV
    - 6개 시드로 Seed Ensemble
    - scale_pos_weight로 클래스 불균형 처리
    ↓
[Threshold Optimization]
    - OOF 예측값으로 F1 최대화 임계값 탐색
    ↓
[Final Prediction]
    - Test 데이터에 최적 임계값 적용
    ↓
[submission.csv]
```

---

## 4. 참고 자료

- **DACON 전력사용량 예측 1위 솔루션**
  - Target Encoding, Seed Ensemble, Custom Objective 등 핵심 전략 참고
  - [원문 블로그](https://blog.naver.com/jaehyunup)

- **Feature Engineering for Tabular Data**
  - Target Encoding with KFold (Leakage 방지)
  - Cyclical Encoding (시간 데이터용)

---

## 5. 실행 방법

```python
# Colab에서 실행
!git pull origin main

# 노트북 열고 Run All
# submission.csv 다운로드 후 제출
```

---

## 6. 개선 가능 포인트

1. **Feature Selection**: 불필요한 피처 제거로 과적합 방지
2. **Hyperparameter Tuning**: Optuna로 더 정밀한 튜닝
3. **Stacking**: XGBoost + LightGBM + CatBoost 스태킹 앙상블
4. **텍스트 피처 활용**: `whyBDA` 등 서술형 답변에 TF-IDF 또는 BERT 임베딩 적용
5. **Cross Validation 전략**: Repeated Stratified KFold (반복 교차검증)

---

*Last Updated: 2026-02-02*
