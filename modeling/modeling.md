# BDA Contest - 모델링 전략 정리

## 1. 대회 개요

- **목표**: BDA 학회 지원자의 **합격 여부(completed)** 예측
- **평가 지표**: F1 Score
- **데이터**: Train 748개, Test 814개
- **Target 불균형**: 0(불합격) 525명 / 1(합격) 223명 (7:3)

---

## 2. 핵심 전략

### 기본 원칙: **Less is More**
> 작은 데이터(748개)에서는 복잡한 기법이 오히려 과적합을 유발함

### 사용 모델

| 모델 | 특징 |
|------|------|
| **TabPFN** | 소규모 데이터 SOTA, 튜닝 불필요 |
| **CatBoost** | Native Categorical 처리 |
| **XGBoost** | 검증된 베이스라인 |

### 앙상블 방식: Simple Blending
```python
# OOF F1 기반 가중 평균
blend = w1*tabpfn + w2*catboost + w3*xgboost
```

---

## 3. 전처리

### 최소 전처리 원칙
- ❌ Target Encoding (과적합 유발)
- ❌ 복잡한 Feature Engineering
- ✅ 결측치만 처리
- ✅ Label Encoding (TabPFN용)

---

## 4. 실험 기록

| 버전 | 기법 | Public F1 |
|------|------|-----------|
| v1 | 기본 모델 | 0.397 |
| v2 | Target Encoding | 0.329 ↓ |
| v3 | Seed Ensemble | 0.382 ↓ |
| v4 | **TabPFN Blend** | ? |

---

## 5. 실행 방법

```python
# Colab
!git pull origin main
# Run All
```

---

*Last Updated: 2026-02-02*
