"""
===========================================
BDA Contest - Final High-Performance Model
===========================================
이 스크립트 하나로 전처리 + 모델링 + 제출파일 생성까지 완료됩니다.
Colab에서 바로 실행하세요:
  !python modeling/run_final.py
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

# pip install이 안 되어 있으면 설치
try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost', '-q'])
    import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'catboost', '-q'])
    from catboost import CatBoostClassifier

# ==========================================
# 1. DATA LOADING
# ==========================================
print("=" * 50)
print("1. Loading Data...")
print("=" * 50)

RAW_TRAIN_URL = 'https://raw.githubusercontent.com/choicompany/bdadacon/refs/heads/main/rawdata/train.csv'
RAW_TEST_URL = 'https://raw.githubusercontent.com/choicompany/bdadacon/refs/heads/main/rawdata/test.csv'

train_raw = pd.read_csv(RAW_TRAIN_URL)
test_raw = pd.read_csv(RAW_TEST_URL)

print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("\n" + "=" * 50)
print("2. Feature Engineering...")
print("=" * 50)

# Helper functions
def count_items(text):
    """콤마로 구분된 항목 개수"""
    if pd.isna(text) or str(text).strip() == '':
        return 0
    return str(text).count(',') + 1

def text_length(text):
    """텍스트 길이"""
    if pd.isna(text):
        return 0
    return len(str(text))

def clean_text(text):
    """특수문자 제거"""
    if pd.isna(text):
        return 'Unknown'
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text))

# Separate target
y = train_raw['completed'].copy()
train_ids = train_raw['ID'].copy()
test_ids = test_raw['ID'].copy()

# Combine for preprocessing
train_x = train_raw.drop(columns=['completed', 'ID'])
test_x = test_raw.drop(columns=['ID'])
train_x['is_train'] = 1
test_x['is_train'] = 0
combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

# 2-1. Derived Features
print("  - Creating derived features...")

# Null count (성실도)
combined['null_count'] = combined.isnull().sum(axis=1)

# Text length features
text_cols = ['whyBDA', 'what_to_gain', 'hope_for_group', 
             'incumbents_lecture_scale_reason', 'interested_company']
for col in text_cols:
    if col in combined.columns:
        combined[f'{col}_len'] = combined[col].apply(text_length)

# Count features (다중 선택)
multi_cols = ['certificate_acquisition', 'desired_certificate', 
              'desired_job', 'desired_job_except_data',
              'onedayclass_topic', 'expected_domain']
for col in multi_cols:
    if col in combined.columns:
        combined[f'{col}_count'] = combined[col].apply(count_items)

# Major interaction
combined['major_combined'] = combined['major1_1'].fillna('') + '_' + combined['major1_2'].fillna('')
combined['major_is_data'] = combined['major_data'].astype(str).apply(lambda x: 1 if x.lower() == 'true' else 0)

# Job type
combined['is_student'] = (combined['job'] == '대학생').astype(int)
combined['is_worker'] = (combined['job'] == '직장인').astype(int)

# Class preference
combined['class1_filled'] = combined['class1'].notna().astype(int)
combined['wants_offline'] = combined['hope_for_group'].astype(str).str.contains('오프라인', na=False).astype(int)

# 2-2. Drop unnecessary columns
drop_cols = ['generation']
combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns], errors='ignore')

# 2-3. Handle Missing & Encoding
print("  - Encoding categorical variables...")

cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = combined.select_dtypes(include=['number', 'bool']).columns.tolist()
if 'is_train' in num_cols:
    num_cols.remove('is_train')

# Fill NA
for col in num_cols:
    combined[col] = combined[col].fillna(-1)

for col in cat_cols:
    combined[col] = combined[col].fillna('Unknown')
    combined[col] = combined[col].apply(clean_text)

# Encoding: High cardinality -> Label, Low -> One-Hot
high_card = [c for c in cat_cols if combined[c].nunique() > 15]
low_card = [c for c in cat_cols if combined[c].nunique() <= 15]

le_dict = {}
for col in high_card:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    le_dict[col] = le

if low_card:
    combined = pd.get_dummies(combined, columns=low_card, drop_first=False, dtype=int)

# 2-4. Split back
train_processed = combined[combined['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
test_processed = combined[combined['is_train'] == 0].drop(columns=['is_train']).reset_index(drop=True)

# Clean column names (XGBoost compatibility)
def clean_columns(cols):
    seen = {}
    result = []
    for c in cols:
        c = re.sub(r'[\[\]<>\s]', '_', str(c))
        if c in seen:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            result.append(c)
    return result

train_processed.columns = clean_columns(train_processed.columns)
test_processed.columns = clean_columns(test_processed.columns)

# Convert to numpy
X = train_processed.values.astype(np.float32)
X_test = test_processed.values.astype(np.float32)
y = y.values.astype(int)

print(f"  - Final Train Shape: {X.shape}")
print(f"  - Final Test Shape: {X_test.shape}")
print(f"  - Target Distribution: 0={sum(y==0)}, 1={sum(y==1)}")

# ==========================================
# 3. MODEL TRAINING (Optimized Ensemble)
# ==========================================
print("\n" + "=" * 50)
print("3. Training Models...")
print("=" * 50)

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

# Class imbalance ratio
scale = sum(y == 0) / sum(y == 1)
print(f"  - Class imbalance ratio: {scale:.2f}")

# Define models with optimized params
xgb_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale,
    'random_state': SEED,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'n_jobs': -1
}

lgb_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1
}

cat_params = {
    'iterations': 500,
    'depth': 5,
    'learning_rate': 0.05,
    'random_seed': SEED,
    'verbose': 0,
    'class_weights': {0: 1, 1: scale}
}

# OOF Predictions Function
def get_oof_predictions(model_class, params, X, y, X_test, n_folds=5):
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = model_class(**params)
        model.fit(X_tr, y_tr)
        
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds
    
    return oof, test_preds

# Train each model
print("  - Training XGBoost...")
xgb_oof, xgb_test = get_oof_predictions(xgb.XGBClassifier, xgb_params, X, y, X_test, N_FOLDS)
xgb_f1 = f1_score(y, (xgb_oof >= 0.5).astype(int))
print(f"    XGBoost OOF F1: {xgb_f1:.4f}")

print("  - Training LightGBM...")
lgb_oof, lgb_test = get_oof_predictions(lgb.LGBMClassifier, lgb_params, X, y, X_test, N_FOLDS)
lgb_f1 = f1_score(y, (lgb_oof >= 0.5).astype(int))
print(f"    LightGBM OOF F1: {lgb_f1:.4f}")

print("  - Training CatBoost...")
cat_oof, cat_test = get_oof_predictions(CatBoostClassifier, cat_params, X, y, X_test, N_FOLDS)
cat_f1 = f1_score(y, (cat_oof >= 0.5).astype(int))
print(f"    CatBoost OOF F1: {cat_f1:.4f}")

# ==========================================
# 4. ENSEMBLE & THRESHOLD OPTIMIZATION
# ==========================================
print("\n" + "=" * 50)
print("4. Ensembling & Threshold Optimization...")
print("=" * 50)

# Stack OOF predictions
oof_stack = np.column_stack([xgb_oof, lgb_oof, cat_oof])
test_stack = np.column_stack([xgb_test, lgb_test, cat_test])

# Meta model
meta = LogisticRegression(random_state=SEED, max_iter=1000)
meta.fit(oof_stack, y)

final_oof = meta.predict_proba(oof_stack)[:, 1]
final_test_probs = meta.predict_proba(test_stack)[:, 1]

print(f"  - Meta model weights: {meta.coef_[0]}")

# Threshold optimization (maximize F1)
print("  - Optimizing threshold for F1...")
best_f1 = 0
best_threshold = 0.5

for th in np.arange(0.2, 0.8, 0.01):
    preds = (final_oof >= th).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th

print(f"  - Best Threshold: {best_threshold:.2f}")
print(f"  - Best OOF F1 Score: {best_f1:.4f}")

# Final predictions
final_preds = (final_test_probs >= best_threshold).astype(int)
print(f"  - Predicted 0: {sum(final_preds == 0)}, 1: {sum(final_preds == 1)}")

# ==========================================
# 5. CREATE SUBMISSION
# ==========================================
print("\n" + "=" * 50)
print("5. Creating Submission...")
print("=" * 50)

submission = pd.DataFrame({
    'ID': test_ids,
    'completed': final_preds
})

# Save
submission.to_csv('submission.csv', index=False)
print(f"  - Saved: submission.csv")
print(submission.head(10))

# Also save to modeling folder if exists
try:
    submission.to_csv('modeling/submission.csv', index=False)
    print(f"  - Also saved to: modeling/submission.csv")
except:
    pass

print("\n" + "=" * 50)
print("DONE! Submit 'submission.csv' to the contest.")
print("=" * 50)
