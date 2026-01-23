
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Settings
# ==========================================
RAW_DATA_DIR = r'c:\Users\choke\bdadacon\rawdata'
PREPROC_DIR = r'c:\Users\choke\bdadacon\preprocessing'
TRAIN_PATH = os.path.join(RAW_DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(RAW_DATA_DIR, 'test.csv')

TRAIN_OUT = os.path.join(PREPROC_DIR, 'train_processed.csv')
TEST_OUT = os.path.join(PREPROC_DIR, 'test_processed.csv')

# Columns to drop
DROP_COLS = ['ID', 'generation']

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def preprocess_pipeline(train_df, test_df):
    print("Starting preprocessing pipeline...")
    
    # 1. Separate Target
    target_col = 'completed'
    y = train_df[target_col]
    
    # Combine for consistent encoding
    train_x = train_df.drop(columns=[target_col])
    
    # Marker for splitting
    train_x['is_train'] = 1
    test_df['is_train'] = 0
    
    combined = pd.concat([train_x, test_df], axis=0, ignore_index=True)
    
    # 2. Drop unnecessary columns
    combined = combined.drop(columns=DROP_COLS, errors='ignore')
    
    # 3. Identify column types
    cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = combined.select_dtypes(include=['number', 'bool']).columns.tolist()
    
    # Exclude marker
    if 'is_train' in num_cols:
        num_cols.remove('is_train')
    
    print(f"Numerical columns: {len(num_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")
    
    # 4. Handle Missing Values
    # - Numerical: Boosting 모델(XGBoost, LightGBM)은 NaN을 자체 처리 가능.
    #   그래도 안전하게 -1 또는 -999로 채워서 "결측" 표시를 명확히 함.
    # - Categorical: 'Unknown' 문자열로 채움 (결측 자체가 하나의 범주가 됨)
    
    for col in num_cols:
        combined[col] = combined[col].fillna(-1)  # 결측 표시용 특수값
        
    for col in cat_cols:
        combined[col] = combined[col].fillna('Unknown')
        combined[col] = combined[col].astype(str)
    
    # 5. Encoding
    # - Low Cardinality (< 15 unique): One-Hot
    # - High Cardinality (>= 15 unique): Label Encoding
    
    onehot_cols = []
    label_cols = []
    
    for col in cat_cols:
        n_unique = combined[col].nunique()
        if n_unique < 15:
            onehot_cols.append(col)
        else:
            label_cols.append(col)
            
    print(f"One-Hot Encoding: {len(onehot_cols)} columns")
    print(f"Label Encoding: {len(label_cols)} columns")
    
    # Apply Label Encoding
    le = LabelEncoder()
    for col in label_cols:
        combined[col] = le.fit_transform(combined[col])
        
    # Apply One-Hot Encoding
    if onehot_cols:
        combined = pd.get_dummies(combined, columns=onehot_cols, drop_first=False, dtype=int)
        
    # 6. Split back
    train_processed = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    test_processed = combined[combined['is_train'] == 0].drop(columns=['is_train'])
    
    # Add target back
    train_processed[target_col] = y.values
    
    return train_processed, test_processed

def main():
    if not os.path.exists(PREPROC_DIR):
        os.makedirs(PREPROC_DIR)
        
    train_df, test_df = load_data()
    
    train_proc, test_proc = preprocess_pipeline(train_df, test_df)
    
    print(f"\nSaving to {PREPROC_DIR}...")
    train_proc.to_csv(TRAIN_OUT, index=False)
    test_proc.to_csv(TEST_OUT, index=False)
    
    print("Preprocessing Done.")
    print(f"Train Shape: {train_proc.shape}")
    print(f"Test Shape: {test_proc.shape}")

if __name__ == "__main__":
    main()
