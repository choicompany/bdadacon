
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import re

# ==========================================
# Settings
# ==========================================
RAW_DATA_DIR = r'c:\Users\choke\bdadacon\rawdata'
PREPROC_DIR = r'c:\Users\choke\bdadacon\preprocessing'
TRAIN_PATH = os.path.join(RAW_DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(RAW_DATA_DIR, 'test.csv')

TRAIN_OUT = os.path.join(PREPROC_DIR, 'train_processed_enhanced.csv')
TEST_OUT = os.path.join(PREPROC_DIR, 'test_processed_enhanced.csv')

# Columns to drop immediately
DROP_COLS = ['ID', 'generation']

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def count_items(text):
    """콤마로 구분된 항목의 개수를 셉니다."""
    if pd.isna(text) or text == '':
        return 0
    return str(text).count(',') + 1

def preprocess_pipeline(train_df, test_df):
    print("Starting ENHANCED preprocessing pipeline...")
    
    # 1. Separate Target
    target_col = 'completed'
    y = train_df[target_col].copy()
    
    train_x = train_df.drop(columns=[target_col])
    
    # Marker for splitting
    train_x['is_train'] = 1
    test_df['is_train'] = 0
    
    combined = pd.concat([train_x, test_df], axis=0, ignore_index=True)
    
    # 2. Advanced Feature Engineering (BEFORE Encoding)
    print("Generating derived features...")
    
    # 2-1. Null Count (성실도 지표)
    combined['null_count'] = combined.isnull().sum(axis=1)
    
    # 2-2. Text Length Features (얼마나 길게 썼는가 - 정성적 평가 요소)
    text_cols = ['whyBDA', 'what_to_gain', 'hope_for_group', 
                 'incumbents_lecture_scale_reason', 'desired_career_path']
    
    for col in text_cols:
        if col in combined.columns:
            # 글자 길이
            combined[f'{col}_len'] = combined[col].astype(str).apply(len)
            # 결측 여부
            combined[f'{col}_isna'] = combined[col].isna().astype(int)

    # 2-3. Count Features (항목 개수)
    # 콤마로 구분된 다중 선택 항목들
    multi_select_cols = ['certificate_acquisition', 'desired_certificate', 
                         'desired_job', 'desired_job_except_data',
                         'onedayclass_topic',  'interested_company']
    
    for col in multi_select_cols:
        if col in combined.columns:
            combined[f'{col}_count'] = combined[col].apply(count_items)

    # 2-4. Major Interaction
    combined['major_full'] = combined['major1_1'].fillna('') + "_" + combined['major1_2'].fillna('')
    combined['major_is_same'] = (combined['major1_1'] == combined['major1_2']).astype(int)

    # 3. Drop unnecessary columns
    combined = combined.drop(columns=DROP_COLS, errors='ignore')

    # 4. Handle Missing Values & Type Conversion
    cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = combined.select_dtypes(include=['number', 'bool']).columns.tolist()
    
    if 'is_train' in num_cols: num_cols.remove('is_train')
    
    # Fill NA
    for col in num_cols:
        combined[col] = combined[col].fillna(-1)
        
    for col in cat_cols:
        combined[col] = combined[col].fillna('Unknown')
        combined[col] = combined[col].astype(str)
        # Clean text slightly
        combined[col] = combined[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9가-힣]', '', x))

    # 5. Encoding Strategy
    # Cardinality 확인
    high_card_cols = []
    low_card_cols = []
    
    for col in cat_cols:
        n_unique = combined[col].nunique()
        if n_unique > 20: 
            high_card_cols.append(col)
        else:
            low_card_cols.append(col)
            
    print(f"High Cardinality (Label Encoding): {len(high_card_cols)}")
    print(f"Low Cardinality (One-Hot Encoding): {len(low_card_cols)}")

    # 5-1. Label Encoding for High Cardinality
    le = LabelEncoder()
    for col in high_card_cols:
        combined[col] = le.fit_transform(combined[col])
        combined[col] = combined[col].astype('category') # LightGBM 등에서 효율적

    # 5-2. One-Hot Encoding for Low Cardinality
    if low_card_cols:
        combined = pd.get_dummies(combined, columns=low_card_cols, drop_first=False, dtype=int)

    # 6. Split back
    train_processed = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    test_processed = combined[combined['is_train'] == 0].drop(columns=['is_train'])
    
    # Add target back
    train_processed[target_col] = y.values
    
    # 7. Target Encoding (Optional/Advanced - KFold Mean Encoding)
    # 여기서는 간단히 하기 위해 생략하되, 복잡한 모델링 파일에서 수행하도록 함.
    # 대신 여기서는 전처리가 완료된 깔끔한 데이터셋을 내보냄.

    return train_processed, test_processed

def main():
    if not os.path.exists(PREPROC_DIR):
        os.makedirs(PREPROC_DIR)
        
    train_df, test_df = load_data()
    
    train_proc, test_proc = preprocess_pipeline(train_df, test_df)
    
    print(f"\nSaving to {PREPROC_DIR}...")
    train_proc.to_csv(TRAIN_OUT, index=False)
    test_proc.to_csv(TEST_OUT, index=False)
    
    print("Enhanced Preprocessing Done.")
    print(f"Train Shape: {train_proc.shape}")
    print(f"Test Shape: {test_proc.shape}")

if __name__ == "__main__":
    main()
