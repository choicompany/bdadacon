
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

# Columns configurations
# Define which columns should be treated as text to generate 'length' features
TEXT_COLS = [
    'whyBDA', 'what_to_gain', 'incumbents_lecture_scale_reason', 
    'interested_company', 'expected_domain', 'onedayclass_topic'
]

# Columns to drop explicitly
DROP_COLS = ['ID', 'generation'] 

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def feature_engineering(df):
    """
    Generate derived features.
    """
    print("Generating derived features...")
    # 1. Text Length Features
    for col in TEXT_COLS:
        if col in df.columns:
            # Fill NaN with empty string to count length 0
            df[col] = df[col].fillna('') 
            df[f'{col}_len'] = df[col].apply(len)
            
    return df

def preprocess_pipeline(train_df, test_df):
    print("Starting preprocessing pipeline...")
    
    # 1. Separate Target
    target_col = 'completed'
    y = train_df[target_col]
    
    # Combine for consistent encoding
    # Drop target from train temporary
    train_x = train_df.drop(columns=[target_col])
    
    # Add a marker to split later
    train_x['is_train'] = 1
    test_df['is_train'] = 0
    
    combined = pd.concat([train_x, test_df], axis=0, ignore_index=True)
    
    # 2. Drop unnecessary columns
    combined = combined.drop(columns=DROP_COLS, errors='ignore')
    
    # 3. Feature Engineering (Text Length)
    combined = feature_engineering(combined)
    
    # 4. Handle Missing Values & Encoding
    # Identify types
    cat_cols = combined.select_dtypes(include=['object', 'category']).columns
    num_cols = combined.select_dtypes(include=['number', 'bool']).columns
    
    # Exclude 'is_train' and derived length cols from categoricals if they slid in
    num_cols = [c for c in num_cols if c != 'is_train']
    
    print(f"Numerical columns: {len(num_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")
    
    # 4.1 Numerical: Median Filling
    for col in num_cols:
        median_val = combined[col].median()
        combined[col] = combined[col].fillna(median_val)
        
    # 4.2 Categorical: Unknown Filling & Encoding
    # Strategy: 
    # If cardinality < 15 -> One-Hot Encoding (users request: "Can I use OneHot?")
    # If cardinality >= 15 -> Label Encoding
    
    onehot_cols = []
    label_cols = []
    
    for col in cat_cols:
        combined[col] = combined[col].fillna('Unknown')
        combined[col] = combined[col].astype(str)
        
        n_unique = combined[col].nunique()
        if n_unique < 15:
            onehot_cols.append(col)
        else:
            label_cols.append(col)
            
    print(f"One-Hot Encoding columns (<15 unique): {len(onehot_cols)}")
    print(f"Label Encoding columns (>=15 unique): {len(label_cols)}")
    
    # Apply Label Encoding
    le = LabelEncoder()
    for col in label_cols:
        combined[col] = le.fit_transform(combined[col])
        
    # Apply One-Hot Encoding
    if onehot_cols:
        combined = pd.get_dummies(combined, columns=onehot_cols, drop_first=False, dtype=int)
        
    # 5. Split back
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
    
    print(f"Saving to {PREPROC_DIR}...")
    train_proc.to_csv(TRAIN_OUT, index=False)
    test_proc.to_csv(TEST_OUT, index=False)
    
    print("Preprocessing Done.")
    print(f"Train Cleaned Shape: {train_proc.shape}")
    print(f"Test Cleaned Shape: {test_proc.shape}")

if __name__ == "__main__":
    main()
