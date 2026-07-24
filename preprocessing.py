# ═══════════════════════════════════════════════════════════════
#  preprocessing.py — UNSW-NB15 Data Preprocessing
#  Federated Learning - Network Traffic Anomaly Detection
#  Author: Piyush Bhati | RVU Summer Internship 2026
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

print("=" * 55)
print("  UNSW-NB15 DATA PREPROCESSING")
print("=" * 55)

# Step 1: Load Dataset
print("\nStep 1: Loading dataset...")
df = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Step 2: Explore
print("\nStep 2: Exploring dataset...")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Class distribution:\n{df['label'].value_counts()}")
print(f"  Attack categories:\n{df['attack_cat'].value_counts()}")

# Step 3: Clean
print("\nStep 3: Cleaning data...")
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Duplicates removed: {before - len(df)}")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
before = len(df)
df.dropna(inplace=True)
print(f"  Missing/Inf removed: {before - len(df)}")

# Step 4: Drop unnecessary columns (errors='ignore' handles missing 'id')
print("\nStep 4: Dropping unnecessary columns...")
df.drop(columns=['id', 'attack_cat'], inplace=True, errors='ignore')
print(f"  Shape after dropping: {df.shape}")

# Step 5: Encode categorical features
print("\nStep 5: Encoding categorical features...")
le = LabelEncoder()
cat_cols = ['proto', 'service', 'state']
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded: {col}")

# Step 6: Separate features and labels
print("\nStep 6: Separating features and labels...")
X = df.drop(columns=['label'])
y = df['label']
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Step 7: Normalize
print("\nStep 7: Normalizing features (Min-Max scaling)...")
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(f"  {X_scaled.shape[1]} features scaled to [0, 1]")

# Step 8: Train-test split
print("\nStep 8: Train-test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {X_train.shape[0]:,} samples")
print(f"  Test : {X_test.shape[0]:,} samples")

# Step 9: Split into 4 FL clients
print("\nStep 9: Creating 4 FL client datasets...")
X_train_r = X_train.reset_index(drop=True)
y_train_r = y_train.reset_index(drop=True)
NUM_CLIENTS = 4
client_size = len(X_train_r) // NUM_CLIENTS

os.makedirs("data/clients", exist_ok=True)
for i in range(NUM_CLIENTS):
    start = i * client_size
    end   = (i+1)*client_size if i < NUM_CLIENTS-1 else len(X_train_r)
    client_df = X_train_r.iloc[start:end].copy()
    client_df['label'] = y_train_r.iloc[start:end].values
    path = f"data/clients/client_{i+1}_train.csv"
    client_df.to_csv(path, index=False)
    print(f"  Client {i+1}: {len(client_df):,} samples -> {path}")

# Step 10: Save processed data
print("\nStep 10: Saving processed datasets...")
os.makedirs("data/processed", exist_ok=True)

train_df = X_train_r.copy()
train_df['label'] = y_train_r.values
test_df  = X_test.reset_index(drop=True).copy()
test_df['label']  = y_test.reset_index(drop=True).values

train_df.to_csv("data/processed/train_cleaned.csv", index=False)
test_df.to_csv("data/processed/test_cleaned.csv", index=False)
print(f"  Saved: data/processed/train_cleaned.csv ({len(train_df):,} rows)")
print(f"  Saved: data/processed/test_cleaned.csv  ({len(test_df):,} rows)")

# Summary
print("\n" + "=" * 55)
print("  PREPROCESSING COMPLETE!")
print("=" * 55)
print(f"  Total records  : {len(df):,}")
print(f"  Features       : {X_scaled.shape[1]}")
print(f"  Train samples  : {len(train_df):,}")
print(f"  Test samples   : {len(test_df):,}")
print(f"  FL clients     : {NUM_CLIENTS}")
print(f"  Normal (0)     : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"  Attack (1)     : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
print("\nNEXT: Run python train_baseline.py")