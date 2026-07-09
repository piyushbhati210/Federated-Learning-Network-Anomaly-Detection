# ═══════════════════════════════════════════════════════════════
#  preprocessing_cicids.py
#  Preprocesses BOTH CICIDS2017 and CICIDS2018 datasets
#  Federated Learning - Network Traffic Anomaly Detection
#  Author: Piyush Bhati | RVU Summer Internship 2026
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import glob

def preprocess_dataset(name, data_path, is_folder=False):
    print("\n" + "=" * 60)
    print(f"  PROCESSING: {name}")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────
    if is_folder:
        csv_files = glob.glob(f"{data_path}/*.csv")
        if not csv_files:
            print(f"❌ No CSV files found in {data_path}")
            return None
        print(f"📂 Found {len(csv_files)} CSV file(s) — loading...")
        dfs = []
        for f in csv_files:
            try:
                temp = pd.read_csv(f, encoding='utf-8', low_memory=False)
                dfs.append(temp)
                print(f"  ✅ {os.path.basename(f)} → {temp.shape}")
            except Exception:
                try:
                    temp = pd.read_csv(f, encoding='latin-1', low_memory=False)
                    dfs.append(temp)
                    print(f"  ✅ {os.path.basename(f)} → {temp.shape}")
                except Exception as e2:
                    print(f"  ⚠️  Skipped: {e2}")
        df = pd.concat(dfs, ignore_index=True)
    else:
        if not os.path.exists(data_path):
            print(f"❌ File not found: {data_path}")
            return None
        try:
            df = pd.read_csv(data_path, encoding='utf-8', low_memory=False)
        except Exception:
            df = pd.read_csv(data_path, encoding='latin-1', low_memory=False)

    df.columns = df.columns.str.strip()
    print(f"\n✅ Total shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"First 5 columns: {list(df.columns[:5])}")

    # ── Find label column ──────────────────────────────────────
    label_col = None
    for possible in ['Label', 'label', 'Class', 'class', 'LABEL',
                     'target', 'Target', 'Attack Type', 'attack_type',
                     'AttackType', 'attack type', 'Type', 'type']:
        if possible in df.columns:
            label_col = possible
            break

    if label_col is None:
        print(f"❌ Label column not found! Columns: {list(df.columns)}")
        return None

    print(f"\n✅ Label column found: '{label_col}'")
    print(f"\nClass distribution (top 10):")
    print(df[label_col].value_counts().head(10))

    # ── Binary label ──────────────────────────────────────────
    normal_values = [
        'BENIGN', 'benign', 'Benign', 'Benign ',
        '0', 'NORMAL', 'normal', 'Normal',
        'Normal Traffic', 'normal traffic', 'NORMAL TRAFFIC'
    ]

    df['binary_label'] = df[label_col].apply(
        lambda x: 0 if str(x).strip() in normal_values else 1
    )

    print(f"\n✅ Binary labels created:")
    print(f"   Normal (0) : {(df['binary_label']==0).sum():,} "
          f"({(df['binary_label']==0).mean()*100:.1f}%)")
    print(f"   Attack (1) : {(df['binary_label']==1).sum():,} "
          f"({(df['binary_label']==1).mean()*100:.1f}%)")

    # ── Drop irrelevant columns ────────────────────────────────
    drop_cols = [c for c in [
        label_col, 'binary_label', 'Flow ID', 'Source IP',
        'Src IP', 'Destination IP', 'Dst IP', 'Source Port',
        'Destination Port', 'Timestamp', 'timestamp',
        'src_ip', 'dst_ip'
    ] if c in df.columns]

    y = df['binary_label'].copy()
    df.drop(columns=drop_cols, inplace=True)

    # ── Clean ─────────────────────────────────────────────────
    print("\n🧹 Cleaning...")

    before = len(df)
    df.drop_duplicates(inplace=True)
    y = y.loc[df.index]
    print(f"  ✅ Duplicates removed : {before - len(df):,}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    mask = df.notna().all(axis=1)
    df = df[mask]
    y = y[mask]
    print(f"  ✅ Inf/NaN removed    : {before - len(df):,}")

    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df.drop(columns=const_cols, inplace=True)
    print(f"  ✅ Constant cols      : {len(const_cols)} removed")
    print(f"  ✅ Clean shape        : {df.shape}")

    # ── Encode ────────────────────────────────────────────────
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    if cat_cols:
        print(f"  ✅ Encoded {len(cat_cols)} categorical columns")

    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    y = y.loc[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)

    # ── Normalize ─────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print(f"  ✅ Normalized {X_scaled.shape[1]} features to [0,1]")

    # ── Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Save ──────────────────────────────────────────────────
    prefix = name.lower().replace(" ", "_").replace("-", "")
    train_df = X_train.reset_index(drop=True).copy()
    train_df['label'] = y_train.reset_index(drop=True).values
    test_df = X_test.reset_index(drop=True).copy()
    test_df['label'] = y_test.reset_index(drop=True).values

    os.makedirs("data/processed", exist_ok=True)
    train_path = f"data/processed/{prefix}_train.csv"
    test_path  = f"data/processed/{prefix}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"\n  ✅ Saved: {train_path}")
    print(f"  ✅ Saved: {test_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ {name} PREPROCESSING COMPLETE")
    print(f"  Total records  : {len(X_scaled):,}")
    print(f"  Features       : {X_scaled.shape[1]}")
    print(f"  Normal traffic : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"  Attack traffic : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    print(f"  Train samples  : {len(train_df):,}")
    print(f"  Test samples   : {len(test_df):,}")
    print(f"{'='*60}")
    return train_df, test_df


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CICIDS2017 + CICIDS2018 PREPROCESSING")
    print("  Federated Learning - Network Traffic Anomaly Detection")
    print("=" * 60)

    preprocess_dataset(
        name="CICIDS2017",
        data_path="data/raw/CICIDS2017",
        is_folder=True
    )

    preprocess_dataset(
        name="CICIDS2018",
        data_path="data/raw/CICIDS2018/02-14-2018.csv",
        is_folder=False
    )

    print("\n" + "=" * 60)
    print("  ✅ BOTH DATASETS PREPROCESSED SUCCESSFULLY!")
    print("  Files saved in data/processed/:")
    print("    cicids2017_train.csv + cicids2017_test.csv")
    print("    cicids2018_train.csv + cicids2018_test.csv")
    print("=" * 60)
    print("\n👉 NEXT: Run train_baseline.py on each dataset!")