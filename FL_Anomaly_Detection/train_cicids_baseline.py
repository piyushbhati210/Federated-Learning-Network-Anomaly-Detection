# ═══════════════════════════════════════════════════════════════
#  train_cicids_baseline.py
#  Train LR, RF and DNN on CICIDS2017 and CICIDS2018
#  Federated Learning - Network Traffic Anomaly Detection
#  Author: Piyush Bhati | RVU Summer Internship 2026
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)
import joblib
import sys
import time
sys.path.append('src')
from model import DNNModel

import warnings
warnings.filterwarnings("ignore")


def train_and_evaluate(dataset_name, train_path, test_path):
    print("\n" + "=" * 60)
    print(f"  TRAINING ON: {dataset_name}")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print(f"\n📂 Loading {dataset_name}...")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_test  = test_df.drop(columns=['label']).values
    y_test  = test_df['label'].values

    print(f"  ✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  ✅ Features: {X_train.shape[1]}")
    print(f"  ✅ Normal: {(y_train==0).sum():,} | "
          f"Attack: {(y_train==1).sum():,}")

    results = {}
    prefix  = dataset_name.lower().replace(" ", "_").replace("-", "")

    # ── 1. Logistic Regression ─────────────────────────────────
    print(f"\n🔵 Training Logistic Regression...")
    start = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42,
                            class_weight='balanced', n_jobs=-1)
    lr.fit(X_train, y_train)
    t = time.time() - start
    y_pred = lr.predict(X_test)
    results['Logistic Regression'] = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'time':      t
    }
    joblib.dump(lr, f"models/local/{prefix}_lr.pkl")
    print(f"  ✅ Accuracy: {results['Logistic Regression']['accuracy']*100:.2f}%"
          f" | F1: {results['Logistic Regression']['f1']:.4f}"
          f" | Time: {t:.1f}s")

    # ── 2. Random Forest ───────────────────────────────────────
    print(f"\n🟢 Training Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    t = time.time() - start
    y_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'time':      t
    }
    joblib.dump(rf, f"models/local/{prefix}_rf.pkl")
    print(f"  ✅ Accuracy: {results['Random Forest']['accuracy']*100:.2f}%"
          f" | F1: {results['Random Forest']['f1']:.4f}"
          f" | Time: {t:.1f}s")

    # ── 3. DNN ─────────────────────────────────────────────────
    print(f"\n🔴 Training DNN...")
    INPUT_DIM = X_train.shape[1]

    class DNNLogits(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32),   nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
    n0 = (yt == 0).sum().item()
    n1 = (yt == 1).sum().item()

    model     = DNNLogits(INPUT_DIM)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n0 / n1]))
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=0.001, weight_decay=1e-4)
    loader    = DataLoader(TensorDataset(Xt, yt),
                           batch_size=512, shuffle=True)

    start = time.time()
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS} — Loss: {loss.item():.4f}")

    t = time.time() - start
    model.eval()
    Xte = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = (torch.sigmoid(model(Xte)) >= 0.5).float().numpy()

    results['DNN'] = {
        'accuracy':  accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall':    recall_score(y_test, preds, zero_division=0),
        'f1':        f1_score(y_test, preds, zero_division=0),
        'time':      t
    }
    torch.save(model.state_dict(),
               f"models/local/{prefix}_dnn.pt")
    print(f"  ✅ Accuracy: {results['DNN']['accuracy']*100:.2f}%"
          f" | F1: {results['DNN']['f1']:.4f}"
          f" | Time: {t:.1f}s")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  📊 {dataset_name} — RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10}"
          f" {'Recall':>10} {'F1':>8}")
    print(f"  {'-'*63}")
    for model_name, r in results.items():
        print(f"  {model_name:<25} "
              f"{r['accuracy']*100:>9.2f}% "
              f"{r['precision']*100:>9.2f}% "
              f"{r['recall']*100:>9.2f}% "
              f"{r['f1']:>8.4f}")
    print(f"{'='*60}")

    return results


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  BASELINE TRAINING — CICIDS2017 + CICIDS2018")
    print("=" * 60)

    all_results = {}

    # CICIDS2017
    all_results['CICIDS2017'] = train_and_evaluate(
        dataset_name="CICIDS2017",
        train_path="data/processed/cicids2017_train.csv",
        test_path="data/processed/cicids2017_test.csv"
    )

    # CICIDS2018
    all_results['CICIDS2018'] = train_and_evaluate(
        dataset_name="CICIDS2018",
        train_path="data/processed/cicids2018_train.csv",
        test_path="data/processed/cicids2018_test.csv"
    )

    # ── Cross-dataset summary ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  🏆 FINAL CROSS-DATASET COMPARISON")
    print("=" * 60)
    print(f"  {'Dataset':<12} {'Model':<25} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'-'*55}")

    # Add UNSW-NB15 results from earlier
    unsw_results = {
        'Logistic Regression': {'accuracy': 0.9323, 'f1': 0.9522},
        'Random Forest':       {'accuracy': 0.9607, 'f1': 0.9714},
        'DNN':                 {'accuracy': 0.9335, 'f1': 0.9498},
    }
    for model_name, r in unsw_results.items():
        print(f"  {'UNSW-NB15':<12} {model_name:<25} "
              f"{r['accuracy']*100:>9.2f}% {r['f1']:>8.4f}")

    for dataset, results in all_results.items():
        for model_name, r in results.items():
            print(f"  {dataset:<12} {model_name:<25} "
                  f"{r['accuracy']*100:>9.2f}% {r['f1']:>8.4f}")

    print("=" * 60)
    print("\n✅ All models trained and saved!")
    print("👉 NEXT: Write the final report!")